#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from steering_utils import (  # noqa: E402
    PromptTooLongError,
    build_preference_prompt,
    default_dropped_samples_path,
    dropped_pair_record,
    input_device,
    load_concept_pairs,
    load_model_and_tokenizer,
    max_length_with_reserved_tokens,
    to_jsonable,
    write_jsonl,
)


ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))

LEADING_LABEL_RE = re.compile(r"^\s*(?:answer\s*:\s*)?([AB])\b", re.IGNORECASE)
LABEL_RE = re.compile(r"\b([AB])\b", re.IGNORECASE)
TIE_RE = re.compile(
    r"\b(tie|equal|equally|both|same|no preference|cannot choose|can't choose|unable to choose|neither)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-judge baseline for RQ2. "
            "Build the original A/B pair prompt, let the model generate an answer, "
            "then parse the response into A/B/tie/invalid/unparsable."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-files", type=Path, nargs="+", required=True)
    parser.add_argument("--task-field", default="")
    parser.add_argument("--positive-field", required=True)
    parser.add_argument("--negative-field", required=True)
    parser.add_argument("--concept-name", default="concept")
    parser.add_argument("--positive-name", default="positive")
    parser.add_argument("--negative-name", default="negative")
    parser.add_argument("--metadata-fields", nargs="+", default=[])
    parser.add_argument("--preference-question", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, default=None)
    parser.add_argument("--dropped-samples-output", type=Path, default=None)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Total token budget including reserved generation tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Generation length budget for the judge answer.",
    )
    parser.add_argument(
        "--truncation-side",
        choices=("right", "left"),
        default="left",
        help="Only used together with --max-length.",
    )
    parser.add_argument("--overflow-policy", choices=("truncate", "drop"), default="truncate")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false")
    parser.set_defaults(prepend_bos=True)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def default_samples_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}.samples.jsonl")


def load_pairs(args: argparse.Namespace):
    pairs = load_concept_pairs(
        args.data_files,
        task_field=args.task_field,
        positive_field=args.positive_field,
        negative_field=args.negative_field,
        concept_name=args.concept_name,
        positive_name=args.positive_name,
        negative_name=args.negative_name,
        metadata_fields=args.metadata_fields,
    )
    if args.limit is not None:
        pairs = pairs[: args.limit]
    return pairs


def generate_completion(model, tokenizer, encoded, *, args: argparse.Namespace) -> str:
    device = input_device(model)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    input_len = int(input_ids.shape[1])

    generate_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": bool(args.do_sample),
        "max_new_tokens": int(args.max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if args.do_sample:
        generate_kwargs["temperature"] = float(args.temperature)
        generate_kwargs["top_p"] = float(args.top_p)

    with torch.no_grad():
        generated = model.generate(**generate_kwargs)
    completion_tokens = generated[:, input_len:]
    return tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)[0]


def parse_judge_completion(text: str) -> tuple[str, str | None]:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return "unparsable", None
    if TIE_RE.search(normalized):
        return "tie", None

    leading = LEADING_LABEL_RE.search(normalized)
    if leading is not None:
        return "parsed", leading.group(1).upper()

    labels = []
    for match in LABEL_RE.finditer(normalized):
        label = match.group(1).upper()
        if label not in labels:
            labels.append(label)

    if len(labels) == 1:
        return "parsed", labels[0]
    if len(labels) > 1:
        return "invalid", None
    return "unparsable", None


def summarize_rows(sample_rows: list[dict[str, Any]], stats: dict[str, int]) -> dict[str, Any]:
    kept = len(sample_rows)
    seen = int(stats.get("seen", kept))
    dropped_as_wrong = max(seen - kept, 0)
    correct_count = int(sum(1 for row in sample_rows if bool(row["is_correct"])))
    parsed_count = int(sum(1 for row in sample_rows if row["parse_status"] == "parsed"))

    parse_status_counts = {
        "parsed": int(sum(1 for row in sample_rows if row["parse_status"] == "parsed")),
        "tie": int(sum(1 for row in sample_rows if row["parse_status"] == "tie")),
        "invalid": int(sum(1 for row in sample_rows if row["parse_status"] == "invalid")),
        "unparsable": int(sum(1 for row in sample_rows if row["parse_status"] == "unparsable")),
    }
    positive_label_counts = {
        "A": int(sum(1 for row in sample_rows if row["positive_label"] == "A")),
        "B": int(sum(1 for row in sample_rows if row["positive_label"] == "B")),
    }
    predicted_label_counts = {
        "A": int(sum(1 for row in sample_rows if row["predicted_label"] == "A")),
        "B": int(sum(1 for row in sample_rows if row["predicted_label"] == "B")),
    }

    by_language: dict[str, dict[str, int]] = {}
    for row in sample_rows:
        language = str(row.get("metadata", {}).get("language", "")).strip()
        if not language:
            continue
        bucket = by_language.setdefault(
            language,
            {"seen": 0, "correct": 0, "parsed": 0, "tie": 0, "invalid": 0, "unparsable": 0},
        )
        bucket["seen"] += 1
        bucket["correct"] += int(bool(row["is_correct"]))
        bucket[row["parse_status"]] += 1

    by_language_summary = {
        language: {
            "seen": values["seen"],
            "correct": values["correct"],
            "parsed": values["parsed"],
            "tie": values["tie"],
            "invalid": values["invalid"],
            "unparsable": values["unparsable"],
            "accuracy": float(values["correct"] / values["seen"]) if values["seen"] > 0 else None,
            "accuracy_parsed_only": float(values["correct"] / values["parsed"]) if values["parsed"] > 0 else None,
        }
        for language, values in sorted(by_language.items())
    }

    return {
        "n_seen": seen,
        "n_kept": kept,
        "n_dropped_as_wrong": dropped_as_wrong,
        "n_correct": correct_count,
        "accuracy": float(correct_count / seen) if seen > 0 else None,
        "accuracy_kept_only": float(correct_count / kept) if kept > 0 else None,
        "accuracy_parsed_only": float(correct_count / parsed_count) if parsed_count > 0 else None,
        "parse_status_counts": parse_status_counts,
        "positive_label_counts": positive_label_counts,
        "predicted_label_counts": predicted_label_counts,
        "by_language": by_language_summary,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.samples_output is None:
        args.samples_output = default_samples_output(args.output)
    if args.dropped_samples_output is None:
        args.dropped_samples_output = default_dropped_samples_path(args.output)

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        allow_network=args.allow_network,
    )
    pairs = load_pairs(args)
    rng = random.Random(args.seed)

    sample_rows: list[dict[str, Any]] = []
    dropped_samples: list[dict[str, Any]] = []
    stats = {"seen": 0, "kept": 0, "skipped_overlength": 0, "skipped_errors": 0}

    for pair in tqdm(pairs, desc="Running LLM judge baseline"):
        stats["seen"] += 1
        try:
            encoded, positive_label, labels = build_preference_prompt(
                tokenizer,
                pair,
                rng=rng,
                max_length=max_length_with_reserved_tokens(args.max_length, args.max_new_tokens),
                prepend_bos=args.prepend_bos,
                truncation_side=args.truncation_side,
                overflow_policy=args.overflow_policy,
                preference_question=args.preference_question,
            )
            completion = generate_completion(model, tokenizer, encoded, args=args)
            parse_status, predicted_label = parse_judge_completion(completion)
            sample_rows.append(
                {
                    "sample_id": pair.sample_id,
                    "task": pair.task,
                    "layer": None,
                    "positive_label": positive_label,
                    "negative_label": labels["negative_label"],
                    "A_side": labels["A_side"],
                    "B_side": labels["B_side"],
                    "predicted_label": predicted_label,
                    "parse_status": parse_status,
                    "is_correct": predicted_label == positive_label,
                    "completion": completion,
                    "prompt_text": encoded.text,
                    "prompt_token_length": int(encoded.input_ids.shape[1]),
                    "metadata": pair.metadata,
                }
            )
            stats["kept"] += 1
        except PromptTooLongError as exc:
            stats["skipped_overlength"] += 1
            dropped_samples.append(
                dropped_pair_record(
                    pair,
                    reason="prompt_too_long",
                    details=exc.details,
                    stage="llm_judge_generate",
                )
            )
        except Exception as exc:
            stats["skipped_errors"] += 1
            dropped_samples.append(
                dropped_pair_record(
                    pair,
                    reason="runtime_error",
                    details={"error": repr(exc)},
                    stage="llm_judge_generate",
                )
            )
            if not args.skip_errors:
                raise

    report = {
        "config": to_jsonable(vars(args)),
        "selection": {
            "data_files": [str(path) for path in args.data_files],
            "num_pairs": len(pairs),
            "limit": args.limit,
        },
        "stats": stats,
        "summary": summarize_rows(sample_rows, stats),
        "scoring_note": (
            "Generation-based LLM judge. "
            "Only parse_status='parsed' contributes a label prediction; "
            "tie/invalid/unparsable count as wrong in accuracy."
        ),
    }

    args.output.write_text(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.samples_output is not None:
        write_jsonl(args.samples_output, sample_rows)
    if args.dropped_samples_output is not None:
        write_jsonl(args.dropped_samples_output, dropped_samples)

    summary = report["summary"]
    accuracy_text = f"{summary['accuracy']:.4f}" if summary["accuracy"] is not None else "None"
    print(f"Saved LLM judge report to {args.output}")
    print(
        f"accuracy={accuracy_text} | "
        f"correct={summary['n_correct']} / seen={summary['n_seen']} | "
        f"parsed={summary['parse_status_counts']['parsed']} | "
        f"tie={summary['parse_status_counts']['tie']} | "
        f"invalid={summary['parse_status_counts']['invalid']} | "
        f"unparsable={summary['parse_status_counts']['unparsable']}"
    )


if __name__ == "__main__":
    main()
