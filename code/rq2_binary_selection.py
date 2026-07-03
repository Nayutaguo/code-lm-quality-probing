#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_answer_token_vectors import (  # noqa: E402
    HookedLastTokenExtractor,
    append_answer_label,
    forward_last_token,
)
from steering_utils import (  # noqa: E402
    PromptTooLongError,
    build_preference_prompt,
    default_dropped_samples_path,
    dropped_pair_record,
    load_concept_pairs,
    load_model_and_tokenizer,
    max_length_with_reserved_tokens,
    num_hidden_states_for_model,
    single_token_id,
    to_jsonable,
    write_jsonl,
)


ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RQ2 binary selection with the original A/B pair prompt and answer-label hidden states. "
            "For each held-out pair, compute score_A and score_B from the chosen layer direction, "
            "then select the higher-scoring side."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vector-path", type=Path, required=True)
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
    parser.add_argument(
        "--estimator",
        default="answer_token_diff",
        help="Estimator name when --vector-path points to a contrast-vector artifact.",
    )
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional prompt cap. Uses the same reserved-token handling as answer-token vector building.",
    )
    parser.add_argument(
        "--truncation-side",
        choices=("right", "left"),
        default="left",
        help="Only used together with --max-length.",
    )
    parser.add_argument("--overflow-policy", choices=("truncate", "drop"), default="truncate")
    parser.add_argument("--dropped-samples-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false")
    parser.set_defaults(prepend_bos=True)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


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


def load_vector(path: Path, *, estimator: str, layer: int) -> tuple[torch.Tensor, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu")

    if artifact.get("kind") == "rq1_fit_directions":
        directions = artifact.get("directions", {})
        if layer not in directions:
            available = sorted(int(item) for item in directions)
            raise KeyError(f"Layer {layer} not found in {path}. Available layers: {available}")
        vector = directions[layer]["unit"].float().cpu()
        return vector, {"artifact_kind": "rq1_fit_directions", "estimator": "rq1_fit_directions", "layer": layer}

    if "estimators" in artifact:
        estimators = artifact["estimators"]
        if estimator not in estimators:
            raise KeyError(f"Estimator {estimator!r} not found in {path}. Available: {list(estimators)}")
        layer_map = estimators[estimator]
        if layer not in layer_map:
            available = sorted(int(item) for item in layer_map)
            raise KeyError(f"Layer {layer} not found for estimator {estimator!r}. Available layers: {available}")
        vector = layer_map[layer]["unit"].float().cpu()
        return vector, {"artifact_kind": "contrast_vector", "estimator": estimator, "layer": layer}

    raise KeyError(f"Unsupported vector artifact at {path}.")


def summarize_rows(sample_rows: list[dict[str, Any]], stats: dict[str, int]) -> dict[str, Any]:
    kept = len(sample_rows)
    seen = int(stats.get("seen", kept))
    correct_count = int(sum(1 for row in sample_rows if bool(row["is_correct"])))
    tie_count = int(sum(1 for row in sample_rows if bool(row["is_tie"])))
    dropped_as_wrong = max(seen - kept, 0)

    by_language: dict[str, dict[str, int]] = {}
    for row in sample_rows:
        language = str(row.get("metadata", {}).get("language", "")).strip()
        if not language:
            continue
        bucket = by_language.setdefault(language, {"seen": 0, "correct": 0, "ties": 0})
        bucket["seen"] += 1
        bucket["correct"] += int(bool(row["is_correct"]))
        bucket["ties"] += int(bool(row["is_tie"]))

    by_language_summary = {
        language: {
            "seen": values["seen"],
            "correct": values["correct"],
            "ties": values["ties"],
            "accuracy": float(values["correct"] / values["seen"]) if values["seen"] > 0 else None,
        }
        for language, values in sorted(by_language.items())
    }

    return {
        "layer": int(sample_rows[0]["layer"]) if sample_rows else None,
        "n_seen": seen,
        "n_kept": kept,
        "n_dropped_as_wrong": dropped_as_wrong,
        "n_correct": correct_count,
        "n_ties": tie_count,
        "accuracy": float(correct_count / seen) if seen > 0 else None,
        "accuracy_kept_only": float(correct_count / kept) if kept > 0 else None,
        "by_language": by_language_summary,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        allow_network=args.allow_network,
    )
    max_layer = num_hidden_states_for_model(model) - 1
    if args.layer < 1 or args.layer > max_layer:
        raise ValueError(f"--layer must be in 1..{max_layer} for hook extraction; got {args.layer}.")

    vector, vector_info = load_vector(args.vector_path, estimator=args.estimator, layer=args.layer)
    pairs = load_pairs(args)
    label_ids = {"A": single_token_id(tokenizer, "A"), "B": single_token_id(tokenizer, "B")}
    rng = random.Random(args.seed)

    sample_rows: list[dict[str, Any]] = []
    dropped_samples: list[dict[str, Any]] = []
    stats = {"seen": 0, "kept": 0, "skipped_overlength": 0, "skipped_errors": 0}

    with HookedLastTokenExtractor(model, [args.layer]) as extractor:
        for pair in tqdm(pairs, desc="Running RQ2 binary selection"):
            stats["seen"] += 1
            try:
                encoded, positive_label, labels = build_preference_prompt(
                    tokenizer,
                    pair,
                    rng=rng,
                    max_length=max_length_with_reserved_tokens(args.max_length, 1),
                    prepend_bos=args.prepend_bos,
                    truncation_side=args.truncation_side,
                    overflow_policy=args.overflow_policy,
                    preference_question=args.preference_question,
                )
                input_ids_a, attention_mask_a = append_answer_label(
                    encoded,
                    label_ids["A"],
                    max_length=args.max_length,
                    truncation_side=args.truncation_side,
                    overflow_policy=args.overflow_policy,
                )
                input_ids_b, attention_mask_b = append_answer_label(
                    encoded,
                    label_ids["B"],
                    max_length=args.max_length,
                    truncation_side=args.truncation_side,
                    overflow_policy=args.overflow_policy,
                )

                hidden_a = forward_last_token(model, extractor, input_ids_a, attention_mask_a)[args.layer]
                hidden_b = forward_last_token(model, extractor, input_ids_b, attention_mask_b)[args.layer]
                score_a = float(torch.dot(hidden_a.float(), vector.float()).item())
                score_b = float(torch.dot(hidden_b.float(), vector.float()).item())
                if score_a > score_b:
                    predicted_label = "A"
                elif score_b > score_a:
                    predicted_label = "B"
                else:
                    predicted_label = None

                is_tie = predicted_label is None
                is_correct = predicted_label == positive_label
                predicted_side = None if predicted_label is None else labels[f"{predicted_label}_side"]

                sample_rows.append(
                    {
                        "sample_id": pair.sample_id,
                        "source": pair.source,
                        "row_index": pair.row_index,
                        "metadata": pair.metadata,
                        "layer": int(args.layer),
                        "positive_label": positive_label,
                        "negative_label": "B" if positive_label == "A" else "A",
                        "A_side": labels["A_side"],
                        "B_side": labels["B_side"],
                        "score_A": score_a,
                        "score_B": score_b,
                        "score_margin_A_minus_B": score_a - score_b,
                        "predicted_label": predicted_label,
                        "predicted_side": predicted_side,
                        "is_tie": is_tie,
                        "is_correct": is_correct,
                        "seq_len_with_answer": int(input_ids_a.shape[1]),
                    }
                )
                stats["kept"] += 1
            except PromptTooLongError as exc:
                stats["skipped_overlength"] += 1
                dropped_samples.append(
                    dropped_pair_record(
                        pair,
                        reason="overlength",
                        details=exc.details,
                        stage="rq2_binary_selection",
                    )
                )
            except Exception:
                stats["skipped_errors"] += 1
                if not args.skip_errors:
                    raise

    if stats["seen"] == 0:
        raise RuntimeError("No pairs were loaded for RQ2 binary selection.")

    summary = summarize_rows(sample_rows, stats)
    report = {
        "config": to_jsonable(vars(args)),
        "vector_info": to_jsonable(vector_info),
        "stats": stats,
        "summary": summary,
    }
    args.output.write_text(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.samples_output is not None:
        write_jsonl(args.samples_output, [to_jsonable(row) for row in sample_rows])

    dropped_output = args.dropped_samples_output or default_dropped_samples_path(args.output)
    if dropped_samples:
        write_jsonl(dropped_output, dropped_samples)
        print(f"Saved dropped samples to {dropped_output}")

    print(f"Saved RQ2 report to {args.output}")
    if args.samples_output is not None:
        print(f"Saved sample scores to {args.samples_output}")
    print(
        f"Layer {args.layer} | accuracy={summary['accuracy']} | "
        f"correct={summary['n_correct']} / seen={summary['n_seen']} | ties={summary['n_ties']}"
    )


if __name__ == "__main__":
    main()
