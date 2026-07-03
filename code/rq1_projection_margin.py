#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
    build_preference_prompt,
    default_dropped_samples_path,
    dropped_pair_record,
    l2_normalize,
    load_concept_pairs,
    load_model_and_tokenizer,
    max_length_with_reserved_tokens,
    num_hidden_states_for_model,
    parse_layer_spec,
    PromptTooLongError,
    single_token_id,
    to_jsonable,
    write_jsonl,
)


ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RQ1: fit a layer-wise quality direction from paired hidden states on one split, "
            "then evaluate projection margins on another split."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--fit-data-files", type=Path, nargs="+", default=None)
    parser.add_argument("--eval-data-files", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--vector-path",
        type=Path,
        default=None,
        help=(
            "Optional saved layer-direction artifact or answer_token_vectors artifact. "
            "If set, reuse its directions and skip fit extraction."
        ),
    )
    parser.add_argument(
        "--fit-output",
        type=Path,
        default=None,
        help="Optional path to save fitted layer directions to disk for later reuse.",
    )
    parser.add_argument("--task-field", default="")
    parser.add_argument("--positive-field", required=True)
    parser.add_argument("--negative-field", required=True)
    parser.add_argument("--concept-name", default="concept")
    parser.add_argument("--positive-name", default="positive")
    parser.add_argument("--negative-name", default="negative")
    parser.add_argument("--metadata-fields", nargs="+", default=[])
    parser.add_argument("--preference-question", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--samples-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--figure-output", type=Path, default=None)
    parser.add_argument("--layers", default="blocks")
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument(
        "--layer-batch-size",
        type=int,
        default=0,
        help="Optional layer chunk size. Use 0 to process all requested layers in one pass (default).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional prompt cap. By default the hook pipeline keeps the full prompt without truncation.",
    )
    parser.add_argument(
        "--truncation-side",
        choices=("right", "left"),
        default="left",
        help="Only used together with --max-length.",
    )
    parser.add_argument("--overflow-policy", choices=("truncate", "drop"), default="truncate")
    parser.add_argument("--fit-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false")
    parser.set_defaults(prepend_bos=True)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--unit-eps", type=float, default=1e-12)
    parser.add_argument("--ci-z", type=float, default=1.96)
    parser.add_argument("--font-family", default="Arial")
    parser.add_argument("--figure-width", type=float, default=8.6)
    parser.add_argument("--figure-height", type=float, default=4.8)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--dropped-samples-output", type=Path, default=None)
    return parser.parse_args()


def default_summary_output(output: Path) -> Path:
    return output.with_suffix(".summary.json")


def default_figure_output(output: Path) -> Path:
    return output.with_suffix(".png")


def default_fit_output(output: Path) -> Path:
    return output.with_suffix(".fit.pt")


def load_pairs(paths: list[Path], args: argparse.Namespace, *, limit: int | None):
    pairs = load_concept_pairs(
        paths,
        task_field=args.task_field,
        positive_field=args.positive_field,
        negative_field=args.negative_field,
        concept_name=args.concept_name,
        positive_name=args.positive_name,
        negative_name=args.negative_name,
        metadata_fields=args.metadata_fields,
    )
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def chunk_layers(layers: list[int], batch_size: int) -> list[list[int]]:
    if batch_size <= 0:
        return [layers]
    return [layers[i : i + batch_size] for i in range(0, len(layers), batch_size)]


def save_fit_artifact(
    output: Path,
    *,
    layers: list[int],
    directions: dict[int, np.ndarray],
    config: dict[str, Any],
    fit_stats: dict[str, int],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "kind": "rq1_fit_directions",
        "layers": [int(layer) for layer in layers],
        "directions": {
            int(layer): {
                "unit": torch.from_numpy(direction).float().cpu(),
                "hidden_size": int(direction.shape[0]),
            }
            for layer, direction in directions.items()
        },
        "fit_stats": fit_stats,
        "config": to_jsonable(config),
    }
    torch.save(artifact, output)


def load_directions_from_artifact(path: Path, layers: list[int]) -> tuple[dict[int, np.ndarray], dict[str, int], dict[int, int]]:
    artifact = torch.load(path, map_location="cpu")
    if artifact.get("kind") == "rq1_fit_directions":
        source = artifact.get("directions", {})
        fit_stats = artifact.get("fit_stats") or {"seen": 0, "kept": 0, "skipped_overlength": 0, "skipped_errors": 0}
        fit_pairs_by_layer = {int(layer): int(fit_stats.get("kept", 0)) for layer in layers}
        return (
            {
                int(layer): source[int(layer)]["unit"].float().cpu().numpy()
                for layer in layers
            },
            fit_stats,
            fit_pairs_by_layer,
        )
    if "estimators" in artifact and "answer_token_diff" in artifact["estimators"]:
        estimator = artifact["estimators"]["answer_token_diff"]
        fit_stats = artifact.get("stats") or {"seen": 0, "kept": 0, "skipped_overlength": 0, "skipped_errors": 0}
        fit_pairs_by_layer = {
            int(layer): int(estimator[int(layer)].get("n_pairs", fit_stats.get("kept", 0)))
            for layer in layers
        }
        return (
            {
                int(layer): estimator[int(layer)]["unit"].float().cpu().numpy()
                for layer in layers
            },
            fit_stats,
            fit_pairs_by_layer,
        )
    raise KeyError(f"Unsupported vector artifact at {path}; expected rq1_fit_directions or answer_token_diff estimators.")


def default_csv_output(output: Path) -> Path:
    return output.with_suffix(".csv")


def paired_effect_size(margins: np.ndarray) -> float | None:
    if margins.size < 2:
        return None
    std = float(margins.std(ddof=1))
    if std == 0.0:
        return None
    return float(margins.mean() / std)


def mean_ci(margins: np.ndarray, z_value: float) -> tuple[float, float] | tuple[None, None]:
    if margins.size == 0:
        return None, None
    if margins.size == 1:
        value = float(margins[0])
        return value, value
    se = float(margins.std(ddof=1) / np.sqrt(margins.size))
    delta = z_value * se
    mean = float(margins.mean())
    return mean - delta, mean + delta


def save_figure(fig, output: Path, *, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass


def plot_layer_profile(report: dict[str, Any], output: Path, args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping RQ1 figure because matplotlib is unavailable: {exc}")
        return

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [args.font_family, "Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
        }
    )

    layers = [int(item["layer"]) for item in report["layers"]]
    mean_margin = [float(item["mean_margin"]) for item in report["layers"]]
    lower = [float(item["ci_low"]) for item in report["layers"]]
    upper = [float(item["ci_high"]) for item in report["layers"]]
    effect = [np.nan if item["cohens_d"] is None else float(item["cohens_d"]) for item in report["layers"]]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(args.figure_width, args.figure_height),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )

    ax_top.plot(layers, mean_margin, color="#2b67b8", linewidth=2.2)
    ax_top.fill_between(layers, lower, upper, color="#9dbce5", alpha=0.35)
    ax_top.axhline(0.0, color="#9aa1a9", linewidth=1.0, linestyle="--")
    ax_top.set_ylabel("Mean Margin", fontsize=11)
    ax_top.set_title("Layer-wise Projection Margin Profile", fontsize=13, pad=8)
    ax_top.grid(axis="y", alpha=0.18, linewidth=0.8, color="#b8c0c8")

    ax_bottom.plot(layers, effect, color="#b85c38", linewidth=2.0)
    ax_bottom.axhline(0.0, color="#9aa1a9", linewidth=1.0, linestyle="--")
    ax_bottom.set_ylabel("Cohen's d", fontsize=11)
    ax_bottom.set_xlabel("Layer", fontsize=11)
    ax_bottom.grid(axis="y", alpha=0.18, linewidth=0.8, color="#b8c0c8")
    ax_bottom.set_xticks(layers)
    save_figure(fig, output, dpi=args.dpi)


def extract_pair_outputs(
    pairs,
    *,
    model,
    tokenizer,
    layers: list[int],
    label_ids: dict[str, int],
    args: argparse.Namespace,
    stage_name: str,
    rng_seed: int,
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    rng = random.Random(rng_seed)
    positive_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    negative_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    sample_rows: list[dict[str, Any]] = []
    dropped_samples: list[dict[str, Any]] = []
    stats = {"seen": 0, "kept": 0, "skipped_overlength": 0, "skipped_errors": 0}

    with HookedLastTokenExtractor(model, layers) as extractor:
        for pair in tqdm(pairs, desc=f"Extracting {stage_name} hidden states"):
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
                negative_label = "B" if positive_label == "A" else "A"
                positive_input_ids, positive_attention_mask = append_answer_label(
                    encoded,
                    label_ids[positive_label],
                    max_length=args.max_length,
                    truncation_side=args.truncation_side,
                    overflow_policy=args.overflow_policy,
                )
                negative_input_ids, negative_attention_mask = append_answer_label(
                    encoded,
                    label_ids[negative_label],
                    max_length=args.max_length,
                    truncation_side=args.truncation_side,
                    overflow_policy=args.overflow_policy,
                )
                positive_outputs = forward_last_token(model, extractor, positive_input_ids, positive_attention_mask)
                negative_outputs = forward_last_token(model, extractor, negative_input_ids, negative_attention_mask)
                for layer in layers:
                    positive_by_layer[layer].append(positive_outputs[layer].float().cpu().numpy())
                    negative_by_layer[layer].append(negative_outputs[layer].float().cpu().numpy())
                sample_rows.append(
                    {
                        "sample_id": pair.sample_id,
                        "source": pair.source,
                        "row_index": pair.row_index,
                        "metadata": pair.metadata,
                        "positive_label": positive_label,
                        "negative_label": negative_label,
                        "A_side": labels["A_side"],
                        "B_side": labels["B_side"],
                        "seq_len_with_answer": int(positive_input_ids.shape[1]),
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
                        stage=stage_name,
                    )
                )
            except Exception:
                stats["skipped_errors"] += 1
                if not args.skip_errors:
                    raise
    return positive_by_layer, negative_by_layer, sample_rows, stats, dropped_samples


def main() -> None:
    args = parse_args()
    if args.vector_path is None and not args.fit_data_files:
        raise ValueError("Pass either --fit-data-files or --vector-path.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = args.csv_output or default_csv_output(args.output)
    summary_output = args.summary_output or default_summary_output(args.output)
    figure_output = args.figure_output or default_figure_output(args.output)
    fit_output = args.fit_output or (None if args.vector_path is not None else default_fit_output(args.output))

    eval_pairs = load_pairs(args.eval_data_files, args, limit=args.eval_limit)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        allow_network=args.allow_network,
    )
    layers = parse_layer_spec(
        args.layers,
        num_hidden_states=num_hidden_states_for_model(model),
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        layer_stride=args.layer_stride,
    )
    label_ids = {"A": single_token_id(tokenizer, "A"), "B": single_token_id(tokenizer, "B")}
    directions: dict[int, np.ndarray]
    fit_stats: dict[str, int]
    fit_pairs_by_layer: dict[int, int]
    fit_dropped: list[dict[str, Any]]
    if args.vector_path is not None:
        print(f"Loading fitted directions from {args.vector_path}")
        directions, fit_stats, fit_pairs_by_layer = load_directions_from_artifact(args.vector_path, layers)
        fit_dropped = []
    else:
        fit_pairs = load_pairs(args.fit_data_files, args, limit=args.fit_limit)
        fit_pos: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        fit_neg: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        fit_stats = {"seen": 0, "kept": 0, "skipped_overlength": 0, "skipped_errors": 0}
        fit_dropped = []
        for chunk_idx, layer_chunk in enumerate(chunk_layers(layers, args.layer_batch_size), start=1):
            print(f"Fitting layers {layer_chunk}" if len(layer_chunk) == len(layers) else f"Fitting layer chunk {chunk_idx}: {layer_chunk}")
            chunk_pos, chunk_neg, chunk_rows, chunk_stats, chunk_dropped = extract_pair_outputs(
                fit_pairs,
                model=model,
                tokenizer=tokenizer,
                layers=layer_chunk,
                label_ids=label_ids,
                args=args,
                stage_name="rq1_fit",
                rng_seed=args.seed,
            )
            for layer in layer_chunk:
                fit_pos[layer] = chunk_pos[layer]
                fit_neg[layer] = chunk_neg[layer]
            if chunk_idx == 1:
                fit_stats = chunk_stats
            fit_dropped.extend(chunk_dropped)
        if fit_stats["kept"] == 0:
            raise RuntimeError(f"Need non-empty fit split. Fit={fit_stats}")
        directions = {}
        for layer in layers:
            fit_pos_matrix = np.stack(fit_pos[layer])
            fit_neg_matrix = np.stack(fit_neg[layer])
            fit_diffs = fit_pos_matrix - fit_neg_matrix
            raw_direction = torch.from_numpy(fit_diffs.mean(axis=0)).float()
            directions[layer] = l2_normalize(raw_direction, eps=args.unit_eps).cpu().numpy()
        fit_pairs_by_layer = {int(layer): int(fit_stats["kept"]) for layer in layers}
        if fit_output is not None:
            save_fit_artifact(
                fit_output,
                layers=layers,
                directions=directions,
                config=vars(args),
                fit_stats=fit_stats,
            )
            print(f"Saved fitted directions to {fit_output}")

    eval_pos: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    eval_neg: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    eval_rows: list[dict[str, Any]] | None = None
    eval_stats: dict[str, int] = {"seen": 0, "kept": 0, "skipped_overlength": 0, "skipped_errors": 0}
    eval_dropped: list[dict[str, Any]] = []
    for chunk_idx, layer_chunk in enumerate(chunk_layers(layers, args.layer_batch_size), start=1):
        print(f"Evaluating layers {layer_chunk}" if len(layer_chunk) == len(layers) else f"Evaluating layer chunk {chunk_idx}: {layer_chunk}")
        chunk_pos, chunk_neg, chunk_rows, chunk_stats, chunk_dropped = extract_pair_outputs(
            eval_pairs,
            model=model,
            tokenizer=tokenizer,
            layers=layer_chunk,
            label_ids=label_ids,
            args=args,
            stage_name="rq1_eval",
            rng_seed=args.seed + 1,
        )
        for layer in layer_chunk:
            eval_pos[layer] = chunk_pos[layer]
            eval_neg[layer] = chunk_neg[layer]
        if chunk_idx == 1:
            eval_rows = chunk_rows
            eval_stats = chunk_stats
        eval_dropped.extend(chunk_dropped)

    if eval_stats["kept"] == 0 or eval_rows is None:
        raise RuntimeError(f"Need non-empty eval split. Eval={eval_stats}")

    layer_rows: list[dict[str, Any]] = []
    sample_output_rows: list[dict[str, Any]] = []
    for layer in layers:
        eval_pos_matrix = np.stack(eval_pos[layer])
        eval_neg_matrix = np.stack(eval_neg[layer])
        unit_direction = directions[layer]
        pos_scores = eval_pos_matrix @ unit_direction
        neg_scores = eval_neg_matrix @ unit_direction
        margins = pos_scores - neg_scores
        mean_margin = float(margins.mean())
        ci_low, ci_high = mean_ci(margins, args.ci_z)
        cohens_d = paired_effect_size(margins)
        sign_rate = float(np.mean(margins > 0))

        layer_rows.append(
            {
                "layer": int(layer),
                "fit_pairs": int(fit_pairs_by_layer.get(int(layer), fit_stats.get("kept", 0))),
                "eval_pairs": int(margins.shape[0]),
                "mean_margin": mean_margin,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "cohens_d": cohens_d,
                "margin_std": float(margins.std(ddof=1)) if margins.size > 1 else 0.0,
                "positive_rate": sign_rate,
                "direction_norm": float(np.linalg.norm(unit_direction)),
            }
        )

        if args.samples_output is not None:
            for meta, pos_score, neg_score, margin in zip(eval_rows, pos_scores, neg_scores, margins):
                sample_output_rows.append(
                    {
                        **meta,
                        "layer": int(layer),
                        "positive_score": float(pos_score),
                        "negative_score": float(neg_score),
                        "margin": float(margin),
                    }
                )

    best_layer = max(layer_rows, key=lambda item: float(item["mean_margin"]))
    report = {
        "config": to_jsonable(vars(args)),
        "fit_stats": fit_stats,
        "eval_stats": eval_stats,
        "fit_source": "vector_path" if args.vector_path is not None else "fit_data_files",
        "fit_output": str(fit_output) if fit_output is not None else None,
        "layers": layer_rows,
        "best_layer_by_mean_margin": best_layer,
    }

    args.output.write_text(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer",
                "fit_pairs",
                "eval_pairs",
                "mean_margin",
                "ci_low",
                "ci_high",
                "cohens_d",
                "margin_std",
                "positive_rate",
                "direction_norm",
            ],
        )
        writer.writeheader()
        for row in layer_rows:
            writer.writerow(row)

    summary_payload = {
        "concept_name": args.concept_name,
        "best_layer": int(best_layer["layer"]),
        "mean_margin": float(best_layer["mean_margin"]),
        "ci_low": float(best_layer["ci_low"]),
        "ci_high": float(best_layer["ci_high"]),
        "cohens_d": None if best_layer["cohens_d"] is None else float(best_layer["cohens_d"]),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(to_jsonable(summary_payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.samples_output is not None:
        write_jsonl(args.samples_output, [to_jsonable(row) for row in sample_output_rows])

    all_dropped = [*fit_dropped, *eval_dropped]
    dropped_output = args.dropped_samples_output or default_dropped_samples_path(args.output)
    if all_dropped:
        write_jsonl(dropped_output, all_dropped)
        print(f"Saved dropped overlength samples to {dropped_output}")

    plot_layer_profile(report, figure_output, args)
    print(f"Saved RQ1 report to {args.output}")
    print(f"Saved layer metrics CSV to {csv_output}")
    print(f"Saved best-layer summary to {summary_output}")
    print(f"Saved layer profile figure to {figure_output}")
    print(
        f"Best layer: {best_layer['layer']} | mean_margin={best_layer['mean_margin']:.6f} | "
        f"95% CI=[{best_layer['ci_low']:.6f}, {best_layer['ci_high']:.6f}] | "
        f"Cohen's d={best_layer['cohens_d']}"
    )


if __name__ == "__main__":
    main()
