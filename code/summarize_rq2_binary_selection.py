#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-dimension RQ2 binary-selection results into a compact summary "
            "with confidence intervals and paired comparisons against the LLM-judge baseline."
        )
    )
    parser.add_argument("--dimensions", nargs="+", required=True)
    parser.add_argument("--ours-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--judge-reports", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--ours-samples",
        type=Path,
        nargs="*",
        default=None,
        help="Optional sample JSONL paths for our method. If omitted, infer from each report path.",
    )
    parser.add_argument(
        "--judge-samples",
        type=Path,
        nargs="*",
        default=None,
        help="Optional sample JSONL paths for the LLM judge. If omitted, infer from each report path.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--ci-z", type=float, default=1.96)
    return parser.parse_args()


def normalize_optional_paths(
    values: list[Path] | None,
    *,
    expected: int,
    label: str,
) -> list[Path | None]:
    if values is None or len(values) == 0:
        return [None] * expected
    if len(values) != expected:
        raise ValueError(f"{label} must contain either 0 paths or exactly {expected} paths.")
    return list(values)


def infer_samples_path(report_path: Path) -> Path | None:
    candidate = report_path.parent.parent / "samples" / f"{report_path.stem}.samples.jsonl"
    return candidate if candidate.exists() else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "summary" not in payload:
        raise KeyError(f"Expected a top-level 'summary' field in {path}.")
    return payload["summary"]


def pct(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, 1)


def wilson_interval(successes: int, total: int, *, z_value: float) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p_hat = float(successes / total)
    z2 = z_value * z_value
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    half = z_value * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total) / denom
    return center - half, center + half


def paired_mean_ci(
    *,
    positive_count: int,
    negative_count: int,
    zero_count: int,
    z_value: float,
) -> tuple[float | None, float | None]:
    total = positive_count + negative_count + zero_count
    if total <= 0:
        return None, None
    mean = float((positive_count - negative_count) / total)
    if total == 1:
        return mean, mean

    sum_sq = (
        positive_count * ((1.0 - mean) ** 2)
        + negative_count * (((-1.0) - mean) ** 2)
        + zero_count * ((0.0 - mean) ** 2)
    )
    variance = sum_sq / (total - 1)
    se = math.sqrt(variance / total)
    delta = z_value * se
    return mean - delta, mean + delta


def mcnemar_exact_pvalue(ours_only_correct: int, judge_only_correct: int) -> float | None:
    discordant = ours_only_correct + judge_only_correct
    if discordant == 0:
        return 1.0

    smaller_tail = min(ours_only_correct, judge_only_correct)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
        for k in range(smaller_tail + 1)
    ]
    max_log = max(log_terms)
    tail_prob = math.exp(max_log) * sum(math.exp(value - max_log) for value in log_terms)
    return min(1.0, 2.0 * tail_prob)


def load_sample_outcomes(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    outcomes: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in outcomes:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}.")
        outcomes[sample_id] = {
            "sample_id": sample_id,
            "is_correct": bool(row.get("is_correct", False)),
            "predicted_label": row.get("predicted_label"),
        }
    return outcomes


def build_row(
    *,
    dimension: str,
    ours_report_path: Path,
    judge_report_path: Path,
    ours_samples_path: Path | None,
    judge_samples_path: Path | None,
    z_value: float,
) -> dict[str, Any]:
    ours_summary = load_summary(ours_report_path)
    judge_summary = load_summary(judge_report_path)

    ours_seen = int(ours_summary.get("n_seen", 0))
    judge_seen = int(judge_summary.get("n_seen", 0))
    if ours_seen != judge_seen:
        raise ValueError(
            f"Seen-count mismatch for {dimension}: ours has {ours_seen}, judge has {judge_seen}."
        )

    ours_correct = int(ours_summary.get("n_correct", 0))
    judge_correct = int(judge_summary.get("n_correct", 0))
    ours_accuracy = float(ours_summary["accuracy"]) if ours_summary.get("accuracy") is not None else None
    judge_accuracy = float(judge_summary["accuracy"]) if judge_summary.get("accuracy") is not None else None
    ours_ci_low, ours_ci_high = wilson_interval(ours_correct, ours_seen, z_value=z_value)
    judge_ci_low, judge_ci_high = wilson_interval(judge_correct, judge_seen, z_value=z_value)

    row: dict[str, Any] = {
        "dimension": dimension,
        "random_accuracy": 0.5,
        "n_seen": ours_seen,
        "ours_correct": ours_correct,
        "judge_correct": judge_correct,
        "ours_accuracy": ours_accuracy,
        "judge_accuracy": judge_accuracy,
        "ours_ci_low": ours_ci_low,
        "ours_ci_high": ours_ci_high,
        "judge_ci_low": judge_ci_low,
        "judge_ci_high": judge_ci_high,
        "delta_vs_judge": None if ours_accuracy is None or judge_accuracy is None else ours_accuracy - judge_accuracy,
        "random_accuracy_pct": pct(0.5),
        "ours_accuracy_pct": pct(ours_accuracy),
        "judge_accuracy_pct": pct(judge_accuracy),
        "ours_ci_low_pct": pct(ours_ci_low),
        "ours_ci_high_pct": pct(ours_ci_high),
        "judge_ci_low_pct": pct(judge_ci_low),
        "judge_ci_high_pct": pct(judge_ci_high),
        "delta_vs_judge_pct": pct(None if ours_accuracy is None or judge_accuracy is None else ours_accuracy - judge_accuracy),
        "ours_report": str(ours_report_path),
        "judge_report": str(judge_report_path),
        "ours_samples": str(ours_samples_path) if ours_samples_path is not None else None,
        "judge_samples": str(judge_samples_path) if judge_samples_path is not None else None,
    }

    if ours_samples_path is None or judge_samples_path is None:
        row.update(
            {
                "paired_n": None,
                "paired_coverage": None,
                "both_correct": None,
                "both_wrong": None,
                "ours_only_correct": None,
                "judge_only_correct": None,
                "delta_ci_low": None,
                "delta_ci_high": None,
                "delta_ci_low_pct": None,
                "delta_ci_high_pct": None,
                "paired_agreement": None,
                "paired_agreement_pct": None,
                "mcnemar_exact_p": None,
                "joint_missing_as_wrong": None,
            }
        )
        return row

    ours_outcomes = load_sample_outcomes(ours_samples_path)
    judge_outcomes = load_sample_outcomes(judge_samples_path)
    sample_ids = set(ours_outcomes) | set(judge_outcomes)

    if len(sample_ids) > ours_seen:
        raise ValueError(
            f"Sample union larger than n_seen for {dimension}: union={len(sample_ids)}, n_seen={ours_seen}."
        )

    both_correct = 0
    both_wrong = 0
    ours_only_correct = 0
    judge_only_correct = 0

    for sample_id in sample_ids:
        ours_correct_flag = bool(ours_outcomes.get(sample_id, {}).get("is_correct", False))
        judge_correct_flag = bool(judge_outcomes.get(sample_id, {}).get("is_correct", False))
        if ours_correct_flag and judge_correct_flag:
            both_correct += 1
        elif ours_correct_flag and not judge_correct_flag:
            ours_only_correct += 1
        elif judge_correct_flag and not ours_correct_flag:
            judge_only_correct += 1
        else:
            both_wrong += 1

    joint_missing_as_wrong = ours_seen - len(sample_ids)
    both_wrong += joint_missing_as_wrong

    paired_n = both_correct + both_wrong + ours_only_correct + judge_only_correct
    delta_ci_low, delta_ci_high = paired_mean_ci(
        positive_count=ours_only_correct,
        negative_count=judge_only_correct,
        zero_count=both_correct + both_wrong,
        z_value=z_value,
    )
    paired_agreement = (
        float((both_correct + both_wrong) / paired_n) if paired_n > 0 else None
    )
    mcnemar_p = mcnemar_exact_pvalue(ours_only_correct, judge_only_correct)

    row.update(
        {
            "paired_n": paired_n,
            "paired_coverage": float(paired_n / ours_seen) if ours_seen > 0 else None,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "ours_only_correct": ours_only_correct,
            "judge_only_correct": judge_only_correct,
            "delta_ci_low": delta_ci_low,
            "delta_ci_high": delta_ci_high,
            "delta_ci_low_pct": pct(delta_ci_low),
            "delta_ci_high_pct": pct(delta_ci_high),
            "paired_agreement": paired_agreement,
            "paired_agreement_pct": pct(paired_agreement),
            "mcnemar_exact_p": mcnemar_p,
            "joint_missing_as_wrong": joint_missing_as_wrong,
        }
    )
    return row


def build_overall_row(rows: list[dict[str, Any]], *, z_value: float) -> dict[str, Any]:
    total_seen = int(sum(int(row["n_seen"]) for row in rows))
    ours_correct = int(sum(int(row["ours_correct"]) for row in rows))
    judge_correct = int(sum(int(row["judge_correct"]) for row in rows))
    ours_accuracy = float(ours_correct / total_seen) if total_seen > 0 else None
    judge_accuracy = float(judge_correct / total_seen) if total_seen > 0 else None
    ours_ci_low, ours_ci_high = wilson_interval(ours_correct, total_seen, z_value=z_value)
    judge_ci_low, judge_ci_high = wilson_interval(judge_correct, total_seen, z_value=z_value)

    paired_rows = [row for row in rows if row.get("paired_n") is not None]
    if paired_rows:
        both_correct = int(sum(int(row["both_correct"]) for row in paired_rows))
        both_wrong = int(sum(int(row["both_wrong"]) for row in paired_rows))
        ours_only_correct = int(sum(int(row["ours_only_correct"]) for row in paired_rows))
        judge_only_correct = int(sum(int(row["judge_only_correct"]) for row in paired_rows))
        paired_n = both_correct + both_wrong + ours_only_correct + judge_only_correct
        delta_ci_low, delta_ci_high = paired_mean_ci(
            positive_count=ours_only_correct,
            negative_count=judge_only_correct,
            zero_count=both_correct + both_wrong,
            z_value=z_value,
        )
        paired_agreement = (
            float((both_correct + both_wrong) / paired_n) if paired_n > 0 else None
        )
        mcnemar_p = mcnemar_exact_pvalue(ours_only_correct, judge_only_correct)
        joint_missing_as_wrong = int(sum(int(row["joint_missing_as_wrong"]) for row in paired_rows))
        paired_coverage = float(paired_n / total_seen) if total_seen > 0 else None
    else:
        both_correct = both_wrong = ours_only_correct = judge_only_correct = 0
        paired_n = None
        delta_ci_low = delta_ci_high = None
        paired_agreement = None
        mcnemar_p = None
        joint_missing_as_wrong = None
        paired_coverage = None

    return {
        "dimension": "Overall",
        "random_accuracy": 0.5,
        "n_seen": total_seen,
        "ours_correct": ours_correct,
        "judge_correct": judge_correct,
        "ours_accuracy": ours_accuracy,
        "judge_accuracy": judge_accuracy,
        "ours_ci_low": ours_ci_low,
        "ours_ci_high": ours_ci_high,
        "judge_ci_low": judge_ci_low,
        "judge_ci_high": judge_ci_high,
        "delta_vs_judge": None if ours_accuracy is None or judge_accuracy is None else ours_accuracy - judge_accuracy,
        "random_accuracy_pct": pct(0.5),
        "ours_accuracy_pct": pct(ours_accuracy),
        "judge_accuracy_pct": pct(judge_accuracy),
        "ours_ci_low_pct": pct(ours_ci_low),
        "ours_ci_high_pct": pct(ours_ci_high),
        "judge_ci_low_pct": pct(judge_ci_low),
        "judge_ci_high_pct": pct(judge_ci_high),
        "delta_vs_judge_pct": pct(None if ours_accuracy is None or judge_accuracy is None else ours_accuracy - judge_accuracy),
        "paired_n": paired_n,
        "paired_coverage": paired_coverage,
        "both_correct": both_correct if paired_rows else None,
        "both_wrong": both_wrong if paired_rows else None,
        "ours_only_correct": ours_only_correct if paired_rows else None,
        "judge_only_correct": judge_only_correct if paired_rows else None,
        "delta_ci_low": delta_ci_low,
        "delta_ci_high": delta_ci_high,
        "delta_ci_low_pct": pct(delta_ci_low),
        "delta_ci_high_pct": pct(delta_ci_high),
        "paired_agreement": paired_agreement,
        "paired_agreement_pct": pct(paired_agreement),
        "mcnemar_exact_p": mcnemar_p,
        "joint_missing_as_wrong": joint_missing_as_wrong,
        "ours_report": None,
        "judge_report": None,
        "ours_samples": None,
        "judge_samples": None,
    }


def main() -> None:
    args = parse_args()
    expected = len(args.dimensions)
    if len(args.ours_reports) != expected or len(args.judge_reports) != expected:
        raise ValueError("--dimensions, --ours-reports, and --judge-reports must have the same length.")

    ours_samples = normalize_optional_paths(args.ours_samples, expected=expected, label="--ours-samples")
    judge_samples = normalize_optional_paths(args.judge_samples, expected=expected, label="--judge-samples")

    rows: list[dict[str, Any]] = []
    for index, dimension in enumerate(args.dimensions):
        ours_report = args.ours_reports[index]
        judge_report = args.judge_reports[index]
        ours_samples_path = ours_samples[index] or infer_samples_path(ours_report)
        judge_samples_path = judge_samples[index] or infer_samples_path(judge_report)
        rows.append(
            build_row(
                dimension=dimension,
                ours_report_path=ours_report,
                judge_report_path=judge_report,
                ours_samples_path=ours_samples_path,
                judge_samples_path=judge_samples_path,
                z_value=args.ci_z,
            )
        )

    overall = build_overall_row(rows, z_value=args.ci_z)
    payload = {
        "config": {
            "dimensions": args.dimensions,
            "ours_reports": [str(path) for path in args.ours_reports],
            "judge_reports": [str(path) for path in args.judge_reports],
            "ours_samples": [row["ours_samples"] for row in rows],
            "judge_samples": [row["judge_samples"] for row in rows],
            "ci_z": args.ci_z,
        },
        "rows": rows,
        "overall": overall,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_rows = [*rows, overall]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dimension",
                "n_seen",
                "random_accuracy",
                "random_accuracy_pct",
                "judge_correct",
                "judge_accuracy",
                "judge_accuracy_pct",
                "judge_ci_low",
                "judge_ci_high",
                "judge_ci_low_pct",
                "judge_ci_high_pct",
                "ours_correct",
                "ours_accuracy",
                "ours_accuracy_pct",
                "ours_ci_low",
                "ours_ci_high",
                "ours_ci_low_pct",
                "ours_ci_high_pct",
                "delta_vs_judge",
                "delta_vs_judge_pct",
                "delta_ci_low",
                "delta_ci_high",
                "delta_ci_low_pct",
                "delta_ci_high_pct",
                "paired_n",
                "paired_coverage",
                "paired_agreement",
                "paired_agreement_pct",
                "both_correct",
                "both_wrong",
                "ours_only_correct",
                "judge_only_correct",
                "joint_missing_as_wrong",
                "mcnemar_exact_p",
                "ours_report",
                "judge_report",
                "ours_samples",
                "judge_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved JSON summary to {args.output_json}")
    print(f"Saved CSV summary to {args.output_csv}")
    for row in csv_rows:
        message = (
            f"{row['dimension']}: "
            f"Ours={row['ours_accuracy_pct']} "
            f"[{row['ours_ci_low_pct']}, {row['ours_ci_high_pct']}] | "
            f"Judge={row['judge_accuracy_pct']} "
            f"[{row['judge_ci_low_pct']}, {row['judge_ci_high_pct']}] | "
            f"Delta={row['delta_vs_judge_pct']}"
        )
        if row["mcnemar_exact_p"] is not None:
            message += f" | McNemar p={row['mcnemar_exact_p']:.6g}"
        print(message)


if __name__ == "__main__":
    main()
