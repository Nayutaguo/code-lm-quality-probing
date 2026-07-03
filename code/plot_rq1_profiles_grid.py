#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple RQ1 projection-margin reports into a paper-style multi-panel figure."
    )
    parser.add_argument("--report-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--x-label", default="Layer")
    parser.add_argument("--margin-label", default="Mean Margin")
    parser.add_argument("--effect-label", default="Cohen's d")
    parser.add_argument("--figure-width", type=float, default=7.1)
    parser.add_argument("--figure-height", type=float, default=6.1)
    parser.add_argument("--font-family", default="Arial")
    parser.add_argument("--title-size", type=float, default=11.5)
    parser.add_argument("--label-size", type=float, default=10.5)
    parser.add_argument("--tick-size", type=float, default=9.2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--x-tick-step", type=int, default=4)
    parser.add_argument("--columns", type=int, default=2)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path)


def default_label(path: Path) -> str:
    stem = path.stem.lower()
    if "safety" in stem:
        return "Security"
    if "efficiency" in stem:
        return "Efficiency"
    if "correctness" in stem or "codeflaws" in stem:
        return "Correctness"
    if "readability" in stem or "test_smells" in stem:
        return "Readability"
    return path.stem.replace("_", " ").title()


def load_report(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = payload["layers"]
    return {
        "label": label,
        "path": str(path),
        "layers": [int(row["layer"]) for row in layers],
        "mean_margin": [float(row["mean_margin"]) for row in layers],
        "ci_low": [float(row["ci_low"]) for row in layers],
        "ci_high": [float(row["ci_high"]) for row in layers],
        "cohens_d": [
            None if row["cohens_d"] is None else float(row["cohens_d"])
            for row in layers
        ],
    }


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


def main() -> None:
    args = parse_args()

    if args.labels is not None and len(args.labels) not in {0, len(args.report_inputs)}:
        raise ValueError("Pass either no labels or one label per report input.")

    report_paths = [resolve_path(path) for path in args.report_inputs]
    labels = args.labels or [default_label(path) for path in report_paths]
    if len(labels) == 0:
        labels = [default_label(path) for path in report_paths]

    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to plot RQ1 profiles: {exc}") from exc

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [args.font_family, "Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    reports = [
        load_report(path, label)
        for path, label in tqdm(
            list(zip(report_paths, labels)),
            desc="Loading RQ1 reports",
        )
    ]

    num_reports = len(reports)
    num_cols = max(1, args.columns)
    num_rows = (num_reports + num_cols - 1) // num_cols

    fig = plt.figure(figsize=(args.figure_width, args.figure_height), constrained_layout=False)
    outer = GridSpec(
        num_rows,
        num_cols,
        figure=fig,
        left=0.07,
        right=0.985,
        bottom=0.07,
        top=0.95 if args.title else 0.975,
        wspace=0.16,
        hspace=0.28,
    )

    margin_line_color = "#2b67b8"
    margin_fill_color = "#9dbce5"
    effect_line_color = "#c85d2e"
    zero_color = "#9aa1a9"
    grid_color = "#d7dfe8"

    for idx, report in enumerate(reports):
        row = idx // num_cols
        col = idx % num_cols
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.06)
        ax_top = fig.add_subplot(inner[0, 0])
        ax_bottom = fig.add_subplot(inner[1, 0], sharex=ax_top)

        layers = report["layers"]
        mean_margin = report["mean_margin"]
        ci_low = report["ci_low"]
        ci_high = report["ci_high"]
        cohens_d = [float("nan") if value is None else value for value in report["cohens_d"]]

        ax_top.plot(layers, mean_margin, color=margin_line_color, linewidth=2.0)
        ax_top.fill_between(layers, ci_low, ci_high, color=margin_fill_color, alpha=0.35)
        ax_top.axhline(0.0, color=zero_color, linewidth=1.0, linestyle="--")
        ax_top.grid(axis="y", alpha=0.55, linewidth=0.8, color=grid_color)
        ax_top.tick_params(axis="both", labelsize=args.tick_size)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_ylabel(args.margin_label, fontsize=args.label_size)
        ax_top.set_title(str(report["label"]), fontsize=args.title_size, pad=6)

        ax_bottom.plot(layers, cohens_d, color=effect_line_color, linewidth=1.8)
        ax_bottom.axhline(0.0, color=zero_color, linewidth=1.0, linestyle="--")
        ax_bottom.grid(axis="y", alpha=0.55, linewidth=0.8, color=grid_color)
        ax_bottom.tick_params(axis="both", labelsize=args.tick_size)
        ax_bottom.set_ylabel(args.effect_label, fontsize=args.label_size)
        if row == num_rows - 1:
            ax_bottom.set_xlabel(args.x_label, fontsize=args.label_size)
        step = max(1, args.x_tick_step)
        ax_bottom.set_xticks(layers[::step])

    total_slots = num_rows * num_cols
    for idx in range(num_reports, total_slots):
        row = idx // num_cols
        col = idx % num_cols
        ax = fig.add_subplot(outer[row, col])
        ax.axis("off")

    if args.title:
        fig.suptitle(args.title, fontsize=args.title_size + 1.5, y=0.995)

    save_figure(fig, resolve_path(args.output), dpi=args.dpi)
    print(f"Saved RQ1 profile grid figure to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
