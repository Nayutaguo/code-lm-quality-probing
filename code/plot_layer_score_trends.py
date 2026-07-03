#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm

from layer_score_plugins import rank_scored_layers


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot layer-wise score trends stored inside one or more steering-vector artifacts."
    )
    parser.add_argument(
        "--vector-paths",
        type=Path,
        nargs="+",
        default=None,
        help="One or more steering-vector artifacts to visualize.",
    )
    parser.add_argument(
        "--json-input",
        type=Path,
        default=None,
        help="Optional exported JSON payload from a previous plot-layer-score run.",
    )
    parser.add_argument(
        "--json-inputs",
        type=Path,
        nargs="+",
        default=None,
        help="Optional list of exported JSON payloads to combine in a shared multi-panel figure.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels, one per --vector-paths entry. Defaults to artifact stem.",
    )
    parser.add_argument(
        "--estimators",
        nargs="*",
        default=None,
        help="Optional estimator names. Pass one value to reuse for every artifact, or one per artifact.",
    )
    parser.add_argument("--plugin", default="layernavigator")
    parser.add_argument("--metric", default="s_score")
    parser.add_argument(
        "--style",
        choices=("line", "decomposition", "bars", "bars-grid"),
        default="line",
        help=(
            "'line' draws the main layer-wise trend. "
            "'decomposition' adds a lower panel for d_score / c_score. "
            "'bars' renders only the stacked d_score / c_score panel for paper-ready figures. "
            "'bars-grid' combines multiple stacked-bar panels with one shared legend."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--x-label", default="Layer")
    parser.add_argument("--y-label", default=None)
    parser.add_argument("--figure-width", type=float, default=9.5)
    parser.add_argument("--figure-height", type=float, default=5.8)
    parser.add_argument(
        "--top-k-annotate",
        type=int,
        default=0,
        help="Annotate the top-k highest-scoring layers for each curve. Use 0 to disable.",
    )
    parser.add_argument(
        "--font-family",
        default="Arial",
        help="Preferred font family for the figure.",
    )
    parser.add_argument(
        "--legend-title",
        default=None,
        help="Optional legend title. Defaults to no legend title for a cleaner paper-style figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for raster outputs.",
    )
    parser.add_argument(
        "--x-tick-step",
        type=int,
        default=1,
        help="Show every Nth layer label on the x-axis.",
    )
    parser.add_argument("--title-size", type=float, default=12.5)
    parser.add_argument("--label-size", type=float, default=11.0)
    parser.add_argument("--tick-size", type=float, default=10.0)
    parser.add_argument("--legend-size", type=float, default=9.5)
    parser.add_argument("--bar-width", type=float, default=0.78)
    parser.add_argument("--grid-columns", type=int, default=2)
    return parser.parse_args()


def normalize_estimators(num_paths: int, estimators: list[str] | None) -> list[str]:
    if not estimators:
        return ["answer_token_diff"] * num_paths
    if len(estimators) == 1:
        return estimators * num_paths
    if len(estimators) != num_paths:
        raise ValueError("Pass either one estimator or one estimator per vector path.")
    return estimators


def normalize_labels(paths: list[Path], labels: list[str] | None) -> list[str]:
    if not labels:
        return [default_short_label(path) for path in paths]
    if len(labels) != len(paths):
        raise ValueError("Pass either no labels or one label per vector path.")
    return [prettify_label(label) for label in labels]


def default_short_label(path: Path) -> str:
    stem = path.stem.lower()
    if "efficodebench" in stem and "cpp" in stem:
        return "Efficiency C++"
    if "efficodebench" in stem:
        return "Efficiency"
    if "code_quality" in stem:
        return "Readability"
    if "marv" in stem:
        return "MaRV"
    if "codeflaws" in stem:
        return "Correctness"
    if "primevul" in stem:
        return "PrimeVul Safety"
    if "safety" in stem:
        return "SafeCoder Safety"
    return prettify_label(path.stem)


def prettify_label(text: str) -> str:
    text = re.sub(r"_+", " ", text.strip())
    text = re.sub(r"\s+", " ", text)
    replacements = {
        "cpp": "C++",
        "primevul": "PrimeVul",
        "marv": "MaRV",
        "safecoder": "SafeCoder",
        "efficodebench": "EffiCodeBench",
        "codeflaws": "Codeflaws",
    }
    parts = []
    for token in text.split(" "):
        lower = token.lower()
        if lower in replacements:
            parts.append(replacements[lower])
        elif lower in {"llama", "llama-3.1-8b"}:
            parts.append(token)
        else:
            parts.append(token.capitalize())
    return " ".join(parts)


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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path)


def configure_matplotlib(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to plot layer score trends: {exc}") from exc

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


def set_layer_ticks(ax, layers: list[int], args: argparse.Namespace) -> None:
    if not layers:
        return
    step = max(1, args.x_tick_step)
    ax.set_xticks(layers[::step])


def load_series_from_json(json_input: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    payload = json.loads(json_input.read_text(encoding="utf-8"))
    series_list = list(payload.get("series", []))
    if not series_list:
        raise ValueError(f"No 'series' entries found in {json_input}")
    return series_list, payload


def load_series_from_json_inputs(json_inputs: list[Path]) -> tuple[list[dict[str, object]], dict[str, object]]:
    combined: list[dict[str, object]] = []
    payload_plugin: str | None = None
    payload_metric: str | None = None
    for path in json_inputs:
        series_list, payload = load_series_from_json(path)
        combined.extend(series_list)
        if payload_plugin is None:
            payload_plugin = str(payload.get("plugin", ""))
        if payload_metric is None:
            payload_metric = str(payload.get("metric", ""))
    return combined, {"plugin": payload_plugin, "metric": payload_metric}


def load_series(
    vector_path: Path,
    *,
    label: str,
    estimator: str,
    plugin: str,
    metric: str,
) -> dict[str, object]:
    artifact = torch.load(vector_path, map_location="cpu")
    ranked = rank_scored_layers(
        artifact=artifact,
        estimator=estimator,
        plugin_name=plugin,
        metric=metric,
    )
    by_layer = sorted(ranked, key=lambda item: item[0])
    artifact = torch.load(vector_path, map_location="cpu")
    score_details = []
    for layer, score in by_layer:
        plugin_scores = artifact["estimators"][estimator][layer].get("scores", {}).get(plugin, {})
        score_details.append(
            {
                "layer": int(layer),
                "score": float(score),
                "s_score": float(plugin_scores.get("s_score")) if plugin_scores.get("s_score") is not None else None,
                "d_score": float(plugin_scores.get("d_score")) if plugin_scores.get("d_score") is not None else None,
                "c_score": float(plugin_scores.get("c_score")) if plugin_scores.get("c_score") is not None else None,
            }
        )
    return {
        "label": label,
        "vector_path": str(vector_path),
        "estimator": estimator,
        "plugin": plugin,
        "metric": metric,
        "layers": [int(layer) for layer, _ in by_layer],
        "scores": [float(score) for _, score in by_layer],
        "score_details": score_details,
        "ranked_layers": [{"layer": int(layer), "score": float(score)} for layer, score in ranked],
    }


def maybe_write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "vector_path", "estimator", "plugin", "metric", "layer", "score"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_series(series_list: list[dict[str, object]], args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to plot layer score trends: {exc}") from exc

    configure_matplotlib(args)

    fig, ax = plt.subplots(figsize=(args.figure_width, args.figure_height), constrained_layout=True)
    colors = [
        "#1f4e79",
        "#b85c38",
        "#2f6b4f",
        "#7a5c99",
        "#c08a00",
        "#4f6d7a",
        "#8c3b5d",
        "#5c677d",
    ]

    for idx, series in enumerate(series_list):
        label = str(series["label"])
        layers = list(series["layers"])
        scores = list(series["scores"])
        color = colors[idx % len(colors)]
        ax.plot(
            layers,
            scores,
            linewidth=2.2,
            color=color,
            label=label,
        )

        if args.top_k_annotate > 0:
            ranked_layers = list(series["ranked_layers"])[: args.top_k_annotate]
            for item in ranked_layers:
                layer = int(item["layer"])
                score = float(item["score"])
                ax.annotate(
                    f"L{layer}",
                    xy=(layer, score),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=color,
                )

    ax.axhline(0.0, color="#9aa1a9", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xlabel(args.x_label, fontsize=args.label_size)
    ax.set_ylabel(args.y_label or "Score", fontsize=args.label_size)
    ax.set_title(args.title or "Layer-wise Score Trend", fontsize=args.title_size, pad=12)
    legend = ax.legend(
        frameon=False,
        title=args.legend_title,
        fontsize=args.legend_size,
        title_fontsize=args.legend_size,
        loc="best",
        handlelength=2.8,
    )
    if legend is not None:
        for line in legend.get_lines():
            line.set_linewidth(2.4)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8, color="#b8c0c8")
    ax.tick_params(axis="both", labelsize=args.tick_size)
    if any(series["layers"] for series in series_list):
        all_layers = sorted({layer for series in series_list for layer in series["layers"]})
        set_layer_ticks(ax, all_layers, args)
    save_figure(fig, args.output, dpi=args.dpi)


def plot_decomposition(series: dict[str, object], args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to plot layer score trends: {exc}") from exc

    configure_matplotlib(args)

    details = list(series["score_details"])
    layers = [int(item["layer"]) for item in details]
    scores = [float(item["score"]) for item in details]
    d_scores = [float(item["d_score"]) if item["d_score"] is not None else 0.0 for item in details]
    c_scores = [float(item["c_score"]) if item["c_score"] is not None else 0.0 for item in details]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(args.figure_width, max(args.figure_height, 6.4)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.08},
        constrained_layout=True,
    )

    ax_top.plot(layers, scores, color="#2f855a", linewidth=2.2)
    ax_top.set_ylabel(args.y_label or "Score", fontsize=args.label_size)
    ax_top.set_title(args.title or f"{series['label']} Layer-wise Score Trend", fontsize=args.title_size, pad=10)
    ax_top.grid(axis="y", alpha=0.18, linewidth=0.8, color="#b8c0c8")
    ax_top.tick_params(axis="both", labelsize=args.tick_size)

    width = args.bar_width
    ax_bottom.bar(layers, d_scores, width=width, color="#b7a1d3", label="Discriminability")
    ax_bottom.bar(layers, c_scores, width=width, bottom=d_scores, color="#e3c06b", label="Consistency")
    ax_bottom.axhline(0.0, color="#9aa1a9", linewidth=1.0, linestyle="--", zorder=0)
    ax_bottom.set_xlabel(args.x_label, fontsize=args.label_size)
    ax_bottom.set_ylabel("Score", fontsize=args.label_size)
    ax_bottom.grid(axis="y", alpha=0.18, linewidth=0.8, color="#b8c0c8")
    ax_bottom.tick_params(axis="both", labelsize=args.tick_size)
    ax_bottom.legend(frameon=False, fontsize=args.legend_size, loc="upper right", handlelength=1.6)
    set_layer_ticks(ax_bottom, layers, args)

    save_figure(fig, args.output, dpi=args.dpi)


def plot_bars(series: dict[str, object], args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to plot layer score trends: {exc}") from exc

    configure_matplotlib(args)

    details = list(series["score_details"])
    layers = [int(item["layer"]) for item in details]
    d_scores = [float(item["d_score"]) if item["d_score"] is not None else 0.0 for item in details]
    c_scores = [float(item["c_score"]) if item["c_score"] is not None else 0.0 for item in details]

    fig, ax = plt.subplots(figsize=(args.figure_width, args.figure_height), constrained_layout=True)
    ax.bar(layers, d_scores, width=args.bar_width, color="#b7a1d3", label="Discriminability")
    ax.bar(
        layers,
        c_scores,
        width=args.bar_width,
        bottom=d_scores,
        color="#e3c06b",
        label="Consistency",
    )
    ax.axhline(0.0, color="#9aa1a9", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xlabel(args.x_label, fontsize=args.label_size)
    ax.set_ylabel(args.y_label or "Score", fontsize=args.label_size)
    if args.title:
        ax.set_title(args.title, fontsize=args.title_size, pad=8)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8, color="#b8c0c8")
    ax.tick_params(axis="both", labelsize=args.tick_size)
    ax.legend(
        frameon=False,
        fontsize=args.legend_size,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.5,
        ncol=2,
        columnspacing=1.2,
    )
    set_layer_ticks(ax, layers, args)

    save_figure(fig, args.output, dpi=args.dpi)


def plot_bars_grid(series_list: list[dict[str, object]], args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to plot layer score trends: {exc}") from exc

    configure_matplotlib(args)

    num_series = len(series_list)
    if num_series == 0:
        raise ValueError("No series available for bars-grid plotting.")
    num_cols = max(1, args.grid_columns)
    num_rows = (num_series + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(args.figure_width, args.figure_height),
        constrained_layout=False,
    )
    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    else:
        axes_list = [axes]

    legend_handles = None
    for idx, (ax, series) in enumerate(zip(axes_list, series_list)):
        details = list(series["score_details"])
        layers = [int(item["layer"]) for item in details]
        d_scores = [float(item["d_score"]) if item["d_score"] is not None else 0.0 for item in details]
        c_scores = [float(item["c_score"]) if item["c_score"] is not None else 0.0 for item in details]

        bars_d = ax.bar(layers, d_scores, width=args.bar_width, color="#b7a1d3", label="Discriminability")
        bars_c = ax.bar(
            layers,
            c_scores,
            width=args.bar_width,
            bottom=d_scores,
            color="#e3c06b",
            label="Consistency",
        )
        if legend_handles is None:
            legend_handles = [bars_d[0], bars_c[0]]
        ax.axhline(0.0, color="#9aa1a9", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_title(str(series["label"]), fontsize=args.title_size, pad=8)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8, color="#b8c0c8")
        ax.tick_params(axis="both", labelsize=args.tick_size)
        set_layer_ticks(ax, layers, args)

        row_idx = idx // num_cols
        col_idx = idx % num_cols
        if row_idx == num_rows - 1:
            ax.set_xlabel(args.x_label, fontsize=args.label_size)
        if col_idx == 0:
            ax.set_ylabel(args.y_label or "Score", fontsize=args.label_size)

    for ax in axes_list[num_series:]:
        ax.axis("off")

    if args.title:
        fig.suptitle(args.title, fontsize=args.title_size + 1.0, y=1.03)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["Discriminability", "Consistency"],
            frameon=False,
            fontsize=args.legend_size,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            columnspacing=1.4,
            handlelength=1.5,
        )

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.1, top=0.9, wspace=0.13, hspace=0.28)

    save_figure(fig, args.output, dpi=args.dpi)


def main() -> None:
    args = parse_args()
    series_list: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    payload_plugin = args.plugin
    payload_metric = args.metric

    if args.json_inputs is not None:
        if args.vector_paths is not None or args.json_input is not None:
            raise ValueError("Use only one of --vector-paths, --json-input, or --json-inputs.")
        json_inputs = [resolve_path(path) for path in args.json_inputs]
        series_list, payload = load_series_from_json_inputs(json_inputs)
        payload_plugin = str(payload.get("plugin", args.plugin))
        payload_metric = str(payload.get("metric", args.metric))
        if args.labels is not None:
            labels = normalize_labels([Path(str(item["vector_path"])) for item in series_list], args.labels)
            for series, label in zip(series_list, labels):
                series["label"] = label
        for series in series_list:
            for layer, score in zip(series["layers"], series["scores"]):
                csv_rows.append(
                    {
                        "label": str(series["label"]),
                        "vector_path": str(series["vector_path"]),
                        "estimator": str(series["estimator"]),
                        "plugin": str(series["plugin"]),
                        "metric": str(series["metric"]),
                        "layer": int(layer),
                        "score": float(score),
                    }
                )
    elif args.json_input is not None:
        if args.vector_paths is not None:
            raise ValueError("Use either --vector-paths or --json-input, not both.")
        json_input = resolve_path(args.json_input)
        series_list, payload = load_series_from_json(json_input)
        payload_plugin = str(payload.get("plugin", args.plugin))
        payload_metric = str(payload.get("metric", args.metric))
        if args.labels is not None:
            labels = normalize_labels([Path(str(item["vector_path"])) for item in series_list], args.labels)
            for series, label in zip(series_list, labels):
                series["label"] = label
        for series in series_list:
            for layer, score in zip(series["layers"], series["scores"]):
                csv_rows.append(
                    {
                        "label": str(series["label"]),
                        "vector_path": str(series["vector_path"]),
                        "estimator": str(series["estimator"]),
                        "plugin": str(series["plugin"]),
                        "metric": str(series["metric"]),
                        "layer": int(layer),
                        "score": float(score),
                    }
                )
    else:
        if not args.vector_paths:
            raise ValueError("Pass --vector-paths or --json-input.")
        vector_paths = [resolve_path(path) for path in args.vector_paths]
        labels = normalize_labels(vector_paths, args.labels)
        estimators = normalize_estimators(len(vector_paths), args.estimators)
        for vector_path, label, estimator in tqdm(
            list(zip(vector_paths, labels, estimators)),
            desc="Loading layer score artifacts",
        ):
            series = load_series(
                vector_path,
                label=label,
                estimator=estimator,
                plugin=args.plugin,
                metric=args.metric,
            )
            series_list.append(series)
            for layer, score in zip(series["layers"], series["scores"]):
                csv_rows.append(
                    {
                        "label": label,
                        "vector_path": str(vector_path),
                        "estimator": estimator,
                        "plugin": args.plugin,
                        "metric": args.metric,
                        "layer": int(layer),
                        "score": float(score),
                    }
                )

    print(f"Rendering {args.style} plot...")
    if args.style == "decomposition":
        if len(series_list) != 1:
            raise ValueError("--style decomposition currently supports exactly one vector artifact.")
        plot_decomposition(series_list[0], args)
    elif args.style == "bars":
        if len(series_list) != 1:
            raise ValueError("--style bars currently supports exactly one series.")
        plot_bars(series_list[0], args)
    elif args.style == "bars-grid":
        plot_bars_grid(series_list, args)
    else:
        plot_series(series_list, args)

    payload = {
        "plugin": payload_plugin,
        "metric": payload_metric,
        "output": str(args.output),
        "series": series_list,
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    if args.csv_output is not None:
        maybe_write_csv(csv_rows, args.csv_output)

    print(f"Saved layer score trend figure to {args.output}")
    if args.json_output is not None:
        print(f"Saved layer score trend JSON to {args.json_output}")
    if args.csv_output is not None:
        print(f"Saved layer score trend CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
