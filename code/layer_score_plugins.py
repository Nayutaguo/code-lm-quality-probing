from __future__ import annotations

from typing import Callable

import torch


LayerScore = dict[str, float]
LayerScorePlugin = Callable[..., LayerScore]


def _normalize_rows(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(eps)


def _standardize_features(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    mean = matrix.mean(dim=0, keepdim=True)
    std = matrix.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (matrix - mean) / std


def layernavigator_score(
    positive: torch.Tensor,
    negative: torch.Tensor,
    vector: torch.Tensor,
    *,
    acts_pre: str | None = "standard",
    eps: float = 1e-12,
) -> LayerScore:
    positive = positive.float()
    negative = negative.float()
    vector = vector.float().reshape(-1)
    if positive.shape != negative.shape:
        raise ValueError(f"Positive/negative shapes must match, got {positive.shape} vs {negative.shape}.")
    if positive.ndim != 2:
        raise ValueError(f"Expected 2D activations, got shape {positive.shape}.")
    if positive.shape[1] != vector.numel():
        raise ValueError(
            f"Activation hidden size {positive.shape[1]} does not match vector size {vector.numel()}."
        )

    all_acts = torch.cat([positive, negative], dim=0)
    if acts_pre == "standard":
        all_acts = _standardize_features(all_acts, eps)
    elif acts_pre not in {None, "none"}:
        raise ValueError(f"Unsupported acts_pre mode: {acts_pre}")

    n_positive = positive.shape[0]
    positive_acts = all_acts[:n_positive]
    negative_acts = all_acts[n_positive:]
    pair_diff = _normalize_rows(positive_acts - negative_acts, eps)

    unit_vector = vector / vector.norm().clamp_min(eps)
    c_score = float((pair_diff @ unit_vector).mean().item())

    mean_total = all_acts.mean(dim=0)
    n_features = all_acts.shape[1]
    s_w = torch.zeros((n_features, n_features), dtype=torch.float32)
    s_b = torch.zeros((n_features, n_features), dtype=torch.float32)
    for class_acts in (positive_acts, negative_acts):
        mean_class = class_acts.mean(dim=0)
        centered = class_acts - mean_class
        s_w += centered.T @ centered
        mean_diff = (mean_class - mean_total).unsqueeze(1)
        s_b += class_acts.shape[0] * (mean_diff @ mean_diff.T)

    s_t = s_w + s_b
    v = unit_vector.reshape(-1, 1)
    numerator = float((v.T @ s_b @ v).item())
    denominator = float((v.T @ s_t @ v).item())
    d_score = numerator / denominator if denominator != 0.0 else float("inf")
    return {
        "s_score": float(d_score + c_score),
        "d_score": float(d_score),
        "c_score": float(c_score),
    }


SCORE_PLUGINS: dict[str, LayerScorePlugin] = {
    "layernavigator": layernavigator_score,
}


def parse_scored_layer_spec(layer_spec: str) -> tuple[str, str, int] | None:
    if not layer_spec.startswith("score:"):
        return None

    parts = layer_spec.split(":")
    if len(parts) == 3:
        _, plugin_name, top_k_text = parts
        metric = "s_score"
    elif len(parts) == 4:
        _, plugin_name, metric, top_k_text = parts
    else:
        raise ValueError(
            "--layers score selection must look like "
            "'score:<plugin>:<top_k>' or 'score:<plugin>:<metric>:<top_k>'."
        )

    try:
        top_k = int(top_k_text)
    except ValueError as exc:
        raise ValueError(f"Invalid top_k in score layer spec: {layer_spec!r}") from exc
    if top_k < 1:
        raise ValueError("--layers score selection requires top_k >= 1.")
    return plugin_name, metric, top_k


def rank_scored_layers(
    *,
    artifact: dict,
    estimator: str,
    plugin_name: str,
    metric: str,
) -> list[tuple[int, float]]:
    if "estimators" not in artifact or estimator not in artifact["estimators"]:
        raise KeyError(f"Estimator {estimator!r} not found in artifact; cannot select score-ranked layers.")

    ranked: list[tuple[int, float]] = []
    for layer in artifact.get("layers", []):
        layer_idx = int(layer)
        layer_info = artifact["estimators"][estimator][layer_idx]
        plugin_scores = layer_info.get("scores", {}).get(plugin_name)
        if plugin_scores is None:
            raise KeyError(
                f"Score plugin {plugin_name!r} not found for layer {layer_idx} under estimator {estimator!r}."
            )
        if metric not in plugin_scores:
            raise KeyError(
                f"Metric {metric!r} not found for score plugin {plugin_name!r}. "
                f"Available: {list(plugin_scores)}"
            )
        ranked.append((layer_idx, float(plugin_scores[metric])))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def describe_scored_layer_selection(
    layer_spec: str,
    *,
    artifact: dict,
    estimator: str,
    preview_top_n: int = 10,
) -> dict | None:
    parsed = parse_scored_layer_spec(layer_spec)
    if parsed is None:
        return None
    plugin_name, metric, top_k = parsed
    ranked = rank_scored_layers(
        artifact=artifact,
        estimator=estimator,
        plugin_name=plugin_name,
        metric=metric,
    )
    resolved_layers = [layer for layer, _ in ranked[:top_k]]
    preview = [
        {"rank": idx + 1, "layer": layer, "score": score}
        for idx, (layer, score) in enumerate(ranked[:preview_top_n])
    ]
    return {
        "mode": "score",
        "plugin": plugin_name,
        "metric": metric,
        "top_k": top_k,
        "resolved_layers": resolved_layers,
        "top_candidates": preview,
    }


def score_estimator_layers(
    *,
    plugin_names: list[str],
    layers: list[int],
    positive_by_layer: dict[int, list[torch.Tensor]],
    negative_by_layer: dict[int, list[torch.Tensor]],
    estimator_layers: dict[int, dict],
    acts_pre: str | None = "standard",
    eps: float = 1e-12,
) -> dict[str, dict[int, LayerScore]]:
    attached: dict[str, dict[int, LayerScore]] = {}
    for plugin_name in plugin_names:
        if plugin_name not in SCORE_PLUGINS:
            raise KeyError(f"Unknown score plugin {plugin_name!r}. Available: {list(SCORE_PLUGINS)}")
        plugin = SCORE_PLUGINS[plugin_name]
        attached[plugin_name] = {}
        for layer in layers:
            positive = torch.stack(positive_by_layer[layer]).float()
            negative = torch.stack(negative_by_layer[layer]).float()
            layer_info = estimator_layers[layer]
            vector = layer_info["unit"] if "unit" in layer_info else layer_info["raw"]
            score = plugin(positive, negative, vector, acts_pre=acts_pre, eps=eps)
            layer_info.setdefault("scores", {})[plugin_name] = score
            attached[plugin_name][layer] = score
    return attached


def score_estimator_matrices(
    *,
    plugin_names: list[str],
    layers: list[int],
    positive_by_layer: dict[int, torch.Tensor],
    negative_by_layer: dict[int, torch.Tensor],
    estimator_layers: dict[int, dict],
    acts_pre: str | None = "standard",
    eps: float = 1e-12,
) -> dict[str, dict[int, LayerScore]]:
    attached: dict[str, dict[int, LayerScore]] = {}
    for plugin_name in plugin_names:
        if plugin_name not in SCORE_PLUGINS:
            raise KeyError(f"Unknown score plugin {plugin_name!r}. Available: {list(SCORE_PLUGINS)}")
        plugin = SCORE_PLUGINS[plugin_name]
        attached[plugin_name] = {}
        for layer in layers:
            positive = positive_by_layer[layer].float()
            negative = negative_by_layer[layer].float()
            layer_info = estimator_layers[layer]
            vector = layer_info["unit"] if "unit" in layer_info else layer_info["raw"]
            score = plugin(positive, negative, vector, acts_pre=acts_pre, eps=eps)
            layer_info.setdefault("scores", {})[plugin_name] = score
            attached[plugin_name][layer] = score
    return attached


def resolve_scored_layers(
    layer_spec: str,
    *,
    artifact: dict,
    estimator: str,
) -> list[int] | None:
    description = describe_scored_layer_selection(
        layer_spec,
        artifact=artifact,
        estimator=estimator,
    )
    if description is None:
        return None
    return list(description["resolved_layers"])
