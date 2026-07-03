#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from steering_utils import (  # noqa: E402
    build_preference_prompt,
    default_dropped_samples_path,
    dropped_pair_record,
    find_block_layers,
    input_device,
    load_concept_pairs,
    load_model_and_tokenizer,
    max_length_with_reserved_tokens,
    num_hidden_states_for_model,
    parse_layer_spec,
    PromptTooLongError,
    single_token_id,
    to_jsonable,
    l2_normalize,
    write_jsonl,
)
from layer_score_plugins import score_estimator_layers  # noqa: E402


ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build BED7-style answer-token steering vectors. For SafeCoder pairs, this "
            "constructs an A/B preference prompt, appends the safe or unsafe answer label, "
            "and estimates hidden(safe_label) - hidden(unsafe_label)."
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
    parser.add_argument(
        "--preference-question",
        default=None,
        help="Optional explicit A/B question. By default a generic concept-preference question is used.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--layers", default="blocks", help="'all', 'blocks', or comma-separated hidden-state indices.")
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--layer-stride", type=int, default=1)
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
    parser.add_argument("--unit-eps", type=float, default=1e-12)
    parser.add_argument(
        "--score-plugin",
        nargs="+",
        default=[],
        help="Optional layer-score plugins to attach to the saved estimator, e.g. layernavigator.",
    )
    parser.add_argument(
        "--score-acts-pre",
        choices=("standard", "none"),
        default="standard",
        help="Activation preprocessing for optional score plugins.",
    )
    return parser.parse_args()


def default_output_path(model_path: Path, concept_name: str, data_name: str) -> Path:
    model_name = model_path.name
    return ROOT / "artifacts/steering_vectors" / model_name / f"{concept_name}_{data_name}_answer_token_vectors.pt"


def data_name_from_paths(paths: list[Path]) -> str:
    return "_".join(path.stem.replace("-", "_") for path in paths)


class HookedLastTokenExtractor:
    """Collect selected block outputs at the final token position during one forward."""

    def __init__(self, model, layers: list[int]):
        if 0 in layers:
            raise ValueError("Hooked answer-token extraction cannot target embedding layer 0.")
        self.model = model
        self.layers = layers
        self.outputs: dict[int, torch.Tensor] = {}
        self._hooks = []

    def __enter__(self):
        blocks = find_block_layers(self.model)
        for layer_idx in self.layers:
            block_idx = layer_idx - 1
            if block_idx < 0 or block_idx >= len(blocks):
                raise ValueError(f"Layer {layer_idx} does not map to a transformer block.")
            self._hooks.append(blocks[block_idx].register_forward_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def reset(self) -> None:
        self.outputs = {}

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.outputs[layer_idx] = hidden[0, -1, :].detach().float().cpu()

        return hook


def backbone_module(model):
    """Return the transformer backbone so extraction does not materialize LM logits."""
    prefix = getattr(model, "base_model_prefix", None)
    if prefix:
        module = getattr(model, prefix, None)
        if module is not None and module is not model:
            return module
    for name in ("model", "transformer", "gpt_neox"):
        module = getattr(model, name, None)
        if module is not None and module is not model:
            return module
    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return base_model
    return model


def append_answer_label(
    encoded,
    label_id: int,
    *,
    max_length: int | None,
    truncation_side: str,
    overflow_policy: str = "truncate",
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = encoded.input_ids[0].tolist()
    if max_length is not None and len(ids) + 1 > max_length:
        if overflow_policy == "drop":
            raise PromptTooLongError(
                "Prompt+answer exceeds max_length and was dropped before truncation.",
                details={
                    "prompt_token_count": int(len(ids)),
                    "required_token_count": int(len(ids) + 1),
                    "max_length": int(max_length),
                    "truncation_side": truncation_side,
                },
            )
        keep = max_length - 1
        if keep < 1:
            raise ValueError("--max-length must leave room for at least one prompt token and one answer label.")
        if truncation_side == "left":
            ids = ids[-keep:]
        else:
            ids = ids[:keep]
    ids = ids + [label_id]
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def forward_last_token(model, extractor: HookedLastTokenExtractor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[int, torch.Tensor]:
    device = input_device(model)
    backbone = backbone_module(model)
    extractor.reset()
    with torch.inference_mode():
        _ = backbone(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=False,
            use_cache=False,
        )
    return dict(extractor.outputs)


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


def main() -> None:
    args = parse_args()
    output = args.output or default_output_path(args.model_path, args.concept_name, data_name_from_paths(args.data_files))
    output.parent.mkdir(parents=True, exist_ok=True)

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
    pairs = load_pairs(args)
    rng = random.Random(args.seed)

    positive_by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    negative_by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    sample_metadata: list[dict[str, Any]] = []
    dropped_samples: list[dict[str, Any]] = []
    stats = {"seen": 0, "kept": 0, "skipped_overlength": 0, "skipped_errors": 0}

    with HookedLastTokenExtractor(model, layers) as extractor:
        for pair in tqdm(pairs, desc="Building answer-token vectors"):
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
                    positive_by_layer[layer].append(positive_outputs[layer])
                    negative_by_layer[layer].append(negative_outputs[layer])
                sample_metadata.append(
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
                        stage="build_answer_token_vectors",
                    )
                )
            except Exception:
                stats["skipped_errors"] += 1
                if not args.skip_errors:
                    raise

    if stats["kept"] == 0:
        raise RuntimeError(f"No answer-token samples were kept. Stats: {stats}")

    estimators = {"answer_token_diff": {}}
    for layer in layers:
        positive = torch.stack(positive_by_layer[layer]).float()
        negative = torch.stack(negative_by_layer[layer]).float()
        diffs = positive - negative
        raw = diffs.mean(dim=0)
        unit = l2_normalize(raw, eps=args.unit_eps)
        estimators["answer_token_diff"][layer] = {
            "raw": raw.cpu(),
            "unit": unit.cpu(),
            "norm": float(raw.float().norm().item()),
            "n_positive": int(positive.shape[0]),
            "n_negative": int(negative.shape[0]),
            "n_pairs": int(diffs.shape[0]),
            "n_used_for_estimator": int(diffs.shape[0]),
            "hidden_size": int(positive.shape[1]),
            "pair_diff_norm_mean": float(diffs.norm(dim=1).mean().item()),
            "pair_diff_norm_std": float(diffs.norm(dim=1).std(unbiased=False).item()),
            "direction": "positive_minus_negative",
            "alignment": "prompt_randomized_answer_label",
        }

    attached_scores = {}
    if args.score_plugin:
        attached_scores = score_estimator_layers(
            plugin_names=args.score_plugin,
            layers=layers,
            positive_by_layer=positive_by_layer,
            negative_by_layer=negative_by_layer,
            estimator_layers=estimators["answer_token_diff"],
            acts_pre=None if args.score_acts_pre == "none" else args.score_acts_pre,
            eps=args.unit_eps,
        )

    artifact = {
        "estimators": estimators,
        "score_plugins": attached_scores,
        "layers": layers,
        "sample_metadata": sample_metadata,
        "stats": stats,
        "config": to_jsonable(vars(args)),
    }
    torch.save(artifact, output)
    dropped_output = args.dropped_samples_output or default_dropped_samples_path(output)
    if dropped_samples:
        write_jsonl(dropped_output, dropped_samples)
        print(f"Saved dropped overlength samples to {dropped_output}")
    print(f"Saved answer-token steering vectors to {output}")
    print("Estimator: answer_token_diff")
    if attached_scores:
        print(f"Attached score plugins: {sorted(attached_scores)}")
    print(f"Layers: {layers}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
