from __future__ import annotations

import difflib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class ConceptPair:
    sample_id: str
    source: str
    row_index: int
    task: str
    positive_text: str
    negative_text: str
    concept_name: str
    positive_name: str
    negative_name: str
    metadata: dict[str, Any]

    @property
    def description(self) -> str:
        return self.task

    @property
    def vul_type(self) -> str:
        return str(self.metadata.get("vul_type", ""))

    @property
    def file_name(self) -> str:
        return str(self.metadata.get("file_name", ""))

    @property
    def func_name(self) -> str:
        return str(self.metadata.get("func_name", ""))


@dataclass
class TokenRegion:
    name: str
    positions: list[int]

    def non_empty(self) -> bool:
        return len(self.positions) > 0


@dataclass
class EncodedPrompt:
    text: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    regions: dict[str, TokenRegion]


class PromptTooLongError(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_concept_pairs(
    paths: Iterable[Path],
    *,
    task_field: str = "description",
    positive_field: str = "positive",
    negative_field: str = "negative",
    concept_name: str = "concept",
    positive_name: str = "positive",
    negative_name: str = "negative",
    metadata_fields: Iterable[str] = (),
) -> list[ConceptPair]:
    pairs: list[ConceptPair] = []
    for path in paths:
        rows = read_jsonl(path)
        source = path.stem
        split = path.parent.name
        for row_index, row in enumerate(rows):
            task = str(row.get(task_field, "")).strip() if task_field else ""
            positive_text = str(row.get(positive_field, ""))
            negative_text = str(row.get(negative_field, ""))
            if not positive_text.strip() or not negative_text.strip():
                continue
            sample_id = f"{split}/{source}/{row_index}"
            pairs.append(
                ConceptPair(
                    sample_id=sample_id,
                    source=source,
                    row_index=row_index,
                    task=task,
                    positive_text=positive_text,
                    negative_text=negative_text,
                    concept_name=concept_name,
                    positive_name=positive_name,
                    negative_name=negative_name,
                    metadata={field: row.get(field, "") for field in metadata_fields},
                )
            )
    return pairs


def pair_metadata(pair: ConceptPair) -> dict[str, Any]:
    metadata = {
        "sample_id": pair.sample_id,
        "source": pair.source,
        "row_index": pair.row_index,
        "concept_name": pair.concept_name,
        "positive_name": pair.positive_name,
        "negative_name": pair.negative_name,
    }
    metadata.update(pair.metadata)
    return metadata


def default_dropped_samples_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.dropped.jsonl")


def dropped_pair_record(
    pair: ConceptPair,
    *,
    reason: str,
    details: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    record = pair_metadata(pair)
    record.update(
        {
            "task": pair.task,
            "stage": stage,
            "reason": reason,
        }
    )
    record.update(details)
    return record


def set_offline_mode(allow_network: bool) -> None:
    if not allow_network:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(
    model_path: Path,
    *,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
    trust_remote_code: bool = False,
    allow_network: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_offline_mode(allow_network)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=not allow_network,
        padding_side="left",
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": not allow_network,
        "low_cpu_mem_usage": True,
    }
    resolved = resolve_dtype(dtype)
    if resolved != "auto":
        kwargs["torch_dtype"] = resolved
    if device_map and device_map != "none":
        kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    except ValueError as exc:
        message = str(exc)
        if "model type `qwen3`" in message or "KeyError: 'qwen3'" in message:
            raise RuntimeError(
                "Qwen3 checkpoints require a newer Transformers version. "
                "Upgrade the active environment with: "
                'pip install -U "transformers>=4.51.0"'
            ) from exc
        raise
    model.eval()
    return model, tokenizer


def input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def num_hidden_states_for_model(model) -> int:
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers) + 1
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers) + 1
    raise ValueError("Cannot infer number of hidden states for this model.")


def parse_layer_spec(
    layer_spec: str,
    *,
    num_hidden_states: int,
    layer_start: int | None = None,
    layer_end: int | None = None,
    layer_stride: int = 1,
) -> list[int]:
    if layer_stride < 1:
        raise ValueError("--layer-stride must be >= 1")

    if layer_spec == "all":
        layers = list(range(num_hidden_states))
    elif layer_spec == "blocks":
        layers = list(range(1, num_hidden_states))
    else:
        layers = [int(part) for part in layer_spec.split(",") if part.strip()]

    if layer_start is not None:
        layers = [layer for layer in layers if layer >= layer_start]
    if layer_end is not None:
        layers = [layer for layer in layers if layer <= layer_end]
    layers = layers[::layer_stride]

    invalid = [layer for layer in layers if layer < 0 or layer >= num_hidden_states]
    if invalid:
        raise ValueError(
            f"Invalid layer indices {invalid}; model exposes hidden-state indices 0..{num_hidden_states - 1}."
        )
    return layers


def max_length_with_reserved_tokens(max_length: int | None, reserved_tokens: int) -> int | None:
    if max_length is None:
        return None
    available = max_length - reserved_tokens
    if available <= 0:
        raise ValueError(
            f"--max-length must leave room for at least {reserved_tokens} reserved token(s)."
        )
    return available


def save_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported save dtype: {dtype_name}")


def build_task_code_text(
    description: str,
    code: str,
    *,
    template: str = "description_plus_code",
) -> tuple[str, list[tuple[int, int]]]:
    if template == "description_plus_code":
        prefix = f"Task:\n{description.strip()}\n\nCode:\n" if description.strip() else "Code:\n"
        text = prefix + code
        return text, [(len(prefix), len(text))]
    if template == "code_only":
        return code, [(0, len(code))]
    raise ValueError(f"Unsupported prompt template: {template}")


def _line_offsets(text: str) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        offsets.append((cursor, end))
        cursor = end
    if not offsets and text:
        offsets.append((0, len(text)))
    return offsets


def changed_char_spans(before: str, after: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    before_offsets = _line_offsets(before)
    after_offsets = _line_offsets(after)
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)

    before_spans: list[tuple[int, int]] = []
    after_spans: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i1 < i2 and before_offsets:
            before_spans.append((before_offsets[i1][0], before_offsets[i2 - 1][1]))
        if j1 < j2 and after_offsets:
            after_spans.append((after_offsets[j1][0], after_offsets[j2 - 1][1]))
    return before_spans, after_spans


def _token_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        return list(encoded["input_ids"]), [tuple(offset) for offset in encoded["offset_mapping"]]
    except Exception:
        encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        return list(encoded["input_ids"]), None


def _prefix_count_positions(
    tokenizer,
    text: str,
    char_spans: list[tuple[int, int]],
    *,
    kept_start: int,
    bos_offset: int,
) -> list[int]:
    positions: set[int] = set()
    for char_start, char_end in char_spans:
        start = len(tokenizer(text[:char_start], add_special_tokens=False)["input_ids"])
        end = len(tokenizer(text[:char_end], add_special_tokens=False)["input_ids"])
        for raw_pos in range(start, max(start, end)):
            new_pos = raw_pos - kept_start + bos_offset
            if new_pos >= bos_offset:
                positions.add(new_pos)
    return sorted(positions)


def encode_text_with_regions(
    tokenizer,
    text: str,
    char_regions: dict[str, list[tuple[int, int]]],
    *,
    max_length: int | None,
    prepend_bos: bool = True,
    truncation_side: str = "right",
    overflow_policy: str = "truncate",
) -> EncodedPrompt:
    raw_ids, offsets = _token_offsets(tokenizer, text)
    bos_id = tokenizer.bos_token_id if prepend_bos else None
    bos_offset = 1 if bos_id is not None else 0
    available = len(raw_ids) if max_length is None else max_length - bos_offset
    if max_length is not None and available <= 0:
        raise ValueError("max_length is too small after reserving space for BOS.")

    if max_length is not None and len(raw_ids) > available:
        if overflow_policy == "drop":
            raise PromptTooLongError(
                "Prompt exceeds max_length and was dropped before truncation.",
                details={
                    "raw_token_count": int(len(raw_ids)),
                    "available_token_count": int(available),
                    "max_length": int(max_length),
                    "prepend_bos": bool(prepend_bos),
                    "truncation_side": truncation_side,
                },
            )
        if truncation_side == "right":
            kept_start, kept_end = 0, available
        elif truncation_side == "left":
            kept_start, kept_end = len(raw_ids) - available, len(raw_ids)
        else:
            raise ValueError("--truncation-side must be right or left")
        kept_ids = raw_ids[kept_start:kept_end]
        kept_offsets = offsets[kept_start:kept_end] if offsets is not None else None
    else:
        kept_start, kept_end = 0, len(raw_ids)
        kept_ids = raw_ids
        kept_offsets = offsets

    input_ids = ([bos_id] if bos_id is not None else []) + kept_ids
    seq_len = len(input_ids)
    regions: dict[str, TokenRegion] = {}

    for name, spans in char_regions.items():
        positions: set[int] = set()
        if kept_offsets is not None:
            for kept_pos, (tok_start, tok_end) in enumerate(kept_offsets):
                if tok_end <= tok_start:
                    continue
                for char_start, char_end in spans:
                    if tok_end > char_start and tok_start < char_end:
                        positions.add(kept_pos + bos_offset)
                        break
        else:
            positions.update(
                _prefix_count_positions(
                    tokenizer,
                    text,
                    spans,
                    kept_start=kept_start,
                    bos_offset=bos_offset,
                )
            )
        regions[name] = TokenRegion(
            name=name,
            positions=[pos for pos in sorted(positions) if 0 <= pos < seq_len],
        )

    ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    mask_tensor = torch.ones_like(ids_tensor)
    return EncodedPrompt(text=text, input_ids=ids_tensor, attention_mask=mask_tensor, regions=regions)


def extraction_prompt(
    tokenizer,
    description: str,
    code: str,
    *,
    paired_code: str | None,
    side: str,
    pooling: str,
    template: str,
    max_length: int | None,
    prepend_bos: bool,
    truncation_side: str,
    overflow_policy: str = "truncate",
) -> EncodedPrompt:
    text, code_spans = build_task_code_text(description, code, template=template)
    char_regions = {"code": code_spans, "all_tokens": [(0, len(text))]}

    if pooling == "changed_mean" and paired_code is not None:
        is_positive_side = side == "positive"
        negative_spans, positive_spans = (
            changed_char_spans(paired_code, code)
            if is_positive_side
            else changed_char_spans(code, paired_code)
        )
        changed_spans = positive_spans if is_positive_side else negative_spans
        prefix_len = code_spans[0][0] if code_spans else 0
        shifted = [(prefix_len + start, prefix_len + end) for start, end in changed_spans if end > start]
        if shifted:
            char_regions["changed"] = shifted

    return encode_text_with_regions(
        tokenizer,
        text,
        char_regions,
        max_length=max_length,
        prepend_bos=prepend_bos,
        truncation_side=truncation_side,
        overflow_policy=overflow_policy,
    )


def select_pool_region(encoded: EncodedPrompt, pooling: str) -> TokenRegion:
    if pooling == "all_tokens_mean":
        return encoded.regions["all_tokens"]
    if pooling == "changed_mean" and "changed" in encoded.regions and encoded.regions["changed"].non_empty():
        return encoded.regions["changed"]
    return encoded.regions["code"]


def pool_hidden(hidden: torch.Tensor, region: TokenRegion, *, pooling: str = "mean") -> torch.Tensor:
    if not region.positions:
        raise ValueError(f"Cannot pool empty token region: {region.name}")

    positions = region.positions
    if pooling in {"mean", "mean_code_span", "changed_mean", "all_tokens_mean"}:
        pos_tensor = torch.tensor(positions, dtype=torch.long, device=hidden.device)
        return hidden[0].index_select(0, pos_tensor).float().mean(dim=0)
    if pooling == "first_code_token":
        return hidden[0, positions[0], :].float()
    if pooling == "last_code_token":
        return hidden[0, positions[-1], :].float()
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def find_block_layers(model) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise ValueError("Cannot locate transformer block layers for hook backend.")


class HookedPooler:
    """Pool selected layer outputs during forward without storing all hidden states."""

    def __init__(self, model, layers: list[int], pooling: str):
        if 0 in layers:
            raise ValueError("Hook backend cannot extract embedding layer 0; use --backend native for layer 0.")
        self.model = model
        self.layers = layers
        self.pooling = pooling
        self._hooks = []
        self._region: TokenRegion | None = None
        self.outputs: dict[int, torch.Tensor] = {}

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

    def set_region(self, region: TokenRegion) -> None:
        self._region = region
        self.outputs = {}

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self._region is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            pooled = pool_hidden(hidden, self._region, pooling=self.pooling)
            self.outputs[layer_idx] = pooled.detach().cpu()

        return hook


class HookedRegionPooler:
    """Pool multiple named regions for selected layers during a single forward."""

    def __init__(self, model, layers: list[int]):
        if 0 in layers:
            raise ValueError("Hook backend cannot extract embedding layer 0; use --backend native for layer 0.")
        self.model = model
        self.layers = layers
        self._hooks = []
        self._regions: dict[str, TokenRegion] = {}
        self.outputs: dict[int, dict[str, torch.Tensor]] = {}

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

    def set_regions(self, regions: dict[str, TokenRegion]) -> None:
        self._regions = {name: region for name, region in regions.items() if region.non_empty()}
        self.outputs = {}

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.outputs[layer_idx] = {}
            for name, region in self._regions.items():
                pooled = pool_hidden(hidden, region, pooling="mean")
                self.outputs[layer_idx][name] = pooled.detach().cpu()

        return hook


def l2_normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vector / vector.float().norm().clamp_min(eps)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    numerator = torch.dot(a.float(), b.float())
    denominator = a.float().norm().clamp_min(eps) * b.float().norm().clamp_min(eps)
    return float((numerator / denominator).item())


def pearsonr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"Expected {text!r} to tokenize to one token, got ids={ids}")
    return int(ids[0])


def build_preference_prompt(
    tokenizer,
    pair: ConceptPair,
    *,
    rng: random.Random,
    max_length: int | None,
    prepend_bos: bool,
    truncation_side: str,
    overflow_policy: str = "truncate",
    preference_question: str | None = None,
    force_positive_label: str | None = None,
) -> tuple[EncodedPrompt, str, dict[str, str]]:
    if force_positive_label is None:
        positive_is_a = rng.random() < 0.5
    elif force_positive_label == "A":
        positive_is_a = True
    elif force_positive_label == "B":
        positive_is_a = False
    else:
        raise ValueError("--force-positive-label must be A, B, or None")
    code_a = pair.positive_text if positive_is_a else pair.negative_text
    code_b = pair.negative_text if positive_is_a else pair.positive_text
    positive_label = "A" if positive_is_a else "B"
    if preference_question is None:
        preference_question = (
            f"Which implementation better satisfies the target criterion "
            f"({pair.concept_name}) and should be preferred?"
        )

    chunks: list[str] = []
    spans: dict[str, list[tuple[int, int]]] = {}

    def append(text: str) -> int:
        start = sum(len(chunk) for chunk in chunks)
        chunks.append(text)
        return start

    task = pair.task.strip()
    if task:
        append("You are reviewing two implementations for the same task.\n\nTask:\n")
        desc_start = append(task)
        desc_end = desc_start + len(task)
        append("\n\nImplementation ")
    else:
        append("You are reviewing two implementations.\n\nImplementation ")
        desc_start = 0
        desc_end = 0
    label_a_start = append("A")
    append(":\n")
    a_start = append(code_a)
    a_end = a_start + len(code_a)
    append("\n\nImplementation ")
    label_b_start = append("B")
    append(":\n")
    b_start = append(code_b)
    b_end = b_start + len(code_b)
    append(f"\n\n{preference_question}\nAnswer with only A or B.\n\nAnswer:\n")
    text = "".join(chunks)

    if task:
        spans["description_span"] = [(desc_start, desc_end)]
    spans["implementation_A_span"] = [(a_start, a_end)]
    spans["implementation_B_span"] = [(b_start, b_end)]
    spans["option_label_A_token"] = [(label_a_start, label_a_start + 1)]
    spans["option_label_B_token"] = [(label_b_start, label_b_start + 1)]

    encoded = encode_text_with_regions(
        tokenizer,
        text,
        spans,
        max_length=max_length,
        prepend_bos=prepend_bos,
        truncation_side=truncation_side,
        overflow_policy=overflow_policy,
    )

    encoded.regions["positive_code_span"] = (
        encoded.regions["implementation_A_span"] if positive_is_a else encoded.regions["implementation_B_span"]
    )
    encoded.regions["negative_code_span"] = (
        encoded.regions["implementation_B_span"] if positive_is_a else encoded.regions["implementation_A_span"]
    )
    last_pos = int(encoded.input_ids.shape[1]) - 1
    encoded.regions["answer_prefix_last_token"] = TokenRegion("answer_prefix_last_token", [last_pos])

    labels = {
        "positive_label": positive_label,
        "negative_label": "B" if positive_label == "A" else "A",
        "A_side": "positive" if positive_is_a else "negative",
        "B_side": "negative" if positive_is_a else "positive",
    }
    return encoded, positive_label, labels


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
