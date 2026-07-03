from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TextSample:
    sample_id: str
    source: str
    row_index: int
    task: str
    text: str
    paired_text: str | None
    metadata: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_jsonl_text_samples(
    paths: Iterable[Path],
    *,
    task_field: str,
    text_field: str,
    paired_text_field: str | None = None,
    metadata_fields: Iterable[str] = (),
    limit: int | None = None,
) -> list[TextSample]:
    samples: list[TextSample] = []
    for path in paths:
        split = path.parent.name
        source = path.stem
        for row_index, row in enumerate(read_jsonl(path)):
            task = str(row.get(task_field, "")).strip() if task_field else ""
            text = str(row.get(text_field, ""))
            if not text.strip():
                continue
            paired_text = str(row.get(paired_text_field, "")) if paired_text_field else None
            samples.append(
                TextSample(
                    sample_id=f"{split}/{source}/{row_index}",
                    source=source,
                    row_index=row_index,
                    task=task,
                    text=text,
                    paired_text=paired_text,
                    metadata={field: row.get(field, "") for field in metadata_fields},
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def artifact_data_name(paths: Iterable[Path]) -> str:
    return "_".join(path.stem.replace("-", "_") for path in paths)
