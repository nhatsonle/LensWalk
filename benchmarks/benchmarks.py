from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .readers import BaseDatasetReader, FieldMap, JSONLReader


@dataclass
class BenchmarkSpec:
    pattern: str
    reader: str = "jsonl"
    field_map: Optional[FieldMap] = None
    validate: bool = True


BENCH_REGISTRY: Dict[str, BenchmarkSpec] = {}


def register_benchmark(name: str, spec: BenchmarkSpec):
    BENCH_REGISTRY[name] = spec


def _looks_like_path(s: str) -> bool:
    return ("/" in s) or s.endswith(".jsonl") or s.endswith(".jsonl.gz") or s.endswith(".json")


def _resolve_path(pattern: str, data_root: Optional[str]) -> str:
    if os.path.isabs(pattern) or os.path.exists(pattern):
        return pattern
    if data_root:
        return os.path.join(data_root, os.path.basename(pattern))
    return pattern


def load_benchmark(
    bench_name_or_path: str,
    *,
    data_root: Optional[str] = None,
    subtitle_root: Optional[str] = None,
    start: int = 0,
    end: int = -1,
    stream: bool = False,
    shuffle: bool = False,
    shuffle_seed: Optional[int] = None,
    shuffle_buffer_size: int = 2048,
) -> Iterable[dict]:
    """Load a registered JSONL benchmark or a direct JSON/JSONL path."""
    if _looks_like_path(bench_name_or_path):
        spec = BenchmarkSpec(
            pattern=bench_name_or_path,
            reader="jsonl",
            field_map=FieldMap(),
            validate=True,
        )
    else:
        if bench_name_or_path not in BENCH_REGISTRY:
            known = ", ".join(sorted(BENCH_REGISTRY.keys()))
            raise KeyError(f"Unknown benchmark '{bench_name_or_path}'. Known: {known}")
        spec = BENCH_REGISTRY[bench_name_or_path]

    pattern = _resolve_path(spec.pattern, data_root)

    if spec.reader != "jsonl":
        raise NotImplementedError(f"Reader '{spec.reader}' is not implemented.")

    reader: BaseDatasetReader = JSONLReader(
        files=pattern,
        field_map=spec.field_map or FieldMap(),
        validate=spec.validate,
        stream=stream,
        video_root=data_root,
        subtitle_root=subtitle_root,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    return reader.select_range(start, end) if start or end >= 0 else reader
