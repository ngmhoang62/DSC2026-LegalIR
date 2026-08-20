"""Execution-only controls and lightweight telemetry for EXP-012b.

Nothing in this module changes retrieval semantics.  Optimizations must opt in
through ``optimized`` and remain comparable with the reference implementation.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class ExecutionConfig:
    profile: str
    bm25_workers: int
    bm25_immutable: bool
    bm25_mmap_mib: int
    leaf_backend: str
    merge_lora: bool
    prefetch_batches: int
    reranker_batch_size: int
    bge_batch_size: int
    pretokenize_training: bool


REFERENCE = ExecutionConfig(
    profile="reference",
    bm25_workers=4,
    bm25_immutable=False,
    bm25_mmap_mib=0,
    leaf_backend="cpu_reference",
    merge_lora=False,
    prefetch_batches=0,
    reranker_batch_size=24,
    bge_batch_size=8,
    pretokenize_training=False,
)

OPTIMIZED = ExecutionConfig(
    profile="optimized",
    bm25_workers=4,
    bm25_immutable=True,
    bm25_mmap_mib=512,
    leaf_backend="cuda_exact",
    merge_lora=True,
    prefetch_batches=2,
    reranker_batch_size=28,
    bge_batch_size=16,
    pretokenize_training=True,
)


def execution_config(profile: str) -> ExecutionConfig:
    if profile == "reference":
        return REFERENCE
    if profile == "optimized":
        return OPTIMIZED
    raise ValueError(f"Unknown execution profile: {profile}")


@dataclass
class StageTelemetry:
    phase_seconds: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + (
                time.perf_counter() - started
            )

    def add(self, name: str, value: float) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + float(value)

    def snapshot(self, device: str | None = None) -> dict[str, Any]:
        memory: dict[str, float | None] = {
            "torch_peak_allocated_mib": None,
            "torch_peak_reserved_mib": None,
            "global_used_mib": None,
        }
        if device and device.startswith("cuda"):
            try:
                import torch

                memory["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                memory["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
                free, total = torch.cuda.mem_get_info()
                memory["global_used_mib"] = (total - free) / 2**20
            except (ImportError, RuntimeError):
                pass
        return {
            "elapsed_seconds": time.perf_counter() - self.started,
            "phase_seconds": {key: round(value, 6) for key, value in self.phase_seconds.items()},
            "counters": self.counters,
            "memory": memory,
        }


def configure_process(config: ExecutionConfig, receipt_path: Path | None = None) -> None:
    """Expose execution-only choices to modules without widening semantic APIs."""
    os.environ["LEGALIR_EXECUTION_PROFILE"] = config.profile
    if receipt_path is not None:
        os.environ["LEGALIR_V3_RECEIPT_PATH"] = str(receipt_path.resolve())


def profile_metadata(config: ExecutionConfig) -> dict[str, Any]:
    return asdict(config)


def read_profile(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
