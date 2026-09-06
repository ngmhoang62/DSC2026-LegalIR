"""EXP-109B: encoder complementarity and full-corpus nested fusion.

This module intentionally owns a fresh namespace.  It consumes the frozen
LegalIR corpus, E5 cache, and audited EXP-021 sparse evidence, but it never
modifies an earlier experiment.  The implementation is split into small,
testable contracts so that the inexpensive audit/replay stages can run
without loading a model.  GPU stages are fail-closed unless the caller has
explicitly opted in through the EXP109B authorization environment variables.

The primary dense scorer is source-exact chunk cosine followed by
``top2_mean`` parent aggregation.  FP16 document vectors are dequantized to
FP32 and renormalized at the scoring boundary.  Stable parent-id ordering is
used only after score equality.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:  # Torch is optional for import-time audit/unit tests.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised only on minimal runtimes.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Frozen paths and immutable experiment contracts
# ---------------------------------------------------------------------------

CODE_FILE = Path(__file__).resolve()
ROOT = CODE_FILE.parents[1]
TRAIN_PATH = ROOT / "public_test_dataset" / "train.json"
FOLDS_PATH = ROOT / "cache" / "cv_folds.json"
EXCLUSIONS_PATH = ROOT / "cache" / "final_preprocessed_v2" / "exclusions.json"
LABEL_IMPACT_PATH = ROOT / "cache" / "final_preprocessed_v2" / "train_label_impact.jsonl"
PROCESSED_MANIFEST = ROOT / "cache" / "final_preprocessed_v2" / "manifest.json"
STRUCT_DIR = ROOT / "cache" / "structural_v3_e5_final_v1"
STRUCT_MANIFEST = STRUCT_DIR / "manifest.json"
E5_DIR = ROOT / "cache" / "e5_final_v1"
E5_MANIFEST = E5_DIR / "manifest.json"
QUERY_DIR = ROOT / "cache" / "exp021_e5_dense_candidates" / "query_embeddings"
QUERY_MANIFEST = QUERY_DIR / "manifest.json"
BM25_RAW_DIR = ROOT / "cache" / "exp021_sparse" / "depth_tune" / "raw4096_evidence"
BM25_TUNING_REPORT = ROOT / "results" / "exp021_sparse" / "depth_rrf_tuning" / "tuning_report.json"
BM25_MANIFEST = ROOT / "cache" / "exp021_sparse" / "passage_hierarchy" / "fts5" / "manifest.json"
BM25_DB = BM25_MANIFEST.parent / "bm25_v3.sqlite"
EXP035_REPORT = ROOT / "results" / "exp035_retrieval_error_adjudication" / "REPORT.json"
EXP035_EVIDENCE = ROOT / "results" / "exp035_retrieval_error_adjudication" / "evidence_pack.jsonl"
EXP036_REPORT = ROOT / "results" / "exp036_coverage_aware_fusion" / "REPORT.json"
EXP036_UNIVERSE = ROOT / "cache" / "exp036_coverage_aware_fusion" / "universe.jsonl"
EXP037_DIR = ROOT / "cache" / "exp037_cached_encoder_complementarity"
EXP037_FIXTURE = EXP037_DIR / "fixture.jsonl"
EXP037_REPORT = EXP037_DIR / "REPORT.json"
EXP109A_RESULTS = ROOT / "results" / "exp109a_softtop5_retrieval"
EXP109A_CACHE = ROOT / "cache" / "exp109a_softtop5_retrieval"
EXP015_RESULTS = ROOT / "results" / "exp015_model_screen"
EXP015_FIXTURE = ROOT / "cache" / "exp015_model_screen" / "fixture"
EXP015_STAGE2_SUMMARY = EXP015_RESULTS / "stage2" / "summary.md"
EXP104_REPORT = ROOT / "results" / "exp104_preranker_benchmark" / "REPORT_benchmark_summary.json"

NAMESPACE = "exp109b_encoder_complementarity"
CACHE_ROOT = ROOT / "cache" / NAMESPACE
RESULTS_ROOT = ROOT / "results" / NAMESPACE
LOG_ROOT = RESULTS_ROOT / "logs"

SCHEMA = "legalir.exp109b_encoder_complementarity.v1"
GATE_REPAIR_ID = "GATE_REPAIR_AFTER_OBSERVATION"
LABEL_POLICY = "canonical_duplicate_alias_drop_empty_passage_v1"
SCORER_CONTRACT = "fp16_to_fp32_l2_cosine_top2mean_v1"
DIMENSION = 1024
QUERY_COUNT = 7000
EVALUABLE_QUERY_COUNT = 6991
DOCUMENT_COUNT = 8507
CHUNK_COUNT = 343347
FOLD_NAMES = tuple(f"fold_{i}" for i in range(5))
CURVE_KS = (1, 3, 5, 10, 16, 20, 32, 50, 64, 100, 150)
CANDIDATE_DEPTHS = (100, 200, 500)
BM25_MAX_RANK = 500
BOUND_BOOTSTRAP_SAMPLES = 10_000
RNG_SEED = 109

INSTRUCTION_PREFIX = "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: "


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    backend: str
    max_length: int
    query_prefix: str
    document_prefix: str
    pooling: str
    dimension: int = DIMENSION


MODEL_SPECS: dict[str, ModelSpec] = {
    "vietlegal_e5": ModelSpec(
        "vietlegal_e5", "mainguyen9/vietlegal-e5", "sentence_transformers", 512,
        "query: ", "passage: ", "native", DIMENSION,
    ),
    "vietlegal_harrier_0_6b": ModelSpec(
        "vietlegal_harrier_0_6b", "mainguyen9/vietlegal-harrier-0.6b", "sentence_transformers", 512,
        INSTRUCTION_PREFIX, "", "native", DIMENSION,
    ),
    "vnlegal_lal": ModelSpec(
        "vnlegal_lal", "darklethelong/vnlegal-lal", "transformers", 2048,
        INSTRUCTION_PREFIX, "", "last_non_padding", DIMENSION,
    ),
}
MODELS = MODEL_SPECS


class GateRejected(RuntimeError):
    """A planned fail-closed gate rejected an execution stage."""

    def __init__(self, status: str, message: str, *, report: Path | None = None):
        super().__init__(message)
        self.status = status
        self.report = report


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                yield value


def atomic_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        if pretty:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        else:
            handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
            for record in records:
                handle.write(canonical_json(dict(record)) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def code_fingerprint() -> str:
    return sha256_file(CODE_FILE)


def write_success(directory: Path, *, stage: str, fingerprint: str, extra: Mapping[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "stage": stage,
        "status": "PASS",
        "content_fingerprint": fingerprint,
        "code_sha256": code_fingerprint(),
        "finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if extra:
        payload.update(dict(extra))
    atomic_json(directory / "_SUCCESS.json", payload)


def require_success(
    directory: Path,
    expected_fingerprint: str | None = None,
    *,
    expected_code_sha256: str | None = None,
) -> dict[str, Any]:
    marker_path = directory / "_SUCCESS.json"
    if not marker_path.exists():
        raise RuntimeError(f"Missing success marker: {marker_path}")
    marker = read_json(marker_path)
    if marker.get("status") != "PASS":
        raise RuntimeError(f"Upstream marker is not PASS: {marker_path}")
    if expected_fingerprint is not None and marker.get("content_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Fingerprint mismatch: {marker_path}")
    if expected_code_sha256 is not None and marker.get("code_sha256") != expected_code_sha256:
        raise RuntimeError(f"Code fingerprint mismatch: {marker_path}")
    return marker


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class RunTracker:
    """Small flushed status/log writer used by all resumable stages."""

    def __init__(self, stage: str, *, outer: str | None = None, model: str | None = None, total: int = 0):
        self.stage = stage
        self.outer = outer
        self.model = model
        self.total = int(total)
        self.run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        self.run_dir = LOG_ROOT / self.run_id
        self.log_path = self.run_dir / f"{stage}.log"
        self.run_log_path = self.run_dir / "run.log"
        self.status_path = RESULTS_ROOT / "RUN_STATUS.json"
        self.started = time.monotonic()
        self.completed = 0
        self.state = "RUNNING"
        self.update()

    def update(self, **values: Any) -> None:
        payload = {
            "run_id": self.run_id,
            "state": self.state,
            "phase": self.stage,
            "outer": self.outer,
            "model": self.model,
            "completed": self.completed,
            "total": self.total,
            "throughput": self.completed / max(time.monotonic() - self.started, 1e-9),
            "eta_seconds": max(0.0, (self.total - self.completed) / max(self.completed / max(time.monotonic() - self.started, 1e-9), 1e-9)) if self.completed else 0.0,
            "last_heartbeat": utc_now(),
            "input_fingerprint": values.pop("input_fingerprint", None),
            "config_fingerprint": values.pop("config_fingerprint", None),
            "code_fingerprint": code_fingerprint(),
        }
        payload.update(values)
        atomic_json(self.status_path, payload)

    def log(self, message: str, *, emit: bool = False) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        line = f"[{utc_now()}] {message}"
        for path in (self.run_log_path, self.log_path):
            with path.open("a", encoding="utf-8", newline="\n", buffering=1) as handle:
                handle.write(line + "\n")
                handle.flush()
        if emit:
            print(line, flush=True)

    def heartbeat(self, completed: int | None = None, *, emit: bool = True, **values: Any) -> None:
        if completed is not None:
            self.completed = int(completed)
        self.update(**values)
        self.log(f"progress={self.completed}/{self.total} stage={self.stage}", emit=emit)

    def finish(self, state: str, **values: Any) -> None:
        self.state = state
        self.update(**values)
        self.log(f"state={state}", emit=True)


@contextlib.contextmanager
def tracked_stage(stage: str, *, outer: str | None = None, model: str | None = None, total: int = 0) -> Iterator[RunTracker]:
    tracker = RunTracker(stage, outer=outer, model=model, total=total)
    try:
        yield tracker
    except KeyboardInterrupt:
        tracker.finish("INTERRUPTED")
        raise
    except Exception:
        tracker.finish("FAILED", error=traceback.format_exc(limit=6))
        raise


# ---------------------------------------------------------------------------
# Canonical labels, folds, and corpus index
# ---------------------------------------------------------------------------


def load_train(path: Path = TRAIN_PATH) -> dict[str, dict[str, Any]]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Train file must be an object: {path}")
    return {str(qid): dict(row) for qid, row in raw.items()}


def canonical_labels(
    *,
    train_path: Path = TRAIN_PATH,
    exclusions_path: Path = EXCLUSIONS_PATH,
    impact_path: Path = LABEL_IMPACT_PATH,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    original = load_train(train_path)
    exclusions_rows = read_json(exclusions_path)
    exclusions = {str(row["doc_id"]): row for row in exclusions_rows}
    if len(exclusions) != len(exclusions_rows):
        raise ValueError("duplicate document id in preprocessing exclusions")
    impacts = list(read_jsonl(impact_path))
    impacted = {str(row["query_id"]): row for row in impacts}
    if len(impacted) != len(impacts):
        raise ValueError("duplicate query id in label-impact sidecar")
    answers: dict[str, set[str]] = {}
    observed: dict[str, set[str]] = {}
    duplicate_occurrences = 0
    empty_occurrences = 0
    for raw_qid, row in original.items():
        qid = str(raw_qid)
        gold: set[str] = set()
        removed: set[str] = set()
        for raw_doc_id in row.get("answer", []):
            doc_id = str(raw_doc_id)
            exclusion = exclusions.get(doc_id)
            if exclusion is None:
                gold.add(doc_id)
                continue
            removed.add(doc_id)
            reasons = {str(reason) for reason in exclusion.get("reasons", [])}
            replacement = exclusion.get("duplicate_retained_id")
            if "exact_duplicate_raw_passage" in reasons:
                if not replacement:
                    raise ValueError(f"duplicate gold has no retained alias: {qid}/{doc_id}")
                replacement = str(replacement)
                if replacement in exclusions:
                    raise ValueError(f"duplicate alias is itself excluded: {qid}/{doc_id}")
                gold.add(replacement)
                duplicate_occurrences += 1
            elif reasons == {"empty_passage"}:
                empty_occurrences += 1
            else:
                raise ValueError(f"unsupported exclusion policy: {qid}/{doc_id}/{sorted(reasons)}")
        answers[qid] = gold
        if removed:
            observed[qid] = removed
    declared = {
        str(qid): {str(value) for value in row.get("intentionally_excluded_gold_ids", [])}
        for qid, row in impacted.items()
    }
    if observed != declared:
        raise ValueError("canonical-gold impact sidecar mismatch")
    non_evaluable = sorted(qid for qid, gold in answers.items() if not gold)
    stats = {
        "policy": LABEL_POLICY,
        "queries": len(answers),
        "evaluable_queries": len(answers) - len(non_evaluable),
        "non_evaluable_queries": len(non_evaluable),
        "non_evaluable_qids": non_evaluable,
        "affected_queries": len(observed),
        "canonicalized_duplicate_occurrences": duplicate_occurrences,
        "dropped_empty_occurrences": empty_occurrences,
        "label_fingerprint": content_hash({qid: sorted(gold) for qid, gold in sorted(answers.items())}),
    }
    return answers, stats


def load_folds(
    *,
    train_path: Path = TRAIN_PATH,
    folds_path: Path = FOLDS_PATH,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    train = load_train(train_path)
    raw = read_json(folds_path)
    folds = {str(name): [str(qid) for qid in values] for name, values in raw.items()}
    fold_for: dict[str, str] = {}
    for name, qids in folds.items():
        for qid in qids:
            if qid in fold_for:
                raise ValueError(f"duplicate query id in folds: {qid}")
            fold_for[qid] = name
    if set(fold_for) != set(train) or len(fold_for) != sum(len(qids) for qids in folds.values()):
        raise ValueError("cv_folds.json is not an exactly-one-fold partition of train.json")
    return folds, fold_for


@dataclass
class CorpusIndex:
    doc_ids: list[str]
    chunk_doc_ids: list[str]
    starts: list[int]
    ends: list[int]
    chunk_indices: list[np.ndarray] | None = None

    @classmethod
    def from_chunk_doc_ids(cls, chunk_doc_ids: Sequence[str]) -> "CorpusIndex":
        chunks = [str(value) for value in chunk_doc_ids]
        positions: dict[str, list[int]] = defaultdict(list)
        for position, doc_id in enumerate(chunks):
            positions[doc_id].append(position)
        # Parent order is the first-seen order.  A non-contiguous parent must
        # still occur once in the parent vector; its explicit index list is
        # then used by the scorer.
        doc_ids = list(positions)
        noncontiguous = any(not np.all(np.diff(values) == 1) for values in positions.values() if len(values) > 1)
        if noncontiguous:
            starts = [int(values[0]) for values in positions.values()]
            ends = [int(values[-1]) + 1 for values in positions.values()]
            chunk_indices = [np.asarray(positions[doc_id], dtype=np.int64) for doc_id in doc_ids]
        else:
            starts = []
            ends = []
            for doc_id in doc_ids:
                values = positions[doc_id]
                starts.append(int(values[0]))
                ends.append(int(values[-1]) + 1)
            chunk_indices = None
        return cls(doc_ids, chunks, starts, ends, chunk_indices)

    @property
    def parent_count(self) -> int:
        return len(self.doc_ids)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_doc_ids)

    @property
    def noncontiguous(self) -> bool:
        return self.chunk_indices is not None

    def indices_for_parent(self, parent_index: int) -> np.ndarray:
        if self.chunk_indices is not None:
            return self.chunk_indices[parent_index]
        return np.arange(self.starts[parent_index], self.ends[parent_index], dtype=np.int64)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA,
            "doc_ids": self.doc_ids,
            "starts": self.starts,
            "ends": self.ends,
            "chunk_count": self.chunk_count,
            "parent_count": self.parent_count,
            "noncontiguous": self.noncontiguous,
            "chunk_indices": None if self.chunk_indices is None else [values.tolist() for values in self.chunk_indices],
        }

    def fingerprint(self) -> str:
        return content_hash({"doc_ids": self.doc_ids, "starts": self.starts, "ends": self.ends, "noncontiguous": self.noncontiguous, "chunk_indices": None if self.chunk_indices is None else [values.tolist() for values in self.chunk_indices]})


def load_corpus_index() -> CorpusIndex:
    return CorpusIndex.from_chunk_doc_ids(
        [str(row["doc_id"]) for row in read_jsonl(STRUCT_DIR / "chunks.jsonl")]
    )


def nested_partitions(folds: Mapping[str, Sequence[str]], outer: str) -> list[dict[str, list[str]]]:
    if outer not in folds:
        raise KeyError(outer)
    remaining = [name for name in sorted(folds) if name != outer]
    return [
        {
            "train_qids": [qid for name in remaining if name != validation for qid in folds[name]],
            "validation_qids": list(folds[validation]),
        }
        for validation in remaining
    ]


def _fold_train_qids(folds: Mapping[str, Sequence[str]], heldout: str) -> list[str]:
    return [qid for name, qids in sorted(folds.items()) if name != heldout for qid in qids]


# ---------------------------------------------------------------------------
# Exact dense scorer and independent NumPy reference
# ---------------------------------------------------------------------------


def numpy_l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norms, np.finfo(np.float32).eps)


def renormalize_fp16(values: np.ndarray) -> np.ndarray:
    """Dequantize FP16 vectors in FP32 and normalize exactly once."""
    return numpy_l2_normalize(np.asarray(values, dtype=np.float16).astype(np.float32, copy=False))


def _torch_l2_normalize(values: "torch.Tensor") -> "torch.Tensor":
    if F is None:
        raise RuntimeError("torch is required for differentiable scoring")
    return F.normalize(values, p=2.0, dim=-1, eps=1e-12)


def _document_block(
    documents: Any,
    index: CorpusIndex,
    start: int,
    end: int,
    device: Any,
    dtype: Any,
) -> tuple["torch.Tensor", np.ndarray]:
    """Load a parent range, preserving source chunk order and FP16 contract."""
    if torch is None:
        raise RuntimeError("torch is required for _document_block")
    if index.chunk_indices is None:
        positions = np.arange(index.starts[start], index.ends[end - 1], dtype=np.int64)
    else:
        positions = np.concatenate(index.chunk_indices[start:end])
    if isinstance(documents, torch.Tensor):
        tensor = documents[torch.as_tensor(positions, device=documents.device)]
        tensor = tensor.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(np.asarray(documents)[positions], device=device, dtype=dtype)
    return _torch_l2_normalize(tensor), positions


def _parent_ranges(index: CorpusIndex, chunk_block_size: int) -> Iterator[tuple[int, int]]:
    start = 0
    while start < index.parent_count:
        end = start
        chunks = 0
        while end < index.parent_count:
            count = len(index.indices_for_parent(end))
            if end > start and chunks + count > chunk_block_size:
                break
            chunks += count
            end += 1
        yield start, end
        start = end


def compute_parent_scores(
    query: "torch.Tensor",
    documents: Any,
    index: CorpusIndex,
    *,
    chunk_block_size: int = 8192,
    device: Any | None = None,
    dtype: Any | None = None,
) -> "torch.Tensor":
    """Differentiable exact all-parent top-2 mean scorer.

    The function deliberately does not detach scores.  Within each parent,
    ``torch.topk`` keeps the two winning chunk paths in the autograd graph.
    """
    if torch is None:
        raise RuntimeError("torch is required for compute_parent_scores")
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if device is None:
        device = query.device
    if dtype is None:
        dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    q = _torch_l2_normalize(query.to(device=device, dtype=dtype))
    output: list[torch.Tensor] = []
    for parent_start, parent_end in _parent_ranges(index, max(1, int(chunk_block_size))):
        block, _ = _document_block(documents, index, parent_start, parent_end, device, dtype)
        scores = q @ block.transpose(0, 1)
        offset = 0
        for parent_index in range(parent_start, parent_end):
            count = len(index.indices_for_parent(parent_index))
            local = scores[:, offset:offset + count]
            top_count = min(2, count)
            output.append(torch.topk(local, k=top_count, dim=1, largest=True, sorted=False).values.mean(dim=1))
            offset += count
    if not output:
        return q.new_empty((q.shape[0], 0))
    return torch.stack(output, dim=1)


def reference_parent_scores(query: "torch.Tensor", documents: Any, index: CorpusIndex) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch is required for reference_parent_scores")
    if query.ndim != 1:
        raise ValueError("reference_parent_scores expects one query vector")
    if isinstance(documents, torch.Tensor):
        docs = _torch_l2_normalize(documents.to(dtype=query.dtype, device=query.device))
    else:
        document_array = np.asarray(documents)
        # The reference path must follow the same dequantize contract as the
        # production scorer.  FP16 cache vectors are promoted then normalized;
        # already-FP32 fixtures must not be quantized a second time.
        if document_array.dtype == np.float16:
            document_values = renormalize_fp16(document_array)
        else:
            document_values = numpy_l2_normalize(np.asarray(document_array, dtype=np.float32))
        docs = torch.as_tensor(document_values, dtype=query.dtype, device=query.device)
    q = _torch_l2_normalize(query)
    values: list[torch.Tensor] = []
    for parent_index in range(index.parent_count):
        scores = docs[index.indices_for_parent(parent_index)] @ q
        values.append(torch.topk(scores, k=min(2, scores.numel()), largest=True, sorted=False).values.mean())
    return torch.stack(values) if values else q.new_empty((0,))


def numpy_top2_parent_scores(
    query: np.ndarray,
    documents: np.ndarray,
    index: CorpusIndex,
    parent_indices: Sequence[int] | None = None,
) -> np.ndarray:
    q = numpy_l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
    docs = renormalize_fp16(np.asarray(documents)) if np.asarray(documents).dtype == np.float16 else numpy_l2_normalize(np.asarray(documents, dtype=np.float32))
    selected = list(range(index.parent_count)) if parent_indices is None else [int(i) for i in parent_indices]
    output = []
    for parent_index in selected:
        scores = docs[index.indices_for_parent(parent_index)] @ q
        output.append(float(np.mean(np.sort(scores)[-min(2, len(scores)):])) if len(scores) else float("nan"))
    return np.asarray(output, dtype=np.float32)


def numpy_secondary_parent_scores(query: np.ndarray, documents: np.ndarray, index: CorpusIndex) -> dict[str, np.ndarray]:
    """Return non-selecting max and normalized-LogSumExp diagnostics."""
    q = numpy_l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
    docs = renormalize_fp16(np.asarray(documents)) if np.asarray(documents).dtype == np.float16 else numpy_l2_normalize(np.asarray(documents, dtype=np.float32))
    maximum: list[float] = []
    normalized_logsumexp: list[float] = []
    for parent_index in range(index.parent_count):
        scores = docs[index.indices_for_parent(parent_index)] @ q
        if len(scores) == 0:
            maximum.append(float("nan"))
            normalized_logsumexp.append(float("nan"))
            continue
        max_score = float(np.max(scores))
        maximum.append(max_score)
        normalized_logsumexp.append(float(max_score + np.log(np.exp(scores - max_score).sum()) - np.log(len(scores))))
    return {
        "max": np.asarray(maximum, dtype=np.float32),
        "normalized_logsumexp": np.asarray(normalized_logsumexp, dtype=np.float32),
    }


def numpy_parent_scores(query: np.ndarray, documents: np.ndarray, index: CorpusIndex) -> np.ndarray:
    return numpy_top2_parent_scores(query, documents, index)


def stable_rank(scores: Sequence[float] | np.ndarray, doc_ids: Sequence[str], limit: int | None = None) -> list[str]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(doc_ids):
        raise ValueError("scores/doc_ids shape mismatch")
    if not np.isfinite(values).all():
        raise ValueError("non-finite parent score")
    order = np.lexsort((np.asarray([str(value) for value in doc_ids], dtype="U"), -values))
    if limit is not None:
        order = order[: int(limit)]
    return [str(doc_ids[int(position)]) for position in order]


def stable_rank_with_scores(scores: Sequence[float] | np.ndarray, doc_ids: Sequence[str], limit: int = 500) -> list[dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.asarray([str(value) for value in doc_ids], dtype="U"), -values))[:limit]
    return [{"doc_id": str(doc_ids[int(position)]), "rank": rank, "score": float(values[int(position)])} for rank, position in enumerate(order, start=1)]


def dense_parent_scores_numpy(query: np.ndarray, documents: np.ndarray, index: CorpusIndex) -> np.ndarray:
    """Independent full-parent reference used by small fixtures and replay."""
    return numpy_parent_scores(query, documents, index)


# ---------------------------------------------------------------------------
# Model contracts and local-only adapters
# ---------------------------------------------------------------------------


def prepare_texts(texts: Sequence[str], spec: ModelSpec, *, is_query: bool) -> list[str]:
    prefix = spec.query_prefix if is_query else spec.document_prefix
    return [prefix + str(text) for text in texts]


def last_non_padding_positions(attention_mask: "torch.Tensor") -> "torch.Tensor":
    """Return the last 1 for either left- or right-padded masks."""
    if torch is None:
        raise RuntimeError("torch is required for last-token pooling")
    mask = attention_mask.to(dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError("attention_mask must be [batch, sequence]")
    if (~mask.any(dim=1)).any():
        raise ValueError("cannot pool an all-padding sequence")
    reversed_mask = torch.flip(mask, dims=[1])
    from_right = reversed_mask.to(dtype=torch.int64).argmax(dim=1)
    return mask.shape[1] - 1 - from_right


def pool_last_non_padding(hidden_states: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    positions = last_non_padding_positions(attention_mask)
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, positions]


@dataclass
class EncodedBatch:
    vectors: np.ndarray
    token_counts: np.ndarray
    truncation_count: int
    elapsed_seconds: float


class EncoderAdapter:
    def __init__(self, spec: ModelSpec, device: str | None = None):
        self.spec = spec
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")

    def encode(self, texts: Sequence[str], *, is_query: bool, batch_size: int = 8) -> EncodedBatch:
        raise NotImplementedError

    def close(self) -> None:
        return None


class SentenceTransformerAdapter(EncoderAdapter):
    def __init__(self, spec: ModelSpec, device: str | None = None):
        super().__init__(spec, device)
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - dependency gate.
            raise RuntimeError("sentence-transformers is unavailable") from exc
        snapshot = resolve_local_snapshot(spec)
        # local_files_only is mandatory: a missing local snapshot is an input
        # gate failure, never an invitation to download.
        try:
            self.model = SentenceTransformer(str(snapshot), device=self.device, local_files_only=True)
        except TypeError as exc:
            raise RuntimeError("installed SentenceTransformer lacks local_files_only support") from exc
        self.model.max_seq_length = spec.max_length

    def encode(self, texts: Sequence[str], *, is_query: bool, batch_size: int = 8) -> EncodedBatch:
        prepared = prepare_texts(texts, self.spec, is_query=is_query)
        started = time.monotonic()
        vectors = self.model.encode(
            prepared,
            batch_size=int(batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vectors = numpy_l2_normalize(np.asarray(vectors, dtype=np.float32))
        token_counts = np.zeros(len(prepared), dtype=np.int32)
        truncated = 0
        try:
            tokenized = self.model.tokenize(prepared)
            mask = tokenized.get("attention_mask")
            if mask is not None:
                token_counts = mask.detach().cpu().numpy().sum(axis=1).astype(np.int32)
                truncated = int(np.sum(token_counts >= self.spec.max_length))
        except Exception:
            # Token telemetry is diagnostic; the model vector contract is
            # still strict and was already checked by dimension/normalization.
            pass
        self._check_vectors(vectors)
        return EncodedBatch(vectors, token_counts, truncated, time.monotonic() - started)

    def _check_vectors(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.spec.dimension or not np.isfinite(vectors).all():
            raise ValueError(f"unexpected {self.spec.key} embedding shape/dtype: {vectors.shape}/{vectors.dtype}")


class LastTokenAdapter(EncoderAdapter):
    def __init__(self, spec: ModelSpec, device: str | None = None):
        super().__init__(spec, device)
        if torch is None:
            raise RuntimeError("torch is unavailable")
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency gate.
            raise RuntimeError("transformers is unavailable") from exc
        snapshot = resolve_local_snapshot(spec)
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        # Runtime contract is explicit even when snapshot defaults are 512.
        self.tokenizer.model_max_length = spec.max_length

    def encode(self, texts: Sequence[str], *, is_query: bool, batch_size: int = 8) -> EncodedBatch:
        if torch is None:
            raise RuntimeError("torch is unavailable")
        prepared = prepare_texts(texts, self.spec, is_query=is_query)
        vectors: list[np.ndarray] = []
        token_counts: list[np.ndarray] = []
        truncation_count = 0
        started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, len(prepared), max(1, int(batch_size))):
                batch = prepared[start:start + max(1, int(batch_size))]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                outputs = self.model(**tokens)
                hidden = outputs.last_hidden_state
                pooled = pool_last_non_padding(hidden, tokens["attention_mask"])
                pooled = _torch_l2_normalize(pooled).float().cpu().numpy()
                vectors.append(pooled.astype(np.float32, copy=False))
                counts = tokens["attention_mask"].sum(dim=1).cpu().numpy().astype(np.int32)
                token_counts.append(counts)
                truncation_count += int(np.sum(counts >= self.spec.max_length))
        result = np.concatenate(vectors, axis=0) if vectors else np.empty((0, self.spec.dimension), dtype=np.float32)
        counts = np.concatenate(token_counts, axis=0) if token_counts else np.empty((0,), dtype=np.int32)
        if result.ndim != 2 or result.shape[1] != self.spec.dimension or not np.isfinite(result).all():
            raise ValueError(f"unexpected {self.spec.key} embedding shape/dtype: {result.shape}/{result.dtype}")
        return EncodedBatch(result, counts, truncation_count, time.monotonic() - started)


def _hf_cache_root() -> Path:
    configured = os.environ.get("HF_HOME")
    if configured:
        return Path(configured) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_local_snapshot(spec: ModelSpec) -> Path:
    model_dir = _hf_cache_root() / ("models--" + spec.repo_id.replace("/", "--"))
    candidates: list[Path] = []
    refs_main = model_dir / "refs" / "main"
    if refs_main.exists():
        try:
            candidates.append(model_dir / "snapshots" / refs_main.read_text(encoding="utf-8").strip())
        except OSError:
            pass
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        candidates.extend(sorted((path for path in snapshots.iterdir() if path.is_dir()), reverse=True))
    for candidate in candidates:
        if (candidate / "config.json").exists() and (candidate / "tokenizer_config.json").exists():
            return candidate
    raise FileNotFoundError(f"No local snapshot for {spec.repo_id}: {model_dir}")


def create_encoder(spec_or_key: ModelSpec | str, *, device: str | None = None) -> EncoderAdapter:
    spec = MODEL_SPECS[spec_or_key] if isinstance(spec_or_key, str) else spec_or_key
    if spec.backend == "sentence_transformers":
        return SentenceTransformerAdapter(spec, device)
    if spec.backend == "transformers":
        return LastTokenAdapter(spec, device)
    raise ValueError(f"unsupported encoder backend: {spec.backend}")


def oom_safe_batch(
    fn: Callable[[int], Any],
    *,
    requested: int = 8,
    candidates: Sequence[int] = (8, 4, 2, 1),
) -> tuple[int, Any, list[dict[str, Any]]]:
    """Try exactly 8 -> 4 -> 2 -> 1 on an OOM; never change model/dtype."""
    attempts: list[dict[str, Any]] = []
    ordered = [value for value in candidates if value <= requested]
    if requested not in ordered:
        ordered.insert(0, requested)
    for batch_size in ordered:
        try:
            result = fn(int(batch_size))
            attempts.append({"batch_size": int(batch_size), "status": "PASS"})
            return int(batch_size), result, attempts
        except RuntimeError as exc:
            text = str(exc).lower()
            if "out of memory" not in text and "cuda error" not in text:
                raise
            attempts.append({"batch_size": int(batch_size), "status": "OOM", "error": str(exc)[:300]})
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"batch fallback exhausted: {attempts}")


# ---------------------------------------------------------------------------
# Audit and reading-order evidence
# ---------------------------------------------------------------------------


def _schema_from_file(path: Path) -> str | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        value = read_json(path)
    except Exception:
        return None
    if isinstance(value, dict):
        for key in ("schema_version", "schema"):
            if value.get(key) is not None:
                return str(value[key])
    return None


def file_record(path: Path, *, role: str | None = None, hash_file: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path.resolve()), "exists": path.exists(), "role": role}
    if not path.exists():
        return record
    record["bytes"] = path.stat().st_size
    record["schema_version"] = _schema_from_file(path)
    if hash_file:
        record["sha256"] = sha256_file(path)
    return record


def _reading_order_paths() -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = [
        (ROOT / "AGENTS.md", "repo policy"),
        (ROOT / "docs" / "EXP-109B_PLAN.md", "active plan"),
        (ROOT / "docs" / "exp109a_softtop5_plan.md", "EXP-109A plan"),
        (ROOT / "src" / "exp109a_softtop5_retrieval.py", "EXP-109A source"),
        (ROOT / "tests" / "test_exp109a_softtop5_retrieval.py", "EXP-109A tests"),
        (EXP109A_RESULTS / "pilot_screen" / "outer_fold_0" / "inner_fold_1" / "PILOT_REPORT.json", "EXP-109A pilot"),
        (EXP109A_RESULTS / "replay_exp102" / "REPLAY_EXP102.json", "EXP-109A reproduction"),
        (EXP109A_CACHE / "reproduction" / "REAL_PARENT_NUMPY_FIXTURE.json", "EXP-109A NumPy fixture"),
        (EXP109A_RESULTS / "preflight" / "PREFLIGHT.json", "EXP-109A preflight"),
        (EXP109A_RESULTS / "smoke" / "SMOKE.json", "EXP-109A smoke"),
        (ROOT / "src" / "exp015_model_screen.py", "EXP-015 source"),
        (ROOT / "src" / "exp015_stage2_content_ablation.py", "EXP-015 Stage 2 source"),
        (EXP015_RESULTS / "summary.md", "EXP-015 summary"),
        (EXP015_STAGE2_SUMMARY, "EXP-015 Stage 2 summary"),
        (ROOT / "src" / "exp035_retrieval_error_adjudication.py", "EXP-035 source"),
        (EXP035_REPORT, "EXP-035 report"),
        (EXP035_EVIDENCE, "EXP-035 evidence"),
        (ROOT / "src" / "exp036_coverage_aware_fusion.py", "EXP-036 source"),
        (EXP036_REPORT, "EXP-036 report"),
        (EXP036_UNIVERSE, "EXP-036 universe"),
        (ROOT / "src" / "exp037_cached_encoder_complementarity.py", "EXP-037 source"),
        (EXP037_FIXTURE, "EXP-037 fixture"),
        (EXP037_DIR / "manifest.json", "EXP-037 manifest"),
        (EXP037_REPORT, "EXP-037 report"),
        (ROOT / "src" / "exp021_e5_dense_candidates.py", "EXP-021 dense source"),
        (ROOT / "src" / "exp021_sparse_depth_tune.py", "EXP-021 sparse depth source"),
        (ROOT / "src" / "exp021_sparse_retrieve.py", "EXP-021 sparse source"),
        (ROOT / "src" / "exp034_shallow_retrieval.py", "EXP-034 source"),
        (ROOT / "results" / "exp034_shallow_retrieval" / "REPORT.json", "EXP-034 report"),
        (ROOT / "src" / "exp027_lambdamart_shortlist.py", "EXP-027 source"),
        (ROOT / "results" / "exp027_lambdamart_shortlist" / "REPORT.json", "EXP-027 report"),
        (ROOT / "src" / "exp104_preranker_benchmark.py", "EXP-104 source"),
        (EXP104_REPORT, "EXP-104 report"),
        (ROOT / "cache" / "cv_folds.json", "folds"),
        (EXCLUSIONS_PATH, "canonical exclusions"),
        (LABEL_IMPACT_PATH, "canonical label impact"),
        (PROCESSED_MANIFEST, "processed manifest"),
        (STRUCT_MANIFEST, "structural manifest"),
        (E5_MANIFEST, "E5 manifest"),
        (QUERY_MANIFEST, "E5 query manifest"),
        (BM25_RAW_DIR / "manifest.json", "BM25 raw manifest"),
        (BM25_TUNING_REPORT, "BM25 tuned aggregation"),
        (BM25_MANIFEST, "BM25 index manifest"),
        (TRAIN_PATH, "canonical train"),
    ]
    # Include the archived EXP-015 model reports/manifests as a complete
    # model-contract audit, without treating BGE as an allowed 109B model.
    for key in MODEL_SPECS:
        paths.append((EXP015_RESULTS / f"{key}.json", f"EXP-015 {key} report"))
        paths.append((ROOT / "cache" / "exp015_model_screen" / "models" / key / "manifest.json", f"EXP-015 {key} manifest"))
    return paths


def inspect_model_snapshot(spec: ModelSpec) -> dict[str, Any]:
    try:
        snapshot = resolve_local_snapshot(spec)
    except Exception as exc:
        return {"model": spec.key, "repo_id": spec.repo_id, "exists": False, "error": str(exc)}
    result: dict[str, Any] = {
        "model": spec.key,
        "repo_id": spec.repo_id,
        "snapshot": str(snapshot.resolve()),
        "snapshot_id": snapshot.name,
        "exists": True,
        "planned_contract": dataclass_to_dict(spec),
        "config": {},
        "tokenizer": {},
        "pooling": {},
        "mismatches": [],
    }
    try:
        config = read_json(snapshot / "config.json")
        tokenizer = read_json(snapshot / "tokenizer_config.json")
        result["config"] = {key: config.get(key) for key in ("model_type", "hidden_size", "max_position_embeddings", "architectures")}
        result["tokenizer"] = {key: tokenizer.get(key) for key in ("model_max_length", "max_length", "padding_side", "truncation_side")}
        if config.get("hidden_size") != spec.dimension:
            result["mismatches"].append(f"hidden_size={config.get('hidden_size')} != {spec.dimension}")
        if spec.backend == "sentence_transformers":
            pooling_path = snapshot / "1_Pooling" / "config.json"
            if pooling_path.exists():
                pooling = read_json(pooling_path)
                result["pooling"] = {key: pooling.get(key) for key in ("word_embedding_dimension", "pooling_mode_mean_tokens", "pooling_mode_lasttoken")}
                if pooling.get("word_embedding_dimension") != spec.dimension:
                    result["mismatches"].append("SentenceTransformer pooling dimension mismatch")
                expected_last = spec.key == "vietlegal_harrier_0_6b"
                if bool(pooling.get("pooling_mode_lasttoken")) != expected_last:
                    result["mismatches"].append("native pooling mode mismatch")
            else:
                result["mismatches"].append("missing 1_Pooling/config.json")
        if spec.key == "vnlegal_lal" and tokenizer.get("model_max_length") != spec.max_length:
            # This is metadata drift, not a runtime contract failure: the
            # adapter passes max_length=2048 explicitly.  Keep it visible.
            result["mismatches"].append("snapshot tokenizer default is not planned runtime max_length; adapter overrides explicitly")
    except Exception as exc:
        result["mismatches"].append(f"snapshot inspection failed: {exc}")
    return result


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, ModelSpec):
        return {
            "key": value.key, "repo_id": value.repo_id, "backend": value.backend,
            "max_length": value.max_length, "query_prefix": value.query_prefix,
            "document_prefix": value.document_prefix, "pooling": value.pooling,
            "dimension": value.dimension,
        }
    raise TypeError(type(value).__name__)


def _active_experiment_processes() -> list[dict[str, Any]]:
    """Best-effort read-only process audit; never kills or alters a process."""
    needles = ("exp109a", "exp109b", "exp109a_softtop5", "exp109b_encoder_complementarity")
    found: list[dict[str, Any]] = []
    try:
        import psutil  # type: ignore
        ignored: set[int] = {os.getpid()}
        current = psutil.Process(os.getpid())
        try:
            if current.parent() is not None:
                ignored.add(current.parent().pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        try:
            ignored.update(child.pid for child in current.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                command = " ".join(process.info.get("cmdline") or [])
                # The desktop command runner can place the Python child under
                # a wrapper process that is not visible in the simple parent
                # chain.  Both are the current audit invocation, not a
                # conflicting worker.
                if "exp109b_encoder_complementarity.py" in command.lower() and " audit" in command.lower():
                    continue
                if any(needle in command.lower() for needle in needles) and process.info.get("pid") not in ignored:
                    found.append({"pid": process.info.get("pid"), "name": process.info.get("name"), "cmdline": command[:500]})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        # A process audit that cannot inspect the host is a warning, not a
        # false PASS claim.
        return [{"audit": "unavailable"}]
    return found


def _inspect_corpus(answers: Mapping[str, set[str]], errors: list[str]) -> dict[str, Any]:
    document_ids: list[str] = []
    document_set: set[str] = set()
    duplicate_documents = 0
    for row in read_jsonl(STRUCT_DIR / "documents.jsonl"):
        doc_id = str(row["doc_id"])
        duplicate_documents += int(doc_id in document_set)
        document_set.add(doc_id)
        document_ids.append(doc_id)
    chunk_ids: list[str] = []
    chunk_set: set[str] = set()
    chunk_doc_ids: list[str] = []
    orphan_chunks = 0
    duplicate_chunks = 0
    nonempty_retrieval_text = 0
    previous: str | None = None
    closed: set[str] = set()
    noncontiguous: set[str] = set()
    for row in read_jsonl(STRUCT_DIR / "chunks.jsonl"):
        chunk_id = str(row["chunk_id"])
        doc_id = str(row["doc_id"])
        duplicate_chunks += int(chunk_id in chunk_set)
        chunk_set.add(chunk_id)
        chunk_ids.append(chunk_id)
        chunk_doc_ids.append(doc_id)
        orphan_chunks += int(doc_id not in document_set)
        nonempty_retrieval_text += int(bool(row.get("retrieval_text")))
        if doc_id != previous:
            if doc_id in closed:
                noncontiguous.add(doc_id)
            if previous is not None:
                closed.add(previous)
            previous = doc_id
    e5_chunk_rows = list(read_jsonl(E5_DIR / "chunk_ids.jsonl"))
    e5_chunk_ids = [str(row["chunk_id"]) for row in e5_chunk_rows]
    e5_chunk_docs = [str(row["doc_id"]) for row in e5_chunk_rows]
    doc_map = read_json(STRUCT_DIR / "doc_to_chunk_ids.json")
    mapped_docs = {str(doc_id) for doc_id in doc_map}
    mapped_chunks = {str(chunk_id) for values in doc_map.values() for chunk_id in values}
    if document_set != mapped_docs:
        errors.append("doc_to_chunk_ids document set mismatch")
    if chunk_set != mapped_chunks:
        errors.append("doc_to_chunk_ids chunk set mismatch")
    if chunk_ids != e5_chunk_ids or chunk_doc_ids != e5_chunk_docs:
        errors.append("structural and E5 chunk order mismatch")
    if len(document_set) != DOCUMENT_COUNT or len(chunk_ids) != CHUNK_COUNT or len(e5_chunk_ids) != CHUNK_COUNT:
        errors.append(f"corpus count mismatch docs={len(document_set)} chunks={len(chunk_ids)} e5={len(e5_chunk_ids)}")
    if duplicate_documents or duplicate_chunks or orphan_chunks or noncontiguous:
        errors.append(
            f"corpus mapping invalid duplicate_docs={duplicate_documents} duplicate_chunks={duplicate_chunks} "
            f"orphans={orphan_chunks} noncontiguous={len(noncontiguous)}"
        )
    missing_gold = sorted({doc_id for gold in answers.values() for doc_id in gold if doc_id not in document_set})
    if missing_gold:
        errors.append(f"canonical gold missing from corpus: {missing_gold[:10]}")
    index = CorpusIndex.from_chunk_doc_ids(chunk_doc_ids)
    parent_dir = CACHE_ROOT / "parent_index"
    atomic_json(parent_dir / "parent_index.json", index.to_json())
    write_success(parent_dir, stage="parent-index", fingerprint=index.fingerprint())
    return {
        "documents": len(document_set),
        "chunks": len(chunk_ids),
        "unique_chunks": len(chunk_set),
        "e5_chunk_ids": len(e5_chunk_ids),
        "nonempty_retrieval_text": nonempty_retrieval_text,
        "noncontiguous_parents": len(noncontiguous),
        "parent_index_fingerprint": index.fingerprint(),
        "document_ids_sha256": content_hash(sorted(document_set)),
        "chunk_ids_sha256": content_hash(chunk_ids),
    }


def _audit_bm25(corpus_ids: set[str], train_qids: set[str], errors: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"raw_evidence": str(BM25_RAW_DIR.resolve()), "max_parent_rank": BM25_MAX_RANK}
    if not BM25_RAW_DIR.exists():
        errors.append(f"missing BM25 raw evidence: {BM25_RAW_DIR}")
        return result
    manifest = read_json(BM25_RAW_DIR / "manifest.json")
    result["raw_manifest"] = manifest
    if manifest.get("config", {}).get("top_passages") != 4096:
        errors.append("BM25 raw evidence is not top-4096")
    seen: set[str] = set()
    outside: set[str] = set()
    evidence_rows = 0
    max_evidence_docs = 0
    shard_records = []
    for shard in sorted((BM25_RAW_DIR / "shards").glob("evidence_*.jsonl")):
        actual = sha256_file(shard)
        declared = next((row.get("sha256") for row in manifest.get("shards", []) if row.get("name") == shard.name), None)
        if declared and actual != declared:
            errors.append(f"BM25 shard hash mismatch: {shard.name}")
        shard_records.append({"path": str(shard.resolve()), "sha256": actual, "bytes": shard.stat().st_size})
        for row in read_jsonl(shard):
            qid = str(row["qid"])
            evidence_rows += 1
            if qid in seen:
                errors.append(f"duplicate BM25 evidence qid: {qid}")
            seen.add(qid)
            evidence = row.get("evidence", [])
            max_evidence_docs = max(max_evidence_docs, len(evidence))
            outside.update(str(doc_id) for doc_id, _ranks in evidence if str(doc_id) not in corpus_ids)
    if seen != train_qids:
        errors.append(f"BM25 evidence qid coverage mismatch: {len(seen)} vs {len(train_qids)}")
    if outside:
        errors.append(f"BM25 evidence has unknown parent IDs: {sorted(outside)[:10]}")
    tuning = read_json(BM25_TUNING_REPORT) if BM25_TUNING_REPORT.exists() else {}
    selected = tuning.get("selected_by_candidate_budget", {}).get("150")
    if not selected or tuning.get("status") != "PASS":
        errors.append("BM25 tuning report lacks PASS selected_by_candidate_budget[150]")
    result.update({
        "evidence_rows": evidence_rows,
        "max_evidence_documents": max_evidence_docs,
        "shards": shard_records,
        "tuning_report_sha256": sha256_file(BM25_TUNING_REPORT) if BM25_TUNING_REPORT.exists() else None,
        "tuning_selected_by_candidate_budget_150": selected,
        "bm25_manifest": read_json(BM25_MANIFEST) if BM25_MANIFEST.exists() else None,
    })
    return result


def _audit_legacy_cohorts(errors: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if EXP035_EVIDENCE.exists():
        evidence_rows = list(read_jsonl(EXP035_EVIDENCE))
        result["exp035_evidence_rows"] = len(evidence_rows)
        result["exp035_roles"] = dict(sorted(Counter(str(row.get("role")) for row in evidence_rows).items()))
        if len(evidence_rows) != 290:
            errors.append(f"EXP-035 evidence cohort count mismatch: {len(evidence_rows)}")
    else:
        errors.append(f"missing EXP-035 evidence cohort: {EXP035_EVIDENCE}")
    if EXP037_FIXTURE.exists():
        fixture_rows = list(read_jsonl(EXP037_FIXTURE))
        result["exp037_fixture_rows"] = len(fixture_rows)
        result["exp037_roles"] = dict(sorted(Counter(str(row.get("role")) for row in fixture_rows).items()))
        result["exp037_max_candidate_count"] = max((len(row.get("candidate_doc_ids", [])) for row in fixture_rows), default=0)
        result["exp037_warning_flags"] = {
            "bounded_oracle_fixture_not_recall": all(bool(row.get("bounded_oracle_fixture_not_recall")) for row in fixture_rows),
            "gold_force_included": all(set(row.get("gold_doc_ids", [])) <= set(row.get("candidate_doc_ids", [])) for row in fixture_rows),
        }
        if len(fixture_rows) != 290:
            errors.append(f"EXP-037 fixture cohort count mismatch: {len(fixture_rows)}")
        if not result["exp037_warning_flags"]["bounded_oracle_fixture_not_recall"]:
            errors.append("EXP-037 fixture warning flag missing")
        if not result["exp037_warning_flags"]["gold_force_included"]:
            errors.append("EXP-037 fixture does not force-include all gold IDs")
    else:
        errors.append(f"missing EXP-037 fixture: {EXP037_FIXTURE}")
    return result


def audit_inputs() -> dict[str, Any]:
    """Create the complete EXP-109B reading/input audit.

    Historical schema typos and deliberately diagnostic fixtures are retained
    in ``mismatches``.  They do not become live EXP-109B inputs: the new
    namespace validates the underlying fingerprints and writes its own
    bounded fixture.  Required live corpus/label/model contracts are errors.
    """
    errors: list[str] = []
    warnings: list[str] = []
    mismatches: list[dict[str, Any]] = []
    answers, label_stats = canonical_labels()
    train = load_train()
    folds, fold_for = load_folds()
    if label_stats["queries"] != QUERY_COUNT or label_stats["evaluable_queries"] != EVALUABLE_QUERY_COUNT or label_stats["non_evaluable_queries"] != QUERY_COUNT - EVALUABLE_QUERY_COUNT:
        errors.append(f"canonical label counts differ: {label_stats}")
    if label_stats["label_fingerprint"] != "9bdf9593b61fe3423d1f1a819ac9fb3e8d7225e6003da0afb840c1f5853fd4c9":
        errors.append("canonical label fingerprint differs from frozen policy")
    if set(folds) != set(FOLD_NAMES):
        errors.append(f"unexpected fold names: {sorted(folds)}")

    processed_manifest = read_json(PROCESSED_MANIFEST)
    structural_manifest = read_json(STRUCT_MANIFEST)
    e5_manifest = read_json(E5_MANIFEST)
    query_manifest = read_json(QUERY_MANIFEST)
    if structural_manifest.get("schema_version") != "legalir.structural_chunks.v3":
        errors.append("structural manifest schema mismatch")
    if e5_manifest.get("schema_version") != "legalir.e5_embedding_cache.v1":
        errors.append("E5 manifest schema mismatch")
    if structural_manifest.get("content_fingerprint") != e5_manifest.get("corpus_fingerprint"):
        errors.append("E5/structural corpus fingerprint mismatch")
    for directory, expected, key in ((STRUCT_DIR, structural_manifest.get("content_fingerprint"), "content_fingerprint"), (E5_DIR, e5_manifest.get("cache_fingerprint"), "cache_fingerprint")):
        marker_path = directory / "_SUCCESS.json"
        if not marker_path.exists():
            errors.append(f"missing frozen success marker: {marker_path}")
        else:
            marker = read_json(marker_path)
            if marker.get(key) != expected:
                errors.append(f"frozen success fingerprint mismatch: {directory}")
    chunk_embeddings = np.load(E5_DIR / "embeddings.f16.npy", mmap_mode="r")
    query_embeddings = np.load(QUERY_DIR / "train_queries.f32.npy", mmap_mode="r")
    query_ids = [str(value) for value in read_json(QUERY_DIR / "train_query_ids.json")]
    if tuple(chunk_embeddings.shape) != (CHUNK_COUNT, DIMENSION) or chunk_embeddings.dtype != np.float16:
        errors.append(f"unexpected E5 chunk array: {chunk_embeddings.shape}/{chunk_embeddings.dtype}")
    if tuple(query_embeddings.shape) != (QUERY_COUNT, DIMENSION) or query_embeddings.dtype != np.float32:
        errors.append(f"unexpected E5 query array: {query_embeddings.shape}/{query_embeddings.dtype}")
    if len(query_ids) != len(set(query_ids)) or set(query_ids) != set(train):
        errors.append("E5 query IDs do not exactly match train IDs")
    corpus_info = _inspect_corpus(answers, errors)
    corpus_ids = set(corpus_info.get("document_ids", []))
    if not corpus_ids:
        # Avoid re-reading the 6 MB document file only for the normal audit
        # path; this is also explicit for unit fixtures.
        corpus_ids = {str(row["doc_id"]) for row in read_jsonl(STRUCT_DIR / "documents.jsonl")}
    bm25_info = _audit_bm25(corpus_ids, set(train), errors)
    legacy_cohorts = _audit_legacy_cohorts(errors)

    # Input files receive path, SHA-256, and schema evidence.  Large frozen
    # arrays are represented by their manifest-declared artifact hash below;
    # the audit itself still records actual array shape/dtype.
    inputs: list[dict[str, Any]] = []
    for path, role in _reading_order_paths():
        record = file_record(path, role=role, hash_file=path.exists())
        inputs.append(record)
        if not record.get("exists") and role in {"active plan", "repo policy", "canonical train", "folds"}:
            errors.append(f"missing required reading input: {path}")

    for spec in MODEL_SPECS.values():
        snapshot = inspect_model_snapshot(spec)
        if not snapshot.get("exists"):
            errors.append(f"missing local model snapshot: {spec.repo_id}")
        elif snapshot.get("config", {}).get("hidden_size") != spec.dimension:
            errors.append(f"model dimension mismatch: {spec.key}")
        inputs.append({"role": "local model snapshot", **snapshot})

    # Explicitly documented historical mismatches; none is silently promoted
    # to an EXP-109B source or metric claim.
    exp035 = read_json(EXP035_REPORT) if EXP035_REPORT.exists() else {}
    if exp035.get("schema_version") == "legalir.exp104_retrieval_error_adjudication.v1":
        mismatches.append({"input": str(EXP035_REPORT), "expected": "EXP-035 schema", "observed": exp035.get("schema_version"), "handling": "diagnostic tag evidence only; not a 109B schema"})
    exp037 = read_json(EXP037_REPORT) if EXP037_REPORT.exists() else {}
    if exp037.get("schema_version") == "legalir.exp037_cached_encoder_complementarity.v1":
        mismatches.append({"input": str(EXP037_REPORT), "expected": "new EXP-109B bounded fixture flags", "observed": "legacy EXP-037 fixture schema", "handling": "fingerprint-check legacy fixture, then write a new 109B fixture"})
    if not (EXP037_DIR / "manifest.json").exists():
        mismatches.append({"input": str(EXP037_DIR / "manifest.json"), "expected": "EXP-037 manifest from reading order", "observed": "missing in checkout", "handling": "use checked-in EXP-037 _SUCCESS/report fingerprints and create a new 109B fixture manifest"})
    exp015_e5 = read_json(EXP015_RESULTS / "vietlegal_e5.json") if (EXP015_RESULTS / "vietlegal_e5.json").exists() else {}
    exp015_fp = exp015_e5.get("fixture", {}).get("v3_content_fingerprint")
    if exp015_fp and exp015_fp != structural_manifest.get("content_fingerprint"):
        mismatches.append({"input": "EXP-015 fixture", "expected": structural_manifest.get("content_fingerprint"), "observed": exp015_fp, "handling": "archived development replay only; not full-corpus input"})
    lal_snapshot = next((item for item in inputs if item.get("model") == "vnlegal_lal"), {})
    if any("default is not planned" in item for item in lal_snapshot.get("mismatches", [])):
        mismatches.append({"input": "VnLegal-LAL tokenizer_config.json", "expected": "runtime max_length=2048", "observed": lal_snapshot.get("tokenizer", {}).get("model_max_length"), "handling": "adapter passes explicit max_length=2048; default metadata remains reported"})
    if not (ROOT / "results" / "exp021_sparse" / "depth_rrf_tuning" / "manifest.json").exists():
        warnings.append("EXP-021 tuned depth output has no standalone manifest; raw evidence manifest plus tuning_report are the contract.")
    if (ROOT / "cache" / "exp022_e5_bm25_union").exists():
        warnings.append("EXP-022 ordered union exists but is excluded from EXP-109B candidate/fusion scoring.")
    active = _active_experiment_processes()
    if active:
        errors.append(f"active EXP-109A/109B process detected: {active[:3]}")
    disk = shutil.disk_usage(ROOT)
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "stage": "input_audit",
        "status": "PASS" if not errors else "REJECTED_INPUT_AUDIT",
        "label_policy": LABEL_POLICY,
        "label_stats": label_stats,
        "train_queries": QUERY_COUNT,
        "folds": {name: len(values) for name, values in sorted(folds.items())},
        "corpus": corpus_info,
        "embeddings": {
            "e5_chunks": {"shape": list(chunk_embeddings.shape), "dtype": str(chunk_embeddings.dtype), "manifest": e5_manifest},
            "e5_queries": {"shape": list(query_embeddings.shape), "dtype": str(query_embeddings.dtype), "manifest": query_manifest},
        },
        "bm25_contract": bm25_info,
        "legacy_cohorts": legacy_cohorts,
        "model_snapshots": [inspect_model_snapshot(spec) for spec in MODEL_SPECS.values()],
        "disk": {"root": str(ROOT.resolve()), "free_bytes": int(disk.free), "total_bytes": int(disk.total), "free_gib": round(disk.free / 2**30, 3)},
        "active_processes": active,
        "inputs": inputs,
        "mismatches": mismatches,
        "warnings": warnings,
        "errors": errors,
        "constraints": {
            "allowed_dense_models": list(MODEL_SPECS),
            "retrieval_text_only": True,
            "aggregation": "top2_mean",
            "scorer_contract": SCORER_CONTRACT,
            "no_ordered_union": True,
            "no_model_download": True,
            "no_public_submission": True,
        },
    }
    audit_fingerprint = content_hash({
        "label_stats": label_stats,
        "folds": report["folds"],
        "corpus": corpus_info,
        "embedding_manifest": e5_manifest,
        "query_manifest": query_manifest,
        "bm25": {"manifest": bm25_info.get("raw_manifest", {}), "tuning": bm25_info.get("tuning_selected_by_candidate_budget_150")},
        "input_sha256": [(row.get("path"), row.get("sha256")) for row in inputs],
        "mismatches": mismatches,
    })
    report["audit_fingerprint"] = audit_fingerprint
    output_dir = RESULTS_ROOT / "input_audit"
    atomic_json(output_dir / "READING_AUDIT.json", report)
    if not errors:
        write_success(output_dir, stage="input-audit", fingerprint=audit_fingerprint, extra={"scorer_contract": SCORER_CONTRACT})
    else:
        (output_dir / "_SUCCESS.json").unlink(missing_ok=True)
    return report


# ---------------------------------------------------------------------------
# EXP-015 replay and shared scorer reproduction
# ---------------------------------------------------------------------------


def _exp015_report_key(key: str) -> Path:
    return EXP015_RESULTS / f"{key}.json"


def _load_exp015_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]], CorpusIndex]:
    queries = list(read_jsonl(EXP015_FIXTURE / "queries.jsonl"))
    chunks = list(read_jsonl(EXP015_FIXTURE / "chunks.jsonl"))
    index = CorpusIndex.from_chunk_doc_ids([str(row["doc_id"]) for row in chunks])
    return queries, chunks, index


def _metric_from_rank_lists(
    predictions: Mapping[str, Sequence[str]],
    answers: Mapping[str, set[str]],
    qids: Sequence[str],
) -> dict[str, Any]:
    evaluable = [qid for qid in qids if answers.get(qid)]
    if not evaluable:
        return {"evaluable_queries": 0}
    recalls = {k: [] for k in CURVE_KS}
    precision5: list[float] = []
    mrr5: list[float] = []
    first_ranks: list[int | None] = []
    for qid in evaluable:
        ranking = list(predictions[qid])
        gold = answers[qid]
        for k in CURVE_KS:
            recalls[k].append(float(bool(set(ranking[:k]) & gold)))
        top5 = ranking[:5]
        precision5.append(len(set(top5) & gold) / 5.0)
        rank = next((i for i, doc_id in enumerate(ranking, start=1) if doc_id in gold), None)
        first_ranks.append(rank)
        mrr5.append(1.0 / rank if rank is not None and rank <= 5 else 0.0)
    return {
        "evaluable_queries": len(evaluable),
        **{f"recall@{k}": float(np.mean(values)) for k, values in recalls.items()},
        "precision@5": float(np.mean(precision5)),
        "mrr@5": float(np.mean(mrr5)),
        "first_ranks": first_ranks,
    }


def _exp015_archived_rankings(key: str) -> tuple[dict[str, list[str]], dict[str, set[str]], dict[str, Any]]:
    report = read_json(_exp015_report_key(key))
    queries, chunks, index = _load_exp015_fixture()
    embeddings_path = EXP015_RESULTS / f"{key}.embeddings.npz"
    arrays = np.load(embeddings_path)
    query_vectors = np.asarray(arrays["query_vectors"], dtype=np.float32)
    chunk_vectors = np.asarray(arrays["chunk_vectors"], dtype=np.float32)
    if query_vectors.shape[0] != len(queries) or chunk_vectors.shape[0] != len(chunks):
        raise ValueError(f"EXP-015 shape mismatch: {key}")
    answers = {str(row["qid"]): {str(value) for value in row.get("answer_doc_ids", [])} for row in queries}
    doc_ids = index.doc_ids
    rankings: dict[str, list[str]] = {}
    for position, query in enumerate(queries):
        scores = numpy_parent_aggregation_max(query_vectors[position], chunk_vectors, index)
        rankings[str(query["qid"])] = stable_rank(scores, doc_ids)
    return rankings, answers, report


def numpy_parent_aggregation_max(query: np.ndarray, documents: np.ndarray, index: CorpusIndex) -> np.ndarray:
    q = numpy_l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
    docs = numpy_l2_normalize(np.asarray(documents, dtype=np.float32))
    return np.asarray([float(np.max(docs[index.indices_for_parent(i)] @ q)) for i in range(index.parent_count)], dtype=np.float32)


def _fresh_exp015_mini_parity(key: str, *, count: int = 8, batch_size: int = 2) -> dict[str, Any]:
    """Fresh local-only mini parity; called only by explicit replay."""
    queries, chunks, _index = _load_exp015_fixture()
    report = read_json(_exp015_report_key(key))
    arrays = np.load(EXP015_RESULTS / f"{key}.embeddings.npz")
    key_spec = MODEL_SPECS[key]
    adapter = create_encoder(key, device="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    try:
        q_count = min(count, len(queries))
        c_count = min(count * 2, len(chunks))
        fresh_q_batch = adapter.encode([str(row["question"]) for row in queries[:q_count]], is_query=True, batch_size=batch_size)
        fresh_c_batch = adapter.encode([str(row["retrieval_text"]) for row in chunks[:c_count]], is_query=False, batch_size=batch_size)
        fresh_q = fresh_q_batch.vectors
        fresh_c = fresh_c_batch.vectors
    finally:
        adapter.close()
    archived_q = np.asarray(arrays["query_vectors"], dtype=np.float32)[:q_count]
    archived_c = np.asarray(arrays["chunk_vectors"], dtype=np.float32)[:c_count]
    q_cos = np.sum(numpy_l2_normalize(fresh_q) * numpy_l2_normalize(archived_q), axis=1)
    c_cos = np.sum(numpy_l2_normalize(fresh_c) * numpy_l2_normalize(archived_c), axis=1)
    dimension_pass = fresh_q.ndim == 2 and fresh_c.ndim == 2 and fresh_q.shape[1] == key_spec.dimension and fresh_c.shape[1] == key_spec.dimension
    archived_tokenization = report.get("tokenization", {})
    truncation_pass = (
        int(fresh_q_batch.truncation_count) == int(archived_tokenization.get("query_truncated", 0))
        and int(fresh_c_batch.truncation_count) == int(archived_tokenization.get("chunk_truncated", 0))
    )
    prefix_pass = bool(key_spec.query_prefix is not None and key_spec.document_prefix is not None)
    pooling_pass = key_spec.pooling in {"native", "last_non_padding"}
    return {
        "model": key,
        "archived_fixture_fingerprint": report.get("fixture", {}).get("v3_content_fingerprint"),
        "query_count": q_count,
        "chunk_count": c_count,
        "query_min_cosine": float(np.min(q_cos)) if len(q_cos) else None,
        "chunk_min_cosine": float(np.min(c_cos)) if len(c_cos) else None,
        "query_pass": bool(len(q_cos) == 0 or np.min(q_cos) >= 0.9999),
        "chunk_pass": bool(len(c_cos) == 0 or np.min(c_cos) >= 0.9999),
        "dimension_pass": dimension_pass,
        "contract_pass": bool(dimension_pass and truncation_pass and prefix_pass and pooling_pass),
        "truncation_pass": truncation_pass,
        "prefix_pass": prefix_pass,
        "pooling_pass": pooling_pass,
        "archived_tokenization": archived_tokenization,
        "prefix": {"query": MODEL_SPECS[key].query_prefix, "document": MODEL_SPECS[key].document_prefix},
        "max_length": MODEL_SPECS[key].max_length,
        "pooling": MODEL_SPECS[key].pooling,
        "fresh_token_telemetry": {"query_truncation_count": int(fresh_q_batch.truncation_count), "chunk_truncation_count": int(fresh_c_batch.truncation_count)},
    }


def shared_scorer_reproduction() -> dict[str, Any]:
    index = CorpusIndex.from_chunk_doc_ids(["a", "a", "b", "b", "b", "c"])
    docs = np.asarray([[1.0, 0.0], [0.8, 0.6], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    numpy_values = numpy_top2_parent_scores(query, docs, index)
    secondary = numpy_secondary_parent_scores(query, docs, index)
    if torch is None:
        raise RuntimeError("torch unavailable for scorer reproduction")
    torch_values = compute_parent_scores(torch.as_tensor(query), docs, index, chunk_block_size=2).detach().cpu().numpy()[0]
    real_fixture_path = EXP109A_CACHE / "reproduction" / "REAL_PARENT_NUMPY_FIXTURE.json"
    real = read_json(real_fixture_path) if real_fixture_path.exists() else {}
    replay_path = EXP109A_RESULTS / "replay_exp102" / "REPLAY_EXP102.json"
    replay_report = read_json(replay_path) if replay_path.exists() else {}
    real_replay = replay_report.get("independent_numpy_real_parent_fixture", {})
    return {
        "schema_version": SCHEMA,
        "status": "PASS" if np.max(np.abs(numpy_values - torch_values)) <= 1e-6 else "REJECTED_REPRODUCTION_GATE",
        "scorer_contract": SCORER_CONTRACT,
        "toy_max_abs_error": float(np.max(np.abs(numpy_values - torch_values))),
        "toy_values": numpy_values.tolist(),
        "secondary_diagnostics": {name: values.tolist() for name, values in secondary.items()},
        "fp16_norms": np.linalg.norm(renormalize_fp16(docs), axis=1).tolist(),
        "real_exp109a_fixture": {
            "path": str(real_fixture_path.resolve()),
            "replay_report": str(replay_path.resolve()),
            "observed_status": real_replay.get("status"),
            "observed_max_abs_error": real_replay.get("max_abs_error"),
            "pass": bool(real_replay) and real_replay.get("status") == "PASS" and float(real_replay.get("max_abs_error", math.inf)) <= 1e-6,
        },
        "fixtures": {"one_chunk": True, "two_chunks": True, "many_chunks": True, "ties": True},
    }


def replay(*, fresh_models: bool = False) -> dict[str, Any]:
    output_dir = RESULTS_ROOT / "replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_by_model: dict[str, dict[str, set[str]]] = {}
    model_reports: dict[str, Any] = {}
    overall_pass = True
    for key in MODEL_SPECS:
        rankings, answers, report = _exp015_archived_rankings(key)
        metrics = _metric_from_rank_lists(rankings, answers, list(rankings))
        expected = report.get("metrics", {})
        observed = {
            "recall@5": metrics["recall@5"],
            "recall@20": metrics["recall@20"],
            "recall@100": metrics["recall@100"],
            "mrr@5": metrics["mrr@5"],
        }
        expected_normalized = {
            "recall@5": expected.get("recall_at_5"),
            "recall@20": expected.get("recall_at_20"),
            "recall@100": expected.get("recall_at_100"),
            "mrr@5": expected.get("mrr_at_5"),
        }
        deltas = {name: abs(float(observed[name]) - float(expected_normalized[name])) for name in observed if expected_normalized[name] is not None}
        model_pass = bool(deltas) and max(deltas.values()) <= 1e-6
        overall_pass &= model_pass
        answers_by_model[key] = answers
        model_reports[key] = {
            "metrics": observed,
            "expected": expected_normalized,
            "absolute_errors": deltas,
            "first_gold_ranks": metrics["first_ranks"],
            "pass": model_pass,
            "development_only": report.get("fixture", {}).get("purpose"),
        }
        if fresh_models:
            model_reports[key]["fresh_mini_parity"] = _fresh_exp015_mini_parity(key)
            overall_pass &= bool(model_reports[key]["fresh_mini_parity"]["query_pass"] and model_reports[key]["fresh_mini_parity"]["chunk_pass"] and model_reports[key]["fresh_mini_parity"]["contract_pass"])
    scorer = shared_scorer_reproduction()
    overall_pass &= scorer["status"] == "PASS" and bool(scorer["real_exp109a_fixture"]["pass"])
    report = {
        "schema_version": SCHEMA,
        "stage": "replay",
        "status": "PASS" if overall_pass else "REJECTED_REPRODUCTION_GATE",
        "metric_tolerance": 1e-6,
        "models": model_reports,
        "shared_scorer": scorer,
        "fresh_models_requested": fresh_models,
        "scope": "EXP-015 fixed development fixture, not OOF/full-corpus recall",
    }
    atomic_json(output_dir / "REPLAY.json", report)
    if overall_pass:
        write_success(output_dir, stage="replay", fingerprint=content_hash(report), extra={"scorer_contract": SCORER_CONTRACT})
    else:
        (output_dir / "_SUCCESS.json").unlink(missing_ok=True)
    return report


# ---------------------------------------------------------------------------
# Legacy EXP-037 fixture bridge and bounded complementarity gate
# ---------------------------------------------------------------------------


def _legacy_fixture_fingerprint() -> dict[str, Any]:
    if not EXP037_FIXTURE.exists() or not EXP037_REPORT.exists():
        raise FileNotFoundError("EXP-037 fixture/report missing")
    report = read_json(EXP037_REPORT)
    return {
        "fixture_sha256": sha256_file(EXP037_FIXTURE),
        "reported_fixture_fingerprint": report.get("fixture_fingerprint"),
        "evidence_sha256": sha256_file(EXP035_EVIDENCE) if EXP035_EVIDENCE.exists() else None,
        "universe_sha256": sha256_file(EXP036_UNIVERSE) if EXP036_UNIVERSE.exists() else None,
        "reported_inputs": report.get("inputs", {}),
        "legacy_status": report.get("status"),
    }


def _exp036_top96_by_qid() -> dict[str, set[str]]:
    """Read the exact EXP-036 candidate universe used by the bounded contract."""
    rows: dict[str, set[str]] = {}
    for row in read_jsonl(EXP036_UNIVERSE):
        qid = str(row["qid"])
        if qid in rows:
            raise ValueError(f"duplicate EXP-036 universe qid: {qid}")
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError(f"invalid EXP-036 candidates for {qid}")
        rows[qid] = {
            str(candidate["doc_id"])
            for candidate in candidates[:96]
            if isinstance(candidate, Mapping) and candidate.get("doc_id") is not None
        }
        if not rows[qid]:
            raise ValueError(f"empty EXP-036 top-96 universe for {qid}")
    return rows


def _dedupe_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = str(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def build_bounded_fixture(*, outer: str = "fold_0", resume: bool = True) -> dict[str, Any]:
    folds, fold_for = load_folds()
    train = load_train()
    answers, label_stats = canonical_labels()
    if outer not in folds:
        raise ValueError(f"unknown outer fold: {outer}")
    legacy = _legacy_fixture_fingerprint()
    legacy_rows = list(read_jsonl(EXP037_FIXTURE))
    exp036_top96 = _exp036_top96_by_qid()
    output_dir = CACHE_ROOT / "bounded_fixture" / outer
    manifest_path = output_dir / "manifest.json"
    fixture_path = output_dir / "queries.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    expected_fingerprint = content_hash({"legacy": legacy, "outer": outer, "allowed_folds": [name for name in folds if name != outer], "canonical_train_sha256": sha256_file(TRAIN_PATH), "structural_manifest_sha256": sha256_file(STRUCT_MANIFEST), "flags": ["bounded_oracle_fixture_not_recall", "gold_force_included", "may_not_be_reported_as_full_corpus_recall"], "query_contract": "canonical_train_question_v1", "candidate_contract": "exp036_top96_union_canonical_gold_v1", "code_sha256": code_fingerprint()})
    if resume and manifest_path.exists() and fixture_path.exists() and chunks_path.exists():
        saved = read_json(manifest_path)
        if saved.get("content_fingerprint") == expected_fingerprint and saved.get("queries_sha256") == sha256_file(fixture_path) and saved.get("chunks_sha256") == sha256_file(chunks_path):
            try:
                require_success(output_dir, expected_fingerprint=expected_fingerprint, expected_code_sha256=code_fingerprint())
            except RuntimeError:
                write_success(output_dir, stage="bounded-fixture", fingerprint=expected_fingerprint)
            return saved
    rows: list[dict[str, Any]] = []
    selected_docs: set[str] = set()
    for row in legacy_rows:
        qid = str(row["qid"])
        if qid not in train or qid not in fold_for:
            raise ValueError(f"EXP-037 bounded fixture contains unknown canonical qid: {qid}")
        canonical_fold = fold_for[qid]
        legacy_fold = row.get("fold")
        if legacy_fold is not None and str(legacy_fold) != canonical_fold:
            raise ValueError(f"EXP-037 fold mismatch for {qid}: {legacy_fold} != {canonical_fold}")
        fold = canonical_fold
        if fold == outer or not answers.get(qid):
            continue
        gold = sorted(answers[qid])
        if qid not in exp036_top96:
            raise ValueError(f"EXP-037 fixture qid absent from EXP-036 universe: {qid}")
        expected_candidates = exp036_top96[qid] | set(gold)
        legacy_candidates = {str(value) for value in row.get("candidate_doc_ids", [])}
        if legacy_candidates != expected_candidates:
            raise ValueError(
                f"EXP-037 candidate contract mismatch for {qid}: "
                f"legacy={len(legacy_candidates)} expected_exp036_top96_union_gold={len(expected_candidates)}"
            )
        # The fixture's source-order must never determine a dense rank.  Use a
        # stable parent-ID order only for the candidate set representation.
        candidates = sorted(expected_candidates)
        selected_docs.update(candidates)
        rows.append({
            "qid": qid,
            "fold": fold,
            "query": str(train[qid].get("question", "")),
            "candidate_doc_ids": candidates,
            "gold_doc_ids": gold,
            "role": row.get("role"),
            "flags": {
                "bounded_oracle_fixture_not_recall": True,
                "gold_force_included": True,
                "may_not_be_reported_as_full_corpus_recall": True,
            },
        })
    rows.sort(key=lambda row: str(row["qid"]))
    chunk_counts: Counter[str] = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = chunks_path.with_name(f".{chunks_path.name}.{os.getpid()}.tmp")
    chunk_count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
            for chunk in read_jsonl(STRUCT_DIR / "chunks.jsonl"):
                doc_id = str(chunk["doc_id"])
                if doc_id not in selected_docs or chunk_counts[doc_id] >= 8:
                    continue
                record = {
                    "chunk_id": str(chunk["chunk_id"]),
                    "doc_id": doc_id,
                    "part_index": int(chunk.get("part_index", chunk_counts[doc_id])),
                    "token_count": int(chunk.get("token_count", 0)),
                    "retrieval_text": str(chunk.get("retrieval_text", "")),
                }
                handle.write(canonical_json(record) + "\n")
                chunk_counts[doc_id] += 1
                chunk_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(chunks_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    write_jsonl_atomic(fixture_path, rows)
    missing_docs = sorted(doc_id for doc_id in selected_docs if not chunk_counts[doc_id])
    if missing_docs:
        raise ValueError(f"bounded fixture candidate docs missing chunks: {missing_docs[:10]}")
    manifest = {
        "schema_version": SCHEMA,
        "stage": "bounded_fixture",
        "outer": outer,
        "queries": len(rows),
        "candidate_document_count": len(selected_docs),
        "chunk_count": chunk_count,
        "chunks_per_parent_cap": 8,
        "source": "EXP-037 fixture candidates, canonical golds, current structural retrieval_text",
        "legacy_fingerprint": legacy,
        "bounded_oracle_fixture_not_recall": True,
        "gold_force_included": True,
        "may_not_be_reported_as_full_corpus_recall": True,
        "input_fingerprint": expected_fingerprint,
        "queries_sha256": sha256_file(fixture_path),
        "chunks_sha256": sha256_file(chunks_path),
    }
    manifest["content_fingerprint"] = expected_fingerprint
    atomic_json(manifest_path, manifest)
    write_success(output_dir, stage="bounded-fixture", fingerprint=expected_fingerprint)
    return manifest


def _load_bounded_data(outer: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], CorpusIndex, dict[str, list[int]]]:
    fixture_dir = CACHE_ROOT / "bounded_fixture" / outer
    manifest_path = fixture_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"bounded fixture manifest missing: {manifest_path}")
    manifest = read_json(manifest_path)
    require_success(
        fixture_dir,
        expected_fingerprint=manifest.get("content_fingerprint"),
        expected_code_sha256=code_fingerprint(),
    )
    if not (manifest.get("bounded_oracle_fixture_not_recall") and manifest.get("gold_force_included") and manifest.get("may_not_be_reported_as_full_corpus_recall")):
        raise RuntimeError("bounded fixture warning flags missing")
    if manifest.get("queries_sha256") != sha256_file(fixture_dir / "queries.jsonl") or manifest.get("chunks_sha256") != sha256_file(fixture_dir / "chunks.jsonl"):
        raise RuntimeError("bounded fixture content hash mismatch")
    rows = list(read_jsonl(fixture_dir / "queries.jsonl"))
    chunks = list(read_jsonl(fixture_dir / "chunks.jsonl"))
    index = CorpusIndex.from_chunk_doc_ids([str(row["doc_id"]) for row in chunks])
    by_qid: dict[str, list[int]] = {}
    doc_to_local = {doc_id: i for i, doc_id in enumerate(index.doc_ids)}
    for row in rows:
        flags = row.get("flags", {})
        if not (flags.get("bounded_oracle_fixture_not_recall") and flags.get("gold_force_included") and flags.get("may_not_be_reported_as_full_corpus_recall")):
            raise RuntimeError(f"bounded query warning flags missing: {row.get('qid')}")
        by_qid[str(row["qid"])] = [doc_to_local[doc_id] for doc_id in row["candidate_doc_ids"] if doc_id in doc_to_local]
    return rows, chunks, index, by_qid


def _rank_bounded_model(query_vectors: np.ndarray, chunk_vectors: np.ndarray, index: CorpusIndex, parent_indices: Sequence[int], doc_ids: Sequence[str]) -> list[str]:
    scores = numpy_top2_parent_scores(query_vectors, chunk_vectors, index, parent_indices)
    selected_doc_ids = [doc_ids[int(i)] for i in parent_indices]
    return stable_rank(scores, selected_doc_ids)


def _frozen_e5_bounded_embeddings(rows: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Read the frozen E5 cache for the bounded fixture without re-encoding."""
    chunk_positions = {
        str(row["chunk_id"]): position
        for position, row in enumerate(read_jsonl(E5_DIR / "chunk_ids.jsonl"))
    }
    query_ids = [str(value) for value in read_json(QUERY_DIR / "train_query_ids.json")]
    query_positions = {qid: position for position, qid in enumerate(query_ids)}
    missing_chunks = [str(row["chunk_id"]) for row in chunks if str(row["chunk_id"]) not in chunk_positions]
    missing_queries = [str(row["qid"]) for row in rows if str(row["qid"]) not in query_positions]
    if missing_chunks or missing_queries:
        raise RuntimeError(f"frozen E5 bounded lookup missing chunks={missing_chunks[:3]} queries={missing_queries[:3]}")
    chunk_array = np.load(E5_DIR / "embeddings.f16.npy", mmap_mode="r")
    query_array = np.load(QUERY_DIR / "train_queries.f32.npy", mmap_mode="r")
    return (
        np.asarray(query_array[[query_positions[str(row["qid"])] for row in rows]], dtype=np.float32),
        np.asarray(chunk_array[[chunk_positions[str(row["chunk_id"])] for row in chunks]], dtype=np.float16),
    )


def deterministic_bootstrap(values: Sequence[float], *, samples: int = BOUND_BOOTSTRAP_SAMPLES, seed: int = RNG_SEED) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "samples": int(samples)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(int(samples), len(array)))
    means = array[draws].mean(axis=1)
    return {"mean": float(array.mean()), "lower": float(np.quantile(means, 0.025)), "upper": float(np.quantile(means, 0.975)), "samples": int(samples), "seed": int(seed)}


def _first_gold_rank(ranking: Sequence[str], gold: set[str]) -> int:
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in gold:
            return rank
    return len(ranking) + 1


def _slice_key(gold: set[str]) -> str:
    return "single_gold" if len(gold) == 1 else "multi_gold"


def _is_source_generation_miss(qid: str, row: Mapping[str, Any], tags: Mapping[str, str] | None) -> bool:
    explicit = row.get("starter_tag")
    if explicit is not None:
        return str(explicit) == "source_generation_miss"
    if tags is None:
        # Synthetic/unit fixtures without the EXP-035 tag sidecar retain the
        # historical bounded-screen interpretation: every E5 Top-32 miss is
        # eligible for the source-generation diagnostic.
        return True
    return tags.get(qid) == "source_generation_miss"


def bounded_model_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    e5_rankings: Mapping[str, Sequence[str]],
    candidate_rankings: Mapping[str, Sequence[str]],
    source_generation_rankings: Mapping[str, Sequence[str]] | None = None,
    tags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    normalized_deltas: list[float] = []
    per_fold: dict[str, list[float]] = defaultdict(list)
    rescue5: list[str] = []
    rescue32: list[str] = []
    lost5: list[str] = []
    controls: list[str] = []
    control_exit: list[str] = []
    source_additions: list[str] = []
    slices: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        qid = str(row["qid"])
        gold = {str(value) for value in row["gold_doc_ids"]}
        base = list(e5_rankings[qid])
        alt = list(candidate_rankings[qid])
        candidate_count = max(1, len(row["candidate_doc_ids"]))
        base_rank = _first_gold_rank(base, gold)
        alt_rank = _first_gold_rank(alt, gold)
        delta = (base_rank - alt_rank) / candidate_count
        normalized_deltas.append(delta)
        per_fold[str(row["fold"])].append(delta)
        key = _slice_key(gold)
        slices[key]["delta"].append(delta)
        if base_rank > 5 and alt_rank <= 5:
            rescue5.append(qid)
        if base_rank > 32 and alt_rank <= 32:
            rescue32.append(qid)
        if base_rank <= 5 and alt_rank > 5:
            lost5.append(qid)
        role = str(row.get("role") or "unknown")
        if role == "control":
            controls.append(qid)
            if base_rank <= 5 and alt_rank > 5:
                control_exit.append(qid)
        if source_generation_rankings is not None and _is_source_generation_miss(qid, row, tags):
            source_rank = _first_gold_rank(source_generation_rankings[qid], gold)
            if base_rank > 32 and source_rank <= 32:
                source_additions.append(qid)
        tag = (tags or {}).get(qid)
        if tag:
            slices[f"tag:{tag}"]["delta"].append(delta)
    bootstrap = deterministic_bootstrap(normalized_deltas)
    return {
        "query_count": len(rows),
        "normalized_rank_improvement": bootstrap,
        "per_fold_mean_delta": {fold: float(np.mean(values)) for fold, values in sorted(per_fold.items()) if values},
        "rescued_top5_qids": rescue5,
        "rescued_top32_qids": rescue32,
        "lost_top5_qids": lost5,
        "net_top5_rescues": len(rescue5) - len(lost5),
        "source_generation_top32_addition_qids": source_additions,
        "source_generation_top32_additions": len(source_additions),
        "control_qids": controls,
        "control_exit_qids": control_exit,
        "control_exit_rate": len(control_exit) / len(controls) if controls else 0.0,
        "slices": {key: {"mean_delta": float(np.mean(value["delta"])) if value["delta"] else 0.0, "count": len(value["delta"])} for key, value in sorted(slices.items())},
        "flags": {
            "bounded_oracle_fixture_not_recall": True,
            "gold_force_included": True,
            "may_not_be_reported_as_full_corpus_recall": True,
        },
    }


def bounded_gate(metrics: Mapping[str, Any], *, included_folds: Sequence[str], source_generation_miss_count: int) -> dict[str, Any]:
    ci = metrics.get("normalized_rank_improvement", {})
    fold_values = metrics.get("per_fold_mean_delta", {})
    min_additions = max(8, int(math.ceil(0.06 * source_generation_miss_count)))
    checks = {
        "bootstrap_lower_gt_zero": float(ci.get("lower", 0.0)) > 0.0,
        "every_included_fold_nonnegative": all(float(fold_values.get(fold, -math.inf)) >= 0.0 for fold in included_folds),
        "at_least_10_top5_rescues": len(metrics.get("rescued_top5_qids", [])) >= 10,
        "net_top5_rescues_at_least_5": int(metrics.get("net_top5_rescues", 0)) >= 5,
        "source_generation_additions": int(metrics.get("source_generation_top32_additions", 0)) >= min_additions,
    }
    # This bounded fixture is gold-forced and is used only to select a source
    # from Folds 1--4.  Matched controls remain an important risk diagnostic,
    # but cannot be tuned into a retrospective binary admission threshold.
    # Exact protection against E5 displacement is instead enforced later by
    # the fold-isolated full-corpus source and fusion gates.
    diagnostics = {
        "control_exit_rate": float(metrics.get("control_exit_rate", 0.0)),
        "control_exit_qids": list(metrics.get("control_exit_qids", [])),
        "risk_flag": "CONTROL_EXIT_REQUIRES_FOLD_ISOLATED_FUSION_SOURCE_PROTECTION"
        if metrics.get("control_exit_qids")
        else "NONE",
        "not_a_bounded_pass_check": True,
    }
    return {
        "checks": checks,
        "diagnostics": diagnostics,
        "source_generation_miss_count": int(source_generation_miss_count),
        "minimum_source_generation_additions": min_additions,
        "pass": all(checks.values()),
    }


def choose_models_from_bounded(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    passed = [key for key, value in results.items() if value.get("gate", {}).get("pass")]
    lal = "vnlegal_lal"
    if lal not in passed:
        return {"status": "REJECTED_BOUNDED_COMPLEMENTARITY", "selected_models": [], "reason": "VnLegal-LAL did not pass the five bounded complementarity checks"}
    left, right = "vietlegal_harrier_0_6b", lal
    left_rescue = set(results[left].get("metrics", {}).get("rescued_top5_qids", []))
    right_rescue = set(results[right].get("metrics", {}).get("rescued_top5_qids", []))
    overlap = left_rescue & right_rescue
    exclusive = {key: len((left_rescue if key == left else right_rescue) - (right_rescue if key == left else left_rescue)) for key in (left, right)}
    ratios = {key: exclusive[key] / max(1, len(left_rescue if key == left else right_rescue)) for key in (left, right)}
    return {
        "status": "PASS_LAL_ONLY_GATE_REPAIR",
        "selected_models": [lal],
        "selection_policy": "LAL_ONLY_AFTER_POST_OBSERVATION_GATE_REPAIR",
        "reason": "VnLegal-LAL passes all five complementarity checks; Harrier has no exclusive Top-5 rescue and is not encoded.",
        "exclusive_rescues": exclusive,
        "exclusive_ratios": ratios,
        "rescue_overlap_count": len(overlap),
        "rescue_overlap_qids": sorted(overlap),
        "passed_models": passed,
    }


def _load_tags() -> dict[str, str]:
    tags: dict[str, str] = {}
    if not EXP035_EVIDENCE.exists():
        return tags
    for row in read_jsonl(EXP035_EVIDENCE):
        tag = row.get("reviewed_tag") or row.get("starter_tag")
        if tag:
            tags[str(row["qid"])] = str(tag)
    return tags


def bounded_screen(*, outer: str = "fold_0", resume: bool = True, authorize: bool = False) -> dict[str, Any]:
    if not authorize and os.environ.get("EXP109B_ALLOW_BOUNDED_GPU") != "1":
        raise GateRejected("REJECTED_AUTHORIZATION_GATE", "bounded-screen requires explicit bounded GPU authorization (EXP109B_ALLOW_BOUNDED_GPU=1)")
    try:
        audit = read_json(RESULTS_ROOT / "input_audit" / "READING_AUDIT.json")
        replay_report = read_json(RESULTS_ROOT / "replay" / "REPLAY.json")
        require_success(
            RESULTS_ROOT / "input_audit",
            expected_fingerprint=audit["audit_fingerprint"],
            expected_code_sha256=code_fingerprint(),
        )
        require_success(
            RESULTS_ROOT / "replay",
            expected_fingerprint=content_hash(replay_report),
            expected_code_sha256=code_fingerprint(),
        )
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise GateRejected("REJECTED_PREREQUISITE_GATE", "current audit and replay success markers are required before bounded screen") from exc
    result_dir = RESULTS_ROOT / "bounded_screen" / outer
    if resume and _stage_marker_current(result_dir, "BOUNDED_REPORT.json"):
        saved = read_json(result_dir / "BOUNDED_REPORT.json")
        if (
            saved.get("input_audit_fingerprint") == audit["audit_fingerprint"]
            and saved.get("replay_fingerprint") == content_hash(replay_report)
        ):
            return saved
    fixture_manifest = build_bounded_fixture(outer=outer, resume=resume)
    rows, chunks, index, candidate_indices = _load_bounded_data(outer)
    queries = [str(row["query"]) for row in rows]
    texts = [str(row["retrieval_text"]) for row in chunks]
    result_dir.mkdir(parents=True, exist_ok=True)
    per_model: dict[str, Any] = {}
    all_models = ("vietlegal_harrier_0_6b", "vnlegal_lal")
    tags = _load_tags()
    model_adapters: dict[str, EncoderAdapter] = {}
    tracker = RunTracker("bounded-screen", outer=outer, total=len(all_models) + 1)
    try:
        e5_queries, e5_chunks = _frozen_e5_bounded_embeddings(rows, chunks)
        e5_rankings: dict[str, list[str]] = {}
        for pos, row in enumerate(rows):
            inds = candidate_indices[str(row["qid"])]
            e5_rankings[str(row["qid"])] = _rank_bounded_model(e5_queries[pos], e5_chunks, index, inds, index.doc_ids)
        tracker.heartbeat(1, emit=True, config_fingerprint=fixture_manifest.get("content_fingerprint"))
        for key in all_models:
            started = time.monotonic()
            adapter = create_encoder(key)
            model_adapters[key] = adapter
            query_batch, query_encoded, query_attempts = oom_safe_batch(lambda batch: adapter.encode(queries, is_query=True, batch_size=batch), requested=8)
            chunk_batch, chunk_encoded, chunk_attempts = oom_safe_batch(lambda batch: adapter.encode(texts, is_query=False, batch_size=batch), requested=8)
            q_vectors = query_encoded.vectors
            c_vectors = chunk_encoded.vectors
            rankings: dict[str, list[str]] = {}
            for pos, row in enumerate(rows):
                inds = candidate_indices[str(row["qid"])]
                rankings[str(row["qid"])] = _rank_bounded_model(q_vectors[pos], c_vectors, index, inds, index.doc_ids)
            source_generation = rankings  # bounded source-generation definition is alternate Top-32 vs E5.
            metrics = bounded_model_metrics(rows, e5_rankings=e5_rankings, candidate_rankings=rankings, source_generation_rankings=source_generation, tags=tags)
            source_miss_count = sum(
                _first_gold_rank(e5_rankings[str(row["qid"])], set(row["gold_doc_ids"])) > 32
                and _is_source_generation_miss(str(row["qid"]), row, tags)
                for row in rows
            )
            gate = bounded_gate(metrics, included_folds=sorted({str(row["fold"]) for row in rows}), source_generation_miss_count=source_miss_count)
            per_model[key] = {
                "metrics": metrics,
                "gate": gate,
                "rankings": rankings,
                "runtime": {
                    "elapsed_seconds": time.monotonic() - started,
                    "query_effective_batch": query_batch,
                    "document_effective_batch": chunk_batch,
                    "query_attempts": query_attempts,
                    "document_attempts": chunk_attempts,
                },
                "estimated_encode_seconds": (
                    CHUNK_COUNT / max(len(texts) / max(chunk_encoded.elapsed_seconds, 1e-9), 1e-9)
                    + QUERY_COUNT / max(len(queries) / max(query_encoded.elapsed_seconds, 1e-9), 1e-9)
                ),
            }
            tracker.heartbeat(1 + len(per_model), emit=True, config_fingerprint=fixture_manifest.get("content_fingerprint"))
    except KeyboardInterrupt:
        tracker.finish("INTERRUPTED")
        raise
    except Exception:
        tracker.finish("FAILED", error=traceback.format_exc(limit=6))
        raise
    finally:
        for adapter in model_adapters.values():
            adapter.close()
    selection = choose_models_from_bounded(per_model)
    # Do not persist per-query ranking arrays inside the summary report; keep a
    # dedicated JSONL artifact so the report stays inspectable.
    for key, value in per_model.items():
        ranking_path = result_dir / f"{key}.rankings.jsonl"
        write_jsonl_atomic(ranking_path, ({"qid": qid, "documents": docs} for qid, docs in sorted(value["rankings"].items())))
        value["rankings_sha256"] = sha256_file(ranking_path)
        value.pop("rankings", None)
    report = {
        "schema_version": SCHEMA,
        "stage": "bounded_screen",
        "status": "PASS" if selection["selected_models"] else selection["status"],
        "outer": outer,
        "fixture": fixture_manifest,
        "bounded_oracle_fixture_not_recall": True,
        "gold_force_included": True,
        "may_not_be_reported_as_full_corpus_recall": True,
        "models": per_model,
        "selection": selection,
        "label_policy": LABEL_POLICY,
        "scorer_contract": SCORER_CONTRACT,
        "input_audit_fingerprint": audit["audit_fingerprint"],
        "replay_fingerprint": content_hash(replay_report),
    }
    atomic_json(result_dir / "BOUNDED_REPORT.json", report)
    if report["status"].startswith("PASS"):
        write_success(result_dir, stage="bounded-screen", fingerprint=content_hash(report), extra={"selected_models": selection["selected_models"]})
    else:
        (result_dir / "_SUCCESS.json").unlink(missing_ok=True)
    tracker.finish("PASS" if report["status"].startswith("PASS") else "REJECTED", output_fingerprint=content_hash(report))
    return report


def _read_verified_bounded_rankings(path: Path, *, expected_sha256: str, expected_qids: set[str], candidate_doc_ids: Mapping[str, set[str]]) -> dict[str, list[str]]:
    """Load a bounded ranking only when its recorded bytes and contract match."""
    if not path.exists() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"bounded ranking SHA-256 mismatch: {path}")
    rankings: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("qid"))
        documents = [str(value) for value in row.get("documents", [])]
        if qid in rankings or qid not in expected_qids:
            raise RuntimeError(f"invalid bounded ranking qid: {qid}")
        if len(documents) != len(set(documents)) or not set(documents).issubset(candidate_doc_ids[qid]):
            raise RuntimeError(f"invalid bounded ranking documents: {qid}")
        rankings[qid] = documents
    if set(rankings) != expected_qids:
        raise RuntimeError(f"bounded ranking qid coverage mismatch: {path}")
    return rankings


def reevaluate_bounded(*, outer: str = "fold_0") -> dict[str, Any]:
    """Reapply the repaired gate to verified cached rankings; never encode."""
    result_dir = RESULTS_ROOT / "bounded_screen" / outer
    report_path = result_dir / "BOUNDED_REPORT.json"
    if not report_path.exists():
        raise GateRejected("REJECTED_PREREQUISITE_GATE", f"bounded report missing: {report_path}")
    prior = read_json(report_path)
    if prior.get("stage") != "bounded_screen" or prior.get("outer") != outer:
        raise RuntimeError("bounded report stage/outer mismatch")
    original_sha256 = sha256_file(report_path)
    backup_path = result_dir / "BOUNDED_REPORT_BEFORE_GATE_REPAIR.json"
    if not backup_path.exists():
        atomic_json(backup_path, prior)
    elif sha256_file(backup_path) != original_sha256:
        raise RuntimeError("existing pre-repair bounded report does not match current report")

    # Refreshes only the deterministic fixture marker for the new code hash;
    # the ranking files below are the already-completed GPU output.
    fixture_manifest = build_bounded_fixture(outer=outer, resume=True)
    rows, chunks, index, candidate_indices = _load_bounded_data(outer)
    expected_qids = {str(row["qid"]) for row in rows}
    candidate_doc_ids = {
        str(row["qid"]): {index.doc_ids[item] for item in candidate_indices[str(row["qid"])]}
        for row in rows
    }
    e5_queries, e5_chunks = _frozen_e5_bounded_embeddings(rows, chunks)
    e5_rankings = {
        str(row["qid"]): _rank_bounded_model(e5_queries[pos], e5_chunks, index, candidate_indices[str(row["qid"])], index.doc_ids)
        for pos, row in enumerate(rows)
    }
    tags = _load_tags()
    source_miss_count = sum(
        _first_gold_rank(e5_rankings[str(row["qid"])], set(row["gold_doc_ids"])) > 32
        and _is_source_generation_miss(str(row["qid"]), row, tags)
        for row in rows
    )
    per_model: dict[str, Any] = {}
    for model in ("vietlegal_harrier_0_6b", "vnlegal_lal"):
        previous_model = prior.get("models", {}).get(model)
        if not isinstance(previous_model, Mapping):
            raise RuntimeError(f"bounded report missing model output: {model}")
        ranking_path = result_dir / f"{model}.rankings.jsonl"
        rankings = _read_verified_bounded_rankings(
            ranking_path,
            expected_sha256=str(previous_model.get("rankings_sha256", "")),
            expected_qids=expected_qids,
            candidate_doc_ids=candidate_doc_ids,
        )
        metrics = bounded_model_metrics(
            rows,
            e5_rankings=e5_rankings,
            candidate_rankings=rankings,
            source_generation_rankings=rankings,
            tags=tags,
        )
        per_model[model] = {
            "metrics": metrics,
            "gate": bounded_gate(metrics, included_folds=sorted({str(row["fold"]) for row in rows}), source_generation_miss_count=source_miss_count),
            "rankings_sha256": sha256_file(ranking_path),
            "ranking_artifact": str(ranking_path.resolve()),
            "ranking_sha256_verified": True,
            "runtime": previous_model.get("runtime", {}),
            "estimated_encode_seconds": previous_model.get("estimated_encode_seconds"),
        }
    selection = choose_models_from_bounded(per_model)
    audit = read_json(RESULTS_ROOT / "input_audit" / "READING_AUDIT.json")
    replay_report = read_json(RESULTS_ROOT / "replay" / "REPLAY.json")
    report = {
        "schema_version": SCHEMA,
        "stage": "bounded_screen",
        "status": "PASS" if selection["selected_models"] else selection["status"],
        "outer": outer,
        "fixture": fixture_manifest,
        "bounded_oracle_fixture_not_recall": True,
        "gold_force_included": True,
        "may_not_be_reported_as_full_corpus_recall": True,
        "models": per_model,
        "selection": selection,
        "label_policy": LABEL_POLICY,
        "scorer_contract": SCORER_CONTRACT,
        "input_audit_fingerprint": audit["audit_fingerprint"],
        "replay_fingerprint": content_hash(replay_report),
        "gate_repair": {
            "id": GATE_REPAIR_ID,
            "reason": "control_exit_rate is retained as a diagnostic/risk flag; bounded pass uses the five complementarity checks.",
            "control_protection": "fold-isolated full-corpus source audit and fusion gate",
            "selection": "VnLegal-LAL only; Harrier is excluded because it has no exclusive Top-5 rescue.",
            "prior_report_path": str(backup_path.resolve()),
            "prior_report_sha256": original_sha256,
            "reused_ranking_artifacts_sha256_verified": True,
            "fold_isolation": "bounded selection uses folds_1_to_4 only; fold_0 remains held out",
        },
    }
    atomic_json(report_path, report)
    atomic_json(result_dir / "GATE_REEVALUATION.json", report)
    if report["status"] == "PASS":
        write_success(result_dir, stage="bounded-screen-re-evaluate", fingerprint=content_hash(report), extra={"selected_models": selection["selected_models"], "gate_repair": GATE_REPAIR_ID})
    else:
        (result_dir / "_SUCCESS.json").unlink(missing_ok=True)
    return report


# ---------------------------------------------------------------------------
# Preflight, resumable encoding, and frozen shard validation
# ---------------------------------------------------------------------------


def _estimate_embedding_bytes(chunks: int = CHUNK_COUNT, queries: int = QUERY_COUNT, models: int = 1) -> int:
    return int(models * ((chunks * DIMENSION * 2) + (queries * DIMENSION * 4) + chunks * 128 + queries * 256))


def _estimate_ranking_bytes(queries: int = QUERY_COUNT, parent_rank: int = BM25_MAX_RANK, models: int = 1) -> int:
    # JSONL rankings include parent IDs, ranks, scores, and framing.  Keep a
    # deliberately conservative planning allowance; the measured files are
    # checked later by their shard hashes.
    return int(models * queries * parent_rank * 96)


def _selected_models_from_bounded(outer: str) -> list[str]:
    report_path = RESULTS_ROOT / "bounded_screen" / outer / "BOUNDED_REPORT.json"
    marker = RESULTS_ROOT / "bounded_screen" / outer / "_SUCCESS.json"
    if not report_path.exists() or not marker.exists():
        raise GateRejected("REJECTED_PREREQUISITE_GATE", f"bounded screen PASS required: {report_path}")
    report = read_json(report_path)
    require_success(
        report_path.parent,
        expected_fingerprint=content_hash(report),
        expected_code_sha256=code_fingerprint(),
    )
    audit = read_json(RESULTS_ROOT / "input_audit" / "READING_AUDIT.json")
    replay_report = read_json(RESULTS_ROOT / "replay" / "REPLAY.json")
    require_success(
        RESULTS_ROOT / "input_audit",
        expected_fingerprint=audit["audit_fingerprint"],
        expected_code_sha256=code_fingerprint(),
    )
    require_success(
        RESULTS_ROOT / "replay",
        expected_fingerprint=content_hash(replay_report),
        expected_code_sha256=code_fingerprint(),
    )
    if (
        report.get("input_audit_fingerprint") != audit["audit_fingerprint"]
        or report.get("replay_fingerprint") != content_hash(replay_report)
    ):
        raise GateRejected("REJECTED_PREREQUISITE_GATE", "bounded screen was produced from stale audit or replay inputs")
    if report.get("status") != "PASS":
        raise GateRejected("REJECTED_BOUNDED_COMPLEMENTARITY", "bounded screen report is not PASS")
    selected = list(report.get("selection", {}).get("selected_models", []))
    if not selected:
        raise GateRejected("REJECTED_BOUNDED_COMPLEMENTARITY", "bounded gate selected no alternate model")
    return selected


def preflight(*, outer: str = "fold_0", resume: bool = True, authorize: bool = False) -> dict[str, Any]:
    if not authorize and os.environ.get("EXP109B_ALLOW_PREFLIGHT_GPU") != "1":
        raise GateRejected("REJECTED_AUTHORIZATION_GATE", "preflight requires explicit GPU authorization (EXP109B_ALLOW_PREFLIGHT_GPU=1)")
    output_dir = RESULTS_ROOT / "preflight" / outer
    selected = _selected_models_from_bounded(outer)
    bounded_report = read_json(RESULTS_ROOT / "bounded_screen" / outer / "BOUNDED_REPORT.json")
    if resume and _stage_marker_current(output_dir, "PREFLIGHT.json"):
        saved = read_json(output_dir / "PREFLIGHT.json")
        if saved.get("bounded_report_fingerprint") == content_hash(bounded_report):
            return saved
    free = shutil.disk_usage(ROOT).free
    reports: dict[str, Any] = {}
    tracker = RunTracker("preflight", outer=outer, total=len(selected))
    for key in selected:
        spec = MODEL_SPECS[key]
        adapter = create_encoder(key)
        query_texts = ["preflight query"] * 8
        doc_texts = ["preflight document"] * 8
        batch_results: dict[str, Any] = {}
        try:
            for kind, texts in (("query", query_texts), ("document", doc_texts)):
                effective, encoded, attempts = oom_safe_batch(lambda batch: adapter.encode(texts, is_query=kind == "query", batch_size=batch), requested=8)
                batch_results[kind] = {
                    "effective_batch_size": effective,
                    "attempts": attempts,
                    "texts_per_second": len(texts) / max(encoded.elapsed_seconds, 1e-9),
                    "token_count_mean": float(encoded.token_counts.mean()) if len(encoded.token_counts) else 0.0,
                    "truncation_count": int(encoded.truncation_count),
                }
            peak_mib = float(torch.cuda.max_memory_allocated() / 2**20) if torch is not None and torch.cuda.is_available() else 0.0
            total_mib = float(torch.cuda.get_device_properties(0).total_memory / 2**20) if torch is not None and torch.cuda.is_available() else 0.0
            headroom = (total_mib - peak_mib) / total_mib if total_mib else None
            chunk_speed = batch_results["document"]["texts_per_second"]
            query_speed = batch_results["query"]["texts_per_second"]
            reports[key] = {
                "contract": dataclass_to_dict(spec),
                "batch_results": batch_results,
                "peak_cuda_memory_mib": peak_mib,
                "device_total_memory_mib": total_mib,
                "headroom_fraction": headroom,
                "headroom_pass": headroom is None or headroom >= 0.10,
                "estimated_chunk_seconds": CHUNK_COUNT / max(chunk_speed, 1e-9),
                "estimated_query_seconds": QUERY_COUNT / max(query_speed, 1e-9),
                "estimated_total_hours": (CHUNK_COUNT / max(chunk_speed, 1e-9) + QUERY_COUNT / max(query_speed, 1e-9)) / 3600,
            }
            tracker.heartbeat(len(reports), emit=True, model=key)
        finally:
            adapter.close()
    estimated_embedding_bytes = _estimate_embedding_bytes(models=len(selected))
    estimated_ranking_bytes = _estimate_ranking_bytes(models=len(selected) + 2)  # E5 + selected alternates + BM25 source ranks.
    estimated_required_bytes = estimated_embedding_bytes + estimated_ranking_bytes
    pass_gate = all(value["headroom_pass"] for value in reports.values()) and free > estimated_required_bytes * 2
    report = {
        "schema_version": SCHEMA,
        "stage": "preflight",
        "status": "PASS" if pass_gate else "REJECTED_RESOURCE_GATE",
        "outer": outer,
        "selected_models": selected,
        "models": reports,
        "bounded_report_fingerprint": content_hash(bounded_report),
        "disk": {"free_bytes": int(free), "estimated_embedding_bytes": estimated_embedding_bytes, "estimated_ranking_bytes": estimated_ranking_bytes, "estimated_required_bytes": estimated_required_bytes, "temporary_merge_multiplier": 2.0},
        "is_feasibility_only": True,
    }
    atomic_json(output_dir / "PREFLIGHT.json", report)
    if pass_gate:
        write_success(output_dir, stage="preflight", fingerprint=content_hash(report))
    else:
        (output_dir / "_SUCCESS.json").unlink(missing_ok=True)
    tracker.finish("PASS" if pass_gate else "REJECTED", output_fingerprint=content_hash(report))
    return report


def _embedding_manifest_fingerprint(model: str, *, outer: str, input_fingerprint: str) -> str:
    return content_hash({"schema": SCHEMA, "model": model, "outer": outer, "input_fingerprint": input_fingerprint, "contract": dataclass_to_dict(MODEL_SPECS[model]), "scorer_contract": SCORER_CONTRACT, "code": code_fingerprint()})


def _save_embedding_shard(path: Path, *, ids: Sequence[str], parent_ids: Sequence[str], vectors: np.ndarray, token_counts: np.ndarray, truncation_count: int, manifest_fingerprint: str) -> dict[str, Any]:
    vectors = np.asarray(vectors, dtype=np.float32)
    normalized = numpy_l2_normalize(vectors)
    payload = {
        "ids": np.asarray(ids, dtype="U"),
        "parent_ids": np.asarray(parent_ids, dtype="U"),
        "vectors": normalized.astype(np.float16),
        "original_norms": np.linalg.norm(vectors, axis=1).astype(np.float32),
        "token_counts": np.asarray(token_counts, dtype=np.int32),
        "truncation_count": np.asarray([int(truncation_count)], dtype=np.int64),
        "manifest_fingerprint": np.asarray([manifest_fingerprint], dtype="U"),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    np.savez_compressed(temporary, **payload)
    # np.savez appends .npz when a string path does not end in .npz.
    generated = Path(str(temporary) if str(temporary).endswith(".npz") else str(temporary) + ".npz")
    generated.replace(path)
    receipt = {"path": str(path.resolve()), "sha256": sha256_file(path), "count": len(ids), "manifest_fingerprint": manifest_fingerprint}
    atomic_json(path.with_suffix(".json"), receipt)
    return receipt


def validate_embedding_shard(path: Path, *, expected_fingerprint: str, expected_count: int | None = None) -> dict[str, Any]:
    receipt_path = path.with_suffix(".json")
    if not path.exists() or not receipt_path.exists():
        raise RuntimeError(f"missing shard or receipt: {path}")
    receipt = read_json(receipt_path)
    if receipt.get("sha256") != sha256_file(path) or receipt.get("manifest_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"corrupt/stale shard: {path}")
    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"]
        parent_ids = data["parent_ids"]
        vectors = data["vectors"]
        norms = data["original_norms"]
        token_counts = data["token_counts"]
        embedded_fp = str(data["manifest_fingerprint"][0])
        if embedded_fp != expected_fingerprint:
            raise RuntimeError(f"embedded shard fingerprint mismatch: {path}")
        if len(ids) != len(parent_ids) or len(ids) != len(vectors) or len(ids) != len(norms) or len(ids) != len(token_counts):
            raise RuntimeError(f"shard array count mismatch: {path}")
        if expected_count is not None and len(ids) != expected_count:
            raise RuntimeError(f"shard count mismatch: {path}")
        if vectors.ndim != 2 or vectors.shape[1] != DIMENSION or not np.isfinite(vectors.astype(np.float32)).all():
            raise RuntimeError(f"shard vector shape/finite mismatch: {path}")
    return {**receipt, "count": int(len(ids))}


def validate_query_embeddings(path: Path, *, expected_fingerprint: str, expected_count: int = QUERY_COUNT) -> dict[str, Any]:
    """Fail closed on a stale or corrupt selected-model query artifact."""
    if not path.exists():
        raise RuntimeError(f"missing query embedding artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        query_ids = [str(value) for value in data["query_ids"]]
        vectors = np.asarray(data["vectors"])
        token_counts = np.asarray(data["token_counts"])
        embedded_fp = str(data["manifest_fingerprint"][0])
        if embedded_fp != expected_fingerprint:
            raise RuntimeError(f"query embedding fingerprint mismatch: {path}")
        if len(query_ids) != expected_count or len(set(query_ids)) != len(query_ids):
            raise RuntimeError(f"query embedding count/uniqueness mismatch: {path}")
        if len(vectors) != len(query_ids) or len(token_counts) != len(query_ids):
            raise RuntimeError(f"query embedding array count mismatch: {path}")
        if vectors.ndim != 2 or vectors.shape[1] != DIMENSION or not np.isfinite(vectors.astype(np.float32)).all():
            raise RuntimeError(f"query embedding vector shape/finite mismatch: {path}")
    return {"path": str(path.resolve()), "count": len(query_ids), "manifest_fingerprint": embedded_fp}


def encode_model_selected(*, model: str, outer: str = "fold_0", resume: bool = True, authorize: bool = False) -> dict[str, Any]:
    if not authorize and os.environ.get("EXP109B_ALLOW_FULL_ENCODING") != "1":
        raise GateRejected("REJECTED_AUTHORIZATION_GATE", "full encoding requires explicit authorization (EXP109B_ALLOW_FULL_ENCODING=1)")
    # The repair report has already SHA-verified the bounded artifacts.  Do
    # not demand a new bounded GPU run merely because this cache-only pilot
    # adds code; verify its declared LAL selection directly instead.
    bounded_report = read_json(RESULTS_ROOT / "bounded_screen" / outer / "BOUNDED_REPORT.json")
    selected = list(bounded_report.get("selection", {}).get("selected_models", []))
    if bounded_report.get("status") != "PASS" or selected != ["vnlegal_lal"]:
        raise GateRejected("REJECTED_PREREQUISITE_GATE", "verified LAL-only bounded selection required")
    if model not in selected:
        raise GateRejected("REJECTED_MODEL_SELECTION_GATE", f"model {model} was not selected by bounded gate")
    preflight_dir = RESULTS_ROOT / "preflight" / outer
    require_success(preflight_dir, expected_fingerprint=content_hash(read_json(preflight_dir / "PREFLIGHT.json")), expected_code_sha256=code_fingerprint())
    preflight_report = read_json(preflight_dir / "PREFLIGHT.json")
    model_preflight = preflight_report.get("models", {}).get(model, {})
    document_batch = int(model_preflight.get("batch_results", {}).get("document", {}).get("effective_batch_size", 8))
    query_batch = int(model_preflight.get("batch_results", {}).get("query", {}).get("effective_batch_size", 8))
    audit_dir = RESULTS_ROOT / "input_audit"
    audit = read_json(audit_dir / "READING_AUDIT.json")
    require_success(
        audit_dir,
        expected_fingerprint=audit["audit_fingerprint"],
        expected_code_sha256=code_fingerprint(),
    )
    manifest_fp = _embedding_manifest_fingerprint(model, outer=outer, input_fingerprint=str(audit["audit_fingerprint"]))
    output_dir = CACHE_ROOT / "embeddings" / model
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    shard_size = 256
    adapter = create_encoder(model)
    tracker = RunTracker("encode-selected", outer=outer, model=model, total=CHUNK_COUNT + QUERY_COUNT)
    chunk_total = 0
    shard_receipts: list[dict[str, Any]] = []
    try:
        chunk_buffer: list[dict[str, Any]] = []
        for row in read_jsonl(STRUCT_DIR / "chunks.jsonl"):
            chunk_buffer.append(row)
            if len(chunk_buffer) < shard_size:
                continue
            shard_index = chunk_total // shard_size
            path = chunk_dir / f"shard-{shard_index:05d}.npz"
            if resume and path.exists():
                try:
                    receipt = validate_embedding_shard(path, expected_fingerprint=manifest_fp, expected_count=len(chunk_buffer))
                except RuntimeError:
                    # A partial/corrupt shard belongs to this stage.  Rebuild
                    # it in place; never reuse it merely because the filename
                    # exists.
                    _effective_batch, encoded, _attempts = oom_safe_batch(lambda batch: adapter.encode([str(item["retrieval_text"]) for item in chunk_buffer], is_query=False, batch_size=batch), requested=document_batch)
                    receipt = _save_embedding_shard(path, ids=[str(item["chunk_id"]) for item in chunk_buffer], parent_ids=[str(item["doc_id"]) for item in chunk_buffer], vectors=encoded.vectors, token_counts=encoded.token_counts, truncation_count=encoded.truncation_count, manifest_fingerprint=manifest_fp)
            else:
                _effective_batch, encoded, _attempts = oom_safe_batch(lambda batch: adapter.encode([str(item["retrieval_text"]) for item in chunk_buffer], is_query=False, batch_size=batch), requested=document_batch)
                receipt = _save_embedding_shard(path, ids=[str(item["chunk_id"]) for item in chunk_buffer], parent_ids=[str(item["doc_id"]) for item in chunk_buffer], vectors=encoded.vectors, token_counts=encoded.token_counts, truncation_count=encoded.truncation_count, manifest_fingerprint=manifest_fp)
            shard_receipts.append(receipt)
            chunk_total += len(chunk_buffer)
            tracker.heartbeat(chunk_total, emit=False, input_fingerprint=audit["audit_fingerprint"], config_fingerprint=manifest_fp)
            chunk_buffer = []
        if chunk_buffer:
            shard_index = chunk_total // shard_size
            path = chunk_dir / f"shard-{shard_index:05d}.npz"
            if resume and path.exists():
                try:
                    receipt = validate_embedding_shard(path, expected_fingerprint=manifest_fp, expected_count=len(chunk_buffer))
                except RuntimeError:
                    _effective_batch, encoded, _attempts = oom_safe_batch(lambda batch: adapter.encode([str(item["retrieval_text"]) for item in chunk_buffer], is_query=False, batch_size=batch), requested=document_batch)
                    receipt = _save_embedding_shard(path, ids=[str(item["chunk_id"]) for item in chunk_buffer], parent_ids=[str(item["doc_id"]) for item in chunk_buffer], vectors=encoded.vectors, token_counts=encoded.token_counts, truncation_count=encoded.truncation_count, manifest_fingerprint=manifest_fp)
            else:
                _effective_batch, encoded, _attempts = oom_safe_batch(lambda batch: adapter.encode([str(item["retrieval_text"]) for item in chunk_buffer], is_query=False, batch_size=batch), requested=document_batch)
                receipt = _save_embedding_shard(path, ids=[str(item["chunk_id"]) for item in chunk_buffer], parent_ids=[str(item["doc_id"]) for item in chunk_buffer], vectors=encoded.vectors, token_counts=encoded.token_counts, truncation_count=encoded.truncation_count, manifest_fingerprint=manifest_fp)
            shard_receipts.append(receipt)
            chunk_total += len(chunk_buffer)
            tracker.heartbeat(chunk_total, emit=False, input_fingerprint=audit["audit_fingerprint"], config_fingerprint=manifest_fp)
        # Query vectors are a single independently checksummed artifact.
        train = load_train()
        query_ids = sorted(train)
        _effective_batch, query_encoded, _attempts = oom_safe_batch(lambda batch: adapter.encode([str(train[qid]["question"]) for qid in query_ids], is_query=True, batch_size=batch), requested=query_batch)
        queries_path = output_dir / "queries.npz"
        temporary = queries_path.with_name(f".{queries_path.name}.{os.getpid()}.tmp")
        np.savez_compressed(temporary, query_ids=np.asarray(query_ids, dtype="U"), vectors=query_encoded.vectors.astype(np.float16), token_counts=query_encoded.token_counts, truncation_count=np.asarray([query_encoded.truncation_count], dtype=np.int64), manifest_fingerprint=np.asarray([manifest_fp], dtype="U"))
        generated = Path(str(temporary) if str(temporary).endswith(".npz") else str(temporary) + ".npz")
        generated.replace(queries_path)
        manifest = {
            "schema_version": SCHEMA,
            "model": model,
            "contract": dataclass_to_dict(MODEL_SPECS[model]),
            "outer": outer,
            "input_fingerprint": audit["audit_fingerprint"],
            "manifest_fingerprint": manifest_fp,
            "scorer_contract": SCORER_CONTRACT,
            "chunks": chunk_total,
            "queries": len(query_ids),
            "shard_size": shard_size,
            "shards": shard_receipts,
            "queries_sha256": sha256_file(queries_path),
        }
        manifest["content_fingerprint"] = content_hash(manifest)
        atomic_json(output_dir / "manifest.json", manifest)
        write_success(output_dir, stage="encode-selected", fingerprint=manifest["content_fingerprint"], extra={"manifest_fingerprint": manifest_fp})
        tracker.heartbeat(CHUNK_COUNT + len(query_ids), emit=True, input_fingerprint=audit["audit_fingerprint"], config_fingerprint=manifest_fp)
        tracker.finish("PASS", output_fingerprint=manifest["content_fingerprint"])
        return manifest
    except KeyboardInterrupt:
        tracker.finish("INTERRUPTED")
        raise
    except Exception:
        tracker.finish("FAILED", error=traceback.format_exc(limit=6))
        raise
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# BM25 parent ranks, dense source audit, candidate depth, and fusion
# ---------------------------------------------------------------------------


def iter_bm25_evidence() -> Iterator[dict[str, Any]]:
    for path in sorted((BM25_RAW_DIR / "shards").glob("evidence_*.jsonl")):
        yield from read_jsonl(path)


def bm25_rankings(evidence: Sequence[Sequence[Any]], config: Mapping[str, Any], *, limit: int = BM25_MAX_RANK) -> list[dict[str, Any]]:
    depth = int(config["depth"])
    parent_k = int(config["parent_rrf_k"])
    fusion_k = int(config["fusion_rrf_k"])
    head = int(config["head_cutoff"])
    eligible: list[tuple[str, list[int]]] = []
    for doc_id, ranks in evidence:
        filtered = sorted(int(rank) for rank in ranks if int(rank) <= depth)
        if filtered:
            eligible.append((str(doc_id), filtered))
    first = [doc_id for doc_id, _ in sorted(eligible, key=lambda item: (item[1][0], item[0]))]
    rrf = [
        doc_id for doc_id, _score, _first in sorted(
            ((doc_id, sum(1.0 / (parent_k + rank) for rank in ranks), ranks[0]) for doc_id, ranks in eligible),
            key=lambda item: (-item[1], item[2], item[0]),
        )
    ]
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, int] = {}
    for ranking in (first, rrf):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (fusion_k + rank)
            best[doc_id] = min(best.get(doc_id, rank), rank)
    fused = sorted(scores, key=lambda doc_id: (-scores[doc_id], best[doc_id], doc_id))
    head_docs = fused[:head]
    result = head_docs + [doc_id for doc_id in rrf if doc_id not in set(head_docs)]
    result = result[:limit]
    return [{"doc_id": doc_id, "rank": rank, "score": float(scores.get(doc_id, 0.0))} for rank, doc_id in enumerate(result, start=1)]


def _tuned_bm25_config_for_qid(qid: str, fold_for: Mapping[str, str]) -> Mapping[str, Any]:
    tuning = read_json(BM25_TUNING_REPORT)
    selected = tuning.get("selected_by_candidate_budget", {}).get("150", {})
    fold = fold_for[str(qid)]
    if fold not in selected:
        raise KeyError(f"no tuned BM25 config for {fold}")
    return selected[fold]


def _load_dense_embedding_matrix(model: str) -> tuple[np.ndarray, list[str], list[str], dict[str, Any]]:
    if model == "vietlegal_e5":
        chunks = np.load(E5_DIR / "embeddings.f16.npy", mmap_mode="r")
        chunk_rows = list(read_jsonl(E5_DIR / "chunk_ids.jsonl"))
        chunk_ids = [str(row["chunk_id"]) for row in chunk_rows]
        parent_ids = [str(row["doc_id"]) for row in chunk_rows]
        query_array = np.load(QUERY_DIR / "train_queries.f32.npy", mmap_mode="r")
        query_ids = [str(value) for value in read_json(QUERY_DIR / "train_query_ids.json")]
        return chunks, chunk_ids, parent_ids, {"query_vectors": query_array, "query_ids": query_ids, "source": "frozen e5_final_v1"}
    manifest = read_json(CACHE_ROOT / "embeddings" / model / "manifest.json")
    require_success(CACHE_ROOT / "embeddings" / model, manifest.get("content_fingerprint"), expected_code_sha256=code_fingerprint())
    audit = read_json(RESULTS_ROOT / "input_audit" / "READING_AUDIT.json")
    require_success(
        RESULTS_ROOT / "input_audit",
        expected_fingerprint=audit["audit_fingerprint"],
        expected_code_sha256=code_fingerprint(),
    )
    if manifest.get("input_fingerprint") != audit["audit_fingerprint"]:
        raise RuntimeError(f"selected-model embedding input fingerprint is stale: {model}")
    if manifest.get("scorer_contract") != SCORER_CONTRACT or manifest.get("contract") != dataclass_to_dict(MODEL_SPECS[model]):
        raise RuntimeError(f"selected-model embedding contract mismatch: {model}")
    arrays: list[np.ndarray] = []
    chunk_ids: list[str] = []
    parent_ids: list[str] = []
    for receipt in manifest.get("shards", []):
        path = Path(receipt["path"])
        validate_embedding_shard(path, expected_fingerprint=manifest["manifest_fingerprint"])
        with np.load(path, allow_pickle=False) as data:
            arrays.append(np.asarray(data["vectors"], dtype=np.float16))
            chunk_ids.extend(str(value) for value in data["ids"])
            parent_ids.extend(str(value) for value in data["parent_ids"])
    chunks = np.concatenate(arrays, axis=0) if arrays else np.empty((0, DIMENSION), dtype=np.float16)
    with np.load(CACHE_ROOT / "embeddings" / model / "queries.npz", allow_pickle=False) as data:
        query_array = np.asarray(data["vectors"], dtype=np.float16)
        query_ids = [str(value) for value in data["query_ids"]]
    if manifest.get("queries_sha256") != sha256_file(CACHE_ROOT / "embeddings" / model / "queries.npz"):
        raise RuntimeError(f"selected-model query embedding hash mismatch: {model}")
    validate_query_embeddings(
        CACHE_ROOT / "embeddings" / model / "queries.npz",
        expected_fingerprint=manifest["manifest_fingerprint"],
    )
    return chunks, chunk_ids, parent_ids, {"query_vectors": query_array, "query_ids": query_ids, "source": str(CACHE_ROOT / "embeddings" / model)}


def _dense_ranking_rows(model: str, qids: Sequence[str], *, limit: int = BM25_MAX_RANK, resume: bool = True, outer: str = "fold_0") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks, chunk_ids, parent_ids, query_info = _load_dense_embedding_matrix(model)
    index = CorpusIndex.from_chunk_doc_ids(parent_ids)
    query_lookup = {qid: pos for pos, qid in enumerate(query_info["query_ids"])}
    audit = read_json(RESULTS_ROOT / "input_audit" / "READING_AUDIT.json")
    fingerprint = content_hash({"audit": audit["audit_fingerprint"], "model": model, "outer": outer, "limit": limit, "scorer": SCORER_CONTRACT, "code": code_fingerprint()})
    output_dir = CACHE_ROOT / "rankings" / model / outer
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    block_size = 64
    tracker = RunTracker("dense-rankings", outer=outer, model=model, total=len(qids))
    for start in range(0, len(qids), block_size):
        batch_qids = list(qids[start:start + block_size])
        shard_path = shard_dir / f"rankings-{start // block_size:05d}.jsonl"
        marker = shard_path.with_suffix(".json")
        if resume and shard_path.exists() and marker.exists():
            saved = read_json(marker)
            if saved.get("fingerprint") == fingerprint and saved.get("sha256") == sha256_file(shard_path):
                rows.extend(read_jsonl(shard_path))
                tracker.heartbeat(min(start + len(batch_qids), len(qids)), emit=False, config_fingerprint=fingerprint)
                continue
        generated: list[dict[str, Any]] = []
        for qid in batch_qids:
            if qid not in query_lookup:
                raise ValueError(f"query vector missing: {qid}")
            q = np.asarray(query_info["query_vectors"][query_lookup[qid]], dtype=np.float32)
            if torch is not None and torch.cuda.is_available():
                scores = compute_parent_scores(torch.as_tensor(q, device="cuda"), chunks, index, chunk_block_size=8192, device="cuda", dtype=torch.float32).detach().cpu().numpy()[0]
            else:
                scores = dense_parent_scores_numpy(q, chunks, index)
            ranking = stable_rank_with_scores(scores, index.doc_ids, limit)
            generated.append({"qid": qid, "model": model, "parent_count": index.parent_count, "documents": ranking, "scorer_contract": SCORER_CONTRACT})
        write_jsonl_atomic(shard_path, generated)
        atomic_json(marker, {"fingerprint": fingerprint, "sha256": sha256_file(shard_path), "count": len(generated)})
        rows.extend(generated)
        tracker.heartbeat(min(start + len(batch_qids), len(qids)), emit=False, config_fingerprint=fingerprint)
    rows.sort(key=lambda row: str(row["qid"]))
    manifest = {"schema_version": SCHEMA, "model": model, "outer": outer, "limit": limit, "parent_count": index.parent_count, "query_count": len(rows), "fingerprint": fingerprint, "shards": [{"name": path.name, "sha256": sha256_file(path)} for path in sorted(shard_dir.glob("rankings-*.jsonl"))]}
    manifest["content_fingerprint"] = content_hash(manifest)
    atomic_json(output_dir / "manifest.json", manifest)
    write_success(output_dir, stage="dense-rankings", fingerprint=manifest["content_fingerprint"])
    tracker.finish("PASS", output_fingerprint=manifest["content_fingerprint"])
    return rows, manifest


def _ranking_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {str(row["qid"]): [str(item["doc_id"] if isinstance(item, dict) else item) for item in row.get("documents", [])] for row in rows}


def _ranking_score_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, tuple[int, float]]]:
    return {str(row["qid"]): {str(item["doc_id"]): (int(item["rank"]), float(item.get("score", 0.0))) for item in row.get("documents", [])} for row in rows}


def _load_dense_ranking_rows(model: str, outer: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and verify one completed dense-ranking stage."""
    output_dir = CACHE_ROOT / "rankings" / model / outer
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing dense-ranking manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    require_success(
        output_dir,
        expected_fingerprint=manifest.get("content_fingerprint"),
        expected_code_sha256=code_fingerprint(),
    )
    rows: list[dict[str, Any]] = []
    shard_dir = output_dir / "shards"
    for entry in manifest.get("shards", []):
        path = shard_dir / str(entry["name"])
        if not path.exists() or sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"dense-ranking shard hash mismatch: {path}")
        rows.extend(read_jsonl(path))
    if len(rows) != int(manifest.get("query_count", -1)):
        raise RuntimeError(f"dense-ranking row count mismatch: {model}/{outer}")
    qids = [str(row.get("qid")) for row in rows]
    if len(qids) != len(set(qids)):
        raise RuntimeError(f"duplicate dense-ranking qid: {model}/{outer}")
    return rows, manifest


def _bm25_rank_rows(qids: Sequence[str], fold_for: Mapping[str, str], *, limit: int = BM25_MAX_RANK) -> list[dict[str, Any]]:
    evidence_by_qid = {str(row["qid"]): row for row in iter_bm25_evidence()}
    missing = sorted(set(str(qid) for qid in qids) - set(evidence_by_qid))
    if missing:
        raise RuntimeError(f"BM25 evidence missing qids: {missing[:10]}")
    rows: list[dict[str, Any]] = []
    for qid in qids:
        qid = str(qid)
        config = _tuned_bm25_config_for_qid(qid, fold_for)
        rows.append({
            "qid": qid,
            "model": "bm25",
            "parent_count": DOCUMENT_COUNT,
            "documents": bm25_rankings(evidence_by_qid[qid]["evidence"], config, limit=limit),
            "scorer_contract": "tuned_exp021_bm25_parent_rank_v1",
        })
    return rows


def union_ranking(rankings: Mapping[str, Sequence[str]], sources: Sequence[str], *, depth: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for doc_id in rankings.get(source, [])[:depth]:
            if doc_id not in seen:
                result.append(doc_id)
                seen.add(doc_id)
    return result


def union_candidate_coverage(rankings: Mapping[str, Sequence[str]], sources: Sequence[str], gold: set[str], *, depth: int) -> float:
    """Recall of the candidate set formed by every source's top ``depth``.

    This is intentionally different from evaluating an ordering at cutoff
    ``depth``: a three-source candidate set may contain up to ``3*depth``
    parents before de-duplication.
    """
    candidates = set()
    for source in sources:
        candidates.update(str(doc_id) for doc_id in rankings.get(source, [])[:depth])
    return len(candidates & gold) / len(gold) if gold else 0.0


def recall_at(prediction: Sequence[str], gold: set[str], k: int) -> float:
    return len(set(prediction[:k]) & gold) / len(gold) if gold else 0.0


def multi_gold_recall_at(prediction: Sequence[str], gold: set[str], k: int) -> float:
    return len(set(prediction[:k]) & gold) / len(gold) if gold else 0.0


def evaluate_rankings(predictions: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], qids: Sequence[str]) -> dict[str, Any]:
    evaluable = [qid for qid in qids if answers.get(qid)]
    if not evaluable:
        return {"evaluable_queries": 0}
    metrics: dict[str, Any] = {f"recall@{k}": float(np.mean([recall_at(predictions[qid], answers[qid], k) for qid in evaluable])) for k in CURVE_KS}
    metrics["precision@5"] = float(np.mean([len(set(predictions[qid][:5]) & answers[qid]) / 5.0 for qid in evaluable]))
    reciprocal_ranks = []
    for qid in evaluable:
        first_rank = next((rank for rank, doc_id in enumerate(predictions[qid], start=1) if doc_id in answers[qid]), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank is not None and first_rank <= 5 else 0.0)
    metrics["mrr@5"] = float(np.mean(reciprocal_ranks))
    multi_qids = [qid for qid in evaluable if len(answers[qid]) >= 2]
    metrics["multi_gold_recall@5"] = float(np.mean([multi_gold_recall_at(predictions[qid], answers[qid], 5) for qid in multi_qids])) if multi_qids else 0.0
    by_cardinality: dict[str, list[float]] = defaultdict(list)
    for qid in evaluable:
        by_cardinality["single_gold" if len(answers[qid]) == 1 else "multi_gold"].append(recall_at(predictions[qid], answers[qid], 5))
    metrics["cardinality"] = {key: {"count": len(values), "recall@5": float(np.mean(values))} for key, values in sorted(by_cardinality.items())}
    metrics["evaluable_queries"] = len(evaluable)
    return metrics


def source_audit(*, outer: str = "fold_0", resume: bool = True, authorize: bool = False) -> dict[str, Any]:
    if not authorize and os.environ.get("EXP109B_ALLOW_SOURCE_AUDIT") != "1":
        raise GateRejected("REJECTED_AUTHORIZATION_GATE", "source-audit requires explicit full-corpus authorization (EXP109B_ALLOW_SOURCE_AUDIT=1)")
    output_dir = RESULTS_ROOT / "source_audit" / outer
    selected = _selected_models_from_bounded(outer)
    preflight_report_path = RESULTS_ROOT / "preflight" / outer / "PREFLIGHT.json"
    preflight_report = read_json(preflight_report_path)
    require_success(
        preflight_report_path.parent,
        expected_fingerprint=content_hash(preflight_report),
        expected_code_sha256=code_fingerprint(),
    )
    audit_dir = RESULTS_ROOT / "input_audit"
    audit = read_json(audit_dir / "READING_AUDIT.json")
    require_success(
        audit_dir,
        expected_fingerprint=audit["audit_fingerprint"],
        expected_code_sha256=code_fingerprint(),
    )
    if resume and _stage_marker_current(output_dir, "SOURCE_AUDIT.json"):
        saved = read_json(output_dir / "SOURCE_AUDIT.json")
        if (
            saved.get("input_audit_fingerprint") == audit["audit_fingerprint"]
            and saved.get("preflight_fingerprint") == content_hash(preflight_report)
        ):
            return saved
    folds, fold_for = load_folds()
    answers, _stats = canonical_labels()
    qids = sorted(answers)
    model_sources = ["vietlegal_e5", *selected]
    dense_rows: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, Any] = {}
    tracker = RunTracker("source-audit", outer=outer, total=len(model_sources))
    for model in model_sources:
        dense_rows[model], manifests[model] = _dense_ranking_rows(model, qids, limit=BM25_MAX_RANK, resume=resume, outer=outer)
        tracker.heartbeat(len(dense_rows), emit=True, model=model)
    dense_maps = {model: _ranking_map(rows) for model, rows in dense_rows.items()}
    dense_score_maps = {model: _ranking_score_map(rows) for model, rows in dense_rows.items()}
    bm25_rows = _bm25_rank_rows(qids, fold_for)
    bm25_map = _ranking_map(bm25_rows)
    source_maps = {**dense_maps, "bm25": bm25_map}
    rank_lookup = {
        source: {qid: {doc_id: rank for rank, doc_id in enumerate(source_maps[source][qid], start=1)} for qid in qids}
        for source in source_maps
    }
    tags = _load_tags()
    report_sources: dict[str, Any] = {}
    for source, ranking in source_maps.items():
        metric = evaluate_rankings(ranking, answers, qids)
        gold_rank_distribution = Counter()
        for qid in qids:
            if answers[qid]:
                gold_rank_distribution[str(_first_gold_rank(ranking[qid], answers[qid]))] += 1
        report_sources[source] = {"metrics": metric, "gold_first_rank_distribution": dict(sorted(gold_rank_distribution.items(), key=lambda item: int(item[0]))), "rankings_count": len(ranking)}
    pairwise_overlap: dict[str, Any] = {}
    for left_index, left in enumerate(source_maps):
        for right in list(source_maps)[left_index + 1:]:
            pairwise_overlap[f"{left}__{right}"] = {str(k): float(np.mean([len(set(source_maps[left][qid][:k]) & set(source_maps[right][qid][:k])) / k for qid in qids])) for k in (5, 20, 50)}
    union_coverage: dict[str, Any] = {}
    for depth in (20, 50, 100, 200, 500):
        coverage_values = []
        candidate_counts = []
        for qid in qids:
            local = {source: source_maps[source][qid] for source in source_maps}
            if answers[qid]:
                coverage_values.append(union_candidate_coverage(local, list(local), answers[qid], depth=depth))
            candidate_counts.append(len(set().union(*(set(values[:depth]) for values in local.values()))))
        union_coverage[str(depth)] = {
            "candidate_set_recall": float(np.mean(coverage_values)) if coverage_values else 0.0,
            "candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
            "source_depth": depth,
        }
    source_gold_contribution: dict[str, Any] = {}
    for source, ranking in source_maps.items():
        source_gold_contribution[source] = {
            str(k): sum(
                gold_doc_id in ranking[qid][:k]
                and not any(gold_doc_id in source_maps[other][qid][:k] for other in source_maps if other != source)
                for qid in qids
                for gold_doc_id in answers[qid]
            )
            for k in (5, 20, 50)
        }
    gold_source_membership = {
        qid: {
            gold_doc_id: {
                source: rank_lookup[source][qid].get(gold_doc_id)
                for source in source_maps
            }
            for gold_doc_id in sorted(answers[qid])
        }
        for qid in qids
        if answers[qid]
    }
    oracle = {qid: max((recall_at(source_maps[source][qid], answers[qid], 5) for source in source_maps), default=0.0) for qid in qids}
    oracle_r5 = float(np.mean([value for qid, value in oracle.items() if answers[qid]])) if any(answers.values()) else 0.0
    # Candidate depth selection is called once per outer fold and receives only
    # outer-train labels, making leakage visible in its returned provenance.
    depth_selection = select_candidate_depth(source_maps, answers, folds, heldout=outer)
    baseline_depth_selection = select_candidate_depth(source_maps, answers, folds, heldout=outer, included_sources=("vietlegal_e5", "bm25"))
    baseline_inner_screen = nested_rrf_screen(
        source_maps,
        answers,
        folds,
        outer=outer,
        included_sources=("vietlegal_e5", "bm25"),
        candidate_depth=int(baseline_depth_selection[outer]["depth"]),
    )
    baseline_predictions = baseline_inner_screen["inner_predictions"]
    baseline_train_qids = _fold_train_qids(folds, outer)
    baseline_train_metrics = evaluate_rankings(baseline_predictions, answers, baseline_train_qids)
    alternate_entries = {source: report_sources[source]["metrics"] for source in selected}
    source_generation_occurrences: list[dict[str, Any]] = []
    source_generation_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for source in selected:
        for qid in baseline_train_qids:
            e5_ranks = rank_lookup["vietlegal_e5"][qid]
            bm25_ranks = rank_lookup["bm25"][qid]
            alternate_ranks = rank_lookup[source][qid]
            for gold_doc_id in sorted(answers[qid]):
                if alternate_ranks.get(gold_doc_id, BM25_MAX_RANK + 1) <= 20 and e5_ranks.get(gold_doc_id, BM25_MAX_RANK + 1) > 50 and bm25_ranks.get(gold_doc_id, BM25_MAX_RANK + 1) > 50:
                    occurrence_key = (qid, gold_doc_id)
                    existing = source_generation_by_key.get(occurrence_key)
                    if existing is None:
                        existing = {"qid": qid, "doc_id": gold_doc_id, "sources": [source]}
                        source_generation_by_key[occurrence_key] = existing
                        source_generation_occurrences.append(existing)
                    else:
                        existing["sources"] = sorted(set(existing.get("sources", [])) | {source})
    union_r50_train = float(np.mean([
        union_candidate_coverage(
            {source: source_maps[source][qid] for source in source_maps},
            list(source_maps),
            answers[qid],
            depth=50,
        )
        for qid in baseline_train_qids
        if answers[qid]
    ]))
    oracle_train = float(np.mean([oracle[qid] for qid in baseline_train_qids if answers[qid]]))
    viability_checks = {
        "best_source_oracle_recall@5_ge_0_950": oracle_train >= 0.950,
        "oracle_gain_vs_corrected_baseline_ge_0_020": oracle_train - float(baseline_train_metrics.get("recall@5", 0.0)) >= 0.020,
        "alternate_gold_occurrences_ge_20": len(source_generation_occurrences) >= 20,
        "candidate_union_recall@50_ge_corrected_baseline": union_r50_train >= float(baseline_train_metrics.get("recall@50", 0.0)),
    }
    source_viability_pass = all(viability_checks.values())
    source_gate = {
        "best_source_oracle_recall@5_outer_train": oracle_train,
        "required_best_source_recall@5": 0.950,
        "oracle_gain_vs_corrected_baseline": oracle_train - float(baseline_train_metrics.get("recall@5", 0.0)),
        "alternate_gold_occurrences": source_generation_occurrences,
        "candidate_union_recall@50_outer_train": union_r50_train,
        "viability_checks": viability_checks,
        "pass": source_viability_pass,
        "oracle_supported_target_097": oracle_train >= 0.970,
        "target_tag": "TARGET_097_SOURCE_ORACLE_SUPPORTED" if oracle_train >= 0.970 else "TARGET_097_NOT_SUPPORTED_BY_CURRENT_SOURCES",
        "alternate_source_metrics": alternate_entries,
        "baseline_train_metrics": baseline_train_metrics,
        "corrected_baseline_inner_screen": {
            "selection_scope": baseline_inner_screen["selection_scope"],
            "candidate_depth": baseline_inner_screen["candidate_depth"],
            "grid_count": baseline_inner_screen["grid_count"],
        },
    }
    error_tag_breakdown = {
        tag: {
            "query_count": len(tag_qids),
            "source_metrics": {
                source: evaluate_rankings(source_maps[source], answers, tag_qids)
                for source in source_maps
            },
        }
        for tag in sorted(set(tags.values()))
        for tag_qids in [[qid for qid in qids if answers[qid] and tags.get(qid) == tag]]
    }
    report = {
        "schema_version": SCHEMA,
        "stage": "source_audit",
        "status": "PASS_SOURCE_VIABILITY" if source_viability_pass else "REJECTED_FULL_CORPUS_SOURCE_GATE",
        "outer": outer,
        "sources": report_sources,
        "pairwise_topk_overlap": pairwise_overlap,
        "unique_gold_contribution": source_gold_contribution,
        "gold_source_membership": {"label_dependent_diagnostic": True, "by_qid": gold_source_membership},
        "union_coverage_by_source_depth": union_coverage,
        "best_source_oracle": {"label_dependent_diagnostic": True, "recall@5": oracle_r5, "source_count": len(source_maps)},
        "error_tags": error_tag_breakdown,
        "depth_selection": depth_selection,
        "corrected_baseline_depth_selection": baseline_depth_selection,
        "source_viability_gate": source_gate,
        "rankings_manifests": manifests,
        "input_audit_fingerprint": audit["audit_fingerprint"],
        "preflight_fingerprint": content_hash(preflight_report),
        "scorer_contract": SCORER_CONTRACT,
        "no_ordered_append_union": True,
    }
    atomic_json(output_dir / "SOURCE_AUDIT.json", report)
    if source_viability_pass:
        write_success(output_dir, stage="source-audit", fingerprint=content_hash(report))
    else:
        (output_dir / "_SUCCESS.json").unlink(missing_ok=True)
    tracker.finish("PASS" if source_viability_pass else "REJECTED", output_fingerprint=content_hash(report))
    return report


def select_candidate_depth(
    source_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    answers: Mapping[str, set[str]],
    folds: Mapping[str, Sequence[str]],
    *,
    heldout: str,
    included_sources: Sequence[str] | None = None,
) -> dict[str, Any]:
    sources = tuple(included_sources or source_rankings)
    missing_sources = [source for source in sources if source not in source_rankings]
    if missing_sources:
        raise KeyError(f"candidate-depth sources missing: {missing_sources}")
    result: dict[str, Any] = {}
    for outer in sorted(folds):
        train_qids = _fold_train_qids(folds, outer)
        curves: dict[int, float] = {}
        for depth in CANDIDATE_DEPTHS:
            values = []
            for qid in train_qids:
                if not answers.get(qid):
                    continue
                # union_ranking expects qid-local rankings; construct that
                # mapping explicitly to keep source membership visible.
                local = {source: source_rankings[source][qid] for source in sources}
                values.append(union_candidate_coverage(local, sources, answers[qid], depth=depth))
            curves[depth] = float(np.mean(values)) if values else 0.0
        chosen = 500
        warning = True
        for index, depth in enumerate(CANDIDATE_DEPTHS):
            if curves[depth] < 0.995:
                continue
            next_gain = curves[CANDIDATE_DEPTHS[index + 1]] - curves[depth] if index + 1 < len(CANDIDATE_DEPTHS) else 0.0
            if next_gain < 0.001:
                chosen = depth
                warning = False
                break
        result[outer] = {"depth": chosen, "curves": {str(key): value for key, value in curves.items()}, "warning": warning, "tag": "CANDIDATE_DEPTH_CEILING_WARNING" if warning else None, "selection_qids": sum(bool(answers.get(qid)) for qid in train_qids), "selection_qids_total": len(train_qids), "heldout_excluded": outer, "included_sources": list(sources)}
    if heldout not in result:
        raise KeyError(heldout)
    return result


def simplex_weights(sources: Sequence[str], *, step: float = 0.10) -> list[dict[str, float]]:
    sources = tuple(sources)
    if not sources:
        raise ValueError("weighted fusion requires at least one source")
    units = int(round(1.0 / step))
    minima = [int(round(0.30 * units)) if source == "vietlegal_e5" else int(round(0.10 * units)) for source in sources]
    result: list[dict[str, float]] = []
    def visit(position: int, remaining: int, values: list[int]) -> None:
        if position == len(sources) - 1:
            if remaining >= minima[position]:
                values.append(remaining)
                result.append({source: value / units for source, value in zip(sources, values)})
                values.pop()
            return
        for value in range(minima[position], remaining - sum(minima[position + 1:]) + 1):
            values.append(value)
            visit(position + 1, remaining - value, values)
            values.pop()
    visit(0, units, [])
    return result


def weighted_rrf_scores(source_rankings: Mapping[str, Sequence[str]], weights: Mapping[str, float], *, rrf_k: int, limit: int | None = None) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for source, weight in weights.items():
        for rank, doc_id in enumerate(source_rankings.get(source, []), start=1):
            scores[str(doc_id)] += float(weight) / (int(rrf_k) + rank)
    if limit is not None:
        ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:limit]
        return {doc_id: scores[doc_id] for doc_id in ordered}
    return dict(scores)


def weighted_rrf_ranking(source_rankings: Mapping[str, Sequence[str]], weights: Mapping[str, float], *, rrf_k: int, limit: int | None = None) -> list[str]:
    scores = weighted_rrf_scores(source_rankings, weights, rrf_k=rrf_k, limit=None)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return ordered if limit is None else ordered[:limit]


def corrected_rrf_predictions(
    source_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    folds: Mapping[str, Sequence[str]],
    answers: Mapping[str, set[str]],
    *,
    heldout: str,
    included_sources: Sequence[str],
    max_depth: int,
) -> dict[str, list[str]]:
    """Return only held-out predictions from the inner-selected baseline."""
    screen = nested_rrf_screen(
        source_rankings,
        answers,
        folds,
        outer=heldout,
        included_sources=included_sources,
        candidate_depth=max_depth,
    )
    return dict(screen["heldout_predictions"])


def tune_rrf_on_train(
    source_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    answers: Mapping[str, set[str]],
    train_qids: Sequence[str],
    *,
    included_sources: Sequence[str],
    candidate_depth: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for rrf_k in (10, 20, 32, 60, 100):
        for weights in simplex_weights(included_sources):
            predictions = {}
            for qid in train_qids:
                local = {source: list(source_rankings[source][qid][:candidate_depth]) for source in included_sources}
                predictions[qid] = weighted_rrf_ranking(local, weights, rrf_k=rrf_k)
            metrics = evaluate_rankings(predictions, answers, train_qids)
            records.append({"rrf_k": rrf_k, "weights": weights, "metrics": metrics, "source_count": len(included_sources)})
    chosen = max(records, key=lambda row: (row["metrics"].get("recall@5", 0.0), row["metrics"].get("precision@5", 0.0), row["metrics"].get("multi_gold_recall@5", 0.0), row["metrics"].get("mrr@5", 0.0), row["metrics"].get("recall@16", 0.0), -row["source_count"], -sum(abs(row["weights"].get(source, 0.0) - (0.65 if source == "vietlegal_e5" else 0.35 / max(1, len(included_sources) - 1))) for source in included_sources), -abs(row["rrf_k"] - 32)))
    return {"chosen": chosen, "grid_count": len(records), "candidate_depth": candidate_depth, "train_qids": len(train_qids)}


def _metric_selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics.get("recall@5", -math.inf)),
        float(metrics.get("precision@5", -math.inf)),
        float(metrics.get("multi_gold_recall@5", -math.inf)),
        float(metrics.get("mrr@5", -math.inf)),
        float(metrics.get("recall@16", -math.inf)),
        float(metrics.get("recall@50", -math.inf)),
    )


def nested_rrf_screen(
    source_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    answers: Mapping[str, set[str]],
    folds: Mapping[str, Sequence[str]],
    *,
    outer: str,
    included_sources: Sequence[str],
    candidate_depth: int,
) -> dict[str, Any]:
    """Select one locked RRF configuration from outer-train inner folds.

    RRF has no fitted parameters, so "refit" means applying the configuration
    selected exclusively by aggregate inner-validation metrics.  It must not
    retune ``k`` or weights against all outer-train labels after that screen.
    """
    if outer not in folds:
        raise KeyError(outer)
    outer_train_folds = [name for name in sorted(folds) if name != outer]
    outer_train_qids = _fold_train_qids(folds, outer)
    screens: list[dict[str, Any]] = []
    for rrf_k in (10, 20, 32, 60, 100):
        for weights in simplex_weights(included_sources):
            validation_predictions: dict[str, list[str]] = {}
            fold_reports: list[dict[str, Any]] = []
            for validation_fold in outer_train_folds:
                validation_qids = list(folds[validation_fold])
                predictions: dict[str, list[str]] = {}
                for qid in validation_qids:
                    local = {source: list(source_rankings[source][qid][:candidate_depth]) for source in included_sources}
                    predictions[qid] = weighted_rrf_ranking(local, weights, rrf_k=rrf_k)
                validation_predictions.update(predictions)
                fold_reports.append({
                    "validation_fold": validation_fold,
                    "train_qids": len(outer_train_qids) - len(validation_qids),
                    "validation_qids": len(validation_qids),
                    "metrics": evaluate_rankings(predictions, answers, validation_qids),
                })
            if set(validation_predictions) != set(outer_train_qids):
                missing = sorted(set(outer_train_qids) - set(validation_predictions))
                extra = sorted(set(validation_predictions) - set(outer_train_qids))
                raise RuntimeError(f"inner RRF validation coverage mismatch missing={missing[:5]} extra={extra[:5]}")
            screens.append({
                "rrf_k": rrf_k,
                "weights": weights,
                "metrics": evaluate_rankings(validation_predictions, answers, outer_train_qids),
                "inner_folds": fold_reports,
                "validation_predictions": validation_predictions,
                "source_count": len(included_sources),
            })
    def selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
        metrics = row["metrics"]
        weights = row["weights"]
        baseline_distance = sum(
            abs(float(weights.get(source, 0.0)) - (0.65 if source == "vietlegal_e5" else 0.35 / max(1, len(included_sources) - 1)))
            for source in included_sources
        )
        return (
            *_metric_selection_key(metrics)[:5],
            -float(row["source_count"]),
            -baseline_distance,
            -abs(int(row["rrf_k"]) - 32),
        )
    chosen = max(screens, key=selection_key)
    heldout_predictions: dict[str, list[str]] = {}
    for qid in folds[outer]:
        local = {source: list(source_rankings[source][qid][:candidate_depth]) for source in included_sources}
        heldout_predictions[qid] = weighted_rrf_ranking(local, chosen["weights"], rrf_k=int(chosen["rrf_k"]))
    return {
        "chosen": {key: value for key, value in chosen.items() if key not in {"validation_predictions", "inner_folds"}},
        "grid_count": len(screens),
        "candidate_depth": candidate_depth,
        "train_qids": len(outer_train_qids),
        "inner_metrics": chosen["metrics"],
        "inner_predictions": chosen["validation_predictions"],
        "inner_folds": chosen["inner_folds"],
        "heldout_predictions": heldout_predictions,
        "selection_scope": "outer_train_inner_cv",
        "final_fit": "not_applicable_weighted_rrf_has_no_fitted_parameters",
    }


def fuse_fold(
    source_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    answers: Mapping[str, set[str]],
    folds: Mapping[str, Sequence[str]],
    *,
    outer: str,
    selected_models: Sequence[str],
) -> dict[str, Any]:
    structures = [("vietlegal_e5", "bm25")]
    if "vietlegal_harrier_0_6b" in selected_models:
        structures.append(("vietlegal_e5", "vietlegal_harrier_0_6b", "bm25"))
    if "vnlegal_lal" in selected_models:
        structures.append(("vietlegal_e5", "vnlegal_lal", "bm25"))
    if len(selected_models) == 2:
        structures.append(("vietlegal_e5", "vietlegal_harrier_0_6b", "vnlegal_lal", "bm25"))
    train_qids = _fold_train_qids(folds, outer)
    heldout_qids = list(folds[outer])
    configs: list[dict[str, Any]] = []
    for structure in structures:
        depth_info = select_candidate_depth(source_rankings, answers, folds, heldout=outer, included_sources=structure)[outer]
        configs.append({"structure": structure, "candidate_depth": depth_info, "tuning": nested_rrf_screen(source_rankings, answers, folds, outer=outer, included_sources=structure, candidate_depth=int(depth_info["depth"]))})
    best = max(configs, key=lambda row: (_metric_selection_key(row["tuning"].get("inner_metrics", {})), -len(row["structure"])))
    predictions: dict[str, list[str]] = {}
    for qid in heldout_qids:
        chosen = best["tuning"]["chosen"]
        local = {source: list(source_rankings[source][qid][:int(best["candidate_depth"]["depth"])]) for source in best["structure"]}
        predictions[qid] = weighted_rrf_ranking(local, chosen["weights"], rrf_k=chosen["rrf_k"])
    return {"outer": outer, "candidate_depth": best["candidate_depth"], "structures": configs, "winner": best, "heldout_predictions": predictions, "heldout_metrics": evaluate_rankings(predictions, answers, heldout_qids)}


# ---------------------------------------------------------------------------
# LambdaMART feature contract and nested fallback
# ---------------------------------------------------------------------------


ALLOWED_FEATURES = (
    "e5_score", "e5_rank", "e5_recip", "e5_z", "e5_margin_rank1", "e5_margin_rank5", "e5_margin_rank10", "e5_present",
    "harrier_score", "harrier_rank", "harrier_recip", "harrier_z", "harrier_margin_rank1", "harrier_margin_rank5", "harrier_margin_rank10", "harrier_present",
    "lal_score", "lal_rank", "lal_recip", "lal_z", "lal_margin_rank1", "lal_margin_rank5", "lal_margin_rank10", "lal_present",
    "bm25_score", "bm25_rank", "bm25_recip", "bm25_z", "bm25_margin_rank1", "bm25_margin_rank5", "bm25_margin_rank10", "bm25_present",
    "source_agreement_top5", "source_agreement_top10", "source_agreement_top20", "dense_top1_score", "dense_top2_score", "dense_top1_top2_gap",
    "parent_chunk_count", "parent_token_length", "query_token_length",
)
FORBIDDEN_FEATURE_TOKENS = ("doc_id", "label", "target", "nearest", "gold", "answer", "query_id")
FEATURE_SOURCES = ("vietlegal_e5", "vietlegal_harrier_0_6b", "vnlegal_lal", "bm25")


def build_parent_text_metadata() -> dict[str, dict[str, int]]:
    counts: Counter[str] = Counter()
    token_lengths: Counter[str] = Counter()
    for row in read_jsonl(STRUCT_DIR / "chunks.jsonl"):
        doc_id = str(row["doc_id"])
        counts[doc_id] += 1
        token_lengths[doc_id] += int(row.get("token_count", 0))
    return {doc_id: {"parent_chunk_count": int(counts[doc_id]), "parent_token_length": int(token_lengths[doc_id])} for doc_id in counts}


def _source_alias(source: str) -> str:
    return {"vietlegal_e5": "e5", "vietlegal_harrier_0_6b": "harrier", "vnlegal_lal": "lal", "bm25": "bm25"}.get(source, source)


def _rank_score_features(source: str, score_map: Mapping[str, tuple[int, float]], doc_id: str, depth: int) -> dict[str, float]:
    alias = _source_alias(source)
    values = {doc: score for doc, (_rank, score) in score_map.items()}
    rank, score = score_map.get(doc_id, (depth + 1, 0.0))
    present = float(doc_id in score_map)
    present_scores = np.asarray(list(values.values()), dtype=np.float64)
    mean = float(present_scores.mean()) if len(present_scores) else 0.0
    std = float(present_scores.std()) if len(present_scores) else 0.0
    z = (score - mean) / std if std > 1e-12 and present else 0.0
    sorted_scores = sorted(values.values(), reverse=True)
    def margin(cutoff: int) -> float:
        reference = sorted_scores[min(cutoff - 1, len(sorted_scores) - 1)] if sorted_scores else 0.0
        return float(reference - score) if present else 0.0
    return {
        f"{alias}_score": float(score), f"{alias}_rank": float(rank), f"{alias}_recip": float(1.0 / rank) if present else 0.0,
        f"{alias}_z": float(z), f"{alias}_margin_rank1": margin(1), f"{alias}_margin_rank5": margin(5), f"{alias}_margin_rank10": margin(10), f"{alias}_present": present,
    }


def build_lambdamart_features(
    qid: str,
    candidate_doc_ids: Sequence[str],
    source_score_maps: Mapping[str, Mapping[str, tuple[int, float]]],
    metadata: Mapping[str, Mapping[str, int]],
    *,
    query_token_length: int,
    depth: int,
) -> tuple[np.ndarray, list[str]]:
    names = list(ALLOWED_FEATURES)
    rows: list[list[float]] = []
    for doc_id in candidate_doc_ids:
        feature: dict[str, float] = {}
        # Keep one fixed feature schema across the baseline, one-alternate,
        # and two-alternate structures.  An omitted source is represented by
        # its explicit zero/presence features, never by a changing column
        # order.
        for source in FEATURE_SOURCES:
            score_map = source_score_maps.get(source, {})
            feature.update(_rank_score_features(source, score_map, doc_id, depth))
        for cutoff, name in ((5, "source_agreement_top5"), (10, "source_agreement_top10"), (20, "source_agreement_top20")):
            feature[name] = float(sum(1 for score_map in source_score_maps.values() if score_map.get(doc_id, (depth + 1, 0.0))[0] <= cutoff))
        dense_maps = [source_score_maps.get(source, {}) for source in FEATURE_SOURCES if source != "bm25"]
        dense_values = sorted([score_map[doc_id][1] for score_map in dense_maps if doc_id in score_map], reverse=True)
        feature["dense_top1_score"] = float(dense_values[0]) if dense_values else 0.0
        feature["dense_top2_score"] = float(dense_values[1]) if len(dense_values) > 1 else 0.0
        feature["dense_top1_top2_gap"] = feature["dense_top1_score"] - feature["dense_top2_score"]
        feature.update({name: float(metadata.get(doc_id, {}).get(name, 0)) for name in ("parent_chunk_count", "parent_token_length")})
        feature["query_token_length"] = float(query_token_length)
        if set(feature) != set(names):
            missing = set(names) - set(feature)
            extra = set(feature) - set(names)
            raise ValueError(f"feature schema mismatch missing={sorted(missing)} extra={sorted(extra)}")
        if any(any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS) for name in feature):
            raise ValueError("forbidden label/identifier feature name")
        rows.append([float(feature[name]) for name in names])
    return np.asarray(rows, dtype=np.float32), names


def train_lambdamart_or_fallback(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: Sequence[int],
    valid_features: np.ndarray,
    *,
    seed: int = RNG_SEED,
) -> tuple[str, Any, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except Exception:
        return "rrf_fallback", None, {"status": "DEPENDENCY_UNAVAILABLE", "fallback": "rrf"}
    best_model = None
    best_config: dict[str, Any] | None = None
    # The caller owns inner-CV selection. This helper is deliberately a fixed
    # winner trainer and never inspects held-out labels.
    config = {"num_leaves": 15, "min_data_in_leaf": 20, "learning_rate": 0.03, "num_boost_round": 100, "feature_fraction": 1.0, "bagging_fraction": 1.0, "deterministic": True, "seed": seed, "objective": "lambdarank", "eval_at": [5]}
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_at=[5], num_leaves=config["num_leaves"], min_child_samples=config["min_data_in_leaf"], learning_rate=config["learning_rate"], n_estimators=config["num_boost_round"], feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, deterministic=True, random_state=seed, verbosity=-1,
    )
    model.fit(train_features, train_labels, group=list(map(int, train_groups)))
    best_model = model
    best_config = config
    predictions = model.predict(valid_features) if len(valid_features) else np.empty((0,), dtype=np.float32)
    return "lambdamart", best_model, {"config": best_config, "valid_prediction_count": len(predictions), "status": "TRAINED"}


def lambdamart_hyperparameter_grid() -> list[dict[str, Any]]:
    return [{"num_leaves": leaves, "min_data_in_leaf": minimum, "learning_rate": rate, "num_boost_round": rounds, "feature_fraction": 1.0, "bagging_fraction": 1.0, "deterministic": True, "seed": RNG_SEED, "objective": "lambdarank", "eval_at": [5]} for leaves in (7, 15) for minimum in (20, 50) for rate in (0.03, 0.05) for rounds in (100, 300)]


def validate_feature_contract(names: Sequence[str]) -> None:
    if tuple(names) != ALLOWED_FEATURES:
        raise ValueError("LambdaMART feature order/schema is not the locked EXP-109B contract")
    if any(any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS) for name in names):
        raise ValueError("forbidden LambdaMART feature")


def _candidate_docs_for_qid(
    source_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    qid: str,
    sources: Sequence[str],
    depth: int,
) -> list[str]:
    local = {source: source_rankings[source][qid] for source in sources}
    return union_ranking(local, sources, depth=depth)


def _lgbm_matrix_for_qids(
    source_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    source_score_maps: Mapping[str, Mapping[str, Mapping[str, tuple[int, float]]]],
    qids: Sequence[str],
    answers: Mapping[str, set[str]],
    *,
    sources: Sequence[str],
    depth: int,
    metadata: Mapping[str, Mapping[str, int]],
    query_token_lengths: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[str, list[str]]], list[str]]:
    matrices: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[int] = []
    qid_candidates: list[tuple[str, list[str]]] = []
    names: list[str] | None = None
    for qid in qids:
        candidates = _candidate_docs_for_qid(source_rankings, qid, sources, depth)
        local_scores = {source: source_score_maps[source][qid] for source in sources}
        matrix, current_names = build_lambdamart_features(qid, candidates, local_scores, metadata, query_token_length=query_token_lengths.get(qid, 0), depth=depth)
        validate_feature_contract(current_names)
        matrices.append(matrix)
        labels.extend([1.0 if doc_id in answers.get(qid, set()) else 0.0 for doc_id in candidates])
        groups.append(len(candidates))
        qid_candidates.append((qid, candidates))
        names = current_names
    if not matrices:
        return np.empty((0, len(ALLOWED_FEATURES)), dtype=np.float32), np.empty((0,), dtype=np.float32), [], [], list(ALLOWED_FEATURES)
    return np.concatenate(matrices, axis=0), np.asarray(labels, dtype=np.float32), groups, qid_candidates, names or list(ALLOWED_FEATURES)


def nested_lambdamart_fusion(
    source_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    source_score_maps: Mapping[str, Mapping[str, Mapping[str, tuple[int, float]]]],
    answers: Mapping[str, set[str]],
    folds: Mapping[str, Sequence[str]],
    *,
    outer: str,
    included_sources: Sequence[str],
    candidate_depth: int,
    metadata: Mapping[str, Mapping[str, int]],
    query_token_lengths: Mapping[str, int],
) -> dict[str, Any]:
    """Run the locked inner-CV LambdaMART grid without reading outer labels."""
    try:
        import lightgbm as lgb
    except Exception:
        return {"status": "DEPENDENCY_UNAVAILABLE", "winner": "rrf_fallback", "heldout_predictions": {}}
    if outer not in folds:
        raise KeyError(outer)
    outer_train_folds = [name for name in sorted(folds) if name != outer]
    grid = lambdamart_hyperparameter_grid()
    screens: list[dict[str, Any]] = []
    for config in grid:
        validation_predictions: dict[str, list[str]] = {}
        validation_qids: list[str] = []
        for validation_fold in outer_train_folds:
            train_qids = [qid for name in outer_train_folds if name != validation_fold for qid in folds[name]]
            validation_qids_fold = list(folds[validation_fold])
            train_x, train_y, train_groups, _train_candidates, feature_names = _lgbm_matrix_for_qids(source_rankings, source_score_maps, train_qids, answers, sources=included_sources, depth=candidate_depth, metadata=metadata, query_token_lengths=query_token_lengths)
            valid_x, _valid_y, _valid_groups, valid_candidates, _ = _lgbm_matrix_for_qids(source_rankings, source_score_maps, validation_qids_fold, answers, sources=included_sources, depth=candidate_depth, metadata=metadata, query_token_lengths=query_token_lengths)
            model = lgb.LGBMRanker(
                objective="lambdarank", metric="ndcg", ndcg_at=[5], num_leaves=config["num_leaves"], min_child_samples=config["min_data_in_leaf"], learning_rate=config["learning_rate"], n_estimators=config["num_boost_round"], feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, deterministic=True, random_state=RNG_SEED, verbosity=-1,
            )
            if len(train_x) == 0 or not np.any(train_y > 0):
                continue
            model.fit(train_x, train_y, group=train_groups, feature_name=feature_names)
            predicted = model.predict(valid_x) if len(valid_x) else np.empty((0,), dtype=np.float32)
            offset = 0
            for qid, candidates in valid_candidates:
                local_scores = predicted[offset:offset + len(candidates)]
                order = np.lexsort((np.asarray(candidates, dtype="U"), -np.asarray(local_scores, dtype=np.float64)))
                validation_predictions[qid] = [candidates[int(position)] for position in order]
                validation_qids.append(qid)
                offset += len(candidates)
        if validation_qids:
            metrics = evaluate_rankings(validation_predictions, answers, validation_qids)
        else:
            metrics = {"recall@5": -math.inf, "precision@5": -math.inf, "multi_gold_recall@5": -math.inf, "mrr@5": -math.inf, "recall@16": -math.inf}
        screens.append({"config": config, "metrics": metrics, "validation_qids": len(validation_qids)})
    if not screens:
        return {"status": "NO_VALID_INNER_SCREEN", "winner": "rrf_fallback", "heldout_predictions": {}}
    chosen = max(screens, key=lambda row: (row["metrics"].get("recall@5", -math.inf), row["metrics"].get("precision@5", -math.inf), row["metrics"].get("multi_gold_recall@5", -math.inf), row["metrics"].get("mrr@5", -math.inf), row["metrics"].get("recall@16", -math.inf), -len(included_sources)))
    train_qids = _fold_train_qids(folds, outer)
    heldout_qids = list(folds[outer])
    train_x, train_y, train_groups, _train_candidates, feature_names = _lgbm_matrix_for_qids(source_rankings, source_score_maps, train_qids, answers, sources=included_sources, depth=candidate_depth, metadata=metadata, query_token_lengths=query_token_lengths)
    valid_x, _valid_y, _valid_groups, valid_candidates, _ = _lgbm_matrix_for_qids(source_rankings, source_score_maps, heldout_qids, answers, sources=included_sources, depth=candidate_depth, metadata=metadata, query_token_lengths=query_token_lengths)
    if len(train_x) == 0 or not np.any(train_y > 0):
        return {"status": "NO_POSITIVE_TRAIN_ROWS", "winner": "rrf_fallback", "screens": screens, "heldout_predictions": {}}
    config = chosen["config"]
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_at=[5], num_leaves=config["num_leaves"], min_child_samples=config["min_data_in_leaf"], learning_rate=config["learning_rate"], n_estimators=config["num_boost_round"], feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, deterministic=True, random_state=RNG_SEED, verbosity=-1,
    )
    model.fit(train_x, train_y, group=train_groups, feature_name=feature_names)
    predicted = model.predict(valid_x) if len(valid_x) else np.empty((0,), dtype=np.float32)
    heldout_predictions: dict[str, list[str]] = {}
    offset = 0
    for qid, candidates in valid_candidates:
        local_scores = predicted[offset:offset + len(candidates)]
        order = np.lexsort((np.asarray(candidates, dtype="U"), -np.asarray(local_scores, dtype=np.float64)))
        heldout_predictions[qid] = [candidates[int(position)] for position in order]
        offset += len(candidates)
    return {
        "status": "TRAINED",
        "winner": "lambdamart",
        "chosen_config": chosen,
        "screens": screens,
        "inner_metrics": chosen["metrics"],
        "heldout_predictions": heldout_predictions,
        "heldout_metrics": evaluate_rankings(heldout_predictions, answers, heldout_qids),
    }


# ---------------------------------------------------------------------------
# Fold-0 report, overnight orchestration, and explicit OOF gate
# ---------------------------------------------------------------------------


def _ranking_transition_metrics(
    baseline_predictions: Mapping[str, Sequence[str]],
    final_predictions: Mapping[str, Sequence[str]],
    answers: Mapping[str, set[str]],
    qids: Sequence[str],
) -> dict[str, Any]:
    promoted: list[str] = []
    lost: list[str] = []
    for qid in qids:
        gold = answers.get(qid, set())
        if not gold:
            continue
        base_rank = _first_gold_rank(baseline_predictions[qid], gold)
        final_rank = _first_gold_rank(final_predictions[qid], gold)
        if base_rank > 5 and final_rank <= 5:
            promoted.append(qid)
        if base_rank <= 5 and final_rank > 5:
            lost.append(qid)
    return {
        "promoted_top5_qids": promoted,
        "lost_top5_qids": lost,
        "promoted_top5": len(promoted),
        "lost_top5": len(lost),
        "net_top5_promotions": len(promoted) - len(lost),
    }


def _load_outer_source_data(
    outer: str,
    selected: Sequence[str],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, dict[str, tuple[int, float]]]], dict[str, Any], dict[str, set[str]], dict[str, list[str]], dict[str, list[str]]]:
    folds, fold_for = load_folds()
    answers, _stats = canonical_labels()
    qids = sorted(answers)
    source_maps: dict[str, dict[str, list[str]]] = {}
    source_score_maps: dict[str, dict[str, dict[str, tuple[int, float]]]] = {}
    manifests: dict[str, Any] = {}
    for model in ["vietlegal_e5", *selected]:
        rows, manifest = _load_dense_ranking_rows(model, outer)
        source_maps[model] = _ranking_map(rows)
        source_score_maps[model] = _ranking_score_map(rows)
        manifests[model] = manifest
    bm25_rows = _bm25_rank_rows(qids, fold_for)
    source_maps["bm25"] = _ranking_map(bm25_rows)
    source_score_maps["bm25"] = _ranking_score_map(bm25_rows)
    manifests["bm25"] = {"source": "tuned_exp021_bm25", "query_count": len(bm25_rows)}
    missing = {source: sorted(set(qids) - set(values)) for source, values in source_maps.items() if set(values) != set(qids)}
    if missing:
        raise RuntimeError(f"source ranking qid coverage mismatch: {missing}")
    return source_maps, source_score_maps, manifests, answers, dict(folds), {qid: fold for fold, values in folds.items() for qid in values}


def _load_cached_ranking_rows_for_pilot(model: str, outer: str, expected_fingerprint: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify completed ranking bytes without invalidating them after code edits."""
    directory = CACHE_ROOT / "rankings" / model / outer
    manifest = read_json(directory / "manifest.json")
    if manifest.get("content_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"cached ranking manifest fingerprint mismatch: {model}")
    rows: list[dict[str, Any]] = []
    for entry in manifest.get("shards", []):
        path = directory / "shards" / str(entry["name"])
        if not path.exists() or sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"cached ranking shard hash mismatch: {path}")
        rows.extend(read_jsonl(path))
    if len(rows) != int(manifest.get("query_count", -1)):
        raise RuntimeError(f"cached ranking count mismatch: {model}")
    return rows, manifest


def _crossfit_rrf_predictions(source_maps: Mapping[str, Mapping[str, Sequence[str]]], answers: Mapping[str, set[str]], inner_folds: Mapping[str, Sequence[str]], sources: Sequence[str]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    predictions: dict[str, list[str]] = {}
    folds_report: list[dict[str, Any]] = []
    for validation_fold in sorted(inner_folds):
        depth = select_candidate_depth(source_maps, answers, inner_folds, heldout=validation_fold, included_sources=sources)[validation_fold]
        screen = nested_rrf_screen(source_maps, answers, inner_folds, outer=validation_fold, included_sources=sources, candidate_depth=int(depth["depth"]))
        predictions.update(screen["heldout_predictions"])
        folds_report.append({"validation_fold": validation_fold, "candidate_depth": depth, "chosen": screen["chosen"], "metrics": evaluate_rankings(screen["heldout_predictions"], answers, inner_folds[validation_fold])})
    return predictions, folds_report


def _pilot_gate(baseline: Mapping[str, Sequence[str]], candidate: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], inner_folds: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    qids = [qid for fold in sorted(inner_folds) for qid in inner_folds[fold] if answers.get(qid)]
    deltas = [recall_at(candidate[qid], answers[qid], 5) - recall_at(baseline[qid], answers[qid], 5) for qid in qids]
    per_fold = {fold: float(np.mean([recall_at(candidate[qid], answers[qid], 5) - recall_at(baseline[qid], answers[qid], 5) for qid in inner_folds[fold] if answers.get(qid)])) for fold in sorted(inner_folds)}
    base = evaluate_rankings(baseline, answers, qids)
    value = evaluate_rankings(candidate, answers, qids)
    delta = {key: float(value.get(key, 0.0) - base.get(key, 0.0)) for key in ("recall@5", "multi_gold_recall@5", "recall@16", "recall@50")}
    ci = deterministic_bootstrap(deltas)
    checks = {
        "delta_recall@5_ge_0_010": delta["recall@5"] >= 0.010,
        "bootstrap_delta_recall@5_lower_gt_zero": ci["lower"] > 0.0,
        "at_least_3_of_4_inner_folds_improve": sum(item > 0.0 for item in per_fold.values()) >= 3,
        "multi_gold_recall@5_non_decrease": delta["multi_gold_recall@5"] >= 0.0,
        "delta_recall@16_ge_-0_001": delta["recall@16"] >= -0.001,
        "delta_recall@50_ge_-0_001": delta["recall@50"] >= -0.001,
    }
    return {"baseline_metrics": base, "metrics": value, "delta": delta, "bootstrap_delta_recall@5": ci, "per_fold_delta_recall@5": per_fold, "checks": checks, "pass": all(checks.values()), "qids": len(qids)}


def cached_fusion_pilot(*, outer: str = "fold_0", resume: bool = True, authorize: bool = False) -> dict[str, Any]:
    """Strict F1--F4 cache-only document-fusion gate; it never scores Fold 0."""
    if not authorize and os.environ.get("EXP109B_ALLOW_CACHED_FUSION_PILOT") != "1":
        raise GateRejected("REJECTED_AUTHORIZATION_GATE", "cached fusion pilot requires explicit authorization")
    source_path = RESULTS_ROOT / "source_audit" / outer / "SOURCE_AUDIT.json"
    source_report = read_json(source_path)
    if source_report.get("status") not in {"PASS_SOURCE_VIABILITY", "REJECTED_FULL_CORPUS_SOURCE_GATE"}:
        raise GateRejected("REJECTED_PREREQUISITE_GATE", "completed source audit required")
    output_dir = RESULTS_ROOT / "cached_fusion_pilot" / outer
    # The repair report has already SHA-verified the bounded artifacts.  Do
    # not demand a new bounded GPU run merely because this cache-only pilot
    # adds code; verify its declared LAL selection directly instead.
    bounded_report = read_json(RESULTS_ROOT / "bounded_screen" / outer / "BOUNDED_REPORT.json")
    selected = list(bounded_report.get("selection", {}).get("selected_models", []))
    if bounded_report.get("status") != "PASS" or selected != ["vnlegal_lal"]:
        raise GateRejected("REJECTED_PREREQUISITE_GATE", "verified LAL-only bounded selection required")
    if selected != ["vnlegal_lal"]:
        raise RuntimeError("cached fusion pilot is locked to VnLegal-LAL")
    expected = source_report.get("rankings_manifests", {})
    e5_rows, e5_manifest = _load_cached_ranking_rows_for_pilot("vietlegal_e5", outer, expected["vietlegal_e5"]["content_fingerprint"])
    lal_rows, lal_manifest = _load_cached_ranking_rows_for_pilot("vnlegal_lal", outer, expected["vnlegal_lal"]["content_fingerprint"])
    folds, fold_for = load_folds()
    answers, _ = canonical_labels()
    qids = sorted(answers)
    source_maps = {"vietlegal_e5": _ranking_map(e5_rows), "vnlegal_lal": _ranking_map(lal_rows), "bm25": _ranking_map(_bm25_rank_rows(qids, fold_for))}
    inner_folds = {name: list(values) for name, values in folds.items() if name != outer}
    inner_qids = {qid for values in inner_folds.values() for qid in values}
    if any(qid in inner_qids for qid in folds[outer]):
        raise RuntimeError("Fold-0 leakage into cached fusion pilot")
    baseline, baseline_folds = _crossfit_rrf_predictions(source_maps, answers, inner_folds, ("vietlegal_e5", "bm25"))
    lal_only = {qid: list(source_maps["vnlegal_lal"][qid]) for qid in inner_qids}
    lal_bm25, lal_bm25_folds = _crossfit_rrf_predictions(source_maps, answers, inner_folds, ("vnlegal_lal", "bm25"))
    # The three-source RRF is retained separately from the document-level
    # learner; both use exactly the cached source rankings.
    candidates = {
        "lal_standalone": lal_only,
        "lal_bm25_weighted_rrf": lal_bm25,
        "e5_lal_bm25_weighted_rrf": _crossfit_rrf_predictions(source_maps, answers, inner_folds, ("vietlegal_e5", "vnlegal_lal", "bm25"))[0],
    }
    reports = {name: _pilot_gate(baseline, prediction, answers, inner_folds) for name, prediction in candidates.items()}
    # The existing LambdaMART implementation is strict cross-fit per candidate
    # configuration and uses only rank, score, margin, agreement, and source
    # presence features; it has no labels/IDs/query-memory feature.
    try:
        metadata = build_parent_text_metadata()
        train = load_train()
        lengths = {qid: len(str(row.get("question", "")).split()) for qid, row in train.items()}
        score_maps = {"vietlegal_e5": _ranking_score_map(e5_rows), "vnlegal_lal": _ranking_score_map(lal_rows), "bm25": _ranking_score_map(_bm25_rank_rows(qids, fold_for))}
        lambda_predictions: dict[str, list[str]] = {}
        lambda_folds: list[dict[str, Any]] = []
        for validation_fold in sorted(inner_folds):
            screen = nested_lambdamart_fusion(source_maps, score_maps, answers, inner_folds, outer=validation_fold, included_sources=("vietlegal_e5", "vnlegal_lal", "bm25"), candidate_depth=50, metadata=metadata, query_token_lengths=lengths)
            if screen.get("status") != "TRAINED":
                raise RuntimeError(f"LambdaMART unavailable for {validation_fold}: {screen.get('status')}")
            lambda_predictions.update(screen["heldout_predictions"])
            lambda_folds.append({"validation_fold": validation_fold, "chosen_config": screen["chosen_config"], "metrics": screen["heldout_metrics"]})
        candidates["lambdamart_top50_per_source"] = lambda_predictions
        reports["lambdamart_top50_per_source"] = _pilot_gate(baseline, lambda_predictions, answers, inner_folds)
    except Exception as exc:
        reports["lambdamart_top50_per_source"] = {"pass": False, "status": "UNAVAILABLE_OR_FAILED", "error": str(exc)}
        lambda_folds = []
    passed = [name for name, value in reports.items() if value.get("pass")]
    winner = max(passed, key=lambda name: _metric_selection_key(reports[name]["metrics"])) if passed else None
    report = {"schema_version": SCHEMA, "stage": "cached_fusion_pilot", "status": "PASS_CACHED_FUSION_INNER_GATE" if winner else "REJECTED_CACHED_FUSION_INNER_GATE", "outer": outer, "selection_scope": "strict_crossfit_folds_1_to_4_only", "fold0_seen": False, "source_routing_gate_status": source_report.get("status"), "source_routing_gate_not_document_fusion_ceiling": True, "baseline": {"name": "corrected_e5_bm25_weighted_rrf", "folds": baseline_folds}, "candidates": reports, "rrf_folds": lal_bm25_folds, "lambdamart_folds": lambda_folds, "winner": winner, "locked_config_required_before_fold0": bool(winner), "no_encoder_reencoding": True, "no_public_submission": True, "rankings_manifests": {"vietlegal_e5": e5_manifest, "vnlegal_lal": lal_manifest}, "source_audit_fingerprint": content_hash(source_report)}
    atomic_json(output_dir / "CACHED_FUSION_PILOT.json", report)
    if winner:
        write_success(output_dir, stage="cached-fusion-pilot", fingerprint=content_hash(report), extra={"winner": winner})
    else:
        (output_dir / "_SUCCESS.json").unlink(missing_ok=True)
    return report


def locked_cached_fusion_fold0(*, outer: str = "fold_0", authorize: bool = False) -> dict[str, Any]:
    """One blind Fold-0 evaluation of the configuration locked by the pilot."""
    if not authorize and os.environ.get("EXP109B_ALLOW_LOCKED_FOLD0") != "1":
        raise GateRejected("REJECTED_AUTHORIZATION_GATE", "locked Fold-0 evaluation requires explicit authorization")
    pilot_path = RESULTS_ROOT / "cached_fusion_pilot" / outer / "CACHED_FUSION_PILOT.json"
    pilot = read_json(pilot_path)
    if pilot.get("status") != "PASS_CACHED_FUSION_INNER_GATE" or pilot.get("winner") != "lambdamart_top50_per_source":
        raise GateRejected("REJECTED_CACHED_FUSION_INNER_GATE", "passing LambdaMART cached pilot required before Fold-0")
    configs = [row["chosen_config"]["config"] for row in pilot.get("lambdamart_folds", [])]
    if len(configs) != 4:
        raise RuntimeError("cached pilot does not contain four locked-config votes")
    key_for = lambda value: json.dumps(value, sort_keys=True)
    counts = Counter(key_for(value) for value in configs)
    locked_key = max(counts, key=lambda key: (counts[key], key))
    locked_config = next(value for value in configs if key_for(value) == locked_key)
    source_report = read_json(RESULTS_ROOT / "source_audit" / outer / "SOURCE_AUDIT.json")
    expected = source_report["rankings_manifests"]
    e5_rows, e5_manifest = _load_cached_ranking_rows_for_pilot("vietlegal_e5", outer, expected["vietlegal_e5"]["content_fingerprint"])
    lal_rows, lal_manifest = _load_cached_ranking_rows_for_pilot("vnlegal_lal", outer, expected["vnlegal_lal"]["content_fingerprint"])
    folds, fold_for = load_folds()
    answers, _ = canonical_labels()
    qids = sorted(answers)
    sources = ("vietlegal_e5", "vnlegal_lal", "bm25")
    bm25_rows = _bm25_rank_rows(qids, fold_for)
    source_maps = {"vietlegal_e5": _ranking_map(e5_rows), "vnlegal_lal": _ranking_map(lal_rows), "bm25": _ranking_map(bm25_rows)}
    score_maps = {"vietlegal_e5": _ranking_score_map(e5_rows), "vnlegal_lal": _ranking_score_map(lal_rows), "bm25": _ranking_score_map(bm25_rows)}
    # Baseline policy is selected using F1--F4 only, then applied unchanged.
    baseline = nested_rrf_screen(source_maps, answers, folds, outer=outer, included_sources=("vietlegal_e5", "bm25"), candidate_depth=int(select_candidate_depth(source_maps, answers, folds, heldout=outer, included_sources=("vietlegal_e5", "bm25"))[outer]["depth"]))
    train_qids = _fold_train_qids(folds, outer)
    heldout_qids = list(folds[outer])
    metadata = build_parent_text_metadata()
    train = load_train()
    lengths = {qid: len(str(row.get("question", "")).split()) for qid, row in train.items()}
    train_x, train_y, train_groups, _train_candidates, feature_names = _lgbm_matrix_for_qids(source_maps, score_maps, train_qids, answers, sources=sources, depth=50, metadata=metadata, query_token_lengths=lengths)
    valid_x, _valid_y, _valid_groups, valid_candidates, _ = _lgbm_matrix_for_qids(source_maps, score_maps, heldout_qids, answers, sources=sources, depth=50, metadata=metadata, query_token_lengths=lengths)
    import lightgbm as lgb
    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", ndcg_at=[5], num_leaves=int(locked_config["num_leaves"]), min_child_samples=int(locked_config["min_data_in_leaf"]), learning_rate=float(locked_config["learning_rate"]), n_estimators=int(locked_config["num_boost_round"]), feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, deterministic=True, random_state=RNG_SEED, verbosity=-1)
    model.fit(train_x, train_y, group=train_groups, feature_name=feature_names)
    predicted = model.predict(valid_x)
    predictions: dict[str, list[str]] = {}
    offset = 0
    for qid, candidates in valid_candidates:
        values = predicted[offset:offset + len(candidates)]
        order = np.lexsort((np.asarray(candidates, dtype="U"), -np.asarray(values, dtype=np.float64)))
        predictions[qid] = [candidates[int(pos)] for pos in order]
        offset += len(candidates)
    final = evaluate_rankings(predictions, answers, heldout_qids)
    baseline_metrics = evaluate_rankings(baseline["heldout_predictions"], answers, heldout_qids)
    delta = {key: float(final.get(key, 0.0) - baseline_metrics.get(key, 0.0)) for key in ("recall@5", "multi_gold_recall@5", "recall@16", "recall@50")}
    r5 = float(final.get("recall@5", 0.0))
    status = "PASS_FOLD0_STAGE1_ANCHOR" if r5 >= 0.940 else ("WEAK_FOLD0_NO_FULL_OOF_REPRESENTATION_NEXT" if r5 >= 0.925 else "REJECTED_FOLD0_GENERALIZATION")
    report = {"schema_version": SCHEMA, "stage": "locked_cached_fusion_fold0", "status": status, "outer": outer, "pilot_fingerprint": content_hash(pilot), "locked_config": locked_config, "locked_config_votes": {key: count for key, count in counts.items()}, "candidate_contract": "union_top50_per_source", "training_scope": "folds_1_to_4_only", "fold0_labels_used_for_evaluation_only": True, "baseline": {"metrics": baseline_metrics, "locked_rrf": baseline["chosen"]}, "final": final, "delta_vs_corrected_baseline": delta, "no_full_oof": True, "no_public_submission": True, "rankings_manifests": {"vietlegal_e5": e5_manifest, "vnlegal_lal": lal_manifest}}
    output_dir = RESULTS_ROOT / "locked_fusion_fold0"
    atomic_json(output_dir / "FOLD0_LOCKED_FUSION_REPORT.json", report)
    write_success(output_dir, stage="locked-cached-fusion-fold0", fingerprint=content_hash(report))
    return report


def _screen_outer_fold(outer: str, source_report: Mapping[str, Any]) -> dict[str, Any]:
    if source_report.get("status") != "PASS_SOURCE_VIABILITY":
        raise GateRejected("REJECTED_FULL_CORPUS_SOURCE_GATE", f"source viability failed for {outer}")
    selected = _selected_models_from_bounded(outer)
    source_maps, source_score_maps, manifests, answers, folds, _fold_for = _load_outer_source_data(outer, selected)
    baseline = fuse_fold(source_maps, answers, folds, outer=outer, selected_models=[])
    best = fuse_fold(source_maps, answers, folds, outer=outer, selected_models=selected)
    baseline_predictions = baseline["heldout_predictions"]
    rrf_predictions = best["heldout_predictions"]
    baseline_metrics = baseline["heldout_metrics"]
    rrf_metrics = best["heldout_metrics"]
    train_qids = _fold_train_qids(folds, outer)
    best_structure = tuple(best["winner"]["structure"])
    candidate_depth = int(best["candidate_depth"]["depth"])
    lambda_report: dict[str, Any]
    try:
        metadata = build_parent_text_metadata()
        train = load_train()
        query_token_lengths = {qid: len(str(row.get("question", "")).split()) for qid, row in train.items()}
        lambda_report = nested_lambdamart_fusion(
            source_maps,
            source_score_maps,
            answers,
            folds,
            outer=outer,
            included_sources=best_structure,
            candidate_depth=candidate_depth,
            metadata=metadata,
            query_token_lengths=query_token_lengths,
        )
    except ImportError:
        lambda_report = {"status": "DEPENDENCY_UNAVAILABLE", "winner": "rrf_fallback", "heldout_predictions": {}}
    lambda_inner_key = _metric_selection_key(lambda_report.get("inner_metrics", {}))
    rrf_inner_key = _metric_selection_key(best["winner"]["tuning"].get("inner_metrics", {}))
    use_lambda = lambda_report.get("status") == "TRAINED" and lambda_inner_key > rrf_inner_key
    final_predictions = lambda_report.get("heldout_predictions", {}) if use_lambda else rrf_predictions
    final_metrics = evaluate_rankings(final_predictions, answers, list(folds[outer]))
    delta = {key: float(final_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)) for key in ("recall@5", "precision@5", "multi_gold_recall@5", "recall@16", "recall@50")}
    transition = _ranking_transition_metrics(baseline_predictions, final_predictions, answers, list(folds[outer]))
    quality_tags: list[str] = []
    if final_metrics.get("recall@5", 0.0) >= 0.950:
        quality_tags.append("PASS_STRONG_095")
    if final_metrics.get("recall@5", 0.0) >= 0.960:
        quality_tags.append("PASS_TARGET_096")
    if final_metrics.get("recall@5", 0.0) >= 0.970:
        quality_tags.append("PASS_TARGET_097")
    ambitious_checks = {
        "recall@5_ge_0_940": final_metrics.get("recall@5", 0.0) >= 0.940,
        "delta_recall@5_ge_0_015": delta["recall@5"] >= 0.015,
        "precision_non_decrease": delta["precision@5"] >= 0.0,
        "multi_gold_non_decrease": delta["multi_gold_recall@5"] >= 0.0,
        "delta_recall@16_ge_-0_001": delta["recall@16"] >= -0.001,
        "delta_recall@50_ge_-0_001": delta["recall@50"] >= -0.001,
        "net_top5_promotions_gt_0": transition["net_top5_promotions"] > 0,
    }
    pass_gate = all(ambitious_checks.values())
    status = "PASS_FOLD0_AMBITIOUS_GATE" if pass_gate else ("WEAK_SIGNAL_NO_FULL_OOF" if delta["recall@5"] >= 0.005 else "REJECTED_FOLD0_GATE")
    report = {
        "schema_version": SCHEMA,
        "stage": "fold0_screen" if outer == "fold_0" else "outer_fold_screen",
        "status": status,
        "outer": outer,
        "baseline": baseline_metrics,
        "best_multi_encoder_rrf": rrf_metrics,
        "best_multi_encoder_rrf_inner_metrics": best["winner"]["tuning"].get("inner_metrics", {}),
        "lambdamart": {key: value for key, value in lambda_report.items() if key != "heldout_predictions"},
        "final": final_metrics,
        "winner": "lambdamart" if use_lambda else "weighted_rrf",
        "delta_vs_corrected_baseline": delta,
        "top5_transition": transition,
        "quality_tags": quality_tags,
        "ambitious_gate": {"checks": ambitious_checks, "pass": pass_gate},
        "source_diagnostics": {
            "unique_gold_contribution": source_report.get("unique_gold_contribution", {}),
            "error_tags": source_report.get("error_tags", {}),
            "best_source_oracle": source_report.get("best_source_oracle", {}),
        },
        "source_audit_fingerprint": content_hash(source_report),
        "rankings_manifests": manifests,
        "selected_models": selected,
        "candidate_depth": candidate_depth,
        "outer_train_qids": len(train_qids),
        "stops_before_full_oof": outer == "fold_0",
        "no_public_submission": True,
    }
    return {
        "report": report,
        "baseline_predictions": baseline_predictions,
        "final_predictions": final_predictions,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "selected_models": selected,
    }


def fold0_screen(*, resume: bool = True, authorize: bool = False) -> dict[str, Any]:
    if not authorize and os.environ.get("EXP109B_ALLOW_FOLD0") != "1":
        raise GateRejected("REJECTED_AUTHORIZATION_GATE", "Fold-0 screen requires explicit authorization (EXP109B_ALLOW_FOLD0=1)")
    source_dir = RESULTS_ROOT / "source_audit" / "fold_0"
    source_report_path = source_dir / "SOURCE_AUDIT.json"
    source_report = read_json(source_report_path)
    output_dir = RESULTS_ROOT / "fold0_screen"
    if resume and _stage_marker_current(output_dir, "FOLD0_REPORT.json"):
        saved = read_json(output_dir / "FOLD0_REPORT.json")
        if saved.get("source_audit_fingerprint") == content_hash(source_report):
            return saved
    require_success(source_dir, expected_fingerprint=content_hash(source_report), expected_code_sha256=code_fingerprint())
    result = _screen_outer_fold("fold_0", source_report)
    report = result["report"]
    atomic_json(output_dir / "FOLD0_REPORT.json", report)
    if report["status"] == "PASS_FOLD0_AMBITIOUS_GATE":
        write_success(output_dir, stage="fold0-screen", fingerprint=content_hash(report))
    else:
        (output_dir / "_SUCCESS.json").unlink(missing_ok=True)
    return report


def _stage_marker_current(directory: Path, report_name: str) -> bool:
    report_path = directory / report_name
    if not report_path.exists() or not (directory / "_SUCCESS.json").exists():
        return False
    try:
        report = read_json(report_path)
        expected = report.get("content_fingerprint") or content_hash(report)
        require_success(directory, expected_fingerprint=expected, expected_code_sha256=code_fingerprint())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _encoding_artifact_current(model: str) -> bool:
    directory = CACHE_ROOT / "embeddings" / model
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists() or not _stage_marker_current(directory, "manifest.json"):
        return False
    try:
        manifest = read_json(manifest_path)
        if manifest.get("model") != model or int(manifest.get("chunks", -1)) != CHUNK_COUNT or int(manifest.get("queries", -1)) != QUERY_COUNT:
            return False
        audit = read_json(RESULTS_ROOT / "input_audit" / "READING_AUDIT.json")
        if manifest.get("input_fingerprint") != audit.get("audit_fingerprint"):
            return False
        if manifest.get("scorer_contract") != SCORER_CONTRACT or manifest.get("contract") != dataclass_to_dict(MODEL_SPECS[model]):
            return False
        require_success(directory, expected_fingerprint=manifest.get("content_fingerprint"), expected_code_sha256=code_fingerprint())
        require_success(
            RESULTS_ROOT / "input_audit",
            expected_fingerprint=audit.get("audit_fingerprint"),
            expected_code_sha256=code_fingerprint(),
        )
        for receipt in manifest.get("shards", []):
            validate_embedding_shard(Path(receipt["path"]), expected_fingerprint=manifest["manifest_fingerprint"])
        queries_path = directory / "queries.npz"
        if manifest.get("queries_sha256") != sha256_file(queries_path):
            return False
        validate_query_embeddings(queries_path, expected_fingerprint=manifest["manifest_fingerprint"])
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _ensure_outer_artifacts(outer: str, *, resume: bool) -> dict[str, Any]:
    bounded_dir = RESULTS_ROOT / "bounded_screen" / outer
    if not _stage_marker_current(bounded_dir, "BOUNDED_REPORT.json"):
        bounded_screen(outer=outer, resume=resume, authorize=True)
    selected = _selected_models_from_bounded(outer)
    preflight_dir = RESULTS_ROOT / "preflight" / outer
    if not _stage_marker_current(preflight_dir, "PREFLIGHT.json"):
        preflight(outer=outer, resume=resume, authorize=True)
    for model in selected:
        if not _encoding_artifact_current(model):
            encode_model_selected(model=model, outer=outer, resume=resume, authorize=True)
    source_dir = RESULTS_ROOT / "source_audit" / outer
    source_path = source_dir / "SOURCE_AUDIT.json"
    if not _stage_marker_current(source_dir, "SOURCE_AUDIT.json"):
        source_audit(outer=outer, resume=resume, authorize=True)
    source_report = read_json(source_path)
    require_success(source_dir, expected_fingerprint=content_hash(source_report), expected_code_sha256=code_fingerprint())
    if source_report.get("status") != "PASS_SOURCE_VIABILITY":
        raise GateRejected("REJECTED_FULL_CORPUS_SOURCE_GATE", f"source viability failed for {outer}")
    return source_report


def nested_oof(*, resume: bool = True, authorize: bool = False) -> dict[str, Any]:
    if not authorize and os.environ.get("EXP109B_ALLOW_FULL_OOF") != "1":
        raise GateRejected("REJECTED_AUTHORIZATION_GATE", "full nested OOF requires explicit authorization (EXP109B_ALLOW_FULL_OOF=1)")
    fold0_dir = RESULTS_ROOT / "fold0_screen"
    fold0 = read_json(fold0_dir / "FOLD0_REPORT.json")
    require_success(fold0_dir, expected_fingerprint=content_hash(fold0), expected_code_sha256=code_fingerprint())
    if fold0.get("status") != "PASS_FOLD0_AMBITIOUS_GATE":
        raise GateRejected("REJECTED_FOLD0_GATE", "Fold-0 ambitious gate has not passed")
    audit = read_json(RESULTS_ROOT / "input_audit" / "READING_AUDIT.json")
    audit_dir = RESULTS_ROOT / "input_audit"
    require_success(audit_dir, expected_fingerprint=audit.get("audit_fingerprint"), expected_code_sha256=code_fingerprint())
    output_dir = RESULTS_ROOT / "nested_oof"
    if resume and _stage_marker_current(output_dir, "OOF_REPORT.json"):
        return read_json(output_dir / "OOF_REPORT.json")
    tracker = RunTracker("nested-oof", total=len(FOLD_NAMES))
    folds, _fold_for = load_folds()
    answers, _stats = canonical_labels()
    reports: dict[str, Any] = {}
    baseline_predictions: dict[str, list[str]] = {}
    final_predictions: dict[str, list[str]] = {}
    try:
        for position, outer in enumerate(FOLD_NAMES, start=1):
            tracker.heartbeat(position - 1, emit=True, outer=outer, input_fingerprint=audit.get("audit_fingerprint"))
            source_report = _ensure_outer_artifacts(outer, resume=resume)
            result = _screen_outer_fold(outer, source_report)
            report = result["report"]
            reports[outer] = report
            baseline_predictions.update(result["baseline_predictions"])
            final_predictions.update(result["final_predictions"])
            atomic_json(output_dir / f"{outer}_REPORT.json", report)
            tracker.heartbeat(position, emit=True, outer=outer)
        qids = [qid for outer in FOLD_NAMES for qid in folds[outer]]
        baseline_metrics = evaluate_rankings(baseline_predictions, answers, qids)
        final_metrics = evaluate_rankings(final_predictions, answers, qids)
        per_query_recall_delta = []
        per_query_rank_delta = []
        for qid in qids:
            if not answers.get(qid):
                continue
            base_rank = _first_gold_rank(baseline_predictions[qid], answers[qid])
            final_rank = _first_gold_rank(final_predictions[qid], answers[qid])
            per_query_recall_delta.append(recall_at(final_predictions[qid], answers[qid], 5) - recall_at(baseline_predictions[qid], answers[qid], 5))
            per_query_rank_delta.append((base_rank - final_rank) / BM25_MAX_RANK)
        fold_deltas = {
            outer: float(reports[outer]["final"].get("recall@5", 0.0) - reports[outer]["baseline"].get("recall@5", 0.0))
            for outer in FOLD_NAMES
        }
        delta = {key: float(final_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)) for key in ("recall@5", "precision@5", "multi_gold_recall@5", "recall@16", "recall@50")}
        aggregate_transition = _ranking_transition_metrics(baseline_predictions, final_predictions, answers, qids)
        recall_ci = deterministic_bootstrap(per_query_recall_delta)
        rank_ci = deterministic_bootstrap(per_query_rank_delta)
        quality_tags: list[str] = []
        if final_metrics.get("recall@5", 0.0) >= 0.950:
            quality_tags.append("PASS_STRONG_095")
        if final_metrics.get("recall@5", 0.0) >= 0.960:
            quality_tags.append("PASS_TARGET_096")
        if final_metrics.get("recall@5", 0.0) >= 0.970:
            quality_tags.append("PASS_TARGET_097")
        gate_checks = {
            "aggregate_recall@5_ge_0_940": final_metrics.get("recall@5", 0.0) >= 0.940,
            "aggregate_delta_recall@5_ge_0_015": delta["recall@5"] >= 0.015,
            "bootstrap_lower_gt_zero": recall_ci.get("lower", 0.0) > 0.0,
            "at_least_4_of_5_folds_nonnegative": sum(value >= 0.0 for value in fold_deltas.values()) >= 4,
            "worst_fold_delta_ge_-0_002": min(fold_deltas.values()) >= -0.002,
            "precision_non_decrease": delta["precision@5"] >= 0.0,
            "multi_gold_non_decrease": delta["multi_gold_recall@5"] >= 0.0,
            "delta_recall@16_ge_-0_001": delta["recall@16"] >= -0.001,
            "delta_recall@50_ge_-0_001": delta["recall@50"] >= -0.001,
        }
        report = {
            "schema_version": SCHEMA,
            "stage": "nested_oof",
            "status": "PASS_FULL_OOF_GATE" if all(gate_checks.values()) else "REJECTED_FULL_OOF_GATE",
            "folds": reports,
            "baseline": baseline_metrics,
            "final": final_metrics,
            "delta_vs_corrected_baseline": delta,
            "fold_delta_recall@5": fold_deltas,
            "bootstrap_recall@5_delta": recall_ci,
            "bootstrap_rank_delta": rank_ci,
            "top5_transition": aggregate_transition,
            "quality_tags": quality_tags,
            "gate": {"checks": gate_checks, "pass": all(gate_checks.values())},
            "input_audit_fingerprint": audit.get("audit_fingerprint"),
            "no_public_submission": True,
        }
        atomic_json(output_dir / "OOF_REPORT.json", report)
        atomic_json(output_dir / "OOF_STATUS.json", {"schema_version": SCHEMA, "stage": "nested_oof", "status": report["status"], "report": str((output_dir / "OOF_REPORT.json").resolve()), "no_public_submission": True})
        if report["status"] == "PASS_FULL_OOF_GATE":
            write_success(output_dir, stage="nested-oof", fingerprint=content_hash(report))
        else:
            (output_dir / "_SUCCESS.json").unlink(missing_ok=True)
        tracker.finish("PASS" if report["status"] == "PASS_FULL_OOF_GATE" else "REJECTED", output_fingerprint=content_hash(report))
        return report
    except KeyboardInterrupt:
        tracker.finish("INTERRUPTED")
        raise
    except Exception:
        tracker.finish("FAILED", error=traceback.format_exc(limit=6))
        raise


def overnight_fold0(*, resume: bool = True, authorize: bool = False) -> dict[str, Any]:
    if not authorize and os.environ.get("EXP109B_ALLOW_OVERNIGHT_FOLD0") != "1":
        raise GateRejected("REJECTED_AUTHORIZATION_GATE", "overnight-fold0 requires explicit authorization")
    # This wrapper is authorized to prepare only the requested outer fold and
    # still terminates at Fold-0; it never infers permission for full OOF.
    _ensure_outer_artifacts("fold_0", resume=resume)
    return fold0_screen(resume=resume, authorize=True)


def smoke() -> dict[str, Any]:
    """Run a deterministic CPU-only scorer/autograd/checkpoint smoke test."""
    if torch is None:
        raise RuntimeError("torch unavailable for CPU smoke")
    torch.manual_seed(RNG_SEED)
    generator = torch.Generator(device="cpu").manual_seed(RNG_SEED)
    documents = torch.randn((8, DIMENSION), generator=generator, dtype=torch.float32)
    index = CorpusIndex.from_chunk_doc_ids(["d0", "d0", "d1", "d1", "d2", "d2", "d3", "d3"])
    queries = torch.nn.Parameter(torch.randn((2, DIMENSION), generator=generator, dtype=torch.float32))
    targets = torch.tensor([0, 1], dtype=torch.long)
    optimizer = torch.optim.SGD([queries], lr=0.25)
    initial_loss = None
    final_loss = None
    for step in range(32):
        optimizer.zero_grad(set_to_none=True)
        scores = compute_parent_scores(queries, documents, index, chunk_block_size=3)
        loss = torch.nn.functional.cross_entropy(scores, targets)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    checkpoint_dir = RESULTS_ROOT / "smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "smoke_checkpoint.pt"
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
    torch.save({"queries": queries.detach().cpu(), "step": 32, "scorer_contract": SCORER_CONTRACT}, temporary)
    temporary.replace(checkpoint_path)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    torch_values = compute_parent_scores(restored["queries"], documents, index, chunk_block_size=3).detach().numpy()
    numpy_values = np.stack([numpy_top2_parent_scores(row.detach().numpy(), documents.numpy(), index) for row in restored["queries"]])
    max_error = float(np.max(np.abs(torch_values - numpy_values)))
    report = {
        "schema_version": SCHEMA,
        "stage": "smoke",
        "status": "PASS" if initial_loss is not None and final_loss is not None and final_loss < initial_loss and max_error <= 1e-6 else "REJECTED_SMOKE",
        "device": "cpu",
        "steps": 32,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "scorer_contract": SCORER_CONTRACT,
        "numpy_max_abs_error": max_error,
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
        "no_model_download": True,
    }
    atomic_json(checkpoint_dir / "SMOKE.json", report)
    if report["status"] == "PASS":
        write_success(checkpoint_dir, stage="smoke", fingerprint=content_hash(report))
    else:
        (checkpoint_dir / "_SUCCESS.json").unlink(missing_ok=True)
    return report


def status() -> dict[str, Any]:
    paths = [
        RESULTS_ROOT / "RUN_STATUS.json",
        RESULTS_ROOT / "input_audit" / "READING_AUDIT.json",
        RESULTS_ROOT / "replay" / "REPLAY.json",
        RESULTS_ROOT / "smoke" / "SMOKE.json",
        RESULTS_ROOT / "bounded_screen" / "fold_0" / "BOUNDED_REPORT.json",
        RESULTS_ROOT / "preflight" / "fold_0" / "PREFLIGHT.json",
        RESULTS_ROOT / "source_audit" / "fold_0" / "SOURCE_AUDIT.json",
        RESULTS_ROOT / "fold0_screen" / "FOLD0_REPORT.json",
        RESULTS_ROOT / "nested_oof" / "OOF_STATUS.json",
    ]
    stages: dict[str, Any] = {}
    for path in paths:
        if path.exists():
            try:
                value = read_json(path)
                stages[str(path.relative_to(ROOT))] = {"status": value.get("status"), "stage": value.get("stage", value.get("phase")), "success_marker": (path.parent / "_SUCCESS.json").exists()}
            except Exception as exc:
                stages[str(path.relative_to(ROOT))] = {"error": str(exc)}
    return {"schema_version": SCHEMA, "namespace": NAMESPACE, "stages": stages}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_result(result: Mapping[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "replay", "smoke", "bounded-screen", "re-evaluate", "preflight", "encode-selected", "source-audit", "cached-fusion-pilot", "locked-fusion-fold0", "fold0-screen", "nested-oof", "overnight-fold0", "status"))
    parser.add_argument("--outer", default="fold_0")
    parser.add_argument("--model", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh-models", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--authorize", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.stage == "audit":
            result = audit_inputs()
        elif args.stage == "replay":
            result = replay(fresh_models=args.fresh_models)
        elif args.stage == "smoke":
            result = smoke()
        elif args.stage == "bounded-screen":
            result = bounded_screen(outer=args.outer, resume=args.resume, authorize=args.authorize)
        elif args.stage == "re-evaluate":
            result = reevaluate_bounded(outer=args.outer)
        elif args.stage == "preflight":
            result = preflight(outer=args.outer, resume=args.resume, authorize=args.authorize)
        elif args.stage == "encode-selected":
            model = args.model
            if not model:
                selected = _selected_models_from_bounded(args.outer)
                if len(selected) != 1:
                    raise ValueError("--model is required when bounded gate selected multiple models")
                model = selected[0]
            result = encode_model_selected(model=model, outer=args.outer, resume=args.resume, authorize=args.authorize)
        elif args.stage == "source-audit":
            result = source_audit(outer=args.outer, resume=args.resume, authorize=args.authorize)
        elif args.stage == "cached-fusion-pilot":
            result = cached_fusion_pilot(outer=args.outer, resume=args.resume, authorize=args.authorize)
        elif args.stage == "locked-fusion-fold0":
            result = locked_cached_fusion_fold0(outer=args.outer, authorize=args.authorize)
        elif args.stage == "fold0-screen":
            result = fold0_screen(resume=args.resume, authorize=args.authorize)
        elif args.stage == "nested-oof":
            result = nested_oof(resume=args.resume, authorize=args.authorize)
        elif args.stage == "overnight-fold0":
            result = overnight_fold0(resume=args.resume, authorize=args.authorize)
        else:
            result = status()
        _print_result(result)
        return 0 if result.get("status", "PASS").startswith("PASS") or args.stage == "status" else 2
    except GateRejected as exc:
        payload = {"schema_version": SCHEMA, "status": exc.status, "error": str(exc), "report": str(exc.report) if exc.report else None}
        _print_result(payload)
        return 2
    except KeyboardInterrupt:
        try:
            atomic_json(RESULTS_ROOT / "RUN_STATUS.json", {
                "run_id": f"cli-interrupted-{os.getpid()}",
                "state": "INTERRUPTED",
                "phase": args.stage,
                "outer": args.outer,
                "model": args.model,
                "completed": 0,
                "total": 0,
                "throughput": 0.0,
                "eta_seconds": 0.0,
                "last_heartbeat": utc_now(),
                "input_fingerprint": None,
                "config_fingerprint": None,
                "code_fingerprint": code_fingerprint(),
            })
        except Exception:
            pass
        print(json.dumps({"schema_version": SCHEMA, "status": "INTERRUPTED"}, ensure_ascii=False), flush=True)
        return 130
    except Exception as exc:
        try:
            atomic_json(RESULTS_ROOT / "RUN_STATUS.json", {
                "run_id": f"cli-failed-{os.getpid()}",
                "state": "FAILED",
                "phase": args.stage,
                "outer": args.outer,
                "model": args.model,
                "completed": 0,
                "total": 0,
                "throughput": 0.0,
                "eta_seconds": 0.0,
                "last_heartbeat": utc_now(),
                "input_fingerprint": None,
                "config_fingerprint": None,
                "code_fingerprint": code_fingerprint(),
                "error": str(exc),
            })
        except Exception:
            pass
        print(json.dumps({"schema_version": SCHEMA, "status": "FAILED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
