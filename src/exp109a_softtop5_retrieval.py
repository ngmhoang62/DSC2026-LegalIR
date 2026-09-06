"""EXP-109A: exact full-corpus soft-top-5 query residual retrieval.

This module is deliberately self-contained and owns only the exp109a
namespace. It consumes the frozen VietLegal-E5 and BM25 artifacts, performs
parent scoring over every chunk, and exposes small, auditable stages:

audit -> test-loss -> replay-exp102 -> preflight -> smoke
-> nested-screen/nested-oof -> report.

The two nested stages are intentionally fail-closed. They require successful
input, math, replay, preflight, and smoke markers before any training is
allowed to start.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


CODE_FILE = Path(__file__).resolve()
ROOT = CODE_FILE.parents[1]
SCHEMA = "legalir.exp109a_softtop5_retrieval.v1"
LABEL_POLICY = "canonical_duplicate_alias_drop_empty_passage_v1"
DIMENSION = 1024
RANK = 32
SEED = 2026
TAU = 0.05
SOFT_K = 5
SOFT_EPS = 1e-6
SCORER_CONTRACT = "fp16_to_fp32_l2_cosine_top2mean_v1"
CURVE_KS = (1, 3, 5, 10, 16, 20, 32, 50, 64, 100, 150)
ARMS = ("A_decoupled", "B_softtop5", "C_hybrid")
ALPHAS = (1.0, 2.0, 5.0, 10.0)
HYBRID_ALPHAS = (1.0, 2.0, 5.0)
LAMBDAS = (0.10, 0.25, 0.50, 1.0)
RRF_WEIGHTS = (0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9)
RRF_KS = (10, 20, 32, 60, 100)
MAX_BM25_DOCUMENTS = 256
PARENT_CHUNK_BLOCK = 8192
CHECKPOINT_QUERY_INTERVAL = 500
PILOT_CONFIGS = (
    {"arm": "A_decoupled", "alpha": None, "lambda_top5": 0.0},
    {"arm": "B_softtop5", "alpha": 2.0, "lambda_top5": 0.0},
    {"arm": "B_softtop5", "alpha": 5.0, "lambda_top5": 0.0},
    {"arm": "C_hybrid", "alpha": 2.0, "lambda_top5": 0.25},
    {"arm": "C_hybrid", "alpha": 5.0, "lambda_top5": 0.50},
)

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
BM25_EVIDENCE_DIR = ROOT / "cache" / "exp021_sparse" / "depth_tune" / "raw4096_evidence"
BM25_EVIDENCE_MANIFEST = BM25_EVIDENCE_DIR / "manifest.json"
BM25_TUNING_REPORT = ROOT / "results" / "exp021_sparse" / "depth_rrf_tuning" / "tuning_report.json"
BM25_DB = ROOT / "cache" / "exp021_sparse" / "passage_hierarchy" / "fts5" / "bm25_v3.sqlite"
BM25_MANIFEST = ROOT / "cache" / "exp021_sparse" / "passage_hierarchy" / "fts5" / "manifest.json"
EXP102_CACHE = ROOT / "cache" / "exp102_mil_nce_retrieval"
EXP102_REPORT = ROOT / "results" / "exp102_mil_nce_retrieval" / "REPORT_full_oof.json"
EXP022_CANDIDATES = ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl"

NAMESPACE_CACHE = ROOT / "cache" / "exp109a_softtop5_retrieval"
RESULTS_DIR = ROOT / "results" / "exp109a_softtop5_retrieval"
LOG_DIR = RESULTS_DIR / "logs"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_numpy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npy")
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def write_success(directory: Path, *, stage: str, fingerprint: str, extra: Mapping[str, Any] | None = None) -> Path:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "stage": stage,
        "status": "PASS",
        "fingerprint": fingerprint,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(dict(extra))
    path = directory / "_SUCCESS.json"
    atomic_json(path, payload)
    return path


def require_success(directory: Path, expected_fingerprint: str | None = None) -> dict[str, Any]:
    marker = directory / "_SUCCESS.json"
    if not marker.exists():
        raise RuntimeError(f"missing success marker: {marker}")
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError(f"stage marker is not PASS: {marker}")
    if expected_fingerprint is not None and value.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(f"fingerprint mismatch for {marker}")
    return value


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class RunLog:
    """Small append-only logger whose status file is safe to inspect mid-run."""

    def __init__(self, stage: str, run_id: str | None = None) -> None:
        self.stage = stage
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
        self.directory = LOG_DIR / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_path = self.directory / f"{stage}.log"
        self.run_log_path = self.directory / "run.log"
        self.status_path = self.directory / "RUN_STATUS.json"
        self.status: dict[str, Any] = {
            "schema_version": SCHEMA,
            "stage": stage,
            "run_id": self.run_id,
            "state": "STARTING",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._write_status()

    def _write_status(self) -> None:
        atomic_json(self.status_path, self.status)
        atomic_json(RESULTS_DIR / "RUN_STATUS.json", self.status)

    def log(self, message: str) -> None:
        for path in (self.log_path, self.run_log_path):
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(message.rstrip() + "\n")

    def update(self, **fields: Any) -> None:
        self.status.update(fields)
        self._write_status()
        compact = " ".join(f"{key}={value}" for key, value in fields.items())
        self.log(compact)
        print(f"[{self.stage}] {compact}", flush=True)

    def finish(self, state: str = "PASS", **fields: Any) -> None:
        self.status.update(fields)
        self.status["state"] = state
        self.status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._write_status()
        self.log(f"state={state}")


def file_record(path: Path, *, schema: str | None = None, hash_file: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"path": relative_path(path), "exists": path.exists()}
    if path.exists():
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path) if hash_file else None
        if schema:
            record["schema_version"] = schema
    return record


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL {path}:{line_number}") from exc


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path, expected_schema: str | None = None) -> dict[str, Any]:
    value = load_json(path)
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise RuntimeError(f"schema mismatch in {path}: {value.get('schema_version')}")
    return value


class ResidualProjection(nn.Module):
    """1024-dimensional rank-32 residual projection with identity initialization."""

    def __init__(self, dimension: int = DIMENSION, rank: int = RANK) -> None:
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        delta = self.up(torch.nn.functional.gelu(self.down(query)))
        return torch.nn.functional.normalize(query + delta, p=2, dim=-1)


class _ImplicitSoftTopK(torch.autograd.Function):
    """SoftTop-k with an analytic implicit threshold gradient."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float, k: int) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError("SoftTop-k custom function expects a one-dimensional vector")
        alpha_value = float(alpha)
        k_value = float(k)
        if alpha_value <= 0:
            raise ValueError("alpha must be positive")
        if not 0 < k_value < float(x.numel()):
            raise ValueError("k must be strictly between zero and the vector length")
        detached = x.detach()
        span = max(20.0 / alpha_value, 20.0)
        lower = -float(detached.max().item()) - span
        upper = -float(detached.min().item()) + span
        with torch.no_grad():
            for _ in range(96):
                middle = (lower + upper) / 2.0
                total = torch.sigmoid(alpha_value * (detached + middle)).sum().item()
                if total > k_value:
                    upper = middle
                else:
                    lower = middle
            threshold = (lower + upper) / 2.0
        output = torch.sigmoid(alpha_value * (x + threshold))
        derivative = alpha_value * output * (1.0 - output)
        ctx.save_for_backward(derivative)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (derivative,) = ctx.saved_tensors
        denominator = derivative.sum().clamp_min(torch.finfo(derivative.dtype).tiny)
        correction = (grad_output * derivative).sum() / denominator
        grad_input = grad_output * derivative - correction * derivative
        return grad_input, None, None


def soft_top_k(scores: torch.Tensor, *, k: int = SOFT_K, alpha: float = 1.0) -> torch.Tensor:
    """Return differentiable soft membership for a full score vector."""
    if scores.ndim == 1:
        return _ImplicitSoftTopK.apply(scores, float(alpha), int(k))
    if scores.ndim != 2:
        raise ValueError("scores must have shape [documents] or [batch, documents]")
    return torch.stack([_ImplicitSoftTopK.apply(row, float(alpha), int(k)) for row in scores], dim=0)


def standardize_scores(scores: torch.Tensor, eps: float = SOFT_EPS) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("standardize_scores expects a full one-dimensional parent vector")
    mean = scores.mean()
    deviation = scores.std(unbiased=False)
    return (scores - mean) / (deviation + eps)


def soft_top5_membership(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    return soft_top_k(standardize_scores(scores), k=SOFT_K, alpha=alpha)


def decoupled_terms(scores: torch.Tensor, gold_mask: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    """Per-positive decoupled loss terms.

    Every positive is scored against all non-golds. Other positive labels are
    intentionally excluded from the denominator.
    """
    if scores.ndim != 1 or gold_mask.ndim != 1 or scores.numel() != gold_mask.numel():
        raise ValueError("scores and gold_mask must be one-dimensional with equal length")
    if tau <= 0:
        raise ValueError("tau must be positive")
    mask = gold_mask.to(device=scores.device, dtype=torch.bool)
    positives = torch.nonzero(mask, as_tuple=False).flatten()
    negatives = torch.nonzero(~mask, as_tuple=False).flatten()
    if positives.numel() == 0:
        raise ValueError("a query must have at least one canonical positive")
    if negatives.numel() == 0:
        raise ValueError("decoupled loss needs at least one non-gold parent")
    denominator = torch.logsumexp(scores.index_select(0, negatives) / tau, dim=0)
    return denominator - scores.index_select(0, positives) / tau


def decoupled_loss(scores: torch.Tensor, gold_mask: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    return decoupled_terms(scores, gold_mask, tau=tau).mean()


def soft_top5_loss(scores: torch.Tensor, gold_indices: Sequence[int], alpha: float) -> torch.Tensor:
    memberships = soft_top5_membership(scores, alpha)
    if not gold_indices:
        raise ValueError("a query must have at least one canonical positive")
    indices = torch.as_tensor(list(gold_indices), device=scores.device, dtype=torch.long)
    return -torch.log(memberships.index_select(0, indices).clamp_min(1e-8)).mean()


def hybrid_loss(
    scores: torch.Tensor,
    gold_mask: torch.Tensor,
    gold_indices: Sequence[int],
    *,
    alpha: float,
    lambda_top5: float,
    tau: float = TAU,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decoupled = decoupled_loss(scores, gold_mask, tau=tau)
    top5 = soft_top5_loss(scores, gold_indices, alpha=alpha)
    return decoupled + float(lambda_top5) * top5, decoupled, top5


def _stable_parent_order(scores: np.ndarray, doc_ids: Sequence[str]) -> np.ndarray:
    ids = np.asarray([str(value) for value in doc_ids], dtype=object)
    return np.lexsort((ids, -np.asarray(scores, dtype=np.float64)))


@dataclass
class CorpusIndex:
    """Parent-major chunk index; offsets are inclusive/exclusive."""

    doc_ids: tuple[str, ...]
    chunk_doc_ids: tuple[str, ...]
    offsets: np.ndarray
    chunk_indices: np.ndarray | None = None

    @classmethod
    def from_chunk_doc_ids(cls, chunk_doc_ids: Sequence[str]) -> "CorpusIndex":
        ids = [str(value) for value in chunk_doc_ids]
        positions: dict[str, list[int]] = defaultdict(list)
        for index, doc_id in enumerate(ids):
            positions[doc_id].append(index)
        doc_ids = tuple(positions)
        ranges = []
        contiguous = True
        for doc_id in doc_ids:
            values = positions[doc_id]
            ranges.append((values[0], values[-1] + 1))
            if values != list(range(values[0], values[-1] + 1)):
                contiguous = False
        offsets = np.asarray(ranges, dtype=np.int64)
        flat_indices = None
        if not contiguous:
            flat_indices = np.asarray([index for doc_id in doc_ids for index in positions[doc_id]], dtype=np.int64)
            offsets = np.asarray(
                [
                    (
                        sum(len(positions[value]) for value in doc_ids[:i]),
                        sum(len(positions[value]) for value in doc_ids[: i + 1]),
                    )
                    for i in range(len(doc_ids))
                ],
                dtype=np.int64,
            )
        return cls(tuple(doc_ids), tuple(ids), offsets, flat_indices)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA,
            "doc_ids": list(self.doc_ids),
            "offsets": self.offsets.tolist(),
            "chunk_count": len(self.chunk_doc_ids),
            "document_count": len(self.doc_ids),
            "noncontiguous": self.chunk_indices is not None,
            "chunk_indices": self.chunk_indices.tolist() if self.chunk_indices is not None else None,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any], chunk_doc_ids: Sequence[str]) -> "CorpusIndex":
        indices = value.get("chunk_indices")
        return cls(
            tuple(map(str, value["doc_ids"])),
            tuple(map(str, chunk_doc_ids)),
            np.asarray(value["offsets"], dtype=np.int64),
            np.asarray(indices, dtype=np.int64) if indices is not None else None,
        )


def iter_parent_blocks(index: CorpusIndex, max_chunks: int = PARENT_CHUNK_BLOCK) -> Iterator[tuple[int, int]]:
    start = 0
    while start < len(index.doc_ids):
        end = start + 1
        chunk_total = 0
        while end <= len(index.doc_ids):
            begin_chunk, end_chunk = map(int, index.offsets[end - 1])
            proposed = chunk_total + end_chunk - begin_chunk
            if end > start + 1 and proposed > max_chunks:
                break
            chunk_total = proposed
            end += 1
        yield start, end - 1
        start = end - 1


def _document_block(
    documents: torch.Tensor | np.ndarray,
    index: CorpusIndex,
    parent_start: int,
    parent_end: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [chunks, dim] documents and [parents, max_chunks] local positions."""
    offsets = index.offsets[parent_start:parent_end]
    lengths = offsets[:, 1] - offsets[:, 0]
    max_length = int(lengths.max())
    positions = np.zeros((len(lengths), max_length), dtype=np.int64)
    if index.chunk_indices is None:
        chunk_start = int(offsets[0, 0])
        chunk_end = int(offsets[-1, 1])
        for row, (begin, end) in enumerate(offsets):
            positions[row, : int(end - begin)] = np.arange(
                int(begin - chunk_start), int(end - chunk_start), dtype=np.int64
            )
        if isinstance(documents, np.ndarray):
            block = torch.from_numpy(np.asarray(documents[chunk_start:chunk_end], dtype=np.float32))
        else:
            block = documents[chunk_start:chunk_end]
    else:
        flat_start = int(offsets[0, 0])
        flat_end = int(offsets[-1, 1])
        selected = index.chunk_indices[flat_start:flat_end]
        cursor = 0
        for row, length in enumerate(lengths):
            next_cursor = cursor + int(length)
            positions[row, : int(length)] = np.arange(cursor, next_cursor, dtype=np.int64)
            cursor = next_cursor
        if isinstance(documents, np.ndarray):
            block = torch.from_numpy(np.asarray(documents[selected], dtype=np.float32))
        else:
            block = documents.index_select(0, torch.as_tensor(selected, dtype=torch.long, device=documents.device))
    # The frozen corpus is stored as FP16.  Casting it back to FP32 does not
    # restore the unit norm it had before quantisation, so dot product here
    # would not be cosine similarity.  Keep this at the single corpus-read
    # boundary so training, evaluation, replay parity, and all scorer blocks
    # share the exact same contract.
    block = torch.nn.functional.normalize(block.to(device=device, dtype=dtype), p=2, dim=-1)
    return block, torch.as_tensor(positions, dtype=torch.long, device=device)


def compute_parent_scores(
    query_batch: torch.Tensor,
    documents: torch.Tensor | np.ndarray,
    index: CorpusIndex,
    *,
    chunk_block_size: int = PARENT_CHUNK_BLOCK,
) -> torch.Tensor:
    """Score every parent using the mean of its two best source-exact chunks.

    Chunk ties are resolved by the original chunk index because the local
    parent position is stable. A one-chunk parent uses that sole score.
    """
    if query_batch.ndim == 1:
        query_batch = query_batch.unsqueeze(0)
    if query_batch.ndim != 2:
        raise ValueError("query_batch must have shape [batch, dimension]")
    if len(index.doc_ids) == 0:
        raise ValueError("corpus has no parents")
    output: list[torch.Tensor] = []
    for parent_start, parent_end in iter_parent_blocks(index, max_chunks=chunk_block_size):
        block, positions = _document_block(
            documents,
            index,
            parent_start,
            parent_end,
            query_batch.device,
            query_batch.dtype,
        )
        chunk_scores = torch.matmul(query_batch, block.transpose(0, 1))
        parent_count, max_length = positions.shape
        expanded_positions = positions.reshape(-1)
        gathered = chunk_scores.index_select(1, expanded_positions).reshape(
            query_batch.shape[0], parent_count, max_length
        )
        lengths = torch.as_tensor(
            index.offsets[parent_start:parent_end, 1] - index.offsets[parent_start:parent_end, 0],
            dtype=torch.long,
            device=query_batch.device,
        )
        valid = torch.arange(max_length, device=query_batch.device).view(1, 1, -1) < lengths.view(1, -1, 1)
        masked = gathered.masked_fill(~valid, -torch.inf)
        order = torch.argsort(masked, dim=-1, descending=True, stable=True)
        first = torch.gather(masked, -1, order[..., :1]).squeeze(-1)
        if max_length == 1:
            aggregate = first
        else:
            second = torch.gather(masked, -1, order[..., 1:2]).squeeze(-1)
            aggregate = torch.where(lengths.view(1, -1) > 1, (first + second) / 2.0, first)
        output.append(aggregate)
    return torch.cat(output, dim=1)


def reference_parent_scores(
    query: torch.Tensor,
    documents: torch.Tensor,
    index: CorpusIndex,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for parent_number, _doc_id in enumerate(index.doc_ids):
        begin, end = map(int, index.offsets[parent_number])
        if index.chunk_indices is None:
            chunk_ids = torch.arange(begin, end, device=documents.device)
        else:
            chunk_ids = torch.as_tensor(index.chunk_indices[begin:end], dtype=torch.long, device=documents.device)
        chunks = documents.index_select(0, chunk_ids).to(dtype=query.dtype)
        chunks = torch.nn.functional.normalize(chunks, p=2, dim=-1)
        scores = chunks.matmul(query)
        ordered = torch.argsort(scores, descending=True, stable=True)
        values.append(scores.index_select(0, ordered[: min(2, len(scores))]).mean())
    return torch.stack(values)


def numpy_top2_parent_scores(
    query: np.ndarray,
    documents: np.ndarray,
    index: CorpusIndex,
    parent_numbers: Sequence[int],
) -> np.ndarray:
    """Independent CPU/NumPy cosine + source-exact top2_mean reference."""
    q = np.asarray(query, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    values: list[float] = []
    for parent_number in parent_numbers:
        begin, end = map(int, index.offsets[int(parent_number)])
        selected = (
            np.arange(begin, end, dtype=np.int64)
            if index.chunk_indices is None
            else np.asarray(index.chunk_indices[begin:end], dtype=np.int64)
        )
        chunks = np.asarray(documents[selected], dtype=np.float32)
        chunks = chunks / np.maximum(np.linalg.norm(chunks, axis=1, keepdims=True), 1e-12)
        scores = np.sort(chunks @ q)[::-1]
        values.append(float(scores[: min(2, len(scores))].mean()))
    return np.asarray(values, dtype=np.float32)


def _real_fixture_selection(inputs: "FrozenInputs") -> tuple[list[str], list[int]]:
    """Stable, declared real-data sample used by the independent Gate-2 check."""
    qids = sorted(inputs.qids, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())[:8]
    parents = sorted(
        range(len(inputs.corpus.doc_ids)),
        key=lambda number: hashlib.sha256(inputs.corpus.doc_ids[number].encode("utf-8")).hexdigest(),
    )[:64]
    return qids, parents


def real_parent_numpy_fixture(inputs: "FrozenInputs", *, device: torch.device) -> dict[str, Any]:
    """Create/verify a frozen real-parent NumPy parity and ranking fixture.

    The first successful run writes the canonical JSON fixture.  Later runs
    recompute it and reject any change rather than silently refreshing it.
    """
    fixture_dir = NAMESPACE_CACHE / "reproduction"
    fixture_path = fixture_dir / "REAL_PARENT_NUMPY_FIXTURE.json"
    qids, parent_numbers = _real_fixture_selection(inputs)
    qid_to_index = _query_index(inputs)
    sample_doc_ids = [inputs.corpus.doc_ids[number] for number in parent_numbers]
    rows: list[dict[str, Any]] = []
    max_abs_error = 0.0
    identity = ResidualProjection().to(device).eval()
    with torch.no_grad():
        for qid in qids:
            raw_query = np.asarray(inputs.query_embeddings[qid_to_index[qid]], dtype=np.float32)
            torch_scores = compute_parent_scores(
                identity(torch.from_numpy(raw_query).to(device).unsqueeze(0)),
                inputs.chunk_embeddings,
                inputs.corpus,
            )[0].detach().cpu().numpy()
            numpy_scores = numpy_top2_parent_scores(raw_query, inputs.chunk_embeddings, inputs.corpus, parent_numbers)
            selected_scores = torch_scores[np.asarray(parent_numbers, dtype=np.int64)]
            max_abs_error = max(max_abs_error, float(np.max(np.abs(selected_scores - numpy_scores))))
            order = _stable_parent_order(torch_scores, inputs.corpus.doc_ids)[:150]
            rows.append({
                "qid": qid,
                "sample_parent_scores": [float(value) for value in numpy_scores],
                "top150_parent_ids": [inputs.corpus.doc_ids[number] for number in order],
            })
    payload = {
        "schema_version": SCHEMA,
        "reference": "independent_numpy_fp16_to_fp32_l2_cosine_top2_mean_v1",
        "query_ids": qids,
        "sample_parent_ids": sample_doc_ids,
        "rows": rows,
        "input_fingerprint": content_hash({"query_ids": inputs.query_ids, "corpus": inputs.corpus.to_json()}),
    }
    computed_hash = content_hash(payload)
    if fixture_path.exists():
        frozen = load_json(fixture_path)
        if content_hash(frozen) != computed_hash:
            raise RuntimeError("frozen real-data NumPy ranking fixture hash mismatch")
    else:
        atomic_json(fixture_path, payload)
    return {
        "status": "PASS" if max_abs_error <= 1e-6 else "FAIL",
        "fixture": relative_path(fixture_path),
        "fixture_hash": computed_hash,
        "queries": len(qids),
        "sampled_real_parents": len(parent_numbers),
        "max_abs_error": max_abs_error,
        "tolerance": 1e-6,
    }


def load_chunk_doc_ids(path: Path = E5_DIR / "chunk_ids.jsonl") -> list[str]:
    return [str(row["doc_id"]) for row in read_jsonl(path)]


def build_parent_index(*, write: bool = True) -> CorpusIndex:
    chunk_doc_ids = load_chunk_doc_ids()
    index = CorpusIndex.from_chunk_doc_ids(chunk_doc_ids)
    if write:
        fingerprint = content_hash(index.to_json())
        output_dir = NAMESPACE_CACHE / "parent_index"
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / "parent_index.json", index.to_json())
        write_success(output_dir, stage="parent-index", fingerprint=fingerprint)
    return index


def canonical_labels() -> tuple[dict[str, set[str]], dict[str, Any]]:
    original = load_json(TRAIN_PATH)
    exclusions = load_json(EXCLUSIONS_PATH)
    by_doc = {str(row["doc_id"]): row for row in exclusions}
    if len(by_doc) != len(exclusions):
        raise ValueError("duplicate document id in preprocessing exclusions")
    impact_rows = list(read_jsonl(LABEL_IMPACT_PATH))
    impacted_by_qid = {str(row["query_id"]): row for row in impact_rows}
    if len(impacted_by_qid) != len(impact_rows):
        raise ValueError("duplicate query id in label impact sidecar")
    answers: dict[str, set[str]] = {}
    observed: dict[str, set[str]] = {}
    duplicate_occurrences = 0
    empty_occurrences = 0
    for raw_qid, row in original.items():
        qid = str(raw_qid)
        canonical: set[str] = set()
        removed: set[str] = set()
        for raw_doc_id in row.get("answer", []):
            doc_id = str(raw_doc_id)
            exclusion = by_doc.get(doc_id)
            if exclusion is None:
                canonical.add(doc_id)
                continue
            removed.add(doc_id)
            reasons = {str(reason) for reason in exclusion.get("reasons", [])}
            replacement = exclusion.get("duplicate_retained_id")
            if "exact_duplicate_raw_passage" in reasons:
                if not replacement:
                    raise ValueError(f"duplicate gold has no retained alias: {qid}/{doc_id}")
                replacement = str(replacement)
                if replacement in by_doc:
                    raise ValueError(f"duplicate gold aliases another excluded document: {qid}/{doc_id}")
                canonical.add(replacement)
                duplicate_occurrences += 1
            elif reasons == {"empty_passage"}:
                empty_occurrences += 1
            else:
                raise ValueError(f"unsupported exclusion policy: {qid}/{doc_id}/{sorted(reasons)}")
        answers[qid] = canonical
        if removed:
            observed[qid] = removed
    declared = {
        qid: {str(value) for value in row.get("intentionally_excluded_gold_ids", [])}
        for qid, row in impacted_by_qid.items()
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


def load_folds() -> tuple[dict[str, list[str]], dict[str, str]]:
    raw = load_json(FOLDS_PATH)
    folds = {str(name): [str(qid) for qid in values] for name, values in raw.items()}
    fold_for = {qid: name for name, qids in folds.items() for qid in qids}
    all_qids = {str(qid) for qid in load_json(TRAIN_PATH)}
    if set(fold_for) != all_qids or sum(len(values) for values in folds.values()) != len(all_qids):
        raise ValueError("cv_folds.json is not a disjoint partition of train.json")
    if len(fold_for) != sum(len(values) for values in folds.values()):
        raise ValueError("duplicate query id in cv_folds.json")
    return folds, fold_for


def _check_expected_hash(path: Path, expected: str | None, errors: list[str]) -> None:
    if expected is None:
        return
    actual = sha256_file(path)
    if actual != expected:
        errors.append(f"sha256 mismatch: {relative_path(path)} expected={expected} actual={actual}")


def _check_frozen_marker(
    directory: Path,
    *,
    expected: str | None,
    expected_key: str | None,
    errors: list[str],
) -> None:
    marker_path = directory / "_SUCCESS.json"
    if not marker_path.exists():
        errors.append(f"missing success marker: {marker_path}")
        return
    marker = load_json(marker_path)
    if expected_key and expected and marker.get(expected_key) != expected:
        errors.append(f"success marker fingerprint mismatch: {relative_path(directory)}")


def audit_inputs() -> dict[str, Any]:
    """Read and validate every frozen input contract without model outputs."""
    errors: list[str] = []
    warnings: list[str] = []
    answers, label_stats = canonical_labels()
    train = load_json(TRAIN_PATH)
    folds, fold_for = load_folds()

    processed_manifest = load_manifest(PROCESSED_MANIFEST)
    structural_manifest = load_manifest(STRUCT_MANIFEST, "legalir.structural_chunks.v3")
    e5_manifest = load_manifest(E5_MANIFEST, "legalir.e5_embedding_cache.v1")
    query_manifest = load_manifest(QUERY_MANIFEST, "legalir.exp021_e5_dense_candidates.v1")
    evidence_manifest = load_manifest(BM25_EVIDENCE_MANIFEST, "legalir.exp021_sparse_depth_tune.v1")
    tuning_report = load_json(BM25_TUNING_REPORT)
    bm25_manifest = load_manifest(BM25_MANIFEST, "legalir.exp012b_v3.v1")

    _check_frozen_marker(
        STRUCT_DIR,
        expected=structural_manifest.get("content_fingerprint"),
        expected_key="content_fingerprint",
        errors=errors,
    )
    _check_frozen_marker(
        E5_DIR,
        expected=e5_manifest.get("cache_fingerprint"),
        expected_key="cache_fingerprint",
        errors=errors,
    )
    _check_frozen_marker(
        QUERY_DIR.parent,
        expected=None,
        expected_key=None,
        errors=errors,
    )
    _check_frozen_marker(
        BM25_EVIDENCE_DIR,
        expected=None,
        expected_key=None,
        errors=errors,
    )

    document_ids: list[str] = []
    document_set: set[str] = set()
    duplicate_documents = 0
    for row in read_jsonl(STRUCT_DIR / "documents.jsonl"):
        document_id = str(row["doc_id"])
        duplicate_documents += document_id in document_set
        document_set.add(document_id)
        document_ids.append(document_id)

    chunk_doc_ids: list[str] = []
    chunk_ids: set[str] = set()
    duplicate_chunks = 0
    orphan_chunks = 0
    noncontiguous: set[str] = set()
    closed: set[str] = set()
    previous_doc: str | None = None
    for row in read_jsonl(STRUCT_DIR / "chunks.jsonl"):
        chunk_id = str(row["chunk_id"])
        doc_id = str(row["doc_id"])
        duplicate_chunks += chunk_id in chunk_ids
        chunk_ids.add(chunk_id)
        chunk_doc_ids.append(doc_id)
        orphan_chunks += doc_id not in document_set
        if doc_id != previous_doc:
            if doc_id in closed:
                noncontiguous.add(doc_id)
            if previous_doc is not None:
                closed.add(previous_doc)
            previous_doc = doc_id

    e5_chunk_doc_ids: list[str] = []
    e5_chunk_id_count = 0
    for row in read_jsonl(E5_DIR / "chunk_ids.jsonl"):
        e5_chunk_id_count += 1
        e5_chunk_doc_ids.append(str(row["doc_id"]))
    doc_map = load_json(STRUCT_DIR / "doc_to_chunk_ids.json")
    map_doc_ids = {str(doc_id) for doc_id in doc_map}
    map_chunk_ids = {str(chunk_id) for values in doc_map.values() for chunk_id in values}

    if document_set != map_doc_ids:
        errors.append("doc_to_chunk_ids document set mismatch")
    if chunk_ids != map_chunk_ids:
        errors.append("doc_to_chunk_ids chunk set mismatch")
    if chunk_doc_ids != e5_chunk_doc_ids:
        errors.append("structural and E5 chunk order mismatch")
    if len(document_ids) != 8507 or len(document_set) != 8507:
        errors.append(f"document count mismatch: {len(document_ids)}")
    if len(chunk_doc_ids) != 343347 or e5_chunk_id_count != 343347:
        errors.append(f"chunk count mismatch: structural={len(chunk_doc_ids)} e5={e5_chunk_id_count}")
    if duplicate_documents or duplicate_chunks or orphan_chunks or noncontiguous:
        errors.append(
            f"corpus mapping invalid duplicates_docs={duplicate_documents} "
            f"duplicates_chunks={duplicate_chunks} orphans={orphan_chunks} noncontiguous={len(noncontiguous)}"
        )

    chunk_embeddings = np.load(E5_DIR / "embeddings.f16.npy", mmap_mode="r")
    query_embeddings = np.load(QUERY_DIR / "train_queries.f32.npy", mmap_mode="r")
    query_ids = [str(value) for value in load_json(QUERY_DIR / "train_query_ids.json")]
    if tuple(chunk_embeddings.shape) != (343347, DIMENSION) or chunk_embeddings.dtype != np.float16:
        errors.append(f"unexpected chunk embedding array: {chunk_embeddings.shape}/{chunk_embeddings.dtype}")
    if tuple(query_embeddings.shape) != (7000, DIMENSION) or query_embeddings.dtype != np.float32:
        errors.append(f"unexpected query embedding array: {query_embeddings.shape}/{query_embeddings.dtype}")
    if set(query_ids) != set(train) or len(query_ids) != len(set(query_ids)):
        errors.append("query embedding IDs do not exactly match train IDs")
    corpus_ids = set(document_set)
    missing_gold = sorted({doc_id for gold in answers.values() for doc_id in gold if doc_id not in corpus_ids})
    if missing_gold:
        errors.append(f"canonical labels missing from corpus: {missing_gold[:5]}")
    if label_stats["evaluable_queries"] != 6991 or label_stats["non_evaluable_queries"] != 9:
        errors.append("canonical label evaluability counts differ from frozen audit")

    selected = tuning_report.get("selected_by_candidate_budget", {}).get("150")
    if not selected:
        errors.append("BM25 tuning report has no selected_by_candidate_budget[150]")
    if tuning_report.get("status") != "PASS":
        errors.append("BM25 tuning report is not PASS")
    if evidence_manifest.get("config", {}).get("top_passages") != 4096:
        errors.append("BM25 evidence is not raw top-4096")
    if bm25_manifest.get("counts", {}).get("passages") != 343347:
        errors.append("BM25 passage count does not match frozen corpus")
    declared_db_hash = bm25_manifest.get("artifact_sha256") or bm25_manifest.get("sha256")
    if isinstance(declared_db_hash, Mapping):
        declared_db_hash = declared_db_hash.get(BM25_DB.name)
    if declared_db_hash:
        _check_expected_hash(BM25_DB, declared_db_hash, errors)
    if not (ROOT / "results" / "exp021_sparse" / "depth_rrf_tuning" / "manifest.json").exists():
        warnings.append("EXP-021 depth_rrf_tuning has no standalone manifest; tuning_report.json and raw evidence manifest are used.")
    if EXP022_CANDIDATES.exists():
        warnings.append("EXP-022 union artifact is present but explicitly excluded from EXP-109A scoring.")
    if len(answers) != 7000 or len(fold_for) != 7000:
        errors.append("train/fold count is not 7000")

    raw_evidence_qids: set[str] = set()
    raw_bm25_missing: set[str] = set()
    for row in read_jsonl(BM25_EVIDENCE_DIR / "shards" / "evidence_0000.jsonl"):
        raw_evidence_qids.add(str(row["qid"]))
        raw_bm25_missing.update(str(doc_id) for doc_id, _ranks in row.get("evidence", []) if str(doc_id) not in corpus_ids)
    for shard in sorted(BM25_EVIDENCE_DIR.joinpath("shards").glob("evidence_*.jsonl"))[1:]:
        for row in read_jsonl(shard):
            qid = str(row["qid"])
            if qid in raw_evidence_qids:
                errors.append(f"duplicate BM25 evidence qid: {qid}")
            raw_evidence_qids.add(qid)
            raw_bm25_missing.update(str(doc_id) for doc_id, _ranks in row.get("evidence", []) if str(doc_id) not in corpus_ids)
    if raw_evidence_qids != set(train):
        errors.append("raw BM25 evidence does not cover exactly the train query IDs")
    if raw_bm25_missing:
        errors.append(f"BM25 evidence has parent IDs outside the frozen corpus: {sorted(raw_bm25_missing)[:5]}")

    input_paths = [
        (TRAIN_PATH, None, True),
        (FOLDS_PATH, None, True),
        (EXCLUSIONS_PATH, None, True),
        (LABEL_IMPACT_PATH, None, True),
        (PROCESSED_MANIFEST, processed_manifest.get("schema_version"), True),
        (STRUCT_MANIFEST, structural_manifest.get("schema_version"), True),
        (STRUCT_DIR / "documents.jsonl", None, True),
        (STRUCT_DIR / "chunks.jsonl", None, True),
        (STRUCT_DIR / "doc_to_chunk_ids.json", None, True),
        (E5_MANIFEST, e5_manifest.get("schema_version"), True),
        (E5_DIR / "chunk_ids.jsonl", None, True),
        (E5_DIR / "embeddings.f16.npy", None, True),
        (QUERY_MANIFEST, query_manifest.get("schema_version"), True),
        (QUERY_DIR / "train_query_ids.json", None, True),
        (QUERY_DIR / "train_queries.f32.npy", None, True),
        (BM25_EVIDENCE_MANIFEST, evidence_manifest.get("schema_version"), True),
        (BM25_TUNING_REPORT, tuning_report.get("schema_version"), True),
        (BM25_DB, bm25_manifest.get("schema_version"), True),
        (BM25_MANIFEST, bm25_manifest.get("schema_version"), True),
        (EXP102_REPORT, None, True),
    ]
    inputs = []
    for path, schema, should_hash in input_paths:
        if path.exists():
            inputs.append(file_record(path, schema=schema, hash_file=should_hash))
        else:
            errors.append(f"missing input: {path}")
            inputs.append(file_record(path, schema=schema, hash_file=False))
    shard_records = []
    for shard in sorted(BM25_EVIDENCE_DIR.joinpath("shards").glob("evidence_*.jsonl")):
        shard_records.append(file_record(shard, hash_file=True))
    declared_shards = {str(row["name"]): str(row["sha256"]) for row in evidence_manifest.get("shards", [])}
    for record in shard_records:
        if record.get("sha256") != declared_shards.get(Path(record["path"]).name):
            errors.append(f"raw BM25 shard hash mismatch: {record['path']}")
    if len(shard_records) != int(evidence_manifest.get("queries", 0) / evidence_manifest.get("config", {}).get("shard_size", 128) + 1):
        warnings.append(f"unexpected raw evidence shard count: {len(shard_records)}")

    index = CorpusIndex.from_chunk_doc_ids(chunk_doc_ids)
    parent_index_dir = NAMESPACE_CACHE / "parent_index"
    parent_index_fingerprint = content_hash(index.to_json())
    atomic_json(parent_index_dir / "parent_index.json", index.to_json())
    write_success(parent_index_dir, stage="parent-index", fingerprint=parent_index_fingerprint)
    report = {
        "schema_version": SCHEMA,
        "stage": "input_audit",
        "status": "PASS" if not errors else "REJECTED_INPUT_GATE",
        "label_policy": LABEL_POLICY,
        "label_stats": label_stats,
        "train_queries": len(train),
        "folds": {name: len(values) for name, values in sorted(folds.items())},
        "corpus": {
            "documents": len(document_set),
            "chunks": len(chunk_doc_ids),
            "unique_chunks": len(chunk_ids),
            "e5_chunk_ids": e5_chunk_id_count,
            "noncontiguous_parents": len(noncontiguous),
            "parent_index_fingerprint": parent_index_fingerprint,
        },
        "frozen_manifests": {
            "processed": processed_manifest,
            "structural": structural_manifest,
            "e5": e5_manifest,
            "query_embeddings": query_manifest,
            "bm25_evidence": evidence_manifest,
            "bm25": bm25_manifest,
        },
        "bm25_contract": {
            "database": relative_path(BM25_DB),
            "raw_evidence": relative_path(BM25_EVIDENCE_DIR),
            "depth_selection": selected,
            "max_parent_rank": MAX_BM25_DOCUMENTS,
            "missing_rank_contribution": 0.0,
            "union_policy": "full dense parent ranking plus BM25 rank contribution; no append-union candidate cap",
        },
        "inputs": inputs,
        "raw_evidence_shards": shard_records,
        "warnings": warnings,
        "errors": errors,
        "audit_fingerprint": content_hash({
            "label_stats": label_stats,
            "corpus": {"documents": len(document_set), "chunks": len(chunk_doc_ids)},
            "input_sha256": [(row["path"], row.get("sha256")) for row in inputs],
            "shards": [(row["path"], row.get("sha256")) for row in shard_records],
        }),
    }
    output_dir = RESULTS_DIR / "input_audit"
    atomic_json(output_dir / "READING_AUDIT.json", report)
    if not errors:
        write_success(output_dir, stage="input-audit", fingerprint=report["audit_fingerprint"], extra={"code_sha256": sha256_file(CODE_FILE), "scorer_contract": SCORER_CONTRACT})
    return report


def _finite_difference_gradient(fn: Any, x: torch.Tensor, epsilon: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    value = x.detach().clone().requires_grad_(True)
    output = fn(value)
    output.backward()
    analytic = value.grad.detach().clone()
    numeric = torch.zeros_like(value)
    for index in range(value.numel()):
        plus = x.detach().clone()
        minus = x.detach().clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (fn(plus) - fn(minus)) / (2.0 * epsilon)
    return analytic, numeric


def math_loss_report() -> dict[str, Any]:
    """Run CPU-only invariants for the loss and implicit-gradient contract."""
    set_seed()
    errors: list[str] = []
    vectors: list[dict[str, Any]] = []
    for length in (8, 17, 64):
        for alpha in (1.0, 2.0, 5.0, 10.0):
            scores = torch.randn(length, dtype=torch.float64)
            membership = soft_top_k(scores, k=5, alpha=alpha)
            vectors.append({
                "length": length,
                "alpha": alpha,
                "sum": float(membership.sum()),
                "min": float(membership.min()),
                "max": float(membership.max()),
            })
            if not torch.isfinite(membership).all() or not (0 <= float(membership.min()) <= float(membership.max()) <= 1):
                errors.append(f"invalid soft membership length={length} alpha={alpha}")
            if abs(float(membership.sum()) - 5.0) > 2e-8:
                errors.append(f"soft membership sum mismatch length={length} alpha={alpha}")

    base = torch.tensor([-2.0, -0.2, 0.1, 0.9, 1.4, 2.8, 3.2, -1.3], dtype=torch.float64)
    perm = torch.tensor([3, 0, 7, 4, 1, 6, 2, 5])
    permuted = soft_top_k(base.index_select(0, perm), k=5, alpha=2.0)
    restored = torch.empty_like(permuted)
    restored[perm] = permuted
    if not torch.allclose(restored, soft_top_k(base, k=5, alpha=2.0), atol=1e-10, rtol=1e-10):
        errors.append("permutation equivariance failed")
    if not torch.allclose(
        soft_top5_membership(base + 17.0, alpha=2.0),
        soft_top5_membership(base, alpha=2.0),
        atol=1e-10,
        rtol=1e-10,
    ):
        errors.append("standardized shift invariance failed")
    finite_x = base.clone().requires_grad_(True)
    finite_loss = -torch.log(soft_top_k(finite_x, k=5, alpha=2.0)).mean()
    finite_loss.backward()
    if finite_x.grad is None or not torch.isfinite(finite_x.grad).all():
        errors.append("soft-top gradient is not finite")
    analytic, numeric = _finite_difference_gradient(
        lambda value: -torch.log(soft_top_k(value, k=5, alpha=2.0)).mean(),
        base,
    )
    max_gradient_error = float((analytic - numeric).abs().max())
    if max_gradient_error > 2e-4:
        errors.append(f"implicit gradient finite-difference error {max_gradient_error}")

    gold_mask = torch.tensor([True, True, False, False, False, False])
    scores = torch.tensor([2.0, 1.0, 0.4, 0.1, -0.3, -0.9], dtype=torch.float64)
    observed_terms = decoupled_terms(scores, gold_mask, tau=0.5)
    expected_denominator = torch.logsumexp(scores[~gold_mask] / 0.5, dim=0)
    expected = expected_denominator - scores[gold_mask] / 0.5
    if not torch.allclose(observed_terms, expected):
        errors.append("decoupled denominator includes an unexpected positive")
    if torch.isclose(
        observed_terms[0],
        torch.logsumexp(scores[1:] / 0.5, dim=0) - scores[0] / 0.5,
    ):
        errors.append("decoupled denominator appears to include other positives")
    hybrid_scores = scores.clone().requires_grad_(True)
    hybrid_total, hybrid_dec, hybrid_top = hybrid_loss(
        hybrid_scores,
        gold_mask,
        [0, 1],
        alpha=2.0,
        lambda_top5=0.25,
    )
    hybrid_total.backward()
    if not torch.isfinite(hybrid_total) or hybrid_scores.grad is None or not torch.isfinite(hybrid_scores.grad).all():
        errors.append("hybrid loss gradient is not finite")

    report = {
        "schema_version": SCHEMA,
        "stage": "math_loss",
        "status": "PASS" if not errors else "REJECTED_MATH_GATE",
        "contract": {
            "soft_top_k": "standardize full parent score vector; solve sum sigmoid(alpha*(x+t))=5; analytic implicit backward",
            "decoupled": "each positive denominator is all non-golds; other positives excluded",
            "tau": TAU,
            "soft_k": SOFT_K,
        },
        "vectors": vectors,
        "finite_difference_max_abs_error": max_gradient_error,
        "hybrid_values": {
            "total": float(hybrid_total.detach()),
            "decoupled": float(hybrid_dec.detach()),
            "top5": float(hybrid_top.detach()),
        },
        "errors": errors,
        "math_fingerprint": content_hash({"vectors": vectors, "error": max_gradient_error, "errors": errors}),
    }
    output_dir = RESULTS_DIR / "math"
    atomic_json(output_dir / "LOSS_MATH.json", report)
    if not errors:
        write_success(output_dir, stage="math-loss", fingerprint=report["math_fingerprint"], extra={"code_sha256": sha256_file(CODE_FILE), "scorer_contract": SCORER_CONTRACT})
    return report


@dataclass
class FrozenInputs:
    train: dict[str, dict[str, Any]]
    qids: list[str]
    folds: dict[str, list[str]]
    fold_for: dict[str, str]
    answers: dict[str, set[str]]
    label_stats: dict[str, Any]
    query_ids: list[str]
    query_embeddings: np.ndarray
    chunk_embeddings: np.ndarray
    corpus: CorpusIndex
    bm25_evidence: dict[str, list[list[Any]]]
    bm25_by_fold: dict[str, dict[str, Any]]


def load_query_embeddings() -> tuple[list[str], np.ndarray]:
    query_ids = [str(value) for value in load_json(QUERY_DIR / "train_query_ids.json")]
    query_embeddings = np.load(QUERY_DIR / "train_queries.f32.npy", mmap_mode="r")
    if query_embeddings.dtype != np.float32 or query_embeddings.shape != (len(query_ids), DIMENSION):
        raise RuntimeError(f"unexpected query embedding matrix: {query_embeddings.shape}/{query_embeddings.dtype}")
    return query_ids, query_embeddings


def load_bm25_evidence() -> dict[str, list[list[Any]]]:
    evidence: dict[str, list[list[Any]]] = {}
    for row in read_jsonl(BM25_EVIDENCE_DIR / "shards" / "evidence_0000.jsonl"):
        evidence[str(row["qid"])] = row["evidence"]
    for shard in sorted(BM25_EVIDENCE_DIR.joinpath("shards").glob("evidence_*.jsonl"))[1:]:
        for row in read_jsonl(shard):
            qid = str(row["qid"])
            if qid in evidence:
                raise ValueError(f"duplicate BM25 evidence qid: {qid}")
            evidence[qid] = row["evidence"]
    if len(evidence) != 7000:
        raise RuntimeError(f"BM25 evidence query count mismatch: {len(evidence)}")
    return evidence


def load_bm25_evidence_from_dir(evidence_dir: Path = BM25_EVIDENCE_DIR) -> dict[str, list[list[Any]]]:
    evidence: dict[str, list[list[Any]]] = {}
    for shard in sorted(evidence_dir.joinpath("shards").glob("evidence_*.jsonl")):
        for row in read_jsonl(shard):
            qid = str(row["qid"])
            if qid in evidence:
                raise ValueError(f"duplicate BM25 evidence qid: {qid}")
            evidence[qid] = row["evidence"]
    return evidence


def load_selected_bm25_configs() -> dict[str, dict[str, Any]]:
    report = load_json(BM25_TUNING_REPORT)
    selected = report.get("selected_by_candidate_budget", {}).get("150")
    if not isinstance(selected, Mapping) or set(selected) != {"fold_0", "fold_1", "fold_2", "fold_3", "fold_4"}:
        raise RuntimeError("BM25 selected_by_candidate_budget[150] is incomplete")
    return {str(fold): dict(config) for fold, config in selected.items()}


def load_frozen_inputs(*, include_bm25: bool = True) -> FrozenInputs:
    train_raw = load_json(TRAIN_PATH)
    train = {str(qid): dict(value) for qid, value in train_raw.items()}
    qids = sorted(train)
    folds, fold_for = load_folds()
    answers, label_stats = canonical_labels()
    query_ids, query_embeddings = load_query_embeddings()
    chunk_embeddings = np.load(E5_DIR / "embeddings.f16.npy", mmap_mode="r")
    chunk_doc_ids = load_chunk_doc_ids()
    corpus = CorpusIndex.from_chunk_doc_ids(chunk_doc_ids)
    evidence = load_bm25_evidence() if include_bm25 else {}
    selected = load_selected_bm25_configs() if include_bm25 else {}
    return FrozenInputs(
        train=train,
        qids=qids,
        folds=folds,
        fold_for=fold_for,
        answers=answers,
        label_stats=label_stats,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        chunk_embeddings=chunk_embeddings,
        corpus=corpus,
        bm25_evidence=evidence,
        bm25_by_fold=selected,
    )


def _sparse_rankings(
    evidence: Sequence[Sequence[Any]],
    *,
    depth: int,
    parent_rrf_k: int,
) -> tuple[list[str], list[str]]:
    eligible: list[tuple[str, list[int]]] = []
    for raw_doc_id, raw_ranks in evidence:
        ranks = [int(value) for value in raw_ranks if int(value) <= int(depth)]
        if ranks:
            eligible.append((str(raw_doc_id), ranks))
    first = [
        doc_id
        for doc_id, _ in sorted(eligible, key=lambda item: (item[1][0], item[0]))[:MAX_BM25_DOCUMENTS]
    ]
    rrf = [
        doc_id
        for doc_id, _score, first_rank in sorted(
            (
                (
                    doc_id,
                    sum(1.0 / (int(parent_rrf_k) + rank) for rank in ranks),
                    ranks[0],
                )
                for doc_id, ranks in eligible
            ),
            key=lambda item: (-item[1], item[2], item[0]),
        )[:MAX_BM25_DOCUMENTS]
    ]
    return first, rrf


def _sparse_cascade(
    first: Sequence[str],
    rrf: Sequence[str],
    *,
    fusion_rrf_k: int,
    head: int,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for ranking in (first, rrf):
        for rank, doc_id in enumerate(ranking, start=1):
            doc_id = str(doc_id)
            scores[doc_id] += 1.0 / (int(fusion_rrf_k) + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    fused = sorted(scores, key=lambda doc_id: (-scores[doc_id], best_rank[doc_id], doc_id))[:MAX_BM25_DOCUMENTS]
    head_ids = set(fused[: int(head)])
    return (fused[: int(head)] + [doc_id for doc_id in rrf if doc_id not in head_ids])[:MAX_BM25_DOCUMENTS]


def sparse_parent_ranking(
    qid: str,
    *,
    evidence: Mapping[str, Sequence[Sequence[Any]]],
    fold_for: Mapping[str, str],
    selected_by_fold: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if qid not in evidence:
        return []
    config = selected_by_fold[fold_for[qid]]
    first, rrf = _sparse_rankings(
        evidence[qid],
        depth=int(config["depth"]),
        parent_rrf_k=int(config["parent_rrf_k"]),
    )
    return _sparse_cascade(
        first,
        rrf,
        fusion_rrf_k=int(config["fusion_rrf_k"]),
        head=int(config["head_cutoff"]),
    )


def weighted_rrf_ranking(
    dense_order: Sequence[str],
    sparse_order: Sequence[str],
    *,
    dense_weight: float,
    rrf_k: int,
    limit: int = 150,
) -> list[str]:
    """Fuse full dense parent ranking with sparse ranks.

    Dense ranks cover every parent. A parent absent from BM25 receives exactly
    zero sparse contribution; no post-hoc append-union is performed.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    dense_weight = float(dense_weight)
    sparse_weight = 1.0 - dense_weight
    seen_dense: set[str] = set()
    for rank, raw_doc_id in enumerate(dense_order, start=1):
        doc_id = str(raw_doc_id)
        if doc_id in seen_dense:
            raise ValueError(f"duplicate dense document: {doc_id}")
        seen_dense.add(doc_id)
        scores[doc_id] = dense_weight / (float(rrf_k) + rank)
        best_rank[doc_id] = rank
    seen_sparse: set[str] = set()
    for rank, raw_doc_id in enumerate(sparse_order, start=1):
        doc_id = str(raw_doc_id)
        if doc_id in seen_sparse:
            raise ValueError(f"duplicate sparse document: {doc_id}")
        seen_sparse.add(doc_id)
        if doc_id not in scores:
            scores[doc_id] = 0.0
            best_rank[doc_id] = rank
        scores[doc_id] += sparse_weight / (float(rrf_k) + rank)
        best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], best_rank[doc_id], doc_id))[: int(limit)]


def candidate_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [{"arm": "A_decoupled", "alpha": None, "lambda_top5": 0.0}]
    configs.extend({"arm": "B_softtop5", "alpha": alpha, "lambda_top5": 0.0} for alpha in ALPHAS)
    configs.extend(
        {"arm": "C_hybrid", "alpha": alpha, "lambda_top5": lambda_top5}
        for alpha in HYBRID_ALPHAS
        for lambda_top5 in LAMBDAS
    )
    return configs


def parameter_label(config: Mapping[str, Any]) -> str:
    arm = str(config["arm"])
    if arm == "A_decoupled":
        return arm
    if arm == "B_softtop5":
        return f"{arm}_alpha{float(config['alpha']):g}"
    return f"{arm}_alpha{float(config['alpha']):g}_lambda{float(config['lambda_top5']):g}"


def arm_loss(
    parent_scores: torch.Tensor,
    gold_indices: Sequence[int],
    *,
    arm: str,
    alpha: float | None = None,
    lambda_top5: float = 0.0,
    tau: float = TAU,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    gold_mask = torch.zeros(parent_scores.numel(), dtype=torch.bool, device=parent_scores.device)
    gold_mask[torch.as_tensor(list(gold_indices), dtype=torch.long, device=parent_scores.device)] = True
    if arm == "A_decoupled":
        value = decoupled_loss(parent_scores, gold_mask, tau=tau)
        return value, {"decoupled": value, "top5": value.detach() * 0.0}
    if alpha is None:
        raise ValueError(f"alpha is required for {arm}")
    if arm == "B_softtop5":
        value = soft_top5_loss(parent_scores, gold_indices, alpha=float(alpha))
        return value, {"decoupled": value.detach() * 0.0, "top5": value}
    if arm == "C_hybrid":
        value, decoupled, top5 = hybrid_loss(
            parent_scores,
            gold_mask,
            gold_indices,
            alpha=float(alpha),
            lambda_top5=float(lambda_top5),
            tau=tau,
        )
        return value, {"decoupled": decoupled, "top5": top5}
    raise ValueError(f"unknown arm: {arm}")


def gold_indices_for_qid(qid: str, inputs: FrozenInputs) -> list[int]:
    parent_index = {doc_id: index for index, doc_id in enumerate(inputs.corpus.doc_ids)}
    missing = [doc_id for doc_id in inputs.answers[qid] if doc_id not in parent_index]
    if missing:
        raise RuntimeError(f"canonical gold absent from corpus for {qid}: {missing}")
    return sorted(parent_index[doc_id] for doc_id in inputs.answers[qid])


def rank_metrics(prediction: Sequence[str], gold: set[str], ks: Sequence[int] = CURVE_KS) -> dict[str, float]:
    values: dict[str, float] = {}
    for k in ks:
        values[f"recall@{k}"] = len(set(map(str, prediction[:k])) & gold) / len(gold) if gold else 0.0
    values["precision@5"] = len(set(map(str, prediction[:5])) & gold) / 5.0 if gold else 0.0
    first_rank = next((rank for rank, doc_id in enumerate(prediction[:5], start=1) if str(doc_id) in gold), None)
    values["mrr@5"] = 1.0 / first_rank if first_rank else 0.0
    return values


def aggregate_metrics(
    predictions: Mapping[str, Sequence[str]],
    answers: Mapping[str, set[str]],
    qids: Sequence[str],
    *,
    baseline_predictions: Mapping[str, Sequence[str]] | None = None,
    dense_scores: Mapping[str, Sequence[float]] | None = None,
    dense_order: Mapping[str, Sequence[str]] | None = None,
    outer_train_qids: Sequence[str] | None = None,
    _include_slices: bool = True,
) -> dict[str, Any]:
    evaluable = [qid for qid in qids if answers.get(qid)]
    per_query: dict[str, dict[str, Any]] = {}
    sums = Counter()
    for qid in evaluable:
        metric = rank_metrics(predictions.get(qid, []), answers[qid])
        per_query[qid] = metric
        sums.update(metric)
    count = len(evaluable)
    metrics = {key: float(value / count) if count else 0.0 for key, value in sums.items()}
    metrics["evaluable_queries"] = count
    metrics["non_evaluable_queries"] = len([qid for qid in qids if not answers.get(qid)])
    metrics["ceiling_gap_to_full_corpus"] = {
        f"recall@{k}": float(1.0 - metrics.get(f"recall@{k}", 0.0)) for k in CURVE_KS
    }

    if _include_slices:
        by_cardinality: dict[str, Any] = {}
        for name, selector in (
            ("single_gold", lambda gold: len(gold) == 1),
            ("multi_gold", lambda gold: len(gold) >= 2),
        ):
            selected_qids = [qid for qid in evaluable if selector(answers[qid])]
            by_cardinality[name] = aggregate_metrics(
                predictions,
                answers,
                selected_qids,
                baseline_predictions=None,
                dense_scores=None,
                dense_order=None,
                outer_train_qids=None,
                _include_slices=False,
            )
        metrics["cardinality"] = by_cardinality

    if baseline_predictions is not None:
        movement: list[int] = []
        margins: list[float] = []
        for qid in evaluable:
            base_rank = {str(doc_id): rank for rank, doc_id in enumerate(baseline_predictions.get(qid, []), start=1)}
            new_rank = {str(doc_id): rank for rank, doc_id in enumerate(predictions.get(qid, []), start=1)}
            movement.extend(
                new_rank[doc_id] - base_rank[doc_id]
                for doc_id in answers[qid]
                if doc_id in base_rank and doc_id in new_rank
            )
            if dense_scores and len(dense_scores.get(qid, [])) >= 6:
                ordered_scores = sorted(map(float, dense_scores[qid]), reverse=True)
                margins.append(ordered_scores[4] - ordered_scores[5])
        metrics["movement"] = {
            "gold_rank_delta_mean": float(np.mean(movement)) if movement else 0.0,
            "gold_rank_delta_median": float(np.median(movement)) if movement else 0.0,
            "gold_rank_delta_count": len(movement),
        }
        metrics["rank5_rank6_margin"] = {
            "dense_score_gap_mean": float(np.mean(margins)) if margins else 0.0,
            "dense_score_gap_median": float(np.median(margins)) if margins else 0.0,
        }
        deltas = [
            per_query[qid]["recall@5"]
            - rank_metrics(baseline_predictions.get(qid, []), answers[qid])["recall@5"]
            for qid in evaluable
        ]
        metrics["improved_queries"] = int(sum(value > 1e-12 for value in deltas))
        metrics["degraded_queries"] = int(sum(value < -1e-12 for value in deltas))
        metrics["tied_queries"] = int(sum(abs(value) <= 1e-12 for value in deltas))

    if outer_train_qids is not None:
        frequencies = Counter(doc_id for qid in outer_train_qids for doc_id in answers.get(qid, set()))
        frequency_groups = {
            "unseen": lambda value: value == 0,
            "once": lambda value: value == 1,
            "2-4": lambda value: 2 <= value <= 4,
            "5+": lambda value: value >= 5,
        }
        frequency_metrics: dict[str, Any] = {}
        for group, selector in frequency_groups.items():
            group_values: list[float] = []
            for qid in evaluable:
                for gold_doc in answers[qid]:
                    if selector(frequencies[gold_doc]):
                        group_values.append(float(gold_doc in set(predictions.get(qid, [])[:5])))
            frequency_metrics[group] = {
                "gold_occurrences": len(group_values),
                "hit_rate_at_5": float(np.mean(group_values)) if group_values else None,
            }
        metrics["label_frequency_slices"] = frequency_metrics
    metrics["per_query"] = per_query
    return metrics


def paired_bootstrap_ci(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    seed: int = SEED,
    samples: int = 2000,
) -> dict[str, float]:
    if len(baseline) != len(treatment):
        raise ValueError("paired bootstrap inputs must have equal length")
    if not baseline:
        return {"delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "samples": 0}
    rng = np.random.default_rng(seed)
    base = np.asarray(baseline, dtype=np.float64)
    treat = np.asarray(treatment, dtype=np.float64)
    delta = treat - base
    draws = rng.integers(0, len(delta), size=(int(samples), len(delta)))
    estimates = delta[draws].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "samples": int(samples),
    }


def _metric_value(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key, 0.0)
    if isinstance(value, Mapping):
        return float(value.get("recall@5", 0.0))
    return float(value)


def _candidate_priority(config: Mapping[str, Any]) -> tuple[int, float, float, str]:
    if "arm" not in config:
        return (9, 0.0, 0.0, fusion_label(config))
    arm = str(config["arm"])
    arm_priority = {"A_decoupled": 0, "C_hybrid": 1, "B_softtop5": 2}.get(arm, 9)
    lambda_value = float(config.get("lambda_top5") or 0.0)
    label = parameter_label(config) if "arm" in config else fusion_label(config)
    return arm_priority, lambda_value, float(config.get("alpha") or 0.0), label


def select_candidate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select with metric tolerances and the plan's explicit tie policy."""
    if not records:
        raise ValueError("cannot select from an empty candidate set")
    ordered_keys = ("recall@5", "multi_gold_recall@5", "precision@5", "recall@16", "mrr@5")
    chosen = records[0]
    for candidate in records[1:]:
        decision: int | None = None
        for key in ordered_keys:
            difference = float(candidate.get(key, 0.0)) - float(chosen.get(key, 0.0))
            if abs(difference) > 1e-4:
                decision = 1 if difference > 0 else -1
                break
        if decision is None:
            if _candidate_priority(candidate.get("config", candidate)) < _candidate_priority(chosen.get("config", chosen)):
                decision = 1
            else:
                decision = -1
        if decision > 0:
            chosen = candidate
    return chosen


def select_fusion(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Fusion selection follows the same held-in-fold validation metrics."""
    if not records:
        raise ValueError("cannot select fusion from empty records")
    return select_candidate(records)


def _query_index(inputs: FrozenInputs) -> dict[str, int]:
    mapping = {qid: index for index, qid in enumerate(inputs.query_ids)}
    if set(mapping) != set(inputs.qids):
        raise RuntimeError("query embedding IDs do not match frozen training IDs")
    return mapping


def _as_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = torch.device(device)
    if value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return value


def dense_rank_one(
    model: nn.Module,
    raw_query: np.ndarray | torch.Tensor,
    inputs: FrozenInputs,
    *,
    device: torch.device,
    with_grad: bool = False,
) -> tuple[list[str], np.ndarray]:
    if isinstance(raw_query, np.ndarray):
        query = torch.from_numpy(np.asarray(raw_query, dtype=np.float32))
    else:
        query = raw_query
    query = query.to(device=device, dtype=torch.float32)
    if query.ndim == 1:
        query = query.unsqueeze(0)
    context = contextlib.nullcontext() if with_grad else torch.no_grad()
    with context:
        projected = model(query)
        scores = compute_parent_scores(
            projected,
            inputs.chunk_embeddings,
            inputs.corpus,
            chunk_block_size=PARENT_CHUNK_BLOCK,
        )
    score_vector = scores[0]
    score_np = score_vector.detach().to("cpu", dtype=torch.float32).numpy()
    order = _stable_parent_order(score_np, inputs.corpus.doc_ids)
    return [inputs.corpus.doc_ids[index] for index in order], score_np


def score_and_fuse(
    model: nn.Module,
    qids: Sequence[str],
    inputs: FrozenInputs,
    *,
    device: torch.device,
    fusion_configs: Sequence[Mapping[str, Any]] | None = None,
    limit: int = 150,
) -> dict[str, dict[str, Any]]:
    """Exact full-parent scoring followed immediately by optional RRF grids."""
    fusion_configs = list(fusion_configs or [{"dense_weight": 1.0, "rrf_k": 60}])
    qid_to_index = _query_index(inputs)
    model.eval()
    outputs: dict[str, dict[str, Any]] = {
        fusion_label(config): {"predictions": {}, "dense_predictions": {}, "dense_scores": {}}
        for config in fusion_configs
    }
    for qid in map(str, qids):
        dense_order, score_vector = dense_rank_one(
            model,
            inputs.query_embeddings[qid_to_index[qid]],
            inputs,
            device=device,
            with_grad=False,
        )
        sparse_order = sparse_parent_ranking(
            qid,
            evidence=inputs.bm25_evidence,
            fold_for=inputs.fold_for,
            selected_by_fold=inputs.bm25_by_fold,
        )
        for config_number, config in enumerate(fusion_configs):
            label = fusion_label(config)
            if float(config["dense_weight"]) >= 0.999999:
                prediction = dense_order[: int(limit)]
            else:
                prediction = weighted_rrf_ranking(
                    dense_order,
                    sparse_order,
                    dense_weight=float(config["dense_weight"]),
                    rrf_k=int(config["rrf_k"]),
                    limit=limit,
                )
            outputs[label]["predictions"][qid] = prediction
            outputs[label]["dense_predictions"][qid] = dense_order[: int(limit)]
            # The dense vector is identical for every fusion candidate. Keep a
            # single copy to avoid multiplying the full 8.5K-parent vector by
            # the 25-point RRF grid in memory.
            if config_number == 0:
                outputs[label]["dense_scores"][qid] = score_vector.tolist()
    return outputs


def fusion_label(config: Mapping[str, Any]) -> str:
    weight = float(config["dense_weight"])
    rrf_k = int(config["rrf_k"])
    return f"dense{weight:g}_bm25{1.0 - weight:g}_rrfk{rrf_k}"


def fusion_grid() -> list[dict[str, Any]]:
    return [
        {"dense_weight": dense_weight, "bm25_weight": 1.0 - dense_weight, "rrf_k": rrf_k}
        for dense_weight in RRF_WEIGHTS
        for rrf_k in RRF_KS
    ]


def _flatten_selection_metrics(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    multi = metrics.get("cardinality", {}).get("multi_gold", {})
    return {
        "config": dict(config),
        "recall@5": float(metrics.get("recall@5", 0.0)),
        "multi_gold_recall@5": float(multi.get("recall@5", 0.0)),
        "precision@5": float(metrics.get("precision@5", 0.0)),
        "recall@16": float(metrics.get("recall@16", 0.0)),
        "mrr@5": float(metrics.get("mrr@5", 0.0)),
        "evaluable_queries": int(metrics.get("evaluable_queries", 0)),
    }


def evaluate_fusion_grid(
    model: nn.Module,
    qids: Sequence[str],
    inputs: FrozenInputs,
    *,
    device: torch.device,
    configs: Sequence[Mapping[str, Any]] | None = None,
    baseline_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    configs = list(configs or fusion_grid())
    scored = score_and_fuse(model, qids, inputs, device=device, fusion_configs=configs)
    records: list[dict[str, Any]] = []
    baseline_predictions = None
    if baseline_config is not None:
        baseline_predictions = scored[fusion_label(baseline_config)]["predictions"]
    for config in configs:
        label = fusion_label(config)
        metric = aggregate_metrics(
            scored[label]["predictions"],
            inputs.answers,
            qids,
            baseline_predictions=baseline_predictions,
            dense_scores=scored[label]["dense_scores"],
            outer_train_qids=None,
        )
        records.append(_flatten_selection_metrics(metric, config))
    selected = select_fusion(records)
    return {
        "selected": dict(selected),
        "records": records,
        "scored": scored,
    }


def _rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    # Checkpoints may be loaded with map_location='cuda'; CPU's default RNG
    # setter still requires a CPU ByteTensor.
    torch.set_rng_state(value["torch"].detach().to(device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state_all([state.detach().to(device="cpu", dtype=torch.uint8) for state in value["cuda"]])


def _training_config(
    config: Mapping[str, Any],
    *,
    query_batch: int = 1,
    epochs: int = 4,
) -> dict[str, Any]:
    return {
        "optimizer": "AdamW",
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "epochs": int(epochs),
        "gradient_clip_norm": 1.0,
        "query_batch": int(query_batch),
        "tau": TAU,
        "soft_k": SOFT_K,
        "dimension": DIMENSION,
        "rank": RANK,
        "arm": str(config["arm"]),
        "alpha": config.get("alpha"),
        "lambda_top5": float(config.get("lambda_top5") or 0.0),
        "early_stopping": False,
        "candidate_policy": "full_corpus_parent_scores",
    }


def _gold_index_cache(inputs: FrozenInputs) -> dict[str, list[int]]:
    parent_index = {doc_id: index for index, doc_id in enumerate(inputs.corpus.doc_ids)}
    values: dict[str, list[int]] = {}
    for qid in inputs.qids:
        values[qid] = sorted(parent_index[doc_id] for doc_id in inputs.answers[qid] if doc_id in parent_index)
    return values


def train_model(
    inputs: FrozenInputs,
    qids: Sequence[str],
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    device: torch.device,
    resume: bool = False,
    query_batch: int = 1,
    epochs: int = 4,
    input_fingerprint: str | None = None,
    allow_orchestration_only_resume: bool = False,
) -> tuple[ResidualProjection, dict[str, Any]]:
    """Train one arm with exact query-boundary checkpoints."""
    qids = [str(qid) for qid in qids if inputs.answers.get(str(qid))]
    if not qids:
        raise ValueError("training split has no evaluable queries")
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = _training_config(config, query_batch=query_batch, epochs=epochs)
    input_fingerprint = input_fingerprint or content_hash({
        "corpus": content_hash(inputs.corpus.to_json()),
        "query_ids": inputs.query_ids,
        "label": inputs.label_stats["label_fingerprint"],
    })
    code_fingerprint = sha256_file(CODE_FILE)
    config_fingerprint = content_hash({"training": training_config, "input": input_fingerprint, "code": code_fingerprint})
    checkpoint_path = output_dir / "checkpoint.pt"
    final_path = output_dir / "model.pt"

    set_seed(SEED)
    model = ResidualProjection(dimension=DIMENSION, rank=RANK).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    state = {
        "epoch": 0,
        "position": 0,
        "order": None,
        "loss_history": [],
        "groups_completed": 0,
    }
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("config_fingerprint") != config_fingerprint:
            compatible = (
                allow_orchestration_only_resume
                and saved.get("training_config") == training_config
                and saved.get("input_fingerprint") == input_fingerprint
            )
            if not compatible:
                raise RuntimeError(f"training checkpoint fingerprint mismatch: {checkpoint_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state.update({key: saved[key] for key in state if key in saved})
        _restore_rng_state(saved["rng"])

    gold_cache = _gold_index_cache(inputs)
    qid_to_index = _query_index(inputs)
    started = time.time()
    while int(state["epoch"]) < int(epochs):
        epoch = int(state["epoch"])
        if state["order"] is None:
            order = list(qids)
            random.Random(SEED + epoch).shuffle(order)
            state["order"] = order
            state["position"] = 0
        order = [str(value) for value in state["order"]]
        model.train()
        position = int(state["position"])
        while position < len(order):
            group = order[position : position + int(query_batch)]
            optimizer.zero_grad(set_to_none=True)
            raw_queries = torch.from_numpy(np.asarray(
                [inputs.query_embeddings[qid_to_index[qid]] for qid in group],
                dtype=np.float32,
            )).to(device=device)
            projected = model(raw_queries)
            parent_scores = compute_parent_scores(
                projected,
                inputs.chunk_embeddings,
                inputs.corpus,
                chunk_block_size=PARENT_CHUNK_BLOCK,
            )
            losses: list[torch.Tensor] = []
            decoupled_graph_terms: list[torch.Tensor] = []
            top5_graph_terms: list[torch.Tensor] = []
            dec_values: list[float] = []
            top_values: list[float] = []
            for row_number, qid in enumerate(group):
                if not gold_cache[qid]:
                    continue
                value, terms = arm_loss(
                    parent_scores[row_number],
                    gold_cache[qid],
                    arm=str(config["arm"]),
                    alpha=config.get("alpha"),
                    lambda_top5=float(config.get("lambda_top5") or 0.0),
                    tau=TAU,
                )
                losses.append(value)
                if str(config["arm"]) == "C_hybrid":
                    decoupled_graph_terms.append(terms["decoupled"])
                    top5_graph_terms.append(terms["top5"])
                dec_values.append(float(terms["decoupled"].detach()))
                top_values.append(float(terms["top5"].detach()))
            if not losses:
                position += len(group)
                state["position"] = position
                continue
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {config['arm']} loss at epoch={epoch} position={position}")
            if str(config["arm"]) == "C_hybrid":
                term_decoupled = torch.stack(decoupled_graph_terms).mean()
                term_top5 = torch.stack(top5_graph_terms).mean()
                decoupled_grad_norm = _tensor_grad_norms(term_decoupled, tuple(model.parameters()))
                top5_grad_norm = _tensor_grad_norms(term_top5, tuple(model.parameters()))
            else:
                decoupled_grad_norm = 0.0
                top5_grad_norm = 0.0
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["gradient_clip_norm"])))
            optimizer.step()
            scheduler.step()
            position += len(group)
            state["position"] = position
            state["groups_completed"] = int(state["groups_completed"]) + 1
            state["loss_history"].append({
                "epoch": epoch,
                "position": position,
                "loss": float(loss.detach()),
                "decoupled": float(np.mean(dec_values)) if dec_values else 0.0,
                "top5": float(np.mean(top_values)) if top_values else 0.0,
                "decoupled_grad_norm": decoupled_grad_norm,
                "top5_grad_norm": top5_grad_norm,
                "grad_norm": grad_norm,
            })
            if int(state["groups_completed"]) % CHECKPOINT_QUERY_INTERVAL == 0:
                _save_training_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    state,
                    training_config,
                    input_fingerprint,
                    code_fingerprint,
                    config_fingerprint,
                )
        state["epoch"] = epoch + 1
        state["position"] = 0
        state["order"] = None
        _save_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            state,
            training_config,
            input_fingerprint,
            code_fingerprint,
            config_fingerprint,
        )
    atomic_torch(final_path, model.state_dict())
    summary = {
        "config": dict(config),
        "training_config": training_config,
        "input_fingerprint": input_fingerprint,
        "code_fingerprint": code_fingerprint,
        "config_fingerprint": config_fingerprint,
        "groups_completed": int(state["groups_completed"]),
        "epochs_completed": int(state["epoch"]),
        "loss_first": state["loss_history"][0]["loss"] if state["loss_history"] else None,
        "loss_last": state["loss_history"][-1]["loss"] if state["loss_history"] else None,
        "elapsed_seconds": time.time() - started,
        "checkpoint": relative_path(checkpoint_path),
        "model": relative_path(final_path),
    }
    atomic_json(output_dir / "TRAINING_SUMMARY.json", summary)
    return model, summary


def _save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: Mapping[str, Any],
    training_config: Mapping[str, Any],
    input_fingerprint: str,
    code_fingerprint: str,
    config_fingerprint: str,
) -> None:
    atomic_torch(
        path,
        {
            "schema_version": SCHEMA,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            **dict(state),
            "training_config": dict(training_config),
            "input_fingerprint": input_fingerprint,
            "code_fingerprint": code_fingerprint,
            "config_fingerprint": config_fingerprint,
            "rng": _rng_state(),
        },
    )


def _tensor_grad_norms(
    value: torch.Tensor,
    parameters: Sequence[torch.Tensor],
) -> float:
    gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    squared = [
        gradient.detach().pow(2).sum()
        for gradient in gradients
        if gradient is not None
    ]
    return float(torch.sqrt(torch.stack(squared).sum())) if squared else 0.0


def _preflight_trial(
    inputs: FrozenInputs,
    *,
    device: torch.device,
    batch_size: int,
    arm: str,
) -> dict[str, Any]:
    qids = [qid for qid in inputs.qids if inputs.answers.get(qid)][: int(batch_size)]
    qid_to_index = _query_index(inputs)
    model = ResidualProjection().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    raw = torch.from_numpy(np.asarray([inputs.query_embeddings[qid_to_index[qid]] for qid in qids], dtype=np.float32)).to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        free_before, total = torch.cuda.mem_get_info(device)
    else:
        free_before, total = 0, 0
    started = time.time()
    try:
        optimizer.zero_grad(set_to_none=True)
        projected = model(raw)
        scores = compute_parent_scores(projected, inputs.chunk_embeddings, inputs.corpus)
        values = []
        gold_cache = _gold_index_cache(inputs)
        for row, qid in enumerate(qids):
            value, _terms = arm_loss(
                scores[row],
                gold_cache[qid],
                arm=arm,
                alpha=2.0 if arm != "A_decoupled" else None,
                lambda_top5=0.25 if arm == "C_hybrid" else 0.0,
            )
            values.append(value)
        torch.stack(values).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        free_after, _ = torch.cuda.mem_get_info(device) if device.type == "cuda" else (0, 0)
        return {
            "batch_size": int(batch_size),
            "arm": arm,
            "status": "PASS",
            "seconds": time.time() - started,
            "peak_memory_bytes": peak,
            "free_before_bytes": int(free_before),
            "free_after_bytes": int(free_after),
            "total_memory_bytes": int(total),
            "headroom_after_peak_ratio": float((total - peak) / total) if total else None,
        }
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower() or "cuda error" in str(exc).lower()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "batch_size": int(batch_size),
            "arm": arm,
            "status": "OOM" if is_oom else "ERROR",
            "error": str(exc),
            "seconds": time.time() - started,
        }


def run_preflight() -> dict[str, Any]:
    output_dir = RESULTS_DIR / "preflight"
    logger = RunLog("preflight")
    try:
        if not torch.cuda.is_available():
            report = {
                "schema_version": SCHEMA,
                "stage": "preflight",
                "status": "REJECTED_PREFLIGHT_NO_CUDA",
                "device": "cpu",
                "errors": ["CUDA is required for the actual full-corpus feasibility gate; no silent CPU fallback."],
            }
            atomic_json(output_dir / "PREFLIGHT.json", report)
            logger.finish(report["status"])
            return report
        inputs = load_frozen_inputs(include_bm25=False)
        device = torch.device("cuda")
        trials = []
        for batch_size in (1, 2, 4, 8):
            for arm in ("A_decoupled", "B_softtop5"):
                logger.update(state="RUNNING", batch_size=batch_size, arm=arm)
                trials.append(_preflight_trial(inputs, device=device, batch_size=batch_size, arm=arm))
        passing = [
            row
            for row in trials
            if row["status"] == "PASS" and float(row.get("headroom_after_peak_ratio") or 0.0) >= 0.10
        ]
        selected_batch = max((int(row["batch_size"]) for row in passing), default=None)
        errors = [] if selected_batch is not None else ["no batch size passed the 10% VRAM headroom gate"]
        report = {
            "schema_version": SCHEMA,
            "stage": "preflight",
            "status": "PASS" if not errors else "REJECTED_PREFLIGHT_GATE",
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "memory_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "corpus": {
                "parents": len(inputs.corpus.doc_ids),
                "chunks": len(inputs.corpus.chunk_doc_ids),
                "scorer": "full-corpus source-exact chunks, top2 mean",
            },
            "trial_grid": trials,
            "selected_query_batch": selected_batch,
            "headroom_requirement": 0.10,
            "errors": errors,
            "preflight_fingerprint": content_hash({"trials": trials, "selected_query_batch": selected_batch}),
        }
        atomic_json(output_dir / "PREFLIGHT.json", report)
        if not errors:
            write_success(output_dir, stage="preflight", fingerprint=report["preflight_fingerprint"], extra={"code_sha256": sha256_file(CODE_FILE), "scorer_contract": SCORER_CONTRACT})
        logger.finish(report["status"], selected_query_batch=selected_batch)
        return report
    except Exception as exc:
        logger.finish("ERROR", error=str(exc))
        raise


def _toy_corpus() -> tuple[CorpusIndex, torch.Tensor, torch.Tensor, list[list[int]]]:
    set_seed(SEED)
    dimension = 16
    chunk_doc_ids = ["d0", "d0", "d1", "d1", "d1", "d2", "d3", "d3", "d4", "d5"]
    documents = torch.randn(len(chunk_doc_ids), dimension, dtype=torch.float32)
    documents = torch.nn.functional.normalize(documents, p=2, dim=-1)
    queries = torch.randn(4, dimension, dtype=torch.float32)
    queries = torch.nn.functional.normalize(queries, p=2, dim=-1)
    gold = [[0], [1], [2], [3]]
    index = CorpusIndex.from_chunk_doc_ids(chunk_doc_ids)
    return index, documents, queries, gold


def _toy_train(
    arm_config: Mapping[str, Any],
    *,
    steps: int,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> tuple[ResidualProjection, list[float]]:
    index, documents, queries, gold = _toy_corpus()
    dimension = documents.shape[1]
    set_seed(SEED)
    model = ResidualProjection(dimension=dimension, rank=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    start = 0
    history: list[float] = []
    if resume and checkpoint_path and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
        history = list(state["history"])
        _restore_rng_state(state["rng"])
    model.train()
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        projected = model(queries)
        scores = compute_parent_scores(projected, documents, index, chunk_block_size=5)
        losses = []
        for row, indices in enumerate(gold):
            value, _terms = arm_loss(
                scores[row],
                indices,
                arm=str(arm_config["arm"]),
                alpha=arm_config.get("alpha"),
                lambda_top5=float(arm_config.get("lambda_top5") or 0.0),
            )
            losses.append(value)
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"toy loss became non-finite for {arm_config}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
        if checkpoint_path and step + 1 == steps // 2:
            atomic_torch(
                checkpoint_path,
                {
                    "schema_version": SCHEMA,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "history": history,
                    "rng": _rng_state(),
                },
            )
    return model, history


def run_smoke() -> dict[str, Any]:
    """Gate-3 smoke: toy resume checks plus real full-corpus GPU passes."""
    output_dir = RESULTS_DIR / "smoke"
    logger = RunLog("smoke")
    errors: list[str] = []
    arms = [
        {"arm": "A_decoupled", "alpha": None, "lambda_top5": 0.0},
        {"arm": "B_softtop5", "alpha": 2.0, "lambda_top5": 0.0},
        {"arm": "C_hybrid", "alpha": 2.0, "lambda_top5": 0.25},
    ]
    results: list[dict[str, Any]] = []
    try:
        index, documents, queries, _gold = _toy_corpus()
        batch_scores = compute_parent_scores(queries, documents, index, chunk_block_size=5)
        reference = torch.stack([reference_parent_scores(query, documents, index) for query in queries])
        if not torch.allclose(batch_scores, reference, atol=1e-6, rtol=1e-6):
            errors.append("toy blockwise parent scorer disagrees with reference scorer")
        for arm_config in arms:
            logger.update(state="RUNNING", arm=parameter_label(arm_config), steps=50)
            checkpoint = output_dir / f"{parameter_label(arm_config)}.checkpoint.pt"
            uninterrupted, full_history = _toy_train(arm_config, steps=50)
            resumed, resumed_history = _toy_train(arm_config, steps=25, checkpoint_path=checkpoint)
            resumed, resumed_history = _toy_train(arm_config, steps=50, checkpoint_path=checkpoint, resume=True)
            max_parameter_difference = max(
                float((uninterrupted.state_dict()[key] - resumed.state_dict()[key]).abs().max())
                for key in uninterrupted.state_dict()
            )
            if max_parameter_difference > 1e-10:
                errors.append(f"checkpoint/resume mismatch for {parameter_label(arm_config)}")
            if not full_history or full_history[-1] > full_history[0] + 1e-7:
                errors.append(f"toy loss did not decrease for {parameter_label(arm_config)}")
            results.append({
                "config": arm_config,
                "steps": 50,
                "loss_first": full_history[0],
                "loss_last": full_history[-1],
                "resume_loss_last": resumed_history[-1],
                "max_resume_parameter_abs_diff": max_parameter_difference,
            })
        full_corpus_trials: list[dict[str, Any]] = []
        if not torch.cuda.is_available():
            errors.append("CUDA is required for Gate-3 full-corpus forward/backward")
        else:
            real_inputs = load_frozen_inputs(include_bm25=False)
            selected_batch = _load_preflight_batch()
            for arm_name in ARMS:
                logger.update(state="RUNNING", arm=arm_name, phase="full_corpus_forward_backward")
                trial = _preflight_trial(
                    real_inputs,
                    device=torch.device("cuda"),
                    batch_size=selected_batch,
                    arm=arm_name,
                )
                full_corpus_trials.append(trial)
                if trial.get("status") != "PASS" or float(trial.get("headroom_after_peak_ratio") or 0.0) < 0.10:
                    errors.append(f"full-corpus forward/backward did not pass for {arm_name}")
        report = {
            "schema_version": SCHEMA,
            "stage": "smoke",
            "status": "PASS" if not errors else "REJECTED_PREFLIGHT_GATE",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "corpus_mode": "synthetic_resume_fixture_plus_real_full_corpus",
            "actual_full_corpus_forward_backward": bool(full_corpus_trials),
            "full_corpus_trials": full_corpus_trials,
            "throughput_and_eta_recorded": bool(full_corpus_trials),
            "arms": results,
            "errors": errors,
            "smoke_fingerprint": content_hash({"arms": results, "errors": errors}),
        }
        atomic_json(output_dir / "SMOKE.json", report)
        if not errors:
            write_success(output_dir, stage="smoke", fingerprint=report["smoke_fingerprint"], extra={"code_sha256": sha256_file(CODE_FILE), "scorer_contract": SCORER_CONTRACT})
        logger.finish(report["status"], arms=len(results))
        return report
    except Exception as exc:
        logger.finish("ERROR", error=str(exc))
        raise


def _replay_metrics_close(observed: Mapping[str, Any], expected: Mapping[str, Any], tolerance: float = 1e-5) -> tuple[bool, dict[str, float]]:
    differences: dict[str, float] = {}
    for key, expected_value in expected.items():
        if isinstance(expected_value, (int, float)) and key in observed:
            differences[key] = abs(float(observed[key]) - float(expected_value))
    return all(value <= tolerance for value in differences.values()), differences


def replay_exp102() -> dict[str, Any]:
    """Replay EXP-102 using its exact evaluator and compare its frozen report."""
    output_dir = RESULTS_DIR / "replay_exp102"
    logger = RunLog("replay-exp102")
    try:
        require_success(RESULTS_DIR / "input_audit")
        if not EXP102_REPORT.exists():
            raise RuntimeError(f"missing EXP-102 reference report: {EXP102_REPORT}")
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        import exp102_mil_nce_retrieval as exp102

        exp102.set_seed(SEED)
        data = exp102.load_corpus_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.update(state="RUNNING", device=str(device), mode="exact_exp102_evaluator")
        documents = torch.from_numpy(np.asarray(data["chunk_embeddings"], dtype=np.float32)).to(device)
        # Match EXP-102's evaluator exactly: its persisted FP16 chunks must be
        # re-normalised after the FP32 dequantisation path.
        documents = documents / torch.norm(documents, dim=-1, keepdim=True).clamp_min(1e-12)
        models: dict[str, nn.Module] = {}
        for fold in sorted(data["folds"]):
            checkpoint = EXP102_CACHE / f"{fold}.pt"
            if not checkpoint.exists():
                raise RuntimeError(f"missing EXP-102 checkpoint: {checkpoint}")
            model = exp102.ResidualProjection(dimension=DIMENSION, rank=RANK).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
            model.eval()
            models[fold] = model

        dense_rankings: dict[str, list[str]] = {}
        rrf_rankings: dict[str, list[str]] = {}
        per_fold: dict[str, Any] = {}
        for fold, heldout in sorted(data["folds"].items()):
            heldout = [str(qid) for qid in heldout]
            dense_metrics, dense_predictions = exp102.evaluate_model(
                data, heldout, models[fold], documents, device, rrf_mode=None, max_cand=150
            )
            rrf_metrics, rrf_predictions = exp102.evaluate_model(
                data,
                heldout,
                models[fold],
                documents,
                device,
                rrf_mode="static",
                static_alpha=0.65,
                rrf_k=32,
                max_cand=150,
            )
            dense_rankings.update(dense_predictions)
            rrf_rankings.update(rrf_predictions)
            per_fold[fold] = {"dense": dense_metrics, "rrf": rrf_metrics}
        all_evaluable = [qid for qid in sorted(data["train_data"]) if qid not in data["non_eval_qids"]]
        observed = {
            "dense": exp102.compute_metrics(dense_rankings, data["train_data"], all_evaluable),
            "rrf": exp102.compute_metrics(rrf_rankings, data["train_data"], all_evaluable),
        }
        expected = load_json(EXP102_REPORT)
        dense_ok, dense_diff = _replay_metrics_close(observed["dense"], expected["metrics_dense"])
        rrf_ok, rrf_diff = _replay_metrics_close(observed["rrf"], expected["metrics_rrf"])
        inputs = load_frozen_inputs(include_bm25=False)
        fixture = real_parent_numpy_fixture(inputs, device=device)
        status = "PASS" if dense_ok and rrf_ok and fixture["status"] == "PASS" else "REJECTED_REPRODUCTION_GATE"
        tolerance = 1e-5
        dense_failures = {key: value for key, value in dense_diff.items() if value > tolerance}
        rrf_failures = {key: value for key, value in rrf_diff.items() if value > tolerance}
        report = {
            "schema_version": SCHEMA,
            "stage": "replay_exp102",
            "status": status,
            "evaluator": "src/exp102_mil_nce_retrieval.py",
            "checkpoint_namespace": relative_path(EXP102_CACHE),
            "observed": observed,
            "reference": {"dense": expected["metrics_dense"], "rrf": expected["metrics_rrf"]},
            "max_abs_difference": {"dense": dense_diff, "rrf": rrf_diff},
            "reproduction_gate": {
                "deterministic_environment_tolerance": 1e-6,
                "maximum_explained_dtype_path_tolerance": tolerance,
                "dense_failures_above_maximum": dense_failures,
                "rrf_failures_above_maximum": rrf_failures,
                "assessment": (
                    "FAIL: at least one reproduced metric exceeds the plan maximum tolerance."
                    if dense_failures or rrf_failures
                    else "PASS"
                ),
            },
            "per_fold": per_fold,
            "independent_numpy_real_parent_fixture": fixture,
            "replay_fingerprint": content_hash({"observed": observed, "reference": expected["metrics_dense"], "rrf": expected["metrics_rrf"]}),
        }
        atomic_json(output_dir / "REPLAY_EXP102.json", report)
        if status == "PASS":
            write_success(output_dir, stage="replay-exp102", fingerprint=report["replay_fingerprint"], extra={"code_sha256": sha256_file(CODE_FILE), "scorer_contract": SCORER_CONTRACT, "fixture_hash": fixture["fixture_hash"]})
        logger.finish(status)
        return report
    except Exception as exc:
        logger.finish("ERROR", error=str(exc))
        raise


def nested_partitions(folds: Mapping[str, Sequence[str]], outer: str) -> list[dict[str, Any]]:
    names = sorted(str(name) for name in folds if str(name) != str(outer))
    if len(names) != 4:
        raise ValueError("strict nested CV requires exactly four inner rotations")
    result = []
    for validation_fold in names:
        validation = [str(qid) for qid in folds[validation_fold]]
        training = [
            str(qid)
            for name in names
            if name != validation_fold
            for qid in folds[name]
        ]
        result.append({
            "validation_fold": validation_fold,
            "train_qids": training,
            "validation_qids": validation,
        })
    return result


def _nested_gate() -> dict[str, Any]:
    required = {
        "input_audit": RESULTS_DIR / "input_audit",
        "math": RESULTS_DIR / "math",
        "replay_exp102": RESULTS_DIR / "replay_exp102",
        "preflight": RESULTS_DIR / "preflight",
        "smoke": RESULTS_DIR / "smoke",
    }
    markers = {}
    errors = []
    code_sha = sha256_file(CODE_FILE)
    for name, directory in required.items():
        try:
            markers[name] = require_success(directory)
            if markers[name].get("scorer_contract") != SCORER_CONTRACT:
                errors.append(f"scorer contract mismatch for {directory / '_SUCCESS.json'}; rerun prerequisite")
        except RuntimeError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("nested training is fail-closed; prerequisite gates failed: " + " | ".join(errors))
    return markers


def fold0_screen_gate(primary: Mapping[str, Any], arm_a: Mapping[str, Any], dense: Mapping[str, Any], arm_a_dense: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the predeclared Gate-4 resource screen; no threshold is tuned."""
    primary_multi = float(primary.get("cardinality", {}).get("multi_gold", {}).get("recall@5", 0.0))
    arm_a_multi = float(arm_a.get("cardinality", {}).get("multi_gold", {}).get("recall@5", 0.0))
    values = {
        "matched_control_dense_delta_recall@5": float(dense["recall@5"] - arm_a_dense["recall@5"]),
        "matched_control_nested_tuned_rrf_delta_recall@5": float(primary["recall@5"] - arm_a["recall@5"]),
        "multi_gold_delta_recall@5": primary_multi - arm_a_multi,
        "recall@50_delta": float(primary["recall@50"] - arm_a["recall@50"]),
        "mrr@5_delta": float(primary["mrr@5"] - arm_a["mrr@5"]),
    }
    criteria = {
        "dense_delta_at_least_0.003": values["matched_control_dense_delta_recall@5"] >= 0.003,
        "rrf_delta_at_least_0.002": values["matched_control_nested_tuned_rrf_delta_recall@5"] >= 0.002,
        "multi_gold_non_decrease": values["multi_gold_delta_recall@5"] >= 0.0,
        "recall50_drop_at_most_0.001": values["recall@50_delta"] >= -0.001,
        "mrr5_drop_at_most_0.003": values["mrr@5_delta"] >= -0.003,
    }
    return {"status": "PASS" if all(criteria.values()) else "REJECTED_FOLD0_SCREEN", "values": values, "criteria": criteria}


def _rank6_to_top5_movement(
    baseline: Mapping[str, Sequence[str]], treatment: Mapping[str, Sequence[str]],
    answers: Mapping[str, set[str]], qids: Sequence[str],
) -> dict[str, int]:
    promoted = demoted = 0
    for qid in qids:
        before = {str(doc): rank for rank, doc in enumerate(baseline.get(qid, []), start=1)}
        after = {str(doc): rank for rank, doc in enumerate(treatment.get(qid, []), start=1)}
        for gold in answers.get(qid, set()):
            if 6 <= before.get(gold, 10**9) <= 10 and after.get(gold, 10**9) <= 5:
                promoted += 1
            if before.get(gold, 10**9) <= 5 and 6 <= after.get(gold, 10**9) <= 10:
                demoted += 1
    return {"gold_rank6_10_to_top5": promoted, "gold_top5_to_rank6_10": demoted, "net": promoted - demoted}


def run_pilot_screen(*, resume: bool = True) -> dict[str, Any]:
    """Cheap Fold-0/inner-fold-1 evidence gate before the 68-run grid."""
    _nested_gate()
    inputs = load_frozen_inputs(include_bm25=True)
    split = next(item for item in nested_partitions(inputs.folds, "fold_0") if item["validation_fold"] == "fold_1")
    query_batch = _load_preflight_batch()
    output_dir = RESULTS_DIR / "pilot_screen" / "outer_fold_0" / "inner_fold_1"
    cache_base = NAMESPACE_CACHE / "nested_screen" / "outer_fold_0" / "inner_0_fold_1"
    logger = RunLog("pilot-screen")
    results: dict[str, dict[str, Any]] = {}
    try:
        for config in PILOT_CONFIGS:
            label = parameter_label(config)
            logger.update(state="RUNNING", outer="fold_0", inner="fold_1", arm=label, phase="pilot_train")
            model, training = train_model(
                inputs, split["train_qids"], config, output_dir=cache_base / label,
                device=torch.device("cuda"), resume=resume, query_batch=query_batch, epochs=4,
                input_fingerprint=load_json(RESULTS_DIR / "input_audit" / "READING_AUDIT.json")["audit_fingerprint"],
                allow_orchestration_only_resume=(label == "A_decoupled"),
            )
            scored = score_and_fuse(model, split["validation_qids"], inputs, device=torch.device("cuda"),
                                    fusion_configs=[{"dense_weight": 1.0, "rrf_k": 60}])
            dense = scored[fusion_label({"dense_weight": 1.0, "rrf_k": 60})]
            results[label] = {
                "config": dict(config), "training": training, "predictions": dense["predictions"],
                "metrics": aggregate_metrics(dense["predictions"], inputs.answers, split["validation_qids"],
                                             dense_scores=dense["dense_scores"], outer_train_qids=split["train_qids"]),
            }
            del model
            torch.cuda.empty_cache()
        baseline = results["A_decoupled"]
        candidates = []
        for label, result in results.items():
            if label == "A_decoupled":
                continue
            metric, base = result["metrics"], baseline["metrics"]
            movement = _rank6_to_top5_movement(baseline["predictions"], result["predictions"], inputs.answers, split["validation_qids"])
            multi_delta = float(metric["cardinality"]["multi_gold"]["recall@5"] - base["cardinality"]["multi_gold"]["recall@5"])
            values = {"recall@5_delta": float(metric["recall@5"] - base["recall@5"]), "multi_gold_recall@5_delta": multi_delta,
                      "recall@16_delta": float(metric["recall@16"] - base["recall@16"]), "recall@50_delta": float(metric["recall@50"] - base["recall@50"]), **movement}
            criteria = {"recall5_gain_at_least_0.5pp": values["recall@5_delta"] >= 0.005,
                        "multi_gold_non_decrease": multi_delta >= 0.0,
                        "recall16_drop_at_most_0.1pp": values["recall@16_delta"] >= -0.001,
                        "recall50_drop_at_most_0.1pp": values["recall@50_delta"] >= -0.001,
                        "at_least_three_net_rank6_10_gold_promotions": movement["net"] >= 3}
            candidates.append({"label": label, "config": result["config"], "values": values, "criteria": criteria, "passes": all(criteria.values())})
        status = "PASS_PILOT_GATE" if any(row["passes"] for row in candidates) else "REJECTED_PILOT_GATE"
        report = {"schema_version": SCHEMA, "stage": "pilot_screen", "status": status, "outer_fold": "fold_0", "inner_validation_fold": "fold_1",
                  "baseline": {"label": "A_decoupled", "metrics": baseline["metrics"], "training": baseline["training"]},
                  "candidates": candidates, "assumptions": {"recall16_50_guardrail": "no drop greater than 0.1pp", "clear_rank_movement": "at least 3 net gold rank-6-to-10 promotions into Top-5"},
                  "pilot_fingerprint": content_hash({"baseline": baseline["metrics"], "candidates": candidates})}
        atomic_json(output_dir / "PILOT_REPORT.json", report)
        if status == "PASS_PILOT_GATE":
            write_success(output_dir, stage="pilot-screen", fingerprint=report["pilot_fingerprint"], extra={"code_sha256": sha256_file(CODE_FILE)})
        logger.finish(status, outer="fold_0", inner="fold_1")
        return report
    except Exception as exc:
        logger.finish("ERROR", error=str(exc))
        raise


def _load_preflight_batch() -> int:
    report = load_json(RESULTS_DIR / "preflight" / "PREFLIGHT.json")
    if report.get("status") != "PASS" or not report.get("selected_query_batch"):
        raise RuntimeError("preflight did not select a safe query batch")
    return int(report["selected_query_batch"])


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _mean_selection_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[canonical_json(record["config"])].append(record)
    output = []
    for key, values in sorted(grouped.items()):
        config = dict(values[0]["config"])
        result: dict[str, Any] = {
            "config": config,
            "inner_rotations": len(values),
            "inner_evaluable_queries": int(sum(int(value.get("evaluable_queries", 0)) for value in values)),
        }
        for metric in ("recall@5", "multi_gold_recall@5", "precision@5", "recall@16", "mrr@5"):
            result[metric] = float(np.mean([float(value.get(metric, 0.0)) for value in values]))
        result["per_inner_rotation"] = [
            {
                "validation_fold": value.get("validation_fold"),
                "recall@5": value.get("recall@5"),
                "multi_gold_recall@5": value.get("multi_gold_recall@5"),
                "precision@5": value.get("precision@5"),
                "recall@16": value.get("recall@16"),
                "mrr@5": value.get("mrr@5"),
            }
            for value in values
        ]
        output.append(result)
    return output


def _arm_part(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arm": str(config["arm"]),
        "alpha": config.get("alpha"),
        "lambda_top5": float(config.get("lambda_top5") or 0.0),
    }


def _fusion_part(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dense_weight": float(config["dense_weight"]),
        "bm25_weight": float(config.get("bm25_weight", 1.0 - float(config["dense_weight"]))),
        "rrf_k": int(config["rrf_k"]),
    }


def run_nested_outer(
    outer: str,
    *,
    screen: bool,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one strict nested outer fold after all manual gates pass."""
    if screen and outer != "fold_0":
        raise ValueError("nested-screen is intentionally limited to --outer fold_0")
    _nested_gate()
    inputs = load_frozen_inputs(include_bm25=True)
    query_batch = _load_preflight_batch()
    base_dir = RESULTS_DIR / ("nested_screen" if screen else "nested_oof") / f"outer_{outer}"
    cache_base = NAMESPACE_CACHE / ("nested_screen" if screen else "nested_oof") / f"outer_{outer}"
    report_path = base_dir / "OUTER_REPORT.json"
    if resume and report_path.exists() and (base_dir / "_SUCCESS.json").exists():
        return load_json(report_path)
    logger = RunLog("nested-screen" if screen else "nested-oof")
    started = time.time()
    outer_train = [
        qid
        for fold, qids in inputs.folds.items()
        if fold != outer
        for qid in qids
    ]
    outer_validation = list(inputs.folds[outer])
    inner_splits = nested_partitions(inputs.folds, outer)
    candidates = candidate_configs()
    all_inner_records: list[dict[str, Any]] = []
    try:
        for inner_number, split in enumerate(inner_splits):
            inner_train = split["train_qids"]
            inner_validation = split["validation_qids"]
            for candidate in candidates:
                label = parameter_label(candidate)
                logger.update(
                    state="RUNNING",
                    outer=outer,
                    inner=split["validation_fold"],
                    arm=label,
                    phase="inner_train",
                )
                candidate_dir = cache_base / f"inner_{inner_number}_{split['validation_fold']}" / label
                model, training_summary = train_model(
                    inputs,
                    inner_train,
                    candidate,
                    output_dir=candidate_dir,
                    device=torch.device("cuda"),
                    resume=resume,
                    query_batch=query_batch,
                    epochs=4,
                    input_fingerprint=load_json(RESULTS_DIR / "input_audit" / "READING_AUDIT.json")["audit_fingerprint"],
                )
                evaluated = evaluate_fusion_grid(
                    model,
                    inner_validation,
                    inputs,
                    device=torch.device("cuda"),
                    configs=fusion_grid(),
                )
                for fusion_record in evaluated["records"]:
                    combined = dict(candidate)
                    combined.update(_fusion_part(fusion_record["config"]))
                    all_inner_records.append({
                        **{key: value for key, value in fusion_record.items() if key != "config"},
                        "config": combined,
                        "validation_fold": split["validation_fold"],
                        "training_summary": training_summary,
                    })
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        averaged = _mean_selection_records(all_inner_records)
        winner = dict(select_candidate(averaged))
        common_candidates = [record for record in averaged if record["config"]["arm"] == "A_decoupled"]
        common_policy = dict(select_candidate(common_candidates))
        # Every arm gets its own inner-selected configuration and one outer
        # retrain.  This is required for the Arm A/B/C and common-policy
        # ablations; selecting only the global winner would make Gate 4's
        # matched-control comparison impossible.
        selected_by_arm = {
            arm: dict(select_candidate([record for record in averaged if record["config"]["arm"] == arm]))
            for arm in ARMS
        }
        common_fusion = _fusion_part(common_policy["config"])
        arm_results: dict[str, dict[str, Any]] = {}
        for arm, selected in selected_by_arm.items():
            arm_config = _arm_part(selected["config"])
            arm_fusion = _fusion_part(selected["config"])
            logger.update(state="RUNNING", outer=outer, arm=parameter_label(arm_config), phase="outer_retrain")
            model, training = train_model(
                inputs, outer_train, arm_config,
                output_dir=cache_base / "outer_retrain" / parameter_label(arm_config),
                device=torch.device("cuda"), resume=resume, query_batch=query_batch, epochs=4,
                input_fingerprint=load_json(RESULTS_DIR / "input_audit" / "READING_AUDIT.json")["audit_fingerprint"],
            )
            fusion_configs = [arm_fusion] + ([] if fusion_label(arm_fusion) == fusion_label(common_fusion) else [common_fusion])
            scored_all = score_and_fuse(model, outer_validation, inputs, device=torch.device("cuda"), fusion_configs=fusion_configs)
            tuned = scored_all[fusion_label(arm_fusion)]
            common = scored_all[fusion_label(common_fusion)]
            arm_results[arm] = {
                "selected": selected,
                "training": training,
                "tuned": tuned,
                "common": common,
                "nested_tuned_metrics": aggregate_metrics(tuned["predictions"], inputs.answers, outer_validation,
                    baseline_predictions=tuned["dense_predictions"], dense_scores=tuned["dense_scores"], outer_train_qids=outer_train),
                "common_policy_metrics": aggregate_metrics(common["predictions"], inputs.answers, outer_validation,
                    baseline_predictions=common["dense_predictions"], dense_scores=common["dense_scores"], outer_train_qids=outer_train),
                "dense_metrics": aggregate_metrics(tuned["dense_predictions"], inputs.answers, outer_validation),
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        winner_arm = str(winner["config"]["arm"])
        winner_result = arm_results[winner_arm]
        winner_scored = winner_result["tuned"]
        primary_metrics = winner_result["nested_tuned_metrics"]
        common_metrics = winner_result["common_policy_metrics"]
        arm_a_result = arm_results["A_decoupled"]
        arm_a_metrics = arm_a_result["nested_tuned_metrics"]
        dense_metrics = winner_result["dense_metrics"]
        arm_a_dense_metrics = arm_a_result["dense_metrics"]
        evaluable_outer = [qid for qid in outer_validation if inputs.answers.get(qid)]
        primary_vs_dense_bootstrap = paired_bootstrap_ci(
            [dense_metrics["per_query"][qid]["recall@5"] for qid in evaluable_outer],
            [primary_metrics["per_query"][qid]["recall@5"] for qid in evaluable_outer],
        )
        common_vs_dense_bootstrap = paired_bootstrap_ci(
            [
                aggregate_metrics(arm_a_result["tuned"]["dense_predictions"], inputs.answers, [qid], _include_slices=False)["recall@5"]
                for qid in evaluable_outer
            ],
            [
                aggregate_metrics(arm_a_result["tuned"]["predictions"], inputs.answers, [qid], _include_slices=False)["recall@5"]
                for qid in evaluable_outer
            ],
        )

        prediction_rows = []
        for qid in outer_validation:
            prediction_rows.append({
                "qid": qid,
                "outer_fold": outer,
                "gold": sorted(inputs.answers[qid]),
                "dense_top150": winner_scored["dense_predictions"].get(qid, []),
                "primary_top150": winner_scored["predictions"].get(qid, []),
                "common_top150": winner_result["common"]["predictions"].get(qid, []),
                "arm_a_top150": arm_a_result["tuned"]["predictions"].get(qid, []),
            })
        _write_jsonl_atomic(base_dir / "outer_predictions.jsonl", prediction_rows)
        screen_gate = fold0_screen_gate(primary_metrics, arm_a_metrics, dense_metrics, arm_a_dense_metrics) if screen else None
        status = str(screen_gate["status"]) if screen_gate is not None else "PASS"
        report = {
            "schema_version": SCHEMA,
            "stage": "nested_screen" if screen else "nested_outer",
            "status": status,
            "outer_fold": outer,
            "outer_train_query_count": len(outer_train),
            "outer_validation_query_count": len(outer_validation),
            "inner_rotations": [
                {
                    "validation_fold": split["validation_fold"],
                    "train_query_count": len(split["train_qids"]),
                    "validation_query_count": len(split["validation_qids"]),
                }
                for split in inner_splits
            ],
            "candidate_grid": {
                "arms": candidate_configs(),
                "fusion": fusion_grid(),
                "selection_order": ["recall@5", "multi_gold_recall@5", "precision@5", "recall@16", "mrr@5"],
                "tie_rule": "within 1e-4: Arm A, then Arm C with smaller lambda, then Arm B",
            },
            "selected_primary": winner,
            "selected_common_arm_a_policy": common_policy,
            "selected_by_arm": selected_by_arm,
            "inner_aggregate_records": averaged,
            "outer_training_by_arm": {arm: result["training"] for arm, result in arm_results.items()},
            "outer_arm_results": {
                arm: {
                    "nested_tuned_metrics": result["nested_tuned_metrics"],
                    "common_policy_metrics": result["common_policy_metrics"],
                    "dense_metrics": result["dense_metrics"],
                }
                for arm, result in arm_results.items()
            },
            "outer_metrics": primary_metrics,
            "matched_arm_a_nested_tuned_metrics": arm_a_metrics,
            "matched_arm_a_dense_metrics": arm_a_dense_metrics,
            "common_policy_metrics": common_metrics,
            "gate4": screen_gate,
            "paired_bootstrap_95ci_vs_dense": {
                "primary": primary_vs_dense_bootstrap,
                "common_policy": common_vs_dense_bootstrap,
            },
            "elapsed_seconds": time.time() - started,
            "outer_fingerprint": content_hash({
                "outer": outer,
                "selected_primary": winner,
                "selected_common": common_policy,
                "primary_metrics": {key: value for key, value in primary_metrics.items() if key != "per_query"},
            }),
        }
        atomic_json(report_path, report)
        if status == "PASS":
            write_success(base_dir, stage="nested-screen" if screen else "nested-outer", fingerprint=report["outer_fingerprint"])
        logger.finish(status, outer=outer)
        return report
    except Exception as exc:
        logger.finish("ERROR", outer=outer, error=str(exc))
        raise


def run_nested_screen(*, outer: str = "fold_0", resume: bool = False) -> dict[str, Any]:
    require_success(RESULTS_DIR / "pilot_screen" / "outer_fold_0" / "inner_fold_1")
    return run_nested_outer(outer, screen=True, resume=resume)


def run_nested_oof(*, resume: bool = False) -> dict[str, Any]:
    _nested_gate()
    screen_dir = RESULTS_DIR / "nested_screen" / "outer_fold_0"
    require_success(screen_dir)
    inputs = load_frozen_inputs(include_bm25=True)
    reports = []
    for outer in sorted(inputs.folds):
        reports.append(run_nested_outer(outer, screen=False, resume=resume))
    primary_predictions: dict[str, list[str]] = {}
    common_predictions: dict[str, list[str]] = {}
    arm_a_predictions: dict[str, list[str]] = {}
    oof_rows = []
    for report in reports:
        outer = report["outer_fold"]
        prediction_path = RESULTS_DIR / "nested_oof" / f"outer_{outer}" / "outer_predictions.jsonl"
        for row in read_jsonl(prediction_path):
            primary_predictions[str(row["qid"])] = row["primary_top150"]
            common_predictions[str(row["qid"])] = row.get("common_top150", row["primary_top150"])
            arm_a_predictions[str(row["qid"])] = row.get("arm_a_top150", row["primary_top150"])
            oof_rows.append(row)
    all_qids = sorted(inputs.qids)
    primary_metrics = aggregate_metrics(primary_predictions, inputs.answers, all_qids)
    common_metrics = aggregate_metrics(common_predictions, inputs.answers, all_qids)
    arm_a_metrics = aggregate_metrics(arm_a_predictions, inputs.answers, all_qids)
    evaluable = [qid for qid in all_qids if inputs.answers.get(qid)]
    per_fold_delta = {
        str(report["outer_fold"]): float(
            report["outer_metrics"]["recall@5"] - report["matched_arm_a_nested_tuned_metrics"]["recall@5"]
        )
        for report in reports
    }
    multi_count = int(arm_a_metrics.get("cardinality", {}).get("multi_gold", {}).get("evaluable_queries", 0))
    multi_delta = float(primary_metrics.get("cardinality", {}).get("multi_gold", {}).get("recall@5", 0.0) - arm_a_metrics.get("cardinality", {}).get("multi_gold", {}).get("recall@5", 0.0))
    paired = paired_bootstrap_ci(
        [arm_a_metrics["per_query"][qid]["recall@5"] for qid in evaluable],
        [primary_metrics["per_query"][qid]["recall@5"] for qid in evaluable],
    )
    deltas = {
        "recall@5": float(primary_metrics["recall@5"] - arm_a_metrics["recall@5"]),
        "multi_gold_recall@5": multi_delta,
        "recall@50": float(primary_metrics["recall@50"] - arm_a_metrics["recall@50"]),
        "recall@150": float(primary_metrics["recall@150"] - arm_a_metrics["recall@150"]),
        "precision@5": float(primary_metrics["precision@5"] - arm_a_metrics["precision@5"]),
        "mrr@5": float(primary_metrics["mrr@5"] - arm_a_metrics["mrr@5"]),
    }
    gate5_criteria = {
        "recall5_delta_at_least_0.005": deltas["recall@5"] >= 0.005,
        "primary_rrf_recall5_at_least_0.9207": float(primary_metrics["recall@5"]) >= 0.9207,
        "multi_gold_requirement": multi_delta >= (0.0 if multi_count < 50 else 0.020),
        "recall50_drop_at_most_0.001": deltas["recall@50"] >= -0.001,
        "recall150_drop_at_most_0.0005": deltas["recall@150"] >= -0.0005,
        "precision5_drop_at_most_0.001": deltas["precision@5"] >= -0.001,
        "mrr5_drop_at_most_0.005": deltas["mrr@5"] >= -0.005,
        "at_least_four_nonnegative_outer_folds": sum(value >= 0.0 for value in per_fold_delta.values()) >= 4,
        "worst_outer_delta_at_least_minus_0.002": min(per_fold_delta.values(), default=-1.0) >= -0.002,
        "paired_ci_no_drop_below_minus_0.001": float(paired["ci95_low"]) >= -0.001,
    }
    status = "PASS_LOSS_GATE" if all(gate5_criteria.values()) else "REJECTED_LOSS_GATE"
    output_dir = RESULTS_DIR / "nested_oof"
    _write_jsonl_atomic(output_dir / "OOF_PREDICTIONS.jsonl", oof_rows)
    report = {
        "schema_version": SCHEMA,
        "stage": "nested_oof",
        "status": status,
        "outer_reports": reports,
        "primary_metrics": primary_metrics,
        "common_policy_metrics": common_metrics,
        "matched_arm_a_metrics": arm_a_metrics,
        "gate5": {"status": status, "deltas": deltas, "per_outer_fold_recall5_delta": per_fold_delta,
                  "paired_bootstrap_95ci": paired, "multi_gold_evaluable_queries": multi_count, "criteria": gate5_criteria},
        "oof_fingerprint": content_hash({
            "outer_fingerprints": [row["outer_fingerprint"] for row in reports],
            "primary_metrics": {key: value for key, value in primary_metrics.items() if key != "per_query"},
        }),
    }
    atomic_json(output_dir / "OOF_REPORT.json", report)
    if status == "PASS_LOSS_GATE":
        write_success(output_dir, stage="nested-oof", fingerprint=report["oof_fingerprint"], extra={"code_sha256": sha256_file(CODE_FILE)})
    return report


def _stage_snapshot(name: str, directory: Path) -> dict[str, Any]:
    marker = directory / "_SUCCESS.json"
    result: dict[str, Any] = {"stage": name, "directory": relative_path(directory), "success": marker.exists()}
    if marker.exists():
        result["marker"] = load_json(marker)
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        result.setdefault("artifacts", []).append(relative_path(path))
        try:
            artifact = load_json(path)
            if isinstance(artifact, Mapping) and artifact.get("status"):
                result["artifact_status"] = artifact["status"]
        except (OSError, json.JSONDecodeError):
            pass
    return result


def write_report() -> dict[str, Any]:
    snapshots = [
        _stage_snapshot("input_audit", RESULTS_DIR / "input_audit"),
        _stage_snapshot("math", RESULTS_DIR / "math"),
        _stage_snapshot("replay_exp102", RESULTS_DIR / "replay_exp102"),
        _stage_snapshot("preflight", RESULTS_DIR / "preflight"),
        _stage_snapshot("smoke", RESULTS_DIR / "smoke"),
        _stage_snapshot("pilot_screen", RESULTS_DIR / "pilot_screen"),
        _stage_snapshot("nested_screen", RESULTS_DIR / "nested_screen"),
        _stage_snapshot("nested_oof", RESULTS_DIR / "nested_oof"),
    ]
    oof_path = RESULTS_DIR / "nested_oof" / "OOF_REPORT.json"
    screen_reports = sorted((RESULTS_DIR / "nested_screen").glob("outer_*/OUTER_REPORT.json")) if (RESULTS_DIR / "nested_screen").exists() else []
    report = {
        "schema_version": SCHEMA,
        "stage": "report",
        "status": "PASS" if oof_path.exists() else "IMPLEMENTED_PENDING_GATES",
        "implementation": {
            "source": relative_path(CODE_FILE),
            "namespace_cache": relative_path(NAMESPACE_CACHE),
            "namespace_results": relative_path(RESULTS_DIR),
            "code_sha256": sha256_file(CODE_FILE),
            "frozen_query_embeddings": relative_path(QUERY_DIR / "train_queries.f32.npy"),
            "frozen_chunk_embeddings": relative_path(E5_DIR / "embeddings.f16.npy"),
            "no_reencoding": True,
            "no_candidate_union_append": True,
            "full_corpus_parent_scoring": True,
            "top2_mean_source_exact": True,
            "early_stopping": False,
        },
        "stages": snapshots,
        "nested_screen_reports": [relative_path(path) for path in screen_reports],
        "nested_oof_report": relative_path(oof_path) if oof_path.exists() else None,
        "next_gate": "Run audit, test-loss, replay-exp102, preflight, and smoke; inspect all reports before nested-screen.",
        "report_fingerprint": content_hash({"snapshots": snapshots, "code_sha256": sha256_file(CODE_FILE)}),
    }
    atomic_json(RESULTS_DIR / "IMPLEMENTATION_REPORT.json", report)
    stage_lines = "\n".join(
        f"| {row['stage']} | {'PASS' if row['success'] else row.get('artifact_status', 'PENDING')} | {row['directory']} |"
        for row in snapshots
    )
    markdown = f"""# EXP-109A implementation report

Generated by src/exp109a_softtop5_retrieval.py report.

Status: **{report['status']}**

The implementation is isolated to the EXP-109A namespace and uses the frozen
VietLegal-E5 query/chunk embeddings and the raw top-4096 BM25 evidence cache.
Training scores all 343,347 chunks, aggregates source-exact parent chunks with
the deterministic top-2 mean, and never appends a candidate union.

| Stage | Status | Directory |
|---|---|---|
{stage_lines}

Nested OOF is deliberately pending until every prerequisite marker is present.
The implementation does not launch nested-screen or nested-oof automatically.

Code SHA-256: {report['implementation']['code_sha256']}
"""
    _write_text_atomic(RESULTS_DIR / "IMPLEMENTATION_REPORT.md", markdown)
    return report


def show_status() -> dict[str, Any]:
    stages = [
        ("input_audit", RESULTS_DIR / "input_audit"),
        ("math", RESULTS_DIR / "math"),
        ("replay_exp102", RESULTS_DIR / "replay_exp102"),
        ("preflight", RESULTS_DIR / "preflight"),
        ("smoke", RESULTS_DIR / "smoke"),
        ("pilot_screen", RESULTS_DIR / "pilot_screen"),
        ("nested_screen", RESULTS_DIR / "nested_screen"),
        ("nested_oof", RESULTS_DIR / "nested_oof"),
    ]
    status = {
        "schema_version": SCHEMA,
        "namespace": relative_path(RESULTS_DIR),
        "stages": [_stage_snapshot(name, directory) for name, directory in stages],
        "run_status": load_json(RESULTS_DIR / "RUN_STATUS.json") if (RESULTS_DIR / "RUN_STATUS.json").exists() else None,
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


def _print_stage_result(report: Mapping[str, Any]) -> None:
    summary = {
        "stage": report.get("stage"),
        "status": report.get("status"),
        "fingerprint": report.get("audit_fingerprint")
        or report.get("math_fingerprint")
        or report.get("replay_fingerprint")
        or report.get("preflight_fingerprint")
        or report.get("smoke_fingerprint")
        or report.get("outer_fingerprint")
        or report.get("oof_fingerprint")
        or report.get("report_fingerprint"),
    }
    print(json.dumps(summary, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "audit",
            "test-loss",
            "replay-exp102",
            "preflight",
            "smoke",
            "pilot-screen",
            "nested-screen",
            "nested-oof",
            "report",
            "status",
        ),
    )
    parser.add_argument("--outer", default="fold_0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.stage == "audit":
            report = audit_inputs()
        elif args.stage == "test-loss":
            report = math_loss_report()
        elif args.stage == "replay-exp102":
            report = replay_exp102()
        elif args.stage == "preflight":
            report = run_preflight()
        elif args.stage == "smoke":
            report = run_smoke()
        elif args.stage == "pilot-screen":
            report = run_pilot_screen(resume=True)
        elif args.stage == "nested-screen":
            report = run_nested_screen(outer=str(args.outer), resume=bool(args.resume))
        elif args.stage == "nested-oof":
            report = run_nested_oof(resume=bool(args.resume))
        elif args.stage == "report":
            report = write_report()
        else:
            show_status()
            return 0
        _print_stage_result(report)
        status = str(report.get("status", "ERROR"))
        return 0 if status in {"PASS", "IMPLEMENTED_PENDING_GATES"} else 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
