"""EXP-109C: candidate-expanded latent-condition late interaction.

This module is intentionally self contained.  It owns the EXP-109C schema,
manifests, reports, checkpoints and authorization barriers; EXP-013 is used
only as a historical reference.  The default code paths are cheap and
fail-closed: no model is loaded, no network access is requested, and no
Fold-0/OOF action is implied by a successful unit test.

The public pure helpers are useful for fixtures and for the later expensive
stages.  Expensive stages are implemented behind explicit authorization flags
and verify the preceding report/success markers before doing work.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

# LightGBM/joblib otherwise probes Windows through a subprocess that is not
# available in the constrained runner.  Pinning the already-known logical
# count is deterministic and avoids a warning being mistaken for a failure.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))

try:  # Optional at import time so audit/tests work without torch.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised on minimal environments.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Paths and immutable contracts
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
E5_QUERY_DIR = ROOT / "cache" / "exp021_e5_dense_candidates" / "query_embeddings"
BM25_RAW_DIR = ROOT / "cache" / "exp021_sparse" / "depth_tune" / "raw4096_evidence"
BM25_TUNING_REPORT = ROOT / "results" / "exp021_sparse" / "depth_rrf_tuning" / "tuning_report.json"
BM25_MANIFEST = ROOT / "cache" / "exp021_sparse" / "passage_hierarchy" / "fts5" / "manifest.json"

EXP109B_CACHE = ROOT / "cache" / "exp109b_encoder_complementarity"
EXP109B_RESULTS = ROOT / "results" / "exp109b_encoder_complementarity"
EXP013_CACHE = ROOT / "cache" / "exp013_slid"
EXP013_RESULTS = ROOT / "results" / "exp013_slid"

NAMESPACE = "exp109c_latent_condition_late_interaction"
CACHE_ROOT = ROOT / "cache" / NAMESPACE
RESULTS_ROOT = ROOT / "results" / NAMESPACE
LOG_ROOT = RESULTS_ROOT / "logs"
MODEL_USE_ELIGIBILITY_PATH = ROOT / "docs" / "exp109c_model_use_eligibility.json"

SCHEMA = "legalir.exp109c_latent_condition_late_interaction.v1"
LABEL_POLICY = "canonical_duplicate_alias_drop_empty_passage_v1"
LABEL_FINGERPRINT = "9bdf9593b61fe3423d1f1a819ac9fb3e8d7225e6003da0afb840c1f5853fd4c9"
STRUCTURAL_FINGERPRINT = "9743fe70ec4092aca28a9855d0236bdedef41d8a458e39d750062b9ef85a37bc"
PROCESSED_CONTENT_FINGERPRINT = "3b7ba47dcfd382f6d670824490777ff98f7f44144a888add664fc1aac6fe5263"
SCORER_CONTRACT = "jina_colbert_v2_128_int8_row_scale_diversity_medoids256_select64_v2"
MODEL_REPO = "jinaai/jina-colbert-v2"
MODEL_SNAPSHOT = "4552c4dc1ffd7d7a635b6a41a1077fe9c9cdd974"
MODEL_PARAMETERS = 559_497_216
DIMENSION = 128
OLD_DIMENSION = 64
RESCUE_DIMENSION = 64
MAX_ANCHORS_PRIMARY = 96
MAX_ANCHORS_RETRY = 128
CONFIG_E_ANCHORS = 256
CONFIG_E_SELECTION_DIMENSION = 64
CONFIG_E_STORAGE_DIMENSION = 128
CONFIG_E_CONTRACT = "config_e_medoids64_select_128_store_score_v1"
CHUNK_COUNT = 343_347
DOCUMENT_COUNT = 8_507
QUERY_COUNT = 7_000
EVALUABLE_QUERY_COUNT = 6_991
FOLD_NAMES = tuple(f"fold_{i}" for i in range(5))
CANDIDATE_DEPTH = 100  # EXP-109C-D100: top-100 from each source, then deduplicate parents.
CANDIDATE_DIAGNOSTIC_DEPTHS = (50, 100, 200)
INDEX_SHARD_SIZE = 256  # bounded-loss checkpoint interval; each receipt is hash-verified
# Score checkpoints are deliberately smaller than index shards: one failed
# late-interaction query batch must not discard hours of exact scoring.
SCORE_QUERY_SHARD_SIZE = 32
SCORER_IMPLEMENTATION_CONTRACT = "d100_batched_chunk_maxsim_masked_fp32_v1"
APPROXIMATE_MICROBATCH = {"max_chunks": 512, "max_document_tokens": 65_536, "max_intermediate_bytes": 192 * 2**20}
EXACT_MICROBATCH = {"max_chunks": 64, "max_document_tokens": 16_384, "max_intermediate_bytes": 192 * 2**20}
FOLD0 = "fold_0"
SEEDS = (109, 110, 111)
FORBIDDEN_FEATURE_TOKENS = ("doc_id", "query_id", "label", "target", "gold", "answer", "nearest")

SCALAR_109B_FEATURES = (
    "e5_score", "e5_rank", "e5_recip", "e5_z", "e5_margin_rank1", "e5_margin_rank5", "e5_margin_rank10", "e5_present",
    "harrier_score", "harrier_rank", "harrier_recip", "harrier_z", "harrier_margin_rank1", "harrier_margin_rank5", "harrier_margin_rank10", "harrier_present",
    "lal_score", "lal_rank", "lal_recip", "lal_z", "lal_margin_rank1", "lal_margin_rank5", "lal_margin_rank10", "lal_present",
    "bm25_score", "bm25_rank", "bm25_recip", "bm25_z", "bm25_margin_rank1", "bm25_margin_rank5", "bm25_margin_rank10", "bm25_present",
    "source_agreement_top5", "source_agreement_top10", "source_agreement_top20", "dense_top1_score", "dense_top2_score", "dense_top1_top2_gap",
    "parent_chunk_count", "parent_token_length", "query_token_length",
)
LATE_FEATURES = (
    "li_chunk_top1_mean", "li_chunk_top2_mean", "li_chunk_top3_mean", "li_chunk_top1_top2_gap",
    "li_two_chunk_union_mean", "li_two_chunk_incremental_gain", "li_full_parent_union_mean", "li_idf_weighted_union_mean",
    "li_token_match_min", "li_token_match_p10", "li_token_match_p25", "li_token_match_median", "li_token_match_mean", "li_token_match_max",
    "li_lower_quartile_mean", "li_selected_chunk_count", "li_selected_token_count", "li_parent_chunk_count", "li_parent_token_count",
    "li_top1_minus_parent_median", "li_union_minus_parent_median", "li_exact_rank_within_candidate_pool",
)
ALLOWED_FEATURES = SCALAR_109B_FEATURES + LATE_FEATURES


class GateRejected(RuntimeError):
    """A planned fail-closed gate rejected the requested stage."""

    def __init__(self, status: str, message: str, *, report: Path | None = None):
        super().__init__(message)
        self.status = status
        self.report = report


@dataclass(frozen=True)
class ChunkMeta:
    chunk_id: str
    doc_id: str
    source_start: int = 0
    source_end: int = 0
    token_start: int = 0
    token_end: int = 0
    original_tokens: int = 0
    retained_tokens: int = 0
    truncated: bool = False
    parent_node_id: str = ""


@dataclass
class ParentIndex:
    """Offset mapping for chunk-bounded parent operations."""

    doc_ids: list[str]
    chunk_doc_ids: list[str]
    positions: dict[str, list[int]]

    @classmethod
    def from_chunk_rows(cls, rows: Iterable[Mapping[str, Any]]) -> "ParentIndex":
        doc_ids: list[str] = []
        positions: dict[str, list[int]] = defaultdict(list)
        chunk_doc_ids: list[str] = []
        for index, row in enumerate(rows):
            doc_id = str(row["doc_id"])
            chunk_doc_ids.append(doc_id)
            positions[doc_id].append(index)
            if doc_id not in positions or len(positions[doc_id]) == 1:
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
        return cls(doc_ids, chunk_doc_ids, dict(positions))

    @classmethod
    def from_chunk_doc_ids(cls, chunk_doc_ids: Sequence[str]) -> "ParentIndex":
        return cls.from_chunk_rows({"doc_id": value} for value in chunk_doc_ids)

    def indices(self, doc_id: str) -> list[int]:
        return self.positions.get(str(doc_id), [])

    @property
    def parent_count(self) -> int:
        return len(self.doc_ids)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_doc_ids)


@dataclass(frozen=True)
class IndexedChunkRef:
    """A lazy reference to one token-vector slice in an index shard."""

    vectors_path: Path
    scales_path: Path
    start: int
    end: int
    metadata: dict[str, Any]
    full_vectors_path: Path | None = None
    full_scales_path: Path | None = None
    full_start: int | None = None
    full_end: int | None = None

    def load(self, cache: "ShardMmapCache | None" = None) -> tuple[np.ndarray, dict[str, Any]]:
        vectors = cache.load(self.vectors_path) if cache is not None else np.load(self.vectors_path, mmap_mode="r")
        scales = cache.load(self.scales_path) if cache is not None else np.load(self.scales_path, mmap_mode="r")
        return dequantize_rows(vectors[self.start:self.end], scales[self.start:self.end]), self.metadata

    def load_full(self, cache: "ShardMmapCache | None" = None) -> tuple[np.ndarray, dict[str, Any]]:
        if self.full_vectors_path is None or self.full_scales_path is None or self.full_start is None or self.full_end is None:
            raise GateRejected("REJECTED_STALE_ARTIFACT", "index shard lacks the required E4 full-token store")
        vectors = cache.load(self.full_vectors_path) if cache is not None else np.load(self.full_vectors_path, mmap_mode="r")
        scales = cache.load(self.full_scales_path) if cache is not None else np.load(self.full_scales_path, mmap_mode="r")
        return dequantize_rows(vectors[self.full_start:self.full_end], scales[self.full_start:self.full_end]), self.metadata


class ShardMmapCache:
    """Per-run mmap handles; vectors remain on disk and are never FP32-cached."""

    def __init__(self) -> None:
        self._arrays: dict[Path, np.ndarray] = {}
        self.opens = 0

    def load(self, path: Path) -> np.ndarray:
        path = path.resolve()
        array = self._arrays.get(path)
        if array is None:
            array = np.load(path, mmap_mode="r")
            self._arrays[path] = array
            self.opens += 1
        return array

    @property
    def open_paths(self) -> int:
        return len(self._arrays)


# ---------------------------------------------------------------------------
# Deterministic IO, hashes, manifests and logging
# ---------------------------------------------------------------------------


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


def code_fingerprint() -> str:
    return sha256_file(CODE_FILE)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
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


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
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


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class RunTracker:
    """Flushes a compact status file and detailed run/stage logs."""

    def __init__(self, stage: str, *, outer: str | None = None, total: int = 0):
        self.stage, self.outer, self.total = stage, outer, int(total)
        self.run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        self.run_dir = LOG_ROOT / self.run_id
        self.log_path = self.run_dir / f"{stage}.log"
        self.run_log_path = self.run_dir / "main.log"
        self.status_path = RESULTS_ROOT / "RUN_STATUS.json"
        self.started = time.monotonic()
        self.completed = 0
        self.state = "RUNNING"
        # Materialize the run directory before any expensive setup.  Some
        # stages legitimately do not call heartbeat until their first shard;
        # without this, RUN_STATUS advertised a run_id whose log directory did
        # not exist and users could not follow a live long-running stage.
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path.touch(exist_ok=True)
        self.log_path.touch(exist_ok=True)
        self.update()
        self.log(f"stage={self.stage} state=RUNNING total={self.total}", emit=True)

    def update(self, **values: Any) -> None:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        throughput = self.completed / elapsed
        payload = {
            "run_id": self.run_id, "state": self.state, "stage": self.stage, "outer": self.outer,
            "completed": self.completed, "total": self.total, "throughput": throughput,
            "eta_seconds": max(0.0, (self.total - self.completed) / max(throughput, 1e-9)) if self.completed else 0.0,
            "private_ram_bytes": private_ram_bytes(), "available_ram_bytes": available_ram_bytes(),
            "vram_bytes": vram_bytes(), "last_heartbeat": utc_now(), "code_fingerprint": code_fingerprint(),
        }
        payload.update(values)
        atomic_json(self.status_path, payload)

    def log(self, message: str, *, emit: bool = False) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        line = f"[{utc_now()}] {message}"
        for path in (self.run_log_path, self.log_path):
            with path.open("a", encoding="utf-8", newline="\n", buffering=1) as handle:
                handle.write(line + "\n")
        if emit:
            print(line, flush=True)

    def heartbeat(self, completed: int | None = None, *, emit: bool = True, **values: Any) -> None:
        if completed is not None:
            self.completed = int(completed)
        self.update(**values)
        self.log(f"stage={self.stage} progress={self.completed}/{self.total}", emit=emit)

    def finish(self, state: str, **values: Any) -> None:
        self.state = state
        self.update(**values)
        self.log(f"stage={self.stage} state={state}", emit=True)


@contextlib.contextmanager
def tracked_stage(stage: str, *, outer: str | None = None, total: int = 0, tracker: RunTracker | None = None) -> Iterator[RunTracker]:
    tracker = tracker or RunTracker(stage, outer=outer, total=total)
    try:
        yield tracker
    except KeyboardInterrupt:
        tracker.finish("INTERRUPTED")
        raise
    except Exception:
        tracker.finish("FAILED", error=traceback.format_exc(limit=5))
        raise
    else:
        # A gate may subsequently reject the stage, but never leave its last
        # heartbeat claiming that an already returned worker is still running.
        tracker.finish("PASS")


def private_ram_bytes() -> int:
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD), ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return int(counters.working_set)
    except Exception:
        return 0


def available_ram_bytes() -> int:
    try:
        import ctypes
        class State(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("memory_load", ctypes.c_uint32), ("total_phys", ctypes.c_uint64), ("avail_phys", ctypes.c_uint64), ("total_page", ctypes.c_uint64), ("avail_page", ctypes.c_uint64), ("total_virtual", ctypes.c_uint64), ("avail_virtual", ctypes.c_uint64), ("avail_extended", ctypes.c_uint64)]
        state = State(); state.length = ctypes.sizeof(State)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))
        return int(state.avail_phys)
    except Exception:
        return 0


def vram_bytes() -> int:
    if torch is None or not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.max_memory_reserved())
    except Exception:
        return 0


def write_success(directory: Path, *, stage: str, fingerprint: str, extra: Mapping[str, Any] | None = None) -> None:
    payload = {"schema_version": SCHEMA, "stage": stage, "status": "PASS", "content_fingerprint": fingerprint, "code_sha256": code_fingerprint(), "finished_at": utc_now()}
    if extra:
        payload.update(dict(extra))
    atomic_json(directory / "_SUCCESS.json", payload)


def require_success(directory: Path, expected_fingerprint: str | None = None, *, expected_code_sha256: str | None = None) -> dict[str, Any]:
    path = directory / "_SUCCESS.json"
    if not path.exists():
        raise GateRejected("REJECTED_PREREQUISITE_GATE", f"Missing success marker: {path}", report=path)
    marker = read_json(path)
    if marker.get("status", "PASS") != "PASS":
        raise GateRejected("REJECTED_PREREQUISITE_GATE", f"Success marker is not PASS: {path}", report=path)
    if expected_fingerprint is not None and marker.get("content_fingerprint") != expected_fingerprint:
        raise GateRejected("REJECTED_STALE_ARTIFACT", f"Fingerprint mismatch: {path}", report=path)
    if expected_code_sha256 is not None and marker.get("code_sha256") not in {None, expected_code_sha256}:
        raise GateRejected("REJECTED_STALE_ARTIFACT", f"Code fingerprint mismatch: {path}", report=path)
    return marker


def write_report(path: Path, report: Mapping[str, Any], *, stage: str, success: bool = False) -> dict[str, Any]:
    payload = dict(report)
    payload.setdefault("schema_version", SCHEMA)
    payload.setdefault("stage", stage)
    payload.setdefault("code_sha256", code_fingerprint())
    payload.setdefault("warnings", [])
    payload.setdefault("claim_boundary", "implementation/audit artifact; not a public or test-generalization result")
    payload["report_fingerprint"] = content_hash({k: v for k, v in payload.items() if k != "report_fingerprint"})
    atomic_json(path, payload)
    if success:
        write_success(path.parent, stage=stage, fingerprint=payload["report_fingerprint"])
    return payload


def require_authorization(stage: str, authorize: bool, env_name: str) -> None:
    value = os.environ.get(env_name, "").lower()
    if not authorize and value not in {"1", "true", "yes"}:
        raise GateRejected("REJECTED_AUTHORIZATION_GATE", f"{stage} requires explicit authorization via --authorize or {env_name}=1")


def require_report_status(path: Path, statuses: Iterable[str], *, failure_status: str = "REJECTED_PREREQUISITE_GATE") -> dict[str, Any]:
    if not path.exists():
        raise GateRejected(failure_status, f"required report is missing: {path}", report=path)
    report = read_json(path)
    if report.get("status") not in set(statuses):
        raise GateRejected(failure_status, f"required report is not in an accepted state: {path} ({report.get('status')})", report=path)
    return report


def expected_candidate_input_fingerprints(*, outer: str = FOLD0) -> dict[str, str]:
    return {
        "vietlegal_e5": str(read_json(EXP109B_CACHE / "rankings" / "vietlegal_e5" / outer / "manifest.json").get("content_fingerprint", "")),
        "vnlegal_lal": str(read_json(EXP109B_CACHE / "rankings" / "vnlegal_lal" / outer / "manifest.json").get("content_fingerprint", "")),
        "bm25_raw_manifest": sha256_file(BM25_RAW_DIR / "manifest.json"),
        "bm25_tuning_report": sha256_file(BM25_TUNING_REPORT),
    }


def require_current_candidate_report(*, outer: str = FOLD0) -> dict[str, Any]:
    path = RESULTS_ROOT / "CANDIDATE_CEILING_REPORT.json"
    report = require_report_status(path, ("PASS_CANDIDATE_CEILING", "PASS_CANDIDATE_CEILING_0995"))
    if report.get("outer") != outer or report.get("candidate_contract", {}).get("primary_depth") != CANDIDATE_DEPTH or report.get("input_fingerprints") != expected_candidate_input_fingerprints(outer=outer):
        raise GateRejected("REJECTED_STALE_ARTIFACT", "candidate ceiling report was not produced from current verified source manifests", report=path)
    return report


# ---------------------------------------------------------------------------
# Canonical labels, folds and structural data
# ---------------------------------------------------------------------------


def load_train(path: Path = TRAIN_PATH) -> dict[str, dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("train.json must be an object keyed by query ID")
    return {str(qid): dict(row) for qid, row in value.items()}


def canonical_labels(*, train_path: Path = TRAIN_PATH, exclusions_path: Path = EXCLUSIONS_PATH, impact_path: Path = LABEL_IMPACT_PATH) -> tuple[dict[str, set[str]], dict[str, Any]]:
    train = load_train(train_path)
    exclusions = {str(row["doc_id"]): row for row in read_json(exclusions_path)}
    impacts = {str(row["query_id"]): row for row in read_jsonl(impact_path)}
    answers: dict[str, set[str]] = {}
    observed: dict[str, set[str]] = {}
    duplicate_occurrences = empty_occurrences = 0
    for qid, row in train.items():
        gold: set[str] = set(); removed: set[str] = set()
        for raw_doc in row.get("answer", []):
            doc_id = str(raw_doc); exclusion = exclusions.get(doc_id)
            if exclusion is None:
                gold.add(doc_id); continue
            removed.add(doc_id); reasons = set(map(str, exclusion.get("reasons", [])))
            replacement = exclusion.get("duplicate_retained_id")
            if reasons == {"exact_duplicate_raw_passage"}:
                if not replacement or str(replacement) in exclusions:
                    raise ValueError(f"invalid duplicate alias for {qid}/{doc_id}")
                gold.add(str(replacement)); duplicate_occurrences += 1
            elif reasons == {"empty_passage"}:
                empty_occurrences += 1
            else:
                raise ValueError(f"unsupported exclusion policy for {qid}/{doc_id}: {reasons}")
        answers[qid] = gold
        if removed:
            observed[qid] = removed
    declared = {qid: set(map(str, row.get("intentionally_excluded_gold_ids", []))) for qid, row in impacts.items()}
    if observed != declared:
        raise ValueError("train_label_impact.jsonl does not match canonical exclusions")
    non_evaluable = sorted(qid for qid, gold in answers.items() if not gold)
    stats = {
        "policy": LABEL_POLICY, "queries": len(answers), "evaluable_queries": len(answers) - len(non_evaluable),
        "non_evaluable_queries": len(non_evaluable), "non_evaluable_qids": non_evaluable,
        "affected_queries": len(observed), "canonicalized_duplicate_occurrences": duplicate_occurrences,
        "dropped_empty_occurrences": empty_occurrences,
        "label_fingerprint": content_hash({qid: sorted(gold) for qid, gold in sorted(answers.items())}),
    }
    return answers, stats


def load_folds(*, train_path: Path = TRAIN_PATH, folds_path: Path = FOLDS_PATH) -> tuple[dict[str, list[str]], dict[str, str]]:
    train = load_train(train_path); raw = read_json(folds_path)
    folds = {str(name): [str(qid) for qid in values] for name, values in raw.items()}
    fold_for: dict[str, str] = {}
    for name, qids in folds.items():
        for qid in qids:
            if qid in fold_for:
                raise ValueError(f"query belongs to multiple folds: {qid}")
            fold_for[qid] = name
    if set(fold_for) != set(train) or sum(map(len, folds.values())) != len(fold_for):
        raise ValueError("cv_folds.json is not an exactly-one-fold partition")
    return folds, fold_for


def nested_partitions(folds: Mapping[str, Sequence[str]], outer: str) -> list[dict[str, list[str]]]:
    if outer not in folds:
        raise KeyError(outer)
    inner_names = [name for name in sorted(folds) if name != outer]
    result = []
    for validation in inner_names:
        train_qids = sorted(qid for name in inner_names if name != validation for qid in folds[name])
        result.append({"validation_fold": validation, "train_qids": train_qids, "validation_qids": sorted(map(str, folds[validation])), "outer_excluded": outer})
    return result


def iter_chunks(path: Path = STRUCT_DIR / "chunks.jsonl") -> Iterator[dict[str, Any]]:
    return read_jsonl(path)


def load_parent_metadata(path: Path = STRUCT_DIR / "chunks.jsonl", documents_path: Path = STRUCT_DIR / "documents.jsonl") -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(documents_path):
        metadata[str(row["doc_id"])] = {"parse_mode": str(row.get("parse_mode", "")), "parent_chunk_count": 0, "parent_token_length": 0}
    for row in read_jsonl(path):
        doc_id = str(row["doc_id"]); item = metadata.setdefault(doc_id, {"parse_mode": "", "parent_chunk_count": 0, "parent_token_length": 0})
        item["parent_chunk_count"] += 1; item["parent_token_length"] += int(row.get("token_count", 0))
    return metadata


def read_chunk_meta(row: Mapping[str, Any]) -> ChunkMeta:
    return ChunkMeta(str(row["chunk_id"]), str(row["doc_id"]), int(row.get("start", 0)), int(row.get("end", 0)), int(row.get("token_start", 0)), int(row.get("token_end", 0)), int(row.get("original_tokens", row.get("token_count", 0))), int(row.get("retained_tokens", row.get("token_count", 0))), bool(row.get("truncated", False)), str(row.get("parent_node_id", "")))


def expected_contract_errors() -> list[str]:
    errors: list[str] = []
    try:
        _answers, stats = canonical_labels()
        if stats["queries"] != QUERY_COUNT or stats["evaluable_queries"] != EVALUABLE_QUERY_COUNT or stats["non_evaluable_queries"] != QUERY_COUNT - EVALUABLE_QUERY_COUNT:
            errors.append(f"canonical label count mismatch: {stats}")
        if stats["label_fingerprint"] != LABEL_FINGERPRINT:
            errors.append(f"canonical label fingerprint mismatch: {stats['label_fingerprint']}")
    except Exception as exc:
        errors.append(f"label audit failed: {exc}")
    try:
        folds, _ = load_folds()
        if set(folds) != set(FOLD_NAMES) or any(len(values) != 1_400 for values in folds.values()):
            errors.append(f"fold contract mismatch: { {key: len(value) for key, value in folds.items()} }")
    except Exception as exc:
        errors.append(f"fold audit failed: {exc}")
    for path in (PROCESSED_MANIFEST, STRUCT_MANIFEST, E5_MANIFEST, E5_QUERY_DIR / "manifest.json", BM25_RAW_DIR / "manifest.json", BM25_TUNING_REPORT):
        if not path.exists():
            errors.append(f"missing live input: {path}")
    if STRUCT_MANIFEST.exists():
        structural = read_json(STRUCT_MANIFEST)
        if structural.get("schema_version") != "legalir.structural_chunks.v3":
            errors.append("structural schema mismatch")
        if structural.get("content_fingerprint") != STRUCTURAL_FINGERPRINT:
            errors.append("structural content fingerprint mismatch")
        counts = structural.get("counts", {})
        if counts.get("documents") != DOCUMENT_COUNT or counts.get("chunks") != CHUNK_COUNT:
            errors.append(f"structural count mismatch: {counts}")
    if E5_MANIFEST.exists():
        e5 = read_json(E5_MANIFEST)
        if e5.get("corpus_fingerprint") != STRUCTURAL_FINGERPRINT or e5.get("chunks") != CHUNK_COUNT or e5.get("dimension") != 1024:
            errors.append("E5 structural/dimension contract mismatch")
    for source in ("vietlegal_e5", "vnlegal_lal"):
        directory = EXP109B_CACHE / "rankings" / source / FOLD0
        try:
            manifest = read_json(directory / "manifest.json")
            require_success(directory, manifest.get("content_fingerprint"))
            if manifest.get("schema_version") != "legalir.exp109b_encoder_complementarity.v1" or int(manifest.get("query_count", -1)) != QUERY_COUNT or int(manifest.get("limit", 0)) < CANDIDATE_DEPTH:
                errors.append(f"{source} ranking manifest contract mismatch")
        except Exception as exc:
            errors.append(f"{source} ranking audit failed: {exc}")
    return errors


# ---------------------------------------------------------------------------
# Phase 0 reading audit
# ---------------------------------------------------------------------------


def file_record(path: Path, role: str, *, hash_file: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path.resolve()), "exists": path.exists(), "role": role, "contract_or_finding": role}
    if path.exists() and path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256_file(path) if hash_file else None})
        try:
            value = read_json(path)
            if isinstance(value, dict):
                record["schema_version"] = value.get("schema_version")
            elif isinstance(value, list):
                record["schema_version"] = "json_list"
        except Exception:
            record["schema_version"] = None
    return record


def reading_order_paths() -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = [
        (ROOT / "AGENTS.md", "repo policy"), (TRAIN_PATH, "canonical train"), (FOLDS_PATH, "fixed folds"),
        (PROCESSED_MANIFEST, "processed manifest"), (EXCLUSIONS_PATH, "duplicate/exclusion mapping"), (LABEL_IMPACT_PATH, "label impact"),
        (STRUCT_MANIFEST, "structural manifest"), (STRUCT_DIR / "chunks.jsonl", "structural chunks schema"), (STRUCT_DIR / "documents.jsonl", "structural documents schema"),
        (STRUCT_DIR / "nodes.jsonl", "structural nodes schema"), (STRUCT_DIR / "doc_to_chunk_ids.json", "parent-to-chunk mapping"),
        (ROOT / "docs" / "EXP-109B_PLAN.md", "EXP-109B anchor plan"), (ROOT / "docs" / "exp109b_runbook.md", "EXP-109B runbook"),
        (ROOT / "src" / "exp109b_encoder_complementarity.py", "EXP-109B anchor source"), (ROOT / "tests" / "test_exp109b_encoder_complementarity.py", "EXP-109B anchor tests"),
        (EXP109B_RESULTS / "source_audit" / FOLD0 / "SOURCE_AUDIT.json", "EXP-109B source audit"),
        (EXP109B_RESULTS / "cached_fusion_pilot" / FOLD0 / "CACHED_FUSION_PILOT.json", "EXP-109B inner pilot"),
        (EXP109B_RESULTS / "locked_fusion_fold0" / "FOLD0_LOCKED_FUSION_REPORT.json", "EXP-109B locked anchor"),
        (ROOT / "docs" / "exp013_runbook.md", "EXP-013 runbook"), (ROOT / "src" / "exp013_core.py", "EXP-013 core reference"),
        (ROOT / "src" / "exp013_model.py", "EXP-013 model reference"), (ROOT / "src" / "exp013_late_interaction.py", "EXP-013 late interaction reference"),
        (ROOT / "src" / "exp013_candidates.py", "EXP-013 candidate reference"), (ROOT / "src" / "exp013_ranker.py", "EXP-013 ranker reference"),
        (ROOT / "src" / "exp013_pipeline.py", "EXP-013 pipeline reference"), (ROOT / "tests" / "test_exp013.py", "EXP-013 tests"),
        (EXP013_RESULTS / "candidate_oracle" / "candidate_oracle.json", "old candidate oracle"), (EXP013_CACHE / "colbert_leaves" / "manifest.json", "old token-index manifest"),
        (EXP013_CACHE / "models" / "model_report.json", "old model report"), (EXP013_CACHE / "colbert_leaves" / "run.log", "old encoding log"),
        (ROOT / "docs" / "exp013b_runbook.md", "EXP-013b runbook"), (ROOT / "results" / "exp013b_cascade" / "candidate_audit" / "candidate_audit.json", "EXP-013b candidate audit"),
        (ROOT / "results" / "exp013b_cascade" / "oof" / "oof_report.json", "EXP-013b OOF report"),
        (ROOT / "results" / "exp035_retrieval_error_adjudication" / "REPORT.json", "negative result EXP-035"),
        (ROOT / "results" / "exp036_coverage_aware_fusion" / "REPORT.json", "negative result EXP-036"),
        (ROOT / "results" / "exp037_cached_encoder_complementarity" / "REPORT.json", "negative result EXP-037"),
        (ROOT / "results" / "exp107_listwise_residual_reranker" / "REPORT.json", "negative result EXP-107"),
        (ROOT / "results" / "exp108_atomic_condition_reranker" / "audit-inputs" / "REPORT.json", "negative result EXP-108"),
        (ROOT / "results" / "exp109a_softtop5_retrieval" / "IMPLEMENTATION_REPORT.json", "negative result EXP-109A"),
    ]
    for model_dir, role in ((EXP109B_CACHE / "rankings" / "vietlegal_e5" / FOLD0, "EXP-109B E5 ranking manifest"), (EXP109B_CACHE / "rankings" / "vnlegal_lal" / FOLD0, "EXP-109B LAL ranking manifest")):
        paths.extend(((model_dir / "manifest.json", role), (model_dir / "_SUCCESS.json", role + " success")))
    paths.extend(((E5_MANIFEST, "current E5 cache manifest"), (E5_QUERY_DIR / "manifest.json", "current E5 query manifest"), (BM25_RAW_DIR / "manifest.json", "tuned BM25 raw manifest"), (BM25_MANIFEST, "BM25 FTS manifest")))
    return paths


def local_jina_snapshot() -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]) / "hub" / f"models--{MODEL_REPO.replace('/', '--')}" / "snapshots" / MODEL_SNAPSHOT)
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{MODEL_REPO.replace('/', '--')}" / "snapshots" / MODEL_SNAPSHOT)
    for path in candidates:
        if str(path) and path.exists():
            return path
    return None


def model_snapshot_record() -> dict[str, Any]:
    snapshot = local_jina_snapshot()
    record: dict[str, Any] = {"repo_id": MODEL_REPO, "expected_snapshot": MODEL_SNAPSHOT, "exists": bool(snapshot), "snapshot": str(snapshot) if snapshot else None, "parameters_expected": MODEL_PARAMETERS, "projection_expected": [DIMENSION, 1024]}
    if snapshot:
        config_path = snapshot / "config.json"
        record["files"] = sorted(path.name for path in snapshot.iterdir() if path.is_file())
        if config_path.exists():
            try:
                config = read_json(config_path); record["config"] = {key: config.get(key) for key in ("model_type", "hidden_size", "num_hidden_layers")}
            except Exception as exc:
                record["config_error"] = str(exc)
        tensors = []
        for safetensor in snapshot.glob("*.safetensors"):
            try:
                from safetensors import safe_open
                with safe_open(str(safetensor), framework="pt", device="cpu") as handle:
                    tensors.extend((key, list(handle.get_slice(key).get_shape())) for key in handle.keys() if key == "linear.weight")
            except Exception as exc:
                record.setdefault("warnings", []).append(f"projection inspection unavailable: {exc}")
        record["projection_tensors"] = tensors
        record["snapshot_fingerprint"] = content_hash({"snapshot": MODEL_SNAPSHOT, "files": record.get("files", []), "projection": tensors})
    return record


def runtime_record() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    try:
        from importlib.metadata import version
        for package in ("numpy", "torch", "transformers", "safetensors", "lightgbm"):
            try: packages[package] = version(package)
            except Exception: packages[package] = None
    except Exception:
        packages = {"numpy": np.__version__}
    cuda: dict[str, Any] = {"available": bool(torch is not None and torch.cuda.is_available())}
    if cuda["available"]:
        cuda.update({"device": torch.cuda.get_device_name(0), "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory), "runtime": torch.version.cuda})
    return {"python": sys.version, "platform": platform.platform(), "packages": packages, "private_ram_bytes": private_ram_bytes(), "available_ram_bytes": available_ram_bytes(), "cuda": cuda}


def model_use_eligibility_record() -> dict[str, Any]:
    if not MODEL_USE_ELIGIBILITY_PATH.exists():
        return {"status": "UNVERIFIED_BLOCKS_REAL_RUN", "path": str(MODEL_USE_ELIGIBILITY_PATH), "reason": "competition terms and Jina model license compatibility have not been recorded", "blocks": ["fidelity-pilot", "encode-corpus", "encode-queries", "score-inner", "train-metric-adapter"]}
    value = read_json(MODEL_USE_ELIGIBILITY_PATH)
    required = {"competition_terms_source", "model_license_source", "approved_for_competition_use"}
    if not isinstance(value, Mapping) or not required.issubset(value):
        return {"status": "INVALID_BLOCKS_REAL_RUN", "path": str(MODEL_USE_ELIGIBILITY_PATH), "reason": f"missing required fields: {sorted(required - set(value) if isinstance(value, Mapping) else required)}"}
    return {"status": "APPROVED" if bool(value.get("approved_for_competition_use")) else "NOT_APPROVED_BLOCKS_REAL_RUN", "path": str(MODEL_USE_ELIGIBILITY_PATH), "competition_terms_source": value["competition_terms_source"], "model_license_source": value["model_license_source"], "approved_for_competition_use": bool(value["approved_for_competition_use"]), "notes": value.get("notes")}


def require_model_use_eligibility() -> dict[str, Any]:
    record = model_use_eligibility_record()
    if record.get("status") != "APPROVED":
        raise GateRejected("REJECTED_MODEL_USE_ELIGIBILITY", f"Jina competition-use eligibility is not approved: {record.get('status')}", report=MODEL_USE_ELIGIBILITY_PATH)
    return record


def audit_inputs() -> dict[str, Any]:
    errors = expected_contract_errors(); warnings: list[str] = []; mismatches: list[dict[str, Any]] = []
    try:
        _answers, label_stats = canonical_labels()
    except Exception as exc:
        label_stats = {"error": str(exc)}
    try:
        folds, _ = load_folds(); fold_counts = {name: len(values) for name, values in sorted(folds.items())}
    except Exception as exc:
        fold_counts = {"error": str(exc)}
    inputs = [file_record(path, role, hash_file=True) for path, role in reading_order_paths()]
    for record in inputs:
        if not record["exists"] and record["role"].startswith("negative result"):
            warnings.append(f"historical input absent; retained as explicit warning: {record['path']}")
    old_manifest = EXP013_CACHE / "colbert_leaves" / "manifest.json"
    if old_manifest.exists() and read_json(old_manifest).get("v3_fingerprint") != STRUCTURAL_FINGERPRINT:
        mismatches.append({"input": str(old_manifest), "observed": read_json(old_manifest).get("v3_fingerprint"), "expected": STRUCTURAL_FINGERPRINT, "handling": "reference/reproduction only; reject for current scoring"})
    if not (ROOT / "cache" / "exp037_cached_encoder_complementarity" / "manifest.json").exists():
        warnings.append("EXP-037 manifest is absent; no legacy force-gold fixture is promoted into EXP-109C")
    if not (ROOT / "results" / "exp107_listwise_residual_reranker" / "REPORT.json").exists():
        warnings.append("EXP-107 report path is absent; no EXP-107 evidence is used")
    if not (ROOT / "results" / "exp109a_softtop5_retrieval" / "PILOT_REPORT.json").exists():
        warnings.append("EXP-109A PILOT_REPORT.json is absent; implementation report is retained as negative context")
    candidate_path = RESULTS_ROOT / "CANDIDATE_CEILING_REPORT.json"
    if candidate_path.exists():
        try:
            candidate = read_json(candidate_path)
            if candidate.get("input_fingerprints") != expected_candidate_input_fingerprints(outer=str(candidate.get("outer", FOLD0))):
                warnings.append("existing CANDIDATE_CEILING_REPORT.json lacks current verified input fingerprints and is invalidated for gating")
        except Exception as exc:
            warnings.append(f"existing candidate ceiling report is unreadable and invalidated: {exc}")
    model = model_snapshot_record()
    if not model["exists"]:
        errors.append("required local Jina snapshot is absent; download is forbidden by this implementation")
    elif model.get("projection_tensors") and model["projection_tensors"] != [("linear.weight", [DIMENSION, 1024])]:
        errors.append(f"Jina projection shape mismatch: {model.get('projection_tensors')}")
    processed = read_json(PROCESSED_MANIFEST) if PROCESSED_MANIFEST.exists() else {}
    structural = read_json(STRUCT_MANIFEST) if STRUCT_MANIFEST.exists() else {}
    eligibility = model_use_eligibility_record(); runtime = runtime_record()
    if eligibility["status"] != "APPROVED":
        warnings.append("model-use eligibility is unverified; real Jina scoring/training remains blocked")
    report = {
        "schema_version": SCHEMA, "stage": "input_audit", "status": "PASS" if not errors else "REJECTED_INPUT_AUDIT",
        "label_policy": LABEL_POLICY, "label_stats": label_stats, "folds": fold_counts,
        "corpus": {"documents": structural.get("counts", {}).get("documents"), "chunks": structural.get("counts", {}).get("chunks"), "parse_modes": structural.get("counts", {}).get("parse_modes"), "structural_fingerprint": structural.get("content_fingerprint"), "processed_fingerprint": processed.get("content_fingerprint")},
        "model_snapshot": model, "inputs": inputs, "mismatches": mismatches, "warnings": warnings, "errors": errors,
        "active_processes": active_experiment_processes(), "disk": disk_record(), "runtime": runtime, "model_use_eligibility": eligibility,
        "constraints": {"candidate_sources": ["vietlegal_e5", "vnlegal_lal", "bm25"], "candidate_depth": CANDIDATE_DEPTH, "dimension": DIMENSION, "no_download": True, "no_public_submission": True, "no_fold0_selection": True, "no_query_memory": True, "old_exp013_index_rejected": True},
    }
    report["audit_fingerprint"] = content_hash({"label_stats": label_stats, "folds": fold_counts, "corpus": report["corpus"], "inputs": [(item["path"], item.get("sha256")) for item in inputs], "mismatches": mismatches})
    return write_report(RESULTS_ROOT / "READING_AUDIT.json", report, stage="input-audit", success=not errors)


def disk_record() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    return {"path": str(ROOT.resolve()), "free_bytes": int(usage.free), "total_bytes": int(usage.total), "free_gib": round(usage.free / 2**30, 3)}


def active_experiment_processes() -> list[dict[str, Any]]:
    # Process enumeration is intentionally best effort and never kills/changes
    # a worker.  On Windows the command line may be unavailable to the caller.
    try:
        import subprocess
        output = subprocess.check_output(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"], text=True, stderr=subprocess.DEVNULL, timeout=5)
        values = json.loads(output) if output.strip() else []
        if isinstance(values, dict): values = [values]
        return [dict(item) for item in values if "exp109a" in str(item.get("CommandLine", "")).lower() or "exp109b" in str(item.get("CommandLine", "")).lower() or "exp109c" in str(item.get("CommandLine", "")).lower()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Jina/ColBERT primitives and exact scorer
# ---------------------------------------------------------------------------


def jina_marker(task: str) -> str:
    if task in {"retrieval.query", "query", "retrieval_query"}:
        return "[QueryMarker] "
    if task in {"retrieval.passage", "document", "passage", "retrieval.document"}:
        return "[DocumentMarker] "
    raise ValueError(f"unsupported Jina task: {task}")


def marker_text(text: str, task: str) -> str:
    return jina_marker(task) + str(text)


def strip_special_token_positions(token_ids: Sequence[int], attention_mask: Sequence[int], special_ids: Iterable[int] = ()) -> np.ndarray:
    specials = {int(value) for value in special_ids}
    return np.asarray([index for index, (token_id, active) in enumerate(zip(token_ids, attention_mask)) if int(active) and int(token_id) not in specials], dtype=np.int64)


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array.copy()
    return array / np.maximum(np.linalg.norm(array, axis=axis, keepdims=True), 1e-12)


def project_and_normalize(hidden: Any, projection: Any, *, dimensions: int = DIMENSION) -> Any:
    if torch is not None and isinstance(hidden, torch.Tensor):
        if hidden.ndim != 3:
            raise ValueError("hidden must be [batch,tokens,hidden]")
        values = projection(hidden)[..., :dimensions]
        if values.shape[-1] != dimensions:
            raise ValueError(f"projection dimension mismatch: {values.shape[-1]} != {dimensions}")
        return F.normalize(values.float(), p=2, dim=-1)
    hidden_np = np.asarray(hidden, dtype=np.float32); projection_np = np.asarray(projection, dtype=np.float32)
    if hidden_np.ndim != 3 or projection_np.ndim != 2 or projection_np.shape[1] != hidden_np.shape[-1]:
        raise ValueError("hidden/projection shape mismatch")
    values = hidden_np @ projection_np[:dimensions].T
    return l2_normalize(values)


def quantize_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("expected [tokens, dimension]")
    if values.shape[0] == 0:
        return np.empty_like(values, dtype=np.int8), np.empty((0,), dtype=np.float16)
    scales = np.maximum(np.max(np.abs(values), axis=1), 1e-8) / 127.0
    return np.clip(np.rint(values / scales[:, None]), -127, 127).astype(np.int8), scales.astype(np.float16)


def dequantize_rows(values: np.ndarray, scales: np.ndarray, *, renormalize: bool = True) -> np.ndarray:
    restored = np.asarray(values, dtype=np.float32) * np.asarray(scales, dtype=np.float32)[:, None]
    return l2_normalize(restored) if renormalize and len(restored) else restored


def numpy_maxsim(query_vectors: np.ndarray, document_vectors: np.ndarray, *, normalize: bool = True) -> tuple[float, np.ndarray]:
    query = l2_normalize(query_vectors) if normalize else np.asarray(query_vectors, dtype=np.float32)
    document = l2_normalize(document_vectors) if normalize else np.asarray(document_vectors, dtype=np.float32)
    if query.ndim != 2 or document.ndim != 2 or query.shape[1] != document.shape[1] or not len(query) or not len(document):
        return float("-inf"), np.full((len(query),), float("-inf"), dtype=np.float32)
    matches = (query @ document.T).max(axis=1).astype(np.float32)
    return float(matches.mean()), matches


def streaming_parent_maxsim(query_vectors: np.ndarray, chunk_vectors: Sequence[np.ndarray], *, normalize: bool = True) -> tuple[float, np.ndarray]:
    """Exact parent MaxSim without concatenating all of a long parent's chunks.

    The running per-query-token maximum is mathematically identical to scoring
    the concatenation, while keeping the memory bound to one chunk.
    """
    query = l2_normalize(query_vectors) if normalize else np.asarray(query_vectors, dtype=np.float32)
    if query.ndim != 2 or not len(query):
        return float("-inf"), np.full((len(query),), float("-inf"), dtype=np.float32)
    matches = np.full((len(query),), float("-inf"), dtype=np.float32)
    seen = False
    for chunk in chunk_vectors:
        document = l2_normalize(chunk) if normalize else np.asarray(chunk, dtype=np.float32)
        if document.ndim != 2 or document.shape[1] != query.shape[1] or not len(document):
            continue
        matches = np.maximum(matches, (query @ document.T).max(axis=1).astype(np.float32))
        seen = True
    return (float(matches.mean()), matches) if seen else (float("-inf"), matches)


def torch_maxsim(query_vectors: Any, document_vectors: Any) -> tuple[Any, Any]:
    if torch is None:
        raise RuntimeError("torch is required for torch_maxsim")
    query = F.normalize(query_vectors.float(), p=2, dim=-1); document = F.normalize(document_vectors.float(), p=2, dim=-1)
    matches = (query @ document.T).amax(dim=1)
    return matches.mean(), matches


def maxsim(query_vectors: np.ndarray, document_vectors: np.ndarray) -> float:
    return numpy_maxsim(query_vectors, document_vectors)[0]


def interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def select_two_chunks(match_vectors: Sequence[np.ndarray], metas: Sequence[Mapping[str, Any]]) -> tuple[int, int | None, np.ndarray]:
    if not match_vectors:
        raise ValueError("at least one chunk is required")
    scores = [float(np.asarray(value).mean()) for value in match_vectors]
    top1 = min(range(len(scores)), key=lambda index: (-scores[index], str(metas[index].get("chunk_id", ""))))
    union = np.asarray(match_vectors[top1], dtype=np.float32)
    if len(match_vectors) == 1:
        return top1, None, union
    candidates: list[tuple[float, int, int, str, int]] = []
    first = metas[top1]
    for index, value in enumerate(match_vectors):
        if index == top1:
            continue
        candidate_union = np.maximum(union, np.asarray(value, dtype=np.float32))
        gain = float(candidate_union.mean())
        overlap = interval_overlap(int(first.get("source_start", 0)), int(first.get("source_end", 0)), int(metas[index].get("source_start", 0)), int(metas[index].get("source_end", 0)))
        retained = int(metas[index].get("retained_tokens", metas[index].get("token_count", 0)))
        candidates.append((gain, -overlap, -retained, str(metas[index].get("chunk_id", "")), index))
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    second = candidates[0][-1]
    return top1, second, np.maximum(union, np.asarray(match_vectors[second], dtype=np.float32))


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if not len(values):
        return {"min": float("-inf"), "p10": float("-inf"), "p25": float("-inf"), "median": float("-inf"), "mean": float("-inf"), "max": float("-inf")}
    return {"min": float(np.min(values)), "p10": float(np.quantile(values, .10)), "p25": float(np.quantile(values, .25)), "median": float(np.median(values)), "mean": float(np.mean(values)), "max": float(np.max(values))}


def latent_condition_features(match_vectors: Sequence[np.ndarray], metas: Sequence[Mapping[str, Any]], *, idf_weights: Sequence[float] | None = None, exact_rank: int = 0, parent_token_count: int | None = None) -> dict[str, float]:
    if not match_vectors or len(match_vectors) != len(metas):
        raise ValueError("match vectors and metadata must be non-empty and aligned")
    vectors = [np.asarray(value, dtype=np.float32) for value in match_vectors]
    scores = np.asarray([float(value.mean()) for value in vectors], dtype=np.float32)
    top1, second, union = select_two_chunks(vectors, metas)
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), str(metas[index].get("chunk_id", ""))))
    top2 = order[1] if len(order) > 1 else None; top3 = order[2] if len(order) > 2 else None
    full_union = np.maximum.reduce(vectors)
    weights = np.asarray(idf_weights if idf_weights is not None else np.ones(len(union)), dtype=np.float32)
    if len(weights) != len(union):
        raise ValueError("idf weight/query token length mismatch")
    weighted = float(np.sum(union * weights) / max(float(np.sum(weights)), 1e-12))
    summary = quantile_summary(full_union)
    parent_median = float(np.median(scores))
    selected_count = 1 if second is None else 2
    selected_tokens = int(metas[top1].get("retained_tokens", metas[top1].get("token_count", 0))) + (0 if second is None else int(metas[second].get("retained_tokens", metas[second].get("token_count", 0))))
    return {
        "li_chunk_top1_mean": float(scores[top1]), "li_chunk_top2_mean": float(scores[top2]) if top2 is not None else float(scores[top1]),
        "li_chunk_top3_mean": float(scores[top3]) if top3 is not None else float(scores[top2]) if top2 is not None else float(scores[top1]),
        "li_chunk_top1_top2_gap": float(scores[top1] - (scores[top2] if top2 is not None else scores[top1])),
        "li_two_chunk_union_mean": float(union.mean()), "li_two_chunk_incremental_gain": float(union.mean() - scores[top1]),
        "li_full_parent_union_mean": float(full_union.mean()), "li_idf_weighted_union_mean": weighted,
        "li_token_match_min": summary["min"], "li_token_match_p10": summary["p10"], "li_token_match_p25": summary["p25"],
        "li_token_match_median": summary["median"], "li_token_match_mean": summary["mean"], "li_token_match_max": summary["max"],
        "li_lower_quartile_mean": float(np.mean(np.sort(full_union)[: max(1, len(full_union) // 4)])),
        "li_selected_chunk_count": float(selected_count), "li_selected_token_count": float(selected_tokens),
        "li_parent_chunk_count": float(len(vectors)), "li_parent_token_count": float(parent_token_count if parent_token_count is not None else sum(int(meta.get("retained_tokens", 0)) for meta in metas)),
        "li_top1_minus_parent_median": float(scores[top1] - parent_median), "li_union_minus_parent_median": float(union.mean() - parent_median),
        "li_exact_rank_within_candidate_pool": float(exact_rank),
    }


# ---------------------------------------------------------------------------
# Label-free current-corpus anchor selection
# ---------------------------------------------------------------------------


def select_anchor_indices(token_ids: Sequence[int], attention_mask: Sequence[int], *, maximum: int = MAX_ANCHORS_PRIMARY, idf: Mapping[int, float] | None = None, structural_prefix_count: int = 0, token_texts: Sequence[str] | None = None, tail_tokens: int = 8, special_ids: Iterable[int] = ()) -> np.ndarray:
    if maximum < 1:
        raise ValueError("maximum anchors must be positive")
    active = strip_special_token_positions(token_ids, attention_mask, special_ids).tolist()
    if len(active) <= maximum:
        return np.asarray(active, dtype=np.int32)
    active_set = set(active); mandatory: set[int] = set(active[: max(0, int(structural_prefix_count))])
    texts = list(token_texts) if token_texts is not None else [""] * len(token_ids)
    # Long legal passages can contain hundreds of numeric pieces. They are
    # useful evidence but cannot all be hard requirements under a 96-anchor
    # contract. Prefix/marker/tail remain mandatory; citation pieces receive a
    # deterministic priority ahead of IDF fill.
    citation_priority: set[int] = set()
    for position in active:
        token = texts[position] if position < len(texts) else ""
        if re.search(r"\d", str(token)) or any(mark in str(token).lower() for mark in ("điều", "khoản", "điểm", "article", "§")):
            citation_priority.add(position)
    mandatory.update(active[-min(tail_tokens, len(active)):])
    mandatory = {position for position in mandatory if position in active_set}
    if len(mandatory) > maximum:
        raise GateRejected("REJECTED_ANCHOR_BUDGET", f"{len(mandatory)} mandatory anchors (structural-prefix/tail) exceed fixed budget {maximum}; do not silently drop them")
    else:
        selected = sorted(mandatory)
        citations = sorted(citation_priority - mandatory, key=lambda position: (-float((idf or {}).get(int(token_ids[position]), 0.0)), position))
        selected.extend(citations[: maximum - len(selected)])
        remaining = [position for position in active if position not in set(selected)]
        remaining.sort(key=lambda position: (-float((idf or {}).get(int(token_ids[position]), 0.0)), position))
        selected.extend(remaining[: maximum - len(selected)])
    return np.asarray(sorted(set(selected)), dtype=np.int32)


def choose_length_by_coverage(lengths: Sequence[int], candidates: Sequence[int], target: float) -> dict[str, Any]:
    values = np.asarray([max(0, int(value)) for value in lengths], dtype=np.int64)
    if not len(values):
        raise ValueError("cannot choose a length from an empty distribution")
    selected = int(max(candidates))
    coverage = 0.0
    curve = {}
    for candidate in sorted(set(map(int, candidates))):
        value = float(np.mean(values <= candidate)); curve[str(candidate)] = value
        if coverage < target and value >= target:
            selected, coverage = candidate, value
    if coverage == 0.0:
        coverage = float(np.mean(values <= selected))
    return {"selected": selected, "target": float(target), "coverage": coverage, "curve": curve, "truncated_count": int(np.sum(values > selected)), "count": int(len(values)), "p50": int(np.quantile(values, .5)), "p99": int(np.quantile(values, .99)), "p999": int(np.quantile(values, .999))}


# ---------------------------------------------------------------------------
# Candidate union and ceiling audit
# ---------------------------------------------------------------------------


def ranking_items(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = row.get("documents", row.get("rankings", row.get("candidates", [])))
    if isinstance(values, Mapping):
        values = next(iter(values.values()), [])
    result = []
    for position, item in enumerate(values, 1):
        if isinstance(item, Mapping):
            doc_id = item.get("doc_id", item.get("document_id", item.get("id")))
            score = float(item.get("score", item.get("bm25_score", 0.0)))
            rank = int(item.get("rank", position))
        else:
            doc_id, score, rank = item, 0.0, position
        if doc_id is not None:
            result.append({"doc_id": str(doc_id), "rank": rank, "score": score})
    return result


def build_candidate_union(source_rankings: Mapping[str, Sequence[Mapping[str, Any]]], *, depth: int = CANDIDATE_DEPTH, sources: Sequence[str] = ("vietlegal_e5", "vnlegal_lal", "bm25")) -> list[dict[str, Any]]:
    if depth < 1:
        raise ValueError("candidate depth must be positive")
    records: dict[str, dict[str, Any]] = {}
    for source in sources:
        for position, item in enumerate(source_rankings.get(source, []), 1):
            if position > depth:
                break
            value = dict(item); doc_id = str(value["doc_id"])
            record = records.setdefault(doc_id, {"doc_id": doc_id, "sources": {}})
            record["sources"][source] = {"rank": int(value.get("rank", position)), "score": float(value.get("score", 0.0))}
    ordered = sorted(records.values(), key=lambda item: (min(int(value["rank"]) for value in item["sources"].values()), str(item["doc_id"])))
    for rank, item in enumerate(ordered, 1):
        item["candidate_rank"] = rank
        item["source_presence"] = len(item["sources"])
    return ordered


def candidate_union_ids(source_rankings: Mapping[str, Sequence[Mapping[str, Any]]], *, depth: int = CANDIDATE_DEPTH, sources: Sequence[str] = ("vietlegal_e5", "vnlegal_lal", "bm25")) -> list[str]:
    return [str(item["doc_id"]) for item in build_candidate_union(source_rankings, depth=depth, sources=sources)]


def recall_fraction(prediction: Sequence[str], gold: set[str], k: int | None = None) -> float:
    if not gold:
        return float("nan")
    values = set(map(str, prediction if k is None else prediction[:k]))
    return float(len(values & gold) / len(gold))


def candidate_ceiling_report(source_rankings_by_qid: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], answers: Mapping[str, set[str]], folds: Mapping[str, Sequence[str]], *, outer: str = FOLD0, report_path: Path | None = None, input_fingerprints: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compute the fixed-depth candidate ceiling without implicit persistence.

    ``report_path`` is deliberately opt-in so a unit fixture can never replace
    the production candidate gate artifact.
    """
    included = sorted(qid for name, qids in folds.items() if name != outer for qid in qids)
    if set(included) - set(source_rankings_by_qid):
        raise ValueError("candidate source rankings do not cover outer-train queries")
    curves: dict[str, Any] = {}
    for depth in CANDIDATE_DIAGNOSTIC_DEPTHS:
        per_qid: dict[str, float] = {}; counts: list[int] = []; absent = 0
        for qid in included:
            pool = candidate_union_ids(source_rankings_by_qid[qid], depth=depth)
            gold = answers.get(qid, set()); per_qid[qid] = recall_fraction(pool, gold); counts.append(len(pool))
            absent += int(bool(gold) and not (set(pool) & gold))
        values = np.asarray([value for value in per_qid.values() if np.isfinite(value)], dtype=np.float64)
        per_fold = {}
        for name, qids in sorted(folds.items()):
            if name == outer: continue
            fold_values = [per_qid[str(qid)] for qid in qids if str(qid) in per_qid and np.isfinite(per_qid[str(qid)])]
            per_fold[name] = float(np.mean(fold_values)) if fold_values else float("nan")
        single = [per_qid[qid] for qid in included if len(answers.get(qid, set())) == 1 and np.isfinite(per_qid[qid])]
        multi = [per_qid[qid] for qid in included if len(answers.get(qid, set())) > 1 and np.isfinite(per_qid[qid])]
        average_candidates = float(np.mean(counts)) if counts else 0.0
        # Each parent score evaluates every query token against the retained
        # token anchors.  This is a deliberately conservative, hardware-free
        # estimate; the measured 5k-chunk checkpoint replaces it later.
        projected_pairs = average_candidates * CONFIG_E_ANCHORS * CONFIG_E_STORAGE_DIMENSION
        curves[str(depth)] = {"candidate_count": {"min": min(counts, default=0), "mean": average_candidates, "max": max(counts, default=0)}, "recall@pool": float(np.mean(values)) if len(values) else 0.0, "per_inner_fold": per_fold, "single_gold": float(np.mean(single)) if single else 0.0, "multi_gold": float(np.mean(multi)) if multi else 0.0, "gold_absent_from_all_sources": absent, "projected_exact_maxsim": {"per_query_token_dimension_multiplications": projected_pairs, "assumption": "candidate parents times Config-E 256 anchors times 128 dimensions; runtime requires measured throughput"}}
        if depth > CANDIDATE_DIAGNOSTIC_DEPTHS[0]:
            previous = curves[str(CANDIDATE_DIAGNOSTIC_DEPTHS[CANDIDATE_DIAGNOSTIC_DEPTHS.index(depth) - 1])]
            curves[str(depth)]["marginal_rescue_vs_previous"] = float(curves[str(depth)]["recall@pool"] - previous["recall@pool"])
    primary = curves[str(CANDIDATE_DEPTH)]
    gate = {"aggregate_ge_0_992": primary["recall@pool"] >= .992, "each_inner_fold_ge_0_990": all(value >= .990 for value in primary["per_inner_fold"].values()), "pass": primary["recall@pool"] >= .992 and all(value >= .990 for value in primary["per_inner_fold"].values())}
    report = {"schema_version": SCHEMA, "stage": "candidate_ceiling", "outer": outer, "candidate_contract": {"sources": ["vietlegal_e5", "vnlegal_lal", "bm25"], "primary_depth": CANDIDATE_DEPTH, "diagnostic_depths": list(CANDIDATE_DIAGNOSTIC_DEPTHS), "union": "unique_parent_ids"}, "input_fingerprints": dict(input_fingerprints or {}), "curves": curves, "gate": gate, "status": "PASS_CANDIDATE_CEILING_0995" if primary["recall@pool"] >= .995 and gate["pass"] else "PASS_CANDIDATE_CEILING" if gate["pass"] else "REJECTED_CANDIDATE_CEILING", "selection_scope": f"outer_train_folds_only; D{CANDIDATE_DEPTH} per source fixed before metric inspection", "claim_boundary": "candidate coverage ceiling only; not Recall@5 or model performance"}
    return write_report(report_path, report, stage="candidate-ceiling", success=gate["pass"]) if report_path is not None else report


def source_maps_from_rows(rows: Sequence[Mapping[str, Any]], source: str) -> dict[str, list[dict[str, Any]]]:
    return {str(row["qid"]): ranking_items(row) for row in rows}


def load_verified_ranking_directory(directory: Path, *, qids: Iterable[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise GateRejected("REJECTED_INPUT_AUDIT", f"ranking manifest missing: {manifest_path}")
    manifest = read_json(manifest_path); require_success(directory, manifest.get("content_fingerprint"))
    wanted = None if qids is None else {str(qid) for qid in qids}; rows: list[dict[str, Any]] = []; observed: set[str] = set(); count = 0
    for item in manifest.get("shards", []):
        path = directory / "shards" / str(item["name"])
        if not path.exists() or sha256_file(path) != item.get("sha256"):
            raise GateRejected("REJECTED_STALE_ARTIFACT", f"ranking shard hash mismatch: {path}")
        for row in read_jsonl(path):
            count += 1; qid = str(row.get("qid"))
            if qid in observed: raise GateRejected("REJECTED_STALE_ARTIFACT", f"duplicate ranking qid: {directory}/{qid}")
            observed.add(qid)
            if wanted is None or qid in wanted: rows.append(row)
    if count != int(manifest.get("query_count", count)):
        raise GateRejected("REJECTED_STALE_ARTIFACT", f"ranking row count mismatch: {directory}")
    if wanted is not None and wanted - observed: raise GateRejected("REJECTED_STALE_ARTIFACT", f"ranking qid coverage mismatch: {directory}")
    return rows, manifest


def load_exp109b_anchor_sources(*, outer: str = FOLD0, qids: Iterable[str] | None = None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    e5_rows, _ = load_verified_ranking_directory(EXP109B_CACHE / "rankings" / "vietlegal_e5" / outer, qids=qids)
    lal_rows, _ = load_verified_ranking_directory(EXP109B_CACHE / "rankings" / "vnlegal_lal" / outer, qids=qids)
    # BM25 raw evidence is read only when the caller requests a ranking.  The
    # raw source is tuned EXP-021 evidence, never the EXP-022 ordered union.
    bm25 = load_tuned_bm25_rankings(qids=qids)
    maps = {"vietlegal_e5": source_maps_from_rows(e5_rows, "vietlegal_e5"), "vnlegal_lal": source_maps_from_rows(lal_rows, "vnlegal_lal"), "bm25": bm25}
    qids = set(maps["vietlegal_e5"]) & set(maps["vnlegal_lal"]) & set(maps["bm25"])
    return {qid: {source: maps[source][qid] for source in maps} for qid in sorted(qids)}


def reproduce_109b_anchor_predictions(source_rankings_by_qid: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], folds: Mapping[str, Sequence[str]], *, outer: str = FOLD0, progress_callback: Callable[[int, str], None] | None = None) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Reproduce the frozen EXP-109B LambdaMART anchor on strict inner folds.

    E4 consumes a *locked* EXP-109B top-16 anchor.  Re-running EXP-109B's
    16-config nested selection grid for every E4 validation fold is both a
    different workflow and 256 expensive fits.  The published strict-inner
    pilot already froze one configuration per held-out fold; this function
    verifies that artifact and performs only the four required cross-fit fits.
    """
    inner_folds = {name: list(map(str, qids)) for name, qids in folds.items() if name != outer}
    if set(inner_folds) != set(FOLD_NAMES) - {outer}:
        raise ValueError("anchor reproduction requires all four non-outer folds")
    try:
        import exp109b_encoder_complementarity as legacy
    except Exception as exc:
        raise GateRejected("REJECTED_ANCHOR_REPRODUCTION", f"EXP-109B anchor helper unavailable: {exc}") from exc
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise GateRejected("REJECTED_ANCHOR_REPRODUCTION", f"LightGBM unavailable for locked EXP-109B anchor: {exc}") from exc
    pilot_path = EXP109B_RESULTS / "cached_fusion_pilot" / outer / "CACHED_FUSION_PILOT.json"
    pilot = read_json(pilot_path) if pilot_path.exists() else {}
    if pilot.get("status") != "PASS_CACHED_FUSION_INNER_GATE" or pilot.get("winner") != "lambdamart_top50_per_source":
        raise GateRejected("REJECTED_ANCHOR_REPRODUCTION", "verified EXP-109B frozen LambdaMART pilot is missing")
    locked: dict[str, dict[str, Any]] = {}
    for row in pilot.get("lambdamart_folds", []):
        validation = str(row.get("validation_fold", "")); config = row.get("chosen_config", {}).get("config")
        if validation in inner_folds and isinstance(config, Mapping): locked[validation] = dict(config)
    if set(locked) != set(inner_folds):
        raise GateRejected("REJECTED_ANCHOR_REPRODUCTION", "EXP-109B pilot lacks a frozen config for one or more strict inner folds")
    rankings: dict[str, dict[str, list[str]]] = {source: {} for source in ("vietlegal_e5", "vnlegal_lal", "bm25")}
    score_maps: dict[str, dict[str, dict[str, tuple[int, float]]]] = {source: {} for source in rankings}
    for qid, sources in source_rankings_by_qid.items():
        for source in rankings:
            values = list(sources.get(source, [])); rankings[source][str(qid)] = [str(item["doc_id"]) for item in values]
            score_maps[source][str(qid)] = {str(item["doc_id"]): (int(item.get("rank", index + 1)), float(item.get("score", 0.0))) for index, item in enumerate(values)}
    metadata = load_parent_metadata()
    train = load_train(); query_lengths = {str(qid): len(str(row.get("question", "")).split()) for qid, row in train.items()}
    predictions: dict[str, list[str]] = {}; fold_reports: dict[str, Any] = {}; answers, _ = canonical_labels()
    for completed, validation in enumerate(sorted(inner_folds), 1):
        train_qids = [qid for name, values in inner_folds.items() if name != validation for qid in values]
        heldout_qids = list(inner_folds[validation])
        train_x, train_y, train_groups, _train_candidates, feature_names = legacy._lgbm_matrix_for_qids(rankings, score_maps, train_qids, answers, sources=("vietlegal_e5", "vnlegal_lal", "bm25"), depth=CANDIDATE_DEPTH, metadata=metadata, query_token_lengths=query_lengths)
        valid_x, _valid_y, _valid_groups, valid_candidates, _ = legacy._lgbm_matrix_for_qids(rankings, score_maps, heldout_qids, answers, sources=("vietlegal_e5", "vnlegal_lal", "bm25"), depth=CANDIDATE_DEPTH, metadata=metadata, query_token_lengths=query_lengths)
        if not len(train_x) or not np.any(train_y > 0): raise GateRejected("REJECTED_ANCHOR_REPRODUCTION", f"no positive training rows for locked anchor {validation}")
        config = locked[validation]
        # ``eval_at``/``ndcg_at`` are reporting-only here (there is no eval
        # set); passing either through sklearn triggers an alias warning.
        # An explicit worker count also avoids sklearn/joblib's missing WMIC
        # physical-core probe on this Windows installation.
        model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_jobs=min(4, os.cpu_count() or 1), num_leaves=int(config["num_leaves"]), min_child_samples=int(config["min_data_in_leaf"]), learning_rate=float(config["learning_rate"]), n_estimators=int(config["num_boost_round"]), feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, deterministic=True, random_state=int(config.get("seed", 109)), verbosity=-1)
        # Keep NumPy fit/predict symmetric.  Passing feature names only at fit
        # causes sklearn to emit a misleading predict-time warning.
        model.fit(train_x, train_y, group=train_groups)
        values = model.predict(valid_x) if len(valid_x) else np.empty((0,), dtype=np.float32); offset = 0; heldout: dict[str, list[str]] = {}
        for qid, candidates in valid_candidates:
            local = values[offset:offset + len(candidates)]; order = np.lexsort((np.asarray(candidates, dtype="U"), -np.asarray(local, dtype=np.float64))); heldout[str(qid)] = [str(candidates[int(position)]) for position in order]; offset += len(candidates)
        if set(heldout) != set(heldout_qids): raise GateRejected("REJECTED_ANCHOR_REPRODUCTION", f"locked anchor prediction coverage mismatch for {validation}")
        predictions.update(heldout); fold_reports[validation] = {"locked_config": config, "train_queries": len(train_qids), "heldout_queries": len(heldout_qids), "pilot_sha256": sha256_file(pilot_path)}
        if progress_callback is not None: progress_callback(completed, validation)
    expected = {str(qid) for name, qids in inner_folds.items() for qid in qids}
    if set(predictions) != expected:
        raise GateRejected("REJECTED_ANCHOR_REPRODUCTION", f"anchor prediction coverage mismatch: {len(predictions)} != {len(expected)}")
    return predictions, {"model": "EXP-109B scalar LambdaMART", "outer_excluded": outer, "fold_reports": fold_reports, "selection_scope": "strict inner folds only; frozen per-fold config from verified cached fusion pilot", "pilot": str(pilot_path.resolve()), "pilot_sha256": sha256_file(pilot_path)}


def load_tuned_bm25_rankings(limit: int = 500, *, qids: Iterable[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    if not (BM25_RAW_DIR / "manifest.json").exists() or not BM25_TUNING_REPORT.exists():
        raise GateRejected("REJECTED_INPUT_AUDIT", "tuned BM25 raw evidence/tuning report is missing")
    wanted = None if qids is None else {str(qid) for qid in qids}; rows: dict[str, dict[str, Any]] = {}
    for path in sorted((BM25_RAW_DIR / "shards").glob("evidence_*.jsonl")):
        for row in read_jsonl(path):
            qid = str(row["qid"])
            if wanted is None or qid in wanted: rows[qid] = row
    if wanted is not None and wanted - set(rows): raise GateRejected("REJECTED_STALE_ARTIFACT", "BM25 qid coverage mismatch")
    tuning = read_json(BM25_TUNING_REPORT).get("selected_by_candidate_budget", {}).get("150", {})
    _folds, fold_for = load_folds(); result: dict[str, list[dict[str, Any]]] = {}
    for qid, row in rows.items():
        config = tuning.get(fold_for.get(qid), tuning.get("fold_0", {"depth": 1024, "parent_rrf_k": 32, "fusion_rrf_k": 32, "head_cutoff": 16}))
        result[qid] = bm25_rankings(row.get("evidence", []), config, limit=limit)
    return result


def bm25_rankings(evidence: Sequence[Any], config: Mapping[str, Any], *, limit: int = 500) -> list[dict[str, Any]]:
    depth, parent_k, fusion_k, head = int(config.get("depth", 1024)), int(config.get("parent_rrf_k", 32)), int(config.get("fusion_rrf_k", 32)), int(config.get("head_cutoff", 16))
    eligible: list[tuple[str, list[int]]] = []
    for item in evidence:
        if isinstance(item, Mapping):
            doc_id, ranks = item.get("doc_id"), item.get("ranks", item.get("passage_ranks", []))
        else:
            doc_id, ranks = item[0], item[1]
        ranks = sorted(int(value) for value in ranks if int(value) <= depth)
        if ranks: eligible.append((str(doc_id), ranks))
    first = sorted(eligible, key=lambda item: (item[1][0], item[0]))
    rrf = sorted(eligible, key=lambda item: (-sum(1.0 / (parent_k + rank) for rank in item[1]), item[1][0], item[0]))
    scores: dict[str, float] = defaultdict(float); best: dict[str, int] = {}
    for ranking in (first, rrf):
        for rank, (doc_id, _ranks) in enumerate(ranking, 1):
            scores[doc_id] += 1.0 / (fusion_k + rank); best[doc_id] = min(best.get(doc_id, rank), rank)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], best[doc_id], doc_id)); head_ids = ordered[:head]; final = head_ids + [doc_id for doc_id in ordered if doc_id not in set(head_ids)]
    return [{"doc_id": doc_id, "rank": rank, "score": float(scores[doc_id])} for rank, doc_id in enumerate(final[:limit], 1)]


# ---------------------------------------------------------------------------
# Reproduction and resource/fidelity gates
# ---------------------------------------------------------------------------


def deterministic_bootstrap(values: Sequence[float], *, samples: int = 10_000, seed: int = 109) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values): return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed); sample = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "lower": float(np.quantile(sample, .025)), "upper": float(np.quantile(sample, .975))}


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(a) != len(b) or not len(a): raise ValueError("correlation inputs must have equal non-zero length")
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort"); result = np.empty(len(values), dtype=np.float64); result[order] = np.arange(len(values), dtype=np.float64); return result
    ra, rb = rank(a), rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0: return 1.0 if np.array_equal(ra, rb) else 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _kendall_agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Tie-aware, bounded Kendall sign agreement for one query's candidates."""
    if len(left) < 2:
        return 1.0
    agree = total = 0
    for i in range(len(left) - 1):
        for j in range(i + 1, len(left)):
            a = int(np.sign(left[i] - left[j])); b = int(np.sign(right[i] - right[j]))
            if a == 0 and b == 0:
                agree += 1
            elif a == 0 or b == 0:
                agree += 0.5
            elif a == b:
                agree += 1
            total += 1
    return float(agree / total) if total else 1.0


def _bin_error(values: np.ndarray, sizes: Sequence[int] | None) -> dict[str, float]:
    if sizes is None or len(sizes) != len(values):
        return {}
    size_values = np.asarray(sizes, dtype=np.int64)
    if not len(size_values):
        return {}
    edges = np.unique(np.quantile(size_values, [0.0, 1 / 3, 2 / 3, 1.0]).astype(np.int64))
    result: dict[str, float] = {}
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (size_values >= low) & (size_values <= high)
        if np.any(mask):
            result[f"{int(low)}-{int(high)}"] = float(values[mask].mean())
    return result


def fidelity_metrics(full_scores: Sequence[float], compressed_scores: Sequence[float], *, qids: Sequence[str] | None = None, gold_by_qid: Mapping[str, Iterable[str]] | None = None, doc_ids: Sequence[str] | None = None, positive_pairs: Sequence[tuple[int, int]] = (), parent_lengths: Sequence[int] | None = None, parent_chunk_counts: Sequence[int] | None = None) -> dict[str, Any]:
    """Measure score/ranking fidelity *within each query's candidate group*.

    Global sorting of MaxSim scores from different queries is invalid; every
    ranking statistic here is grouped by qid. ``positive_pairs`` is retained
    for backwards-compatible small fixtures, while grouped positive/negative
    pairs are generated automatically for the real rescue fixture.
    """
    full, compressed = np.asarray(full_scores, dtype=np.float64), np.asarray(compressed_scores, dtype=np.float64)
    if full.shape != compressed.shape or not len(full): raise ValueError("fidelity score vectors must have equal non-zero shape")
    absolute = np.abs(full - compressed); relative = absolute / np.maximum(np.abs(full), 1e-8)
    group_ids = ["__single_group__"] * len(full) if qids is None else list(map(str, qids))
    if len(group_ids) != len(full): raise ValueError("qid count must equal fidelity score count")
    ids = [str(i) for i in range(len(full))] if doc_ids is None else list(map(str, doc_ids))
    if len(ids) != len(full): raise ValueError("doc-id count must equal fidelity score count")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, qid in enumerate(group_ids): groups[qid].append(index)
    top5: list[float] = []; spearmans: list[float] = []; kendalls: list[float] = []
    full_recall: list[float] = []; compressed_recall: list[float] = []; recall_retention: list[float] = []
    agreement: list[float] = []; conditional: list[float] = []; false_flips = rescued = 0
    for qid, indices in sorted(groups.items()):
        f, c = full[indices], compressed[indices]
        full_order = sorted(range(len(indices)), key=lambda pos: (-f[pos], ids[indices[pos]]))
        compressed_order = sorted(range(len(indices)), key=lambda pos: (-c[pos], ids[indices[pos]]))
        top5.append(float(len(set(full_order[:5]) & set(compressed_order[:5])) / max(1, min(5, len(indices)))))
        spearmans.append(spearman_correlation(f, c)); kendalls.append(_kendall_agreement(f, c))
        gold = {str(value) for value in (gold_by_qid or {}).get(qid, ())}
        if gold:
            fgold = recall_fraction([ids[indices[pos]] for pos in full_order], gold, 5)
            cgold = recall_fraction([ids[indices[pos]] for pos in compressed_order], gold, 5)
            full_recall.append(fgold); compressed_recall.append(cgold)
            if fgold > 0: recall_retention.append(cgold / fgold)
        positives = [pos for pos, index in enumerate(indices) if ids[index] in gold]
        negatives = [pos for pos, index in enumerate(indices) if ids[index] not in gold]
        for pos in positives:
            for neg in negatives:
                reference = int(np.sign(f[pos] - f[neg])); observed = int(np.sign(c[pos] - c[neg]))
                agreement.append(float(reference == observed))
                if reference >= 0:
                    conditional.append(float(observed >= 0))
                    false_flips += int(observed < 0)
                rescued += int(reference < 0 and observed >= 0)
    # Unit fixtures without qids still exercise the explicit pair contract.
    if positive_pairs:
        for a, b in positive_pairs:
            reference = int(np.sign(full[a] - full[b])); observed = int(np.sign(compressed[a] - compressed[b]))
            agreement.append(float(reference == observed))
            if reference >= 0: conditional.append(float(observed >= 0)); false_flips += int(observed < 0)
            rescued += int(reference < 0 and observed >= 0)
    monotonic = None
    if parent_lengths is not None and len(parent_lengths) == len(absolute) and len(absolute) > 2:
        monotonic = spearman_correlation(parent_lengths, absolute)
    return {"mean_absolute_error": float(np.mean(absolute)), "max_absolute_error": float(np.max(absolute)), "mean_relative_error": float(np.mean(relative)), "spearman": float(np.mean(spearmans)), "kendall_agreement": float(np.mean(kendalls)), "top5_agreement": float(np.mean(top5)), "full_gold_recall_at5": float(np.mean(full_recall)) if full_recall else None, "compressed_gold_recall_at5": float(np.mean(compressed_recall)) if compressed_recall else None, "gold_recall_at5_retention": float(np.mean(recall_retention)) if recall_retention else None, "pairwise_sign_agreement": float(np.mean(agreement)) if agreement else 1.0, "conditional_positive_order_retention": float(np.mean(conditional)) if conditional else 1.0, "false_flips_vs_reference": false_flips, "rescued_reference_misorderings": rescued, "length_error_spearman": monotonic, "error_by_parent_length": _bin_error(absolute, parent_lengths), "error_by_parent_chunk_count": _bin_error(absolute, parent_chunk_counts), "no_monotonic_error_explosion": monotonic is None or monotonic < .95}


def fidelity_gate(metrics: Mapping[str, Any], *, anchors: int) -> dict[str, Any]:
    checks = {"mean_absolute_error_le_0_010": float(metrics.get("mean_absolute_error", math.inf)) <= .010, "top5_agreement_ge_0_97": float(metrics.get("top5_agreement", 0.0)) >= .97, "conditional_positive_order_retention_ge_0_97": float(metrics.get("conditional_positive_order_retention", 0.0)) >= .97, "no_monotonic_error_explosion": bool(metrics.get("no_monotonic_error_explosion", False))}
    return {"anchors": anchors, "checks": checks, "pass": all(checks.values()), "status": "PASS_FIDELITY" if all(checks.values()) else "REJECTED_COMPRESSION_FIDELITY_GATE"}


def _old_reference_maxsim(query: np.ndarray, document: np.ndarray) -> float:
    """Independent, deliberately small NumPy reference for the EXP-013 store."""
    if query.ndim != 2 or document.ndim != 2 or not len(query) or not len(document):
        raise ValueError("old reference requires non-empty rank-2 token matrices")
    return float(np.max(np.asarray(query, dtype=np.float64) @ np.asarray(document, dtype=np.float64).T, axis=1).mean())


def generate_real_old_pairs(*, required: int = 20) -> list[dict[str, Any]]:
    """Freeze a deterministic, diverse sample from the actual EXP-013 token stores.

    This is evidence generation, not new annotation: qids and document spans are
    selected only from the old stores and their observed lengths.
    """
    query_dir = EXP013_CACHE / "colbert_queries" / "train"; leaf_dir = EXP013_CACHE / "colbert_leaves"
    qids = [str(value) for value in read_json(query_dir / "qids.json")]
    offsets = np.load(query_dir / "query_offsets.i64.npy", mmap_mode="r")
    queries = np.load(query_dir / "query_vectors.f16.npy", mmap_mode="r")
    vectors = np.load(leaf_dir / "token_vectors.int8.npy", mmap_mode="r"); scales = np.load(leaf_dir / "token_scales.f16.npy", mmap_mode="r")
    passages = list(read_jsonl(leaf_dir / "passages.jsonl"))
    if len(qids) != len(offsets) - 1 or not passages:
        raise GateRejected("REJECTED_REPRODUCTION_GATE", "old EXP-013 stores are incomplete")
    # Quantiles of both query and passage lengths force short/long and multi-span
    # coverage without using relevance labels.
    qorder = sorted(range(len(qids)), key=lambda i: (int(offsets[i + 1] - offsets[i]), qids[i]))
    porder = sorted(range(len(passages)), key=lambda i: (int(passages[i]["token_end"]) - int(passages[i]["token_start"]), str(passages[i].get("doc_id", "")), str(passages[i].get("chunk_id", ""))))
    positions = np.linspace(0, min(len(qorder), len(porder)) - 1, required, dtype=int)
    pairs: list[dict[str, Any]] = []
    for ordinal, pos in enumerate(positions.tolist()):
        qi, passage_index = qorder[pos], porder[(pos * 37 + ordinal) % len(porder)]
        passage = passages[passage_index]
        q = np.asarray(queries[int(offsets[qi]):int(offsets[qi + 1])], dtype=np.float32)
        d = dequantize_rows(vectors[int(passage["token_start"]):int(passage["token_end"])], scales[int(passage["token_start"]):int(passage["token_end"])])
        if q.shape[1] != OLD_DIMENSION or d.shape[1] != OLD_DIMENSION:
            raise GateRejected("REJECTED_REPRODUCTION_GATE", "old EXP-013 store does not have 64-d vectors")
        packed, row_scales = quantize_rows(d); restored = dequantize_rows(packed, row_scales)
        pairs.append({"qid": qids[qi], "doc_id": str(passage.get("doc_id", "")), "chunk_id": str(passage.get("chunk_id", "")), "query_shape": list(q.shape), "document_shape": list(d.shape), "query_vectors_64": q.tolist(), "document_vectors_64": d.tolist(), "expected_maxsim": _old_reference_maxsim(q, d), "expected_int8_maxsim": _old_reference_maxsim(q, restored), "selection": {"ordinal": ordinal, "query_length_rank": pos, "passage_length_rank": int((pos * 37 + ordinal) % len(porder))}})
    return pairs


def reproduction_report(*, generate_fixture: bool = True) -> dict[str, Any]:
    rng = np.random.default_rng(109); query = l2_normalize(rng.normal(size=(5, DIMENSION)).astype(np.float32)); document = l2_normalize(rng.normal(size=(17, DIMENSION)).astype(np.float32)); packed, scales = quantize_rows(document); restored = dequantize_rows(packed, scales)
    numpy_score, matches = numpy_maxsim(query, document); quant_score, _ = numpy_maxsim(query, restored)
    torch_parity = None
    if torch is not None:
        torch_score, _ = torch_maxsim(torch.tensor(query), torch.tensor(document)); torch_parity = abs(float(torch_score) - numpy_score)
    old_manifest = read_json(EXP013_CACHE / "colbert_leaves" / "manifest.json") if (EXP013_CACHE / "colbert_leaves" / "manifest.json").exists() else {}
    real_fixture_path = CACHE_ROOT / "reproduction" / "real_pairs.json"
    if generate_fixture:
        real_fixture_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(real_fixture_path, generate_real_old_pairs())
    real_pairs = read_json(real_fixture_path) if real_fixture_path.exists() else []
    if not isinstance(real_pairs, list): real_pairs = []
    real_errors: list[str] = []
    for index, pair in enumerate(real_pairs):
        try:
            query64 = np.asarray(pair["query_vectors_64"], dtype=np.float32); document64 = np.asarray(pair["document_vectors_64"], dtype=np.float32)
            if query64.ndim != 2 or document64.ndim != 2 or query64.shape[1] != OLD_DIMENSION or document64.shape[1] != OLD_DIMENSION: raise ValueError("expected non-empty 64-d query/document vectors")
            reference = float(pair["expected_maxsim"]); observed, _ = numpy_maxsim(query64, document64)
            packed_pair, scales_pair = quantize_rows(document64); quantized, _ = numpy_maxsim(query64, dequantize_rows(packed_pair, scales_pair))
            if abs(observed - reference) > 1e-5: raise ValueError(f"reference mismatch {observed} vs {reference}")
            if "expected_int8_maxsim" in pair and abs(quantized - float(pair["expected_int8_maxsim"])) > 1e-5: raise ValueError("int8 reference mismatch")
        except Exception as exc:
            real_errors.append(f"pair {index}: {exc}")
    real_count = len(real_pairs); synthetic_gate = (torch_parity is None or torch_parity <= 1e-5) and old_manifest.get("v3_fingerprint") != STRUCTURAL_FINGERPRINT
    full_gate = synthetic_gate and real_count >= 20 and not real_errors
    report = {"schema_version": SCHEMA, "stage": "reproduce_exp013", "status": "PASS" if full_gate else "REJECTED_REPRODUCTION_GATE", "marker_contract": {"query": jina_marker("query"), "document": jina_marker("document")}, "numpy_torch_maxsim_abs_error": torch_parity, "int8_score_delta": abs(numpy_score - quant_score), "stable_tie_break": stable_rank([1.0, 1.0], ["b", "a"]) == ["a", "b"], "old_index": {"v3_fingerprint": old_manifest.get("v3_fingerprint"), "current_structural_fingerprint": STRUCTURAL_FINGERPRINT, "rejected_for_current_scoring": True}, "real_old_pairs": {"required": 20, "fixture": str(real_fixture_path.resolve()), "fixture_sha256": sha256_file(real_fixture_path) if real_fixture_path.exists() else None, "executed": real_count, "errors": real_errors, "claim": "each real pair must carry frozen 64-d vectors and expected FP32/int8 MaxSim references"}, "gate": {"synthetic_primitives": synthetic_gate, "torch_maxsim_le_1e-5": torch_parity is None or torch_parity <= 1e-5, "old_index_stale_rejected": old_manifest.get("v3_fingerprint") != STRUCTURAL_FINGERPRINT, "real_old_pairs_ge_20": real_count >= 20, "real_pair_references_pass": not real_errors}}
    return write_report(RESULTS_ROOT / "REPRODUCTION_REPORT.json", report, stage="reproduce-exp013", success=full_gate)


def stable_rank(scores: Sequence[float], doc_ids: Sequence[str], limit: int | None = None) -> list[str]:
    if len(scores) != len(doc_ids) or any(not np.isfinite(float(score)) for score in scores):
        raise ValueError("scores/doc_ids mismatch or non-finite score")
    order = sorted(range(len(doc_ids)), key=lambda index: (-float(scores[index]), str(doc_ids[index])))
    return [str(doc_ids[index]) for index in order[:limit]]


def exact_refinement_set(approximate_scores: Mapping[str, float], anchor_ranking: Sequence[str], *, top_k: int = 16) -> list[str]:
    """Frozen label-free E4 policy: top-k approximate union top-k LambdaMART."""
    if top_k != 16:
        raise ValueError("Config E4 locks top_k=16; it is not a tuning parameter")
    approximate = stable_rank(list(approximate_scores.values()), list(approximate_scores), limit=top_k)
    anchor = [str(doc_id) for doc_id in anchor_ranking if str(doc_id) in approximate_scores][:top_k]
    return list(dict.fromkeys(approximate + anchor))


def cascade_refined_ranking(approximate_scores: Mapping[str, float], exact_scores: Mapping[str, float], *, limit: int | None = None) -> list[str]:
    """Rank exact-refined parents by exact score and all others by approximate."""
    if not set(exact_scores).issubset(approximate_scores):
        raise ValueError("exact scores must be a subset of approximate candidates")
    scores = [float(exact_scores.get(doc_id, approximate_scores[doc_id])) for doc_id in approximate_scores]
    return stable_rank(scores, list(approximate_scores), limit=limit)


def estimate_index_bytes(*, chunks: int = CHUNK_COUNT, anchors: int = MAX_ANCHORS_PRIMARY, dimensions: int = DIMENSION) -> dict[str, int]:
    vector_bytes = chunks * anchors * dimensions * 1; scale_bytes = chunks * anchors * 2; passage_bytes = chunks * 180; checkpoint_bytes = vector_bytes // 8
    return {"vectors_int8": vector_bytes, "scales_fp16": scale_bytes, "passage_sidecars": passage_bytes, "temporary_checkpoints": checkpoint_bytes, "total": vector_bytes + scale_bytes + passage_bytes + checkpoint_bytes}


def estimate_dual_index_bytes(*, chunks: int = CHUNK_COUNT, anchors: int = CONFIG_E_ANCHORS, dimensions: int = DIMENSION, full_tokens_per_chunk: int = 1024) -> dict[str, int]:
    approximate = estimate_index_bytes(chunks=chunks, anchors=anchors, dimensions=dimensions)
    full_vectors = chunks * full_tokens_per_chunk * dimensions
    full_scales = chunks * full_tokens_per_chunk * 2
    total = approximate["total"] + full_vectors + full_scales
    return {"approximate_store_total": approximate["total"], "full_vectors_int8_upper_bound": full_vectors, "full_scales_fp16_upper_bound": full_scales, "total_upper_bound": total, "full_tokens_per_chunk_upper_bound": full_tokens_per_chunk}


def resource_preflight(*, authorize: bool = False, device: str = "cuda") -> dict[str, Any]:
    require_authorization("preflight", authorize, "EXP109C_ALLOW_PREFLIGHT_GPU")
    require_report_status(RESULTS_ROOT / "READING_AUDIT.json", ("PASS",))
    require_report_status(RESULTS_ROOT / "REPRODUCTION_REPORT.json", ("PASS",), failure_status="REJECTED_REPRODUCTION_GATE")
    if device != "cuda":
        raise GateRejected("REJECTED_RESOURCE_GATE", "EXP-109C does not silently fall back from CUDA")
    if torch is None or not torch.cuda.is_available():
        raise GateRejected("REJECTED_RESOURCE_GATE", "CUDA is unavailable for the required preflight")
    snapshot = local_jina_snapshot()
    if snapshot is None:
        raise GateRejected("REJECTED_INPUT_AUDIT", "local Jina snapshot is missing; download is forbidden")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True, fix_mistral_regex=True)
    # The remote Jina model also constructs tokenizer helpers while loading;
    # propagate the same corrected Mistral regex setting used above.
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True, fix_mistral_regex=True).to(device).half().eval()
    query_lengths = [len(tokenizer(marker_text(str(row.get("question", "")), "query"), truncation=False, add_special_tokens=True)["input_ids"]) for row in load_train().values()]
    doc_lengths: list[int] = []
    for row in read_jsonl(STRUCT_DIR / "chunks.jsonl"):
        doc_lengths.append(len(tokenizer(marker_text(str(row.get("retrieval_text", "")), "document"), truncation=False, add_special_tokens=True)["input_ids"]))
    query_budget = choose_length_by_coverage(query_lengths, (64, 96, 128), .995); doc_budget = choose_length_by_coverage(doc_lengths, (512, 768, 1024), .999)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        query_batch = tokenizer([marker_text("quy định điều kiện áp dụng", "query")], padding=True, truncation=True, max_length=query_budget["selected"], return_tensors="pt").to(device)
        document_batch = tokenizer([marker_text("Điều 1. Nội dung quy định", "document")], padding=True, truncation=True, max_length=doc_budget["selected"], return_tensors="pt").to(device)
        q_hidden = model(**query_batch).last_hidden_state; d_hidden = model(**document_batch).last_hidden_state
        projection = getattr(model, "colbert_projection", None)
        if projection is None:
            try:
                from safetensors.torch import load_file
                weights = load_file(str(next(snapshot.glob("*.safetensors"))), device="cpu"); projection = torch.nn.Linear(weights["linear.weight"].shape[1], weights["linear.weight"].shape[0], bias=False); projection.weight.data.copy_(weights["linear.weight"]); projection = projection.to(device=device, dtype=torch.float16)
            except Exception as exc:
                del model; torch.cuda.empty_cache(); raise GateRejected("REJECTED_RESOURCE_GATE", f"projection load failed: {exc}") from exc
        q = project_and_normalize(q_hidden, projection); d = project_and_normalize(d_hidden, projection); _score, _ = torch_maxsim(q[0], d[0])
        packed, scales = quantize_rows(d[0].detach().cpu().numpy()); restored = torch.as_tensor(dequantize_rows(packed, scales), dtype=torch.float32, device=device); _quant_score, _ = torch_maxsim(q[0], restored)
    peak = int(torch.cuda.max_memory_reserved()); total = int(torch.cuda.get_device_properties(device).total_memory); headroom = (total - peak) / max(total, 1)
    parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    del model; torch.cuda.empty_cache()
    required = estimate_index_bytes(anchors=CONFIG_E_ANCHORS, dimensions=CONFIG_E_STORAGE_DIMENSION)
    disk = disk_record(); disk_pass = disk["free_bytes"] >= int(required["total"] * 1.25)
    private_ram = private_ram_bytes(); available_ram = available_ram_bytes(); quantization_delta = abs(float(_score) - float(_quant_score))
    report = {"schema_version": SCHEMA, "stage": "preflight", "status": "PASS" if headroom >= .10 and parameters < 4_000_000_000 and disk_pass and private_ram < 5.5 * 2**30 and available_ram >= .75 * 2**30 else "REJECTED_RESOURCE_GATE", "device": device, "parameters": parameters, "query_budget": query_budget, "document_budget": doc_budget, "peak_vram_bytes": peak, "vram_total_bytes": total, "vram_headroom_fraction": headroom, "private_ram_bytes": private_ram, "private_ram_target_bytes": int(5.5 * 2**30), "available_ram_bytes": available_ram, "quantized_parent_score_abs_delta": quantization_delta, "disk": {**disk, "estimate": required, "headroom_required_fraction": .25, "pass": disk_pass}, "contract": {"model": MODEL_REPO, "snapshot": MODEL_SNAPSHOT, "dimension": CONFIG_E_STORAGE_DIMENSION, "anchors_per_chunk": CONFIG_E_ANCHORS, "selection_dimension": CONFIG_E_SELECTION_DIMENSION, "selector": "diversity_medoids_farthest_first", "config_e_contract": CONFIG_E_CONTRACT, "encode_dtype": "float16_cuda", "stored_dtype": "int8+float16_scale", "query_dtype": "float16"}, "claim_boundary": "Config E resource feasibility only; no retrieval metric"}
    return write_report(RESULTS_ROOT / "PREFLIGHT.json", report, stage="preflight", success=report["status"] == "PASS")


# ---------------------------------------------------------------------------
# Resumable token index and query store
# ---------------------------------------------------------------------------


def shard_fingerprint(config: Mapping[str, Any]) -> str:
    return content_hash({"schema": SCHEMA, "structural_fingerprint": STRUCTURAL_FINGERPRINT, **dict(config)})


def save_index_shard(shard_dir: Path, shard_number: int, vectors: np.ndarray, scales: np.ndarray, passages: Sequence[Mapping[str, Any]], *, fingerprint: str) -> dict[str, Any]:
    shard_dir.mkdir(parents=True, exist_ok=True); stem = f"shard-{shard_number:05d}"
    vector_path, scale_path, passage_path = shard_dir / f"{stem}.vectors.int8.npy", shard_dir / f"{stem}.scales.f16.npy", shard_dir / f"{stem}.passages.jsonl"
    atomic_npy(vector_path, np.asarray(vectors, dtype=np.int8)); atomic_npy(scale_path, np.asarray(scales, dtype=np.float16)); write_jsonl_atomic(passage_path, passages)
    receipt = {"name": stem, "vectors": vector_path.name, "scales": scale_path.name, "passages": passage_path.name, "count": len(passages), "token_vectors": int(len(vectors)), "fingerprint": fingerprint, "sha256": {path.name: sha256_file(path) for path in (vector_path, scale_path, passage_path)}}
    atomic_json(shard_dir / f"{stem}.json", receipt)
    return receipt


def save_dual_index_shard(shard_dir: Path, shard_number: int, approximate_vectors: np.ndarray, approximate_scales: np.ndarray, full_vectors: np.ndarray, full_scales: np.ndarray, passages: Sequence[Mapping[str, Any]], *, fingerprint: str) -> dict[str, Any]:
    """Commit E4's approximate and full-token stores atomically as one receipt."""
    shard_dir.mkdir(parents=True, exist_ok=True); stem = f"shard-{shard_number:05d}"
    paths = {"vectors": shard_dir / f"{stem}.vectors.int8.npy", "scales": shard_dir / f"{stem}.scales.f16.npy", "full_vectors": shard_dir / f"{stem}.full.vectors.int8.npy", "full_scales": shard_dir / f"{stem}.full.scales.f16.npy", "passages": shard_dir / f"{stem}.passages.jsonl"}
    atomic_npy(paths["vectors"], np.asarray(approximate_vectors, dtype=np.int8)); atomic_npy(paths["scales"], np.asarray(approximate_scales, dtype=np.float16)); atomic_npy(paths["full_vectors"], np.asarray(full_vectors, dtype=np.int8)); atomic_npy(paths["full_scales"], np.asarray(full_scales, dtype=np.float16)); write_jsonl_atomic(paths["passages"], passages)
    receipt = {"name": stem, **{key: value.name for key, value in paths.items()}, "count": len(passages), "token_vectors": int(len(approximate_vectors)), "full_token_vectors": int(len(full_vectors)), "dual_store": True, "fingerprint": fingerprint, "sha256": {path.name: sha256_file(path) for path in paths.values()}}
    atomic_json(shard_dir / f"{stem}.json", receipt)
    return receipt


def verify_index_shard(shard_dir: Path, receipt: Mapping[str, Any], *, fingerprint: str) -> dict[str, Any]:
    if receipt.get("fingerprint") != fingerprint:
        raise GateRejected("REJECTED_STALE_ARTIFACT", f"index shard fingerprint mismatch: {receipt.get('name')}")
    keys = ("vectors", "scales", "passages", "full_vectors", "full_scales") if receipt.get("dual_store") else ("vectors", "scales", "passages")
    for key in keys:
        path = shard_dir / str(receipt[key])
        if not path.exists() or sha256_file(path) != receipt.get("sha256", {}).get(path.name):
            raise GateRejected("REJECTED_STALE_ARTIFACT", f"index shard hash mismatch: {path}")
    return dict(receipt)


def _load_model_for_encoding(*, device: str, allow_download: bool = False) -> tuple[Any, Any, Any]:
    if allow_download:
        raise GateRejected("REJECTED_DOWNLOAD_GATE", "EXP-109C never downloads a model from this namespace")
    snapshot = local_jina_snapshot()
    if snapshot is None:
        raise GateRejected("REJECTED_INPUT_AUDIT", "Jina snapshot missing")
    if torch is None:
        raise RuntimeError("torch is required for Jina encoding")
    from transformers import AutoModel, AutoTokenizer
    from safetensors.torch import load_file
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True, fix_mistral_regex=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True).to(device).half().eval()
    files = list(snapshot.glob("*.safetensors")); weights = load_file(str(files[0]), device="cpu") if files else {}
    if "linear.weight" not in weights or tuple(weights["linear.weight"].shape) != (DIMENSION, 1024):
        raise GateRejected("REJECTED_INPUT_AUDIT", "Jina linear.weight projection shape is not [128,1024]")
    projection = torch.nn.Linear(1024, DIMENSION, bias=False); projection.weight.data.copy_(weights["linear.weight"]); projection = projection.to(device=device, dtype=torch.float16).eval()
    return model, tokenizer, projection


def build_or_load_document_idf(tokenizer: Any, *, batch_size: int = 64) -> tuple[dict[int, float], dict[str, Any]]:
    """Build label-free document-frequency IDF over current structural chunks."""
    output = CACHE_ROOT / "idf"; manifest_path = output / "manifest.json"; values_path = output / "token_idf.json"
    if manifest_path.exists() and values_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("structural_fingerprint") == STRUCTURAL_FINGERPRINT and manifest.get("snapshot") == MODEL_SNAPSHOT and sha256_file(values_path) == manifest.get("files", {}).get(values_path.name):
            return {int(key): float(value) for key, value in read_json(values_path).items()}, manifest
    frequencies: Counter[int] = Counter(); total = 0; batch: list[str] = []; special_ids = set(map(int, getattr(tokenizer, "all_special_ids", ())))
    def consume(texts: Sequence[str]) -> None:
        nonlocal total
        encoded = tokenizer([marker_text(text, "document") for text in texts], add_special_tokens=True, truncation=False, padding=False)
        for token_ids in encoded["input_ids"]:
            frequencies.update(set(int(token_id) for token_id in token_ids if int(token_id) not in special_ids)); total += 1
    for row in iter_chunks():
        batch.append(str(row.get("retrieval_text", "")))
        if len(batch) >= max(1, batch_size): consume(batch); batch = []
    if batch: consume(batch)
    if total != CHUNK_COUNT:
        raise GateRejected("REJECTED_INPUT_AUDIT", f"IDF tokenization count mismatch: {total} != {CHUNK_COUNT}")
    values = {str(token_id): float(math.log((total + 1) / (count + 1)) + 1.0) for token_id, count in sorted(frequencies.items())}
    atomic_json(values_path, values)
    manifest = {"schema_version": SCHEMA, "stage": "document-idf", "structural_fingerprint": STRUCTURAL_FINGERPRINT, "snapshot": MODEL_SNAPSHOT, "chunk_count": total, "files": {values_path.name: sha256_file(values_path)}}; manifest["content_fingerprint"] = content_hash(manifest); atomic_json(manifest_path, manifest)
    return {int(key): float(value) for key, value in values.items()}, manifest


def encode_jina_texts(model: Any, tokenizer: Any, projection: Any, texts: Sequence[str], *, task: str, max_length: int, device: str, dimensions: int = DIMENSION) -> tuple[list[np.ndarray], list[np.ndarray], list[int], int]:
    if torch is None: raise RuntimeError("torch unavailable")
    marked = [marker_text(text, task) for text in texts]; special_ids = set(map(int, getattr(tokenizer, "all_special_ids", ())))
    untruncated = tokenizer(marked, padding=False, truncation=False, add_special_tokens=True)
    raw_lengths = [len(values) for values in untruncated["input_ids"]]
    original_lengths = [sum(int(token_id) not in special_ids for token_id in values) for values in untruncated["input_ids"]]
    encoded = tokenizer(marked, padding=True, truncation=True, max_length=max_length, return_tensors="pt", return_attention_mask=True)
    token_ids = encoded["input_ids"].cpu().numpy(); masks = encoded["attention_mask"].cpu().numpy(); truncated = sum(int(length > max_length) for length in raw_lengths)
    with torch.inference_mode():
        output = model(**{key: value.to(device) for key, value in encoded.items()}, return_dict=True); hidden = getattr(output, "last_hidden_state", output[0]); projected = project_and_normalize(hidden, projection, dimensions=dimensions).cpu().numpy()
    vectors, ids = [], []
    for index in range(len(texts)):
        positions = strip_special_token_positions(token_ids[index], masks[index], special_ids); vectors.append(np.asarray(projected[index, positions], dtype=np.float32)); ids.append(np.asarray(token_ids[index, positions], dtype=np.int64))
    return vectors, ids, original_lengths, truncated


def encode_corpus(*, resume: bool = True, authorize: bool = False, device: str = "cuda", batch_size: int = 8) -> dict[str, Any]:
    require_authorization("encode-corpus", authorize, "EXP109C_ALLOW_CORPUS_ENCODING")
    require_model_use_eligibility()
    preflight = read_json(RESULTS_ROOT / "PREFLIGHT.json") if (RESULTS_ROOT / "PREFLIGHT.json").exists() else {}
    if preflight.get("status") != "PASS": raise GateRejected("REJECTED_PREREQUISITE_GATE", "PASS PREFLIGHT.json is required before corpus encoding")
    require_report_status(RESULTS_ROOT / "CONFIG_E_PRODUCTION_PREFLIGHT.json", ("PASS_CONFIG_E_PRODUCTION_PREFLIGHT",))
    if device != "cuda" or torch is None or not torch.cuda.is_available():
        raise GateRejected("REJECTED_RESOURCE_GATE", "exact corpus encoding requires CUDA; CPU fallback is forbidden")
    document_budget = int(preflight.get("document_budget", {}).get("selected", 0))
    if document_budget not in {512, 768, 1024}: raise GateRejected("REJECTED_PREREQUISITE_GATE", "preflight document tokenizer budget is invalid")
    config = {"model": MODEL_REPO, "snapshot": MODEL_SNAPSHOT, "dimension": CONFIG_E_STORAGE_DIMENSION, "maximum_anchors": CONFIG_E_ANCHORS, "selection_dimension": CONFIG_E_SELECTION_DIMENSION, "selector": "diversity_medoids_farthest_first", "mandatory_policy": "prefix4_tail8", "document_max_length": document_budget, "shard_size": INDEX_SHARD_SIZE, "structural_fingerprint": STRUCTURAL_FINGERPRINT, "scorer_contract": SCORER_CONTRACT, "config_e_contract": CONFIG_E_CONTRACT, "dual_store": {"approximate": "256 medoids x 128d int8", "exact": "all document tokens x 128d int8", "exact_parent_scoring": "streaming_per_query_token_max"}}
    fingerprint = shard_fingerprint(config); output = CACHE_ROOT / "index"; shard_dir = output / "shards"; output.mkdir(parents=True, exist_ok=True)

    def existing_receipts() -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for receipt_path in sorted(shard_dir.glob("shard-*.json")):
            receipt = read_json(receipt_path)
            name = str(receipt.get("name", receipt_path.stem))
            try:
                number = int(name.rsplit("-", 1)[1])
            except (IndexError, ValueError) as exc:
                raise GateRejected("REJECTED_STALE_ARTIFACT", f"invalid index shard receipt: {receipt_path}") from exc
            verify_index_shard(shard_dir, receipt, fingerprint=fingerprint)
            result[number] = receipt
        return result

    if resume and (output / "manifest.json").exists() and (output / "_SUCCESS.json").exists():
        manifest = read_json(output / "manifest.json")
        if manifest.get("fingerprint") != fingerprint:
            raise GateRejected("REJECTED_STALE_ARTIFACT", "completed EXP-109C index uses a different scorer configuration")
        require_success(output, manifest.get("content_fingerprint"))
        for receipt in manifest.get("shards", []):
            verify_index_shard(shard_dir, receipt, fingerprint=fingerprint)
        return manifest

    completed = existing_receipts() if resume else {}
    # If a crash happened after the final shard was committed but before the
    # manifest marker, reconstruct the manifest without loading the model.
    if completed and sum(int(item.get("count", 0)) for item in completed.values()) == CHUNK_COUNT:
        receipts = [completed[number] for number in sorted(completed)]
        manifest = {"schema_version": SCHEMA, "stage": "index", "structural_fingerprint": STRUCTURAL_FINGERPRINT, "config": config, "fingerprint": fingerprint, "shards": receipts, "counts": {"chunks": CHUNK_COUNT, "token_vectors": sum(int(item["token_vectors"]) for item in receipts), "full_token_vectors": sum(int(item["full_token_vectors"]) for item in receipts)}}
        manifest["content_fingerprint"] = content_hash(manifest); atomic_json(output / "manifest.json", manifest)
        write_success(output, stage="encode-corpus", fingerprint=manifest["content_fingerprint"], extra={"shard_count": len(receipts)})
        return manifest

    chunks_path = STRUCT_DIR / "chunks.jsonl"; model, tokenizer, projection = _load_model_for_encoding(device=device)
    receipts: list[dict[str, Any]] = []; total = 0
    try:
        rows_iter = iter_chunks(chunks_path); buffer: list[dict[str, Any]] = []; shard_number = 0
        with tracked_stage("encode-corpus", total=CHUNK_COUNT) as tracker:
            for row in rows_iter:
                buffer.append(row)
                if len(buffer) < INDEX_SHARD_SIZE: continue
                if shard_number in completed:
                    receipt = completed[shard_number]
                    if int(receipt.get("count", -1)) != len(buffer):
                        raise GateRejected("REJECTED_STALE_ARTIFACT", f"index shard count mismatch at shard {shard_number}")
                else:
                    receipt = _encode_chunk_buffer(buffer, shard_dir, shard_number, model, tokenizer, projection, fingerprint, document_max_length=document_budget, device=device, batch_size=batch_size)
                receipts.append(receipt); total += len(buffer); buffer = []; shard_number += 1; tracker.heartbeat(total, emit=False)
            if buffer:
                if shard_number in completed:
                    receipt = completed[shard_number]
                    if int(receipt.get("count", -1)) != len(buffer):
                        raise GateRejected("REJECTED_STALE_ARTIFACT", f"index shard count mismatch at shard {shard_number}")
                else:
                    receipt = _encode_chunk_buffer(buffer, shard_dir, shard_number, model, tokenizer, projection, fingerprint, document_max_length=document_budget, device=device, batch_size=batch_size)
                receipts.append(receipt); total += len(buffer); tracker.heartbeat(total, emit=False)
    finally:
        del model, tokenizer, projection
        if torch is not None and device.startswith("cuda"): torch.cuda.empty_cache()
    manifest = {"schema_version": SCHEMA, "stage": "index", "structural_fingerprint": STRUCTURAL_FINGERPRINT, "config": config, "fingerprint": fingerprint, "shards": receipts, "counts": {"chunks": total, "token_vectors": sum(item["token_vectors"] for item in receipts), "full_token_vectors": sum(item["full_token_vectors"] for item in receipts)}}; manifest["content_fingerprint"] = content_hash(manifest); atomic_json(output / "manifest.json", manifest)
    for receipt in receipts: verify_index_shard(shard_dir, receipt, fingerprint=fingerprint)
    write_success(output, stage="encode-corpus", fingerprint=manifest["content_fingerprint"], extra={"shard_count": len(receipts)})
    write_report(RESULTS_ROOT / "INDEX_MANIFEST.json", {"stage": "index", "status": "PASS", "index_manifest": str((output / "manifest.json").resolve()), "content_fingerprint": manifest["content_fingerprint"], "shard_count": len(receipts), "chunk_count": total, "claim_boundary": "current-corpus token index integrity; no retrieval metric"}, stage="encode-corpus")
    return manifest


def _encode_chunk_buffer(rows: Sequence[Mapping[str, Any]], shard_dir: Path, shard_number: int, model: Any, tokenizer: Any, projection: Any, fingerprint: str, *, document_max_length: int, device: str, batch_size: int) -> dict[str, Any]:
    vectors_parts: list[np.ndarray] = []; scales_parts: list[np.ndarray] = []; full_vectors_parts: list[np.ndarray] = []; full_scales_parts: list[np.ndarray] = []; passages: list[dict[str, Any]] = []; cursor = full_cursor = 0
    for start in range(0, len(rows), max(1, batch_size)):
        batch = rows[start:start + batch_size]; vectors, token_ids, original_lengths, _truncated = encode_jina_texts(model, tokenizer, projection, [str(row.get("retrieval_text", "")) for row in batch], task="document", max_length=document_max_length, device=device)
        for row, value, ids, original in zip(batch, vectors, token_ids, original_lengths):
            keep = config_e_anchor_indices(value)
            packed, scales = quantize_rows(l2_normalize(value[keep])); full_packed, full_scales = quantize_rows(l2_normalize(value)); vectors_parts.append(packed); scales_parts.append(scales); full_vectors_parts.append(full_packed); full_scales_parts.append(full_scales); meta = read_chunk_meta(row); passages.append({"chunk_id": meta.chunk_id, "doc_id": meta.doc_id, "node_id": meta.parent_node_id, "parent_node_id": meta.parent_node_id, "source_start": meta.source_start, "source_end": meta.source_end, "token_start": cursor, "token_end": cursor + len(packed), "full_token_start": full_cursor, "full_token_end": full_cursor + len(full_packed), "original_tokens": int(original), "retained_tokens": len(packed), "full_retained_tokens": len(full_packed), "truncated": bool(original > len(value))}); cursor += len(packed); full_cursor += len(full_packed)
    vectors = np.concatenate(vectors_parts, axis=0) if vectors_parts else np.empty((0, DIMENSION), dtype=np.int8); scales = np.concatenate(scales_parts, axis=0) if scales_parts else np.empty((0,), dtype=np.float16)
    full_vectors = np.concatenate(full_vectors_parts, axis=0) if full_vectors_parts else np.empty((0, DIMENSION), dtype=np.int8); full_scales = np.concatenate(full_scales_parts, axis=0) if full_scales_parts else np.empty((0,), dtype=np.float16)
    return save_dual_index_shard(shard_dir, shard_number, vectors, scales, full_vectors, full_scales, passages, fingerprint=fingerprint)


def encode_queries(*, resume: bool = True, authorize: bool = False, device: str = "cuda", batch_size: int = 16) -> dict[str, Any]:
    require_authorization("encode-queries", authorize, "EXP109C_ALLOW_QUERY_ENCODING")
    require_model_use_eligibility()
    preflight = read_json(RESULTS_ROOT / "PREFLIGHT.json") if (RESULTS_ROOT / "PREFLIGHT.json").exists() else {}
    if preflight.get("status") != "PASS": raise GateRejected("REJECTED_PREREQUISITE_GATE", "PASS PREFLIGHT.json is required before query encoding")
    require_report_status(RESULTS_ROOT / "CONFIG_E_PRODUCTION_PREFLIGHT.json", ("PASS_CONFIG_E_PRODUCTION_PREFLIGHT",))
    if device != "cuda" or torch is None or not torch.cuda.is_available():
        raise GateRejected("REJECTED_RESOURCE_GATE", "exact query encoding requires CUDA; CPU fallback is forbidden")
    query_budget = int(preflight.get("query_budget", {}).get("selected", 0))
    if query_budget not in {64, 96, 128}: raise GateRejected("REJECTED_PREREQUISITE_GATE", "preflight query tokenizer budget is invalid")
    output = CACHE_ROOT / "queries"; output.mkdir(parents=True, exist_ok=True); train = load_train(); qids = sorted(train)
    if resume and (output / "manifest.json").exists() and (output / "_SUCCESS.json").exists():
        manifest = read_json(output / "manifest.json")
        if manifest.get("structural_fingerprint") != STRUCTURAL_FINGERPRINT or manifest.get("snapshot") != MODEL_SNAPSHOT or manifest.get("query_max_length") not in {64, 96, 128} or not (output / "query_token_ids.i64.npy").exists():
            raise GateRejected("REJECTED_STALE_ARTIFACT", "completed EXP-109C query store uses a different input/model snapshot")
        require_success(output, manifest.get("content_fingerprint"))
        return manifest
    model, tokenizer, projection = _load_model_for_encoding(device=device); pieces: list[np.ndarray] = []; id_pieces: list[np.ndarray] = []; offsets = [0]; truncated_total = 0; truncated_qids: list[str] = []; original_lengths: dict[str, int] = {}
    try:
        with tracked_stage("encode-queries", total=len(qids)) as tracker:
            for start in range(0, len(qids), max(1, batch_size)):
                batch_ids = qids[start:start + batch_size]; values, ids, originals, truncated = encode_jina_texts(model, tokenizer, projection, [str(train[qid]["question"]) for qid in batch_ids], task="query", max_length=query_budget, device=device); truncated_total += truncated
                for qid, value, token_ids, original in zip(batch_ids, values, ids, originals):
                    pieces.append(value.astype(np.float16)); id_pieces.append(token_ids.astype(np.int64)); offsets.append(offsets[-1] + len(value)); original_lengths[qid] = int(original)
                    if original > len(value): truncated_qids.append(qid)
                tracker.heartbeat(start + len(batch_ids), emit=False)
    finally:
        del model, tokenizer, projection
        if torch is not None and device.startswith("cuda"): torch.cuda.empty_cache()
    vectors = np.concatenate(pieces, axis=0) if pieces else np.empty((0, DIMENSION), dtype=np.float16); token_ids = np.concatenate(id_pieces, axis=0) if id_pieces else np.empty((0,), dtype=np.int64); atomic_npy(output / "query_vectors.f16.npy", vectors); atomic_npy(output / "query_token_ids.i64.npy", token_ids); atomic_npy(output / "query_offsets.i64.npy", np.asarray(offsets, dtype=np.int64)); atomic_json(output / "qids.json", qids); atomic_json(output / "truncation.json", {"count": truncated_total, "query_ids": truncated_qids, "queries": len(qids), "max_length": query_budget, "original_token_lengths": original_lengths}); manifest = {"schema_version": SCHEMA, "stage": "queries", "structural_fingerprint": STRUCTURAL_FINGERPRINT, "model": MODEL_REPO, "snapshot": MODEL_SNAPSHOT, "dimension": CONFIG_E_STORAGE_DIMENSION, "config_e_contract": CONFIG_E_CONTRACT, "query_max_length": query_budget, "query_count": len(qids), "token_vectors": len(vectors), "files": {name: sha256_file(output / name) for name in ("query_vectors.f16.npy", "query_token_ids.i64.npy", "query_offsets.i64.npy", "qids.json", "truncation.json")}}; manifest["content_fingerprint"] = content_hash(manifest); atomic_json(output / "manifest.json", manifest); write_success(output, stage="encode-queries", fingerprint=manifest["content_fingerprint"]); return manifest


# ---------------------------------------------------------------------------
# Exact score sidecars and frozen inner features
# ---------------------------------------------------------------------------


def load_index_records(index_dir: Path = CACHE_ROOT / "index") -> tuple[dict[str, list[tuple[np.ndarray, dict[str, Any]]]], dict[str, Any]]:
    """Materialize an index for small fixtures only.

    The production scorer uses :func:`load_index_catalog` below so that the
    343k-chunk corpus is never dequantized into one private-RAM object.
    """
    manifest = read_json(index_dir / "manifest.json"); require_success(index_dir, manifest.get("content_fingerprint")); if_fingerprint = manifest.get("fingerprint")
    by_doc: dict[str, list[tuple[np.ndarray, dict[str, Any]]]] = defaultdict(list)
    for receipt in manifest.get("shards", []):
        verified = verify_index_shard(index_dir / "shards", receipt, fingerprint=if_fingerprint); vectors = np.load(index_dir / "shards" / verified["vectors"], mmap_mode="r"); scales = np.load(index_dir / "shards" / verified["scales"], mmap_mode="r"); passages = list(read_jsonl(index_dir / "shards" / verified["passages"]))
        for row in passages:
            start, end = int(row["token_start"]), int(row["token_end"]); by_doc[str(row["doc_id"])].append((dequantize_rows(vectors[start:end], scales[start:end]), row))
    return dict(by_doc), manifest


def load_index_catalog(index_dir: Path = CACHE_ROOT / "index") -> tuple[dict[str, list[IndexedChunkRef]], dict[str, Any]]:
    """Load verified metadata and memmap paths without materializing vectors."""
    manifest = read_json(index_dir / "manifest.json")
    require_success(index_dir, manifest.get("content_fingerprint"))
    fingerprint = manifest.get("fingerprint")
    by_doc: dict[str, list[IndexedChunkRef]] = defaultdict(list)
    for receipt in manifest.get("shards", []):
        verified = verify_index_shard(index_dir / "shards", receipt, fingerprint=fingerprint)
        if not verified.get("dual_store"):
            raise GateRejected("REJECTED_STALE_ARTIFACT", "EXP-109C E4 requires dual-store index receipts")
        vectors_path = index_dir / "shards" / str(verified["vectors"])
        scales_path = index_dir / "shards" / str(verified["scales"])
        full_vectors_path = index_dir / "shards" / str(verified["full_vectors"])
        full_scales_path = index_dir / "shards" / str(verified["full_scales"])
        for row in read_jsonl(index_dir / "shards" / str(verified["passages"])):
            by_doc[str(row["doc_id"])].append(IndexedChunkRef(vectors_path, scales_path, int(row["token_start"]), int(row["token_end"]), dict(row), full_vectors_path, full_scales_path, int(row["full_token_start"]), int(row["full_token_end"])))
    return dict(by_doc), manifest


def score_parent_from_chunks(query_vectors: np.ndarray, chunks: Sequence[tuple[np.ndarray, Mapping[str, Any]]], *, idf_weights: Sequence[float] | None = None, exact_rank: int = 0) -> dict[str, Any]:
    vectors: list[np.ndarray] = []; metas: list[Mapping[str, Any]] = []; scores: list[float] = []; matches: list[np.ndarray] = []
    for document, meta in chunks:
        score, match = numpy_maxsim(query_vectors, document); scores.append(score); matches.append(match); vectors.append(document); metas.append(meta)
    features = latent_condition_features(matches, metas, idf_weights=idf_weights, exact_rank=exact_rank, parent_token_count=sum(int(meta.get("retained_tokens", 0)) for meta in metas)); return {"features": features, "chunk_scores": scores, "selected_chunk_ids": [str(metas[index].get("chunk_id")) for index in sorted(range(len(scores)), key=lambda index: (-scores[index], str(metas[index].get("chunk_id", ""))))[:2]], "selected_chunk_provenance": [{key: metas[index].get(key) for key in ("chunk_id", "source_start", "source_end", "parent_node_id")} for index in sorted(range(len(scores)), key=lambda index: (-scores[index], str(metas[index].get("chunk_id", ""))))[:2]]}


def score_parent_from_chunks_cuda(query_vectors: np.ndarray, chunks: Sequence[tuple[np.ndarray, Mapping[str, Any]]], *, idf_weights: Sequence[float] | None = None, exact_rank: int = 0, device: str = "cuda") -> dict[str, Any]:
    """Exact MaxSim in bounded CUDA blocks; no corpus-sized GPU tensor exists."""
    if torch is None or device != "cuda" or not torch.cuda.is_available():
        raise GateRejected("REJECTED_RESOURCE_GATE", "exact scoring requires CUDA; CPU fallback is forbidden")
    query = torch.as_tensor(query_vectors, dtype=torch.float32, device=device); vectors: list[np.ndarray] = []; metas: list[Mapping[str, Any]] = []; scores: list[float] = []; matches: list[np.ndarray] = []
    with torch.inference_mode():
        for document, meta in chunks:
            if not len(document): continue
            document_cuda = torch.as_tensor(document, dtype=torch.float32, device=device)
            score, match = torch_maxsim(query, document_cuda)
            scores.append(float(score)); matches.append(match.detach().cpu().numpy().astype(np.float32)); vectors.append(document); metas.append(meta)
    if not matches: raise ValueError("parent has no retained token vectors")
    features = latent_condition_features(matches, metas, idf_weights=idf_weights, exact_rank=exact_rank, parent_token_count=sum(int(meta.get("retained_tokens", 0)) for meta in metas))
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], str(metas[index].get("chunk_id", ""))))[:2]
    return {"features": features, "chunk_scores": scores, "selected_chunk_ids": [str(metas[index].get("chunk_id")) for index in order], "selected_chunk_provenance": [{key: metas[index].get(key) for key in ("chunk_id", "source_start", "source_end", "parent_node_id")} for index in order]}


def score_parent_from_match_vectors(matches: Sequence[np.ndarray], metas: Sequence[Mapping[str, Any]], *, idf_weights: Sequence[float] | None = None, exact_rank: int = 0) -> dict[str, Any]:
    """Reference feature aggregation over one independently computed match vector/chunk."""
    if not matches or len(matches) != len(metas):
        raise ValueError("match vectors and metadata must be non-empty and aligned")
    arrays = [np.asarray(match, dtype=np.float32) for match in matches]
    features = latent_condition_features(arrays, metas, idf_weights=idf_weights, exact_rank=exact_rank, parent_token_count=sum(int(meta.get("retained_tokens", 0)) for meta in metas))
    scores = [float(match.mean()) for match in arrays]
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], str(metas[index].get("chunk_id", ""))))[:2]
    return {"features": features, "chunk_scores": scores, "selected_chunk_ids": [str(metas[index].get("chunk_id")) for index in order], "selected_chunk_provenance": [{key: metas[index].get(key) for key in ("chunk_id", "source_start", "source_end", "parent_node_id")} for index in order]}


def _is_cuda_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def batched_chunk_matches_cuda(query_vectors: np.ndarray, chunks: Sequence[tuple[np.ndarray, Mapping[str, Any]]], *, policy: Mapping[str, int], device: str = "cuda") -> tuple[list[np.ndarray], dict[str, float]]:
    """FP32 masked MaxSim for independent chunks, bounded by tokens and temporary bytes.

    Padding is masked to ``-inf`` before max reduction.  A zero-padded row is
    therefore unable to beat a valid negative cosine match.  The output order
    is exactly the input chunk order so feature aggregation remains unchanged.
    """
    if torch is None or F is None or device != "cuda" or not torch.cuda.is_available():
        raise GateRejected("REJECTED_RESOURCE_GATE", "batched exact scoring requires CUDA")
    if not chunks:
        raise ValueError("at least one chunk is required")
    query_np = np.asarray(query_vectors, dtype=np.float32)
    if query_np.ndim != 2 or not len(query_np):
        raise ValueError("query vectors must be non-empty [tokens, dimension]")
    valid = [(np.asarray(document, dtype=np.float32), meta) for document, meta in chunks if len(document)]
    if len(valid) != len(chunks):
        raise ValueError("empty chunk is invalid for batched scorer")
    max_chunks = max(1, int(policy["max_chunks"])); max_tokens = max(1, int(policy["max_document_tokens"])); max_intermediate = max(1, int(policy["max_intermediate_bytes"]))
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes / max(total_bytes, 1) < .15:
        max_chunks = min(max_chunks, 16)
        max_tokens = min(max_tokens, 4_096)
    torch.cuda.synchronize(device); total_started = time.perf_counter(); h2d_seconds = 0.0; gpu_seconds = 0.0; batches = 0; oom_retries = 0
    query = F.normalize(torch.as_tensor(query_np, dtype=torch.float32, device=device), p=2, dim=-1)

    def run_batch(batch: Sequence[tuple[np.ndarray, Mapping[str, Any]]]) -> list[np.ndarray]:
        nonlocal h2d_seconds, gpu_seconds, batches, oom_retries
        width = max(len(document) for document, _meta in batch)
        host = np.zeros((len(batch), width, query_np.shape[1]), dtype=np.float32)
        mask_host = np.zeros((len(batch), width), dtype=np.bool_)
        for index, (document, _meta) in enumerate(batch):
            host[index, :len(document)] = document; mask_host[index, :len(document)] = True
        try:
            torch.cuda.synchronize(device); h2d_started = time.perf_counter()
            documents = torch.as_tensor(host, dtype=torch.float32, device=device)
            mask = torch.as_tensor(mask_host, dtype=torch.bool, device=device)
            torch.cuda.synchronize(device); h2d_seconds += time.perf_counter() - h2d_started
            gpu_started = time.perf_counter(); documents = F.normalize(documents, p=2, dim=-1)
            similarity = torch.einsum("qd,bld->bql", query, documents)
            similarity.masked_fill_(~mask[:, None, :], float("-inf"))
            matches = similarity.amax(dim=-1)
            result = matches.detach().cpu().numpy().astype(np.float32)
            torch.cuda.synchronize(device); gpu_seconds += time.perf_counter() - gpu_started; batches += 1
            return [result[index] for index in range(len(batch))]
        except RuntimeError as exc:
            if not _is_cuda_oom(exc) or len(batch) == 1:
                raise
            oom_retries += 1; torch.cuda.empty_cache()
            midpoint = len(batch) // 2
            return run_batch(batch[:midpoint]) + run_batch(batch[midpoint:])

    output: list[np.ndarray] = []; pending: list[tuple[np.ndarray, Mapping[str, Any]]] = []; pending_tokens = 0; pending_width = 0
    for document, meta in valid:
        prospective_count = len(pending) + 1; prospective_tokens = pending_tokens + len(document); prospective_width = max(pending_width, len(document)); intermediate = prospective_count * len(query_np) * prospective_width * np.dtype(np.float32).itemsize
        if pending and (prospective_count > max_chunks or prospective_tokens > max_tokens or intermediate > max_intermediate):
            output.extend(run_batch(pending)); pending = []; pending_tokens = 0; pending_width = 0
        pending.append((document, meta)); pending_tokens += len(document); pending_width = max(pending_width, len(document))
    if pending:
        output.extend(run_batch(pending))
    torch.cuda.synchronize(device)
    return output, {"batches": float(batches), "chunks": float(len(output)), "h2d_seconds": h2d_seconds, "gpu_seconds": gpu_seconds, "wall_seconds": time.perf_counter() - total_started, "oom_retries": float(oom_retries), "free_vram_bytes_before": float(free_bytes), "total_vram_bytes": float(total_bytes)}


def load_indexed_chunk_block(items: Sequence[tuple[int, str, Mapping[str, Any], IndexedChunkRef]], *, mmap_cache: ShardMmapCache, full_tokens: bool) -> list[tuple[np.ndarray, Mapping[str, Any]]]:
    """Vectorized per-shard dequantization for a bounded reference block."""
    grouped: dict[tuple[Path, Path], list[tuple[int, IndexedChunkRef]]] = defaultdict(list)
    for index, (_position, _doc_id, _candidate, reference) in enumerate(items):
        vector_path = reference.full_vectors_path if full_tokens else reference.vectors_path; scale_path = reference.full_scales_path if full_tokens else reference.scales_path
        if vector_path is None or scale_path is None:
            raise GateRejected("REJECTED_STALE_ARTIFACT", "index reference lacks the required full-token store")
        grouped[(vector_path, scale_path)].append((index, reference))
    output: list[tuple[np.ndarray, Mapping[str, Any]] | None] = [None] * len(items)
    for (vector_path, scale_path), references in grouped.items():
        vectors, scales = mmap_cache.load(vector_path), mmap_cache.load(scale_path); raw_parts: list[np.ndarray] = []; scale_parts: list[np.ndarray] = []; lengths: list[int] = []
        for _index, reference in references:
            start = int(reference.full_start if full_tokens else reference.start); end = int(reference.full_end if full_tokens else reference.end)
            raw_parts.append(np.asarray(vectors[start:end])); scale_parts.append(np.asarray(scales[start:end])); lengths.append(end - start)
        restored = dequantize_rows(np.concatenate(raw_parts, axis=0), np.concatenate(scale_parts, axis=0)); cursor = 0
        for (index, reference), length in zip(references, lengths):
            output[index] = (restored[cursor:cursor + length], reference.metadata); cursor += length
    if any(value is None for value in output):
        raise AssertionError("incomplete indexed chunk block")
    return [value for value in output if value is not None]


def score_candidate_pool_batched_cuda(query_vectors: np.ndarray, candidates: Sequence[Mapping[str, Any]], by_doc: Mapping[str, Sequence[IndexedChunkRef]], *, idf_weights: Sequence[float], mmap_cache: ShardMmapCache, full_tokens: bool, device: str = "cuda", return_chunk_matches: bool = False) -> Any:
    """Score all chunks for a candidate set in shard-grouped, bounded CUDA batches."""
    started = time.perf_counter()
    original: list[tuple[int, str, Mapping[str, Any], IndexedChunkRef]] = []
    for candidate in candidates:
        doc_id = str(candidate["doc_id"])
        for reference in by_doc.get(doc_id, ()):
            original.append((len(original), doc_id, candidate, reference))
    if not original:
        raise GateRejected("REJECTED_SCORE_CONTRACT", "candidate set has no indexed chunks")
    ordered = sorted(original, key=lambda item: (str(item[3].full_vectors_path if full_tokens else item[3].vectors_path), int(item[3].full_start if full_tokens else item[3].start), item[1]))
    policy = EXACT_MICROBATCH if full_tokens else APPROXIMATE_MICROBATCH
    by_position: dict[int, tuple[np.ndarray, Mapping[str, Any]]] = {}; diagnostics: defaultdict[str, float] = defaultdict(float); io_seconds = 0.0; pending: list[tuple[int, str, Mapping[str, Any], IndexedChunkRef]] = []; pending_tokens = 0
    def consume(block: Sequence[tuple[int, str, Mapping[str, Any], IndexedChunkRef]]) -> None:
        nonlocal io_seconds
        io_started = time.perf_counter(); loaded = load_indexed_chunk_block(block, mmap_cache=mmap_cache, full_tokens=full_tokens); io_seconds += time.perf_counter() - io_started
        matches, batch_diagnostics = batched_chunk_matches_cuda(query_vectors, loaded, policy=policy, device=device)
        for key, value in batch_diagnostics.items(): diagnostics[key] += float(value)
        for (position, _doc_id, _candidate, _reference), match, (_document, meta) in zip(block, matches, loaded): by_position[position] = (match, meta)
    for item in ordered:
        reference = item[3]; length = int((reference.full_end - reference.full_start) if full_tokens else (reference.end - reference.start))
        if pending and pending_tokens + length > int(policy["max_document_tokens"]):
            consume(pending); pending = []; pending_tokens = 0
        pending.append(item); pending_tokens += length
    if pending:
        consume(pending)
    grouped_matches: dict[str, list[np.ndarray]] = defaultdict(list); grouped_metas: dict[str, list[Mapping[str, Any]]] = defaultdict(list); grouped_candidate: dict[str, Mapping[str, Any]] = {}
    for position, doc_id, candidate, _reference in original:
        match, meta = by_position[position]; grouped_matches[doc_id].append(match); grouped_metas[doc_id].append(meta); grouped_candidate[doc_id] = candidate
    scored = {doc_id: {"candidate": grouped_candidate[doc_id], "late": score_parent_from_match_vectors(grouped_matches[doc_id], grouped_metas[doc_id], idf_weights=idf_weights, exact_rank=int(grouped_candidate[doc_id]["candidate_rank"]))} for doc_id in grouped_matches}
    kernel_wall_seconds = diagnostics.pop("wall_seconds")
    diagnostics.update({"kernel_wall_seconds": kernel_wall_seconds, "wall_seconds": time.perf_counter() - started, "io_dequant_seconds": io_seconds, "mmap_open_paths": float(mmap_cache.open_paths), "parents": float(len(scored)), "tier": "exact_full_token" if full_tokens else "approximate_medoids"})
    return (scored, diagnostics, dict(grouped_matches)) if return_chunk_matches else (scored, diagnostics)


def reference_chunk_matches_cuda(query_vectors: np.ndarray, chunks: Sequence[tuple[np.ndarray, Mapping[str, Any]]], *, device: str = "cuda") -> list[np.ndarray]:
    """Pre-optimization CUDA reference, kept solely for scorer parity evidence."""
    if torch is None or device != "cuda" or not torch.cuda.is_available():
        raise GateRejected("REJECTED_RESOURCE_GATE", "reference scoring requires CUDA")
    query = torch.as_tensor(query_vectors, dtype=torch.float32, device=device); output: list[np.ndarray] = []
    with torch.inference_mode():
        for document, _meta in chunks:
            _score, match = torch_maxsim(query, torch.as_tensor(document, dtype=torch.float32, device=device))
            output.append(match.detach().cpu().numpy().astype(np.float32))
    return output


def score_candidate_pool_reference_cuda(query_vectors: np.ndarray, candidates: Sequence[Mapping[str, Any]], by_doc: Mapping[str, Sequence[IndexedChunkRef]], *, idf_weights: Sequence[float], full_tokens: bool, device: str = "cuda") -> tuple[dict[str, dict[str, Any]], dict[str, list[np.ndarray]], dict[str, float]]:
    """Intentionally unbatched production-reference scorer for parity/benchmark only."""
    started = time.perf_counter(); io_seconds = 0.0; result: dict[str, dict[str, Any]] = {}; match_rows: dict[str, list[np.ndarray]] = {}
    for candidate in candidates:
        doc_id = str(candidate["doc_id"]); refs = by_doc.get(doc_id, ())
        if not refs:
            continue
        io_started = time.perf_counter(); chunks = [reference.load_full() if full_tokens else reference.load() for reference in refs]; io_seconds += time.perf_counter() - io_started
        matches = reference_chunk_matches_cuda(query_vectors, chunks, device=device)
        result[doc_id] = {"candidate": candidate, "late": score_parent_from_match_vectors(matches, [meta for _document, meta in chunks], idf_weights=idf_weights, exact_rank=int(candidate["candidate_rank"]))}
        match_rows[doc_id] = matches
    return result, match_rows, {"wall_seconds": time.perf_counter() - started, "io_dequant_seconds": io_seconds, "parents": float(len(result)), "tier": "exact_full_token" if full_tokens else "approximate_medoids"}


def build_late_feature_vector(record: Mapping[str, Any], *, names: Sequence[str] = ALLOWED_FEATURES) -> np.ndarray:
    validate_feature_contract(names)
    if any(name not in record for name in names):
        raise ValueError(f"missing feature(s): {sorted(set(names) - set(record))}")
    return np.asarray([float(record[name]) for name in names], dtype=np.float32)


def _source_alias(source: str) -> str:
    return {"vietlegal_e5": "e5", "vietlegal_harrier_0_6b": "harrier", "vnlegal_lal": "lal", "bm25": "bm25"}.get(source, source)


def source_scalar_features(source: str, ranking: Sequence[Mapping[str, Any]], doc_id: str, *, depth: int) -> dict[str, float]:
    """Build the frozen EXP-109B scalar block without labels or IDs as features."""
    alias = _source_alias(source); values = {str(item["doc_id"]): float(item.get("score", 0.0)) for item in ranking[:depth]}
    ranks = {str(item["doc_id"]): int(item.get("rank", index + 1)) for index, item in enumerate(ranking[:depth])}
    present = str(doc_id) in ranks; score = values.get(str(doc_id), 0.0); rank = ranks.get(str(doc_id), depth + 1)
    array = np.asarray(list(values.values()), dtype=np.float64); std = float(array.std()) if len(array) else 0.0; mean = float(array.mean()) if len(array) else 0.0
    ordered = sorted(values.values(), reverse=True)
    def margin(cutoff: int) -> float:
        reference = ordered[min(cutoff - 1, len(ordered) - 1)] if ordered else 0.0
        return float(reference - score) if present else 0.0
    return {f"{alias}_score": score, f"{alias}_rank": float(rank), f"{alias}_recip": 1.0 / rank if present else 0.0, f"{alias}_z": (score - mean) / std if present and std > 1e-12 else 0.0, f"{alias}_margin_rank1": margin(1), f"{alias}_margin_rank5": margin(5), f"{alias}_margin_rank10": margin(10), f"{alias}_present": float(present)}


def build_combined_feature_record(qid: str, doc_id: str, source_rankings: Mapping[str, Sequence[Mapping[str, Any]]], late_features: Mapping[str, float], *, metadata: Mapping[str, Any] | None = None, query_token_length: int = 0, depth: int = CANDIDATE_DEPTH) -> dict[str, Any]:
    """Join 109B scalar features with the required EXP-109C late block.

    ``qid``/``doc_id`` are retained only as row identity for grouping and are
    deliberately excluded by ``build_late_feature_vector`` from the matrix.
    """
    metadata = metadata or {}; record: dict[str, Any] = {"qid": str(qid), "doc_id": str(doc_id)}
    for source in ("vietlegal_e5", "vietlegal_harrier_0_6b", "vnlegal_lal", "bm25"):
        record.update(source_scalar_features(source, source_rankings.get(source, []), doc_id, depth=depth))
    source_ranks = {source: {str(item["doc_id"]): int(item.get("rank", index + 1)) for index, item in enumerate(source_rankings.get(source, [])[:depth])} for source in source_rankings}
    for cutoff, name in ((5, "source_agreement_top5"), (10, "source_agreement_top10"), (20, "source_agreement_top20")):
        record[name] = float(sum(1 for ranks in source_ranks.values() if ranks.get(str(doc_id), depth + 1) <= cutoff))
    dense_scores = sorted([float(item.get("score", 0.0)) for source in ("vietlegal_e5", "vnlegal_lal", "vietlegal_harrier_0_6b") for item in source_rankings.get(source, [])[:depth] if str(item["doc_id"]) == str(doc_id)], reverse=True)
    record["dense_top1_score"] = dense_scores[0] if dense_scores else 0.0; record["dense_top2_score"] = dense_scores[1] if len(dense_scores) > 1 else 0.0; record["dense_top1_top2_gap"] = record["dense_top1_score"] - record["dense_top2_score"]
    record["parent_chunk_count"] = float(metadata.get("parent_chunk_count", 0)); record["parent_token_length"] = float(metadata.get("parent_token_length", 0)); record["query_token_length"] = float(query_token_length)
    record.update({name: float(late_features[name]) for name in LATE_FEATURES})
    return record


def validate_feature_contract(names: Sequence[str]) -> None:
    if tuple(names) != ALLOWED_FEATURES:
        raise ValueError("EXP-109C feature schema/order mismatch")
    if any(any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS) for name in names):
        raise ValueError("identifier/label-dependent feature is forbidden")


def deterministic_inner_selection(qids: Sequence[str], folds: Mapping[str, Sequence[str]], qrow: Mapping[str, int], offsets: np.ndarray, *, outer: str, target_query_count: int | None) -> tuple[list[str], dict[str, Any]]:
    """Pre-score, label-free stratified sample with equal F1-F4 allocation."""
    inner_folds = [name for name in sorted(folds) if name != outer]
    if target_query_count is None:
        selected = sorted(qids); policy = "full_strict_inner"
    else:
        if target_query_count not in {1024, 2048} or target_query_count % len(inner_folds):
            raise ValueError("pilot query count must be 1024 or 2048 and divide four inner folds")
        per_fold = target_query_count // len(inner_folds); allowed = set(qids); selected = []
        for fold in inner_folds:
            fold_qids = [str(qid) for qid in folds[fold] if str(qid) in allowed]
            ordered = sorted(fold_qids, key=lambda qid: (int(offsets[qrow[qid] + 1] - offsets[qrow[qid]]), qid)); chosen: list[str] = []
            for index in range(8):
                start = round(index * len(ordered) / 8); end = round((index + 1) * len(ordered) / 8); take = per_fold // 8 + int(index < per_fold % 8)
                chosen.extend(sorted(ordered[start:end], key=lambda qid: content_hash({"exp": "109c-d100", "outer": outer, "fold": fold, "qid": qid}))[:take])
            if len(chosen) != per_fold: raise GateRejected("REJECTED_SCORE_CONTRACT", f"pilot allocation failed for {fold}")
            selected.extend(chosen)
        selected = sorted(selected); policy = f"d100_stratified_hash_{target_query_count}"
    payload = {"schema_version": SCHEMA, "outer": outer, "candidate_source_depth": CANDIDATE_DEPTH, "policy": policy, "query_ids": selected, "per_fold": {name: sum(qid in set(map(str, folds[name])) for qid in selected) for name in inner_folds}}
    payload["query_selection_fingerprint"] = content_hash(payload)
    return selected, payload


def score_inner(*, outer: str = FOLD0, resume: bool = True, authorize: bool = False, device: str = "cuda", target_query_count: int | None = None) -> dict[str, Any]:
    require_authorization("score-inner", authorize, "EXP109C_ALLOW_EXACT_SCORING")
    require_model_use_eligibility()
    require_current_candidate_report(outer=outer)
    benchmark = require_report_status(RESULTS_ROOT / "BATCHED_SCORER_BENCHMARK.json", ("PASS_BATCHED_SCORER_PROMOTION",))
    if benchmark.get("scorer_implementation_contract") != SCORER_IMPLEMENTATION_CONTRACT:
        raise GateRejected("REJECTED_STALE_ARTIFACT", "batched scorer benchmark was produced by a different implementation contract")
    if device != "cuda" or torch is None or not torch.cuda.is_available(): raise GateRejected("REJECTED_RESOURCE_GATE", "exact scoring requires CUDA; CPU fallback is forbidden")
    # Establish the durable run log before verifying/loading the large dual
    # index.  That setup can take minutes and must not look like an invisible
    # or unobservable worker to the experiment owner.
    tracker = RunTracker("score-inner", outer=outer, total=0)
    tracker.log("stage=score-inner loading query/index/anchor prerequisites", emit=True)
    query_dir = CACHE_ROOT / "queries"; index_dir = CACHE_ROOT / "index"; query_manifest = read_json(query_dir / "manifest.json") if (query_dir / "manifest.json").exists() else {}; require_success(query_dir); by_doc, index_manifest = load_index_catalog(index_dir); qvectors = np.load(query_dir / "query_vectors.f16.npy", mmap_mode="r"); qtoken_ids = np.load(query_dir / "query_token_ids.i64.npy", mmap_mode="r"); offsets = np.load(query_dir / "query_offsets.i64.npy", mmap_mode="r"); qids = [str(value) for value in read_json(query_dir / "qids.json")]; qrow = {qid: index for index, qid in enumerate(qids)}; source_rows = load_exp109b_anchor_sources(outer=outer); anchor_predictions, anchor_provenance = _e4_anchor_predictions(outer=outer); parent_metadata = load_parent_metadata(); idf_path = CACHE_ROOT / "idf" / "token_idf.json"; idf_manifest_path = CACHE_ROOT / "idf" / "manifest.json"
    if index_manifest.get("config", {}).get("config_e_contract") != CONFIG_E_CONTRACT or query_manifest.get("config_e_contract") != CONFIG_E_CONTRACT:
        raise GateRejected("REJECTED_STALE_ARTIFACT", "index/query stores do not satisfy the locked Config E 256x128d contract")
    if not idf_path.exists() or not idf_manifest_path.exists(): raise GateRejected("REJECTED_PREREQUISITE_GATE", "current-corpus IDF artifact is required for exact scoring")
    idf_manifest = read_json(idf_manifest_path)
    if sha256_file(idf_path) != idf_manifest.get("files", {}).get(idf_path.name) or idf_manifest.get("structural_fingerprint") != STRUCTURAL_FINGERPRINT: raise GateRejected("REJECTED_STALE_ARTIFACT", "current-corpus scorer IDF artifact is stale")
    idf = {int(key): float(value) for key, value in read_json(idf_path).items()}; output = late_score_cache_dir(outer); shard_dir = output / "shards"; output.mkdir(parents=True, exist_ok=True); mmap_cache = ShardMmapCache(); performance: defaultdict[str, float] = defaultdict(float)

    def score_query(qid: str) -> dict[str, Any]:
        start, end = int(offsets[qrow[qid]]), int(offsets[qrow[qid] + 1]); q = np.asarray(qvectors[start:end], dtype=np.float32); query_idf = np.asarray([idf.get(int(token_id), 1.0) for token_id in qtoken_ids[start:end]], dtype=np.float32); candidates = build_candidate_union(source_rows[qid], depth=CANDIDATE_DEPTH)
        approximate, approximate_diagnostics = score_candidate_pool_batched_cuda(q, candidates, by_doc, idf_weights=query_idf, mmap_cache=mmap_cache, full_tokens=False, device=device)
        accumulate_numeric_diagnostics(performance, "approximate", approximate_diagnostics)
        if not approximate:
            raise GateRejected("REJECTED_SCORE_CONTRACT", f"no indexed candidates for query {qid}")
        if qid not in anchor_predictions:
            raise GateRejected("REJECTED_ANCHOR_REPRODUCTION", f"strict E4 LambdaMART anchor is missing query {qid}")
        approximate_scores = {doc_id: float(value["late"]["features"]["li_full_parent_union_mean"]) for doc_id, value in approximate.items()}
        exact_ids = set(exact_refinement_set(approximate_scores, anchor_predictions[qid]))
        exact_candidates = [approximate[doc_id]["candidate"] for doc_id in sorted(exact_ids) if doc_id in approximate]
        exact, exact_diagnostics = score_candidate_pool_batched_cuda(q, exact_candidates, by_doc, idf_weights=query_idf, mmap_cache=mmap_cache, full_tokens=True, device=device)
        accumulate_numeric_diagnostics(performance, "exact", exact_diagnostics)
        scores = []
        for doc_id, value in approximate.items():
            candidate = value["candidate"]; late = value["late"]; tier = "approximate_medoids"
            if doc_id in exact_ids:
                late = exact[doc_id]["late"]
                tier = "exact_full_token"
            late["features"] = build_combined_feature_record(qid, doc_id, source_rows[qid], late["features"], metadata=parent_metadata.get(doc_id), query_token_length=len(q), depth=CANDIDATE_DEPTH)
            scores.append({"doc_id": doc_id, "candidate_rank": int(candidate["candidate_rank"]), "source_presence": int(candidate.get("source_presence", 0)), "scoring_tier": tier, **late})
        scores.sort(key=lambda item: (-float(item["features"]["li_two_chunk_union_mean"]), str(item["doc_id"])))
        for rank, item in enumerate(scores, 1): item["features"]["li_exact_rank_within_candidate_pool"] = rank
        return {"qid": qid, "query_tokens": len(q), "scores": scores, "exact_set_size": len(exact_ids), "structural_fingerprint": STRUCTURAL_FINGERPRINT, "index_fingerprint": index_manifest.get("content_fingerprint")}

    folds, _ = load_folds(); inner_qids = {str(qid) for name, values in folds.items() if name != outer for qid in values}
    available_qids = [qid for qid in sorted(source_rows) if qid in qrow and qid in inner_qids]; qid_list, selection = deterministic_inner_selection(available_qids, folds, qrow, offsets, outer=outer, target_query_count=target_query_count); selection_dir = CACHE_ROOT / "score_selections" / outer; selection_dir.mkdir(parents=True, exist_ok=True); atomic_json(selection_dir / f"{selection['query_selection_fingerprint']}.json", selection); tracker.total = len(qid_list); tracker.update(); tracker.log(f"stage=score-inner prerequisites ready total={tracker.total} selection={selection['query_selection_fingerprint']}", emit=True); receipts: list[dict[str, Any]] = []
    completed: dict[str, dict[str, Any]] = {}
    if resume:
        for receipt_path in sorted(shard_dir.glob("scores-*.json")):
            receipt = read_json(receipt_path); name = str(receipt.get("name", receipt_path.stem))
            path = shard_dir / name
            if not path.exists() or receipt.get("sha256") != sha256_file(path):
                raise GateRejected("REJECTED_STALE_ARTIFACT", f"score shard hash mismatch: {path}")
            if receipt.get("scorer_implementation_contract") != SCORER_IMPLEMENTATION_CONTRACT:
                raise GateRejected("REJECTED_STALE_ARTIFACT", f"score shard was produced by a different scorer implementation: {receipt_path}")
            shard_qids = tuple(map(str, receipt.get("query_ids", ())))
            if receipt.get("candidate_source_depth") != CANDIDATE_DEPTH or not shard_qids or len(shard_qids) != int(receipt.get("count", -1)) or set(shard_qids) - set(qid_list) or any(qid in completed for qid in shard_qids):
                raise GateRejected("REJECTED_STALE_ARTIFACT", f"score shard query-selection contract mismatch: {receipt_path}")
            for qid in shard_qids: completed[qid] = receipt
    if resume and (output / "manifest.json").exists() and (output / "_SUCCESS.json").exists():
        previous = read_json(output / "manifest.json")
        if previous.get("scorer_implementation_contract") != SCORER_IMPLEMENTATION_CONTRACT:
            raise GateRejected("REJECTED_STALE_ARTIFACT", "completed late-score cache was produced by a different scorer implementation contract")
        require_success(output, previous.get("content_fingerprint"))
        if set(qid_list) == set(completed): return previous
    with tracked_stage("score-inner", outer=outer, total=len(qid_list), tracker=tracker) as tracker:
        for receipt in {id(value): value for value in completed.values()}.values(): receipts.append(receipt)
        pending_qids = [qid for qid in qid_list if qid not in completed]
        tracker.heartbeat(len(qid_list) - len(pending_qids), emit=True)
        for shard_number, start in enumerate(range(0, len(pending_qids), SCORE_QUERY_SHARD_SIZE)):
            shard_qids = pending_qids[start:start + SCORE_QUERY_SHARD_SIZE]
            shard_rows = [score_query(qid) for qid in shard_qids]
            shard_id = content_hash({"contract": SCORER_IMPLEMENTATION_CONTRACT, "qids": shard_qids})[:16]
            path = shard_dir / f"scores-{shard_id}.jsonl"; count = write_jsonl_atomic(path, shard_rows); receipt = {"name": path.name, "count": count, "query_ids": shard_qids, "query_selection_fingerprint": selection["query_selection_fingerprint"], "candidate_source_depth": CANDIDATE_DEPTH, "index_fingerprint": index_manifest.get("content_fingerprint"), "sha256": sha256_file(path), "scorer_implementation_contract": SCORER_IMPLEMENTATION_CONTRACT}; atomic_json(shard_dir / f"scores-{shard_id}.json", receipt); receipts.append(receipt); tracker.heartbeat(len(qid_list) - len(pending_qids) + start + len(shard_qids), emit=False)
    manifest = {"schema_version": SCHEMA, "stage": "score-inner", "outer": outer, "query_count": len(qid_list), "query_selection": selection, "candidate_depth": CANDIDATE_DEPTH, "query_shard_size": SCORE_QUERY_SHARD_SIZE, "scorer_implementation_contract": SCORER_IMPLEMENTATION_CONTRACT, "scoring_policy": "Config E approximate medoids all candidates; exact full-token top16 approximate union top16 strict EXP-109B LambdaMART", "anchor_provenance": anchor_provenance, "shards": receipts, "index_fingerprint": index_manifest.get("content_fingerprint"), "structural_fingerprint": STRUCTURAL_FINGERPRINT, "performance": dict(performance), "status": "PASS", "claim_boundary": "frozen late score/features; no Fold-0 selection claim"}; manifest["content_fingerprint"] = content_hash(manifest); atomic_json(output / "manifest.json", manifest)
    for receipt in receipts:
        path = shard_dir / receipt["name"]
        if receipt.get("sha256") != sha256_file(path): raise GateRejected("REJECTED_STALE_ARTIFACT", f"score shard hash mismatch: {path}")
    write_success(output, stage="score-inner", fingerprint=manifest["content_fingerprint"])
    completed_rows, _ = load_cached_score_rows(outer=outer); report = {"schema_version": SCHEMA, "stage": "score-inner", "outer": outer, "query_count": len(qid_list), "candidate_depth": CANDIDATE_DEPTH, "shards": receipts, "index_fingerprint": index_manifest.get("content_fingerprint"), "structural_fingerprint": STRUCTURAL_FINGERPRINT, "idf_fingerprint": idf_manifest.get("content_fingerprint"), "length_lottery_diagnostics": length_lottery_diagnostics(completed_rows, parent_metadata), "status": "PASS", "claim_boundary": "frozen late score/features; no Fold-0 selection claim"}; write_report(RESULTS_ROOT / "INNER_LATE_SCORE_REPORT.json", report, stage="score-inner")
    return manifest


def _rank_late_scores(values: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(values, key=lambda doc_id: (-float(values[doc_id]["late"]["features"]["li_two_chunk_union_mean"]), str(doc_id)))


def accumulate_numeric_diagnostics(target: dict[str, float], prefix: str, diagnostics: Mapping[str, Any]) -> None:
    """Keep descriptive diagnostic labels out of numeric performance totals."""
    for key, value in diagnostics.items():
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            name = f"{prefix}_{key}"
            target[name] = target.get(name, 0.0) + float(value)


def _pool_parity(reference: Mapping[str, Mapping[str, Any]], reference_matches: Mapping[str, Sequence[np.ndarray]], batched: Mapping[str, Mapping[str, Any]], batched_matches: Mapping[str, Sequence[np.ndarray]]) -> dict[str, Any]:
    documents_equal = set(reference) == set(batched); max_match_delta = 0.0; max_feature_delta = 0.0; feature_equal = documents_equal; match_equal = documents_equal
    if documents_equal:
        for doc_id in reference:
            left_matches, right_matches = reference_matches[doc_id], batched_matches[doc_id]
            if len(left_matches) != len(right_matches): match_equal = False; continue
            for left, right in zip(left_matches, right_matches):
                max_match_delta = max(max_match_delta, float(np.max(np.abs(np.asarray(left) - np.asarray(right)))))
            for name in LATE_FEATURES:
                delta = abs(float(reference[doc_id]["late"]["features"][name]) - float(batched[doc_id]["late"]["features"][name])); max_feature_delta = max(max_feature_delta, delta)
    approximate_ranking_equal = documents_equal and _rank_late_scores(reference) == _rank_late_scores(batched)
    return {"documents_equal": documents_equal, "match_vectors_equal_shape": match_equal, "max_abs_match_delta": max_match_delta, "max_abs_feature_delta": max_feature_delta, "approximate_ranking_equal": approximate_ranking_equal, "strict_pass": documents_equal and match_equal and approximate_ranking_equal and max_match_delta <= 1e-6 and max_feature_delta <= 1e-6}


def run_batched_scorer_benchmark(*, outer: str = FOLD0, authorize: bool = False, device: str = "cuda", query_count: int = 16) -> dict[str, Any]:
    """Promotion gate for mathematically equivalent batched score implementation."""
    require_authorization("benchmark-batched-scorer", authorize, "EXP109C_ALLOW_EXACT_SCORING")
    require_model_use_eligibility(); require_current_candidate_report(outer=outer)
    if device != "cuda" or torch is None or not torch.cuda.is_available(): raise GateRejected("REJECTED_RESOURCE_GATE", "batched scorer benchmark requires CUDA")
    query_dir = CACHE_ROOT / "queries"; index_dir = CACHE_ROOT / "index"; require_success(query_dir); by_doc, index_manifest = load_index_catalog(index_dir)
    qvectors = np.load(query_dir / "query_vectors.f16.npy", mmap_mode="r"); qtoken_ids = np.load(query_dir / "query_token_ids.i64.npy", mmap_mode="r"); offsets = np.load(query_dir / "query_offsets.i64.npy", mmap_mode="r"); qids = [str(value) for value in read_json(query_dir / "qids.json")]; qrow = {qid: index for index, qid in enumerate(qids)}
    source_rows = load_exp109b_anchor_sources(outer=outer); anchors, provenance = _e4_anchor_predictions(outer=outer); idf_path = CACHE_ROOT / "idf" / "token_idf.json"; idf_manifest_path = CACHE_ROOT / "idf" / "manifest.json"
    if not idf_path.exists() or not idf_manifest_path.exists(): raise GateRejected("REJECTED_PREREQUISITE_GATE", "current-corpus IDF artifact is required")
    idf_manifest = read_json(idf_manifest_path); idf = {int(key): float(value) for key, value in read_json(idf_path).items()}
    if sha256_file(idf_path) != idf_manifest.get("files", {}).get(idf_path.name): raise GateRejected("REJECTED_STALE_ARTIFACT", "IDF artifact hash mismatch")
    folds, _ = load_folds(); inner = {str(qid) for name, values in folds.items() if name != outer for qid in values}; available = [qid for qid in sorted(source_rows) if qid in inner and qid in qrow and qid in anchors]
    if len(available) < query_count: raise GateRejected("REJECTED_SCORE_CONTRACT", "not enough strict-inner queries for benchmark")
    lengths = sorted(available, key=lambda qid: (int(offsets[qrow[qid] + 1] - offsets[qrow[qid]]), qid)); positions = np.linspace(0, len(lengths) - 1, num=query_count, dtype=int).tolist(); selected = [lengths[position] for position in positions]
    report_rows: list[dict[str, Any]] = []; reference_seconds = 0.0; batched_seconds = 0.0; mmap_cache = ShardMmapCache()
    with tracked_stage("benchmark-batched-scorer", outer=outer, total=len(selected)) as tracker:
        for completed, qid in enumerate(selected, 1):
            start, end = int(offsets[qrow[qid]]), int(offsets[qrow[qid] + 1]); query = np.asarray(qvectors[start:end], dtype=np.float32); weights = np.asarray([idf.get(int(token_id), 1.0) for token_id in qtoken_ids[start:end]], dtype=np.float32); candidates = build_candidate_union(source_rows[qid], depth=CANDIDATE_DEPTH)
            ref_approx, ref_approx_matches, ref_approx_timing = score_candidate_pool_reference_cuda(query, candidates, by_doc, idf_weights=weights, full_tokens=False, device=device)
            batched_approx, batch_approx_timing, batch_approx_matches = score_candidate_pool_batched_cuda(query, candidates, by_doc, idf_weights=weights, mmap_cache=mmap_cache, full_tokens=False, device=device, return_chunk_matches=True)
            approx_parity = _pool_parity(ref_approx, ref_approx_matches, batched_approx, batch_approx_matches)
            ref_scores = {doc_id: float(value["late"]["features"]["li_full_parent_union_mean"]) for doc_id, value in ref_approx.items()}; batch_scores = {doc_id: float(value["late"]["features"]["li_full_parent_union_mean"]) for doc_id, value in batched_approx.items()}; ref_exact_ids = exact_refinement_set(ref_scores, anchors[qid]); batch_exact_ids = exact_refinement_set(batch_scores, anchors[qid])
            ref_exact_candidates = [ref_approx[doc_id]["candidate"] for doc_id in ref_exact_ids]; batch_exact_candidates = [batched_approx[doc_id]["candidate"] for doc_id in batch_exact_ids]
            ref_exact, ref_exact_matches, ref_exact_timing = score_candidate_pool_reference_cuda(query, ref_exact_candidates, by_doc, idf_weights=weights, full_tokens=True, device=device)
            batch_exact, batch_exact_timing, batch_exact_matches = score_candidate_pool_batched_cuda(query, batch_exact_candidates, by_doc, idf_weights=weights, mmap_cache=mmap_cache, full_tokens=True, device=device, return_chunk_matches=True)
            exact_parity = _pool_parity(ref_exact, ref_exact_matches, batch_exact, batch_exact_matches)
            ref_final = dict(ref_approx); ref_final.update(ref_exact); batch_final = dict(batched_approx); batch_final.update(batch_exact)
            final_ranking_equal = _rank_late_scores(ref_final) == _rank_late_scores(batch_final)
            reference_seconds += ref_approx_timing["wall_seconds"] + ref_exact_timing["wall_seconds"]; batched_seconds += batch_approx_timing["wall_seconds"] + batch_exact_timing["wall_seconds"]
            report_rows.append({"qid": qid, "query_tokens": len(query), "approximate": approx_parity, "exact_refinement_set_equal": ref_exact_ids == batch_exact_ids, "exact_refinement_set": ref_exact_ids, "exact": exact_parity, "final_ranking_equal": final_ranking_equal, "top5_equal": _rank_late_scores(ref_final)[:5] == _rank_late_scores(batch_final)[:5], "timing": {"reference_seconds": ref_approx_timing["wall_seconds"] + ref_exact_timing["wall_seconds"], "batched_seconds": batch_approx_timing["wall_seconds"] + batch_exact_timing["wall_seconds"], "batched_approximate": batch_approx_timing, "batched_exact": batch_exact_timing}})
            tracker.heartbeat(completed, emit=True)
    strict = all(row["approximate"]["strict_pass"] and row["exact"]["strict_pass"] and row["exact_refinement_set_equal"] and row["final_ranking_equal"] and row["top5_equal"] for row in report_rows); speedup = reference_seconds / max(batched_seconds, 1e-12); report = {"schema_version": SCHEMA, "stage": "benchmark-batched-scorer", "outer": outer, "status": "PASS_BATCHED_SCORER_PROMOTION" if strict and speedup >= 2.0 else "REJECTED_BATCHED_SCORER_PROMOTION", "scorer_implementation_contract": SCORER_IMPLEMENTATION_CONTRACT, "fixture_policy": "16 deterministic strict-inner queries stratified by query token length; all candidate parents, approximate and full exact tiers", "query_count": len(selected), "anchor_provenance": provenance, "index_fingerprint": index_manifest.get("content_fingerprint"), "reference_seconds": reference_seconds, "batched_seconds": batched_seconds, "speedup": speedup, "minimum_speedup": 2.0, "mmap_open_paths": mmap_cache.open_paths, "rows": report_rows, "gate": {"per_chunk_match_and_feature_parity": strict, "exact_set_equality": all(row["exact_refinement_set_equal"] for row in report_rows), "final_ranking_parity": all(row["final_ranking_equal"] for row in report_rows), "top5_agreement": all(row["top5_equal"] for row in report_rows), "speedup_ge_2": speedup >= 2.0}, "claim_boundary": "implementation parity/performance evidence only; not a retrieval metric"}
    return write_report(RESULTS_ROOT / "BATCHED_SCORER_BENCHMARK.json", report, stage="benchmark-batched-scorer", success=report["status"] == "PASS_BATCHED_SCORER_PROMOTION")


def late_score_cache_dir(outer: str, *, target_query_count: int | None = None) -> Path:
    """Single versioned D100 cache, expanded by query ID from 1024 to full."""
    return CACHE_ROOT / "late_scores" / outer / SCORER_IMPLEMENTATION_CONTRACT / "incremental"


def load_cached_score_rows(*, outer: str = FOLD0, target_query_count: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only hash-verified query score shards."""
    output = late_score_cache_dir(outer)
    manifest = read_json(output / "manifest.json") if (output / "manifest.json").exists() else {}
    require_success(output, manifest.get("content_fingerprint"))
    rows: list[dict[str, Any]] = []
    for receipt in manifest.get("shards", []):
        path = output / "shards" / str(receipt["name"])
        if not path.exists() or sha256_file(path) != receipt.get("sha256"):
            raise GateRejected("REJECTED_STALE_ARTIFACT", f"score shard hash mismatch: {path}")
        rows.extend(read_jsonl(path))
    if len(rows) != int(manifest.get("query_count", len(rows))):
        raise GateRejected("REJECTED_STALE_ARTIFACT", "score shard query count mismatch")
    return rows, manifest


def length_lottery_diagnostics(score_rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    full_scores: list[float] = []; two_scores: list[float] = []; chunk_counts: list[float] = []; token_counts: list[float] = []; by_mode: dict[str, list[float]] = defaultdict(list)
    for query in score_rows:
        for row in query.get("scores", []):
            doc_id = str(row["doc_id"]); features = row.get("features", {}); meta = metadata.get(doc_id, {})
            full_scores.append(float(features.get("li_full_parent_union_mean", 0.0))); two_scores.append(float(features.get("li_two_chunk_union_mean", 0.0))); chunk_counts.append(float(meta.get("parent_chunk_count", features.get("li_parent_chunk_count", 0.0)))); token_counts.append(float(meta.get("parent_token_length", features.get("li_parent_token_count", 0.0)))); by_mode[str(meta.get("parse_mode", "unknown"))].append(float(features.get("li_full_parent_union_mean", 0.0)))
    def corr(left: Sequence[float], right: Sequence[float]) -> float | None:
        return spearman_correlation(left, right) if len(left) > 2 and len(left) == len(right) else None
    return {"records": len(full_scores), "spearman": {"full_parent_vs_chunk_count": corr(full_scores, chunk_counts), "full_parent_vs_token_count": corr(full_scores, token_counts), "two_chunk_vs_chunk_count": corr(two_scores, chunk_counts), "two_chunk_vs_token_count": corr(two_scores, token_counts)}, "full_parent_mean_by_parse_mode": {mode: float(np.mean(values)) for mode, values in sorted(by_mode.items())}}


def fidelity_pilot_report(*, full_scores: Sequence[float], compressed_scores: Sequence[float], positive_pairs: Sequence[tuple[int, int]] = (), parent_lengths: Sequence[int] | None = None, retry_full_scores: Sequence[float] | None = None, retry_compressed_scores: Sequence[float] | None = None, report_path: Path | None = None) -> dict[str, Any]:
    """Evaluate the locked 96-anchor pilot and, once only, the 128 retry."""
    attempts: list[dict[str, Any]] = []
    first = fidelity_metrics(full_scores, compressed_scores, positive_pairs=positive_pairs, parent_lengths=parent_lengths)
    first_gate = fidelity_gate(first, anchors=MAX_ANCHORS_PRIMARY)
    attempts.append({"anchors": MAX_ANCHORS_PRIMARY, "metrics": first, "gate": first_gate})
    chosen = attempts[0]
    if not first_gate["pass"]:
        if retry_full_scores is None or retry_compressed_scores is None:
            report = {"schema_version": SCHEMA, "stage": "fidelity-pilot", "status": "REJECTED_COMPRESSION_FIDELITY_GATE", "attempts": attempts, "retry": {"allowed": True, "executed": False, "reason": "128-anchor retry input was not supplied"}, "claim_boundary": "compression fidelity only; no retrieval metric"}
            return write_report(report_path, report, stage="fidelity-pilot") if report_path is not None else report
        retry = fidelity_metrics(retry_full_scores, retry_compressed_scores, positive_pairs=positive_pairs, parent_lengths=parent_lengths)
        retry_gate = fidelity_gate(retry, anchors=MAX_ANCHORS_RETRY)
        attempts.append({"anchors": MAX_ANCHORS_RETRY, "metrics": retry, "gate": retry_gate})
        chosen = attempts[-1]
    report = {"schema_version": SCHEMA, "stage": "fidelity-pilot", "status": "PASS_FIDELITY" if chosen["gate"]["pass"] else "REJECTED_COMPRESSION_FIDELITY_GATE", "attempts": attempts, "selected_attempt": chosen["anchors"], "retry": {"allowed": True, "executed": len(attempts) == 2}, "claim_boundary": "compression fidelity only; no retrieval metric"}
    return write_report(report_path, report, stage="fidelity-pilot") if report_path is not None else report


def z_normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    values = np.asarray(list(scores.values()), dtype=np.float64); mean = float(values.mean()) if len(values) else 0.0; std = float(values.std()) if len(values) else 0.0; return {key: float((value - mean) / std) if std > 1e-12 else 0.0 for key, value in scores.items()}


def residual_blend(anchor: Mapping[str, float], late: Mapping[str, float], alpha: float) -> dict[str, float]:
    if alpha not in {0.15, 0.30, 0.45}: raise ValueError("EXP-109C residual blend alpha is locked to {0.15,0.30,0.45}")
    a, l = z_normalize_scores(anchor), z_normalize_scores(late); ids = set(a) | set(l); return {doc_id: (1 - alpha) * a.get(doc_id, 0.0) + alpha * l.get(doc_id, 0.0) for doc_id in ids}


def select_six_negatives(candidate_rows: Sequence[Mapping[str, Any]], gold_ids: Iterable[str], *, epoch: int = 0) -> list[str]:
    gold = {str(value) for value in gold_ids}
    pool = {str(row["doc_id"]) for row in candidate_rows}
    if not pool:
        return []
    eligible = [row for row in candidate_rows if str(row["doc_id"]) not in gold]; selected: list[str] = []
    def add(rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            doc_id = str(row["doc_id"])
            if doc_id not in gold and doc_id in pool and doc_id not in selected:
                selected.append(doc_id)
            if len(selected) >= 6: return
    add(sorted(eligible, key=lambda row: (float(row.get("anchor_rank", row.get("candidate_rank", 999))), str(row["doc_id"])))[:2])
    rotating = sorted([row for row in eligible if 6 <= int(row.get("anchor_rank", row.get("candidate_rank", 999))) <= 32], key=lambda row: (int(row.get("anchor_rank", row.get("candidate_rank", 999))), str(row["doc_id"])))
    if rotating: add(rotating[epoch % len(rotating):] + rotating[: epoch % len(rotating)])
    add(sorted(eligible, key=lambda row: (-float(row.get("late_coverage", row.get("late_score", -math.inf))), float(row.get("anchor_score", math.inf)), str(row["doc_id"]))))
    source_confusers = [row for row in eligible if int(row.get("source_presence", row.get("channel_count", 0))) >= 2]
    add(sorted(source_confusers, key=lambda row: (int(row.get("anchor_rank", row.get("candidate_rank", 999))), str(row["doc_id"]))))
    add(sorted(eligible, key=lambda row: (int(row.get("candidate_rank", 999)), str(row["doc_id"]))))
    return selected[:6]


def multi_positive_loss(positive_scores: Sequence[float], negative_scores: Sequence[float], *, temperature: float = .05) -> float:
    if temperature <= 0 or not positive_scores: raise ValueError("positive scores and positive temperature are required")
    negatives = np.asarray(negative_scores, dtype=np.float64) / temperature; losses = []
    for positive in positive_scores:
        logits = np.concatenate(([float(positive) / temperature], negatives)); maximum = float(np.max(logits)); losses.append(-(float(positive) / temperature - maximum - math.log(float(np.exp(logits - maximum).sum()))))
    return float(np.mean(losses))


def ensemble_seed_scores(scores_by_seed: Mapping[int, Mapping[str, float]], *, seeds: Sequence[int] = SEEDS) -> dict[str, float]:
    if tuple(seeds) != SEEDS or set(scores_by_seed) != set(SEEDS): raise ValueError("EXP-109C seed ensemble must use exactly 109,110,111")
    normalized = [z_normalize_scores(scores_by_seed[seed]) for seed in seeds]; ids = sorted(set().union(*(values.keys() for values in normalized))); return {doc_id: float(np.mean([values.get(doc_id, 0.0) for values in normalized])) for doc_id in ids}


def smooth_max(values: Any, temperature: float = .05) -> Any:
    """Differentiable log-sum-exp max used only by the adapter training path."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if torch is not None and isinstance(values, torch.Tensor):
        return temperature * torch.logsumexp(values / temperature, dim=0)
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array)) if len(array) else float("-inf")
    return float(temperature * (maximum / temperature + math.log(float(np.exp(array / temperature - maximum / temperature).sum())))) if len(array) else float("-inf")


def adapted_parent_score(adapter: Any, query_vectors: Any, chunk_vectors: Sequence[Any], *, temperature: float = .05) -> Any:
    if not chunk_vectors:
        return torch.tensor(float("-inf")) if torch is not None else float("-inf")
    query = adapter.transform_query(query_vectors)
    scores = []
    for chunks in chunk_vectors:
        document = adapter.transform_document(chunks)
        score, _ = torch_maxsim(query, document)
        scores.append(score)
    return smooth_max(torch.stack(scores), temperature) if torch is not None else smooth_max(scores, temperature)


def _chunk_bag(value: Any) -> list[Any]:
    """Normalize one cached positive/negative parent bag to a chunk list."""
    if torch is not None and isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, np.ndarray):
        return [value]
    return list(value)


def torch_multi_positive_loss(positive_scores: Sequence[Any], negative_scores: Sequence[Any], *, temperature: float = .05) -> Any:
    """Each positive has its own denominator containing the shared negatives."""
    if torch is None or not positive_scores or temperature <= 0:
        raise ValueError("torch, positives and a positive temperature are required")
    negatives = torch.stack(list(negative_scores)) if negative_scores else torch.empty(0, device=positive_scores[0].device)
    losses = []
    for positive in positive_scores:
        denominator = torch.cat((positive.reshape(1), negatives)).logsumexp(dim=0)
        losses.append(-(positive / temperature - (torch.cat((positive.reshape(1), negatives)) / temperature).logsumexp(dim=0)))
    return torch.stack(losses).mean()


def train_adapter_batches(batches: Sequence[Mapping[str, Any]], *, rank: int, learning_rate: float, temperature: float, epochs: int = 3, seed: int = 109) -> dict[str, Any]:
    """Train one frozen-embedding adapter from parent-level weak supervision.

    Each batch contains ``query`` (QxD), ``positive_chunks`` (list of token
    arrays per gold parent) and ``negative_chunks`` (the same for negatives).
    This helper never loads a backbone and applies the six-negative limit.
    """
    if torch is None:
        raise RuntimeError("torch is required for metric adapter training")
    if rank not in {4, 8} or learning_rate not in {.001, .003} or temperature not in {.05, .10} or epochs > 3:
        raise ValueError("adapter hyperparameters are outside the locked EXP-109C grid")
    torch.manual_seed(seed); adapter = LowRankMetricAdapter(DIMENSION, rank, seed=seed); optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=.0001); losses: list[float] = []
    for _epoch in range(epochs):
        for batch in batches:
            query = torch.as_tensor(batch["query"], dtype=torch.float32); positives = [[torch.as_tensor(chunk, dtype=torch.float32) for chunk in _chunk_bag(value)] for value in batch.get("positive_chunks", [])]; negatives = [[torch.as_tensor(chunk, dtype=torch.float32) for chunk in _chunk_bag(value)] for value in batch.get("negative_chunks", [])][:6]
            if not positives or not negatives: continue
            optimizer.zero_grad(set_to_none=True); positive_scores = [adapted_parent_score(adapter, query, value, temperature=temperature) for value in positives]; negative_scores = [adapted_parent_score(adapter, query, value, temperature=temperature) for value in negatives]; loss = torch_multi_positive_loss(positive_scores, negative_scores, temperature=temperature); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    state = {name: value.detach().cpu().numpy().tolist() for name, value in adapter.state_dict().items()}
    return {"rank": rank, "learning_rate": learning_rate, "temperature": temperature, "seed": seed, "epochs": epochs, "losses": losses, "final_loss": losses[-1] if losses else None, "backbone_frozen": True, "negative_limit": 6, "identity_initialization": True, "state": state, "state_fingerprint": content_hash(state)}


def build_adapter_batches_from_cache(score_rows: Sequence[Mapping[str, Any]], *, outer: str = FOLD0, allowed_qids: Sequence[str] | None = None, max_batches: int | None = None) -> list[dict[str, Any]]:
    """Materialize parent-level weak-supervision bags from verified cache shards."""
    query_dir = CACHE_ROOT / "queries"; require_success(query_dir); qvectors = np.load(query_dir / "query_vectors.f16.npy", mmap_mode="r"); offsets = np.load(query_dir / "query_offsets.i64.npy", mmap_mode="r"); qids = [str(value) for value in read_json(query_dir / "qids.json")]; qrow = {qid: index for index, qid in enumerate(qids)}
    catalog, _ = load_index_catalog(); answers, _ = canonical_labels(); folds, _ = load_folds(); inner_qids = [str(qid) for name, values in folds.items() if name != outer for qid in values]
    if allowed_qids is not None: inner_qids = [str(qid) for qid in allowed_qids]
    by_qid = {str(row["qid"]): list(row.get("scores", [])) for row in score_rows}; batches: list[dict[str, Any]] = []
    for qid in sorted(inner_qids):
        if qid not in by_qid or qid not in qrow or not answers.get(qid): continue
        candidates = by_qid[qid]; positive_rows = [row for row in candidates if str(row["doc_id"]) in answers[qid]]
        if not positive_rows: continue
        negative_ids = select_six_negatives([{"doc_id": row["doc_id"], "candidate_rank": row.get("candidate_rank", 999), "anchor_rank": row.get("candidate_rank", 999), "late_coverage": row.get("features", {}).get("li_two_chunk_union_mean", -math.inf), "anchor_score": row.get("features", {}).get("e5_score", math.inf), "source_presence": row.get("source_presence", 0)} for row in candidates], answers[qid])
        rows_by_id = {str(row["doc_id"]): row for row in candidates}; positive_bags: list[list[np.ndarray]] = []; negative_bags: list[list[np.ndarray]] = []
        for row, target in ((value, positive_bags) for value in positive_rows):
            references = catalog.get(str(row["doc_id"]), []); order = np.argsort(-np.asarray(row.get("chunk_scores", []), dtype=np.float64), kind="mergesort")[:3]
            target.append([references[int(index)].load()[0] for index in order if int(index) < len(references)])
        for doc_id in negative_ids:
            row = rows_by_id[doc_id]; references = catalog.get(doc_id, []); order = np.argsort(-np.asarray(row.get("chunk_scores", []), dtype=np.float64), kind="mergesort")[:3]
            negative_bags.append([references[int(index)].load()[0] for index in order if int(index) < len(references)])
        positive_bags = [bag for bag in positive_bags if bag]; negative_bags = [bag for bag in negative_bags if bag]
        if positive_bags and negative_bags:
            position = qrow[qid]; query = np.asarray(qvectors[int(offsets[position]):int(offsets[position + 1])], dtype=np.float32); batches.append({"qid": qid, "query": query, "positive_chunks": positive_bags, "negative_chunks": negative_bags})
        if max_batches is not None and len(batches) >= max_batches: break
    return batches


def _prediction_metrics(predictions: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], qids: Sequence[str]) -> dict[str, float]:
    evaluated = [str(qid) for qid in qids if answers.get(str(qid))]
    if not evaluated:
        return {**{f"recall@{cutoff}": 0.0 for cutoff in (1, 3, 5, 10, 16, 50, CANDIDATE_DEPTH)}, "precision@5": 0.0, "mrr@5": 0.0, "single_gold_recall@5": 0.0, "multi_gold_recall@5": 0.0, "evaluable_queries": 0.0}
    def recall(qid: str, k: int) -> float: return recall_fraction(predictions.get(qid, []), answers[qid], k)
    values = [recall(qid, 5) for qid in evaluated]; first = [recall(qid, 1) for qid in evaluated]; precision = [len(set(predictions.get(qid, [])[:5]) & answers[qid]) / 5.0 for qid in evaluated]
    mrr = []
    for qid in evaluated:
        ranks = [index + 1 for index, doc_id in enumerate(predictions.get(qid, [])[:5]) if str(doc_id) in answers[qid]]; mrr.append(1.0 / min(ranks) if ranks else 0.0)
    multi = [recall(qid, 5) for qid in evaluated if len(answers[qid]) > 1]; single = [recall(qid, 5) for qid in evaluated if len(answers[qid]) == 1]
    result = {f"recall@{cutoff}": float(np.mean([recall(qid, cutoff) for qid in evaluated])) for cutoff in (1, 3, 5, 10, 16, 50, CANDIDATE_DEPTH)}
    result.update({"precision@5": float(np.mean(precision)), "mrr@5": float(np.mean(mrr)), "single_gold_recall@5": float(np.mean(single)) if single else 0.0, "multi_gold_recall@5": float(np.mean(multi)) if multi else 0.0, "evaluable_queries": float(len(evaluated))})
    return result


def _lgbm_matrix(score_by_qid: Mapping[str, Sequence[Mapping[str, Any]]], qids: Sequence[str], answers: Mapping[str, set[str]], feature_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[str, list[str]]]]:
    matrices: list[np.ndarray] = []; labels: list[float] = []; groups: list[int] = []; candidates: list[tuple[str, list[str]]] = []
    for qid in qids:
        rows = list(score_by_qid.get(str(qid), []))
        if not rows: continue
        matrix = np.asarray([[float(row.get("features", {}).get(name, 0.0)) for name in feature_names] for row in rows], dtype=np.float32)
        docs = [str(row["doc_id"]) for row in rows]
        matrices.append(matrix); labels.extend(float(doc_id in answers.get(str(qid), set())) for doc_id in docs); groups.append(len(docs)); candidates.append((str(qid), docs))
    return (np.concatenate(matrices, axis=0) if matrices else np.empty((0, len(feature_names)), dtype=np.float32), np.asarray(labels, dtype=np.float32), groups, candidates)


def _fit_lambdamart_predictions(train_x: np.ndarray, train_y: np.ndarray, train_groups: Sequence[int], valid_x: np.ndarray, valid_candidates: Sequence[tuple[str, Sequence[str]]], config: Mapping[str, Any]) -> dict[str, list[str]]:
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise GateRejected("REJECTED_DEPENDENCY_GATE", f"lightgbm is required for EXP-109C frozen screen: {exc}") from exc
    if not len(train_x) or not np.any(train_y > 0): raise GateRejected("REJECTED_PREREQUISITE_GATE", "no positive LambdaMART train rows")
    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_jobs=min(4, os.cpu_count() or 1), num_leaves=int(config["num_leaves"]), min_child_samples=int(config["min_data_in_leaf"]), learning_rate=float(config["learning_rate"]), n_estimators=int(config["num_boost_round"]), feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, deterministic=True, random_state=109, verbosity=-1)
    model.fit(train_x, train_y, group=list(map(int, train_groups)))
    predicted = model.predict(valid_x) if len(valid_x) else np.empty((0,), dtype=np.float32); offset = 0; result: dict[str, list[str]] = {}
    for qid, docs in valid_candidates:
        local = predicted[offset:offset + len(docs)]; order = np.lexsort((np.asarray(docs, dtype="U"), -np.asarray(local, dtype=np.float64))); result[qid] = [str(docs[int(index)]) for index in order]; offset += len(docs)
    return result


def _frozen_lambdamart_grid() -> list[dict[str, Any]]:
    return [{"num_leaves": leaves, "min_data_in_leaf": minimum, "learning_rate": rate, "num_boost_round": rounds} for leaves in (7, 15) for minimum in (50, 100) for rate in (.03, .05) for rounds in (200, 400)]


def nested_inner_lambdamart(score_rows: Sequence[Mapping[str, Any]], answers: Mapping[str, set[str]], folds: Mapping[str, Sequence[str]], *, outer: str, feature_names: Sequence[str]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Nested F1--F4 config selection; no Fold-0 label enters this path."""
    validate_feature_contract(tuple(feature_names)) if tuple(feature_names) == ALLOWED_FEATURES else None
    score_by_qid = {str(row["qid"]): list(row.get("scores", [])) for row in score_rows}; inner_names = [name for name in sorted(folds) if name != outer]; predictions: dict[str, list[str]] = {}; selections: dict[str, Any] = {}
    for heldout in inner_names:
        tuning_names = [name for name in inner_names if name != heldout]; config_scores: list[dict[str, Any]] = []
        for config in _frozen_lambdamart_grid():
            validation_predictions: dict[str, list[str]] = {}
            for validation in tuning_names:
                train_qids = [str(qid) for name in tuning_names if name != validation for qid in folds[name]]; valid_qids = [str(qid) for qid in folds[validation]]
                train_x, train_y, groups, _ = _lgbm_matrix(score_by_qid, train_qids, answers, feature_names); valid_x, _valid_y, _valid_groups, valid_candidates = _lgbm_matrix(score_by_qid, valid_qids, answers, feature_names)
                validation_predictions.update(_fit_lambdamart_predictions(train_x, train_y, groups, valid_x, valid_candidates, config))
            metrics = _prediction_metrics(validation_predictions, answers, [str(qid) for name in tuning_names for qid in folds[name]])
            config_scores.append({"config": config, "metrics": metrics})
        chosen = max(config_scores, key=lambda item: (item["metrics"]["recall@5"], item["metrics"]["precision@5"], item["metrics"]["multi_gold_recall@5"], item["metrics"]["mrr@5"]))
        train_qids = [str(qid) for name in tuning_names for qid in folds[name]]; valid_qids = [str(qid) for qid in folds[heldout]]
        train_x, train_y, groups, _ = _lgbm_matrix(score_by_qid, train_qids, answers, feature_names); valid_x, _valid_y, _valid_groups, valid_candidates = _lgbm_matrix(score_by_qid, valid_qids, answers, feature_names)
        predictions.update(_fit_lambdamart_predictions(train_x, train_y, groups, valid_x, valid_candidates, chosen["config"])); selections[heldout] = {"selected": chosen, "candidate_depth": CANDIDATE_DEPTH, "outer_excluded": outer}
    return predictions, {"feature_names": list(feature_names), "fold_selections": selections, "selection_scope": "nested F1-F4 only; Fold-0 excluded"}


def rank_from_score_rows(score_rows: Sequence[Mapping[str, Any]], *, score_key: str = "li_two_chunk_union_mean") -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    for row in score_rows:
        values = row.get("scores", [])
        ordered = sorted(values, key=lambda item: (-float(item.get("features", {}).get(score_key, -math.inf)), str(item.get("doc_id", ""))))
        predictions[str(row["qid"])] = [str(item["doc_id"]) for item in ordered]
    return predictions


def rank_by_normalized_feature_block(score_rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> dict[str, list[str]]:
    """Deterministic diagnostic rank for a frozen late feature block."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[str(row["qid"])].extend(row.get("scores", []))
    predictions: dict[str, list[str]] = {}
    for qid, values in grouped.items():
        local: dict[str, list[float]] = {name: [float(item.get("features", {}).get(name, 0.0)) for item in values] for name in feature_names}
        means = {name: float(np.mean(scores)) if scores else 0.0 for name, scores in local.items()}
        stds = {name: float(np.std(scores)) if scores else 0.0 for name, scores in local.items()}
        scored = []
        for index, item in enumerate(values):
            score = sum((local[name][index] - means[name]) / stds[name] for name in feature_names if stds[name] > 1e-12)
            scored.append((score, str(item["doc_id"])))
        predictions[qid] = [doc_id for _score, doc_id in sorted(scored, key=lambda value: (-value[0], value[1]))]
    return predictions


def frozen_inner_screen_report(score_rows: Sequence[Mapping[str, Any]], source_rankings_by_qid: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], answers: Mapping[str, set[str]], folds: Mapping[str, Sequence[str]], *, outer: str = FOLD0, anchor_predictions: Mapping[str, Sequence[str]] | None = None, anchor_provenance: Mapping[str, Any] | None = None, late_predictions: Mapping[str, Sequence[str]] | None = None, model_views: Mapping[str, Mapping[str, Sequence[str]]] | None = None, report_path: Path | None = None) -> dict[str, Any]:
    late_predictions = {str(qid): list(map(str, values)) for qid, values in (late_predictions or rank_from_score_rows(score_rows)).items()}
    if anchor_predictions is None:
        fallback_anchor: dict[str, list[str]] = {}
        for qid, sources in source_rankings_by_qid.items():
            candidates = build_candidate_union(sources, depth=CANDIDATE_DEPTH)
            fallback_anchor[qid] = [str(item["doc_id"]) for item in candidates]
        anchor_predictions = fallback_anchor
        anchor_provenance = {"model": "candidate-union fallback", "usable_for_gate": False, "reason": "EXP-109B LambdaMART predictions were not supplied"}
    else:
        anchor_predictions = {str(qid): list(map(str, values)) for qid, values in anchor_predictions.items()}
        anchor_provenance = dict(anchor_provenance or {"model": "EXP-109B scalar LambdaMART", "usable_for_gate": True})
    included = sorted(qid for name, qids in folds.items() if name != outer for qid in qids)
    base_metrics = _prediction_metrics(anchor_predictions, answers, included); late_metrics = _prediction_metrics(late_predictions, answers, included)
    per_fold: dict[str, Any] = {}; deltas: list[float] = []
    for name, qids in sorted(folds.items()):
        if name == outer: continue
        base = _prediction_metrics(anchor_predictions, answers, qids); late = _prediction_metrics(late_predictions, answers, qids); delta = late["recall@5"] - base["recall@5"]; deltas.append(delta); per_fold[name] = {"anchor": base, "late": late, "delta_recall@5": delta}
    delta = late_metrics["recall@5"] - base_metrics["recall@5"]
    views: dict[str, Mapping[str, Sequence[str]]] = {
        "standalone_jina_two_chunk": rank_from_score_rows(score_rows),
        "full_parent_union_diagnostic": rank_from_score_rows(score_rows, score_key="li_full_parent_union_mean"),
        "frozen_late_feature_block_diagnostic": rank_by_normalized_feature_block(score_rows, LATE_FEATURES[:-1]),
    }
    views.update(model_views or {})
    deltas_per_query = [recall_fraction(late_predictions.get(qid, []), answers[qid], 5) - recall_fraction(anchor_predictions.get(qid, []), answers[qid], 5) for qid in included if answers.get(qid)]
    transitions = {"wins": 0, "losses": 0, "ties": 0, "gold_into_top5": 0, "gold_out_of_top5": 0}
    oracle_values: list[float] = []
    for qid in included:
        if not answers.get(qid): continue
        base_value = recall_fraction(anchor_predictions.get(qid, []), answers[qid], 5); late_value = recall_fraction(late_predictions.get(qid, []), answers[qid], 5)
        transitions["wins" if late_value > base_value else "losses" if late_value < base_value else "ties"] += 1
        transitions["gold_into_top5"] += int(base_value == 0 and late_value > 0); transitions["gold_out_of_top5"] += int(base_value > 0 and late_value == 0); oracle_values.append(max(base_value, late_value))
    report = {"schema_version": SCHEMA, "stage": "frozen-inner-screen", "outer": outer, "selection_scope": "strict inner F1-F4; outer fold excluded", "fold0_seen": False, "anchor_provenance": anchor_provenance, "anchor": base_metrics, "late": late_metrics, "views": {name: _prediction_metrics(prediction, answers, included) for name, prediction in views.items()}, "delta_recall@5": delta, "multi_gold_delta": late_metrics["multi_gold_recall@5"] - base_metrics["multi_gold_recall@5"], "folds_non_negative": sum(value >= 0 for value in deltas), "worst_fold_delta": min(deltas, default=0.0), "per_fold": per_fold, "bootstrap": deterministic_bootstrap(deltas_per_query), "transitions": transitions, "choice_oracle_gain": float(np.mean(oracle_values) - base_metrics["recall@5"]) if oracle_values else 0.0, "choice_oracle_label_dependent_not_deployable": True, "claim_boundary": "inner cross-fit feature-screen evidence only; not Fold-0/public Recall"}
    gate_input = dict(report); gate_input["anchor_reproduction_pass"] = bool(anchor_provenance.get("usable_for_gate", False))
    report["gate"] = frozen_inner_gate(gate_input) if gate_input["anchor_reproduction_pass"] else {"pass": False, "status": "REJECTED_ANCHOR_REPRODUCTION", "checks": {}, "reason": "a real EXP-109B LambdaMART inner anchor is required"}
    report["status"] = "PASS_FROZEN_LATE_INTERACTION_GATE" if report["gate"]["pass"] else "REJECTED_FROZEN_LATE_INTERACTION_GATE"
    return write_report(report_path, report, stage="frozen-inner-screen") if report_path is not None else report


def run_frozen_inner_screen(*, outer: str = FOLD0) -> dict[str, Any]:
    """Run the cheap, cached-only inner screen after exact score shards exist."""
    score_rows, _score_manifest = load_cached_score_rows(outer=outer)
    source_rows = load_exp109b_anchor_sources(outer=outer); answers, _ = canonical_labels(); folds, _ = load_folds()
    anchor, anchor_meta = nested_inner_lambdamart(score_rows, answers, folds, outer=outer, feature_names=SCALAR_109B_FEATURES)
    raw, raw_meta = nested_inner_lambdamart(score_rows, answers, folds, outer=outer, feature_names=SCALAR_109B_FEATURES + ("li_two_chunk_union_mean",))
    full, full_meta = nested_inner_lambdamart(score_rows, answers, folds, outer=outer, feature_names=ALLOWED_FEATURES)
    standalone = rank_from_score_rows(score_rows)
    selection = {"schema_version": SCHEMA, "stage": "full-latent-selection-metadata", "outer": outer, "winner": "D_exp109b_plus_full_latent_block_lambdamart", "feature_names": list(ALLOWED_FEATURES), "candidate_contract": {"sources": ["vietlegal_e5", "vnlegal_lal", "bm25"], "per_source_depth": CANDIDATE_DEPTH, "union": "unique_parent_ids", "post_union_truncation": None}, "nested_metadata": full_meta, "score_manifest_sha256": sha256_file(late_score_cache_dir(outer) / "manifest.json"), "fold0_seen": False}
    selection["selection_fingerprint"] = content_hash(selection)
    atomic_json(RESULTS_ROOT / "FULL_LATENT_SELECTION_METADATA.json", selection)
    report = frozen_inner_screen_report(score_rows, source_rows, answers, folds, outer=outer, anchor_predictions=anchor, anchor_provenance={"model": "EXP-109B scalar LambdaMART reproduced", "usable_for_gate": True, **anchor_meta}, late_predictions=full, model_views={"B_jina_exact_maxsim_standalone": standalone, "C_exp109b_plus_raw_late_lambdamart": raw, "D_exp109b_plus_full_latent_block_lambdamart": full}, report_path=RESULTS_ROOT / "FROZEN_INNER_SCREEN.json")
    report["full_latent_selection_metadata_sha256"] = sha256_file(RESULTS_ROOT / "FULL_LATENT_SELECTION_METADATA.json")
    atomic_json(RESULTS_ROOT / "FROZEN_INNER_SCREEN.json", report)
    return report


def fold_qids_without_labels(*, outer: str = FOLD0) -> dict[str, list[str]]:
    """Read only the frozen fold membership JSON; never load training labels."""
    raw = read_json(FOLDS_PATH)
    folds = {str(name): [str(qid) for qid in values] for name, values in raw.items()}
    if set(folds) != set(FOLD_NAMES) or outer not in folds:
        raise GateRejected("REJECTED_FOLD_CONTRACT", "frozen fold membership is incomplete")
    if len({qid for values in folds.values() for qid in values}) != sum(len(values) for values in folds.values()):
        raise GateRejected("REJECTED_FOLD_CONTRACT", "fold membership overlaps")
    return folds


def frozen_config_key(config: Mapping[str, Any]) -> tuple[int, int, float, int]:
    return (int(config["num_leaves"]), int(config["min_data_in_leaf"]), float(config["learning_rate"]), int(config["num_boost_round"]))


def select_final_frozen_config(fold_selections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Majority-vote final config from F1--F4 nested selections only."""
    grouped: dict[tuple[int, int, float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for fold, value in sorted(fold_selections.items()):
        selected = value.get("selected", {}) if isinstance(value, Mapping) else {}
        config = selected.get("config") if isinstance(selected, Mapping) else None
        if not isinstance(config, Mapping): raise GateRejected("REJECTED_FROZEN_WINNER_LOCK", f"missing full-latent config for {fold}")
        grouped[frozen_config_key(config)].append(selected)
    if len(grouped) == 0: raise GateRejected("REJECTED_FROZEN_WINNER_LOCK", "no full-latent nested selections")
    def score(item: tuple[tuple[int, int, float, int], list[Mapping[str, Any]]]) -> tuple[Any, ...]:
        key, selections = item
        metrics = [dict(value.get("metrics", {})) for value in selections]
        mean = lambda name: float(np.mean([float(value.get(name, -math.inf)) for value in metrics]))
        return (len(selections), mean("recall@5"), mean("precision@5"), mean("multi_gold_recall@5"), mean("mrr@5"))
    # The final ``min`` below supplies the explicit smallest-config tie break
    # without depending on mapping insertion order.
    best_key, selections = max(grouped.items(), key=lambda item: score(item)[:5])
    tied = [item for item in grouped.items() if score(item)[:5] == score((best_key, selections))[:5]]
    if len(tied) > 1: best_key, selections = min(tied, key=lambda item: item[0])
    return {"config": {"num_leaves": best_key[0], "min_data_in_leaf": best_key[1], "learning_rate": best_key[2], "num_boost_round": best_key[3]}, "votes": len(selections), "folds": sorted(str(name) for name, value in fold_selections.items() if frozen_config_key(value["selected"]["config"]) == best_key), "policy": "majority_votes_then_mean_recall5_precision5_multigold_mrr_then_smallest_tuple"}


def frozen_winner_lock(*, outer: str = FOLD0) -> dict[str, Any]:
    """Lock the passed D100 full-latent winner before any Fold-0 scoring."""
    frozen_path = RESULTS_ROOT / "FROZEN_INNER_SCREEN.json"; frozen = require_report_status(frozen_path, ("PASS_FROZEN_LATE_INTERACTION_GATE",))
    score_rows, score_manifest = load_cached_score_rows(outer=outer)
    metadata_path = RESULTS_ROOT / "FULL_LATENT_SELECTION_METADATA.json"
    if not metadata_path.exists():
        # This is cache-only F1--F4 reproduction, never Jina scoring or Fold-0 access.
        answers, _ = canonical_labels(); folds, _ = load_folds(); _predictions, full_meta = nested_inner_lambdamart(score_rows, answers, folds, outer=outer, feature_names=ALLOWED_FEATURES)
        metadata = {"schema_version": SCHEMA, "stage": "full-latent-selection-metadata", "outer": outer, "winner": "D_exp109b_plus_full_latent_block_lambdamart", "feature_names": list(ALLOWED_FEATURES), "nested_metadata": full_meta, "score_manifest_sha256": sha256_file(late_score_cache_dir(outer) / "manifest.json"), "fold0_seen": False}; metadata["selection_fingerprint"] = content_hash(metadata); atomic_json(metadata_path, metadata)
    metadata = read_json(metadata_path)
    if tuple(metadata.get("feature_names", ())) != ALLOWED_FEATURES or metadata.get("winner") != "D_exp109b_plus_full_latent_block_lambdamart": raise GateRejected("REJECTED_FROZEN_WINNER_LOCK", "full-latent selection metadata contract mismatch")
    final_config = select_final_frozen_config(metadata.get("nested_metadata", {}).get("fold_selections", {}))
    fold_membership = fold_qids_without_labels(outer=outer); inner_qids = sorted(qid for name, values in fold_membership.items() if name != outer for qid in values); fold0_qids = sorted(fold_membership[outer])
    # Persist only the labels already admitted by the F1--F4 frozen screen.
    # Subsequent score/train/predict stages read this reduced artifact rather
    # than the canonical training file, so their code path cannot inspect F0.
    answers, _ = canonical_labels()
    inner_answers = {qid: sorted(answers[qid]) for qid in inner_qids}
    labels_path = RESULTS_ROOT / "F1_F4_TRAINING_LABELS_LOCK.json"
    labels_lock = {"schema_version": SCHEMA, "stage": "frozen-winner-lock", "outer": outer, "query_ids_sha256": content_hash(inner_qids), "query_count": len(inner_qids), "answers": inner_answers, "fold0_labels_present": False, "fold0_seen": False}
    labels_lock["content_fingerprint"] = content_hash(labels_lock)
    atomic_json(labels_path, labels_lock)
    lock = {"schema_version": SCHEMA, "stage": "frozen-winner-lock", "status": "PASS_FROZEN_WINNER_LOCK", "decision": "close_metric_adapter_and_evaluate_passed_frozen_winner", "winner": "D_exp109b_plus_full_latent_block_lambdamart", "feature_names": list(ALLOWED_FEATURES), "candidate_contract": {"sources": ["vietlegal_e5", "vnlegal_lal", "bm25"], "per_source_depth": CANDIDATE_DEPTH, "union": "unique_parent_ids", "post_union_truncation": None}, "config_e_contract": CONFIG_E_CONTRACT, "scorer_implementation_contract": SCORER_IMPLEMENTATION_CONTRACT, "final_config": final_config, "inner_training_qids_sha256": content_hash(inner_qids), "fold0_query_ids_sha256": content_hash(fold0_qids), "fold0_query_count": len(fold0_qids), "frozen_inner_screen_sha256": sha256_file(frozen_path), "frozen_inner_screen_fingerprint": frozen.get("report_fingerprint"), "full_latent_selection_metadata_sha256": sha256_file(metadata_path), "score_manifest_sha256": sha256_file(late_score_cache_dir(outer) / "manifest.json"), "score_manifest_fingerprint": score_manifest.get("content_fingerprint"), "f1_f4_training_labels_lock_sha256": sha256_file(labels_path), "code_sha256": code_fingerprint(), "fold0_seen_at_lock_time": False, "fold0_seen": False, "no_adapter_training": True, "no_full_oof": True, "no_public_submission": True, "claim_boundary": "locked frozen winner only; Fold-0 labels and metrics have not been read"}
    lock["lock_fingerprint"] = content_hash(lock)
    return write_report(RESULTS_ROOT / "FROZEN_WINNER_LOCK.json", lock, stage="frozen-winner-lock", success=True)


def _frozen_lock() -> dict[str, Any]:
    return require_report_status(RESULTS_ROOT / "FROZEN_WINNER_LOCK.json", ("PASS_FROZEN_WINNER_LOCK",))


def _locked_inner_answers(lock: Mapping[str, Any]) -> dict[str, set[str]]:
    path = RESULTS_ROOT / "F1_F4_TRAINING_LABELS_LOCK.json"
    if not path.exists() or sha256_file(path) != lock.get("f1_f4_training_labels_lock_sha256"):
        raise GateRejected("REJECTED_FROZEN_WINNER_LOCK", "F1-F4 training-label lock is missing or stale", report=path)
    payload = read_json(path)
    if payload.get("fold0_labels_present") is not False or payload.get("fold0_seen") is not False:
        raise GateRejected("REJECTED_FOLD_ISOLATION", "training-label lock is not Fold-0 isolated", report=path)
    return {str(qid): set(map(str, values)) for qid, values in payload.get("answers", {}).items()}


def frozen_fold0_score_dir(lock: Mapping[str, Any]) -> Path:
    return CACHE_ROOT / "late_scores" / "fold_0_locked" / str(lock["lock_fingerprint"])


def _fit_frozen_ranker(train_x: np.ndarray, train_y: np.ndarray, groups: Sequence[int], config: Mapping[str, Any], feature_names: Sequence[str]) -> Any:
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise GateRejected("REJECTED_DEPENDENCY_GATE", f"lightgbm is required for frozen winner: {exc}") from exc
    if len(train_x) == 0 or not np.any(train_y > 0):
        raise GateRejected("REJECTED_PREREQUISITE_GATE", "final frozen ranker has no positive training rows")
    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_jobs=min(4, os.cpu_count() or 1), num_leaves=int(config["num_leaves"]), min_child_samples=int(config["min_data_in_leaf"]), learning_rate=float(config["learning_rate"]), n_estimators=int(config["num_boost_round"]), feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, deterministic=True, random_state=109, verbosity=-1)
    # Feature order is cryptographically locked in the model manifest; keep
    # arrays here to avoid sklearn's false feature-name warning at inference.
    model.fit(train_x, train_y, group=list(map(int, groups)))
    return model


def _final_model_paths() -> tuple[Path, Path, Path]:
    return (RESULTS_ROOT / "FINAL_SCALAR_MODEL.txt", RESULTS_ROOT / "FINAL_FULL_LATENT_MODEL.txt", RESULTS_ROOT / "FINAL_MODEL_MANIFEST.json")


def _train_frozen_model_from_inner(lock: Mapping[str, Any], feature_names: Sequence[str]) -> tuple[Any, dict[str, Any]]:
    score_rows, score_manifest = load_cached_score_rows(outer=FOLD0)
    answers = _locked_inner_answers(lock); memberships = fold_qids_without_labels(outer=FOLD0)
    inner_qids = sorted(qid for name, values in memberships.items() if name != FOLD0 for qid in values)
    if set(inner_qids) != set(answers) or content_hash(inner_qids) != lock.get("inner_training_qids_sha256"):
        raise GateRejected("REJECTED_FOLD_ISOLATION", "F1-F4 label lock does not exactly match winner lock")
    by_qid = {str(row["qid"]): list(row.get("scores", [])) for row in score_rows}
    x, y, groups, _ = _lgbm_matrix(by_qid, inner_qids, answers, feature_names)
    model = _fit_frozen_ranker(x, y, groups, lock["final_config"]["config"], feature_names)
    detail = {"feature_names": list(feature_names), "feature_names_sha256": content_hash(list(feature_names)), "training_qids_sha256": content_hash(inner_qids), "training_query_count": len(inner_qids), "score_manifest_fingerprint": score_manifest.get("content_fingerprint"), "final_config": lock["final_config"], "fold0_labels_read": False}
    return model, detail


def _save_frozen_model(model: Any, path: Path, detail: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))
    if not path.exists():
        raise GateRejected("REJECTED_MODEL_CONTRACT", f"LightGBM did not write {path}")
    result = dict(detail); result.update({"path": str(path.resolve()), "sha256": sha256_file(path)})
    return result


def _load_frozen_booster(path: Path, expected_sha256: str) -> Any:
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise GateRejected("REJECTED_DEPENDENCY_GATE", f"lightgbm is required for frozen winner: {exc}") from exc
    if not path.exists() or sha256_file(path) != expected_sha256:
        raise GateRejected("REJECTED_MODEL_CONTRACT", f"frozen model is missing or hash-mismatched: {path}")
    return lgb.Booster(model_file=str(path))


def _scalar_anchor_for_fold0(lock: Mapping[str, Any], source_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], parent_metadata: Mapping[str, Mapping[str, Any]], query_lengths: Mapping[str, int]) -> tuple[Any, dict[str, Any]]:
    scalar_path, _full_path, manifest_path = _final_model_paths()
    if manifest_path.exists():
        manifest = read_json(manifest_path); scalar = manifest.get("scalar_model", {})
        if scalar.get("feature_names") == list(SCALAR_109B_FEATURES):
            return _load_frozen_booster(scalar_path, str(scalar.get("sha256", ""))), scalar
    model, detail = _train_frozen_model_from_inner(lock, SCALAR_109B_FEATURES)
    scalar = _save_frozen_model(model, scalar_path, detail)
    # This pre-score scalar anchor is deterministic and F1--F4-only.  The
    # full model is deliberately not trained until locked F0 features exist.
    atomic_json(RESULTS_ROOT / "FROZEN_SCALAR_ANCHOR_MANIFEST.json", {"schema_version": SCHEMA, "stage": "score-fold0-frozen", "scalar_model": scalar, "lock_fingerprint": lock["lock_fingerprint"], "fold0_labels_read": False, "source_rows_fingerprint": content_hash(source_rows), "query_lengths_fingerprint": content_hash(query_lengths)})
    return model.booster_, scalar


def _validate_fold0_score_row(row: Mapping[str, Any], source_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[str], str]:
    expected = build_candidate_union(source_rows, depth=CANDIDATE_DEPTH)
    expected_ids = [str(item["doc_id"]) for item in expected]
    if any(int(provenance["rank"]) > CANDIDATE_DEPTH for item in expected for provenance in item.get("sources", {}).values()):
        raise GateRejected("REJECTED_FOLD0_CANDIDATE_CONTRACT", f"source contribution exceeds D100 for query {row.get('qid')}")
    recorded_ids = row.get("candidate_ids")
    if recorded_ids is not None and [str(doc_id) for doc_id in recorded_ids] != expected_ids:
        raise GateRejected("REJECTED_FOLD0_CANDIDATE_CONTRACT", f"stored candidate ID union differs for query {row.get('qid')}")
    if row.get("candidate_ids_sha256") is not None and row.get("candidate_ids_sha256") != content_hash(expected_ids):
        raise GateRejected("REJECTED_FOLD0_CANDIDATE_CONTRACT", f"stored candidate ID hash differs for query {row.get('qid')}")
    observed_scores = list(row.get("scores", []))
    observed_by_rank = [str(item["doc_id"]) for item in sorted(observed_scores, key=lambda value: int(value.get("candidate_rank", math.inf)))]
    if observed_by_rank != expected_ids:
        raise GateRejected("REJECTED_FOLD0_CANDIDATE_CONTRACT", f"candidate union differs for query {row.get('qid')}")
    expected_ranks = {str(item["doc_id"]): int(item["candidate_rank"]) for item in expected}
    if {str(item["doc_id"]): int(item.get("candidate_rank", -1)) for item in observed_scores} != expected_ranks:
        raise GateRejected("REJECTED_FOLD0_CANDIDATE_CONTRACT", f"candidate ranks differ for query {row.get('qid')}")
    if len(observed_by_rank) != len(set(observed_by_rank)):
        raise GateRejected("REJECTED_FOLD0_CANDIDATE_CONTRACT", f"duplicate candidate parent for query {row.get('qid')}")
    if set(source_rows) != {"vietlegal_e5", "vnlegal_lal", "bm25"}:
        raise GateRejected("REJECTED_FOLD0_CANDIDATE_CONTRACT", "source union has an unexpected source set")
    for item in observed_scores:
        features = item.get("features", {})
        if any(name not in features or not np.isfinite(float(features[name])) for name in ALLOWED_FEATURES):
            raise GateRejected("REJECTED_FOLD0_FEATURE_CONTRACT", f"missing/non-finite full-latent feature for {row.get('qid')}")
    return expected_ids, content_hash(expected_ids)


def score_fold0_frozen(*, outer: str = FOLD0, resume: bool = True, authorize: bool = False, device: str = "cuda") -> dict[str, Any]:
    """Materialize D100-source-union F0 late features without reading F0 labels."""
    require_authorization("score-fold0-frozen", authorize, "EXP109C_ALLOW_FOLD0")
    if outer != FOLD0 or device != "cuda" or torch is None or not torch.cuda.is_available():
        raise GateRejected("REJECTED_RESOURCE_GATE", "locked Fold-0 scoring requires CUDA on the designated Fold-0 only")
    lock = _frozen_lock(); fold_membership = fold_qids_without_labels(outer=outer); qid_list = sorted(fold_membership[outer])
    output = frozen_fold0_score_dir(lock); shard_dir = output / "shards"; output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and (output / "_SUCCESS.json").exists():
        existing = read_json(output / "manifest.json"); require_success(output, existing.get("content_fingerprint"))
        if existing.get("winner_lock_fingerprint") == lock["lock_fingerprint"] and existing.get("query_ids_sha256") == content_hash(qid_list):
            return existing
        raise GateRejected("REJECTED_STALE_ARTIFACT", "existing locked Fold-0 cache belongs to another winner lock")
    query_dir = CACHE_ROOT / "queries"; index_dir = CACHE_ROOT / "index"; require_success(query_dir)
    query_manifest = read_json(query_dir / "manifest.json"); by_doc, index_manifest = load_index_catalog(index_dir)
    if query_manifest.get("config_e_contract") != CONFIG_E_CONTRACT or index_manifest.get("config", {}).get("config_e_contract") != CONFIG_E_CONTRACT:
        raise GateRejected("REJECTED_STALE_ARTIFACT", "query/index store fails locked Config E contract")
    qvectors = np.load(query_dir / "query_vectors.f16.npy", mmap_mode="r"); qtoken_ids = np.load(query_dir / "query_token_ids.i64.npy", mmap_mode="r"); offsets = np.load(query_dir / "query_offsets.i64.npy", mmap_mode="r")
    all_qids = [str(value) for value in read_json(query_dir / "qids.json")]; qrow = {qid: index for index, qid in enumerate(all_qids)}
    source_rows = load_exp109b_anchor_sources(outer=outer, qids=set(qid_list)); parent_metadata = load_parent_metadata()
    if set(source_rows) != set(qid_list) or set(qid_list) - set(qrow):
        raise GateRejected("REJECTED_FOLD0_CANDIDATE_CONTRACT", "Fold-0 source/query coverage is incomplete")
    idf_path = CACHE_ROOT / "idf" / "token_idf.json"; idf_manifest_path = CACHE_ROOT / "idf" / "manifest.json"
    if not idf_path.exists() or not idf_manifest_path.exists(): raise GateRejected("REJECTED_PREREQUISITE_GATE", "IDF artifacts are missing")
    idf_manifest = read_json(idf_manifest_path)
    if sha256_file(idf_path) != idf_manifest.get("files", {}).get(idf_path.name): raise GateRejected("REJECTED_STALE_ARTIFACT", "IDF artifact hash mismatch")
    idf = {int(key): float(value) for key, value in read_json(idf_path).items()}
    query_lengths = {qid: int(offsets[qrow[qid] + 1] - offsets[qrow[qid]]) for qid in qid_list}
    scalar_booster, scalar_detail = _scalar_anchor_for_fold0(lock, source_rows, parent_metadata, query_lengths)
    mmap_cache = ShardMmapCache(); receipts: list[dict[str, Any]] = []; completed: set[str] = set(); performance: defaultdict[str, float] = defaultdict(float)
    if resume:
        for receipt_path in sorted(shard_dir.glob("scores-*.json")):
            receipt = read_json(receipt_path); path = shard_dir / str(receipt.get("name", "")); shard_qids = [str(qid) for qid in receipt.get("query_ids", ())]
            if not path.exists() or receipt.get("sha256") != sha256_file(path) or receipt.get("winner_lock_fingerprint") != lock["lock_fingerprint"] or not shard_qids or set(shard_qids) - set(qid_list) or completed.intersection(shard_qids):
                raise GateRejected("REJECTED_STALE_ARTIFACT", f"locked F0 shard receipt is invalid: {receipt_path}")
            for row in read_jsonl(path):
                if str(row.get("qid")) not in shard_qids: raise GateRejected("REJECTED_STALE_ARTIFACT", f"locked F0 shard row mismatch: {path}")
                _validate_fold0_score_row(row, source_rows[str(row["qid"])])
            completed.update(shard_qids); receipts.append(receipt)

    def score_query(qid: str) -> dict[str, Any]:
        start, end = int(offsets[qrow[qid]]), int(offsets[qrow[qid] + 1]); query = np.asarray(qvectors[start:end], dtype=np.float32); weights = np.asarray([idf.get(int(token), 1.0) for token in qtoken_ids[start:end]], dtype=np.float32)
        candidates = build_candidate_union(source_rows[qid], depth=CANDIDATE_DEPTH)
        scalar_rows = [build_combined_feature_record(qid, str(item["doc_id"]), source_rows[qid], {name: 0.0 for name in LATE_FEATURES}, metadata=parent_metadata.get(str(item["doc_id"])), query_token_length=len(query), depth=CANDIDATE_DEPTH) for item in candidates]
        scalar_x = np.asarray([[float(row[name]) for name in SCALAR_109B_FEATURES] for row in scalar_rows], dtype=np.float32)
        scalar_anchor = stable_rank(scalar_booster.predict(scalar_x), [str(item["doc_id"]) for item in candidates])
        approximate, approximate_diag = score_candidate_pool_batched_cuda(query, candidates, by_doc, idf_weights=weights, mmap_cache=mmap_cache, full_tokens=False, device=device)
        accumulate_numeric_diagnostics(performance, "approximate", approximate_diag)
        approximate_scores = {doc_id: float(value["late"]["features"]["li_full_parent_union_mean"]) for doc_id, value in approximate.items()}
        exact_ids = exact_refinement_set(approximate_scores, scalar_anchor)
        exact_candidates = [approximate[doc_id]["candidate"] for doc_id in exact_ids]
        exact, exact_diag = score_candidate_pool_batched_cuda(query, exact_candidates, by_doc, idf_weights=weights, mmap_cache=mmap_cache, full_tokens=True, device=device)
        accumulate_numeric_diagnostics(performance, "exact", exact_diag)
        scores: list[dict[str, Any]] = []
        for doc_id, value in approximate.items():
            candidate, late = value["candidate"], value["late"]; tier = "approximate_medoids"
            if doc_id in exact:
                late, tier = exact[doc_id]["late"], "exact_full_token"
            features = build_combined_feature_record(qid, doc_id, source_rows[qid], late["features"], metadata=parent_metadata.get(doc_id), query_token_length=len(query), depth=CANDIDATE_DEPTH)
            scores.append({"doc_id": doc_id, "candidate_rank": int(candidate["candidate_rank"]), "source_presence": int(candidate.get("source_presence", 0)), "scoring_tier": tier, **late, "features": features})
        scores.sort(key=lambda item: (-float(item["features"]["li_two_chunk_union_mean"]), str(item["doc_id"])))
        for rank, item in enumerate(scores, 1): item["features"]["li_exact_rank_within_candidate_pool"] = rank
        row = {"qid": qid, "query_tokens": len(query), "scores": scores, "exact_set_size": len(exact_ids), "candidate_ids": [str(item["doc_id"]) for item in candidates], "candidate_ids_sha256": content_hash([str(item["doc_id"]) for item in candidates]), "structural_fingerprint": STRUCTURAL_FINGERPRINT, "index_fingerprint": index_manifest.get("content_fingerprint"), "labels_read": False}
        _validate_fold0_score_row(row, source_rows[qid]); return row

    with tracked_stage("score-fold0-frozen", outer=outer, total=len(qid_list)) as tracker:
        tracker.heartbeat(len(completed), emit=True)
        pending = [qid for qid in qid_list if qid not in completed]
        for start in range(0, len(pending), SCORE_QUERY_SHARD_SIZE):
            shard_qids = pending[start:start + SCORE_QUERY_SHARD_SIZE]; rows = [score_query(qid) for qid in shard_qids]
            shard_id = content_hash({"lock": lock["lock_fingerprint"], "qids": shard_qids})[:16]; path = shard_dir / f"scores-{shard_id}.jsonl"; shard_dir.mkdir(parents=True, exist_ok=True)
            count = write_jsonl_atomic(path, rows); receipt = {"name": path.name, "count": count, "query_ids": shard_qids, "sha256": sha256_file(path), "winner_lock_fingerprint": lock["lock_fingerprint"], "candidate_source_depth": CANDIDATE_DEPTH, "scorer_implementation_contract": SCORER_IMPLEMENTATION_CONTRACT}; atomic_json(shard_dir / f"scores-{shard_id}.json", receipt); receipts.append(receipt); tracker.heartbeat(len(completed) + start + len(shard_qids), emit=True)
    counts: list[int] = []; candidate_hashes: dict[str, str] = {}
    for receipt in receipts:
        for row in read_jsonl(shard_dir / receipt["name"]):
            ids, digest = _validate_fold0_score_row(row, source_rows[str(row["qid"])]); counts.append(len(ids)); candidate_hashes[str(row["qid"])] = digest
    if set(candidate_hashes) != set(qid_list): raise GateRejected("REJECTED_FOLD0_CANDIDATE_CONTRACT", "Fold-0 score coverage is incomplete")
    manifest = {"schema_version": SCHEMA, "stage": "score-fold0-frozen", "status": "PASS_FOLD0_SCORE_MANIFEST", "winner_lock_fingerprint": lock["lock_fingerprint"], "query_count": len(qid_list), "query_ids_sha256": content_hash(qid_list), "candidate_contract": {"sources": ["vietlegal_e5", "vnlegal_lal", "bm25"], "per_source_depth": CANDIDATE_DEPTH, "union": "unique_parent_ids", "post_union_truncation": None, "candidate_count": {"min": min(counts), "mean": float(np.mean(counts)), "max": max(counts)}}, "candidate_ids_sha256_by_query": candidate_hashes, "feature_names": list(ALLOWED_FEATURES), "config_e_contract": CONFIG_E_CONTRACT, "scorer_implementation_contract": SCORER_IMPLEMENTATION_CONTRACT, "scalar_anchor_model_sha256": scalar_detail["sha256"], "index_fingerprint": index_manifest.get("content_fingerprint"), "idf_fingerprint": idf_manifest.get("content_fingerprint"), "shards": receipts, "performance": dict(performance), "fold0_labels_read": False, "claim_boundary": "pre-label locked Fold-0 features only; no Fold-0 metric or model selection"}
    manifest["content_fingerprint"] = content_hash(manifest); atomic_json(output / "manifest.json", manifest); write_success(output, stage="score-fold0-frozen", fingerprint=manifest["content_fingerprint"])
    return write_report(RESULTS_ROOT / "FOLD0_SCORE_MANIFEST.json", manifest, stage="score-fold0-frozen")


def _load_locked_fold0_score_rows(lock: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = frozen_fold0_score_dir(lock); manifest_path = output / "manifest.json"
    if not manifest_path.exists(): raise GateRejected("REJECTED_PREREQUISITE_GATE", "locked Fold-0 feature manifest is missing", report=manifest_path)
    manifest = read_json(manifest_path); require_success(output, manifest.get("content_fingerprint"))
    if manifest.get("winner_lock_fingerprint") != lock.get("lock_fingerprint"):
        raise GateRejected("REJECTED_STALE_ARTIFACT", "Fold-0 feature cache belongs to another winner lock")
    rows: list[dict[str, Any]] = []
    for receipt in manifest.get("shards", []):
        path = output / "shards" / str(receipt.get("name", ""))
        if not path.exists() or sha256_file(path) != receipt.get("sha256"):
            raise GateRejected("REJECTED_STALE_ARTIFACT", f"locked F0 score shard hash mismatch: {path}")
        rows.extend(read_jsonl(path))
    expected = sorted(fold_qids_without_labels(outer=FOLD0)[FOLD0]); by_qid = {str(row.get("qid")): row for row in rows}
    if set(by_qid) != set(expected) or len(rows) != len(by_qid):
        raise GateRejected("REJECTED_FOLD0_CANDIDATE_CONTRACT", "locked Fold-0 score rows do not cover exactly Fold-0")
    source_rows = load_exp109b_anchor_sources(outer=FOLD0, qids=set(expected))
    for qid, row in by_qid.items(): _validate_fold0_score_row(row, source_rows[qid])
    return [by_qid[qid] for qid in expected], manifest


def _rank_model_rows(booster: Any, score_rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    for row in score_rows:
        scores = list(row.get("scores", [])); docs = [str(item["doc_id"]) for item in scores]
        matrix = np.asarray([[float(item.get("features", {}).get(name, 0.0)) for name in feature_names] for item in scores], dtype=np.float32)
        if len(docs) == 0 or len(docs) != len(set(docs)) or not np.isfinite(matrix).all():
            raise GateRejected("REJECTED_PREDICTION_CONTRACT", f"invalid locked score row for {row.get('qid')}")
        predictions[str(row["qid"])] = stable_rank(booster.predict(matrix), docs)
    return predictions


def train_final_frozen(*, outer: str = FOLD0, authorize: bool = False) -> dict[str, Any]:
    """Train F1--F4 scalar/full models and lock their pre-label Fold-0 predictions."""
    require_authorization("train-final-frozen", authorize, "EXP109C_ALLOW_FOLD0")
    if outer != FOLD0: raise GateRejected("REJECTED_FOLD_ISOLATION", "frozen final train path is defined only for Fold-0")
    lock = _frozen_lock(); fold0_rows, score_manifest = _load_locked_fold0_score_rows(lock)
    scalar_path, full_path, manifest_path = _final_model_paths()
    scalar_model, scalar_detail = _train_frozen_model_from_inner(lock, SCALAR_109B_FEATURES)
    full_model, full_detail = _train_frozen_model_from_inner(lock, ALLOWED_FEATURES)
    scalar_info = _save_frozen_model(scalar_model, scalar_path, scalar_detail); full_info = _save_frozen_model(full_model, full_path, full_detail)
    model_manifest = {"schema_version": SCHEMA, "stage": "train-final-frozen", "status": "PASS_FINAL_FROZEN_MODELS", "winner_lock_fingerprint": lock["lock_fingerprint"], "scalar_model": scalar_info, "full_latent_model": full_info, "candidate_contract": lock["candidate_contract"], "fold0_score_manifest_fingerprint": score_manifest.get("content_fingerprint"), "fold0_labels_read": False, "claim_boundary": "F1-F4-trained frozen models; no Fold-0 labels or evaluation"}
    model_manifest["content_fingerprint"] = content_hash(model_manifest); atomic_json(manifest_path, model_manifest)
    scalar_predictions = _rank_model_rows(scalar_model.booster_, fold0_rows, SCALAR_109B_FEATURES); full_predictions = _rank_model_rows(full_model.booster_, fold0_rows, ALLOWED_FEATURES)
    qids = sorted(fold_qids_without_labels(outer=outer)[outer]); scalar_predictions = _validated_predictions(scalar_predictions, qids, label="fold0-scalar-prelabel"); full_predictions = _validated_predictions(full_predictions, qids, label="fold0-full-latent-prelabel")
    scalar_prediction_path = RESULTS_ROOT / "FOLD0_SCALAR_PREDICTIONS.json"; full_prediction_path = RESULTS_ROOT / "FOLD0_FULL_LATENT_PREDICTIONS.json"
    scalar_payload = {"schema_version": SCHEMA, "stage": "train-final-frozen", "model_sha256": scalar_info["sha256"], "feature_names": list(SCALAR_109B_FEATURES), "query_ids_sha256": content_hash(qids), "predictions": scalar_predictions, "fold0_labels_read": False}; scalar_payload["prediction_fingerprint"] = content_hash(scalar_predictions); atomic_json(scalar_prediction_path, scalar_payload)
    full_payload = {"schema_version": SCHEMA, "stage": "train-final-frozen", "model_sha256": full_info["sha256"], "feature_names": list(ALLOWED_FEATURES), "query_ids_sha256": content_hash(qids), "predictions": full_predictions, "fold0_labels_read": False}; full_payload["prediction_fingerprint"] = content_hash(full_predictions); atomic_json(full_prediction_path, full_payload)
    coverage = {qid: len(full_predictions[qid]) for qid in qids}
    lock_payload = {"schema_version": SCHEMA, "stage": "train-final-frozen", "status": "PASS_FOLD0_PREDICTION_LOCK", "winner_lock_sha256": sha256_file(RESULTS_ROOT / "FROZEN_WINNER_LOCK.json"), "winner_lock_fingerprint": lock["lock_fingerprint"], "model_manifest_sha256": sha256_file(manifest_path), "scalar_prediction_sha256": sha256_file(scalar_prediction_path), "full_prediction_sha256": sha256_file(full_prediction_path), "scalar_prediction_fingerprint": scalar_payload["prediction_fingerprint"], "full_prediction_fingerprint": full_payload["prediction_fingerprint"], "query_count": len(qids), "query_ids_sha256": content_hash(qids), "candidate_count_by_query": coverage, "top5": {qid: full_predictions[qid][:5] for qid in qids}, "labels_read": False, "timestamp": utc_now(), "claim_boundary": "prediction lock written before Fold-0 label access"}
    lock_payload["content_fingerprint"] = content_hash(lock_payload); return write_report(RESULTS_ROOT / "FOLD0_PREDICTION_LOCK.json", lock_payload, stage="train-final-frozen", success=True)


def _frozen_transfer_status(*, scalar: Mapping[str, Any], full: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> str:
    delta = float(full["recall@5"]) - float(scalar["recall@5"])
    promote = delta >= .005 and float(bootstrap.get("mean", 0.0)) > 0 and float(full["recall@1"]) >= float(scalar["recall@1"]) and float(full["mrr@5"]) >= float(scalar["mrr@5"]) and float(full["multi_gold_recall@5"]) >= float(scalar["multi_gold_recall@5"]) - .002
    return "PROMOTE_FROZEN_JINA" if promote else "KEEP_WEAK_FROZEN_JINA" if delta > 0 else "REJECT_FROZEN_JINA_TRANSFER"


def evaluate_fold0_frozen(*, outer: str = FOLD0, authorize: bool = False) -> dict[str, Any]:
    """The single authorized Fold-0 label read, strictly after prediction lock."""
    require_authorization("evaluate-fold0-frozen", authorize, "EXP109C_ALLOW_FOLD0")
    report_path = RESULTS_ROOT / "FROZEN_WINNER_FOLD0_REPORT.json"
    if report_path.exists(): raise GateRejected("REJECTED_FOLD0_ALREADY_EVALUATED", "Fold-0 evaluation is locked to one label-read event", report=report_path)
    lock = _frozen_lock(); prediction_lock_path = RESULTS_ROOT / "FOLD0_PREDICTION_LOCK.json"; prediction_lock = require_report_status(prediction_lock_path, ("PASS_FOLD0_PREDICTION_LOCK",))
    if prediction_lock.get("winner_lock_fingerprint") != lock.get("lock_fingerprint") or prediction_lock.get("labels_read") is not False:
        raise GateRejected("REJECTED_PREDICTION_CONTRACT", "Fold-0 prediction lock is not compatible with frozen winner")
    scalar_path = RESULTS_ROOT / "FOLD0_SCALAR_PREDICTIONS.json"; full_path = RESULTS_ROOT / "FOLD0_FULL_LATENT_PREDICTIONS.json"
    if sha256_file(scalar_path) != prediction_lock.get("scalar_prediction_sha256") or sha256_file(full_path) != prediction_lock.get("full_prediction_sha256"):
        raise GateRejected("REJECTED_PREDICTION_CONTRACT", "prediction artifact hash differs from the pre-label lock")
    qids = sorted(fold_qids_without_labels(outer=outer)[outer]); scalar_predictions = _validated_predictions(read_json(scalar_path).get("predictions", {}), qids, label="fold0-scalar"); full_predictions = _validated_predictions(read_json(full_path).get("predictions", {}), qids, label="fold0-full-latent")
    answers, _ = canonical_labels(); answers = {qid: answers[qid] for qid in qids}
    scalar_metrics = _prediction_metrics(scalar_predictions, answers, qids); full_metrics = _prediction_metrics(full_predictions, answers, qids)
    deltas = [recall_fraction(full_predictions[qid], answers[qid], 5) - recall_fraction(scalar_predictions[qid], answers[qid], 5) for qid in qids if answers[qid]]; bootstrap = deterministic_bootstrap(deltas)
    transitions = {"wins": 0, "losses": 0, "ties": 0, "gold_into_top5": 0, "gold_out_of_top5": 0}; oracle: list[float] = []
    for qid in qids:
        base = recall_fraction(scalar_predictions[qid], answers[qid], 5); value = recall_fraction(full_predictions[qid], answers[qid], 5); transitions["wins" if value > base else "losses" if value < base else "ties"] += 1; transitions["gold_into_top5"] += int(base == 0 and value > 0); transitions["gold_out_of_top5"] += int(base > 0 and value == 0); oracle.append(max(base, value))
    report = {"schema_version": SCHEMA, "stage": "evaluate-fold0-frozen", "outer": outer, "status": _frozen_transfer_status(scalar=scalar_metrics, full=full_metrics, bootstrap=bootstrap), "matched_scalar_anchor": scalar_metrics, "full_latent_winner": full_metrics, "historical_exp109b_fold0_recall@5": .9331783500238435, "delta_winner_vs_matched_scalar": full_metrics["recall@5"] - scalar_metrics["recall@5"], "delta_winner_vs_exp109b_historical_fold0": full_metrics["recall@5"] - .9331783500238435, "bootstrap": bootstrap, "transitions": transitions, "choice_oracle_recall@5": float(np.mean(oracle)), "choice_oracle_label_dependent_not_deployable": True, "milestones": {"PASS_095": full_metrics["recall@5"] >= .950, "PASS_096": full_metrics["recall@5"] >= .960, "PASS_097": full_metrics["recall@5"] >= .970}, "prediction_lock_sha256": sha256_file(prediction_lock_path), "winner_lock_fingerprint": lock["lock_fingerprint"], "no_adapter_training": True, "no_full_oof": True, "no_public_submission": True, "claim_boundary": "one locked Fold-0 transfer evaluation; not OOF, public, or submission Recall"}
    return write_report(report_path, report, stage="evaluate-fold0-frozen", success=True)


def d100_futility_decision(report: Mapping[str, Any], *, target_query_count: int) -> dict[str, Any]:
    """Conservative, precommitted D100 escalation decision; never a Fold-0 claim."""
    delta = float(report.get("delta_recall@5", 0.0)); bootstrap = report.get("bootstrap", {}); upper = float(bootstrap.get("upper", 0.0)); lower = float(bootstrap.get("lower", 0.0)); oracle = float(report.get("choice_oracle_gain", 0.0)); folds = int(report.get("folds_non_negative", 0)); multi = float(report.get("multi_gold_delta", 0.0)); worst = float(report.get("worst_fold_delta", 0.0))
    reject_checks = {"delta_le_zero": delta <= 0.0, "bootstrap_upper_lt_0_005": upper < .005, "oracle_lt_0_015": oracle < .015, "at_most_one_fold_improves": folds <= 1, "multi_gold_delta_le_zero": multi <= 0.0}
    continue_checks = {"stable_delta_ge_0_003": delta >= .003 and folds >= 3 and multi >= 0.0, "bootstrap_lower_gt_zero": lower > 0.0, "oracle_ge_0_020_and_worst_ge_minus_0_005": oracle >= .020 and worst >= -.005}
    # A futility stop requires the complete precommitted adverse pattern.  A
    # single weak diagnostic (especially a boundary-value oracle) is evidence
    # of ambiguity, not authorization to discard the reusable full-inner run.
    if all(reject_checks.values()): status = f"REJECTED_D100_FUTILITY_{target_query_count}"; action = "stop_before_next_budget"
    elif any(continue_checks.values()): status = f"CONTINUE_D100_FUTILITY_{target_query_count}"; action = "run_next_precommitted_budget"
    else: status = f"AMBIGUOUS_D100_FUTILITY_{target_query_count}"; action = "run_next_precommitted_budget"
    return {"status": status, "action": action, "rejection_policy": "all_reject_checks_required", "reject_checks": reject_checks, "continue_checks": continue_checks, "target_query_count": target_query_count}


def run_d100_futility_screen(*, outer: str = FOLD0, target_query_count: int) -> dict[str, Any]:
    """Evaluate a deterministic D100 pilot only; full screen remains a separate gate."""
    score_rows, manifest = load_cached_score_rows(outer=outer, target_query_count=target_query_count)
    selection = manifest.get("query_selection", {}); selected = set(map(str, selection.get("query_ids", ())))
    if len(selected) != target_query_count or {str(row.get("qid")) for row in score_rows} != selected:
        raise GateRejected("REJECTED_STALE_ARTIFACT", "D100 score cache does not exactly match the requested deterministic futility selection")
    source_rows = load_exp109b_anchor_sources(outer=outer, qids=selected); answers, _ = canonical_labels(); all_folds, _ = load_folds()
    folds = {name: [str(qid) for qid in qids if str(qid) in selected] for name, qids in all_folds.items()}
    anchor, anchor_meta = nested_inner_lambdamart(score_rows, answers, folds, outer=outer, feature_names=SCALAR_109B_FEATURES)
    raw, _raw_meta = nested_inner_lambdamart(score_rows, answers, folds, outer=outer, feature_names=SCALAR_109B_FEATURES + ("li_two_chunk_union_mean",))
    full, _full_meta = nested_inner_lambdamart(score_rows, answers, folds, outer=outer, feature_names=ALLOWED_FEATURES)
    standalone = rank_from_score_rows(score_rows)
    report = frozen_inner_screen_report(score_rows, source_rows, answers, folds, outer=outer, anchor_predictions=anchor, anchor_provenance={"model": "EXP-109B scalar LambdaMART reproduced", "usable_for_gate": True, **anchor_meta}, late_predictions=full, model_views={"B_jina_exact_maxsim_standalone": standalone, "C_exp109b_plus_raw_late_lambdamart": raw, "D_exp109b_plus_full_latent_block_lambdamart": full})
    report.update({"stage": "d100-futility-screen", "query_selection": selection, "futility": d100_futility_decision(report, target_query_count=target_query_count), "claim_boundary": "precommitted D100 inner-pilot futility evidence only; not the full frozen gate or Fold-0/public Recall"})
    report["status"] = report["futility"]["status"]
    return write_report(RESULTS_ROOT / f"D100_FUTILITY_{target_query_count}_SCREEN.json", report, stage="d100-futility-screen")


# ---------------------------------------------------------------------------
# Adapter, gates and orchestration reports
# ---------------------------------------------------------------------------


if torch is not None:
    class LowRankMetricAdapter(torch.nn.Module):
        def __init__(self, dimensions: int = DIMENSION, rank: int = 4, *, seed: int = 109):
            super().__init__()
            if rank not in {4, 8}: raise ValueError("adapter rank must be 4 or 8")
            generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
            self.query_u = torch.nn.Parameter(torch.zeros(dimensions, rank)); self.query_v = torch.nn.Parameter(torch.randn(rank, dimensions, generator=generator) * .001)
            self.document_u = torch.nn.Parameter(torch.zeros(dimensions, rank)); self.document_v = torch.nn.Parameter(torch.randn(rank, dimensions, generator=generator) * .001)

        @staticmethod
        def _apply(values: Any, left: Any, right: Any) -> Any:
            transformed = values @ (torch.eye(values.shape[-1], device=values.device, dtype=values.dtype) + left @ right)
            return F.normalize(transformed.float(), p=2, dim=-1)

        def transform_query(self, values: Any) -> Any:
            return self._apply(values, self.query_u, self.query_v)

        def transform_document(self, values: Any) -> Any:
            return self._apply(values, self.document_u, self.document_v)

        def forward(self, query: Any, document: Any) -> tuple[Any, Any]:
            return self.transform_query(query), self.transform_document(document)
else:  # pragma: no cover
    LowRankMetricAdapter = None  # type: ignore[assignment,misc]


def adapter_identity_score_parity(*, dimensions: int = DIMENSION, rank: int = 4) -> float:
    if torch is None: return float("nan")
    adapter = LowRankMetricAdapter(dimensions, rank); q = F.normalize(torch.randn(3, dimensions), p=2, dim=-1); d = F.normalize(torch.randn(5, dimensions), p=2, dim=-1); aq, ad = adapter(q, d); return float(torch.max(torch.abs(q - aq)).item() + torch.max(torch.abs(d - ad)).item())


def frozen_inner_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {"delta_ge_0_005": float(metrics.get("delta_recall@5", -math.inf)) >= .005, "bootstrap_mean_gt_0": float(metrics.get("bootstrap", {}).get("mean", -math.inf)) > 0, "three_folds_non_negative": int(metrics.get("folds_non_negative", 0)) >= 3, "multi_gold_non_decrease": float(metrics.get("multi_gold_delta", -math.inf)) >= 0}
    oracle = float(metrics.get("choice_oracle_gain", 0.0)); alternative = all(checks.values()) or (oracle >= .020 and float(metrics.get("worst_fold_delta", -math.inf)) >= -.005)
    return {"checks": checks, "choice_oracle_branch": oracle >= .020 and float(metrics.get("worst_fold_delta", -math.inf)) >= -.005, "pass": alternative, "status": "PASS_FROZEN_LATE_INTERACTION_GATE" if alternative else "REJECTED_FROZEN_LATE_INTERACTION_GATE"}


def final_inner_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "aggregate_recall@5_ge_0_945": float(metrics.get("recall@5", 0.0)) >= .945, "delta_vs_109b_ge_0_015": float(metrics.get("delta_vs_109b", -math.inf)) >= .015,
        "bootstrap_lower_ge_0_008": float(metrics.get("bootstrap_lower", -math.inf)) >= .008, "at_least_3_of_4_folds_improve": int(metrics.get("folds_improved", 0)) >= 3,
        "worst_fold_ge_minus_0_002": float(metrics.get("worst_fold_delta", -math.inf)) >= -.002, "multi_gold_recall_ge_0_760": float(metrics.get("multi_gold_recall@5", 0.0)) >= .760,
        "multi_gold_delta_ge_0_030": float(metrics.get("multi_gold_delta", -math.inf)) >= .030, "precision_non_decrease": bool(metrics.get("precision_non_decrease", False)),
        "mrr_safety": float(metrics.get("mrr_delta", -math.inf)) >= -.001, "recall1_safety": float(metrics.get("recall1_delta", -math.inf)) >= -.002,
    }
    return {"checks": checks, "pass": all(checks.values()), "status": "PASS_FINAL_INNER_GATE" if all(checks.values()) else "WEAK_INNER_SIGNAL_NO_FOLD0"}


def _validated_predictions(predictions: Mapping[str, Sequence[str]], qids: Sequence[str], *, label: str) -> dict[str, list[str]]:
    expected = {str(qid) for qid in qids}; observed = {str(qid) for qid in predictions}
    if observed != expected:
        raise GateRejected("REJECTED_PREDICTION_CONTRACT", f"{label} qid coverage mismatch: missing={len(expected-observed)} unexpected={len(observed-expected)}")
    result: dict[str, list[str]] = {}
    for qid in sorted(expected):
        docs = [str(doc_id) for doc_id in predictions[qid]]
        if len(docs) != len(set(docs)):
            raise GateRejected("REJECTED_PREDICTION_CONTRACT", f"{label} has duplicate document IDs for {qid}")
        result[qid] = docs
    return result


def evaluated_prediction_report(*, stage: str, predictions: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], qids: Sequence[str], anchor_predictions: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Evaluate only after a complete, deterministic prediction set is present."""
    checked = _validated_predictions(predictions, qids, label=stage); metrics = _prediction_metrics(checked, answers, qids)
    report: dict[str, Any] = {"stage": stage, "predictions": checked, "metrics": metrics, "prediction_fingerprint": content_hash(checked)}
    if anchor_predictions is not None:
        anchor = _validated_predictions(anchor_predictions, qids, label=f"{stage}-anchor"); base = _prediction_metrics(anchor, answers, qids)
        deltas = [recall_fraction(checked[qid], answers[qid], 5) - recall_fraction(anchor[qid], answers[qid], 5) for qid in qids if answers.get(qid)]
        report.update({"anchor_metrics": base, "delta_recall@5": metrics["recall@5"] - base["recall@5"], "bootstrap": deterministic_bootstrap(deltas), "fold_predictions_validated": True})
    return report


def train_metric_adapter(*, outer: str = FOLD0, resume: bool = True, authorize: bool = False) -> dict[str, Any]:
    """Run fold-isolated adapter training; no stage may emit a synthetic PASS."""
    require_authorization("train-metric-adapter", authorize, "EXP109C_ALLOW_ADAPTER_TRAINING")
    require_model_use_eligibility()
    frozen_path = RESULTS_ROOT / "FROZEN_INNER_SCREEN.json"
    frozen = read_json(frozen_path) if frozen_path.exists() else {}
    if frozen.get("status") != "PASS_FROZEN_LATE_INTERACTION_GATE":
        raise GateRejected("REJECTED_PREREQUISITE_GATE", "a passing FROZEN_INNER_SCREEN is required before adapter training", report=frozen_path)
    score_rows, _ = load_cached_score_rows(outer=outer); answers, _ = canonical_labels(); folds, _ = load_folds()
    # Batches and checkpoints are rebuilt separately for every held-out inner
    # fold. The scorer deliberately covers every candidate row in that fold;
    # it never evaluates only the training negatives.
    by_qid = {str(row["qid"]): row for row in score_rows}; predictions: dict[str, list[str]] = {}; folds_report: dict[str, Any] = {}
    for heldout in [name for name in sorted(folds) if name != outer]:
        train_qids = [str(qid) for name, qids in folds.items() if name not in {outer, heldout} for qid in qids]
        valid_qids = [str(qid) for qid in folds[heldout]]
        batches = build_adapter_batches_from_cache([by_qid[qid] for qid in train_qids if qid in by_qid], outer=outer, allowed_qids=train_qids)
        if not batches:
            raise GateRejected("REJECTED_ADAPTER_DATA_GATE", f"no fold-isolated adapter batches for {heldout}")
        # The first runtime implementation uses the predeclared grid and
        # retains each state for audited later candidate-pool inference.
        trials = [train_adapter_batches(batches, rank=rank, learning_rate=lr, temperature=temp, seed=seed) for rank in (4, 8) for lr in (.001, .003) for temp in (.05, .10) for seed in SEEDS]
        chosen = min(trials, key=lambda item: (float(item["final_loss"] if item["final_loss"] is not None else math.inf), item["rank"], item["learning_rate"], item["temperature"], item["seed"]))
        # Candidate-pool inference is intentionally fail-closed until the
        # score shards contain raw adapter scores. This is a data-contract
        # error rather than the former unconditional evaluator rejection.
        if not all("adapter_scores" in by_qid[qid] for qid in valid_qids if qid in by_qid):
            raise GateRejected("REJECTED_ADAPTER_SCORE_CACHE_MISSING", "adapter candidate-pool scores must be materialized by the scorer before cross-fit evaluation")
        for qid in valid_qids:
            values = by_qid[qid]["adapter_scores"]
            predictions[qid] = stable_rank([float(item["score"]) for item in values], [str(item["doc_id"]) for item in values])
        folds_report[heldout] = {"train_qids": len(train_qids), "validation_qids": len(valid_qids), "trials": len(trials), "chosen": chosen}
    inner_qids = [str(qid) for name, qids in folds.items() if name != outer for qid in qids]
    anchor = rank_from_score_rows(score_rows)
    evaluated = evaluated_prediction_report(stage="metric-adapter", predictions=predictions, answers=answers, qids=inner_qids, anchor_predictions={qid: anchor[qid] for qid in inner_qids})
    report = {"schema_version": SCHEMA, "stage": "metric-adapter", "outer": outer, "status": "PASS_ADAPTER_CROSSFIT", "folds": folds_report, "inner_metrics": evaluated["metrics"], "evaluation": evaluated, "selection_scope": "F1-F4 fold-isolated; early selection data never includes held-out inner fold", "claim_boundary": "inner adapter evidence only; not Fold-0/public Recall"}
    return write_report(RESULTS_ROOT / "METRIC_ADAPTER_REPORT.json", report, stage="metric-adapter")


def write_final_inner_gate(*, adapter_report: Mapping[str, Any] | None = None, frozen_report: Mapping[str, Any] | None = None, outer: str = FOLD0) -> dict[str, Any]:
    """Materialize the final inner gate only from explicit adapter metrics."""
    adapter = dict(adapter_report or (read_json(RESULTS_ROOT / "METRIC_ADAPTER_REPORT.json") if (RESULTS_ROOT / "METRIC_ADAPTER_REPORT.json").exists() else {}))
    frozen = dict(frozen_report or (read_json(RESULTS_ROOT / "FROZEN_INNER_SCREEN.json") if (RESULTS_ROOT / "FROZEN_INNER_SCREEN.json").exists() else {}))
    metrics = adapter.get("inner_metrics")
    if not isinstance(metrics, Mapping):
        raise GateRejected("REJECTED_PREREQUISITE_GATE", "METRIC_ADAPTER_REPORT.json must contain evaluated inner_metrics before final gate")
    gate = final_inner_gate(metrics)
    report = {"schema_version": SCHEMA, "stage": "final-inner-gate", "outer": outer, "status": gate["status"], "metrics": dict(metrics), "gate": gate, "frozen_screen_status": frozen.get("status"), "claim_boundary": "strict inner selection gate only; not Fold-0/public Recall"}
    return write_report(RESULTS_ROOT / "FINAL_INNER_GATE.json", report, stage="final-inner-gate")


def _position_stratified_indices(length: int, maximum: int, mandatory: Sequence[int] = ()) -> np.ndarray:
    required = sorted(set(int(value) for value in mandatory if 0 <= int(value) < length))
    if len(required) > maximum: raise GateRejected("REJECTED_ANCHOR_BUDGET", "mandatory anchor count exceeds rescue budget")
    if length <= maximum: return np.arange(length, dtype=np.int32)
    positions = np.linspace(0, length - 1, maximum - len(required), dtype=int).tolist()
    return np.asarray(sorted(set(required + positions))[:maximum], dtype=np.int32)


def _diversity_medoids_reference(values: np.ndarray, maximum: int, mandatory: Sequence[int] = ()) -> np.ndarray:
    """Slow, transparent reference retained only for E3 parity checks."""
    if len(values) <= maximum: return np.arange(len(values), dtype=np.int32)
    chosen = sorted(set(int(value) for value in mandatory if 0 <= int(value) < len(values)))
    if len(chosen) > maximum: raise GateRejected("REJECTED_ANCHOR_BUDGET", "mandatory anchor count exceeds rescue budget")
    if not chosen: chosen = [0]
    while len(chosen) < maximum:
        similarities = values @ values[np.asarray(chosen)].T
        minimum_distance = 1.0 - similarities.max(axis=1)
        minimum_distance[np.asarray(chosen)] = -np.inf
        chosen.append(int(np.argmax(minimum_distance)))
    return np.asarray(sorted(chosen), dtype=np.int32)


def _diversity_medoids(values: np.ndarray, maximum: int, mandatory: Sequence[int] = ()) -> np.ndarray:
    """Deterministic incremental farthest-first medoids, O(n*k*d)."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values): raise ValueError("medoids require a non-empty [tokens,dimension] matrix")
    if len(values) <= maximum: return np.arange(len(values), dtype=np.int32)
    chosen = sorted(set(int(value) for value in mandatory if 0 <= int(value) < len(values)))
    if len(chosen) > maximum: raise GateRejected("REJECTED_ANCHOR_BUDGET", "mandatory anchor count exceeds rescue budget")
    if not chosen: chosen = [0]
    # The reference's distance after the initial mandatory set is the maximum
    # similarity to any selected medoid. Compute it once, then update it with
    # only the newly chosen medoid on each iteration.
    min_distance = 1.0 - (values @ values[np.asarray(chosen)].T).max(axis=1)
    min_distance[np.asarray(chosen)] = -np.inf
    while len(chosen) < maximum:
        next_index = int(np.argmax(min_distance)); chosen.append(next_index)
        min_distance = np.minimum(min_distance, 1.0 - (values @ values[next_index]))
        min_distance[np.asarray(chosen)] = -np.inf
    return np.asarray(sorted(chosen), dtype=np.int32)


def config_e_mandatory_indices(length: int) -> list[int]:
    """Locked marker/prefix/tail policy shared by pilot and corpus encoder."""
    return list(range(min(4, length))) + list(range(max(0, length - 8), length))


def config_e_anchor_indices(values128: np.ndarray) -> np.ndarray:
    """Choose positions in renormalized 64d geometry; retain 128d vectors."""
    values = np.asarray(values128, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != CONFIG_E_STORAGE_DIMENSION:
        raise ValueError("Config E requires 128-d token vectors")
    geometry64 = l2_normalize(values[:, :CONFIG_E_SELECTION_DIMENSION])
    return _diversity_medoids(geometry64, CONFIG_E_ANCHORS, config_e_mandatory_indices(len(values)))


def _rescue_chunk_vectors(chunks64: Sequence[np.ndarray], query64: np.ndarray, selector: str) -> np.ndarray:
    if selector == "all": return np.concatenate(chunks64, axis=0)
    if selector == "top8-full":
        ranked = sorted(range(len(chunks64)), key=lambda index: (-numpy_maxsim(query64, chunks64[index])[0], index))[:8]
        return np.concatenate([chunks64[index] for index in ranked], axis=0)
    values: list[np.ndarray] = []
    for chunk in chunks64:
        mandatory = list(range(min(4, len(chunk)))) + list(range(max(0, len(chunk) - 8), len(chunk)))
        indices = _position_stratified_indices(len(chunk), 256, mandatory) if selector == "position" else _diversity_medoids(chunk, 256, mandatory)
        values.append(chunk[indices])
    return np.concatenate(values, axis=0)


def _rescue_selection(*, answers: Mapping[str, set[str]], folds: Mapping[str, Sequence[str]], train: Mapping[str, Mapping[str, Any]], metadata: Mapping[str, Mapping[str, Any]], excluded_qids: Iterable[str] = ()) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    excluded = {str(qid) for qid in excluded_qids}
    included = [str(qid) for name, qids in sorted(folds.items()) if name != FOLD0 for qid in qids if str(qid) not in excluded]
    ordered = sorted(included, key=lambda qid: (len(str(train[qid].get("question", "")).split()), qid))
    source_qids = [ordered[int(index)] for index in np.linspace(0, len(ordered) - 1, min(768, len(ordered)), dtype=int)]
    sources = load_exp109b_anchor_sources(outer=FOLD0, qids=source_qids)
    viable: list[tuple[int, str, list[str]]] = []
    for qid in sorted(sources):
        ranked = build_candidate_union(sources[qid], depth=CANDIDATE_DEPTH); gold = answers.get(qid, set())
        positive = [str(row["doc_id"]) for row in ranked if str(row["doc_id"]) in gold]
        if not positive: continue
        chosen = list(dict.fromkeys(positive))
        for low, high, count in ((1, 5, 5), (6, 20, 5), (21, 100, 5), (101, 200, 5)):
            band = [str(row["doc_id"]) for row in ranked if low <= int(row["candidate_rank"]) <= high and str(row["doc_id"]) not in gold]
            if band: chosen.extend(band[int(index)] for index in np.linspace(0, len(band) - 1, min(count, len(band)), dtype=int))
        for row in ranked:
            if len(chosen) >= 20: break
            if str(row["doc_id"]) not in chosen: chosen.append(str(row["doc_id"]))
        chosen = list(dict.fromkeys(chosen))[:20]
        if len(chosen) == 20:
            length = int(np.median([int(metadata.get(doc_id, {}).get("parent_token_length", 0)) for doc_id in chosen]))
            viable.append((length, qid, chosen))
    if len(viable) < 32: raise GateRejected("REJECTED_PREREQUISITE_GATE", "rescue fidelity fixture needs 32 inner queries with 20 current candidates")
    viable.sort(key=lambda value: (value[0], value[1])); selected = [viable[int(index)] for index in np.linspace(0, len(viable) - 1, 32, dtype=int)]
    records = [(qid, doc_id) for _length, qid, docs in selected for doc_id in docs]
    modes = Counter(str(metadata.get(doc_id, {}).get("parse_mode", "unknown")) for _qid, doc_id in records)
    if not any("struct" in mode for mode in modes) or not any("fallback" in mode for mode in modes):
        raise GateRejected("REJECTED_PREREQUISITE_GATE", f"rescue fixture lacks required structured/fallback coverage: {dict(modes)}")
    return records, {"queries": [qid for _length, qid, _docs in selected], "excluded_query_count": len(excluded), "candidate_count_per_query": 20, "rank_bands": ["1-5", "6-20", "21-100", "101-200"], "parse_modes": dict(modes)}


def generate_fidelity_fixture(*, device: str, query_max_length: int, document_max_length: int) -> dict[str, Any]:
    """Generate the fixed 32-query x 20-candidate Matryoshka rescue fixture."""
    answers, _ = canonical_labels(); folds, _ = load_folds(); train = load_train(); metadata = load_parent_metadata()
    selected, sampling = _rescue_selection(answers=answers, folds=folds, train=train, metadata=metadata)
    selected_docs = {doc_id for _qid, doc_id in selected}; chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in iter_chunks():
        if str(chunk["doc_id"]) in selected_docs: chunks_by_doc[str(chunk["doc_id"])].append(chunk)
    missing = selected_docs - set(chunks_by_doc)
    if missing: raise GateRejected("REJECTED_INPUT_AUDIT", f"rescue fixture selected parents without structural chunks: {sorted(missing)[:3]}")
    progress_path = CACHE_ROOT / "fidelity_rescue" / "progress.json"; selection_fingerprint = content_hash({"records": selected, "sampling": sampling, "dimension": RESCUE_DIMENSION})
    progress = read_json(progress_path) if progress_path.exists() else {}
    if progress.get("selection_fingerprint") != selection_fingerprint: progress = {}
    records: list[dict[str, Any]] = list(progress.get("records", [])); scores: dict[str, list[float]] = {key: list(progress.get("scores", {}).get(key, [])) for key in ("full128", "full64", "control_idf128", "A_position64", "B_diversity64", "C_all64", "D_top8_full64")}
    if len(records) > len(selected) or any(len(values) != len(records) for values in scores.values()): raise GateRejected("REJECTED_STALE_ARTIFACT", "rescue progress arrays are inconsistent", report=progress_path)
    model, tokenizer, projection = _load_model_for_encoding(device=device); idf, idf_manifest = build_or_load_document_idf(tokenizer)
    query_cache: dict[str, np.ndarray] = {}
    try:
        with tracked_stage("fidelity-rescue", total=len(selected)) as tracker:
            for index, (qid, doc_id) in enumerate(selected[len(records):], start=len(records)):
                if qid not in query_cache:
                    query_cache[qid] = encode_jina_texts(model, tokenizer, projection, [str(train[qid].get("question", ""))], task="query", max_length=query_max_length, device=device)[0][0]
                q128 = query_cache[qid]; q64 = l2_normalize(q128[:, :RESCUE_DIMENSION])
                chunks128: list[np.ndarray] = []; chunks64: list[np.ndarray] = []; control: list[np.ndarray] = []
                for chunk in sorted(chunks_by_doc[doc_id], key=lambda row: str(row["chunk_id"])):
                    values, token_ids, _lengths, _ = encode_jina_texts(model, tokenizer, projection, [str(chunk.get("retrieval_text", ""))], task="document", max_length=document_max_length, device=device)
                    value, ids = values[0], token_ids[0]; chunks128.append(value); chunks64.append(l2_normalize(value[:, :RESCUE_DIMENSION]))
                    pieces = tokenizer.convert_ids_to_tokens(ids.tolist())
                    control.append(value[select_anchor_indices(ids.tolist(), [1] * len(ids), maximum=128, idf=idf, token_texts=pieces)])
                full128 = np.concatenate(chunks128, axis=0); full64 = np.concatenate(chunks64, axis=0)
                candidates = {"control_idf128": np.concatenate(control, axis=0), "A_position64": _rescue_chunk_vectors(chunks64, q64, "position"), "B_diversity64": _rescue_chunk_vectors(chunks64, q64, "diversity"), "C_all64": _rescue_chunk_vectors(chunks64, q64, "all"), "D_top8_full64": _rescue_chunk_vectors(chunks64, q64, "top8-full")}
                scores["full128"].append(numpy_maxsim(q128, full128)[0]); scores["full64"].append(numpy_maxsim(q64, full64)[0])
                for name, vectors in candidates.items():
                    packed, scales = quantize_rows(vectors); query = q128 if name == "control_idf128" else q64
                    scores[name].append(numpy_maxsim(query, dequantize_rows(packed, scales))[0])
                records.append({"qid": qid, "doc_id": doc_id, "positive": doc_id in answers.get(qid, set()), "parent_token_length": int(metadata.get(doc_id, {}).get("parent_token_length", len(full128))), "parent_chunk_count": len(chunks128)})
                atomic_json(progress_path, {"schema_version": SCHEMA, "stage": "fidelity-rescue-progress", "selection_fingerprint": selection_fingerprint, "completed": len(records), "total": len(selected), "records": records, "scores": scores})
                tracker.heartbeat(index + 1, emit=(index == 0 or (index + 1) % 8 == 0 or index + 1 == len(selected)))
    finally:
        del model, tokenizer, projection
        torch.cuda.empty_cache()
    return {"schema_version": SCHEMA, "stage": "automated-fidelity-rescue-fixture", "sampling_policy": sampling, "selection_fingerprint": selection_fingerprint, "input_fingerprints": {"candidate": require_current_candidate_report().get("report_fingerprint"), "idf": idf_manifest["content_fingerprint"]}, "records": records, "scores": scores}


def _invalidate_legacy_fidelity_report() -> None:
    path = RESULTS_ROOT / "FIDELITY_PILOT_REPORT.json"
    if path.exists():
        legacy = RESULTS_ROOT / "invalidated" / "FIDELITY_PILOT_REPORT.invalid_metric_contract.json"
        if not legacy.exists(): shutil.copy2(path, legacy)
    status = read_json(RESULTS_ROOT / "RUN_STATUS.json") if (RESULTS_ROOT / "RUN_STATUS.json").exists() else {}
    status.update({"state": "REJECTED", "stage": "fidelity-pilot", "status": "REJECTED_COMPRESSION_FIDELITY_GATE_INVALID_METRIC_CONTRACT", "last_heartbeat": utc_now(), "reason": "legacy positive-order and global top5 metrics were invalid"})
    atomic_json(RESULTS_ROOT / "RUN_STATUS.json", status)


def run_fidelity_pilot(*, authorize: bool = False) -> dict[str, Any]:
    """Run the one approved grouped Matryoshka fidelity rescue; never index corpus."""
    require_authorization("fidelity-pilot", authorize, "EXP109C_ALLOW_FIDELITY_PILOT"); require_model_use_eligibility(); require_current_candidate_report()
    if torch is None or not torch.cuda.is_available(): raise GateRejected("REJECTED_RESOURCE_GATE", "fidelity pilot requires CUDA; CPU fixture generation is forbidden")
    preflight = read_json(RESULTS_ROOT / "PREFLIGHT.json") if (RESULTS_ROOT / "PREFLIGHT.json").exists() else {}
    if preflight.get("status") != "PASS": raise GateRejected("REJECTED_PREREQUISITE_GATE", "PASS PREFLIGHT.json is required before fidelity pilot")
    _invalidate_legacy_fidelity_report()
    fixture = generate_fidelity_fixture(device="cuda", query_max_length=int(preflight["query_budget"]["selected"]), document_max_length=int(preflight["document_budget"]["selected"]))
    fixture_path = CACHE_ROOT / "fidelity_rescue" / "fixture.json"; atomic_json(fixture_path, fixture)
    records = fixture["records"]; common = {"qids": [row["qid"] for row in records], "doc_ids": [row["doc_id"] for row in records], "gold_by_qid": {qid: canonical_labels()[0].get(qid, set()) for qid in {row["qid"] for row in records}}, "parent_lengths": [row["parent_token_length"] for row in records], "parent_chunk_counts": [row["parent_chunk_count"] for row in records]}
    dimension = fidelity_metrics(fixture["scores"]["full128"], fixture["scores"]["full64"], **common); dimension_gate = fidelity_gate(dimension, anchors=-1)
    configs = {"Control": (128, "control_idf128", "idf current 128d"), "A": (256, "A_position64", "mandatory + position-stratified 64d"), "B": (256, "B_diversity64", "mandatory + embedding-diversity medoids 64d"), "C": (0, "C_all64", "all tokens 64d int8 oracle"), "D": (0, "D_top8_full64", "full tokens in query-aware top-8 chunks 64d")}
    attempts = []
    for name, (anchors, score_key, selector) in configs.items():
        reference = fixture["scores"]["full128"] if name == "Control" else fixture["scores"]["full64"]
        metrics = fidelity_metrics(reference, fixture["scores"][score_key], **common); gate = fidelity_gate(metrics, anchors=anchors)
        attempts.append({"config": name, "dimension": DIMENSION if name == "Control" else RESCUE_DIMENSION, "anchors_per_chunk": anchors if anchors else "all", "selector": selector, "metrics": metrics, "gate": gate})
    passed = [item["config"] for item in attempts if item["gate"]["pass"] and (item["config"] == "Control" or dimension_gate["pass"])]
    status = "PASS_FIDELITY_RESCUE" if passed else "REJECTED_COMPRESSION_FIDELITY_RESCUE_GATE"
    report = {"schema_version": SCHEMA, "stage": "fidelity-pilot", "status": status, "legacy_report": {"status": "REJECTED_COMPRESSION_FIDELITY_GATE_INVALID_METRIC_CONTRACT", "path": str((RESULTS_ROOT / "invalidated" / "FIDELITY_PILOT_REPORT.invalid_metric_contract.json").resolve())}, "fixture": {"path": str(fixture_path.resolve()), "queries": 32, "candidates_per_query": 20, "pairs": len(records), "sampling": fixture["sampling_policy"]}, "dimension_fidelity": {"reference": "full-token 128d", "candidate": "full-token 64d", "metrics": dimension, "gate": dimension_gate}, "attempts": attempts, "passed_configurations": passed, "claim_boundary": "grouped compression/dimension fidelity only; no corpus encoding or retrieval claim"}
    report = write_report(RESULTS_ROOT / "FIDELITY_PILOT_REPORT.json", report, stage="fidelity-pilot", success=status == "PASS_FIDELITY_RESCUE")
    run_status = read_json(RESULTS_ROOT / "RUN_STATUS.json"); run_status.update({"state": "PASS" if status.startswith("PASS") else "REJECTED", "status": status, "stage": "fidelity-pilot", "last_heartbeat": utc_now()}); atomic_json(RESULTS_ROOT / "RUN_STATUS.json", run_status)
    return report


def _config_e_common(records: Sequence[Mapping[str, Any]], answers: Mapping[str, set[str]]) -> dict[str, Any]:
    return {"qids": [str(row["qid"]) for row in records], "doc_ids": [str(row["doc_id"]) for row in records], "gold_by_qid": {str(row["qid"]): answers.get(str(row["qid"]), set()) for row in records}, "parent_lengths": [int(row["parent_token_length"]) for row in records], "parent_chunk_counts": [int(row["parent_chunk_count"]) for row in records]}


def _config_e_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    base = fidelity_gate(metrics, anchors=CONFIG_E_ANCHORS)
    checks = dict(base["checks"])
    full_recall, compressed_recall = metrics.get("full_gold_recall_at5"), metrics.get("compressed_gold_recall_at5")
    checks["gold_recall_at5_no_decrease"] = full_recall is not None and compressed_recall is not None and float(compressed_recall) >= float(full_recall) - 1e-12
    return {"anchors": CONFIG_E_ANCHORS, "checks": checks, "pass": all(checks.values()), "status": "PASS_CONFIG_E_FIDELITY" if all(checks.values()) else "REJECTED_CONFIG_E_FIDELITY_GATE"}


def _record_terminal_stage_status(stage: str, status: str) -> None:
    """Keep RUN_STATUS truthful after a normal-returning gate rejection."""
    value = read_json(RESULTS_ROOT / "RUN_STATUS.json") if (RESULTS_ROOT / "RUN_STATUS.json").exists() else {}
    value.update({"stage": stage, "state": "PASS" if status.startswith("PASS") else "REJECTED", "status": status, "last_heartbeat": utc_now(), "code_fingerprint": code_fingerprint()})
    atomic_json(RESULTS_ROOT / "RUN_STATUS.json", value)


def _config_e_score_fixture(*, stage: str, records_input: Sequence[Mapping[str, Any]], sampling: Mapping[str, Any], output: Path, device: str, query_max_length: int, document_max_length: int) -> dict[str, Any]:
    """Score a frozen 32x20 fixture without changing its query/candidate sample."""
    answers, _ = canonical_labels(); train = load_train(); metadata = load_parent_metadata()
    selected = [(str(row["qid"]), str(row["doc_id"])) for row in records_input]
    if len(selected) != 640 or len({qid for qid, _doc_id in selected}) != 32:
        raise GateRejected("REJECTED_FIDELITY_FIXTURE_CONTRACT", "Config E requires exactly 32 queries x 20 candidates")
    per_query = Counter(qid for qid, _doc_id in selected)
    if any(count != 20 for count in per_query.values()): raise GateRejected("REJECTED_FIDELITY_FIXTURE_CONTRACT", "each Config E fixture query must have exactly 20 candidates")
    selected_docs = {doc_id for _qid, doc_id in selected}; chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in iter_chunks():
        if str(chunk["doc_id"]) in selected_docs: chunks_by_doc[str(chunk["doc_id"])].append(chunk)
    if selected_docs - set(chunks_by_doc): raise GateRejected("REJECTED_INPUT_AUDIT", "Config E fixture has parents missing structural chunks")
    fingerprint = content_hash({"stage": stage, "records": selected, "sampling": dict(sampling), "contract": CONFIG_E_CONTRACT})
    progress_path = output / "progress.json"; progress = read_json(progress_path) if progress_path.exists() else {}
    if progress.get("fingerprint") != fingerprint: progress = {}
    records: list[dict[str, Any]] = list(progress.get("records", [])); full_scores = list(progress.get("full128_scores", [])); config_scores = list(progress.get("config_e_scores", []))
    if not (len(records) == len(full_scores) == len(config_scores) <= len(selected)): raise GateRejected("REJECTED_STALE_ARTIFACT", "Config E fixture progress is inconsistent", report=progress_path)
    model, tokenizer, projection = _load_model_for_encoding(device=device); query_cache: dict[str, np.ndarray] = {}
    try:
        with tracked_stage(stage, total=len(selected)) as tracker:
            for index, (qid, doc_id) in enumerate(selected[len(records):], start=len(records)):
                if qid not in query_cache:
                    query_cache[qid] = encode_jina_texts(model, tokenizer, projection, [str(train[qid].get("question", ""))], task="query", max_length=query_max_length, device=device)[0][0]
                q128 = query_cache[qid]; full_chunks: list[np.ndarray] = []; retained_chunks: list[np.ndarray] = []
                for chunk in sorted(chunks_by_doc[doc_id], key=lambda row: str(row["chunk_id"])):
                    values = encode_jina_texts(model, tokenizer, projection, [str(chunk.get("retrieval_text", ""))], task="document", max_length=document_max_length, device=device)[0][0]
                    full_chunks.append(values); retained_chunks.append(values[config_e_anchor_indices(values)])
                full = np.concatenate(full_chunks, axis=0); retained = np.concatenate(retained_chunks, axis=0); packed, scales = quantize_rows(retained)
                full_scores.append(numpy_maxsim(q128, full)[0]); config_scores.append(numpy_maxsim(q128, dequantize_rows(packed, scales))[0])
                records.append({"qid": qid, "doc_id": doc_id, "positive": doc_id in answers.get(qid, set()), "parent_token_length": int(metadata.get(doc_id, {}).get("parent_token_length", len(full))), "parent_chunk_count": len(full_chunks), "retained_token_count": int(len(retained))})
                atomic_json(progress_path, {"schema_version": SCHEMA, "stage": stage, "fingerprint": fingerprint, "completed": len(records), "total": len(selected), "records": records, "full128_scores": full_scores, "config_e_scores": config_scores})
                tracker.heartbeat(index + 1, emit=(index == 0 or (index + 1) % 8 == 0 or index + 1 == len(selected)))
    finally:
        del model, tokenizer, projection
        torch.cuda.empty_cache()
    common = _config_e_common(records, answers); metrics = fidelity_metrics(full_scores, config_scores, **common); gate = _config_e_gate(metrics)
    fixture = {"schema_version": SCHEMA, "stage": stage, "contract": CONFIG_E_CONTRACT, "selection_geometry": "renormalized_first_64d", "storage_scoring_dimension": CONFIG_E_STORAGE_DIMENSION, "anchors_per_chunk": CONFIG_E_ANCHORS, "sampling": dict(sampling), "records": records, "full128_scores": full_scores, "config_e_scores": config_scores, "fingerprint": fingerprint}
    atomic_json(output / "fixture.json", fixture)
    return {"fixture": fixture, "metrics": metrics, "gate": gate}


def run_config_e_existing_fixture(*, authorize: bool = False) -> dict[str, Any]:
    require_authorization("config-e-existing", authorize, "EXP109C_ALLOW_FIDELITY_PILOT"); require_model_use_eligibility(); require_current_candidate_report()
    old = read_json(CACHE_ROOT / "fidelity_rescue" / "fixture.json") if (CACHE_ROOT / "fidelity_rescue" / "fixture.json").exists() else {}
    records = old.get("records") if isinstance(old, Mapping) else None
    if not isinstance(records, list): raise GateRejected("REJECTED_PREREQUISITE_GATE", "the completed rescue fixture is required for Config E1")
    preflight = require_report_status(RESULTS_ROOT / "PREFLIGHT.json", ("PASS",))
    scored = _config_e_score_fixture(stage="config-e-existing", records_input=records, sampling={"source": "fidelity_rescue existing fixture", "selection_fingerprint": old.get("selection_fingerprint")}, output=CACHE_ROOT / "config_e" / "existing", device="cuda", query_max_length=int(preflight["query_budget"]["selected"]), document_max_length=int(preflight["document_budget"]["selected"]))
    report = {"schema_version": SCHEMA, "stage": "config-e-existing", "status": scored["gate"]["status"], "fixture": str((CACHE_ROOT / "config_e" / "existing" / "fixture.json").resolve()), "metrics": scored["metrics"], "gate": scored["gate"], "contract": CONFIG_E_CONTRACT, "claim_boundary": "E1 existing-fixture compression fidelity only; no corpus encoding"}
    result = write_report(RESULTS_ROOT / "CONFIG_E_EXISTING_FIDELITY_REPORT.json", report, stage="config-e-existing", success=scored["gate"]["pass"]); _record_terminal_stage_status("config-e-existing", result["status"]); return result


def run_config_e_confirmation_fixture(*, authorize: bool = False) -> dict[str, Any]:
    require_authorization("config-e-confirmation", authorize, "EXP109C_ALLOW_FIDELITY_PILOT"); require_model_use_eligibility(); require_report_status(RESULTS_ROOT / "CONFIG_E_EXISTING_FIDELITY_REPORT.json", ("PASS_CONFIG_E_FIDELITY",))
    existing = read_json(CACHE_ROOT / "config_e" / "existing" / "fixture.json"); excluded = {str(row["qid"]) for row in existing.get("records", [])}
    answers, _ = canonical_labels(); folds, _ = load_folds(); train = load_train(); selected, sampling = _rescue_selection(answers=answers, folds=folds, train=train, metadata=load_parent_metadata(), excluded_qids=excluded)
    if excluded & {qid for qid, _doc_id in selected}: raise GateRejected("REJECTED_FIDELITY_FIXTURE_CONTRACT", "confirmation fixture overlaps E1 query ids")
    preflight = require_report_status(RESULTS_ROOT / "PREFLIGHT.json", ("PASS",))
    records = [{"qid": qid, "doc_id": doc_id} for qid, doc_id in selected]
    scored = _config_e_score_fixture(stage="config-e-confirmation", records_input=records, sampling={**sampling, "disjoint_from_existing_queries": True}, output=CACHE_ROOT / "config_e" / "confirmation", device="cuda", query_max_length=int(preflight["query_budget"]["selected"]), document_max_length=int(preflight["document_budget"]["selected"]))
    report = {"schema_version": SCHEMA, "stage": "config-e-confirmation", "status": scored["gate"]["status"], "fixture": str((CACHE_ROOT / "config_e" / "confirmation" / "fixture.json").resolve()), "metrics": scored["metrics"], "gate": scored["gate"], "contract": CONFIG_E_CONTRACT, "excluded_existing_queries": len(excluded), "claim_boundary": "E2 disjoint confirmation only; no corpus encoding"}
    result = write_report(RESULTS_ROOT / "CONFIG_E_CONFIRMATION_REPORT.json", report, stage="config-e-confirmation", success=scored["gate"]["pass"]); _record_terminal_stage_status("config-e-confirmation", result["status"]); return result


def write_e2_cutoff_diagnostic() -> dict[str, Any]:
    """Persist the audited E2 near-tie; it is diagnostic, never a promotion."""
    fixture_path = CACHE_ROOT / "config_e" / "confirmation" / "fixture.json"
    fixture = read_json(fixture_path) if fixture_path.exists() else {}
    records = fixture.get("records", [])
    full = fixture.get("full128_scores", [])
    approximate = fixture.get("config_e_scores", [])
    rows = [(str(row.get("doc_id")), float(full[index]), float(approximate[index])) for index, row in enumerate(records) if str(row.get("qid")) == "51028"]
    if len(rows) != 20 or not any(doc == "166505" for doc, _f, _a in rows):
        raise GateRejected("REJECTED_E2_DIAGNOSTIC_CONTRACT", "E2 fixture no longer contains the audited query 51028/gold 166505")
    ids = [row[0] for row in rows]
    full_rank = stable_rank([row[1] for row in rows], ids)
    approx_rank = stable_rank([row[2] for row in rows], ids)
    gold = "166505"; full_neighbor = full_rank[5]; approx_neighbor = approx_rank[4]
    full_map, approx_map = {doc: score for doc, score, _ in rows}, {doc: score for doc, _, score in rows}
    report = {"schema_version": SCHEMA, "stage": "config-e2-cutoff-diagnostic", "status": "PASS_DIAGNOSTIC_ONLY", "fixture": str(fixture_path.resolve()), "query_id": "51028", "gold_doc_id": gold, "full": {"gold_rank": full_rank.index(gold) + 1, "neighbor_rank6": full_neighbor, "gold_score": full_map[gold], "neighbor_score": full_map[full_neighbor], "margin": full_map[gold] - full_map[full_neighbor]}, "config_e": {"gold_rank": approx_rank.index(gold) + 1, "neighbor_rank5": approx_neighbor, "gold_score": approx_map[gold], "neighbor_score": approx_map[approx_neighbor], "margin": approx_map[approx_neighbor] - approx_map[gold]}, "claim_boundary": "post-hoc diagnosis of frozen E2 only; not an E4 gate or a change to E2 rejection"}
    return write_report(RESULTS_ROOT / "E2_CUTOFF_DIAGNOSTIC.json", report, stage="config-e2-cutoff-diagnostic")


def _e4_anchor_predictions(*, outer: str = FOLD0) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Reproduce and cache strict EXP-109B cross-fit LambdaMART predictions."""
    path = CACHE_ROOT / "config_e" / "e4" / "anchor_predictions.json"
    # These predictions are an immutable EXP-109B anchor.  Their validity is
    # determined by the frozen pilot evidence and input contract, not by every
    # subsequent EXP-109C scoring implementation edit.  Coupling this cache to
    # code_fingerprint() made an unrelated scorer change retrain four inner
    # LambdaMART models before an otherwise resumable score run.
    pilot_path = EXP109B_RESULTS / "cached_fusion_pilot" / outer / "CACHED_FUSION_PILOT.json"
    pilot_sha256 = sha256_file(pilot_path) if pilot_path.exists() else None
    fingerprint = content_hash({"structural": STRUCTURAL_FINGERPRINT, "labels": LABEL_FINGERPRINT, "outer": outer, "pilot_sha256": pilot_sha256, "contract": "exp109b_strict_crossfit_lambdamart_top16_v1"})
    if path.exists():
        cached = read_json(path)
        predictions = cached.get("predictions")
        provenance = dict(cached.get("provenance", {}))
        # Migrate the pre-contract cache only when its own provenance proves it
        # came from this exact frozen EXP-109B pilot and outer exclusion.
        compatible_legacy_cache = (
            provenance.get("pilot_sha256") == pilot_sha256
            and provenance.get("outer_excluded") == outer
        )
        if isinstance(predictions, Mapping) and (cached.get("fingerprint") == fingerprint or compatible_legacy_cache):
            if cached.get("fingerprint") != fingerprint:
                atomic_json(path, {"fingerprint": fingerprint, "predictions": predictions, "provenance": provenance})
            return {str(qid): list(map(str, value)) for qid, value in predictions.items()}, provenance
    # Anchor reproduction is CPU-only but can be several minutes.  Use the
    # normal tracker rather than a hand-written RUN_STATUS so its run_id has a
    # matching log directory and every fold becomes a durable checkpoint.
    with tracked_stage("config-e4-anchor-crossfit", outer=outer, total=4) as tracker:
        tracker.log("loading complete strict-inner source rankings", emit=True)
        folds, _ = load_folds()
        # The model must be trained on the full strict inner partitions.
        # Loading only fixture qids would silently change training data.
        source_rows = load_exp109b_anchor_sources(outer=outer)
        def checkpoint(completed: int, validation: str) -> None:
            tracker.heartbeat(completed, emit=True, status="RUNNING_EXP109B_STRICT_CROSSFIT_ANCHOR", checkpoint_unit="inner validation fold", last_completed_fold=validation)
        predictions, provenance = reproduce_109b_anchor_predictions(source_rows, folds, outer=outer, progress_callback=checkpoint)
    atomic_json(path, {"fingerprint": fingerprint, "predictions": predictions, "provenance": provenance})
    return predictions, provenance


def _e4_gate(*, full_top5_contained: float, cascade_agreement: float, full_recall: float, cascade_recall: float, conditional_retention: float, exact_sizes: Sequence[int], candidate_counts: Sequence[int]) -> dict[str, Any]:
    checks = {"full_reference_top5_contained_in_exact_set": full_top5_contained >= 1.0 - 1e-12, "cascade_top5_agreement_ge_0_99": cascade_agreement >= .99, "gold_recall_at5_no_decrease": cascade_recall >= full_recall - 1e-12, "conditional_positive_order_retention_ge_0_99": conditional_retention >= .99, "no_exact_set_explosion": bool(exact_sizes) and max(exact_sizes) <= 32 and all(size <= count for size, count in zip(exact_sizes, candidate_counts))}
    passed = all(checks.values())
    return {"checks": checks, "pass": passed, "status": "PASS_CONFIG_E4_EXACT_REFINEMENT" if passed else "REJECTED_CONFIG_E4_EXACT_REFINEMENT_GATE"}


def run_config_e4_exact_refinement(*, authorize: bool = False) -> dict[str, Any]:
    """Disjoint E4 confirmation of the locked approximate-to-exact cascade."""
    require_authorization("config-e4-exact-refinement", authorize, "EXP109C_ALLOW_FIDELITY_PILOT"); require_model_use_eligibility(); require_current_candidate_report()
    require_report_status(RESULTS_ROOT / "CONFIG_E_EXISTING_FIDELITY_REPORT.json", ("PASS_CONFIG_E_FIDELITY",))
    write_e2_cutoff_diagnostic()
    e1 = read_json(CACHE_ROOT / "config_e" / "existing" / "fixture.json"); e2 = read_json(CACHE_ROOT / "config_e" / "confirmation" / "fixture.json")
    excluded = {str(row["qid"]) for item in (e1, e2) for row in item.get("records", [])}
    answers, _ = canonical_labels(); folds, _ = load_folds(); train = load_train(); metadata = load_parent_metadata()
    selected, sampling = _rescue_selection(answers=answers, folds=folds, train=train, metadata=metadata, excluded_qids=excluded)
    if excluded & {qid for qid, _doc in selected}: raise GateRejected("REJECTED_FIDELITY_FIXTURE_CONTRACT", "E4 overlaps E1/E2 query ids")
    preflight = require_report_status(RESULTS_ROOT / "PREFLIGHT.json", ("PASS",))
    anchors, provenance = _e4_anchor_predictions()
    output = CACHE_ROOT / "config_e" / "e4"; progress_path = output / "progress.json"
    pairs_by_qid: dict[str, list[str]] = defaultdict(list)
    for qid, doc_id in selected: pairs_by_qid[qid].append(doc_id)
    selected_docs = {doc for _qid, doc in selected}; chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in iter_chunks():
        if str(chunk["doc_id"]) in selected_docs: chunks_by_doc[str(chunk["doc_id"])].append(chunk)
    fingerprint = content_hash({"stage": "config-e4-exact-refinement", "records": selected, "excluded": sorted(excluded), "policy": "top16_config_e_union_top16_exp109b_lambdamart", "contract": CONFIG_E_CONTRACT})
    progress = read_json(progress_path) if progress_path.exists() else {}
    groups = list(progress.get("groups", [])) if progress.get("fingerprint") == fingerprint else []
    if len(groups) > 32 or len({str(group.get("qid")) for group in groups}) != len(groups): raise GateRejected("REJECTED_STALE_ARTIFACT", "E4 progress groups are inconsistent")
    remaining = [qid for qid in sampling["queries"] if qid not in {str(group["qid"]) for group in groups}]
    model, tokenizer, projection = _load_model_for_encoding(device="cuda"); started = time.monotonic()
    try:
        with tracked_stage("config-e4-exact-refinement", total=32) as tracker:
            for qid in remaining:
                q = encode_jina_texts(model, tokenizer, projection, [str(train[qid]["question"])], task="query", max_length=int(preflight["query_budget"]["selected"]), device="cuda")[0][0]
                approximate: dict[str, float] = {}; full: dict[str, float] = {}; chunks_cache: dict[str, list[np.ndarray]] = {}; cost: dict[str, dict[str, int]] = {}
                for doc_id in pairs_by_qid[qid]:
                    chunks = [encode_jina_texts(model, tokenizer, projection, [str(row.get("retrieval_text", ""))], task="document", max_length=int(preflight["document_budget"]["selected"]), device="cuda")[0][0] for row in sorted(chunks_by_doc[doc_id], key=lambda row: str(row["chunk_id"]))]
                    retained = np.concatenate([chunk[config_e_anchor_indices(chunk)] for chunk in chunks], axis=0); packed, scales = quantize_rows(retained)
                    approximate[doc_id] = numpy_maxsim(q, dequantize_rows(packed, scales))[0]
                    full[doc_id] = streaming_parent_maxsim(q, chunks)[0]; chunks_cache[doc_id] = chunks
                    cost[doc_id] = {"chunks": len(chunks), "full_token_vectors": int(sum(len(chunk) for chunk in chunks)), "approximate_token_vectors": int(len(retained))}
                exact_ids = exact_refinement_set(approximate, anchors.get(qid, [])); exact = {doc: streaming_parent_maxsim(q, chunks_cache[doc])[0] for doc in exact_ids}
                reference_rank = stable_rank(list(full.values()), list(full)); cascade_rank = cascade_refined_ranking(approximate, exact)
                groups.append({"qid": qid, "doc_ids": pairs_by_qid[qid], "full_scores": full, "approximate_scores": approximate, "exact_doc_ids": exact_ids, "exact_scores": exact, "reference_rank": reference_rank, "cascade_rank": cascade_rank, "cost": cost})
                atomic_json(progress_path, {"schema_version": SCHEMA, "stage": "config-e4-exact-refinement", "fingerprint": fingerprint, "completed": len(groups), "total": 32, "groups": groups})
                tracker.heartbeat(len(groups), emit=True)
    finally:
        del model, tokenizer, projection; torch.cuda.empty_cache()
    if len(groups) != 32: raise GateRejected("REJECTED_STALE_ARTIFACT", "E4 confirmation did not complete 32 queries")
    top5_contained = []; agreements = []; full_recall = []; cascade_recall = []; conditional = []; recovered = lost = 0; sizes = []; counts = []; exact_tokens = []; exact_chunks = []
    for group in groups:
        qid = str(group["qid"]); reference, cascade, exact_ids = group["reference_rank"], group["cascade_rank"], set(group["exact_doc_ids"])
        top5_contained.append(float(set(reference[:5]).issubset(exact_ids))); agreements.append(float(len(set(reference[:5]) & set(cascade[:5])) / 5)); gold = answers.get(qid, set()); fr = recall_fraction(reference, gold, 5); cr = recall_fraction(cascade, gold, 5); full_recall.append(fr); cascade_recall.append(cr); recovered += int(cr > fr); lost += int(cr < fr); sizes.append(len(exact_ids)); counts.append(len(group["doc_ids"])); exact_tokens.extend(group["cost"][doc]["full_token_vectors"] for doc in exact_ids); exact_chunks.extend(group["cost"][doc]["chunks"] for doc in exact_ids)
        for doc in gold & set(group["doc_ids"]):
            if group["full_scores"][doc] >= max((group["full_scores"][other] for other in group["doc_ids"] if other not in gold), default=float("-inf")):
                conditional.append(float(cascade.index(doc) <= reference.index(doc)))
    metrics = {"full_reference_top5_contained_in_exact_set": float(np.mean(top5_contained)), "cascade_top5_agreement": float(np.mean(agreements)), "full_gold_recall_at5": float(np.mean(full_recall)), "cascade_gold_recall_at5": float(np.mean(cascade_recall)), "conditional_positive_order_retention": float(np.mean(conditional)) if conditional else 1.0, "gold_recovered": recovered, "gold_lost": lost, "exact_set_size": quantile_summary(np.asarray(sizes, dtype=np.float32)), "exact_cost": {"parent_full_token_vectors": quantile_summary(np.asarray(exact_tokens, dtype=np.float32)), "parent_chunk_count": quantile_summary(np.asarray(exact_chunks, dtype=np.float32))}, "elapsed_seconds": time.monotonic() - started}
    gate = _e4_gate(full_top5_contained=metrics["full_reference_top5_contained_in_exact_set"], cascade_agreement=metrics["cascade_top5_agreement"], full_recall=metrics["full_gold_recall_at5"], cascade_recall=metrics["cascade_gold_recall_at5"], conditional_retention=metrics["conditional_positive_order_retention"], exact_sizes=sizes, candidate_counts=counts)
    fixture_path = output / "fixture.json"; atomic_json(fixture_path, {"schema_version": SCHEMA, "stage": "config-e4-exact-refinement", "fingerprint": fingerprint, "policy": "top16_config_e_union_top16_exp109b_lambdamart", "sampling": {**sampling, "disjoint_from_e1_e2": True}, "groups": groups, "anchor_provenance": provenance})
    report = {"schema_version": SCHEMA, "stage": "config-e4-exact-refinement", "status": gate["status"], "fixture": str(fixture_path.resolve()), "policy": "top16 Config E union top16 strict cross-fit EXP-109B LambdaMART; maximum 32", "metrics": metrics, "gate": gate, "excluded_e1_e2_queries": len(excluded), "claim_boundary": "E4 disjoint 32x20 confirmation in its fixture candidate universe; no corpus encoding"}
    result = write_report(RESULTS_ROOT / "CONFIG_E4_EXACT_REFINEMENT_REPORT.json", report, stage="config-e4-exact-refinement", success=gate["pass"]); _record_terminal_stage_status("config-e4-exact-refinement", result["status"]); return result


def run_config_e_production_preflight(*, authorize: bool = False, device: str = "cuda", benchmark_chunks: int = 5_000) -> dict[str, Any]:
    """E3: real-chunk selector parity, hash/resume probe, and 5k throughput."""
    require_authorization("config-e-production-preflight", authorize, "EXP109C_ALLOW_PREFLIGHT_GPU")
    require_report_status(RESULTS_ROOT / "CONFIG_E4_EXACT_REFINEMENT_REPORT.json", ("PASS_CONFIG_E4_EXACT_REFINEMENT",))
    preflight = require_report_status(RESULTS_ROOT / "PREFLIGHT.json", ("PASS",))
    if device != "cuda" or torch is None or not torch.cuda.is_available(): raise GateRejected("REJECTED_RESOURCE_GATE", "Config E preflight requires CUDA")
    positions = set(np.linspace(0, CHUNK_COUNT - 1, benchmark_chunks, dtype=int).tolist()); samples: list[dict[str, Any]] = []
    for index, row in enumerate(iter_chunks()):
        if index in positions: samples.append(row)
    if len(samples) != benchmark_chunks: raise GateRejected("REJECTED_INPUT_AUDIT", "could not assemble deterministic 5k Config E benchmark")
    output = CACHE_ROOT / "config_e" / "production_preflight"; probe_dir = output / "shard_probe"; fingerprint = shard_fingerprint({"config_e_contract": CONFIG_E_CONTRACT, "anchors": CONFIG_E_ANCHORS, "dimension": CONFIG_E_STORAGE_DIMENSION, "selection_dimension": CONFIG_E_SELECTION_DIMENSION})
    model, tokenizer, projection = _load_model_for_encoding(device=device); parity_errors: list[str] = []; retained_counts: list[int] = []; vectors_probe: list[np.ndarray] = []; scales_probe: list[np.ndarray] = []; full_vectors_probe: list[np.ndarray] = []; full_scales_probe: list[np.ndarray] = []; passages_probe: list[dict[str, Any]] = []; cursor = full_cursor = 0; started = time.monotonic()
    try:
        with tracked_stage("config-e-production-preflight", total=len(samples)) as tracker:
            for start in range(0, len(samples), 8):
                batch = samples[start:start + 8]
                vectors, _ids, originals, _truncated = encode_jina_texts(model, tokenizer, projection, [str(row.get("retrieval_text", "")) for row in batch], task="document", max_length=int(preflight["document_budget"]["selected"]), device=device)
                for local, (row, value, original) in enumerate(zip(batch, vectors, originals)):
                    selected = config_e_anchor_indices(value); retained = value[selected]; retained_counts.append(len(retained))
                    if start + local < 20:
                        reference = _diversity_medoids_reference(l2_normalize(value[:, :CONFIG_E_SELECTION_DIMENSION]), CONFIG_E_ANCHORS, config_e_mandatory_indices(len(value)))
                        if not np.array_equal(selected, reference): parity_errors.append(f"chunk {start + local}: optimized/reference medoid positions differ")
                    packed, scales = quantize_rows(retained); full_packed, full_scales = quantize_rows(value)
                    if start + local < 20:
                        meta = read_chunk_meta(row); vectors_probe.append(packed); scales_probe.append(scales); full_vectors_probe.append(full_packed); full_scales_probe.append(full_scales); passages_probe.append({"chunk_id": meta.chunk_id, "doc_id": meta.doc_id, "token_start": cursor, "token_end": cursor + len(packed), "full_token_start": full_cursor, "full_token_end": full_cursor + len(full_packed), "original_tokens": int(original), "retained_tokens": len(packed), "full_retained_tokens": len(full_packed)}); cursor += len(packed); full_cursor += len(full_packed)
                tracker.heartbeat(min(start + len(batch), len(samples)), emit=(start == 0 or (start + len(batch)) % 256 == 0 or start + len(batch) == len(samples)))
    finally:
        del model, tokenizer, projection
        torch.cuda.empty_cache()
    elapsed = max(time.monotonic() - started, 1e-9); probe_vectors = np.concatenate(vectors_probe, axis=0); probe_scales = np.concatenate(scales_probe, axis=0); probe_full_vectors = np.concatenate(full_vectors_probe, axis=0); probe_full_scales = np.concatenate(full_scales_probe, axis=0)
    receipt = save_dual_index_shard(probe_dir, 0, probe_vectors, probe_scales, probe_full_vectors, probe_full_scales, passages_probe, fingerprint=fingerprint); verified = verify_index_shard(probe_dir, receipt, fingerprint=fingerprint)
    # Re-verifying the same receipt is the resume path: it must accept a
    # committed shard without invoking the model or altering a byte.
    resume_verified = verify_index_shard(probe_dir, read_json(probe_dir / "shard-00000.json"), fingerprint=fingerprint)
    disk_estimate = estimate_dual_index_bytes(anchors=CONFIG_E_ANCHORS, dimensions=CONFIG_E_STORAGE_DIMENSION, full_tokens_per_chunk=int(preflight["document_budget"]["selected"]))
    disk = disk_record(); disk_pass = int(disk["free_bytes"]) >= int(disk_estimate["total_upper_bound"] * 1.25)
    passed = not parity_errors and len(retained_counts) == benchmark_chunks and max(retained_counts, default=0) <= CONFIG_E_ANCHORS and verified == resume_verified and bool(receipt.get("dual_store")) and disk_pass
    report = {"schema_version": SCHEMA, "stage": "config-e-production-preflight", "status": "PASS_CONFIG_E_PRODUCTION_PREFLIGHT" if passed else "REJECTED_CONFIG_E_PRODUCTION_PREFLIGHT", "contract": {"name": CONFIG_E_CONTRACT, "storage_scoring_dimension": CONFIG_E_STORAGE_DIMENSION, "selection_dimension": CONFIG_E_SELECTION_DIMENSION, "anchors_per_chunk": CONFIG_E_ANCHORS, "selector": "incremental_farthest_first_diversity_medoids", "dual_store": True, "exact_parent_scoring": "streaming_per_query_token_max"}, "real_chunk_parity": {"required": 20, "executed": min(20, len(samples)), "errors": parity_errors}, "benchmark": {"chunks": len(samples), "elapsed_seconds": elapsed, "throughput_chunks_per_second": len(samples) / elapsed, "estimated_full_encode_seconds": CHUNK_COUNT / (len(samples) / elapsed), "retained_tokens": {"min": min(retained_counts, default=0), "mean": float(np.mean(retained_counts)) if retained_counts else 0.0, "max": max(retained_counts, default=0)}}, "disk": {**disk, "required_bytes": int(disk_estimate["total_upper_bound"] * 1.25), "headroom_fraction": .25, "pass": disk_pass}, "disk_estimate": disk_estimate, "hash_resume_probe": {"receipt": receipt, "dual_store": bool(receipt.get("dual_store")), "verified": bool(verified), "resume_verified": bool(resume_verified)}, "claim_boundary": "E3 production integration feasibility only; no corpus index or retrieval result"}
    result = write_report(RESULTS_ROOT / "CONFIG_E_PRODUCTION_PREFLIGHT.json", report, stage="config-e-production-preflight", success=report["status"] == "PASS_CONFIG_E_PRODUCTION_PREFLIGHT"); _record_terminal_stage_status("config-e-production-preflight", result["status"]); return result


def nested_oof(*, resume: bool = True, authorize: bool = False) -> dict[str, Any]:
    require_authorization("nested-oof", authorize, "EXP109C_ALLOW_FULL_OOF")
    fold0 = read_json(RESULTS_ROOT / "FOLD0_REPORT.json") if (RESULTS_ROOT / "FOLD0_REPORT.json").exists() else {}
    if fold0.get("status") not in {"PASS_FOLD0_STRONG_USER_DECISION", "PASS_TARGET_097", "PASS_TARGET_098"}:
        raise GateRejected("REJECTED_FOLD0_AUTHORIZATION_BARRIER", "full OOF requires an explicitly authorized strong Fold-0 report")
    final = read_json(RESULTS_ROOT / "FINAL_INNER_GATE.json") if (RESULTS_ROOT / "FINAL_INNER_GATE.json").exists() else {}
    if final.get("status") != "PASS_FINAL_INNER_GATE": raise GateRejected("REJECTED_PREREQUISITE_GATE", "final inner gate must pass before full OOF")
    predictions_path = RESULTS_ROOT / "OOF_PREDICTIONS.json"
    if not predictions_path.exists():
        raise GateRejected("REJECTED_FULL_OOF_PREREQUISITE", "full OOF prediction artifact is missing", report=predictions_path)
    artifact = read_json(predictions_path); answers, _ = canonical_labels(); folds, _ = load_folds()
    predictions = artifact.get("predictions") if isinstance(artifact, Mapping) else None
    if not isinstance(predictions, Mapping): raise GateRejected("REJECTED_PREDICTION_CONTRACT", "OOF_PREDICTIONS.json must contain predictions")
    qids = [str(qid) for qids in folds.values() for qid in qids]
    evaluated = evaluated_prediction_report(stage="nested-oof", predictions=predictions, answers=answers, qids=qids)
    metrics = evaluated["metrics"]; per_fold = {name: _prediction_metrics(evaluated["predictions"], answers, qids) for name, qids in folds.items()}
    report = {"schema_version": SCHEMA, "stage": "nested-oof", "status": "PASS_FULL_OOF_EVALUATED", "metrics": metrics, "per_fold": per_fold, "evaluation": evaluated, "claim_boundary": "evaluated full OOF only; promotion still requires comparative anchor analysis"}
    return write_report(RESULTS_ROOT / "FULL_OOF_REPORT.json", report, stage="nested-oof")


def locked_fold0(*, outer: str = FOLD0, authorize: bool = False) -> dict[str, Any]:
    require_authorization("locked-fold0", authorize, "EXP109C_ALLOW_FOLD0")
    final = read_json(RESULTS_ROOT / "FINAL_INNER_GATE.json") if (RESULTS_ROOT / "FINAL_INNER_GATE.json").exists() else {}
    if final.get("status") != "PASS_FINAL_INNER_GATE": raise GateRejected("REJECTED_FOLD0_AUTHORIZATION_BARRIER", "Fold-0 requires PASS_FINAL_INNER_GATE")
    predictions_path = RESULTS_ROOT / "FOLD0_PREDICTIONS.json"
    if not predictions_path.exists():
        raise GateRejected("REJECTED_FOLD0_PREREQUISITE", "locked Fold-0 prediction artifact is missing", report=predictions_path)
    artifact = read_json(predictions_path); answers, _ = canonical_labels(); folds, _ = load_folds()
    predictions = artifact.get("predictions") if isinstance(artifact, Mapping) else None
    if not isinstance(predictions, Mapping): raise GateRejected("REJECTED_PREDICTION_CONTRACT", "FOLD0_PREDICTIONS.json must contain predictions")
    qids = [str(qid) for qid in folds[outer]]; evaluated = evaluated_prediction_report(stage="locked-fold0", predictions=predictions, answers=answers, qids=qids)
    recall5 = evaluated["metrics"]["recall@5"]
    status_value = "REJECTED_FOLD0_REPRESENTATION" if recall5 < .950 else "WEAK_FOLD0_NO_FULL_OOF" if recall5 < .960 else "PASS_FOLD0_STRONG_USER_DECISION" if recall5 < .970 else "PASS_TARGET_097" if recall5 < .980 else "PASS_TARGET_098"
    report = {"schema_version": SCHEMA, "stage": "locked-fold0", "outer": outer, "status": status_value, "metrics": evaluated["metrics"], "evaluation": evaluated, "prediction_artifact_sha256": sha256_file(predictions_path), "claim_boundary": "one locked Fold-0 evaluation; not full OOF/public Recall"}
    return write_report(RESULTS_ROOT / "FOLD0_REPORT.json", report, stage="locked-fold0")


def status() -> dict[str, Any]:
    result: dict[str, Any] = {"namespace": NAMESPACE, "status": "PASS_STATUS_READ_ONLY", "code_sha256": code_fingerprint(), "artifacts": {}}
    for name in ("READING_AUDIT.json", "REPRODUCTION_REPORT.json", "PREFLIGHT.json", "CANDIDATE_CEILING_REPORT.json", "FIDELITY_PILOT_REPORT.json", "CONFIG_E_EXISTING_FIDELITY_REPORT.json", "CONFIG_E_CONFIRMATION_REPORT.json", "CONFIG_E_PRODUCTION_PREFLIGHT.json", "INDEX_MANIFEST.json", "INNER_LATE_SCORE_REPORT.json", "FROZEN_INNER_SCREEN.json", "FULL_LATENT_SELECTION_METADATA.json", "FROZEN_WINNER_LOCK.json", "FOLD0_SCORE_MANIFEST.json", "FINAL_MODEL_MANIFEST.json", "FOLD0_PREDICTION_LOCK.json", "FROZEN_WINNER_FOLD0_REPORT.json", "METRIC_ADAPTER_REPORT.json", "FINAL_INNER_GATE.json", "FOLD0_REPORT.json", "FULL_OOF_REPORT.json", "RUN_STATUS.json"):
        path = RESULTS_ROOT / name; result["artifacts"][name] = {"exists": path.exists(), "status": read_json(path).get("status") if path.exists() and path.suffix == ".json" else None}
    result["active_processes"] = active_experiment_processes(); return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("stage", choices=("audit", "reproduce-exp013", "preflight", "candidate-ceiling", "fidelity-pilot", "config-e-existing", "config-e-confirmation", "config-e2-cutoff-diagnostic", "config-e4-exact-refinement", "config-e-production-preflight", "encode-corpus", "encode-queries", "benchmark-batched-scorer", "score-inner", "d100-futility-screen", "frozen-inner-screen", "frozen-winner-lock", "score-fold0-frozen", "train-final-frozen", "evaluate-fold0-frozen", "train-metric-adapter", "final-inner-gate", "locked-fold0", "nested-oof", "status"))
    value.add_argument("--outer", default=FOLD0); value.add_argument("--resume", action="store_true"); value.add_argument("--authorize", action="store_true"); value.add_argument("--device", default="cuda"); value.add_argument("--batch-size", type=int, default=8); value.add_argument("--query-count", type=int, choices=(1024, 2048)); return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.stage == "audit": result = audit_inputs()
        elif args.stage == "reproduce-exp013": result = reproduction_report()
        elif args.stage == "preflight": result = resource_preflight(authorize=args.authorize, device=args.device)
        elif args.stage == "status": result = status()
        elif args.stage == "candidate-ceiling":
            audit = read_json(RESULTS_ROOT / "READING_AUDIT.json") if (RESULTS_ROOT / "READING_AUDIT.json").exists() else {}
            if audit.get("status") != "PASS": raise GateRejected("REJECTED_INPUT_AUDIT", "PASS READING_AUDIT.json is required before candidate ceiling")
            answers, _stats = canonical_labels(); folds, _ = load_folds(); source_rows = load_exp109b_anchor_sources(outer=args.outer)
            fingerprints = expected_candidate_input_fingerprints(outer=args.outer)
            result = candidate_ceiling_report(source_rows, answers, folds, outer=args.outer, report_path=RESULTS_ROOT / "CANDIDATE_CEILING_REPORT.json", input_fingerprints=fingerprints)
        elif args.stage == "fidelity-pilot":
            result = run_fidelity_pilot(authorize=args.authorize)
        elif args.stage == "config-e-existing": result = run_config_e_existing_fixture(authorize=args.authorize)
        elif args.stage == "config-e-confirmation": result = run_config_e_confirmation_fixture(authorize=args.authorize)
        elif args.stage == "config-e2-cutoff-diagnostic": result = write_e2_cutoff_diagnostic()
        elif args.stage == "config-e4-exact-refinement": result = run_config_e4_exact_refinement(authorize=args.authorize)
        elif args.stage == "config-e-production-preflight": result = run_config_e_production_preflight(authorize=args.authorize, device=args.device)
        elif args.stage == "encode-corpus": result = encode_corpus(resume=args.resume, authorize=args.authorize, device=args.device, batch_size=args.batch_size)
        elif args.stage == "encode-queries": result = encode_queries(resume=args.resume, authorize=args.authorize, device=args.device, batch_size=args.batch_size)
        elif args.stage == "benchmark-batched-scorer": result = run_batched_scorer_benchmark(outer=args.outer, authorize=args.authorize, device=args.device)
        elif args.stage == "score-inner": result = score_inner(outer=args.outer, resume=args.resume, authorize=args.authorize, device=args.device, target_query_count=args.query_count)
        elif args.stage == "d100-futility-screen":
            if args.query_count is None: raise ValueError("d100-futility-screen requires --query-count")
            result = run_d100_futility_screen(outer=args.outer, target_query_count=args.query_count)
        elif args.stage == "frozen-inner-screen": result = run_frozen_inner_screen(outer=args.outer)
        elif args.stage == "frozen-winner-lock": result = frozen_winner_lock(outer=args.outer)
        elif args.stage == "score-fold0-frozen": result = score_fold0_frozen(outer=args.outer, resume=args.resume, authorize=args.authorize, device=args.device)
        elif args.stage == "train-final-frozen": result = train_final_frozen(outer=args.outer, authorize=args.authorize)
        elif args.stage == "evaluate-fold0-frozen": result = evaluate_fold0_frozen(outer=args.outer, authorize=args.authorize)
        elif args.stage == "train-metric-adapter": result = train_metric_adapter(outer=args.outer, resume=args.resume, authorize=args.authorize)
        elif args.stage == "final-inner-gate":
            result = write_final_inner_gate(outer=args.outer)
        elif args.stage == "locked-fold0": result = locked_fold0(outer=args.outer, authorize=args.authorize)
        elif args.stage == "nested-oof": result = nested_oof(resume=args.resume, authorize=args.authorize)
        else: raise AssertionError(args.stage)
        print(json.dumps({"stage": args.stage, "status": result.get("status"), "report": str(RESULTS_ROOT.resolve())}, ensure_ascii=False, sort_keys=True))
        status_value = str(result.get("status", ""))
        return 2 if status_value.startswith(("REJECTED_", "WEAK_")) else 0
    except GateRejected as exc:
        print(json.dumps({"stage": args.stage, "status": exc.status, "error": str(exc), "report": str(exc.report) if exc.report else None}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
