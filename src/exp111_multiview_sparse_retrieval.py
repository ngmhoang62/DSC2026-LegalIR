"""EXP-111: resumable, fold-zero-blind multi-view sparse retrieval.

All pre-lock commands deserialize only labels for Folds 1--4.  Index building is
label-free and streams source JSONL into SQLite; it never materializes the
corpus in Python memory.  Fold 0 is a separately authorized terminal stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "exp111_multiview_sparse"
RESULTS = ROOT / "results" / "exp111_multiview_sparse"
V3 = ROOT / "cache" / "structural_v3_e5_final_v1"
PRE = ROOT / "cache" / "final_preprocessed_v2"
TRAIN = ROOT / "public_test_dataset" / "train.json"
FOLDS = ROOT / "cache" / "cv_folds.json"
EXP021 = ROOT / "results" / "exp021_sparse" / "depth_rrf_tuning"
EXP021_DB = ROOT / "cache" / "exp021_sparse" / "passage_hierarchy" / "fts5" / "bm25_v3.sqlite"
SCHEMA = "legalir.exp111_multiview_sparse.v1"
LABEL_POLICY = "canonical_duplicate_alias_drop_empty_passage_v1"
INNER_FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")
CURVE = (1, 5, 10, 16, 20, 30, 50, 80, 100, 200)
VIEWS = ("v1_surface_structural", "v2_windows", "v3_parent", "v5_citation")
SOURCE_VIEWS = ("v0_control", "v1_surface_structural", "v2_w384", "v2_w512", "v4_bigram", "v4_trigram", "v3_parent", "v5_citation")
WORD = re.compile(r"[\w]+", re.UNICODE)
CITATION = re.compile(r"(?<!\w)(\d{1,3})\s*/\s*((?:19|20)\d{2})\s*/\s*([A-Za-zĐđ-]{2,20}(?:\s*-\s*[A-Za-zĐđ]{1,20}){0,4})(?!\w)", re.UNICODE)


class GateRejected(RuntimeError):
    """Expected gate stop with process exit 2."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL {path}:{number}") from exc


def rss_bytes() -> int | None:
    """Current process RSS on POSIX and Windows, without requiring psutil."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception: pass
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        if os.name != "nt": return None
    try:
        import ctypes
        class COUNTERS(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = COUNTERS(); counters.cb = ctypes.sizeof(COUNTERS)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb): return int(counters.WorkingSetSize)
    except Exception: pass
    return None


def system_available_ram_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception: pass
    if os.name != "nt": return None
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        state = MEMORYSTATUSEX(); state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)): return int(state.ullAvailPhys)
    except Exception: pass
    return None


@dataclass
class Run:
    stage: str
    total: int = 0
    run_id: str = ""
    started: float = 0.0
    log: Path | None = None
    interrupted: bool = False
    peak_rss: int = 0

    def __enter__(self) -> "Run":
        self.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
        self.started = time.monotonic()
        self.log = RESULTS / "logs" / self.run_id / f"{self.stage}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.status("RUNNING", 0, note="started")
        self.line(f"START stage={self.stage} total={self.total} log={self.log}")
        return self

    def line(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
        print(line, flush=True)
        assert self.log is not None
        with self.log.open("a", encoding="utf-8", newline="\n", buffering=1) as handle:
            handle.write(line + "\n")

    def status(self, state: str, completed: int, **extra: Any) -> None:
        elapsed = max(0.001, time.monotonic() - self.started) if self.started else 0.0
        eta = ((self.total - completed) * elapsed / completed) if completed and self.total else None
        current_rss = rss_bytes(); self.peak_rss = max(self.peak_rss, current_rss or 0)
        atomic_json(RESULTS / "RUN_STATUS.json", {"schema_version": SCHEMA, "run_id": self.run_id,
            "state": state, "stage": self.stage, "completed": completed, "total": self.total,
            "eta_seconds": eta, "last_heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "log_path": str(self.log) if self.log else None, "rss_bytes": current_rss, "peak_rss_bytes": self.peak_rss,
            "available_system_ram_bytes": system_available_ram_bytes(), **extra})

    def heartbeat(self, completed: int, **extra: Any) -> None:
        self.status("RUNNING", completed, **extra)
        elapsed = max(0.001, time.monotonic() - self.started)
        rate = completed / elapsed if completed else 0.0
        eta = (self.total - completed) / rate if rate else None
        self.line(f"HEARTBEAT stage={self.stage} completed={completed}/{self.total} qps={rate:.3f} estimated_eta_seconds={eta} rss_bytes={rss_bytes()}")

    def __exit__(self, kind: Any, error: Any, _trace: Any) -> bool:
        if kind is KeyboardInterrupt:
            self.status("INTERRUPTED", 0, error="KeyboardInterrupt")
            self.line("INTERRUPTED checkpoint preserved")
            return False
        if kind:
            self.status("FAILED", 0, error=f"{kind.__name__}: {error}")
            self.line(f"FAILED {kind.__name__}: {error}")
            return False
        self.status("PASS", self.total)
        self.line(f"COMPLETE elapsed_seconds={time.monotonic() - self.started:.2f}")
        return False


def load_folds() -> tuple[dict[str, list[str]], dict[str, str]]:
    raw = json.loads(FOLDS.read_text(encoding="utf-8"))
    folds = {str(name): [str(qid) for qid in values] for name, values in raw.items()}
    if set(folds) != {f"fold_{number}" for number in range(5)}:
        raise GateRejected(f"unexpected folds: {sorted(folds)}")
    by_qid: dict[str, str] = {}
    for name, qids in folds.items():
        for qid in qids:
            if qid in by_qid:
                raise GateRejected(f"duplicate qid in folds: {qid}")
            by_qid[qid] = name
    return folds, by_qid


def iter_train_rows(select_qids: set[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Decode selected qids only; Fold-0 answer objects never enter Python objects.

    The train file is a top-level JSON object.  This tiny streaming parser scans
    a value as JSON text, then calls json.loads only after the qid has been
    admitted.  It deliberately avoids a normal json.load(TRAIN) before lock.
    """
    decoder = json.JSONDecoder()
    text = TRAIN.read_text(encoding="utf-8")
    pos = 0
    while pos < len(text) and text[pos].isspace(): pos += 1
    if pos >= len(text) or text[pos] != "{": raise GateRejected("train root is not object")
    pos += 1
    while True:
        while pos < len(text) and text[pos].isspace(): pos += 1
        if pos < len(text) and text[pos] == "}": return
        raw_qid, pos = decoder.raw_decode(text, pos)
        qid = str(raw_qid)
        while pos < len(text) and text[pos].isspace(): pos += 1
        if pos >= len(text) or text[pos] != ":": raise GateRejected("malformed train delimiter")
        pos += 1
        start = pos
        while pos < len(text) and text[pos].isspace(): pos += 1
        if pos >= len(text) or text[pos] != "{": raise GateRejected(f"train row {qid} is not object")
        depth, quoted, escaped = 0, False, False
        while pos < len(text):
            char = text[pos]
            if quoted:
                if escaped: escaped = False
                elif char == "\\": escaped = True
                elif char == '"': quoted = False
            else:
                if char == '"': quoted = True
                elif char == "{": depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        pos += 1
                        break
            pos += 1
        if qid in select_qids:
            yield qid, json.loads(text[start:pos])
        while pos < len(text) and text[pos].isspace(): pos += 1
        if pos < len(text) and text[pos] == ",": pos += 1; continue
        if pos < len(text) and text[pos] == "}": return
        raise GateRejected("malformed train separator")


def canonical_inner() -> tuple[dict[str, str], dict[str, set[str]], dict[str, list[str]], dict[str, str], dict[str, Any]]:
    folds, fold_for = load_folds()
    inner = {qid for fold in INNER_FOLDS for qid in folds[fold]}
    rows = dict(iter_train_rows(inner))
    if set(rows) != inner: raise GateRejected("inner train qid coverage mismatch")
    exclusions = {str(item["doc_id"]): item for item in json.loads((PRE / "exclusions.json").read_text(encoding="utf-8"))}
    impact = {str(item["query_id"]): item for item in read_jsonl(PRE / "train_label_impact.jsonl")}
    questions, answers = {}, {}
    duplicate_count = empty_count = 0
    for qid, row in rows.items():
        questions[qid] = str(row["question"])
        gold: set[str] = set()
        dropped: set[str] = set()
        for item in row.get("answer", []):
            doc_id = str(item); exclusion = exclusions.get(doc_id)
            if exclusion is None: gold.add(doc_id); continue
            dropped.add(doc_id); reasons = set(exclusion["reasons"])
            if reasons == {"exact_duplicate_raw_passage"} and exclusion.get("duplicate_retained_id"):
                gold.add(str(exclusion["duplicate_retained_id"])); duplicate_count += 1
            elif reasons == {"empty_passage"}: empty_count += 1
            else: raise GateRejected(f"unsupported canonicalization {qid}/{doc_id}")
        declared = set(impact.get(qid, {}).get("intentionally_excluded_gold_ids", []))
        if dropped != declared: raise GateRejected(f"impact mismatch {qid}")
        answers[qid] = gold
    stats = {"policy": LABEL_POLICY, "scope": "folds_1_to_4_only", "queries": len(rows),
        "evaluable_queries": sum(bool(value) for value in answers.values()),
        "non_evaluable_queries": sum(not value for value in answers.values()),
        "canonicalized_duplicate_occurrences": duplicate_count, "dropped_empty_occurrences": empty_count,
        "label_fingerprint": fingerprint({qid: sorted(gold) for qid, gold in sorted(answers.items())}),
        "fold0_labels_deserialized": False}
    return questions, answers, folds, fold_for, stats


def surface(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return [match.group(0) for match in WORD.finditer(normalized)]


def fts_or(tokens: Sequence[str]) -> str:
    clean = [token for token in tokens if WORD.fullmatch(token)]
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in clean) if clean else '"__exp111_no_token__"'


def fts_phrase(tokens: Sequence[str]) -> str:
    return '"' + " ".join(token.replace('"', '""') for token in tokens if WORD.fullmatch(token)) + '"'


class ExactFtsSession:
    """The actual FTS5 tokenizer, reusable without retaining source text."""
    def __init__(self, max_cache: int = 4096) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE VIRTUAL TABLE token_input USING fts5(text, tokenize='unicode61 tokenchars _')")
        self.conn.execute("CREATE VIRTUAL TABLE token_vocab USING fts5vocab(token_input, 'row')")
        self.cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self.max_cache = max_cache

    def terms(self, text: str) -> list[str]:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return list(cached)
        self.conn.execute("DELETE FROM token_input")
        self.conn.execute("INSERT INTO token_input(text) VALUES(?)", (text,))
        terms = tuple(str(row[0]) for row in self.conn.execute("SELECT term FROM token_vocab ORDER BY term"))
        self.cache[key] = terms
        if len(self.cache) > self.max_cache: self.cache.popitem(last=False)
        return list(terms)

    def sequence(self, text: str) -> list[str]:
        """Normalize each lexical unit separately so phrase order is preserved."""
        output: list[str] = []
        for token in surface(text):
            normalized = self.terms(token)
            if len(normalized) == 1: output.append(normalized[0])
        return output

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ExactFtsSession": return self
    def __exit__(self, *_: Any) -> None: self.close()


def exact_fts_terms(text: str) -> list[str]:
    """Ask SQLite FTS5 itself for unicode61-normalized terms, never approximate it in Python."""
    with ExactFtsSession(max_cache=0) as tokenizer:
        return tokenizer.terms(text)


def exact_fts_token_sequence(text: str) -> list[str]:
    with ExactFtsSession(max_cache=0) as tokenizer:
        return tokenizer.sequence(text)


def phrase_expression_terms(conn: sqlite3.Connection, table: str, normalized: Sequence[str], size: int, *, total_windows: int | None = None, df_cache: MutableMapping[str, int] | None = None) -> str:
    """Build an exact FTS phrase from already-FTS-normalized terms.

    The optional connection/cache supports audit streaming without opening a
    SQLite database twice per query.  Its result is intentionally identical to
    ``phrase_expression``; only the lifetime of read-only resources changes.
    """
    if len(normalized) < size: return '"__exp111_no_token__"'
    denominator = total_windows if total_windows is not None else int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    informative = set()
    for token in sorted(set(normalized)):
        doc_frequency = df_cache.get(token) if df_cache is not None else None
        if doc_frequency is None:
            row = conn.execute(f"SELECT doc FROM {table}_vocab WHERE term=?", (token,)).fetchone()
            doc_frequency = int(row[0]) if row else 0
            if df_cache is not None: df_cache[token] = doc_frequency
        if doc_frequency < denominator * .20: informative.add(token)
    phrases = [fts_phrase(normalized[index:index + size]) for index in range(len(normalized) - size + 1) if set(normalized[index:index + size]) & informative]
    return " OR ".join(phrases) if phrases else '"__exp111_no_token__"'


def phrase_expression(database: Path, table: str, tokens: Sequence[str], size: int) -> str:
    """Quoted phrases with exact FTS terms and a window-row DF denominator."""
    if len(tokens) < size: return '"__exp111_no_token__"'
    conn = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        normalized = exact_fts_token_sequence(" ".join(tokens))
        return phrase_expression_terms(conn, table, normalized, size)
    finally: conn.close()


def citation_tokens(text: str) -> list[str]:
    result = []
    for number, year, issuer in CITATION.findall(unicodedata.normalize("NFC", text).casefold()):
        tail = "_".join(surface(issuer))
        if tail: result.append(f"cite_{number}_{year}_{tail}")
    return sorted(set(result))


def robust_z(values: Sequence[float]) -> list[float]:
    if not values: return []
    med = float(statistics.median(values)); mad = float(statistics.median(abs(value - med) for value in values))
    scale = 1.4826 * mad
    if scale <= 1e-12: return [0.0] * len(values)
    return [(value - med) / scale for value in values]


def metric(predictions: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], ks: Sequence[int] = CURVE) -> dict[str, float]:
    qids = [qid for qid, gold in answers.items() if gold]
    output: dict[str, float] = {"evaluable_queries": float(len(qids))}
    for k in ks:
        output[f"recall@{k}"] = float(np.mean([len(set(predictions.get(qid, [])[:k]) & answers[qid]) / len(answers[qid]) for qid in qids])) if qids else 0.0
    output["precision@5"] = float(np.mean([len(set(predictions.get(qid, [])[:5]) & answers[qid]) / 5.0 for qid in qids])) if qids else 0.0
    output["mrr@5"] = float(np.mean([next((1.0 / rank for rank, doc in enumerate(predictions.get(qid, [])[:5], 1) if doc in answers[qid]), 0.0) for qid in qids])) if qids else 0.0
    multi = [qid for qid in qids if len(answers[qid]) > 1]
    output["multi_gold_recall@5"] = float(np.mean([len(set(predictions.get(qid, [])[:5]) & answers[qid]) / len(answers[qid]) for qid in multi])) if multi else 0.0
    return output


def write_success(directory: Path, stage: str, payload: Mapping[str, Any]) -> None:
    atomic_json(directory / "_SUCCESS.json", {"schema_version": SCHEMA, "stage": stage, "fingerprint": fingerprint(payload)})


def input_audit() -> dict[str, Any]:
    with Run("audit") as run:
        manifest = json.loads((V3 / "manifest.json").read_text(encoding="utf-8"))
        success = json.loads((V3 / "_SUCCESS.json").read_text(encoding="utf-8"))
        questions, answers, folds, fold_for, labels = canonical_inner()
        disk = shutil.disk_usage(CACHE.parent)
        exp021_ok = EXP021_DB.exists() and (EXP021 / "oof_rankings.jsonl").exists() and (EXP021 / "tuning_report.json").exists()
        result = {"schema_version": SCHEMA, "status": "PASS_INPUT_AUDIT" if exp021_ok and disk.free >= 25 * 1024**3 else "REJECTED_READING_OR_INPUT_GATE",
            "v3_fingerprint": manifest.get("content_fingerprint"), "v3_success_fingerprint": success.get("content_fingerprint"),
            "counts": {"parents": manifest["counts"]["documents"], "chunks": manifest["counts"]["chunks"], "inner_queries": len(questions), **labels},
            "folds": {name: len(qids) for name, qids in sorted(folds.items())}, "fold_partition_qids": len(fold_for),
            "disk_free_bytes": disk.free, "preflight_minimum_bytes": 25 * 1024**3,
            "exp021_control_present": exp021_ok, "fold0_labels_deserialized": False,
            "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (PRE / "manifest.json", PRE / "exclusions.json", PRE / "train_label_impact.jsonl", V3 / "manifest.json", V3 / "_SUCCESS.json", FOLDS)}}
        atomic_json(RESULTS / "INPUT_AUDIT.json", result)
        if result["status"] != "PASS_INPUT_AUDIT": raise GateRejected(result["status"])
        write_success(RESULTS, "audit", result); return result


def _v0_search(query: str, limit: int = 2048) -> list[dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "src"))
    from exp012b_bm25 import BM25Searcher, default_segmenter, safe_fts_query
    with BM25Searcher(EXP021_DB, profile="legal_structure") as searcher:
        return searcher.search_expression(safe_fts_query(default_segmenter(query)), limit=limit)


def exp021_config(fold: str) -> dict[str, int]:
    report = json.loads((EXP021 / "tuning_report.json").read_text(encoding="utf-8"))
    chosen = report["selected_by_fold"].get(fold)
    if not chosen: raise GateRejected(f"missing EXP-021 selected config for {fold}")
    return {key: int(chosen[key]) for key in ("depth", "parent_rrf_k", "fusion_rrf_k", "head_cutoff")}


def aggregate_v0_exp021(hits: Sequence[Mapping[str, Any]], config: Mapping[str, int], top: int = 256) -> list[dict[str, Any]]:
    """Byte-for-byte ranking logic of EXP-021 `_rankings` plus `_cascade`."""
    by_doc: dict[str, list[Mapping[str, Any]]] = defaultdict(list); seen: dict[str, set[str]] = defaultdict(set)
    for hit in hits:
        if int(hit["rank"]) > int(config["depth"]): continue
        doc, node = str(hit["doc_id"]), str(hit["parent_node_id"])
        if node not in seen[doc] and len(by_doc[doc]) < 3:
            by_doc[doc].append(hit); seen[doc].add(node)
    for values in by_doc.values(): values.sort(key=lambda row: (int(row["rank"]), str(row["chunk_id"])))
    first = [doc for doc, _rows in sorted(by_doc.items(), key=lambda item: (int(item[1][0]["rank"]), item[0]))[:256]]
    rrf_scores = {doc: sum(1.0 / (int(config["parent_rrf_k"]) + int(row["rank"])) for row in rows) for doc, rows in by_doc.items()}
    rrf = sorted(rrf_scores, key=lambda doc: (-rrf_scores[doc], int(by_doc[doc][0]["rank"]), doc))[:256]
    fusion: dict[str, float] = defaultdict(float); best_rank: dict[str, int] = {}
    for ranking in (first, rrf):
        for rank, doc in enumerate(ranking, 1):
            fusion[doc] += 1.0 / (int(config["fusion_rrf_k"]) + rank)
            best_rank[doc] = min(best_rank.get(doc, rank), rank)
    fused = sorted(fusion, key=lambda doc: (-fusion[doc], best_rank[doc], doc))[:256]
    head = fused[:int(config["head_cutoff"])]
    order = (head + [doc for doc in rrf if doc not in set(head)])[:256]
    records = []
    for rank, doc in enumerate(order[:top], 1):
        rows = by_doc[doc]; best = rows[0]
        records.append({"doc_id": doc, "rank": rank, "raw_score": float(fusion.get(doc, rrf_scores[doc])), "best_unit_id": str(best["chunk_id"]), "best_score": -float(best["score"]), "second_score": -float(rows[1]["score"]) if len(rows) > 1 else None, "score_gap": None, "matching_units": len(rows), "rank_key": int(best["rank"]), "v0_config": dict(config)})
    for row, z in zip(records, robust_z([float(row["raw_score"]) for row in records])): row["robust_z"] = z
    return records


def _db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def build_surface_structural(resume: bool) -> dict[str, Any]:
    """V1: streamed structural raw chunks, not V0's Underthesea cache."""
    directory = CACHE / "v1_surface_structural"; database = directory / "index.sqlite"; directory.mkdir(parents=True, exist_ok=True)
    config = {"view": "v1", "v3": sha256(V3 / "manifest.json"), "tokenizer": "surface_unicode61_v1"}
    marker = directory / "manifest.json"
    if resume and marker.exists() and database.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("config_hash") == fingerprint(config) and saved.get("database_sha256") == sha256(database): return saved
    state_path = directory / "BUILD_STATE.json"
    if database.exists() and resume:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        if state.get("config_hash") != fingerprint(config):
            raise GateRejected("REJECTED_RESUME_FINGERPRINT_GATE: V1 partial index config differs")
    else:
        database.unlink(missing_ok=True); state_path.unlink(missing_ok=True)
    total = int(json.loads((V3 / "manifest.json").read_text(encoding="utf-8"))["counts"]["chunks"])
    with Run("build-index-v1", total) as run:
        conn = _db(database)
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS units USING fts5(text, unit_id UNINDEXED, doc_id UNINDEXED, node_id UNINDEXED, tokenize='unicode61 tokenchars _')")
            existing = int(conn.execute("SELECT count(*) FROM units").fetchone()[0])
            for count, row in enumerate(read_jsonl(V3 / "chunks.jsonl"), 1):
                if count <= existing: continue
                conn.execute("INSERT INTO units(text,unit_id,doc_id,node_id) VALUES(?,?,?,?)", (" ".join(surface(str(row["raw_text"]))), row["chunk_id"], row["doc_id"], row["parent_node_id"]))
                if count % 8192 == 0:
                    conn.commit(); atomic_json(state_path, {"schema_version": SCHEMA, "config_hash": fingerprint(config), "committed_records": count})
                if count % 16384 == 0: run.heartbeat(count, database_bytes=database.stat().st_size)
            conn.commit()
        finally: conn.close()
    result = {"schema_version": SCHEMA, "view": "v1_surface_structural", "config": config, "config_hash": fingerprint(config), "records": total, "database_sha256": sha256(database), "database_bytes": database.stat().st_size}
    atomic_json(marker, result); write_success(directory, "build-index-v1", result); return result


def word_spans(text: str) -> list[tuple[int, int]]:
    # Offsets are always against the exact source string; normalization is only
    # applied to indexed token text after this source slice is selected.
    return [(match.start(), match.end()) for match in WORD.finditer(text)]


def windows(doc_id: str, text: str, width: int, overlap: int) -> Iterator[dict[str, Any]]:
    if not 0 <= overlap < width: raise ValueError("invalid overlap")
    spans = word_spans(text); step = width - overlap
    if not spans:
        return
    for start in range(0, len(spans), step):
        end = min(len(spans), start + width); first, last = spans[start], spans[end - 1]
        yield {"window_id": f"{doc_id}:{start}:{end}", "doc_id": doc_id, "word_start": start, "word_end": end,
               "char_start": first[0], "char_end": last[1], "raw_text_hash": hashlib.sha256(text[first[0]:last[1]].encode("utf-8")).hexdigest(),
               "text": " ".join(surface(text[first[0]:last[1]]))}
        if end == len(spans): break


def iter_parent_sources() -> Iterator[tuple[str, str, str]]:
    """Yield exact document nodes once, with raw source text, streaming nodes."""
    yielded = set()
    for row in read_jsonl(V3 / "nodes.jsonl"):
        if row.get("kind") == "document":
            doc_id = str(row["doc_id"])
            if doc_id not in yielded:
                yielded.add(doc_id); yield doc_id, str(row["raw_text"]), str(row.get("node_id", ""))


def build_windows_and_parent(resume: bool) -> dict[str, Any]:
    directory = CACHE / "v2_v3"; dbpath = directory / "index.sqlite"; directory.mkdir(parents=True, exist_ok=True)
    config = {"views": ["v2_windows", "v3_parent", "v5_citation"], "v3": sha256(V3 / "manifest.json"), "windows": [[384, 96], [512, 128]], "tokenizer": "surface_unicode61_v1"}
    marker = directory / "manifest.json"
    if resume and marker.exists() and dbpath.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("config_hash") == fingerprint(config) and saved.get("database_sha256") == sha256(dbpath): return saved
    state_path = directory / "BUILD_STATE.json"
    if dbpath.exists() and resume:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        if state.get("config_hash") != fingerprint(config):
            raise GateRejected("REJECTED_RESUME_FINGERPRINT_GATE: V2/V3/V5 partial index config differs")
    else:
        dbpath.unlink(missing_ok=True); state_path.unlink(missing_ok=True)
    total = int(json.loads((V3 / "manifest.json").read_text(encoding="utf-8"))["counts"]["documents"])
    with Run("build-index-v2-v3-v5", total) as run:
        conn = _db(dbpath)
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS local384 USING fts5(text,unit_id UNINDEXED,doc_id UNINDEXED,word_start UNINDEXED,word_end UNINDEXED,char_start UNINDEXED,char_end UNINDEXED,raw_text_hash UNINDEXED, tokenize='unicode61 tokenchars _')")
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS local512 USING fts5(text,unit_id UNINDEXED,doc_id UNINDEXED,word_start UNINDEXED,word_end UNINDEXED,char_start UNINDEXED,char_end UNINDEXED,raw_text_hash UNINDEXED, tokenize='unicode61 tokenchars _')")
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS parents USING fts5(text,doc_id UNINDEXED,raw_text_hash UNINDEXED,raw_length UNINDEXED, tokenize='unicode61 tokenchars _')")
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS citations USING fts5(tokens,doc_id UNINDEXED, tokenize='unicode61 tokenchars _')")
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS local384_vocab USING fts5vocab(local384, 'row')")
            existing = int(conn.execute("SELECT count(*) FROM parents").fetchone()[0])
            count = existing; windows_count = int(conn.execute("SELECT count(*) FROM local384").fetchone()[0])
            for count, (doc_id, text, _node) in enumerate(iter_parent_sources(), 1):
                if count <= existing: continue
                raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest(); normalized = " ".join(surface(text))
                conn.execute("INSERT INTO parents(text,doc_id,raw_text_hash,raw_length) VALUES(?,?,?,?)", (normalized, doc_id, raw_hash, len(text)))
                tokens = citation_tokens(text)
                if tokens: conn.execute("INSERT INTO citations(tokens,doc_id) VALUES(?,?)", (" ".join(tokens), doc_id))
                for width, overlap in ((384, 96), (512, 128)):
                    table = f"local{width}"
                    for record in windows(doc_id, text, width, overlap) or ():
                        conn.execute(f"INSERT INTO {table}(text,unit_id,doc_id,word_start,word_end,char_start,char_end,raw_text_hash) VALUES(?,?,?,?,?,?,?,?)", (record["text"], record["window_id"], doc_id, record["word_start"], record["word_end"], record["char_start"], record["char_end"], record["raw_text_hash"]))
                        windows_count += 1
                if count % 128 == 0:
                    conn.commit(); atomic_json(state_path, {"schema_version": SCHEMA, "config_hash": fingerprint(config), "committed_records": count, "committed_windows": windows_count})
                if count % 256 == 0: run.heartbeat(count, windows=windows_count, database_bytes=dbpath.stat().st_size)
            conn.commit()
        finally: conn.close()
    result = {"schema_version": SCHEMA, "views": config["views"], "config": config, "config_hash": fingerprint(config), "records": count, "windows": windows_count, "database_sha256": sha256(dbpath), "database_bytes": dbpath.stat().st_size}
    atomic_json(marker, result); write_success(directory, "build-index-v2-v3-v5", result); return result


def search_fts(database: Path, table: str, expression: str, limit: int, columns: str) -> list[dict[str, Any]]:
    if expression == '"__exp111_no_token__"': return []
    conn = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(f"SELECT {columns}, bm25({table}) AS score FROM {table} WHERE {table} MATCH ? ORDER BY score ASC, doc_id ASC, rowid ASC LIMIT ?", (expression, limit)).fetchall()
    finally: conn.close()
    return [dict(zip(columns.split(","), row[:-1])) | {"score": float(row[-1]), "unit_rank": rank} for rank, row in enumerate(rows, 1)]


def aggregate_units(rows: Sequence[Mapping[str, Any]], *, top: int, nonredundant: bool = False, second_lambda: float = 0.6) -> list[dict[str, Any]]:
    by_doc: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: by_doc[str(row["doc_id"])].append(row)
    out = []
    for doc, values in by_doc.items():
        values = sorted(values, key=lambda row: (float(row["score"]), int(row["unit_rank"])))
        best = values[0]; second = None
        if nonredundant:
            for candidate in values[1:]:
                a, b = int(best["word_start"]), int(best["word_end"]); c, d = int(candidate["word_start"]), int(candidate["word_end"])
                overlap = max(0, min(b, d) - max(a, c)) / max(1, min(b-a, d-c))
                if overlap <= 0.50 or abs((a+b) - (c+d)) / 2 >= 192:
                    second = candidate; break
        # SQLite bm25 is a negative relevance score; smaller is better.
        score = -float(best["score"]) + (second_lambda * -float(second["score"]) if second else 0.0)
        out.append({"doc_id": doc, "raw_score": score, "best_unit_id": str(best.get("unit_id", doc)), "best_score": -float(best["score"]),
                    "second_score": (-float(second["score"]) if second else None), "score_gap": ((-float(best["score"])) - (-float(second["score"])) if second else None), "matching_units": len(values), "rank_key": int(best["unit_rank"])})
    out.sort(key=lambda item: (-float(item["raw_score"]), int(item["rank_key"]), str(item["doc_id"])))
    out = out[:top]
    for rank, row in enumerate(out, 1): row["rank"] = rank
    zs = robust_z([float(row["raw_score"]) for row in out])
    for row, z in zip(out, zs): row["robust_z"] = z
    return out


def source_rows(query: str, view: str, *, limit: int = 500, v0_config: Mapping[str, int] | None = None) -> list[dict[str, Any]]:
    if view == "v0_control":
        if v0_config is None: raise GateRejected("V0 requires per-fold frozen EXP-021 config")
        return aggregate_v0_exp021(_v0_search(query, int(v0_config["depth"])), v0_config, min(limit, 256))
    if view == "v1_surface_structural":
        rows = search_fts(CACHE / "v1_surface_structural" / "index.sqlite", "units", fts_or(surface(query)), 2048, "unit_id,doc_id,node_id")
        rows = [row | {"word_start": 0, "word_end": 0} for row in rows]
        return aggregate_units(rows, top=limit)
    if view in {"v2_w384", "v2_w512"}:
        table = "local384" if view.endswith("384") else "local512"
        rows = search_fts(CACHE / "v2_v3" / "index.sqlite", table, fts_or(surface(query)), 3000, "unit_id,doc_id,word_start,word_end,char_start,char_end,raw_text_hash")
        return aggregate_units(rows, top=limit, nonredundant=True, second_lambda=0.6)
    if view in {"v4_bigram", "v4_trigram"}:
        database = CACHE / "v2_v3" / "index.sqlite"; size = 2 if view == "v4_bigram" else 3
        expression = phrase_expression(database, "local384", surface(query), size)
        rows = search_fts(database, "local384", expression, 1500, "unit_id,doc_id,word_start,word_end,char_start,char_end,raw_text_hash")
        return aggregate_units(rows, top=limit, nonredundant=True, second_lambda=0.6)
    if view == "v3_parent":
        rows = search_fts(CACHE / "v2_v3" / "index.sqlite", "parents", fts_or(surface(query)), 500, "doc_id,raw_text_hash,raw_length")
        rows = [row | {"unit_id": row["doc_id"], "word_start": 0, "word_end": 0} for row in rows]
        return aggregate_units(rows, top=limit)
    if view == "v5_citation":
        tokens = citation_tokens(query)
        if not tokens: return []
        rows = search_fts(CACHE / "v2_v3" / "index.sqlite", "citations", fts_or(tokens), 500, "doc_id")
        rows = [row | {"unit_id": row["doc_id"], "word_start": 0, "word_end": 0} for row in rows]
        return aggregate_units(rows, top=limit)
    raise ValueError(view)


def run_reproduction(resume: bool) -> dict[str, Any]:
    """Inner-only replay plus a source-owned 100-query raw-score fixture."""
    del resume
    questions, answers, folds, fold_for, stats = canonical_inner()
    output = RESULTS / "REPRODUCTION_REPORT.json"; fixture = CACHE / "v0_reproduction_fixture.jsonl"
    qids = sorted(questions, key=lambda qid: hashlib.sha256(qid.encode()).hexdigest())[:100]
    with Run("reproduce", len(qids)) as run:
        generated = []
        for position, qid in enumerate(qids, 1):
            config = exp021_config(fold_for[qid]); hits = _v0_search(questions[qid], config["depth"])
            generated.append({"qid": qid, "fold": fold_for[qid], "config": config, "parents": aggregate_v0_exp021(hits, config, 200), "raw_hits": [{"chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "rank": row["rank"], "score": row["score"]} for row in hits]})
            if position % 20 == 0: run.heartbeat(position)
        frozen = {str(row["qid"]): [str(doc) for doc in row["documents"]][:200] for row in read_jsonl(EXP021 / "oof_rankings.jsonl")}
        mismatch = next((item["qid"] for item in generated if [row["doc_id"] for row in item["parents"]] != frozen.get(item["qid"], [])[:200]), None)
        exact = mismatch is None
        if exact:
            fixture.parent.mkdir(parents=True, exist_ok=True)
            provenance = {"reference": str((EXP021 / "oof_rankings.jsonl").relative_to(ROOT)), "reference_sha256": sha256(EXP021 / "oof_rankings.jsonl"), "owning_aggregation_source": "src/exp021_sparse_depth_tune.py", "owning_source_sha256": sha256(ROOT / "src" / "exp021_sparse_depth_tune.py"), "records": generated}
            fixture.write_text(canonical(provenance) + "\n", encoding="utf-8")
        rankings = {str(row["qid"]): [str(item) for item in row["documents"]] for row in read_jsonl(EXP021 / "oof_rankings.jsonl") if str(row["qid"]) in answers}
        replay = metric(rankings, answers)
        result = {"schema_version": SCHEMA, "status": "PASS_REPRODUCTION" if exact else "REJECTED_REPRODUCTION_GATE", "scope": "folds_1_to_4_only", "fold0_labels_deserialized": False,
            "fixture_qids": len(qids), "fixture_sha256": sha256(fixture) if fixture.exists() else None, "scorer_top200_parent_exact": exact, "first_mismatch_qid": mismatch,
            "exp021_inner_replay_metrics": replay, "labels": stats,
            "deviation": "Full-five-fold replay is intentionally not run pre-lock because plan section 5 forbids loading Fold-0 labels."}
        atomic_json(output, result)
        if not exact: raise GateRejected(result["status"])
        write_success(RESULTS, "reproduce", result); return result


TRACE_DB = CACHE / "lexical_trace" / "source_offsets.sqlite"


def iter_jsonl_offsets(path: Path) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """Read JSONL records with offsets into the original UTF-8 source file."""
    with path.open("rb") as handle:
        while True:
            offset = handle.tell(); line = handle.readline()
            if not line: return
            if line.strip(): yield offset, len(line), json.loads(line.decode("utf-8"))


def build_lexical_trace(resume: bool) -> dict[str, Any]:
    """Resumable source-offset sidecar; source text is never copied or mutated."""
    directory = TRACE_DB.parent; directory.mkdir(parents=True, exist_ok=True)
    config = {"nodes_sha256": sha256(V3 / "nodes.jsonl"), "chunks_sha256": sha256(V3 / "chunks.jsonl"), "schema": "source_offsets_v1"}
    state_path = directory / "BUILD_STATE.json"; manifest_path = directory / "manifest.json"
    if resume and TRACE_DB.exists() and manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("config_hash") == fingerprint(config) and saved.get("database_sha256") == sha256(TRACE_DB): return saved
    if TRACE_DB.exists() and not resume: TRACE_DB.unlink()
    parse_modes = {str(row["doc_id"]): str(row.get("parse_mode", "unknown")) for row in read_jsonl(V3 / "documents.jsonl")}
    conn = _db(TRACE_DB)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS documents(doc_id TEXT PRIMARY KEY, offset INTEGER NOT NULL, length INTEGER NOT NULL, parse_mode TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, offset INTEGER NOT NULL, length INTEGER NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL)")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(doc_id)")
        with Run("build-lexical-trace", 8507 + 343347) as run:
            existing_docs = int(conn.execute("SELECT count(*) FROM documents").fetchone()[0])
            for seen, (offset, length, row) in enumerate(iter_jsonl_offsets(V3 / "nodes.jsonl"), 1):
                if row.get("kind") == "document": conn.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?)", (str(row["doc_id"]), offset, length, parse_modes.get(str(row["doc_id"]), "unknown")))
                if seen % 8192 == 0: conn.commit()
                if seen % 65536 == 0: run.heartbeat(min(existing_docs + seen, 8507), phase="document_offsets")
            conn.commit()
            existing_chunks = int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
            for seen, (offset, length, row) in enumerate(iter_jsonl_offsets(V3 / "chunks.jsonl"), 1):
                conn.execute("INSERT OR IGNORE INTO chunks VALUES(?,?,?,?,?,?)", (str(row["chunk_id"]), str(row["doc_id"]), offset, length, int(row["start"]), int(row["end"])))
                if seen % 8192 == 0: conn.commit()
                if seen % 16384 == 0:
                    atomic_json(state_path, {"schema_version": SCHEMA, "config_hash": fingerprint(config), "document_records": int(conn.execute("SELECT count(*) FROM documents").fetchone()[0]), "chunk_records": int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])})
                    run.heartbeat(8507 + min(seen, 343347), phase="chunk_offsets")
            conn.commit()
        documents = int(conn.execute("SELECT count(*) FROM documents").fetchone()[0]); chunks = int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
    finally: conn.close()
    if documents != 8507 or chunks != 343347: raise GateRejected("REJECTED_LEXICAL_TRACE_COUNT_GATE")
    result = {"schema_version": SCHEMA, "config": config, "config_hash": fingerprint(config), "documents": documents, "chunks": chunks, "database_sha256": sha256(TRACE_DB), "database_bytes": TRACE_DB.stat().st_size}
    atomic_json(manifest_path, result); write_success(directory, "build-lexical-trace", result); return result


def trace_row(table: str, identifier: str) -> dict[str, Any]:
    with TraceReader() as reader:
        return reader.row(table, identifier)


def trace_chunks(doc_id: str) -> list[dict[str, Any]]:
    with TraceReader() as reader:
        return reader.chunks(doc_id)


class TraceReader:
    """Read exact source records through the offset sidecar, with persistent handles."""
    def __init__(self) -> None:
        self.conn = sqlite3.connect(f"file:{TRACE_DB.as_posix()}?mode=ro", uri=True)
        self.nodes = (V3 / "nodes.jsonl").open("rb")
        self.chunks_source = (V3 / "chunks.jsonl").open("rb")

    def row(self, table: str, identifier: str) -> dict[str, Any]:
        columns = "offset,length,parse_mode" if table == "documents" else "offset,length"
        key = "doc_id" if table == "documents" else "chunk_id"
        row = self.conn.execute(f"SELECT {columns} FROM {table} WHERE {key}=?", (identifier,)).fetchone()
        if row is None: raise GateRejected(f"missing lexical trace {table}/{identifier}")
        handle = self.nodes if table == "documents" else self.chunks_source
        handle.seek(int(row[0])); payload = json.loads(handle.read(int(row[1])).decode("utf-8"))
        if table == "documents": payload["parse_mode"] = str(row[2])
        return payload

    def chunks(self, doc_id: str) -> list[dict[str, Any]]:
        ids = [str(row[0]) for row in self.conn.execute("SELECT chunk_id FROM chunks WHERE doc_id=? ORDER BY start, chunk_id", (doc_id,))]
        return [self.row("chunks", chunk_id) for chunk_id in ids]

    def close(self) -> None:
        self.nodes.close(); self.chunks_source.close(); self.conn.close()

    def __enter__(self) -> "TraceReader": return self
    def __exit__(self, *_: Any) -> None: self.close()


def token_coverage(query_terms: Sequence[str], text: str, tokenizer: ExactFtsSession | None = None) -> float:
    needed = set(query_terms)
    tokens = tokenizer.terms(text) if tokenizer else exact_fts_terms(text)
    return len(needed & set(tokens)) / len(needed) if needed else 0.0


def shortest_phrase_span(query_tokens: Sequence[str], text: str) -> int | None:
    terms = surface(text); phrases = [tuple(query_tokens[pos:pos + width]) for width in (2, 3) for pos in range(max(0, len(query_tokens) - width + 1))]
    matches = [width for phrase in phrases for width in [len(phrase)] if any(tuple(terms[pos:pos + width]) == phrase for pos in range(max(0, len(terms) - width + 1)))]
    return min(matches) if matches else None


def shortest_proximity_span(query_tokens: Sequence[str], text: str) -> int | None:
    needed = set(query_tokens); terms = surface(text)
    if not needed: return None
    positions = [(index, token) for index, token in enumerate(terms) if token in needed]
    have: dict[str, int] = defaultdict(int); left = 0; best: int | None = None
    for right, (position, token) in enumerate(positions):
        have[token] += 1
        while len(have) == len(needed):
            width = position - positions[left][0] + 1; best = width if best is None else min(best, width)
            left_token = positions[left][1]; have[left_token] -= 1
            if have[left_token] == 0: del have[left_token]
            left += 1
    return best


def v0_term_document_frequency(terms: Sequence[str], conn: sqlite3.Connection | None = None) -> dict[str, int]:
    owned = conn is None
    conn = conn or sqlite3.connect(f"file:{EXP021_DB.as_posix()}?mode=ro", uri=True)
    try:
        return {term: int(conn.execute("SELECT count(*) FROM passages WHERE passages MATCH ?", (fts_or([term]),)).fetchone()[0]) for term in terms}
    finally:
        if owned: conn.close()


def source_snippet(text: str, query_tokens: Sequence[str], limit: int = 240) -> str:
    lower = text.casefold(); positions = [lower.find(token.casefold()) for token in query_tokens if lower.find(token.casefold()) >= 0]
    start = max(0, min(positions) - 80) if positions else 0
    return text[start:start + limit]


def lexical_audit(resume: bool) -> dict[str, Any]:
    questions, answers, folds, fold_for, stats = canonical_inner()
    trace = build_lexical_trace(resume)
    directory = CACHE / "lexical_audit"; shard_dir = directory / "shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    qids = sorted(questions); shard_size = 64
    config = {"trace": trace["database_sha256"], "v0_fixture": sha256(CACHE / "v0_reproduction_fixture.jsonl"), "code": sha256(Path(__file__)), "scope": "folds_1_to_4_only", "shard_size": shard_size}
    config_hash = fingerprint(config); records: list[dict[str, Any]] = []
    sys.path.insert(0, str(ROOT / "src")); from exp012b_bm25 import default_segmenter
    v0_conn = sqlite3.connect(f"file:{EXP021_DB.as_posix()}?mode=ro", uri=True)
    try:
        with TraceReader() as reader, ExactFtsSession() as tokenizer, Run("lexical-audit", len(qids)) as run:
            completed = 0
            for shard, start in enumerate(range(0, len(qids), shard_size)):
                selected = qids[start:start + shard_size]; path = shard_dir / f"audit-{shard:04d}.jsonl"; marker = path.with_suffix(".json")
                valid = resume and path.exists() and marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("config_hash") == config_hash and json.loads(marker.read_text(encoding="utf-8")).get("sha256") == sha256(path)
                if valid:
                    shard_records = list(read_jsonl(path))
                else:
                    shard_records = []
                    for qid in selected:
                        query = questions[qid]; raw_tokens = surface(query)
                        under_text = default_segmenter(query); under_tokens = WORD.findall(under_text.casefold()); fts_tokens = tokenizer.sequence(under_text)
                        term_df = v0_term_document_frequency(fts_tokens, v0_conn)
                        oov_terms = [term for term in fts_tokens if term_df[term] == 0]
                        high_df_terms = [term for term in fts_tokens if term_df[term] >= 0.20 * 343347]
                        control = source_rows(query, "v0_control", limit=100, v0_config=exp021_config(fold_for[qid]))
                        ranks = [row["doc_id"] for row in control]; first = next((rank for rank, doc in enumerate(ranks, 1) if doc in answers[qid]), None)
                        gold_docs = [(doc_id, reader.row("documents", doc_id)) for doc_id in sorted(answers[qid])]
                        whole = {doc_id: token_coverage(fts_tokens, str(row["raw_text"]), tokenizer) for doc_id, row in gold_docs}
                        gold_chunks = [(doc_id, chunk) for doc_id, _doc in gold_docs for chunk in reader.chunks(doc_id)]
                        best_chunk = max(((token_coverage(fts_tokens, str(chunk["raw_text"]), tokenizer), doc_id, chunk) for doc_id, chunk in gold_chunks), default=(0.0, None, None), key=lambda item: item[0])
                        window_cov = {f"w{width}_o{overlap}": max((token_coverage(fts_tokens, record["text"], tokenizer) for doc_id, doc in gold_docs for record in windows(doc_id, str(doc["raw_text"]), width, overlap) or ()), default=0.0) for width, overlap in ((384, 96), (512, 128))}
                        confuser = next((row for row in control if row["doc_id"] not in answers[qid]), None)
                        confuser_doc = reader.row("documents", str(confuser["doc_id"])) if confuser else None
                        phrase_gold = max((shortest_phrase_span(fts_tokens, str(row["raw_text"])) or 0 for _doc_id, row in gold_docs), default=0) or None
                        phrase_confuser = shortest_phrase_span(fts_tokens, str(confuser_doc["raw_text"])) if confuser_doc else None
                        proximity_gold = min((value for _doc_id, row in gold_docs if (value := shortest_proximity_span(fts_tokens, str(row["raw_text"]))) is not None), default=None)
                        proximity_confuser = shortest_proximity_span(fts_tokens, str(confuser_doc["raw_text"])) if confuser_doc else None
                        tags: list[str] = []
                        if raw_tokens != under_tokens or raw_tokens != fts_tokens: tags.append("tokenization_split_merge")
                        if citation_tokens(query): tags.append("legal_identifier_normalization")
                        if any(any(char.isdigit() for char in token) for token in raw_tokens): tags.append("numeric_unit_normalization")
                        if oov_terms: tags.append("oov_surface_or_parser")
                        if high_df_terms: tags.append("high_df_term_dilution")
                        if max(whole.values(), default=0.0) - best_chunk[0] >= .30: tags.append("distributed_across_chunks")
                        if best_chunk[2] and best_chunk[2].get("source_kind") == "fallback": tags.append("fallback_parser")
                        if phrase_gold and phrase_confuser: tags.append("phrase_confuser")
                        if proximity_gold is not None and proximity_confuser is not None and proximity_confuser <= proximity_gold: tags.append("proximity_confuser")
                        if first is None and max(whole.values(), default=0.0) < .40: tags.append("low_source_coverage")
                        if not tags: tags.append("unclassified")
                        shard_records.append({"qid": qid, "fold": fold_for[qid], "raw_surface_tokens": raw_tokens, "underthesea_tokens": under_tokens, "fts_normalized_tokens": fts_tokens, "fts_term_document_frequency": term_df, "oov_fts_terms": oov_terms, "high_df_fts_terms": high_df_terms, "citation_tokens": citation_tokens(query), "gold_doc_ids": sorted(answers[qid]), "v0_first_gold_rank": first, "whole_parent_coverage": whole, "best_structural_chunk_coverage": best_chunk[0], "best_structural_chunk": None if best_chunk[2] is None else {"doc_id": best_chunk[1], "chunk_id": best_chunk[2]["chunk_id"], "start": best_chunk[2]["start"], "end": best_chunk[2]["end"]}, "best_fixed_window_coverage": window_cov, "phrase_span_gold": phrase_gold, "phrase_span_top_confuser": phrase_confuser, "proximity_span_gold": proximity_gold, "proximity_span_top_confuser": proximity_confuser, "parse_modes": {doc_id: row.get("parse_mode", "unknown") for doc_id, row in gold_docs}, "failure_tags": tags, "examples": {"gold": [{"doc_id": doc_id, "source_sha256": hashlib.sha256(str(row["raw_text"]).encode("utf-8")).hexdigest(), "snippet": source_snippet(str(row["raw_text"]), raw_tokens)} for doc_id, row in gold_docs], "top_confuser": None if confuser_doc is None else {"doc_id": confuser["doc_id"], "score": confuser["raw_score"], "best_unit_id": confuser["best_unit_id"], "source_sha256": hashlib.sha256(str(confuser_doc["raw_text"]).encode("utf-8")).hexdigest(), "snippet": source_snippet(str(confuser_doc["raw_text"]), raw_tokens)}}})
                    temp = path.with_suffix(".tmp"); temp.write_text("".join(canonical(row) + "\n" for row in shard_records), encoding="utf-8"); temp.replace(path)
                    atomic_json(marker, {"config_hash": config_hash, "qids": selected, "records": len(shard_records), "sha256": sha256(path)})
                records.extend(shard_records); completed += len(selected); run.heartbeat(completed, shard=shard)
    finally: v0_conn.close()
    tag_counts = {tag: sum(tag in record["failure_tags"] for record in records) for tag in sorted({tag for record in records for tag in record["failure_tags"]})}
    source_exact_examples = {tag: [record["qid"] for record in records if tag in record["failure_tags"]][:20] for tag in tag_counts}
    path = RESULTS / "LEXICAL_FAILURE_AUDIT.json"; payload = {"schema_version": SCHEMA, "status": "PASS_LEXICAL_AUDIT", "scope": "folds_1_to_4_only", "fold0_labels_deserialized": False, "labels": stats, "config": config, "tag_counts": tag_counts, "source_exact_examples_qids": source_exact_examples, "records": records}
    atomic_json(path, payload); write_success(RESULTS, "lexical-audit", {"records_sha256": sha256(path)}); return {"records": len(records), "path": str(path)}


def build_index(view: str, resume: bool) -> dict[str, Any]:
    if view == "v0_control":
        if not EXP021_DB.exists(): raise GateRejected("REJECTED_READING_OR_INPUT_GATE: missing EXP-021 DB")
        return {"view": view, "status": "REUSED_EXACT_EXP021_CONTROL", "database_sha256": sha256(EXP021_DB)}
    if view == "v1_surface_structural": return build_surface_structural(resume)
    if view in {"v2_windows", "v3_parent", "v5_citation"}: return build_windows_and_parent(resume)
    raise ValueError(view)


def ranked_docs(rows: Sequence[Mapping[str, Any]]) -> list[str]: return [str(row["doc_id"]) for row in rows]


def candidate_coverage(rankings: Mapping[str, Mapping[str, Sequence[str]]], answers: Mapping[str, set[str]], qids: Sequence[str], k: int) -> tuple[dict[str, float], dict[str, set[str]]]:
    sets = {qid: set(doc for source in rankings.values() for doc in source[qid][:k]) for qid in qids}
    values = {qid: len(sets[qid] & answers[qid]) / len(answers[qid]) if answers[qid] else 0.0 for qid in qids}
    return {f"candidate_coverage@{k}": float(np.mean([values[qid] for qid in qids if answers[qid]])), "evaluable_queries": float(sum(bool(answers[qid]) for qid in qids))}, sets


def best_source_oracle(rankings: Mapping[str, Mapping[str, Sequence[str]]], answers: Mapping[str, set[str]], qids: Sequence[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    chosen, provenance = {}, {}
    for qid in qids:
        view, docs = max(rankings.items(), key=lambda item: (len(set(item[1][qid][:5]) & answers[qid]) / len(answers[qid]) if answers[qid] else 0.0, len(set(item[1][qid][:5]) & answers[qid]), -list(rankings).index(item[0])))
        chosen[qid], provenance[qid] = list(docs[qid]), view
    return chosen, provenance


def bounded_screen(resume: bool) -> dict[str, Any]:
    del resume
    required = [CACHE / "v1_surface_structural" / "index.sqlite", CACHE / "v2_v3" / "index.sqlite"]
    if any(not path.exists() for path in required): raise GateRejected("REJECTED_BOUNDED_SPARSE_COMPLEMENTARITY_GATE: required indexes absent")
    questions, answers, folds, fold_for, stats = canonical_inner()
    qids = sorted(questions, key=lambda qid: hashlib.sha256(("exp111-bounded:" + qid).encode()).hexdigest())[:512]
    views = ("v0_control", "v1_surface_structural", "v2_w384", "v2_w512", "v4_bigram", "v4_trigram", "v3_parent", "v5_citation")
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {view: {} for view in views}
    with Run("bounded-screen", len(qids) * len(views)) as run:
        completed = 0
        for qid in qids:
            for view in views:
                all_rows[view][qid] = source_rows(questions[qid], view, limit=500, v0_config=exp021_config(fold_for[qid]) if view == "v0_control" else None); completed += 1
                if completed % 64 == 0: run.heartbeat(completed)
    pred = {view: {qid: ranked_docs(rows[qid]) for qid in qids} for view, rows in all_rows.items()}
    metrics = {view: metric(pred[view], {qid: answers[qid] for qid in qids}) for view in views}
    oracle, oracle_provenance = best_source_oracle(pred, {qid: answers[qid] for qid in qids}, qids)
    union50, union50_sets = candidate_coverage(pred, {qid: answers[qid] for qid in qids}, qids, 50)
    v0_50, _ = candidate_coverage({"v0_control": pred["v0_control"]}, {qid: answers[qid] for qid in qids}, qids, 50)
    unique20_pairs = {(qid, gold) for qid in qids for gold in answers[qid] if gold not in set(pred["v0_control"][qid][:20]) and any(gold in set(pred[view][qid][:20]) for view in views if view != "v0_control")}
    per_view_unique20 = {view: sorted((qid, gold) for qid, gold in unique20_pairs if gold in set(pred[view][qid][:20])) for view in views if view != "v0_control"}
    fused = {qid: hierarchical_rrf({view: pred[view][qid] for view in views}) for qid in qids}
    baseline = metrics["v0_control"]["recall@5"]; delta = metric(fused, {qid: answers[qid] for qid in qids})["recall@5"] - baseline
    oracle_metric = metric(oracle, {qid: answers[qid] for qid in qids})
    conditions = [oracle_metric["recall@5"] - baseline >= .010, len(unique20_pairs) >= 10, delta >= .010]
    passed = sum(conditions) >= 2 and union50["candidate_coverage@50"] >= v0_50["candidate_coverage@50"]
    result = {"schema_version": SCHEMA, "status": "PASS_BOUNDED_SPARSE_COMPLEMENTARITY_GATE" if passed else "REJECTED_BOUNDED_SPARSE_COMPLEMENTARITY_GATE", "scope": "stable_hash_512_folds_1_to_4", "fold0_labels_deserialized": False, "labels": stats, "metrics": metrics, "best_source_choice_oracle": {"metrics": oracle_metric, "provenance": oracle_provenance}, "deployable_fused_metrics": metric(fused, {qid: answers[qid] for qid in qids}), "unique_qid_gold_recoveries_at20": sorted(unique20_pairs), "per_view_unique_recovery_membership_at20": per_view_unique20, "candidate_set_coverage": {"union_top50_per_source": union50, "v0_top50": v0_50, "mean_unique_candidates": float(np.mean([len(value) for value in union50_sets.values()]))}, "gate_conditions": conditions, "views": views}
    atomic_json(RESULTS / "BOUNDED_SOURCE_SCREEN.json", result)
    if not passed: raise GateRejected(result["status"])
    write_success(RESULTS, "bounded-screen", result); return result


def hierarchical_rrf(rankings: Mapping[str, Sequence[str]], k: int = 20, family_weights: Mapping[str, float] | None = None) -> list[str]:
    family = {"v0_control": "structural", "v1_surface_structural": "structural", "v2_w384": "local", "v2_w512": "local", "v4_bigram": "local", "v4_trigram": "local", "v3_parent": "global", "v5_citation": "specialist"}
    family_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for view, docs in rankings.items():
        for rank, doc in enumerate(docs, 1): family_scores[family[view]][str(doc)] = max(family_scores[family[view]][str(doc)], 1.0 / (k + rank))
    scores: dict[str, float] = defaultdict(float)
    family_weights = dict(family_weights or {"structural": 1.0, "local": 1.0, "global": 1.0, "specialist": 1.0})
    for name, values in family_scores.items():
        for doc, value in values.items(): scores[doc] += family_weights.get(name, 0.0) * value
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


def family_weight_grid() -> Iterator[dict[str, float]]:
    """Deterministic simplex step .10, minimum .10 for required active families."""
    for structural in range(1, 11):
        for local in range(1, 11 - structural):
            for global_ in range(1, 11 - structural - local):
                specialist = 10 - structural - local - global_
                yield {"structural": structural / 10, "local": local / 10, "global": global_ / 10, "specialist": specialist / 10}


COMPACT_SCORE_FIELDS = ("doc_id", "rank", "raw_score", "robust_z", "matching_units", "score_gap")


def score_v2_directory(stage: str = "inner") -> Path:
    return CACHE / "source_scores_v2" / stage


def score_v2_config(qids: Sequence[str], stage: str = "inner") -> dict[str, Any]:
    return {"schema": "source_scores_v2_compact_v1", "stage": stage, "qids": list(qids), "views": SOURCE_VIEWS,
        "top_parents": 500, "fields": COMPACT_SCORE_FIELDS, "v1": sha256(CACHE / "v1_surface_structural" / "index.sqlite"),
        "v2": sha256(CACHE / "v2_v3" / "index.sqlite"), "v0": sha256(EXP021_DB), "scorer_source": sha256(Path(__file__))}


def compact_score_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Lossless for all current downstream score/feature consumers; preserves ranking order."""
    return {field: row.get(field) for field in COMPACT_SCORE_FIELDS}


def compact_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"qid": str(record["qid"]), "sources": {view: [compact_score_row(row) for row in record["sources"][view][:500]] for view in SOURCE_VIEWS}}


def shard_paths(directory: Path, shard: int) -> tuple[Path, Path]:
    path = directory / f"scores-{shard:04d}.jsonl"; return path, path.with_suffix(".json")


def verified_shard(path: Path, marker: Path, qids: Sequence[str], config_hash: str | None = None) -> bool:
    if not path.exists() or not marker.exists(): return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    return saved.get("sha256") == sha256(path) and saved.get("qids") == list(qids) and (config_hash is None or saved.get("config_hash") == config_hash)


def migrate_old_score_shard(shard: int, *, stage: str = "inner") -> dict[str, Any]:
    """One old shard at a time; never mutates the legacy cache."""
    old_dir = CACHE / "source_scores" / stage; old_path, old_marker = shard_paths(old_dir, shard)
    saved = json.loads(old_marker.read_text(encoding="utf-8"))
    if not verified_shard(old_path, old_marker, saved["qids"], saved.get("config_hash")): raise GateRejected("REJECTED_OLD_SHARD_HASH_GATE")
    new_dir = score_v2_directory(stage); new_dir.mkdir(parents=True, exist_ok=True); path, marker = shard_paths(new_dir, shard)
    migration = {"schema": "source_scores_v2_migration_v1", "source_path": str(old_path.relative_to(ROOT)), "source_sha256": sha256(old_path), "source_config_hash": saved["config_hash"], "fields": COMPACT_SCORE_FIELDS}
    migration_hash = fingerprint(migration)
    if verified_shard(path, marker, saved["qids"], migration_hash): return json.loads(marker.read_text(encoding="utf-8"))
    records = [compact_source_record(row) for row in read_jsonl(old_path)]
    temp = path.with_suffix(".tmp"); temp.write_text("".join(canonical(row) + "\n" for row in records), encoding="utf-8"); temp.replace(path)
    result = {"schema_version": SCHEMA, "config_hash": migration_hash, "qids": saved["qids"], "records": len(records), "sha256": sha256(path), "migration": migration}
    atomic_json(marker, result); return result


def score_v2_shard(shard: int, *, stage: str = "inner") -> dict[str, Any]:
    """Exactly one 64-query shard per child process; it exits after atomic commit."""
    questions, _answers, _folds, fold_for, _labels = canonical_inner(); qids = sorted(questions); shard_size = 64
    selected = qids[shard * shard_size:(shard + 1) * shard_size]
    if not selected: raise GateRejected("REJECTED_SOURCE_SHARD_RANGE_GATE")
    directory = score_v2_directory(stage); directory.mkdir(parents=True, exist_ok=True); path, marker = shard_paths(directory, shard)
    config_hash = fingerprint(score_v2_config(qids, stage))
    if verified_shard(path, marker, selected, config_hash): return json.loads(marker.read_text(encoding="utf-8"))
    rows = []
    with Run(f"score-v2-shard-{shard:04d}", len(selected) * len(SOURCE_VIEWS)) as run:
        for number, qid in enumerate(selected, 1):
            sources = {view: source_rows(questions[qid], view, limit=500, v0_config=exp021_config(fold_for[qid]) if view == "v0_control" else None) for view in SOURCE_VIEWS}
            rows.append(compact_source_record({"qid": qid, "sources": sources}))
            run.heartbeat(number * len(SOURCE_VIEWS), shard=shard)
        temp = path.with_suffix(".tmp"); temp.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8"); temp.replace(path)
        result = {"schema_version": SCHEMA, "config_hash": config_hash, "qids": selected, "records": len(rows), "sha256": sha256(path), "scoring": score_v2_config(qids, stage)}
        atomic_json(marker, result)
    return result


def run_score_v2_children(*, stage: str = "inner", start_shard: int = 24, child_timeout_seconds: int = 7200) -> dict[str, Any]:
    """Visible parent orchestrator; every scorer child handles one shard then exits."""
    questions, _answers, _folds, _fold_for, _labels = canonical_inner(); qids = sorted(questions); total_shards = expected_inner_shards(qids)
    if not 0 <= start_shard < total_shards: raise GateRejected("REJECTED_SOURCE_SHARD_RANGE_GATE")
    directory = score_v2_directory(stage); durations: deque[float] = deque(maxlen=8); completed = start_shard
    with Run("score-v2-orchestrator", total_shards) as run:
        for shard in range(start_shard, total_shards):
            selected = qids[shard * 64:(shard + 1) * 64]; path, marker = shard_paths(directory, shard)
            if path.exists() and marker.exists() and verified_shard(path, marker, selected):
                completed = shard + 1; run.heartbeat(completed, shard=shard, skipped_verified=True); continue
            started = time.monotonic()
            command = [sys.executable, "-u", str(Path(__file__)), "score-v2-shard", "--shard", str(shard)]
            try:
                child = subprocess.run(command, cwd=ROOT, timeout=child_timeout_seconds, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8")
            except subprocess.TimeoutExpired as exc:
                run.status("FAILED", completed, failure="CHILD_TIMEOUT", shard=shard, timeout_seconds=child_timeout_seconds)
                raise GateRejected("REJECTED_SOURCE_CHILD_TIMEOUT_GATE") from exc
            elapsed = time.monotonic() - started; durations.append(elapsed)
            if child.returncode != 0 or not verified_shard(path, marker, selected):
                run.status("FAILED", completed, failure="CHILD_EXIT_OR_HASH", shard=shard, returncode=child.returncode, stderr_tail=child.stderr[-2000:])
                raise GateRejected("REJECTED_SOURCE_CHILD_EXIT_GATE")
            completed = shard + 1; rolling = float(statistics.median(durations)); eta = rolling * (total_shards - completed)
            run.heartbeat(completed, shard=shard, child_elapsed_seconds=elapsed, rolling_median_shard_seconds=rolling, rolling_eta_seconds=eta, parent_rss_after_child=rss_bytes(), available_system_ram_bytes=system_available_ram_bytes())
    result = {"status": "PASS_SOURCE_V2_ORCHESTRATION", "stage": stage, "start_shard": start_shard, "total_shards": total_shards, "completed_shards": completed, "fold0_called": False}
    atomic_json(score_v2_directory(stage) / "orchestrator_manifest.json", result); return result


def iter_v2_shards(stage: str = "inner") -> Iterator[tuple[int, Path, dict[str, Any]]]:
    directory = score_v2_directory(stage)
    for path in sorted(directory.glob("scores-*.jsonl")):
        marker = path.with_suffix(".json"); saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("sha256") != sha256(path): raise GateRejected("REJECTED_V2_SHARD_HASH_GATE")
        yield int(path.stem.split("-")[-1]), path, saved


def mark_interrupted_resource_unbounded() -> dict[str, Any]:
    """Reconcile the stale legacy launcher state without altering legacy shard bytes."""
    old_dir = CACHE / "source_scores" / "inner"; verified = []
    for shard in range(24):
        path, marker = shard_paths(old_dir, shard); saved = json.loads(marker.read_text(encoding="utf-8"))
        if not verified_shard(path, marker, saved["qids"], saved["config_hash"]): raise GateRejected("REJECTED_OLD_SHARD_HASH_GATE")
        verified.append({"shard": shard, "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    previous = json.loads((RESULTS / "RUN_STATUS.json").read_text(encoding="utf-8"))
    payload = {"schema_version": SCHEMA, "state": "INTERRUPTED_RESOURCE_UNBOUNDED_ACCUMULATOR", "stage": "score-sources-inner-legacy-v1", "last_heartbeat": previous.get("last_heartbeat"), "legacy_run_id": previous.get("run_id"), "verified_shards": len(verified), "verified_shard_range": [0, 23], "legacy_shards": verified, "recovery": "source_scores_v2 disk-backed migration required; legacy source bytes are immutable"}
    atomic_json(RESULTS / "RUN_STATUS.json", payload); atomic_json(RESULTS / "INTERRUPTED_RESOURCE_UNBOUNDED_ACCUMULATOR.json", payload); return payload


def parity_old_v2(shards: Sequence[int] = (0, 1, 2, 3)) -> dict[str, Any]:
    """Exact document order and scalar parity over migrated legacy shard records."""
    old_dir = CACHE / "source_scores" / "inner"; new_dir = score_v2_directory("inner"); checked = 0
    for shard in shards:
        old_path, _old_marker = shard_paths(old_dir, shard); new_path, _new_marker = shard_paths(new_dir, shard)
        for old, new in zip(read_jsonl(old_path), read_jsonl(new_path), strict=True):
            if old["qid"] != new["qid"]: raise GateRejected("REJECTED_V2_PARITY_QID_GATE")
            for view in SOURCE_VIEWS:
                left, right = old["sources"][view][:500], new["sources"][view]
                if [row["doc_id"] for row in left] != [row["doc_id"] for row in right]: raise GateRejected("REJECTED_V2_PARITY_ORDER_GATE")
                for a, b in zip(left, right, strict=True):
                    for field in COMPACT_SCORE_FIELDS[1:]:
                        x, y = a.get(field), b.get(field)
                        if x is None or y is None:
                            if x != y: raise GateRejected("REJECTED_V2_PARITY_SCALAR_GATE")
                        elif abs(float(x) - float(y)) > 1e-12: raise GateRejected("REJECTED_V2_PARITY_SCALAR_GATE")
            checked += 1
    return {"status": "PASS_OLD_V2_PARITY", "shards": list(shards), "queries": checked, "fields": COMPACT_SCORE_FIELDS}


def parity_old_v2_streaming_metrics(shards: Sequence[int] = (0, 1, 2, 3)) -> dict[str, Any]:
    """The v2 streaming sufficient statistics must equal legacy dict-style metrics."""
    _questions, answers, _folds, _fold_for, _labels = canonical_inner(); old_dir = CACHE / "source_scores" / "inner"; new_dir = score_v2_directory("inner")
    old_totals = {view: MetricSums() for view in SOURCE_VIEWS}; new_totals = {view: MetricSums() for view in SOURCE_VIEWS}
    for shard in shards:
        old_path, _ = shard_paths(old_dir, shard); new_path, _ = shard_paths(new_dir, shard)
        for old, new in zip(read_jsonl(old_path), read_jsonl(new_path), strict=True):
            for view in SOURCE_VIEWS:
                old_totals[view].add([str(row["doc_id"]) for row in old["sources"][view]], answers[str(old["qid"])])
                new_totals[view].add([str(row["doc_id"]) for row in new["sources"][view]], answers[str(new["qid"])])
    left, right = {view: item.report() for view, item in old_totals.items()}, {view: item.report() for view, item in new_totals.items()}
    if canonical(left) != canonical(right): raise GateRejected("REJECTED_V2_STREAMING_METRIC_PARITY_GATE")
    return {"status": "PASS_V2_STREAMING_METRIC_PARITY", "shards": list(shards), "metrics": left}


def bootstrap_delta(left: Mapping[str, Sequence[str]], right: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], *, seed: int = 111, samples: int = 2000) -> dict[str, float]:
    qids = np.asarray([qid for qid, gold in answers.items() if gold], dtype=object); rng = np.random.default_rng(seed)
    values = np.asarray([len(set(right[qid][:5]) & answers[qid]) / len(answers[qid]) - len(set(left[qid][:5]) & answers[qid]) / len(answers[qid]) for qid in qids], dtype=np.float64)
    means = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(samples)])
    return {"mean": float(values.mean()), "lower": float(np.quantile(means, .025)), "upper": float(np.quantile(means, .975)), "samples": samples, "seed": seed}


@dataclass
class MetricSums:
    count: int = 0
    multi_count: int = 0
    values: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.values is None: self.values = defaultdict(float)

    def add(self, docs: Sequence[str], gold: set[str]) -> None:
        if not gold: return
        self.count += 1
        for k in CURVE: self.values[f"recall@{k}"] += len(set(docs[:k]) & gold) / len(gold)
        self.values["precision@5"] += len(set(docs[:5]) & gold) / 5.0
        self.values["mrr@5"] += next((1.0 / rank for rank, doc in enumerate(docs[:5], 1) if doc in gold), 0.0)
        if len(gold) > 1:
            self.multi_count += 1; self.values["multi_gold_recall@5"] += len(set(docs[:5]) & gold) / len(gold)

    def report(self) -> dict[str, float]:
        output = {"evaluable_queries": float(self.count)}
        for key, value in self.values.items(): output[key] = value / (self.multi_count if key == "multi_gold_recall@5" and self.multi_count else max(1, self.count))
        return output


@dataclass
class CandidateCoverageSums:
    """Streaming set coverage for the union of each source's top-K candidates.

    This deliberately has no ranking semantics.  A concatenated source order is
    a deployable ranking candidate, but it is not a candidate-set ceiling: it
    can evict V0 ranks 101--K before those ranks are measured.
    """
    count: int = 0
    values: dict[int, float] | None = None

    def __post_init__(self) -> None:
        if self.values is None: self.values = defaultdict(float)

    def add(self, docs_by_view: Mapping[str, Sequence[str]], gold: set[str], ks: Sequence[int] = CURVE) -> None:
        if not gold: return
        self.count += 1
        for k in ks:
            candidates = set(doc for docs in docs_by_view.values() for doc in docs[:k])
            self.values[k] += len(candidates & gold) / len(gold)

    def report(self) -> dict[str, float]:
        output = {"evaluable_queries": float(self.count)}
        for k, value in self.values.items(): output[f"candidate_coverage@{k}_per_source"] = value / max(1, self.count)
        return output


def expected_inner_shards(qids: Sequence[str], shard_size: int = 64) -> int: return math.ceil(len(qids) / shard_size)


def full_source_audit(resume: bool) -> dict[str, Any]:
    del resume
    required = [CACHE / "v1_surface_structural" / "index.sqlite", CACHE / "v2_v3" / "index.sqlite"]
    if any(not path.exists() for path in required): raise GateRejected("REJECTED_FULL_SOURCE_INPUT_GATE: sparse indexes absent")
    questions, answers, folds, fold_for, labels = canonical_inner(); qids = sorted(questions); expected = expected_inner_shards(qids)
    shards = list(iter_v2_shards("inner"))
    if [number for number, _path, _meta in shards] != list(range(expected)): raise GateRejected("REJECTED_V2_SOURCE_COMPLETENESS_GATE")
    totals = {view: MetricSums() for view in SOURCE_VIEWS}; union_coverage = CandidateCoverageSums()
    by_fold = {fold: {view: MetricSums() for view in SOURCE_VIEWS} for fold in INNER_FOLDS}; exclusive = {view: 0 for view in SOURCE_VIEWS}
    phrase = {"bigram": 0, "trigram": 0}; processed = 0; database = CACHE / "v2_v3" / "index.sqlite"
    with Run("full-source-audit-streaming", len(qids)) as run, sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as phrase_conn, ExactFtsSession(max_cache=65536) as phrase_tokenizer:
        phrase_total_windows = int(phrase_conn.execute("SELECT count(*) FROM local384").fetchone()[0]); phrase_df_cache: dict[str, int] = {}
        for shard, path, _meta in shards:
            for record in read_jsonl(path):
                qid = str(record["qid"]); gold = answers[qid]; fold = fold_for[qid]
                docs_by_view = {view: [str(row["doc_id"]) for row in record["sources"][view]] for view in SOURCE_VIEWS}
                for view, docs in docs_by_view.items(): totals[view].add(docs, gold); by_fold[fold][view].add(docs, gold)
                union_coverage.add(docs_by_view, gold)
                for view, docs in docs_by_view.items():
                    exclusive[view] += sum(gold_doc in set(docs[:50]) and not any(gold_doc in set(other[:50]) for other_view, other in docs_by_view.items() if other_view != view) for gold_doc in gold)
                terms = phrase_tokenizer.sequence(" ".join(surface(questions[qid])))
                phrase["bigram"] += int(phrase_expression_terms(phrase_conn, "local384", terms, 2, total_windows=phrase_total_windows, df_cache=phrase_df_cache) != '"__exp111_no_token__"')
                phrase["trigram"] += int(phrase_expression_terms(phrase_conn, "local384", terms, 3, total_windows=phrase_total_windows, df_cache=phrase_df_cache) != '"__exp111_no_token__"')
                processed += 1
            run.heartbeat(processed, shard=shard, rolling_shards=1)
    metrics = {view: total.report() for view, total in totals.items()}; union_metric = union_coverage.report()
    phrase_eligibility = {name: value / len(qids) for name, value in phrase.items()}
    coverage_ks = (50, 100, 200)
    gates = {
        "v0_exact_exp021_inner": abs(metrics["v0_control"]["recall@5"] - 0.836721497109482) <= 1e-12,
        "union_set_coverage_not_below_v0_at_50_100_200": all(union_metric[f"candidate_coverage@{k}_per_source"] >= metrics["v0_control"][f"recall@{k}"] for k in coverage_ks),
        "bigram_expression_nonempty_rate_ge_001": phrase_eligibility["bigram"] >= .01,
        "trigram_expression_nonempty_rate_ge_001": phrase_eligibility["trigram"] >= .01,
    }
    status = "PASS_FULL_SOURCE_AUDIT" if all(gates.values()) else "REJECTED_FULL_SOURCE_AUDIT_GATE"
    result = {"schema_version": SCHEMA, "status": status, "scope": "folds_1_to_4_only", "fold0_labels_deserialized": False, "labels": labels, "streaming": True,
        "metrics": metrics, "union_candidate_ceiling": union_metric, "exclusive_qid_gold_pair_count_at50": exclusive, "phrase_expression_eligible_rate": phrase_eligibility, "gates": gates,
        "per_fold": {fold: {view: by_fold[fold][view].report() for view in SOURCE_VIEWS} for fold in INNER_FOLDS}}
    atomic_json(RESULTS / "FULL_SOURCE_AUDIT.json", result)
    if status != "PASS_FULL_SOURCE_AUDIT": raise GateRejected(status)
    write_success(RESULTS, "full-source-audit", result); return result


def sparse_feature_rows_from_sources(sources: Mapping[str, Sequence[Mapping[str, Any]]], query: str) -> tuple[list[str], list[list[float]], list[str]]:
    """Fixed-order, label-free feature matrix from one compact disk record."""
    views = SOURCE_VIEWS
    families = (("v0_control", "v1_surface_structural"), ("v2_w384", "v2_w512", "v4_bigram", "v4_trigram"), ("v3_parent",), ("v5_citation",))
    candidates = list(dict.fromkeys(doc for family in families for doc in hierarchical_rrf({view: ranked_docs(sources[view]) for view in family})[:50]))
    by_view = {view: {str(row["doc_id"]): row for row in sources[view][:500]} for view in views}
    tokens = surface(query); values: list[list[float]] = []
    names = [f"{view}_{feature}" for view in views for feature in ("present", "rank_recip", "raw_score", "robust_z", "matching_units", "gap")] + ["source_agreement", "query_token_count", "query_unique_ratio", "query_has_citation", "query_has_numeric"]
    for doc in candidates:
        row: list[float] = []
        present = 0
        for view in views:
            item = by_view[view].get(doc)
            if item is None:
                row.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)); continue
            present += 1
            row.extend((1.0, 1.0 / float(item["rank"]), float(item["raw_score"]), float(item.get("robust_z", 0.0)), float(item.get("matching_units", 0)), float(item.get("score_gap") or 0.0)))
        row.extend((float(present), float(len(tokens)), float(len(set(tokens)) / max(1, len(tokens))), float(bool(citation_tokens(query))), float(any(any(char.isdigit() for char in token) for token in tokens))))
        values.append(row)
    return candidates, values, names


def sparse_feature_rows(scores: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], questions: Mapping[str, str], qid: str) -> tuple[list[str], list[list[float]], list[str]]:
    return sparse_feature_rows_from_sources({view: scores[view][qid] for view in SOURCE_VIEWS}, questions[qid])


def feature_v2_directory(stage: str = "inner") -> Path: return CACHE / "sparse_features_v2" / stage


def build_feature_v2_shard(shard: int, *, stage: str = "inner") -> dict[str, Any]:
    """Emit query-group feature shards without loading other retrieval shards."""
    questions, answers, _folds, fold_for, _labels = canonical_inner(); source_path, source_marker = shard_paths(score_v2_directory(stage), shard)
    if not source_path.exists() or not source_marker.exists(): raise GateRejected("REJECTED_V2_SOURCE_COMPLETENESS_GATE")
    source_hash = sha256(source_path); directory = feature_v2_directory(stage); directory.mkdir(parents=True, exist_ok=True); path, marker = shard_paths(directory, shard)
    config = {"schema": "sparse_features_v2_v1", "source_sha256": source_hash, "source_marker_sha256": sha256(source_marker), "fields": "sparse_feature_rows_from_sources_v1"}; config_hash = fingerprint(config)
    if path.exists() and marker.exists() and verified_shard(path, marker, json.loads(source_marker.read_text(encoding="utf-8"))["qids"], config_hash): return json.loads(marker.read_text(encoding="utf-8"))
    rows = []
    for source in read_jsonl(source_path):
        qid = str(source["qid"]); docs, features, names = sparse_feature_rows_from_sources(source["sources"], questions[qid])
        rows.append({"qid": qid, "fold": fold_for[qid], "docs": docs, "features": np.asarray(features, dtype=np.float32).tolist(), "labels": [int(doc in answers[qid]) for doc in docs], "feature_names": names})
    temp = path.with_suffix(".tmp"); temp.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8"); temp.replace(path)
    result = {"schema_version": SCHEMA, "config_hash": config_hash, "qids": [row["qid"] for row in rows], "records": len(rows), "sha256": sha256(path), "feature_source": config}
    atomic_json(marker, result); return result


def run_feature_v2_children(*, stage: str = "inner", child_timeout_seconds: int = 3600) -> dict[str, Any]:
    """One visible child per feature shard; parent retains no feature matrices."""
    questions, _answers, _folds, _fold_for, _labels = canonical_inner(); qids = sorted(questions); total_shards = expected_inner_shards(qids)
    source_dir, directory = score_v2_directory(stage), feature_v2_directory(stage); directory.mkdir(parents=True, exist_ok=True)
    durations: deque[float] = deque(maxlen=8); completed = 0
    with Run("feature-v2-orchestrator", total_shards) as run:
        for shard in range(total_shards):
            selected = qids[shard * 64:(shard + 1) * 64]; source_path, source_marker = shard_paths(source_dir, shard); path, marker = shard_paths(directory, shard)
            if not source_path.exists() or not source_marker.exists(): raise GateRejected("REJECTED_V2_SOURCE_COMPLETENESS_GATE")
            config = {"schema": "sparse_features_v2_v1", "source_sha256": sha256(source_path), "source_marker_sha256": sha256(source_marker), "fields": "sparse_feature_rows_from_sources_v1"}
            if path.exists() and marker.exists() and verified_shard(path, marker, selected, fingerprint(config)):
                completed = shard + 1; run.heartbeat(completed, shard=shard, skipped_verified=True); continue
            started = time.monotonic(); command = [sys.executable, "-u", str(Path(__file__)), "build-feature-v2-shard", "--shard", str(shard)]
            try:
                child = subprocess.run(command, cwd=ROOT, timeout=child_timeout_seconds, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8")
            except subprocess.TimeoutExpired as exc:
                run.status("FAILED", completed, failure="CHILD_TIMEOUT", shard=shard, timeout_seconds=child_timeout_seconds); raise GateRejected("REJECTED_FEATURE_CHILD_TIMEOUT_GATE") from exc
            elapsed = time.monotonic() - started; durations.append(elapsed)
            if child.returncode != 0 or not verified_shard(path, marker, selected, fingerprint(config)):
                run.status("FAILED", completed, failure="CHILD_EXIT_OR_HASH", shard=shard, returncode=child.returncode, stderr_tail=child.stderr[-2000:]); raise GateRejected("REJECTED_FEATURE_CHILD_EXIT_GATE")
            completed = shard + 1; rolling = float(statistics.median(durations))
            run.heartbeat(completed, shard=shard, child_elapsed_seconds=elapsed, rolling_median_shard_seconds=rolling, rolling_eta_seconds=rolling * (total_shards - completed), parent_rss_after_child=rss_bytes(), available_system_ram_bytes=system_available_ram_bytes())
    result = {"status": "PASS_FEATURE_V2_ORCHESTRATION", "stage": stage, "total_shards": total_shards, "completed_shards": completed, "fold0_called": False}
    atomic_json(directory / "orchestrator_manifest.json", result); return result


def iter_feature_v2_shards(stage: str = "inner") -> Iterator[tuple[int, Path, dict[str, Any]]]:
    directory = feature_v2_directory(stage)
    for path in sorted(directory.glob("scores-*.jsonl")):
        marker = path.with_suffix(".json"); saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("sha256") != sha256(path): raise GateRejected("REJECTED_FEATURE_V2_SHARD_HASH_GATE")
        yield int(path.stem.split("-")[-1]), path, saved


def crossfit_sparse_lambdamart(scores: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], questions: Mapping[str, str], answers: Mapping[str, set[str]], folds: Mapping[str, Sequence[str]]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise GateRejected("REJECTED_LAMBDAMART_DEPENDENCY_GATE") from exc
    predictions: dict[str, list[str]] = {}; reports = []; feature_order: list[str] | None = None
    config = {"objective": "lambdarank", "num_leaves": 7, "min_data_in_leaf": 50, "learning_rate": .05, "num_boost_round": 300, "feature_fraction": 1.0, "bagging_fraction": 1.0, "deterministic": True, "seed": 111, "eval_at": [5], "source": "EXP-109B majority winner"}
    for heldout in INNER_FOLDS:
        train_qids = [qid for fold in INNER_FOLDS if fold != heldout for qid in folds[fold] if answers[qid]]
        x: list[list[float]] = []; y: list[int] = []; groups: list[int] = []
        training_positive_pairs = 0; training_without_gold = 0
        for qid in train_qids:
            docs, rows, names = sparse_feature_rows(scores, questions, qid)
            if feature_order is None: feature_order = names
            if names != feature_order: raise GateRejected("REJECTED_FEATURE_ORDER_GATE")
            positives = [int(doc in answers[qid]) for doc in docs]
            training_positive_pairs += sum(positives); training_without_gold += int(not any(positives))
            x.extend(rows); y.extend(positives); groups.append(len(docs))
        model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", ndcg_at=[5], num_leaves=7, min_child_samples=50, learning_rate=.05, n_estimators=300, feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, deterministic=True, random_state=111, verbosity=-1)
        model.fit(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32), group=groups, feature_name=feature_order)
        heldout_positive_pairs = 0; heldout_without_gold = 0
        for qid in folds[heldout]:
            docs, rows, names = sparse_feature_rows(scores, questions, qid)
            if names != feature_order: raise GateRejected("REJECTED_FEATURE_ORDER_GATE")
            positives = sum(doc in answers[qid] for doc in docs)
            heldout_positive_pairs += positives; heldout_without_gold += int(bool(answers[qid]) and positives == 0)
            values = model.predict(np.asarray(rows, dtype=np.float32)) if rows else np.asarray([])
            predictions[qid] = [doc for _score, doc in sorted(zip(values.tolist(), docs), key=lambda item: (-item[0], item[1]))]
        reports.append({"validation_fold": heldout, "training_folds": [fold for fold in INNER_FOLDS if fold != heldout], "training_qids": len(train_qids), "feature_count": len(feature_order or ()), "candidate_positive_coverage": {"training_positive_qid_gold_pairs": training_positive_pairs, "training_queries_without_gold_candidate": training_without_gold, "heldout_positive_qid_gold_pairs": heldout_positive_pairs, "heldout_queries_without_gold_candidate": heldout_without_gold}, "config": config})
    if any("doc_id" in name or "label" in name or "fold" in name for name in feature_order or ()): raise GateRejected("REJECTED_FEATURE_LEAKAGE_GATE")
    return predictions, {"status": "PASS_CROSSFIT_LAMBDAMART", "config": config, "feature_order": feature_order, "folds": reports, "candidate_contract": "union_top50_per_active_sparse_family"}


def frozen_inner_sparse(resume: bool) -> dict[str, Any]:
    # Do not accidentally revive the legacy all-shards accumulator.  This
    # command is reopened only after the disk-backed feature-shard reader.
    raise GateRejected("REJECTED_DISK_BACKED_LAMBDAMART_IMPLEMENTATION_GATE")
    questions, answers, folds, fold_for, labels = canonical_inner(); qids = sorted(questions)
    scores = source_score_cache(qids=qids, questions=questions, resume=resume, stage="inner")
    rankings = {view: {qid: ranked_docs(rows) for qid, rows in source.items()} for view, source in scores.items()}
    predictions: dict[str, list[str]] = {}; selected: dict[str, Any] = {}
    grid = (5, 10, 20, 40)
    weights_grid = tuple(family_weight_grid())
    for heldout in INNER_FOLDS:
        train = {qid: answers[qid] for fold in INNER_FOLDS if fold != heldout for qid in folds[fold]}
        choices = []
        for k in grid:
            for weights in weights_grid:
                candidate = {qid: hierarchical_rrf({view: ranks[qid][:100] for view, ranks in rankings.items()}, k, weights) for qid in train}
                values = metric(candidate, train)
                choices.append((values, k, weights))
        best_values, best_k, best_weights = max(choices, key=lambda item: (item[0]["recall@5"], item[0]["multi_gold_recall@5"], item[0]["precision@5"], item[0]["mrr@5"], -sum(value > 0 for value in item[2].values()), -item[1]))
        selected[heldout] = {"rrf_k": best_k, "family_weights": best_weights, "selection_metrics": best_values, "selection_folds": [fold for fold in INNER_FOLDS if fold != heldout]}
        for qid in folds[heldout]: predictions[qid] = hierarchical_rrf({view: ranks[qid] for view, ranks in rankings.items()}, best_k, best_weights)
    lgbm_predictions, lgbm_report = crossfit_sparse_lambdamart(scores, questions, answers, folds)
    rrf_metric = metric(predictions, answers); lgbm_metric = metric(lgbm_predictions, answers)
    winner_name, final_predictions, final = ("sparse_lambdamart", lgbm_predictions, lgbm_metric) if lgbm_metric["recall@5"] > rrf_metric["recall@5"] else ("hierarchical_weighted_rrf", predictions, rrf_metric)
    baseline = rankings["v0_control"]
    base = metric(baseline, answers); delta = bootstrap_delta(baseline, final_predictions, answers)
    fold_delta = {fold: metric({qid: final_predictions[qid] for qid in folds[fold]}, {qid: answers[qid] for qid in folds[fold]})["recall@5"] - metric({qid: baseline[qid] for qid in folds[fold]}, {qid: answers[qid] for qid in folds[fold]})["recall@5"] for fold in INNER_FOLDS}
    checks = {"recall_ge_090": final["recall@5"] >= .9, "delta_ge_040": delta["mean"] >= .04, "three_folds_positive": sum(value > 0 for value in fold_delta.values()) >= 3, "worst_ge_minus_002": min(fold_delta.values()) >= -.002, "bootstrap_lower_gt_zero": delta["lower"] > 0, "multi_non_decrease": final["multi_gold_recall@5"] >= base["multi_gold_recall@5"], "quality_non_decrease": final["precision@5"] >= base["precision@5"] - .002 and final["mrr@5"] >= base["mrr@5"] - .002}
    status = "PROMOTE_SPARSE_WINNER" if all(checks.values()) else ("KEEP_COMPLEMENT_ONLY" if delta["mean"] > 0 else "REJECTED_SPARSE_PROMOTION_GATE")
    result = {"schema_version": SCHEMA, "status": status, "scope": "strict_crossfit_folds_1_to_4_only", "fold0_labels_deserialized": False, "labels": labels, "winner": winner_name, "selections": selected, "metrics": final, "rrf_metrics": rrf_metric, "lambdamart_metrics": lgbm_metric, "lambdamart": lgbm_report, "baseline_v0_metrics": base, "bootstrap_delta_recall@5": delta, "per_fold_delta_recall@5": fold_delta, "checks": checks}
    atomic_json(RESULTS / "FROZEN_INNER_SPARSE_REPORT.json", result)
    if status == "REJECTED_SPARSE_PROMOTION_GATE": raise GateRejected(status)
    write_success(RESULTS, "frozen-inner-sparse", result); return result


def dense_complement(resume: bool) -> dict[str, Any]:
    """Fail closed unless the immutable EXP-109B per-source F1--F4 cache exists."""
    del resume
    questions, answers, _folds, _fold_for, labels = canonical_inner()
    anchor = ROOT / "results" / "exp109b_encoder_complementarity" / "anchor_inner_predictions.jsonl"
    anchor_manifest = anchor.with_suffix(".manifest.json")
    if not anchor.exists() or not anchor_manifest.exists(): raise GateRejected("REJECTED_DENSE_ANCHOR_INPUT_GATE")
    manifest = json.loads(anchor_manifest.read_text(encoding="utf-8")); rows = list(read_jsonl(anchor))
    if any(row.get("fold0_included") for row in rows): raise GateRejected("REJECTED_FOLD0_LEAKAGE_GATE: anchor includes Fold0")
    pred = {str(row["qid"]): [str(value["doc_id"]) for value in row["prediction"]] for row in rows}
    replay = metric(pred, answers)
    expected = 0.9251057869956494
    anchor_ok = len(pred) == len(questions) and abs(replay["recall@5"] - expected) <= 1e-12 and manifest.get("fold0_read") is False
    source_candidates = [ROOT / "cache" / "exp109b_encoder_complementarity" / "source_rankings_inner.jsonl", ROOT / "results" / "exp109b_encoder_complementarity" / "sources_top50_inner.jsonl"]
    source_cache = next((path for path in source_candidates if path.exists()), None)
    result = {"schema_version": SCHEMA, "stage": "dense-complement", "scope": "folds_1_to_4_only", "fold0_labels_deserialized": False, "labels": labels,
        "anchor_manifest_sha256": sha256(anchor_manifest), "anchor_predictions_sha256": sha256(anchor), "anchor_replay_metrics": replay, "anchor_exact": anchor_ok,
        "required_source_cache": [str(path) for path in source_candidates], "source_cache_found": str(source_cache) if source_cache else None,
        "status": "PASS_DENSE_INPUT_GATE" if anchor_ok and source_cache else "REJECTED_DENSE_SOURCE_CACHE_GATE"}
    atomic_json(RESULTS / "DENSE_COMPLEMENT_REPORT.json", result)
    if not anchor_ok: raise GateRejected("REJECTED_DENSE_ANCHOR_REPRODUCTION_GATE")
    if source_cache is None: raise GateRejected(result["status"])
    raise GateRejected("DEFERRED_DENSE_COMPARISON_NOT_IMPLEMENTED: exact E5/LAL source-cache schema must be audited before B/C candidate construction")


def overnight_inner(resume: bool) -> dict[str, Any]:
    """Sequential inner-only orchestration. It never invokes Fold 0."""
    bounded = json.loads((RESULTS / "BOUNDED_SOURCE_SCREEN.json").read_text(encoding="utf-8")) if (RESULTS / "BOUNDED_SOURCE_SCREEN.json").exists() else None
    if not bounded or bounded.get("status") != "PASS_BOUNDED_SPARSE_COMPLEMENTARITY_GATE": raise GateRejected("REJECTED_BOUNDED_SPARSE_COMPLEMENTARITY_GATE")
    full = full_source_audit(resume); frozen = frozen_inner_sparse(resume)
    dense = dense_complement(resume)
    return {"status": "PASS_OVERNIGHT_INNER", "full_source": full.get("status"), "frozen": frozen.get("status"), "dense": dense.get("status"), "fold0_called": False}


def unsupported(stage: str) -> dict[str, Any]:
    raise GateRejected(f"{stage} is fail-closed until bounded gate and cached-source manifests pass; Fold-0 remains unauthorized")


def status() -> dict[str, Any]:
    return json.loads((RESULTS / "RUN_STATUS.json").read_text(encoding="utf-8")) if (RESULTS / "RUN_STATUS.json").exists() else {"state": "NOT_STARTED"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "reproduce", "lexical-audit", "build-index", "bounded-screen", "full-source-audit", "frozen-inner-sparse", "dense-complement", "overnight-inner", "fold0", "status", "mark-interrupted", "migrate-score-shard", "score-v2-shard", "score-v2-orchestrate", "build-feature-v2-shard", "feature-v2-orchestrate", "parity-old-v2", "parity-streaming-metrics"))
    parser.add_argument("--view", choices=("v0_control", *VIEWS))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-fold0", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.stage == "audit": result = input_audit()
        elif args.stage == "reproduce": result = run_reproduction(args.resume)
        elif args.stage == "lexical-audit": result = lexical_audit(args.resume)
        elif args.stage == "build-index": result = build_index(args.view, args.resume)
        elif args.stage == "bounded-screen": result = bounded_screen(args.resume)
        elif args.stage == "full-source-audit": result = full_source_audit(args.resume)
        elif args.stage == "frozen-inner-sparse": result = frozen_inner_sparse(args.resume)
        elif args.stage == "dense-complement": result = dense_complement(args.resume)
        elif args.stage == "mark-interrupted": result = mark_interrupted_resource_unbounded()
        elif args.stage == "migrate-score-shard":
            if args.shard is None: raise GateRejected("REJECTED_SOURCE_SHARD_RANGE_GATE")
            result = migrate_old_score_shard(args.shard)
        elif args.stage == "score-v2-shard":
            if args.shard is None: raise GateRejected("REJECTED_SOURCE_SHARD_RANGE_GATE")
            result = score_v2_shard(args.shard)
        elif args.stage == "score-v2-orchestrate": result = run_score_v2_children(start_shard=args.shard if args.shard is not None else 24)
        elif args.stage == "build-feature-v2-shard":
            if args.shard is None: raise GateRejected("REJECTED_SOURCE_SHARD_RANGE_GATE")
            result = build_feature_v2_shard(args.shard)
        elif args.stage == "feature-v2-orchestrate": result = run_feature_v2_children()
        elif args.stage == "parity-old-v2": result = parity_old_v2()
        elif args.stage == "parity-streaming-metrics": result = parity_old_v2_streaming_metrics()
        elif args.stage == "overnight-inner": result = overnight_inner(args.resume)
        elif args.stage == "status": result = status()
        elif args.stage == "fold0":
            if not args.authorize_fold0 or os.getenv("EXP111_ALLOW_FOLD0") != "1": raise GateRejected("REJECTED_AUTHORIZATION_GATE")
            result = unsupported("fold0")
        else: result = unsupported(args.stage)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True); return 0
    except GateRejected as error:
        atomic_json(RESULTS / "RUN_STATUS.json", {"schema_version": SCHEMA, "state": "REJECTED", "stage": args.stage, "error": str(error), "last_heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(str(error), file=sys.stderr, flush=True); return 2


if __name__ == "__main__":
    raise SystemExit(main())
