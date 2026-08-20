"""Shared deterministic infrastructure for the v3-native EXP-012b pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


PIPELINE_SCHEMA = "legalir.exp012b_v3.v1"
SOURCE_RANKING_SCHEMA = "legalir.exp012b_v3.ranking.v1"
EVIDENCE_SCHEMA = "legalir.exp012b_v3.evidence.v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(canonical_json(record) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return count


def atomic_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            if pretty:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
            else:
                handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _artifact_receipts(cache_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = cache_dir / name
        stat = path.stat()
        receipts[name] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": expected,
        }
    return receipts


def load_v3_manifest(
    cache_dir: Path,
    *,
    full_verify: bool | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    with (cache_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "legalir.structural_chunks.v3":
        raise ValueError(f"Not a structural-v3 cache: {cache_dir}")
    if full_verify is None:
        full_verify = os.environ.get("LEGALIR_EXECUTION_PROFILE") != "optimized"
    if receipt_path is None:
        configured = os.environ.get("LEGALIR_V3_RECEIPT_PATH")
        receipt_path = Path(configured) if configured else None
    receipt_payload: dict[str, Any] = {}
    if receipt_path is not None and receipt_path.exists():
        receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    trusted = receipt_payload.get("integrity_receipts", {})
    if receipt_payload.get("v3_fingerprint") != manifest.get("content_fingerprint"):
        trusted = {}
    verified: dict[str, dict[str, Any]] = {}
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = cache_dir / name
        stat = path.stat()
        receipt = trusted.get(name, {})
        stat_matches = (
            receipt.get("path") == str(path.resolve())
            and int(receipt.get("size", -1)) == stat.st_size
            and int(receipt.get("mtime_ns", -1)) == stat.st_mtime_ns
            and receipt.get("sha256") == expected
        )
        actual = expected if full_verify is False and stat_matches else sha256_file(path)
        if actual != expected:
            raise ValueError(f"Stale/corrupt v3 artifact {name}: {actual} != {expected}")
        verified[name] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": actual,
        }
    manifest["integrity_receipts"] = verified
    return manifest


def artifact_manifest(
    *, stage: str, inputs: dict[str, Any], config: dict[str, Any], files: Iterable[Path]
) -> dict[str, Any]:
    file_hashes = {path.name: sha256_file(path) for path in files}
    fingerprint = content_hash({"inputs": inputs, "config": config, "files": file_hashes})
    return {
        "schema_version": PIPELINE_SCHEMA,
        "stage": stage,
        "inputs": inputs,
        "config": config,
        "artifact_sha256": file_hashes,
        "content_fingerprint": fingerprint,
    }


def require_success(stage_dir: Path, expected_v3_fingerprint: str | None = None) -> dict[str, Any]:
    success_path = stage_dir / "_SUCCESS.json"
    if not success_path.exists():
        raise RuntimeError(f"Upstream stage incomplete: {success_path}")
    with success_path.open("r", encoding="utf-8") as handle:
        marker = json.load(handle)
    if expected_v3_fingerprint and marker.get("v3_fingerprint") != expected_v3_fingerprint:
        raise RuntimeError(f"Upstream stage has stale v3 fingerprint: {stage_dir}")
    return marker


class StageLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.log_path = run_dir / "run.log"
        self.status_path = run_dir / "status.json"
        self.telemetry: dict[str, Any] = {}

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8", newline="\n", buffering=1) as handle:
            handle.write(line + "\n")

    def status(self, **values: Any) -> None:
        atomic_json(
            self.status_path,
            {"schema_version": PIPELINE_SCHEMA, "pid": os.getpid(), **values},
        )

    def set_telemetry(self, telemetry: dict[str, Any]) -> None:
        self.telemetry = telemetry


@contextmanager
def stage_run(
    run_dir: Path,
    stage: str,
    *,
    total: int | None = None,
    v3_fingerprint: str | None = None,
) -> Iterator[StageLogger]:
    # A failed re-run must never leave a stale success marker from an older
    # configuration behind.  Downstream stages use this file as a hard gate.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "_SUCCESS.json").unlink(missing_ok=True)
    logger = StageLogger(run_dir)
    started = time.time()
    logger.status(stage=stage, state="RUNNING", completed=0, total=total, started_at=started)
    logger.log(f"START stage={stage} total={total}")
    try:
        yield logger
    except BaseException as error:
        logger.status(
            stage=stage,
            state="FAILED",
            completed=None,
            total=total,
            started_at=started,
            finished_at=time.time(),
            error=f"{type(error).__name__}: {error}",
        )
        logger.log(f"FAILED stage={stage}: {type(error).__name__}: {error}")
        raise
    else:
        finished = time.time()
        logger.status(
            stage=stage,
            state="COMPLETE",
            completed=total,
            total=total,
            started_at=started,
            finished_at=finished,
        )
        atomic_json(
            run_dir / "_SUCCESS.json",
            {
                "schema_version": PIPELINE_SCHEMA,
                "stage": stage,
                "v3_fingerprint": v3_fingerprint,
                "finished_at": finished,
                "telemetry": logger.telemetry,
            },
        )
        logger.log(f"COMPLETE stage={stage} elapsed_seconds={finished-started:.1f}")


def load_queries(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result: dict[str, str] = {}
    for qid, record in payload.items():
        query = str(record.get("question", "")).strip()
        if not query:
            raise ValueError(f"Missing question for qid={qid}")
        result[str(qid)] = query
    return result


def load_answers(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(qid): {str(doc) for doc in row.get("answer", [])} for qid, row in payload.items()}
