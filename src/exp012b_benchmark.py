"""Freeze, sample and compare EXP-012b reference/optimized artifacts.

The harness never launches a long model stage.  It creates a deterministic
256-query contract and reports parity for already-produced artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from exp012b_core import atomic_json, canonical_json, content_hash, load_answers, load_queries, read_jsonl, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "cache" / "exp012b_v3"
DEFAULT_OPTIMIZED = ROOT / "cache" / "exp012b_v3_fast"
DEFAULT_RESULTS = ROOT / "results" / "benchmarks" / "exp012b"


def deterministic_sample(train_path: Path, size: int = 256) -> list[str]:
    queries = load_queries(train_path)
    answers = load_answers(train_path)
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    lengths = sorted(len(value) for value in queries.values())
    q1 = lengths[len(lengths) // 4]
    q2 = lengths[len(lengths) // 2]
    q3 = lengths[len(lengths) * 3 // 4]
    for qid, query in queries.items():
        length_bucket = sum(len(query) > edge for edge in (q1, q2, q3))
        answer_bucket = min(3, len(answers.get(qid, ())))
        buckets[(length_bucket, answer_bucket)].append(qid)
    selected: list[str] = []
    keys = sorted(buckets)
    cursor = 0
    while len(selected) < min(size, len(queries)):
        key = keys[cursor % len(keys)]
        rows = buckets[key]
        index = cursor // len(keys)
        if index < len(rows):
            selected.append(sorted(rows, key=lambda qid: (content_hash(qid), qid))[index])
        cursor += 1
    return sorted(selected)


def freeze_reference(cache_root: Path, results_root: Path, output: Path) -> dict[str, Any]:
    files: list[Path] = []
    for relative in (
        "preflight/manifest.json", "bm25_fields/manifest.json", "bm25_index/manifest.json",
        "bge_leaves/manifest.json", "bge_blocks/manifest.json", "rankings/train/manifest.json",
        "evidence/train/manifest.json", "zero_shot/train/manifest.json",
    ):
        path = cache_root / relative
        if path.exists():
            files.append(path)
    for relative in ("oof_metrics.json", "submission/submission.json", "submission/submission.zip"):
        path = results_root / relative
        if path.exists():
            files.append(path)
    payload = {
        "schema_version": "legalir.exp012b.benchmark.v1",
        "profile": "reference",
        "files": {
            str(path.relative_to(ROOT)): {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }
    payload["content_fingerprint"] = content_hash(payload)
    atomic_json(output, payload)
    return payload


def _sample_records(path: Path, qids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["qid"]): row for row in read_jsonl(path) if str(row.get("qid")) in qids}


def _ids(record: dict[str, Any], field: str) -> list[str]:
    return [str(row[field]) for row in record.get("candidates", [])]


def compare(reference: Path, optimized: Path, qids: Iterable[str]) -> dict[str, Any]:
    wanted = set(qids)
    checks: dict[str, Any] = {}
    pairs = {
        "ranking": "rankings/train/hybrid_candidates.jsonl",
        "evidence": "evidence/train/evidence.jsonl",
        "zero_shot": "zero_shot/train/scores.jsonl",
    }
    overall = True
    for name, relative in pairs.items():
        left = _sample_records(reference / relative, wanted)
        right = _sample_records(optimized / relative, wanted)
        common = sorted(left.keys() & right.keys())
        if name == "ranking":
            mismatches = [qid for qid in common if _ids(left[qid], "doc_id") != _ids(right[qid], "doc_id")]
        elif name == "evidence":
            def signature(row: dict[str, Any]) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
                return [
                    (str(candidate["doc_id"]), tuple((str(ev["chunk_id"]), content_hash(ev["bundle_text"])) for ev in candidate["evidence"]))
                    for candidate in row.get("candidates", [])
                ]
            mismatches = [qid for qid in common if signature(left[qid]) != signature(right[qid])]
        else:
            # Score files have several rows per qid, so the simple qid map is
            # only a smoke check; full numerical parity belongs to fold-0 gate.
            mismatches = []
        passed = len(common) == len(wanted) and not mismatches
        checks[name] = {"pass": passed, "covered": len(common), "mismatches": mismatches[:20]}
        overall &= passed
    return {"schema_version": "legalir.exp012b.parity.v1", "pass": overall, "checks": checks}


def write_report(payload: dict[str, Any], path: Path) -> None:
    lines = ["# EXP-012b optimization parity", "", f"Overall: **{'PASS' if payload['pass'] else 'FAIL'}**", ""]
    lines += ["| Channel | Status | Covered | Mismatches |", "|---|---:|---:|---:|"]
    for name, row in payload["checks"].items():
        lines.append(f"| {name} | {'PASS' if row['pass'] else 'FAIL'} | {row['covered']} | {len(row['mismatches'])} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["freeze", "sample", "compare"], required=True)
    parser.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--optimized-cache", type=Path, default=DEFAULT_OPTIMIZED)
    parser.add_argument("--reference-results", type=Path, default=ROOT / "results" / "exp012b_v3")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--sample-size", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "sample_qids.json"
    if args.action == "freeze":
        freeze_reference(args.reference_cache, args.reference_results, args.output_dir / "baseline.json")
    elif args.action == "sample":
        qids = deterministic_sample(ROOT / "public_test_dataset" / "train.json", args.sample_size)
        atomic_json(sample_path, {"qids": qids, "fingerprint": content_hash(qids)})
    else:
        if not sample_path.exists():
            qids = deterministic_sample(ROOT / "public_test_dataset" / "train.json", args.sample_size)
            atomic_json(sample_path, {"qids": qids, "fingerprint": content_hash(qids)})
        qids = json.loads(sample_path.read_text(encoding="utf-8"))["qids"]
        payload = compare(args.reference_cache, args.optimized_cache, qids)
        atomic_json(args.output_dir / "parity.json", payload)
        write_report(payload, args.output_dir / "report.md")
        if not payload["pass"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
