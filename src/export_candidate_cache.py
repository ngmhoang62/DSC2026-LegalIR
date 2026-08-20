"""Validate source rankings and export deterministic EXP-012b candidate caches.

This module is intentionally model-free. Retrieval programs write the source
ranking schema documented by ``SOURCE_SCHEMA_VERSION``; this exporter performs
validation, RRF fusion, evidence routing, fingerprinting, and the audit gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from structural_chunker_v3 import canonical_json, sha256_file


SOURCE_SCHEMA_VERSION = "legalir.source_ranking.v1"
CANDIDATE_SCHEMA_VERSION = "legalir.candidates.v1"
SOURCE_ORDER = ("bge", "e5", "bm25")


class RankingValidationError(ValueError):
    pass


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RankingValidationError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc


def load_chunk_references(cache_dir: Path) -> tuple[set[str], dict[str, str]]:
    document_ids = {
        str(record["doc_id"]) for record in read_jsonl(cache_dir / "documents.jsonl")
    }
    chunk_to_doc: dict[str, str] = {}
    for chunk in read_jsonl(cache_dir / "chunks.jsonl"):
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in chunk_to_doc:
            raise RankingValidationError(f"Duplicate chunk ID in v3 cache: {chunk_id}")
        chunk_to_doc[chunk_id] = str(chunk["doc_id"])
    return document_ids, chunk_to_doc


def _validate_ranked_items(
    items: Any,
    *,
    item_type: str,
    qid: str,
    document_ids: set[str],
    chunk_to_doc: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise RankingValidationError(f"{qid}: {item_type} must be a list")
    ranks: set[int] = set()
    identities: set[str] = set()
    validated: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise RankingValidationError(f"{qid}: malformed {item_type} item")
        try:
            rank = int(item["rank"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RankingValidationError(
                f"{qid}: {item_type} item requires numeric rank and score"
            ) from exc
        if rank < 1 or rank in ranks:
            raise RankingValidationError(f"{qid}: duplicate or non-positive {item_type} rank {rank}")
        ranks.add(rank)
        doc_id = str(item.get("doc_id", ""))
        if doc_id not in document_ids:
            raise RankingValidationError(f"{qid}: unknown document {doc_id}")
        if item_type == "documents":
            identity = doc_id
            normalized = {"doc_id": doc_id, "rank": rank, "score": score}
        else:
            chunk_id = str(item.get("chunk_id", ""))
            if chunk_id not in chunk_to_doc:
                raise RankingValidationError(f"{qid}: unknown chunk {chunk_id}")
            if chunk_to_doc[chunk_id] != doc_id:
                raise RankingValidationError(
                    f"{qid}: chunk {chunk_id} belongs to {chunk_to_doc[chunk_id]}, not {doc_id}"
                )
            identity = chunk_id
            normalized = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "rank": rank,
                "score": score,
            }
        if identity in identities:
            raise RankingValidationError(f"{qid}: duplicate {item_type} identity {identity}")
        identities.add(identity)
        validated.append(normalized)
    validated.sort(key=lambda item: (item["rank"], item.get("chunk_id", item["doc_id"])))
    return validated


def load_source_ranking(
    path: Path,
    expected_source: str,
    document_ids: set[str],
    chunk_to_doc: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], str]:
    records: dict[str, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    for record in read_jsonl(path):
        if record.get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise RankingValidationError(
                f"{path}: expected schema_version={SOURCE_SCHEMA_VERSION}"
            )
        source = str(record.get("source", "")).casefold()
        if source != expected_source:
            raise RankingValidationError(
                f"{path}: source={source!r}, expected {expected_source!r}"
            )
        fingerprint = str(record.get("source_fingerprint", ""))
        if not fingerprint:
            raise RankingValidationError(f"{path}: missing source_fingerprint")
        fingerprints.add(fingerprint)
        qid = str(record.get("qid", ""))
        query = str(record.get("query", ""))
        if not qid or not query:
            raise RankingValidationError(f"{path}: qid and query are required")
        if qid in records:
            raise RankingValidationError(f"{path}: duplicate qid {qid}")
        documents = _validate_ranked_items(
            record.get("documents"),
            item_type="documents",
            qid=qid,
            document_ids=document_ids,
            chunk_to_doc=chunk_to_doc,
        )
        chunks = _validate_ranked_items(
            record.get("chunks", []),
            item_type="chunks",
            qid=qid,
            document_ids=document_ids,
            chunk_to_doc=chunk_to_doc,
        )
        records[qid] = {
            "qid": qid,
            "query": query,
            "documents": documents,
            "chunks": chunks,
        }
    if not records:
        raise RankingValidationError(f"{path}: no ranking records")
    if len(fingerprints) != 1:
        raise RankingValidationError(f"{path}: inconsistent source_fingerprint values")
    return records, next(iter(fingerprints))


def _select_evidence(
    doc_id: str,
    source_records: dict[str, dict[str, Any]],
    evidence_per_doc: int,
) -> list[dict[str, Any]]:
    per_source: dict[str, list[dict[str, Any]]] = {}
    occurrence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in SOURCE_ORDER:
        ranked = [
            item for item in source_records[source]["chunks"] if item["doc_id"] == doc_id
        ]
        per_source[source] = ranked
        for item in ranked:
            occurrence[item["chunk_id"]].append(
                {"source": source, "rank": item["rank"], "score": item["score"]}
            )

    selected: list[str] = []
    selected_set: set[str] = set()
    # First pass preserves the required semantic/lexical diversity.
    for source in SOURCE_ORDER:
        for item in per_source[source]:
            if item["chunk_id"] not in selected_set:
                selected.append(item["chunk_id"])
                selected_set.add(item["chunk_id"])
                break
        if len(selected) >= evidence_per_doc:
            break

    # Fill missing slots with the next-best unique evidence across any source.
    combined = sorted(
        (
            item["rank"],
            SOURCE_ORDER.index(source),
            item["chunk_id"],
        )
        for source in SOURCE_ORDER
        for item in per_source[source]
    )
    for _, _, chunk_id in combined:
        if len(selected) >= evidence_per_doc:
            break
        if chunk_id not in selected_set:
            selected.append(chunk_id)
            selected_set.add(chunk_id)

    return [
        {
            "chunk_id": chunk_id,
            "provenance": sorted(
                occurrence[chunk_id],
                key=lambda item: (SOURCE_ORDER.index(item["source"]), item["rank"]),
            ),
        }
        for chunk_id in selected
    ]


def fuse_query(
    qid: str,
    source_records: dict[str, dict[str, Any]],
    *,
    rrf_k: int = 60,
    top_docs: int = 50,
    evidence_per_doc: int = 3,
) -> dict[str, Any]:
    query_values = {source_records[source]["query"] for source in SOURCE_ORDER}
    if len(query_values) != 1:
        raise RankingValidationError(f"{qid}: query text differs between ranking sources")
    rrf_scores: dict[str, float] = defaultdict(float)
    best_ranks: dict[str, int] = {}
    source_doc_items: dict[str, dict[str, dict[str, Any]]] = {}
    for source in SOURCE_ORDER:
        by_doc = {item["doc_id"]: item for item in source_records[source]["documents"]}
        source_doc_items[source] = by_doc
        for doc_id, item in by_doc.items():
            rrf_scores[doc_id] += 1.0 / (rrf_k + item["rank"])
            best_ranks[doc_id] = min(best_ranks.get(doc_id, item["rank"]), item["rank"])
    ordered = sorted(
        rrf_scores,
        key=lambda doc_id: (-rrf_scores[doc_id], best_ranks[doc_id], doc_id),
    )[:top_docs]

    candidates = []
    for fused_rank, doc_id in enumerate(ordered, start=1):
        per_source = {
            source: {
                "rank": source_doc_items[source][doc_id]["rank"],
                "score": source_doc_items[source][doc_id]["score"],
            }
            for source in SOURCE_ORDER
            if doc_id in source_doc_items[source]
        }
        candidates.append(
            {
                "doc_id": doc_id,
                "fused_rank": fused_rank,
                "rrf_score": rrf_scores[doc_id],
                "best_source_rank": best_ranks[doc_id],
                "sources": per_source,
                "evidence": _select_evidence(doc_id, source_records, evidence_per_doc),
            }
        )
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "qid": qid,
        "query": next(iter(query_values)),
        "candidates": candidates,
    }


def _candidate_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def export_candidates(
    *,
    chunk_cache_dir: Path,
    audit_report_path: Path,
    ranking_paths: dict[str, Path],
    output_dir: Path,
    split: str,
    rrf_k: int = 60,
    top_docs: int = 50,
    evidence_per_doc: int = 3,
    validate_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    with audit_report_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit.get("status") != "PASS":
        raise RankingValidationError(
            f"Candidate export blocked: structural audit status is {audit.get('status')!r}"
        )
    with (chunk_cache_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        chunk_manifest = json.load(handle)
    if audit.get("content_fingerprint") != chunk_manifest.get("content_fingerprint"):
        raise RankingValidationError("Audit report does not match the structural cache fingerprint")

    document_ids, chunk_to_doc = load_chunk_references(chunk_cache_dir)
    rankings: dict[str, dict[str, dict[str, Any]]] = {}
    source_fingerprints: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for source in SOURCE_ORDER:
        path = ranking_paths[source]
        rankings[source], source_fingerprints[source] = load_source_ranking(
            path, source, document_ids, chunk_to_doc
        )
        input_hashes[source] = sha256_file(path)

    qid_sets = {source: set(records) for source, records in rankings.items()}
    reference_qids = qid_sets[SOURCE_ORDER[0]]
    for source in SOURCE_ORDER[1:]:
        if qid_sets[source] != reference_qids:
            raise RankingValidationError(
                f"Query set mismatch for {source}: "
                f"missing={sorted(reference_qids - qid_sets[source])[:20]}, "
                f"extra={sorted(qid_sets[source] - reference_qids)[:20]}"
            )

    config = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "split": split,
        "rrf_k": rrf_k,
        "top_docs": top_docs,
        "evidence_per_doc": evidence_per_doc,
        "source_order": list(SOURCE_ORDER),
        "tie_break": ["rrf_score_desc", "best_source_rank_asc", "doc_id_asc"],
    }
    fingerprint_inputs = {
        "config": config,
        "chunk_content_fingerprint": chunk_manifest["content_fingerprint"],
        "audit_sha256": sha256_file(audit_report_path),
        "source_fingerprints": source_fingerprints,
        "ranking_file_sha256": input_hashes,
    }
    fingerprint = _candidate_fingerprint(fingerprint_inputs)

    fused_records: list[dict[str, Any]] = []
    pool_sizes: list[int] = []
    evidence_counts: list[int] = []
    for qid in sorted(reference_qids):
        per_source = {source: rankings[source][qid] for source in SOURCE_ORDER}
        fused = fuse_query(
            qid,
            per_source,
            rrf_k=rrf_k,
            top_docs=top_docs,
            evidence_per_doc=evidence_per_doc,
        )
        fused["cache_fingerprint"] = fingerprint
        fused_records.append(fused)
        pool_sizes.append(len(fused["candidates"]))
        evidence_counts.extend(len(candidate["evidence"]) for candidate in fused["candidates"])

    summary = {
        **config,
        "candidate_fingerprint": fingerprint,
        "chunk_content_fingerprint": chunk_manifest["content_fingerprint"],
        "audit_sha256": fingerprint_inputs["audit_sha256"],
        "source_fingerprints": source_fingerprints,
        "ranking_file_sha256": input_hashes,
        "counts": {
            "queries": len(fused_records),
            "candidate_pool_min": min(pool_sizes),
            "candidate_pool_max": max(pool_sizes),
            "candidate_pool_mean": sum(pool_sizes) / len(pool_sizes),
            "evidence_min": min(evidence_counts) if evidence_counts else 0,
            "evidence_max": max(evidence_counts) if evidence_counts else 0,
            "evidence_mean": (
                sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0
            ),
        },
        "validated_only": validate_only,
    }
    if validate_only:
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_candidates.jsonl"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        existing_fingerprint = existing.get("candidate_fingerprint")
        if existing_fingerprint != fingerprint and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_dir}: fingerprint changed from "
                f"{existing_fingerprint} to {fingerprint}; pass --overwrite explicitly"
            )
        if existing_fingerprint == fingerprint and output_path.exists() and not overwrite:
            summary["no_op_existing_cache"] = True
            return summary

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in fused_records:
            handle.write(canonical_json(record) + "\n")
    temporary.replace(output_path)
    summary["validated_only"] = False
    summary["artifact_sha256"] = {output_path.name: sha256_file(output_path)}
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-cache-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--bm25", type=Path, required=True)
    parser.add_argument("--bge", type=Path, required=True)
    parser.add_argument("--e5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "public"), required=True)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-docs", type=int, default=50)
    parser.add_argument("--evidence-per-doc", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = export_candidates(
            chunk_cache_dir=args.chunk_cache_dir,
            audit_report_path=args.audit_report,
            ranking_paths={"bm25": args.bm25, "bge": args.bge, "e5": args.e5},
            output_dir=args.output_dir,
            split=args.split,
            rrf_k=args.rrf_k,
            top_docs=args.top_docs,
            evidence_per_doc=args.evidence_per_doc,
            validate_only=args.validate_only,
            overwrite=args.overwrite,
        )
    except (RankingValidationError, FileExistsError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
