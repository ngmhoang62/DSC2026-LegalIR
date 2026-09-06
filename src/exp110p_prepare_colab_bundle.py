"""Stream a compact, hash-verified EXP-110P bundle for Google Colab.

This exporter never uploads the corpus, corpus embeddings, SQLite BM25 index,
or Jina index.  It reads verified ranking shards sequentially and materializes
only the top-50 rows needed by the specialist.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from exp110p_semantic_label_prototype import (
    FOLD0,
    INNER_FOLDS,
    LABEL_FINGERPRINT,
    LABEL_POLICY,
    SCHEMA,
    atomic_json,
    canonical_json,
    canonical_labels,
    content_hash,
    inner_qids,
    load_folds,
    read_json,
    read_jsonl,
    sha256_file,
    validate_fold_partition,
    write_jsonl_atomic,
)

CODE_FILE = Path(__file__).resolve()
ROOT = CODE_FILE.parents[1]
DEFAULT_TRAIN = ROOT / "public_test_dataset" / "train.json"
DEFAULT_FOLDS = ROOT / "cache" / "cv_folds.json"
DEFAULT_EXCLUSIONS = ROOT / "cache" / "final_preprocessed_v2" / "exclusions.json"
DEFAULT_IMPACT = ROOT / "cache" / "final_preprocessed_v2" / "train_label_impact.jsonl"
DEFAULT_QUERY_DIR = ROOT / "cache" / "exp021_e5_dense_candidates" / "query_embeddings"
DEFAULT_STRUCT_DIR = ROOT / "cache" / "structural_v3_e5_final_v1"
DEFAULT_E5_RANK_DIR = ROOT / "cache" / "exp109b_encoder_complementarity" / "rankings" / "vietlegal_e5" / FOLD0
DEFAULT_LAL_RANK_DIR = ROOT / "cache" / "exp109b_encoder_complementarity" / "rankings" / "vnlegal_lal" / FOLD0
DEFAULT_BM25_EVIDENCE = ROOT / "cache" / "exp021_sparse" / "depth_tune" / "raw4096_evidence"
DEFAULT_BM25_TUNING = ROOT / "results" / "exp021_sparse" / "depth_rrf_tuning" / "tuning_report.json"
DEFAULT_PILOT = ROOT / "results" / "exp109b_encoder_complementarity" / "cached_fusion_pilot" / FOLD0 / "CACHED_FUSION_PILOT.json"
DEFAULT_ANCHOR_CANDIDATES = (
    ROOT / "results" / "exp109b_encoder_complementarity" / "anchor_inner_predictions.jsonl",
    ROOT / "cache" / "exp109b_encoder_complementarity" / "anchor_inner_predictions.jsonl",
)


def _atomic_copy(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with source.open("rb") as src, temporary.open("wb") as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    temporary.replace(destination)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "source": str(source)}


def _verify_success(directory: Path, manifest: Mapping[str, Any]) -> None:
    marker = directory / "_SUCCESS.json"
    if not marker.exists() or read_json(marker).get("status") != "PASS":
        raise RuntimeError(f"missing PASS marker: {directory}")
    expected = manifest.get("content_fingerprint")
    if expected and read_json(marker).get("content_fingerprint") != expected:
        raise RuntimeError(f"manifest/success fingerprint mismatch: {directory}")


def _ranking_rows_top50(directory: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = read_json(directory / "manifest.json")
    _verify_success(directory, manifest)
    if int(manifest.get("limit", 0)) < 500 or int(manifest.get("query_count", 0)) != 7000:
        raise RuntimeError(f"ranking prerequisite is not a complete 7,000-query top-500+ store: {directory}")
    output: dict[str, list[dict[str, Any]]] = {}
    shard_dir = directory / "shards"
    for entry in manifest.get("shards", []):
        path = shard_dir / str(entry["name"])
        if not path.exists() or sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"ranking shard hash mismatch: {path}")
        for row in read_jsonl(path):
            qid = str(row["qid"])
            if qid in output:
                raise RuntimeError(f"duplicate ranking qid: {qid}")
            documents = row.get("documents", row.get("rankings", []))
            output[qid] = [
                {"doc_id": str(item["doc_id"]), "rank": int(item.get("rank", i + 1)), "score": float(item.get("score", 0.0))}
                for i, item in enumerate(documents[:500])
            ]
    if len(output) != 7000:
        raise RuntimeError(f"ranking query count mismatch: {directory}: {len(output)}")
    return output, {"directory": str(directory), "manifest_sha256": sha256_file(directory / "manifest.json"), "content_fingerprint": manifest.get("content_fingerprint"), "model": manifest.get("model"), "query_count": len(output)}


def _bm25_rankings(evidence: Sequence[Sequence[Any]], config: Mapping[str, Any], *, limit: int = 50) -> list[dict[str, Any]]:
    depth = int(config["depth"])
    parent_k = int(config["parent_rrf_k"])
    fusion_k = int(config["fusion_rrf_k"])
    head = int(config["head_cutoff"])
    eligible: list[tuple[str, list[int]]] = []
    for doc_id, ranks in evidence:
        filtered = sorted(int(rank) for rank in ranks if int(rank) <= depth)
        if filtered:
            eligible.append((str(doc_id), filtered))
    first = [doc for doc, _ranks in sorted(eligible, key=lambda item: (item[1][0], item[0]))]
    rrf = [doc for doc, _score, _first in sorted(((doc, sum(1.0 / (parent_k + rank) for rank in ranks), ranks[0]) for doc, ranks in eligible), key=lambda item: (-item[1], item[2], item[0]))]
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, int] = {}
    for ranking in (first, rrf):
        for rank, doc in enumerate(ranking, 1):
            scores[doc] += 1.0 / (fusion_k + rank)
            best[doc] = min(best.get(doc, rank), rank)
    fused = sorted(scores, key=lambda doc: (-scores[doc], best[doc], doc))
    head_docs = fused[:head]
    result = head_docs + [doc for doc in rrf if doc not in set(head_docs)]
    result = result[:limit]
    return [{"doc_id": doc, "rank": rank, "score": float(scores[doc])} for rank, doc in enumerate(result, 1)]


def _bm25_top50(evidence_dir: Path, tuning_path: Path, fold_for: Mapping[str, str], *, limit: int = 500) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    tuning = read_json(tuning_path)
    selected = tuning.get("selected_by_candidate_budget", {}).get("150")
    if not isinstance(selected, dict):
        raise RuntimeError("tuned BM25 report lacks selected_by_candidate_budget[150]")
    output: dict[str, list[dict[str, Any]]] = {}
    manifest = read_json(evidence_dir / "manifest.json")
    for entry in manifest.get("shards", []):
        path = evidence_dir / "shards" / str(entry["name"])
        if not path.exists() or sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"BM25 evidence shard hash mismatch: {path}")
        for row in read_jsonl(path):
            qid = str(row["qid"])
            if qid in output:
                raise RuntimeError(f"duplicate BM25 qid: {qid}")
            fold = fold_for[qid]
            output[qid] = _bm25_rankings(row["evidence"], selected[fold], limit=limit)
    if len(output) != len(fold_for):
        raise RuntimeError(f"BM25 query count mismatch: {len(output)} != {len(fold_for)}")
    return output, {"evidence_manifest_sha256": sha256_file(evidence_dir / "manifest.json"), "tuning_sha256": sha256_file(tuning_path), "query_count": len(output), "feature_context_top_k": int(limit), "contract": "tuned_exp021_bm25_parent_aggregation_v1"}


def _write_source_rows(destination: Path, source_maps: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], qids: Sequence[str]) -> dict[str, Any]:
    def rows() -> Iterable[Mapping[str, Any]]:
        for qid in sorted(map(str, qids)):
            yield {
                "qid": qid,
                "sources": {
                    source: [dict(item, source=source) for item in source_maps[source][qid]]
                    for source in ("vietlegal_e5", "vnlegal_lal", "bm25")
                },
            }
    count = write_jsonl_atomic(destination, rows())
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "rows": count, "sources": ["vietlegal_e5", "vnlegal_lal", "bm25"], "candidate_top_k": 50, "feature_context_top_k": 500}


def _populate_dense_context(conn: sqlite3.Connection, table: str, directory: Path, required_qids: set[str]) -> dict[str, Any]:
    """Persist only F1-F4 top-500 rows on disk; never build a 7k map in RAM."""
    manifest = read_json(directory / "manifest.json")
    _verify_success(directory, manifest)
    if int(manifest.get("limit", 0)) < 500 or int(manifest.get("query_count", 0)) != 7000:
        raise RuntimeError(f"ranking prerequisite is not a complete 7,000-query top-500+ store: {directory}")
    conn.execute(f"CREATE TABLE {table} (qid TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    count = 0
    for entry in manifest.get("shards", []):
        path = directory / "shards" / str(entry["name"])
        if not path.exists() or sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"ranking shard hash mismatch: {path}")
        batch: list[tuple[str, str]] = []
        for row in read_jsonl(path):
            qid = str(row["qid"])
            if qid not in required_qids:
                continue
            docs = row.get("documents", row.get("rankings", []))[:500]
            if len(docs) < 500:
                raise RuntimeError(f"dense feature context shorter than 500: {table}/{qid}")
            payload = [{"doc_id": str(item["doc_id"]), "rank": int(item.get("rank", index + 1)), "score": float(item.get("score", 0.0))} for index, item in enumerate(docs)]
            batch.append((qid, canonical_json(payload)))
            count += 1
            if len(batch) >= 100:
                conn.executemany(f"INSERT INTO {table}(qid,payload) VALUES (?,?)", batch); conn.commit(); batch.clear()
        if batch:
            conn.executemany(f"INSERT INTO {table}(qid,payload) VALUES (?,?)", batch); conn.commit()
    if count != len(required_qids):
        raise RuntimeError(f"dense inner coverage mismatch for {table}: {count} != {len(required_qids)}")
    return {"directory": str(directory), "manifest_sha256": sha256_file(directory / "manifest.json"), "content_fingerprint": manifest.get("content_fingerprint"), "model": manifest.get("model"), "query_count": count, "feature_context_top_k": 500}


def _populate_bm25_context(conn: sqlite3.Connection, evidence_dir: Path, tuning_path: Path, fold_for: Mapping[str, str], required_qids: set[str]) -> dict[str, Any]:
    tuning = read_json(tuning_path)
    selected = tuning.get("selected_by_candidate_budget", {}).get("150")
    if not isinstance(selected, dict):
        raise RuntimeError("tuned BM25 report lacks selected_by_candidate_budget[150]")
    manifest = read_json(evidence_dir / "manifest.json")
    conn.execute("CREATE TABLE bm25 (qid TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    count = 0
    for entry in manifest.get("shards", []):
        path = evidence_dir / "shards" / str(entry["name"])
        if not path.exists() or sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"BM25 evidence shard hash mismatch: {path}")
        batch: list[tuple[str, str]] = []
        for row in read_jsonl(path):
            qid = str(row["qid"])
            if qid not in required_qids:
                continue
            payload = _bm25_rankings(row["evidence"], selected[fold_for[qid]], limit=500)
            if len(payload) < 50:
                raise RuntimeError(f"BM25 candidate context shorter than 50: {qid}")
            batch.append((qid, canonical_json(payload)))
            count += 1
            if len(batch) >= 100:
                conn.executemany("INSERT INTO bm25(qid,payload) VALUES (?,?)", batch); conn.commit(); batch.clear()
        if batch:
            conn.executemany("INSERT INTO bm25(qid,payload) VALUES (?,?)", batch); conn.commit()
    if count != len(required_qids):
        raise RuntimeError(f"BM25 inner coverage mismatch: {count} != {len(required_qids)}")
    return {"evidence_manifest_sha256": sha256_file(evidence_dir / "manifest.json"), "tuning_sha256": sha256_file(tuning_path), "query_count": count, "feature_context_top_k": 500, "contract": "tuned_exp021_bm25_parent_aggregation_v1"}


def _write_source_rows_from_sqlite(destination: Path, database: Path, qids: Sequence[str]) -> dict[str, Any]:
    def rows() -> Iterable[Mapping[str, Any]]:
        with sqlite3.connect(database) as conn:
            for qid in sorted(map(str, qids)):
                values = {}
                for source, table in (("vietlegal_e5", "e5"), ("vnlegal_lal", "lal"), ("bm25", "bm25")):
                    row = conn.execute(f"SELECT payload FROM {table} WHERE qid=?", (qid,)).fetchone()
                    if row is None:
                        raise RuntimeError(f"missing streamed source context: {source}/{qid}")
                    values[source] = [dict(item, source=source) for item in json.loads(row[0])]
                yield {"qid": qid, "sources": values}
    count = write_jsonl_atomic(destination, rows())
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "rows": count, "sources": ["vietlegal_e5", "vnlegal_lal", "bm25"], "candidate_top_k": 50, "feature_context_top_k": 500, "storage": "streamed_sqlite_temp"}


def _copy_anchor(anchor_path: Path, destination: Path, qids: set[str], fold_for: Mapping[str, str]) -> dict[str, Any]:
    seen: set[str] = set()

    def rows() -> Iterable[Mapping[str, Any]]:
        for row in read_jsonl(anchor_path):
            qid = str(row["qid"])
            if qid not in qids or fold_for[qid] == FOLD0:
                raise RuntimeError(f"Fold 0 or unknown qid in anchor predictions: {qid}")
            if qid in seen:
                raise RuntimeError(f"duplicate anchor qid: {qid}")
            seen.add(qid)
            docs = row.get("prediction", row.get("documents", row.get("ranking")))
            if not isinstance(docs, list):
                raise RuntimeError(f"anchor prediction list missing: {qid}")
            values = [{"doc_id": str(x.get("doc_id")) if isinstance(x, Mapping) else str(x)} for x in docs]
            if len({x["doc_id"] for x in values}) != len(values):
                raise RuntimeError(f"duplicate anchor document: {qid}")
            output = {"qid": qid, "prediction": values, "fold0_included": False}
            if isinstance(row.get("raw_scores"), Mapping):
                output["raw_scores"] = {str(doc): float(score) for doc, score in row["raw_scores"].items()}
            yield output
    count = write_jsonl_atomic(destination, rows())
    if seen != qids:
        raise RuntimeError(f"anchor prediction coverage mismatch: expected {len(qids)}, observed {len(seen)}")
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "rows": count, "fold0_included": False, "source": str(anchor_path)}


def _copy_impact_as_json(impact_path: Path, destination: Path) -> dict[str, Any]:
    rows = list(read_jsonl(impact_path))
    atomic_json(destination, rows)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "rows": len(rows), "source": str(impact_path)}


def _locked_configs(pilot_path: Path, destination: Path) -> dict[str, Any]:
    pilot = read_json(pilot_path)
    configs: dict[str, Any] = {}
    for item in pilot.get("lambdamart_folds", []):
        fold = str(item.get("validation_fold"))
        if fold in INNER_FOLDS:
            chosen = item.get("chosen_config", {}).get("config")
            if not isinstance(chosen, Mapping):
                raise RuntimeError(f"missing chosen LambdaMART config: {fold}")
            configs[fold] = dict(chosen)
    if set(configs) != set(INNER_FOLDS):
        raise RuntimeError(f"locked configs do not cover F1-F4: {sorted(configs)}")
    payload = {"schema_version": SCHEMA, "stage": "exp109b-locked-anchor-config", "winner": "lambdamart_top50_per_source", "configs_by_heldout_fold": configs, "fold0_included": False, "source_report": str(pilot_path), "source_report_sha256": sha256_file(pilot_path)}
    atomic_json(destination, payload)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "fold0_included": False, "configs": sorted(configs)}


def _parent_metadata(struct_dir: Path, destination: Path) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    token_lengths: dict[str, int] = defaultdict(int)
    chunks_path = struct_dir / "chunks.jsonl"
    for row in read_jsonl(chunks_path):
        doc = str(row["doc_id"])
        counts[doc] += 1
        token_lengths[doc] += int(row.get("token_count", 0))
    docs_path = struct_dir / "documents.jsonl"
    def rows() -> Iterable[Mapping[str, Any]]:
        for row in read_jsonl(docs_path):
            doc = str(row["doc_id"])
            yield {"doc_id": doc, "parent_chunk_count": int(counts.get(doc, 0)), "parent_token_length": int(token_lengths.get(doc, 0))}
    count = write_jsonl_atomic(destination, rows())
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "rows": count, "includes_document_label": False}


def _manifest_record(input_dir: Path, rel: str) -> dict[str, Any]:
    path = input_dir / rel
    return {"path": rel.replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def export_bundle(
    output_root: Path,
    *,
    train_path: Path = DEFAULT_TRAIN,
    folds_path: Path = DEFAULT_FOLDS,
    exclusions_path: Path = DEFAULT_EXCLUSIONS,
    impact_path: Path = DEFAULT_IMPACT,
    query_dir: Path = DEFAULT_QUERY_DIR,
    struct_dir: Path = DEFAULT_STRUCT_DIR,
    e5_rank_dir: Path = DEFAULT_E5_RANK_DIR,
    lal_rank_dir: Path = DEFAULT_LAL_RANK_DIR,
    bm25_evidence_dir: Path = DEFAULT_BM25_EVIDENCE,
    bm25_tuning_path: Path = DEFAULT_BM25_TUNING,
    pilot_path: Path = DEFAULT_PILOT,
    anchor_path: Path | None = None,
    source_mode: str = "memory",
) -> dict[str, Any]:
    output_root = Path(output_root)
    input_dir = output_root / "input"
    code_dir = output_root / "code"
    input_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    train_raw = read_json(train_path)
    if not isinstance(train_raw, dict):
        raise RuntimeError("train/folds must be JSON objects")
    answers, label_stats = canonical_labels(train_path, exclusions_path, impact_path)
    qids = sorted(map(str, train_raw))
    folds, fold_for = load_folds(folds_path)
    validate_fold_partition(folds, qids, outer=FOLD0)
    if set(qids) != set(fold_for) or len(fold_for) != len(qids):
        raise RuntimeError("train/folds mismatch")
    inner = inner_qids(folds, outer=FOLD0)
    if label_stats.get("label_fingerprint") != LABEL_FINGERPRINT:
        raise RuntimeError("canonical label fingerprint mismatch")
    if anchor_path is None:
        anchor_path = next((path for path in DEFAULT_ANCHOR_CANDIDATES if path.exists()), None)
    if anchor_path is None or not anchor_path.exists():
        raise RuntimeError("required verified EXP-109B anchor_inner_predictions.jsonl is absent; refusing to fabricate bundle input")
    copied: list[dict[str, Any]] = []
    for source, rel in ((train_path, "train.json"), (folds_path, "cv_folds.json"), (exclusions_path, "exclusions.json"), (query_dir / "train_queries.f32.npy", "e5_query_embeddings/train_queries.f32.npy"), (query_dir / "train_query_ids.json", "e5_query_embeddings/train_query_ids.json"), (query_dir / "manifest.json", "e5_query_embeddings/manifest.json"), (pilot_path, "exp109b_pilot_report.json")):
        copied.append(_atomic_copy(Path(source), input_dir / rel))
    copied.append(_copy_impact_as_json(impact_path, input_dir / "label_impact_report.json"))
    if source_mode == "memory":
        # Fast path: safe when EXP-109C is stopped.  It avoids SQLite JSON
        # serialization/deserialization and writes the bundle in one pass.
        source_e5, e5_meta = _ranking_rows_top50(e5_rank_dir)
        source_lal, lal_meta = _ranking_rows_top50(lal_rank_dir)
        source_bm25, bm25_meta = _bm25_top50(bm25_evidence_dir, bm25_tuning_path, fold_for, limit=500)
        source_report = _write_source_rows(input_dir / "exp109b_sources_top50.jsonl", {"vietlegal_e5": source_e5, "vnlegal_lal": source_lal, "bm25": source_bm25}, inner)
    elif source_mode == "streaming":
        # Bounded-memory fallback for concurrent high-RAM experiments.
        database = output_root / "STREAMING_SOURCE_CONTEXT.sqlite"
        if database.exists():
            database.unlink()
        with sqlite3.connect(database) as conn:
            conn.execute("PRAGMA journal_mode=OFF"); conn.execute("PRAGMA synchronous=OFF")
            required_inner = set(inner)
            e5_meta = _populate_dense_context(conn, "e5", e5_rank_dir, required_inner)
            lal_meta = _populate_dense_context(conn, "lal", lal_rank_dir, required_inner)
            bm25_meta = _populate_bm25_context(conn, bm25_evidence_dir, bm25_tuning_path, fold_for, required_inner)
        source_report = _write_source_rows_from_sqlite(input_dir / "exp109b_sources_top50.jsonl", database, inner)
    else:
        raise ValueError(f"unknown source_mode: {source_mode}")
    anchor_report = _copy_anchor(anchor_path, input_dir / "exp109b_anchor_inner_predictions.jsonl", set(inner), fold_for)
    configs_report = _locked_configs(pilot_path, input_dir / "exp109b_locked_configs.json")
    parent_report = _parent_metadata(struct_dir, input_dir / "parent_metadata_minimal.jsonl")
    copied.extend([source_report, anchor_report, configs_report, parent_report])
    exp024_report = ROOT / "results" / "exp024_memory_lexical" / "report.json"
    copied.append(_atomic_copy(exp024_report, input_dir / "exp024_report.json"))
    copied.append(_atomic_copy(CODE_FILE, code_dir / CODE_FILE.name))
    copied.append(_atomic_copy(CODE_FILE.with_name("exp110p_semantic_label_prototype.py"), code_dir / "exp110p_semantic_label_prototype.py"))
    materializer = CODE_FILE.with_name("exp110p_materialize_exp109b_anchor.py")
    if materializer.exists():
        copied.append(_atomic_copy(materializer, code_dir / materializer.name))
    relative_files = [
        "train.json", "cv_folds.json", "exclusions.json", "label_impact_report.json",
        "e5_query_embeddings/train_queries.f32.npy", "e5_query_embeddings/train_query_ids.json", "e5_query_embeddings/manifest.json",
        "exp109b_sources_top50.jsonl", "exp109b_anchor_inner_predictions.jsonl", "exp109b_locked_configs.json", "exp109b_pilot_report.json",
        "parent_metadata_minimal.jsonl", "exp024_report.json",
    ]
    manifest = {
        "schema_version": SCHEMA,
        "stage": "prepare-colab-bundle",
        "label_policy": LABEL_POLICY,
        "label_fingerprint": LABEL_FINGERPRINT,
        "folds_fingerprint": sha256_file(folds_path),
        "query_count": len(qids),
        "inner_query_count": len(inner),
        "evaluable_query_count": int(label_stats["evaluable_queries"]),
        "document_count": int(parent_report["rows"]),
        "source_manifests": {"vietlegal_e5": e5_meta, "vnlegal_lal": lal_meta, "bm25": bm25_meta},
        "files": {_rel: _manifest_record(input_dir, _rel) for _rel in relative_files},
        "code_files": {
            CODE_FILE.name: {"sha256": sha256_file(code_dir / CODE_FILE.name), "bytes": (code_dir / CODE_FILE.name).stat().st_size},
            "exp110p_semantic_label_prototype.py": {"sha256": sha256_file(code_dir / "exp110p_semantic_label_prototype.py"), "bytes": (code_dir / "exp110p_semantic_label_prototype.py").stat().st_size},
            **({materializer.name: {"sha256": sha256_file(code_dir / materializer.name), "bytes": (code_dir / materializer.name).stat().st_size}} if materializer.exists() else {}),
        },
        "exporter_code_sha256": sha256_file(CODE_FILE),
        "fold0_predictions_included": False,
        "fold0_read": False,
        "inner_only_source_rows": True,
        "source_rows_candidate_top_k": 50,
        "source_rows_feature_context_top_k": 500,
        "source_rankings_include_fold_labels": False,
        "late_interaction_scores_included": False,
        "lexical_query_memory_included": False,
        "label_stats": label_stats,
        "bundle_contract": "compact_query_label_source_top50_candidate_top500_context_v2",
        "source_export_mode": source_mode,
    }
    atomic_json(input_dir / "INPUT_MANIFEST.json", manifest)
    result = {"status": "PASS", "output_root": str(output_root.resolve()), "input_manifest": str((input_dir / "INPUT_MANIFEST.json").resolve()), "input_manifest_sha256": sha256_file(input_dir / "INPUT_MANIFEST.json"), "files": copied, "fold0_predictions_included": False, "label_stats": label_stats}
    atomic_json(output_root / "EXPORT_REPORT.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-predictions", type=Path)
    parser.add_argument("--source-mode", choices=("memory", "streaming"), default="memory")
    args = parser.parse_args(argv)
    print(json.dumps(export_bundle(args.output, anchor_path=args.anchor_predictions, source_mode=args.source_mode), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
