"""EXP-021: E5 chunk retrieval aggregated into parent-document candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from exp012b_core import atomic_json, canonical_json, load_answers, read_jsonl, sha256_file, stage_run, write_jsonl
from exp012b_tuning import load_folds


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "mainguyen9/vietlegal-e5"
SCHEMA = "legalir.exp021_e5_dense_candidates.v1"
AGGREGATIONS = ("max", "top2_mean", "top4_mean", "logsumexp")
CURVE_KS = (20, 32, 50, 80, 100, 120, 150)
MAX_CANDIDATES = 150
CHUNK_DEPTH = 4096
EVIDENCE_PER_DOCUMENT = 4


def _code_sha256() -> str:
    return sha256_file(Path(__file__))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_e5_cache(corpus_dir: Path, e5_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    corpus = _json(corpus_dir / "manifest.json")
    cache = _json(e5_dir / "manifest.json")
    if not (corpus_dir / "_SUCCESS.json").exists() or not (e5_dir / "_SUCCESS.json").exists():
        raise RuntimeError("Structural corpus or E5 cache lacks a success marker")
    if cache.get("corpus_fingerprint") != corpus.get("content_fingerprint"):
        raise RuntimeError("E5 cache does not belong to structural corpus")
    if cache.get("model_id") != MODEL_ID or cache.get("query_prefix") != "query: " or cache.get("dimension") != 1024:
        raise RuntimeError("Unexpected E5 model/query prefix/dimension")
    if cache.get("storage_dtype") != "float16" or cache.get("embedding_compute_dtype") != "float32":
        raise RuntimeError("Unexpected E5 precision policy")
    for name, expected in cache.get("artifact_sha256", {}).items():
        if sha256_file(e5_dir / name) != expected:
            raise RuntimeError(f"Corrupt E5 cache artifact: {name}")
    return corpus, cache


def _load_chunk_index(e5_dir: Path) -> tuple[list[str], list[str]]:
    ids = list(read_jsonl(e5_dir / "chunk_ids.jsonl"))
    return [str(row["chunk_id"]) for row in ids], [str(row["doc_id"]) for row in ids]


def aggregate_chunk_hits(
    hits: Iterable[tuple[str, str, float]], *, evidence_limit: int = EVIDENCE_PER_DOCUMENT
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate score-sorted chunk hits into deterministic document rankings."""
    by_doc: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for chunk_id, doc_id, score in hits:
        if len(by_doc[doc_id]) < evidence_limit:
            by_doc[doc_id].append((chunk_id, float(score)))
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in AGGREGATIONS}
    for doc_id, evidence in by_doc.items():
        scores = [row[1] for row in evidence]
        values = {
            "max": scores[0],
            "top2_mean": sum(scores[:2]) / min(2, len(scores)),
            "top4_mean": sum(scores[:4]) / min(4, len(scores)),
            "logsumexp": math.log(sum(math.exp(value - scores[0]) for value in scores[:4])) + scores[0],
        }
        evidence_rows = [{"chunk_id": chunk_id, "chunk_score": score} for chunk_id, score in evidence]
        for name, score in values.items():
            output[name].append({"doc_id": doc_id, "aggregate_score": score, "evidence": evidence_rows})
    for name in output:
        output[name].sort(key=lambda row: (-float(row["aggregate_score"]), str(row["doc_id"])))
        for rank, row in enumerate(output[name], 1):
            row["rank"] = rank
    return output


def _recall_mrr(rankings: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], qids: Iterable[str], k: int) -> tuple[float, float]:
    recall_sum = 0.0
    reciprocal_sum = 0.0
    count = 0
    for qid in qids:
        gold = answers[qid]
        if not gold:
            continue
        predicted = set(rankings[qid][:k])
        recall_sum += len(predicted & gold) / len(gold)
        first = next((index for index, doc_id in enumerate(rankings[qid][:k], 1) if doc_id in gold), None)
        reciprocal_sum += 0.0 if first is None else 1.0 / first
        count += 1
    return recall_sum / max(count, 1), reciprocal_sum / max(count, 1)


def nested_select(
    rankings: Mapping[str, Mapping[str, Sequence[str]]], retained_answers: Mapping[str, set[str]], folds: Mapping[str, Sequence[str]]
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, dict[str, float]]]:
    all_qids = sorted(retained_answers)
    chosen: dict[str, str] = {}
    oof: dict[str, list[str]] = {}
    fold_metrics: dict[str, dict[str, float]] = {}
    for fold, heldout in sorted(folds.items()):
        heldout_set = set(map(str, heldout))
        train_qids = [qid for qid in all_qids if qid not in heldout_set]
        keys: list[tuple[float, float, int, str]] = []
        for priority, name in enumerate(AGGREGATIONS):
            recall, mrr = _recall_mrr(rankings[name], retained_answers, train_qids, MAX_CANDIDATES)
            keys.append((recall, mrr, -priority, name))
        _, _, _, selected = max(keys)
        chosen[fold] = selected
        for qid in heldout:
            oof[str(qid)] = list(rankings[selected][str(qid)][:MAX_CANDIDATES])
        recall, mrr = _recall_mrr(oof, retained_answers, map(str, heldout), MAX_CANDIDATES)
        fold_metrics[fold] = {"retained_recall@150": recall, "retained_mrr@150": mrr}
    return chosen, oof, fold_metrics


def _query_fingerprint(queries: Mapping[str, str], folds: Mapping[str, Sequence[str]]) -> str:
    return _hash_json({"queries": dict(sorted(queries.items())), "folds": {name: list(map(str, values)) for name, values in sorted(folds.items())}})


def _encode_queries(
    queries: Mapping[str, str],
    output_dir: Path,
    query_fingerprint: str,
    e5_cache_fingerprint: str,
    *,
    batch_size: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    ids = sorted(queries)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path, ids_path, manifest_path = output_dir / "train_queries.f32.npy", output_dir / "train_query_ids.json", output_dir / "manifest.json"
    if matrix_path.exists() and ids_path.exists() and manifest_path.exists():
        manifest = _json(manifest_path)
        fingerprints_match = not (
            manifest.get("query_fingerprint") != query_fingerprint
            or manifest.get("e5_cache_fingerprint") != e5_cache_fingerprint
            or manifest.get("encoder_code_sha256") != _code_sha256()
        )
        if fingerprints_match:
            cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            matrix = np.load(matrix_path, mmap_mode="r")
            if cached_ids != ids or matrix.shape != (len(ids), 1024) or matrix.dtype != np.float32:
                raise RuntimeError("Existing query cache is malformed")
            return ids, matrix
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(device).eval()
    matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(len(ids), 1024))
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        texts = ["query: " + queries[qid] for qid in batch_ids]
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=True, return_tensors="pt")
        if int(encoded["attention_mask"].sum(dim=1).max()) > 512:
            raise RuntimeError("Query input exceeds E5 512-token budget")
        with torch.inference_mode():
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            weights = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1e-9)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float32)
        matrix[start:start + len(batch_ids)] = vectors
    matrix.flush()
    atomic_json(ids_path, ids)
    atomic_json(manifest_path, {"schema_version": SCHEMA, "query_fingerprint": query_fingerprint, "e5_cache_fingerprint": e5_cache_fingerprint, "encoder_code_sha256": _code_sha256(), "model_id": MODEL_ID, "query_prefix": "query: ", "dimension": 1024, "dtype": "float32", "artifacts": {matrix_path.name: sha256_file(matrix_path), ids_path.name: sha256_file(ids_path)}})
    return ids, np.load(matrix_path, mmap_mode="r")


def run_train_oof(*, corpus_dir: Path, e5_dir: Path, train_path: Path, folds_path: Path, preprocessing_dir: Path, cache_dir: Path, results_dir: Path, batch_size: int = 32, device: str = "cuda") -> dict[str, Any]:
    corpus, e5 = _validate_e5_cache(corpus_dir, e5_dir)
    answers = load_answers(train_path)
    folds = load_folds(folds_path)
    queries = {str(key): str(value["question"]) for key, value in _json(train_path).items()}
    if set(queries) != set(answers) or set().union(*(set(map(str, values)) for values in folds.values())) != set(queries):
        raise RuntimeError("Train queries, answers and fixed folds differ")
    excluded = {str(row["doc_id"]) for row in json.loads((preprocessing_dir / "exclusions.json").read_text(encoding="utf-8"))}
    retained_answers = {qid: values - excluded for qid, values in answers.items()}
    query_fp = _query_fingerprint(queries, folds)
    qids, query_matrix = _encode_queries(
        queries,
        cache_dir / "query_embeddings",
        query_fp,
        str(e5["cache_fingerprint"]),
        batch_size=batch_size,
        device=device,
    )
    chunk_ids, chunk_docs = _load_chunk_index(e5_dir)
    if len(chunk_ids) != int(e5["chunks"]):
        raise RuntimeError("Chunk-ID order length does not match E5 cache")
    # The cache is a read-only FP16 memmap. Materialize a writable FP32 copy
    # before creating the tensor, both to make the precision transition explicit
    # and to avoid PyTorch's undefined-behavior warning for read-only arrays.
    document_matrix = np.array(
        np.load(e5_dir / e5["embedding_file"], mmap_mode="r"), dtype=np.float32, copy=True
    )
    docs = torch.from_numpy(document_matrix).to(device)
    config_dir = cache_dir / "config_rankings"
    config_dir.mkdir(parents=True, exist_ok=True)
    temp_handles = {name: (config_dir / f".{name}.jsonl.tmp").open("w", encoding="utf-8", newline="\n") for name in AGGREGATIONS}
    started = time.perf_counter()
    with stage_run(results_dir, "e5-dense-train-oof", total=len(qids), v3_fingerprint=corpus["content_fingerprint"]) as logger:
        try:
            for start in range(0, len(qids), batch_size):
                batch_qids = qids[start:start + batch_size]
                # Query embeddings are also read-only memmap slices.  Score from
                # a writable FP32 copy so PyTorch never receives a non-writable
                # NumPy view.
                query_batch = np.array(
                    query_matrix[start:start + len(batch_qids)], dtype=np.float32, copy=True
                )
                vectors = torch.from_numpy(query_batch).to(device)
                with torch.inference_mode():
                    scores, indices = torch.topk(vectors @ docs.T, CHUNK_DEPTH, dim=1)
                for qid, row_scores, row_indices in zip(batch_qids, scores.cpu().numpy(), indices.cpu().numpy()):
                    hits = [(chunk_ids[int(index)], chunk_docs[int(index)], float(score)) for score, index in zip(row_scores, row_indices)]
                    aggregated = aggregate_chunk_hits(hits)
                    for name, candidates in aggregated.items():
                        if len(candidates) < MAX_CANDIDATES:
                            raise RuntimeError(f"{qid} has only {len(candidates)} unique documents in Top-{CHUNK_DEPTH}")
                        temp_handles[name].write(canonical_json({"qid": qid, "query": queries[qid], "aggregation": name, "candidates": candidates[:MAX_CANDIDATES]}) + "\n")
                completed = start + len(batch_qids)
                elapsed = time.perf_counter() - started
                logger.status(stage="e5-dense-train-oof", state="RUNNING", completed=completed, total=len(qids), eta_seconds=round((len(qids)-completed)/(completed/max(elapsed, 1e-9)), 1))
                if completed % 256 == 0 or completed == len(qids):
                    logger.log(f"progress={completed}/{len(qids)}")
        finally:
            for handle in temp_handles.values():
                handle.close()
        for name in AGGREGATIONS:
            (config_dir / f".{name}.jsonl.tmp").replace(config_dir / f"{name}.jsonl")
        rankings = {name: {str(row["qid"]): [str(item["doc_id"]) for item in row["candidates"]] for row in read_jsonl(config_dir / f"{name}.jsonl")} for name in AGGREGATIONS}
        chosen, oof, fold_metrics = nested_select(rankings, retained_answers, folds)
        selected_records = []
        config_rows = {name: {str(row["qid"]): row for row in read_jsonl(config_dir / f"{name}.jsonl")} for name in AGGREGATIONS}
        for qid in qids:
            fold = next(name for name, values in folds.items() if qid in set(map(str, values)))
            row = config_rows[chosen[fold]][qid]
            selected_records.append({**row, "fold": fold, "selected_by_nested_oof": True})
        candidates_path = cache_dir / "train_oof_candidates.jsonl"
        write_jsonl(candidates_path, selected_records)
        curve = {}
        for name, values in rankings.items():
            curve[name] = {str(k): _recall_mrr(values, retained_answers, qids, k)[0] for k in CURVE_KS}
        retained_curve = {str(k): _recall_mrr(oof, retained_answers, qids, k)[0] for k in CURVE_KS}
        all_curve = {str(k): _recall_mrr(oof, answers, qids, k)[0] for k in CURVE_KS}
        evidence_counts = [len(row["evidence"]) for record in selected_records for row in record["candidates"]]
        excluded_gold_ids = set().union(*(values & excluded for values in answers.values()))
        report = {"schema_version": SCHEMA, "status": "PASS", "corpus_fingerprint": corpus["content_fingerprint"], "e5_cache_fingerprint": e5["cache_fingerprint"], "query_fingerprint": query_fp, "code_sha256": _code_sha256(), "config": {"chunk_depth": CHUNK_DEPTH, "max_candidates": MAX_CANDIDATES, "aggregations": list(AGGREGATIONS), "curve_ks": list(CURVE_KS), "dimension": 1024, "score_dtype": "float32"}, "selected_aggregation_by_fold": chosen, "fold_metrics": fold_metrics, "oof": {"retained_candidate_recall_curve": retained_curve, "all_gold_candidate_recall_curve": all_curve}, "aggregation_curves_retained": curve, "unavoidable_exclusions": {"excluded_corpus_documents": len(excluded), "unique_gold_document_ids": len(excluded_gold_ids), "queries_with_removed_gold": sum(bool(values & excluded) for values in answers.values()), "gold_id_occurrences": sum(len(values & excluded) for values in answers.values())}, "evidence": {"documents": len(selected_records) * MAX_CANDIDATES, "mean_chunks_per_document": sum(evidence_counts) / max(len(evidence_counts), 1), "max_chunks_per_document": max(evidence_counts, default=0)}, "runtime_seconds": round(time.perf_counter() - started, 3)}
        exp014_path = ROOT / "results" / "exp014" / "candidate_audit" / "candidate_audit.json"
        if exp014_path.exists():
            exp014 = _json(exp014_path)
            report["exp014_reference"] = {"candidate_recall_curve": exp014.get("metrics", {}), "pool_size": exp014.get("pool_size", {}), "caveat": "Different structural corpus and hybrid candidate channels; candidate-stage reference only."}
        report_path = results_dir / "oof_report.json"
        atomic_json(report_path, report)
        files = [candidates_path, report_path, *(config_dir / f"{name}.jsonl" for name in AGGREGATIONS)]
        file_hashes = {str(path): sha256_file(path) for path in files}
        artifact_hashes = {
            ("cache/" + str(path.relative_to(cache_dir)) if path.is_relative_to(cache_dir) else "results/" + path.name): digest
            for path, digest in ((path, file_hashes[str(path)]) for path in files)
        }
        manifest = {"schema_version": SCHEMA, "stage": "e5-dense-train-oof", "content_fingerprint": _hash_json({"report": report, "files": file_hashes}), "inputs": {"corpus_fingerprint": corpus["content_fingerprint"], "e5_cache_fingerprint": e5["cache_fingerprint"], "query_fingerprint": query_fp, "folds_sha256": sha256_file(folds_path), "preprocessing_exclusions_sha256": sha256_file(preprocessing_dir / "exclusions.json")}, "config": report["config"], "artifact_sha256": artifact_hashes}
        atomic_json(cache_dir / "manifest.json", manifest)
        atomic_json(cache_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "stage": "e5-dense-train-oof", "corpus_fingerprint": corpus["content_fingerprint"], "content_fingerprint": manifest["content_fingerprint"]})
        logger.set_telemetry({"queries": len(qids), "runtime_seconds": report["runtime_seconds"]})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(__import__("sys").stdout, "reconfigure"):
        __import__("sys").stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "cache" / "structural_v3_e5_final_v1")
    parser.add_argument("--e5-dir", type=Path, default=ROOT / "cache" / "e5_final_v1")
    parser.add_argument("--train-file", type=Path, default=ROOT / "public_test_dataset" / "train.json")
    parser.add_argument("--folds-file", type=Path, default=ROOT / "cache" / "cv_folds.json")
    parser.add_argument("--preprocessing-dir", type=Path, default=ROOT / "cache" / "final_preprocessed_v2")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "exp021_e5_dense_candidates")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "exp021_e5_dense_candidates")
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    print(json.dumps(run_train_oof(
        corpus_dir=args.corpus_dir, e5_dir=args.e5_dir, train_path=args.train_file,
        folds_path=args.folds_file, preprocessing_dir=args.preprocessing_dir,
        cache_dir=args.cache_dir, results_dir=args.results_dir,
        batch_size=args.batch_size, device=args.device,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
