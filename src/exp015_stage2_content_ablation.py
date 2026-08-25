"""Stage-2 paired raw-text versus v3-marker retrieval screen for EXP-015."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import exp015_model_screen as base


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "exp015_model_screen" / "stage2"
RESULTS = ROOT / "results" / "exp015_model_screen" / "stage2"
SEED = 15016
QUERY_COUNT = 512
BACKGROUND_DOCUMENTS = 1536
CHUNKS_PER_DOCUMENT = 4
MODEL_KEYS = ("vnlegal_lal", "vietlegal_e5", "vietlegal_harrier_0_6b")
VIEWS = ("raw_text", "retrieval_text")


def paths() -> dict[str, Path]:
    return {"queries": CACHE / "queries.jsonl", "chunks": CACHE / "chunks.jsonl", "manifest": CACHE / "manifest.json", "success": CACHE / "_SUCCESS.json"}


def build_fixture(force: bool = False) -> dict[str, Any]:
    out = paths()
    if out["success"].exists() and not force:
        return json.loads(out["manifest"].read_text(encoding="utf-8"))
    if any(path.exists() for path in out.values()) and not force:
        raise RuntimeError("Partial stage-2 fixture exists; inspect it before --force-fixture")
    train = base.load_train()
    folds = json.loads(base.FOLDS_PATH.read_text(encoding="utf-8"))
    document_ids = {str(row["doc_id"]) for row in base.read_jsonl(base.V3_DIR / "documents.jsonl")}
    eligible = [str(qid) for qid in folds["fold_0"] if str(qid) in train and base.answer_ids(train[str(qid)]) and set(base.answer_ids(train[str(qid)])).issubset(document_ids)]
    qids = sorted(eligible, key=lambda qid: base.digest({"stage": 2, "seed": SEED, "qid": qid}))[:QUERY_COUNT]
    if len(qids) != QUERY_COUNT:
        raise RuntimeError("Insufficient eligible questions")
    positives = {doc_id for qid in qids for doc_id in base.answer_ids(train[qid])}
    background = sorted(document_ids - positives, key=lambda doc_id: base.digest({"stage": 2, "seed": SEED, "doc": doc_id}))[:BACKGROUND_DOCUMENTS]
    selected_docs = positives | set(background)
    per_doc: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in base.read_jsonl(base.V3_DIR / "chunks.jsonl"):
        if str(row["doc_id"]) in selected_docs:
            per_doc[str(row["doc_id"])].append((base.digest({"stage": 2, "seed": SEED, "chunk": row["chunk_id"]}), row))
    rows = [row for doc_id in sorted(selected_docs) for _, row in sorted(per_doc[doc_id])[:CHUNKS_PER_DOCUMENT]]
    rows.sort(key=lambda row: str(row["chunk_id"]))
    available_docs = {str(row["doc_id"]) for row in rows}
    queries = [{"qid": qid, "question": str(train[qid]["question"]), "answer_doc_ids": base.answer_ids(train[qid])} for qid in qids]
    if any(not set(row["answer_doc_ids"]).issubset(available_docs) for row in queries):
        raise RuntimeError("A positive parent has no retained candidate chunk")
    CACHE.mkdir(parents=True, exist_ok=True)
    for key, records in (("queries", queries), ("chunks", rows)):
        with out[key].open("w", encoding="utf-8", newline="\n") as handle:
            for row in records:
                handle.write(base.canonical_json(row) + "\n")
    v3 = json.loads((base.V3_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest = {"experiment": "exp015_stage2_content_ablation", "interpretation": "fold_0_development_only_not_oof", "v3_content_fingerprint": v3["content_fingerprint"], "seed": SEED, "query_count": len(queries), "positive_document_count": len(positives), "background_document_count": len(background), "document_count": len(available_docs), "chunk_count": len(rows), "chunks_per_document_cap": CHUNKS_PER_DOCUMENT, "views": list(VIEWS), "query_ids_sha256": base.digest(qids), "chunk_ids_sha256": base.digest([row["chunk_id"] for row in rows])}
    base.write_json(out["manifest"], manifest)
    base.write_json(out["success"], {"manifest_sha256": base.sha256_file(out["manifest"])})
    return manifest


def load_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    out = paths()
    if not out["success"].exists():
        raise RuntimeError("Run --stage build-fixture first")
    return list(base.read_jsonl(out["queries"])), list(base.read_jsonl(out["chunks"])), json.loads(out["manifest"].read_text(encoding="utf-8"))


def run_one(key: str, view: str, *, device: str, batch_size: int, force: bool) -> dict[str, Any]:
    if key not in MODEL_KEYS or view not in VIEWS:
        raise ValueError("Unsupported model or view")
    report_path = RESULTS / f"{key}__{view}.json"
    vector_path = RESULTS / f"{key}__{view}.npz"
    if report_path.exists() and vector_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    queries, chunks, fixture = load_fixture()
    spec = base.MODELS[key]
    encoder = base.create_encoder(spec, device, allow_download=False)
    docs = [spec.document_prefix + str(row[view]) for row in chunks]
    questions = [spec.query_prefix + str(row["question"]) for row in queries]
    start = time.perf_counter()
    doc_vectors, effective_batch = base.encode_oom_safe(encoder, docs, batch_size, device)
    query_vectors, effective_batch = base.encode_oom_safe(encoder, questions, effective_batch, device)
    evaluation = base.evaluate(queries, chunks, query_vectors, doc_vectors)
    token_counts = encoder.token_counts(docs)
    elapsed = time.perf_counter() - start
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(vector_path, query_vectors=query_vectors.astype(np.float32), chunk_vectors=doc_vectors.astype(np.float32))
    report = {"experiment": "exp015_stage2_content_ablation", "interpretation": "fold_0_development_only_not_oof", "model": {"key": key, "repo_id": spec.repo_id, "backend": spec.backend, "max_length": spec.max_length, "query_prefix": spec.query_prefix, "document_prefix": spec.document_prefix}, "view": view, "fixture": fixture, "runtime": {"device": device, "initial_batch_size": batch_size, "effective_batch_size": effective_batch, "elapsed_seconds": round(elapsed, 3), "texts_per_second": round((len(docs)+len(questions))/elapsed, 3)}, "tokenization": {"chunk_truncated": int(sum(count > spec.max_length for count in token_counts)), "chunk_tokens_p95": float(np.percentile(token_counts, 95))}, "embeddings": {"dimension": int(doc_vectors.shape[1])}, **evaluation}
    base.write_json(report_path, report)
    return report


def summarize() -> Path:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in RESULTS.glob("*.json") if "__" in path.stem]
    reports.sort(key=lambda row: (row["model"]["key"], row["view"]))
    lines = ["# EXP-015 stage 2 — raw text vs v3 markers", "", "> Fold-0 development ablation only; not OOF.", "", "| Model | View | R@5 | R@20 | R@100 | MRR@5 | trunc. chunks |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in reports:
        m = row["metrics"]
        lines.append(f"| `{row['model']['key']}` | `{row['view']}` | {m['recall_at_5']:.4f} | {m['recall_at_20']:.4f} | {m['recall_at_100']:.4f} | {m['mrr_at_5']:.4f} | {row['tokenization']['chunk_truncated']} |")
    paired: dict[str, Any] = {}
    by_key = {(row["model"]["key"], row["view"]): row for row in reports}
    for key in MODEL_KEYS:
        raw = by_key.get((key, "raw_text"))
        marked = by_key.get((key, "retrieval_text"))
        if not raw or not marked:
            continue
        raw_ranks = {row["qid"]: row["first_relevant_rank"] for row in raw["per_query"]}
        marked_ranks = {row["qid"]: row["first_relevant_rank"] for row in marked["per_query"]}
        deltas = [raw_ranks[qid] - marked_ranks[qid] for qid in raw_ranks]
        paired[key] = {
            "queries": len(deltas),
            "marker_improves_rank": sum(delta > 0 for delta in deltas),
            "marker_worsens_rank": sum(delta < 0 for delta in deltas),
            "same_rank": sum(delta == 0 for delta in deltas),
            "mean_rank_delta_raw_minus_marker": float(np.mean(deltas)),
            "recall_at_5_delta": marked["metrics"]["recall_at_5"] - raw["metrics"]["recall_at_5"],
            "mrr_at_5_delta": marked["metrics"]["mrr_at_5"] - raw["metrics"]["mrr_at_5"],
        }
    lines.extend(["", "## Paired raw-to-marker deltas", "", "| Model | Better rank | Worse rank | Same | ΔR@5 | ΔMRR@5 |", "|---|---:|---:|---:|---:|---:|"])
    for key, delta in paired.items():
        lines.append(f"| `{key}` | {delta['marker_improves_rank']} | {delta['marker_worsens_rank']} | {delta['same_rank']} | {delta['recall_at_5_delta']:+.4f} | {delta['mrr_at_5_delta']:+.4f} |")
    out = RESULTS / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    base.write_json(RESULTS / "paired_deltas.json", paired)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build-fixture", "run-one", "run-all", "summarize"), required=True)
    parser.add_argument("--model", choices=MODEL_KEYS)
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-fixture", action="store_true")
    args = parser.parse_args()
    if args.stage == "build-fixture": print(json.dumps(build_fixture(args.force_fixture), ensure_ascii=False, indent=2))
    elif args.stage == "run-one":
        if not args.model or not args.view: parser.error("--model and --view are required")
        print(json.dumps(run_one(args.model, args.view, device=args.device, batch_size=args.batch_size, force=args.force), ensure_ascii=False, indent=2))
    elif args.stage == "run-all":
        for key in MODEL_KEYS:
            for view in VIEWS:
                print(f"[stage2] {key} {view}", flush=True)
                run_one(key, view, device=args.device, batch_size=args.batch_size, force=args.force)
        print(summarize())
    else: print(summarize())


if __name__ == "__main__": main()
