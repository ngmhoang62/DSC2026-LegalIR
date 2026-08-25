"""Evaluate one sparse configuration against the labelled training questions."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from exp012b_bm25 import (
    BM25Searcher, aggregate_bm25_documents, default_segmenter, safe_fts_query,
    title_folded_fts_query,
)
from exp012b_core import artifact_manifest, atomic_json, canonical_json, load_v3_manifest, stage_run
from exp012b_retrieval import evaluate_rankings


def retrieve_and_evaluate_bm25(
    *, database_path: Path, v3_dir: Path, train_path: Path, output_dir: Path,
    profile: str = "balanced", top_passages: int = 2000, top_documents: int = 100,
    workers: int = 3, query_mode: str = "default",
    aggregation: str = "rrf3",
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    v3 = load_v3_manifest(v3_dir)
    train = json.loads(train_path.read_text(encoding="utf-8"))
    qids = sorted(map(str, train))
    answers = {qid: set(map(str, train[qid]["answer"])) for qid in qids}
    # Underthesea's loaded CRF tokenizer uses mutable module-global state and
    # is not safe when invoked simultaneously from our SQLite worker threads.
    # Prepare all FTS expressions on the owning (main) thread; each worker then
    # does only read-only SQLite work with an immutable expression.
    if query_mode not in {"default", "title_folded"}:
        raise ValueError(f"Unknown query mode: {query_mode}")
    expressions = {
        qid: (
            title_folded_fts_query(str(train[qid]["question"]))
            if query_mode == "title_folded"
            else safe_fts_query(default_segmenter(str(train[qid]["question"])))
        )
        for qid in qids
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rankings.jsonl"
    predictions: dict[str, list[str]] = {}
    state = threading.local()
    searchers: list[BM25Searcher] = []
    searcher_lock = threading.Lock()

    def process(qid: str) -> tuple[str, str, list[dict[str, Any]]]:
        searcher = getattr(state, "searcher", None)
        if searcher is None:
            searcher = BM25Searcher(database_path, profile=profile)
            state.searcher = searcher
            with searcher_lock:
                searchers.append(searcher)
        hits = searcher.search_expression(expressions[qid], limit=top_passages)
        return qid, str(train[qid]["question"]), aggregate_bm25_documents(
            hits, top_docs=top_documents, strategy=aggregation
        )

    try:
        with stage_run(
            output_dir, "retrieve-bm25", total=len(qids), v3_fingerprint=v3["content_fingerprint"]
        ) as logger:
            temporary = output_path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for position, (qid, query, docs) in enumerate(executor.map(process, qids), start=1):
                        predictions[qid] = [str(row["doc_id"]) for row in docs]
                        handle.write(canonical_json({
                            "qid": qid, "query": query, "documents": docs,
                        }) + "\n")
                        if position % 64 == 0 or position == len(qids):
                            logger.status(
                                stage="retrieve-bm25", state="RUNNING", completed=position,
                                total=len(qids), profile=profile, emit_log=True,
                            )
            temporary.replace(output_path)
            metrics = evaluate_rankings(predictions, answers, ks=(1, 5, 10, 20, 50, 100))
            report = {
                "v3_fingerprint": v3["content_fingerprint"], "profile": profile,
                "top_passages": top_passages, "top_documents": top_documents,
                "queries": len(qids), "metrics": metrics,
            }
            report_path = output_dir / "metrics.json"
            atomic_json(report_path, report)
            logger.log(" ".join(f"{name}={value:.6f}" for name, value in metrics.items()))
            result = artifact_manifest(
                stage="retrieve-bm25",
                inputs={"v3_fingerprint": v3["content_fingerprint"]},
                config={"profile": profile, "query_mode": query_mode, "aggregation": aggregation, "top_passages": top_passages, "top_documents": top_documents},
                files=[output_path, report_path],
            )
            result["counts"] = {"queries": len(qids)}
            atomic_json(output_dir / "manifest.json", result)
            return result
    finally:
        for searcher in searchers:
            searcher.close()
