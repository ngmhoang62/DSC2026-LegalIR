"""Measure actual FTS5 retrieval cost at candidate-passage depths on a fixed query sample."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from exp012b_bm25 import BM25Searcher, aggregate_bm25_documents, default_segmenter, safe_fts_query
from exp012b_core import atomic_json
from exp012b_retrieval import weighted_rrf


def benchmark(*, database_path: Path, train_path: Path, output_path: Path, sample_size: int, workers: int) -> dict:
    train = json.loads(train_path.read_text(encoding="utf-8"))
    qids = sorted(map(str, train))
    selected = [qids[round(index * (len(qids) - 1) / (sample_size - 1))] for index in range(sample_size)]
    expressions = {qid: safe_fts_query(default_segmenter(str(train[qid]["question"]))) for qid in selected}
    results = {}
    for depth in (1024, 2048, 4096):
        state = threading.local(); searchers: list[BM25Searcher] = []; lock = threading.Lock()
        def one(qid: str) -> tuple[int, int]:
            searcher = getattr(state, "searcher", None)
            if searcher is None:
                searcher = BM25Searcher(database_path, profile="legal_structure")
                state.searcher = searcher
                with lock: searchers.append(searcher)
            hits = searcher.search_expression(expressions[qid], limit=depth)
            first = aggregate_bm25_documents(hits, top_docs=100, strategy="first_passage")
            rrf3 = aggregate_bm25_documents(hits, top_docs=100, strategy="rrf3")
            weighted_rrf({"first": first, "rrf3": rrf3}, limit=100)
            return len(hits), len({str(hit["doc_id"]) for hit in hits})
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            counts = list(executor.map(one, selected))
        elapsed = time.perf_counter() - started
        for searcher in searchers: searcher.close()
        results[str(depth)] = {
            "sample_queries": sample_size, "workers": workers, "elapsed_seconds": elapsed,
            "seconds_per_query_wall": elapsed / sample_size,
            "mean_returned_passages": sum(row[0] for row in counts) / sample_size,
            "mean_unique_documents_in_passage_pool": sum(row[1] for row in counts) / sample_size,
            "linear_extrapolation_7000_seconds": elapsed / sample_size * 7000,
        }
        print(f"depth={depth} elapsed={elapsed:.2f}s per_query={elapsed / sample_size:.4f}s", flush=True)
    report = {"database": str(database_path), "sample_size": sample_size, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True); atomic_json(output_path, report)
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(benchmark(database_path=args.database, train_path=args.train_file, output_path=args.output, sample_size=args.sample_size, workers=args.workers), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
