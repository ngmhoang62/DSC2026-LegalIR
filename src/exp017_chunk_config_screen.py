"""Screen EXP-017 chunk corpora with a fixed parent fixture and VietLegal-E5."""

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
RESULTS = ROOT / "results" / "exp017_chunk_config"
FIXTURE = ROOT / "cache" / "exp015_model_screen" / "stage2"
SEED = 17017
MAX_CHUNKS_PER_DOCUMENT = 4
MODEL_KEY = "vietlegal_e5"
CORPORA = {
    "b384_o32": ROOT / "cache" / "exp016_split_article_v3",
    "b512_o32": ROOT / "cache" / "exp017_b512_o32",
    "b768_o32": ROOT / "cache" / "exp017_b768_o32",
}


def read_jsonl(path: Path):
    yield from base.read_jsonl(path)


def fixture_parent_ids() -> tuple[list[dict[str, Any]], set[str]]:
    queries = list(read_jsonl(FIXTURE / "queries.jsonl"))
    parents = {str(row["doc_id"]) for row in read_jsonl(FIXTURE / "chunks.jsonl")}
    return queries, parents


def select_chunks(corpus: Path, parents: set[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in read_jsonl(corpus / "chunks.jsonl"):
        doc_id = str(row["doc_id"])
        if doc_id in parents:
            grouped[doc_id].append((base.digest({"seed": SEED, "chunk": row["chunk_id"]}), row))
    missing = sorted(parents - set(grouped))
    if missing:
        raise RuntimeError(f"Candidate parents missing from {corpus}: {missing[:10]}")
    rows = [row for doc_id in sorted(parents) for _, row in sorted(grouped[doc_id])[:MAX_CHUNKS_PER_DOCUMENT]]
    return sorted(rows, key=lambda row: str(row["chunk_id"]))


def evaluate(key: str, *, device: str, batch_size: int, force: bool) -> dict[str, Any]:
    if key not in CORPORA:
        raise ValueError(f"Unknown key {key}")
    out = RESULTS / f"{key}.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    corpus = CORPORA[key]
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    queries, parents = fixture_parent_ids()
    chunks = select_chunks(corpus, parents)
    spec = base.MODELS[MODEL_KEY]
    encoder = base.create_encoder(spec, device, allow_download=False)
    doc_texts = [spec.document_prefix + str(row["retrieval_text"]) for row in chunks]
    query_texts = [spec.query_prefix + str(row["question"]) for row in queries]
    start = time.perf_counter()
    vectors, actual_batch = base.encode_oom_safe(encoder, doc_texts, batch_size, device)
    q_vectors, actual_batch = base.encode_oom_safe(encoder, query_texts, actual_batch, device)
    elapsed = time.perf_counter() - start
    result = base.evaluate(queries, chunks, q_vectors, vectors)
    counts = encoder.token_counts(doc_texts)
    report = {"experiment": "exp017_chunk_config", "interpretation": "fold_0_development_only_not_oof", "corpus_key": key, "corpus_fingerprint": manifest["content_fingerprint"], "chunk_config": {key: manifest[key] for key in ("max_passage_tokens", "token_window", "token_overlap", "heading_policy")}, "fixture": {"queries": len(queries), "parents": len(parents), "chunks": len(chunks), "selection_seed": SEED, "max_chunks_per_document": MAX_CHUNKS_PER_DOCUMENT}, "runtime": {"elapsed_seconds": round(elapsed, 3), "effective_batch_size": actual_batch, "texts_per_second": round((len(doc_texts)+len(query_texts))/elapsed, 3)}, "tokenization": {"p95": float(np.percentile(counts,95)), "truncated": int(sum(count > spec.max_length for count in counts))}, **result}
    RESULTS.mkdir(parents=True, exist_ok=True)
    base.write_json(out, report)
    return report


def summarize() -> Path:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in RESULTS.glob("b*_o*.json")]
    reports.sort(key=lambda row: row["chunk_config"]["max_passage_tokens"])
    lines = ["# EXP-017 phase-1 chunk-size screen", "", "> Fold-0 development screen only; tokenizer, parser and overlap are fixed.", "", "| Config | Max/window/overlap | Fixture chunks | R@5 | R@20 | R@100 | MRR@5 | sec | E5 trunc. |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in reports:
        cfg, m, rt, tok = row["chunk_config"], row["metrics"], row["runtime"], row["tokenization"]
        lines.append(f"| `{row['corpus_key']}` | {cfg['max_passage_tokens']}/{cfg['token_window']}/{cfg['token_overlap']} | {row['fixture']['chunks']} | {m['recall_at_5']:.4f} | {m['recall_at_20']:.4f} | {m['recall_at_100']:.4f} | {m['mrr_at_5']:.4f} | {rt['elapsed_seconds']:.1f} | {tok['truncated']} |")
    output = RESULTS / "phase1_summary.md"
    output.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("run-one", "summarize"))
    parser.add_argument("--config", choices=tuple(CORPORA))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "run-one":
        if not args.config: parser.error("--config is required")
        print(json.dumps(evaluate(args.config, device=args.device, batch_size=args.batch_size, force=args.force), ensure_ascii=False, indent=2))
    else: print(summarize())


if __name__ == "__main__": main()
