"""Evaluate E5-tokenizer chunk-size corpora on the fixed EXP-015 stage-2 fixture."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import exp015_model_screen as base

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "exp018_e5_chunk_config"
FIXTURE = ROOT / "cache" / "exp015_model_screen" / "stage2"
SEED = 18018
MAX_PER_DOCUMENT = 4
CORPORA = {
    "e5_384_o32": ROOT / "cache" / "exp018_e5_384_o32",
    "e5_508_o32": ROOT / "cache" / "exp018_e5_508_o32",
}


def parents_and_queries() -> tuple[list[dict[str, Any]], set[str]]:
    queries = list(base.read_jsonl(FIXTURE / "queries.jsonl"))
    parents = {str(row["doc_id"]) for row in base.read_jsonl(FIXTURE / "chunks.jsonl")}
    return queries, parents


def fixture_chunks(corpus: Path, parents: set[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in base.read_jsonl(corpus / "chunks.jsonl"):
        doc_id = str(row["doc_id"])
        if doc_id in parents:
            grouped[doc_id].append((base.digest({"seed": SEED, "chunk_id": row["chunk_id"]}), row))
    missing = parents - set(grouped)
    if missing:
        raise RuntimeError(f"Missing candidate parents: {sorted(missing)[:10]}")
    return sorted([row for doc_id in sorted(parents) for _, row in sorted(grouped[doc_id])[:MAX_PER_DOCUMENT]], key=lambda row: str(row["chunk_id"]))


def run(key: str, device: str, batch_size: int, force: bool) -> dict[str, Any]:
    output = RESULTS / f"{key}.json"
    if output.exists() and not force:
        return json.loads(output.read_text(encoding="utf-8"))
    corpus = CORPORA[key]
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    queries, parents = parents_and_queries()
    chunks = fixture_chunks(corpus, parents)
    spec = base.MODELS["vietlegal_e5"]
    encoder = base.create_encoder(spec, device, allow_download=False)
    documents = [spec.document_prefix + str(row["retrieval_text"]) for row in chunks]
    questions = [spec.query_prefix + str(row["question"]) for row in queries]
    start = time.perf_counter()
    doc_vectors, actual_batch = base.encode_oom_safe(encoder, documents, batch_size, device)
    query_vectors, actual_batch = base.encode_oom_safe(encoder, questions, actual_batch, device)
    result = base.evaluate(queries, chunks, query_vectors, doc_vectors)
    counts = encoder.token_counts(documents)
    over_limit = int(sum(count > spec.max_length for count in counts))
    if over_limit:
        raise RuntimeError(f"E5 input truncation invariant failed: {over_limit} fixture chunks > {spec.max_length}")
    report = {"experiment": "exp018_e5_chunk_config", "interpretation": "fold_0_development_only_not_oof", "corpus_key": key, "corpus_fingerprint": manifest["content_fingerprint"], "chunk_config": {field: manifest[field] for field in ("tokenizer", "max_passage_tokens", "token_window", "token_overlap", "heading_policy")}, "fixture": {"queries": len(queries), "parents": len(parents), "chunks": len(chunks), "seed": SEED, "max_chunks_per_document": MAX_PER_DOCUMENT}, "runtime": {"elapsed_seconds": round(time.perf_counter()-start,3), "effective_batch_size": actual_batch}, "e5_input_tokens": {"p95": float(np.percentile(counts,95)), "max": int(max(counts)), "over_512": over_limit}, **result}
    RESULTS.mkdir(parents=True, exist_ok=True)
    base.write_json(output, report)
    return report


def summarize() -> Path:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in RESULTS.glob("e5_*.json")]
    reports.sort(key=lambda row: row["chunk_config"]["max_passage_tokens"])
    lines = ["# EXP-018 E5-tokenizer chunk-size screen", "", "> Fold-0 development screen only. All E5 inputs satisfy the 512-token invariant.", "", "| Config | text max/window/overlap | Fixture chunks | E5 input max | R@5 | R@20 | R@100 | MRR@5 | sec |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in reports:
        cfg, m = row["chunk_config"], row["metrics"]
        lines.append(f"| `{row['corpus_key']}` | {cfg['max_passage_tokens']}/{cfg['token_window']}/{cfg['token_overlap']} | {row['fixture']['chunks']} | {row['e5_input_tokens']['max']} | {m['recall_at_5']:.4f} | {m['recall_at_20']:.4f} | {m['recall_at_100']:.4f} | {m['mrr_at_5']:.4f} | {row['runtime']['elapsed_seconds']:.1f} |")
    output = RESULTS / "phase1_summary.md"
    output.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run-one", "summarize"), required=True)
    parser.add_argument("--config", choices=tuple(CORPORA))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "run-one":
        if not args.config: parser.error("--config is required")
        print(json.dumps(run(args.config,args.device,args.batch_size,args.force),ensure_ascii=False,indent=2))
    else: print(summarize())


if __name__ == "__main__": main()
