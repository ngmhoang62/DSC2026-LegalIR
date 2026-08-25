"""Screen 512-token Qwen3 Harrier/LAL stacks on the fixed EXP-015 fixture."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import exp015_model_screen as base


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "exp020_qwen_stack_screen"
FIXTURE = ROOT / "cache" / "exp015_model_screen" / "stage2"
CORPUS = ROOT / "cache" / "exp020_qwen3_511_o64"
E5_BASELINE = ROOT / "results" / "exp019_e5_overlap" / "e5_508_o64.json"
SEED = 18018
MAX_PER_DOCUMENT = 4
MODEL_KEYS = ("vietlegal_harrier_0_6b", "vnlegal_lal")
MODEL_LIMIT = 512
CONTENT_FIELDS = ("retrieval_text", "raw_text")


def parents_and_queries() -> tuple[list[dict[str, Any]], set[str]]:
    queries = list(base.read_jsonl(FIXTURE / "queries.jsonl"))
    parents = {str(row["doc_id"]) for row in base.read_jsonl(FIXTURE / "chunks.jsonl")}
    return queries, parents


def fixture_chunks(parents: set[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in base.read_jsonl(CORPUS / "chunks.jsonl"):
        doc_id = str(row["doc_id"])
        if doc_id in parents:
            grouped[doc_id].append((base.digest({"seed": SEED, "chunk_id": row["chunk_id"]}), row))
    missing = sorted(parents - set(grouped))
    if missing:
        raise RuntimeError(f"Missing fixture parents from Qwen corpus: {missing[:10]}")
    selected = [
        row
        for doc_id in sorted(parents)
        for _, row in sorted(grouped[doc_id])[:MAX_PER_DOCUMENT]
    ]
    return sorted(selected, key=lambda row: str(row["chunk_id"]))


def spec_for(key: str) -> base.ModelSpec:
    if key not in MODEL_KEYS:
        raise ValueError(f"Unknown model key: {key}")
    # LAL's 2048-token native context is intentionally capped for this fair,
    # GPU-practical comparison with the 512-token Harrier.
    return replace(base.MODELS[key], max_length=MODEL_LIMIT)


def paired_rank_movement(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int]:
    candidate_ranks = {str(row["qid"]): int(row["first_relevant_rank"]) for row in candidate["per_query"]}
    baseline_ranks = {str(row["qid"]): int(row["first_relevant_rank"]) for row in baseline["per_query"]}
    if set(candidate_ranks) != set(baseline_ranks):
        raise RuntimeError("Qwen and E5 reports do not contain identical query IDs")
    improved = worsened = unchanged = entered_top5 = left_top5 = 0
    for qid, rank in candidate_ranks.items():
        reference = baseline_ranks[qid]
        improved += rank < reference
        worsened += rank > reference
        unchanged += rank == reference
        entered_top5 += rank <= 5 < reference
        left_top5 += reference <= 5 < rank
    return {
        "improved_first_relevant_rank": improved,
        "worsened_first_relevant_rank": worsened,
        "unchanged_first_relevant_rank": unchanged,
        "entered_top5": entered_top5,
        "left_top5": left_top5,
    }


def report_path(key: str, content_field: str) -> Path:
    suffix = "" if content_field == "retrieval_text" else f"__{content_field}"
    return RESULTS / f"{key}{suffix}.json"


def run(key: str, content_field: str, *, device: str, batch_size: int, force: bool) -> dict[str, Any]:
    if content_field not in CONTENT_FIELDS:
        raise ValueError(f"Unknown content field: {content_field}")
    output = report_path(key, content_field)
    if output.exists() and not force:
        return json.loads(output.read_text(encoding="utf-8"))
    if not E5_BASELINE.exists():
        raise FileNotFoundError(f"Required E5-508/o64 baseline missing: {E5_BASELINE}")
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    queries, parents = parents_and_queries()
    chunks = fixture_chunks(parents)
    spec = spec_for(key)
    documents = [str(row[content_field]) for row in chunks]
    questions = [spec.query_prefix + str(row["question"]) for row in queries]
    encoder = base.create_encoder(spec, device, allow_download=False)
    document_counts = encoder.token_counts(documents)
    question_counts = encoder.token_counts(questions)
    over_limit = int(sum(count > MODEL_LIMIT for count in document_counts))
    if over_limit:
        raise RuntimeError(f"Qwen input truncation invariant failed: {over_limit} documents > {MODEL_LIMIT}")
    start = time.perf_counter()
    document_vectors, actual_batch = base.encode_oom_safe(encoder, documents, batch_size, device)
    query_vectors, actual_batch = base.encode_oom_safe(encoder, questions, actual_batch, device)
    result = base.evaluate(queries, chunks, query_vectors, document_vectors)
    baseline = json.loads(E5_BASELINE.read_text(encoding="utf-8"))
    report = {
        "experiment": "exp020_qwen_stack_screen",
        "interpretation": "fold_0_development_only_not_oof",
        "model": {
            "key": spec.key,
            "repo_id": spec.repo_id,
            "backend": spec.backend,
            "effective_max_length": MODEL_LIMIT,
            "query_prefix": spec.query_prefix,
            "document_prefix": spec.document_prefix,
        },
        "document_content_field": content_field,
        "corpus": {
            "path": str(CORPUS),
            "content_fingerprint": manifest["content_fingerprint"],
            "chunk_config": {field: manifest[field] for field in ("tokenizer", "max_passage_tokens", "token_window", "token_overlap", "heading_policy")},
        },
        "fixture": {"queries": len(queries), "parents": len(parents), "chunks": len(chunks), "seed": SEED, "max_chunks_per_document": MAX_PER_DOCUMENT},
        "runtime": {"elapsed_seconds": round(time.perf_counter() - start, 3), "effective_batch_size": actual_batch},
        "qwen_input_tokens": {
            "document_p95": float(np.percentile(document_counts, 95)),
            "document_max": int(max(document_counts)),
            "document_over_512": over_limit,
            "query_p95": float(np.percentile(question_counts, 95)),
            "query_max": int(max(question_counts)),
            "query_over_512": int(sum(count > MODEL_LIMIT for count in question_counts)),
        },
        "paired_vs_e5_508_o64": paired_rank_movement(result, baseline),
        **result,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    base.write_json(output, report)
    del encoder, document_vectors, query_vectors
    gc.collect()
    if device.startswith("cuda"):
        import torch
        torch.cuda.empty_cache()
    return report


def summarize() -> Path:
    reports = [
        json.loads(report_path(key, content_field).read_text(encoding="utf-8"))
        for key in MODEL_KEYS
        for content_field in CONTENT_FIELDS
        if report_path(key, content_field).exists()
    ]
    for row in reports:
        # Reports emitted before the raw-text ablation did not need this
        # explicit field; preserve them as the original retrieval-text runs.
        row.setdefault("document_content_field", "retrieval_text")
    reports.sort(key=lambda row: (-row["metrics"]["recall_at_5"], row["model"]["key"], row["document_content_field"]))
    lines = [
        "# EXP-020 Qwen3 512-token stack screen", "",
        "> Fold-0 development screen only; both Qwen stacks use the same parser, Qwen3 chunk corpus, 512-input cap, parent fixture, and max-over-chunks aggregation.", "",
        "| Model | Document text | R@5 | R@20 | R@100 | MRR@5 | Doc input max | Truncated docs | Batch | sec | Entered/left top-5 vs E5 |", 
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reports:
        metrics, tokens, paired = row["metrics"], row["qwen_input_tokens"], row["paired_vs_e5_508_o64"]
        lines.append(
            f"| `{row['model']['key']}` | `{row['document_content_field']}` | {metrics['recall_at_5']:.4f} | {metrics['recall_at_20']:.4f} | {metrics['recall_at_100']:.4f} | {metrics['mrr_at_5']:.4f} | {tokens['document_max']} | {tokens['document_over_512']} | {row['runtime']['effective_batch_size']} | {row['runtime']['elapsed_seconds']:.1f} | {paired['entered_top5']}/{paired['left_top5']} |"
        )
    output = RESULTS / "summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("preflight", "run-one", "summarize"))
    parser.add_argument("--model", choices=MODEL_KEYS)
    parser.add_argument("--content-field", choices=CONTENT_FIELDS, default="retrieval_text")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "preflight":
        required = [CORPUS / "manifest.json", CORPUS / "chunks.jsonl", FIXTURE / "queries.jsonl", FIXTURE / "chunks.jsonl", E5_BASELINE]
        missing = [str(path) for path in required if not path.exists()]
        print(json.dumps({"missing_inputs": missing, "model_limit": MODEL_LIMIT, "models": list(MODEL_KEYS)}, ensure_ascii=False, indent=2))
    elif args.stage == "run-one":
        if not args.model:
            parser.error("--model is required with --stage run-one")
        print(json.dumps(run(args.model, args.content_field, device=args.device, batch_size=args.batch_size, force=args.force), ensure_ascii=False, indent=2))
    else:
        print(summarize())


if __name__ == "__main__":
    main()
