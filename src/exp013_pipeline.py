"""EXP-013 command line: selective late-interaction document retrieval.

The first four stages are intentionally separated from evaluation.  They are a
one-time corpus build and have hard integrity/resource gates before any costly
OOF experiment is allowed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exp012b_core import atomic_json, stage_run
from exp013_core import DEFAULT_DIMENSIONS, DEFAULT_MODEL, require_exp013_success, v3_preflight
from exp013_candidates import audit_candidate_oracle, build_candidate_union, build_query_memory, retrieve_colbert_prototypes
from exp013_late_interaction import build_document_prototypes, encode_colbert_queries, encode_colbert_v3
from exp013_model import colbert_encode, load_colbert_model
from exp013_ranker import build_feature_rows, score_exact_maxsim, train_lambdamart_oof


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cache" / "exp013_slid"
DEFAULT_RESULTS = ROOT / "results" / "exp013_slid"
DEFAULT_V3 = ROOT / "cache" / "structural_v3"
DEFAULT_AUDIT = ROOT / "results" / "audits" / "chunker_v3" / "audit.json"
DEFAULT_TRAIN = ROOT / "public_test_dataset" / "train.json"
DEFAULT_PUBLIC = ROOT / "public_test_dataset" / "public-official.json"
DEFAULT_FOLDS = ROOT / "cache" / "cv_folds.json"
DEFAULT_REFERENCE_RANKINGS = ROOT / "cache" / "exp012b_v3" / "rankings"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", required=True, choices=(
        "preflight", "prepare-models", "encode-colbert-v3", "build-document-prototypes", "status",
        "encode-colbert-queries", "retrieve-colbert-prototypes", "build-query-memory", "build-candidate-union",
        "audit-candidate-oracle", "score-exact-maxsim", "build-features", "train-lambdamart-oof",
    ))
    value.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    value.add_argument("--v3-dir", type=Path, default=DEFAULT_V3)
    value.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    value.add_argument("--model", default=DEFAULT_MODEL)
    value.add_argument("--device", default="cuda")
    value.add_argument("--batch-size", type=int, default=12)
    value.add_argument("--split", choices=("train", "public"), default="train")
    value.add_argument("--reference-rankings", type=Path, default=DEFAULT_REFERENCE_RANKINGS,
        help="Existing EXP-012 source-ranking directory; only read as BM25/BGE candidate channels.")
    value.add_argument("--allow-download", action="store_true",
        help="Permit Hugging Face to download the model.  Never implied by another flag.")
    value.add_argument("--trust-remote-code", action="store_true",
        help="Permit Jina's model implementation to execute. Required only for model stages.")
    return value


def model_preflight(*, model_name: str, device: str, allow_download: bool, trust_remote_code: bool) -> dict:
    """Load once, verify a token-level model surface, then promptly release VRAM."""
    if not trust_remote_code:
        raise RuntimeError("Jina ColBERT v2 requires explicit --trust-remote-code; no code was loaded.")
    model, tokenizer = load_colbert_model(model_name, device=device, allow_download=allow_download,
                                          trust_remote_code=trust_remote_code)
    import torch
    vectors, _, _ = colbert_encode(
        model, tokenizer, ["quy định về điều kiện áp dụng"], task="retrieval.query", device=device,
        dimensions=DEFAULT_DIMENSIONS,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    payload = {"model": model_name, "device": device, "parameters": parameter_count,
               "tokenizer_class": type(tokenizer).__name__, "model_class": type(model).__name__,
               "expected_dimensions": DEFAULT_DIMENSIONS, "smoke_shape": list(vectors[0].shape)}
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def main() -> int:
    args = parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.stage == "status":
        for name in ("preflight", "models", "colbert_leaves", "document_prototypes", "colbert_queries", "colbert_rankings", "query_memory", "candidates"):
            path = args.cache_root / name / "status.json"
            if path.exists():
                print(f"{name}: {path.read_text(encoding='utf-8').strip()}")
            else:
                print(f"{name}: NOT_STARTED")
        return 0
    if args.stage == "preflight":
        output = args.cache_root / "preflight"
        v3 = v3_preflight(args.v3_dir, args.audit)
        with stage_run(output, "preflight", total=1, v3_fingerprint=v3["content_fingerprint"]) as logger:
            atomic_json(output / "preflight.json", {"v3_fingerprint": v3["content_fingerprint"], "counts": v3["counts"]})
            logger.log(f"v3_fingerprint={v3['content_fingerprint']} chunks={v3['counts']['chunks']}")
        return 0
    v3 = v3_preflight(args.v3_dir, args.audit)
    require_exp013_success(args.cache_root / "preflight", v3["content_fingerprint"])
    if args.stage == "prepare-models":
        output = args.cache_root / "models"
        with stage_run(output, "prepare-models", total=1, v3_fingerprint=v3["content_fingerprint"]) as logger:
            report = model_preflight(model_name=args.model, device=args.device, allow_download=args.allow_download,
                                    trust_remote_code=args.trust_remote_code)
            atomic_json(output / "model_report.json", report)
            logger.log(f"model={args.model} parameters={report['parameters']}")
        return 0
    require_exp013_success(args.cache_root / "models", v3["content_fingerprint"])
    if args.stage == "encode-colbert-v3":
        encode_colbert_v3(args.v3_dir, args.cache_root / "colbert_leaves", v3_fingerprint=v3["content_fingerprint"],
            model_name=args.model, device=args.device, batch_size=args.batch_size,
            allow_download=args.allow_download, trust_remote_code=args.trust_remote_code)
        return 0
    if args.stage == "build-document-prototypes":
        require_exp013_success(args.cache_root / "colbert_leaves", v3["content_fingerprint"])
        build_document_prototypes(args.cache_root / "colbert_leaves", args.cache_root / "document_prototypes",
                                  v3_fingerprint=v3["content_fingerprint"])
        return 0
    query_path = DEFAULT_TRAIN if args.split == "train" else DEFAULT_PUBLIC
    query_dir = args.cache_root / "colbert_queries" / args.split
    ranking_dir = args.cache_root / "colbert_rankings" / args.split
    memory_dir = args.cache_root / "query_memory"
    candidate_dir = args.cache_root / "candidates"
    exact_dir = args.cache_root / "exact" / args.split
    feature_dir = args.cache_root / "features" / args.split
    oracle_dir = args.results_root / "candidate_oracle"
    if args.stage == "encode-colbert-queries":
        encode_colbert_queries(query_path, query_dir, v3_fingerprint=v3["content_fingerprint"],
            model_name=args.model, device=args.device, batch_size=max(1, args.batch_size),
            allow_download=args.allow_download, trust_remote_code=args.trust_remote_code)
        return 0
    if args.stage == "retrieve-colbert-prototypes":
        require_exp013_success(query_dir, v3["content_fingerprint"])
        require_exp013_success(args.cache_root / "document_prototypes", v3["content_fingerprint"])
        retrieve_colbert_prototypes(query_dir, args.cache_root / "document_prototypes", ranking_dir,
            v3_fingerprint=v3["content_fingerprint"], device=args.device, batch_size=min(4, max(1, args.batch_size)))
        return 0
    if args.stage == "build-query-memory":
        build_query_memory(split=args.split, train_path=DEFAULT_TRAIN, query_path=query_path,
            folds_path=DEFAULT_FOLDS, output_dir=memory_dir / args.split,
            v3_fingerprint=v3["content_fingerprint"])
        return 0
    if args.stage == "build-candidate-union":
        require_exp013_success(ranking_dir, v3["content_fingerprint"])
        require_exp013_success(memory_dir / args.split, v3["content_fingerprint"])
        source_path = args.reference_rankings / args.split / "source_rankings.jsonl"
        build_candidate_union(split=args.split, source_rankings_path=source_path,
            colbert_rankings_path=ranking_dir / "rankings.jsonl",
            memory_rankings_path=memory_dir / args.split / f"{args.split}_rankings.jsonl",
            output_dir=candidate_dir / args.split, v3_fingerprint=v3["content_fingerprint"])
        return 0
    if args.stage == "audit-candidate-oracle":
        if args.split != "train":
            raise RuntimeError("Candidate oracle requires --split train")
        require_exp013_success(candidate_dir / args.split, v3["content_fingerprint"])
        audit_candidate_oracle(candidates_path=candidate_dir / "train" / "train_candidates.jsonl",
            train_path=DEFAULT_TRAIN, output_dir=oracle_dir, v3_fingerprint=v3["content_fingerprint"])
        return 0
    if args.stage == "score-exact-maxsim":
        if args.split == "train":
            require_exp013_success(oracle_dir, v3["content_fingerprint"])
            oracle = json.loads((oracle_dir / "candidate_oracle.json").read_text(encoding="utf-8"))
            if not oracle.get("pass"):
                raise RuntimeError("Exact MaxSim blocked: candidate Recall@64 oracle did not pass")
        require_exp013_success(candidate_dir / args.split, v3["content_fingerprint"])
        require_exp013_success(query_dir, v3["content_fingerprint"])
        require_exp013_success(args.cache_root / "colbert_leaves", v3["content_fingerprint"])
        score_exact_maxsim(candidates_path=candidate_dir / args.split / f"{args.split}_candidates.jsonl",
            query_dir=query_dir, leaf_dir=args.cache_root / "colbert_leaves", output_dir=exact_dir,
            v3_fingerprint=v3["content_fingerprint"], device=args.device)
        return 0
    if args.stage == "build-features":
        require_exp013_success(candidate_dir / args.split, v3["content_fingerprint"])
        require_exp013_success(exact_dir, v3["content_fingerprint"])
        output = feature_dir / "features.jsonl"
        with stage_run(feature_dir, "build-features", total=1, v3_fingerprint=v3["content_fingerprint"]) as logger:
            count = build_feature_rows(candidate_dir / args.split / f"{args.split}_candidates.jsonl",
                exact_dir / "exact_scores.jsonl", output)
            from exp013_core import write_stage_manifest
            write_stage_manifest(feature_dir, stage="build-features", v3_fingerprint=v3["content_fingerprint"],
                config={"split": args.split}, files=[output], counts={"feature_rows": count})
            logger.log(f"feature_rows={count}")
        return 0
    if args.stage == "train-lambdamart-oof":
        if args.split != "train":
            raise RuntimeError("train-lambdamart-oof only accepts --split train")
        require_exp013_success(feature_dir, v3["content_fingerprint"])
        train_lambdamart_oof(features_path=feature_dir / "features.jsonl", train_path=DEFAULT_TRAIN,
            folds_path=DEFAULT_FOLDS, output_dir=args.results_root / "lambdamart_oof", v3_fingerprint=v3["content_fingerprint"])
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
