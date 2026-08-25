"""Build auditable sparse-retrieval artifacts over the final structural corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from exp012b_bm25 import build_fts5_index, tokenize_v3_fields
from exp021_sparse_audit import audit_sparse_inputs
from exp021_sparse_error_audit import audit_sparse_errors
from exp021_title_audit import audit_title_lexicon
from exp021_aggregation_fusion import audit_aggregation_fusion
from exp021_aggregation_oof import nested_aggregation_oof
from exp021_sparse_retrieve import retrieve_and_evaluate_bm25


def _utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("input-audit", "tokenize", "build-index", "retrieve", "error-audit", "title-audit", "aggregation-fusion", "aggregation-oof", "status"))
    parser.add_argument("--v3-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--field-mode", choices=("passage_only", "passage_hierarchy", "passage_hierarchy_scope", "all_fields", "passage_title_folded"), default="passage_only")
    parser.add_argument("--processed-contexts", type=Path)
    parser.add_argument("--preprocessing-manifest", type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile", choices=("balanced", "legal_structure", "heading_priority", "title_auxiliary"), default="balanced")
    parser.add_argument("--query-mode", choices=("default", "title_folded"), default="default")
    parser.add_argument("--aggregation", choices=("rrf3", "first_passage"), default="rrf3")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    variant_dir = args.cache_root / args.field_mode
    if args.stage == "input-audit":
        required = (args.processed_contexts, args.preprocessing_manifest, args.train_file)
        if any(value is None for value in required):
            parser.error("input-audit requires --processed-contexts, --preprocessing-manifest, and --train-file")
        result = audit_sparse_inputs(
            v3_dir=args.v3_dir,
            processed_contexts=args.processed_contexts,
            preprocessing_manifest=args.preprocessing_manifest,
            train_path=args.train_file,
            output_dir=args.results_root / "input_audit",
        )
    elif args.stage == "tokenize":
        result = tokenize_v3_fields(
            args.v3_dir, variant_dir / "bm25_fields", workers=args.workers,
            field_mode=args.field_mode, resume=args.resume,
        )
    elif args.stage == "build-index":
        result = build_fts5_index(
            variant_dir / "bm25_fields", variant_dir / "fts5", resume=args.resume,
        )
    elif args.stage == "retrieve":
        if args.train_file is None:
            parser.error("retrieve requires --train-file")
        ranking_dir = variant_dir / (
            "rankings_train" if args.aggregation == "rrf3" else f"rankings_train_{args.aggregation}"
        )
        result = retrieve_and_evaluate_bm25(
            database_path=variant_dir / "fts5" / "bm25_v3.sqlite", v3_dir=args.v3_dir,
            train_path=args.train_file, output_dir=ranking_dir,
            profile=args.profile, workers=args.workers, query_mode=args.query_mode,
            aggregation=args.aggregation,
        )
    elif args.stage == "error-audit":
        if args.train_file is None:
            parser.error("error-audit requires --train-file")
        report = audit_sparse_errors(
            rankings_path=variant_dir / "rankings_train" / "rankings.jsonl",
            train_path=args.train_file, v3_dir=args.v3_dir,
            output_dir=args.results_root / args.field_mode / "error_audit",
        )
        result = {
            "report_path": str(args.results_root / args.field_mode / "error_audit" / "error_audit.json"),
            "queries": report["queries"],
            "first_gold_rank_buckets": report["first_gold_rank_buckets"],
        }
    elif args.stage == "title-audit":
        if args.train_file is None:
            parser.error("title-audit requires --train-file")
        result = audit_title_lexicon(
            v3_dir=args.v3_dir, train_path=args.train_file,
            output_dir=args.results_root / "title_lexicon_audit",
        )
    elif args.stage == "aggregation-fusion":
        if args.train_file is None:
            parser.error("aggregation-fusion requires --train-file")
        result = audit_aggregation_fusion(
            rrf_path=args.cache_root / "passage_hierarchy" / "rankings_train" / "rankings.jsonl",
            first_path=args.cache_root / "passage_hierarchy" / "rankings_train_first_passage" / "rankings.jsonl",
            train_path=args.train_file,
            output_path=args.results_root / "aggregation_fusion.json",
        )
    elif args.stage == "aggregation-oof":
        if args.train_file is None:
            parser.error("aggregation-oof requires --train-file")
        result = nested_aggregation_oof(
            rrf_path=args.cache_root / "passage_hierarchy" / "rankings_train" / "rankings.jsonl",
            first_path=args.cache_root / "passage_hierarchy" / "rankings_train_first_passage" / "rankings.jsonl",
            train_path=args.train_file, folds_path=args.cache_root.parent / "cv_folds.json",
            output_path=args.results_root / "aggregation_oof.json",
        )
    else:
        reports = []
        for status_path in sorted(args.cache_root.glob("**/status.json")):
            status = json.loads(status_path.read_text(encoding="utf-8"))
            reports.append({"stage_dir": str(status_path.parent.relative_to(args.cache_root)), **status})
        print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)
        return 0
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
