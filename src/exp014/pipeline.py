"""EXP-014 Neuro-Symbolic Legal Retrieval command-line pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

# Ensure src/ is in sys.path for root module imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exp012b_core import atomic_json, read_jsonl, stage_run
from exp012b_tuning import load_folds
from exp014.candidates import audit_candidates, build_candidates, build_query_memory
from exp014.capsules import build_capsules
from exp014.core import BGE_RERANKER_MODEL, DEFAULT_RERANKER, preflight, require_success, write_manifest
from exp014.fusion import _apply_dynamic_cutoff, _fuse, evaluate_oof
from exp014.ranker import audit_shortlist, score_preranker, train_final_preranker, train_preranker_oof
from exp014.reranker import score_bge_lora, score_reranker
from exp014.train_lora import train_lora_bge_reranker


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE, DEFAULT_RESULTS = ROOT / "cache" / "exp014", ROOT / "results" / "exp014"
DEFAULT_V3, DEFAULT_AUDIT = ROOT / "cache" / "structural_v3", ROOT / "results" / "audits" / "chunker_v3" / "audit.json"
DEFAULT_EXP012 = ROOT / "cache" / "exp012b_v3"
TRAIN, PUBLIC, FOLDS = ROOT / "public_test_dataset" / "train.json", ROOT / "public_test_dataset" / "public-official.json", ROOT / "cache" / "cv_folds.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", required=True, choices=(
        "preflight", "build-query-memory", "build-candidates", "audit-candidates",
        "train-preranker-oof", "audit-shortlist", "build-capsules", "train-lora",
        "score-qwen-oof", "score-public", "evaluate-oof", "train-final", "public-submission", "status"
    ))
    value.add_argument("--split", choices=("train", "public"), default="train")
    value.add_argument("--fold", type=int, choices=range(5), default=0)
    value.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    value.add_argument("--v3-dir", type=Path, default=DEFAULT_V3)
    value.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    value.add_argument("--upstream-cache", type=Path, default=DEFAULT_EXP012)
    value.add_argument("--model", default=DEFAULT_RERANKER)
    value.add_argument("--device", default="cuda")
    value.add_argument("--allow-download", action="store_true")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--max-length", type=int, default=768)
    return value


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "rankings": args.upstream_cache / "rankings",
        "lookup": args.upstream_cache / "chunk_lookup" / "chunk_offsets.sqlite",
        "memory": args.cache_root / "query_memory",
        "candidates": args.cache_root / "candidates",
        "preranker": args.cache_root / "preranker",
        "capsules": args.cache_root / "capsules",
        "lora": args.cache_root / "lora",
        "qwen": args.cache_root / "qwen",
    }


def _shortlist_k(paths: dict[str, Path]) -> int:
    audit_file = paths["preranker"] / "shortlist_audit" / "shortlist_audit.json"
    if not audit_file.exists():
        return 25
    return int(json.loads(audit_file.read_text(encoding="utf-8")).get("shortlist_k", 25))


def _status(cache: Path) -> None:
    for path in sorted(cache.rglob("status.json")):
        print(f"{path.relative_to(cache)}: {path.read_text(encoding='utf-8').strip()}")


def main() -> int:
    args = parser().parse_args()
    paths = _paths(args)
    
    if args.stage == "status":
        _status(args.cache_root)
        return 0
        
    v3 = preflight(args.v3_dir, args.audit)
    fingerprint = v3["content_fingerprint"]
    
    if args.stage == "preflight":
        out = args.cache_root / "preflight"
        with stage_run(out, "preflight", total=1, v3_fingerprint=fingerprint) as logger:
            path = out / "preflight.json"
            atomic_json(path, {"v3_fingerprint": fingerprint, "counts": v3["counts"]})
            write_manifest(out, stage="preflight", v3_fingerprint=fingerprint, config={}, files=[path], counts={"documents": int(v3["counts"]["documents"])})
            logger.log(f"v3_fingerprint={fingerprint}")
        return 0
        
    require_success(args.cache_root / "preflight", v3_fingerprint=fingerprint)
    split_path = TRAIN if args.split == "train" else PUBLIC
    source = paths["rankings"] / args.split / "source_rankings.jsonl"
    memory = paths["memory"] / args.split / f"{args.split}_memory.jsonl"
    candidates = paths["candidates"] / args.split / f"{args.split}_candidates.jsonl"
    
    if args.stage == "build-query-memory":
        build_query_memory(split=args.split, train_path=TRAIN, query_path=split_path, folds_path=FOLDS, output_dir=paths["memory"] / args.split, v3_fingerprint=fingerprint)
        return 0
        
    if args.stage == "build-candidates":
        require_success(paths["memory"] / args.split, v3_fingerprint=fingerprint)
        build_candidates(split=args.split, rankings_path=source, memory_path=memory, v3_dir=args.v3_dir, output_dir=paths["candidates"] / args.split, v3_fingerprint=fingerprint)
        return 0
        
    if args.stage == "audit-candidates":
        if args.split != "train":
            raise RuntimeError("candidate audit requires --split train")
        require_success(paths["candidates"] / "train", v3_fingerprint=fingerprint)
        audit_candidates(candidates_path=candidates, memory_path=memory, train_path=TRAIN, folds_path=FOLDS, v3_dir=args.v3_dir, output_dir=args.results_root / "candidate_audit", v3_fingerprint=fingerprint)
        return 0
        
    if args.stage == "train-preranker-oof":
        require_success(args.results_root / "candidate_audit", v3_fingerprint=fingerprint)
        train_preranker_oof(candidates_path=candidates, train_path=TRAIN, folds_path=FOLDS, output_dir=paths["preranker"] / "oof", v3_fingerprint=fingerprint)
        return 0
        
    if args.stage == "audit-shortlist":
        require_success(paths["preranker"] / "oof", v3_fingerprint=fingerprint)
        audit_shortlist(predictions_path=paths["preranker"] / "oof" / "oof_predictions.jsonl", train_path=TRAIN, output_dir=paths["preranker"] / "shortlist_audit", v3_fingerprint=fingerprint)
        return 0
        
    if args.stage == "build-capsules":
        require_success(paths["preranker"] / "shortlist_audit", v3_fingerprint=fingerprint)
        if args.split == "train":
            pred = paths["preranker"] / "oof" / "oof_predictions.jsonl"
        else:
            public_pred_dir = paths["preranker"] / "public"
            pred = public_pred_dir / "public_predictions.jsonl"
            if not pred.exists():
                final_model_path = paths["preranker"] / "final" / "final.pkl"
                if not final_model_path.exists():
                    train_final_preranker(candidates_path=paths["candidates"] / "train" / "train_candidates.jsonl",
                                          train_path=TRAIN, output_dir=paths["preranker"] / "final", v3_fingerprint=fingerprint)
                public_pred_dir.mkdir(parents=True, exist_ok=True)
                score_preranker(candidates_path=candidates, model_path=final_model_path, output_path=pred)
                
        build_capsules(split=args.split, candidates_path=candidates, preranker_path=pred, rankings_path=source, v3_dir=args.v3_dir, lookup_db=paths["lookup"], output_dir=paths["capsules"] / args.split, v3_fingerprint=fingerprint, tokenizer_name=args.model, allow_download=args.allow_download, max_length=args.max_length, shortlist_k=_shortlist_k(paths), resume=args.resume, ranking_dir=paths["rankings"] / args.split, bge_dir=args.upstream_cache / "bge_leaves", bm25_db=args.upstream_cache / "bm25_index" / "bm25_v3.sqlite")
        return 0
        
    if args.stage == "train-lora":
        require_success(paths["capsules"] / "train", v3_fingerprint=fingerprint)
        train_lora_bge_reranker(capsules_path=paths["capsules"] / "train" / "train_capsules.jsonl", train_path=TRAIN, output_dir=paths["lora"], v3_fingerprint=fingerprint, device=args.device, allow_download=args.allow_download)
        return 0
        
    if args.stage == "score-qwen-oof":
        require_success(paths["capsules"] / "train", v3_fingerprint=fingerprint)
        fold_qids = {str(value) for value in load_folds(FOLDS)[f"fold_{args.fold}"]}
        adapter_dir = paths["lora"] / "bge_lora_adapter"
        if adapter_dir.exists():
            score_bge_lora(capsules_path=paths["capsules"] / "train" / "train_capsules.jsonl", adapter_path=adapter_dir, output_dir=paths["qwen"] / f"fold_{args.fold}", v3_fingerprint=fingerprint, device=args.device, allow_download=args.allow_download, max_length=args.max_length, batch_size=args.batch_size, fold_qids=fold_qids)
        else:
            score_reranker(capsules_path=paths["capsules"] / "train" / "train_capsules.jsonl", output_dir=paths["qwen"] / f"fold_{args.fold}", v3_fingerprint=fingerprint, model_name=args.model, device=args.device, allow_download=args.allow_download, max_length=args.max_length, batch_size=args.batch_size, fold_qids=fold_qids, resume=args.resume)
        return 0
        
    if args.stage == "evaluate-oof":
        require_success(paths["qwen"] / "fold_0", v3_fingerprint=fingerprint)
        # Check available fold scores
        available_qwen = {fold: paths["qwen"] / f"fold_{fold}" / "scores.jsonl" for fold in range(5) if (paths["qwen"] / f"fold_{fold}" / "scores.jsonl").exists()}
        if not available_qwen:
            raise RuntimeError("No Reranker fold scores found for evaluate-oof")
        evaluate_oof(candidates_path=candidates, preranker_path=paths["preranker"] / "oof" / "oof_predictions.jsonl", qwen_paths=available_qwen, train_path=TRAIN, folds_path=FOLDS, baseline_results=ROOT / "results" / "exp013b_cascade" / "oof", output_dir=args.results_root / "oof", v3_fingerprint=fingerprint)
        return 0
        
    if args.stage == "train-final":
        require_success(paths["preranker"] / "oof", v3_fingerprint=fingerprint)
        train_final_preranker(candidates_path=paths["candidates"] / "train" / "train_candidates.jsonl", train_path=TRAIN, output_dir=paths["preranker"] / "final", v3_fingerprint=fingerprint)
        return 0
        
    if args.stage == "score-public":
        require_success(paths["candidates"] / "public", v3_fingerprint=fingerprint)
        public_candidates = paths["candidates"] / "public" / "public_candidates.jsonl"
        public_source = paths["rankings"] / "public" / "source_rankings.jsonl"
        public_preranker = paths["preranker"] / "public"
        pred = public_preranker / "public_predictions.jsonl"
        
        final_model_path = paths["preranker"] / "final" / "final.pkl"
        if not final_model_path.exists():
            train_final_preranker(candidates_path=paths["candidates"] / "train" / "train_candidates.jsonl",
                                  train_path=TRAIN, output_dir=paths["preranker"] / "final", v3_fingerprint=fingerprint)
                                  
        with stage_run(public_preranker, "score-preranker-public", total=1, v3_fingerprint=fingerprint) as logger:
            count = score_preranker(candidates_path=public_candidates, model_path=final_model_path, output_path=pred)
            write_manifest(public_preranker, stage="score-preranker-public", v3_fingerprint=fingerprint, config={"model": "final.pkl"}, files=[pred], counts={"queries": count})
            logger.log(f"queries={count}")
            
        build_capsules(split="public", candidates_path=public_candidates, preranker_path=pred, rankings_path=public_source, v3_dir=args.v3_dir, lookup_db=paths["lookup"], output_dir=paths["capsules"] / "public", v3_fingerprint=fingerprint, tokenizer_name=args.model, allow_download=args.allow_download, max_length=args.max_length, shortlist_k=_shortlist_k(paths), resume=args.resume, ranking_dir=paths["rankings"] / "public", bge_dir=args.upstream_cache / "bge_leaves", bm25_db=args.upstream_cache / "bm25_index" / "bm25_v3.sqlite")
        
        adapter_dir = paths["lora"] / "bge_lora_adapter"
        if adapter_dir.exists():
            score_bge_lora(capsules_path=paths["capsules"] / "public" / "public_capsules.jsonl", adapter_path=adapter_dir, output_dir=paths["qwen"] / "public", v3_fingerprint=fingerprint, device=args.device, allow_download=args.allow_download, max_length=args.max_length, batch_size=args.batch_size)
        else:
            score_reranker(capsules_path=paths["capsules"] / "public" / "public_capsules.jsonl", output_dir=paths["qwen"] / "public", v3_fingerprint=fingerprint, model_name=args.model, device=args.device, allow_download=args.allow_download, max_length=args.max_length, batch_size=args.batch_size, resume=args.resume)
        return 0
        
    if args.stage == "public-submission":
        require_success(paths["qwen"] / "public", v3_fingerprint=fingerprint)
        public_candidates = paths["candidates"] / "public" / "public_candidates.jsonl"
        candidates_by_qid = {str(row["qid"]): {str(item["doc_id"]): item for item in row["candidates"]} for row in read_jsonl(public_candidates)}
        lam = {str(row["qid"]): row["candidates"] for row in read_jsonl(paths["preranker"] / "public" / "public_predictions.jsonl")}
        qwen = {str(row["qid"]): row["scores"] for row in read_jsonl(paths["qwen"] / "public" / "scores.jsonl")}
        
        expected_public = set(json.loads(PUBLIC.read_text(encoding="utf-8")))
        if set(lam) != expected_public or set(candidates_by_qid) != expected_public:
            raise RuntimeError("public query set mismatch")
            
        modal_config = {"qwen": 0.8, "lambda": 0.2, "bge": 0.05}
        oof_report_path = args.results_root / "oof" / "oof_report.json"
        if oof_report_path.exists():
            try:
                modal_config = json.loads(oof_report_path.read_text(encoding="utf-8")).get("modal_public_fusion", modal_config)
            except Exception:
                pass
                
        payload = {}
        for qid in sorted(lam):
            fused_ranked = _fuse(lam[qid], qwen.get(qid, []), candidates_by_qid[qid], modal_config, apply_entity_boost=True)
            # Apply dynamic cutoff to boost precision
            final_answers = _apply_dynamic_cutoff(fused_ranked, candidates_by_qid[qid])
            payload[qid] = {"answer": final_answers[:5]}
            
        out = args.results_root / "submission"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "submission.json"
        
        with stage_run(out, "public-submission", total=len(payload), v3_fingerprint=fingerprint) as logger:
            atomic_json(path, payload)
            archive = out / "submission.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
                handle.write(path, arcname="submission.json")
            write_manifest(out, stage="public-submission", v3_fingerprint=fingerprint, config={"top_k": 5, "modal_fusion": modal_config}, files=[path, archive], counts={"queries": len(payload)})
            logger.log(f"submission={archive}")
        return 0
        
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
