"""EXP-013b Selective Legal Cascade command-line pipeline."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

from exp012b_core import atomic_json, read_jsonl, stage_run, write_jsonl
from exp012b_tuning import load_folds
from exp013b_candidates import audit_candidates, build_candidates, build_query_memory
from exp013b_capsules import build_capsules
from exp013b_core import QWEN_MODEL, preflight, require_success, write_manifest
from exp013b_fusion import _ambiguity, _fuse, evaluate_oof, fit_router, screen_fold
from exp013b_qwen import benchmark_qwen, prepare_qwen, score_qwen
from exp013b_ranker import audit_shortlist, score_preranker, train_final_preranker, train_preranker_oof


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE, DEFAULT_RESULTS = ROOT / "cache" / "exp013b_cascade", ROOT / "results" / "exp013b_cascade"
DEFAULT_V3, DEFAULT_AUDIT = ROOT / "cache" / "structural_v3", ROOT / "results" / "audits" / "chunker_v3" / "audit.json"
DEFAULT_EXP012 = ROOT / "cache" / "exp012b_v3"
TRAIN, PUBLIC, FOLDS = ROOT / "public_test_dataset" / "train.json", ROOT / "public_test_dataset" / "public-official.json", ROOT / "cache" / "cv_folds.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", required=True, choices=("preflight", "build-query-memory", "build-candidates", "audit-candidates", "train-preranker-oof", "audit-shortlist", "prepare-qwen", "build-capsules", "benchmark-qwen", "score-qwen-oof", "score-qwen-screen", "fit-router", "evaluate-oof", "train-final", "score-public", "public-submission", "status"))
    value.add_argument("--split", choices=("train", "public"), default="train")
    value.add_argument("--fold", type=int, choices=range(5), default=0)
    value.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE); value.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    value.add_argument("--v3-dir", type=Path, default=DEFAULT_V3); value.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    value.add_argument("--upstream-cache", type=Path, default=DEFAULT_EXP012); value.add_argument("--model", default=QWEN_MODEL); value.add_argument("--device", default="cuda")
    value.add_argument("--allow-download", action="store_true"); value.add_argument("--resume", action="store_true"); value.add_argument("--batch-size", type=int, default=8); value.add_argument("--max-length", type=int, default=768)
    value.add_argument("--pairs", type=int, default=2048, help="Number of capsule pairs used by benchmark-qwen.")
    return value


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {"rankings": args.upstream_cache / "rankings", "lookup": args.upstream_cache / "chunk_lookup" / "chunk_offsets.sqlite", "memory": args.cache_root / "query_memory", "candidates": args.cache_root / "candidates", "preranker": args.cache_root / "preranker", "capsules": args.cache_root / "capsules", "qwen": args.cache_root / "qwen", "router": args.cache_root / "router"}


def _shortlist_k(paths: dict[str, Path]) -> int:
    return int(json.loads((paths["preranker"] / "shortlist_audit" / "shortlist_audit.json").read_text(encoding="utf-8"))["shortlist_k"])


def _qwen_runtime(paths: dict[str, Path], model: str) -> tuple[int, int]:
    report = json.loads((paths["qwen"] / "benchmark" / "benchmark.json").read_text(encoding="utf-8"))
    selected = report.get("selected")
    if report.get("status") != "PASS" or not selected:
        raise RuntimeError("Qwen benchmark has no passing runtime configuration")
    manifest = json.loads((paths["qwen"] / "benchmark" / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config", {}).get("model") != model:
        raise RuntimeError("requested scorer model differs from benchmarked model")
    return int(selected["max_length"]), int(selected["batch_size"])


def _status(cache: Path) -> None:
    for path in sorted(cache.rglob("status.json")):
        print(f"{path.relative_to(cache)}: {path.read_text(encoding='utf-8').strip()}")


def main() -> int:
    args = parser().parse_args(); paths = _paths(args)
    if args.batch_size < 1 or args.max_length < 128: raise SystemExit("invalid batch size or max length")
    if args.stage == "status": _status(args.cache_root); return 0
    v3 = preflight(args.v3_dir, args.audit); fingerprint = v3["content_fingerprint"]
    if args.stage == "preflight":
        out = args.cache_root / "preflight"
        with stage_run(out, "preflight", total=1, v3_fingerprint=fingerprint) as logger:
            path = out / "preflight.json"; atomic_json(path, {"v3_fingerprint": fingerprint, "counts": v3["counts"]}); write_manifest(out, stage="preflight", v3_fingerprint=fingerprint, config={}, files=[path], counts={"documents": int(v3["counts"]["documents"])}); logger.log(f"v3_fingerprint={fingerprint}")
        return 0
    require_success(args.cache_root / "preflight", v3_fingerprint=fingerprint)
    split_path = TRAIN if args.split == "train" else PUBLIC
    source = paths["rankings"] / args.split / "source_rankings.jsonl"; memory = paths["memory"] / args.split / f"{args.split}_memory.jsonl"; candidates = paths["candidates"] / args.split / f"{args.split}_candidates.jsonl"
    if args.stage == "build-query-memory":
        build_query_memory(split=args.split, train_path=TRAIN, query_path=split_path, folds_path=FOLDS, output_dir=paths["memory"] / args.split, v3_fingerprint=fingerprint); return 0
    if args.stage == "build-candidates":
        require_success(paths["memory"] / args.split, v3_fingerprint=fingerprint); build_candidates(split=args.split, rankings_path=source, memory_path=memory, v3_dir=args.v3_dir, output_dir=paths["candidates"] / args.split, v3_fingerprint=fingerprint); return 0
    if args.stage == "audit-candidates":
        if args.split != "train": raise RuntimeError("candidate audit requires --split train")
        require_success(paths["candidates"] / "train", v3_fingerprint=fingerprint); audit_candidates(candidates_path=candidates, memory_path=memory, train_path=TRAIN, folds_path=FOLDS, v3_dir=args.v3_dir, output_dir=args.results_root / "candidate_audit", v3_fingerprint=fingerprint); return 0
    if args.stage == "train-preranker-oof":
        require_success(args.results_root / "candidate_audit", v3_fingerprint=fingerprint); train_preranker_oof(candidates_path=candidates, train_path=TRAIN, folds_path=FOLDS, output_dir=paths["preranker"] / "oof", v3_fingerprint=fingerprint); return 0
    if args.stage == "audit-shortlist":
        require_success(paths["preranker"] / "oof", v3_fingerprint=fingerprint); audit_shortlist(predictions_path=paths["preranker"] / "oof" / "oof_predictions.jsonl", train_path=TRAIN, output_dir=paths["preranker"] / "shortlist_audit", v3_fingerprint=fingerprint); return 0
    if args.stage == "prepare-qwen":
        prepare_qwen(output_dir=paths["qwen"] / "prepared", v3_fingerprint=fingerprint, model_name=args.model, device=args.device, allow_download=args.allow_download); return 0
    if args.stage == "build-capsules":
        require_success(paths["preranker"] / "shortlist_audit", v3_fingerprint=fingerprint); require_success(paths["qwen"] / "prepared", v3_fingerprint=fingerprint)
        pred = paths["preranker"] / "oof" / "oof_predictions.jsonl" if args.split == "train" else paths["preranker"] / "public_predictions.jsonl"
        build_capsules(split=args.split, candidates_path=candidates, preranker_path=pred, rankings_path=source, v3_dir=args.v3_dir, lookup_db=paths["lookup"], output_dir=paths["capsules"] / args.split, v3_fingerprint=fingerprint, tokenizer_name=args.model, allow_download=args.allow_download, max_length=args.max_length, shortlist_k=_shortlist_k(paths), resume=args.resume, ranking_dir=paths["rankings"] / args.split, bge_dir=args.upstream_cache / "bge_leaves", bm25_db=args.upstream_cache / "bm25_index" / "bm25_v3.sqlite"); return 0
    if args.stage == "benchmark-qwen":
        require_success(paths["capsules"] / "train", v3_fingerprint=fingerprint); require_success(paths["qwen"] / "prepared", v3_fingerprint=fingerprint)
        prepared = json.loads((paths["qwen"] / "prepared" / "manifest.json").read_text(encoding="utf-8"))
        if prepared.get("config", {}).get("model") != args.model: raise RuntimeError("requested benchmark model differs from prepared model")
        benchmark_qwen(capsules_path=paths["capsules"] / "train" / "train_capsules.jsonl", output_dir=paths["qwen"] / "benchmark", v3_fingerprint=fingerprint, model_name=args.model, device=args.device, allow_download=args.allow_download, pairs=args.pairs, max_length=args.max_length); return 0
    if args.stage == "score-qwen-oof":
        if args.fold == 0: raise RuntimeError("fold 0 uses score-qwen-screen")
        require_success(paths["router"], v3_fingerprint=fingerprint)
        folds = load_folds(FOLDS); fold_qids = {str(value) for value in folds[f"fold_{args.fold}"]}
        score_length, score_batch = _qwen_runtime(paths, args.model)
        score_qwen(capsules_path=paths["capsules"] / "train" / "train_capsules.jsonl", output_dir=paths["qwen"] / f"fold_{args.fold}", v3_fingerprint=fingerprint, model_name=args.model, device=args.device, allow_download=args.allow_download, max_length=score_length, batch_size=score_batch, fold_qids=fold_qids, resume=args.resume); return 0
    if args.stage == "score-qwen-screen":
        if args.fold != 0: raise RuntimeError("screen is defined only for fold 0")
        require_success(paths["qwen"] / "benchmark", v3_fingerprint=fingerprint)
        fold0_dir = paths["qwen"] / "fold_0"
        manifest_path = fold0_dir / "manifest.json"
        model_matches = False
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
                if m.get("config", {}).get("model") == args.model:
                    model_matches = True
            except Exception:
                pass
        try:
            if not model_matches:
                raise RuntimeError("model in fold0 cache does not match requested model")
            require_success(fold0_dir, v3_fingerprint=fingerprint)
        except RuntimeError:
            score_length, score_batch = _qwen_runtime(paths, args.model)
            fold0_qids = {str(value) for value in load_folds(FOLDS)["fold_0"]}
            score_qwen(capsules_path=paths["capsules"] / "train" / "train_capsules.jsonl", output_dir=fold0_dir,
                v3_fingerprint=fingerprint, model_name=args.model, device=args.device, allow_download=args.allow_download,
                max_length=score_length, batch_size=score_batch, fold_qids=fold0_qids, resume=args.resume)
        screen_fold(fold=0, candidates_path=candidates, preranker_path=paths["preranker"] / "oof" / "oof_predictions.jsonl", qwen_path=fold0_dir / "scores.jsonl", train_path=TRAIN, folds_path=FOLDS, output_dir=args.results_root / "qwen_screen", v3_fingerprint=fingerprint); return 0
    if args.stage == "fit-router":
        require_success(args.results_root / "qwen_screen", v3_fingerprint=fingerprint); fit_router(candidates_path=candidates, preranker_path=paths["preranker"] / "oof" / "oof_predictions.jsonl", qwen_path=paths["qwen"] / "fold_0" / "scores.jsonl", train_path=TRAIN, folds_path=FOLDS, output_dir=paths["router"], v3_fingerprint=fingerprint); return 0
    if args.stage == "evaluate-oof":
        require_success(paths["router"], v3_fingerprint=fingerprint); evaluate_oof(candidates_path=candidates, preranker_path=paths["preranker"] / "oof" / "oof_predictions.jsonl", qwen_paths={fold: paths["qwen"] / f"fold_{fold}" / "scores.jsonl" for fold in range(5)}, router_path=paths["router"] / "router.json", train_path=TRAIN, folds_path=FOLDS, baseline_results=ROOT / "results" / "exp012b_v3" / "lora", output_dir=args.results_root / "oof", v3_fingerprint=fingerprint); return 0
    if args.stage == "train-final":
        require_success(args.results_root / "oof", v3_fingerprint=fingerprint); train_final_preranker(candidates_path=paths["candidates"] / "train" / "train_candidates.jsonl", train_path=TRAIN, output_dir=paths["preranker"] / "final", v3_fingerprint=fingerprint); return 0
    if args.stage == "score-public":
        require_success(args.results_root / "oof", v3_fingerprint=fingerprint); require_success(paths["preranker"] / "final", v3_fingerprint=fingerprint); require_success(paths["candidates"] / "public", v3_fingerprint=fingerprint)
        public_candidates = paths["candidates"] / "public" / "public_candidates.jsonl"; public_source = paths["rankings"] / "public" / "source_rankings.jsonl"
        public_preranker = paths["preranker"] / "public"; pred = public_preranker / "public_predictions.jsonl"
        with stage_run(public_preranker, "score-preranker-public", total=1, v3_fingerprint=fingerprint) as logger:
            count = score_preranker(candidates_path=public_candidates, model_path=paths["preranker"] / "final" / "final.pkl", output_path=pred)
            write_manifest(public_preranker, stage="score-preranker-public", v3_fingerprint=fingerprint,
                config={"model": "final.pkl"}, files=[pred], counts={"queries": count}); logger.log(f"queries={count}")
        build_capsules(split="public", candidates_path=public_candidates, preranker_path=pred, rankings_path=public_source, v3_dir=args.v3_dir, lookup_db=paths["lookup"], output_dir=paths["capsules"] / "public", v3_fingerprint=fingerprint, tokenizer_name=args.model, allow_download=args.allow_download, max_length=args.max_length, shortlist_k=_shortlist_k(paths), resume=args.resume, ranking_dir=paths["rankings"] / "public", bge_dir=args.upstream_cache / "bge_leaves", bm25_db=args.upstream_cache / "bm25_index" / "bm25_v3.sqlite")
        candidate_rows = {str(row["qid"]): {str(item["doc_id"]): item for item in row["candidates"]} for row in read_jsonl(public_candidates)}
        lambda_rows = {str(row["qid"]): row["candidates"] for row in read_jsonl(pred)}
        router = json.loads((paths["router"] / "router.json").read_text(encoding="utf-8"))["router"]
        fraction = float(router["fraction"])
        if fraction < 1.0:
            ordered = sorted(lambda_rows, key=lambda qid: (-_ambiguity(lambda_rows[qid], candidate_rows[qid]), qid))
            routed_qids = set(ordered[:math.ceil(len(ordered) * fraction)])
        else:
            routed_qids = set(lambda_rows)
        score_length, score_batch = _qwen_runtime(paths, args.model)
        score_qwen(capsules_path=paths["capsules"] / "public" / "public_capsules.jsonl", output_dir=paths["qwen"] / "public", v3_fingerprint=fingerprint, model_name=args.model, device=args.device, allow_download=args.allow_download, max_length=score_length, batch_size=score_batch, fold_qids=routed_qids, resume=args.resume)
        return 0
    if args.stage == "public-submission":
        require_success(args.results_root / "oof", v3_fingerprint=fingerprint); require_success(paths["qwen"] / "public", v3_fingerprint=fingerprint)
        public_candidates = paths["candidates"] / "public" / "public_candidates.jsonl"
        candidates_by_qid = {str(row["qid"]): {str(item["doc_id"]): item for item in row["candidates"]} for row in read_jsonl(public_candidates)}; lam = {str(row["qid"]): row["candidates"] for row in read_jsonl(paths["preranker"] / "public" / "public_predictions.jsonl")}; qwen = {str(row["qid"]): row["scores"] for row in read_jsonl(paths["qwen"] / "public" / "scores.jsonl")}
        expected_public = set(json.loads(PUBLIC.read_text(encoding="utf-8")))
        if set(lam) != expected_public or set(candidates_by_qid) != expected_public: raise RuntimeError("public query set mismatch")
        router = json.loads((paths["router"] / "router.json").read_text(encoding="utf-8"))["router"]
        modal = json.loads((args.results_root / "oof" / "oof_report.json").read_text(encoding="utf-8"))["modal_public_fusion"]
        fraction = float(router["fraction"])
        if fraction < 1.0:
            ordered = sorted(lam, key=lambda qid: (-_ambiguity(lam[qid], candidates_by_qid[qid]), qid))
            routed = set(ordered[:math.ceil(len(ordered)*fraction)])
        else:
            routed = set(lam)
        if not routed <= set(qwen): raise RuntimeError("public Qwen scores do not match frozen router selection")
        payload = {qid: {"answer": (_fuse(lam[qid], qwen[qid], candidates_by_qid[qid], modal) if qid in qwen else [str(row["doc_id"]) for row in lam[qid]])[:5]} for qid in sorted(lam)}
        out = args.results_root / "submission"; out.mkdir(parents=True, exist_ok=True); path = out / "submission.json"
        with stage_run(out, "public-submission", total=len(payload), v3_fingerprint=fingerprint) as logger:
            atomic_json(path, payload); archive = out / "submission.zip"; 
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle: handle.write(path, arcname="submission.json")
            write_manifest(out, stage="public-submission", v3_fingerprint=fingerprint, config={"top_k": 5, "router_fraction": router["fraction"]}, files=[path, archive], counts={"queries": len(payload)}); logger.log(f"submission={archive}")
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__": raise SystemExit(main())
