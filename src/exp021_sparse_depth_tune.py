"""Resumable overnight depth/aggregation tuning for the final sparse branch.

Run ``all`` once.  It retrieves top-4096 FTS passages, caches only the
per-document ranks needed by first-passage/RRF3, then performs a nested-OOF
grid search offline.  No dense artifact is read or modified.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from exp012b_bm25 import BM25Searcher, default_segmenter, safe_fts_query
from exp012b_core import atomic_json, canonical_json, content_hash, sha256_file, stage_run


DEPTHS = (1024, 2048, 4096)
RRF_KS = (32, 64, 128)
HEAD_CUTOFFS = (16, 32, 64)
RANK_KS = (1, 5, 10, 16, 30, 50, 80, 100, 120, 150, 180)
SHARD_SIZE = 128
MAX_DOCUMENTS = 256
SCHEMA = "legalir.exp021_sparse_depth_tune.v1"


def _load_train(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    train = json.loads(path.read_text(encoding="utf-8"))
    return train, sorted(map(str, train))


def _evidence_from_hits(hits: Iterable[dict[str, Any]]) -> list[list[Any]]:
    """Keep only ranks that can affect first-passage or RRF3 aggregation."""
    by_doc: dict[str, list[int]] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    for hit in hits:
        doc_id = str(hit["doc_id"])
        parent_id = str(hit["parent_node_id"])
        if parent_id in parents[doc_id] or len(by_doc.get(doc_id, [])) >= 3:
            continue
        parents[doc_id].add(parent_id)
        by_doc.setdefault(doc_id, []).append(int(hit["rank"]))
    return [[doc_id, ranks] for doc_id, ranks in sorted(by_doc.items(), key=lambda item: (item[1][0], item[0]))]


def retrieve_evidence(
    *, database_path: Path, train_path: Path, cache_dir: Path, workers: int, resume: bool
) -> dict[str, Any]:
    train, qids = _load_train(train_path)
    output_dir = cache_dir / "raw4096_evidence"
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    config = {"database_sha256": sha256_file(database_path), "top_passages": 4096, "workers": workers, "shard_size": SHARD_SIZE}
    config_hash = content_hash(config)
    with stage_run(output_dir, "retrieve-raw4096-evidence", total=len(qids)) as logger:
        logger.log("phase=segment-queries mode=single-thread (Underthesea thread safety)")
        expressions = {qid: safe_fts_query(default_segmenter(str(train[qid]["question"]))) for qid in qids}
        logger.log("phase=fts5-retrieval top_passages=4096")
        complete = 0; shard_paths: list[Path] = []
        for shard_index, start in enumerate(range(0, len(qids), SHARD_SIZE)):
            shard_qids = qids[start:start + SHARD_SIZE]
            path = shard_dir / f"evidence_{shard_index:04d}.jsonl"
            marker = shard_dir / f"evidence_{shard_index:04d}.json"
            if resume and path.exists() and marker.exists():
                saved = json.loads(marker.read_text(encoding="utf-8"))
                if saved.get("config_hash") == config_hash and saved.get("sha256") == sha256_file(path):
                    complete += len(shard_qids); shard_paths.append(path)
                    logger.status(stage="retrieve-raw4096-evidence", state="RUNNING", completed=complete, total=len(qids), shard=shard_index, emit_log=True)
                    continue
            state = threading.local(); searchers: list[BM25Searcher] = []; lock = threading.Lock()
            def one(qid: str) -> dict[str, Any]:
                searcher = getattr(state, "searcher", None)
                if searcher is None:
                    searcher = BM25Searcher(database_path, profile="legal_structure")
                    state.searcher = searcher
                    with lock: searchers.append(searcher)
                hits = searcher.search_expression(expressions[qid], limit=4096)
                return {"qid": qid, "evidence": _evidence_from_hits(hits)}
            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    rows = list(executor.map(one, shard_qids))
            finally:
                for searcher in searchers: searcher.close()
            temporary = path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
                for row in rows: handle.write(canonical_json(row) + "\n")
            temporary.replace(path)
            atomic_json(marker, {"config_hash": config_hash, "sha256": sha256_file(path), "queries": len(rows)})
            complete += len(rows); shard_paths.append(path)
            logger.status(stage="retrieve-raw4096-evidence", state="RUNNING", completed=complete, total=len(qids), shard=shard_index, emit_log=True)
        manifest = {
            "schema_version": SCHEMA, "stage": "retrieve-raw4096-evidence", "config": config,
            "shards": [{"name": path.name, "sha256": sha256_file(path)} for path in shard_paths],
            "queries": len(qids),
        }
        manifest["content_fingerprint"] = content_hash(manifest)
        atomic_json(output_dir / "manifest.json", manifest)
        return manifest


def _rankings(evidence: Sequence[Sequence[Any]], depth: int, parent_rrf_k: int) -> tuple[list[str], list[str]]:
    eligible = [(str(doc_id), [int(rank) for rank in ranks if int(rank) <= depth]) for doc_id, ranks in evidence]
    eligible = [(doc_id, ranks) for doc_id, ranks in eligible if ranks]
    first = [doc_id for doc_id, _ in sorted(eligible, key=lambda item: (item[1][0], item[0]))[:MAX_DOCUMENTS]]
    rrf = [
        doc_id for doc_id, _, _ in sorted(
            ((doc_id, sum(1.0 / (parent_rrf_k + rank) for rank in ranks), ranks[0]) for doc_id, ranks in eligible),
            key=lambda item: (-item[1], item[2], item[0]),
        )[:MAX_DOCUMENTS]
    ]
    return first, rrf


def _cascade(first: Sequence[str], rrf: Sequence[str], fusion_rrf_k: int, head: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float); best: dict[str, int] = {}
    for ranking in (first, rrf):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (fusion_rrf_k + rank)
            best[doc_id] = min(best.get(doc_id, rank), rank)
    fused = sorted(scores, key=lambda doc_id: (-scores[doc_id], best[doc_id], doc_id))[:MAX_DOCUMENTS]
    result = fused[:head] + [doc_id for doc_id in rrf if doc_id not in set(fused[:head])]
    return result[:MAX_DOCUMENTS]


def _recall(prediction: Sequence[str], answers: set[str], k: int) -> float:
    return len(set(prediction[:k]) & answers) / len(answers) if answers else 0.0


def _iter_evidence(shard_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(shard_dir.glob("evidence_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip(): yield json.loads(line)


def tune_oof(*, evidence_dir: Path, train_path: Path, folds_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))
    train, qids = _load_train(train_path)
    answers = {qid: set(map(str, train[qid]["answer"])) for qid in qids}
    folds = {name: set(map(str, values)) for name, values in json.loads(folds_path.read_text(encoding="utf-8")).items()}
    fold_for = {qid: name for name, values in folds.items() for qid in values}
    if set(fold_for) != set(qids): raise RuntimeError("folds must partition all train queries")
    configs = [(depth, parent_k, fusion_k, head) for depth in DEPTHS for parent_k in RRF_KS for fusion_k in RRF_KS for head in HEAD_CUTOFFS]
    sums: dict[tuple[int, int, int, int], dict[str, Counter[str]]] = {
        config: {fold: Counter() for fold in folds} for config in configs
    }
    counts = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    with stage_run(output_dir, "tune-depth-rrf-oof", total=len(qids)) as logger:
        for position, row in enumerate(_iter_evidence(evidence_dir / "shards"), start=1):
            qid = str(row["qid"]); evidence = row["evidence"]; fold = fold_for[qid]; counts[fold] += 1
            cached: dict[tuple[int, int], tuple[list[str], list[str]]] = {}
            for depth in DEPTHS:
                for parent_k in RRF_KS:
                    cached[(depth, parent_k)] = _rankings(evidence, depth, parent_k)
            for config in configs:
                depth, parent_k, fusion_k, head = config
                first, rrf = cached[(depth, parent_k)]
                prediction = _cascade(first, rrf, fusion_k, head)
                for k in RANK_KS:
                    sums[config][fold][f"recall@{k}"] += _recall(prediction, answers[qid], k)
            if position % SHARD_SIZE == 0 or position == len(qids):
                logger.status(stage="tune-depth-rrf-oof", state="RUNNING", completed=position, total=len(qids), emit_log=True)
        def select_by_metric(metric_k: int) -> dict[str, tuple[int, int, int, int]]:
            chosen: dict[str, tuple[int, int, int, int]] = {}
            for heldout in folds:
                training = [name for name in folds if name != heldout]
                chosen[heldout] = max(
                    configs,
                    key=lambda config: (
                        sum(sums[config][fold][f"recall@{metric_k}"] / counts[fold] for fold in training),
                        sum(sums[config][fold]["recall@5"] / counts[fold] for fold in training),
                        sum(sums[config][fold]["recall@10"] / counts[fold] for fold in training),
                        -config[0], -config[1], -config[2], -config[3],
                    ),
                )
            return chosen
        selected = select_by_metric(5)
        candidate_selected = {k: select_by_metric(k) for k in (30, 50, 80, 100, 120, 150, 180)}
        # The head configuration optimizes R@5; each candidate budget gets
        # its own nested selection, since larger K is otherwise monotonic.
        oof_sums = Counter(); oof_predictions: list[dict[str, Any]] = []
        candidate_oof_sums = Counter()
        for row in _iter_evidence(evidence_dir / "shards"):
            qid = str(row["qid"]); depth, parent_k, fusion_k, head = selected[fold_for[qid]]
            first, rrf = _rankings(row["evidence"], depth, parent_k)
            prediction = _cascade(first, rrf, fusion_k, head)
            for k in RANK_KS: oof_sums[f"recall@{k}"] += _recall(prediction, answers[qid], k)
            for budget, selected_for_budget in candidate_selected.items():
                depth_b, parent_b, fusion_b, head_b = selected_for_budget[fold_for[qid]]
                first_b, rrf_b = _rankings(row["evidence"], depth_b, parent_b)
                prediction_b = _cascade(first_b, rrf_b, fusion_b, head_b)
                candidate_oof_sums[f"recall@{budget}"] += _recall(prediction_b, answers[qid], budget)
            oof_predictions.append({"qid": qid, "fold": fold_for[qid], "documents": prediction})
        prediction_path = output_dir / "oof_rankings.jsonl"
        with prediction_path.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
            for row in oof_predictions: handle.write(canonical_json(row) + "\n")
        selected_json = {fold: {"depth": value[0], "parent_rrf_k": value[1], "fusion_rrf_k": value[2], "head_cutoff": value[3]} for fold, value in sorted(selected.items())}
        modal = Counter(selected.values()).most_common(1)[0][0]
        report = {
            "schema_version": SCHEMA, "status": "PASS", "evidence_fingerprint": manifest["content_fingerprint"],
            "grid": {"depths": DEPTHS, "parent_rrf_k": RRF_KS, "fusion_rrf_k": RRF_KS, "head_cutoffs": HEAD_CUTOFFS, "candidate_document_ks": (30, 50, 80, 100, 120, 150, 180)},
            "selected_by_fold": selected_json,
            "selected_by_candidate_budget": {
                str(budget): {
                    fold: {"depth": value[0], "parent_rrf_k": value[1], "fusion_rrf_k": value[2], "head_cutoff": value[3]}
                    for fold, value in sorted(selected_for_budget.items())
                }
                for budget, selected_for_budget in candidate_selected.items()
            },
            "modal_configuration": {"depth": modal[0], "parent_rrf_k": modal[1], "fusion_rrf_k": modal[2], "head_cutoff": modal[3]},
            "oof_recall_curve": {name: value / len(qids) for name, value in sorted(oof_sums.items())},
            "candidate_budget_oof_recall": {name: value / len(qids) for name, value in sorted(candidate_oof_sums.items())},
            "candidate_document_curve_note": "Each R@K uses its own nested-OFF selection; choose deployment K from the dense-union budget, not by maximizing monotonic R@K.",
            "oof_rankings_sha256": sha256_file(prediction_path),
        }
        atomic_json(output_dir / "tuning_report.json", report)
        logger.log("selected=" + json.dumps(selected_json, ensure_ascii=False, sort_keys=True))
        logger.log("oof=" + json.dumps(report["oof_recall_curve"], ensure_ascii=False, sort_keys=True))
        return report


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(__import__("sys").stdout, "reconfigure"): __import__("sys").stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("retrieve-evidence", "tune", "all"))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--folds-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.stage in {"retrieve-evidence", "all"}:
        retrieval = retrieve_evidence(database_path=args.database, train_path=args.train_file, cache_dir=args.cache_root, workers=args.workers, resume=args.resume)
        print(json.dumps(retrieval, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"tune", "all"}:
        report = tune_oof(evidence_dir=args.cache_root / "raw4096_evidence", train_path=args.train_file, folds_path=args.folds_file, output_dir=args.results_root / "depth_rrf_tuning")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
