"""Materialize the missing F1--F4 EXP-109B cached-fusion anchor.

This is a repair utility, not an EXP-110P model.  It uses the frozen
EXP-109B cached E5/LAL/BM25 ranking stores and the four configs recorded in
the verified cached-fusion pilot.  It never evaluates, scores, or writes a
Fold-0 prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import exp109b_encoder_complementarity as exp109b


EXPECTED = {
    "recall@5": 0.9251057869956494,
    "recall@1": 0.6753531199713928,
    "mrr@5": 0.8044072948328268,
    "precision@5": 0.19749687108886105,
    "multi_gold_recall@5": 0.7174585218702866,
}
INNER_FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")


def _locked_configs(pilot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for row in pilot.get("lambdamart_folds", []):
        fold = str(row.get("validation_fold"))
        config = row.get("chosen_config", {}).get("config")
        if fold in INNER_FOLDS and isinstance(config, Mapping):
            configs[fold] = dict(config)
    if set(configs) != set(INNER_FOLDS):
        raise RuntimeError(f"pilot lacks one locked config per inner fold: {sorted(configs)}")
    return configs


def materialize(output: Path) -> dict[str, Any]:
    pilot_path = exp109b.RESULTS_ROOT / "cached_fusion_pilot" / "fold_0" / "CACHED_FUSION_PILOT.json"
    pilot = exp109b.read_json(pilot_path)
    if pilot.get("status") != "PASS_CACHED_FUSION_INNER_GATE" or pilot.get("winner") != "lambdamart_top50_per_source":
        raise RuntimeError("verified passing EXP-109B cached-fusion pilot is required")
    configs = _locked_configs(pilot)
    expected_manifests = pilot.get("rankings_manifests", {})
    e5_rows, e5_manifest = exp109b._load_cached_ranking_rows_for_pilot(
        "vietlegal_e5", "fold_0", expected_manifests["vietlegal_e5"]["content_fingerprint"]
    )
    lal_rows, lal_manifest = exp109b._load_cached_ranking_rows_for_pilot(
        "vnlegal_lal", "fold_0", expected_manifests["vnlegal_lal"]["content_fingerprint"]
    )
    folds, fold_for = exp109b.load_folds()
    if any(name not in folds for name in INNER_FOLDS):
        raise RuntimeError("missing EXP-109B inner fold")
    inner_folds = {name: list(folds[name]) for name in INNER_FOLDS}
    inner_qids = [qid for name in INNER_FOLDS for qid in inner_folds[name]]
    answers, _ = exp109b.canonical_labels()
    if set(inner_qids) - set(answers):
        raise RuntimeError("inner label coverage mismatch")
    source_rankings = {
        "vietlegal_e5": exp109b._ranking_map(e5_rows),
        "vnlegal_lal": exp109b._ranking_map(lal_rows),
        "bm25": exp109b._ranking_map(exp109b._bm25_rank_rows(inner_qids, fold_for)),
    }
    score_maps = {
        "vietlegal_e5": exp109b._ranking_score_map(e5_rows),
        "vnlegal_lal": exp109b._ranking_score_map(lal_rows),
        "bm25": exp109b._ranking_score_map(exp109b._bm25_rank_rows(inner_qids, fold_for)),
    }
    metadata = exp109b.build_parent_text_metadata()
    train = exp109b.load_train()
    lengths = {qid: len(str(train[qid].get("question", "")).split()) for qid in inner_qids}
    import lightgbm as lgb

    predictions: dict[str, list[str]] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    for heldout in INNER_FOLDS:
        train_qids = [qid for name in INNER_FOLDS if name != heldout for qid in inner_folds[name]]
        valid_qids = list(inner_folds[heldout])
        train_x, train_y, train_groups, _train_candidates, names = exp109b._lgbm_matrix_for_qids(
            source_rankings, score_maps, train_qids, answers, sources=("vietlegal_e5", "vnlegal_lal", "bm25"),
            depth=50, metadata=metadata, query_token_lengths=lengths,
        )
        valid_x, _valid_y, _valid_groups, valid_candidates, _ = exp109b._lgbm_matrix_for_qids(
            source_rankings, score_maps, valid_qids, answers, sources=("vietlegal_e5", "vnlegal_lal", "bm25"),
            depth=50, metadata=metadata, query_token_lengths=lengths,
        )
        config = configs[heldout]
        model = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", ndcg_at=[5], num_leaves=int(config["num_leaves"]),
            min_child_samples=int(config["min_data_in_leaf"]), learning_rate=float(config["learning_rate"]),
            n_estimators=int(config["num_boost_round"]), feature_fraction=1.0, bagging_fraction=1.0,
            bagging_freq=0, deterministic=True, random_state=exp109b.RNG_SEED, verbosity=-1,
        )
        model.fit(train_x, train_y, group=train_groups, feature_name=names)
        scores = np.asarray(model.predict(valid_x), dtype=np.float64)
        offset = 0
        for qid, candidates in valid_candidates:
            local = scores[offset: offset + len(candidates)]
            order = np.lexsort((np.asarray(candidates, dtype="U"), -local))
            ranked = [candidates[int(position)] for position in order]
            predictions[qid] = ranked
            raw_scores[qid] = {doc: float(score) for doc, score in zip(candidates, local)}
            offset += len(candidates)
    if set(predictions) != set(inner_qids):
        raise RuntimeError("anchor coverage mismatch")
    metrics = exp109b.evaluate_rankings(predictions, answers, inner_qids)
    if any(abs(float(metrics[key]) - expected) > 1e-12 for key, expected in EXPECTED.items()):
        raise RuntimeError(f"anchor metric mismatch: {metrics}")
    output = Path(output)
    exp109b.write_jsonl_atomic(output, (
        {"qid": qid, "prediction": [{"doc_id": doc} for doc in predictions[qid]], "raw_scores": raw_scores[qid], "fold0_included": False}
        for qid in sorted(inner_qids)
    ))
    report = {
        "stage": "exp109b-inner-anchor-materialization",
        "status": "PASS_RECONSTRUCTED_INNER_ANCHOR",
        "output": str(output.resolve()),
        "sha256": exp109b.sha256_file(output),
        "metrics": metrics,
        "expected_metrics": EXPECTED,
        "locked_configs": configs,
        "source_manifests": {"vietlegal_e5": e5_manifest, "vnlegal_lal": lal_manifest},
        "reference_status": "reconstructed_from_verified_EXP109B_inputs_not_independent_exported_reference",
        "fold0_read": False,
        "fold0_predictions_included": False,
    }
    exp109b.atomic_json(output.with_suffix(".manifest.json"), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(materialize(args.output), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
