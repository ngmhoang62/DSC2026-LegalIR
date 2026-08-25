"""EXP-028: shortlist-first nested LambdaMART on the immutable EXP-022 pool.

This experiment deliberately reuses EXP-027's provenance-repaired feature
matrix.  The only learned operation is a fold-isolated ranker; retrieval,
candidate membership, and the 150-parent budget are immutable.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from exp012b_core import artifact_manifest, atomic_json, load_v3_manifest, read_jsonl, require_success, sha256_file, stage_run, write_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp026_lambdamart_capsules import FEATURE_BLOCKS, FEATURE_SETS, _columns
from exp027_lambdamart_shortlist import K_GRID, retained_answers

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "legalir.exp028_lambdamart_shortlist.v1"
FLOOR = 0.985
SEED = 2028
PARAM_GRID = tuple(
    {"num_leaves": leaves, "min_child_samples": child, "n_estimators": trees, "reg_lambda": l2}
    for leaves, child, trees, l2 in itertools.product((15, 31, 63), (20, 50), (250, 500), (0.0, 2.0))
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_features(path: Path) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    schema = _json(path / "feature_schema.json")
    columns = schema.get("columns", [])
    if tuple(columns) != _columns("all"):
        raise ValueError("EXP-027 feature schema does not contain the expected 24 columns")
    return np.load(path / "features.f32.npy", mmap_mode="r"), list(read_jsonl(path / "query_index.jsonl")), columns


def _candidate_ids(path: Path) -> dict[str, list[str]]:
    result = {}
    for row in read_jsonl(path):
        qid = str(row["qid"]); ids = [str(x["doc_id"]) for x in row["candidates"]]
        if qid in result or len(ids) != 150 or len(ids) != len(set(ids)):
            raise ValueError(f"invalid immutable candidate pool for {qid}")
        result[qid] = ids
    return result


def audit(*, candidates: Path, sidecar: Path, feature_dir: Path, folds_path: Path, train: Path, preprocessing: Path, v3: Path, output_dir: Path) -> dict[str, Any]:
    """Bind every reused input and validate membership/order without rewriting it."""
    require_success(candidates.parent); require_success(sidecar.parent); require_success(feature_dir)
    v3_manifest = load_v3_manifest(v3)
    fixed = _candidate_ids(candidates)
    data, index, columns = _load_features(feature_dir)
    if len(data) != len(fixed) * 150 or len(index) != len(fixed):
        raise ValueError("feature row/query count mismatch")
    side = {str(x["qid"]): x for x in read_jsonl(sidecar)}
    if set(side) != set(fixed): raise ValueError("sidecar query set mismatch")
    for item in index:
        qid = str(item["qid"])
        if item["doc_ids"] != fixed.get(qid) or int(item["end"]) - int(item["start"]) != 150:
            raise ValueError(f"feature membership/order mismatch: {qid}")
        if side[qid].get("candidate_ids") != fixed[qid]:
            raise ValueError(f"sidecar membership/order mismatch: {qid}")
    answers, impact = retained_answers(train, preprocessing / "exclusions.json", preprocessing / "train_label_impact.jsonl")
    folds = load_folds(folds_path); folded = {str(q) for qs in folds.values() for q in qs}
    if set(answers) != set(fixed) or folded != set(fixed): raise ValueError("train/fold/candidate query set mismatch")
    report = {
        "schema_version": SCHEMA, "status": "PASS", "queries": len(fixed), "candidates": len(fixed) * 150,
        "columns": list(columns), "retained_label_impact": impact, "v3_fingerprint": v3_manifest["content_fingerprint"],
        "inputs": {"candidates_sha256": sha256_file(candidates), "sidecar_sha256": sha256_file(sidecar),
                   "features_manifest_sha256": sha256_file(feature_dir / "manifest.json"), "folds_sha256": sha256_file(folds_path),
                   "train_sha256": sha256_file(train), "preprocessing_manifest_sha256": sha256_file(preprocessing / "manifest.json"),
                   "exclusions_sha256": sha256_file(preprocessing / "exclusions.json"), "v3_manifest_sha256": sha256_file(v3 / "manifest.json")},
    }
    with stage_run(output_dir, "exp028-input-audit", total=len(fixed), v3_fingerprint=v3_manifest["content_fingerprint"]) as log:
        atomic_json(output_dir / "input_audit.json", report); log.set_telemetry(report)
    atomic_json(output_dir / "manifest.json", artifact_manifest(stage="exp028-input-audit", inputs=report["inputs"], config={"candidate_limit": 150, "columns": list(columns)}, files=[output_dir / "input_audit.json"]))
    return report


def _fit(train_idx: list[dict[str, Any]], data: np.ndarray, answers: dict[str, set[str]], columns: list[str], feature_set: str, params: dict[str, Any]):
    from lightgbm import LGBMRanker
    positions = [columns.index(x) for x in _columns(feature_set)]
    rows = np.concatenate([np.arange(x["start"], x["end"]) for x in train_idx])
    x = np.asarray(data[rows][:, positions]); y = np.asarray([int(doc in answers[str(item["qid"])] ) for item in train_idx for doc in item["doc_ids"]], dtype=np.int32)
    model = LGBMRanker(objective="lambdarank", metric="ndcg", learning_rate=.04, random_state=42, deterministic=True, force_col_wise=True, n_jobs=-1, verbosity=-1, **params)
    model.fit(x, y, group=[item["end"] - item["start"] for item in train_idx], eval_at=[5])
    return model, positions


def _predict(model: Any, positions: list[int], test_idx: list[dict[str, Any]], data: np.ndarray) -> dict[str, list[str]]:
    result = {}
    for item in test_idx:
        scores = model.predict(np.asarray(data[item["start"]:item["end"], positions]))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), str(item["doc_ids"][i])))
        result[str(item["qid"])] = [str(item["doc_ids"][i]) for i in order]
    return result


def _metrics(pred: dict[str, list[str]], answers: dict[str, set[str]], ks: tuple[int, ...] = K_GRID) -> dict[str, float]:
    return evaluate_rankings(pred, answers, ks=(5, *ks))


def _screen(feature_set: str, params: dict[str, Any], outer: str, folds: dict[str, list[str]], by_qid: dict[str, dict[str, Any]], data: np.ndarray, answers: dict[str, set[str]], columns: list[str]) -> dict[str, Any]:
    outer_train = [q for name, qids in folds.items() if name != outer for q in qids]
    per_inner = {}
    for inner in sorted(set(folds) - {outer}):
        inner_set = set(folds[inner]); train_ids = [q for q in outer_train if q not in inner_set]
        model, positions = _fit([by_qid[q] for q in train_ids], data, answers, columns, feature_set, params)
        pred = _predict(model, positions, [by_qid[q] for q in folds[inner]], data)
        per_inner[inner] = _metrics(pred, {q: answers[q] for q in folds[inner]})
    viable = [k for k in K_GRID if all(value[f"recall@{k}"] >= FLOOR for value in per_inner.values())]
    diagnostic_k = viable[0] if viable else 100
    recalls = [value[f"recall@{diagnostic_k}"] for value in per_inner.values()]
    return {"feature_set": feature_set, "params": params, "per_inner": per_inner, "viable_k": viable[0] if viable else None,
            "diagnostic_k": diagnostic_k, "worst_inner_recall": min(recalls), "mean_inner_recall": float(np.mean(recalls))}


def choose_screen(screens: list[dict[str, Any]]) -> dict[str, Any]:
    """Strict shortlist-first selection; deterministic even when no screen passes."""
    if not screens: raise ValueError("empty screen list")
    def key(x: dict[str, Any]):
        viable = x["viable_k"] is not None
        return (-int(viable), x["viable_k"] if viable else 10**9, -x["worst_inner_recall"], -x["mean_inner_recall"], FEATURE_SETS.index(x["feature_set"]), PARAM_GRID.index(x["params"]))
    return sorted(screens, key=key)[0]


def nested_oof(*, feature_dir: Path, train: Path, preprocessing: Path, folds_path: Path, output_dir: Path, v3: Path) -> dict[str, Any]:
    data, index, columns = _load_features(feature_dir); by_qid = {x["qid"]: x for x in index}; answers, _ = retained_answers(train, preprocessing / "exclusions.json", preprocessing / "train_label_impact.jsonl"); folds = load_folds(folds_path)
    selected_predictions, family_predictions, selections = {}, {name: {} for name in FEATURE_SETS}, {}
    with stage_run(output_dir, "exp028-nested-oof", total=len(index), v3_fingerprint=load_v3_manifest(v3)["content_fingerprint"]) as log:
        for outer, heldout in sorted(folds.items()):
            screens = [_screen(feature, param, outer, folds, by_qid, data, answers, columns) for feature in FEATURE_SETS for param in PARAM_GRID]
            chosen = choose_screen(screens); family_chosen = {feature: choose_screen([x for x in screens if x["feature_set"] == feature]) for feature in FEATURE_SETS}
            outer_train = [q for name, qids in folds.items() if name != outer for q in qids]
            for feature, selected in family_chosen.items():
                model, positions = _fit([by_qid[q] for q in outer_train], data, answers, columns, feature, selected["params"])
                family_predictions[feature].update(_predict(model, positions, [by_qid[q] for q in heldout], data))
            # Avoid depending on identity semantics above: score the chosen model explicitly.
            model, positions = _fit([by_qid[q] for q in outer_train], data, answers, columns, chosen["feature_set"], chosen["params"])
            selected_predictions.update(_predict(model, positions, [by_qid[q] for q in heldout], data))
            selections[outer] = {"chosen": chosen, "feature_family_choices": family_chosen}
            log.status(stage="exp028-nested-oof", state="RUNNING", completed=len(selected_predictions), total=len(index)); log.log(f"outer={outer} k={chosen['viable_k']} worst_inner={chosen['worst_inner_recall']:.6f}")
    write_jsonl(output_dir / "oof_predictions.jsonl", ({"qid": q, "doc_ids": ids} for q, ids in sorted(selected_predictions.items())))
    write_jsonl(output_dir / "ablation_predictions.jsonl", ({"feature_set": f, "qid": q, "doc_ids": ids} for f in FEATURE_SETS for q, ids in sorted(family_predictions[f].items())))
    per_outer = {}
    for outer, qids in sorted(folds.items()):
        selected = selections[outer]["chosen"]; k = selected["viable_k"]
        metric = _metrics({q: selected_predictions[q] for q in qids}, {q: answers[q] for q in qids})
        per_outer[outer] = {"selected_k": k, "selection_passed": k is not None, "heldout_recall_at_selected_k": metric.get(f"recall@{k}") if k else None, "heldout_metrics": metric}
    viable_outer = all(x["selection_passed"] and x["heldout_recall_at_selected_k"] >= FLOOR for x in per_outer.values())
    deployment_k = max(x["selected_k"] for x in per_outer.values()) if all(x["selection_passed"] for x in per_outer.values()) else None
    report = {"schema_version": SCHEMA, "denominator": "retained_gold", "floor": FLOOR, "k_grid": list(K_GRID), "parameter_grid": list(PARAM_GRID), "selections": selections, "per_outer": per_outer,
              "global_oof_curve_diagnostic_only": _metrics(selected_predictions, answers), "deployment_k": deployment_k, "deployment_pairs": len(index) * deployment_k if deployment_k else None,
              "status": "PASS" if viable_outer and deployment_k <= 100 else "FAIL", "selection_note": "K is selected exclusively from inner folds; global curve is post-hoc diagnostic."}
    atomic_json(output_dir / "oof_report.json", report)
    atomic_json(output_dir / "manifest.json", artifact_manifest(stage="exp028-nested-oof", inputs={"features_manifest_sha256": sha256_file(feature_dir / "manifest.json"), "train_sha256": sha256_file(train), "folds_sha256": sha256_file(folds_path)}, config={"floor": FLOOR, "k_grid": list(K_GRID), "parameter_grid": list(PARAM_GRID), "feature_sets": list(FEATURE_SETS)}, files=[output_dir / "oof_predictions.jsonl", output_dir / "ablation_predictions.jsonl", output_dir / "oof_report.json"]))
    return report


def ablation(*, predictions: Path, oof: Path, train: Path, preprocessing: Path, folds_path: Path, output_dir: Path, v3: Path) -> dict[str, Any]:
    answers, _ = retained_answers(train, preprocessing / "exclusions.json", preprocessing / "train_label_impact.jsonl"); folds = load_folds(folds_path); report_oof = _json(oof)
    by_family: dict[str, dict[str, list[str]]] = {f: {} for f in FEATURE_SETS}
    for row in read_jsonl(predictions): by_family[row["feature_set"]][str(row["qid"])] = [str(x) for x in row["doc_ids"]]
    result = {"schema_version": SCHEMA, "denominator": "retained_gold", "feature_sets": {}}
    for feature in FEATURE_SETS:
        outer = {}
        for name, qids in sorted(folds.items()):
            selected = report_oof["selections"][name]["feature_family_choices"][feature]; k = selected["viable_k"]
            metric = _metrics({q: by_family[feature][q] for q in qids}, {q: answers[q] for q in qids})
            outer[name] = {"inner_selected_k": k, "heldout_recall_at_selected_k": metric.get(f"recall@{k}") if k else None, "heldout_metrics": metric}
        result["feature_sets"][feature] = {"per_outer": outer, "global_oof_curve_diagnostic_only": _metrics(by_family[feature], answers)}
    with stage_run(output_dir, "exp028-ablation", total=len(FEATURE_SETS), v3_fingerprint=load_v3_manifest(v3)["content_fingerprint"]) as log:
        atomic_json(output_dir / "ablation_report.json", result); log.set_telemetry(result)
    atomic_json(output_dir / "manifest.json", artifact_manifest(stage="exp028-ablation", inputs={"ablation_predictions_sha256": sha256_file(predictions), "oof_report_sha256": sha256_file(oof)}, config={"floor": FLOOR}, files=[output_dir / "ablation_report.json"]))
    return result


def _permuted(data: np.ndarray, item: dict[str, Any], position: int, seed: int) -> np.ndarray:
    values = np.asarray(data[item["start"]:item["end"], :]).copy(); rng = np.random.default_rng(seed); values[:, position] = values[rng.permutation(len(values)), position]; return values


def importance(*, feature_dir: Path, train: Path, preprocessing: Path, folds_path: Path, oof: Path, output_dir: Path, v3: Path) -> dict[str, Any]:
    data, index, columns = _load_features(feature_dir); by_qid = {x["qid"]: x for x in index}; answers, _ = retained_answers(train, preprocessing / "exclusions.json", preprocessing / "train_label_impact.jsonl"); folds = load_folds(folds_path); selections = _json(oof)["selections"]
    gain, split, permutation = defaultdict(float), defaultdict(float), {name: {"per_fold": {}, "predictions": {}} for name in columns}
    with stage_run(output_dir, "exp028-permutation-importance", total=len(columns) * len(folds), v3_fingerprint=load_v3_manifest(v3)["content_fingerprint"]) as log:
        completed = 0
        for fold_number, (outer, heldout) in enumerate(sorted(folds.items())):
            choice = selections[outer]["chosen"]; train_ids = [q for name, qs in folds.items() if name != outer for q in qs]
            model, positions = _fit([by_qid[q] for q in train_ids], data, answers, columns, choice["feature_set"], choice["params"])
            for name, value in zip(_columns(choice["feature_set"]), model.feature_importances_): gain[name] += float(value)
            for name, value in zip(_columns(choice["feature_set"]), model.booster_.feature_importance(importance_type="split")): split[name] += float(value)
            base = _predict(model, positions, [by_qid[q] for q in heldout], data); k = choice["viable_k"] or 100; base_metric = _metrics(base, {q: answers[q] for q in heldout})
            for column in columns:
                pos = columns.index(column); pred = {}
                for item in [by_qid[q] for q in heldout]:
                    x = _permuted(data, item, pos, SEED + fold_number * 1009 + pos)
                    scores = model.predict(x[:, positions]); order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), str(item["doc_ids"][i])))
                    pred[item["qid"]] = [str(item["doc_ids"][i]) for i in order]
                metric = _metrics(pred, {q: answers[q] for q in heldout})
                permutation[column]["predictions"].update(pred)
                permutation[column]["per_fold"][outer] = {"selected_k": k, "delta_recall_at_k": base_metric[f"recall@{k}"] - metric[f"recall@{k}"], "delta_recall_at_5": base_metric["recall@5"] - metric["recall@5"], "delta_precision_at_5": base_metric["precision@5"] - metric["precision@5"]}
                completed += 1; log.status(stage="exp028-permutation-importance", state="RUNNING", completed=completed, total=len(columns) * len(folds))
    result = {"schema_version": SCHEMA, "seed": SEED, "gain": dict(sorted(gain.items(), key=lambda x: -x[1])), "split": dict(sorted(split.items(), key=lambda x: -x[1])), "permutation": {}}
    for column, value in permutation.items():
        deltas = list(value["per_fold"].values())
        result["permutation"][column] = {"per_fold": value["per_fold"], "mean_delta_recall_at_selected_k": float(np.mean([x["delta_recall_at_k"] for x in deltas])), "mean_delta_recall_at_5": float(np.mean([x["delta_recall_at_5"] for x in deltas])), "mean_delta_precision_at_5": float(np.mean([x["delta_precision_at_5"] for x in deltas]))}
    with stage_run(output_dir, "exp028-permutation-importance-finalize", total=len(columns), v3_fingerprint=load_v3_manifest(v3)["content_fingerprint"]) as log:
        atomic_json(output_dir / "importance_report.json", result); log.set_telemetry(result)
    atomic_json(output_dir / "manifest.json", artifact_manifest(stage="exp028-importance", inputs={"feature_manifest_sha256": sha256_file(feature_dir / "manifest.json"), "oof_report_sha256": sha256_file(oof)}, config={"seed": SEED, "permutation_scope": "within_query"}, files=[output_dir / "importance_report.json"]))
    return result


def report(*, audit_path: Path, oof_path: Path, ablation_path: Path, importance_path: Path, output_dir: Path, v3: Path) -> dict[str, Any]:
    payload = {"schema_version": SCHEMA, "hypothesis": "shortlist-first nested selection can reduce K without changing retrieval membership", "audit": _json(audit_path), "oof": _json(oof_path), "ablation": _json(ablation_path), "importance": _json(importance_path), "exp027_comparison": {"prior_k": 64, "prior_pairs": 448000}, "capsules": "EXP-027 capsule inventory is reused only if EXP-028 passes; no capsules rebuilt here.", "reranker": "Not run."}
    with stage_run(output_dir, "exp028-report", total=1, v3_fingerprint=load_v3_manifest(v3)["content_fingerprint"]) as log:
        atomic_json(output_dir / "REPORT.json", payload); log.set_telemetry({"status": payload["oof"]["status"]})
    atomic_json(output_dir / "manifest.json", artifact_manifest(stage="exp028-report", inputs={"audit_sha256": sha256_file(audit_path), "oof_sha256": sha256_file(oof_path), "ablation_sha256": sha256_file(ablation_path), "importance_sha256": sha256_file(importance_path)}, config={"reranker": "not_run"}, files=[output_dir / "REPORT.json"]))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("stage", choices=("audit", "nested-oof", "ablation", "importance", "report")); p.add_argument("--cache-root", type=Path, default=ROOT / "cache" / "exp028_lambdamart_shortlist"); p.add_argument("--results-root", type=Path, default=ROOT / "results" / "exp028_lambdamart_shortlist"); p.add_argument("--candidates", type=Path, default=ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl"); p.add_argument("--sidecar", type=Path, default=ROOT / "cache" / "exp027_lambdamart_shortlist" / "provenance" / "provenance_sidecar.jsonl"); p.add_argument("--features", type=Path, default=ROOT / "cache" / "exp027_lambdamart_shortlist" / "features"); p.add_argument("--train", type=Path, default=ROOT / "public_test_dataset" / "train.json"); p.add_argument("--folds", type=Path, default=ROOT / "cache" / "cv_folds.json"); p.add_argument("--preprocessing", type=Path, default=ROOT / "cache" / "final_preprocessed_v2"); p.add_argument("--v3", type=Path, default=ROOT / "cache" / "structural_v3_e5_final_v1"); a = p.parse_args(argv)
    if a.stage == "audit": value = audit(candidates=a.candidates, sidecar=a.sidecar, feature_dir=a.features, folds_path=a.folds, train=a.train, preprocessing=a.preprocessing, v3=a.v3, output_dir=a.results_root / "audit")
    elif a.stage == "nested-oof": value = nested_oof(feature_dir=a.features, train=a.train, preprocessing=a.preprocessing, folds_path=a.folds, output_dir=a.results_root / "oof", v3=a.v3)
    elif a.stage == "ablation": value = ablation(predictions=a.results_root / "oof" / "ablation_predictions.jsonl", oof=a.results_root / "oof" / "oof_report.json", train=a.train, preprocessing=a.preprocessing, folds_path=a.folds, output_dir=a.results_root / "ablation", v3=a.v3)
    elif a.stage == "importance": value = importance(feature_dir=a.features, train=a.train, preprocessing=a.preprocessing, folds_path=a.folds, oof=a.results_root / "oof" / "oof_report.json", output_dir=a.results_root / "importance", v3=a.v3)
    else: value = report(audit_path=a.results_root / "audit" / "input_audit.json", oof_path=a.results_root / "oof" / "oof_report.json", ablation_path=a.results_root / "ablation" / "ablation_report.json", importance_path=a.results_root / "importance" / "importance_report.json", output_dir=a.results_root, v3=a.v3)
    print(json.dumps(value, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
