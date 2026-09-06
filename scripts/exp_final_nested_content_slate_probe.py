"""Nested fifth-slot selection with query-conditioned E5 content geometry."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp_final_nested_slate_probe as common  # noqa: E402
from exp_final.data import Data  # noqa: E402
from exp_final.slate import FoldLabelStats, dense_relation_features  # noqa: E402

OUT = ROOT / "results/exp_final_retrieval/nested_content_slate_probe"
CACHE = ROOT / "cache/exp_final_retrieval/content_slate_features"


def load_sources():
    memory = {}
    for fold in range(5):
        memory.update(common.read(ROOT / f"results/exp_final_retrieval/memory_ltr_probe/fold_{fold}/PREDICTIONS.json"))
    return {
        "xgb131": common.read(ROOT / "results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json"),
        "lgb131": common.read(ROOT / "results/gemini/exp_authority_131d/lgbm_131d_OOF_PREDICTIONS.json"),
        "profile": common.read(ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json"),
        "memory": memory,
        "kernel": common.read(ROOT / "results/exp_final_retrieval/kernel_ltr_probe/l15_t5/PREDICTIONS.json"),
    }


def content_cache(base, qids):
    path = CACHE / "e5_top10_content_relation.f32.npy"
    ids_path = CACHE / "qids.json"
    marker_path = CACHE / "manifest.json"
    expected = {
        "contract": "normalized-mean-of-query-top2-parent-chunks-v1",
        "qids": qids,
        "base_digest": common.sha(ROOT / "results/exp_final_retrieval/gemini_evidence_audit/GEMINI_EVIDENCE_AUDIT.json"),
        "matrix_sha": common.sha(ROOT / "cache/e5_final_v1/embeddings.f16.npy"),
        "query_sha": common.sha(ROOT / "cache/exp021_e5_dense_candidates/query_embeddings/train_queries.f32.npy"),
    }
    if path.exists() and ids_path.exists() and marker_path.exists():
        marker = common.read(marker_path)
        comparable = {key: marker[key] for key in expected}
        if comparable != expected or marker.get("sha256") != common.sha(path):
            raise ValueError("Content feature cache provenance mismatch")
        values = np.load(path, mmap_mode="r")
        if values.shape != (len(qids), 5, 24):
            raise ValueError("Content feature cache shape mismatch")
        return {qid: values[index] for index, qid in enumerate(qids)}
    CACHE.mkdir(parents=True, exist_ok=True)
    data = Data()
    matrix = data.matrix("e5")
    values = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+", dtype=np.float32, shape=(len(qids), 5, 24))
    for qi, qid in enumerate(qids):
        query = data.query_vector(qid, "e5")
        vectors = []
        parent_scores = []
        for doc in base[qid][:10]:
            rows = np.asarray(matrix[data.positions[data.doc_row[doc]]], dtype=np.float32)
            rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
            scores = rows @ query
            count = min(2, len(scores))
            selected = np.argpartition(scores, -count)[-count:]
            vector = rows[selected].mean(0)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            vectors.append(vector)
            parent_scores.append(float(scores[selected].mean()))
        for candidate_index in range(5, 10):
            relation = dense_relation_features(query, vectors, candidate_index)
            values[qi, candidate_index - 5] = np.concatenate([
                relation,
                np.asarray([
                    parent_scores[candidate_index], parent_scores[4],
                    parent_scores[candidate_index] - parent_scores[4],
                ], dtype=np.float32),
            ])
        if (qi + 1) % 100 == 0 or qi + 1 == len(qids):
            print(f"content_geometry={qi+1}/{len(qids)}", flush=True)
    values.flush()
    del values
    path.with_suffix(".tmp.npy").replace(path)
    common.write(ids_path, qids)
    common.write(marker_path, {**expected, "shape": [len(qids), 5, 24], "dtype": "float32", "sha256": common.sha(path)})
    values = np.load(path, mmap_mode="r")
    return {qid: values[index] for index, qid in enumerate(qids)}


def append_content(x, owners, content):
    counters = {}
    rows = np.empty((len(x), 24), dtype=np.float32)
    for index, qid in enumerate(owners):
        local = counters.get(qid, 0)
        rows[index] = content[qid][local]
        counters[qid] = local + 1
    if any(count != 5 for count in counters.values()):
        raise ValueError("Content rows do not align with five candidate rows per query")
    return np.concatenate([x, rows], axis=1)


def fit_actionable(family, x, y):
    mask = y != 1
    target = (y[mask] == 2).astype(np.int8)
    if family == "action_lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.2, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=112))
    elif family == "action_lgb":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(objective="binary", num_leaves=7, min_child_samples=15, learning_rate=.03, n_estimators=100, class_weight="balanced", deterministic=True, force_col_wise=True, n_jobs=4, random_state=112, verbosity=-1)
    else:
        raise ValueError(family)
    model.fit(x[mask], target)
    return model


def action_utility(model, x):
    probability = model.predict_proba(x)[:, 1]
    return probability - (1 - probability)


def main():
    import exp109b_encoder_complementarity as old

    labels, _ = old.canonical_labels()
    folds = common.read(ROOT / "cache/cv_folds.json")
    train = common.read(ROOT / "public_test_dataset/train.json")
    questions = {str(qid): row["question"] for qid, row in train.items()}
    titles = common.load_titles()
    sources = load_sources()
    base = common.blend([sources["xgb131"], sources["lgb131"], sources["profile"]])
    all_qids = [qid for fold in range(5) for qid in folds[f"fold_{fold}"] if labels.get(qid)]
    content = content_cache(base, all_qids)
    report = {
        "status": "COMPLETE_STRICT_NESTED_CONTENT_SLATE_PROBE",
        "protocol": "three inner folds fit; one calibration fold selects complete policy; outer prediction lock precedes evaluation",
        "gemini_namespace_written": False,
        "base": common.metrics(base, labels, all_qids),
        "folds": {},
    }
    oof = {}
    for outer in range(5):
        calibration = common.CALIBRATION[outer]
        inner_folds = [fold for fold in range(5) if fold not in (outer, calibration)]
        inner_qids = [qid for fold in inner_folds for qid in folds[f"fold_{fold}"] if labels.get(qid)]
        cal_qids = [qid for qid in folds[f"fold_{calibration}"] if labels.get(qid)]
        outer_qids = [qid for qid in folds[f"fold_{outer}"] if labels.get(qid)]
        print(f"outer={outer} fit={inner_folds} calibration={calibration}", flush=True)
        stats = FoldLabelStats(labels, inner_qids)
        x_train, y_train, train_owners, _ = common.build_rows(inner_qids, base, sources, questions, titles, stats, labels, True)
        x_cal, _, cal_owners, cal_candidates = common.build_rows(cal_qids, base, sources, questions, titles, stats, labels, False)
        x_outer, _, outer_owners, outer_candidates = common.build_rows(outer_qids, base, sources, questions, titles, stats, labels, False)
        x_train = append_content(x_train, train_owners, content)
        x_cal = append_content(x_cal, cal_owners, content)
        x_outer = append_content(x_outer, outer_owners, content)
        trials = [{"recipe": {"family": "noop", "risk": 0., "action_rate": 0.}, "metrics": common.metrics(base, labels, cal_qids), "actions": 0}]
        utilities = {}
        for family in ("lr", "lgb"):
            model = common.fit_model(family, x_train, y_train)
            for risk in common.RISKS:
                utilities[(family, risk)] = common.utilities(model, x_cal, risk)
        action_models = {}
        for family in ("action_lr", "action_lgb"):
            action_models[family] = fit_actionable(family, x_train, y_train)
            utilities[(family, 1.)] = action_utility(action_models[family], x_cal)
        for (family, risk), values in utilities.items():
            selected = common.best_per_query(cal_qids, cal_owners, cal_candidates, values)
            for rate in common.ACTION_RATES[1:]:
                predictions, events = common.apply_rate(base, cal_qids, selected, rate)
                trials.append({"recipe": {"family": family, "risk": risk, "action_rate": rate}, "metrics": common.metrics(predictions, labels, cal_qids), "actions": len(events), "outcomes": common.event_outcomes(events, labels)})
        winner = max(trials, key=lambda row: common.selection_key(row["metrics"], row["actions"]))
        recipe = winner["recipe"]
        if recipe["family"] == "noop":
            predictions = {qid: list(base[qid]) for qid in outer_qids};events = []
        else:
            if recipe["family"].startswith("action_"):
                model = action_models[recipe["family"]];values = action_utility(model, x_outer)
            else:
                model = common.fit_model(recipe["family"], x_train, y_train);values = common.utilities(model, x_outer, recipe["risk"])
            selected = common.best_per_query(outer_qids, outer_owners, outer_candidates, values)
            predictions, events = common.apply_rate(base, outer_qids, selected, recipe["action_rate"])
        fold_dir = OUT / f"fold_{outer}"
        common.write(fold_dir / "PREDICTIONS_LOCK.json", predictions)
        lock = {"outer_fold": outer, "calibration_fold": calibration, "inner_folds": inner_folds, "recipe": recipe, "calibration_winner": winner, "prediction_sha256": common.sha(fold_dir / "PREDICTIONS_LOCK.json")}
        common.write(fold_dir / "SELECTION_LOCK.json", lock)
        value = common.metrics(predictions, labels, outer_qids);baseline = common.metrics(base, labels, outer_qids)
        report["folds"][f"fold_{outer}"] = {"selection_lock": lock, "base": baseline, "slate": value, "delta": value["recall_at_5"] - baseline["recall_at_5"], "actions": len(events), "outcomes": common.event_outcomes(events, labels), "top_calibration_trials": sorted(trials, key=lambda row: common.selection_key(row["metrics"], row["actions"]), reverse=True)[:10]}
        oof.update(predictions)
        print(f"outer={outer} recipe={recipe} delta={report['folds'][f'fold_{outer}']['delta']:+.9f} outcomes={report['folds'][f'fold_{outer}']['outcomes']}", flush=True)
    common.write(OUT / "OOF_PREDICTIONS.json", oof)
    report["oof"] = common.metrics(oof, labels, all_qids)
    report["delta"] = report["oof"]["recall_at_5"] - report["base"]["recall_at_5"]
    report["nonnegative_folds"] = sum(row["delta"] >= 0 for row in report["folds"].values())
    common.write(OUT / "NESTED_CONTENT_SLATE_REPORT.json", report)
    print(json.dumps({"base": report["base"], "oof": report["oof"], "delta": report["delta"], "nonnegative_folds": report["nonnegative_folds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
