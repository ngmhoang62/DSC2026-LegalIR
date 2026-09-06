"""Strict nested probe for learning which document should occupy rank five.

Gemini artifacts are consumed read-only as research evidence.  All code and
outputs from this probe live in the EXP-final namespace.  Each outer fold uses
three folds for fitting and one disjoint calibration fold for complete recipe
selection.  Outer labels are only read after predictions have been locked.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from exp_final.slate import FoldLabelStats, replacement_class, slate_features  # noqa: E402

OUT = ROOT / "results/exp_final_retrieval/nested_slate_probe"
CALIBRATION = {0: 4, 1: 0, 2: 1, 3: 2, 4: 3}
ACTION_RATES = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
RISKS = (1.0, 2.0, 4.0)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(predictions, labels, qids):
    recall = []
    precision = []
    multi = []
    single = []
    mrr = []
    for qid in qids:
        gold = labels.get(qid, set())
        if not gold:
            continue
        top = predictions[qid][:5]
        hits = len(set(top) & gold)
        value = hits / len(gold)
        recall.append(value)
        precision.append(hits / 5)
        (single if len(gold) == 1 else multi).append(value)
        mrr.append(next((1 / rank for rank, doc in enumerate(top, 1) if doc in gold), 0.0))
    return {
        "recall_at_5": float(np.mean(recall)),
        "precision_at_5": float(np.mean(precision)),
        "single_gold_recall_at_5": float(np.mean(single)),
        "multi_gold_recall_at_5": float(np.mean(multi)),
        "mrr_at_5": float(np.mean(mrr)),
        "queries": len(recall),
    }


def selection_key(value, actions):
    return (
        value["recall_at_5"],
        value["precision_at_5"],
        value["multi_gold_recall_at_5"],
        value["mrr_at_5"],
        -actions,
    )


def blend(systems, weights=(0.3, 0.4, 0.3), k=10, depth=64):
    result = {}
    for qid in systems[0]:
        scores = {}
        for system, weight in zip(systems, weights):
            for rank, doc in enumerate(system[qid][:depth], 1):
                scores[doc] = scores.get(doc, 0.0) + weight / (k + rank)
        result[qid] = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return result


def load_titles():
    db_path = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    titles = {
        str(doc): json.loads(payload).get("retrieval_name", "")
        for doc, payload in db.execute("SELECT doc,payload FROM documents")
    }
    db.close()
    return titles


def build_rows(qids, base, systems, questions, titles, stats, labels, training):
    rows = []
    targets = []
    owners = []
    candidates = []
    for offset, qid in enumerate(qids, 1):
        order = base[qid]
        local_systems = {name: ranking[qid] for name, ranking in systems.items()}
        query_gold = labels[qid] if training else None
        for candidate_index in range(5, min(10, len(order))):
            rows.append(slate_features(
                questions[qid], order, candidate_index, local_systems, titles,
                stats, query_gold=query_gold,
            ))
            targets.append(replacement_class(labels[qid], order[4], order[candidate_index]))
            owners.append(qid)
            candidates.append(order[candidate_index])
        if offset % 1000 == 0:
            print(f"features={offset}/{len(qids)}", flush=True)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(targets, dtype=np.int8),
        np.asarray(owners, dtype=object),
        np.asarray(candidates, dtype=object),
    )


def fit_model(family, x, y):
    if family == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.2, class_weight="balanced", solver="lbfgs",
                max_iter=2000, random_state=112,
            ),
        )
    elif family == "lgb":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, num_leaves=7,
            min_child_samples=20, learning_rate=0.03, n_estimators=100,
            class_weight="balanced", feature_fraction=1.0,
            bagging_fraction=1.0, deterministic=True, force_col_wise=True,
            n_jobs=4, random_state=112, verbosity=-1,
        )
    else:
        raise ValueError(family)
    model.fit(x, y)
    classes = model[-1].classes_ if family == "lr" else model.classes_
    if list(map(int, classes)) != [0, 1, 2]:
        raise ValueError(f"Training split lacks a replacement class: {classes}")
    return model


def utilities(model, x, risk):
    probabilities = model.predict_proba(x)
    return probabilities[:, 2] - risk * probabilities[:, 0]


def best_per_query(qids, owners, candidates, values):
    selected = {}
    for row, (qid, doc, value) in enumerate(zip(owners, candidates, values)):
        item = (float(value), str(doc), row)
        if qid not in selected or item[:2] > selected[qid][:2]:
            selected[qid] = item
    if set(selected) != set(qids):
        raise ValueError("Missing query in slate utility rows")
    return selected


def apply_rate(base, qids, selected, rate):
    predictions = {qid: list(base[qid]) for qid in qids}
    count = min(len(qids), int(np.ceil(rate * len(qids))))
    chosen = sorted(qids, key=lambda qid: (-selected[qid][0], qid))[:count]
    events = []
    for qid in chosen:
        candidate = selected[qid][1]
        old = predictions[qid][4]
        index = predictions[qid].index(candidate)
        predictions[qid].pop(index)
        predictions[qid].insert(4, candidate)
        events.append({"qid": qid, "displaced": old, "candidate": candidate, "utility": selected[qid][0]})
    return predictions, events


def event_outcomes(events, labels):
    wins = losses = neutral = 0
    gain = 0.0
    for event in events:
        gold = labels[event["qid"]]
        before = float(event["displaced"] in gold) / len(gold)
        after = float(event["candidate"] in gold) / len(gold)
        delta = after - before
        gain += delta
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            neutral += 1
    return {"wins": wins, "losses": losses, "neutral": neutral, "query_recall_points": gain}


def main():
    import exp109b_encoder_complementarity as old

    source_paths = {
        "xgb131": ROOT / "results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json",
        "lgb131": ROOT / "results/gemini/exp_authority_131d/lgbm_131d_OOF_PREDICTIONS.json",
        "profile": ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json",
        "memory": ROOT / "results/exp_final_retrieval/memory_ltr_probe/fold_0/PREDICTIONS.json",
        "kernel": ROOT / "results/exp_final_retrieval/kernel_ltr_probe/l15_t5/PREDICTIONS.json",
    }
    # The memory prediction file is fold-specific; merge all five sealed files.
    memory = {}
    for fold in range(5):
        memory.update(read(ROOT / f"results/exp_final_retrieval/memory_ltr_probe/fold_{fold}/PREDICTIONS.json"))
    sources = {
        "xgb131": read(source_paths["xgb131"]),
        "lgb131": read(source_paths["lgb131"]),
        "profile": read(source_paths["profile"]),
        "memory": memory,
        "kernel": read(source_paths["kernel"]),
    }
    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache/cv_folds.json")
    train = read(ROOT / "public_test_dataset/train.json")
    questions = {str(qid): row["question"] for qid, row in train.items()}
    titles = load_titles()
    base = blend([sources["xgb131"], sources["lgb131"], sources["profile"]])
    all_qids = [qid for fold in range(5) for qid in folds[f"fold_{fold}"] if labels.get(qid)]
    report = {
        "status": "COMPLETE_STRICT_NESTED_SLATE_PROBE",
        "protocol": "three inner folds fit; one disjoint calibration fold selects complete recipe; no refit; outer labels read after prediction lock",
        "gemini_namespace_written": False,
        "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for name, path in source_paths.items()},
        "base": metrics(base, labels, all_qids),
        "folds": {},
    }
    oof = {}
    locks = {}
    for outer in range(5):
        calibration = CALIBRATION[outer]
        inner_folds = [fold for fold in range(5) if fold not in (outer, calibration)]
        inner_qids = [qid for fold in inner_folds for qid in folds[f"fold_{fold}"] if labels.get(qid)]
        cal_qids = [qid for qid in folds[f"fold_{calibration}"] if labels.get(qid)]
        outer_qids = [qid for qid in folds[f"fold_{outer}"] if labels.get(qid)]
        print(f"outer={outer} fit={inner_folds} calibration={calibration}", flush=True)
        stats = FoldLabelStats(labels, inner_qids)
        x_train, y_train, _, _ = build_rows(inner_qids, base, sources, questions, titles, stats, labels, True)
        x_cal, _, cal_owners, cal_candidates = build_rows(cal_qids, base, sources, questions, titles, stats, labels, False)
        x_outer, _, outer_owners, outer_candidates = build_rows(outer_qids, base, sources, questions, titles, stats, labels, False)
        class_counts = {str(label): int((y_train == label).sum()) for label in (0, 1, 2)}
        trials = [{"recipe": {"family": "noop", "risk": 0.0, "action_rate": 0.0}, "metrics": metrics(base, labels, cal_qids), "actions": 0}]
        models = {}
        for family in ("lr", "lgb"):
            model = fit_model(family, x_train, y_train)
            models[family] = model
            for risk in RISKS:
                selected = best_per_query(cal_qids, cal_owners, cal_candidates, utilities(model, x_cal, risk))
                for rate in ACTION_RATES[1:]:
                    predictions, events = apply_rate(base, cal_qids, selected, rate)
                    trials.append({
                        "recipe": {"family": family, "risk": risk, "action_rate": rate},
                        "metrics": metrics(predictions, labels, cal_qids),
                        "actions": len(events),
                        "outcomes": event_outcomes(events, labels),
                    })
        winner = max(trials, key=lambda row: selection_key(row["metrics"], row["actions"]))
        recipe = winner["recipe"]
        if recipe["family"] == "noop":
            outer_predictions = {qid: list(base[qid]) for qid in outer_qids}
            outer_events = []
        else:
            model = models[recipe["family"]]
            selected = best_per_query(
                outer_qids, outer_owners, outer_candidates,
                utilities(model, x_outer, recipe["risk"]),
            )
            outer_predictions, outer_events = apply_rate(base, outer_qids, selected, recipe["action_rate"])
        # Prediction lock precedes any outer metric calculation.
        fold_dir = OUT / f"fold_{outer}"
        write(fold_dir / "PREDICTIONS_LOCK.json", outer_predictions)
        lock = {
            "outer_fold": outer,
            "calibration_fold": calibration,
            "inner_folds": inner_folds,
            "recipe": recipe,
            "training_class_counts": class_counts,
            "calibration_winner": winner,
            "prediction_sha256": sha(fold_dir / "PREDICTIONS_LOCK.json"),
        }
        write(fold_dir / "SELECTION_LOCK.json", lock)
        outer_value = metrics(outer_predictions, labels, outer_qids)
        base_value = metrics(base, labels, outer_qids)
        outcomes = event_outcomes(outer_events, labels)
        report["folds"][f"fold_{outer}"] = {
            "selection_lock": lock,
            "base": base_value,
            "slate": outer_value,
            "delta": outer_value["recall_at_5"] - base_value["recall_at_5"],
            "actions": len(outer_events),
            "outcomes": outcomes,
            "top_calibration_trials": sorted(trials, key=lambda row: selection_key(row["metrics"], row["actions"]), reverse=True)[:10],
        }
        oof.update(outer_predictions)
        locks[f"fold_{outer}"] = recipe
        print(f"outer={outer} recipe={recipe} delta={report['folds'][f'fold_{outer}']['delta']:+.9f} actions={len(outer_events)} outcomes={outcomes}", flush=True)
    write(OUT / "OOF_PREDICTIONS.json", oof)
    aggregate = metrics(oof, labels, all_qids)
    report["oof"] = aggregate
    report["delta"] = aggregate["recall_at_5"] - report["base"]["recall_at_5"]
    report["nonnegative_folds"] = sum(row["delta"] >= 0 for row in report["folds"].values())
    report["selected_recipes"] = locks
    write(OUT / "NESTED_SLATE_REPORT.json", report)
    print(json.dumps({"base": report["base"], "oof": aggregate, "delta": report["delta"], "nonnegative_folds": report["nonnegative_folds"], "selected_recipes": locks}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
