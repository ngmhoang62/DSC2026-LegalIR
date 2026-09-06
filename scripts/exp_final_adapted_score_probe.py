"""Development diagnostic for continuous adapted-E5 score features.

The EXP-112 adapter produced a valid OOF ranking for every query, but later
EXP-FINAL probes exposed only a rank-truncated view of that signal.  This
probe preserves the existing OOF systems and asks a narrow question: does a
level-2 ranker benefit materially from the adapter's raw parent score,
full-corpus rank, cutoff margins, and query drift?

This is deliberately labelled a development diagnostic.  Although every
base prediction is OOF for its own query, a production-quality outer estimate
would require nested base-model fits so that level-2 training artifacts also
exclude the target outer fold.  A weak result here rejects that expensive
nested experiment; a strong result only authorizes implementing it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/exp_final_retrieval/adapted_score_probe"
DEPTH = 64


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def adapted_feature_names():
    return [
        "adapted_full_rank",
        "adapted_full_recip",
        "adapted_raw_score",
        "adapted_score_z_top500",
        "adapted_score_minus_top1",
        "adapted_score_minus_rank5",
        "adapted_score_minus_rank6",
        "adapted_score_minus_rank10",
        "adapted_inside_top5",
        "adapted_inside_top10",
        "adapted_query_drift",
        "adapted_rank5_rank6_gap",
        "adapted_top10_score_std",
    ]


def load_systems_and_scores():
    sys.path.insert(0, str(ROOT / "scripts"))
    import exp_final_meta_ltr_probe as meta

    systems = meta.load_systems()
    folds = read(ROOT / "cache/cv_folds.json")
    compact = {}
    started = time.monotonic()
    completed = 0
    for fold in range(5):
        folder = ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{fold}/test-query"
        for qid in folds[f"fold_{fold}"]:
            qid = str(qid)
            if qid not in systems["memory"]:
                continue
            row = read(folder / f"{qid}.json")
            order = list(map(str, row["order"]))
            scores = np.asarray(row["scores"], dtype=np.float32)
            if len(order) != len(scores) or len(order) != 8507:
                raise ValueError(f"Adapted full-corpus row contract mismatch: {qid}")
            wanted = set().union(*(set(systems[name][qid][:DEPTH]) for name in systems))
            positions = {doc: index for index, doc in enumerate(order) if doc in wanted}
            if len(positions) != len(wanted):
                missing = sorted(wanted - set(positions))[:5]
                raise ValueError(f"Adapted scores missing candidate(s) {qid}: {missing}")
            context = scores[:500]
            mean = float(context.mean())
            std = float(context.std())
            cut = {rank: float(scores[rank - 1]) for rank in (1, 5, 6, 10)}
            query_values = {
                "frozen_query_cosine": float(row["frozen_query_cosine"]),
                "mean_top500": mean,
                "std_top500": std,
                "rank5_rank6_gap": cut[5] - cut[6],
                "top10_std": float(scores[:10].std()),
            }
            docs = {}
            for doc in wanted:
                index = positions[doc]
                score = float(scores[index])
                docs[doc] = (index + 1, score, (score - mean) / std if std > 1e-12 else 0.0,
                             score - cut[1], score - cut[5], score - cut[6], score - cut[10])
            compact[qid] = (query_values, docs)
            completed += 1
            if completed % 500 == 0:
                print(f"compact_adapted={completed}/6991 elapsed={time.monotonic()-started:.1f}s", flush=True)
    if set(compact) != set(systems["memory"]):
        raise ValueError("Adapted compact qid scope mismatch")
    return systems, compact


def continuous_features(qid, docs, compact):
    query_values, values = compact[qid]
    result = np.zeros((len(docs), len(adapted_feature_names())), dtype=np.float32)
    drift = 1.0 - query_values["frozen_query_cosine"]
    for row, doc in enumerate(docs):
        rank, score, z, m1, m5, m6, m10 = values[doc]
        result[row] = (
            rank,
            1.0 / rank,
            score,
            z,
            m1,
            m5,
            m6,
            m10,
            float(rank <= 5),
            float(rank <= 10),
            drift,
            query_values["rank5_rank6_gap"],
            query_values["top10_std"],
        )
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite adapted feature: {qid}")
    return result


def fit_ranker(x, y, groups):
    import lightgbm as lgb

    model = lgb.LGBMRanker(
        objective="lambdarank",
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=7,
        min_child_samples=50,
        lambdarank_truncation_level=30,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        deterministic=True,
        force_col_wise=True,
        n_jobs=4,
        random_state=7112,
        verbosity=-1,
    )
    model.fit(x, y, group=groups, eval_at=[5])
    return model


def main():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import exp109b_encoder_complementarity as old
    import exp_final_meta_ltr_probe as meta

    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache/cv_folds.json")
    systems, compact = load_systems_and_scores()
    names = list(systems)
    base_names = meta.feature_names(names)
    predictions = {}
    fold_reports = {}
    feature_importances = {}
    for outer in range(5):
        train_qids = [q for fold in range(5) if fold != outer for q in folds[f"fold_{fold}"] if labels.get(q)]
        test_qids = [q for q in folds[f"fold_{outer}"] if labels.get(q)]
        train_rows, train_groups, target = [], [], []
        for index, qid in enumerate(train_qids):
            docs, rank_features = meta.rows_for_query(qid, systems, names)
            raw_features = continuous_features(qid, docs, compact)
            train_rows.append(np.concatenate((rank_features, raw_features), axis=1))
            train_groups.append(len(docs))
            target.extend(doc in labels[qid] for doc in docs)
            if (index + 1) % 1000 == 0:
                print(f"outer={outer} train={index+1}/{len(train_qids)}", flush=True)
        xtrain = np.concatenate(train_rows)
        model = fit_ranker(xtrain, np.asarray(target, dtype=np.int8), train_groups)
        del xtrain, train_rows

        local = {}
        for index, qid in enumerate(test_qids):
            docs, rank_features = meta.rows_for_query(qid, systems, names)
            raw_features = continuous_features(qid, docs, compact)
            score = model.predict(np.concatenate((rank_features, raw_features), axis=1))
            order = [docs[row] for row in sorted(range(len(docs)), key=lambda row: (-float(score[row]), docs[row]))]
            seen = set(order)
            local[qid] = order + [doc for doc in systems["memory"][qid] if doc not in seen]
            if (index + 1) % 400 == 0:
                print(f"outer={outer} score={index+1}/{len(test_qids)}", flush=True)
        predictions.update(local)
        baseline = meta.metrics(systems["memory"], labels, test_qids)
        measured = meta.metrics(local, labels, test_qids)
        fold_reports[f"fold_{outer}"] = {
            "baseline": baseline,
            "continuous_adapted_ltr": measured,
            "delta": measured["recall_at_5"] - baseline["recall_at_5"],
        }
        all_names = base_names + adapted_feature_names()
        importance = sorted(zip(all_names, model.feature_importances_.tolist()), key=lambda item: (-item[1], item[0]))
        feature_importances[f"fold_{outer}"] = importance[:25]
        print(f"outer={outer} recall={measured['recall_at_5']:.9f} delta={fold_reports[f'fold_{outer}']['delta']:+.9f}", flush=True)
        write(OUT / f"fold_{outer}/PREDICTIONS.json", local)

    qids = [q for q in systems["memory"] if labels.get(q)]
    baseline = meta.metrics(systems["memory"], labels, qids)
    measured = meta.metrics(predictions, labels, qids)
    deltas = [fold_reports[f"fold_{fold}"]["delta"] for fold in range(5)]
    report = {
        "status": "COMPLETE_EXPOSED_ADAPTED_SCORE_DIAGNOSTIC",
        "scope_warning": (
            "Each adapted score is OOF for its query, but level-2 outer isolation is not nested: "
            "base adapters used to form level-2 training features may have seen the target fold. "
            "This report estimates upside only and cannot promote a final system."
        ),
        "hypothesis": "Continuous adapted-E5 cutoff evidence is materially stronger than rank-only stacking.",
        "depth": DEPTH,
        "feature_names": base_names + adapted_feature_names(),
        "baseline": baseline,
        "continuous_adapted_ltr": measured,
        "delta": measured["recall_at_5"] - baseline["recall_at_5"],
        "fold_deltas": deltas,
        "nonnegative_folds": sum(delta >= 0 for delta in deltas),
        "folds": fold_reports,
        "feature_importances": feature_importances,
        "decision": (
            "AUTHORIZE_NESTED_ADAPTED_FEATURES"
            if measured["recall_at_5"] - baseline["recall_at_5"] >= 0.005 and sum(delta >= 0 for delta in deltas) >= 4
            else "REJECT_NESTED_ADAPTED_FEATURES"
        ),
    }
    write(OUT / "ADAPTED_SCORE_DIAGNOSTIC.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
