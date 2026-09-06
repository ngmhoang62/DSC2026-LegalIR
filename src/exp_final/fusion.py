from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .contracts import CACHE, Progress, digest, read, rank, rrf, sha, write


SOURCE_FIELDS = ("score", "rank", "recip", "z", "margin_rank1", "margin_rank5", "margin_rank10", "present", "score_available", "rank_known")
META = ("source_agreement_top5", "source_agreement_top10", "source_agreement_top20", "dense_top1_score", "dense_top2_score", "dense_top1_top2_gap", "parent_chunk_count", "parent_token_length", "query_token_length")


def feature_names(block):
    sources = ["e5", "lal", "bm25"] + (["trigram"] if block >= 1 else [])
    names = [s+"_"+f for s in sources for f in SOURCE_FIELDS] + list(META)
    if block >= 2:
        from exp109c_latent_condition_late_interaction import LATE_FEATURES
        names += ["jina_scored"] + list(LATE_FEATURES)
    return names


def features(data, store, q, docs, block):
    sources = ["e5", "lal", "bm25"] + (["trigram"] if block >= 1 else [])
    dependencies = [store.get(q, s) for s in sources + (["jina"] if block >= 2 else [])]
    signature = digest([data.fingerprint, q, docs, block, "features-v2", dependencies])
    path = CACHE / "features" / f"{signature}.npy"
    legacy = CACHE.parent / "exp112_task_adaptive_retrieval" / "features" / f"{signature}.npy"
    if not path.exists() and legacy.exists():
        marker = legacy.with_suffix('.json')
        if not marker.exists() or read(marker)["sha256"] != sha(legacy):
            raise ValueError("Legacy EXP-112 feature cache hash missing/mismatch")
        return np.load(legacy, mmap_mode="r")
    if path.exists():
        marker = path.with_suffix('.json')
        if not marker.exists() or read(marker)["sha256"] != sha(path):
            raise ValueError("Feature cache hash missing/mismatch")
        return np.load(path, mmap_mode="r")
    columns, source_ranks, dense_values = [], [], []
    for source in sources:
        rows = store.get(q, source)
        if rows is None:
            raise ValueError(f"Missing frozen source {source}/{q}")
        mapping = {r["doc_id"]: (int(r["rank"]), float(r["score"])) for r in rows}
        values = np.array([r["score"] for r in rows], dtype=np.float64)
        known = np.array([d in mapping for d in docs], dtype=bool)
        scores = np.array([mapping.get(d, (501, 0.))[1] for d in docs])
        available = known.copy()
        if source in ("e5", "lal"):
            missing = [d for d, k in zip(docs, known) if not k]
            if missing:
                scores[~known] = data.exact(q, missing, source)
                available[:] = True
            dense_values.append(scores)
        ranks = np.array([mapping.get(d, (501, 0.))[0] for d in docs], dtype=float)
        source_ranks.append(ranks)
        std, mean = (float(values.std()), float(values.mean())) if len(values) else (0., 0.)
        normalized = np.where(available, (scores - mean) / std, 0.) if std > 1e-12 else np.zeros_like(scores)
        sorted_values = np.sort(values)[::-1]
        margins = [np.where(available, (sorted_values[min(k-1, len(values)-1)] if len(values) else 0.)-scores, 0.) for k in (1, 5, 10)]
        columns.extend([scores, ranks, np.where(known, 1/ranks, 0.), normalized, *margins, known, available, known])
    columns.extend([(np.array(source_ranks) <= k).sum(0) for k in (5, 10, 20)])
    dv = np.sort(np.array(dense_values), axis=0)
    columns.extend([dv[-1], dv[-2], dv[-1]-dv[-2],
                    [data.metadata[d]["parent_chunk_count"] for d in docs],
                    [data.metadata[d]["parent_token_length"] for d in docs], [len(data.questions[q].split())]*len(docs)])
    if block >= 2:
        from exp109c_latent_condition_late_interaction import LATE_FEATURES
        js = store.get(q, "jina")
        if js is None:
            raise ValueError("Jina block requested without sanitized query features")
        jm = {r["doc_id"]: r["features"] for r in js}
        columns.append([d in jm for d in docs])
        columns.extend([[jm.get(d, {}).get(k, 0.) for d in docs] for k in LATE_FEATURES])
    x = np.array(columns, dtype=np.float32).T
    if x.shape != (len(docs), len(feature_names(block))) or not np.isfinite(x).all():
        raise ValueError("Feature schema/non-finite mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy"); np.save(tmp, x); tmp.replace(path)
    write(path.with_suffix('.json'), dict(signature=signature, sha256=sha(path), dtype=str(x.dtype), shape=list(x.shape)))
    return x


def fit_ranker(data, store, qids, family, block, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    signature = digest([data.fingerprint, qids, family, block, feature_names(block), sha(Path(__file__))])
    marker = path.with_suffix(".json")
    if marker.exists():
        if read(marker)["signature"] != signature:
            raise ValueError("ML resume scope mismatch")
        if read(marker).get("sha256") != sha(path):
            raise ValueError("ML checkpoint hash mismatch")
        with path.open("rb") as f:
            return pickle.load(f)
    sizes = [len(set(store.candidates(q)) | data.gold[q]) for q in qids if data.gold[q]]
    temp = path.with_suffix(".matrix.npy")
    x = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(sum(sizes), len(feature_names(block))))
    y = np.empty(sum(sizes), dtype=np.int8)
    offset = 0
    progress = Progress('ml_features_'+path.stem, len(qids))
    for qi, q in enumerate(qids):
        if not data.gold[q]:
            continue
        docs = list(dict.fromkeys(store.candidates(q) + sorted(data.gold[q])))
        x[offset:offset+len(docs)] = features(data, store, q, docs, block)
        y[offset:offset+len(docs)] = [d in data.gold[q] for d in docs]
        offset += len(docs)
        progress.update(qi+1)
    if family == "lm":
        import lightgbm as lgb
        model = lgb.LGBMRanker(objective="lambdarank", num_leaves=7, min_child_samples=50, learning_rate=.05,
                              n_estimators=300, feature_fraction=1., bagging_fraction=1., deterministic=True,
                              force_col_wise=True, n_jobs=4, random_state=112, verbosity=-1)
        model.fit(x, y, group=sizes, eval_at=[5])
    elif family == "lr":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", solver="liblinear", max_iter=2000, random_state=112))
        model.fit(x, y)
    else:
        raise ValueError(family)
    with path.with_suffix(".tmp").open("wb") as f:
        pickle.dump(model, f)
    path.with_suffix(".tmp").replace(path)
    write(marker, {"signature": signature, "sha256": sha(path), "training_qids": qids, "family": family, "block": block, "feature_names": feature_names(block)})
    del x
    return model


def upstream(data, store, q, model, recipe, adapted=None):
    frozen = store.candidates(q)
    if recipe["family"] == "rrf":
        sources = store.rankings(q)
        order, scores = rrf([sources[s][:100] for s in ("e5", "lal", "bm25")])
        return {"order": order, "scores": scores}
    docs = frozen if recipe["pool"] == "frozen" else list(dict.fromkeys(frozen + adapted["order"][:100]))
    x = features(data, store, q, docs, recipe["block"])
    scores = model.decision_function(x) if recipe["family"] == "lr" else model.predict(x)
    ml = rank(scores, docs)
    if recipe["pool"] == "frozen":
        sm = dict(zip(docs, map(float, scores)))
        return {"order": ml, "scores": [sm[d] for d in ml]}
    beta = recipe["beta"]
    # Keep adapted global ranks, not ranks recomputed in the union.
    ar = {d: i for i, d in enumerate(adapted["order"], 1)}
    mr = {d: i for i, d in enumerate(ml, 1)}
    values = [(1-beta)/(32+mr[d]) + beta/(32+ar[d]) for d in docs]
    order = rank(values, docs); sm = dict(zip(docs, values))
    return {"order": order, "scores": [sm[d] for d in order]}
