"""Supervised legal case-linker probe.

Rank support queries for a new query, then propagate their known legal-document
labels.  The experiment is fold-isolated: an outer fold is never used in the
pair ranker, label-frequency features, or case memory supporting that fold.
This is a development probe over previously exposed CV folds, not a final lock.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/exp_final_retrieval/memory_ltr_probe"
OUT = ROOT / "results/exp_final_retrieval/case_linker_probe"
TOKEN = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[a-zA-ZÀ-ỹĐđ]+", re.UNICODE)
NUMBER = re.compile(r"[0-9]+(?:[.,][0-9]+)*")


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(path)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def plain(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn").replace("đ", "d")


def text_state(text):
    accented = tuple(TOKEN.findall(text.lower()))
    bare = tuple(TOKEN.findall(plain(text)))
    return {
        "tokens": set(accented), "bare": set(bare), "numbers": set(NUMBER.findall(text)),
        "prefix": tuple(bare[:3]), "length": len(accented),
    }


def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def top_indices(scores, n, exclude=None):
    values = np.asarray(scores, dtype=np.float32)
    if exclude is not None:
        values = values.copy(); values[exclude] = -np.inf
    take = min(n, len(values) - (1 if exclude is not None else 0))
    part = np.argpartition(-values, take - 1)[:take]
    return sorted(part.tolist(), key=lambda i: (-float(values[i]), i))


def pair_features(anchor, candidate, lal, e5, lal_rank, e5_rank, states, support_labels, frequency):
    a, b = states[anchor], states[candidate]
    labels = support_labels[candidate]
    freqs = [frequency[d] for d in labels]
    length_ratio = min(a["length"], b["length"]) / max(1, max(a["length"], b["length"]))
    return [
        lal, e5, max(lal, e5), min(lal, e5), (lal + e5) / 2, abs(lal - e5),
        1.0 / (32 + lal_rank), 1.0 / (32 + e5_rank),
        jaccard(a["tokens"], b["tokens"]), jaccard(a["bare"], b["bare"]),
        jaccard(a["numbers"], b["numbers"]), float(bool(a["numbers"] and a["numbers"] == b["numbers"])),
        float(a["prefix"] == b["prefix"]), length_ratio, math.log1p(b["length"]),
        len(labels), math.log1p(min(freqs) if freqs else 0), math.log1p(max(freqs) if freqs else 0),
    ]


FEATURE_NAMES = (
    "lal_cos", "e5_cos", "dense_max", "dense_min", "dense_mean", "dense_gap",
    "lal_recip_rank", "e5_recip_rank", "token_jaccard", "bare_token_jaccard",
    "number_jaccard", "same_nonempty_numbers", "same_question_prefix", "length_ratio",
    "candidate_log_length", "candidate_gold_cardinality", "candidate_min_label_log_frequency",
    "candidate_max_label_log_frequency",
)


def metrics(rankings, labels, qids):
    recall, precision, multi, mrr = [], [], [], []
    for q in qids:
        gold = labels.get(q, set())
        if not gold: continue
        top = rankings[q][:5]; hits = len(set(top) & gold); value = hits / len(gold)
        recall.append(value); precision.append(hits / 5)
        if len(gold) > 1: multi.append(value)
        first = next((i for i, d in enumerate(top, 1) if d in gold), None)
        mrr.append(0.0 if first is None else 1.0 / first)
    return {"recall@5": float(np.mean(recall)), "precision@5": float(np.mean(precision)),
            "multi_gold_recall@5": float(np.mean(multi)), "mrr@5": float(np.mean(mrr)), "queries": len(recall)}


def fuse(base, expert, weight, k=32):
    br = {d: i for i, d in enumerate(base, 1)}; er = {d: i for i, d in enumerate(expert, 1)}
    docs = set(br) | set(er)
    score = {d: ((1-weight)/(k+br[d]) if d in br else 0.0) + (weight/(k+er[d]) if d in er else 0.0) for d in docs}
    return sorted(docs, key=lambda d: (-score[d], d))


def run_outer(outer, ids, lal_vectors, e5_vectors, labels, folds, states, base):
    import lightgbm as lgb

    test = [str(q) for q in folds[f"fold_{outer}"] if labels.get(str(q))]
    support = [str(q) for f in range(5) if f != outer for q in folds[f"fold_{f}"] if labels.get(str(q))]
    row = {q: i for i, q in enumerate(ids)}
    si = [row[q] for q in support]; ti = [row[q] for q in test]
    frequency = Counter(d for q in support for d in labels[q])
    by_doc = defaultdict(list)
    for i, q in enumerate(support):
        for d in labels[q]: by_doc[d].append(i)
    support_labels = {q: labels[q] for q in support}
    support_lal = lal_vectors[si]; support_e5 = e5_vectors[si]
    lal_train = np.asarray(support_lal @ support_lal.T, dtype=np.float32)
    e5_train = np.asarray(support_e5 @ support_e5.T, dtype=np.float32)
    np.fill_diagonal(lal_train, -np.inf); np.fill_diagonal(e5_train, -np.inf)
    train_x, train_y, groups = [], [], []
    for ai, anchor in enumerate(support):
        ltop = top_indices(lal_train[ai], 48); etop = top_indices(e5_train[ai], 48)
        candidates = list(dict.fromkeys(ltop + etop))
        positive_indices = set(j for d in labels[anchor] for j in by_doc[d] if j != ai)
        # Inject the most semantically plausible positive links.  Without this,
        # low lexical similarity can make a train group contain no positives.
        injected = sorted(positive_indices, key=lambda j: (-max(float(lal_train[ai,j]), float(e5_train[ai,j])), support[j]))[:16]
        candidates = list(dict.fromkeys(candidates + injected))
        if not positive_indices:
            continue
        lr = {j: r for r, j in enumerate(ltop, 1)}; er = {j: r for r, j in enumerate(etop, 1)}
        for j in candidates:
            shared = labels[anchor] & labels[support[j]]
            rarity = max((1.0 / math.sqrt(frequency[d]) for d in shared), default=0.0)
            relevance = 0 if not shared else (3 if labels[anchor] == labels[support[j]] else 2 if rarity >= .5 else 1)
            train_x.append(pair_features(anchor, support[j], float(lal_train[ai,j]), float(e5_train[ai,j]), lr.get(j, 500), er.get(j, 500), states, support_labels, frequency))
            train_y.append(relevance)
        groups.append(len(candidates))
        if (ai + 1) % 1000 == 0: print(f"outer={outer} case_train_features={ai+1}/{len(support)}", flush=True)
    del lal_train, e5_train
    x = np.asarray(train_x, dtype=np.float32); y = np.asarray(train_y, dtype=np.int8)
    model = lgb.LGBMRanker(objective="lambdarank", num_leaves=15, min_child_samples=30, learning_rate=.05,
                          n_estimators=300, lambdarank_truncation_level=32, deterministic=True,
                          force_col_wise=True, n_jobs=4, random_state=2112, verbosity=-1)
    positive_rows = int(np.count_nonzero(y))
    model.fit(x, y, group=groups, eval_at=[16, 32]); del x, y, train_x, train_y

    lal_test = np.asarray(lal_vectors[ti] @ support_lal.T, dtype=np.float32)
    e5_test = np.asarray(e5_vectors[ti] @ support_e5.T, dtype=np.float32)
    case_rankings = {}; case_ceiling = []
    for qi, q in enumerate(test):
        ltop = top_indices(lal_test[qi], 64); etop = top_indices(e5_test[qi], 64)
        candidates = list(dict.fromkeys(ltop + etop)); lr = {j:r for r,j in enumerate(ltop,1)}; er = {j:r for r,j in enumerate(etop,1)}
        tx = np.asarray([pair_features(q, support[j], float(lal_test[qi,j]), float(e5_test[qi,j]), lr.get(j,500), er.get(j,500), states, support_labels, frequency) for j in candidates], dtype=np.float32)
        scores = model.predict(tx)
        neighbours = [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), support[candidates[i]]))]
        reachable = set().union(*(labels[support[j]] for j in neighbours)) if neighbours else set()
        current = set(base[q][:5]) & labels[q]
        case_ceiling.append(min(5, len(current | (reachable & labels[q]))) / len(labels[q]))
        votes = defaultdict(float)
        for rank, j in enumerate(neighbours[:64], 1):
            docs = labels[support[j]]
            for d in docs:
                votes[d] += 1.0 / ((32 + rank) * max(1, len(docs)) * math.sqrt(frequency[d]))
        case_rankings[q] = sorted(votes, key=lambda d: (-votes[d], d))
    result = {"outer": outer, "case_candidate_oracle_recall@5": float(np.mean(case_ceiling)), "case_rankings": case_rankings}
    write(OUT / f"fold_{outer}" / "CASE_RANKINGS.json", case_rankings)
    write(OUT / f"fold_{outer}" / "PAIR_MODEL.json", {"feature_names": FEATURE_NAMES, "training_groups": len(groups), "training_pairs": int(sum(groups)), "positive_rows": positive_rows})
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--outer", default="0", choices=["0","1","2","3","4","all"]); args = parser.parse_args()
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import exp109b_encoder_complementarity as old
    labels, _ = old.canonical_labels(); folds = read(ROOT / "cache/cv_folds.json"); train = read(ROOT / "public_test_dataset/train.json")
    base = {}
    for f in range(5): base.update(read(BASE / f"fold_{f}" / "PREDICTIONS.json"))
    e5_dir = ROOT / "cache/exp021_e5_dense_candidates/query_embeddings"
    ids = list(map(str, read(e5_dir / "train_query_ids.json"))); e5 = normalize(np.load(e5_dir / "train_queries.f32.npy", mmap_mode="r"))
    with np.load(ROOT / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz", allow_pickle=False) as z:
        l_ids = list(map(str, z["query_ids"].tolist())); lv = normalize(z["vectors"])
    lrow = {q:i for i,q in enumerate(l_ids)}; lal = lv[[lrow[q] for q in ids]]
    states = {q: text_state(train[q]["question"]) for q in ids}
    outers = range(5) if args.outer == "all" else [int(args.outer)]
    results = [run_outer(f, ids, lal, e5, labels, folds, states, base) for f in outers]
    all_case = {}
    for f in range(5):
        path = OUT / f"fold_{f}" / "CASE_RANKINGS.json"
        if path.exists(): all_case.update(read(path))
    evaluable = [q for q in base if labels.get(q) and q in all_case]
    if not evaluable:
        return
    baseline = metrics(base, labels, evaluable); trials = []
    for weight in (.01,.02,.03,.05,.075,.10,.15,.20):
        ranked = {q:fuse(base[q], all_case[q], weight) for q in evaluable}; score = metrics(ranked, labels, evaluable)
        trials.append({"weight": weight, "metrics": score, "delta": score["recall@5"]-baseline["recall@5"], "multi_delta": score["multi_gold_recall@5"]-baseline["multi_gold_recall@5"]})
    trials.sort(key=lambda r:(r["metrics"]["recall@5"],r["metrics"]["precision@5"],r["metrics"]["mrr@5"]),reverse=True)
    report = {"status":"COMPLETE_CASE_LINKER_PROBE" if len(results)==5 else "COMPLETE_CASE_LINKER_PARTIAL", "completed_folds":[r["outer"] for r in results], "scope_queries":len(evaluable), "baseline":baseline, "case_candidate_oracle_by_fold":{f"fold_{r['outer']}":r["case_candidate_oracle_recall@5"] for r in results}, "top_trials":trials[:10]}
    write(OUT / "CASE_LINKER_REPORT.json", report); print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
