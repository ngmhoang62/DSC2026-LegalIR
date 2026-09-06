"""Strict outer-fold kernel-posterior specialist for EXP-FINAL.

This adapts the teammate repository's TF-IDF kernel posterior without copying
its small-block tuning protocol.  Each outer test fold is absent from both the
TF-IDF fit and label memory.  For ranker-training rows, the complete source
fold of the target query is removed from label memory, so neither self matches
nor same-fold labels can leak into the supervised specialist features.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CACHE = ROOT / "cache/exp112_task_adaptive_retrieval"
MEMORY = ROOT / "results/exp_final_retrieval/memory_ltr_probe"
OUT = ROOT / "results/exp_final_retrieval/kernel_ltr_probe"
NEIGHBOR_DEPTH = 400
POWER = 2.0
KERNEL_NAMES = (
    "kernel_word_seen", "kernel_word_top1", "kernel_word_top2",
    "kernel_word_top3", "kernel_word_robust", "kernel_word_norm",
    "kernel_word_recip_rank", "kernel_char_seen", "kernel_char_top1",
    "kernel_char_top2", "kernel_char_top3", "kernel_char_robust",
    "kernel_char_norm", "kernel_char_recip_rank", "kernel_mix_raw",
    "kernel_mix_recip_rank", "kernel_mix_rare_raw",
    "kernel_mix_rare_recip_rank", "kernel_mix_prior_raw",
    "kernel_mix_prior_recip_rank", "kernel_log_label_frequency",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def metrics(rankings, labels, qids):
    recall, precision, multi, mrr = [], [], [], []
    for qid in qids:
        gold = labels.get(qid, set())
        if not gold:
            continue
        top = rankings[qid][:5]
        hits = len(set(top) & gold)
        value = hits / len(gold)
        recall.append(value)
        precision.append(hits / 5)
        if len(gold) > 1:
            multi.append(value)
        first = next((rank for rank, doc in enumerate(top, 1) if doc in gold), None)
        mrr.append(0.0 if first is None else 1.0 / first)
    return {
        "recall_at_5": float(np.mean(recall)),
        "precision_at_5": float(np.mean(precision)),
        "multi_gold_recall_at_5": float(np.mean(multi)),
        "mrr_at_5": float(np.mean(mrr)),
        "queries": len(recall),
    }


def top_indices(row, allowed, depth=NEIGHBOR_DEPTH):
    row = row.tocoo()
    keep = allowed[row.col]
    cols, values = row.col[keep], row.data[keep]
    if len(values) > depth:
        part = np.argpartition(values, -depth)[-depth:]
        cols, values = cols[part], values[part]
    order = np.lexsort((cols, -values))
    return cols[order], values[order]


def channel_evidence(cols, values, support_qids, labels):
    per_doc = defaultdict(list)
    for col, value in zip(cols, values):
        transformed = float(value) ** POWER
        for doc in labels.get(support_qids[int(col)], ()):
            bucket = per_doc[doc]
            if len(bucket) < 3:
                bucket.append(transformed)
    result = {}
    for doc, evidence in per_doc.items():
        padded = evidence + [0.0] * (3 - len(evidence))
        result[doc] = (padded[0], padded[1], padded[2],
                       padded[0] + 0.40 * padded[1] + 0.15 * padded[2])
    return result


def rankmap(scores):
    order = sorted(scores, key=lambda doc: (-float(scores[doc]), doc))
    return order, {doc: rank for rank, doc in enumerate(order, 1)}


def kernel_features(word_ev, char_ev, docs, frequency):
    word_scores = {doc: values[3] for doc, values in word_ev.items()}
    char_scores = {doc: values[3] for doc, values in char_ev.items()}
    word_max = max(word_scores.values(), default=1.0)
    char_max = max(char_scores.values(), default=1.0)
    word_norm = {doc: score / max(word_max, 1e-12) for doc, score in word_scores.items()}
    char_norm = {doc: score / max(char_max, 1e-12) for doc, score in char_scores.items()}
    _, word_ranks = rankmap(word_scores)
    _, char_ranks = rankmap(char_scores)
    max_frequency = max(frequency.values(), default=1)
    mixed = {}
    for doc in set(word_scores) | set(char_scores):
        mixed[doc] = 0.5 * word_norm.get(doc, 0.0) + 0.5 * char_norm.get(doc, 0.0)
    variants = []
    for prior_power in (0.0, -0.35, 0.35):
        score = {
            doc: value * (((frequency.get(doc, 0) + 0.5) / (max_frequency + 0.5)) ** prior_power)
            for doc, value in mixed.items()
        }
        _, ranks = rankmap(score)
        variants.append((score, ranks))
    output = np.zeros((len(docs), len(KERNEL_NAMES)), dtype=np.float32)
    for row, doc in enumerate(docs):
        w = word_ev.get(doc, (0.0, 0.0, 0.0, 0.0))
        c = char_ev.get(doc, (0.0, 0.0, 0.0, 0.0))
        output[row, :7] = (
            float(doc in word_ev), w[0], w[1], w[2], w[3], word_norm.get(doc, 0.0),
            1.0 / (32 + word_ranks[doc]) if doc in word_ranks else 0.0,
        )
        output[row, 7:14] = (
            float(doc in char_ev), c[0], c[1], c[2], c[3], char_norm.get(doc, 0.0),
            1.0 / (32 + char_ranks[doc]) if doc in char_ranks else 0.0,
        )
        column = 14
        for scores, ranks in variants:
            output[row, column] = scores.get(doc, 0.0)
            output[row, column + 1] = 1.0 / (32 + ranks[doc]) if doc in ranks else 0.0
            column += 2
        output[row, 20] = math.log1p(frequency.get(doc, 0))
    if not np.isfinite(output).all():
        raise ValueError("Non-finite kernel feature")
    return output, variants[0][0]


def feature_batches(query_matrix, support_matrix, query_qids, support_qids, labels,
                    docs_by_query, allowed_by_query, frequency_by_query, batch_size=64):
    for begin in range(0, len(query_qids), batch_size):
        end = min(len(query_qids), begin + batch_size)
        similarity = (query_matrix[begin:end] @ support_matrix.T).tocsr()
        for local, qid in enumerate(query_qids[begin:end]):
            allowed = allowed_by_query(qid)
            wc, wv = top_indices(similarity.getrow(local), allowed)
            yield qid, wc, wv


def build_channel(vectorizer, train_text, test_text):
    support = vectorizer.fit_transform(train_text)
    test = vectorizer.transform(test_text)
    return support, test


def run():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import lightgbm as lgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    import exp109b_encoder_complementarity as old
    from exp_final.data import Data, SourceStore
    from exp_final.fusion import features
    from exp_final_memory_ltr_probe import memory_features, normalize, support_index

    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache/cv_folds.json")
    fold_of = {str(q): fold for fold in range(5) for q in folds[f"fold_{fold}"]}
    data = Data()
    store = SourceStore(LEGACY_CACHE / "sources.sqlite")
    store.jina_enabled = True
    with np.load(ROOT / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz", allow_pickle=False) as archive:
        vector_ids = list(map(str, archive["query_ids"].tolist()))
        lal_vectors = normalize(archive["vectors"])
    qrow = {qid: row for row, qid in enumerate(vector_ids)}
    configs = {
        "l7_t30": dict(num_leaves=7, min_child_samples=50, lambdarank_truncation_level=30),
        "l15_t5": dict(num_leaves=15, min_child_samples=50, lambdarank_truncation_level=5),
    }
    predictions = {name: {} for name in configs}
    baseline, kernel_direct, fold_reports = {}, {}, {}
    started = time.time()
    for outer in range(5):
        folder = OUT / f"fold_{outer}"
        folder.mkdir(parents=True, exist_ok=True)
        marker = read(LEGACY_CACHE / f"outer/fold_{outer}/outer-ml.json")
        train_qids = [q for q in map(str, marker["training_qids"]) if labels.get(q)]
        test_qids = [q for q in folds[f"fold_{outer}"] if labels.get(q)]
        train_text = [data.questions[q] for q in train_qids]
        test_text = [data.questions[q] for q in test_qids]
        word_support, word_test = build_channel(
            TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 3),
                            min_df=2, max_df=.995, max_features=180_000, sublinear_tf=True,
                            dtype=np.float32), train_text, test_text)
        char_support, char_test = build_channel(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                            max_features=160_000, sublinear_tf=True, dtype=np.float32),
            train_text, test_text)
        train_docs, groups, target = [], [], []
        for qid in train_qids:
            docs = list(dict.fromkeys(store.candidates(qid) + sorted(labels[qid])))
            train_docs.append(docs)
            groups.append(len(docs))
            target.extend(doc in labels[qid] for doc in docs)
        base = np.load(MEMORY / f"fold_{outer}/train_augmented.f32.npy", mmap_mode="r")
        if sum(groups) != len(base):
            raise ValueError(f"Outer {outer} base row mismatch")
        extra_path = folder / "train_kernel.f32.npy"
        extra = np.lib.format.open_memmap(extra_path, mode="w+", dtype=np.float32,
                                          shape=(len(base), len(KERNEL_NAMES)))
        offsets = np.cumsum([0] + groups)
        train_fold_indices = {
            fold: np.asarray([fold_of[q] != fold for q in train_qids], dtype=bool)
            for fold in sorted(set(fold_of[q] for q in train_qids))
        }
        train_frequency = {
            fold: Counter(doc for index, qid in enumerate(train_qids) if train_fold_indices[fold][index]
                          for doc in labels[qid])
            for fold in train_fold_indices
        }
        for begin in range(0, len(train_qids), 64):
            end = min(len(train_qids), begin + 64)
            word_sim = (word_support[begin:end] @ word_support.T).tocsr()
            char_sim = (char_support[begin:end] @ char_support.T).tocsr()
            for local, query_index in enumerate(range(begin, end)):
                qid = train_qids[query_index]
                held_fold = fold_of[qid]
                allowed = train_fold_indices[held_fold]
                wc, wv = top_indices(word_sim.getrow(local), allowed)
                cc, cv = top_indices(char_sim.getrow(local), allowed)
                word_ev = channel_evidence(wc, wv, train_qids, labels)
                char_ev = channel_evidence(cc, cv, train_qids, labels)
                values, _ = kernel_features(word_ev, char_ev, train_docs[query_index], train_frequency[held_fold])
                extra[offsets[query_index]:offsets[query_index + 1]] = values
            if end % 512 == 0 or end == len(train_qids):
                print(f"outer={outer} kernel_train={end}/{len(train_qids)} elapsed={time.time()-started:.1f}s", flush=True)
        extra.flush()
        combined_path = folder / "train_combined.f32.npy"
        combined = np.lib.format.open_memmap(combined_path, mode="w+", dtype=np.float32,
                                             shape=(len(base), base.shape[1] + len(KERNEL_NAMES)))
        combined[:, :base.shape[1]] = base
        combined[:, base.shape[1]:] = extra
        combined.flush()
        del extra
        by_doc, frequency = support_index(labels, train_qids)
        lal_support = lal_vectors[[qrow[q] for q in train_qids]]
        lal_test = np.asarray(lal_vectors[[qrow[q] for q in test_qids]] @ lal_support.T, dtype=np.float32)
        all_allowed = np.ones(len(train_qids), dtype=bool)
        test_rows, test_docs, test_groups = [], [], []
        local_kernel = {}
        for begin in range(0, len(test_qids), 64):
            end = min(len(test_qids), begin + 64)
            word_sim = (word_test[begin:end] @ word_support.T).tocsr()
            char_sim = (char_test[begin:end] @ char_support.T).tocsr()
            for local, test_index in enumerate(range(begin, end)):
                qid = test_qids[test_index]
                docs = store.candidates(qid)
                wc, wv = top_indices(word_sim.getrow(local), all_allowed)
                cc, cv = top_indices(char_sim.getrow(local), all_allowed)
                word_ev = channel_evidence(wc, wv, train_qids, labels)
                char_ev = channel_evidence(cc, cv, train_qids, labels)
                kernel, direct_scores = kernel_features(word_ev, char_ev, docs, frequency)
                xb = np.asarray(features(data, store, qid, docs, 2))
                xm = memory_features(lal_test[test_index], docs, train_qids, labels, by_doc, frequency)
                test_rows.append(np.concatenate([xb, xm, kernel], axis=1))
                test_docs.append(docs)
                test_groups.append(len(docs))
                direct = sorted(direct_scores, key=lambda doc: (-direct_scores[doc], doc))
                local_kernel[qid] = direct + [doc for doc in docs if doc not in direct_scores]
            print(f"outer={outer} kernel_test={end}/{len(test_qids)} elapsed={time.time()-started:.1f}s", flush=True)
        test_matrix = np.concatenate(test_rows)
        test_ends = np.cumsum([0] + test_groups)
        legacy = read(MEMORY / f"fold_{outer}/PREDICTIONS.json")
        baseline.update(legacy)
        kernel_direct.update(local_kernel)
        fold_reports[f"fold_{outer}"] = {}
        for name, config in configs.items():
            model = lgb.LGBMRanker(objective="lambdarank", learning_rate=.05, n_estimators=300,
                                   feature_fraction=1., bagging_fraction=1., deterministic=True,
                                   force_col_wise=True, n_jobs=4, random_state=5112, verbosity=-1,
                                   **config)
            model.fit(combined, np.asarray(target, dtype=np.int8), group=groups, eval_at=[5])
            scores = model.predict(test_matrix)
            local_predictions = {}
            for index, (qid, docs) in enumerate(zip(test_qids, test_docs)):
                values = scores[test_ends[index]:test_ends[index + 1]]
                local_predictions[qid] = [docs[row] for row in sorted(
                    range(len(docs)), key=lambda row: (-float(values[row]), docs[row]))]
            predictions[name].update(local_predictions)
            fold_reports[f"fold_{outer}"][name] = metrics(local_predictions, labels, test_qids)
            print(f"outer={outer} kernel_ltr={name} recall={fold_reports[f'fold_{outer}'][name]['recall_at_5']:.9f}", flush=True)
        write(folder / "FEATURE_MANIFEST.json", {
            "outer": outer, "feature_names": list(KERNEL_NAMES), "power": POWER,
            "neighbor_depth": NEIGHBOR_DEPTH, "tfidf_fit_qids": train_qids,
            "test_qids": test_qids, "train_feature_label_support": "exclude_target_query_fold",
            "test_feature_label_support": "all_outer_training_folds",
        })
        del combined, base, test_matrix, word_support, word_test, char_support, char_test
    qids = [qid for qid in baseline if labels.get(qid)]
    baseline_metrics = metrics(baseline, labels, qids)
    aggregate = []
    for name, ranked in predictions.items():
        result = metrics(ranked, labels, qids)
        fold_deltas = []
        for fold in range(5):
            ids = [q for q in folds[f"fold_{fold}"] if labels.get(q)]
            fold_deltas.append(fold_reports[f"fold_{fold}"][name]["recall_at_5"] - metrics(baseline, labels, ids)["recall_at_5"])
        aggregate.append({
            "system": name, "config": configs[name], "metrics": result,
            "delta": result["recall_at_5"] - baseline_metrics["recall_at_5"],
            "fold_deltas": fold_deltas, "nonnegative_folds": sum(value >= 0 for value in fold_deltas),
        })
        write(OUT / name / "PREDICTIONS.json", ranked)
    direct_metrics = metrics(kernel_direct, labels, qids)
    oracle = {}
    for qid in qids:
        gold = labels[qid]
        left = len(set(baseline[qid][:5]) & gold)
        right = len(set(kernel_direct[qid][:5]) & gold)
        oracle[qid] = kernel_direct[qid] if right > left else baseline[qid]
    aggregate.sort(key=lambda row: (row["metrics"]["recall_at_5"], row["metrics"]["precision_at_5"], row["metrics"]["mrr_at_5"]), reverse=True)
    report = {
        "status": "COMPLETE_STRICT_KERNEL_LTR_PROBE",
        "scope_warning": "Outer-isolated development OOF; architecture selection has seen these folds historically.",
        "baseline": baseline_metrics, "kernel_direct": direct_metrics,
        "choice_oracle": metrics(oracle, labels, qids), "folds": fold_reports,
        "aggregate": aggregate, "feature_names": list(KERNEL_NAMES),
        "neighbor_depth": NEIGHBOR_DEPTH, "power": POWER,
    }
    write(OUT / "KERNEL_LTR_REPORT.json", report)
    store.close()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()
