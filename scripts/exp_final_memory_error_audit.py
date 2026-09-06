"""Audit residual errors after the strongest fold-isolated case-memory LTR."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results/exp_final_retrieval/memory_ltr_probe"
OUT = ROOT / "results/exp_final_retrieval/memory_error_audit"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import exp109b_encoder_complementarity as old
    from exp_final.data import SourceStore

    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache/cv_folds.json")
    train = read(ROOT / "public_test_dataset/train.json")
    predictions = {}
    for fold in range(5):
        predictions.update(read(PRED / f"fold_{fold}" / "PREDICTIONS.json"))
    with np.load(ROOT / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz", allow_pickle=False) as z:
        ids = list(map(str, z["query_ids"].tolist())); vectors = normalize(z["vectors"])
    qrow = {q: i for i, q in enumerate(ids)}
    store = SourceStore(ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite")

    totals = Counter(); fold_rows = {}; samples = []
    for fold in range(5):
        test = [str(q) for q in folds[f"fold_{fold}"] if labels.get(str(q))]
        support = [str(q) for f in range(5) if f != fold for q in folds[f"fold_{f}"] if labels.get(str(q))]
        by_doc = defaultdict(list)
        for q in support:
            for doc in labels[q]: by_doc[doc].append(q)
        similarity = np.asarray(vectors[[qrow[q] for q in test]] @ vectors[[qrow[q] for q in support]].T, dtype=np.float32)
        local = Counter()
        for qi, q in enumerate(test):
            candidates = store.candidates(q)
            rank = {doc: i for i, doc in enumerate(predictions[q], 1)}
            for doc in labels[q]:
                local["gold_assignments"] += 1
                hit = rank.get(doc, 10**9) <= 5
                local["retrieved"] += int(hit)
                if hit: continue
                local["missed"] += 1
                seen = bool(by_doc.get(doc))
                local["missed_seen_label"] += int(seen)
                local["missed_unseen_label"] += int(not seen)
                local["missed_in_content_union"] += int(doc in candidates)
                local["missed_outside_content_union"] += int(doc not in candidates)
                supporters = by_doc.get(doc, [])
                nearest = []
                if supporters:
                    support_pos = {sq: i for i, sq in enumerate(support)}
                    nearest = sorted(((float(similarity[qi, support_pos[sq]]), sq) for sq in supporters), reverse=True)[:3]
                    local["missed_seen_nearest_ge_080"] += int(nearest[0][0] >= .80)
                    local["missed_seen_nearest_ge_085"] += int(nearest[0][0] >= .85)
                    local["missed_seen_nearest_ge_090"] += int(nearest[0][0] >= .90)
                if len(samples) < 160:
                    samples.append({
                        "fold": fold, "qid": q, "question": train[q]["question"], "missed_gold": doc,
                        "final_rank": None if doc not in rank else rank[doc], "in_content_union": doc in candidates,
                        "support_count": len(supporters),
                        "nearest_same_label_queries": [
                            {"similarity": score, "qid": sq, "question": train[sq]["question"], "gold": sorted(labels[sq])}
                            for score, sq in nearest
                        ],
                        "top5": predictions[q][:5], "all_gold": sorted(labels[q]),
                    })
        fold_rows[f"fold_{fold}"] = dict(local)
        totals.update(local)
    store.close()
    report = {
        "status": "COMPLETE_MEMORY_ERROR_AUDIT",
        "totals": dict(totals),
        "seen_label_ceiling_recall": 1.0 - totals["missed_unseen_label"] / totals["gold_assignments"],
        "residual_miss_seen_fraction": totals["missed_seen_label"] / totals["missed"],
        "folds": fold_rows,
        "sample_count": len(samples),
    }
    write(OUT / "MEMORY_ERROR_AUDIT.json", report)
    write(OUT / "MEMORY_ERROR_SAMPLES.json", samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
