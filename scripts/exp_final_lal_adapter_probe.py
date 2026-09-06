"""Fold-4 pilot for task-adapting the strongest frozen dense source (LAL).

EXP-112 adapted E5 only.  Frozen LAL is the stronger content retriever and is
also the representation behind the strongest semantic case-memory system.
This probe preserves the LAL document bank and trains only LoRA query-side
parameters with the same full-corpus, multi-positive objective.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/exp_final_retrieval/lal_adapter_probe/fold_4"
OUT = ROOT / "results/exp_final_retrieval/lal_adapter_probe"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def metrics(rankings, labels, qids):
    recall, precision, single, multi, mrr = [], [], [], [], []
    for qid in qids:
        gold = labels.get(qid, set())
        if not gold:
            continue
        top = rankings[qid][:5]
        hits = len(set(top) & gold)
        value = hits / len(gold)
        recall.append(value)
        precision.append(hits / 5)
        (single if len(gold) == 1 else multi).append(value)
        first = next((rank for rank, doc in enumerate(top, 1) if doc in gold), None)
        mrr.append(0.0 if first is None else 1.0 / first)
    return {
        "recall_at_5": float(np.mean(recall)),
        "precision_at_5": float(np.mean(precision)),
        "single_gold_recall_at_5": float(np.mean(single)),
        "multi_gold_recall_at_5": float(np.mean(multi)),
        "mrr_at_5": float(np.mean(mrr)),
        "queries": len(recall),
    }


def rrf(left, right, weight, constant=32):
    left_rank = {doc: rank for rank, doc in enumerate(left, 1)}
    right_rank = {doc: rank for rank, doc in enumerate(right, 1)}
    docs = set(left_rank) | set(right_rank)
    scores = {
        doc: (1.0 - weight) / (constant + left_rank[doc]) if doc in left_rank else 0.0
        for doc in docs
    }
    for doc in docs:
        if doc in right_rank:
            scores[doc] += weight / (constant + right_rank[doc])
    return sorted(docs, key=lambda doc: (-scores[doc], doc))


def choice_oracle(left, right, labels, qids):
    chosen = {}
    counts = {"left": 0, "right": 0, "tie": 0}
    for qid in qids:
        gold = labels[qid]
        a = len(set(left[qid][:5]) & gold)
        b = len(set(right[qid][:5]) & gold)
        if b > a:
            chosen[qid] = right[qid]
            counts["right"] += 1
        else:
            chosen[qid] = left[qid]
            counts["left" if a > b else "tie"] += 1
    return {"metrics": metrics(chosen, labels, qids), "query_choices": counts}


def identity_preflight(data, qids):
    import torch
    from exp_final.learning import QueryEncoder

    sample = qids[:16]
    model = QueryEncoder(source="lal")
    model.eval()
    with torch.no_grad():
        actual = model([data.questions[q] for q in sample]).cpu().numpy()
    expected = np.stack([data.query_vector(q, "lal") for q in sample])
    cosines = np.sum(actual * expected, axis=1)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "queries": sample,
        "minimum_cosine": float(cosines.min()),
        "mean_cosine": float(cosines.mean()),
        "maximum_abs_vector_error": float(np.max(np.abs(actual - expected))),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "peak_vram": int(torch.cuda.max_memory_reserved()),
    }
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # Cached LAL queries were emitted by another transformers loading path, so
    # use a score-equivalence tolerance rather than bit identity.
    if result["minimum_cosine"] < 0.9999:
        raise ValueError(f"LAL identity-LoRA parity failed: {result}")
    write(OUT / "IDENTITY_PREFLIGHT.json", result)
    return result


def load_rankings(folder, qids):
    rows = {}
    drifts = []
    for index, qid in enumerate(qids):
        row = read(folder / f"{qid}.json")
        rows[qid] = list(map(str, row["order"]))
        drifts.append(1.0 - float(row["frozen_query_cosine"]))
        if (index + 1) % 300 == 0:
            print(f"load_rankings={index+1}/{len(qids)}", flush=True)
    return rows, {"mean": float(np.mean(drifts)), "p95": float(np.quantile(drifts, 0.95)), "max": float(np.max(drifts))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--benchmark-updates", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    import exp109b_encoder_complementarity as old
    from exp_final.data import Data, SourceStore
    from exp_final.learning import train_query, score_queries

    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache/cv_folds.json")
    train_qids = [q for fold in range(4) for q in folds[f"fold_{fold}"] if labels.get(q)]
    test_qids = [q for q in folds["fold_4"] if labels.get(q)]
    data = Data()
    store = SourceStore(ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite")
    store.jina_enabled = True
    identity = identity_preflight(data, test_qids)
    if args.preflight_only:
        store.close()
        print(json.dumps(identity, ensure_ascii=False, indent=2), flush=True)
        return

    if args.benchmark_updates:
        result = train_query(
            data, store, train_qids, CACHE / f"benchmark-{args.benchmark_updates}",
            epochs=1, nominal_epochs=2, microbatch=4, max_updates=args.benchmark_updates,
            positive_policy="all", learning_rate=5e-5, source="lal",
        )
        write(OUT / "TRAIN_BENCHMARK.json", result)
        store.close()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    started = time.monotonic()
    training = train_query(
        data, store, train_qids, CACHE / "query", epochs=2, nominal_epochs=2,
        microbatch=4, positive_policy="all", learning_rate=5e-5, source="lal",
    )
    memory = read(ROOT / "results/exp_final_retrieval/memory_ltr_probe/fold_4/PREDICTIONS.json")
    frozen = {q: [row["doc_id"] for row in store.get(q, "lal")] for q in test_qids}
    systems = {"frozen_lal": metrics(frozen, labels, test_qids), "memory": metrics(memory, labels, test_qids)}
    rankings = {}
    drift = {}
    trials = []
    oracles = {}
    for epoch in (1, 2):
        output = CACHE / f"test-query-{epoch}"
        score_queries(data, test_qids, CACHE / "query" / f"epoch-{epoch}.pt", output, batch_size=4, source="lal")
        ranked, drift[f"epoch_{epoch}"] = load_rankings(output, test_qids)
        rankings[epoch] = ranked
        systems[f"adapted_lal_epoch{epoch}"] = metrics(ranked, labels, test_qids)
        oracles[f"memory_vs_adapted_lal_epoch{epoch}"] = choice_oracle(memory, ranked, labels, test_qids)
        for weight in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
            fused = {q: rrf(memory[q], ranked[q], weight) for q in test_qids}
            measured = metrics(fused, labels, test_qids)
            trials.append({
                "epoch": epoch,
                "weight": weight,
                "metrics": measured,
                "delta_vs_memory": measured["recall_at_5"] - systems["memory"]["recall_at_5"],
            })
    trials.sort(key=lambda row: (row["metrics"]["recall_at_5"], row["metrics"]["precision_at_5"], row["metrics"]["mrr_at_5"]), reverse=True)
    report = {
        "status": "COMPLETE_LAL_ADAPTER_FOLD4_PROBE",
        "scope": "Fold 4 OOF; LAL query tower trained on Folds 0-3; exposed development fold.",
        "hypothesis": "Adapting the strongest dense source is more useful than adapting E5 alone.",
        "identity_preflight": identity,
        "training": training,
        "systems": systems,
        "query_drift": drift,
        "choice_oracles": oracles,
        "top_fusions": trials[:20],
        "runtime_seconds": time.monotonic() - started,
    }
    write(OUT / "LAL_ADAPTER_REPORT.json", report)
    store.close()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
