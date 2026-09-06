"""Strict Fold-4 pilot for dual document--case LAL query adaptation.

The existing strongest system is a fold-isolated LAL semantic case-memory LTR.
Vanilla query-to-document LAL adaptation improved standalone retrieval but added
only +0.119pp when fused with that incumbent on Fold 4.  This experiment asks a
different question: can the LAL query space itself be trained to cluster legal
questions that share a supplied parent label while retaining document retrieval?

The first screen intentionally uses a non-learned label-prototype source.  This
avoids training a downstream model on in-sample adapter features.  If it shows
useful strict OOF signal, a later nested cross-fit can train an LTR safely.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/exp_final_retrieval/dual_case_adapter_probe/fold_4"
OUT = ROOT / "results/exp_final_retrieval/dual_case_adapter_probe"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def metrics(rankings, labels, qids):
    recall, precision, single, multi, mrr = [], [], [], [], []
    for qid in qids:
        gold = labels.get(qid, set())
        if not gold:
            continue
        top = rankings[qid][:5]
        hits = len(set(top) & gold); value = hits / len(gold)
        recall.append(value); precision.append(hits / 5)
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


def weighted_rrf(rankings, weights, constant=32):
    if set(rankings) != set(weights):
        raise ValueError("Every ranked source needs exactly one weight")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("RRF weights must sum to one")
    scores = defaultdict(float)
    for name, order in rankings.items():
        for rank, doc in enumerate(order, 1):
            scores[doc] += weights[name] / (constant + rank)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


def bounded_screen_rrf(rankings, weights, depth=200, constant=32):
    """Exact top-five screen for this probe's bounded specialist weights.

    Every trial keeps at least 70% weight on the incumbent. Its fifth-ranked
    document alone scores at least ``.70/(32+5)``. A document below rank 200 in
    every source scores at most ``1/(32+201)``, so it cannot enter the top five.
    The union of each source's top 200 is therefore sufficient for this grid.
    """
    if weights.get("incumbent", weights.get("base", 0.0)) < .70 - 1e-12:
        raise ValueError("Bounded top-200 proof requires anchor weight >= 0.70")
    return weighted_rrf({name: order[:depth] for name, order in rankings.items()}, weights, constant)


def choice_oracle(left, right, labels, qids):
    chosen, counts = {}, {"left": 0, "right": 0, "tie": 0}
    for qid in qids:
        gold = labels[qid]
        a = len(set(left[qid][:5]) & gold); b = len(set(right[qid][:5]) & gold)
        if b > a:
            chosen[qid] = right[qid]; counts["right"] += 1
        else:
            chosen[qid] = left[qid]; counts["left" if a > b else "tie"] += 1
    return {"metrics": metrics(chosen, labels, qids), "query_choices": counts}


def prototype_orders(target_vectors, support_vectors, support_qids, labels, all_doc_ids):
    """Rank seen labels by max, top-two mean, and log-mean-exp similarity."""
    by_doc = defaultdict(list)
    for index, qid in enumerate(support_qids):
        for doc in labels[qid]:
            by_doc[doc].append(index)
    seen_docs = sorted(by_doc)
    support_positions = [np.asarray(by_doc[doc], dtype=np.int64) for doc in seen_docs]
    result = {"max": [], "top2": [], "logmeanexp": []}
    for vector in target_vectors:
        similarities = np.asarray(support_vectors @ vector, dtype=np.float32)
        scores = {name: np.empty(len(seen_docs), dtype=np.float32) for name in result}
        for row, positions in enumerate(support_positions):
            local = similarities[positions]
            take = min(2, len(local)); top = np.partition(local, len(local) - take)[-take:]
            maximum = float(top.max())
            scores["max"][row] = maximum
            scores["top2"][row] = float(top.mean())
            scaled = 20.0 * (local - maximum)
            scores["logmeanexp"][row] = maximum + float(np.log(np.exp(scaled).mean()) / 20.0)
        for name in result:
            order = sorted(range(len(seen_docs)), key=lambda i: (-float(scores[name][i]), seen_docs[i]))
            result[name].append([seen_docs[i] for i in order])
    return result


def load_full_rankings(folder, qids):
    rankings, drift = {}, []
    for index, qid in enumerate(qids):
        row = read(folder / f"{qid}.json")
        rankings[qid] = list(map(str, row["order"])); drift.append(1.0 - float(row["frozen_query_cosine"]))
        if (index + 1) % 300 == 0:
            print(f"load_rankings={index + 1}/{len(qids)}", flush=True)
    return rankings, {
        "mean": float(np.mean(drift)), "p95": float(np.quantile(drift, .95)),
        "max": float(np.max(drift)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-updates", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    import exp109b_encoder_complementarity as old
    from exp_final.data import Data, SourceStore
    from exp_final.learning import train_query_with_case, score_queries, encode_query_vectors

    labels, _ = old.canonical_labels(); folds = read(ROOT / "cache/cv_folds.json")
    train_qids = [q for fold in range(4) for q in folds[f"fold_{fold}"] if labels.get(q)]
    test_qids = [q for q in folds["fold_4"] if labels.get(q)]
    data = Data(); store = SourceStore(ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite")
    store.jina_enabled = True
    if args.benchmark_updates:
        value = train_query_with_case(
            data, store, train_qids, CACHE / f"benchmark-{args.benchmark_updates}",
            epochs=1, nominal_epochs=2, max_updates=args.benchmark_updates,
        )
        write(OUT / "TRAIN_BENCHMARK.json", value); store.close()
        print(json.dumps(value, ensure_ascii=False, indent=2), flush=True); return

    started = time.monotonic()
    training = train_query_with_case(
        data, store, train_qids, CACHE / "query", epochs=2, nominal_epochs=2,
    )
    incumbent = read(ROOT / "results/exp_final_retrieval/memory_ltr_probe/fold_4/PREDICTIONS.json")
    systems = {"incumbent_memory": metrics(incumbent, labels, test_qids)}
    trials, oracles, representation = [], {}, {}
    # Frozen prototype is an essential control: it isolates loss-induced gains
    # from gains obtainable by merely exposing label prototypes to RRF.
    frozen_support = normalize(np.stack([data.query_vector(q, "lal") for q in train_qids]))
    frozen_target = normalize(np.stack([data.query_vector(q, "lal") for q in test_qids]))
    frozen_proto = prototype_orders(frozen_target, frozen_support, train_qids, labels, data.doc_ids)

    for epoch in (1, 2):
        checkpoint = CACHE / "query" / f"epoch-{epoch}.pt"
        content_dir = CACHE / f"content-epoch-{epoch}"
        score_queries(data, test_qids, checkpoint, content_dir, batch_size=4, source="lal")
        content, drift = load_full_rankings(content_dir, test_qids)
        encode_query_vectors(data, train_qids, checkpoint, CACHE / f"vectors-epoch-{epoch}-train", batch_size=8)
        encode_query_vectors(data, test_qids, checkpoint, CACHE / f"vectors-epoch-{epoch}-test", batch_size=8)
        support = normalize(np.load(CACHE / f"vectors-epoch-{epoch}-train/vectors.f32.npy", mmap_mode="r"))
        target = normalize(np.load(CACHE / f"vectors-epoch-{epoch}-test/vectors.f32.npy", mmap_mode="r"))
        adapted_proto = prototype_orders(target, support, train_qids, labels, data.doc_ids)
        representation[f"epoch_{epoch}"] = {"query_drift": drift}
        systems[f"adapted_content_epoch{epoch}"] = metrics(content, labels, test_qids)
        for family, proto_modes in (("frozen", frozen_proto), ("adapted", adapted_proto)):
            for mode, orders in proto_modes.items():
                proto = {qid: orders[i] for i, qid in enumerate(test_qids)}
                key = f"{family}_prototype_{mode}_epoch{epoch}"
                # Frozen representation is identical across epochs; only emit
                # it once while retaining uniform trial code below.
                if family == "frozen" and epoch == 2:
                    continue
                systems[key] = metrics(proto, labels, test_qids)
                oracles[f"incumbent_vs_{key}"] = choice_oracle(incumbent, proto, labels, test_qids)
                for content_weight in (0.0, .05, .10, .15):
                    for prototype_weight in (.025, .05, .10, .15):
                        if content_weight + prototype_weight > .30:
                            continue
                        fused = {
                            qid: bounded_screen_rrf(
                                {"incumbent": incumbent[qid], "content": content[qid], "prototype": proto[qid]},
                                {"incumbent": 1 - content_weight - prototype_weight,
                                 "content": content_weight, "prototype": prototype_weight},
                            ) for qid in test_qids
                        }
                        measured = metrics(fused, labels, test_qids)
                        trials.append({
                            "epoch": epoch, "prototype_family": family, "prototype_mode": mode,
                            "content_weight": content_weight, "prototype_weight": prototype_weight,
                            "metrics": measured,
                            "delta_vs_incumbent": measured["recall_at_5"] - systems["incumbent_memory"]["recall_at_5"],
                        })
    trials.sort(key=lambda row: (
        row["metrics"]["recall_at_5"], row["metrics"]["precision_at_5"],
        row["metrics"]["multi_gold_recall_at_5"], row["metrics"]["mrr_at_5"],
    ), reverse=True)
    report = {
        "status": "COMPLETE_DUAL_CASE_ADAPTER_FOLD4_PROBE",
        "scope": "Fold 4 OOF; dual LAL tower trained on Folds 0-3; non-learned prototype screen; exposed development fold.",
        "training": training,
        "systems": systems,
        "representation": representation,
        "choice_oracles": oracles,
        "top_fusions": trials[:30],
        "runtime_seconds": time.monotonic() - started,
    }
    write(OUT / "DUAL_CASE_ADAPTER_REPORT.json", report); store.close()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
