"""Bounded Fold-4 probe for trainable label prototypes.

This is a hypothesis test, not a five-fold result.  Hyperparameters and the RRF
weight are selected on a deterministic holdout inside F0--F3 using frozen LAL
vectors.  The selected recipe is then refit on all F0--F3 and evaluated once on
Fold 4 with both frozen-LAL and the already trained dual-case representation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp_final.xmc import (  # noqa: E402
    TrainablePrototypeXMC,
    XMCTrainConfig,
    initial_label_prototypes,
    l2_normalize,
    score_topk,
    train_epoch,
)

OUT = ROOT / "results" / "exp_final_retrieval" / "trainable_xmc_probe"
DUAL = ROOT / "cache" / "exp_final_retrieval" / "dual_case_adapter_probe" / "fold_4"
BASE_PATH = ROOT / "results" / "exp_final_retrieval" / "profile_ltr_probe" / "l15_t5" / "PREDICTIONS.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def metrics(rankings, labels, qids):
    recall, precision, single, multi, reciprocal = [], [], [], [], []
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
        reciprocal.append(next((1 / rank for rank, doc in enumerate(top, 1) if doc in gold), 0.0))
    return {
        "recall_at_5": float(np.mean(recall)),
        "precision_at_5": float(np.mean(precision)),
        "single_gold_recall_at_5": float(np.mean(single)),
        "multi_gold_recall_at_5": float(np.mean(multi)),
        "mrr_at_5": float(np.mean(reciprocal)),
        "queries": len(recall),
    }


def metric_key(value):
    return (
        value["recall_at_5"], value["precision_at_5"],
        value["multi_gold_recall_at_5"], value["mrr_at_5"],
    )


def fuse(base, expert, weight, constant):
    base_rank = {doc: rank for rank, doc in enumerate(base, 1)}
    expert_rank = {doc: rank for rank, doc in enumerate(expert, 1)}
    scores = {}
    for doc, rank in base_rank.items():
        scores[doc] = (1.0 - weight) / (constant + rank)
    for doc, rank in expert_rank.items():
        scores[doc] = scores.get(doc, 0.0) + weight / (constant + rank)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


def rankings_from_topk(values, indices, qids, label_ids):
    rankings = {}
    for qid, row_values, row_indices in zip(qids, values, indices):
        pairs = [(str(label_ids[int(i)]), float(v)) for v, i in zip(row_values, row_indices)]
        pairs.sort(key=lambda item: (-item[1], item[0]))
        rankings[qid] = [doc for doc, _ in pairs]
    return rankings


def prototype_oracle(left, right, labels, qids):
    chosen, choices = {}, {"left": 0, "right": 0, "tie": 0}
    for qid in qids:
        gold = labels[qid]
        lhits = len(set(left[qid][:5]) & gold)
        rhits = len(set(right[qid][:5]) & gold)
        side = "right" if rhits > lhits else "left" if lhits > rhits else "tie"
        choices[side] += 1
        chosen[qid] = right[qid] if side == "right" else left[qid]
    return {"metrics": metrics(chosen, labels, qids), "query_choices": choices}


def hash_bucket(qid, modulo=5):
    return int(hashlib.sha256(str(qid).encode()).hexdigest()[:16], 16) % modulo


def frozen_lal_vectors(qids):
    path = ROOT / "cache" / "exp109b_encoder_complementarity" / "embeddings" / "vnlegal_lal" / "queries.npz"
    with np.load(path, allow_pickle=False) as archive:
        ids = list(map(str, archive["query_ids"].tolist()))
        vectors = np.asarray(archive["vectors"], dtype=np.float32)
    row = {qid: index for index, qid in enumerate(ids)}
    return l2_normalize(vectors[[row[qid] for qid in qids]])


def cached_vectors(folder, qids):
    root = DUAL / folder
    ids = list(map(str, read(root / "query_ids.json")))
    vectors = np.load(root / "vectors.f32.npy", mmap_mode="r")
    row = {qid: index for index, qid in enumerate(ids)}
    return l2_normalize(np.asarray(vectors[[row[qid] for qid in qids]], dtype=np.float32))


def label_problem(vectors, qids, labels):
    label_ids = sorted(set().union(*(labels[qid] for qid in qids)))
    label_row = {doc: row for row, doc in enumerate(label_ids)}
    positive_rows = [[label_row[doc] for doc in sorted(labels[qid])] for qid in qids]
    prototypes = initial_label_prototypes(vectors, positive_rows, len(label_ids))
    return label_ids, positive_rows, prototypes


def new_model(prototypes, config, device):
    torch.manual_seed(config.seed)
    model = TrainablePrototypeXMC(prototypes, query_rank=config.query_rank).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    return model, optimizer


def evaluate_expert(model, vectors, qids, label_ids, device):
    values, indices = score_topk(model, vectors, topk=200, device=device)
    return rankings_from_topk(values, indices, qids, label_ids)


def select_recipe(train_qids, val_qids, labels, base, device):
    all_qids = train_qids + val_qids
    all_vectors = frozen_lal_vectors(all_qids)
    train_vectors = all_vectors[:len(train_qids)]
    val_vectors = all_vectors[len(train_qids):]
    label_ids, positive_rows, prototypes = label_problem(train_vectors, train_qids, labels)
    trials, best = [], None
    for query_rank in (0, 32):
        config = XMCTrainConfig(query_rank=query_rank)
        model, optimizer = new_model(prototypes, config, device)
        for epoch in range(1, 9):
            losses = train_epoch(
                model, optimizer, train_vectors, positive_rows, config, epoch - 1, device=device,
            )
            expert = evaluate_expert(model, val_vectors, val_qids, label_ids, device)
            standalone = metrics(expert, labels, val_qids)
            for constant in (0, 32):
                for weight in (0.02, 0.05, 0.10, 0.15, 0.20):
                    fused = {q:fuse(base[q], expert[q], weight, constant) for q in val_qids}
                    value = metrics(fused, labels, val_qids)
                    trial = {
                        "query_rank": query_rank, "epoch": epoch, "constant": constant,
                        "weight": weight, "losses": losses, "standalone": standalone,
                        "metrics": value,
                    }
                    trials.append(trial)
                    key = (*metric_key(value), -query_rank, -epoch, -constant, -weight)
                    if best is None or key > best[0]:
                        best = (key, copy.deepcopy(trial))
            print(
                f"[xmc-select] rank={query_rank} epoch={epoch}/8 "
                f"expert_R5={standalone['recall_at_5']:.6f} loss={losses['loss']:.4f}",
                flush=True,
            )
        del model, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()
    trials.sort(key=lambda row:(*metric_key(row["metrics"]), -row["query_rank"], -row["epoch"]), reverse=True)
    return best[1], trials[:30]


def refit_and_evaluate(name, train_vectors, test_vectors, train_qids, test_qids,
                       labels, base, recipe, device):
    label_ids, positives, prototypes = label_problem(train_vectors, train_qids, labels)
    config = XMCTrainConfig(query_rank=int(recipe["query_rank"]))
    model, optimizer = new_model(prototypes, config, device)
    history = []
    for epoch in range(1, int(recipe["epoch"]) + 1):
        history.append(train_epoch(model, optimizer, train_vectors, positives, config, epoch - 1, device=device))
        print(f"[xmc-refit] {name} epoch={epoch}/{recipe['epoch']} loss={history[-1]['loss']:.4f}", flush=True)
    expert = evaluate_expert(model, test_vectors, test_qids, label_ids, device)
    fused = {
        q:fuse(base[q], expert[q], float(recipe["weight"]), int(recipe["constant"]))
        for q in test_qids
    }
    diagnostic_trials = []
    for constant in (0, 32):
        for weight in (0.02, 0.05, 0.10, 0.15, 0.20):
            ranked = {q:fuse(base[q], expert[q], weight, constant) for q in test_qids}
            value = metrics(ranked, labels, test_qids)
            diagnostic_trials.append({"constant":constant, "weight":weight, "metrics":value})
    diagnostic_trials.sort(key=lambda row:metric_key(row["metrics"]), reverse=True)
    base_metrics = metrics(base, labels, test_qids)
    quarter = {}
    for bucket in range(4):
        qids = [q for q in test_qids if hash_bucket(q, 4) == bucket]
        bm, fm = metrics(base, labels, qids), metrics(fused, labels, qids)
        quarter[str(bucket)] = {
            "queries": len(qids), "base": bm, "fused": fm,
            "delta": fm["recall_at_5"] - bm["recall_at_5"],
        }
    result = {
        "label_count": len(label_ids), "history": history,
        "expert": metrics(expert, labels, test_qids),
        "locked_fusion": metrics(fused, labels, test_qids),
        "locked_delta": metrics(fused, labels, test_qids)["recall_at_5"] - base_metrics["recall_at_5"],
        "choice_oracle": prototype_oracle(base, expert, labels, test_qids),
        "quarters": quarter,
        "posthoc_diagnostic_top5": diagnostic_trials[:5],
    }
    write(OUT / f"{name}_expert_top200.json", expert)
    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    import exp109b_encoder_complementarity as legacy

    started = time.monotonic()
    labels, _ = legacy.canonical_labels()
    folds = read(ROOT / "cache" / "cv_folds.json")
    base = {str(q):list(map(str, ranking)) for q, ranking in read(BASE_PATH).items()}
    outer_train = [q for fold in range(4) for q in folds[f"fold_{fold}"] if labels.get(q)]
    test_qids = [q for q in folds["fold_4"] if labels.get(q)]
    internal_val = [q for q in outer_train if hash_bucket(q, 5) == 0]
    internal_train = [q for q in outer_train if q not in set(internal_val)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[xmc] device={device} inner_train={len(internal_train)} "
        f"inner_val={len(internal_val)} fold4={len(test_qids)}",
        flush=True,
    )
    base_val = metrics(base, labels, internal_val)
    recipe, selection_trials = select_recipe(internal_train, internal_val, labels, base, device)
    selection_lock = {
        "scope": "F0-F3 deterministic hash holdout; frozen LAL only",
        "base": base_val, "recipe": {k:recipe[k] for k in ("query_rank", "epoch", "constant", "weight")},
        "selected_metrics": recipe["metrics"], "top_trials": selection_trials,
    }
    write(OUT / "SELECTION_LOCK.json", selection_lock)
    print(f"[xmc] locked={selection_lock['recipe']} val_R5={recipe['metrics']['recall_at_5']:.6f}", flush=True)

    frozen_train = frozen_lal_vectors(outer_train)
    frozen_test = frozen_lal_vectors(test_qids)
    adapted_train = cached_vectors("vectors-epoch-2-train", outer_train)
    adapted_test = cached_vectors("vectors-epoch-2-test", test_qids)
    systems = {
        "frozen_lal": refit_and_evaluate(
            "frozen_lal", frozen_train, frozen_test, outer_train, test_qids,
            labels, base, recipe, device,
        ),
        "dual_adapted_epoch2": refit_and_evaluate(
            "dual_adapted_epoch2", adapted_train, adapted_test, outer_train, test_qids,
            labels, base, recipe, device,
        ),
    }
    report = {
        "status": "COMPLETE_TRAINABLE_XMC_FOLD4_DEVELOPMENT_PROBE",
        "scope": "Strict Fold4 labels withheld from XMC fitting; exposed development Fold4, not OOF.",
        "hypothesis": "Trainable label prototypes can realize repeated-label recall that fixed centroids/RRF leave unused.",
        "base": metrics(base, labels, test_qids),
        "selection_lock": selection_lock,
        "systems": systems,
        "runtime_seconds": time.monotonic() - started,
    }
    write(OUT / "TRAINABLE_XMC_REPORT.json", report)
    print(json.dumps({
        "status": report["status"], "base": report["base"],
        "recipe": selection_lock["recipe"],
        "systems": {
            name:{k:value[k] for k in ("expert", "locked_fusion", "locked_delta", "choice_oracle", "quarters")}
            for name,value in systems.items()
        },
        "runtime_seconds": report["runtime_seconds"],
        "artifact": str(OUT / "TRAINABLE_XMC_REPORT.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
