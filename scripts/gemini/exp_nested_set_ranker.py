"""Strict nested residual Set-Transformer over top-10 candidate slate in Gemini namespace.

Evaluates whether joint slate interaction and cross-candidate self-attention can
recover missing secondary golds and resolve Rank 5/6 boundary decisions.

Protocol:
- 5 outer folds.
- For each outer fold: 3 inner folds for fitting, 1 disjoint calibration fold for
  checkpoint (epoch) and fusion alpha selection, 1 held-out outer fold for blind test.
- Predictions locked before outer labels are read.
- Zero label leakage, strict reproducibility.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics, paired_bootstrap
from gemini.set_ranker import ResidualSetRanker, multi_positive_set_loss

OUT = ROOT / "results/gemini/exp_nested_set_ranker"
CACHE = ROOT / "cache/gemini/exp_nested_set_ranker"
EPOCHS = (5, 10, 15, 20)
ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
CALIBRATION = {0: 4, 1: 0, 2: 1, 3: 2, 4: 3}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
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


def selection_key(value: dict[str, Any], actions: int) -> tuple[float, float, float, float, int]:
    return (
        float(value["recall_at_5"]),
        float(value["precision_at_5"]),
        float(value.get("multi_gold_recall_at_5", 0.0)),
        float(value.get("mrr_at_5", 0.0)),
        -actions,
    )


def blend(systems: list[dict[str, list[str]]], weights: tuple[float, ...] = (0.3, 0.4, 0.3), k: int = 10, depth: int = 64) -> dict[str, list[str]]:
    result = {}
    first_sys = systems[0]
    for qid in first_sys:
        scores: dict[str, float] = {}
        for system, weight in zip(systems, weights):
            for rank, doc in enumerate(system[qid][:depth], 1):
                scores[doc] = scores.get(doc, 0.0) + weight / (k + rank)
        result[qid] = sorted(scores.keys(), key=lambda doc: (-scores[doc], doc))
    return result


def load_systems() -> dict[str, dict[str, list[str]]]:
    memory: dict[str, list[str]] = {}
    for fold in range(5):
        memory.update(read_json(ROOT / f"results/exp_final_retrieval/memory_ltr_probe/fold_{fold}/PREDICTIONS.json"))
    return {
        "xgb": read_json(ROOT / "results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json"),
        "lgb": read_json(ROOT / "results/gemini/exp_authority_131d/lgbm_131d_OOF_PREDICTIONS.json"),
        "profile": read_json(ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json"),
        "memory": memory,
        "kernel": read_json(ROOT / "results/exp_final_retrieval/kernel_ltr_probe/l15_t5/PREDICTIONS.json"),
    }


def load_query_vectors(qids: list[str]) -> np.ndarray:
    folder = ROOT / "cache/exp021_e5_dense_candidates/query_embeddings"
    ids = read_json(folder / "train_query_ids.json")
    vectors = np.load(folder / "train_queries.f32.npy", mmap_mode="r")
    id_to_idx = {str(q): i for i, q in enumerate(ids)}
    out = np.empty((len(qids), vectors.shape[1]), dtype=np.float32)
    for i, q in enumerate(qids):
        v = np.array(vectors[id_to_idx[q]], dtype=np.float32)
        norm = float(np.linalg.norm(v))
        out[i] = v / max(norm, 1e-12)
    return out


def materialize_features(
    qids: list[str],
    folds: dict[str, list[str]],
    base: dict[str, list[str]],
    systems: dict[str, dict[str, list[str]]],
) -> np.ndarray:
    path = CACHE / "top10_features_143d.f32.npy"
    manifest_path = CACHE / "manifest.json"
    inputs = [ROOT / f"cache/gemini/exp_authority_131d/fold_{fold}/test_131d.f32.npy" for fold in range(5)]
    contract = {
        "version": "gemini-131d-plus-six-oof-ranks-v1",
        "num_queries": len(qids),
        "input_hashes": [sha(p) for p in inputs],
    }
    if manifest_path.exists() and path.exists():
        manifest = read_json(manifest_path)
        if all(manifest.get(k) == contract[k] for k in contract) and manifest.get("sha256") == sha(path):
            print("Loaded cached 143D slate features from cache.", flush=True)
            return np.load(path, mmap_mode="r")

    CACHE.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.npy")
    values = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(len(qids), 10, 143))
    row = {q: i for i, q in enumerate(qids)}

    all_orders = {"base": base, **systems}

    for fold in range(5):
        fold_qids = [q for q in folds[f"fold_{fold}"] if q in row]
        docs = read_json(ROOT / f"cache/gemini/exp_unified_ltr/fold_{fold}/test_docs.json")
        groups = read_json(ROOT / f"cache/gemini/exp_unified_ltr/fold_{fold}/test_groups.json")
        matrix = np.load(inputs[fold], mmap_mode="r")
        ends = np.cumsum([0] + groups)
        if len(fold_qids) != len(docs):
            raise ValueError(f"Fold {fold} test-doc query count mismatch: {len(fold_qids)} vs {len(docs)}")

        for index, (qid, candidates) in enumerate(zip(fold_qids, docs)):
            mapping = {doc: position for position, doc in enumerate(candidates)}
            chosen = base[qid][:10]
            if any(doc not in mapping for doc in chosen):
                raise ValueError(f"Base top10 outside candidate pool for {qid}")
            base_rows = np.asarray(matrix[ends[index] : ends[index + 1]])[[mapping[doc] for doc in chosen]]
            extra = []
            for order in all_orders.values():
                ranks = {doc: rank for rank, doc in enumerate(order[qid], 1)}
                rank_values = np.asarray([ranks.get(doc, 1000) for doc in chosen], dtype=np.float32)
                extra.extend((rank_values, np.where(rank_values < 1000, 1.0 / (32.0 + rank_values), 0.0)))
            values[row[qid]] = np.concatenate([base_rows, np.asarray(extra, dtype=np.float32).T], axis=1)
        print(f"Materialized fold {fold} ({len(fold_qids)} queries)", flush=True)

    values.flush()
    del values
    tmp_path.replace(path)
    write_json(manifest_path, {**contract, "shape": [len(qids), 10, 143], "sha256": sha(path)})
    print(f"Successfully materialized and cached {path} (shape: [{len(qids)}, 10, 143])", flush=True)
    return np.load(path, mmap_mode="r")


def train_checkpoints(
    features: np.ndarray,
    queries: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    outer: int,
) -> tuple[ResidualSetRanker, dict[int, dict[str, Any]], np.ndarray, np.ndarray]:
    torch.manual_seed(112 + outer)
    np.random.seed(112 + outer)
    random.seed(112 + outer)

    train_feats = np.asarray(features[indices])
    mean = np.asarray(train_feats.mean(axis=(0, 1)), dtype=np.float32)
    std = np.asarray(train_feats.std(axis=(0, 1)), dtype=np.float32)
    std = np.where(std > 1e-6, std, 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualSetRanker(143, queries.shape[1], hidden=128, heads=4, layers=2, depth=10, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    states: dict[int, dict[str, Any]] = {}
    eligible = indices[targets[indices].sum(axis=1) > 0]

    for epoch in range(1, max(EPOCHS) + 1):
        model.train()
        generator = np.random.default_rng(112 + outer * 100 + epoch)
        order = generator.permutation(eligible)
        losses = []
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            x = torch.from_numpy((np.asarray(features[batch]) - mean) / std).to(device)
            q = torch.from_numpy(queries[batch]).to(device)
            y = torch.from_numpy(targets[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = multi_positive_set_loss(model(x, q), y, multi_weight=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        avg_loss = float(np.mean(losses))
        if epoch in EPOCHS:
            states[epoch] = copy.deepcopy(model.state_dict())
            print(f"  [Outer {outer}] Epoch {epoch:02d}: loss = {avg_loss:.6f} (saved checkpoint)", flush=True)

    return model, states, mean, std


@torch.inference_mode()
def predict_logits(
    model: ResidualSetRanker,
    state: dict[str, Any],
    features: np.ndarray,
    queries: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.load_state_dict(state)
    model.eval()
    result = []
    for start in range(0, len(indices), 256):
        batch = indices[start : start + 256]
        x = torch.from_numpy((np.asarray(features[batch]) - mean) / std).to(device)
        q = torch.from_numpy(queries[batch]).to(device)
        result.append(model(x, q).cpu().numpy())
    return np.concatenate(result, axis=0)


def fused_predictions(
    base: dict[str, list[str]],
    qids: list[str],
    logits: np.ndarray,
    alpha: float,
) -> dict[str, list[str]]:
    result = {}
    for qi, qid in enumerate(qids):
        docs = base[qid][:10]
        model_order = sorted(range(10), key=lambda idx: (-float(logits[qi, idx]), docs[idx]))
        model_rank = np.empty(10, dtype=np.int16)
        for rank, idx in enumerate(model_order, 1):
            model_rank[idx] = rank
        values = (1.0 - alpha) / (32.0 + np.arange(1, 11)) + alpha / (32.0 + model_rank)
        head = [docs[idx] for idx in sorted(range(10), key=lambda idx: (-float(values[idx]), docs[idx]))]
        result[qid] = head + base[qid][10:]
    return result


def main():
    start_time = time.time()
    print("=" * 80, flush=True)
    print("GEMINI NESTED SET-RANKER PROBE", flush=True)
    print("Strict 5-Fold Nested CV with Disjoint Calibration & Model Lock", flush=True)
    print("=" * 80, flush=True)

    labels, label_audit = get_canonical_labels()
    folds = get_cv_folds()
    systems = load_systems()
    base = blend([systems["xgb"], systems["lgb"], systems["profile"]])
    qids = [q for fold in range(5) for q in folds[f"fold_{fold}"] if labels.get(q)]
    row = {q: i for i, q in enumerate(qids)}

    print(f"Loaded {len(qids)} evaluable queries across 5 folds.", flush=True)
    base_metrics = compute_metrics(base, labels, qids)
    print(f"Base 131D Tri-Blend: Recall@5 = {base_metrics['recall_at_5']:.6f}, Prec@5 = {base_metrics['precision_at_5']:.6f}, Multi@5 = {base_metrics['multi_gold_recall_at_5']:.6f}, MRR@5 = {base_metrics['mrr_at_5']:.6f}", flush=True)

    features = materialize_features(qids, folds, base, systems)
    queries = load_query_vectors(qids)
    targets = np.asarray([[doc in labels[q] for doc in base[q][:10]] for q in qids], dtype=np.float32)

    report: dict[str, Any] = {
        "status": "COMPLETE_GEMINI_NESTED_SET_RANKER",
        "protocol": "Three folds fit; one disjoint calibration fold selects (epoch, alpha); prediction locked before reading outer test labels",
        "base_metrics": base_metrics,
        "folds": {},
    }
    oof: dict[str, list[str]] = {}

    for outer in range(5):
        cal_fold = CALIBRATION[outer]
        inner = [f for f in range(5) if f not in (outer, cal_fold)]
        train_qids = [q for f in inner for q in folds[f"fold_{f}"] if labels.get(q)]
        cal_qids = [q for q in folds[f"fold_{cal_fold}"] if labels.get(q)]
        outer_qids = [q for q in folds[f"fold_{outer}"] if labels.get(q)]

        ti = np.asarray([row[q] for q in train_qids])
        ci = np.asarray([row[q] for q in cal_qids])
        oi = np.asarray([row[q] for q in outer_qids])

        print(f"\n--- Outer Fold {outer} (Train: {inner}, Cal: {cal_fold}, Test: {outer}) ---", flush=True)
        model, states, mean, std = train_checkpoints(features, queries, targets, ti, outer)

        # Calibration selection
        trials = []
        for epoch, state in states.items():
            cal_logits = predict_logits(model, state, features, queries, ci, mean, std)
            for alpha in ALPHAS:
                pred = fused_predictions(base, cal_qids, cal_logits, alpha)
                val = compute_metrics(pred, labels, cal_qids)
                actions = sum(pred[q][:5] != base[q][:5] for q in cal_qids)
                trials.append({"epoch": epoch, "alpha": alpha, "metrics": val, "actions": actions})

        winner = max(trials, key=lambda item: selection_key(item["metrics"], item["actions"]))
        print(f"Calibration Winner: Epoch {winner['epoch']}, Alpha {winner['alpha']:.2f} (Cal R@5 = {winner['metrics']['recall_at_5']:.6f}, Actions = {winner['actions']})", flush=True)

        # Outer test prediction & LOCK before evaluating
        outer_logits = predict_logits(model, states[winner["epoch"]], features, queries, oi, mean, std)
        outer_pred = fused_predictions(base, outer_qids, outer_logits, winner["alpha"])

        fold_dir = OUT / f"fold_{outer}"
        write_json(fold_dir / "PREDICTIONS_LOCK.json", outer_pred)
        torch.save({"state": states[winner["epoch"]], "mean": mean, "std": std, "winner": winner}, fold_dir / "model.pt")

        lock = {
            "outer_fold": outer,
            "calibration_fold": cal_fold,
            "inner_folds": inner,
            "winner": winner,
            "prediction_sha256": sha(fold_dir / "PREDICTIONS_LOCK.json"),
            "model_sha256": sha(fold_dir / "model.pt"),
        }
        write_json(fold_dir / "SELECTION_LOCK.json", lock)

        # Evaluate outer fold
        outer_metrics = compute_metrics(outer_pred, labels, outer_qids)
        outer_base_metrics = compute_metrics(base, labels, outer_qids)
        delta_r5 = outer_metrics["recall_at_5"] - outer_base_metrics["recall_at_5"]

        report["folds"][f"fold_{outer}"] = {
            "lock": lock,
            "base": outer_base_metrics,
            "set_ranker": outer_metrics,
            "delta_recall_at_5": delta_r5,
            "top_cal_trials": sorted(trials, key=lambda item: selection_key(item["metrics"], item["actions"]), reverse=True)[:5],
        }
        oof.update(outer_pred)
        print(f"Outer Fold {outer} Blind Test: R@5 = {outer_metrics['recall_at_5']:.6f} (Base = {outer_base_metrics['recall_at_5']:.6f}, Delta = {delta_r5:+.6f})", flush=True)

        del model, states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(OUT / "OOF_PREDICTIONS.json", oof)
    oof_metrics = compute_metrics(oof, labels, qids)
    delta_oof = oof_metrics["recall_at_5"] - base_metrics["recall_at_5"]
    report["oof_metrics"] = oof_metrics
    report["delta_recall_at_5"] = delta_oof
    report["positive_folds"] = sum(v["delta_recall_at_5"] > 0 for v in report["folds"].values())
    report["nonnegative_folds"] = sum(v["delta_recall_at_5"] >= 0 for v in report["folds"].values())
    write_json(OUT / "NESTED_SET_RANKER_REPORT.json", report)

    # Paired bootstrap
    boot = paired_bootstrap(base, oof, labels, qids, k=5, n_boot=10000)
    report["paired_bootstrap"] = boot
    write_json(OUT / "NESTED_SET_RANKER_REPORT.json", report)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80, flush=True)
    print("FINAL 5-FOLD OOF EVALUATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Base 131D Tri-Blend Recall@5: {base_metrics['recall_at_5']:.6f}", flush=True)
    print(f"Set-Ranker OOF Recall@5:     {oof_metrics['recall_at_5']:.6f} (Delta: {delta_oof:+.6f})", flush=True)
    print(f"Set-Ranker OOF Precision@5:  {oof_metrics['precision_at_5']:.6f} (Delta: {oof_metrics['precision_at_5'] - base_metrics['precision_at_5']:+.6f})", flush=True)
    print(f"Set-Ranker OOF Multi@5:      {oof_metrics['multi_gold_recall_at_5']:.6f} (Delta: {oof_metrics['multi_gold_recall_at_5'] - base_metrics['multi_gold_recall_at_5']:+.6f})", flush=True)
    print(f"Non-negative Folds:          {report['nonnegative_folds']}/5", flush=True)
    print(f"Paired Bootstrap p-value:    {boot['p_value']:.4f}", flush=True)
    print(f"Total time elapsed:          {elapsed:.1f}s", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
