from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NS = "exp_final_retrieval"
CACHE = ROOT / "cache" / NS
RESULTS = ROOT / "results" / NS
SCHEMA = "exp_final.v1"
KS = (1, 3, 5, 10, 16, 20, 32, 50, 64, 100, 200)
V0 = dict(depth=1024, parent_rrf_k=32, fusion_rrf_k=32, head_cutoff=16)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def lock(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Immutable lock mismatch: {path}")
    else:
        write(path, value)
    return value


def records(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class DiskRows(Mapping):
    """Lazy query shards; do not retain full-corpus Python ranking dictionaries."""
    def __init__(self, directory, qids):
        self.directory=Path(directory);self.qids=tuple(qids);self.members=set(qids)
    def __getitem__(self,q):
        if q not in self.members:raise KeyError(q)
        return read(self.directory/f'{q}.json')
    def __iter__(self):return iter(self.qids)
    def __len__(self):return len(self.qids)


def splits(folds, outer):
    names = [f"fold_{i}" for i in range(5)]
    cal = names[(names.index(outer) - 1) % 5]
    train = [q for f in names if f not in (outer, cal) for q in folds[f]]
    valid, test = list(folds[cal]), list(folds[outer])
    if set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test):
        raise ValueError("Fold overlap")
    return train, valid, test


def seed_all(seed=112):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rank(scores, ids):
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite ranking scores")
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), str(ids[i])))]


def zscore(values):
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite normalization input")
    return (x - x.mean()) / x.std() if len(x) and x.std() > 1e-12 else np.zeros_like(x)


def union(sources, depth=100):
    return list(dict.fromkeys(d for source in sources for d in source[:depth]))


def rrf(rankings, weights=None):
    weights = weights or [1 / len(rankings)] * len(rankings)
    scores = {}
    for ranking, weight in zip(rankings, weights):
        for i, d in enumerate(ranking, 1):
            scores[d] = scores.get(d, 0.0) + weight / (32 + i)
    order = rank(list(scores.values()), list(scores))
    return order, [scores[d] for d in order]


def metrics(predictions, gold, *, exclude_empty=False, output_count=False, allow_empty_predictions=False):
    sums = {f"recall@{k}": [] for k in KS}
    precision, reciprocal, single, multi = [], [], [], []
    for q, g in gold.items():
        if q not in predictions:
            raise ValueError(f"Missing prediction {q}")
        g = set(g)
        if not g and exclude_empty:
            continue
        if not g:
            raise ValueError("Empty original gold")
        p = predictions[q]
        if len(p) != len(set(p)):
            raise ValueError("Duplicate prediction")
        for k in KS:
            sums[f"recall@{k}"].append(len(set(p[:k]) & g) / len(g))
        emitted = p if output_count else p[:5]
        if not emitted and allow_empty_predictions:
            precision.append(0.0)
            r = 0.0
            (single if len(g) == 1 else multi).append(r)
            reciprocal.append(0.0)
            continue
        if not 1 <= len(emitted) <= 5:
            raise ValueError("Invalid official prediction count")
        precision.append(len(set(emitted) & g) / len(emitted))
        r = len(set(emitted) & g) / len(g)
        (single if len(g) == 1 else multi).append(r)
        reciprocal.append(next((1 / i for i, d in enumerate(p[:5], 1) if d in g), 0.0))
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {**{k: mean(v) for k, v in sums.items()}, "precision@5": mean(precision), "mrr@5": mean(reciprocal),
            "single_gold_recall@5": mean(single), "multi_gold_recall@5": mean(multi), "queries": len(precision)}


def report_metrics(pred, original, canonical, *, allow_empty_predictions=False):
    return {
        "official": metrics(pred, original, allow_empty_predictions=allow_empty_predictions),
        "canonical": metrics(
            pred,
            canonical,
            exclude_empty=True,
            allow_empty_predictions=allow_empty_predictions,
        ),
    }


def metric_key(m):
    a, b = m["official"], m["canonical"]
    return tuple(round(float(v), 12) for v in (a["recall@5"], a["precision@5"], b["multi_gold_recall@5"], b["recall@5"], a["mrr@5"]))


def correct(order, scores, ce, alpha=0.0, threshold=None):
    if alpha == 0:
        return list(order), list(scores)
    n = min(50, len(order))
    a = zscore(scores[:n])
    confidence = float(a[4] - a[5]) if n > 5 else 0.0
    if threshold is not None and confidence > threshold:
        return list(order), list(scores)
    b = zscore([ce[d] for d in order[:n]])
    mixed = (1 - alpha) * a + alpha * b
    indices = sorted(range(n), key=lambda i: (-float(mixed[i]), i))
    return [order[i] for i in indices] + list(order[n:]), [float(mixed[i]) for i in indices] + list(scores[n:])


def prefix(order, scores, threshold=None):
    if threshold is None:
        return list(order[:5])
    z = zscore(scores[:50])
    n = 1
    while n < min(5, len(order)) and z[0] - z[n] <= threshold:
        n += 1
    return list(order[:n])


def select_threshold(rows, original):
    fixed = {q: r["order"][:5] for q, r in rows.items()}
    if metrics(fixed, original)["recall@5"] < .98:
        return None
    candidates = []
    for t in (6., 4., 3., 2.):
        p = {q: prefix(r["order"], r["scores"], t) for q, r in rows.items()}
        if all(set(p[q]) & set(original[q]) == set(fixed[q]) & set(original[q]) for q in p):
            if metrics(p, original, output_count=True)["precision@5"] > metrics(fixed, original)["precision@5"]:
                candidates.append((metrics(p, original, output_count=True)["precision@5"], t))
    return max(candidates)[1] if candidates else None


def deployment_recipe(locks):
    fields = ("epoch", "family", "block", "beta", "pool", "alpha", "confidence")
    signatures = [tuple(x.get(k) for k in fields) for x in locks]
    counts = Counter(signatures)
    costs = lambda x: (int(x.get("block", 0)), float(x.get("alpha", 0)), int(x.get("epoch", 0)), str(x.get("family", "")))
    i = min(range(len(locks)), key=lambda i: (-counts[signatures[i]], costs(locks[i]), i))
    return dict(locks[i])


def export_submission(directory, predictions, query_ids, doc_ids, *, member="submission.json", recall_first=None):
    directory = Path(directory)
    if set(predictions) != set(query_ids):
        raise ValueError("Public qid set mismatch")
    allowed = set(doc_ids)
    for p in predictions.values():
        if not 1 <= len(p) <= 5 or len(set(p)) != len(p) or not set(p) <= allowed:
            raise ValueError("Invalid submission IDs/count")
    payload = {q: {"answer": predictions[q]} for q in sorted(predictions)}
    recall_first = recall_first or predictions
    for p in recall_first.values():
        if not 1 <= len(p) <= 5 or len(set(p)) != len(p) or not set(p) <= allowed:
            raise ValueError('Invalid recall-first IDs/count')
    if set(recall_first) != set(predictions) or any(not set(predictions[q]) <= set(recall_first[q]) for q in predictions):
        raise ValueError("Adaptive output must be a subset of recall-first output")
    write(directory / "submission_recall_first.json", {q: {"answer": recall_first[q]} for q in sorted(recall_first)})
    write(directory / "submission.json", payload)
    with zipfile.ZipFile(directory / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(directory / "submission.json", member)
    with zipfile.ZipFile(directory / "submission.zip") as z:
        if z.namelist() != [member] or z.read(member) != (directory / "submission.json").read_bytes():
            raise ValueError("ZIP verification failed")
    result = {"status": "COMPLETE_SUBMISSION", "queries": len(payload), "uploaded": False,
              "files": {n: sha(directory / n) for n in ("submission.json", "submission_recall_first.json", "submission.zip")}}
    write(directory / "SUBMISSION_MANIFEST.json", result)
    return result


class Progress:
    def __init__(self, stage, total=0):
        import psutil
        self.process = psutil.Process()
        self.stage, self.total, self.started, self.last = stage, total, time.monotonic(), 0.
        self.path = RESULTS / "logs" / (stage.replace("/", "_") + ".log")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.update(0, force=True)

    def update(self, completed, state="RUNNING", force=False, **extra):
        import psutil
        now = time.monotonic()
        if not force and now - self.last < 30:
            return
        self.last = now
        elapsed = now - self.started
        value = dict(stage=self.stage, state=state, pid=os.getpid(), completed=completed, total=self.total,
                     elapsed_seconds=elapsed, eta_seconds=max(0,self.total-completed)*elapsed/completed if completed else None,
                     rss_bytes=self.process.memory_info().rss, available_ram_bytes=psutil.virtual_memory().available,
                     timestamp=time.time(), log_path=str(self.path), **extra)
        write(RESULTS / "RUN_STATUS.json", value)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")
            f.flush()
        print(json.dumps(value, ensure_ascii=False), flush=True)
