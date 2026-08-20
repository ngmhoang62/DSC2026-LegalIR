import json
import math
from itertools import product
from pathlib import Path
import sys
sys.path.append("d:/Study/DSC2026/LegalIR/src")

from exp012b_core import load_answers, read_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp013b_core import rank_ids
from exp013b_fusion import _ambiguity, _fuse

folds = load_folds(Path('cache/cv_folds.json'))
answers = load_answers(Path('public_test_dataset/train.json'))
qids = {str(v) for v in folds['fold_0']}

candidates = {str(r['qid']): {str(i['doc_id']): i for i in r['candidates']} for r in read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl')) if str(r['qid']) in qids}
lam = {str(r['qid']): r['candidates'] for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl')) if str(r['qid']) in qids}
qwen = {str(r['qid']): r['scores'] for r in read_jsonl(Path('cache/exp013b_cascade/qwen/fold_0/scores.jsonl')) if str(r['qid']) in qids}
gold = {qid: answers[qid] for qid in qids}

plain = {qid: [str(row['doc_id']) for row in lam[qid]] for qid in qids}
lambda_metrics = evaluate_rankings(plain, gold, ks=(5,))

grid = [dict(zip(("qwen", "lambda", "bge"), values)) for values in product((1., 2., 3., 4.), (.5, 1., 2.), (0., .5))]
best_config = {"qwen": 3.0, "lambda": 1.0, "bge": 0.5}
best_qwen_metrics = evaluate_rankings({qid: _fuse(lam[qid], qwen[qid], candidates[qid], best_config) for qid in qids}, gold, ks=(5,))

for config in grid:
    fused = {qid: _fuse(lam[qid], qwen[qid], candidates[qid], config) for qid in qids}
    m = evaluate_rankings(fused, gold, ks=(5,))
    if (m["recall@5"], m["precision@5"]) > (best_qwen_metrics["recall@5"], best_qwen_metrics["precision@5"]):
        best_qwen_metrics = m
        best_config = config

print("Best Config:", best_config)
print("Lambda metrics:", lambda_metrics)
print("Best fused metrics:", best_qwen_metrics)
print("Recall gain:", best_qwen_metrics["recall@5"] - lambda_metrics["recall@5"])
print("Precision delta:", best_qwen_metrics["precision@5"] - lambda_metrics["precision@5"])

# Test router
full = {qid: _fuse(lam[qid], qwen[qid], candidates[qid], best_config) for qid in qids}
full_gain = evaluate_rankings(full, gold, ks=(5,))["recall@5"] - evaluate_rankings(plain, gold, ks=(5,))["recall@5"]
ordered = sorted(qids, key=lambda qid: (-_ambiguity(lam[qid], candidates[qid]), qid))
selected = None
for fraction in (.30, .35):
    routed = set(ordered[:math.ceil(len(ordered) * fraction)])
    mixed = {qid: full[qid] if qid in routed else plain[qid] for qid in qids}
    gain = evaluate_rankings(mixed, gold, ks=(5,))["recall@5"] - evaluate_rankings(plain, gold, ks=(5,))["recall@5"]
    print(f"Fraction {fraction}: gain={gain:.6f}, full_gain={full_gain:.6f}, ratio={gain/full_gain:.2f}")
    if gain >= .8 * full_gain:
        selected = {"fraction": fraction, "fold0_gain": gain, "full_gain": full_gain}
        break
print("Router Selected:", selected)
