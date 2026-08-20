import json
from pathlib import Path
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("d:/Study/DSC2026/LegalIR/src")

from exp012b_core import load_answers, read_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp013b_core import rank_ids

answers = load_answers(Path('public_test_dataset/train.json'))
queries_data = json.load(open('public_test_dataset/train.json', 'r', encoding='utf-8'))
lam_oof = {str(r['qid']): r['candidates'] for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl'))}
cands_data = {str(r['qid']): {str(c['doc_id']): c for c in r['candidates']} for r in read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl'))}

# Check upper bounds with different K
print("--- SHORTLIST CANDIDATE UPPER BOUNDS ---")
for k in [5, 10, 15, 20, 24, 32, 40, 50]:
    oracle = {}
    for qid in lam_oof:
        docs = [str(c['doc_id']) for c in lam_oof[qid][:k]]
        g = answers[qid]
        # Oracle sort: gold first
        reordered = [d for d in docs if d in g] + [d for d in docs if d not in g]
        oracle[qid] = reordered
    m = evaluate_rankings(oracle, answers, ks=(5,))
    print(f"Top-{k:2d} Candidate Ceiling (Recall@5 Upper Bound): {m['recall@5']:.5f} ({m['recall@5']*100:.2f}%)")
