import json
from pathlib import Path
import sys
sys.path.append("d:/Study/DSC2026/LegalIR/src")

from exp012b_core import load_answers, read_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp013b_core import rank_ids

folds = load_folds(Path('cache/cv_folds.json'))
answers = load_answers(Path('public_test_dataset/train.json'))
qids = {str(v) for v in folds['fold_0']}

lam = {str(r['qid']): r['candidates'] for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl')) if str(r['qid']) in qids}
cand = {str(r['qid']): {str(i['doc_id']): i for i in r['candidates']} for r in read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl')) if str(r['qid']) in qids}
qwen = {str(r['qid']): r['scores'] for r in read_jsonl(Path('cache/exp013b_cascade/qwen/fold_0/scores.jsonl')) if str(r['qid']) in qids}
gold = {qid: answers[qid] for qid in qids}

# Check pure reranker ranking
pure_rerank = {}
for qid in qids:
    q_sc = {str(x['doc_id']): float(x['qwen_score']) for x in qwen[qid]}
    pure_rerank[qid] = rank_ids(q_sc)

m_pure = evaluate_rankings(pure_rerank, gold, ks=(1, 3, 5, 10, 24))
print("Pure BGE-Reranker on Top 24 Candidates:", m_pure)

# Check oracle ranking from Top 24
oracle_24 = {}
for qid in qids:
    docs = [str(x['doc_id']) for x in lam[qid][:24]]
    # Put gold docs first if present
    g = gold[qid]
    reordered = [d for d in docs if d in g] + [d for d in docs if d not in g]
    oracle_24[qid] = reordered

print("Oracle Top 24 upper bound Recall@5:", evaluate_rankings(oracle_24, gold, ks=(5,)))

# Analyze queries where LambdaMART won vs where Reranker won
lambda_wins = 0
reranker_wins = 0
both_hit = 0
both_miss = 0

for qid in qids:
    g = gold[qid]
    lam_top5 = set(str(x['doc_id']) for x in lam[qid][:5])
    rerank_top5 = set(pure_rerank[qid][:5])
    
    lam_hit = bool(lam_top5 & g)
    rerank_hit = bool(rerank_top5 & g)
    
    if lam_hit and not rerank_hit:
        lambda_wins += 1
    elif rerank_hit and not lam_hit:
        reranker_wins += 1
    elif lam_hit and rerank_hit:
        both_hit += 1
    else:
        both_miss += 1

print(f"Comparison: LambdaMART wins={lambda_wins}, Reranker wins={reranker_wins}, Both hit={both_hit}, Both miss={both_miss}")
