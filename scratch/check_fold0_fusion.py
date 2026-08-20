import json
import numpy as np
import sys
sys.path.append("d:/Study/DSC2026/LegalIR/src")

from exp012b_core import load_answers, read_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp013b_core import rank_ids

from pathlib import Path
folds = load_folds(Path('cache/cv_folds.json'))
answers = load_answers(Path('public_test_dataset/train.json'))
qids = {str(v) for v in folds['fold_0']}

lam = {str(r['qid']): r['candidates'] for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl')) if str(r['qid']) in qids}
cand = {str(r['qid']): {str(i['doc_id']): i for i in r['candidates']} for r in read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl')) if str(r['qid']) in qids}
qwen = {str(r['qid']): r['scores'] for r in read_jsonl(Path('cache/exp013b_cascade/qwen/fold_0/scores.jsonl')) if str(r['qid']) in qids}
gold = {qid: answers[qid] for qid in qids}

plain = {qid: [str(row['doc_id']) for row in lam[qid]] for qid in qids}
base_metrics = evaluate_rankings(plain, gold, ks=(5, 10, 20, 24))
print("LambdaMART Base:", base_metrics)

best_recall = 0
best_cfg = None

for w_qwen in [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    for w_lam in [0.2, 0.5, 1.0, 2.0]:
        for w_bge in [0.0, 0.2, 0.5, 1.0]:
            fused = {}
            for qid in qids:
                q_sc = {str(x['doc_id']): float(x['qwen_score']) for x in qwen[qid]}
                q_rk = {doc: i for i, doc in enumerate(rank_ids(q_sc), 1)}
                res = {}
                for row in lam[qid]:
                    doc_id = str(row['doc_id'])
                    bge_rank = int(cand[qid][doc_id].get('bge_rank', 999))
                    res[doc_id] = w_lam / (60 + int(row['rank']))
                    if doc_id in q_rk: 
                        res[doc_id] += w_qwen / (60 + q_rk[doc_id])
                    if bge_rank < 999: 
                        res[doc_id] += w_bge / (60 + bge_rank)
                fused[qid] = rank_ids(res)
            m = evaluate_rankings(fused, gold, ks=(5,))
            if m['recall@5'] > best_recall:
                best_recall = m['recall@5']
                best_cfg = (w_qwen, w_lam, w_bge, m)

print(f"Best Configuration: w_qwen={best_cfg[0]}, w_lam={best_cfg[1]}, w_bge={best_cfg[2]} -> {best_cfg[3]}")
