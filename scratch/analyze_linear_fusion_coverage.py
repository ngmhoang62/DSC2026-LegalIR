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

folds = load_folds(Path('cache/cv_folds.json'))
answers = load_answers(Path('public_test_dataset/train.json'))

lam_oof = {str(r['qid']): r['candidates'] for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl'))}
cand_train = {str(r['qid']): {str(i['doc_id']): i for i in r['candidates']} for r in read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl'))}

# Load all 5 folds of Qwen/BGE reranker scores
qwen_by_fold = {}
for f in range(5):
    p = Path(f'cache/exp013b_cascade/qwen/fold_{f}/scores.jsonl')
    if p.exists():
        for r in read_jsonl(p):
            qwen_by_fold[str(r['qid'])] = r['scores']

print(f"Total queries with Reranker scores: {len(qwen_by_fold)}/7000")

# Analyze LambdaMART base metrics
plain = {qid: [str(row['doc_id']) for row in lam_oof[qid]] for qid in lam_oof}
print("LambdaMART Full 5-Fold Base:", evaluate_rankings(plain, answers, ks=(5, 10, 20, 24)))

# Check what happens with Linear Score Normalization vs RRF on all queries
def zscore(vals):
    arr = np.array(vals, dtype=np.float32)
    std = arr.std()
    return (arr - arr.mean()) / (std if std > 1e-5 else 1.0)

for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    fused_linear = {}
    for qid in lam_oof:
        lam_docs = [str(r['doc_id']) for r in lam_oof[qid]]
        lam_scores = [float(r['lambda_score']) for r in lam_oof[qid]]
        lam_z = zscore(lam_scores)
        
        if qid in qwen_by_fold:
            q_map = {str(x['doc_id']): float(x['qwen_score']) for x in qwen_by_fold[qid]}
            q_vals = [q_map.get(d, -20.0) for d in lam_docs]
            q_z = zscore(q_vals)
            
            combined = {d: (1.0 - alpha) * l_z + alpha * qz for d, l_z, qz in zip(lam_docs, lam_z, q_z)}
            fused_linear[qid] = rank_ids(combined)
        else:
            fused_linear[qid] = lam_docs
            
    m = evaluate_rankings(fused_linear, answers, ks=(5,))
    print(f"Linear Fusion (alpha={alpha:.1f}): Recall@5={m['recall@5']:.5f}, Prec@5={m['precision@5']:.5f}")
