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
queries_data = json.load(open('public_test_dataset/train.json', 'r', encoding='utf-8'))

lam_oof = {str(r['qid']): r['candidates'] for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl'))}
cands_data = {str(r['qid']): {str(c['doc_id']): c for c in r['candidates']} for r in read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl'))}

# Load all 5 folds of Qwen/BGE reranker scores
qwen_by_fold = {}
for f in range(5):
    p = Path(f'cache/exp013b_cascade/qwen/fold_{f}/scores.jsonl')
    if p.exists():
        for r in read_jsonl(p):
            qwen_by_fold[str(r['qid'])] = {str(x['doc_id']): float(x['qwen_score']) for x in r['scores']}

def zscore(arr):
    a = np.array(arr, dtype=np.float32)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-5 else 1.0)

# Check document title exact n-gram match bonus
v3_dir = Path('cache/structural_v3')
docs_manifest = json.load(open(v3_dir / 'manifest.json', 'r', encoding='utf-8'))
doc_metadata = {str(r['doc_id']): str(r.get('document_label', '')) for r in read_jsonl(v3_dir / 'documents.jsonl')}

best_recall = 0
best_params = None

for w_lam in [1.0, 1.5, 2.0]:
    for w_qwen in [1.0, 2.0, 3.0, 4.0]:
        for w_exact_title in [0.0, 0.5, 1.0, 1.5]:
            for w_rrf in [0.0, 0.5, 1.0]:
                fused = {}
                for qid in lam_oof:
                    q_text = queries_data[qid]['question'].lower()
                    doc_list = [str(r['doc_id']) for r in lam_oof[qid]]
                    lam_scores = [float(r['lambda_score']) for r in lam_oof[qid]]
                    lam_z = zscore(lam_scores)
                    
                    scores_dict = {}
                    has_qwen = qid in qwen_by_fold
                    qwen_map = qwen_by_fold.get(qid, {})
                    q_vals = [qwen_map.get(d, -15.0) for d in doc_list]
                    q_z = zscore(q_vals) if has_qwen else np.zeros_like(lam_z)
                    
                    for idx, doc_id in enumerate(doc_list):
                        cand_info = cands_data[qid].get(doc_id, {})
                        bge_rk = cand_info.get('bge_rank', 999)
                        bm25_rk = cand_info.get('bm25_rank', 999)
                        
                        # Title match bonus
                        title = doc_metadata.get(doc_id, '').lower().replace('-', ' ')
                        title_words = [w for w in title.split() if len(w) > 2]
                        title_overlap = sum(1 for w in title_words if w in q_text) / max(len(title_words), 1)
                        
                        # RRF channel bonus
                        rrf = (1.0 / (60 + bge_rk)) + (1.0 / (60 + bm25_rk))
                        
                        s = w_lam * lam_z[idx] + (w_qwen * q_z[idx] if has_qwen else 0.0) + w_exact_title * title_overlap + w_rrf * rrf * 10.0
                        scores_dict[doc_id] = s
                        
                    fused[qid] = rank_ids(scores_dict)
                    
                m = evaluate_rankings(fused, answers, ks=(5,))
                if m['recall@5'] > best_recall:
                    best_recall = m['recall@5']
                    best_params = (w_lam, w_qwen, w_exact_title, w_rrf, m)

print(f"Best Ensemble Configuration:")
print(f"Params (w_lam, w_qwen, w_title, w_rrf): {best_params[:4]}")
print(f"Metrics: {best_params[4]}")
