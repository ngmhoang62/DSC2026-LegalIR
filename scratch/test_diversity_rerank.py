import json
from pathlib import Path
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("d:/Study/DSC2026/LegalIR/src")

from exp012b_core import load_answers, read_jsonl
from exp012b_retrieval import evaluate_rankings
from exp013b_core import rank_ids

answers = load_answers(Path('public_test_dataset/train.json'))
oof_report = json.load(open('results/exp013b_cascade/oof/oof_report.json', 'r', encoding='utf-8'))
v3_dir = Path('cache/structural_v3')
doc_meta = {str(r['doc_id']): str(r.get('document_label', '')) for r in read_jsonl(v3_dir / 'documents.jsonl')}

preds = oof_report['predictions']
base_metrics = evaluate_rankings(preds, answers, ks=(5, 10, 20, 24))
print("Base OOF Metrics:", base_metrics)

def get_normalized_title(doc_id):
    label = doc_meta.get(doc_id, '')
    words = [w for w in label.lower().replace('-', ' ').split() if len(w) > 2 and not w.isdigit()]
    return set(words)

# Test Soft Redundancy Penalty (MMR-style on top 24)
for penalty in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
    diversified_preds = {}
    for qid in preds:
        ranked_docs = preds[qid][:24]
        selected = []
        candidates_pool = list(ranked_docs)
        
        # Initial score is -rank
        scores = {d: 24 - idx for idx, d in enumerate(candidates_pool)}
        
        while len(selected) < 5 and candidates_pool:
            if not selected:
                best_doc = candidates_pool.pop(0)
                selected.append(best_doc)
            else:
                best_doc = None
                best_score = -9999.0
                best_idx = -1
                
                for idx, d in enumerate(candidates_pool):
                    d_words = get_normalized_title(d)
                    
                    # Compute max jaccard overlap with already selected docs
                    max_sim = 0.0
                    for s in selected:
                        s_words = get_normalized_title(s)
                        if d_words and s_words:
                            sim = len(d_words & s_words) / len(d_words | s_words)
                            if sim > max_sim:
                                max_sim = sim
                                
                    # If two documents share >70% of words in title, apply penalty
                    redundancy_penalty = penalty * (max_sim if max_sim > 0.6 else 0.0)
                    adj_score = scores[d] - redundancy_penalty * 10.0
                    
                    if adj_score > best_score:
                        best_score = adj_score
                        best_doc = d
                        best_idx = idx
                        
                selected.append(best_doc)
                candidates_pool.pop(best_idx)
                
        remaining = [d for d in ranked_docs if d not in selected]
        diversified_preds[qid] = selected + remaining
        
    m = evaluate_rankings(diversified_preds, answers, ks=(5,))
    print(f"Redundancy Penalty {penalty:.1f}: Recall@5={m['recall@5']:.5f}, Prec@5={m['precision@5']:.5f}")
