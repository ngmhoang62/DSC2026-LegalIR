import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import json
from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics, paired_bootstrap
from gemini.kinship import load_doc_labels, apply_guarded_inverse_law
from pathlib import Path

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
labels, meta = get_canonical_labels()
folds = get_cv_folds()
eval_qids = [q for f in range(5) for q in folds[f'fold_{f}'] if labels.get(q)]

from scratch.eval_h57_superseded import (
    blended, old_to_new,
    apply_kinship_promotion, apply_multi_statute_promotion,
    apply_inverse_kinship_promotion, apply_deep_statutory_kinship,
    apply_topic_law_promotion, apply_preamble_citation_kinship,
    apply_hierarchical_midrank_inverse_kinship, apply_technical_standard_kinship,
    apply_superseded_dedup, doc_preambles, questions
)

# Test top_k=2 vs top_k=3 vs top_k=4 in apply_guarded_inverse_law
for tk in [2, 3, 4]:
    all_preds = {}
    fold_recalls = []
    tot_prom = 0
    for f in range(5):
        f_eval_qids = [q for q in folds[f'fold_{f}'] if labels.get(q)]
        f1, _ = apply_kinship_promotion(blended, doc_labels, f_eval_qids, top_k=2, cand_max=9)
        f2, _ = apply_multi_statute_promotion(f1, doc_labels, questions, f_eval_qids)
        f3, _ = apply_inverse_kinship_promotion(f2, doc_labels, f_eval_qids, top_k=2, cand_max=9)
        f4, _ = apply_deep_statutory_kinship(f3, doc_labels, f_eval_qids, top_k=2, cand_max=15)
        f5, p_cnt = apply_guarded_inverse_law(f4, doc_labels, f_eval_qids, top_k=tk, cand_max=12)
        tot_prom += p_cnt
        f6, _ = apply_topic_law_promotion(f5, doc_labels, questions, f_eval_qids, cand_max=6)
        f7, _ = apply_preamble_citation_kinship(f6, doc_labels, doc_preambles, f_eval_qids, top_k=2, cand_max=8)
        f8, _ = apply_hierarchical_midrank_inverse_kinship(f7, doc_labels, f_eval_qids, cand_max=12)
        f9, _ = apply_technical_standard_kinship(f8, doc_labels, f_eval_qids, cand_max=10)
        f_final, _ = apply_superseded_dedup(f9, f_eval_qids)
        all_preds.update(f_final)
        m = compute_metrics(f_final, labels, f_eval_qids)
        fold_recalls.append(m['recall_at_5'])
        
    m_all = compute_metrics(all_preds, labels, eval_qids)
    rec = m_all['recall_at_5']
    f_str = str([round(x, 6) for x in fold_recalls])
    print(f"top_k={tk}: Recall@5 = {rec:.6f} | Folds: {f_str} (promotions={tot_prom})")
