import json
import sqlite3
import re
from pathlib import Path
from collections import Counter

ROOT = Path('D:/Study/DSC2026/LegalIR')
import sys
sys.path.insert(0, str(ROOT / 'src'))

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics
from gemini.kinship import load_doc_labels

labels, _ = get_canonical_labels()
folds = get_cv_folds()
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')

all_qids = sorted([q for q in labels if labels[q]])

# 1. Single-gold vs Multi-gold
single_gold_qids = [q for q in all_qids if len(labels[q]) == 1]
multi_gold_qids = [q for q in all_qids if len(labels[q]) >= 2]

# 2. Statutory type of gold documents
has_law_qids = []
decree_only_qids = []
other_qids = []

for q in all_qids:
    g_types = []
    for d in labels[q]:
        lbl = doc_labels.get(d, '').lower()
        if lbl.startswith('luat') or lbl.startswith('bo luat'):
            g_types.append('law')
        elif lbl.startswith('nghi dinh') or lbl.startswith('thong tu'):
            g_types.append('decree_circular')
        else:
            g_types.append('other')
    if 'law' in g_types:
        has_law_qids.append(q)
    elif 'decree_circular' in g_types:
        decree_only_qids.append(q)
    else:
        other_qids.append(q)

# 3. Gold frequency in corpus
gold_counts = Counter(d for q in all_qids for d in labels[q])
frequent_gold_qids = [q for q in all_qids if any(gold_counts[d] >= 10 for d in labels[q])]
rare_gold_qids = [q for q in all_qids if all(gold_counts[d] <= 3 for d in labels[q])]

# 4. Temporal era of gold documents
year_regex = re.compile(r'\b(20\d{2}|19\d{2})\b')
modern_qids = []
legacy_qids = []
for q in all_qids:
    years = []
    for d in labels[q]:
        lbl = doc_labels.get(d, '')
        m = year_regex.search(lbl)
        if m:
            years.append(int(m.group(1)))
    if years:
        if max(years) >= 2018:
            modern_qids.append(q)
        else:
            legacy_qids.append(q)

slices = {
    'all_queries': all_qids,
    'single_gold': single_gold_qids,
    'multi_gold': multi_gold_qids,
    'has_primary_law': has_law_qids,
    'decree_circular_only': decree_only_qids,
    'other_statutes': other_qids,
    'frequent_golds_ge10': frequent_gold_qids,
    'rare_golds_le3': rare_gold_qids,
    'modern_golds_ge2018': modern_qids,
    'legacy_golds_lt2018': legacy_qids,
    'fold_0': [q for q in folds['fold_0'] if labels.get(q)],
    'fold_1': [q for q in folds['fold_1'] if labels.get(q)],
    'fold_2': [q for q in folds['fold_2'] if labels.get(q)],
    'fold_3': [q for q in folds['fold_3'] if labels.get(q)],
    'fold_4': [q for q in folds['fold_4'] if labels.get(q)],
}

# Evaluate Authoritative Baseline on all frozen slices
base_preds = json.loads((ROOT / 'results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json').read_text(encoding='utf-8'))
baseline_scorecard = {}
for s_name, qids in slices.items():
    m = compute_metrics(base_preds, labels, qids)
    baseline_scorecard[s_name] = {
        'count': len(qids),
        'recall_at_5': m['recall_at_5'],
        'precision_at_5': m['precision_at_5'],
        'multi_gold_recall_at_5': m['multi_gold_recall_at_5'],
        'mrr_at_5': m['mrr_at_5'],
    }

diagnostic_payload = {
    'status': 'FROZEN_ROBUSTNESS_DIAGNOSTIC_PROTOCOL',
    'created_at': '2026-09-07T16:47:00+07:00',
    'slice_counts': {k: len(v) for k, v in slices.items()},
    'baseline_scorecard': baseline_scorecard,
    'slices': slices,
}

out_path = ROOT / 'results/gemini/ROBUSTNESS_DIAGNOSTIC_SLICES.json'
out_path.write_text(json.dumps(diagnostic_payload, ensure_ascii=False, indent=2), encoding='utf-8')
print('Robustness diagnostic protocol frozen and written to', out_path)
print('\nBASELINE FROZEN SCORECARD:')
for s_name, sc in baseline_scorecard.items():
    cnt = sc['count']
    r5 = sc['recall_at_5']
    p5 = sc['precision_at_5']
    print(f'  {s_name:22s} (N={cnt:4d}): R@5={r5:.6f} | Prec@5={p5:.6f}')
