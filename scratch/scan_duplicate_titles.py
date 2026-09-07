import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import json
import re
from collections import defaultdict
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels
from pathlib import Path

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
labels, meta = get_canonical_labels()
eval_qids = [qid for qid, gold in labels.items() if gold]
preds = json.load(open('results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json'))

# Normalize doc label: strip trailing IDs/numbers
def normalize_title(lbl):
    # remove trailing numbers / IDs
    lbl = re.sub(r'\b\d{5,}\b', '', lbl) # remove long doc IDs
    lbl = re.sub(r'\s+', ' ', lbl).strip()
    return lbl

norm_to_docs = defaultdict(set)
for d, lbl in doc_labels.items():
    nt = normalize_title(lbl)
    if len(nt) > 10:
        norm_to_docs[nt].add(d)

print(f"Total normalized titles with >= 2 docs: {sum(1 for v in norm_to_docs.values() if len(v) >= 2)}")

# Check co-occurrence in Top 5
dup_cooccur = []
for nt, doc_set in norm_to_docs.items():
    if len(doc_set) < 2:
        continue
    doc_list = list(doc_set)
    for i in range(len(doc_list)):
        for j in range(i+1, len(doc_list)):
            d1, d2 = doc_list[i], doc_list[j]
            co_cnt = 0
            both_gold = 0
            d1_gold = 0
            d2_gold = 0
            neither_gold = 0
            for qid in eval_qids:
                top5 = preds[qid][:5]
                if d1 in top5 and d2 in top5:
                    co_cnt += 1
                    golds = labels[qid]
                    is1 = d1 in golds
                    is2 = d2 in golds
                    if is1 and is2:
                        both_gold += 1
                    elif is1:
                        d1_gold += 1
                    elif is2:
                        d2_gold += 1
                    else:
                        neither_gold += 1
            if co_cnt > 0:
                dup_cooccur.append((co_cnt, d1, d2, nt, d1_gold, d2_gold, both_gold, neither_gold))

dup_cooccur.sort(key=lambda x: x[0], reverse=True)
print(f"Total duplicate title pairs appearing together in Top 5: {len(dup_cooccur)}")

for co_cnt, d1, d2, nt, d1_g, d2_g, both_g, neither_g in dup_cooccur:
    print(f"\n({co_cnt}x) Title: {nt[:60]}")
    print(f"    d1: {d1} [GOLD={d1_g}] | d2: {d2} [GOLD={d2_g}] | Both: {both_g} | Neither: {neither_g}")
