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

def core_title(lbl):
    # remove all digits
    t = re.sub(r'\d+', '', lbl)
    return ' '.join(t.split())

core_map = defaultdict(list)
for d, lbl in doc_labels.items():
    c = core_title(lbl)
    if len(c) > 10:
        core_map[c].append((d, lbl))

print(f"Cores with >= 2 docs: {sum(1 for v in core_map.values() if len(v) >= 2)}")

# Check co-occurrences in Top 5
co_pairs = []
for c, doc_list in core_map.items():
    if len(doc_list) < 2:
        continue
    for i in range(len(doc_list)):
        for j in range(i+1, len(doc_list)):
            d1, lbl1 = doc_list[i]
            d2, lbl2 = doc_list[j]
            co_cnt = 0
            both_g = 0
            d1_g = 0
            d2_g = 0
            neither_g = 0
            for qid in eval_qids:
                top5 = preds[qid][:5]
                if d1 in top5 and d2 in top5:
                    co_cnt += 1
                    golds = labels[qid]
                    is1 = d1 in golds
                    is2 = d2 in golds
                    if is1 and is2:
                        both_g += 1
                    elif is1:
                        d1_g += 1
                    elif is2:
                        d2_g += 1
                    else:
                        neither_g += 1
            if co_cnt > 0:
                co_pairs.append((co_cnt, d1, d2, c, lbl1, lbl2, d1_g, d2_g, both_g, neither_g))

co_pairs.sort(key=lambda x: x[0], reverse=True)
print(f"Total co-occurring core pairs in Top 5: {len(co_pairs)}")

for co_cnt, d1, d2, c, lbl1, lbl2, d1_g, d2_g, both_g, neither_g in co_pairs[:25]:
    print(f"\n({co_cnt}x) Core: '{c[:40]}'")
    print(f"    d1: {d1} [G={d1_g}] | {lbl1[:45]}")
    print(f"    d2: {d2} [G={d2_g}] | {lbl2[:45]}")
    print(f"    Both Gold: {both_g} | Neither Gold: {neither_g}")
