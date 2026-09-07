import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import json
import re
from pathlib import Path
from collections import defaultdict
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
preds = json.load(open('results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json'))
labels, meta = get_canonical_labels()

eval_qids = [qid for qid, gold in labels.items() if gold]

def parse_doc(doc_id, lbl):
    years = re.findall(r'\b(19\d\d|20\d\d)\b', lbl)
    if not years:
        return None, None
    year = int(years[0])
    core = re.sub(r'\b(19\d\d|20\d\d)\b', '', lbl)
    core = re.sub(r'\b\d+\b', '', core)
    core = ' '.join(core.split())
    return core, year

core_to_docs = defaultdict(list)
for doc_id, lbl in doc_labels.items():
    core, year = parse_doc(doc_id, lbl)
    if core and len(core) > 8:
        core_to_docs[core].append((year, doc_id, lbl))

print(f'Total distinct cores: {len(core_to_docs)}')

multi_year_cores = {k: v for k, v in core_to_docs.items() if len({y for y, _, _ in v}) >= 2}
print(f'Cores with multiple years: {len(multi_year_cores)}')

redundant_pairs = []
for core, doc_list in multi_year_cores.items():
    doc_list.sort(key=lambda x: x[0])
    for i in range(len(doc_list)):
        for j in range(i+1, len(doc_list)):
            old_year, old_id, old_lbl = doc_list[i]
            new_year, new_id, new_lbl = doc_list[j]
            if old_year == new_year:
                continue
            
            both_top5 = 0
            old_gold = 0
            new_gold = 0
            for qid in eval_qids:
                top5 = preds[qid][:5]
                if old_id in top5 and new_id in top5:
                    both_top5 += 1
                    golds = labels[qid]
                    if old_id in golds:
                        old_gold += 1
                    if new_id in golds:
                        new_gold += 1
            if both_top5 > 0:
                redundant_pairs.append((both_top5, old_gold, new_gold, old_year, new_year, old_id, new_id, core, old_lbl, new_lbl))

redundant_pairs.sort(key=lambda x: x[0], reverse=True)
print(f'Total pairs appearing together in Top 5: {len(redundant_pairs)}')
for both_top5, old_gold, new_gold, old_year, new_year, old_id, new_id, core, old_lbl, new_lbl in redundant_pairs[:25]:
    print(f'\nCore: "{core}" ({both_top5} times both in Top 5)')
    print(f'  Old ({old_year}): {old_id} | Gold count: {old_gold} | {old_lbl}')
    print(f'  New ({new_year}): {new_id} | Gold count: {new_gold} | {new_lbl}')
