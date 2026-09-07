import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import json
import re
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels
from pathlib import Path

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
labels, meta = get_canonical_labels()
eval_qids = [qid for qid, gold in labels.items() if gold]

preds = json.load(open('results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json'))

queries = {}
with open('cache/exp012b_v3/rankings/train/query_rows.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        queries[str(item.get('query_id') or item.get('id') or item.get('qid'))] = item.get('query') or ''

# Let's extract statutory citations from Decrees in Top 3..4
# E.g. 'nghi dinh 61 2018 nd cp' -> citation pattern '61 2018'
def extract_statute_tokens(lbl):
    m = re.findall(r'\b(?:nghi dinh|luat|thong tu)\s+(\d+)\s+(20\d\d|19\d\d)\b', lbl)
    return [f'{num} {yr}' for num, yr in m]

wins = 0
losses = 0
ties = 0
tested = []

for qid in eval_qids:
    top = preds[qid]
    top5 = top[:5]
    d5 = top5[4]
    lbl5 = doc_labels.get(d5, '')
    
    # Authority Guard for Rank 5
    if lbl5.startswith('luat') or lbl5.startswith('bo luat') or 'sua doi' in lbl5 or 'bo sung' in lbl5:
        continue
        
    # Check documents at Rank 3 and 4
    mid_docs = top[2:4] # Rank 3 and 4
    mid_citations = []
    for d in mid_docs:
        lbl = doc_labels.get(d, '')
        mid_citations.extend(extract_statute_tokens(lbl))
        
    if not mid_citations:
        continue
        
    # Check candidate pool at Rank 6..8
    found_cand_idx = None
    for idx_offset, d_cand in enumerate(top[5:8]):
        cand_lbl = doc_labels.get(d_cand, '')
        # Must be an implementing circular or decree
        if not (cand_lbl.startswith('thong tu') or cand_lbl.startswith('nghi dinh')):
            continue
        if 'sua doi' in cand_lbl or 'bo sung' in cand_lbl:
            continue
        # Check if candidate guides the mid-rank document
        for cit in mid_citations:
            if re.search(rf'(?:huong dan|thi hanh|quy dinh chi tiet)[^.\n]{{0,40}}?{cit}', cand_lbl):
                found_cand_idx = 5 + idx_offset
                break
        if found_cand_idx is not None:
            break
            
    if found_cand_idx is not None:
        cand_doc = top[found_cand_idx]
        golds = labels[qid]
        is_d5_g = d5 in golds
        is_cand_g = cand_doc in golds
        
        tested.append((qid, d5, lbl5[:35], cand_doc, doc_labels.get(cand_doc, '')[:45], found_cand_idx+1, is_d5_g, is_cand_g, queries.get(qid, '')))
        if is_cand_g and not is_d5_g:
            wins += 1
        elif is_d5_g and not is_cand_g:
            losses += 1
        else:
            ties += 1

print(f"Testing Mid-Rank Forward Kinship (Rank 3..4 Decree/Law -> Rank 6..8 Implementing Circular):")
print(f"Total tested queries: {len(tested)}")
print(f"Wins: {wins} | Losses: {losses} | Ties: {ties} | Net: +{wins - losses}")

for qid, d5, lbl5, cand, cand_lbl, r, is_d5_g, is_cand_g, qtext in tested:
    outcome = "WIN" if (is_cand_g and not is_d5_g) else ("LOSS" if (is_d5_g and not is_cand_g) else "TIE")
    print(f"[{outcome}] QID {qid} (Rank {r} -> 5): D5={d5} ({lbl5}) [GOLD={is_d5_g}]")
    print(f"     Cand: {cand} ({cand_lbl}) [GOLD={is_cand_g}]")
    print(f"     Query: {qtext[:80]}")
