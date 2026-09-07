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

queries = {}
with open('cache/exp012b_v3/rankings/train/query_rows.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        queries[str(item.get('query_id') or item.get('id') or item.get('qid'))] = item.get('query') or ''

from scratch.eval_h57_superseded import p_post

target_doc = '166505' # NĐ 204/2004
target_lbl = doc_labels.get(target_doc, '')

SALARY_TERMS = {'he so luong', 'bang luong', 'ngach', 'bac luong', 'chuc danh nghe nghiep', 'xep luong'}

wins = 0
losses = 0
ties = 0
tested_qids = []

import unicodedata

def strip_accents(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    return text.replace('đ', 'd').replace('Đ', 'D').lower()

for qid in eval_qids:
    top = p_post[qid]
    top5 = top[:5]
    if target_doc in top5:
        continue
    if target_doc not in top[5:10]:
        continue
        
    qtext = strip_accents(queries.get(qid, ''))
    # check if query matches salary terms
    matched_term = next((t for t in SALARY_TERMS if t in qtext), None)
    if not matched_term:
        continue
        
    golds = labels[qid]
    d5 = top5[4]
    
    # Guard: do not displace gold d5
    is_d5_gold = d5 in golds
    is_target_gold = target_doc in golds
    
    tested_qids.append((qid, qtext, d5, doc_labels.get(d5, '')[:35], is_d5_gold, is_target_gold, top.index(target_doc) + 1))
    
    if is_target_gold and not is_d5_gold:
        wins += 1
    elif is_d5_gold and not is_target_gold:
        losses += 1
    else:
        ties += 1

print(f"Target Doc: {target_doc} ({target_lbl})")
print(f"Tested queries where target is at ranks 6-10 and query has salary terms: {len(tested_qids)}")
print(f"Wins: {wins} | Losses: {losses} | Ties: {ties} | Net: +{wins - losses}")

for qid, qtext, d5, lbl5, is_d5_gold, is_target_gold, r in tested_qids:
    outcome = "WIN" if (is_target_gold and not is_d5_gold) else ("LOSS" if (is_d5_gold and not is_target_gold) else "TIE")
    print(f"[{outcome}] QID {qid} (Rank {r}): {qtext[:70]}")
    print(f"    D5: {d5} ({lbl5}) [GOLD={is_d5_gold}] | Target [GOLD={is_target_gold}]")
