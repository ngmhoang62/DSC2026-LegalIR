import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import json
import re
import unicodedata
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels
from pathlib import Path

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
labels, meta = get_canonical_labels()
eval_qids = [qid for qid, gold in labels.items() if gold]

from scratch.eval_h57_complete import all_preds

queries = {}
with open('cache/exp012b_v3/rankings/train/query_rows.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        queries[str(item.get('query_id') or item.get('id') or item.get('qid'))] = item.get('query') or ''

def strip_accents(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    return text.replace('đ', 'd').replace('Đ', 'D').lower()

QD595_ID = '285041'
QD595_PHRASES = ['dong bao hiem', 'tham gia bao hiem', 'so bao hiem', 'thu bao hiem', 'cap so', 'bao hiem that nghiep']

wins = 0
losses = 0
ties = 0
tested = []

for qid in eval_qids:
    top = all_preds[qid]
    top5 = top[:5]
    if QD595_ID in top5:
        continue
    # look in ranks 6..8
    if QD595_ID not in top[5:8]:
        continue
        
    qtext = strip_accents(queries.get(qid, ''))
    matched = any(p in qtext for p in QD595_PHRASES)
    if not matched:
        continue
        
    d5 = top5[4]
    lbl5 = doc_labels.get(d5, '')
    if lbl5.startswith('luat') or lbl5.startswith('bo luat') or 'sua doi' in lbl5 or 'bo sung' in lbl5:
        continue
        
    golds = labels[qid]
    is_d5_g = d5 in golds
    is_target_g = QD595_ID in golds
    r = top.index(QD595_ID) + 1
    
    if is_target_g and not is_d5_g:
        wins += 1
        tested.append(('WIN', qid, r, d5, lbl5, queries.get(qid, '')))
    elif is_d5_g and not is_target_g:
        losses += 1
        tested.append(('LOSS', qid, r, d5, lbl5, queries.get(qid, '')))
    else:
        ties += 1
        tested.append(('TIE', qid, r, d5, lbl5, queries.get(qid, '')))

print(f"Testing QD 595 promotion from Ranks 6..8:")
print(f"Wins: {wins} | Losses: {losses} | Ties: {ties} | Net: +{wins - losses}")
for outcome, qid, r, d5, lbl5, qtext in tested:
    print(f"[{outcome}] QID {qid} (Rank {r} -> 5): D5={d5} ({lbl5[:35]})")
    print(f"     Query: {qtext[:80]}")
