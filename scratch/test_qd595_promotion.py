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

from scratch.eval_h57_superseded import p_post

queries = {}
with open('cache/exp012b_v3/rankings/train/query_rows.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        queries[str(item.get('query_id') or item.get('id') or item.get('qid'))] = item.get('query') or ''

def strip_accents(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    return text.replace('đ', 'd').replace('Đ', 'D').lower()

target_id = '285041'
KEY_PHRASES = ['dong bao hiem', 'tham gia bao hiem', 'so bao hiem', 'thu bao hiem', 'cap so']

wins = 0
losses = 0
ties = 0
tested = []

for qid in eval_qids:
    top = p_post[qid]
    top5 = top[:5]
    if target_id in top5:
        continue
    if len(top) <= 5 or top[5] != target_id:
        continue
        
    qtext = strip_accents(queries.get(qid, ''))
    matched = any(p in qtext for p in KEY_PHRASES)
    if not matched:
        continue
        
    d5 = top5[4]
    lbl5 = doc_labels.get(d5, '')
    # Guard: never displace primary law or amendment
    if lbl5.startswith('luat') or lbl5.startswith('bo luat') or 'sua doi' in lbl5 or 'bo sung' in lbl5:
        continue
        
    golds = labels[qid]
    is_d5_g = d5 in golds
    is_target_g = target_id in golds
    
    if is_target_g and not is_d5_g:
        wins += 1
        tested.append(('WIN', qid, d5, lbl5, queries.get(qid, '')))
    elif is_d5_g and not is_target_g:
        losses += 1
        tested.append(('LOSS', qid, d5, lbl5, queries.get(qid, '')))
    else:
        ties += 1
        tested.append(('TIE', qid, d5, lbl5, queries.get(qid, '')))

print(f"Testing 285041 promotion from Rank 6 when query has insurance collection phrases:")
print(f"Wins: {wins} | Losses: {losses} | Ties: {ties} | Net: +{wins - losses}")
for outcome, qid, d5, lbl5, qtext in tested:
    print(f"[{outcome}] QID {qid}: D5={d5} ({lbl5[:35]})")
    print(f"     Query: {qtext[:80]}")
