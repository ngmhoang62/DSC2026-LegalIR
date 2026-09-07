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

# Let's inspect state corporations / specific entities
# Phrases like 'tong cong ty ...' or 'tap doan ...'
CORP_PATTERNS = [
    r'tong cong ty\s+([a-z\s]+?)(?:\s+viet nam|\s+mien|\s+phai|\s+co|\s+duoc|$)',
    r'tap doan\s+([a-z\s]+?)(?:\s+viet nam|\s+phai|\s+co|\s+duoc|$)',
]

wins = 0
losses = 0
ties = 0
cases = []

for qid in eval_qids:
    qtext = strip_accents(queries.get(qid, ''))
    # check if query mentions a specific enterprise/corp
    corp_match = None
    for pat in CORP_PATTERNS:
        m = re.search(pat, qtext)
        if m:
            corp_name = m.group(1).strip()
            if len(corp_name) >= 3:
                corp_match = corp_name
                break
    if not corp_match:
        continue
        
    top = p_post[qid]
    d5 = top[4]
    lbl5 = strip_accents(doc_labels.get(d5, ''))
    
    # If D5 already matches the corp, no need to swap
    if corp_match in lbl5:
        continue
        
    # Check if any doc in 6..10 matches the corp
    for r, d_cand in enumerate(top[5:10], start=6):
        lbl_cand = strip_accents(doc_labels.get(d_cand, ''))
        if corp_match in lbl_cand:
            # Candidate matches exact corp!
            golds = labels[qid]
            is_d5_g = d5 in golds
            is_cand_g = d_cand in golds
            cases.append((qid, corp_match, d5, lbl5[:40], d_cand, lbl_cand[:40], r, is_d5_g, is_cand_g))
            if is_cand_g and not is_d5_g:
                wins += 1
            elif is_d5_g and not is_cand_g:
                losses += 1
            else:
                ties += 1
            break

print(f"Total corporate entity queries tested: {len(cases)}")
print(f"Wins: {wins} | Losses: {losses} | Ties: {ties} | Net: +{wins - losses}")
for qid, corp, d5, lbl5, d_cand, lbl_cand, r, is_d5_g, is_cand_g in cases:
    outcome = "WIN" if (is_cand_g and not is_d5_g) else ("LOSS" if (is_d5_g and not is_cand_g) else "TIE")
    print(f"[{outcome}] QID {qid} (Rank {r} -> 5): Entity='{corp}'")
    print(f"    D5: {d5} ({lbl5}) [GOLD={is_d5_g}]")
    print(f"    Cand: {d_cand} ({lbl_cand}) [GOLD={is_cand_g}]")
