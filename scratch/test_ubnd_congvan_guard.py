import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import json
import re
import unicodedata
from pathlib import Path
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
preds = json.load(open('results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json'))
labels, meta = get_canonical_labels()

eval_qids = [qid for qid, gold in labels.items() if gold]

queries = {}
with open('cache/exp012b_v3/rankings/train/query_rows.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        queries[str(item.get('query_id') or item.get('id') or item.get('qid'))] = item.get('query') or ''

def strip_accents(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    return text.replace('đ', 'd').replace('Đ', 'D').lower()

PROVINCE_TERMS = {
    'tinh', 'thanh pho', 'tp', 'tphcm', 'ha noi', 'da nang', 'hai phong', 'can tho',
    'ubnd', 'hoi dong nhan dan', 'hdnd', 'dia phuong', 'tinh uy', 'quan', 'huyen', 'xa'
}

DISPATCH_TERMS = {
    'cong van', 'thong bao', 'huong dan cua cuc', 'tong cuc'
}

# Let's test swapping Rank 5 and Rank 6
wins = 0
losses = 0
ties = 0
swapped = []

for qid in eval_qids:
    top = preds[qid]
    d5 = top[4]
    d6 = top[5] if len(top) > 5 else None
    if not d6:
        continue
        
    lbl5 = doc_labels.get(d5, '')
    lbl6 = doc_labels.get(d6, '')
    qtext = queries.get(qid, '')
    qclean = strip_accents(qtext)
    
    # Check if D5 is provincial or dispatch
    is_ubnd_5 = bool('ubnd' in lbl5 or 'qd ubnd' in lbl5)
    is_cv_5 = bool(lbl5.startswith('cong van') or lbl5.startswith('thong bao'))
    
    if not (is_ubnd_5 or is_cv_5):
        continue
        
    # Check if query asks for local/provincial
    if is_ubnd_5:
        # if query explicitly has province terms, don't demote
        has_local = any(re.search(rf'\b{t}\b', qclean) for t in PROVINCE_TERMS)
        if has_local:
            continue
            
    if is_cv_5:
        # if query explicitly has dispatch terms, don't demote
        has_disp = any(re.search(rf'\b{t}\b', qclean) for t in DISPATCH_TERMS)
        if has_disp:
            continue
            
    # Check if D6 is normative
    is_normative_6 = bool(
        lbl6.startswith('luat') or lbl6.startswith('bo luat') or
        lbl6.startswith('nghi dinh') or lbl6.startswith('thong tu') or
        lbl6.startswith('nghi quyet') or lbl6.startswith('quyet dinh') and 'ubnd' not in lbl6
    )
    if not is_normative_6:
        continue
        
    # Check topic overlap between query and D6
    q_words = set(re.findall(r'[a-z]{3,}', qclean)) - {'nhu', 'the', 'nao', 'duoc', 'khong', 'quy', 'dinh', 'theo', 'phap', 'luat', 'cho', 'biet', 'trong', 'truong', 'hop', 'co'}
    lbl6_words = set(re.findall(r'[a-z]{3,}', strip_accents(lbl6))) - {'nghi', 'dinh', 'thong', 'luat', 'quyet', 'nam', 'huong', 'dan', 'so'}
    
    overlap = q_words & lbl6_words
    # Topic guard: must share at least 2 substantive words (or 1 long word >= 5 chars)
    if len(overlap) < 2 and not any(len(w) >= 5 for w in overlap):
        continue
        
    # Evaluate swap
    golds = labels[qid]
    top5_old = top[:5]
    top5_new = top[:4] + [d6]
    
    old_hits = len(set(top5_old) & golds)
    new_hits = len(set(top5_new) & golds)
    
    old_rec = old_hits / len(golds)
    new_rec = new_hits / len(golds)
    
    if new_rec > old_rec:
        wins += 1
        swapped.append(('WIN', qid, old_rec, new_rec, d5, lbl5, d6, lbl6, overlap, qtext))
    elif new_rec < old_rec:
        losses += 1
        swapped.append(('LOSS', qid, old_rec, new_rec, d5, lbl5, d6, lbl6, overlap, qtext))
    else:
        ties += 1

print(f"Total evaluated candidates for swap: {wins + losses + ties}")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Ties: {ties}")
print(f"Net Gain: +{wins - losses}")

for outcome, qid, old_rec, new_rec, d5, lbl5, d6, lbl6, overlap, qtext in swapped:
    print(f"\n[{outcome}] QID {qid}: Recall {old_rec:.3f} -> {new_rec:.3f}")
    print(f"  Query: {qtext[:100]}")
    print(f"  D5 (demoted): {d5} ({lbl5[:60]})")
    print(f"  D6 (promoted): {d6} ({lbl6[:60]})")
    print(f"  Overlap: {overlap}")
