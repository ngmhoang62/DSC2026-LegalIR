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

STOPWORDS = {
    'nhu', 'the', 'nao', 'duoc', 'khong', 'quy', 'dinh', 'theo', 'phap', 'luat',
    'cho', 'biet', 'trong', 'truong', 'hop', 'co', 'va', 'cua', 'tai', 've', 'cac',
    'nhung', 'gi', 'thi', 'den', 'tu', 'do', 'ra', 'sao', 'muc', 'nao', 'ai', 'khi'
}

PROVINCE_TERMS = {
    'tinh', 'thanh pho', 'tp', 'tphcm', 'ha noi', 'da nang', 'hai phong', 'can tho',
    'ubnd', 'hoi dong nhan dan', 'hdnd', 'dia phuong', 'tinh uy', 'quan', 'huyen', 'xa'
}

DISPATCH_TERMS = {
    'cong van', 'thong bao', 'huong dan cua cuc', 'tong cuc'
}

def get_substantive_tokens(text: str) -> set[str]:
    cleaned = strip_accents(text)
    words = re.findall(r'[a-z]{3,}', cleaned)
    return {w for w in words if w not in STOPWORDS}

# Let's test different thresholds of topic overlap
for min_overlap in [1, 2]:
    wins = 0
    losses = 0
    ties = 0
    details = []

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
        
        is_ubnd_5 = bool('ubnd' in lbl5 or 'qd ubnd' in lbl5)
        is_cv_5 = bool(lbl5.startswith('cong van') or lbl5.startswith('thong bao'))
        
        if not (is_ubnd_5 or is_cv_5):
            continue
            
        # If query asks for local/provincial, don't demote UBND
        if is_ubnd_5 and any(re.search(rf'\b{t}\b', qclean) for t in PROVINCE_TERMS):
            continue
        # If query asks for dispatch, don't demote Cong Van
        if is_cv_5 and any(re.search(rf'\b{t}\b', qclean) for t in DISPATCH_TERMS):
            continue
            
        # Check if D6 is normative
        is_normative_6 = bool(
            lbl6.startswith('luat') or lbl6.startswith('bo luat') or
            lbl6.startswith('nghi dinh') or lbl6.startswith('thong tu') or
            lbl6.startswith('nghi quyet') or (lbl6.startswith('quyet dinh') and 'ubnd' not in lbl6)
        )
        if not is_normative_6:
            continue
            
        q_tokens = get_substantive_tokens(qtext)
        lbl6_tokens = get_substantive_tokens(lbl6) - {'nghi', 'dinh', 'thong', 'luat', 'quyet', 'nam', 'huong', 'dan', 'so', 'ban', 'hanh'}
        
        # Also check overlap with Top 1-4 doc labels (topical context)
        top4_tokens = set()
        for d in top[:4]:
            top4_tokens |= (get_substantive_tokens(doc_labels.get(d, '')) - {'nghi', 'dinh', 'thong', 'luat', 'quyet', 'nam', 'huong', 'dan', 'so', 'ban', 'hanh'})
            
        overlap_query = q_tokens & lbl6_tokens
        overlap_top4 = top4_tokens & lbl6_tokens
        
        # Guard: must share topic words with query OR strong overlap with Top 4
        if len(overlap_query) < min_overlap and len(overlap_top4) < (min_overlap + 1):
            continue
            
        golds = labels[qid]
        is_d5_gold = d5 in golds
        is_d6_gold = d6 in golds
        
        if is_d6_gold and not is_d5_gold:
            wins += 1
            details.append(('WIN', qid, d5, lbl5, d6, lbl6, overlap_query, overlap_top4, qtext))
        elif is_d5_gold and not is_d6_gold:
            losses += 1
            details.append(('LOSS', qid, d5, lbl5, d6, lbl6, overlap_query, overlap_top4, qtext))
        else:
            ties += 1
            
    print(f"\n--- min_overlap={min_overlap} ---")
    print(f"Wins: {wins} | Losses: {losses} | Ties: {ties} | Net: +{wins - losses}")
    for item in details:
        print(f"[{item[0]}] QID {item[1]}: D5={item[2]} ({item[3][:35]}) -> D6={item[4]} ({item[5][:35]})")
        print(f"     Query: {item[8][:80]}")
        print(f"     Overlap Q: {item[6]} | Overlap Top4: {item[7]}")
