import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import json
import re
from collections import Counter, defaultdict
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

r6_10_misses = []
doc_type_counts = Counter()
ministry_counts = Counter()

for qid in eval_qids:
    top = preds[qid]
    top5 = top[:5]
    golds = labels[qid]
    missing = golds - set(top5)
    for g in missing:
        if g in top:
            r = top.index(g) + 1
            if 6 <= r <= 10:
                lbl = doc_labels.get(g, '')
                # determine doc type
                dtype = 'other'
                if lbl.startswith('luat') or lbl.startswith('bo luat'):
                    dtype = 'luat'
                elif lbl.startswith('nghi dinh'):
                    dtype = 'nghi dinh'
                elif lbl.startswith('thong tu'):
                    dtype = 'thong tu'
                elif lbl.startswith('quyet dinh'):
                    dtype = 'quyet dinh'
                elif lbl.startswith('nghi quyet'):
                    dtype = 'nghi quyet'
                elif lbl.startswith('cong van') or lbl.startswith('thong bao'):
                    dtype = 'cong van/thong bao'
                elif lbl.startswith('qcvn') or lbl.startswith('tcvn') or lbl.startswith('tieu chuan'):
                    dtype = 'standard'
                doc_type_counts[dtype] += 1
                
                # ministry/issuer
                issuers = re.findall(r'\b(btc|bca|bqp|byt|bgddt|bxd|bct|btnmt|bldtbxh|bgtvt|bkhdt|bnv|bnnptnt|bhxh|tandtc|vksndtc|ubnd)\b', lbl)
                for iss in issuers:
                    ministry_counts[iss] += 1
                    
                r6_10_misses.append((r, qid, g, dtype, lbl, top[:5]))

print(f"Total missing golds at Ranks 6..10: {len(r6_10_misses)}")
print("\nDocument Type Distribution of Missing Golds at 6..10:")
for dt, cnt in doc_type_counts.most_common():
    print(f"  {dt}: {cnt} ({cnt/len(r6_10_misses)*100:.1f}%)")

print("\nIssuer Distribution of Missing Golds at 6..10:")
for iss, cnt in ministry_counts.most_common(12):
    print(f"  {iss}: {cnt}")

print("\n--- SAMPLE CIRCULARS (THONG TU) AT RANKS 6-7 ---")
tt_misses = [x for x in r6_10_misses if x[3] == 'thong tu' and x[0] in (6, 7)]
for r, qid, g, _, lbl, top5 in tt_misses[:10]:
    qtext = queries.get(qid, '')[:70]
    print(f"\nQID {qid} (Rank {r}): {g} ({lbl[:55]})")
    print(f"  Query: {qtext}")
    print(f"  Top 5: {[d + ' (' + doc_labels.get(d, '')[:20] + ')' for d in top5]}")
