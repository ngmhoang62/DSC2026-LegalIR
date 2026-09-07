import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import json
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels
from pathlib import Path
from collections import Counter

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

zero_hit_qids = []
zero_hit_golds_at_6_10 = []

for qid in eval_qids:
    top = p_post[qid]
    top5 = top[:5]
    golds = labels[qid]
    hits = set(top5) & golds
    if not hits: # ZERO HIT!
        zero_hit_qids.append(qid)
        golds_in_6_10 = [g for g in golds if g in top[5:10]]
        if golds_in_6_10:
            for g in golds_in_6_10:
                r = top.index(g) + 1
                zero_hit_golds_at_6_10.append((qid, r, g, doc_labels.get(g, ''), top[:5]))

print(f"Total zero-hit queries in p_post: {len(zero_hit_qids)}")
print(f"Zero-hit queries with gold at ranks 6..10: {len(set(x[0] for x in zero_hit_golds_at_6_10))}")
print(f"Total golds at ranks 6..10 for zero-hit queries: {len(zero_hit_golds_at_6_10)}")

c = Counter(x[1] for x in zero_hit_golds_at_6_10)
print("Gold rank distribution in zero-hit queries:")
for r in sorted(c.keys()):
    print(f"  Rank {r}: {c[r]}")

print("\n--- SAMPLE ZERO-HIT QUERIES WITH GOLD AT RANK 6-8 ---")
for qid, r, g, lbl, top5 in zero_hit_golds_at_6_10[:15]:
    qtext = queries.get(qid, '')
    print(f"\nQID {qid} | Gold at Rank {r}: {g} ({lbl[:50]})")
    print(f"  Query: {qtext[:90]}")
    print(f"  Top 5: {[d + ' (' + doc_labels.get(d, '')[:20] + ')' for d in top5]}")
