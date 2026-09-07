import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
from gemini.labels import get_canonical_labels
from pathlib import Path

ROOT = Path('.')
labels, meta = get_canonical_labels()
eval_qids = [qid for qid, gold in labels.items() if gold]

from scratch.eval_h57_superseded import p_post, xgb_tuned, lgb145, xgb131, prof

zero_hit_qids = set()
for qid in eval_qids:
    top5 = p_post[qid][:5]
    if not (set(top5) & labels[qid]):
        zero_hit_qids.add(qid)

print(f"Total zero-hit queries: {len(zero_hit_qids)}")

# Measure agreement among the 4 models for each query
# Signal 1: Number of models that agree on Rank 1
# Signal 2: Size of union of top 3 from all 4 models (if high, models disagree; if 3, perfect agreement)
# Signal 3: How many of the 4 models have p_post[0] in their top 1?

agreements_zero = []
agreements_hit = []

union_sizes_zero = []
union_sizes_hit = []

for qid in eval_qids:
    top1s = [
        xgb_tuned.get(qid, [''])[0],
        lgb145.get(qid, [''])[0],
        xgb131.get(qid, [''])[0],
        prof.get(qid, [''])[0],
    ]
    # top 3 union size across 4 models
    t3_union = set(
        xgb_tuned.get(qid, [])[:3] +
        lgb145.get(qid, [])[:3] +
        xgb131.get(qid, [])[:3] +
        prof.get(qid, [])[:3]
    )
    union_size = len(t3_union)
    
    # agreement: max count of same doc at rank 1
    from collections import Counter
    max_top1_count = Counter(top1s).most_common(1)[0][1]
    
    if qid in zero_hit_qids:
        agreements_zero.append(max_top1_count)
        union_sizes_zero.append(union_size)
    else:
        agreements_hit.append(max_top1_count)
        union_sizes_hit.append(union_size)

print(f"Mean Top-1 Model Agreement: Hit queries = {np.mean(agreements_hit):.2f} vs Zero-hit = {np.mean(agreements_zero):.2f}")
print(f"Mean Top-3 Model Union Size: Hit queries = {np.mean(union_sizes_hit):.2f} vs Zero-hit = {np.mean(union_sizes_zero):.2f}")

# Distribution of Top-1 agreement:
print("\nTop-1 Agreement Distribution:")
print("Count=4 (all 4 agree):")
print(f"  Hits: {sum(1 for x in agreements_hit if x == 4)} ({sum(1 for x in agreements_hit if x == 4)/len(agreements_hit)*100:.1f}%)")
print(f"  Zero-hits: {sum(1 for x in agreements_zero if x == 4)} ({sum(1 for x in agreements_zero if x == 4)/len(agreements_zero)*100:.1f}%)")

print("Count=1 (all 4 disagree):")
print(f"  Hits: {sum(1 for x in agreements_hit if x == 1)} ({sum(1 for x in agreements_hit if x == 1)/len(agreements_hit)*100:.1f}%)")
print(f"  Zero-hits: {sum(1 for x in agreements_zero if x == 1)} ({sum(1 for x in agreements_zero if x == 1)/len(agreements_zero)*100:.1f}%)")
