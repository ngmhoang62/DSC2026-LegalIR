import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import json
import re
from pathlib import Path
from collections import defaultdict
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
preds = json.load(open('results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json'))
labels, meta = get_canonical_labels()

eval_qids = [qid for qid, gold in labels.items() if gold]

# Let's inspect all docs in corpus
# A document is superseded if there is a newer document with the same law/decree subject
# Let's do a more comprehensive scan of all docs that co-occur in Top 10

# Let's build a map: for each pair of docs (d1, d2) that co-occur in top 5:
# if d1 and d2 have high token overlap in their title and d1 is older than d2:
# check gold distribution!

def extract_year(lbl):
    years = re.findall(r'\b(19\d\d|20\d\d)\b', lbl)
    return int(years[0]) if years else None

pair_stats = defaultdict(lambda: {'both_top5': 0, 'old_gold': 0, 'new_gold': 0, 'neither_gold': 0, 'both_gold': 0})

for qid in eval_qids:
    top5 = preds[qid][:5]
    golds = labels[qid]
    
    for i in range(len(top5)):
        for j in range(i+1, len(top5)):
            d1, d2 = top5[i], top5[j]
            lbl1 = doc_labels.get(d1, '')
            lbl2 = doc_labels.get(d2, '')
            y1, y2 = extract_year(lbl1), extract_year(lbl2)
            if y1 is None or y2 is None or y1 == y2:
                continue
            
            old_d, new_d = (d1, d2) if y1 < y2 else (d2, d1)
            old_lbl, new_lbl = (lbl1, lbl2) if y1 < y2 else (lbl2, lbl1)
            old_y, new_y = min(y1, y2), max(y1, y2)
            
            # Check title similarity (Jaccard of words without numbers)
            w1 = set(re.findall(r'[a-z]+', old_lbl)) - {'so', 'nam', 'ngay', 'thang', 've', 'cua', 'tai'}
            w2 = set(re.findall(r'[a-z]+', new_lbl)) - {'so', 'nam', 'ngay', 'thang', 've', 'cua', 'tai'}
            if not w1 or not w2:
                continue
            jaccard = len(w1 & w2) / len(w1 | w2)
            
            # If jaccard > 0.5, they are likely the same statutory instrument in different years!
            if jaccard >= 0.45:
                pair_key = (old_d, new_d, old_lbl, new_lbl, old_y, new_y, round(jaccard, 2))
                stats = pair_stats[pair_key]
                stats['both_top5'] += 1
                is_old = old_d in golds
                is_new = new_d in golds
                if is_old and is_new:
                    stats['both_gold'] += 1
                elif is_old:
                    stats['old_gold'] += 1
                elif is_new:
                    stats['new_gold'] += 1
                else:
                    stats['neither_gold'] += 1

print(f"Total candidate superseded pairs found in Top 5: {len(pair_stats)}")

# Sort by frequency
sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]['both_top5'], reverse=True)

print("\n--- ZERO OLD-GOLD PAIRS (Safe to Dedup) ---")
safe_pairs = []
for (old_d, new_d, old_lbl, new_lbl, old_y, new_y, jac), stats in sorted_pairs:
    if stats['old_gold'] == 0 and stats['both_gold'] == 0:
        safe_pairs.append((old_d, new_d, old_lbl, new_lbl, stats))
        print(f"\n({stats['both_top5']}x) Old ({old_y}): {old_d} ({old_lbl[:60]})")
        print(f"       New ({new_y}): {new_d} ({new_lbl[:60]})")
        print(f"       New Gold: {stats['new_gold']} | Neither: {stats['neither_gold']} | Jac: {jac}")

print("\n--- NON-ZERO OLD-GOLD PAIRS (Unsafe) ---")
for (old_d, new_d, old_lbl, new_lbl, old_y, new_y, jac), stats in sorted_pairs:
    if stats['old_gold'] > 0 or stats['both_gold'] > 0:
        print(f"\n({stats['both_top5']}x) Old ({old_y}): {old_d} [OLD_GOLD={stats['old_gold']}, BOTH={stats['both_gold']}]")
        print(f"       New ({new_y}): {new_d} [NEW_GOLD={stats['new_gold']}]")
        print(f"       Old: {old_lbl[:50]} | New: {new_lbl[:50]}")
