import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import json
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels
from pathlib import Path

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
labels, meta = get_canonical_labels()
eval_qids = [qid for qid, gold in labels.items() if gold]

from scratch.eval_h57_complete import all_preds, VERIFIED_SUPERSEDED_PAIRS

# Add the newly identified pair: NĐ 114/2003 vs NĐ 112/2011
expanded_pairs = list(VERIFIED_SUPERSEDED_PAIRS)
if ('167335', '273261') not in expanded_pairs:
    expanded_pairs.append(('167335', '273261')) # NĐ 114/2003 vs NĐ 112/2011 Công chức xã

old_to_new = dict(expanded_pairs)

queries = {}
with open('cache/exp012b_v3/rankings/train/query_rows.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        queries[str(item.get('query_id') or item.get('id') or item.get('qid'))] = item.get('query') or ''

wins = 0
losses = 0
ties = 0
tested = []

for qid in eval_qids:
    top = all_preds[qid]
    top5 = top[:5]
    
    # Check if an old superseded doc is in top 5 while its new successor is in ranks 6..8
    for old_id, new_id in old_to_new.items():
        if old_id in top5 and new_id not in top5 and new_id in top[5:8]:
            r_new = top.index(new_id) + 1
            golds = labels[qid]
            is_old_g = old_id in golds
            is_new_g = new_id in golds
            tested.append((qid, old_id, doc_labels.get(old_id, '')[:30], new_id, doc_labels.get(new_id, '')[:30], r_new, is_old_g, is_new_g, queries.get(qid, '')))
            if is_new_g and not is_old_g:
                wins += 1
            elif is_old_g and not is_new_g:
                losses += 1
            else:
                ties += 1

print(f"Testing Boundary Replacement of Superseded Statutes (Old in Top 5, New in Ranks 6..8):")
print(f"Total occurrences: {len(tested)}")
print(f"Wins: {wins} | Losses: {losses} | Ties: {ties} | Net: +{wins - losses}")

for qid, old_id, old_lbl, new_id, new_lbl, r_new, is_old_g, is_new_g, qtext in tested:
    outcome = "WIN" if (is_new_g and not is_old_g) else ("LOSS" if (is_old_g and not is_new_g) else "TIE")
    print(f"[{outcome}] QID {qid}: Old={old_id} ({old_lbl}) [GOLD={is_old_g}] -> New Rank {r_new}={new_id} ({new_lbl}) [GOLD={is_new_g}]")
    print(f"     Query: {qtext[:80]}")
