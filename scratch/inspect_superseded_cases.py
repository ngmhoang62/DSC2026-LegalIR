import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
preds = json.load(open('results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json'))
labels, meta = get_canonical_labels()

ZERO_LOSS_PAIRS = [
    ('143446', '129823'),
    ('249551', '200355'),
    ('300749', '160120'),
    ('98167', '107019'),
    ('123271', '156'),
    ('288898', '177345'),
    ('78507', '65586'),
    ('111542', '247734'),
    ('174178', '13920'),
    ('256632', '175879'),
    ('173016', '166766'),
    ('293796', '189230'),
    ('10590', '122192'),
    ('180016', '251264'),
]
old_to_new = dict(ZERO_LOSS_PAIRS)
eval_qids = [qid for qid, gold in labels.items() if gold]

queries = {}
with open('cache/exp012b_v3/rankings/train/query_rows.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        queries[str(item.get('query_id') or item.get('id') or item.get('qid'))] = item.get('query') or ''

print('Inspecting candidates in queries where superseded doc is in top 5 and a gold is at ranks 6-10...')
for qid in eval_qids:
    top = preds[qid]
    top5 = top[:5]
    for old_id, new_id in old_to_new.items():
        if old_id in top5 and new_id in top5:
            golds = labels[qid]
            # check if gold is at ranks 6..10
            r6_10_golds = [g for g in golds if g in top[5:10]]
            if r6_10_golds:
                print(f"\n=======================================================")
                print(f"QID {qid}: Query: {queries.get(qid, '')[:120]}")
                print(f"Matched pair: {old_id} ({doc_labels.get(old_id, '')}) vs {new_id} ({doc_labels.get(new_id, '')})")
                print("Top 5:")
                for r, d in enumerate(top5):
                    print(f"  {r+1}. {d}: {doc_labels.get(d, '')} {'[GOLD]' if d in golds else ''}")
                print("Ranks 6..10:")
                for r, d in enumerate(top[5:10], start=6):
                    print(f"  {r}. {d}: {doc_labels.get(d, '')} {'[GOLD]' if d in golds else ''}")
