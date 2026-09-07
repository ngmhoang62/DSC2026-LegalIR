import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import json
import re
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels
from pathlib import Path

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
labels, meta = get_canonical_labels()
eval_qids = [qid for qid, gold in labels.items() if gold]

queries = {}
with open('cache/exp012b_v3/rankings/train/query_rows.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        queries[str(item.get('query_id') or item.get('id') or item.get('qid'))] = item.get('query') or ''

from scratch.eval_h57_superseded import p_post

salary_docs = {
    '166505': 'NĐ 204/2004 Hệ số lương',
    '43443': 'NĐ 38/2019 Mức lương cơ sở 1.49M',
    '41395': 'NĐ 24/2023 Mức lương cơ sở 1.8M',
    '284539': 'NQ 69/2022 Tăng lương cơ sở',
}

print("Scanning salary-related queries...")
salary_queries = []
for qid in eval_qids:
    qtext = queries.get(qid, '').lower()
    if 'lương cơ sở' in qtext or 'hệ số lương' in qtext or 'bảng lương' in qtext or 'tăng mức lương' in qtext or 'tính lương' in qtext or 'mức lương của' in qtext:
        golds = labels[qid]
        top = p_post[qid]
        top5 = top[:5]
        hits = set(top5) & golds
        is_miss = len(hits) < len(golds)
        salary_queries.append((qid, qtext, golds, top[:10], is_miss))

print(f"Total salary queries: {len(salary_queries)}")
miss_cnt = sum(1 for x in salary_queries if x[4])
print(f"Queries with missing gold: {miss_cnt} / {len(salary_queries)}")

print("\n--- SAMPLE SALARY QUERIES WITH MISSING GOLD ---")
for qid, qtext, golds, top10, _ in [x for x in salary_queries if x[4]][:15]:
    print(f"\nQID {qid}: {qtext[:100]}")
    gold_names = [f"{g}: {doc_labels.get(g, '')[:35]}" for g in golds]
    print(f"  Golds ({len(golds)}): {', '.join(gold_names)}")
    print("  Top 5 in predictions:")
    for r, d in enumerate(top10[:5]):
        is_g = " [GOLD]" if d in golds else ""
        print(f"    {r+1}. {d}: {doc_labels.get(d, '')[:45]}{is_g}")
    print("  Ranks 6..10:")
    for r, d in enumerate(top10[5:10], start=6):
        is_g = " [GOLD]" if d in golds else ""
        print(f"    {r}. {d}: {doc_labels.get(d, '')[:45]}{is_g}")
