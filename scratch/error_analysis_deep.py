import json
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("d:/Study/DSC2026/LegalIR/src")

from exp012b_core import load_answers, read_jsonl

answers = load_answers(Path('public_test_dataset/train.json'))
queries_data = json.load(open('public_test_dataset/train.json', 'r', encoding='utf-8'))
oof_report = json.load(open('results/exp013b_cascade/oof/oof_report.json', 'r', encoding='utf-8'))
v3_dir = Path('cache/structural_v3')
doc_meta = {str(r['doc_id']): str(r.get('document_label', '')) for r in read_jsonl(v3_dir / 'documents.jsonl')}

preds = oof_report['predictions']

fail_count = 0
reasons = {
    "gold_not_in_top24": 0,
    "gold_at_rank_6_to_10": 0,
    "gold_at_rank_11_to_24": 0,
    "law_vs_decree_conflict": 0,
    "year_superseded_conflict": 0
}

sample_cases = []

for qid in sorted(answers):
    gold_set = answers[qid]
    top5 = set(preds[qid][:5])
    top24 = preds[qid][:24]
    
    hit = bool(gold_set & top5)
    if not hit:
        fail_count += 1
        gold_doc = list(gold_set)[0]
        q_text = queries_data[qid]['question']
        
        gold_rank = (top24.index(gold_doc) + 1) if gold_doc in top24 else 999
        
        if gold_rank == 999:
            reasons["gold_not_in_top24"] += 1
        elif 6 <= gold_rank <= 10:
            reasons["gold_at_rank_6_to_10"] += 1
        else:
            reasons["gold_at_rank_11_to_24"] += 1
            
        top1_doc = preds[qid][0]
        gold_label = doc_meta.get(gold_doc, gold_doc)
        top1_label = doc_meta.get(top1_doc, top1_doc)
        
        if len(sample_cases) < 10:
            sample_cases.append({
                "qid": qid,
                "question": q_text,
                "gold_doc": gold_doc,
                "gold_label": gold_label,
                "gold_rank": gold_rank,
                "top1_doc": top1_doc,
                "top1_label": top1_label,
                "top5_labels": [doc_meta.get(d, d) for d in preds[qid][:5]]
            })

print(f"Total Failures: {fail_count}/7000 ({fail_count/7000*100:.2f}%)")
print(f"Breakdown: {reasons}")
print("\n--- SAMPLE ERROR CASES ---")
for idx, c in enumerate(sample_cases, 1):
    print(f"\n[Case {idx}] QID {c['qid']} (Gold Rank: {c['gold_rank']})")
    print(f"Question: {c['question']}")
    print(f"Gold: [{c['gold_doc']}] {c['gold_label']}")
    print(f"Top-1 Predicted: [{c['top1_doc']}] {c['top1_label']}")
    print("Top-5 Predicted:")
    for rank, l in enumerate(c['top5_labels'], 1):
        print(f"  {rank}. {l}")
