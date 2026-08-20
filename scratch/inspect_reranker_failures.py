import json
from pathlib import Path
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("d:/Study/DSC2026/LegalIR/src")

from exp012b_core import load_answers, read_jsonl
from exp012b_tuning import load_folds
from exp013b_core import rank_ids

folds = load_folds(Path('cache/cv_folds.json'))
answers = load_answers(Path('public_test_dataset/train.json'))
queries_data = json.load(open('public_test_dataset/train.json', 'r', encoding='utf-8'))
qids = {str(v) for v in folds['fold_0']}

lam = {str(r['qid']): r['candidates'] for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl')) if str(r['qid']) in qids}
qwen = {str(r['qid']): r['scores'] for r in read_jsonl(Path('cache/exp013b_cascade/qwen/fold_0/scores.jsonl')) if str(r['qid']) in qids}
capsules = {str(r['qid']): {str(c['doc_id']): c for c in r['candidates']} for r in read_jsonl(Path('cache/exp013b_cascade/capsules/train/train_capsules.jsonl')) if str(r['qid']) in qids}

count = 0
for qid in qids:
    g = answers[qid]
    q_sc = {str(x['doc_id']): float(x['qwen_score']) for x in qwen[qid]}
    pure_rerank = rank_ids(q_sc)
    
    lam_top5 = set(str(x['doc_id']) for x in lam[qid][:5])
    rerank_top5 = set(pure_rerank[:5])
    
    if (lam_top5 & g) and not (rerank_top5 & g):
        count += 1
        gold_doc = list(g)[0]
        q_text = queries_data[qid]['question']
        print(f"\n--- Case {count}: QID {qid} ---")
        print(f"Question: {q_text}")
        print(f"Gold doc: {gold_doc}")
        print(f"Lambda rank of gold: {next((x['rank'] for x in lam[qid] if str(x['doc_id']) == gold_doc), None)}")
        print(f"Reranker rank of gold: {pure_rerank.index(gold_doc) + 1 if gold_doc in pure_rerank else 'N/A'}")
        
        # Look at capsule of gold vs capsule of rank 1
        if gold_doc in capsules[qid]:
            print("\n[GOLD DOC CAPSULE]:")
            print(capsules[qid][gold_doc]['document'][:400])
        top1_doc = pure_rerank[0]
        if top1_doc in capsules[qid]:
            print(f"\n[TOP 1 FALSE POSITIVE CAPSULE (doc {top1_doc})]:")
            print(capsules[qid][top1_doc]['document'][:400])
            
        if count >= 3:
            break
