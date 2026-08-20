import json
import math
import time
from pathlib import Path
import sys
import io
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("d:/Study/DSC2026/LegalIR/src")

from exp012b_core import load_answers, read_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp013b_core import rank_ids
from exp013b_capsules import ScopedEvidenceFallback, _render, PersistentChunkReader, load_document_metadata
from exp013b_fusion import _fuse

folds = load_folds(Path('cache/cv_folds.json'))
answers = load_answers(Path('public_test_dataset/train.json'))
qids = {str(v) for v in folds['fold_0']}

candidates = {str(r['qid']): r for r in read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl')) if str(r['qid']) in qids}
predictions = {str(r['qid']): r for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl')) if str(r['qid']) in qids}
gold = {qid: answers[qid] for qid in qids}

v3_dir = Path('cache/structural_v3')
lookup_db = Path('cache/exp012b_v3/chunk_lookup/chunk_offsets.sqlite')
ranking_dir = Path('cache/exp012b_v3/rankings/train')
bge_dir = Path('cache/exp012b_v3/bge_leaves')
bm25_db = Path('cache/exp012b_v3/bm25_index/bm25_v3.sqlite')

# Load all wanted docs for fold 0
wanted_docs = set()
for qid in qids:
    for item in predictions[qid]['candidates'][:24]:
        wanted_docs.add(str(item['doc_id']))

print(f"Total unique shortlisted docs for fold 0: {len(wanted_docs)}")
tok = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
m = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3', torch_dtype=torch.float16).cuda().eval()

docs = load_document_metadata(v3_dir)
scope_ids = {str(node_id) for row in docs.values() for node_id in row.get("scope_node_ids", [])}
scope_nodes = {str(row["node_id"]): str(row.get("raw_text", "")) for row in read_jsonl(v3_dir / "nodes.jsonl") if str(row["node_id"]) in scope_ids}

print("Running Scoped refinement on fold 0...")
new_capsules = {}
with ScopedEvidenceFallback(wanted_docs=wanted_docs, ranking_dir=ranking_dir, bge_dir=bge_dir, bm25_db=bm25_db) as scoped, PersistentChunkReader(v3_dir, lookup_db) as reader:
    for qid in qids:
        query_str = str(candidates[qid]["query"])
        selected = predictions[qid]["candidates"][:24]
        selections = {}
        need = set()
        for rank_row in selected:
            doc_id = str(rank_row["doc_id"])
            ids = scoped.select(qid=qid, query=query_str, doc_id=doc_id)
            selections[doc_id] = ids
            need.update(c_id for c_id, _ in ids)
        chunks = reader.load(need)
        
        rows = []
        for rank_row in selected:
            doc_id = str(rank_row["doc_id"])
            evidence = [chunks[c_id] for c_id, _ in selections[doc_id]]
            scope_text = "\n".join(scope_nodes.get(str(node_id), "") for node_id in docs[doc_id].get("scope_node_ids", []))
            query_val, doc_val = _render(tok, query_str, str(docs[doc_id].get("document_label", doc_id)), evidence, max_length=768, scope_text=scope_text)
            rows.append({"doc_id": doc_id, "query": query_val, "document": doc_val})
        new_capsules[qid] = rows

print("Scoring with BGE-Reranker...")
new_qwen_scores = {}
batch_size = 16
for qid in qids:
    rows = new_capsules[qid]
    pairs = [(r["query"], r["document"]) for r in rows]
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        inp = tok([p[0] for p in batch], [p[1] for p in batch], padding=True, truncation=True, max_length=768, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            vals = m(**inp).logits.reshape(-1).float().cpu().tolist()
            scores.extend(vals)
    new_qwen_scores[qid] = [{"doc_id": r["doc_id"], "qwen_score": s} for r, s in zip(rows, scores)]

# Evaluate pure reranker and fused
cand_map = {str(r['qid']): {str(i['doc_id']): i for i in r['candidates']} for r in read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl')) if str(r['qid']) in qids}
lam_map = {str(r['qid']): r['candidates'] for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl')) if str(r['qid']) in qids}

best_recall = 0
best_cfg = None
for w_qwen in [1.0, 2.0, 3.0, 4.0, 5.0]:
    for w_lam in [0.5, 1.0, 2.0]:
        fused = {}
        for qid in qids:
            fused[qid] = _fuse(lam_map[qid], new_qwen_scores[qid], cand_map[qid], {"qwen": w_qwen, "lambda": w_lam, "bge": 0.0})
        met = evaluate_rankings(fused, gold, ks=(5,))
        if met["recall@5"] > best_recall:
            best_recall = met["recall@5"]
            best_cfg = (w_qwen, w_lam, met)

print(f"\n==========================================")
print(f"NEW Fused Result with Scoped Capsules: {best_cfg}")
print(f"==========================================")
