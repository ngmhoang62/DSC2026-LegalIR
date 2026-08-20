import json
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
from exp013b_capsules import _render, PersistentChunkReader, load_document_metadata
from exp013b_fusion import _fuse

t0 = time.time()
print("1. Loading folds, metadata and models...")
folds = load_folds(Path('cache/cv_folds.json'))
answers = load_answers(Path('public_test_dataset/train.json'))
qids_list = [str(v) for v in folds['fold_0']]
qids = set(qids_list)
gold = {qid: answers[qid] for qid in qids}

candidates = {str(r['qid']): r for r in read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl')) if str(r['qid']) in qids}
predictions = {str(r['qid']): r for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl')) if str(r['qid']) in qids}

v3_dir = Path('cache/structural_v3')
lookup_db = Path('cache/exp012b_v3/chunk_lookup/chunk_offsets.sqlite')
ranking_dir = Path('cache/exp012b_v3/rankings/train')
bge_dir = Path('cache/exp012b_v3/bge_leaves')

docs = load_document_metadata(v3_dir)
scope_ids = {str(node_id) for row in docs.values() for node_id in row.get("scope_node_ids", [])}
scope_nodes = {str(row["node_id"]): str(row.get("raw_text", "")) for row in read_jsonl(v3_dir / "nodes.jsonl") if str(row["node_id"]) in scope_ids}

# 2. Fast GPU Matrix Multiplication for exact top chunk retrieval
print("2. Computing GPU matrix multiplication for top chunk retrieval...")
query_rows = list(read_jsonl(ranking_dir / "query_rows.jsonl"))
qid_to_row = {str(row["qid"]): int(row["row"]) for row in query_rows}
query_embeddings = np.load(ranking_dir / "query_embeddings.f16.npy", mmap_mode="r")

fold0_rows = [qid_to_row[qid] for qid in qids_list]
Q = torch.from_numpy(np.asarray(query_embeddings[fold0_rows], dtype=np.float32)).cuda() # (1400, 1024)

leaf_rows = list(read_jsonl(bge_dir / "leaf_rows.jsonl"))
leaf_vectors = np.memmap(bge_dir / "leaf_embeddings.f16", dtype=np.float16, mode="r").reshape(len(leaf_rows), 1024)
L = torch.from_numpy(np.asarray(leaf_vectors, dtype=np.float32)).cuda() # (435300, 1024)

# Doc to leaf indices mapping
doc_to_leaf_indices = {}
for idx, r in enumerate(leaf_rows):
    doc_id = str(r["doc_id"])
    if doc_id not in doc_to_leaf_indices:
        doc_to_leaf_indices[doc_id] = []
    doc_to_leaf_indices[doc_id].append(idx)

# Compute top chunks per candidate doc for all 1400 queries
print(f"Matrix shapes: Q={Q.shape}, L={L.shape}. Time elapsed: {time.time()-t0:.2f}s")
print("3. Extracting best chunks per candidate doc...")

selections_by_qid = {}
all_needed_chunks = set()

for q_idx, qid in enumerate(qids_list):
    q_vec = Q[q_idx] # (1024,)
    selected_candidates = predictions[qid]["candidates"][:24]
    doc_selections = {}
    
    for cand_row in selected_candidates:
        doc_id = str(cand_row["doc_id"])
        indices = doc_to_leaf_indices.get(doc_id, [])
        if not indices:
            doc_selections[doc_id] = []
            continue
            
        doc_leaf_mat = L[indices] # (num_chunks, 1024)
        scores = (doc_leaf_mat @ q_vec).cpu().numpy()
        sorted_order = np.argsort(-scores)
        
        # Pick top 2 chunks with different parents if possible
        chosen = []
        seen_parents = set()
        for pos in sorted_order:
            row_info = leaf_rows[indices[pos]]
            parent = str(row_info["parent_node_id"])
            chunk_id = str(row_info["chunk_id"])
            if parent not in seen_parents:
                chosen.append((chunk_id, parent))
                seen_parents.add(parent)
            if len(chosen) == 2:
                break
        if not chosen and sorted_order.size > 0:
            row_info = leaf_rows[indices[sorted_order[0]]]
            chosen.append((str(row_info["chunk_id"]), str(row_info["parent_node_id"])))
            
        doc_selections[doc_id] = chosen
        all_needed_chunks.update(c_id for c_id, _ in chosen)
        
    selections_by_qid[qid] = doc_selections

print(f"Chunks selection done in {time.time()-t0:.2f}s! Needed unique chunks: {len(all_needed_chunks)}")

# 4. Render capsules
tok = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
m = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3', torch_dtype=torch.float16).cuda().eval()

print("4. Loading chunk text and rendering capsules...")
new_capsules = {}
with PersistentChunkReader(v3_dir, lookup_db) as reader:
    chunks_data = reader.load(all_needed_chunks)
    for qid in qids_list:
        query_str = str(candidates[qid]["query"])
        selected_candidates = predictions[qid]["candidates"][:24]
        rows = []
        for cand_row in selected_candidates:
            doc_id = str(cand_row["doc_id"])
            ids = selections_by_qid[qid][doc_id]
            evidence = [chunks_data[c_id] for c_id, _ in ids if c_id in chunks_data]
            scope_text = "\n".join(scope_nodes.get(str(node_id), "") for node_id in docs[doc_id].get("scope_node_ids", []))
            query_val, doc_val = _render(tok, query_str, str(docs[doc_id].get("document_label", doc_id)), evidence, max_length=768, scope_text=scope_text)
            rows.append({"doc_id": doc_id, "query": query_val, "document": doc_val})
        new_capsules[qid] = rows

print(f"Capsules rendered in {time.time()-t0:.2f}s! Now scoring on GPU...")

# 5. Fast Batch Scoring on GPU
batch_size = 32
new_qwen_scores = {}
all_pairs = []
pair_indices = []

for qid in qids_list:
    rows = new_capsules[qid]
    for r in rows:
        all_pairs.append((r["query"], r["document"][:2500]))
        pair_indices.append((qid, r["doc_id"]))

print(f"Total pairs to score: {len(all_pairs)}. Batch size: {batch_size}")
all_scores = []
with torch.inference_mode():
    for i in range(0, len(all_pairs), batch_size):
        batch = all_pairs[i:i+batch_size]
        inp = tok([p[0] for p in batch], [p[1] for p in batch], padding=True, truncation=True, max_length=768, return_tensors='pt').to('cuda')
        vals = m(**inp).logits.reshape(-1).float().cpu().tolist()
        all_scores.extend(vals)

for (qid, doc_id), s in zip(pair_indices, all_scores):
    if qid not in new_qwen_scores:
        new_qwen_scores[qid] = []
    new_qwen_scores[qid].append({"doc_id": doc_id, "qwen_score": s})

print(f"Scoring complete in {time.time()-t0:.2f}s!")

# 6. Evaluate
cand_map = {str(r['qid']): {str(i['doc_id']): i for i in r['candidates']} for r in read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl')) if str(r['qid']) in qids}
lam_map = {str(r['qid']): r['candidates'] for r in read_jsonl(Path('cache/exp013b_cascade/preranker/oof/oof_predictions.jsonl')) if str(r['qid']) in qids}

plain = {qid: [str(row['doc_id']) for row in lam_map[qid]] for qid in qids}
lambda_m = evaluate_rankings(plain, gold, ks=(5,))
print(f"\nLambdaMART Baseline: {lambda_m}")

best_recall = 0
best_cfg = None
for w_qwen in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
    for w_lam in [0.5, 1.0, 1.5, 2.0]:
        fused = {}
        for qid in qids_list:
            fused[qid] = _fuse(lam_map[qid], new_qwen_scores[qid], cand_map[qid], {"qwen": w_qwen, "lambda": w_lam, "bge": 0.0})
        met = evaluate_rankings(fused, gold, ks=(5,))
        if met["recall@5"] > best_recall:
            best_recall = met["recall@5"]
            best_cfg = (w_qwen, w_lam, met)

print(f"\n=======================================================")
print(f"NEW FUSED RESULT WITH DENSE-SCOPED CAPSULES:")
print(f"Best Config: w_qwen={best_cfg[0]}, w_lam={best_cfg[1]}")
print(f"Metrics: {best_cfg[2]}")
print(f"Recall@5 Gain: +{best_cfg[2]['recall@5'] - lambda_m['recall@5']:.5f}")
print(f"=======================================================")
