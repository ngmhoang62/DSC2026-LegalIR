import json
import glob
import os
import pickle
import zipfile
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from evaluator import eval_retrieval

def load_article_chunks_v2_corpus(cache_chunks_path: str):
    print(f"Loading Parent-Aware Article Chunks v2 from {cache_chunks_path}...", flush=True)
    with open(cache_chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = data['chunks']
    doc_to_chunk_ids = data['doc_to_chunk_ids']
    
    chunk_ids = [c['chunk_id'] for c in chunks]
    chunk_doc_ids = [c['doc_id'] for c in chunks]
    
    return chunks, chunk_ids, chunk_doc_ids, doc_to_chunk_ids

def load_cached_v2_embeddings_gpu(bgem3_cache_path: str, e5_cache_path: str, device: str = 'cuda'):
    if not os.path.exists(bgem3_cache_path):
        raise FileNotFoundError(f"Missing BGE-M3 v2 embeddings cache: {bgem3_cache_path}")
    if not os.path.exists(e5_cache_path):
        raise FileNotFoundError(f"Missing E5-Large v2 embeddings cache: {e5_cache_path}")

    print(f"Loading cached BGE-M3 v2 chunk embeddings from {bgem3_cache_path}...", flush=True)
    with open(bgem3_cache_path, 'rb') as f:
        bgem3_np = pickle.load(f)
    bgem3_gpu = torch.from_numpy(bgem3_np).to(device=device, dtype=torch.float16)

    print(f"Loading cached E5-Large v2 chunk embeddings from {e5_cache_path}...", flush=True)
    with open(e5_cache_path, 'rb') as f:
        e5_np = pickle.load(f)
    e5_gpu = torch.from_numpy(e5_np).to(device=device, dtype=torch.float16)

    return bgem3_gpu, e5_gpu

def get_bm25_index(selected_contexts_dir: str, cache_bm25_path: str):
    filepaths = glob.glob(os.path.join(selected_contexts_dir, "context_*.json"))
    doc_ids = []
    corpus_texts = []
    
    for fp in filepaths:
        with open(fp, 'r', encoding='utf-8') as f:
            doc = json.load(f)
            doc_id = str(doc['id'])
            name = doc.get('name', '')
            passage = doc.get('passage', '')
            full_text = f"{name} {passage}".strip()
            doc_ids.append(doc_id)
            corpus_texts.append(full_text)

    if os.path.exists(cache_bm25_path):
        print(f"Loading BM25 underthesea tokenized corpus from {cache_bm25_path}...", flush=True)
        with open(cache_bm25_path, 'rb') as f:
            tokenized_corpus = pickle.load(f)
    else:
        print("Tokenizing full corpus with underthesea for BM25...", flush=True)
        tokenized_corpus = [word_tokenize(text.lower()) for text in tqdm(corpus_texts)]
        os.makedirs(os.path.dirname(cache_bm25_path), exist_ok=True)
        with open(cache_bm25_path, 'wb') as f:
            pickle.dump(tokenized_corpus, f)

    bm25 = BM25Okapi(tokenized_corpus)
    return doc_ids, bm25

def compute_parent_consensus_scores(top_chunk_indices, sims, chunk_doc_ids_arr, 
                                     alpha: float = 0.3, beta: float = 0.05) -> list:
    doc_sims_map = {}
    for idx in top_chunk_indices:
        d_id = chunk_doc_ids_arr[idx]
        sim = float(sims[idx])
        if d_id not in doc_sims_map:
            doc_sims_map[d_id] = []
        doc_sims_map[d_id].append(sim)

    doc_scores = {}
    for d_id, sim_list in doc_sims_map.items():
        sim_list.sort(reverse=True)
        s_max = sim_list[0]
        s_top3_mean = float(np.mean(sim_list[:3]))
        n_hits = len(sim_list)
        
        # Structural consensus score
        consensus_score = s_max + alpha * s_top3_mean + beta * np.log1p(n_hits)
        doc_scores[d_id] = consensus_score

    sorted_docs = [d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)]
    return sorted_docs

def rrf_3way_fusion(bm25_rank_list: list, bgem3_rank_list: list, e5_rank_list: list, 
                    w_bm25: float = 1.0, w_bgem3: float = 1.0, w_e5: float = 1.0,
                    top_k: int = 5, rrf_k: int = 60) -> list:
    scores = {}

    for rank, doc_id in enumerate(bm25_rank_list):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_bm25 / (rrf_k + rank + 1)

    for rank, doc_id in enumerate(bgem3_rank_list):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_bgem3 / (rrf_k + rank + 1)

    for rank, doc_id in enumerate(e5_rank_list):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_e5 / (rrf_k + rank + 1)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:top_k]]

def eval_parent_child_v2_consensus(bgem3_model, e5_model, train_path: str, folds_path: str, 
                                    chunk_doc_ids: list, bgem3_gpu: torch.Tensor, e5_gpu: torch.Tensor, 
                                    doc_ids: list, bm25: BM25Okapi, top_k: int = 5,
                                    alpha: float = 0.3, beta: float = 0.05):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(folds_path, 'r', encoding='utf-8') as f:
        folds = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    chunk_doc_ids_arr = np.array(chunk_doc_ids)
    all_fold_metrics = []

    print("\n--- Evaluating EXP-011c (Parent-Aware Chunks v2 + Consensus Retrieval) on 5-Fold Local CV ---", flush=True)
    for fold_name, val_qids in folds.items():
        y_true = {qid: train_data[qid]['answer'] for qid in val_qids}
        y_pred = {}

        for qid in tqdm(val_qids, desc=f"Evaluating {fold_name}"):
            query_text = train_data[qid]['question']

            # 1. BM25 Doc Ranks
            query_tokens = word_tokenize(query_text.lower())
            bm25_scores = bm25.get_scores(query_tokens)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:100]
            bm25_top_doc_ids = doc_ids_arr[bm25_top_indices].tolist()

            # 2. BGE-M3 Dense Chunk Parent-Child Consensus Ranks (GPU Accelerated)
            q_bgem3_np = bgem3_model.encode(query_text, normalize_embeddings=True)
            q_bgem3_torch = torch.from_numpy(q_bgem3_np).to(device=bgem3_gpu.device, dtype=torch.float16)
            bgem3_sims_gpu = torch.matmul(bgem3_gpu, q_bgem3_torch)
            bgem3_top_k_res = torch.topk(bgem3_sims_gpu, k=500)
            bgem3_top_chunk_indices = bgem3_top_k_res.indices.cpu().numpy()
            bgem3_sims = bgem3_sims_gpu.cpu().numpy()
            
            bgem3_top_doc_ids = compute_parent_consensus_scores(
                bgem3_top_chunk_indices, bgem3_sims, chunk_doc_ids_arr, alpha=alpha, beta=beta
            )[:100]

            # 3. E5-Large Dense Chunk Parent-Child Consensus Ranks (GPU Accelerated)
            e5_query = f"query: {query_text}"
            q_e5_np = e5_model.encode(e5_query, normalize_embeddings=True)
            q_e5_torch = torch.from_numpy(q_e5_np).to(device=e5_gpu.device, dtype=torch.float16)
            e5_sims_gpu = torch.matmul(e5_gpu, q_e5_torch)
            e5_top_k_res = torch.topk(e5_sims_gpu, k=500)
            e5_top_chunk_indices = e5_top_k_res.indices.cpu().numpy()
            e5_sims = e5_sims_gpu.cpu().numpy()
            
            e5_top_doc_ids = compute_parent_consensus_scores(
                e5_top_chunk_indices, e5_sims, chunk_doc_ids_arr, alpha=alpha, beta=beta
            )[:100]

            # 4. 3-Way RRF Fusion on Parent Consensus Ranks
            fused_doc_ids = rrf_3way_fusion(bm25_top_doc_ids, bgem3_top_doc_ids, e5_top_doc_ids, top_k=top_k)
            y_pred[qid] = {"answer": fused_doc_ids}

        res = eval_retrieval(y_pred, y_true)
        all_fold_metrics.append(res)
        print(f"  {fold_name}: Recall@{top_k} = {res['recall']:.4f} | Precision@{top_k} = {res['precision']:.4f}", flush=True)

    mean_recall = float(np.mean([m['recall'] for m in all_fold_metrics]))
    mean_precision = float(np.mean([m['precision'] for m in all_fold_metrics]))
    print(f"\n>>> EXP-011c (Parent-Aware Chunks v2 + Consensus) 5-Fold CV Mean Recall@{top_k}: {mean_recall:.4f} | Mean Precision@{top_k}: {mean_precision:.4f} <<<\n", flush=True)
    return mean_recall, mean_precision

def generate_public_submission(bgem3_model, e5_model, public_path: str, 
                               chunk_doc_ids: list, bgem3_gpu: torch.Tensor, e5_gpu: torch.Tensor, 
                               doc_ids: list, bm25: BM25Okapi, res_dir: str, 
                               mean_recall: float, mean_precision: float, top_k: int = 5,
                               alpha: float = 0.3, beta: float = 0.05):
    os.makedirs(res_dir, exist_ok=True)
    out_json = os.path.join(res_dir, "submission.json")
    out_zip = os.path.join(res_dir, "submission.zip")
    out_txt = os.path.join(res_dir, "metrics.txt")

    with open(public_path, 'r', encoding='utf-8') as f:
        public_data = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    chunk_doc_ids_arr = np.array(chunk_doc_ids)
    submission = {}

    print("Generating Public Test predictions for EXP-011c (Parent-Aware Chunks v2 + Consensus)...", flush=True)
    for qid, val in tqdm(public_data.items(), desc="Public Test Predictions"):
        query_text = val['question']

        # 1. BM25 Ranks
        query_tokens = word_tokenize(query_text.lower())
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:100]
        bm25_top_doc_ids = doc_ids_arr[bm25_top_indices].tolist()

        # 2. BGE-M3 Dense Chunk Parent Consensus Ranks (GPU Accelerated)
        q_bgem3_np = bgem3_model.encode(query_text, normalize_embeddings=True)
        q_bgem3_torch = torch.from_numpy(q_bgem3_np).to(device=bgem3_gpu.device, dtype=torch.float16)
        bgem3_sims_gpu = torch.matmul(bgem3_gpu, q_bgem3_torch)
        bgem3_top_k_res = torch.topk(bgem3_sims_gpu, k=500)
        bgem3_top_chunk_indices = bgem3_top_k_res.indices.cpu().numpy()
        bgem3_sims = bgem3_sims_gpu.cpu().numpy()
        
        bgem3_top_doc_ids = compute_parent_consensus_scores(
            bgem3_top_chunk_indices, bgem3_sims, chunk_doc_ids_arr, alpha=alpha, beta=beta
        )[:100]

        # 3. E5-Large Dense Chunk Parent Consensus Ranks (GPU Accelerated)
        e5_query = f"query: {query_text}"
        q_e5_np = e5_model.encode(e5_query, normalize_embeddings=True)
        q_e5_torch = torch.from_numpy(q_e5_np).to(device=e5_gpu.device, dtype=torch.float16)
        e5_sims_gpu = torch.matmul(e5_gpu, q_e5_torch)
        e5_top_k_res = torch.topk(e5_sims_gpu, k=500)
        e5_top_chunk_indices = e5_top_k_res.indices.cpu().numpy()
        e5_sims = e5_sims_gpu.cpu().numpy()
        
        e5_top_doc_ids = compute_parent_consensus_scores(
            e5_top_chunk_indices, e5_sims, chunk_doc_ids_arr, alpha=alpha, beta=beta
        )[:100]

        # 4. 3-Way RRF Fusion
        fused_doc_ids = rrf_3way_fusion(bm25_top_doc_ids, bgem3_top_doc_ids, e5_top_doc_ids, top_k=top_k)
        submission[qid] = {"answer": fused_doc_ids}

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname="submission.json")

    metrics_content = f"""Experiment: EXP-011c (Parent-Aware Chunks v2 + Consensus Retrieval)
Category: hierarchical/parent_child_v2_consensus
Description: Parent-Aware Article Chunks v2 enriched with Chapter and Scope (Điều 1) + Parent Document Consensus Scoring (MaxP + Top3-Mean + Log(Hits) Consensus).
Total Model Parameters: 0B (BM25) + 560M (BGE-M3) + 560M (E5-Large) = 1.12B (<4B constraint).
Hyperparameters: alpha={alpha}, beta={beta}
-------------------------------------------------
Local 5-Fold CV Mean Recall@5    : {mean_recall:.4f}
Local 5-Fold CV Mean Precision@5 : {mean_precision:.4f}
-------------------------------------------------
"""
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(metrics_content)

    print(f"Saved submission JSON: {out_json}", flush=True)
    print(f"Saved submission ZIP: {out_zip}", flush=True)
    print(f"Saved metrics TXT: {out_txt}", flush=True)

if __name__ == "__main__":
    base_dir = "d:/Study/DSC2026/LegalIR"
    selected_contexts_dir = os.path.join(base_dir, "public_test_dataset/selected-contexts")
    train_file = os.path.join(base_dir, "public_test_dataset/train.json")
    public_file = os.path.join(base_dir, "public_test_dataset/public-official.json")
    folds_file = os.path.join(base_dir, "cache/cv_folds.json")
    cache_chunks = os.path.join(base_dir, "cache/article_chunks_v2.json")
    cache_bm25 = os.path.join(base_dir, "cache/corpus_bm25_underthesea_tokens.pkl")
    cache_bgem3_emb = os.path.join(base_dir, "cache/bgem3_article_chunk_v2_embeddings.pkl")
    cache_e5_emb = os.path.join(base_dir, "cache/e5_article_chunk_v2_embeddings.pkl")
    res_dir = os.path.join(base_dir, "results/hierarchical/parent_child_consensus_scope")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)

    print("Loading BGE-M3 and E5-Large models in FP16...", flush=True)
    bgem3_model = SentenceTransformer('BAAI/bge-m3', device=device).half()
    bgem3_model.max_seq_length = 512

    e5_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device).half()
    e5_model.max_seq_length = 512

    chunks, chunk_ids, chunk_doc_ids, doc_to_chunk_ids = load_article_chunks_v2_corpus(cache_chunks)
    bgem3_gpu, e5_gpu = load_cached_v2_embeddings_gpu(cache_bgem3_emb, cache_e5_emb, device=device)
    doc_ids, bm25 = get_bm25_index(selected_contexts_dir, cache_bm25)

    mean_recall, mean_precision = eval_parent_child_v2_consensus(
        bgem3_model, e5_model, train_file, folds_file, 
        chunk_doc_ids, bgem3_gpu, e5_gpu, 
        doc_ids, bm25, top_k=5, alpha=0.3, beta=0.05
    )
    
    generate_public_submission(
        bgem3_model, e5_model, public_file, 
        chunk_doc_ids, bgem3_gpu, e5_gpu, 
        doc_ids, bm25, res_dir, mean_recall, mean_precision, top_k=5,
        alpha=0.3, beta=0.05
    )
