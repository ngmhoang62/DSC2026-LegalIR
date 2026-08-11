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

def load_article_chunks_corpus(cache_chunks_path: str):
    print(f"Loading Article-Level Chunks from {cache_chunks_path}...", flush=True)
    with open(cache_chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = data['chunks']
    doc_to_chunk_ids = data['doc_to_chunk_ids']
    
    chunk_ids = [c['chunk_id'] for c in chunks]
    chunk_doc_ids = [c['doc_id'] for c in chunks]
    
    return chunks, chunk_ids, chunk_doc_ids, doc_to_chunk_ids

def load_bgem3_embeddings(bgem3_cache_path: str):
    if not os.path.exists(bgem3_cache_path):
        raise FileNotFoundError(f"Missing BGE-M3 embeddings cache: {bgem3_cache_path}. Run EXP-008 first!")

    print(f"Loading cached BGE-M3 chunk embeddings from {bgem3_cache_path}...", flush=True)
    with open(bgem3_cache_path, 'rb') as f:
        bgem3_embeddings = pickle.load(f)
    return bgem3_embeddings

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

def hierarchical_rrf_fusion(coarse_doc_ids: list, fine_chunk_sorted_doc_ids: list, top_k: int = 5, rrf_k: int = 60) -> list:
    scores = {}
    for rank, doc_id in enumerate(coarse_doc_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        
    for rank, doc_id in enumerate(fine_chunk_sorted_doc_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:top_k]]

def eval_hierarchical_retrieval(model, train_path: str, folds_path: str, 
                                chunk_doc_ids: list, chunk_embeddings: np.ndarray, 
                                doc_ids: list, bm25: BM25Okapi, top_k: int = 5):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(folds_path, 'r', encoding='utf-8') as f:
        folds = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    chunk_doc_ids_arr = np.array(chunk_doc_ids)
    all_fold_metrics = []

    print("\n--- Evaluating EXP-011 (Hierarchical Coarse-to-Fine Retrieval BGE-M3) on 5-Fold Local CV ---", flush=True)
    for fold_name, val_qids in folds.items():
        y_true = {qid: train_data[qid]['answer'] for qid in val_qids}
        y_pred = {}

        val_queries = [train_data[qid]['question'] for qid in val_qids]
        
        # Batch encode all query embeddings in FP16 on GPU
        print(f"  Encoding {len(val_queries)} queries for {fold_name} in batch on GPU...", flush=True)
        q_embs = model.encode(val_queries, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

        for idx, qid in enumerate(val_qids):
            query_text = val_queries[idx]
            q_emb = q_embs[idx]

            # 1. Coarse Pass: Retrieve Top 150 candidate documents using BM25
            query_tokens = word_tokenize(query_text.lower())
            bm25_scores = bm25.get_scores(query_tokens)
            coarse_top_indices = np.argsort(bm25_scores)[::-1][:150]
            coarse_doc_ids = doc_ids_arr[coarse_top_indices].tolist()
            coarse_doc_set = set(coarse_doc_ids)

            # 2. Fine Pass: Calculate similarity ONLY for chunks of Top 150 candidate documents
            chunk_sims = np.dot(chunk_embeddings, q_emb)
            top_chunk_indices = np.argsort(chunk_sims)[::-1][:500]
            
            doc_max_sims = {}
            for c_idx in top_chunk_indices:
                d_id = chunk_doc_ids_arr[c_idx]
                if d_id in coarse_doc_set:
                    sim = chunk_sims[c_idx]
                    if d_id not in doc_max_sims or sim > doc_max_sims[d_id]:
                        doc_max_sims[d_id] = sim

            fine_sorted_doc_ids = [d for d, _ in sorted(doc_max_sims.items(), key=lambda x: x[1], reverse=True)]

            # 3. Hierarchical RRF Fusion
            fused_doc_ids = hierarchical_rrf_fusion(coarse_doc_ids, fine_sorted_doc_ids, top_k=top_k)
            y_pred[qid] = {"answer": fused_doc_ids}

        res = eval_retrieval(y_pred, y_true)
        all_fold_metrics.append(res)
        print(f"  {fold_name}: Recall@{top_k} = {res['recall']:.4f} | Precision@{top_k} = {res['precision']:.4f}", flush=True)

    mean_recall = float(np.mean([m['recall'] for m in all_fold_metrics]))
    mean_precision = float(np.mean([m['precision'] for m in all_fold_metrics]))
    print(f"\n>>> EXP-011 (Hierarchical Coarse-to-Fine BGE-M3) 5-Fold CV Mean Recall@{top_k}: {mean_recall:.4f} | Mean Precision@{top_k}: {mean_precision:.4f} <<<\n", flush=True)
    return mean_recall, mean_precision

def generate_public_submission(model, public_path: str, chunk_doc_ids: list, chunk_embeddings: np.ndarray, doc_ids: list, bm25: BM25Okapi, res_dir: str, mean_recall: float, mean_precision: float, top_k: int = 5):
    os.makedirs(res_dir, exist_ok=True)
    out_json = os.path.join(res_dir, "submission.json")
    out_zip = os.path.join(res_dir, "submission.zip")
    out_txt = os.path.join(res_dir, "metrics.txt")

    with open(public_path, 'r', encoding='utf-8') as f:
        public_data = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    chunk_doc_ids_arr = np.array(chunk_doc_ids)
    submission = {}

    public_qids = list(public_data.keys())
    public_queries = [public_data[qid]['question'] for qid in public_qids]

    print("Encoding public test queries in batch on GPU...", flush=True)
    q_embs = model.encode(public_queries, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    print("Generating Public Test predictions for EXP-011 (Hierarchical Coarse-to-Fine)...", flush=True)
    for idx, qid in enumerate(tqdm(public_qids)):
        query_text = public_queries[idx]
        q_emb = q_embs[idx]

        # 1. Coarse Pass: Retrieve Top 150 candidate documents using BM25
        query_tokens = word_tokenize(query_text.lower())
        bm25_scores = bm25.get_scores(query_tokens)
        coarse_top_indices = np.argsort(bm25_scores)[::-1][:150]
        coarse_doc_ids = doc_ids_arr[coarse_top_indices].tolist()
        coarse_doc_set = set(coarse_doc_ids)

        # 2. Fine Pass: Calculate similarity ONLY for chunks of Top 150 candidate documents
        chunk_sims = np.dot(chunk_embeddings, q_emb)
        top_chunk_indices = np.argsort(chunk_sims)[::-1][:500]
        
        doc_max_sims = {}
        for c_idx in top_chunk_indices:
            d_id = chunk_doc_ids_arr[c_idx]
            if d_id in coarse_doc_set:
                sim = chunk_sims[c_idx]
                if d_id not in doc_max_sims or sim > doc_max_sims[d_id]:
                    doc_max_sims[d_id] = sim

        fine_sorted_doc_ids = [d for d, _ in sorted(doc_max_sims.items(), key=lambda x: x[1], reverse=True)]

        # 3. Hierarchical RRF Fusion
        fused_doc_ids = hierarchical_rrf_fusion(coarse_doc_ids, fine_sorted_doc_ids, top_k=top_k)
        submission[qid] = {"answer": fused_doc_ids}

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname="submission.json")

    metrics_content = f"""Experiment: EXP-011 (Hierarchical Coarse-to-Fine Retrieval: BM25 + BGE-M3 Article Chunks FP16)
Category: hierarchical/bgem3_coarse_to_fine
Description: 2-Pass Hierarchical Retrieval: Coarse BM25 candidate selection (Top 150) -> Fine Article-Level Chunk re-scoring with BAAI/bge-m3 FP16 -> RRF Fusion.
Total Model Parameters: 0B (BM25) + 560M (BGE-M3) = 560M (<4B constraint).
-------------------------------------------------
Local 5-Fold CV Mean Recall@5    : {mean_recall:.4f}
Local 5-Fold CV Mean Precision@5 : {mean_precision:.4f}
-------------------------------------------------
Public Test Recall@5            : 
Public Test Precision@5         : 
"""
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(metrics_content)

    print(f"Saved submission JSON: {out_json}", flush=True)
    print(f"Saved submission ZIP: {out_zip}", flush=True)
    print(f"Saved metrics TXT: {out_txt}", flush=True)

if __name__ == "__main__":
    base_dir = "d:/Study/DSC2026/LegalIR"
    if not os.path.exists(base_dir):
        base_dir = "/content/drive/MyDrive/DSC2026/LegalIR"

    selected_contexts_dir = os.path.join(base_dir, "public_test_dataset/selected-contexts")
    train_file = os.path.join(base_dir, "public_test_dataset/train.json")
    public_file = os.path.join(base_dir, "public_test_dataset/public-official.json")
    folds_file = os.path.join(base_dir, "cache/cv_folds.json")
    cache_chunks = os.path.join(base_dir, "cache/article_chunks.json")
    cache_bm25 = os.path.join(base_dir, "cache/corpus_bm25_underthesea_tokens.pkl")
    cache_bgem3_emb = os.path.join(base_dir, "cache/bgem3_article_chunk_embeddings.pkl")
    res_dir = os.path.join(base_dir, "results/hierarchical/bgem3_coarse_to_fine")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)

    print("Loading BGE-M3 model in FP16 for query encoding...", flush=True)
    model = SentenceTransformer('BAAI/bge-m3', device=device).half()
    model.max_seq_length = 512

    chunks, chunk_ids, chunk_doc_ids, doc_to_chunk_ids = load_article_chunks_corpus(cache_chunks)
    chunk_embeddings = load_bgem3_embeddings(cache_bgem3_emb)
    doc_ids, bm25 = get_bm25_index(selected_contexts_dir, cache_bm25)

    mean_recall, mean_precision = eval_hierarchical_retrieval(model, train_file, folds_file, chunk_doc_ids, chunk_embeddings, doc_ids, bm25, top_k=5)
    generate_public_submission(model, public_file, chunk_doc_ids, chunk_embeddings, doc_ids, bm25, res_dir, mean_recall, mean_precision, top_k=5)
