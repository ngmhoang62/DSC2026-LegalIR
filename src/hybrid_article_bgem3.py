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
    chunk_texts = [c['chunk_text'] for c in chunks]
    chunk_doc_ids = [c['doc_id'] for c in chunks]
    
    return chunks, chunk_ids, chunk_texts, chunk_doc_ids, doc_to_chunk_ids

def get_or_build_bgem3_chunk_embeddings(model, chunk_texts, cache_emb_path: str, batch_size: int = 32):
    if os.path.exists(cache_emb_path):
        print(f"Loading BGE-M3 chunk embeddings from cache: {cache_emb_path}", flush=True)
        with open(cache_emb_path, 'rb') as f:
            chunk_embeddings = pickle.load(f)
    else:
        print(f"Encoding {len(chunk_texts)} Article Chunks with BGE-M3 (FP16) on local GPU...", flush=True)
        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        os.makedirs(os.path.dirname(cache_emb_path), exist_ok=True)
        with open(cache_emb_path, 'wb') as f:
            pickle.dump(chunk_embeddings, f)
        print(f"Saved BGE-M3 chunk embeddings to {cache_emb_path}", flush=True)

    return chunk_embeddings

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

def rrf_fusion(dense_rank_dict: dict, bm25_rank_dict: dict, top_k: int = 5, rrf_k: int = 60) -> list:
    scores = {}
    for rank, doc_id in enumerate(bm25_rank_dict):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        
    for rank, doc_id in enumerate(dense_rank_dict):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:top_k]]

def eval_hybrid_bgem3_chunked(model, train_path: str, folds_path: str, chunk_doc_ids: list, chunk_embeddings: np.ndarray, doc_ids: list, bm25: BM25Okapi, top_k: int = 5):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(folds_path, 'r', encoding='utf-8') as f:
        folds = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    chunk_doc_ids_arr = np.array(chunk_doc_ids)
    all_fold_metrics = []

    print("\n--- Evaluating Hybrid BM25 + BGE-M3 (Article-Level Chunking FP16) on 5-Fold Local CV ---", flush=True)
    for fold_name, val_qids in folds.items():
        y_true = {qid: train_data[qid]['answer'] for qid in val_qids}
        y_pred = {}

        for qid in val_qids:
            query_text = train_data[qid]['question']
            
            # 1. BM25 Retrieval
            query_tokens = word_tokenize(query_text.lower())
            bm25_scores = bm25.get_scores(query_tokens)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:100]
            bm25_top_doc_ids = doc_ids_arr[bm25_top_indices].tolist()

            # 2. Dense Chunk Retrieval
            q_emb = model.encode(query_text, normalize_embeddings=True)
            chunk_sims = np.dot(chunk_embeddings, q_emb)
            
            top_chunk_indices = np.argsort(chunk_sims)[::-1][:500]
            doc_max_sims = {}
            for idx in top_chunk_indices:
                d_id = chunk_doc_ids_arr[idx]
                sim = chunk_sims[idx]
                if d_id not in doc_max_sims or sim > doc_max_sims[d_id]:
                    doc_max_sims[d_id] = sim

            dense_sorted_doc_ids = [d for d, _ in sorted(doc_max_sims.items(), key=lambda x: x[1], reverse=True)[:100]]

            # 3. RRF Fusion
            fused_doc_ids = rrf_fusion(dense_sorted_doc_ids, bm25_top_doc_ids, top_k=top_k)
            y_pred[qid] = {"answer": fused_doc_ids}

        res = eval_retrieval(y_pred, y_true)
        all_fold_metrics.append(res)
        print(f"  {fold_name}: Recall@{top_k} = {res['recall']:.4f} | Precision@{top_k} = {res['precision']:.4f}", flush=True)

    mean_recall = float(np.mean([m['recall'] for m in all_fold_metrics]))
    mean_precision = float(np.mean([m['precision'] for m in all_fold_metrics]))
    print(f"\n>>> EXP-008 (BGE-M3 Article Chunks FP16) 5-Fold CV Mean Recall@{top_k}: {mean_recall:.4f} | Mean Precision@{top_k}: {mean_precision:.4f} <<<\n", flush=True)
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

    print("Generating Public Test predictions for Hybrid BM25 + BGE-M3 (Article Chunks FP16)...", flush=True)
    for qid, val in tqdm(public_data.items()):
        query_text = val['question']
        
        # 1. BM25 Retrieval
        query_tokens = word_tokenize(query_text.lower())
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:100]
        bm25_top_doc_ids = doc_ids_arr[bm25_top_indices].tolist()

        # 2. Dense Chunk Retrieval
        q_emb = model.encode(query_text, normalize_embeddings=True)
        chunk_sims = np.dot(chunk_embeddings, q_emb)
        top_chunk_indices = np.argsort(chunk_sims)[::-1][:500]
        doc_max_sims = {}
        for idx in top_chunk_indices:
            d_id = chunk_doc_ids_arr[idx]
            sim = chunk_sims[idx]
            if d_id not in doc_max_sims or sim > doc_max_sims[d_id]:
                doc_max_sims[d_id] = sim

        dense_sorted_doc_ids = [d for d, _ in sorted(doc_max_sims.items(), key=lambda x: x[1], reverse=True)[:100]]

        # 3. RRF Fusion
        fused_doc_ids = rrf_fusion(dense_sorted_doc_ids, bm25_top_doc_ids, top_k=top_k)
        submission[qid] = {"answer": fused_doc_ids}

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname="submission.json")

    metrics_content = f"""Experiment: EXP-008 (Hybrid BM25 + BGE-M3 on Article-Level Chunks FP16)
Category: hybrid/bgem3_article_chunks
Description: Hybrid BM25 (underthesea) + BAAI/bge-m3 FP16 dense retrieval on 276,230 Article-level chunks.
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
    res_dir = os.path.join(base_dir, "results/hybrid/bgem3_article_chunks")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)

    print("Loading BGE-M3 model in FP16 half precision for local GPU acceleration...", flush=True)
    model = SentenceTransformer('BAAI/bge-m3', device=device).half()
    model.max_seq_length = 512

    chunks, chunk_ids, chunk_texts, chunk_doc_ids, doc_to_chunk_ids = load_article_chunks_corpus(cache_chunks)
    chunk_embeddings = get_or_build_bgem3_chunk_embeddings(model, chunk_texts, cache_bgem3_emb, batch_size=32)

    doc_ids, bm25 = get_bm25_index(selected_contexts_dir, cache_bm25)

    mean_recall, mean_precision = eval_hybrid_bgem3_chunked(model, train_file, folds_file, chunk_doc_ids, chunk_embeddings, doc_ids, bm25, top_k=5)
    generate_public_submission(model, public_file, chunk_doc_ids, chunk_embeddings, doc_ids, bm25, res_dir, mean_recall, mean_precision, top_k=5)
