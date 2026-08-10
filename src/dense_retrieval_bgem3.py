import json
import glob
import os
import pickle
import zipfile
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from evaluator import eval_retrieval

def load_corpus(selected_contexts_dir: str):
    filepaths = glob.glob(os.path.join(selected_contexts_dir, "context_*.json"))
    doc_ids = []
    corpus_texts = []
    doc_map = {}

    print(f"Loading {len(filepaths)} document contexts...", flush=True)
    for fp in filepaths:
        with open(fp, 'r', encoding='utf-8') as f:
            doc = json.load(f)
            doc_id = str(doc['id'])
            name = doc.get('name', '')
            passage = doc.get('passage', '')
            full_text = f"{name} {passage}".strip()
            
            doc_ids.append(doc_id)
            corpus_texts.append(full_text)
            doc_map[doc_id] = full_text

    return doc_ids, corpus_texts, doc_map

def get_or_build_dense_embeddings(model, corpus_texts, cache_path: str, batch_size: int = 64):
    if os.path.exists(cache_path):
        print(f"Loading BGE-M3 dense embeddings from cache: {cache_path}", flush=True)
        with open(cache_path, 'rb') as f:
            doc_embeddings = pickle.load(f)
    else:
        print("Encoding corpus documents with BAAI/bge-m3 on GPU (max_seq_length=512)...", flush=True)
        doc_embeddings = model.encode(
            corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(doc_embeddings, f)
        print(f"Saved BGE-M3 dense embeddings to cache: {cache_path}", flush=True)

    return doc_embeddings

def reciprocal_rank_fusion(bm25_scores, dense_scores, k=60, top_k_candidates=50):
    bm25_ranks = np.argsort(bm25_scores)[::-1][:top_k_candidates]
    dense_ranks = np.argsort(dense_scores)[::-1][:top_k_candidates]

    rrf_score_dict = {}
    for rank, idx in enumerate(bm25_ranks):
        rrf_score_dict[idx] = rrf_score_dict.get(idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, idx in enumerate(dense_ranks):
        rrf_score_dict[idx] = rrf_score_dict.get(idx, 0.0) + 1.0 / (k + rank + 1)

    sorted_candidates = sorted(rrf_score_dict.items(), key=lambda item: item[1], reverse=True)
    return [idx for idx, score in sorted_candidates]

def run_evaluations(train_path: str, folds_path: str, doc_ids: list, bm25: BM25Okapi, doc_embeddings: np.ndarray, model: SentenceTransformer, top_k: int = 5):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(folds_path, 'r', encoding='utf-8') as f:
        folds = json.load(f)

    doc_ids_arr = np.array(doc_ids)

    all_val_qids = []
    for val_qids in folds.values():
        all_val_qids.extend(val_qids)

    print("Encoding validation questions with BGE-M3...", flush=True)
    val_questions = [train_data[qid]['question'] for qid in all_val_qids]
    q_embeddings_all = model.encode(
        val_questions,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    qid_to_emb = {qid: q_embeddings_all[idx] for idx, qid in enumerate(all_val_qids)}

    # 1. Pure Dense BGE-M3 Evaluation
    print("\n--- Evaluating Pure Dense Retrieval (BAAI/bge-m3) on 5-Fold Local CV ---", flush=True)
    dense_fold_metrics = []
    for fold_name, val_qids in folds.items():
        y_true = {qid: train_data[qid]['answer'] for qid in val_qids}
        y_pred = {}

        q_embs = np.array([qid_to_emb[qid] for qid in val_qids])
        sim_matrix = q_embs @ doc_embeddings.T

        for idx, qid in enumerate(val_qids):
            scores = sim_matrix[idx]
            top_indices = np.argsort(scores)[::-1][:top_k]
            y_pred[qid] = {"answer": doc_ids_arr[top_indices].tolist()}

        res = eval_retrieval(y_pred, y_true)
        dense_fold_metrics.append(res)
        print(f"  {fold_name}: Recall@{top_k} = {res['recall']:.4f} | Precision@{top_k} = {res['precision']:.4f}", flush=True)

    dense_mean_recall = float(np.mean([m['recall'] for m in dense_fold_metrics]))
    dense_mean_precision = float(np.mean([m['precision'] for m in dense_fold_metrics]))
    print(f">>> Dense BGE-M3 5-Fold Mean Recall@{top_k}: {dense_mean_recall:.4f} | Mean Precision@{top_k}: {dense_mean_precision:.4f} <<<\n", flush=True)

    # 2. Hybrid (BM25 + BGE-M3 + RRF) Evaluation
    print("\n--- Evaluating Hybrid Retrieval (BM25 + BGE-M3 + RRF k=60) on 5-Fold Local CV ---", flush=True)
    hybrid_fold_metrics = []
    for fold_name, val_qids in folds.items():
        y_true = {qid: train_data[qid]['answer'] for qid in val_qids}
        y_pred = {}

        q_embs = np.array([qid_to_emb[qid] for qid in val_qids])
        sim_matrix = q_embs @ doc_embeddings.T

        for idx, qid in enumerate(val_qids):
            query_text = train_data[qid]['question']
            query_tokens = word_tokenize(query_text.lower(), format="text").split()
            bm25_scores = bm25.get_scores(query_tokens)
            dense_scores = sim_matrix[idx]

            hybrid_indices = reciprocal_rank_fusion(bm25_scores, dense_scores, k=60, top_k_candidates=50)[:top_k]
            y_pred[qid] = {"answer": doc_ids_arr[hybrid_indices].tolist()}

        res = eval_retrieval(y_pred, y_true)
        hybrid_fold_metrics.append(res)
        print(f"  {fold_name}: Recall@{top_k} = {res['recall']:.4f} | Precision@{top_k} = {res['precision']:.4f}", flush=True)

    hybrid_mean_recall = float(np.mean([m['recall'] for m in hybrid_fold_metrics]))
    hybrid_mean_precision = float(np.mean([m['precision'] for m in hybrid_fold_metrics]))
    print(f">>> Hybrid (BM25 + BGE-M3) 5-Fold Mean Recall@{top_k}: {hybrid_mean_recall:.4f} | Mean Precision@{top_k}: {hybrid_mean_precision:.4f} <<<\n", flush=True)

    return (dense_mean_recall, dense_mean_precision), (hybrid_mean_recall, hybrid_mean_precision)

def save_submissions(public_path: str, doc_ids: list, bm25: BM25Okapi, doc_embeddings: np.ndarray, model: SentenceTransformer, dense_res_dir: str, hybrid_res_dir: str, dense_metrics: tuple, hybrid_metrics: tuple, top_k: int = 5):
    with open(public_path, 'r', encoding='utf-8') as f:
        public_data = json.load(f)

    q_ids = list(public_data.keys())
    public_questions = [public_data[qid]['question'] for qid in q_ids]

    print("Encoding Public Test questions with BGE-M3...", flush=True)
    q_embeddings = model.encode(
        public_questions,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    dense_sim_matrix = q_embeddings @ doc_embeddings.T
    doc_ids_arr = np.array(doc_ids)

    # 1. Pure Dense BGE-M3 Submission
    os.makedirs(dense_res_dir, exist_ok=True)
    dense_sub = {}
    for idx, qid in enumerate(q_ids):
        scores = dense_sim_matrix[idx]
        top_indices = np.argsort(scores)[::-1][:top_k]
        dense_sub[qid] = {"answer": doc_ids_arr[top_indices].tolist()}

    with open(os.path.join(dense_res_dir, "submission.json"), 'w', encoding='utf-8') as f:
        json.dump(dense_sub, f, ensure_ascii=False, indent=4)
    with zipfile.ZipFile(os.path.join(dense_res_dir, "submission.zip"), 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(dense_res_dir, "submission.json"), arcname="submission.json")

    with open(os.path.join(dense_res_dir, "metrics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"""Experiment: Dense Retrieval (bge-m3)
Category: dense/bge-m3
Description: Dense Embedding Retrieval using BAAI/bge-m3.
-------------------------------------------------
Local 5-Fold CV Mean Recall@5    : {dense_metrics[0]:.4f}
Local 5-Fold CV Mean Precision@5 : {dense_metrics[1]:.4f}
-------------------------------------------------
Public Test Recall@5            : 
Public Test Precision@5         : 
""")

    # 2. Hybrid (BM25 + BGE-M3 + RRF) Submission
    os.makedirs(hybrid_res_dir, exist_ok=True)
    hybrid_sub = {}
    for idx, qid in enumerate(q_ids):
        query_text = public_data[qid]['question']
        query_tokens = word_tokenize(query_text.lower(), format="text").split()
        bm25_scores = bm25.get_scores(query_tokens)
        dense_scores = dense_sim_matrix[idx]

        hybrid_indices = reciprocal_rank_fusion(bm25_scores, dense_scores, k=60, top_k_candidates=50)[:top_k]
        hybrid_sub[qid] = {"answer": doc_ids_arr[hybrid_indices].tolist()}

    with open(os.path.join(hybrid_res_dir, "submission.json"), 'w', encoding='utf-8') as f:
        json.dump(hybrid_sub, f, ensure_ascii=False, indent=4)
    with zipfile.ZipFile(os.path.join(hybrid_res_dir, "submission.zip"), 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(hybrid_res_dir, "submission.json"), arcname="submission.json")

    with open(os.path.join(hybrid_res_dir, "metrics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"""Experiment: Hybrid Retrieval (BM25 + BGE-M3 + RRF k=60)
Category: hybrid/bm25-bge-m3
Description: Reciprocal Rank Fusion of BM25 Lexical + BAAI/bge-m3 Dense embeddings.
-------------------------------------------------
Local 5-Fold CV Mean Recall@5    : {hybrid_metrics[0]:.4f}
Local 5-Fold CV Mean Precision@5 : {hybrid_metrics[1]:.4f}
-------------------------------------------------
Public Test Recall@5            : 
Public Test Precision@5         : 
""")

    print(f"Saved Dense BGE-M3 results to: {dense_res_dir}", flush=True)
    print(f"Saved Hybrid BM25+BGE-M3 results to: {hybrid_res_dir}", flush=True)

if __name__ == "__main__":
    base_dir = "d:/Study/DSC2026/LegalIR"
    selected_contexts_dir = os.path.join(base_dir, "public_test_dataset/selected-contexts")
    train_file = os.path.join(base_dir, "public_test_dataset/train.json")
    public_file = os.path.join(base_dir, "public_test_dataset/public-official.json")
    folds_file = os.path.join(base_dir, "cache/cv_folds.json")
    bm25_cache_file = os.path.join(base_dir, "cache/corpus_bm25_tokens.pkl")
    dense_cache_file = os.path.join(base_dir, "cache/corpus_dense_bgem3_embeddings.pkl")

    dense_res_dir = os.path.join(base_dir, "results/dense/bge-m3")
    hybrid_res_dir = os.path.join(base_dir, "results/hybrid/bm25-bge-m3")

    model_name = "BAAI/bge-m3"
    print(f"Loading model: {model_name}...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    model = SentenceTransformer(model_name, device=device)
    
    # Crucial speed fix: Cap max sequence length to 512
    model.max_seq_length = 512
    print(f"Set model max_seq_length = {model.max_seq_length}", flush=True)

    # Load BM25
    with open(bm25_cache_file, 'rb') as f:
        tokenized_corpus = pickle.load(f)
    bm25 = BM25Okapi(tokenized_corpus)

    doc_ids, corpus_texts, doc_map = load_corpus(selected_contexts_dir)
    doc_embeddings = get_or_build_dense_embeddings(model, corpus_texts, dense_cache_file, batch_size=64)

    dense_metrics, hybrid_metrics = run_evaluations(train_file, folds_file, doc_ids, bm25, doc_embeddings, model, top_k=5)
    save_submissions(public_file, doc_ids, bm25, doc_embeddings, model, dense_res_dir, hybrid_res_dir, dense_metrics, hybrid_metrics, top_k=5)
