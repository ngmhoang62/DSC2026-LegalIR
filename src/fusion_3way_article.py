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

def load_cached_embeddings(bgem3_cache_path: str, e5_cache_path: str):
    if not os.path.exists(bgem3_cache_path):
        raise FileNotFoundError(f"Missing BGE-M3 embeddings cache: {bgem3_cache_path}. Run EXP-008 first!")
    if not os.path.exists(e5_cache_path):
        raise FileNotFoundError(f"Missing E5-Large embeddings cache: {e5_cache_path}. Run EXP-009 first!")

    print(f"Loading cached BGE-M3 chunk embeddings from {bgem3_cache_path}...", flush=True)
    with open(bgem3_cache_path, 'rb') as f:
        bgem3_embeddings = pickle.load(f)

    print(f"Loading cached E5-Large chunk embeddings from {e5_cache_path}...", flush=True)
    with open(e5_cache_path, 'rb') as f:
        e5_embeddings = pickle.load(f)

    return bgem3_embeddings, e5_embeddings

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

def rrf_3way_fusion(bm25_rank_list: list, bgem3_rank_list: list, e5_rank_list: list, 
                    w_bm25: float = 1.0, w_bgem3: float = 1.0, w_e5: float = 1.0,
                    top_k: int = 5, rrf_k: int = 60) -> list:
    """
    Combines 3 retrieval engines (BM25 + BGE-M3 + E5-Large) using weighted Reciprocal Rank Fusion.
    """
    scores = {}

    # 1. BM25 ranks
    for rank, doc_id in enumerate(bm25_rank_list):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_bm25 / (rrf_k + rank + 1)

    # 2. BGE-M3 ranks
    for rank, doc_id in enumerate(bgem3_rank_list):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_bgem3 / (rrf_k + rank + 1)

    # 3. E5-Large ranks
    for rank, doc_id in enumerate(e5_rank_list):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_e5 / (rrf_k + rank + 1)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:top_k]]

def eval_3way_fusion(bgem3_model, e5_model, train_path: str, folds_path: str, 
                     chunk_doc_ids: list, bgem3_embeddings: np.ndarray, e5_embeddings: np.ndarray, 
                     doc_ids: list, bm25: BM25Okapi, top_k: int = 5):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(folds_path, 'r', encoding='utf-8') as f:
        folds = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    chunk_doc_ids_arr = np.array(chunk_doc_ids)
    all_fold_metrics = []

    print("\n--- Evaluating EXP-010 (3-Way Fusion: BM25 + BGE-M3 + E5-Large) on 5-Fold Local CV ---", flush=True)
    for fold_name, val_qids in folds.items():
        y_true = {qid: train_data[qid]['answer'] for qid in val_qids}
        y_pred = {}

        for qid in val_qids:
            query_text = train_data[qid]['question']

            # 1. BM25 Ranks
            query_tokens = word_tokenize(query_text.lower())
            bm25_scores = bm25.get_scores(query_tokens)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:100]
            bm25_top_doc_ids = doc_ids_arr[bm25_top_indices].tolist()

            # 2. BGE-M3 Dense Chunk Ranks
            q_bgem3 = bgem3_model.encode(query_text, normalize_embeddings=True)
            bgem3_sims = np.dot(bgem3_embeddings, q_bgem3)
            bgem3_top_chunk_indices = np.argsort(bgem3_sims)[::-1][:500]
            bgem3_doc_max = {}
            for idx in bgem3_top_chunk_indices:
                d_id = chunk_doc_ids_arr[idx]
                sim = bgem3_sims[idx]
                if d_id not in bgem3_doc_max or sim > bgem3_doc_max[d_id]:
                    bgem3_doc_max[d_id] = sim
            bgem3_top_doc_ids = [d for d, _ in sorted(bgem3_doc_max.items(), key=lambda x: x[1], reverse=True)[:100]]

            # 3. E5-Large Dense Chunk Ranks
            e5_query = f"query: {query_text}"
            q_e5 = e5_model.encode(e5_query, normalize_embeddings=True)
            e5_sims = np.dot(e5_embeddings, q_e5)
            e5_top_chunk_indices = np.argsort(e5_sims)[::-1][:500]
            e5_doc_max = {}
            for idx in e5_top_chunk_indices:
                d_id = chunk_doc_ids_arr[idx]
                sim = e5_sims[idx]
                if d_id not in e5_doc_max or sim > e5_doc_max[d_id]:
                    e5_doc_max[d_id] = sim
            e5_top_doc_ids = [d for d, _ in sorted(e5_doc_max.items(), key=lambda x: x[1], reverse=True)[:100]]

            # 4. 3-Way RRF Fusion
            fused_doc_ids = rrf_3way_fusion(bm25_top_doc_ids, bgem3_top_doc_ids, e5_top_doc_ids, top_k=top_k)
            y_pred[qid] = {"answer": fused_doc_ids}

        res = eval_retrieval(y_pred, y_true)
        all_fold_metrics.append(res)
        print(f"  {fold_name}: Recall@{top_k} = {res['recall']:.4f} | Precision@{top_k} = {res['precision']:.4f}", flush=True)

    mean_recall = float(np.mean([m['recall'] for m in all_fold_metrics]))
    mean_precision = float(np.mean([m['precision'] for m in all_fold_metrics]))
    print(f"\n>>> EXP-010 (3-Way Fusion Article Chunks) 5-Fold CV Mean Recall@{top_k}: {mean_recall:.4f} | Mean Precision@{top_k}: {mean_precision:.4f} <<<\n", flush=True)
    return mean_recall, mean_precision

def generate_public_submission(bgem3_model, e5_model, public_path: str, 
                               chunk_doc_ids: list, bgem3_embeddings: np.ndarray, e5_embeddings: np.ndarray, 
                               doc_ids: list, bm25: BM25Okapi, res_dir: str, 
                               mean_recall: float, mean_precision: float, top_k: int = 5):
    os.makedirs(res_dir, exist_ok=True)
    out_json = os.path.join(res_dir, "submission.json")
    out_zip = os.path.join(res_dir, "submission.zip")
    out_txt = os.path.join(res_dir, "metrics.txt")

    with open(public_path, 'r', encoding='utf-8') as f:
        public_data = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    chunk_doc_ids_arr = np.array(chunk_doc_ids)
    submission = {}

    print("Generating Public Test predictions for EXP-010 (3-Way Fusion)...", flush=True)
    for qid, val in tqdm(public_data.items()):
        query_text = val['question']

        # 1. BM25 Ranks
        query_tokens = word_tokenize(query_text.lower())
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:100]
        bm25_top_doc_ids = doc_ids_arr[bm25_top_indices].tolist()

        # 2. BGE-M3 Dense Chunk Ranks
        q_bgem3 = bgem3_model.encode(query_text, normalize_embeddings=True)
        bgem3_sims = np.dot(bgem3_embeddings, q_bgem3)
        bgem3_top_chunk_indices = np.argsort(bgem3_sims)[::-1][:500]
        bgem3_doc_max = {}
        for idx in bgem3_top_chunk_indices:
            d_id = chunk_doc_ids_arr[idx]
            sim = bgem3_sims[idx]
            if d_id not in bgem3_doc_max or sim > bgem3_doc_max[d_id]:
                bgem3_doc_max[d_id] = sim
        bgem3_top_doc_ids = [d for d, _ in sorted(bgem3_doc_max.items(), key=lambda x: x[1], reverse=True)[:100]]

        # 3. E5-Large Dense Chunk Ranks
        e5_query = f"query: {query_text}"
        q_e5 = e5_model.encode(e5_query, normalize_embeddings=True)
        e5_sims = np.dot(e5_embeddings, q_e5)
        e5_top_chunk_indices = np.argsort(e5_sims)[::-1][:500]
        e5_doc_max = {}
        for idx in e5_top_chunk_indices:
            d_id = chunk_doc_ids_arr[idx]
            sim = e5_sims[idx]
            if d_id not in e5_doc_max or sim > e5_doc_max[d_id]:
                e5_doc_max[d_id] = sim
        e5_top_doc_ids = [d for d, _ in sorted(e5_doc_max.items(), key=lambda x: x[1], reverse=True)[:100]]

        # 4. 3-Way RRF Fusion
        fused_doc_ids = rrf_3way_fusion(bm25_top_doc_ids, bgem3_top_doc_ids, e5_top_doc_ids, top_k=top_k)
        submission[qid] = {"answer": fused_doc_ids}

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname="submission.json")

    metrics_content = f"""Experiment: EXP-010 (3-Way Fusion: BM25 + BGE-M3 + Multilingual-E5-Large on Article Chunks)
Category: ensemble/3way_fusion_article
Description: 3-Way RRF Fusion of BM25 (underthesea) + BAE-M3 + Multilingual-E5-Large on 276,230 Article-level chunks.
Total Model Parameters: 0B (BM25) + 560M (BGE-M3) + 560M (E5-Large) = 1.12B (<4B constraint).
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
    cache_e5_emb = os.path.join(base_dir, "cache/e5_article_chunk_embeddings.pkl")
    res_dir = os.path.join(base_dir, "results/ensemble/3way_fusion_article")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device for query encoding: {device}", flush=True)

    print("Loading BGE-M3 and E5-Large models in FP16 for fast query encoding...", flush=True)
    bgem3_model = SentenceTransformer('BAAI/bge-m3', device=device).half()
    bgem3_model.max_seq_length = 512

    e5_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device).half()
    e5_model.max_seq_length = 512

    chunks, chunk_ids, chunk_doc_ids, doc_to_chunk_ids = load_article_chunks_corpus(cache_chunks)
    bgem3_embeddings, e5_embeddings = load_cached_embeddings(cache_bgem3_emb, cache_e5_emb)
    doc_ids, bm25 = get_bm25_index(selected_contexts_dir, cache_bm25)

    mean_recall, mean_precision = eval_3way_fusion(
        bgem3_model, e5_model, train_file, folds_file, 
        chunk_doc_ids, bgem3_embeddings, e5_embeddings, 
        doc_ids, bm25, top_k=5
    )
    
    generate_public_submission(
        bgem3_model, e5_model, public_file, 
        chunk_doc_ids, bgem3_embeddings, e5_embeddings, 
        doc_ids, bm25, res_dir, mean_recall, mean_precision, top_k=5
    )
