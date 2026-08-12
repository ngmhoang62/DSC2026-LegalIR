import json
import glob
import os
import pickle
import zipfile
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
    chunk_id_to_index = {c['chunk_id']: i for i, c in enumerate(chunks)}
    
    return chunks, chunk_ids, chunk_doc_ids, doc_to_chunk_ids, chunk_id_to_index

def load_cached_embeddings(bgem3_cache_path: str, e5_cache_path: str, device: str):
    if not os.path.exists(bgem3_cache_path):
        raise FileNotFoundError(f"Missing BGE-M3 embeddings cache: {bgem3_cache_path}")
    if not os.path.exists(e5_cache_path):
        raise FileNotFoundError(f"Missing E5-Large embeddings cache: {e5_cache_path}")

    print(f"Loading BGE-M3 embeddings cache onto GPU ({device})...", flush=True)
    with open(bgem3_cache_path, 'rb') as f:
        bgem3_np = pickle.load(f)
    bgem3_gpu = torch.from_numpy(bgem3_np).half().to(device)
    del bgem3_np

    print(f"Loading E5-Large embeddings cache onto GPU ({device})...", flush=True)
    with open(e5_cache_path, 'rb') as f:
        e5_np = pickle.load(f)
    e5_gpu = torch.from_numpy(e5_np).half().to(device)
    del e5_np

    return bgem3_gpu, e5_gpu

def get_bm25_index(selected_contexts_dir: str, cache_bm25_path: str, doc_to_chunk_ids: dict):
    if os.path.exists(cache_bm25_path):
        print(f"Loading BM25 underthesea tokenized corpus from {cache_bm25_path}...", flush=True)
        with open(cache_bm25_path, 'rb') as f:
            tokenized_corpus = pickle.load(f)
        doc_ids = list(doc_to_chunk_ids.keys())
        bm25 = BM25Okapi(tokenized_corpus)
        return doc_ids, bm25
    
    print("Tokenizing full corpus with underthesea for BM25...", flush=True)
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

    tokenized_corpus = [word_tokenize(text.lower()) for text in tqdm(corpus_texts)]
    os.makedirs(os.path.dirname(cache_bm25_path), exist_ok=True)
    with open(cache_bm25_path, 'wb') as f:
        pickle.dump(tokenized_corpus, f)

    bm25 = BM25Okapi(tokenized_corpus)
    return doc_ids, bm25

@torch.inference_mode()
def batch_predict_reranker_scores(reranker_model, reranker_tokenizer, pairs: list, device: str, batch_size: int = 64) -> np.ndarray:
    all_scores = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking GPU Batches", leave=False):
        batch_pairs = pairs[i:i+batch_size]
        inputs = reranker_tokenizer(
            batch_pairs, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(device)
        
        logits = reranker_model(**inputs).logits
        if logits.shape[-1] == 1:
            scores = logits.view(-1).cpu().numpy()
        else:
            scores = logits[:, 1].cpu().numpy()
        all_scores.extend(scores)
        
    return np.array(all_scores, dtype=np.float32)

@torch.inference_mode()
def evaluate_queries_two_stage_batched(qids: list, qid_to_query: dict, bgem3_model, e5_model, reranker_model, reranker_tokenizer,
                                       chunks: list, chunk_doc_ids_arr: np.ndarray, doc_to_chunk_ids: dict, chunk_id_to_index: dict,
                                       bgem3_gpu: torch.Tensor, e5_gpu: torch.Tensor,
                                       doc_ids_arr: np.ndarray, bm25: BM25Okapi, device: str,
                                       top_n_dense: int = 30, top_n_bm25_docs: int = 15, top_k: int = 5, batch_size: int = 64) -> dict:
    print(f"  Stage 1: Accelerated GPU Vector Search for {len(qids)} queries...", flush=True)
    queries_text = [qid_to_query[qid] for qid in qids]

    # Pre-tokenize queries for BM25
    qid_to_tokens = {qid: word_tokenize(qid_to_query[qid].lower()) for qid in qids}

    # 1. BGE-M3 Dense Search ON GPU (Fast matrix multiplication)
    q_bgem3_vecs = bgem3_model.encode(queries_text, batch_size=64, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).half()
    bgem3_sims_matrix = torch.matmul(q_bgem3_vecs, bgem3_gpu.T).cpu().numpy() # shape: (num_q, num_chunks)

    # 2. E5-Large Dense Search ON GPU (Fast matrix multiplication)
    e5_queries_text = [f"query: {q}" for q in queries_text]
    q_e5_vecs = e5_model.encode(e5_queries_text, batch_size=64, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).half()
    e5_sims_matrix = torch.matmul(q_e5_vecs, e5_gpu.T).cpu().numpy() # shape: (num_q, num_chunks)

    query_cand_meta = []
    all_pairs = []

    for i, qid in enumerate(qids):
        query_text = qid_to_query[qid]
        candidate_indices = set()

        # BGE-M3 Top Chunks
        bgem3_top_idx = np.argsort(bgem3_sims_matrix[i])[::-1][:top_n_dense]
        candidate_indices.update(bgem3_top_idx)

        # E5-Large Top Chunks
        e5_top_idx = np.argsort(e5_sims_matrix[i])[::-1][:top_n_dense]
        candidate_indices.update(e5_top_idx)

        # BM25 Top Docs -> Chunks
        query_tokens = qid_to_tokens[qid]
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_doc_idx = np.argsort(bm25_scores)[::-1][:top_n_bm25_docs]
        bm25_top_doc_ids = doc_ids_arr[bm25_top_doc_idx]

        for d_id in bm25_top_doc_ids:
            c_ids = doc_to_chunk_ids.get(d_id, [])
            for c_id in c_ids[:3]:
                c_idx = chunk_id_to_index.get(c_id)
                if c_idx is not None:
                    candidate_indices.add(c_idx)

        cand_idx_list = list(candidate_indices)
        start_idx = len(all_pairs)
        for idx in cand_idx_list:
            all_pairs.append((query_text, chunks[idx]['chunk_text']))
        
        query_cand_meta.append({
            'qid': qid,
            'cand_chunk_indices': cand_idx_list,
            'start_pair_idx': start_idx,
            'num_pairs': len(cand_idx_list)
        })

    print(f"  Stage 2: GPU Reranking {len(all_pairs)} pairs across {len(qids)} queries...", flush=True)
    all_scores = batch_predict_reranker_scores(reranker_model, reranker_tokenizer, all_pairs, device=device, batch_size=batch_size)

    predictions = {}
    for meta in query_cand_meta:
        qid = meta['qid']
        cand_indices = meta['cand_chunk_indices']
        start_idx = meta['start_pair_idx']
        num_pairs = meta['num_pairs']

        if num_pairs == 0:
            predictions[qid] = {"answer": doc_ids_arr[:top_k].tolist()}
            continue

        q_scores = all_scores[start_idx : start_idx + num_pairs]
        doc_max_score = {}
        for c_idx, score in zip(cand_indices, q_scores):
            d_id = chunk_doc_ids_arr[c_idx]
            if d_id not in doc_max_score or score > doc_max_score[d_id]:
                doc_max_score[d_id] = float(score)

        sorted_docs = sorted(doc_max_score.items(), key=lambda x: x[1], reverse=True)
        predictions[qid] = {"answer": [d_id for d_id, _ in sorted_docs[:top_k]]}

    return predictions

def eval_exp012_cv(bgem3_model, e5_model, reranker_model, reranker_tokenizer, train_path: str, folds_path: str,
                   chunks: list, chunk_doc_ids: list, doc_to_chunk_ids: dict, chunk_id_to_index: dict,
                   bgem3_gpu: torch.Tensor, e5_gpu: torch.Tensor,
                   doc_ids: list, bm25: BM25Okapi, device: str, top_k: int = 5, batch_size: int = 64):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(folds_path, 'r', encoding='utf-8') as f:
        folds = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    chunk_doc_ids_arr = np.array(chunk_doc_ids)
    qid_to_query = {qid: train_data[qid]['question'] for qid in train_data}
    all_fold_metrics = []

    print("\n=== Evaluating EXP-012 (GPU-Accelerated BGE-Reranker-v2-m3) on 5-Fold Local CV ===", flush=True)
    for fold_name, val_qids in folds.items():
        y_true = {qid: train_data[qid]['answer'] for qid in val_qids}
        
        y_pred = evaluate_queries_two_stage_batched(
            val_qids, qid_to_query, bgem3_model, e5_model, reranker_model, reranker_tokenizer,
            chunks, chunk_doc_ids_arr, doc_to_chunk_ids, chunk_id_to_index,
            bgem3_gpu, e5_gpu, doc_ids_arr, bm25, device=device, top_k=top_k, batch_size=batch_size
        )

        res = eval_retrieval(y_pred, y_true)
        all_fold_metrics.append(res)
        print(f"  {fold_name}: Recall@{top_k} = {res['recall']:.4f} | Precision@{top_k} = {res['precision']:.4f}", flush=True)

    mean_recall = float(np.mean([m['recall'] for m in all_fold_metrics]))
    mean_precision = float(np.mean([m['precision'] for m in all_fold_metrics]))
    print(f"\n>>> EXP-012 (Cross-Encoder BGE-Reranker-v2-m3) 5-Fold CV Mean Recall@{top_k}: {mean_recall:.4f} | Mean Precision@{top_k}: {mean_precision:.4f} <<<\n", flush=True)
    return mean_recall, mean_precision

def generate_public_submission(bgem3_model, e5_model, reranker_model, reranker_tokenizer, public_path: str,
                               chunks: list, chunk_doc_ids: list, doc_to_chunk_ids: dict, chunk_id_to_index: dict,
                               bgem3_gpu: torch.Tensor, e5_gpu: torch.Tensor,
                               doc_ids: list, bm25: BM25Okapi, res_dir: str, device: str,
                               mean_recall: float, mean_precision: float, top_k: int = 5, batch_size: int = 64):
    os.makedirs(res_dir, exist_ok=True)
    out_json = os.path.join(res_dir, "submission.json")
    out_zip = os.path.join(res_dir, "submission.zip")
    out_txt = os.path.join(res_dir, "metrics.txt")

    with open(public_path, 'r', encoding='utf-8') as f:
        public_data = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    chunk_doc_ids_arr = np.array(chunk_doc_ids)
    qids = list(public_data.keys())
    qid_to_query = {qid: public_data[qid]['question'] for qid in qids}

    print("Generating Public Test predictions for EXP-012 (GPU-Accelerated Reranker)...", flush=True)
    submission = evaluate_queries_two_stage_batched(
        qids, qid_to_query, bgem3_model, e5_model, reranker_model, reranker_tokenizer,
        chunks, chunk_doc_ids_arr, doc_to_chunk_ids, chunk_id_to_index,
        bgem3_gpu, e5_gpu, doc_ids_arr, bm25, device=device, top_k=top_k, batch_size=batch_size
    )

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname="submission.json")

    metrics_content = f"""Experiment: EXP-012 (Cross-Encoder Stage-2 Reranker with BAAI/bge-reranker-v2-m3)
Category: reranker/bge_reranker_v2_m3
Description: Two-Stage Retrieval combining Stage-1 Candidate Pool (BM25 + BGE-M3 + E5-Large) with Stage-2 BAAI/bge-reranker-v2-m3 Cross-Encoder Reranking and MaxP Aggregation.
Total Model Parameters: 0B (BM25) + 560M (BGE-M3) + 560M (E5-Large) + 560M (BGE-Reranker) = 1.68B (<4B constraint).
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
    res_dir = os.path.join(base_dir, "results/reranker/bge_reranker_v2_m3")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device for query encoding and reranking: {device}", flush=True)

    print("1. Loading First-Stage Bi-Encoders in FP16...", flush=True)
    bgem3_model = SentenceTransformer('BAAI/bge-m3', device=device).half()
    bgem3_model.max_seq_length = 512

    e5_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device).half()
    e5_model.max_seq_length = 512

    print("2. Loading Second-Stage BAAI/bge-reranker-v2-m3 Cross-Encoder in FP16...", flush=True)
    reranker_model_name = "BAAI/bge-reranker-v2-m3"
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        reranker_model_name, 
        torch_dtype=torch.float16
    ).to(device)
    reranker_model.eval()

    print("3. Loading Corpus & Caches...", flush=True)
    chunks, chunk_ids, chunk_doc_ids, doc_to_chunk_ids, chunk_id_to_index = load_article_chunks_corpus(cache_chunks)
    bgem3_gpu, e5_gpu = load_cached_embeddings(cache_bgem3_emb, cache_e5_emb, device=device)
    doc_ids, bm25 = get_bm25_index(selected_contexts_dir, cache_bm25, doc_to_chunk_ids)

    print("4. Running Local 5-Fold CV Evaluation...", flush=True)
    mean_recall, mean_precision = eval_exp012_cv(
        bgem3_model, e5_model, reranker_model, reranker_tokenizer, train_file, folds_file,
        chunks, chunk_doc_ids, doc_to_chunk_ids, chunk_id_to_index,
        bgem3_gpu, e5_gpu, doc_ids, bm25, device=device, top_k=5, batch_size=64
    )

    print("5. Generating Public Submission...", flush=True)
    generate_public_submission(
        bgem3_model, e5_model, reranker_model, reranker_tokenizer, public_file,
        chunks, chunk_doc_ids, doc_to_chunk_ids, chunk_id_to_index,
        bgem3_gpu, e5_gpu, doc_ids, bm25, res_dir, device=device,
        mean_recall=mean_recall, mean_precision=mean_precision, top_k=5, batch_size=64
    )
