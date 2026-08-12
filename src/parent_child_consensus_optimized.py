import json
import os
import pickle
import zipfile
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from sentence_transformers import SentenceTransformer
from evaluator import eval_retrieval

def rrf_3way(bm25_ranks, bgem3_ranks, e5_ranks, top_k=5, rrf_k=60):
    scores = {}
    for rank, doc_id in enumerate(bm25_ranks):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(bgem3_ranks):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(e5_ranks):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:top_k]]

def compute_parent_consensus_scores(top_chunk_indices, sims_array, chunk_doc_ids_arr, candidate_top_k=100, alpha=0.3, beta=0.02):
    doc_sims_map = {}
    for idx in top_chunk_indices[:candidate_top_k]:
        d_id = chunk_doc_ids_arr[idx]
        sim = float(sims_array[idx])
        if d_id not in doc_sims_map:
            doc_sims_map[d_id] = []
        doc_sims_map[d_id].append(sim)

    doc_scores = {}
    for d_id, sim_list in doc_sims_map.items():
        sim_list.sort(reverse=True)
        s_max = sim_list[0]
        s_top3_mean = float(np.mean(sim_list[:3]))
        n_hits = len(sim_list)
        consensus_score = s_max + alpha * s_top3_mean + beta * np.log1p(n_hits)
        doc_scores[d_id] = consensus_score

    sorted_docs = [d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)]
    return sorted_docs

def get_bm25_index(selected_contexts_dir, cache_bm25_path):
    import glob
    filepaths = glob.glob(os.path.join(selected_contexts_dir, "context_*.json"))
    doc_ids = []
    for fp in filepaths:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
            doc_ids.append(str(data['id']))
    doc_ids_arr = np.array(doc_ids)

    with open(cache_bm25_path, 'rb') as f:
        tokenized_corpus = pickle.load(f)

    bm25 = BM25Okapi(tokenized_corpus)
    return doc_ids_arr, bm25

def load_article_chunks_corpus(chunks_cache_path):
    with open(chunks_cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = data['chunks']
    chunk_ids = np.array([c['chunk_id'] for c in chunks])
    chunk_doc_ids = np.array([c['doc_id'] for c in chunks])
    doc_to_chunk_ids = data['doc_to_chunk_ids']
    return chunks, chunk_ids, chunk_doc_ids, doc_to_chunk_ids

def load_cached_embeddings_gpu(bgem3_pkl_path, e5_pkl_path, device='cuda'):
    print(f"Loading cached BGE-M3 chunk embeddings from {bgem3_pkl_path}...", flush=True)
    with open(bgem3_pkl_path, 'rb') as f:
        bgem3_np = pickle.load(f)
    bgem3_gpu = torch.from_numpy(bgem3_np).to(device=device, dtype=torch.float16)

    print(f"Loading cached E5-Large chunk embeddings from {e5_pkl_path}...", flush=True)
    with open(e5_pkl_path, 'rb') as f:
        e5_np = pickle.load(f)
    e5_gpu = torch.from_numpy(e5_np).to(device=device, dtype=torch.float16)

    return bgem3_gpu, e5_gpu

def eval_optimized_consensus(bgem3_model, e5_model, train_file, folds_file, chunk_doc_ids, bgem3_gpu, e5_gpu, doc_ids_arr, bm25, top_k=5, candidate_top_k=100, alpha=0.3, beta=0.02):
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(folds_file, 'r', encoding='utf-8') as f:
        folds = json.load(f)

    device = bgem3_gpu.device
    fold_metrics = []

    print("\n--- Evaluating EXP-011d (Optimized Parent-Child Consensus Retrieval) on 5-Fold Local CV ---", flush=True)

    for fold_name, qids in folds.items():
        y_true = {}
        y_pred = {}

        queries = [train_data[qid]['question'] for qid in qids]
        
        # Batch query encoding on GPU
        bgem3_query_embs = bgem3_model.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        e5_queries = [f"query: {q}" for q in queries]
        e5_query_embs = e5_model.encode(e5_queries, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

        bgem3_q_gpu = torch.from_numpy(bgem3_query_embs).to(device=device, dtype=torch.float16)
        e5_q_gpu = torch.from_numpy(e5_query_embs).to(device=device, dtype=torch.float16)

        # Batch similarity matrix multiplication on GPU
        bgem3_sims_mat = torch.matmul(bgem3_gpu, bgem3_q_gpu.T)  # (N_chunks, N_queries)
        e5_sims_mat = torch.matmul(e5_gpu, e5_q_gpu.T)          # (N_chunks, N_queries)

        bgem3_top_k_res = torch.topk(bgem3_sims_mat, k=candidate_top_k, dim=0)
        e5_top_k_res = torch.topk(e5_sims_mat, k=candidate_top_k, dim=0)

        bgem3_indices_mat = bgem3_top_k_res.indices.cpu().numpy()
        bgem3_sims_cpu = bgem3_sims_mat.cpu().numpy()

        e5_indices_mat = e5_top_k_res.indices.cpu().numpy()
        e5_sims_cpu = e5_sims_mat.cpu().numpy()

        for idx, qid in enumerate(qids):
            query_text = train_data[qid]['question']
            y_true[qid] = train_data[qid]['answer']

            # 1. BM25 Document search
            query_tokens = word_tokenize(query_text.lower())
            bm25_scores = bm25.get_scores(query_tokens)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:100]
            bm25_docs = doc_ids_arr[bm25_top_indices].tolist()

            # 2. BGE-M3 Optimized Consensus
            bgem3_top_idx = bgem3_indices_mat[:, idx]
            bgem3_sims_q = bgem3_sims_cpu[:, idx]
            bgem3_docs = compute_parent_consensus_scores(bgem3_top_idx, bgem3_sims_q, chunk_doc_ids, candidate_top_k=candidate_top_k, alpha=alpha, beta=beta)

            # 3. E5-Large Optimized Consensus
            e5_top_idx = e5_indices_mat[:, idx]
            e5_sims_q = e5_sims_cpu[:, idx]
            e5_docs = compute_parent_consensus_scores(e5_top_idx, e5_sims_q, chunk_doc_ids, candidate_top_k=candidate_top_k, alpha=alpha, beta=beta)

            # 4. RRF 3-Way Fusion
            fused_top_docs = rrf_3way(bm25_docs, bgem3_docs, e5_docs, top_k=top_k)
            y_pred[qid] = {"answer": fused_top_docs}

        metrics = eval_retrieval(y_pred, y_true)
        fold_metrics.append(metrics)
        print(f"  {fold_name}: Recall@{top_k} = {metrics['recall']:.4f} | Precision@{top_k} = {metrics['precision']:.4f}", flush=True)

    mean_recall = float(np.mean([m['recall'] for m in fold_metrics]))
    mean_precision = float(np.mean([m['precision'] for m in fold_metrics]))

    print(f"\n=================================================", flush=True)
    print(f"EXP-011d Local 5-Fold CV Mean Recall@{top_k}    : {mean_recall:.4f}", flush=True)
    print(f"EXP-011d Local 5-Fold CV Mean Precision@{top_k} : {mean_precision:.4f}", flush=True)
    print(f"=================================================\n", flush=True)

    return mean_recall, mean_precision

def generate_public_submission(bgem3_model, e5_model, public_file, chunk_doc_ids, bgem3_gpu, e5_gpu, doc_ids_arr, bm25, output_dir, mean_recall, mean_precision, top_k=5, candidate_top_k=100, alpha=0.3, beta=0.02):
    os.makedirs(output_dir, exist_ok=True)
    with open(public_file, 'r', encoding='utf-8') as f:
        public_data = json.load(f)

    device = bgem3_gpu.device
    qids = list(public_data.keys())
    queries = [public_data[qid]['question'] for qid in qids]

    print(f"Generating predictions for {len(qids)} public test queries...", flush=True)

    bgem3_query_embs = bgem3_model.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    e5_queries = [f"query: {q}" for q in queries]
    e5_query_embs = e5_model.encode(e5_queries, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    bgem3_q_gpu = torch.from_numpy(bgem3_query_embs).to(device=device, dtype=torch.float16)
    e5_q_gpu = torch.from_numpy(e5_query_embs).to(device=device, dtype=torch.float16)

    bgem3_sims_mat = torch.matmul(bgem3_gpu, bgem3_q_gpu.T)
    e5_sims_mat = torch.matmul(e5_gpu, e5_q_gpu.T)

    bgem3_top_k_res = torch.topk(bgem3_sims_mat, k=candidate_top_k, dim=0)
    e5_top_k_res = torch.topk(e5_sims_mat, k=candidate_top_k, dim=0)

    bgem3_indices_mat = bgem3_top_k_res.indices.cpu().numpy()
    bgem3_sims_cpu = bgem3_sims_mat.cpu().numpy()

    e5_indices_mat = e5_top_k_res.indices.cpu().numpy()
    e5_sims_cpu = e5_sims_mat.cpu().numpy()

    results = {}
    for idx, qid in enumerate(qids):
        query_text = public_data[qid]['question']
        query_tokens = word_tokenize(query_text.lower())
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:100]
        bm25_docs = doc_ids_arr[bm25_top_indices].tolist()

        bgem3_top_idx = bgem3_indices_mat[:, idx]
        bgem3_sims_q = bgem3_sims_cpu[:, idx]
        bgem3_docs = compute_parent_consensus_scores(bgem3_top_idx, bgem3_sims_q, chunk_doc_ids, candidate_top_k=candidate_top_k, alpha=alpha, beta=beta)

        e5_top_idx = e5_indices_mat[:, idx]
        e5_sims_q = e5_sims_cpu[:, idx]
        e5_docs = compute_parent_consensus_scores(e5_top_idx, e5_sims_q, chunk_doc_ids, candidate_top_k=candidate_top_k, alpha=alpha, beta=beta)

        fused_top_docs = rrf_3way(bm25_docs, bgem3_docs, e5_docs, top_k=top_k)
        results[qid] = {"answer": fused_top_docs}

    json_path = os.path.join(output_dir, "submission.json")
    zip_path = os.path.join(output_dir, "submission.zip")
    metrics_path = os.path.join(output_dir, "metrics.txt")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submission.json")

    metrics_text = (
        f"Experiment: EXP-011d (Optimized Parent-Child Consensus Retrieval)\n"
        f"Category: hierarchical/parent_child_consensus_optimized\n"
        f"Description: Tight Candidate Window (K=100) Parent-Child Consensus Retrieval (MaxP + 0.3*Top3-Mean + 0.02*Log(Hits)).\n"
        f"Total Model Parameters: 0B (BM25) + 560M (BGE-M3) + 560M (E5-Large) = 1.12B (<4B constraint).\n"
        f"Hyperparameters: candidate_top_k=100, alpha=0.3, beta=0.02\n"
        f"-------------------------------------------------\n"
        f"Local 5-Fold CV Mean Recall@5    : {mean_recall:.4f}\n"
        f"Local 5-Fold CV Mean Precision@5 : {mean_precision:.4f}\n"
        f"-------------------------------------------------\n"
    )

    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(metrics_text)

    print(f"Saved submission JSON: {json_path}", flush=True)
    print(f"Saved submission ZIP: {zip_path}", flush=True)
    print(f"Saved metrics TXT: {metrics_path}", flush=True)

if __name__ == "__main__":
    base_dir = "d:/Study/DSC2026/LegalIR"
    selected_contexts_dir = os.path.join(base_dir, "public_test_dataset/selected-contexts")
    train_file = os.path.join(base_dir, "public_test_dataset/train.json")
    public_file = os.path.join(base_dir, "public_test_dataset/public-official.json")
    folds_file = os.path.join(base_dir, "cache/cv_folds.json")
    cache_chunks = os.path.join(base_dir, "cache/article_chunks.json")
    cache_bm25 = os.path.join(base_dir, "cache/corpus_bm25_underthesea_tokens.pkl")
    cache_bgem3_emb = os.path.join(base_dir, "cache/bgem3_article_chunk_embeddings.pkl")
    cache_e5_emb = os.path.join(base_dir, "cache/e5_article_chunk_embeddings.pkl")
    output_dir = os.path.join(base_dir, "results/hierarchical/parent_child_consensus_optimized")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)

    print("Loading BGE-M3 and E5-Large models in FP16...", flush=True)
    bgem3_model = SentenceTransformer('BAAI/bge-m3', device=device).half()
    bgem3_model.max_seq_length = 512

    e5_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device).half()
    e5_model.max_seq_length = 512

    chunks, chunk_ids, chunk_doc_ids, doc_to_chunk_ids = load_article_chunks_corpus(cache_chunks)
    bgem3_gpu, e5_gpu = load_cached_embeddings_gpu(cache_bgem3_emb, cache_e5_emb, device=device)
    doc_ids_arr, bm25 = get_bm25_index(selected_contexts_dir, cache_bm25)

    mean_recall, mean_precision = eval_optimized_consensus(
        bgem3_model, e5_model, train_file, folds_file, 
        chunk_doc_ids, bgem3_gpu, e5_gpu, 
        doc_ids_arr, bm25, top_k=5, candidate_top_k=100, alpha=0.3, beta=0.02
    )
    
    generate_public_submission(
        bgem3_model, e5_model, public_file, 
        chunk_doc_ids, bgem3_gpu, e5_gpu, 
        doc_ids_arr, bm25, output_dir, mean_recall, mean_precision, top_k=5,
        candidate_top_k=100, alpha=0.3, beta=0.02
    )
