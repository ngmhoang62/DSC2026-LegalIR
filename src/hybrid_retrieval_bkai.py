import json
import glob
import os
import pickle
import zipfile
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from underthesea import word_tokenize
from evaluator import eval_retrieval

def reciprocal_rank_fusion(bm25_scores, dense_scores, k=60, top_k_candidates=50):
    """
    RRF combination of BM25 and Dense similarity scores for a single query.
    """
    # Rank indices
    bm25_ranks = np.argsort(bm25_scores)[::-1][:top_k_candidates]
    dense_ranks = np.argsort(dense_scores)[::-1][:top_k_candidates]

    rrf_score_dict = {}

    # Score BM25 candidates
    for rank, idx in enumerate(bm25_ranks):
        rrf_score_dict[idx] = rrf_score_dict.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # Score Dense candidates
    for rank, idx in enumerate(dense_ranks):
        rrf_score_dict[idx] = rrf_score_dict.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # Sort candidates by combined RRF score
    sorted_candidates = sorted(rrf_score_dict.items(), key=lambda item: item[1], reverse=True)
    return [idx for idx, score in sorted_candidates]

def run_hybrid_eval(train_path: str, folds_path: str, doc_ids: list, bm25: BM25Okapi, doc_embeddings: np.ndarray, model: SentenceTransformer, top_k: int = 5, rrf_k: int = 60):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(folds_path, 'r', encoding='utf-8') as f:
        folds = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    all_fold_metrics = []

    print(f"\n--- Evaluating Hybrid Retrieval (BM25 + vietnamese-bi-encoder + RRF k={rrf_k}) on 5-Fold Local CV ---", flush=True)
    for fold_name, val_qids in folds.items():
        y_true = {qid: train_data[qid]['answer'] for qid in val_qids}
        y_pred = {}

        val_questions = [train_data[qid]['question'] for qid in val_qids]
        q_embeddings = model.encode(
            val_questions,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        dense_similarity_matrix = q_embeddings @ doc_embeddings.T

        for idx, qid in enumerate(val_qids):
            query_text = train_data[qid]['question']
            query_tokens = word_tokenize(query_text.lower(), format="text").split()
            bm25_scores = bm25.get_scores(query_tokens)
            dense_scores = dense_similarity_matrix[idx]

            # Reciprocal Rank Fusion
            hybrid_top_indices = reciprocal_rank_fusion(bm25_scores, dense_scores, k=rrf_k, top_k_candidates=50)[:top_k]
            pred_doc_ids = doc_ids_arr[hybrid_top_indices].tolist()
            y_pred[qid] = {"answer": pred_doc_ids}

        res = eval_retrieval(y_pred, y_true)
        all_fold_metrics.append(res)
        print(f"  {fold_name}: Recall@{top_k} = {res['recall']:.4f} | Precision@{top_k} = {res['precision']:.4f}", flush=True)

    mean_recall = float(np.mean([m['recall'] for m in all_fold_metrics]))
    mean_precision = float(np.mean([m['precision'] for m in all_fold_metrics]))
    print(f"\n>>> 5-Fold CV Mean Recall@{top_k}: {mean_recall:.4f} | Mean Precision@{top_k}: {mean_precision:.4f} <<<\n", flush=True)
    return mean_recall, mean_precision

def generate_public_submission(public_path: str, doc_ids: list, bm25: BM25Okapi, doc_embeddings: np.ndarray, model: SentenceTransformer, res_dir: str, mean_recall: float, mean_precision: float, top_k: int = 5, rrf_k: int = 60):
    os.makedirs(res_dir, exist_ok=True)
    out_json = os.path.join(res_dir, "submission.json")
    out_zip = os.path.join(res_dir, "submission.zip")
    out_txt = os.path.join(res_dir, "metrics.txt")

    with open(public_path, 'r', encoding='utf-8') as f:
        public_data = json.load(f)

    q_ids = list(public_data.items())
    public_questions = [v['question'] for k, v in q_ids]

    print("Generating predictions for Public Test with Hybrid RRF...", flush=True)
    q_embeddings = model.encode(
        public_questions,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    dense_similarity_matrix = q_embeddings @ doc_embeddings.T
    doc_ids_arr = np.array(doc_ids)

    submission = {}
    for idx, (qid, val) in enumerate(q_ids):
        query_text = val['question']
        query_tokens = word_tokenize(query_text.lower(), format="text").split()
        bm25_scores = bm25.get_scores(query_tokens)
        dense_scores = dense_similarity_matrix[idx]

        hybrid_top_indices = reciprocal_rank_fusion(bm25_scores, dense_scores, k=rrf_k, top_k_candidates=50)[:top_k]
        pred_doc_ids = doc_ids_arr[hybrid_top_indices].tolist()
        submission[qid] = {"answer": pred_doc_ids}

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname="submission.json")

    metrics_content = f"""Experiment: Hybrid Retrieval (BM25 + vietnamese-bi-encoder + RRF k={rrf_k})
Category: hybrid/bm25-vietnamese-bi-encoder
Description: Reciprocal Rank Fusion of BM25 Lexical + vietnamese-bi-encoder Dense embeddings.
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
    selected_contexts_dir = os.path.join(base_dir, "public_test_dataset/selected-contexts")
    train_file = os.path.join(base_dir, "public_test_dataset/train.json")
    public_file = os.path.join(base_dir, "public_test_dataset/public-official.json")
    folds_file = os.path.join(base_dir, "cache/cv_folds.json")
    bm25_cache_file = os.path.join(base_dir, "cache/corpus_bm25_tokens.pkl")
    dense_cache_file = os.path.join(base_dir, "cache/corpus_dense_bkai_embeddings.pkl")
    res_dir = os.path.join(base_dir, "results/hybrid/bm25-vietnamese-bi-encoder")

    # Load BM25
    with open(bm25_cache_file, 'rb') as f:
        tokenized_corpus = pickle.load(f)
    bm25 = BM25Okapi(tokenized_corpus)

    # Load Dense
    with open(dense_cache_file, 'rb') as f:
        doc_embeddings = pickle.load(f)

    # Load Doc IDs
    filepaths = glob.glob(os.path.join(selected_contexts_dir, "context_*.json"))
    doc_ids = [str(json.load(open(fp, 'r', encoding='utf-8'))['id']) for fp in filepaths]

    model_name = "bkai-foundation-models/vietnamese-bi-encoder"
    model = SentenceTransformer(model_name, device="cuda")

    # 1. Run 5-Fold Local CV
    mean_recall, mean_precision = run_hybrid_eval(train_file, folds_file, doc_ids, bm25, doc_embeddings, model, top_k=5, rrf_k=60)

    # 2. Save submission files and metrics.txt
    generate_public_submission(public_file, doc_ids, bm25, doc_embeddings, model, res_dir, mean_recall, mean_precision, top_k=5, rrf_k=60)
