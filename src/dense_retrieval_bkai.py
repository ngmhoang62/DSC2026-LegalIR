import json
import glob
import os
import pickle
import zipfile
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
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

def get_or_build_dense_embeddings(model, corpus_texts, cache_path: str, batch_size: int = 32):
    if os.path.exists(cache_path):
        print(f"Loading dense corpus embeddings from cache: {cache_path}", flush=True)
        with open(cache_path, 'rb') as f:
            doc_embeddings = pickle.load(f)
    else:
        print("Encoding corpus documents with vietnamese-bi-encoder...", flush=True)
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
        print(f"Saved dense embeddings to cache: {cache_path}", flush=True)

    return doc_embeddings

def run_dense_eval(train_path: str, folds_path: str, doc_ids: list, doc_embeddings: np.ndarray, model: SentenceTransformer, top_k: int = 5):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(folds_path, 'r', encoding='utf-8') as f:
        folds = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    all_fold_metrics = []

    print("\n--- Evaluating Dense Retrieval (bkai-bi-encoder) on 5-Fold Local CV ---", flush=True)
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

        similarity_matrix = q_embeddings @ doc_embeddings.T

        for idx, qid in enumerate(val_qids):
            scores = similarity_matrix[idx]
            top_indices = np.argsort(scores)[::-1][:top_k]
            pred_doc_ids = doc_ids_arr[top_indices].tolist()
            y_pred[qid] = {"answer": pred_doc_ids}

        res = eval_retrieval(y_pred, y_true)
        all_fold_metrics.append(res)
        print(f"  {fold_name}: Recall@{top_k} = {res['recall']:.4f} | Precision@{top_k} = {res['precision']:.4f}", flush=True)

    mean_recall = float(np.mean([m['recall'] for m in all_fold_metrics]))
    mean_precision = float(np.mean([m['precision'] for m in all_fold_metrics]))
    print(f"\n>>> 5-Fold CV Mean Recall@{top_k}: {mean_recall:.4f} | Mean Precision@{top_k}: {mean_precision:.4f} <<<\n", flush=True)
    return mean_recall, mean_precision

def generate_public_submission(public_path: str, doc_ids: list, doc_embeddings: np.ndarray, model: SentenceTransformer, res_dir: str, mean_recall: float, mean_precision: float, top_k: int = 5):
    os.makedirs(res_dir, exist_ok=True)
    out_json = os.path.join(res_dir, "submission.json")
    out_zip = os.path.join(res_dir, "submission.zip")
    out_txt = os.path.join(res_dir, "metrics.txt")

    with open(public_path, 'r', encoding='utf-8') as f:
        public_data = json.load(f)

    q_ids = list(public_data.keys())
    public_questions = [public_data[qid]['question'] for qid in q_ids]

    print("Generating predictions for Public Test with Dense Bi-Encoder...", flush=True)
    q_embeddings = model.encode(
        public_questions,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    similarity_matrix = q_embeddings @ doc_embeddings.T
    doc_ids_arr = np.array(doc_ids)

    submission = {}
    for idx, qid in enumerate(q_ids):
        scores = similarity_matrix[idx]
        top_indices = np.argsort(scores)[::-1][:top_k]
        pred_doc_ids = doc_ids_arr[top_indices].tolist()
        submission[qid] = {"answer": pred_doc_ids}

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname="submission.json")

    metrics_content = f"""Experiment: Dense Retrieval (vietnamese-bi-encoder)
Category: dense/vietnamese-bi-encoder
Description: Dense Embedding Retrieval using bkai-foundation-models/vietnamese-bi-encoder.
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
    cache_file = os.path.join(base_dir, "cache/corpus_dense_bkai_embeddings.pkl")
    res_dir = os.path.join(base_dir, "results/dense/vietnamese-bi-encoder")

    model_name = "bkai-foundation-models/vietnamese-bi-encoder"
    print(f"Loading model: {model_name}...", flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    model = SentenceTransformer(model_name, device=device)

    doc_ids, corpus_texts, doc_map = load_corpus(selected_contexts_dir)
    doc_embeddings = get_or_build_dense_embeddings(model, corpus_texts, cache_file)

    # 1. Run 5-Fold Local CV
    mean_recall, mean_precision = run_dense_eval(train_file, folds_file, doc_ids, doc_embeddings, model, top_k=5)

    # 2. Save submission files and metrics.txt
    generate_public_submission(public_file, doc_ids, doc_embeddings, model, res_dir, mean_recall, mean_precision, top_k=5)
