import json
import glob
import os
import pickle
import zipfile
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from evaluator import eval_retrieval

def preprocess_single_text(text: str) -> list:
    if not text:
        return []
    return word_tokenize(text.lower(), format="text").split()

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

def get_or_build_bm25_tokens(doc_ids, corpus_texts, cache_path: str):
    if os.path.exists(cache_path):
        print(f"Loading tokenized corpus from cache: {cache_path}", flush=True)
        with open(cache_path, 'rb') as f:
            tokenized_corpus = pickle.load(f)
    else:
        num_cores = max(1, mp.cpu_count() - 1)
        print(f"Tokenizing corpus with underthesea on {num_cores} CPU cores...", flush=True)
        with mp.Pool(processes=num_cores) as pool:
            tokenized_corpus = list(tqdm(pool.imap(preprocess_single_text, corpus_texts, chunksize=100), total=len(corpus_texts)))
            
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(tokenized_corpus, f)
        print(f"Saved tokenized corpus to cache: {cache_path}", flush=True)
        
    return tokenized_corpus

def run_bm25_eval(train_path: str, folds_path: str, doc_ids: list, bm25: BM25Okapi, top_k: int = 5):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(folds_path, 'r', encoding='utf-8') as f:
        folds = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    all_fold_metrics = []

    print("\n--- Evaluating BM25 Baseline on 5-Fold Local CV ---", flush=True)
    for fold_name, val_qids in folds.items():
        y_true = {qid: train_data[qid]['answer'] for qid in val_qids}
        y_pred = {}

        for qid in val_qids:
            query_text = train_data[qid]['question']
            query_tokens = preprocess_single_text(query_text)
            scores = bm25.get_scores(query_tokens)
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

def generate_public_submission(public_path: str, doc_ids: list, bm25: BM25Okapi, res_dir: str, mean_recall: float, mean_precision: float, top_k: int = 5):
    os.makedirs(res_dir, exist_ok=True)
    out_json = os.path.join(res_dir, "submission.json")
    out_zip = os.path.join(res_dir, "submission.zip")
    out_txt = os.path.join(res_dir, "metrics.txt")

    with open(public_path, 'r', encoding='utf-8') as f:
        public_data = json.load(f)

    doc_ids_arr = np.array(doc_ids)
    submission = {}

    print("Generating predictions for Public Test...", flush=True)
    for qid, val in tqdm(public_data.items()):
        query_text = val['question']
        query_tokens = preprocess_single_text(query_text)
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        pred_doc_ids = doc_ids_arr[top_indices].tolist()
        submission[qid] = {"answer": pred_doc_ids}

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname="submission.json")

    metrics_content = f"""Experiment: BM25 Lexical Baseline
Category: baselines/bm25
Description: BM25Okapi retrieval with underthesea Vietnamese word tokenization.
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
    cache_file = os.path.join(base_dir, "cache/corpus_bm25_tokens.pkl")
    res_dir = os.path.join(base_dir, "results/baselines/bm25")

    doc_ids, corpus_texts, doc_map = load_corpus(selected_contexts_dir)
    tokenized_corpus = get_or_build_bm25_tokens(doc_ids, corpus_texts, cache_file)

    print("Building BM25 Index...", flush=True)
    bm25 = BM25Okapi(tokenized_corpus)

    # 1. Run 5-Fold Local CV
    mean_recall, mean_precision = run_bm25_eval(train_file, folds_file, doc_ids, bm25, top_k=5)

    # 2. Save submission files and metrics.txt
    generate_public_submission(public_file, doc_ids, bm25, res_dir, mean_recall, mean_precision, top_k=5)
