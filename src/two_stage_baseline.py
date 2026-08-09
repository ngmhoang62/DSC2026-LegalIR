import json
import os
import pickle
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from evaluator import eval_retrieval

class LegalRerankerDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples  # list of (question, text, label)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, doc_text, label = self.samples[idx]
        encoding = self.tokenizer(
            q,
            doc_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item

def generate_hard_negatives(train_data, val_qids, doc_ids, doc_map, bm25, top_k_bm25=50, max_negatives=5):
    """
    Generate (question, doc_text, label) training samples.
    Positives: ground truth docs (label=1.0)
    Hard Negatives: top BM25 docs that are NOT ground truth (label=0.0)
    """
    doc_ids_arr = np.array(doc_ids)
    samples = []
    
    # Preprocess questions
    for qid in tqdm(val_qids, desc="Mining hard negatives"):
        query_text = train_data[qid]['question']
        gt_docs = set(train_data[qid]['answer'])
        
        # Add positives
        for gtd in gt_docs:
            if gtd in doc_map:
                samples.append((query_text, doc_map[gtd], 1.0))
        
        # Get BM25 top_k
        query_tokens = word_tokenize(query_text.lower(), format="text").split()
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k_bm25]
        top_bm25_docs = doc_ids_arr[top_indices]
        
        neg_count = 0
        for d_id in top_bm25_docs:
            if d_id not in gt_docs and d_id in doc_map:
                samples.append((query_text, doc_map[d_id], 0.0))
                neg_count += 1
                if neg_count >= max_negatives:
                    break

    print(f"Generated {len(samples)} training pairs for Cross-Encoder.")
    return samples

def train_reranker(model, train_loader, val_loader, device, epochs=2, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    return model

def rerank_eval(model, tokenizer, q_dict, doc_ids, doc_map, bm25, device, stage1_top_k=50, final_top_k=5, max_length=512):
    """
    Rerank Stage 1 (BM25) Top-50 candidates using Fine-tuned Cross-Encoder.
    """
    model.eval()
    model.to(device)
    doc_ids_arr = np.array(doc_ids)
    y_pred = {}

    for qid, val in tqdm(q_dict.items(), desc="Reranking candidates"):
        query_text = val['question']
        query_tokens = word_tokenize(query_text.lower(), format="text").split()
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:stage1_top_k]
        candidate_ids = doc_ids_arr[top_indices]

        # Prepare pairs
        pairs = [(query_text, doc_map[cid]) for cid in candidate_ids if cid in doc_map]
        actual_cids = [cid for cid in candidate_ids if cid in doc_map]

        if not pairs:
            y_pred[qid] = {"answer": list(candidate_ids[:final_top_k])}
            continue

        # Inference in small batches
        pair_scores = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                inputs = tokenizer(
                    [p[0] for p in batch_pairs],
                    [p[1] for p in batch_pairs],
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(-1).cpu().numpy()
                if logits.ndim == 0:
                    logits = [logits.item()]
                pair_scores.extend(logits)

        # Xếp hạng lại theo score giảm dần
        sorted_indices = np.argsort(pair_scores)[::-1][:final_top_k]
        reranked_doc_ids = [actual_cids[idx] for idx in sorted_indices]
        y_pred[qid] = {"answer": reranked_doc_ids}

    return y_pred

if __name__ == "__main__":
    print("Stage 2 Reranker script template ready.")
