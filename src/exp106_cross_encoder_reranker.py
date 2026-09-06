"""EXP-106: Heavy Cross-Encoder Reranker & Graph-Aware Synergy Fusion (Stage 2).

Key Invariants:
1. Strict K=50 candidate pool from EXP-102 MIL-NCE RRF (349,550 pairs, Recall@50 = 98.53%).
2. Canonical Dual-Evidence Article Extraction: Intact legal articles parsed by boundaries + verified metadata.
3. Fast Memory-Efficient Streaming: Document-level article caching + streaming JSONL storage.
4. Multi-Gold Safe: Multi-positive contrastive LoRA training + Set-theoretic negative filtering.
5. Hardware Safety: Batch size 16 during inference (stays strictly within 4.5GB VRAM, zero PCIe paging).
6. Graph-Aware Synergy Fusion: Within-query Z-score scaling + Symbolic law number boost + Citation graph propagation.
7. 5-Fold Stability Audit: Evaluates heldout OOF across all 5 folds.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(r"D:\Study\DSC2026\LegalIR")
TRAIN_PATH = ROOT / "public_test_dataset" / "train.json"
FOLDS_PATH = ROOT / "cache" / "cv_folds.json"
AUDIT_EXP034 = ROOT / "results" / "exp034_shallow_retrieval" / "audit" / "REPORT.json"
EXP102_CACHE = ROOT / "cache" / "exp102_mil_nce_retrieval"
CORPUS_DIR = ROOT / "public_test_dataset" / "selected-contexts"
DOC_METADATA_PATH = ROOT / "cache" / "exp030_legal_evidence_routing_canonical_v1" / "metadata" / "document_metadata.jsonl"
CHUNK_IDS_JSONL = ROOT / "cache" / "e5_final_v1" / "chunk_ids.jsonl"
RESULTS_DIR = ROOT / "results" / "exp106_cross_encoder_reranker"
CACHE_DIR = ROOT / "cache" / "exp106_cross_encoder_reranker"

MODEL_ID = "BAAI/bge-reranker-v2-m3"
KS = (1, 5, 16, 24, 32, 50)
SEED = 2026

LAW_NUM_PATTERN = re.compile(r"(\d+[\d/]+[A-ZĐa-zđ\d-]+)", re.IGNORECASE)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zscore(arr: Sequence[float]) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    s = float(a.std())
    return (a - a.mean()) / (s if s > 1e-5 else 1.0)


def evaluate_rankings(pred: dict[str, list[str]], answers: dict[str, set[str]], ks: tuple[int, ...] = KS):
    recalls = {k: [] for k in ks}
    rr_5 = []
    
    for qid, gold_set in answers.items():
        if not gold_set or qid not in pred:
            continue
        predicted = pred[qid]
        
        for k in ks:
            pred_k = set(predicted[:k])
            recalls[k].append(len(pred_k & gold_set) / len(gold_set))
            
        first_gold_rank = None
        for r, doc_id in enumerate(predicted[:5], 1):
            if doc_id in gold_set:
                first_gold_rank = r
                break
        rr_5.append(1.0 / first_gold_rank if first_gold_rank else 0.0)
        
    return {
        **{f"recall@{k}": float(np.mean(recalls[k])) for k in ks},
        "mrr@5": float(np.mean(rr_5)),
        "evaluable_queries": len(rr_5),
    }


def load_doc_metadata_and_relations() -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, str]]:
    doc_metadata = {}
    doc_relations = {}
    doc_num_to_id = {}
    
    if DOC_METADATA_PATH.exists():
        for line in DOC_METADATA_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = str(item["doc_id"])
            doc_metadata[doc_id] = item
            official = item.get("official_title", {})
            num = official.get("number")
            if num:
                doc_num_to_id[num.strip().lower()] = doc_id
            doc_relations[doc_id] = item.get("relations", [])
    return doc_metadata, doc_relations, doc_num_to_id


class DocumentArticleCache:
    """Pre-parses and caches document legal articles in RAM to avoid repeated disk reads."""
    def __init__(self, doc_metadata: dict[str, dict[str, Any]], corpus_dir: Path = CORPUS_DIR):
        self.doc_metadata = doc_metadata
        self.corpus_dir = corpus_dir
        self.cache: dict[str, tuple[str, list[str]]] = {}
        
    def get(self, doc_id: str) -> tuple[str, list[str]]:
        doc_id = str(doc_id)
        if doc_id in self.cache:
            return self.cache[doc_id]
            
        meta = self.doc_metadata.get(doc_id, {})
        official = meta.get("official_title", {})
        if official.get("status") == "VERIFIED":
            doc_type = official.get("document_type", "VĂN BẢN")
            number = official.get("number", "")
            display_text = official.get("display_text", "")
            title_str = f"[{doc_type} {number}]: {display_text}"
        else:
            title_str = f"[VĂN BẢN]: {meta.get('normalized_label', f'Tài liệu {doc_id}')}"

        doc_file = self.corpus_dir / f"context_{doc_id}.json"
        if not doc_file.exists():
            self.cache[doc_id] = (title_str, [])
            return self.cache[doc_id]
            
        doc_json = json.loads(doc_file.read_text(encoding="utf-8"))
        passage = doc_json.get("passage", "")
        articles = [a.strip() for a in re.split(r"\n(?=(?:Điều\s+\d+|Chương\s+[IVXLCDM\d]+|Mục\s+\d+))", passage) if a.strip()]
        self.cache[doc_id] = (title_str, articles)
        return self.cache[doc_id]

    def build_capsule(self, doc_id: str, query: str, max_chars: int = 1400) -> str:
        title_str, articles = self.get(doc_id)
        if not articles:
            return title_str
            
        q_words = set(re.findall(r"\w+", query.lower()))
        scored_articles = []
        for art in articles:
            art_words = set(re.findall(r"\w+", art.lower()))
            overlap = len(q_words & art_words)
            scored_articles.append((art, overlap))
            
        scored_articles.sort(key=lambda x: -x[1])
        
        selected_parts = []
        current_len = 0
        for art, overlap in scored_articles:
            if overlap == 0 and selected_parts:
                break
            selected_parts.append(art)
            current_len += len(art)
            if current_len >= max_chars or len(selected_parts) >= 2:
                break
                
        if not selected_parts and articles:
            selected_parts.append(articles[0][:max_chars])
            
        body = "\n\n".join(selected_parts)
        return f"{title_str}\n\n[NỘI DUNG ĐIỀU KHOẢN TRỌNG TÂM]:\n{body}"


def build_and_cache_canonical_dual_capsules() -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]], dict[str, list[str]]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    capsule_file = CACHE_DIR / "capsules_k50_canonical_dual.jsonl"
    
    train_data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    all_qids = sorted(train_data.keys())
    answers = {qid: set(map(str, train_data[qid]["answer"])) for qid in all_qids}
    
    if capsule_file.exists():
        print(f"Loading cached canonical dual capsules from streaming JSONL: {capsule_file}", flush=True)
        capsules = {}
        with open(capsule_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    capsules[str(item["qid"])] = item["capsules"]
        return capsules, answers, folds
        
    print("[1/3] Loading embeddings and document metadata...", flush=True)
    doc_metadata, _, _ = load_doc_metadata_and_relations()
    doc_cache = DocumentArticleCache(doc_metadata)
    
    import exp102_mil_nce_retrieval as exp102
    data = exp102.load_corpus_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    documents = torch.from_numpy(np.asarray(data["chunk_embeddings"], dtype=np.float32)).to(device)
    documents = documents / torch.norm(documents, dim=-1, keepdim=True).clamp_min(1e-12)
    
    doc_to_chunk_indices = data["doc_to_chunk_indices"]
    qid_to_idx = data["qid_to_idx"]
    bm25_ranks = data["bm25_ranks"]
    
    print("[2/3] Extracting Top 50 candidates & assembling intact legal articles (Fast Document Cache)...", flush=True)
    capsules = {}
    t_start = time.time()
    
    with open(capsule_file, "w", encoding="utf-8") as out_f:
        with torch.no_grad():
            for fold_name, heldout_qids in sorted(folds.items()):
                heldout_qids = list(map(str, heldout_qids))
                ckpt_path = EXP102_CACHE / f"{fold_name}.pt"
                model = exp102.ResidualProjection(1024, 32).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.eval()
                
                for qid in heldout_qids:
                    q_text = train_data[qid]["question"]
                    q_idx = qid_to_idx[qid]
                    q_vec = torch.from_numpy(data["query_embeddings"][q_idx]).to(device).unsqueeze(0)
                    q_proj = model(q_vec)
                    q_proj = q_proj / torch.norm(q_proj, dim=-1, keepdim=True).clamp_min(1e-12)
                    sims = torch.mm(q_proj, documents.T).squeeze(0).cpu().numpy()
                    
                    # Dense doc scores
                    dense_doc_scores = {}
                    for doc_id, c_indices in doc_to_chunk_indices.items():
                        dense_doc_scores[doc_id] = float(np.max(sims[c_indices]))
                        
                    # RRF candidate pool
                    dense_sorted = sorted(dense_doc_scores.keys(), key=lambda d: -dense_doc_scores[d])
                    scored = []
                    for r_d, doc_id in enumerate(dense_sorted[:100], 1):
                        r_b = bm25_ranks.get(qid, {}).get(doc_id, 0)
                        score = 0.65 / (32.0 + float(r_d)) + 0.35 * (1.0 / (32.0 + float(r_b)) if r_b > 0 else 0.0)
                        scored.append((doc_id, score, r_d))
                    scored.sort(key=lambda x: -x[1])
                    cands = [x[0] for x in scored[:50]]
                    
                    q_capsules = []
                    for r, doc_id in enumerate(cands, 1):
                        capsule_text = doc_cache.build_capsule(doc_id, q_text)
                        q_capsules.append({
                            "doc_id": doc_id,
                            "rank": r,
                            "rrf_score": 1.0 / (32.0 + float(r)),
                            "question": q_text,
                            "capsule_text": capsule_text,
                        })
                    capsules[qid] = q_capsules
                    
                    # Stream directly to JSONL to prevent high memory usage
                    out_f.write(json.dumps({"qid": qid, "capsules": q_capsules}, ensure_ascii=False) + "\n")
                    
    print(f"[3/3] Successfully extracted and streamed {len(capsules)} query capsule lists in {time.time()-t_start:.1f}s to: {capsule_file}", flush=True)
    return capsules, answers, folds


class LegalPairTrainDataset(Dataset):
    def __init__(self, train_qids: list[str], capsules: dict[str, list[dict[str, Any]]], answers: dict[str, set[str]]):
        self.pairs: list[tuple[str, str, float]] = []
        
        for qid in train_qids:
            if qid not in capsules or qid not in answers:
                continue
            gold_set = answers[qid]
            q_caps = capsules[qid]
            q_text = q_caps[0]["question"]
            
            # Multi-Positive Expansion
            pos_caps = [c for c in q_caps if str(c["doc_id"]) in gold_set]
            # Strict Set-Theoretic True Negatives
            neg_caps = [c for c in q_caps if str(c["doc_id"]) not in gold_set]
            
            if not pos_caps or not neg_caps:
                continue
                
            for pos in pos_caps:
                self.pairs.append((q_text, pos["capsule_text"], 1.0))
                
            for neg in neg_caps[:2]:
                self.pairs.append((q_text, neg["capsule_text"], 0.0))
                
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        return self.pairs[idx]


def train_lora_fold(
    fold_name: str,
    train_qids: list[str],
    capsules: dict[str, list[dict[str, Any]]],
    answers: dict[str, set[str]],
    device: torch.device,
    epochs: int = 2,
    lr: float = 1e-4,
    batch_size: int = 8,
    accum_steps: int = 2,
):
    print(f"\n========================================================")
    print(f"TRAINING LORA ON {fold_name.upper()} ({len(train_qids)} queries)")
    print(f"========================================================")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=1,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.train()
    
    dataset = LegalPairTrainDataset(train_qids, capsules, answers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"Dataset size: {len(dataset)} pairs | DataLoader: {len(dataloader)} batches/epoch")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.BCEWithLogitsLoss()
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad()
        
        for step, (queries, docs, labels) in enumerate(dataloader, 1):
            inputs = tokenizer(
                list(queries),
                list(docs),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            target = labels.float().unsqueeze(-1).to(device)
            
            with torch.amp.autocast("cuda"):
                outputs = model(**inputs)
                loss = criterion(outputs.logits, target) / accum_steps
                
            scaler.scale(loss).backward()
            running_loss += loss.item() * accum_steps
            step_count += 1
            
            if step % accum_steps == 0 or step == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            if step % 250 == 0:
                print(f"  [Epoch {epoch}/{epochs}] Step {step}/{len(dataloader)} | Loss: {running_loss/step_count:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)
                
        print(f"--> [Epoch {epoch} Finished] Avg Loss: {running_loss/step_count:.4f} (Elapsed: {time.time()-start_time:.1f}s)", flush=True)
        
    adapter_dir = CACHE_DIR / f"lora_adapter_{fold_name}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter for {fold_name} to: {adapter_dir}", flush=True)
    
    return model, tokenizer


def score_heldout_with_lora(
    model: nn.Module,
    tokenizer: Any,
    heldout_qids: list[str],
    capsules: dict[str, list[dict[str, Any]]],
    device: torch.device,
    batch_size: int = 16,  # Strictly bounded to 16 for zero PCIe paging
) -> dict[str, dict[str, float]]:
    model.eval()
    all_pairs = []
    pair_map = []
    
    for qid in heldout_qids:
        for item in capsules[qid]:
            all_pairs.append((item["question"], item["capsule_text"]))
            pair_map.append((qid, item["doc_id"]))
            
    print(f"Scoring {len(all_pairs)} heldout candidate pairs (batch_size={batch_size})...", flush=True)
    scores_dict = defaultdict(dict)
    start_t = time.time()
    
    with torch.no_grad():
        for i in range(0, len(all_pairs), batch_size):
            batch_pairs = all_pairs[i:i + batch_size]
            queries = [p[0] for p in batch_pairs]
            docs = [p[1] for p in batch_pairs]
            
            inputs = tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            
            with torch.amp.autocast("cuda"):
                outputs = model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).float().cpu().numpy()
                
            for j, score in enumerate(batch_scores):
                qid, doc_id = pair_map[i + j]
                scores_dict[qid][doc_id] = float(score)
                
            if (i + batch_size) % 10000 < batch_size or (i + batch_size) >= len(all_pairs):
                elapsed = time.time() - start_t
                rate = (i + len(batch_pairs)) / max(elapsed, 1e-5)
                print(f"  Scored {min(i + batch_size, len(all_pairs))}/{len(all_pairs)} pairs | Rate: {rate:.1f} pairs/s | ETA: {(len(all_pairs) - (i + batch_size))/max(rate, 1e-5):.0f}s", flush=True)
                
    return scores_dict


def run_full_exp106():
    print("==========================================================")
    print("RUNNING EXP-106: Heavy Cross-Encoder Reranker (Stage 2)")
    print("==========================================================")
    
    set_seed()
    capsules, answers, folds = build_and_cache_canonical_dual_capsules()
    doc_metadata, doc_relations, doc_num_to_id = load_doc_metadata_and_relations()
    train_data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    
    audit_data = json.loads(AUDIT_EXP034.read_text(encoding="utf-8")) if AUDIT_EXP034.exists() else {}
    non_eval_qids = set(map(str, audit_data.get("label_stats", {}).get("non_evaluable_qids", [])))
    
    all_qids = sorted(answers.keys())
    evaluable_qids = [q for q in all_qids if q not in non_eval_qids]
    eval_answers = {q: answers[q] for q in evaluable_qids}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU device: {device} ({torch.cuda.get_device_name(0)})")
    
    oof_ce_scores = {}
    
    for fold_name, heldout_qids in sorted(folds.items()):
        heldout_qids = list(map(str, heldout_qids))
        outer_train_qids = [q for f, q_list in folds.items() if f != fold_name for q in q_list]
        
        scores_file = CACHE_DIR / f"scores_{fold_name}.json"
        
        if scores_file.exists():
            print(f"Loading cached scores for {fold_name} from: {scores_file}", flush=True)
            fold_scores = json.loads(scores_file.read_text(encoding="utf-8"))
        else:
            model, tokenizer = train_lora_fold(
                fold_name=fold_name,
                train_qids=outer_train_qids,
                capsules=capsules,
                answers=answers,
                device=device,
                epochs=2,
                lr=1e-4,
                batch_size=8,
                accum_steps=2,
            )
            fold_scores = score_heldout_with_lora(model, tokenizer, heldout_qids, capsules, device, batch_size=16)
            scores_file.write_text(json.dumps(fold_scores, ensure_ascii=False), encoding="utf-8")
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
        oof_ce_scores.update(fold_scores)
        
    print("\n========================================================")
    print("EVALUATING 5-FOLD OOF GRAPH-AWARE SYNERGY FUSION")
    print("========================================================")
    
    results_summary = {}
    
    grid_params = [
        (0.3, 0.7, 0.0),
        (0.4, 0.6, 0.0),
        (0.5, 0.5, 0.0),
        (0.3, 0.7, 0.4),
        (0.4, 0.6, 0.4),
        (0.5, 0.5, 0.4),
        (0.0, 1.0, 0.0),  # Pure Stage 1 baseline for comparison
    ]
    
    for w_ce, w_s1, gamma in grid_params:
        pred_rankings = {}
        
        for qid in evaluable_qids:
            q_text = train_data[qid]["question"]
            q_matches = set(LAW_NUM_PATTERN.findall(q_text))
            
            c_list = capsules[qid]
            raw_ce = [oof_ce_scores[qid].get(item["doc_id"], -20.0) for item in c_list]
            raw_s1 = [item["rrf_score"] for item in c_list]
            
            z_ce = zscore(raw_ce)
            z_s1 = zscore(raw_s1)
            
            fused = {}
            for i, item in enumerate(c_list):
                doc_id = item["doc_id"]
                score = w_ce * z_ce[i] + w_s1 * z_s1[i]
                
                # Symbolic Law Entity Priority Boost
                cap_text = item["capsule_text"]
                if q_matches:
                    for m in q_matches:
                        if len(m) >= 4 and m.lower() in cap_text.lower():
                            score += 10.0
                            break
                fused[doc_id] = score
                
            # Citation Graph Propagation
            if gamma > 0:
                top_cand = max(fused.keys(), key=lambda d: fused[d])
                relations = doc_relations.get(top_cand, [])
                for rel in relations:
                    rel_num = rel.get("number", "").strip().lower()
                    if rel_num in doc_num_to_id:
                        target_id = doc_num_to_id[rel_num]
                        if target_id in fused:
                            fused[target_id] += gamma * 2.0
                            
            order = sorted(fused.keys(), key=lambda d: -fused[d])
            pred_rankings[qid] = order
            
        m_agg = evaluate_rankings(pred_rankings, eval_answers)
        
        fold_metrics = {}
        all_folds_pass = True
        
        for fold_name, heldout_qids in sorted(folds.items()):
            heldout_eval = {str(q): answers[str(q)] for q in heldout_qids if str(q) not in non_eval_qids}
            heldout_pred = {str(q): pred_rankings[str(q)] for q in heldout_qids if str(q) not in non_eval_qids}
            f_m = evaluate_rankings(heldout_pred, heldout_eval)
            fold_metrics[fold_name] = f_m
            if f_m["recall@5"] < 0.935:
                all_folds_pass = False
                
        run_key = f"synergy_wce{w_ce:.1f}_ws1{w_s1:.1f}_gamma{gamma:.1f}"
        results_summary[run_key] = {
            "w_ce": w_ce,
            "w_s1": w_s1,
            "gamma": gamma,
            "aggregate_metrics": m_agg,
            "fold_metrics": fold_metrics,
            "all_5_folds_pass_gate": all_folds_pass,
        }
        
        print(f"[{run_key}] Aggregate R@1: {m_agg['recall@1']*100:.2f}% | R@5: {m_agg['recall@5']*100:.2f}% | R@16: {m_agg['recall@16']*100:.2f}% | MRR@5: {m_agg['mrr@5']:.4f} | Gate Pass: {all_folds_pass}")

    report_path = RESULTS_DIR / "REPORT.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved final EXP-106 report to: {report_path}")
    return results_summary


if __name__ == "__main__":
    run_full_exp106()
