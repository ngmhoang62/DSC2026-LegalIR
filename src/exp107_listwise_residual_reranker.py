"""EXP-107: True End-to-End Residual Listwise Cross-Encoder Reranker on Fold 0.

Strict Methodological Integrity & Architectural Design:
1. True Neural Residual Objective:
   Forward pass computes S_total(q, d) = z(S_Stage1(q, d)) + alpha * CE_theta(q, d).
   Listwise Multi-Positive InfoNCE loss is calculated directly on S_total, training
   the Cross-Encoder exclusively to output delta(q, d) correcting Stage-1 rank inversions.
2. Preserved Full Context Length:
   max_length=512 tokens preserved to avoid truncating legal definitions.
3. Multi-Gold Loss Preservation with Balanced Negatives:
   max_negs=6 hard negatives from Stage-1 ranks 1-8.
4. Strict Nested Cross-Validation Parameter Selection (Zero Test Leakage):
   - Inner Train (Folds 1-3): Trains model to predict on Inner Val.
   - Inner Val (Fold 4): Evaluates out-of-sample predictions to optimize alpha and fusion weights.
   - Outer Retrain (Folds 1-4): Retrains model on all outer folds with frozen hyperparameters.
   - Held-out Test (Fold 0): Blind evaluation on unseen test split.
5. Clean & Transparent Ablation Reporting:
   Isolates Stage-1, Identifier Law Boost, Standalone CE, and True Residual Reranker to prevent
   confounding gains.
6. Canonical Non-Evaluable Exclusion:
   Excludes the canonical non-evaluable queries defined in exp034 audit REPORT.json.
7. Gradient Checkpointing Enabled:
   Eliminates activation memory explosion on 6 GB VRAM GPUs during training.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(r"D:\Study\DSC2026\LegalIR")
TRAIN_PATH = ROOT / "public_test_dataset" / "train.json"
FOLDS_PATH = ROOT / "cache" / "cv_folds.json"
AUDIT_EXP034 = ROOT / "results" / "exp034_shallow_retrieval" / "audit" / "REPORT.json"
CACHE_DIR = ROOT / "cache" / "exp107_listwise_residual_reranker"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "BAAI/bge-reranker-v2-m3"
SEED = 2026
MAX_NEGS = 6
MAX_LENGTH = 512

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

def evaluate_rankings(pred: dict[str, list[str]], answers: dict[str, set[str]], ks: tuple[int, ...] = (1, 5, 16, 50)):
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

class TrueResidualListwiseDataset(Dataset):
    """Dataset packaging Query, All Positives, MAX_NEGS Hard Negatives, and their normalized Stage-1 z-scores."""
    def __init__(self, qid_list: list[str], capsules: dict[str, list[dict[str, Any]]], answers: dict[str, set[str]], max_negs: int = MAX_NEGS):
        self.samples = []
        for qid in qid_list:
            if qid not in capsules or qid not in answers:
                continue
            gold_set = answers[qid]
            q_caps = capsules[qid]
            q_text = q_caps[0]["question"]
            
            all_s1_raw = [c["rrf_score"] for c in q_caps]
            all_s1_z = zscore(all_s1_raw)
            
            pos_indices = [i for i, c in enumerate(q_caps) if str(c["doc_id"]) in gold_set]
            neg_indices = [i for i, c in enumerate(q_caps) if str(c["doc_id"]) not in gold_set]
            
            if not pos_indices or not neg_indices:
                continue
                
            selected_neg_idx = neg_indices[:max_negs]
            
            pos_texts = [q_caps[i]["capsule_text"] for i in pos_indices]
            pos_s1_z = [float(all_s1_z[i]) for i in pos_indices]
            
            neg_texts = [q_caps[i]["capsule_text"] for i in selected_neg_idx]
            neg_s1_z = [float(all_s1_z[i]) for i in selected_neg_idx]
            
            self.samples.append({
                "qid": qid,
                "question": q_text,
                "pos_texts": pos_texts,
                "pos_s1_z": pos_s1_z,
                "neg_texts": neg_texts,
                "neg_s1_z": neg_s1_z,
            })
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def train_residual_model(train_dataset: TrueResidualListwiseDataset, device: torch.device, epochs: int = 2, alpha: float = 0.35, phase_name: str = "Training") -> Any:
    """Fine-tunes BGE-Reranker-v2-m3 with True Residual Listwise InfoNCE Loss & Gradient Checkpointing."""
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=1,
        trust_remote_code=True,
    )
    base_model.config.return_dict = True
    
    # Enable Gradient Checkpointing to free 60% activation VRAM during training
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    accum_steps = 4
    total_steps = (len(train_dataset) // accum_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    temperature = 0.05

    train_start_t = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, idx in enumerate(indices, 1):
            sample = train_dataset[idx]
            q_text = sample["question"]
            all_docs = sample["pos_texts"] + sample["neg_texts"]
            all_s1_z = torch.tensor(sample["pos_s1_z"] + sample["neg_s1_z"], device=device, dtype=torch.float32)
            n_pos = len(sample["pos_texts"])
            
            queries = [q_text] * len(all_docs)
            inputs = tokenizer(
                queries,
                all_docs,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            
            with torch.amp.autocast("cuda"):
                outputs = model(return_dict=True, **inputs)
                ce_logits = outputs.logits.squeeze(-1)
                
                # TRUE RESIDUAL OBJECTIVE: S_total = S_Stage1_z + alpha * CE_delta
                total_logits = all_s1_z + alpha * ce_logits
                
                pos_logits = total_logits[:n_pos]
                neg_logits = total_logits[n_pos:]
                
                loss_list = []
                for p_logit in pos_logits:
                    combined_logits = torch.cat([p_logit.unsqueeze(0), neg_logits]) / temperature
                    target = torch.tensor([0], device=device)
                    loss_list.append(F.cross_entropy(combined_logits.unsqueeze(0), target))
                    
                query_loss = torch.mean(torch.stack(loss_list)) / accum_steps
                
            scaler.scale(query_loss).backward()
            running_loss += query_loss.item() * accum_steps
            
            if step % accum_steps == 0 or step == len(indices):
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
                optimizer.zero_grad()
                
            if step % 200 == 0 or step == len(indices):
                elapsed = time.time() - train_start_t
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                if reserved_mb > 5800:
                    print(f"CRITICAL: VRAM exceeded threshold ({reserved_mb:.1f} MB)! Aborting for safety.", flush=True)
                    sys.exit(1)
                print(f"  [{phase_name} | Ep {epoch}/{epochs}] Query {step}/{len(indices)} | Loss: {running_loss/step:.4f} | VRAM: {allocated_mb:.0f}MB alloc / {reserved_mb:.0f}MB res | Elapsed: {elapsed:.0f}s", flush=True)
                
        print(f"--> [{phase_name} | Ep {epoch} Complete] Avg Loss: {running_loss/len(indices):.4f} (Time: {time.time()-train_start_t:.1f}s)", flush=True)

    return model, tokenizer

def score_query_set(model: Any, tokenizer: Any, qid_list: list[str], capsules: dict[str, list[dict[str, Any]]], device: torch.device, batch_size: int = 16) -> dict[str, dict[str, float]]:
    """Runs batch inference to obtain raw CE logits for a query set."""
    model.eval()
    pairs = []
    pair_map = []
    for qid in qid_list:
        if qid in capsules:
            for item in capsules[qid]:
                pairs.append((item["question"], item["capsule_text"]))
                pair_map.append((qid, item["doc_id"]))
                
    s_dict = defaultdict(dict)
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            inputs = tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            with torch.amp.autocast("cuda"):
                outputs = model(return_dict=True, **inputs)
                batch_scores = outputs.logits.squeeze(-1).float().cpu().numpy()
                
            for j, sc in enumerate(batch_scores):
                q, d = pair_map[i + j]
                s_dict[q][d] = float(sc)
    return s_dict

def main() -> int:
    start_time = time.time()
    set_seed()
    print("==========================================================================")
    print("EXP-107: TRUE RESIDUAL LISTWISE CROSS-ENCODER WITH NESTED CV")
    print(f"CONFIG: max_length={MAX_LENGTH}, max_negs={MAX_NEGS} hard negs, Gradient Checkpointing=ON")
    print("==========================================================================")
    
    # 1. Load Data & Exclude Canonical Non-Evaluable Queries
    print("\n[1/6] Loading metadata, labels, CV folds, and canonical non-evaluable QIDs...", flush=True)
    train_data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    
    audit_data = json.loads(AUDIT_EXP034.read_text(encoding="utf-8")) if AUDIT_EXP034.exists() else {}
    non_eval_qids = set(map(str, audit_data.get("label_stats", {}).get("non_evaluable_qids", [])))
    print(f"Canonical Non-Evaluable QIDs excluded: {len(non_eval_qids)}")
    
    answers = {qid: set(map(str, train_data[qid]["answer"])) for qid in train_data if str(qid) not in non_eval_qids}

    # Heldout Test Split (Fold 0)
    heldout_0 = [str(q) for q in folds["fold_0"] if str(q) not in non_eval_qids]
    heldout_0_answers = {q: answers[q] for q in heldout_0}
    
    # Inner Splits for Strict Nested CV
    inner_train_qids = [str(q) for f in ["fold_1", "fold_2", "fold_3"] for q in folds[f] if str(q) not in non_eval_qids]
    inner_val_qids = [str(q) for q in folds["fold_4"] if str(q) not in non_eval_qids]
    inner_val_answers = {q: answers[q] for q in inner_val_qids}
    outer_train_qids = inner_train_qids + inner_val_qids

    print(f"Fold 0 Test Held-out queries:  {len(heldout_0)}")
    print(f"Inner Train queries (F1-F3):    {len(inner_train_qids)}")
    print(f"Inner Val queries (Fold 4):    {len(inner_val_qids)}")
    print(f"Outer Train queries (F1-F4):    {len(outer_train_qids)}")

    # Load candidate capsules
    capsules = {}
    capsule_file = ROOT / "cache" / "exp106_cross_encoder_reranker" / "capsules_k50_canonical_dual.jsonl"
    print(f"Loading candidate capsules from: {capsule_file.name}...", flush=True)
    with open(capsule_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = str(item["qid"])
            if qid not in non_eval_qids:
                capsules[qid] = item["capsules"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0)})")
    allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    print(f"Initial GPU Memory: {allocated_mb:.1f} MB allocated, {reserved_mb:.1f} MB reserved.")

    # 2. STEP A: INNER TRAINING (FOLDS 1-3) FOR OUT-OF-SAMPLE WEIGHT SELECTION
    print("\n==========================================================================")
    print("STEP A: INNER TRAINING ON FOLDS 1-3 & PARAMETER SELECTION ON FOLD 4")
    print("==========================================================================")
    inner_train_dataset = TrueResidualListwiseDataset(inner_train_qids, capsules, answers, max_negs=MAX_NEGS)
    print(f"Inner Train Dataset size: {len(inner_train_dataset)} queries")
    
    print("\nTraining Inner Model on Folds 1-3 (2 Epochs)...", flush=True)
    inner_model, inner_tokenizer = train_residual_model(inner_train_dataset, device, epochs=2, alpha=0.35, phase_name="Inner-Train")
    
    print("\nScoring Inner Validation Split (Fold 4, Out-of-Sample)...", flush=True)
    inner_val_scores = score_query_set(inner_model, inner_tokenizer, inner_val_qids, capsules, device, batch_size=16)
    
    # Grid search optimal residual weights on out-of-sample Fold 4
    LAW_NUM_PATTERN = re.compile(r"(\d+[\d/]+[A-ZĐa-zđ\d-]+)", re.IGNORECASE)
    best_val_r5 = -1.0
    best_weights = (0.35, 0.65)
    
    for w_ce in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        for w_s1 in [0.50, 0.55, 0.60, 0.65, 0.70]:
            pred_val = {}
            for qid in inner_val_qids:
                q_text = train_data[qid]["question"]
                q_matches = set(LAW_NUM_PATTERN.findall(q_text))
                c_list = capsules[qid]
                raw_ce = [inner_val_scores[qid].get(c["doc_id"], -20.0) for c in c_list]
                raw_s1 = [c["rrf_score"] for c in c_list]
                z_ce = zscore(raw_ce)
                z_s1 = zscore(raw_s1)
                
                fused = {}
                for i, c in enumerate(c_list):
                    d = c["doc_id"]
                    score = w_ce * z_ce[i] + w_s1 * z_s1[i]
                    if q_matches:
                        for m in q_matches:
                            if len(m) >= 4 and m.lower() in c["capsule_text"].lower():
                                score += 10.0
                                break
                    fused[d] = score
                pred_val[qid] = sorted(fused.keys(), key=lambda d: -fused[d])
                
            m_val = evaluate_rankings(pred_val, inner_val_answers)
            if m_val["recall@5"] > best_val_r5:
                best_val_r5 = m_val["recall@5"]
                best_weights = (w_ce, w_s1)

    w_ce_opt, w_s1_opt = best_weights
    print(f"\n[FROZEN HYPERPARAMETERS SELECTED FROM FOLD 4 OUT-OF-SAMPLE]:")
    print(f"  w_ce = {w_ce_opt:.2f}, w_s1 = {w_s1_opt:.2f} (Inner Val Peak Recall@5: {best_val_r5*100:.2f}%)")

    # Clean up inner model to free VRAM
    del inner_model, inner_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Freed Inner Model VRAM: {torch.cuda.memory_allocated() / (1024 * 1024):.1f} MB allocated.")

    # 3. STEP B: OUTER RETRAINING (FOLDS 1-4) & BLIND TEST ON FOLD 0
    print("\n==========================================================================")
    print("STEP B: OUTER RETRAINING (FOLDS 1-4) & BLIND EVALUATION ON FOLD 0")
    print("==========================================================================")
    outer_train_dataset = TrueResidualListwiseDataset(outer_train_qids, capsules, answers, max_negs=MAX_NEGS)
    print(f"Outer Train Dataset size: {len(outer_train_dataset)} queries")
    
    print(f"\nTraining Final Model on all Outer Folds (F1-F4) with alpha={w_ce_opt:.2f} (2 Epochs)...", flush=True)
    outer_model, outer_tokenizer = train_residual_model(outer_train_dataset, device, epochs=2, alpha=w_ce_opt, phase_name="Outer-Train")
    
    adapter_out = CACHE_DIR / "lora_adapter_fold0_true_residual"
    outer_model.save_pretrained(adapter_out)
    outer_tokenizer.save_pretrained(adapter_out)
    print(f"Saved True Residual LoRA adapter to: {adapter_out}", flush=True)

    print("\nScoring Blind Test Heldout Split (Fold 0)...", flush=True)
    heldout_0_scores = score_query_set(outer_model, outer_tokenizer, heldout_0, capsules, device, batch_size=16)
    
    scores_out = CACHE_DIR / "scores_heldout_fold0_true_residual.json"
    scores_out.write_text(json.dumps(heldout_0_scores, ensure_ascii=False), encoding="utf-8")

    # 4. STEP C: TRANSPARENT ABLATION AUDIT REPORT
    print("\n==========================================================================")
    print("STEP C: OFFICIAL ABLATION REPORT ON HELDOUT FOLD 0 (1,398 QUERIES)")
    print("==========================================================================")

    # 1. Baseline Stage 1 Alone
    s1_pred_0 = {q: [c["doc_id"] for c in capsules[q]] for q in heldout_0}
    m_s1 = evaluate_rankings(s1_pred_0, heldout_0_answers)

    # 2. Stage 1 + Law Identifier Boost Alone (Ablation)
    pred_s1_boost = {}
    for qid in heldout_0:
        q_matches = set(LAW_NUM_PATTERN.findall(train_data[qid]["question"]))
        c_list = capsules[qid]
        z_s1 = zscore([c["rrf_score"] for c in c_list])
        fused = {}
        for i, c in enumerate(c_list):
            score = float(z_s1[i])
            if q_matches:
                for m in q_matches:
                    if len(m) >= 4 and m.lower() in c["capsule_text"].lower():
                        score += 10.0
                        break
            fused[c["doc_id"]] = score
        pred_s1_boost[qid] = sorted(fused.keys(), key=lambda d: -fused[d])
    m_s1_boost = evaluate_rankings(pred_s1_boost, heldout_0_answers)

    # 3. Standalone Cross-Encoder Alone (Ablation)
    pred_ce_raw = {}
    for qid in heldout_0:
        c_list = capsules[qid]
        scored = [(c["doc_id"], heldout_0_scores[qid].get(c["doc_id"], -20.0)) for c in c_list]
        scored.sort(key=lambda x: -x[1])
        pred_ce_raw[qid] = [x[0] for x in scored]
    m_ce_raw = evaluate_rankings(pred_ce_raw, heldout_0_answers)

    # 4. True Residual Synergy Fusion (Stage 1 + Residual CE, WITHOUT Identifier Boost)
    pred_residual_no_boost = {}
    for qid in heldout_0:
        c_list = capsules[qid]
        z_ce = zscore([heldout_0_scores[qid].get(c["doc_id"], -20.0) for c in c_list])
        z_s1 = zscore([c["rrf_score"] for c in c_list])
        fused = {c["doc_id"]: w_ce_opt * z_ce[i] + w_s1_opt * z_s1[i] for i, c in enumerate(c_list)}
        pred_residual_no_boost[qid] = sorted(fused.keys(), key=lambda d: -fused[d])
    m_residual_no_boost = evaluate_rankings(pred_residual_no_boost, heldout_0_answers)

    # 5. Full EXP-107 System
    pred_full = {}
    for qid in heldout_0:
        q_matches = set(LAW_NUM_PATTERN.findall(train_data[qid]["question"]))
        c_list = capsules[qid]
        z_ce = zscore([heldout_0_scores[qid].get(c["doc_id"], -20.0) for c in c_list])
        z_s1 = zscore([c["rrf_score"] for c in c_list])
        fused = {}
        for i, c in enumerate(c_list):
            doc_id = c["doc_id"]
            score = w_ce_opt * z_ce[i] + w_s1_opt * z_s1[i]
            if q_matches:
                for m in q_matches:
                    if len(m) >= 4 and m.lower() in c["capsule_text"].lower():
                        score += 10.0
                        break
            fused[doc_id] = score
        pred_full[qid] = sorted(fused.keys(), key=lambda d: -fused[d])
    m_full = evaluate_rankings(pred_full, heldout_0_answers)

    # Multi-Gold Query Breakdown on Full System
    multi_gold_qids = [q for q in heldout_0 if len(heldout_0_answers[q]) >= 2]
    single_gold_qids = [q for q in heldout_0 if len(heldout_0_answers[q]) == 1]
    
    r5_single = [len(set(pred_full[q][:5]) & heldout_0_answers[q]) / len(heldout_0_answers[q]) for q in single_gold_qids]
    r5_multi = [len(set(pred_full[q][:5]) & heldout_0_answers[q]) / len(heldout_0_answers[q]) for q in multi_gold_qids]

    print("\n==========================================================================")
    print("STEP C: OFFICIAL ABLATION REPORT ON HELDOUT FOLD 0 (1,398 QUERIES)")
    print("==========================================================================")
    
    print("\n--- CLEAN ABLATION BENCHMARK TABLE (FOLD 0) ---")
    print(f"1. Stage-1 Baseline:            Recall@1 = {m_s1['recall@1']*100:.2f}% | Recall@5 = {m_s1['recall@5']*100:.2f}% | Recall@16 = {m_s1['recall@16']*100:.2f}% | MRR@5 = {m_s1['mrr@5']:.4f}")
    print(f"2. Stage-1 + Law Identifier:    Recall@1 = {m_s1_boost['recall@1']*100:.2f}% | Recall@5 = {m_s1_boost['recall@5']*100:.2f}% | Recall@16 = {m_s1_boost['recall@16']*100:.2f}% | MRR@5 = {m_s1_boost['mrr@5']:.4f}")
    print(f"3. Raw Residual CE Alone:       Recall@1 = {m_ce_raw['recall@1']*100:.2f}% | Recall@5 = {m_ce_raw['recall@5']*100:.2f}% | Recall@16 = {m_ce_raw['recall@16']*100:.2f}% | MRR@5 = {m_ce_raw['mrr@5']:.4f}")
    print(f"4. True Residual Synergy (NoId):Recall@1 = {m_residual_no_boost['recall@1']*100:.2f}% | Recall@5 = {m_residual_no_boost['recall@5']*100:.2f}% | Recall@16 = {m_residual_no_boost['recall@16']*100:.2f}% | MRR@5 = {m_residual_no_boost['mrr@5']:.4f}")
    print(f"5. Full EXP-107 Final System:   Recall@1 = {m_full['recall@1']*100:.2f}% | Recall@5 = {m_full['recall@5']*100:.2f}% | Recall@16 = {m_full['recall@16']*100:.2f}% | MRR@5 = {m_full['mrr@5']:.4f}")

    print("\n--- AUDIT GATE CHECKLIST ---")
    print(f"Single-Gold Recall@5 ({len(single_gold_qids)} queries): {np.mean(r5_single)*100:.2f}%")
    print(f"Multi-Gold Recall@5 ({len(multi_gold_qids)} queries):   {np.mean(r5_multi)*100:.2f}% (EXP-106 was 71.82%)")
    print(f"Overall Fold 0 Recall@5:                      {m_full['recall@5']*100:.2f}%")
    
    passed_gate1 = bool(np.mean(r5_multi) >= 0.84)
    passed_gate2 = bool(m_full['recall@5'] >= 0.9420)
    passed_gate3 = bool(m_full['recall@1'] >= 0.6950 and m_full['mrr@5'] >= 0.8150)
    
    print("\nAUDIT RESULTS:")
    print(f"  * Gate 1 (Multi-Gold Recall@5 >= 84.0%): {'PASSED [OK]' if passed_gate1 else 'FAILED [X]'}")
    print(f"  * Gate 2 (Fold 0 Overall Recall@5 >= 94.20%): {'PASSED [OK]' if passed_gate2 else 'FAILED [X]'}")
    print(f"  * Gate 3 (Recall@1 >= 69.50%, MRR@5 >= 0.8150): {'PASSED [OK]' if passed_gate3 else 'FAILED [X]'}")

    report = {
        "exp_id": "EXP-107",
        "methodology": "Strict Nested CV Parameter Selection + True Residual InfoNCE Loss",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fold_0_heldout_count": len(heldout_0),
        "frozen_optimal_weights_from_fold4": {"w_ce": w_ce_opt, "w_s1": w_s1_opt},
        "ablation_results": {
            "stage1_baseline": m_s1,
            "stage1_plus_identifier": m_s1_boost,
            "raw_residual_ce": m_ce_raw,
            "true_residual_synergy_no_id": m_residual_no_boost,
            "full_exp107_system": m_full,
        },
        "multi_gold_recall5": float(np.mean(r5_multi)),
        "single_gold_recall5": float(np.mean(r5_single)),
        "audit_gates": {
            "gate_1_multi_gold_passed": passed_gate1,
            "gate_2_overall_passed": passed_gate2,
            "gate_3_precision_passed": passed_gate3,
            "overall_audit_passed": passed_gate1 and passed_gate2 and passed_gate3
        },
        "total_elapsed_sec": time.time() - start_time
    }
    
    report_file = CACHE_DIR / "REPORT_FOLD0.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved verified audit report to: {report_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
