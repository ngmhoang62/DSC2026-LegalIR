"""Fast 1-epoch LoRA Fine-Tuning for BAAI/bge-reranker-v2-m3 on Hard Negatives with Gradient Checkpointing."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src/ is in sys.path for root module imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from exp012b_core import atomic_json, load_answers, read_jsonl, stage_run
from exp014.core import BGE_RERANKER_MODEL, write_manifest


class LegalPairDataset(Dataset):
    """Pairwise dataset with 1 positive capsule and K hard negative capsules."""

    def __init__(self, records: list[dict[str, Any]], answers: dict[str, set[str]]):
        self.pairs: list[tuple[str, str, float]] = []
        for record in records:
            qid = str(record["qid"])
            if qid not in answers:
                continue
            gold_docs = answers[qid]
            candidates = record["candidates"]
            
            positives = [c for c in candidates if str(c["doc_id"]) in gold_docs]
            negatives = [c for c in candidates if str(c["doc_id"]) not in gold_docs]
            
            if not positives or not negatives:
                continue
            
            pos = positives[0]
            # Top 2 hard negatives for maximum signal-to-noise ratio and fast training
            for neg in negatives[:2]:
                self.pairs.append((str(pos["query"]), str(pos["document"]), 1.0))
                self.pairs.append((str(neg["query"]), str(neg["document"]), 0.0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, doc, label = self.pairs[idx]
        return query, doc, label


def train_lora_bge_reranker(*, capsules_path: Path, train_path: Path, output_dir: Path, v3_fingerprint: str,
                           model_name: str = BGE_RERANKER_MODEL, device: str = "cuda",
                           batch_size: int = 4, gradient_accumulation_steps: int = 4,
                           epochs: int = 1, lr: float = 1e-4, allow_download: bool = True) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    answers = load_answers(train_path)
    records = list(read_jsonl(capsules_path))
    dataset = LegalPairDataset(records, answers)
    
    with stage_run(output_dir, "train-lora", total=len(dataset), v3_fingerprint=v3_fingerprint) as logger:
        logger.log(f"phase=init_model model={model_name} total_pairs={len(dataset)} batch_size={batch_size} accum_steps={gradient_accumulation_steps}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not allow_download, trust_remote_code=True)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, local_files_only=not allow_download, trust_remote_code=True
        )
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable()
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(base_model, lora_config)
        model.to(device)
        model.train()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")
        criterion = nn.BCEWithLogitsLoss()
        
        started = time.perf_counter()
        total_steps = len(dataloader) * epochs
        step = 0
        loss_val = 0.0
        optimizer.zero_grad()
        
        for epoch in range(epochs):
            for batch_idx, (queries, docs, labels) in enumerate(dataloader, 1):
                step += 1
                encoded = tokenizer(
                    list(queries), list(docs), padding=True, truncation=True, max_length=512, return_tensors="pt"
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                targets = torch.as_tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
                
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(**encoded).logits
                    loss = criterion(logits, targets) / gradient_accumulation_steps
                    
                scaler.scale(loss).backward()
                
                if batch_idx % gradient_accumulation_steps == 0 or batch_idx == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                loss_val = float(loss.item()) * gradient_accumulation_steps
                if step % (10 * gradient_accumulation_steps) == 0 or step == total_steps:
                    rate = step * batch_size / max(time.perf_counter() - started, 1e-9)
                    logger.log(f"epoch={epoch}/{epochs} step={step}/{total_steps} loss={loss_val:.4f} rate={rate:.1f}_pairs/s")
                    
        adapter_dir = output_dir / "bge_lora_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        
        report = {"model": model_name, "epochs": epochs, "pairs": len(dataset), "final_loss": loss_val}
        atomic_json(output_dir / "lora_report.json", report)
        logger.log(f"LoRA fine-tuning complete! Adapter saved to {adapter_dir}")
        
        del model, base_model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            
        return write_manifest(output_dir, stage="train-lora", v3_fingerprint=v3_fingerprint,
                              config={"model": model_name, "epochs": epochs, "batch_size": batch_size, "effective_batch_size": batch_size * gradient_accumulation_steps, "lr": lr},
                              files=[output_dir / "lora_report.json"], counts={"pairs": len(dataset)})
