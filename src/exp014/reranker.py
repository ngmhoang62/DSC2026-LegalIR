"""Official Cross-Encoder Neural Reranking (BGE-Reranker-v2-m3 & LoRA) with bounded batches and resumable JSONL output."""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src/ is in sys.path for root module imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from exp012b_core import atomic_json, read_jsonl, sha256_file, stage_run, write_jsonl
from exp014.core import DEFAULT_RERANKER, hash_payload, write_manifest


def _load_reranker(*, model_name: str, device: str, allow_download: bool):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", local_files_only=not allow_download, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float16,
        local_files_only=not allow_download, trust_remote_code=True).to(device).eval()
    return model, tokenizer


def score_reranker(*, capsules_path: Path, output_dir: Path, v3_fingerprint: str, model_name: str = DEFAULT_RERANKER, device: str = "cuda",
                   allow_download: bool = True, max_length: int = 768, batch_size: int = 16, fold_qids: set[str] | None = None, resume: bool = False) -> dict[str, Any]:
    import torch
    records = [record for record in read_jsonl(capsules_path) if fold_qids is None or str(record["qid"]) in fold_qids]
    capsule_sha = sha256_file(capsules_path)
    score_fingerprint = hash_payload({"capsules": capsule_sha, "model": model_name, "max_length": max_length,
                                      "batch_size": batch_size, "qids": sorted(str(row["qid"]) for row in records)})
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scores.jsonl"
    existing = {str(row["qid"]): row for row in read_jsonl(path) if row.get("score_fingerprint") == score_fingerprint} if resume and path.exists() else {}
    
    with stage_run(output_dir, "score-reranker", total=len(records), v3_fingerprint=v3_fingerprint) as logger:
        expected_qids = {str(row["qid"]) for row in records}
        if set(existing) == expected_qids:
            write_jsonl(path, [existing[qid] for qid in sorted(existing)])
            logger.log(f"resume_cache_hit queries={len(existing)} model_load_skipped=true")
            return write_manifest(output_dir, stage="score-reranker", v3_fingerprint=v3_fingerprint,
                config={"model": model_name, "max_length": max_length, "batch_size": batch_size},
                inputs={"capsules_sha256": capsule_sha}, files=[path], counts={"queries": len(existing)})
                
        model, tokenizer = _load_reranker(model_name=model_name, device=device, allow_download=allow_download)
        result = dict(existing)
        started = time.perf_counter()
        completed = 0
        
        for position, record in enumerate(records, 1):
            qid = str(record["qid"])
            if qid in result:
                continue
            candidates = record["candidates"]
            pairs = [(str(candidate["query"]), str(candidate["document"])) for candidate in candidates]
            scores: list[float] = []
            
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start:start + batch_size]
                encoded = tokenizer([p[0] for p in chunk], [p[1] for p in chunk], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.inference_mode():
                    logits = model(**encoded).logits.reshape(-1).cpu().tolist()
                    scores.extend(logits)
                
            scored_rows = [{"doc_id": str(candidate["doc_id"]), "lambda_rank": int(candidate["lambda_rank"]),
                            "lambda_score": float(candidate["lambda_score"]), "qwen_score": float(score)}
                           for candidate, score in zip(candidates, scores)]
            result[qid] = {"schema_version": "legalir.exp014.scores.v1", "v3_fingerprint": v3_fingerprint,
                           "score_fingerprint": score_fingerprint, "qid": qid, "scores": scored_rows}
            completed += 1
            
            if position % 64 == 0 or position == len(records):
                logger.status(stage="score-reranker", state="RUNNING", completed=position, total=len(records))
            if position % 64 == 0 or position == len(records):
                write_jsonl(path, [result[known_qid] for known_qid in sorted(result)])
            if position % 128 == 0 or position == len(records):
                rate = completed * len(candidates) / max(time.perf_counter() - started, 1e-9)
                logger.log(f"progress={position}/{len(records)} rate={rate:.2f}_pairs/s eta_seconds={(len(records)-position)*len(candidates)/max(rate,1e-9):.0f}")
                
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            
        records_out = [result[qid] for qid in sorted(result)]
        write_jsonl(path, records_out)
        return write_manifest(output_dir, stage="score-reranker", v3_fingerprint=v3_fingerprint,
            config={"model": model_name, "max_length": max_length, "batch_size": batch_size},
            inputs={"capsules_sha256": capsule_sha}, files=[path],
            counts={"queries": len(records_out), "pairs": sum(len(row["scores"]) for row in records_out)})


def score_bge_lora(*, capsules_path: Path, adapter_path: Path, output_dir: Path, v3_fingerprint: str,
                   device: str = "cuda", allow_download: bool = True, max_length: int = 768, batch_size: int = 16,
                   fold_qids: set[str] | None = None) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    records = [record for record in read_jsonl(capsules_path) if fold_qids is None or str(record["qid"]) in fold_qids]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scores.jsonl"
    
    with stage_run(output_dir, "score-bge-lora", total=len(records), v3_fingerprint=v3_fingerprint) as logger:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=not allow_download, trust_remote_code=True)
        base = AutoModelForSequenceClassification.from_pretrained(DEFAULT_RERANKER, dtype=torch.float16, local_files_only=not allow_download, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(adapter_path)).to(device).eval()
        
        results = []
        started = time.perf_counter()
        for position, record in enumerate(records, 1):
            qid = str(record["qid"])
            candidates = record["candidates"]
            pairs = [(str(c["query"]), str(c["document"])) for c in candidates]
            scores: list[float] = []
            
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start:start + batch_size]
                encoded = tokenizer([p[0] for p in chunk], [p[1] for p in chunk], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.inference_mode():
                    logits = model(**encoded).logits.reshape(-1).cpu().tolist()
                    scores.extend(logits)
                    
            scored_rows = [{"doc_id": str(c["doc_id"]), "bge_lora_score": float(s), "qwen_score": float(s),
                            "lambda_rank": int(c.get("lambda_rank", 1)), "lambda_score": float(c.get("lambda_score", 0.0))}
                           for c, s in zip(candidates, scores)]
            results.append({"qid": qid, "scores": scored_rows})
            
            if position % 128 == 0 or position == len(records):
                rate = position * len(candidates) / max(time.perf_counter() - started, 1e-9)
                logger.log(f"progress={position}/{len(records)} rate={rate:.2f}_pairs/s")
                
        write_jsonl(path, results)
        del model, base
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            
        return write_manifest(output_dir, stage="score-bge-lora", v3_fingerprint=v3_fingerprint,
                              config={"adapter": str(adapter_path), "max_length": max_length, "batch_size": batch_size},
                              files=[path], counts={"queries": len(results)})
