"""Official Qwen3-Reranker scoring with bounded batches and resumable JSONL output."""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from exp012b_core import atomic_json, read_jsonl, sha256_file, stage_run, write_jsonl
from exp013b_core import QWEN_INSTRUCTION, QWEN_MODEL, hash_payload, write_manifest


PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def _load_qwen(*, model_name: str, device: str, allow_download: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", local_files_only=not allow_download, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    if model_name != QWEN_MODEL:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float16,
            local_files_only=not allow_download, trust_remote_code=True).to(device).eval()
        if hasattr(model, "new") and hasattr(model.new, "embeddings") and hasattr(model.new.embeddings, "position_ids"):
            max_pos = getattr(model.config, "max_position_embeddings", 8192)
            model.new.embeddings.position_ids = torch.arange(max_pos, device=device)
        return model, tokenizer, None, None
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, attn_implementation="sdpa",
                                                  local_files_only=not allow_download, trust_remote_code=True).to(device).eval()
    yes = tokenizer.convert_tokens_to_ids("yes")
    no = tokenizer.convert_tokens_to_ids("no")
    if not isinstance(yes, int) or not isinstance(no, int) or yes < 0 or no < 0:
        raise RuntimeError("Qwen tokenizer does not expose yes/no score tokens")
    return model, tokenizer, yes, no


def _pairs_to_inputs(tokenizer: Any, pairs: list[tuple[str, str]], *, max_length: int, device: str):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    values = [f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}" for query, document in pairs]
    encoded = tokenizer(values, padding=False, truncation=True, max_length=max_length - len(prefix) - len(suffix), return_attention_mask=False)
    rows = [{"input_ids": prefix + ids + suffix} for ids in encoded["input_ids"]]
    padded = tokenizer.pad(rows, padding=True, return_tensors="pt", max_length=max_length)
    return {name: value.to(device) for name, value in padded.items()}


def _score(model: Any, tokenizer: Any, yes: int | None, no: int | None, pairs: list[tuple[str, str]], *, max_length: int, device: str) -> list[float]:
    import torch
    if yes is None or no is None:
        encoded = tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            return model(**encoded).logits.reshape(-1).detach().float().cpu().tolist()
    inputs = _pairs_to_inputs(tokenizer, pairs, max_length=max_length, device=device)
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, :]
        pair = torch.stack((logits[:, no], logits[:, yes]), dim=1)
        return torch.nn.functional.log_softmax(pair, dim=1)[:, 1].detach().float().cpu().tolist()


def prepare_qwen(*, output_dir: Path, v3_fingerprint: str, model_name: str, device: str, allow_download: bool) -> dict[str, Any]:
    import torch
    with stage_run(output_dir, "prepare-qwen", total=1, v3_fingerprint=v3_fingerprint) as logger:
        model, tokenizer, yes, no = _load_qwen(model_name=model_name, device=device, allow_download=allow_download)
        report = {"model": model_name, "tokenizer": type(tokenizer).__name__, "model_class": type(model).__name__,
                  "yes_token_id": yes, "no_token_id": no, "instruction": QWEN_INSTRUCTION,
                  "parameters": sum(parameter.numel() for parameter in model.parameters())}
        path = output_dir / "model_report.json"
        atomic_json(path, report)
        del model
        if device.startswith("cuda"): torch.cuda.empty_cache()
        return write_manifest(output_dir, stage="prepare-qwen", v3_fingerprint=v3_fingerprint,
            config={"model": model_name, "backend": "qwen_yes_no" if yes is not None else "gte_sequence_classification", "attn_implementation": "sdpa" if yes is not None else None, "dtype": "float16"}, files=[path], counts={"models": 1})


def benchmark_qwen(*, capsules_path: Path, output_dir: Path, v3_fingerprint: str, model_name: str, device: str,
                   allow_download: bool, pairs: int = 2048, max_length: int = 768) -> dict[str, Any]:
    import torch
    data = [(str(row["query"]), str(row["document"])) for record in read_jsonl(capsules_path) for row in record["candidates"]][:pairs]
    if not data: raise ValueError("no capsule pairs available")
    attempts: list[dict[str, Any]] = []
    with stage_run(output_dir, "benchmark-qwen", total=len(data), v3_fingerprint=v3_fingerprint) as logger:
        model, tokenizer, yes, no = _load_qwen(model_name=model_name, device=device, allow_download=allow_download)
        selected = None
        for length in (max_length, 512) if max_length != 512 else (512,):
            safe_batch = None
            for batch in (16, 12, 8, 6, 4, 2, 1):
                try:
                    if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(device)
                    warmup = data[:min(len(data), max(64, batch))]
                    for start in range(0, len(warmup), batch):
                        _score(model, tokenizer, yes, no, warmup[start:start + batch], max_length=length, device=device)
                    peak = int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0
                    attempts.append({"phase": "warmup", "max_length": length, "batch_size": batch, "peak_bytes": peak})
                    if peak <= int(5.5 * 1024**3): safe_batch = batch; break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    attempts.append({"phase": "warmup", "max_length": length, "batch_size": batch, "oom": True})
            if safe_batch is None: continue
            if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter(); done = 0
            for start in range(0, len(data), safe_batch):
                _score(model, tokenizer, yes, no, data[start:start + safe_batch], max_length=length, device=device)
                done += min(safe_batch, len(data) - start)
            seconds = time.perf_counter() - started
            peak = int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0
            result = {"phase": "full", "max_length": length, "batch_size": safe_batch,
                      "pairs_per_second": done / max(seconds, 1e-9), "peak_bytes": peak}
            attempts.append(result)
            if peak <= int(5.5 * 1024**3) and result["pairs_per_second"] >= 15:
                selected = result
            if selected: break
        del model; gc.collect()
        if device.startswith("cuda"): torch.cuda.empty_cache()
        report = {"status": "PASS" if selected else "FAIL", "selected": selected, "attempts": attempts,
                  "fallback": "Alibaba-NLP/gte-multilingual-reranker-base" if not selected else None}
        path = output_dir / "benchmark.json"; atomic_json(path, report)
        if not selected: raise RuntimeError("Qwen benchmark did not satisfy 15 pairs/s and 5.5GB gates; use --model Alibaba-NLP/gte-multilingual-reranker-base")
        logger.log(f"batch={selected['batch_size']} length={selected['max_length']} throughput={selected['pairs_per_second']:.2f}/s")
        return write_manifest(output_dir, stage="benchmark-qwen", v3_fingerprint=v3_fingerprint, config={"pairs": len(data), "model": model_name}, files=[path], counts={"pairs": len(data)})


def score_qwen(*, capsules_path: Path, output_dir: Path, v3_fingerprint: str, model_name: str, device: str,
               allow_download: bool, max_length: int, batch_size: int, fold_qids: set[str] | None = None, resume: bool = False) -> dict[str, Any]:
    import torch
    records = [record for record in read_jsonl(capsules_path) if fold_qids is None or str(record["qid"]) in fold_qids]
    capsule_sha = sha256_file(capsules_path)
    score_fingerprint = hash_payload({"capsules": capsule_sha, "model": model_name, "max_length": max_length,
                                      "batch_size": batch_size, "instruction": QWEN_INSTRUCTION,
                                      "qids": sorted(str(row["qid"]) for row in records)})
    path = output_dir / "scores.jsonl"
    existing = {str(row["qid"]): row for row in read_jsonl(path) if row.get("score_fingerprint") == score_fingerprint} if resume and path.exists() else {}
    with stage_run(output_dir, "score-qwen", total=len(records), v3_fingerprint=v3_fingerprint) as logger:
        expected_qids = {str(row["qid"]) for row in records}
        if set(existing) == expected_qids:
            write_jsonl(path, [existing[qid] for qid in sorted(existing)])
            logger.log(f"resume_cache_hit queries={len(existing)} model_load_skipped=true")
            return write_manifest(output_dir, stage="score-qwen", v3_fingerprint=v3_fingerprint,
                config={"model": model_name, "max_length": max_length, "requested_batch_size": batch_size,
                        "actual_batch_size": batch_size, "length_buckets": True, "instruction": QWEN_INSTRUCTION},
                inputs={"capsules_sha256": capsule_sha, "routed_qids_fingerprint": hash_payload(sorted(expected_qids))},
                files=[path], counts={"queries": len(existing), "pairs": sum(len(row["scores"]) for row in existing.values())})
        model, tokenizer, yes, no = _load_qwen(model_name=model_name, device=device, allow_download=allow_download)
        result = dict(existing); started = time.perf_counter(); completed = 0; actual_batch = batch_size
        for record in records:
            qid = str(record["qid"])
            if qid in result:
                completed += 1; continue
            candidates = record["candidates"]
            order = sorted(range(len(candidates)), key=lambda index: (len(tokenizer(candidates[index]["query"], candidates[index]["document"], add_special_tokens=False)["input_ids"]), index))
            score_by_index: dict[int, float] = {}; cursor = 0
            while cursor < len(order):
                indices = order[cursor:cursor + actual_batch]
                try:
                    values = _score(model, tokenizer, yes, no,
                        [(candidates[index]["query"], candidates[index]["document"]) for index in indices],
                        max_length=max_length, device=device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if actual_batch == 1: raise
                    actual_batch = max(1, actual_batch // 2); logger.log(f"oom_retry new_batch_size={actual_batch}"); continue
                score_by_index.update(zip(indices, values)); cursor += len(indices)
            scores = [score_by_index[index] for index in range(len(candidates))]
            if len(scores) != len(candidates) or not np.isfinite(scores).all(): raise RuntimeError(f"invalid Qwen scores qid={qid}")
            result[qid] = {"qid": qid, "score_fingerprint": score_fingerprint,
                           "scores": [{"doc_id": str(row["doc_id"]), "qwen_score": float(score), "lambda_rank": int(row["lambda_rank"])} for row, score in zip(candidates, scores)]}
            completed += 1
            if completed % 64 == 0 or completed == len(records):
                write_jsonl(path, [result[known_qid] for known_qid in sorted(result)])
                logger.status(stage="score-qwen", state="RUNNING", completed=completed, total=len(records))
            if completed % 256 == 0 or completed == len(records):
                rate = completed / max(time.perf_counter() - started, 1e-9); logger.log(f"progress={completed}/{len(records)} rate={rate:.2f}_queries_per_second eta_seconds={(len(records)-completed)/max(rate,1e-9):.0f}")
        write_jsonl(path, [result[qid] for qid in sorted(result)])
        del model; gc.collect()
        if device.startswith("cuda"): torch.cuda.empty_cache()
        return write_manifest(output_dir, stage="score-qwen", v3_fingerprint=v3_fingerprint,
            config={"model": model_name, "max_length": max_length, "requested_batch_size": batch_size,
                    "actual_batch_size": actual_batch, "length_buckets": True, "instruction": QWEN_INSTRUCTION},
            inputs={"capsules_sha256": capsule_sha, "routed_qids_fingerprint": hash_payload(sorted(str(row["qid"]) for row in records))},
            files=[path], counts={"queries": len(result), "pairs": sum(len(row["scores"]) for row in result.values())})
