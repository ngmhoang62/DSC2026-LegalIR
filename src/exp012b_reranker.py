"""Cached zero-shot reranking and fold-safe pairwise LoRA training."""

from __future__ import annotations

import gc
import itertools
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from exp012b_core import (
    PIPELINE_SCHEMA,
    atomic_json,
    canonical_json,
    content_hash,
    read_jsonl,
    sha256_file,
    stage_run,
)
from exp012b_retrieval import weighted_rrf


DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
SCORING_STRATEGY = "length_bucketed_adaptive_v1"


def _evidence_offsets(
    evidence_path: Path,
    source_fingerprint: str,
    logger: Any,
) -> list[dict[str, Any]]:
    """Build a tiny QID-to-byte-offset cache once for fold-filtered scoring."""
    index_path = evidence_path.with_name(f"{evidence_path.stem}_qid_offsets.json")
    source_size = evidence_path.stat().st_size
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            payload.get("source_fingerprint") == source_fingerprint
            and int(payload.get("source_size", -1)) == source_size
        ):
            return list(payload["records"])
    logger.log(f"phase=index-evidence source_bytes={source_size}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with evidence_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in seen:
                raise ValueError(f"Duplicate evidence qid: {qid}")
            seen.add(qid)
            records.append({"qid": qid, "offset": offset, "ordinal": len(records)})
            if len(records) % 256 == 0:
                logger.status(
                    stage="score-lora-fold", state="RUNNING", phase="index-evidence",
                    completed=handle.tell(), total=source_size,
                )
    atomic_json(
        index_path,
        {
            "schema_version": f"{PIPELINE_SCHEMA}.qid_offsets.v1",
            "source_fingerprint": source_fingerprint,
            "source_size": source_size,
            "records": records,
        },
    )
    return records


def _filtered_evidence_records(
    evidence_path: Path,
    qids_filter: set[str] | None,
    source_fingerprint: str,
    logger: Any,
) -> Iterator[dict[str, Any]]:
    if qids_filter is None:
        yield from read_jsonl(evidence_path)
        return
    offsets = _evidence_offsets(evidence_path, source_fingerprint, logger)
    selected = [row for row in offsets if row["qid"] in qids_filter]
    missing = sorted(qids_filter - {row["qid"] for row in selected})
    if missing:
        raise ValueError(f"Missing evidence qids: {missing[:10]}")
    with evidence_path.open("rb") as handle:
        for item in selected:
            handle.seek(int(item["offset"]))
            row = json.loads(handle.readline())
            if str(row.get("qid")) != item["qid"]:
                raise RuntimeError(f"Stale evidence offset for {item['qid']}")
            yield row


def unload_cuda(*objects: Any) -> None:
    for value in objects:
        del value
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load_reranker(model_name: str, device: str):
    # Every pipeline model is expected to exist in the local HF cache. This
    # also prevents Transformers' PEFT adapter probe from retrying the network.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=True,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa" if device.startswith("cuda") else "eager",
    ).to(device)
    model.eval()
    return model, tokenizer


def _score_pairs_with_stats(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    *,
    device: str,
    batch_size: int = 24,
    max_length: int = 512,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    import torch

    if batch_size < 2:
        raise ValueError("Reranker batch size must be at least 2")
    started = time.perf_counter()
    actual_batch = batch_size
    query_cache: dict[str, str] = {}
    for query, _ in pairs:
        if query not in query_cache:
            ids = tokenizer(
                query, add_special_tokens=False, return_attention_mask=False
            )["input_ids"][:128]
            query_cache[query] = tokenizer.decode(ids, skip_special_tokens=True).strip()
    # Tokenize once per shard. Keeping only input_ids is compact enough for a
    # 64-query shard and removes duplicate CPU tokenization before every GPU
    # batch. Stable length sorting substantially reduces padding FLOPs.
    tokenized = tokenizer(
        [query_cache[query] for query, _ in pairs],
        [passage for _, passage in pairs],
        add_special_tokens=True,
        truncation="only_second",
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    lengths = [len(input_ids) for input_ids in tokenized]
    order = sorted(range(len(tokenized)), key=lambda index: (lengths[index], index))
    output = np.empty(len(pairs), dtype=np.float32)
    cursor = 0
    padded_tokens = 0
    useful_tokens = 0
    maximum_batch = 0

    def target_batch(length: int) -> int:
        if length <= 256:
            return actual_batch * 2
        if length <= 384:
            return max(actual_batch, actual_batch * 4 // 3)
        return actual_batch

    with torch.inference_mode():
        while cursor < len(order):
            selected = order[cursor : cursor + target_batch(lengths[order[cursor]])]
            try:
                encoded = tokenizer.pad(
                    [{"input_ids": tokenized[index]} for index in selected],
                    padding=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(device, non_blocking=device.startswith("cuda"))
                    for key, value in encoded.items()
                }
                logits = model(**encoded, return_dict=True).logits.reshape(-1).float().cpu().numpy()
                if np.any(~np.isfinite(logits)):
                    raise ValueError("Reranker produced NaN/Inf")
                output[selected] = logits
                maximum_batch = max(maximum_batch, len(selected))
                useful_tokens += sum(lengths[index] for index in selected)
                padded_tokens += len(selected) * int(encoded["input_ids"].shape[1])
                cursor += len(selected)
            except RuntimeError as error:
                if "out of memory" not in str(error).casefold() or actual_batch <= 2:
                    raise
                actual_batch = max(2, actual_batch // 2)
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    ordered_lengths = sorted(lengths)
    percentile = lambda fraction: (
        ordered_lengths[min(len(ordered_lengths) - 1, round((len(ordered_lengths) - 1) * fraction))]
        if ordered_lengths else 0
    )
    stats = {
        "strategy": SCORING_STRATEGY,
        "seconds": elapsed,
        "pairs_per_second": len(pairs) / elapsed if elapsed else 0.0,
        "useful_tokens_per_second": useful_tokens / elapsed if elapsed else 0.0,
        "padding_fraction": (
            (padded_tokens - useful_tokens) / padded_tokens if padded_tokens else 0.0
        ),
        "length_p50": percentile(0.50),
        "length_p95": percentile(0.95),
        "maximum_batch_used": maximum_batch,
    }
    return output, actual_batch, stats


def score_pairs(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    *,
    device: str,
    batch_size: int = 24,
    max_length: int = 512,
) -> tuple[np.ndarray, int]:
    scores, actual_batch, _ = _score_pairs_with_stats(
        model,
        tokenizer,
        pairs,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    return scores, actual_batch


def _score_input_ids_with_stats(
    model: Any,
    tokenizer: Any,
    tokenized: Sequence[Sequence[int]],
    *,
    device: str,
    batch_size: int = 24,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Score already-tokenized pairs while preserving stable output order."""
    import torch

    started = time.perf_counter()
    actual_batch = batch_size
    lengths = [len(row) for row in tokenized]
    order = sorted(range(len(tokenized)), key=lambda index: (lengths[index], index))
    output = np.empty(len(tokenized), dtype=np.float32)
    cursor = padded_tokens = useful_tokens = maximum_batch = 0

    def target_batch(length: int) -> int:
        if length <= 256:
            return actual_batch * 2
        if length <= 384:
            return max(actual_batch, actual_batch * 4 // 3)
        return actual_batch

    with torch.inference_mode():
        while cursor < len(order):
            selected = order[cursor : cursor + target_batch(lengths[order[cursor]])]
            try:
                encoded = tokenizer.pad(
                    [
                        {
                            # Packed token caches are NumPy memmaps. Convert
                            # scalar IDs at the tokenizer boundary because
                            # Transformers rejects numpy.int64 values even
                            # though PyTorch accepts them.
                            "input_ids": [int(token_id) for token_id in tokenized[index]]
                        }
                        for index in selected
                    ],
                    padding=True, pad_to_multiple_of=8, return_tensors="pt",
                )
                encoded = {key: value.to(device, non_blocking=device.startswith("cuda")) for key, value in encoded.items()}
                logits = model(**encoded, return_dict=True).logits.reshape(-1).float().cpu().numpy()
                if np.any(~np.isfinite(logits)):
                    raise ValueError("Reranker produced NaN/Inf")
                output[selected] = logits
                maximum_batch = max(maximum_batch, len(selected))
                useful_tokens += sum(lengths[index] for index in selected)
                padded_tokens += len(selected) * int(encoded["input_ids"].shape[1])
                cursor += len(selected)
            except RuntimeError as error:
                if "out of memory" not in str(error).casefold() or actual_batch <= 2:
                    raise
                actual_batch = max(2, actual_batch // 2)
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    ordered_lengths = sorted(lengths)
    percentile = lambda fraction: ordered_lengths[min(len(ordered_lengths) - 1, round((len(ordered_lengths) - 1) * fraction))] if ordered_lengths else 0
    return output, actual_batch, {
        "strategy": SCORING_STRATEGY + "+pretokenized",
        "seconds": elapsed,
        "pairs_per_second": len(tokenized) / elapsed if elapsed else 0.0,
        "useful_tokens_per_second": useful_tokens / elapsed if elapsed else 0.0,
        "padding_fraction": (padded_tokens - useful_tokens) / padded_tokens if padded_tokens else 0.0,
        "length_p50": percentile(0.50), "length_p95": percentile(0.95),
        "maximum_batch_used": maximum_batch,
    }


def score_evidence_records(
    evidence_path: Path,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_RERANKER,
    device: str = "cuda",
    batch_size: int = 24,
    shard_queries: int = 64,
    model: Any | None = None,
    tokenizer: Any | None = None,
    resume: bool = False,
    qids_filter: set[str] | None = None,
    stage_name: str = "score-zero-shot",
    model_loader: Callable[[], tuple[Any, Any]] | None = None,
    execution_metadata: Mapping[str, Any] | None = None,
    token_cache_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_manifest_path = evidence_path.parent / "manifest.json"
    evidence_fingerprint = None
    evidence_manifest: dict[str, Any] = {}
    if evidence_manifest_path.exists():
        evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
        evidence_fingerprint = evidence_manifest.get("content_fingerprint")

    token_reader = None
    if token_cache_dir is not None:
        from exp012b_token_cache import TokenCacheReader

        token_reader = TokenCacheReader(token_cache_dir)
        if token_reader.manifest.get("source_fingerprint") != evidence_fingerprint:
            raise RuntimeError("Token cache was built from different evidence")
    if token_reader is not None and qids_filter is None:
        record_count = int(token_reader.manifest["counts"]["queries"])
    elif qids_filter is None and evidence_manifest.get("queries") is not None:
        record_count = int(evidence_manifest["queries"])
    elif qids_filter is not None:
        record_count = len(qids_filter)
    else:
        record_count = sum(1 for _ in read_jsonl(evidence_path))
    own_model = model is None
    source_fingerprint = (
        token_reader.manifest["content_fingerprint"]
        if token_reader is not None else evidence_fingerprint or sha256_file(evidence_path)
    )
    model_path = Path(model_name)
    model_identity: Any = model_name
    if model_path.is_dir():
        adapter_files = [
            path for path in (
                model_path / "adapter_config.json",
                model_path / "adapter_model.safetensors",
                model_path / "adapter_model.bin",
            ) if path.exists()
        ]
        model_identity = {
            "path": str(model_path.resolve()),
            "files": {path.name: sha256_file(path) for path in adapter_files},
        }
    config_hash = content_hash(
        {
            "evidence_fingerprint": source_fingerprint,
            "model": model_identity,
            "max_length": 512,
            "shard_queries": shard_queries,
            "base_batch_size": batch_size,
            "scoring_strategy": SCORING_STRATEGY,
            "pad_to_multiple_of": 8,
            "stage_name": stage_name,
            "qids_filter_hash": content_hash(sorted(qids_filter)) if qids_filter is not None else None,
            "execution": dict(execution_metadata or {}),
        }
    )
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    completed_shards: list[Path] = []
    with stage_run(output_dir, stage_name, total=record_count) as logger:
        logger.log(
            f"phase=prepare-evidence queries={record_count} filtered={qids_filter is not None}"
        )
        record_iterator = iter(
            token_reader.query_records(qids_filter)
            if token_reader is not None
            else _filtered_evidence_records(evidence_path, qids_filter, source_fingerprint, logger)
        )
        # Prime the generator before loading the GPU model. For fold-filtered
        # scoring this builds/validates the 1.3GB evidence offset index while
        # VRAM is still free, then puts the first record back unchanged.
        try:
            first_record = next(record_iterator)
        except StopIteration:
            if record_count:
                raise RuntimeError("Evidence source is empty")
        else:
            record_iterator = itertools.chain((first_record,), record_iterator)
        if model is None or tokenizer is None:
            logger.log(f"phase=load-model model={model_name}")
            if model_loader is not None:
                model, tokenizer = model_loader()
            else:
                model, tokenizer = _load_reranker(model_name, device)
            logger.log("phase=score model_loaded=true")
        actual_batch = batch_size
        scoring_seconds = 0.0
        scored_pairs = 0
        try:
          for shard_index, start in enumerate(range(0, record_count, shard_queries)):
              shard_records = []
              for _ in range(min(shard_queries, record_count - start)):
                  try:
                      shard_records.append(next(record_iterator))
                  except StopIteration as error:
                      raise RuntimeError(
                          f"Evidence query count is below manifest/filter count at row {start}"
                      ) from error
              shard_path = shard_dir / f"scores_{shard_index:04d}.jsonl"
              marker_path = shard_dir / f"scores_{shard_index:04d}.json"
              if resume and shard_path.exists() and marker_path.exists():
                  marker = json.loads(marker_path.read_text(encoding="utf-8"))
                  if (
                      marker.get("config_hash") == config_hash
                      and marker.get("sha256") == sha256_file(shard_path)
                  ):
                      completed_shards.append(shard_path)
                      previous = marker.get("score_stats", {})
                      scoring_seconds += float(previous.get("seconds", 0.0))
                      scored_pairs += int(marker.get("pairs", 0))
                      actual_batch = min(
                          actual_batch, int(marker.get("actual_batch_size", actual_batch))
                      )
                      logger.status(
                          stage=stage_name, state="RUNNING", phase="score",
                          completed=start + len(shard_records), total=record_count,
                          shard=shard_index, resumed=True,
                      )
                      continue
              pairs: list[tuple[str, str]] = []
              cached_input_ids: list[Sequence[int]] = []
              identities: list[tuple[str, str, dict[str, Any]]] = []
              seen: set[tuple[str, str]] = set()
              for record in shard_records:
                  qid = str(record["qid"])
                  if token_reader is not None:
                      pair_rows = record["pairs"]
                      iterator = (({"doc_id": row["doc_id"]}, row) for row in pair_rows)
                  else:
                      query = record["query"]
                      iterator = (
                          (candidate, evidence)
                          for candidate in record["candidates"]
                          for evidence in candidate["evidence"]
                      )
                  for candidate, evidence in iterator:
                          key = (qid, evidence["chunk_id"])
                          if key in seen:
                              continue
                          seen.add(key)
                          if token_reader is not None:
                              cached_input_ids.append(evidence["input_ids"])
                          else:
                              pairs.append((query, evidence["bundle_text"]))
                          identities.append((qid, candidate["doc_id"], evidence))
              if token_reader is not None:
                  logits, actual_batch, score_stats = _score_input_ids_with_stats(
                      model, tokenizer, cached_input_ids, device=device, batch_size=actual_batch
                  )
              else:
                  logits, actual_batch, score_stats = _score_pairs_with_stats(
                      model, tokenizer, pairs, device=device,
                      batch_size=actual_batch, max_length=512,
                  )
              scoring_seconds += float(score_stats["seconds"])
              scored_pairs += len(identities)
              with shard_path.open("w", encoding="utf-8", newline="\n") as handle:
                  for (qid, doc_id, evidence), score in zip(identities, logits):
                      handle.write(
                          canonical_json(
                              {
                                  "schema_version": PIPELINE_SCHEMA,
                                  "qid": qid,
                                  "doc_id": doc_id,
                                  "chunk_id": evidence["chunk_id"],
                                  "bundle_hash": evidence.get("bundle_hash") or content_hash(evidence["bundle_text"]),
                                  "score": float(score),
                              }
                          )
                          + "\n"
                      )
              atomic_json(
                  marker_path,
                  {
                      "config_hash": config_hash,
                      "shard": shard_index,
                      "query_start": start,
                      "query_end": start + len(shard_records),
                      "pairs": len(identities),
                      "actual_batch_size": actual_batch,
                      "score_stats": score_stats,
                      "sha256": sha256_file(shard_path),
                  },
              )
              completed_shards.append(shard_path)
              logger.status(
                  stage=stage_name,
                  state="RUNNING",
                  phase="score",
                  completed=start + len(shard_records),
                  total=record_count,
                  shard=shard_index,
                  pairs=len(pairs),
                  actual_batch_size=actual_batch,
                  pairs_scored=scored_pairs,
                  pairs_per_second=(scored_pairs / scoring_seconds if scoring_seconds else 0.0),
                  eta_seconds=(
                      (record_count - start - len(shard_records))
                      * (scoring_seconds / (start + len(shard_records)))
                      if start + len(shard_records) and scoring_seconds else None
                  ),
              )
              logger.log(
                  f"shard={shard_index} queries={len(shard_records)} pairs={len(identities)} "
                  f"pairs_per_second={score_stats['pairs_per_second']:.2f} "
                  f"padding_fraction={score_stats['padding_fraction']:.3f}"
              )
          try:
              next(record_iterator)
          except StopIteration:
              pass
          else:
              raise RuntimeError("Evidence query count exceeds manifest/filter count")
        finally:
          if own_model and model is not None:
              unload_cuda(model, tokenizer)
        merged = output_dir / "scores.jsonl"
        seen_keys: set[tuple[str, str]] = set()
        count = 0
        with merged.open("w", encoding="utf-8", newline="\n") as output:
            for shard in completed_shards:
                for row in read_jsonl(shard):
                    key = (row["qid"], row["chunk_id"])
                    if key in seen_keys:
                        raise ValueError(f"Duplicate reranker score key: {key}")
                    seen_keys.add(key)
                    output.write(canonical_json(row) + "\n")
                    count += 1
        result = {
            "schema_version": PIPELINE_SCHEMA,
            "stage": stage_name,
            "model": model_name,
            "config_hash": config_hash,
            "counts": {"queries": record_count, "pairs": count},
            "actual_batch_size": actual_batch,
            "scoring_strategy": SCORING_STRATEGY,
            "scoring_seconds": scoring_seconds,
            "mean_pairs_per_second": (
                scored_pairs / scoring_seconds if scoring_seconds else None
            ),
            "execution": dict(execution_metadata or {}),
            "artifact_sha256": {merged.name: sha256_file(merged)},
        }
        if device.startswith("cuda"):
            try:
                import torch

                free, total = torch.cuda.mem_get_info()
                result["memory"] = {
                    "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    "global_used_mib_at_finish": (total - free) / 2**20,
                }
            except RuntimeError:
                pass
        result["content_fingerprint"] = content_hash(result)
        atomic_json(output_dir / "manifest.json", result)
    return result


def aggregate_ce_documents(
    score_rows: Iterable[dict[str, Any]], gamma: float = 0.2
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in score_rows:
        grouped[str(row["qid"])][str(row["doc_id"])].append(float(row["score"]))
    result: dict[str, list[dict[str, Any]]] = {}
    for qid, documents in grouped.items():
        rows = []
        for doc_id, values in documents.items():
            ordered = sorted(values, reverse=True)
            best = ordered[0]
            top2 = ordered[:2] if len(ordered) > 1 else [best, best]
            rows.append(
                {
                    "doc_id": doc_id,
                    "score": best + gamma * (float(np.mean(top2)) - best),
                    "max_score": best,
                }
            )
        rows.sort(key=lambda row: (-row["score"], row["doc_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        result[qid] = rows
    return result


def fuse_stage1_and_ce(
    stage1: Sequence[dict[str, Any]],
    ce: Sequence[dict[str, Any]],
    *,
    ce_weight: float = 2.0,
    limit: int = 5,
) -> list[dict[str, Any]]:
    return weighted_rrf(
        {"stage1": stage1, "ce": ce},
        weights={"stage1": 1.0, "ce": ce_weight},
        limit=limit,
    )


def mine_pairwise_examples(
    *,
    qid: str,
    query: str,
    answers: set[str],
    positive_bundles: Mapping[str, dict[str, Any]],
    channel_rankings: Mapping[str, Sequence[dict[str, Any]]],
    best_bundle_by_doc: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    negatives: list[tuple[str, str]] = []
    for channel in ("bm25", "bge", "hybrid"):
        for row in channel_rankings[channel]:
            doc_id = str(row["doc_id"])
            if doc_id not in answers and doc_id in best_bundle_by_doc:
                negatives.append((channel, doc_id))
                break
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    weight = 1.0 / max(1, len(answers))
    for positive_doc in sorted(answers):
        if positive_doc not in positive_bundles:
            raise ValueError(f"Missing positive bundle for {qid}/{positive_doc}")
        for source, negative_doc in negatives:
            negative = best_bundle_by_doc[negative_doc]
            key = (positive_doc, negative_doc, negative["chunk_id"])
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "qid": qid,
                    "query": query,
                    "positive_doc_id": positive_doc,
                    "positive_chunk_id": positive_bundles[positive_doc]["chunk_id"],
                    "positive_text": positive_bundles[positive_doc]["bundle_text"],
                    "negative_doc_id": negative_doc,
                    "negative_chunk_id": negative["chunk_id"],
                    "negative_text": negative["bundle_text"],
                    "negative_source": source,
                    "sample_weight": weight,
                }
            )
    return output


def pairwise_softplus_loss(positive_scores: Any, negative_scores: Any, weights: Any | None = None):
    import torch
    import torch.nn.functional as functional

    losses = functional.softplus(-(positive_scores - negative_scores))
    if weights is not None:
        losses = losses * weights
    return losses.mean()


def train_lora_pairwise(
    train_pairs_path: Path,
    validation_pairs_path: Path,
    output_dir: Path,
    *,
    base_model: str = DEFAULT_RERANKER,
    device: str = "cuda",
    epochs: int = 2,
    learning_rate: float = 1e-4,
    gradient_accumulation: int = 16,
    pair_batch_size: int = 4,
    seed: int = 42,
    train_on_validation: bool = False,
    resume: bool = False,
    checkpoint_pairs: int = 2048,
    stage_name: str = "train-lora-fold",
    gradient_checkpointing: bool = True,
    pretokenize: bool = False,
) -> dict[str, Any]:
    """Train one adapter with deterministic, optimizer-safe mid-epoch resume."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if pair_batch_size < 1:
        raise ValueError("pair_batch_size must be positive")
    if checkpoint_pairs < pair_batch_size:
        raise ValueError("checkpoint_pairs must be at least pair_batch_size")
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_adapter = output_dir / "resume_adapter"
    state_path = output_dir / "training_state.pt"
    config_hash = content_hash(
        {
            "train_sha256": sha256_file(train_pairs_path),
            "validation_sha256": sha256_file(validation_pairs_path),
            "base_model": base_model,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "gradient_accumulation": gradient_accumulation,
            "pair_batch_size": pair_batch_size,
            "seed": seed,
            "train_on_validation": train_on_validation,
            "gradient_checkpointing": gradient_checkpointing,
            "pretokenize": pretokenize,
        }
    )

    def bucketed_rows(rows: Sequence[dict[str, Any]], epoch_seed: int) -> list[dict[str, Any]]:
        # Character length is a cheap stable proxy for token length. Shuffle
        # length-local buckets, then shuffle bucket order: padding falls while
        # stochastic order remains deterministic for every epoch.
        ordered = sorted(
            rows,
            key=lambda row: (
                max(len(row["positive_text"]), len(row["negative_text"])) + len(row["query"]),
                row["qid"],
                row["positive_chunk_id"],
                row["negative_chunk_id"],
            ),
        )
        rng = random.Random(epoch_seed)
        buckets = [ordered[start : start + 64] for start in range(0, len(ordered), 64)]
        for bucket in buckets:
            rng.shuffle(bucket)
        rng.shuffle(buckets)
        return [row for bucket in buckets for row in bucket]

    model = None
    tokenizer = None
    result: dict[str, Any] = {}
    with stage_run(output_dir, stage_name, total=epochs) as logger:
        logger.log("phase=load-pairs")
        train_rows = list(read_jsonl(train_pairs_path))
        validation_rows = list(read_jsonl(validation_pairs_path))
        if train_on_validation:
            train_rows.extend(validation_rows)
        logger.log(
            f"phase=load-model train_pairs={len(train_rows)} validation_pairs={len(validation_rows)}"
        )
        saved_state = None
        if resume and state_path.exists() and resume_adapter.exists():
            candidate = torch.load(state_path, map_location="cpu", weights_only=False)
            if candidate.get("config_hash") != config_hash:
                raise RuntimeError("Cannot resume LoRA training with changed inputs/config")
            saved_state = candidate
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        if pretokenize:
            logger.log("phase=pretokenize-pairs")
            unique_rows = train_rows + ([] if train_on_validation else validation_rows)
            for start in range(0, len(unique_rows), 256):
                rows = unique_rows[start : start + 256]
                queries = [row["query"] for row in rows]
                encoded = tokenizer(
                    queries + queries,
                    [row["positive_text"] for row in rows]
                    + [row["negative_text"] for row in rows],
                    padding=False,
                    truncation="only_second",
                    max_length=512,
                    return_attention_mask=False,
                )["input_ids"]
                split = len(rows)
                for index, row in enumerate(rows):
                    row["_positive_input_ids"] = encoded[index]
                    row["_negative_input_ids"] = encoded[split + index]
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa" if device.startswith("cuda") else "eager",
        )
        if gradient_checkpointing:
            base.gradient_checkpointing_enable()
        base.config.use_cache = False
        if saved_state is not None:
            model = PeftModel.from_pretrained(base, resume_adapter, is_trainable=True).to(device)
        else:
            lora = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["query", "value"],
                modules_to_save=["classifier"],
                bias="none",
            )
            model = get_peft_model(base, lora).to(device)
        if device.startswith("cuda"):
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
            model.enable_input_require_grads()
            torch.cuda.reset_peak_memory_stats()

        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
        total_updates = max(1, math.ceil(len(train_rows) * epochs / gradient_accumulation))
        warmup_updates = max(1, round(total_updates * 0.06))

        def learning_rate_scale(step: int) -> float:
            if step < warmup_updates:
                return (step + 1) / warmup_updates
            return max(0.0, (total_updates - step) / max(1, total_updates - warmup_updates))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
        accumulation_steps = max(1, math.ceil(gradient_accumulation / pair_batch_size))
        best_validation = math.inf
        best_epoch = 0
        start_epoch = 1
        start_row = 0
        global_update = 0
        if saved_state is not None:
            optimizer.load_state_dict(saved_state["optimizer"])
            scheduler.load_state_dict(saved_state["scheduler"])
            scaler.load_state_dict(saved_state["scaler"])
            start_epoch = int(saved_state["epoch"])
            start_row = int(saved_state["next_row"])
            global_update = int(saved_state["global_update"])
            best_validation = float(saved_state["best_validation"])
            best_epoch = int(saved_state["best_epoch"])
            torch.set_rng_state(saved_state["torch_rng"])
            if device.startswith("cuda") and saved_state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(saved_state["cuda_rng"])
            logger.log(
                f"phase=resume epoch={start_epoch} next_row={start_row} updates={global_update}"
            )

        def encode_pair_batch(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], Any]:
            if pretokenize:
                encoded = tokenizer.pad(
                    [
                        {"input_ids": row["_positive_input_ids"]} for row in rows
                    ] + [
                        {"input_ids": row["_negative_input_ids"]} for row in rows
                    ],
                    padding=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
            else:
                queries = [row["query"] for row in rows]
                passages = [row["positive_text"] for row in rows] + [row["negative_text"] for row in rows]
                encoded = tokenizer(
                    queries + queries,
                    passages,
                    padding=True,
                    pad_to_multiple_of=8,
                    truncation="only_second",
                    max_length=512,
                    return_tensors="pt",
                )
            encoded = {
                key: value.to(device, non_blocking=device.startswith("cuda"))
                for key, value in encoded.items()
            }
            weights = torch.tensor(
                [row.get("sample_weight", 1.0) for row in rows],
                dtype=torch.float32,
                device=device,
            )
            return encoded, weights

        def validation_loss() -> float:
            model.eval()
            losses: list[float] = []
            validation_batch = max(pair_batch_size, 8)
            with torch.inference_mode():
                for validation_start in range(0, len(validation_rows), validation_batch):
                    rows = validation_rows[validation_start : validation_start + validation_batch]
                    encoded, weights = encode_pair_batch(rows)
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")
                    ):
                        logits = model(**encoded, return_dict=True).logits.reshape(-1).float()
                    split = len(rows)
                    losses.append(
                        float(pairwise_softplus_loss(logits[:split], logits[split:], weights).cpu())
                    )
            return float(np.mean(losses))

        def save_resume_state(epoch: int, next_row: int) -> None:
            model.save_pretrained(resume_adapter, save_embedding_layers=False)
            payload = {
                "config_hash": config_hash,
                "epoch": epoch,
                "next_row": next_row,
                "global_update": global_update,
                "best_validation": best_validation,
                "best_epoch": best_epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
            }
            temporary = state_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(state_path)

        for epoch in range(start_epoch, epochs + 1):
            epoch_rows = bucketed_rows(train_rows, seed + epoch)
            row_cursor = start_row if epoch == start_epoch else 0
            if row_cursor < 0 or row_cursor > len(epoch_rows):
                raise RuntimeError(f"Invalid LoRA resume row: {row_cursor}/{len(epoch_rows)}")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_started = time.perf_counter()
            batches = 0
            checkpoint_cursor = row_cursor
            for row_start in range(row_cursor, len(epoch_rows), pair_batch_size):
                rows = epoch_rows[row_start : row_start + pair_batch_size]
                encoded, weights = encode_pair_batch(rows)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")
                ):
                    logits = model(**encoded, return_dict=True).logits.reshape(-1).float()
                    split = len(rows)
                    loss = pairwise_softplus_loss(
                        logits[:split], logits[split:], weights
                    ) / accumulation_steps
                scaler.scale(loss).backward()
                batches += 1
                processed = row_start + len(rows)
                final_batch = processed == len(epoch_rows)
                optimizer_boundary = batches % accumulation_steps == 0 or final_batch
                if optimizer_boundary:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_update += 1
                    if processed - checkpoint_cursor >= checkpoint_pairs or final_batch:
                        save_resume_state(epoch, processed)
                        checkpoint_cursor = processed
                        logger.log(
                            f"phase=train checkpoint epoch={epoch} pairs={processed}/{len(epoch_rows)}"
                        )
                if batches % 128 == 0 or final_batch:
                    elapsed = time.perf_counter() - epoch_started
                    completed_this_run = max(0, processed - row_cursor)
                    rate = completed_this_run / elapsed if elapsed else 0.0
                    remaining_pairs = (len(epoch_rows) - processed) + (epochs - epoch) * len(train_rows)
                    logger.status(
                        stage=stage_name,
                        state="RUNNING",
                        phase="train",
                        completed=epoch - 1,
                        total=epochs,
                        epoch=epoch,
                        epoch_pairs_completed=processed,
                        epoch_pairs_total=len(epoch_rows),
                        pairs_per_second=rate,
                        eta_seconds=remaining_pairs / rate if rate else None,
                        optimizer_updates=global_update,
                        resumable_from_pair=checkpoint_cursor,
                    )
            logger.log(f"phase=validate epoch={epoch}")
            mean_validation = validation_loss()
            epoch_seconds = time.perf_counter() - epoch_started
            logger.log(
                f"epoch={epoch} validation_pairwise_loss={mean_validation:.6f} "
                f"pairs_per_second={max(0, len(epoch_rows)-row_cursor) / max(epoch_seconds, 1e-9):.2f}"
            )
            if train_on_validation or mean_validation < best_validation:
                best_validation = mean_validation
                best_epoch = epoch
                checkpoint = output_dir / "best_adapter"
                model.save_pretrained(checkpoint, save_embedding_layers=False)
                tokenizer.save_pretrained(checkpoint)
            if epoch < epochs:
                save_resume_state(epoch + 1, 0)
            logger.status(
                stage=stage_name,
                state="RUNNING",
                phase="epoch-complete",
                completed=epoch,
                total=epochs,
                best_epoch=best_epoch,
                best_validation_loss=best_validation,
            )
            start_row = 0

        result = {
            "schema_version": PIPELINE_SCHEMA,
            "stage": stage_name,
            "base_model": base_model,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "train_pairs": len(train_rows),
            "validation_pairs": len(validation_rows),
            "pair_batch_size": pair_batch_size,
            "effective_pairs_per_update": pair_batch_size * accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "pretokenize": pretokenize,
            "attention_implementation": "sdpa" if device.startswith("cuda") else "eager",
            "peak_vram_mib": (
                torch.cuda.max_memory_reserved() / (1024 * 1024)
                if device.startswith("cuda") else None
            ),
            "train_on_validation": train_on_validation,
            "resume_supported": True,
            "checkpoint_pairs": checkpoint_pairs,
            "config_hash": config_hash,
            "seed": seed,
        }
        atomic_json(output_dir / "manifest.json", result)
        state_path.unlink(missing_ok=True)
        if resume_adapter.exists():
            shutil.rmtree(resume_adapter)
    if model is not None:
        unload_cuda(model, tokenizer)
    return result
