# EXP-012b v3-native runbook

This pipeline uses structural-v3 passages end to end. BM25 and BGE-M3 are
independent retrieval channels; neither is a hard gate. Their parent-document
rankings are fused, then a cross-encoder scores at most three diverse evidence
passages per parent.

All commands below run in the foreground. Run one command at a time. The
pipeline never starts a hidden watcher or polling process. `--stage status` is
a read-only, one-shot snapshot that should only be invoked manually.

## Resource rules

- BM25 stages are CPU/disk only.
- BGE leaf embeddings are FP16 memory maps and resumable by verified shard.
- Retrieval keeps the block matrix on GPU, but leaf embeddings on CPU/disk.
- The BGE model is unloaded before the cross-encoder stages begin.
- Evidence routing uses a byte-offset index and 32-query batches; it does not
  load all 435,316 v3 passages into RAM. Routing output is checkpointed every
  32 queries.
- Cross-encoder scoring uses explicit SDPA, tokenizes each shard once, sorts
  pairs stably by length, and uses adaptive 48/32/24 batches. It is resumable
  every 64 queries by verified shard and halves its base batch on CUDA OOM.
- LoRA loads only the reranker. BGE-M3 is not resident during training.
- LoRA batches four positive/negative pairs into one 8-sequence forward,
  retains FP32 trainable master weights, and uses FP16 autocast plus gradient
  checkpointing for the frozen backbone.
- LoRA mining uses exact rowid-bounded FTS5, four bounded CPU workers, one
  persistent SQLite connection per worker, per-QID route checkpoints, and
  64-query teacher shards.
- LoRA training checkpoints adapter, optimizer, scheduler, scaler, and RNG
  state every 2,048 pairs. `--resume` continues at the saved epoch/row.
- Hybrid retrieval checkpoints every 64 queries and uses four bounded CPU
  workers. The retrieval depths and FP32 leaf scores are unchanged.

## Stage order

Use `d:\Study\DSC2026\dsc_env\Scripts\python.exe` from the LegalIR root.
The commands below are PowerShell commands; they deliberately use the full
path so another `python` on `PATH` cannot be selected accidentally.

```powershell
python src/exp012b_pipeline.py --stage preflight
python src/exp012b_pipeline.py --stage build-chunk-lookup
python src/exp012b_pipeline.py --stage tokenize-bm25 --resume
python src/exp012b_pipeline.py --stage build-bm25 --resume
python src/exp012b_pipeline.py --stage encode-bge-leaves --resume
python src/exp012b_pipeline.py --stage build-bge-blocks
python src/exp012b_pipeline.py --stage retrieve-hybrid --split train --resume
python src/exp012b_pipeline.py --stage route-evidence --split train --resume
python src/exp012b_pipeline.py --stage score-zero-shot --split train --resume
python src/exp012b_pipeline.py --stage evaluate-zero-shot
```

Stop if Stage-1 hybrid Recall@50 is below 0.965 or below the best individual
BM25/BGE channel. Also stop before LoRA when the zero-shot promotion floors in
`results/exp012b_v3/zero_shot_metrics.json` fail.

`evaluate-zero-shot` writes its own log/status/success marker under
`results/exp012b_v3/zero_shot_evaluation`. Its promotion gate requires both the
legacy floors and non-regression against the v3 Stage-1 Recall@5 and
Precision@5. `mine-lora`, including fold 0, refuses to run unless that success
marker exists and the complete gate passes.

## Measured scorer configuration on this machine

The completed train evidence artifact contains 620,715 unique query-passage
pairs. A representative sample had median pair length 388 tokens and P95 467
tokens, so stable length sorting materially reduces padding.

On the RTX 4050 Laptop GPU, the original eager batch-8 path scored about
61.5-63.9 pairs/second. The selected SDPA + one-pass tokenization + adaptive
length-bucket path scored 74.25 pairs/second over 710 real pairs, a measured
1.23x speedup. Every one of the eight sampled queries retained the exact same
Top-5 document order; maximum FP16 logit drift was 0.015625.

At the measured rate, pure forward compute for the full zero-shot artifact is
approximately 2 hours 19 minutes. Allow additional time for model loading,
JSON parsing, shard hashing, disk writes, and laptop power/thermal variation.
The scorer now writes about 110 short checkpoints instead of 28 long ones, so
an interruption normally loses no more than one 64-query shard.

The command remains unchanged; the measured settings are reproducible defaults:

```powershell
python src/exp012b_pipeline.py --stage score-zero-shot --split train --resume
```

During scoring, `status.json` reports completed queries, cumulative scored
pairs, observed pairs/second, and an ETA. `run.log` records one compact line per
shard. No polling or background watcher is required.

After the zero-shot gate passes, run fold 0 first:

```powershell
python src/exp012b_pipeline.py --stage mine-lora --fold 0 --resume
python src/exp012b_pipeline.py --stage train-lora-fold --fold 0 --resume
python src/exp012b_pipeline.py --stage score-lora-fold --fold 0 --resume
python src/exp012b_pipeline.py --stage evaluate-lora-fold --fold 0
```

Folds 1-4 are code-gated until fold 0 improves Recall@5 by at least 0.005 while
losing no more than 0.001 Precision@5 versus zero-shot. If it passes, repeat the
four commands for folds 1 through 4, keeping `--resume` on mine/train/score,
then run:

```powershell
python src/exp012b_pipeline.py --stage evaluate-oof
python src/exp012b_pipeline.py --stage mine-final --resume
python src/exp012b_pipeline.py --stage train-final --resume
python src/exp012b_pipeline.py --stage retrieve-hybrid --split public --resume
python src/exp012b_pipeline.py --stage route-evidence --split public --resume
python src/exp012b_pipeline.py --stage score-final-public --resume
python src/exp012b_pipeline.py --stage public-submission
```

The final adapter and submission stages are blocked unless OOF improves
Recall@5 by at least 0.005 versus the zero-shot result while losing no more
than 0.001 Precision@5. This replaces the obsolete EXP-012 floors of 0.8634
and 0.1841. Public fusion uses the modal fold-isolated LoRA configuration, not
a configuration selected on public data.

## Manual status and recovery

```powershell
python src/exp012b_pipeline.py --stage status
```

Each stage owns `run.log`, `status.json`, and `_SUCCESS.json`. A re-run removes
the old success marker before doing work, so a failed retry cannot authorize a
downstream stage. For resumable stages, pass `--resume`; completed shards are
accepted only when their configuration and content hashes match.

`status` now includes the active phase and ETA when the stage can estimate it.
Mining phases are `select-negatives`, `segment-queries`, `route-documents`,
`teacher-score`, and `write-pairs`. A Ctrl+C marks the stage FAILED instead of
leaving a misleading RUNNING status.

The default `--workers 4` is deliberately bounded for this laptop. Increasing
it may increase SQLite/disk contention and must be benchmarked before use.

## Quality boundary

The enabled optimizations preserve the candidate depths, exact FTS5 BM25
formula, FP32 leaf refinement, evidence limits, RRF weights, and reranker
inputs. The following are not enabled because they may change Recall/Precision:

- reducing BM25/BGE Top-2000 or parent Top-100;
- reducing Top-8 leaf refinement or the three evidence passages;
- FP16/approximate leaf search, ANN, or chunk pruning;
- changing the reranker maximum length or LoRA effective batch semantics.

The current environment must contain `peft` before the first LoRA training
stage. Installing packages is intentionally not performed by this runbook.

## Optimized profile (post-EXP-012b reference run)

The completed submission is frozen under `cache/exp012b_v3` and
`results/exp012b_v3`. Never point an optimized output at either directory.
`--execution-profile optimized` selects `cache/exp012b_v3_fast` and
`results/exp012b_v3_fast`; completed reference stages are read-only upstream
fallbacks, so unchanged BM25/BGE/evidence files are neither copied nor rebuilt.

Create the performance contract once:

```powershell
python src/exp012b_benchmark.py --action freeze
python src/exp012b_benchmark.py --action sample
python src/exp012b_pipeline.py --execution-profile optimized --stage preflight
```

For the first low-risk scorer benchmark, reuse reference evidence and adapter,
but create all new artifacts in the fast namespace:

```powershell
python src/exp012b_pipeline.py --execution-profile optimized --stage build-token-cache --split train --resume
python src/exp012b_pipeline.py --execution-profile optimized --stage score-lora-fold --fold 0 --resume
```

The token cache stores each exact max-512 pair once as packed `uint32` IDs.
Zero-shot and every LoRA fold can reuse it. Optimized LoRA inference merges the
adapter into the in-memory base model with `safe_merge=True`; it never writes a
merged model to disk. Promote this path only after fold-0 logits/Top-5 and
metrics pass the benchmark contract.

For a full optimized Stage-1 experiment, run the normal stage order with the
profile flag. The optimized profile uses cached structural integrity receipts,
four-process Underthesea passage segmentation, BGE batch 16 with OOM fallback,
immutable/mmap SQLite reads, exact FP32 CUDA candidate-leaf refinement, and a
persistent evidence chunk reader. Candidate depths, RRF, Top-8, evidence
budgets, max length and model identities are unchanged.

```powershell
python src/exp012b_pipeline.py --execution-profile optimized --stage tokenize-bm25 --workers 4 --resume
python src/exp012b_pipeline.py --execution-profile optimized --stage build-bm25 --resume
python src/exp012b_pipeline.py --execution-profile optimized --stage encode-bge-leaves --resume
python src/exp012b_pipeline.py --execution-profile optimized --stage build-bge-blocks
python src/exp012b_pipeline.py --execution-profile optimized --stage retrieve-hybrid --split train --workers 4 --resume
python src/exp012b_pipeline.py --execution-profile optimized --stage route-evidence --split train --resume
python src/exp012b_pipeline.py --execution-profile optimized --stage build-token-cache --split train --resume
```

Prepare teacher routes/scores once before fold materialization:

```powershell
python src/exp012b_pipeline.py --execution-profile optimized --stage prepare-lora-teacher --resume
python src/exp012b_pipeline.py --execution-profile optimized --stage mine-lora --fold 0 --resume
```

Training remains conservative by default: pair batch 4 and checkpointing on.
Candidates `--pair-batch-size 8` and `--no-gradient-checkpointing` are explicit
experiments, not promoted defaults. Each must rerun train/score/evaluate fold 0
and pass the metric gate before folds 1-4. Optimized training pretokenizes pairs
once and restricts optimizer/gradient clipping to trainable parameters.

After matching artifacts exist, compare the fixed sample:

```powershell
python src/exp012b_benchmark.py --action compare
```

A non-zero exit means parity failed; do not promote the fast profile. Full
Stage-1 Recall@50, zero-shot, fold-0 and OOF metric gates remain mandatory.
