# EXP-109A runbook

## Scope

EXP-109A is an isolated exact full-corpus parent-retrieval experiment. It
reads the frozen VietLegal-E5 query/chunk embedding caches and the raw
top-4096 BM25 evidence cache. It does not re-encode queries or chunks, change
the corpus, add a retriever, or reuse the EXP-022 append-union candidate
artifact.

The implementation is:

    src/exp109a_softtop5_retrieval.py
    tests/test_exp109a_softtop5_retrieval.py

Its private artifacts are limited to:

    cache/exp109a_softtop5_retrieval/
    results/exp109a_softtop5_retrieval/

## Frozen scoring contract

- VietLegal-E5 query and chunk embeddings remain frozen.
- Each parent is scored from every source-exact chunk belonging to that
  parent. The parent score is the mean of its two highest chunk scores; a
  one-chunk parent uses that score. Chunk ties use source chunk order.
- The train/evaluation vector has all 8,507 parents. There is no K150/K256
  training shortcut.
- Arm A is matched decoupled loss with all non-golds in every denominator.
- Arm B standardizes the full parent score vector, solves the SoftTop-5
  sigmoid threshold, and uses the analytic implicit backward derivative.
- Arm C is Arm A plus lambda times Arm B. The raw term gradients are logged;
  no rescaling is applied.
- All real arms use AdamW, learning rate 2e-4, weight decay 0.01, four
  epochs, gradient clipping 1, and no early stopping.
- Weighted RRF uses the full dense parent ranking plus BM25 rank contribution.
  Dense weights are 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9; RRF k is 10, 20,
  32, 60, or 100. A missing BM25 rank contributes zero.

## Gate order

Run from D:\Study\DSC2026\LegalIR with the bundled environment:

    D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py audit
    D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py test-loss
    D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py replay-exp102
    D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py preflight
    D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py smoke
    D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py report

The required reports are, respectively:

    results/exp109a_softtop5_retrieval/input_audit/READING_AUDIT.json
    results/exp109a_softtop5_retrieval/math/LOSS_MATH.json
    results/exp109a_softtop5_retrieval/replay_exp102/REPLAY_EXP102.json
    results/exp109a_softtop5_retrieval/preflight/PREFLIGHT.json
    results/exp109a_softtop5_retrieval/smoke/SMOKE.json

Only a PASS marker from all five prerequisite stages unlocks nested training.
The CUDA preflight must select the largest batch size with at least 10 percent
headroom. Smoke is a CPU synthetic full-corpus fixture and verifies all arms,
finite gradients, 50 steps, and exact checkpoint resume. It is not a public
or OOF score.

## Nested execution

These commands are manual gates and are never launched by the implementation:

    D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py nested-screen --outer fold_0 --resume
    D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py nested-oof --resume

The screen runs four inner rotations for outer fold_0. Inspect its
OUTER_REPORT.json, selection records, leakage checks, loss curves, VRAM, and
error slices before deciding whether to launch the full nested OOF. The OOF
stage also requires a successful fold_0 screen marker.

## Resume and inspection

Check progress without loading large artifacts:

    D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py status

Training checkpoints are atomic and include model, optimizer, scheduler,
epoch/query position, query order, RNG states, configuration, input
fingerprint, code fingerprint, and loss history. A changed input or source
fingerprint fails closed instead of silently resuming an incompatible run.

## Metric scope

Reports distinguish full-corpus dense ranking, nested inner validation, outer
fold, aggregate OOF, candidate coverage, single-gold, multi-gold, label
frequency slices, parent movement, rank-5/rank-6 margin, improved/degraded/
tied queries, paired within-fold bootstrap intervals, and the full-corpus
ceiling gap. No outer-fold label is used for inner selection.
