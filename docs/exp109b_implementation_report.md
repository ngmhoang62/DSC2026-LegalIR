# EXP-109B implementation handover

Date: 2026-08-29

## Implemented

- Added the isolated EXP-109B source, tests, cache namespace, results namespace,
  and runbook.
- Added the complete reading/input audit with canonical-label, fold, corpus,
  frozen E5, BM25, EXP-035/037, local-snapshot, disk, and process checks.
- Implemented the locked E5, Harrier, and LAL contracts; LAL handles both
  left- and right-padding last-token pooling.
- Implemented exact FP16-to-FP32 re-normalization and blockwise `top2_mean`
  parent scoring, with max and normalized-LogSumExp diagnostics kept out of
  winner selection.
- Implemented the EXP-037-derived, gold-forced bounded screen, all requested
  gate metrics, source-generation tagging, rescue overlap, and one/two-model
  selection policy.
- Implemented preflight, effective batch-size reuse, resumable/checksummed
  encoding shards, exact full-parent ranking, source viability, corrected
  weighted RRF, nested LambdaMART selection, Fold-0 gating, and guarded
  five-fold OOF orchestration.
- Added flushed run status/logging, code/input/scorer fingerprints, and
  owning-stage rebuild behavior for corrupt or partial shards.

## Verified artifacts

- `results/exp109b_encoder_complementarity/input_audit/READING_AUDIT.json`:
  `PASS`, 0 hard errors, 5 explicitly recorded checkout mismatches.
- `results/exp109b_encoder_complementarity/replay/REPLAY.json`:
  `PASS`; all three archived EXP-015 metric replays have zero absolute error;
  independent real-parent scorer parity is `5.960464477539063e-08`.
- `results/exp109b_encoder_complementarity/smoke/SMOKE.json`:
  `PASS`; 32 CPU steps, finite checkpoint, NumPy scorer error
  `1.862645149230957e-08`.
- EXP-109B tests: `23 passed`.
- Required regression suite: `79 passed`.

## Follow-up implementation audit (2026-08-29)

- Fixed the NumPy reference scorer so FP32 fixtures are normalized directly
  instead of being silently requantized through FP16.
- Fixed a `source_audit` runtime failure where gold-source membership read its
  rank lookup before construction.  Unique source contribution now counts
  query--gold-document occurrences, and EXP-035 tag slices include per-source
  metrics.
- Made the source-viability corrected baseline use outer-train inner-CV
  predictions, rather than evaluating a configuration tuned on those same
  labels.  RRF now locks `k` and weights from aggregate inner validation
  instead of retuning them on all outer-train labels; reports retain this
  provenance.
- Bound bounded/preflight/full-encoding resumes to current audit/replay/model
  fingerprints, validate selected-model query artifacts, and verify the
  legacy fixture against the real EXP-036 top-96 universe before reuse.
- Verified the real bounded contract read-only: `290/290` EXP-037 rows match
  `EXP-036 top-96 ∪ canonical gold`.  The expanded EXP-109B suite is
  `26 passed`; the regression suite is `82 passed`.

## Intentionally not executed

The following stages remain fail-closed and were not launched in this turn:

- fresh local model mini-encoding (`replay --fresh-models`);
- bounded GPU screen;
- GPU preflight and 343,347-chunk full encoding;
- full-corpus source audit, Fold-0 screen, and nested OOF;
- public submission.

No model was downloaded, no frozen corpus/old experiment namespace was edited,
and no public result was created. Run the guarded commands in
`docs/exp109b_runbook.md` only after the corresponding explicit authorization
and after inspecting the current success markers.
