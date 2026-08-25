# LegalIR — Handover: EXP-033 evidence-first Capsule v2

You are continuing in `D:\Study\DSC2026\LegalIR`. Communicate in Vietnamese.
Lead with artifacts and measured results; do not guess current processes,
hardware state, cache presence, scores or completion.

## Read before changing anything

1. Read `AGENTS.md` completely; it is the stable contract.
2. Inspect the real schemas, manifests, fingerprints and `_SUCCESS.json` files
   for the frozen preprocessing, Structural v3, E5/BM25 candidates, nested
   LambdaMART and K=64 capsules.
3. Read at least:
   - `results/exp030_legal_evidence_routing_canonical_v1/REPORT.json`;
   - `results/exp031_capsule_diagnostics/REPORT.json`;
   - `results/exp031_model_specific_validation/REPORT.json` and
     `GATE_REASSESSMENT.json`;
   - `results/exp032_capsule_gap_audit/REPORT.json`;
   - `src/exp030_legal_evidence_routing.py` and
     `src/exp032_capsule_gap_audit.py`.
4. Recheck active processes and emitted artifacts. Do not resume an old runner
   merely because a status file exists, and do not hand-edit generated JSON.

## Frozen contract

- Retrieval remains VietLegal-E5 @100 plus up to 50 novel BM25 parents, exactly
  150 unique parent document IDs/query. No third retriever, query memory,
  citation/entity channel, corpus change or public submission.
- LambdaMART remains fold-isolated and K=64 remains the conservative shortlist.
  Every prediction is a parent document ID and final output contains exactly
  five IDs.
- Use `canonical_duplicate_alias_drop_empty_passage_v1`: exact duplicates map
  to `duplicate_retained_id`; only truly empty passages are dropped. Require
  exactly 6,991 evaluable and 9 non-evaluable queries plus a matching label
  fingerprint before downstream work.
- For each reranker outer split, outer-train queries use inner-crossfit
  LambdaMART inside outer-train; outer-heldout queries use a pre-ranker trained
  only on all outer-train. No heldout label may choose evidence, rendering,
  thresholds, fusion, checkpoints or hyperparameters.

## What failed and what remains usable

- EXP-030/031 are `REJECTED`; do not promote their capsule variants and do not
  resume their training runner. Their immutable artifacts remain diagnostic
  evidence.
- Fresh bounded EXP-031 validation rejected model-specific routing. BGE mean
  delta Recall@5 was -0.0046875 and Precision@5 -0.00125; GTE mean delta
  Recall@5 was -0.0015625. Read the reports for per-fold details.
- EXP-032 audited 320 queries / 20,480 pairs and found the implementation gap:
  only 11 queries had explicit-scope and 19 explicit-subject signals, yet an
  auxiliary view was selected for about 93% of both gold and negatives. Of 420
  gold auxiliary views, 58.81% exceeded 512 tokens and 16.19% placed the answer
  marker after token 512. The old router also used candidate-level max over a
  variable number of views and reused upstream top-two evidence without a new
  in-document search.
- The conclusion is not that all chunk-derived capsules are useless. The
  rejected implementation was metadata-heavy rather than a true
  query-conditioned, in-document evidence pack.
- The 320 EXP-031 queries have been inspected. They may be reported as
  diagnostic data but must not be advertised as an independent confirmation
  set. Report a sensitivity slice excluding them in EXP-033.

## Next experiment: `exp033_in_document_evidence_routing`

Create new `src/cache/results/exp033_in_document_evidence_routing` namespaces;
do not overwrite Structural v3 or EXP-022/027/028/030/031/032.

Provide CLI commands:

```text
audit-inputs
sample-scope-audit
finalize-scope-audit
build-parent-index
score-in-document
build-capsules-v2
screen-evidence
preflight-bge
train-bge
evaluate-bge
report
overnight
```

### Phase A — source-exact scope audit

- Deterministically sample 200 unique documents: 50 scope-of-regulation, 50
  applicable-subject, 40 combined, 30 rejected legacy-scope, and 30 with no
  typed scope but raw-text scope signals.
- Store source offsets, raw span, parser label, review label, boundary
  completeness, missed span and rationale in fingerprinted JSONL.
- The agent reviews all 200 and exports a balanced 30-item spot-check for the
  user. Stop with `WAITING_SCOPE_SPOTCHECK`; do not continue automatically.
- Require at least 27/30 agreement. Otherwise adjudicate disagreements and
  issue another 30-item spot-check.
- Parser gate: micro precision >=0.95, each-class precision >=0.90, boundary
  accuracy >=0.95 and stratified-weighted recall >=0.90.
- If it fails, repair only the EXP-033 metadata sidecar from observed error
  classes, retain exact offsets and use structural sibling boundaries. Rebuild
  Structural v3 only if the audit proves parent/offset/hierarchy corruption.

### Phase B — in-document evidence selection

- Build a fingerprinted `doc_id -> chunk/embedding rows` index over the frozen
  343,347 Structural-v3 chunks. Reuse `cache/e5_final_v1` embeddings and
  EXP-021 train-query embeddings; do not re-encode the corpus.
- Use final passage+hierarchy FTS5 and `BM25Searcher.search_document()` for
  same-parent lexical evidence.
- Compare current upstream top-two evidence, in-parent E5 top-one, E5 top-two
  with MMR lambda in {0.70, 0.85}, in-parent BM25 top-two, and hybrid reciprocal
  E5/BM25 ranks with dense weight in {0.25, 0.50, 0.75}.
- Choose the primary chunk by query relevance. Add a secondary only when it is
  non-duplicate, has cosine redundancy below 0.90 and fits the budget. Prefer
  clause/parent/sibling expansion that completes the legal proposition.
- Score scope and applicable-subject spans directly against the query. Select
  at most one span. Regex is a high-precision prior, not an exhaustive router;
  uncertain cases omit scope.

### Phase C — one answer-first capsule per document

Use Vietnamese markers only:

```text
[BẰNG CHỨNG CHÍNH]
[VỊ TRÍ PHÁP LÝ]
[BẰNG CHỨNG BỔ SUNG]
[PHẠM VI LIÊN QUAN] / [ĐỐI TƯỢNG LIÊN QUAN]
[VĂN BẢN]
```

- Do not create variable auxiliary views or take their maximum.
- For BGE, enforce the actual query-document pair at <=512 tokenizer tokens.
  Audit queries first; if a query exceeds 128 tokens, preserve head 96 plus
  tail 32 and report it.
- After query and special tokens, primary evidence receives at least 55% of
  the document allowance; identity plus path at most 20%; optional
  applicability at most 15%; secondary uses the remainder.
- Truncate only on sentence/clause boundaries. Assert answer evidence remains
  present after tokenization. Use a verified accented official title when
  available and the unaccented normalized label only as fallback/audit identity.

### Phase D — evidence gate before GPU training

- Select every selector, weight, scope threshold and renderer in inner folds;
  evaluate full nested OOF on 6,991 evaluable queries and report the slice that
  excludes the 320 EXP-031 diagnostic queries.
- Cheap residual grid: ranking window W in {16,25,32,50,64} and evidence weight
  alpha in {0.10,0.25,0.50,0.75}, using per-query standardized LambdaMART and
  evidence scores.
- Continue to BGE training only if aggregate Recall@5 improves by at least
  0.002 over original LambdaMART, Precision@5 does not decrease, at least four
  outer folds are non-negative, worst-fold delta is >=-0.002, the exclude-320
  sensitivity is non-negative, and candidate membership/oracle are unchanged.
- On failure emit `REJECTED_EVIDENCE_GATE`, produce error slices and stop. The
  user explicitly chose not to run even a diagnostic BGE fold after this gate
  fails.

### Phase E — protected BGE residual training

- Use only `BAAI/bge-reranker-v2-m3` after the evidence gate; do not repeat the
  six-model screen or GTE in EXP-033.
- Preflight a real 512-token backward, optimizer state and checkpoint reload on
  the 6GB GPU. Use full FP32 only if peak VRAM <=5.4GB; otherwise FP32 LoRA
  rank 16, alpha 32, dropout 0.05; use FP16 LoRA only after measured FP32 OOM.
  Do not switch to QLoRA without new approval.
- Train three fixed epochs, effective batch 32, with no outer-heldout early
  stopping. Use multi-positive listwise loss.
- For each positive/epoch choose eight deterministic hard negatives: two
  LambdaMART top-five false positives, two ranks 6-16, two lexical/structural
  legal near-misses, one source-diverse negative and one seeded rank-33-64 tail;
  backfill deterministically when a category is unavailable.
- Final ranking is a protected residual. Inner-select alpha in
  {0.10,0.25,0.50,0.75}, standardized promotion margin tau in
  {0,0.25,0.50,1.00}, and W in {16,25,32,50,64}. An outside-top-five candidate
  displaces the current head only after clearing the promotion margin;
  otherwise preserve LambdaMART order.
- Promotion requires all five folds, aggregate Recall@5 delta >=0.002,
  non-decreasing Precision@5 and no fold below -0.005 Recall delta. Absolute
  Recall@5 >=0.97 is the final target. Passing the relative gate below 0.97 is
  `PROMISING_NOT_FINAL`, not a submission result.

## Tests, scheduler and reporting

- Add unit tests for canonical labels, fold isolation, immutable candidate
  membership, chunk/embedding/document mapping, same-parent BM25, scope spans
  and sampler, deterministic selectors, one-view answer-first rendering,
  actual tokenizer budget, protected residual, exact top-five output, resume
  and hard-stop behavior.
- Integration-smoke one query/fold through index -> evidence -> capsule ->
  scoring; test scope detected/combined/rejected/missing fixtures; if the
  evidence gate passes, run one BGE optimizer step and checkpoint reload before
  full training.
- Run retained EXP-030/031 regressions plus EXP-033 tests before a long job.
- `overnight` is sequential, fingerprinted and resumable. Every job has logs,
  manifest and `_SUCCESS.json` or `_FAILED.json`. Maintain `RUN_STATUS.json`
  and `state.jsonl` with weighted percent, ETA, retry, VRAM and paths. Integrity,
  fold, membership and hash errors are never retried.
- The final report must include parser confusion/boundary audit, selector and
  K-window ablations, token/truncation audit, LambdaMART anchor, Recall@5 and
  Precision@5, rank rescue/harm, query/document/error slices, latency/VRAM,
  completed-fold coverage and an explicit promotion decision.

Do not launch a public submission. Do not continue past either manual
spot-check or evidence-gate stop without satisfying its stated condition.
