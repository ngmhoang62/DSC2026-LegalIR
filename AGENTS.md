# LegalIR — Project Instructions and Stable Context

## Mission

LegalIR retrieves the Vietnamese legal documents that answer a user question.
The output for each query is an ordered list of document IDs, with at most five
predictions. This repository is an experimental research codebase, not a
general legal-advice system.

## Competition contract

- Primary metric: Recall@5.
- Secondary metric: Precision@5, used as a tie-breaker when Recall@5 is tied.
- A query can have at most five relevant document IDs.
- The project uses fixed five-fold stratified cross-validation for local
  validation. Never use labels from a held-out fold to build retrieval,
  features, query-memory, thresholds, routing, or fusion for that fold.
- Submitted models must respect the competition's active-parameter limit of
  4.0B.
- Public leaderboard results are evidence, not a tuning target. Select model
  choices, thresholds, and fusion settings from fold-isolated local validation
  only.

## Data and correctness invariants

- `public_test_dataset/train.json` and the fixed fold definition are the
  sources of training labels and CV membership. Inspect their real schema
  before using them; do not infer field names or paths.
- Every prediction is a parent **document ID**, never a chunk ID.
- The structural corpus represents a document hierarchy:
  `Document -> Chapter -> Section -> Article -> Clause/Point`.
- Chunk identity is offset-based. Preserve raw text, offsets, parent links and
  manifest fingerprints; do not hand-edit generated cache artifacts.
- Generated artifact manifests, fingerprints and `_SUCCESS.json` markers are
  integrity gates. A mismatch means rebuild the affected artifact through its
  owning stage; never edit JSON merely to make a fingerprint match.
- A successful unit/smoke test is not a claim of metric reproduction. Report
  empirical CV or public results only after reading the corresponding emitted
  artifacts.

## Engineering rules

- Inspect files with `rg`/directory listing before describing, changing or
  deleting them. Do not invent source structure, cache layout, results or
  scores.
- Keep each experiment in its own namespace (`src/expNNN...`,
  `cache/expNNN...`, `results/expNNN...`). Do not overwrite a prior experiment
  or structural corpus in place.
- Preserve pipeline and reusable method modules. Delete only clearly disposable
  scratch/temporary artifacts, and state exactly what was removed.
- Use deterministic ordering, fixed seeds where randomness is involved,
  artifact fingerprints, resumable query-level checkpoints, periodic progress
  logs and an ETA for long stages.
- Do not run expensive corpus encoding, full retrieval, training, model
  download, or public submission unless the user explicitly asks. First run
  cheap audits, schema checks and small fixtures when appropriate.
- Avoid silent leakage: any feature that depends on labels, nearest training
  queries, calibration, router fitting, threshold selection, or fusion tuning
  must be fold-isolated.
- Keep dependencies reproducible in `requirements.txt` whenever a new runtime
  dependency is introduced.

## Research posture

- Start from a measurable hypothesis and define the candidate-oracle ceiling,
  ablation, acceptance gate, expected runtime and stop condition before adding
  expensive components.
- Analyze failures by query class, candidate-source coverage, rank movement,
  document type and evidence quality; aggregate metrics alone are insufficient.
- Rich structural context is a resource, not proof that every retrieval stage
  must process every chunk. Prefer coarse-to-fine retrieval and selective
  evidence construction where it preserves candidate recall.
- Treat all recent scores, cache locations, active runs, installed models and
  experiment-specific architectures as volatile. Obtain them from the user or
  inspect current manifests/results at the beginning of a new task.

## Evidence-capsule and heavy-reranker rules

- The canonical downstream label policy is
  `canonical_duplicate_alias_drop_empty_passage_v1`: map an exact-duplicate
  gold identity to preprocessing's `duplicate_retained_id`, and drop a gold
  identity only when its source passage is truly empty. The current frozen
  train data has 6,991 evaluable and 9 non-evaluable queries; audit these exact
  counts and the label fingerprint before training or evaluation.
- A capsule is one deterministic representation of one query-parent-document
  pair. Put answer evidence first and enforce the selected model's actual pair
  tokenizer budget. Scope, applicable subjects, identity and legal relations
  are optional context; they must never push all answer evidence outside the
  model input.
- Do not score a variable number of auxiliary views and take their uncorrected
  maximum. That gives documents with more views more chances to obtain a high
  score. Any multi-instance aggregation must be learned or calibrated inside
  the training fold and compared with a one-capsule baseline.
- Once a parent document enters the experiment's locked shortlist K, evidence
  selection may search all chunks belonging to that parent. `K=64` was the
  conservative EXP-027/028 historical budget; EXP-106--108 deliberately use
  K=50 for reranker cost. Do not silently substitute one contract for another.
  Existing upstream E5 evidence and same-parent BM25
  fallback are provenance, not proof that their first two chunks contain the
  answer. Preserve the selected chunk IDs, source offsets, ancestry and scores.
- Treat scope parsing and query-time scope use as separate questions. Measure
  parser precision, recall and boundary quality against source-exact annotated
  spans before relying on it. At query time select at most the relevant span;
  do not concatenate every scope node in a document.
- Preserve the strongest locked Stage-1 ranking for the experiment as the
  anchor. Historically this was EXP-027 LambdaMART; the strongest newer
  document-level anchor is EXP-109B's E5+LAL+BM25 LambdaMART. A heavy reranker
  must be evaluated as a fold-isolated residual/correction with a documented
  fallback, not be granted unrestricted authority merely because its candidate
  oracle is high.
- Prefer FP32. Use full fine-tuning only after a real backward plus optimizer
  preflight fits the GPU with at least 10% VRAM headroom. Otherwise try FP32
  LoRA, then FP16 LoRA only after measured FP32 OOM. Do not silently switch to
  QLoRA or another quantization policy.

## Repository orientation

- `src/`: source code. Historical experiments and current pipelines coexist;
  inspect ownership before modifying shared modules.
- `cache/`: generated, fingerprinted intermediate artifacts; never manually
  reformat or edit them.
- `results/`: metrics, audits, logs and submissions; use these as the source of
  truth for completed experiment outcomes.
- `tests/`: retained regression and synthetic tests.
- `docs/`: runbooks and design notes.
- `HANDOVER_PROMPT.md`: copyable prompt for starting a new conversation.

## Metric scopes and current baselines

Never compare a candidate ceiling, a 128/512-query development fixture, strict
inner F1--F4, five-fold OOF, one locked Fold-0 transfer, and a public submission
as if they were the same evaluation.

| Scope | Strongest verified result currently recorded | Meaning |
|---|---:|---|
| Full five-fold OOF | EXP-013b Recall@5 `0.9305047619` | Best verified whole-OOF result; historical BGE/Qwen/Lambda cascade and old representation. |
| Strict inner F1--F4 | EXP-109C Recall@5 `0.9333899517` | EXP-109B scalar anchor plus frozen Jina latent features; selection evidence only. |
| Locked Fold-0 | EXP-109B Recall@5 `0.9331783500` | Best verified Fold-0 transfer. EXP-109C reached `0.9325226514` and did not replace it. |
| Current architecture anchor | EXP-109B E5+VnLegal-LAL+BM25 LambdaMART | Default anchor for new complementarity work until a later locked gate passes. |

There is no verified local `0.96--0.97` result in this repository. Before
reporting a "new best", cite the artifact and metric scope explicitly.

## Completed optimization registry

Use this registry to avoid repeating old tuning. Read the referenced result and
manifest before reuse; create a new experiment namespace for a genuinely
different candidate/feature contract.

### Parsing, chunking and corpus representation

- EXP-016 accepted split-article heading recognition: 149 documents moved from
  fallback to structured with zero reverse regressions. This validates parser
  coverage, not retrieval gain.
- EXP-017 compared BGE-tokenized 384/512/768 windows on a fixed Fold-0 fixture;
  768 had the best Recall@5 (`0.8203`) but caused 2,712 E5 truncations.
- EXP-018 switched to the actual E5 tokenizer. `508/476/32` beat
  `384/352/32` (`0.8242` vs `0.8203` fixture Recall@5) while respecting the
  512-token E5 limit.
- EXP-019 found overlap 0/32/64 identical on that fixture (`0.8242` Recall@5).
  Do not reopen overlap tuning without a new failure hypothesis.
- EXP-020 showed `retrieval_text` better than `raw_text` for both Harrier and
  LAL on the same development fixture.
- The current structural-v3 corpus has 8,507 parent documents and 343,347
  structural chunks. Verify its manifest/fingerprint rather than rebuilding it
  casually.

### Dense retriever screening and training

- EXP-015 is a 128-query fixed Fold-0 development screen with equal corpus,
  formatting and max-over-chunks aggregation. Recall@5: VietLegal-E5 `0.9609`,
  VnLegal-LAL `0.9453`, VietLegal-Harrier `0.9297`, multilingual-E5-large
  `0.9141`, Qwen3-0.6B `0.9062`, BGE-M3 `0.8984`, Vietnamese-Legal-Embedding
  `0.8906`. These are not OOF scores; they justify E5 as anchor and LAL/Harrier
  only as complementarity candidates.
- EXP-101 tested supervised embedding alignment. Orthogonal mean-document plus
  RRF reached OOF Recall@5 `0.9078809524`; ridge fell to `0.8055714286`.
- EXP-102 MIL-NCE residual projection produced dense OOF Recall@5
  `0.9010966481`; static E5+BM25 RRF reached `0.9153959853`, Recall@50
  `0.9853287560`, MRR@5 `0.7756997091`.
- EXP-103 unconditional citation/preamble expansion reduced EXP-102 RRF
  Recall@5 from `0.9153959853` to `0.9047871072`; do not repeat it unchanged.
- EXP-109A tested decoupled/SoftTop-5 retrieval objectives and stopped at
  `REJECTED_PILOT_GATE`; no full nested/OOF winner exists.
- EXP-109C tested Jina-ColBERT late interaction. Jina standalone inner Recall@5
  was `0.8386822814`; latent features improved the scalar inner anchor
  `0.9254186781 -> 0.9333899517` (`+0.797pp`, CI95 lower `+0.420pp`, 4/4 folds
  positive). Locked Fold-0 was `0.9325226514`, only `+0.250pp` versus its
  matched scalar and `-0.066pp` versus historical EXP-109B. Status:
  `KEEP_WEAK_FROZEN_JINA`; keep it as a diagnostic source, not current winner.

### Sparse/BM25 optimization

- EXP-021 is the canonical completed sparse study on structural-v3. Passage-only
  RRF3 Recall@5 was `0.78819`; hierarchy raised it to `0.79348`; scope lowered
  it to `0.73232`; folded title lowered it to `0.77858`. Retain hierarchy and
  reject scope/title fields for this exact contract.
- EXP-021 aggregation Recall@5: first-passage `0.81306`; equal fusion of
  first-passage and RRF3 `0.82618`; head-20 fusion/RRF3-tail `0.82618`.
  Nested tuning selected modal head policy: passage depth `1024`, parent RRF
  `k=32`, fusion RRF `k=32`, head cutoff `16`, reaching OOF Recall@5 `0.83514`
  and Recall@50 `0.96575`.
- EXP-021 sparse candidate Recall@K curve is already measured: K30 `0.95308`,
  K50 `0.96590`, K80 `0.97361`, K100 `0.97654`, K120 `0.97860`, K150
  `0.98053`, K180 `0.98298`. These are ceilings, not top-5 rankings.
- Raw `document_label` is not a useful default signal: almost all labels are
  unaccented slugs, and isolated folded-title indexing hurt EXP-021. Do not
  reintroduce it without a new normalization and bounded ablation.

### Candidate union and rank fusion

- EXP-022's historical pool is E5@100 plus up to 50 novel BM25 parents, exactly
  150 unique parents/query. It remains the input contract for EXP-026--033, not
  a universal current policy. EXP-023 is its quota ablation, not a replacement
  candidate list.
- EXP-034's fold-isolated E5/BM25 RRF reached Recall@5 `0.9020049588`,
  Recall@32 `0.9776402994`, Recall@64 `0.9862942831`. Fixed/adaptive shortlist
  gates failed; do not cite the candidate ceiling as top-5 performance.
- EXP-108 rebuilt full-union weighted RRF over weights `0.55/0.45`,
  `0.65/0.35`, `0.75/0.25`; selections varied by calibration fold. Its
  downstream evidence gate failed, so it did not replace EXP-109B.
- EXP-109B is the current document-level fusion anchor. Strict F1--F4 weighted
  RRF reached `0.9187287681`; LambdaMART reached `0.9251057870`, `+2.078pp`
  over corrected E5+BM25 with 4/4 positive folds and CI95 lower `+1.582pp`.
  Locked Fold-0 reached `0.9331783500`, `+3.070pp` over corrected E5+BM25
  (`0.9024797330`). Status `WEAK_FOLD0_NO_FULL_OOF_REPRESENTATION_NEXT`; no
  full five-fold OOF exists.

### LambdaMART and ML pre-ranker optimization

- EXP-026 tried two LambdaMART settings. EXP-028 then nested-tuned 24 settings:
  leaves `{15,31,63}`, minimum child `{20,50}`, trees `{250,500}`, L2 `{0,2}`,
  plus feature-family selection. Parameters varied strongly by outer fold;
  there is no stable universal old-pool optimum. EXP-028 selected K50 on two
  outer folds and K64 on three, failed its all-fold shortlist gate, and had
  diagnostic OOF Recall@5 `0.9192826823`.
- EXP-027 is the strongest verified historical LambdaMART shortlist on the
  immutable EXP-022 pool: full OOF Recall@5 `0.9207611962` on retained gold.
  K64 was its conservative downstream budget, not a permanent Stage-1 rule.
- EXP-104 compared model families on an old EXP-102 feature matrix. OOF
  Recall@5: Logistic Regression `0.9156820674`, XGBoost listwise
  `0.9142039765`, CatBoost listwise `0.9137390931`, LightGBM LambdaRank
  `0.9127258857`, LightGBM rank_xendcg `0.9111643542`. Several nominal 28D
  features were deterministic placeholders/transforms of rank and the selected
  family had no independent outer confirmation; do not overgeneralize it.
- EXP-105 rank ensemble reached `0.9148834215` and did not beat EXP-104
  Logistic Regression. CatBoost/XGBoost/ensemble are not automatic reruns.
- EXP-109B tuned the newer source-fusion LambdaMART contract. Its majority
  locked configuration is `objective=lambdarank`, `eval_at=5`,
  `num_leaves=7`, `min_data_in_leaf=50`, `learning_rate=0.05`,
  `num_boost_round=300`, full feature/bagging fractions and deterministic seed.
  Reuse it by default for new source-fusion screens. Do not reopen a broad grid
  merely because the experiment ID changed; a standardized balanced Logistic
  Regression is acceptable only as a cheap distinct-bias challenger.

### Evidence selection, reranking and other closed/active branches

- EXP-030/031 rejected metadata-heavy/variable-view capsule routing. EXP-033
  failed its evidence gate; scope routing is not validated.
- EXP-106/107 explored protected/residual reranking at K50. EXP-107 has no
  complete result namespace in this checkout; do not quote conversational
  metrics as repository-verified results.
- EXP-108 atomic condition-aware global+local packages had zero silent
  truncation but failed `REJECTED_EVIDENCE_GATE`: exact-anchor retention
  `89/128 = 0.6953125` was below `0.80`; retainable-anchor retention was
  `77/84 = 0.9166667`. It never reached reranker training.
- EXP-024/025 are historical query-memory/both-source-miss diagnostics; any
  label-derived memory must remain strictly fold-isolated.
- EXP-034 adaptive K routing failed its inner gate and saved no pairs over K64.
  EXP-035 is error adjudication, not a promoted retriever. EXP-036 stopped after
  its Fold-0 Recall@32 promotion floor failed.
- EXP-110P semantic label-prototype is implemented for Colab but has no
  accepted metric result yet. Recheck Drive artifacts and exact EXP-109B anchor
  reproduction before use.

### Primary result artifacts

- EXP-013b OOF: `results/exp013b_cascade/oof/oof_report.json`.
- EXP-015 model screen: `results/exp015_model_screen/summary.md`.
- EXP-021 sparse study and tuned config:
  `results/exp021_sparse/REPORT.md` and
  `results/exp021_sparse/depth_rrf_tuning/tuning_report.json`.
- EXP-027/028 LambdaMART:
  `results/exp027_lambdamart_shortlist/REPORT.json` and
  `results/exp028_lambdamart_shortlist/oof/oof_report.json`.
- EXP-034 rank-fusion/shortlist audit:
  `results/exp034_shallow_retrieval/REPORT.json`.
- EXP-102/103 retrieval:
  `results/exp102_mil_nce_retrieval/REPORT_full_oof.json` and
  `results/exp103_preamble_citation/REPORT_seed3_cit3.json`.
- EXP-104/105 model-family screens:
  `results/exp104_preranker_benchmark/REPORT_benchmark_summary.json` and
  `results/exp105_synergy_preranker/REPORT.json`.
- EXP-108 evidence rejection:
  `results/exp108_atomic_condition_reranker/evidence/fold_0/REPORT.json`.
- EXP-109B inner and locked Fold-0:
  `results/exp109b_encoder_complementarity/cached_fusion_pilot/fold_0/CACHED_FUSION_PILOT.json`
  and
  `results/exp109b_encoder_complementarity/locked_fusion_fold0/FOLD0_LOCKED_FUSION_REPORT.json`.
- EXP-109C inner and locked Fold-0:
  `results/exp109c_latent_condition_late_interaction/FROZEN_INNER_SCREEN.json`
  and
  `results/exp109c_latent_condition_late_interaction/FROZEN_WINNER_FOLD0_REPORT.json`.
- Active status file:
  `results/exp110p_semantic_label_prototype/IMPLEMENTATION_REPORT.json`.

## Rules for the next experiment

- Identify the component above first. If the same input, candidate, feature and
  metric contract was already tuned, reuse its locked result instead of opening
  another grid.
- A changed representation or independent source can justify a small,
  pre-registered model/config screen. State exactly what changed and why the old
  result does not transfer.
- Keep Fold 0 absent until source/view/config/model selection is locked. A
  promising F1--F4 result authorizes at most one locked Fold-0 confirmation,
  not further tuning against Fold 0.
- Never overwrite EXP-022/027/028, EXP-102, EXP-109B/C, or active EXP-110P
  namespaces.

## Historical EXP-022--033 handoff (not the current global retrieval policy)

The bullets below describe the frozen contract of that historical branch only.
They must not block explicitly approved later retrieval work such as
EXP-109B/C.

- Retrieval is **frozen**. It uses only VietLegal-E5 and BM25; do not add, tune, or route any other candidate source (including query-memory, entity/citation, char lexical backoff, Qwen retrieval, or other dense models) in this stage.
- The fixed candidate policy is **E5 anchor @100 + up to 50 novel BM25 parent documents**, capped at exactly **150 unique parent document IDs/query**. Append BM25 documents absent from the E5 anchor in BM25 rank order. If fewer than 50 are novel, backfill first from E5 ranks 101--150 and then BM25, preserving all available E5/BM25 ranks, scores, and evidence provenance.
- The fixed-policy artifact is `cache/exp022_e5_bm25_union/train_oof_candidates.jsonl`; verify its manifest and `results/exp022_e5_bm25_union/union_report.json` before use. EXP-023 is a quota-ablation artifact, not the downstream candidate list. Candidate coverage is a ceiling, not a LambdaMART/reranker or public result.
- EXP-027 repaired the sparse-provenance sidecar for this immutable pool; EXP-028 then evaluated fold-isolated, shortlist-first LambdaMART over the same 24 non-label feature columns. These artifacts are evidence only: any new ranking experiment stays in a new namespace and must not overwrite them.
- No stable evidence supports a shortlist below **K=64**. EXP-028's strict nested selection rejected K=50 because two held-out folds fell below retained Recall@K=0.985. Use K=64 as the conservative pre-reranker shortlist budget unless a separately approved, fold-safe experiment improves it; this is 448,000 query-document pairs for the 7,000 training queries.
- The existing K=64 capsule inventory is provenance-bound to E5 evidence with frozen, same-parent BM25 fallback. It retains chunk IDs, raw text, offsets and structural context, but its 768-token value is an estimate. A reranker stage must implement and audit a model-specific pair renderer/tokenizer hard budget before training or scoring.
- EXP-030/031 showed that metadata-heavy, variable-view capsule routing is not
  an accepted method. The next approved research stage is evidence-first:
  audit scope parsing, select query-conditioned evidence across all chunks of
  each K=64 parent, render one answer-first capsule, and pass a nested evidence
  gate before any further heavy-reranker fine-tuning. It may not change
  retrieval, candidate membership, the 150-document budget, or launch a public
  submission without explicit approval.
- For downstream supervised-reranker OOF, do not blindly reuse one global cross-fitted LambdaMART/capsule artifact as training input for every outer fold: a ranking for an outer-train query may have been produced using labels from that outer-heldout fold. Within each reranker outer split, derive pre-ranker shortlists/capsules for outer-train queries by inner cross-fitting only inside outer-train; score outer-heldout queries with a pre-ranker trained on all outer-train. A single full-train LambdaMART is deferred until test inference preparation.

## Communication

- Work in Vietnamese unless the user requests otherwise.
- Lead with evidence and outcome. Clearly separate verified facts, hypotheses
  and recommendations.
- If a choice materially changes research direction, compute cost or existing
  artifacts, explain the trade-off and obtain direction before acting.
