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
- Once a parent document enters K=64, evidence selection may search all chunks
  belonging to that parent. Existing upstream E5 evidence and same-parent BM25
  fallback are provenance, not proof that their first two chunks contain the
  answer. Preserve the selected chunk IDs, source offsets, ancestry and scores.
- Treat scope parsing and query-time scope use as separate questions. Measure
  parser precision, recall and boundary quality against source-exact annotated
  spans before relying on it. At query time select at most the relevant span;
  do not concatenate every scope node in a document.
- Preserve LambdaMART as the ranking anchor. A heavy reranker must be evaluated
  as a fold-isolated residual/correction with a documented fallback, not be
  granted unrestricted authority to reorder K=64 merely because its candidate
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

## Retrieval and pre-reranking handoff (verify emitted manifests before using)

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
