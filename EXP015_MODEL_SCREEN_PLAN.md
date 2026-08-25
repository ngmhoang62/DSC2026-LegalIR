# EXP-015 — Model / tokenizer retrieval screen

## Purpose

Run a small, reproducible **development screen** of candidate embedding-model
and tokenizer stacks on the existing `structural_v3` chunks.  This experiment
does not alter the chunker, the raw datasets, or the planned preprocessing
policy in `NEW_PLAN.md`.

## Candidates

- `BAAI/bge-m3` — current multilingual baseline.
- `darklethelong/vnlegal-lal`
- `bqbbao6/vietnamese-legal-embedding`
- `mainguyen9/vietlegal-harrier-0.6b`
- `intfloat/multilingual-e5-large-instruct`
- `Qwen/Qwen3-Embedding-0.6B`
- `mainguyen9/vietlegal-e5`

Public legal-domain fine-tuned models are permitted by the competition rules
as confirmed by the team.  The experiment downloads model weights only; it
does not add any external training or retrieval data.

## Fixed protocol

1. Use the frozen current `cache/structural_v3` corpus.  It is deliberately
   unfiltered: the pending empty-passage and duplicate-document policy is not
   silently mixed into this experiment.
2. Build one deterministic fixture from `fold_0` of `cache/cv_folds.json`:
   128 held-out training questions, every answer parent document, and 384
   deterministic distractor documents.
3. Retain at most eight deterministically selected current v3 chunks per
   parent document.  The cap is identical for answer and distractor documents,
   so document length cannot inflate one candidate's index size.
4. Encode chunk `retrieval_text`, normalize vectors, score with cosine, then
   aggregate chunk scores to parent documents by `max`.
5. Report Recall@5, Recall@20, Recall@100, MRR@5, throughput, embedding
   dimension, effective batch size, and token-truncation counts.  The fixture
   manifest records corpus fingerprint, selected question IDs, document/chunk
   identity hashes, prompts, and model-specific token limits.

## Fairness and interpretation

All candidates share the identical fixture, parent aggregation, ranking logic,
and runtime measurement.  They do **not** share an artificial universal
tokenizer/prompt: each model uses the query/document formatting and pooling
method stated by its model card.  That is necessary to test a model *stack*
rather than a deliberately misconfigured checkpoint.

The fold-0 result is only a screening signal.  No winner from EXP-015 may be
presented as an out-of-fold result or frozen for the final system until it has
been re-evaluated under the repository's strict fold-isolated selection
procedure.

## Outputs

- `cache/exp015_model_screen/fixture/`: deterministic fixture and manifest.
- `cache/exp015_model_screen/models/<key>/`: resumable normalized embeddings
  and per-model metadata.
- `results/exp015_model_screen/`: per-model JSON reports and Markdown table.

