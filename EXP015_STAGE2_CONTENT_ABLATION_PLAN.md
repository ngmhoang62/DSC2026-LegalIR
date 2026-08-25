# EXP-015 stage 2 — corpus scale and content-view ablation

## Question

Does the stage-1 result arise because `vietlegal-e5` is intrinsically better on
LegalIR, or because it happens to match v3's marked `retrieval_text`?

## Fixed evaluation protocol

- Models: `vnlegal_lal`, `vietlegal_e5`, and `vietlegal_harrier_0_6b` only.
- Questions: 512 deterministic eligible questions from `fold_0`.
- Candidate parents: all positive parents for those questions plus 1,536
  deterministic distractors (about 2,048 parents total).
- Candidate chunks: four deterministic chunks per parent.  This makes the
  candidate universe four times larger than stage 1 without letting a long
  legal document receive more max-pooling opportunities than a short one.
- Views, evaluated on precisely the same chunk IDs:
  - `raw_text`: leaf text only;
  - `retrieval_text`: existing v3 markers/structural context plus leaf text.
- Retrieval: normalized cosine, then `max` over a parent's retained chunks.
- Metrics: Recall@5/20/100, MRR@5, and paired per-query rank deltas.

## Interpretation guardrails

This is still a fold-0 development ablation, not an OOF selection result.  It
answers the narrow content-view question.  It must not decide a final model,
tokenizer, or preprocessing policy; that needs a later fold-isolated test.

