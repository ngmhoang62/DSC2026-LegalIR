# EXP-023 — E5 + BM25 quota ablation and handoff

## Default candidate policy for the next pre-reranker stage

Use **E5 anchor @ 100 + up to 50 novel BM25 documents**, with a fixed cap of 150 parent-document candidates/query.
Append BM25 documents only when absent from the E5 anchor, in BM25 rank order. If fewer than 50 are novel, backfill first from the E5 tail and then BM25 until the list contains exactly 150 unique parent IDs. Preserve source ranks/provenance for LambdaMART.

This is the modal nested-OOF quota choice (3/5 folds); it is a fixed downstream policy, not a claim that a per-query or public optimum was selected. The fold-specific candidates in this EXP-023 cache remain an ablation artifact.

## Nested OOF quota screen

Selection for each held-out fold maximizes retained-gold Recall@150 on the other four folds, then MRR@150, then fixed grid order.

- OOF retained Recall@150: `0.990337`
- OOF retained MRR@150: `0.728117`
- Actual novel BM25 additions: mean `41.973`, min `18`, max `50`.

| Held-out fold | E5 anchor | BM25 novel max | Retained Recall@150 | MRR@150 |
|---|---:|---:|---:|---:|
| fold_0 | 100 | 50 | 0.989628 | 0.735576 |
| fold_1 | 120 | 30 | 0.991225 | 0.722625 |
| fold_2 | 100 | 50 | 0.990886 | 0.727376 |
| fold_3 | 100 | 50 | 0.990052 | 0.731994 |
| fold_4 | 120 | 30 | 0.989895 | 0.723002 |

## Remaining retained-gold misses

There are `126` retained gold occurrences absent from the fold-selected 150-candidate lists.

- **both_model_miss** (`86`): the gold parent is absent from E5's stored Top-150 *and* from the selected BM25 ranking list. This is bounded by the rankings inspected here, so it is a handoff set for independent channels (Qwen dense or query-memory/lexical), **not** proof that the document is unretrievable or that either channel will rescue it.
- **budget_allocation_miss** (`40`): the gold parent occurs in at least one inspected E5/BM25 source ranking, but is omitted by this 150-document quota/allocation. It is a candidate-budget problem, not evidence of a semantic or lexical failure.

The sample rows and per-occurrence ranks are in `REMAINING_MISSES.md` and `remaining_misses.jsonl`. Any claim that the first class is semantic versus lexical requires a fold-isolated rescue audit from the proposed Qwen and query-memory channels.

## Scope

These are candidate-stage OOF coverage metrics only. They do not measure LambdaMART or final reranker quality, and they must not be read as a public/submission result.
