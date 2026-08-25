# EXP-017 — chunk size and overlap ablation

## Goal

Measure chunk-size effects independently from parser repair.  EXP-016 is the
frozen baseline; all candidates use its split-article-heading parser.

## Phase 1: size (overlap fixed)

| Key | Max tokens | Window | Overlap |
|---|---:|---:|---:|
| `b384_o32` | 384 | 352 | 32 |
| `b512_o32` | 512 | 480 | 32 |
| `b768_o32` | 768 | 736 | 32 |

`b384_o32` reuses `cache/exp016_split_article_v3`.  The window remains 32
tokens below the maximum, exactly as in v3.  This is a controlled size change,
not a tokenizer or parser change.

## Phase 2: overlap (deferred)

After phase 1, sweep overlap `0`, `32`, and `64` only for the selected size.
This phase is intentionally not pre-run: selecting its size from phase-1
retrieval evidence prevents a 3-by-3 rebuild grid.

## Gates and measurement

Every corpus must pass `audit_structural_chunks_v3.py`, retain all 8,532 source
documents and all train ground-truth parents, and respect its configured token
budget.  Retrieval uses a fixed stage-2 parent candidate fixture (512 fold-0
queries and 1,983 parent documents), `mainguyen9/vietlegal-e5`, normalized
cosine and max-over-chunks parent aggregation.  The comparison reports
Recall@5/20/100, MRR@5, chunk/node counts, token percentiles and encode time.

This is fold-0 development evidence only; it cannot freeze the final setting.
