# EXP-018 — target-tokenizer chunk-size ablation

## Motivation

EXP-017 showed that BGE-tokenizer chunk budgets do not transfer directly to
VietLegal-E5: E5 truncated 687 fixture chunks at nominal BGE 512 and 2,712 at
nominal BGE 768.  EXP-018 therefore measures size using the target encoder's
own tokenizer.

## Exact E5 accounting

VietLegal-E5 requires `passage: ` for documents. Its tokenizer emits two
prefix tokens and adds two single-sequence special tokens. With a model limit
of 512, retrieval text must be at most **508** tokens:

`508 retrieval_text + 2 passage-prefix + 2 special = 512`.

## Phase 1 configurations

| Key | Tokenizer | Retrieval-text max | Window | Overlap |
|---|---|---:|---:|---:|
| `e5_384_o32` | `mainguyen9/vietlegal-e5` | 384 | 352 | 32 |
| `e5_508_o32` | `mainguyen9/vietlegal-e5` | 508 | 476 | 32 |

The split-article parser of EXP-016, all sources, and all other policies stay
fixed. EXP-017 is retained as evidence of the tokenizer mismatch, not used to
select a size.

## Gates

Both corpora must audit `PASS`; no encoded document text may exceed E5's 512
input-token limit after prefix/special-token accounting. A fixed 512-query,
1,983-parent fixture and VietLegal-E5 dense retrieval compare Recall@5/20/100
and MRR@5. Only then may overlap `0/32/64` be swept for the selected size.
