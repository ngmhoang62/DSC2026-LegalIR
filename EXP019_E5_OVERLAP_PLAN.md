# EXP-019 — E5-508 overlap ablation

## Selection input

EXP-018 selected the E5-safe 508-token retrieval-text budget for this
development-only phase: it had higher Recall@5 (0.8242 versus 0.8203) and
fewer corpus chunks than E5-384, while every model input remained at most 512
tokens. The 2/512-query margin is not a final-model claim.

## Configurations

All runs use the EXP-016 parser, `mainguyen9/vietlegal-e5` tokenizer, max
retrieval text 508, window 476, and only change overlap:

| Key | Overlap |
|---|---:|
| `e5_508_o0` | 0 |
| `e5_508_o32` | 32 (reuse EXP-018) |
| `e5_508_o64` | 64 |

## Gates

Audit every new corpus; preserve all source/ground-truth documents and no E5
input can exceed 512 after `passage: ` and special tokens. Evaluate the fixed
512-query, 1,983-parent fixture with VietLegal-E5 and max-over-chunks parent
aggregation. The result remains fold-0 development evidence only.
