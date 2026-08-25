# EXP-020 — Qwen3 stack screen at the practical 512-token limit

## Decision question

After EXP-019, retain the structural-parser policy of EXP-016 and choose
`overlap=64` as the conservative candidate.  This screen asks whether the
Qwen3-tokenized stacks `VietLegal-Harrier-0.6B` or `VNLegal-LAL` beat the
current VietLegal-E5 candidate before a later parser rebuild and corpus-wide
embedding cache are committed.

This is a development screen, not OOF model selection.

## Exact Qwen3 accounting

The locally installed `Qwen/Qwen3-Embedding-0.6B` fast tokenizer adds one
single-sequence special token.  Harrier and LAL require no document prefix.
Consequently, the user-selected 512-token practical model budget is enforced
as:

`511 retrieval_text + 0 document-prefix tokens + 1 special token = 512`.

LAL supports longer inputs in its model card, but it is intentionally capped
to 512 here because the target GPU cannot practically encode a corpus at its
2048-token maximum.  This makes Harrier and LAL directly comparable at the
same Qwen3 chunk boundaries.

## Fixed configuration

| Item | Value |
|---|---|
| Parser | EXP-016 split-article-heading enabled |
| Chunk tokenizer | `Qwen/Qwen3-Embedding-0.6B` |
| Retrieval-text maximum | 511 tokenizer tokens |
| Oversized-leaf window | 479 tokenizer tokens |
| Overlap | 64 tokenizer tokens |
| Corpus | `cache/exp020_qwen3_511_o64` |
| Audit output | `results/exp020_qwen3_511_o64_audit` |
| Evaluation fixture | Existing 512-query / 1,983-parent EXP-015 stage-2 fixture |
| Per-parent evidence | Deterministic cap of 4 chunks |
| Parent score | Maximum cosine similarity over that parent's chunks |

## Models and formatting

| Model | Effective max input | Query formatting | Document formatting |
|---|---:|---|---|
| `mainguyen9/vietlegal-harrier-0.6b` | 512 | documented legal instruction plus `Query: ` | raw `retrieval_text` |
| `darklethelong/vnlegal-lal` | 512 | same documented legal instruction plus `Query: ` | raw `retrieval_text` |

Both use their documented Qwen-family last-token representation and cosine
similarity.  LAL's code path deliberately overrides its native 2048 limit to
512 only for this experiment.

## Commands

Run from `LegalIR` with `D:\Study\DSC2026\dsc_env\Scripts\python.exe`:

```powershell
& 'D:\Study\DSC2026\dsc_env\Scripts\python.exe' src/structural_chunker_v3.py `
  --contexts-dir public_test_dataset/selected-contexts `
  --output-dir cache/exp020_qwen3_511_o64 `
  --tokenizer Qwen/Qwen3-Embedding-0.6B `
  --max-passage-tokens 511 --token-window 479 --token-overlap 64 `
  --allow-split-article-heading --workers 4

& 'D:\Study\DSC2026\dsc_env\Scripts\python.exe' src/audit_structural_chunks_v3.py `
  --contexts-dir public_test_dataset/selected-contexts `
  --cache-dir cache/exp020_qwen3_511_o64 `
  --output-dir results/exp020_qwen3_511_o64_audit

& 'D:\Study\DSC2026\dsc_env\Scripts\python.exe' src/exp020_qwen_stack_screen.py --stage run-one --model vietlegal_harrier_0_6b
& 'D:\Study\DSC2026\dsc_env\Scripts\python.exe' src/exp020_qwen_stack_screen.py --stage run-one --model vnlegal_lal
& 'D:\Study\DSC2026\dsc_env\Scripts\python.exe' src/exp020_qwen_stack_screen.py --stage summarize
```

## Gates and stop condition

- The new structural audit must be `PASS`, preserve all source and
  ground-truth documents, and match its corpus fingerprint.
- Every encoded document must be at most 512 Qwen3 input tokens.  Any
  truncation is a hard failure.
- The query IDs and parent candidates must exactly equal EXP-015 stage-2.
- Record paired first-relevant-rank movements versus E5-508/o64.

If neither Qwen3 model provides a material and stable improvement on the
fixed fixture, do not test LAL at 1024/2048 and retain E5 as the working
candidate.  If one is competitive, it becomes only the working candidate;
final selection still requires later fold-isolated evaluation.
