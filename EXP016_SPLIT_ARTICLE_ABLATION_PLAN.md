# EXP-016 — split-article-heading parser ablation

## Hypothesis

The 149 v3 fallback documents classified as `split_Dieu_number_heading` become
structured when the parser accepts a physical line containing `Điều` followed
by a numeric article label one to three lines later.

## Isolated change

- New opt-in flag: `--allow-split-article-heading`.
- Regex only matches `Điều` at a physical line start followed by a simple
  numeric label (`1`, `1a`, `1đ`) within at most three line breaks.
- Tokenizer, 384-token budget, 352-token window, 32-token overlap, source
  files, and all other parsing rules remain the same as `structural_v3`.

## Namespace and acceptance gates

- Corpus: `cache/exp016_split_article_v3`.
- Audit: `results/exp016_split_article_v3_audit`.
- Required: audit `PASS`; all 8,532 documents and every train ground-truth
  parent represented; no token-budget violation.
- Primary measurement: transitions from baseline fallback to candidate
  structured, especially the 149 pre-audited IDs.
- Secondary: total nodes/chunks, fallback count, split reasons and token
  percentiles relative to frozen `structural_v3`.

No embedding, retrieval, model selection, tokenizer, or token-budget change is
part of EXP-016.
