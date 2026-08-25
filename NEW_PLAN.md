# LegalIR — Preprocessing and Structural-Corpus Notes for the Next Pipeline Plan

> Status: working decisions only. The raw dataset, existing caches,
> experiments, and submissions remain unchanged. The EXP-016--EXP-020
> development results below select a working chunking/retrieval configuration;
> they are not fold-isolated final-model evidence.

## Planned preprocessing decisions

### 1. Fill missing document names from their existing links

- Population: 1,125 documents have no `name` field.
- In the preprocessed corpus, populate a missing `name` deterministically from
  the corresponding `link`: use the final URL path segment and remove its
  file extension (normally `.aspx`).
- Normalize the retrieval-facing `name` for **every** document, whether it was
  supplied or derived: replace hyphen separators (`-`) with spaces and compact
  repeated whitespace.  This does not attempt to restore Vietnamese diacritics
  or rewrite legal identifiers; retain the original supplied/derived value and
  derivation provenance in the preprocessing manifest.
- This is an internal transformation of metadata already supplied in the
  dataset; it must not fetch or add external content.
- Preserve `public_test_dataset/selected-contexts/` unchanged. The
  preprocessing artifact should record the derivation rule/provenance so it
  can be reproduced and audited.

### 2. Exclude documents with missing or empty passages

- Population: 20 documents have an empty `passage`.
- Exclude these documents from the **preprocessed retrieval corpus/index**.
- Do not delete or modify the raw source files.
- When later evaluating train data, explicitly account for queries whose
  ground-truth answers include an excluded empty-passage document; do not
  attribute this unavoidable candidate loss to retrieval quality.

### 3. Collapse exact duplicate raw passages while preserving one document ID

Four exact raw-passage duplicate groups were inspected. The links within each
group are highly similar, so the identical passages appear to be legitimate
duplicates associated with alternative website paths. The preprocessed corpus
will retain exactly one manually selected document from each group and exclude
the other members from the retrieval corpus/index:

| Duplicate group | Retain | Exclude |
|---|---|---|
| 1 | `context_84226` | `context_121575` |
| 2 | `context_206810` | `context_158189`, `context_184972` |
| 3 | `context_280171` | `context_254937` |
| 4 | `context_277743` | `context_35337` |

- The raw files remain intact.
- The future preprocessing manifest must record every retained and excluded ID
  and the duplicate-group reason.
- When later evaluating train data, explicitly account for labels pointing to
  an excluded duplicate ID; this is distinct from a retrieval miss.

## Planned structural chunking and dense-retrieval configuration

### 4. Working retrieval stack: VietLegal-E5

- Select `mainguyen9/vietlegal-e5` as the working dense retrieval model and
  its own tokenizer as the chunking tokenizer.
- The model's required formatting is `query: ` for questions and `passage: `
  for document retrieval text.  Keep this formatting unchanged in encoding and
  retrieval.
- The practical model-input budget is **512 tokens**.  Exact E5 accounting is:

  `508 retrieval_text + 2 passage-prefix tokens + 2 single-sequence special tokens = 512`.

- Therefore the structural chunker configuration is
  `max_passage_tokens=508`, `token_window=476`, and
  `token_overlap=64`.  The numbers 508/476 are internal text/window caps; the
  selected model-input budget remains 512.
- `overlap=64` is deliberately retained as a cheap boundary safeguard.  In
  the prior full-corpus screen it created only 22 more chunks than overlap 0,
  while avoiding a no-overlap policy at window boundaries.
- Do not pursue LAL at 1024/2048 tokens.  The deployment budget is 512, and
  long-context LAL encoding is not practical on the available GPU.

### 5. Evidence supporting the working stack

All figures below are one fixed fold-0 development fixture (512 queries,
1,983 candidate parent documents, at most four chunks per parent), so they
are selection evidence only, not an OOF claim.

| Stack / document representation | R@5 | MRR@5 |
|---|---:|---:|
| VietLegal-E5, E5-tokenized `retrieval_text`, 508/476/64 | **0.8242** | **0.6541** |
| VietLegal-Harrier, Qwen3-tokenized `retrieval_text`, 512 input | 0.8184 | 0.6484 |
| VNLegal-LAL, Qwen3-tokenized `retrieval_text`, 512 input | 0.7930 | 0.6237 |
| VietLegal-Harrier, raw passage | 0.7832 | 0.6099 |
| VNLegal-LAL, raw passage | 0.7461 | 0.5674 |

- Structural `retrieval_text` is retained.  Against raw passage it improved
  R@5 by 3.52 points for Harrier and 4.69 points for LAL on the same fixture.
- EXP-020 verified zero document-input truncations for its Qwen3 models; the
  corresponding E5 508-token configuration had zero E5 inputs above 512.

### 6. Parser policy to carry into the next corpus

- Preserve the document hierarchy and offset-backed chunk identity:
  `Document -> Chapter -> Section -> Article -> Clause/Point`, with Annex as
  a structural boundary/source kind.
- Enable the EXP-016 split-article rule: recognise an article heading when
  `Điều` occupies one physical line and its numeric label begins the next
  line.  The parser must still not interpret ordinary in-sentence references
  such as `theo Điều 12` as headings.
- EXP-016 converted all 149 identified split-article documents from fallback
  to structured parsing.  The remaining fallback documents are accepted for
  now; they must remain auditable and use window chunking rather than being
  silently dropped.
- Preserve raw text, offsets, parent IDs, chunk IDs, source links and
  manifests.  Never hand-edit generated cache files.

### 7. Marker/title budgets: fixed after direct inspection; no tuning ablation

The final prefix maximums, measured in VietLegal-E5 tokenizer tokens, are:

| Marker | Maximum heading tokens |
|---|---:|
| `[Văn bản]` | 48 |
| `[Chương]` | 32 |
| `[Mục]` | 64 |
| `[Điều]` | 24 |

- These are upper bounds, not padding: an absent field is omitted and a short
  heading consumes only its actual tokens.
- These values were set once after deterministic direct-passage samples and a
  full-corpus E5-tokenizer audit; they are not a retrieval-score tuning grid.
  An arithmetic mean is evidence about typical length, but is not used as the
  cap because it would truncate roughly half of headings.
- Audit evidence: normalized document labels have mean 28.04 and only 1/8,532
  exceeds 48; chapter headings have mean 12.01/P95 20; section headings have
  mean 26.33/P95 54; article headings have mean 7.29/P99 22.  The former
  `[Mục]=24` cap clipped 3,661/7,703 section headings (47.5%), which is why it
  is raised to 64 (its P99).  `[Điều]` is reduced to 24 (its P99 is 22).
  `[Văn bản] + [Chương] + [Mục] + [Điều]` still has the same 168-token
  worst-case total as the prior 48/32/24/64 configuration, so the correction
  does not enlarge the prefix budget at the expense of content.
- Do not run a title-budget ablation unless later OOF error analysis identifies
  a concrete title-truncation failure.  Any such change must be a separate,
  controlled experiment.
- Annex is real corpus content, not a spurious parser pattern: the audit found
  9,611 Annex headings and direct raw-passage samples beginning with `PHỤ LỤC`.
  Retain Annex as a parser boundary/source kind.  It is not currently a
  dedicated retrieval-prefix line (`[Phụ lục]`), and no new Annex prefix is
  introduced in the final build without its own ablation.

### 8. Required final-build gates

1. Apply the planned preprocessing transformation into a new, manifested
   corpus namespace: derive the 1,125 missing names, exclude the 20 empty
   passages, and apply the manually selected duplicate retention policy.
   Never modify `public_test_dataset/selected-contexts/`.
2. Build a new structural corpus with the E5 configuration in section 4 and
   the parser policy in section 6.  Do not overwrite EXP-016--EXP-020 caches.
3. Audit the corpus before encoding: audit status `PASS`, source-document
   accounting, parent/offset round-trip coverage, parse-mode counts, token
   P100 at most 508 for `retrieval_text`, the fixed section-7 prefix caps, and
   explicit accounting for any ground-truth IDs intentionally excluded by
   preprocessing.
4. Only after the audit passes, encode and cache VietLegal-E5 document
   embeddings.  Bind embedding caches to the structural-corpus fingerprint,
   model ID, tokenizer, prefixes, max length, dtype and normalization policy;
   make encoding resumable.
5. Downstream retrieval must aggregate chunk scores to parent document IDs and
   output at most five parent IDs.  Any later model/fusion/threshold choice
   must use fold-isolated OOF evaluation, not the development fixture above.
