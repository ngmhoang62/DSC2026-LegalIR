# EXP-111 — Multi-view Sparse Retrieval and Lexical Geometry Fusion

## 0. Mục đích của tài liệu

Đây là implementation plan tự chứa ngữ cảnh cho một agent ở conversation khác. Agent triển khai không được dựa vào lịch sử chat. Trước khi viết code, agent phải đọc các file được liệt kê, tạo `READING_AUDIT.json`, kiểm tra artifact/fingerprint thật và ghi lại mọi khác biệt giữa plan với workspace hiện tại.

EXP-111 mở lại nhánh sparse retrieval theo yêu cầu mới của người dùng. Quyết định này ghi đè phần handoff cũ trong `AGENTS.md` nói retrieval đã frozen, nhưng chỉ trong namespace EXP-111. Tuyệt đối không sửa hoặc ghi đè EXP-021, EXP-022, EXP-109B, EXP-109C, EXP-110P hay corpus structural-v3.

Mục tiêu nghiên cứu là xác định xem BM25 hiện tại mới chỉ tối ưu depth/aggregation hay đã thực sự chạm trần lexical retrieval; sau đó xây một sparse ensemble có các retrieval geometry độc lập đủ mạnh để cải thiện Stage 1.

Không dùng LLM, không sinh dữ liệu, không annotate nhãn/evidence mới, không tải model. EXP-111 là CPU/disk experiment; GPU không phải điều kiện chạy.

---

## 1. Kết luận audit dẫn đến EXP-111

### 1.1 Timeline đúng của các experiment cũ

Agent phải kiểm chứng lại từ source/report, nhưng context hiện đã xác nhận:

- EXP-013/013b dùng BM25/BGE và query-memory trong candidate/fusion pipeline cũ. BM25 chưa phải nhánh multi-view sparse hiện đại.
- EXP-014 kế thừa source rankings cũ, tạo candidate union và LambdaMART/reranking. Nó không phải một ablation tokenizer/BM25 độc lập.
- EXP-015 là model screen; EXP-016 audit parser/corpus; EXP-017–019 chủ yếu screen chunk configuration cho dense E5; EXP-020 screen Qwen/prefix. Không được mô tả EXP-015–020 là đã exhaustively tune BM25.
- EXP-021 mới là tuning sparse chính trên corpus final structural-v3: Underthesea, SQLite FTS5, structural passage, hierarchy, aggregation, retrieval depth và RRF.
- EXP-024/025 đã thử query-memory/char-word lexical backoff và identifier/entity diagnostics. Raw char backoff chỉ rescue rất ít; query-memory rescue bị tập trung vào vài frequent labels và không ổn định theo fold. Không lặp lại các nhánh đó trong core EXP-111.
- EXP-026/027 và EXP-109B chứng minh document-level LambdaMART từ rank/score/agreement features có thể cải thiện head ranking. EXP-109B strict-inner F1–F4 đạt `Recall@5=0.9251057869956494`; locked Fold 0 đạt `0.9331783500238435`.

Các số EXP-013b/014/109B thuộc scope khác nhau. Không so aggregate OOF với một Fold-0 như cùng một benchmark.

### 1.2 Current tuned sparse contract của EXP-021

Nguồn sự thật:

- `src/exp012b_bm25.py`
- `src/exp021_sparse_retrieve.py`
- `src/exp021_sparse_depth_tune.py`
- `results/exp021_sparse/REPORT.md`
- `results/exp021_sparse/depth_rrf_tuning/tuning_report.json`
- `results/exp021_sparse/depth_rrf_tuning/oof_rankings.jsonl`

Contract hiện tại:

```text
corpus: 8,507 parents / 343,347 structural chunks
tokenizer: Underthesea word_tokenize(format="text"), casefold
index: SQLite FTS5 unicode61, underscore là token character
query: quoted tokens nối bằng OR
fields: hierarchy weight 2 + passage content weight 1
parent aggregation: first-passage + RRF3
selected depth: 1,024 passages
head/tail: fused top 16, sau đó RRF3 tail
```

EXP-021 đã chứng minh:

- Scope field làm giảm mạnh head recall.
- Folded title làm giảm Recall@5; `document_label` phần lớn là slug không dấu và không được tái sử dụng như evidence text.
- First passage tốt cho head; RRF3 hữu ích hơn ở tail.
- Depth/RRF tuning tăng head recall, nhưng cache chỉ giữ ranks, không giữ đủ raw BM25 scores để thử scorer/aggregation mới.
- OOF curve được báo cáo: R@5 `0.8351357143`, R@16 `0.9262904762`, R@50 `0.9657523810`, R@100 `0.9767404762`.

Các con số trên là metric của artifact EXP-021 và phải được reproduce bằng canonical evaluator trước khi dùng làm baseline EXP-111.

### 1.3 Direct failure audit đã đo trước khi viết plan

Audit read-only trên `oof_rankings.jsonl`, train labels và source-exact chunks cho thấy:

- Trong 6,989 query có retained raw gold và ranking ở phép audit nhanh, 1,014 query không có gold trong top 5.
- First-gold rank bucket: 441 ở rank 6–10; 258 ở 11–20; 150 ở 21–50; 60 ở 51–100; 105 ở trên 100/không có.
- Với 1,014 top-5 misses, surface informative-token coverage trung bình của toàn gold parent là `0.9351`; best gold chunk là `0.7971`.
- Với matched rank-1 controls, các giá trị tương ứng là `0.9836` và `0.9252`.
- Chỉ 7/1,014 misses có gold-parent token coverage dưới 0.4. Vì vậy broad lexical gap không phải failure mode chính.
- 128/1,014 misses có khoảng cách `whole-parent coverage - best-chunk coverage >= 0.30`, là bằng chứng cho distributed evidence/chunk geometry.
- Top-1 confuser có trung bình parent coverage `0.9431`, nhỉnh hơn gold `0.9351`; scorer OR-unigram thường thấy gần như cùng bộ từ ở gold và confuser.
- Gold chứa ít nhất một surface bigram của query ở 1,003/1,014 misses và trigram ở 887/1,014. Binary phrase existence không đủ; graded phrase/proximity score vẫn là hypothesis hợp lệ.
- Numeric queries và strict legal-identifier queries yếu hơn tổng thể trong audit nhanh. Chúng là specialist slice nhỏ, không được phép làm global identifier boost như EXP-107.
- Raw `document_label` overlap rất thấp trong audit và title-folded từng làm giảm score. Không dùng slug title làm global retrieval field.

Lưu ý quan trọng: audit nhanh dùng surface regex để mô tả lexical coverage, không phải exact Underthesea/FTS tokenizer. EXP-111 phải tái tạo audit bằng exact token streams. Ngoài ra, `error_audit.json` cũ báo slice fallback rất thấp trên một pre-tune aggregation, trong khi slicing trực tiếp tuned OOF rankings cho kết quả khác. Agent phải ghi rõ provenance và không trộn hai slice.

### 1.4 Điều thực sự đáng tham khảo từ repo `dsc2026-legalir-repro`

Đọc code, không chỉ README:

- `benchmark_burst_v4_full_sqlite.py`
- `tune_burst_phrases.py`
- `tune_burst_score_ltr.py`
- `tune_burst_pairwise.py`
- `tune_burst_memory.py`
- `tune_burst_supervised_profile_bm25.py`
- `tune_burst_graph_posterior.py`
- `run_burst_multistage_submission.py`

Phần transferable, label-free:

1. Full-document BM25 view.
2. Fixed sliding-window BM25 view, độc lập với structural parser.
3. Parent score từ best local window cộng discounted second non-redundant window.
4. Bigram/trigram phrase retrieval.
5. Rank/score/agreement fusion thay vì coi một BM25 ranking là đủ.

Phần không được bê nguyên vào core:

- Query memory dùng qrels.
- Supervised document profiles xây từ labelled queries.
- Co-relevance graph dùng gold labels.
- Các contiguous block 100-query tuning/validation trong repro không tương đương fixed five-fold protocol của project.
- Repo không chứa locally verifiable result artifacts cho claim sparse R@5 0.94+. Claim đó là hypothesis/motivation, không phải baseline có thể so trực tiếp.

Linh hồn nên lấy là **diversity of lexical geometry**, không phải một weight cụ thể từ repo.

---

## 2. Research questions và target

EXP-111 phải trả lời tuần tự:

1. Underthesea compound-token view có bỏ sót tín hiệu mà raw surface tokens giữ được không?
2. Fixed overlapping windows có sửa parser/fallback và chunk-boundary errors không?
3. Full-parent BM25 có bổ sung global-document evidence mà best structural chunk bỏ lỡ không?
4. Phrase/proximity có phân biệt gold khỏi lexical confuser tốt hơn OR-unigram không?
5. Các view trên có tạo complementarity thật hay chỉ nhân bản một tín hiệu BM25 tương quan?
6. Sparse winner có thay thế/bổ sung BM25 cũ trong EXP-109B để tăng Stage-1 Recall@5 không?

Targets, theo scope:

- Sparse standalone inner F1–F4: tối thiểu `0.90`, ambitious `0.93–0.94`.
- Corrected E5+LAL+sparse inner fusion: tối thiểu `0.935`, milestone `0.94+`.
- Candidate ceiling: union sparse sources @100/source phải đủ cao để top-5 target không bị chặn.
- Stretch objective cuối cùng vẫn là Stage-1 `0.96–0.97`, nhưng EXP-111 không được hứa đạt mốc đó trước khi source oracle, candidate ceiling và strict-inner fusion cùng ủng hộ.

Primary metric là canonical Recall@5. Secondary: Precision@5, R@1, MRR@5, R@10/16/30/50/80/100, single/multi-gold Recall@5.

---

## 3. Namespace và deliverables

Không sửa experiment cũ. Tạo:

```text
src/exp111_multiview_sparse_retrieval.py
tests/test_exp111_multiview_sparse_retrieval.py
docs/exp111_runbook.md
cache/exp111_multiview_sparse/
results/exp111_multiview_sparse/
```

Artifacts bắt buộc:

```text
results/exp111_multiview_sparse/READING_AUDIT.json
results/exp111_multiview_sparse/INPUT_AUDIT.json
results/exp111_multiview_sparse/REPRODUCTION_REPORT.json
results/exp111_multiview_sparse/LEXICAL_FAILURE_AUDIT.json
results/exp111_multiview_sparse/BOUNDED_SOURCE_SCREEN.json
results/exp111_multiview_sparse/FULL_SOURCE_AUDIT.json
results/exp111_multiview_sparse/FROZEN_INNER_SPARSE_REPORT.json
results/exp111_multiview_sparse/DENSE_COMPLEMENT_REPORT.json
results/exp111_multiview_sparse/LOCKED_CONFIG.json
results/exp111_multiview_sparse/FOLD0_REPORT.json   # chỉ sau authorization
results/exp111_multiview_sparse/RUN_STATUS.json
```

Mỗi generated cache có manifest, code/input fingerprints, record count, SHA-256 và `_SUCCESS.json`. Không sửa JSON thủ công để pass gate.

---

## 4. Mandatory reading order

Agent phải đọc toàn bộ các file sau trước khi code:

1. `AGENTS.md`.
2. `cache/final_preprocessed_v2/manifest.json`, `exclusions.json`, `train_label_impact.jsonl`.
3. `cache/structural_v3_e5_final_v1/manifest.json`, `_SUCCESS.json`, schemas/examples của `documents.jsonl`, `chunks.jsonl`, `nodes.jsonl`.
4. `src/exp012b_bm25.py`.
5. `src/exp021_sparse_retrieve.py`, `exp021_sparse_depth_tune.py`, `exp021_sparse_error_audit.py`.
6. Toàn bộ `results/exp021_sparse/REPORT.md` và reports nêu ở §1.2.
7. `src/exp013*_*.py`, `src/exp014/candidates.py`, `src/exp014/ranker.py`, và relevant EXP-013/014 reports để xác định đúng historical scope.
8. `src/exp015_model_screen.py`, `exp015_stage2_content_ablation.py`, `exp017_chunk_config_screen.py`, `exp018_e5_chunk_config_screen.py`, `exp019_e5_overlap_screen.py` để không gán nhầm dense screens thành BM25 tuning.
9. `results/exp024_memory_lexical/report.json`, `results/exp025_both_model_miss/report.json`.
10. `src/exp027_lambdamart_shortlist.py`.
11. `docs/EXP-109B_PLAN.md`, `src/exp109b_encoder_complementarity.py`, cached fusion pilot và locked Fold-0 reports.
12. Các file repro liệt kê ở §1.4.

`READING_AUDIT.json` phải ghi path, SHA-256, điều học được, phần được reuse, phần bị cấm reuse. Nếu path/artifact thiếu hoặc fingerprints không hợp lệ, dừng `REJECTED_READING_OR_INPUT_GATE`.

---

## 5. Fold isolation và canonical labels

Canonical policy bắt buộc:

```text
canonical_duplicate_alias_drop_empty_passage_v1
6,991 evaluable queries
9 non-evaluable queries
```

Agent phải reuse canonicalization implementation đã được kiểm chứng trong các experiment mới hơn; không tự viết mapping khác. Audit exact label fingerprint.

Outer holdout là Fold 0:

- Mọi tokenizer/view/window/aggregation/depth/RRF/LambdaMART choice dùng Folds 1–4.
- Fold 0 labels không được load vào process trước khi `LOCKED_CONFIG.json` được tạo.
- Label-free corpus indexes có thể xây cho toàn corpus vì không dùng qrels.
- Với cross-fit F1–F4, khi đánh giá một validation fold, weights/depth/model được fit hoặc chọn từ ba fold còn lại.
- EXP-111 không được đọc public labels, tune theo leaderboard hay tự tạo submission.

CLI Fold-0 phải yêu cầu đồng thời:

```text
--authorize-fold0
EXP111_ALLOW_FOLD0=1
```

---

## 6. Phase 0 — Input audit và exact reproduction

### 6.1 Input audit

Kiểm tra:

- 8,507 retained parent documents.
- 343,347 structural chunks.
- 7,000 train queries; 6,991 evaluable; 9 non-evaluable.
- Fold membership unique và phủ đủ 7,000 query.
- Final preprocessing exclusions/duplicate mappings đúng manifest.
- EXP-021 FTS database, rankings và tuning report fingerprint hợp lệ.
- EXP-109B E5/LAL/BM25 cached ranking manifests hợp lệ nếu đi tới dense complement phase.
- Disk trống tối thiểu 25 GiB trước build; RAM preflight không được materialize toàn corpus text.

### 6.2 Reproduction contract

Hai mức:

1. Evaluator replay: đọc EXP-021 OOF ranking, canonicalize labels, reproduce exact metric curve.
2. Scorer fixture: chọn ít nhất 100 real queries cố định bằng hash từ F1–F4, chạy current `BM25Searcher`/aggregation và compare exact top-200 document IDs, ranks và raw FTS scores với frozen fixture được tạo một lần từ owning EXP-021 code.

Tolerance:

- IDs/rank order: exact.
- Raw SQLite BM25 score: `abs diff <=1e-12` trên cùng environment.
- Metric: `<=1e-12`.

Không được nới gate vì tie hypothesis chưa trace. Nếu deterministic ties tồn tại, lock tie-break bằng `doc_id/chunk_id` và tạo evidence first-mismatch.

Gate:

```text
PASS_REPRODUCTION
REJECTED_REPRODUCTION_GATE
```

---

## 7. Phase 1 — Exact lexical failure audit

Không tune từ audit labels; đây là diagnostic.

### 7.1 Audit tokenizer

Đối với mọi query và gold/top confuser:

- Lưu raw surface tokens.
- Lưu exact Underthesea tokens.
- Lưu exact tokens sau FTS5 normalization.
- OOV/query token document-frequency statistics.
- Compound-token split/merge disagreements.
- Diacritic/`đ`/Unicode normalization differences.
- Punctuation, slash, dash, decimal, percent, currency, date, legal citation behavior.

Không dùng một handwritten stopword list làm production filter. Stopword candidates được suy ra từ corpus DF và chỉ được screen như query-policy ablation.

### 7.2 Audit matching và geometry

Cho mỗi query:

- Gold first-rank và rank bucket.
- Whole-parent query-term coverage.
- Best structural-chunk coverage.
- Best fixed-window coverage cho window configs ở Phase 2.
- Bigram/trigram counts và proximity spans ở gold vs top-1/top-5 confusers.
- Number of matching chunks; repeated-term dispersion.
- Gold parse mode, source length, chunk count.
- Legal citation/numeric flags.
- Label-noise suspicion chỉ là diagnostic; không sửa nhãn.

Failure tags tối thiểu:

```text
tokenization_split_merge
rare_or_oov_surface
legal_identifier_normalization
numeric_unit_normalization
distributed_across_chunks
structural_boundary
fallback_parser
phrase_confuser
high_df_term_dilution
near_duplicate_confuser
likely_label_or_corpus_mismatch
unclassified
```

Xuất ít nhất 20 source-exact examples mỗi major class nếu có, gồm qid, canonical gold, top docs, matched terms, offsets/chunk IDs và scores. Không copy hàng MB raw text vào report; dùng short source-exact snippets + hashes.

---

## 8. Phase 2 — Build label-free sparse views

Tất cả views dùng canonical retained parent IDs. Không dùng `document_label` làm global evidence field.

### V0 — Structural Underthesea control

- Exact EXP-021 passage+hierarchy representation.
- Rerun score collection để giữ raw FTS score, passage rank, parent IDs và top-3 distinct parent-node hits.
- Đây là control, không phải artifact copy thiếu scores.

### V1 — Structural surface-token view

- Cùng structural chunks/hierarchy với V0.
- Tokenizer đơn giản Unicode surface words, lower/casefold, FTS5 `unicode61`.
- Không Vietnamese word segmentation; mục tiêu là diversity, không thay thế V0 mặc định.
- Preserve slash/dash-derived normalized tokens qua specialist extractor, không nhét punctuation thô vào FTS expression.

### V2 — Fixed overlapping local-window view

Build từ exact retained raw passage, độc lập structural parser.

Bounded configs:

```text
W384/O96
W512/O128
```

Word offsets phải map được về source character offsets. Mỗi window lưu:

```json
{
  "window_id": "doc:start:end",
  "doc_id": "...",
  "word_start": 0,
  "word_end": 0,
  "char_start": 0,
  "char_end": 0,
  "raw_text_hash": "..."
}
```

Windows cover toàn source, deterministic, không silent gaps. Overlap là chủ ý.

Parent aggregation candidates:

```text
best_window_score
best + lambda * second_nonredundant
lambda in {0.30, 0.60}
```

Second window chỉ hợp lệ nếu overlap ratio với best `<=0.50` hoặc center distance đủ lớn. Không được thưởng hai overlapping copies của cùng evidence.

### V3 — Full-parent surface BM25

- Một FTS row/retained parent, raw passage source-exact.
- Không prepend slug label.
- Full-parent BM25 là global-context complement; không được giả định nó tự mạnh hơn local BM25.
- Lưu document length và normalized raw score.

### V4 — Phrase/proximity local view

Reuse V2 index; không cần duplicate corpus database nếu FTS schema cho phép.

- Query bigrams và trigrams từ surface tokens.
- Chỉ tạo phrase nếu chứa ít nhất một/two informative terms theo corpus DF policy.
- Exact quoted phrase OR retrieval cho bigram và trigram tách riêng.
- Không dùng binary “phrase exists” làm final score; giữ BM25 phrase score, ranks, counts và shortest matched span nếu khả dụng.
- Phrase view được xem là một family với local-window view để tránh multiplicity bias.

### V5 — Legal citation specialist

High-precision regex/normalizer cho:

```text
33/2023/NĐ-CP
123/2020/NĐ-CP
80/2021/TT-BTC
Điều/Khoản/Điểm references
dates, percentages, money and measurement composites
```

Sinh synthetic index tokens ổn định, ví dụ `cite_33_2023_nd_cp`; query và source dùng cùng parser. Citation match là candidate/feature specialist, không phải global additive boost. Nếu parser confidence thấp, trả empty specialist features.

### Index constraints

- Dùng SQLite FTS5 để giữ RAM bounded; không thêm dependency mới.
- Contentless index chỉ khi source sidecar đủ để trace offsets.
- WAL/temp/cache policy phải được benchmark và documented.
- Build resume theo committed shard; marker có input/code/config hash.
- Không load 343k chunks hay full raw passages thành Python objects cùng lúc.

---

## 9. Phase 3 — Retrieval, score preservation và parent features

Depth để score source, không phải final K:

```text
V0 structural hits: 2,048 passages
V1 structural hits: 2,048 passages
V2 local hits:      3,000 windows
V3 full parents:      500 parents
V4 bigram:          1,500 windows
V4 trigram:         1,500 windows
V5 citation: all exact matches, capped deterministically at 500
```

Agent có thể giảm depth chỉ sau bounded latency/coverage proof; không giảm âm thầm.

Mỗi source ranking lưu ít nhất top 500 parents hoặc toàn bộ retrieved parents nếu ít hơn:

```json
{
  "qid": "...",
  "source": "...",
  "documents": [
    {
      "doc_id": "...",
      "rank": 1,
      "raw_score": 0.0,
      "robust_z": 0.0,
      "best_unit_id": "...",
      "best_score": 0.0,
      "second_score": 0.0,
      "score_gap": 0.0,
      "matching_units": 0
    }
  ]
}
```

Score normalization trong query:

- max normalization chỉ là feature.
- median/MAD robust z-score là feature.
- rank/reciprocal rank luôn giữ.
- Không thay raw source score bằng reciprocal final rank như lỗi từng gặp ở EXP-107.

Candidate curves bắt buộc ở:

```text
K = 1, 5, 10, 16, 20, 30, 50, 80, 100, 200
```

Không chọn một K=150 tùy ý và gọi đó là Stage-1 metric.

---

## 10. Phase 4 — Bounded source screen

Chọn trước 512 qids bằng stable hash từ F1–F4, không dùng labels để chọn cohort. Error cohorts từ Phase 1 chỉ báo cáo riêng, không dùng tune.

So sánh:

- V0 control.
- V1.
- V2 configs.
- V3.
- V4 bigram/trigram.
- V5 citation slice.
- Family-level oracle và full sparse source oracle, đánh dấu label-dependent diagnostic.

Mỗi new view báo:

- Standalone R@K.
- Unique gold additions @5/@20/@50 so V0.
- Pairwise top-K overlap.
- Gold-in/confuser-out movements.
- Parse-mode and failure-tag gains/losses.
- Runtime/query, database size, peak RSS.

Gate để chạy full F1–F4: ít nhất hai trong ba điều kiện:

1. Full sparse choice-oracle gain over V0 @5 `>=0.010`.
2. Ít nhất 10 unique gold occurrences vào top 20 từ new views trên bounded cohort.
3. Best learned/rank-fused bounded ranking tăng R@5 `>=0.010` và không giảm multi-gold.

Đồng thời candidate union R@50 không thấp hơn V0. Nếu fail: `REJECTED_BOUNDED_SPARSE_COMPLEMENTARITY_GATE`.

Không dùng source-oracle threshold 0.95 làm hard reject; EXP-109B đã cho thấy document-level fusion có thể vượt một source-routing oracle đơn giản.

---

## 11. Phase 5 — Full F1–F4 source audit

Chỉ score F1–F4. F0 qids không xuất hiện trong reports/cache ở phase này.

Báo cáo:

- Canonical standalone metrics mỗi view/family.
- Source oracle và union candidate ceiling.
- Per-view exclusive gold contributions.
- First-gold rank transitions vs V0.
- 4-fold breakdown.
- Single/multi-gold, structured/fallback, identifier/numeric, short-query slices.
- Bootstrap CIs cho paired deltas.

Candidate viability expectations, không phải mọi mục đều hard gate:

```text
union sparse sources @50/source >= 0.985
union sparse sources @100/source >= 0.992
```

Nếu thấp hơn, report phải ghi ceiling warning; vẫn được chạy fusion nếu unique head signal tồn tại.

---

## 12. Phase 6 — Hierarchical sparse fusion

### 12.1 Tránh source multiplicity bias

Không cho ba phrase/window variants ba lá phiếu ngang với một full-doc view. Fuse theo family:

```text
structural family: V0 + V1
local family: V2 unigram + V4 bigram + V4 trigram
global family: V3
specialist family: V5
```

Mỗi family chọn subconfig bằng cross-fit, rồi family rankings mới đi vào final fusion.

### 12.2 Cross-fitted weighted RRF

Với mỗi validation fold trong F1–F4:

- Dùng ba folds còn lại để chọn window config, lambda, family weights và RRF k.
- Apply unchanged lên validation fold.
- Grid coarse, deterministic; không hàng nghìn correlated trials.

Grid:

```text
RRF k in {5, 10, 20, 40}
family weights theo simplex step 0.10
minimum structural/local/global weight 0.10 nếu family còn active
specialist weight có thể 0
```

Selection order: R@5, multi-gold R@5, Precision@5, MRR@5, ít families hơn.

### 12.3 Cross-fitted sparse LambdaMART

Reuse audited pure helper logic từ EXP-109B hoặc copy vào EXP-111 namespace; không gọi CLI/stages của EXP-109B.

Candidate set là union top-50 per active sparse family. Báo mean/min/max unique parents/query.

Allowed features:

- Per-view/family rank, reciprocal rank, raw score, robust z.
- Presence and source agreement at top 5/10/20/50.
- Best/second local score, nonredundant gap, matching-unit count.
- Phrase rank/score/count.
- Exact citation/numeric match flags.
- Query token count, unique-token ratio, high-DF ratio, citation/numeric flags.
- Parent length, structural chunk count, fixed-window count, parse mode.
- Score margins and rank dispersions.

Forbidden:

- `doc_id` or document label as categorical/learned feature.
- Fold/row order.
- Nearest labelled query, query memory, label frequency, supervised profile or graph.
- Gold-derived inference feature.

Use the fixed, already-audited EXP-109B LightGBM hyperparameter family. Nếu thử nhiều configs, mỗi validation fold phải chọn config chỉ từ ba folds còn lại. Không tune trực tiếp trên aggregate F1–F4 rồi báo cùng aggregate là unbiased.

Sparse winner là best giữa weighted RRF và LambdaMART theo exact cross-fit predictions.

### Sparse promotion gate

`PROMOTE_SPARSE_WINNER` nếu:

- R@5 `>=0.900`.
- Delta vs canonical EXP-021 V0 `>=+0.040` absolute.
- Ít nhất 3/4 folds dương; worst fold `>=-0.002`.
- Bootstrap 95% CI lower bound của R@5 delta `>0`.
- Multi-gold R@5 không giảm.
- Precision@5/MRR@5 không giảm quá 0.002.

Nếu sparse gain dương nhưng chưa đủ gate, gắn `KEEP_COMPLEMENT_ONLY` và vẫn được đi Phase 7 khi có exclusive recovery signal. Không gọi nó standalone breakthrough.

---

## 13. Phase 7 — Complementarity với current Stage 1

Đây là phase cache-only; không encode E5/LAL lại.

Anchor bắt buộc:

```text
EXP-109B strict cross-fit F1–F4
E5 + LAL + old tuned BM25
LambdaMART top-50/source
Recall@5 = 0.9251057869956494
```

Reproduce exact anchor candidate order, feature order, scores và predictions trước khi thay source. Aggregate metric match nhưng predictions khác thì reproduction fail.

So sánh strict cross-fit:

```text
A. E5 + LAL + old BM25                         # exact anchor
B. E5 + LAL + EXP-111 sparse winner            # replace old BM25
C. E5 + LAL + old BM25 + sparse families       # augment
D. E5 + EXP-111 sparse winner                   # cost-aware diagnostic
```

Candidate policy là top-50 **mỗi source/family**, unique union, không truncate thành 50 total. Lưu pool min/mean/max.

Config selection dùng F1–F4 cross-fit như Phase 6. Không đọc F0.

### Dense complement promotion gate

Winner phải:

- R@5 `>=0.935`.
- Delta vs exact EXP-109B anchor `>=+0.0075`.
- Bootstrap 95% CI lower bound `>0`.
- Ít nhất 3/4 folds dương; worst fold `>=-0.002`.
- Multi-gold R@5 không giảm.
- Precision@5 và MRR@5 không giảm quá 0.002.
- Candidate union @50/source không thấp hơn anchor.

Nếu `>=0.94`, đánh dấu `PASS_INNER_094_MILESTONE`. Nếu delta dương nhưng dưới gate, `KEEP_WEAK_SPARSE_COMPLEMENT`; không đọc F0.

---

## 14. Phase 8 — Lock và Fold-0 evaluation

Chỉ tạo `LOCKED_CONFIG.json` sau khi Phase 7 pass. File chứa:

- Active views/families.
- Tokenizers, window config, lambda.
- Source depths.
- Candidate contract.
- RRF/LambdaMART config and feature order.
- Training qids and fold isolation proof.
- Input/code/artifact hashes.
- Exact anchor reproduction hash.

Sau explicit authorization:

1. Train final sparse/fusion model trên F1–F4.
2. Score F0 label-free.
3. Lock predictions/hash.
4. Chỉ sau đó load canonical F0 labels để evaluate một lần.

Báo cáo F0:

- Anchor EXP-109B matched under exact same candidate/evaluator contract.
- EXP-111 winner.
- R@1/5/10/16/30/50/100, Precision@5, MRR@5.
- Single/multi-gold.
- Per-view rescues/losses.
- Query-level wins/losses/ties and bootstrap CI.
- Candidate ceiling and gap to 0.96/0.97.

Promotion to full OOF chỉ khi:

- F0 R@5 `>=0.945`.
- Delta vs matched EXP-109B F0 `>=+0.005`.
- Multi-gold, Precision@5 và MRR@5 không giảm.

Không tự chạy full OOF hoặc public submission. Báo cáo và chờ user quyết định.

---

## 15. Tests bắt buộc

### Unit tests

- Canonical duplicate/drop policy and 6991/9 counts.
- Fold isolation and F0 absence.
- Surface/Underthesea/FTS token fixture parity.
- Unicode NFC/NFD, `đ`, slash, dash, decimals, percentages, dates.
- Legal citation normalization positive/negative cases.
- Window coverage, offsets, overlap and deterministic IDs.
- Nonredundant second-window aggregation.
- Phrase query escaping and no FTS operator injection.
- Parent dedup and deterministic tie-break.
- Raw score preservation and robust z edge cases.
- RRF family multiplicity protection.
- Candidate union top-50/source semantics.
- LambdaMART feature order/dtype and missing-source handling.
- No label/doc-ID feature leakage.
- Resume/checkpoint/fingerprint invalidation.
- Gate pass/reject/keep-weak behavior.

### Integration tests

- 20 real parents: source round-trip and window offsets.
- 20 real queries: V0 scorer exact parity.
- NumPy/reference implementation for aggregation/fusion.
- Fixture with phrase specialist rescue.
- Fixture with overlapping duplicate windows not double-counted.
- Fixture with citation confuser.
- Cross-fit test proving validation labels never enter training/selection.
- Interrupted index/scoring resume produces identical hashes.

Tests phải pass trước mọi full index/score run.

---

## 16. CLI, orchestration và logging

CLI đề xuất:

```powershell
python -u src/exp111_multiview_sparse_retrieval.py audit --resume
python -u src/exp111_multiview_sparse_retrieval.py reproduce --resume
python -u src/exp111_multiview_sparse_retrieval.py lexical-audit --resume
python -u src/exp111_multiview_sparse_retrieval.py build-index --view <view> --resume
python -u src/exp111_multiview_sparse_retrieval.py bounded-screen --resume
python -u src/exp111_multiview_sparse_retrieval.py full-source-audit --resume
python -u src/exp111_multiview_sparse_retrieval.py frozen-inner-sparse --resume
python -u src/exp111_multiview_sparse_retrieval.py dense-complement --resume
python -u src/exp111_multiview_sparse_retrieval.py overnight-inner --resume
python -u src/exp111_multiview_sparse_retrieval.py fold0 --resume --authorize-fold0
python -u src/exp111_multiview_sparse_retrieval.py status
```

`overnight-inner` dừng sau Phase 7; không chạy Fold0.

Logs:

```text
results/exp111_multiview_sparse/logs/<run-id>/overnight.log
results/exp111_multiview_sparse/logs/<run-id>/<stage>.log
```

Terminal hiển thị stage, progress, query throughput, DB size, RSS, ETA, heartbeat mỗi 5 phút, gate result và log path. Detailed records ghi file, flush định kỳ; không spam từng query.

`RUN_STATUS.json`:

```json
{
  "run_id": "...",
  "state": "RUNNING|PASS|REJECTED|FAILED|INTERRUPTED|KEEP_WEAK",
  "stage": "...",
  "completed": 0,
  "total": 0,
  "eta_seconds": 0,
  "last_heartbeat": "...",
  "log_path": "..."
}
```

Exit codes: 0 pass/completed, 2 gate rejection, 1 crash. Ctrl+C flush checkpoint/status.

---

## 17. Resource budget và ETA contract

EXP-111 chạy CPU/disk, không cần GPU. Không chiếm tài nguyên Colab của EXP-110P.

Expected local budget phải được benchmark trước khi hứa ETA:

- Indexes/caches: 5–15 GiB dự kiến; preflight yêu cầu 25 GiB free.
- RAM: target dưới 3 GiB RSS; SQLite streaming, không corpus-wide Python objects.
- Bounded screen: khoảng vài chục phút sau khi indexes tồn tại.
- Full F1–F4 source scoring: dự kiến 2–8 giờ tùy SQLite throughput.
- Cross-fit LambdaMART: thường dưới 1 giờ sau khi source scores cached.

Agent phải benchmark 128 queries/view, báo measured ms/query, p50/p95 và extrapolated ETA trước full scoring. ETA phải gắn nhãn estimated và được cập nhật từ actual progress.

---

## 18. Stop rules và interpretation

- Fail reproduction: sửa fidelity, không bypass.
- Fail bounded complementarity: dừng; không build thêm supervised profile/query memory.
- New tokenizer yếu standalone nhưng có exclusive rescues: có thể giữ làm specialist.
- Source oracle thấp không tự động kết luận fusion vô dụng.
- Sparse standalone mạnh nhưng không bổ sung E5/LAL: không thay Stage 1 chỉ vì standalone đẹp.
- Dense complement không vượt anchor có CI dương: đóng EXP-111, giữ audit/index artifacts.
- Không post-hoc nới gate sau khi thấy F0.
- Không diễn giải candidate ceiling, source oracle, inner F1–F4, Fold0 và public score như cùng một metric.

---

## 19. Báo cáo implementation trước khi chạy

Sau khi implement/test nhưng trước full run, agent phải trả về:

1. Files đã tạo/sửa.
2. Checklist từng mục plan: implemented, deferred hoặc deviation có lý do.
3. Test command và exact result.
4. Input/artifact fingerprints.
5. Reproduction result và first mismatch nếu fail.
6. Index/scorer benchmark, disk/RAM, ETA.
7. Stages được phép chạy và stages đang fail-closed.
8. Exact command để resume.

Không được nói “workflow hoàn chỉnh” nếu Fold0/public stages chỉ là stub.
