# EXP-033 — Evidence-first Capsule v2 và BGE residual reranking

## Tóm tắt

- Không khôi phục runner EXP-030/031 và không xem các capsule cũ là kết quả có thể quảng bá. Giữ EXP-032 làm bằng chứng chẩn đoán: scope được cấp gần như đại trà, có bias do `max` trên số view khác nhau, và answer evidence đôi khi bị đẩy khỏi cửa sổ 512 tokens.
- Giữ nguyên retrieval E5@100 + BM25 novel@50, pool 150 parent documents, nested LambdaMART và K=64. Không xây retriever thứ ba, không thay corpus và không submission.
- Xây EXP-033 theo đúng chuỗi: audit parser → tìm evidence trong từng shortlisted document → Capsule v2 answer-first → nested screen → chỉ khi evidence gate đạt mới fine-tune BGE.
- BGE không được tự do thay thế LambdaMART. Nó là bộ residual correction có ngưỡng bảo vệ top đầu.

## Cập nhật ngữ cảnh và handover

- Cập nhật `AGENTS.md` bằng các quy tắc ổn định:

  - Canonical gold dùng `canonical_duplicate_alias_drop_empty_passage_v1`: 6.991 evaluable và 9 non-evaluable.
  - Capsule phải có đúng một representation/document, answer-first, tokenizer-audited; cấm variable-view `max`.
  - Scope, đối tượng và quan hệ pháp lý là evidence tùy chọn được chọn theo query, không được prepend đại trà hoặc đẩy answer evidence khỏi budget.
  - Sau khi parent document vào K64, phải cho phép tìm evidence trên toàn bộ chunks của chính document đó; upstream top-2 chunks không mặc nhiên là answer-bearing.
  - Heavy reranker là residual trên LambdaMART và phải có fallback; không được coi candidate oracle là chất lượng reranker.
  - Ưu tiên FP32. Full fine-tuning chỉ khi backward+optimizer preflight thật sự vừa VRAM với 10% headroom; nếu không dùng FP32 LoRA, rồi mới FP16 LoRA. Không tự động dùng QLoRA.
  - Thay đoạn “next stage là benchmark heavy reranker” đã lỗi thời bằng EXP-033 evidence-first workflow.

- Ghi đè hoàn toàn `HANDOVER_PROMPT.md`:

  - Nêu rõ EXP-030/031 `REJECTED`, không resume runner cũ.
  - Liệt kê bắt buộc phải đọc `AGENTS.md`, EXP-030/031 reports, gate reassessment và EXP-032 audit.
  - Chép đầy đủ mục tiêu, giao thức, gates, CLI, tests và stop conditions của plan này.
  - Cảnh báo 320 query EXP-031 đã được dùng để chẩn đoán; không gọi kết quả EXP-033 là independent test. Báo riêng sensitivity khi loại 320 query này.
  - Yêu cầu kiểm tra process/artifact hiện tại thay vì giả định runner hoặc cache còn tồn tại.
  - Hướng dẫn agent mới bắt đầu bằng audit/read-only, sau đó implement EXP-033; phải dừng tại checkpoint spot-check parser và dừng trước training nếu evidence gate thất bại.

## Implementation EXP-033

- Tạo namespace `exp033_in_document_evidence_routing` với CLI:

  - `audit-inputs`
  - `sample-scope-audit`
  - `finalize-scope-audit`
  - `build-parent-index`
  - `score-in-document`
  - `build-capsules-v2`
  - `screen-evidence`
  - `preflight-bge`
  - `train-bge`
  - `evaluate-bge`
  - `report`
  - `overnight`

### 1. Audit parser Phạm vi/Đối tượng

- Lấy 200 document không trùng, có source-exact spans, stratified:

  - 50 `scope_of_regulation`.
  - 50 `applicable_subjects`.
  - 40 `combined`.
  - 30 `rejected_legacy_scope`.
  - 30 không có typed scope nhưng có tín hiệu từ raw text.

- Annotation JSONL lưu document/node ID, raw span, offsets, parser label, human-review label, boundary completeness, missed span, rationale và source fingerprint.
- Agent annotate đủ 200; xuất 30 mẫu cân bằng để người dùng spot-check.
- Nếu agreement dưới 27/30, adjudicate toàn bộ disagreement và xuất thêm 30 mẫu mới.
- Parser được chấp nhận khi micro precision ≥0,95, từng class precision ≥0,90, boundary accuracy ≥0,95 và stratified-weighted recall ≥0,90.
- Nếu fail, chỉ sửa parser/metadata sidecar trong EXP-033:

  - Giữ exact source offsets.
  - Rút span theo sibling structural boundary thay vì nối/cắt mù 1.800 ký tự.
  - Bổ sung rule chỉ từ error taxonomy đã annotate.
  - Không rebuild Structural v3 trừ khi audit chứng minh sai parent/offset/hierarchy.

### 2. In-document evidence selector

- Tạo fingerprinted `doc_id → chunk rows/embedding indices` từ 343.347 chunks.
- Tái sử dụng frozen VietLegal-E5 chunk embeddings và train-query embeddings; không encode lại corpus.
- Dùng `BM25Searcher.search_document()` trên final `passage_hierarchy` FTS5 để bảo đảm lexical evidence luôn thuộc đúng parent.
- So sánh fold-isolated các selector:

  - Current upstream E5 top-2.
  - In-parent E5 top-1.
  - In-parent E5 top-2 với MMR λ ∈ {0,70; 0,85}.
  - In-parent BM25 top-2.
  - Hybrid E5/BM25 reciprocal-rank với dense weight ∈ {0,25; 0,50; 0,75}.

- Primary chunk là chunk điểm cao nhất. Secondary chỉ được nhận khi khác primary, redundancy cosine <0,90 và còn token budget.
- Clause-aware expansion lấy heading/parent/sibling liền kề để hoàn chỉnh Điều/Khoản/Điểm; không thay bằng một chunk chủ đề xa chỉ để tăng diversity.
- Scope/đối tượng được score trực tiếp với query. Chỉ chọn tối đa một span; regex chỉ là high-precision prior, semantic score xử lý trường hợp regex không nhận diện. Khi không chắc chắn, bỏ scope.

### 3. Capsule v2

- Mỗi query-document chỉ có một capsule, dùng marker tiếng Việt:

  - `[BẰNG CHỨNG CHÍNH]`
  - `[VỊ TRÍ PHÁP LÝ]`
  - `[BẰNG CHỨNG BỔ SUNG]`
  - `[PHẠM VI LIÊN QUAN]` hoặc `[ĐỐI TƯỢNG LIÊN QUAN]`
  - `[VĂN BẢN]`

- Render answer-first; không dùng multi-view và không lấy `max` giữa các representation.
- BGE pair budget đúng 512 tokens:

  - Audit query trước; query quá 128 tokens dùng head 96 + tail 32 và phải được báo cáo.
  - Tính document allowance sau query và special tokens bằng tokenizer thật.
  - Primary evidence nhận ít nhất 55% document allowance.
  - Identity + structural path tối đa 20%.
  - Applicability tối đa 15% và chỉ xuất hiện khi router chọn.
  - Secondary dùng phần còn lại.
  - Cắt tại sentence/clause boundary; assert marker và một phần evidence chính luôn còn trong input.
  - Không dựa vào tokenizer right-truncation sau khi render; pair hoàn chỉnh phải `≤512`.

- Identity dùng official accented title khi `VERIFIED`, kèm normalized unaccented label làm fallback/audit identity; không tự sinh dấu tiếng Việt.

### 4. Evidence screen và gate trước training

- Mọi selector, weight, scope threshold và capsule policy được chọn trong inner folds của từng outer fold.
- Đánh giá full nested OOF trên 6.991 evaluable queries, đồng thời báo sensitivity khi loại 320 query EXP-031.
- Cheap residual score dùng per-query standardized LambdaMART và selector evidence score; grid chỉ gồm:

  - Window `W ∈ {16, 25, 32, 50, 64}`.
  - Evidence weight `α ∈ {0,10; 0,25; 0,50; 0,75}`.

- Chỉ được sang BGE training nếu so với LambdaMART gốc:

  - Aggregate Recall@5 tăng ít nhất 0,002.
  - Precision@5 không giảm.
  - Ít nhất 4/5 outer folds không âm.
  - Worst-fold Recall@5 delta ≥−0,002.
  - Sensitivity loại 320 query không có delta âm.
  - Candidate membership, oracle Recall@64 và canonical labels không đổi.

- Nếu gate fail: scheduler dừng trước GPU training, report error slices và selector alternatives; không “chạy BGE thử cho biết”.

### 5. BGE residual fine-tuning

- Chỉ dùng `BAAI/bge-reranker-v2-m3` sau khi evidence gate đạt; không rerun GTE hoặc six-model screen trong EXP-033.
- Preflight trên GPU 6GB với sequence 512, backward, optimizer state, checkpoint save/reload và peak VRAM:

  - Full FP32 nếu peak ≤5,4GB.
  - Nếu không, FP32 LoRA rank 16, alpha 32, dropout 0,05.
  - Nếu FP32 LoRA vẫn vượt 5,4GB, FP16 LoRA cùng cấu hình.
  - QLoRA không được dùng nếu chưa có phê duyệt mới.

- Train ba epoch, effective batch 32, gradient checkpointing khi cần, không heldout early stopping.
- Mỗi group chứa tất cả positives còn trong K64 và tám hard negatives/positive:

  - Hai false positives trong LambdaMART top 5.
  - Hai candidates rank 6–16.
  - Hai lexical/structural legal near-misses.
  - Một source-diverse negative.
  - Một seeded tail negative rank 33–64.

- Dùng multi-positive listwise loss; không gắn positive label cho một arbitrary passage không qua evidence selector.
- Final ranking là protected residual:

  - Chuẩn hóa LambdaMART và BGE score trong từng query.
  - Grid inner-only: `α ∈ {0,10; 0,25; 0,50; 0,75}`, promotion margin `τ ∈ {0; 0,25; 0,50; 1,00}` standard deviations, window `W ∈ {16,25,32,50,64}`.
  - Candidate ngoài top 5 chỉ thay vị trí hiện tại khi vượt promotion margin; nếu không giữ LambdaMART order.
  - Luôn trả đúng năm parent document IDs, không adaptive output threshold.

- Promotion gate cuối:

  - Đủ cả năm outer folds.
  - Aggregate Recall@5 tăng ≥0,002 so với selected zero-shot/LambdaMART anchor.
  - Precision@5 không giảm.
  - Không fold nào giảm Recall@5 quá 0,005.
  - Recall@5 ≥0,97 mới được gọi là đạt mục tiêu cuối; nếu chỉ qua promotion gate nhưng dưới 0,97 thì ghi `PROMISING_NOT_FINAL`, không submission.

## Scheduler, artifacts và báo cáo

- `overnight` chạy tuần tự, fingerprint-bound và resumable, nhưng có hai hard stops:

  - `WAITING_SCOPE_SPOTCHECK` trước khi finalize parser.
  - `REJECTED_EVIDENCE_GATE` trước BGE training nếu selector không đạt.

- Mỗi job có log, manifest, `_SUCCESS.json`/`_FAILED.json`, retry chỉ cho lỗi tải/I/O/OOM có phân loại; integrity, membership, fold hoặc hash failure không retry.
- `RUN_STATUS.json` và `state.jsonl` ghi phase, phần trăm theo weighted job units, ETA, retry, peak VRAM, paths và failure reason.
- Report cuối phải gồm parser confusion/boundary audit, evidence-selection ablation, token/truncation audit, LambdaMART anchor, nested zero-shot/trained metrics, rank rescue/harm, query/document/error slices, K-window ablation, latency/VRAM, coverage và kết luận promotion rõ ràng.

## Tests và acceptance

- Unit tests:

  - Canonical labels đúng 6.991 + 9.
  - Fold isolation và không dùng heldout để chọn selector/router/fusion.
  - Candidate membership/order và parent IDs bất biến.
  - Chunk ID ↔ embedding row ↔ document ID chính xác.
  - BM25 search không thoát khỏi requested parent.
  - Scope offsets/source spans, annotation schema và stratified sampler deterministic.
  - Selector/MMR/hybrid deterministic, seeded tie-breaking.
  - Một capsule/document, marker tiếng Việt, answer-first và actual pair ≤512.
  - Answer evidence không thể bị scope làm biến mất.
  - Protected residual fallback và luôn đúng top 5.
  - Resume, hard-stop và failure isolation của scheduler.

- Integration smoke:

  - Một query/fold qua toàn bộ parent-index → selector → capsule → scoring.
  - Scope repair fixture cho detected, combined, rejected và missing cases.
  - Một BGE optimizer step cho strategy được preflight chọn, save/reload adapter hoặc full checkpoint.
  - Resume sau simulated interruption và xác minh fingerprint mismatch buộc rebuild.

- Trước full run, chạy toàn bộ tests EXP-030/031 còn liên quan cùng tests EXP-033; không sửa artifact cũ để ép pass.

## Giả định đã khóa

- Retrieval, corpus, pool 150, canonical labels, nested LambdaMART và K=64 vẫn authoritative.
- EXP-030/031 là rejected research evidence; EXP-032 là diagnostic audit, không phải accepted method.
- Agent tạo 200 scope annotations; người dùng spot-check 30 trước khi pipeline tiếp tục.
- Không full BGE training nếu evidence selector gate thất bại.
- Không public submission trong EXP-033.
