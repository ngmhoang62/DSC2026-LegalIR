# EXP-030 — Legal Evidence Routing và fine-tuning theo khả năng GPU

## Mục tiêu và ràng buộc

- Giữ nguyên frozen E5@100 + BM25 novel@50, cascade nested theo outer fold và shortlist đúng K=64.
- Reranker luôn trả đúng 5 văn bản; báo cáo canonical-gold Recall@5 và Precision@5. Exact-duplicate gold được ánh xạ sang `duplicate_retained_id`; gold có passage nguồn thực sự rỗng bị loại. Gate dữ liệu bắt buộc xác nhận 6.991 query evaluable và 9 query non-evaluable trên train hiện tại.
- Không dùng heldout outer để chọn capsule, model, precision hoặc chế độ fine-tuning.
- Ưu tiên FP32. Model nhỏ chỉ full fine-tune khi parameter audit và optimizer smoke thực trên GPU 6 GB cùng pass. Thứ tự fallback là full FP32 → full FP16 → LoRA FP32 → LoRA FP16 → QLoRA (chỉ khi bitsandbytes khả dụng).

## Giả thuyết có thể bác bỏ

### H1 — Tên có dấu bổ sung tín hiệu cho label không dấu

`document_label` của 8.507/8.507 văn bản đều không dấu. Không tự thêm dấu. Chỉ nhận `TÊN CHÍNH THỨC` khi đó là exact span có dấu trong header nguồn, viết hoa, đúng loại văn bản và overlap token với label ít nhất 0,50.

- Baseline: `unaccented_base`.
- Ablation: `title_base`, `both_base`.
- Bác bỏ nếu bounded gate không đạt mean ΔRecall@5 ≥ 0,003, có outer giảm quá 0,005, hoặc không dương ở ít nhất 4/5 outer trên cả BGE và GTE.

### H2 — Scope typed và ancestry thật giúp phân biệt legal near-miss

Không tin trực tiếp `scope_node_ids` legacy. Chỉ nhận `Phạm vi điều chỉnh`, `Đối tượng áp dụng` và heading kết hợp; lưu exact span/offset. Structural path được dựng từ `chunk.parent_node_id` và ancestry Điều → Khoản → Điểm, không dùng scope node làm parent.

- Ablation: `both_base` → `typed_scope` → `multi_view`.
- Query regex chỉ là hint. Base evidence luôn tồn tại; khi không chắc, router giữ cả hai auxiliary view có lexical score gần nhau.
- Bác bỏ theo bounded gate như H1, sau đó phải qua full confirmation paired bootstrap trên toàn bộ bốn inner folds của từng outer, ở cả BGE và GTE; mỗi outer yêu cầu CI95 thấp > 0.

### H3 — Fine-tuning phải cải thiện gold survival, không chỉ mean

- Mỗi positive trong K64 được ghép với 8 negative duy nhất/epoch: top-rank, E5-only, BM25-only, cùng legal family, scope/actor confuser và seeded tail.
- Ba epoch cố định, effective accumulation 32, không heldout early stopping.
- Fine-tuned model chỉ promotable khi đủ 5 outer, aggregate ΔRecall@5 ≥ 0,002, ΔPrecision@5 ≥ 0 và không outer nào giảm Recall@5 quá 0,005.
- Mục tiêu nghiên cứu Recall@5 ≥ 0,970 được báo riêng; không đạt thì xuất error buckets 6–10, 11–32 và 33–64 rồi thiết kế can thiệp kế tiếp, không tạo metric giả.

## Capsule tiếng Việt

Các marker duy nhất: `[VĂN BẢN]`, `[TÊN CHÍNH THỨC]`, `[TÊN CHUẨN HÓA]`, `[PHẠM VI ĐIỀU CHỈNH]`, `[ĐỐI TƯỢNG ÁP DỤNG]`, `[VỊ TRÍ TRONG VĂN BẢN]`, `[BẰNG CHỨNG TRẢ LỜI]`, `[QUAN HỆ PHÁP LÝ]`.

Mỗi candidate có base view; applicability và legal-relation là view riêng để scope dài không đẩy answer evidence khỏi budget. Mọi pair được kiểm tra bằng tokenizer thật ở 512 token. Jina vẫn nhận một native call gồm đủ 64 documents; runtime packing context 4.096 kích hoạt block aggregation có sẵn trong remote code để workload 64×512 vừa GPU 6 GB.

## Số audit và preflight đã đo

- 7.000 query đều có dấu; 8.507 label đều không dấu; gold-count: 6.447 query có 1 gold, 485 có 2, 53 có 3, 14 có 4, 1 có 5.
- 6.864 title exact-source VERIFIED; 1.643 MISSING, không sinh title giả.
- 4.590 văn bản có typed scope; 6.672 legacy scope node bị từ chối vì không đủ điều kiện heading.
- Smoke capsule thật: 1 query, 64 candidates, 126/126 evidence resolve parent và structural path; candidate membership/order giữ nguyên.
- Parameter thực: Vietnamese 569,10M; BGE 569,10M; GTE 306,07M; mMARCO 117,64M; Qwen 596,35M; Jina 596,84M.
- Score/backward/save/reload LoRA FP32 pass cho Vietnamese, BGE, GTE, mMARCO và Qwen. Jina native K=2/8/16/32/64 pass, zero-shot-only vì không có backward contract công bố.
- mMARCO full FP32 optimizer smoke pass với 24 comparisons, 3 optimizer updates và full-checkpoint reload.
- `bitsandbytes` hiện không khả dụng; QLoRA không được giả vờ là đã preflight.

## Scheduler và dừng sớm

`overnight` đóng băng code/input/config fingerprint, ghi `state.jsonl`, `RUN_STATUS.json`, log riêng và marker theo job. Score artifacts có qid byte-offset index để worker không parse lặp file capsule ~1,5 GB. Model/fold failures được cô lập; integrity/fingerprint/membership failures chặn dependencies. Nếu capsule không qua bounded hoặc full gate, scheduler dừng trước full six-model selection và fine-tuning.

Không chạy benchmark dài chỉ vì code compile. Điều kiện khởi chạy là unit/regression pass, metadata/capsule smoke pass, đủ 6 model preflight và scheduler dry-run/failure-isolation pass.
