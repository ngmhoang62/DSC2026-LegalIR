# EXP-013b — Selective Legal Cascade

## Tóm tắt

Thay thế nhánh SLID/ColBERT đã fail bằng cascade tận dụng hoàn toàn artifacts EXP-012b và structural-v3 hiện có:

1. Candidate pool: `BGE Top-120 ∪ BM25 Top-50 ∪ fold-safe query-memory Top-10`.
2. LambdaMART fold-isolated rút khoảng 139 candidates xuống Top-24 hoặc Top-32.
3. Tạo một legal evidence capsule/document từ tối đa hai chunks v3 bổ sung nhau.
4. `Qwen/Qwen3-Reranker-0.6B` chỉ chấm shortlist và chỉ chạy cho các query khó sau fold-0 screening.
5. Giữ output Top-5 cố định; adaptive top-k nằm ngoài EXP-013b.

Local oracle đã xác nhận candidate pool mới đạt Recall **0.99022**, đủ ceiling cho mục tiêu Recall@5 ≥0.95. Không encode lại 435k chunks và không dùng ColBERT artifacts lỗi.

## Thay đổi triển khai

### 1. Namespace và candidate stage

Tạo entrypoint riêng `src/exp013b_pipeline.py`; artifacts nằm tại `cache/exp013b_cascade/` và `results/exp013b_cascade/`. Không sửa/xóa EXP-012b, EXP-013 hoặc structural-v3.

`build-candidates --split train|public`:

- Đọc frozen `source_rankings.jsonl` của EXP-012b.
- Union theo document ID, không cắt bằng RRF:
  - BGE leaf Top-120;
  - BM25 Top-50;
  - query-memory Top-10.
- Train memory phải loại toàn bộ query cùng CV fold trước khi đọc answer; public memory được dùng toàn bộ train.
- Lưu source rank/raw score, channel agreement, document/chunk statistics và deterministic candidate rank.
- Tie-break: số kênh giảm dần, best source rank tăng dần, BGE rank, BM25 rank, memory rank, `doc_id`.

`audit-candidates` hard-fail nếu:

- query set/fingerprint sai;
- duplicate document;
- train Recall@pool <0.990;
- bất kỳ ground-truth document ID nào mất khỏi v3;
- pool trung bình vượt 145 hoặc pool tối đa vượt 180.

### 2. LambdaMART shortlist và evidence capsule

`train-preranker-oof` huấn luyện năm LambdaMART models bằng feature rẻ:

- BGE/BM25/memory ranks và raw scores;
- reciprocal ranks, channel overlap/agreement;
- query length;
- document length, chunk/article count, fallback/structured mode;
- score gaps và rank margins;
- document type và hierarchy availability.

Không dùng ColBERT score, Qwen score hoặc label của held-out fold.

Shortlist được khóa tự động:

- Dùng Top-24 nếu OOF Recall@24 ≥0.975.
- Nếu không, dùng Top-32 khi Recall@32 ≥0.975.
- Nếu cả hai fail, dừng trước Qwen và cải tiến pre-ranker.

`build-capsules` tạo đúng một input/document:

- `[Văn bản]` document label;
- `[Phạm vi]` scope nếu có;
- `[Cấu trúc]` Chapter/Section/Article;
- passage BGE tốt nhất;
- passage BM25 tốt nhất có parent khác; nếu trùng thì dùng BGE structural complement.
- Query tối đa 128 tokens; toàn bộ Qwen pair tối đa 768 tokens.
- Với memory-only document chưa có `evidence_by_doc`, chạy scoped BGE refinement và scoped BM25 chỉ cho document thiếu.
- Giữ full chunk IDs/provenance trong metadata; không đưa answer label vào capsule cache.

### 3. Qwen screening, selective routing và OOF

Dùng trực tiếp interface chính thức của [`Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B):

- `AutoModelForCausalLM`, FP16, PyTorch SDPA; không yêu cầu FlashAttention/vLLM.
- Left padding, length buckets và adaptive batch.
- Score là log-probability `yes` so với `no`.
- Instruction cố định:

  > Given a Vietnamese legal question, determine whether the legal document contains provisions that directly answer the question. Prioritize exact legal conditions, subjects, exceptions, penalties, procedures, and scope of application.

Stages:

1. `benchmark-qwen --pairs 2048` đo throughput, peak VRAM và batch an toàn.
2. Nếu peak VRAM >5.5 GB, giảm batch; nếu throughput <15 pairs/s ở 768 tokens, thử 512 tokens.
3. Nếu vẫn <15 pairs/s, dừng Qwen và chạy cùng fold-0 screen bằng `Alibaba-NLP/gte-multilingual-reranker-base`; không chạy toàn OOF với model quá chậm.
4. `score-qwen-screen --fold 0 --resume` chấm toàn bộ shortlist fold 0.
5. Promotion gate fold 0:
   - Recall@5 tăng ≥0.008 so với LambdaMART;
   - Precision@5 giảm không quá 0.001;
   - không có NaN/Inf hoặc query thiếu score.
6. Nếu pass, khóa model/instruction/max-length/batch.

Selective routing sau screening:

- Ambiguity score dựa trên Lambda margin Top-5/Top-6, BGE–Lambda disagreement, channel overlap và score entropy.
- Route tối đa 30% query khó; nếu rule giữ dưới 80% fold-0 gain của full Qwen thì tăng lên tối đa 35%.
- Nếu 35% vẫn không giữ được 80% gain, hard-fail vì kiến trúc không đạt đồng thời speed/quality.
- Query không route giữ LambdaMART ranking.
- Query được route dùng weighted RRF của Qwen rank, Lambda rank và BGE rank.
- Fusion weights chọn nested trên bốn folds còn lại từ grid cố định; public dùng modal OOF configuration, không tuning trên public.

`evaluate-oof` promotion gate:

- Recall@5 ≥0.950;
- Precision@5 ≥0.1971;
- candidate Recall ≥0.990;
- worst-fold Recall@5 không thấp hơn EXP-012b tương ứng;
- bootstrap 95% CI của Recall gain so với EXP-012b không có cận dưới âm.

Chỉ sau gate này mới chạy final/public:

- train final LambdaMART trên 7,000 queries;
- build candidates/capsules public;
- áp dụng cùng shortlist K, route fraction, Qwen configuration và modal fusion;
- tạo submission Top-5.

Nếu OOF đạt 0.95 nhưng bị giới hạn bởi candidate misses, lập EXP-014 bổ sung `Qwen3-Embedding-0.6B`. Không encode corpus bằng model này trong EXP-013b.

## Thời gian dự kiến trên RTX 4050 6GB

| Công việc | Thời gian dự kiến |
|---|---:|
| Candidate train + audit | 1–2 phút |
| 5-fold LambdaMART + shortlist audit | 3–8 phút |
| Route evidence và build train capsules | 10–25 phút |
| Download/warm-up Qwen lần đầu | 5–20 phút |
| Benchmark 2,048 pairs | 2–5 phút |
| Full Qwen fold-0 screen | 15–45 phút |
| Selective Qwen folds 1–4 | 20–60 phút |
| Fusion + OOF evaluation | 1–3 phút |
| **Tổng OOF lần đầu** | **khoảng 55–130 phút; dự kiến thực tế 75–105 phút** |
| Các lần tuning chỉ dùng score cache | dưới 10 phút |
| Public 1,000 queries | khoảng 8–20 phút |
| GTE fallback nếu Qwen fail | cộng thêm khoảng 20–45 phút |

Không stage nào được phép chạy nhiều giờ mà không báo trước: benchmark throughput phải hoàn thành trước Qwen fold-0 và dự báo ETA từ số pairs thực tế. Mọi stage dài ghi status mỗi 64 queries, log ETA mỗi 256 queries và hỗ trợ `--resume`.

## Kiểm thử và nghiệm thu

- Candidate union deterministic và đúng Recall 0.99022 trên artifact hiện tại.
- Query-memory không đọc label cùng fold.
- Mọi shortlist document có capsule; fallback evidence chỉ chạy khi nguồn cũ thiếu.
- Capsule ≤768 tokens và có tối đa hai unique parent chunks.
- Qwen prompt/token IDs `yes`/`no` đúng; batch và single-pair scores tương đương trong tolerance.
- Resume không ghi trùng hoặc bỏ query.
- Router không vượt route budget và cho kết quả byte-identical.
- OOF predictions chứa đúng một lần mỗi qid và chỉ dùng fold model tương ứng.
- Public submission chỉ được tạo sau OOF promotion gate.

## Giả định đã khóa

- Mục tiêu chính EXP-013b là Recall@5 ≥0.95; 0.99 là stretch goal, không cam kết từ candidate ceiling 0.99022.
- Qwen zero-shot trước; chưa LoRA/fine-tune transformer.
- Candidate pool dùng BGE-120 + BM25-50 + memory-10.
- Shortlist ưu tiên K=24, tự chuyển K=32 theo gate.
- Route budget tối đa 35%.
- Giữ nguyên toàn bộ artifacts ColBERT lỗi để audit/postmortem; không sử dụng downstream và không xóa.
