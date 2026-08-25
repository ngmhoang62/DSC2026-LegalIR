# EXP-021 — Báo cáo audit và ablation nhánh sparse (BM25)

Ngày tổng hợp: 2026-08-23  
Phạm vi: retrieval sparse trên corpus structural-v3 cuối; không chạy dense retrieval, reranker hay public submission.

## 1. Mục tiêu

Xây dựng một nhánh sparse có thể tái lập và kiểm toán được, thay vì tiếp tục sử dụng BM25/index cũ vốn gắn với corpus structural trước đó. Câu hỏi được kiểm tra theo thứ tự là:

1. BM25 chỉ trên nội dung structural chunk có làm baseline sạch đủ mạnh không?
2. Metadata nào thực sự giúp: hierarchy, scope hay document title?
3. Nhiều chunks của một document nên được aggregate thế nào?
4. Cấu hình chọn từ ablation có ổn định dưới nested OOF hay chỉ tốt do chọn trên toàn bộ train?

Không có kết quả nào trong báo cáo này là điểm public-test hoặc submission score.

## 2. Dữ liệu và tính toàn vẹn artifact

| Mục | Giá trị đã kiểm tra |
|---|---:|
| Preprocessing fingerprint | `3b7ba47dcfd382f6d670824490777ff98f7f44144a888add664fc1aac6fe5263` |
| Structural-v3 fingerprint | `9743fe70ec4092aca28a9855d0236bdedef41d8a458e39d750062b9ef85a37bc` |
| Documents | 8.507 |
| Structural chunks | 343.347 |
| Train queries | 7.000 |
| Structured / fallback documents | 7.288 / 1.219 |
| Documents có scope nodes | 5.556 |
| Queries có legal identifier theo regex hẹp | 46 |

Input audit kiểm tra `_SUCCESS.json`, số lượng document/chunk, và liên kết giữa structural manifest với preprocessing manifest. Artifact: [sparse_input_audit.json](input_audit/sparse_input_audit.json).

Corpus có phân bố rất lệch về số chunks/document: median 26, P90 86, P99 242 và max 3.981. Vì vậy document-level aggregation là một phần quan trọng của sparse retrieval, không phải chi tiết triển khai nhỏ.

## 3. Thiết kế retrieval được kiểm tra

### 3.1 Đơn vị index

Mỗi record FTS5 tương ứng một structural chunk. Nội dung chính (`passage_content`) là `raw_text` của chunk, sau Vietnamese word segmentation bằng Underthesea. SQLite FTS5 dùng inverted index và `bm25(...)` để lấy top passages; nó không phải model học sâu.

BM25 query dùng các token được quote và nối OR. Điều này cho phép partial lexical match và vô hiệu hoá FTS operators nếu chúng xuất hiện trong câu hỏi.

### 3.2 Cách tính document ranking

Hai cách aggregate chunks thành document được so sánh:

- `RRF3`: lấy tối đa ba passages thuộc các `parent_node_id` khác nhau, rồi cộng `1 / (60 + global_passage_rank)`.
- `first-passage`: mỗi document chỉ nhận evidence passage tốt nhất.

`first-passage` có trực giác phù hợp cho head ranking: một chunk chứa đúng điều luật nên mạnh hơn document có nhiều chunks khớp các từ rải rác. `RRF3` có thể tốt hơn ở tail vì thưởng cho document có nhiều evidence lexical độc lập.

### 3.3 Logging và khả năng resume

Tokenization tạo shard 8.192 passages/shard, có checksum marker và resume. Build index và retrieval đều có `status.json`; log runtime hiện in đồng thời ra terminal và file với tiến độ shard/commit/query. Query segmentation được chạy tuần tự trước khi dùng SQLite worker threads vì Underthesea có global CRF state không thread-safe.

## 4. Baseline passage-only

Configuration:

- Field: chỉ `passage_content`; không đọc/index title, hierarchy hay scope.
- Index: SQLite FTS5 (`unicode61`, giữ `_` trong compound token).
- Aggregation: `RRF3`.
- Passages/documents retrieved: 2.000 / 100.

Artifacts:

- Fields: `cache/exp021_sparse/passage_only/bm25_fields/`
- FTS5: `cache/exp021_sparse/passage_only/fts5/bm25_v3.sqlite`
- Train rankings: `cache/exp021_sparse/passage_only/rankings_train/`

| Metric | Passage-only |
|---|---:|
| Recall@1 | 0,47914 |
| Recall@5 | 0,78819 |
| Recall@10 | 0,86824 |
| Recall@20 | 0,93267 |
| Recall@50 | 0,96503 |
| Recall@100 | 0,97598 |
| Precision@5 | 0,16726 |

Đây là baseline mới trên fingerprint corpus cuối, không được trộn với BM25 score/artifact EXP-012b cũ.

## 5. Error audit của baseline

Các tín hiệu đáng chú ý:

| Slice | Recall@5 | Diễn giải |
|---|---:|---|
| Gold structured | 0,80449 | Mức baseline tốt. |
| Gold fallback | 0,10945 | Điểm yếu lớn; fallback thiếu hierarchy/scope tự nhiên. |
| Gold có scope | 0,81829 | Tốt hơn nhóm không scope. |
| Gold không scope | 0,46566 | Khó hơn rõ rệt. |
| Query có legal identifier | 0,60507 | Mẫu nhỏ (46 query), nhưng yếu hơn query không identifier 0,78940. |

Lưu ý: slice theo gold document có thể có hơn 7.000 entries vì một query có nhiều gold documents. Có 13 gold document occurrences không tồn tại trong corpus cuối; các occurrence này không thể được retrieval tìm lại.

Artifact: [passage-only error audit](passage_only/error_audit/error_audit.json).

## 6. Ablation fields

Tất cả ablation dùng corpus/fingerprint giống nhau. Delta dưới đây so với passage-only.

| Cấu hình | R@1 | R@5 | R@10 | R@50 | Quyết định |
|---|---:|---:|---:|---:|---|
| Passage-only + RRF3 | 0,47914 | 0,78819 | 0,86824 | 0,96503 | Baseline |
| Passage + hierarchy (weight hierarchy:passage = 2:1), RRF3 | 0,48154 | 0,79348 | 0,86915 | 0,96637 | Giữ |
| Passage + hierarchy + scope, RRF3 | 0,42364 | 0,73232 | 0,83540 | 0,94378 | Loại |
| Passage + title-folded (title weight 0,25), RRF3 | 0,46323 | 0,77858 | 0,86263 | 0,96480 | Loại |

### 6.1 Hierarchy: giữ

Hierarchy tạo lift nhỏ nhưng nhất quán: +0,00529 Recall@5 và +0,00135 Recall@50. Error audit cho thấy gain chủ yếu đến từ gold documents structured/có scope; fallback gần như không đổi, phù hợp vì fallback không có hierarchy có nghĩa.

### 6.2 Scope: loại

Scope field được tạo từ heading scope article cộng tối đa 1.000 ký tự mở đầu. Dù full scope text không bị loại khỏi passage, field scope vẫn bị lặp trên các chunks thuộc document. Kết quả giảm mạnh 0,06115 Recall@5. Suy luận hợp lý là repeated scope term frequency làm các document phù hợp “phạm vi” nhưng không chứa evidence chính thắng ranking. Đây là inference từ cơ chế và ablation, không phải một causal proof tách riêng từng term.

### 6.3 Raw title: không thử trực tiếp; title-folded: đã thử và loại

Title lexicon audit xác nhận quan sát ban đầu về title không dấu:

| Kiểm tra | Kết quả |
|---|---:|
| Title có dấu tiếng Việt | 20 / 8.507 |
| Query có dấu tiếng Việt | 7.000 / 7.000 |
| Query có overlap token với title gold, dạng raw | 10,37% |
| Query có overlap token với title gold, sau fold dấu | 57,71% |

Vì raw title chắc chắn yếu, chỉ title-folded mới được index. Query được tách field: token có dấu chỉ match `passage_content`; bản fold dấu chỉ match `document_label`, do đó không làm folded query term nhiễu passage. Dù coverage tăng, title vẫn giảm Recall@5 0,00961 so với baseline. Title pháp lý thường quá tổng quát và lặp qua toàn bộ chunks của một document; overlap không đồng nghĩa với discriminatory signal.

Artifact: [title lexicon audit](title_lexicon_audit/title_lexicon_audit.json).

## 7. Ablation aggregation: câu trả lời cho giá trị của chunking

So sánh trên index `passage + hierarchy`:

| Aggregation | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| RRF3 | 0,48154 | 0,79348 | 0,86915 | 0,93300 | 0,96637 | 0,97705 |
| First-passage | 0,50472 | 0,81306 | 0,88652 | 0,92810 | 0,96159 | 0,97418 |
| Equal fusion of two rankings | 0,50315 | 0,82618 | 0,90165 | 0,93723 | 0,96436 | 0,97619 |
| Fusion head-20, RRF3 tail | 0,50315 | 0,82618 | 0,90165 | 0,93723 | 0,96637 | 0,97705 |

Kết luận trực tiếp:

- Chunking giúp BM25 vì chunk tốt nhất định vị evidence chính xác: first-passage cao hơn RRF3 ở Recall@1/5/10.
- Nhưng cộng evidence từ nhiều chunks làm hại head ranking: document chứa keyword ở nhiều vị trí có thể vượt document có một evidence passage rất đúng.
- RRF3 vẫn có ích ở tail, vì phục hồi Recall@50/100.

Do đó câu trả lời không phải “chunking có ích hay không” một cách tuyệt đối. Chunking là hữu ích; aggregation phải phân biệt head precision và tail coverage.

Artifact: [aggregation fusion audit](aggregation_fusion.json).

## 8. Final depth, RRF và candidate-budget tuning

### 8.1 Một lần retrieve top-4096, nhiều ablation offline

FTS5 được chạy một lần với top-4096 passages/query. Thay vì lưu toàn bộ raw chunks, cache giữ với mỗi document tối đa ba passage ranks thuộc parent khác nhau; đó là đủ để tái tạo first-passage và RRF3 với mọi depth prefix. Raw evidence gồm 7.000 query, 55 shards và fingerprint `1088eea666c1d481925c3ec16b899a99c17bb477291b614275268e6598a66645`.

Grid nested OOF:

| Hyperparameter | Giá trị |
|---|---|
| Passage depth | 1.024, 2.048, 4.096 |
| Parent RRF k | 32, 64, 128 |
| Fusion RRF k | 32, 64, 128 |
| Fusion-head cutoff | 16, 32, 64 |
| Candidate document budget | 30, 50, 80, 100, 120, 150, 180 |

Có hai RRF khác nhau: parent-RRF gộp tối đa ba parent passages thành document; fusion-RRF gộp hai document rankings (`first-passage` và RRF3). Chúng được tune riêng.

### 8.2 Cấu hình head-ranking được OOF chọn

Mỗi held-out fold chọn config trên bốn folds còn lại bằng Recall@5, tie-break Recall@10. Modal config là:

```text
depth = 1024
parent-RRF k = 32
fusion-RRF k = 32
fusion head cutoff = 16
```

Bốn trên năm folds chọn đúng config trên; fold 1 chỉ khác fusion-RRF k=128. Do đó depth 1.024 là lựa chọn ổn định cho head ranking trong grid đã xét, không phải giả định rằng 4.096 luôn xấu.

| OOF metric, head-selected config | Giá trị |
|---|---:|
| Recall@1 | 0,50761 |
| Recall@5 | 0,83514 |
| Recall@10 | 0,90016 |
| Recall@16 | 0,92629 |
| Recall@30 | 0,95334 |
| Recall@50 | 0,96575 |
| Recall@80 | 0,97413 |
| Recall@100 | 0,97674 |
| Recall@120 | 0,97845 |
| Recall@150 | 0,98110 |
| Recall@180 | 0,98339 |

So với best aggregation trước depth/RRF tune (Recall@5=0,82618), head tune tăng thêm 0,00896 tuyệt đối.

### 8.3 Candidate-budget curves

Vì Recall@K tăng đơn điệu theo K, không có một “K tối ưu” chỉ từ retrieval metric. Vì vậy mỗi budget được nested-select riêng; đây là candidate coverage có thể đạt trong grid, không phải một config public-deployment cố định.

| Sparse document budget | Nested-OOF Recall@K |
|---:|---:|
| 30 | 0,95308 |
| 50 | 0,96590 |
| 80 | 0,97361 |
| 100 | 0,97654 |
| 120 | 0,97860 |
| 150 | 0,98053 |
| 180 | 0,98298 |

Điểm gãy thực dụng là 80→100 (+0,00293) và 100→120 (+0,00206); còn 120→180 tăng tổng +0,00438 với thêm 60 documents. Budget cuối phải được chọn theo candidate cap của union với dense/reranker, không chỉ theo sparse curve.

### 8.4 Runtime depth benchmark

Benchmark trực tiếp trên 128 query với ba workers cho thấy top-4096 không chậm hơn đáng kể top-2048 ở workload này:

| Depth | Wall time/128 query | Ngoại suy 7.000 query | Unique docs trung bình/passage pool |
|---:|---:|---:|---:|
| 1.024 | 69,73 s | 63,6 phút | 342 |
| 2.048 | 73,50 s | 67,0 phút | 646 |
| 4.096 | 72,71 s | 66,3 phút | 1.193 |

Tuning head chọn 1.024, nhưng cache 4.096 vẫn hữu ích để derive/tune tail budgets offline. Runtime benchmark là sample 128 query, không phải SLA production.

Artifacts: [raw-4096 evidence manifest](../../cache/exp021_sparse/depth_tune/raw4096_evidence/manifest.json), [final tuning report](depth_rrf_tuning/tuning_report.json), [OOF rankings](depth_rrf_tuning/oof_rankings.jsonl), [depth benchmark](depth_benchmark.json).

## 9. Cấu hình sparse đề xuất và phạm vi hiệu lực

### Head ranking / evidence routing

```text
Corpus: structural_v3_e5_final_v1
Segmenter: Underthesea
Index unit: structural raw-text chunk
Indexed fields: passage_content (weight 1.0) + hierarchy_text (weight 2.0)
Không index: scope_text, document title
Passage retrieval depth: 1024
Parent aggregation: RRF3, k=32
Document fusion: equal-RRF(first-passage, parent-RRF3), k=32
Final order: fusion top-16, sau đó RRF3 tail không trùng document
```

Đây là config head được OOF chọn. Nó phù hợp để rank candidate đầu và chọn evidence; không tự động quyết định sparse candidate budget trong union với dense.

### Candidate coverage

Các budget được báo là curve: 30/50/80/100/120/150/180. Nếu cần một mốc sparse đơn lẻ trước khi biết union budget cuối, 100 là trade-off hợp lý về coverage/cost; đó là recommendation vận hành, không phải optimum thống kê duy nhất. Khi dense candidate cap đã cố định, chọn K tương ứng từ bảng OOF thay vì tự suy diễn.

## 10. Những gì đã bị loại hoặc không chạy

| Hướng | Trạng thái | Evidence |
|---|---|---|
| Scope field | Loại | Recall@5 giảm 0,79348 → 0,73232. |
| Title-folded field | Loại | Recall@5 giảm 0,78819 → 0,77858. |
| Raw title | Không index | 8.487/8.507 title không dấu; raw query-title overlap chỉ 10,37%. |
| Fallback lexical backoff | Không chạy trong conversation này | Đã được giao cho conversation khác theo yêu cầu. |
| Identifier-specific normalization/backoff | Chưa benchmark | Không được coi là đã bị loại. |
| Neural sparse | Không chạy | Ngoài scope theo yêu cầu. |
| BM25L/BM25+ | Không chạy | Chưa có evidence length-normalization là nguyên nhân chính. |

## 11. Uncertainties và giới hạn

1. 46 legal-identifier queries là mẫu nhỏ. Chúng yếu hơn baseline chung, nhưng chưa đủ để kết luận identifier normalization có hay không có ích.
2. Fallback gold có Recall@5 rất thấp (0,10945) nhưng Recall@100 vẫn 0,90547. Điều này chứng minh lỗi head ranking nặng, không chứng minh duy nhất rằng chunk length hay tokenizer là nguyên nhân.
3. Candidate-budget nested OOF chọn config khác nhau giữa folds, nhất là tại K lớn. Không được deploy trực tiếp các config theo fold cho public query; config modal head là lựa chọn fixed hiện có, còn K phải chốt cùng budget dense.
4. OOF chỉ bao phủ grid đã xét, không chứng minh optimality trên mọi BM25 parameter, tokenizer hay sparse technique.
5. Chưa đo end-to-end union sparse+dense, candidate recall sau dedup/cap, reranker score hay public-test performance. Vì vậy những con số ở đây không được dùng thay leaderboard/submission metric.
6. Tuning cache chỉ lưu ranks, không lưu BM25 score hoặc full raw passages. Nó đủ cho first-passage/RRF rank aggregation grid, nhưng không đủ để đánh giá một scoring rule mới cần raw term score.

## 12. File và log chính

- [Input audit](input_audit/sparse_input_audit.json)
- [Passage-only train metrics](../../cache/exp021_sparse/passage_only/rankings_train/metrics.json)
- [Hierarchy train metrics](../../cache/exp021_sparse/passage_hierarchy/rankings_train/metrics.json)
- [Scope train metrics](../../cache/exp021_sparse/passage_hierarchy_scope/rankings_train/metrics.json)
- [Title-folded train metrics](../../cache/exp021_sparse/passage_title_folded/rankings_train/metrics.json)
- [First-passage train metrics](../../cache/exp021_sparse/passage_hierarchy/rankings_train_first_passage/metrics.json)
- [Aggregation fusion](aggregation_fusion.json)
- [Nested OOF](aggregation_oof.json)
- [Final depth/RRF/budget tuning](depth_rrf_tuning/tuning_report.json)
- [Overnight runtime log](depth_tune_overnight.log)

Mọi FTS5 index, tokenized fields, ranking JSONL và manifest tương ứng nằm trong `cache/exp021_sparse/`. Các log runtime nằm trong `results/exp021_sparse/`.
