# EXP-109B — Encoder Complementarity, Full-Corpus Retrieval & Nested Fusion

## 1. Tóm tắt và giả thuyết

EXP-109B kiểm tra giả thuyết:

> Plateau Recall@5 hiện tại chủ yếu đến từ representation của VietLegal-E5. VietLegal-Harrier hoặc VnLegal-LAL tuy có standalone Recall@5 thấp hơn E5 nhưng có thể tìm đúng những gold mà E5 và BM25 bỏ sót. Nếu complementarity đủ lớn, fusion ở Stage 1 có thể nâng Recall@5 lên vùng `0.94–0.95` trước reranking.

Đây không phải continuation của SoftTop-5 và không fine-tune encoder. EXP-109A đã cho thấy:

- SoftTop-5 tốt nhất chỉ tăng `+0.4116pp` Recall@5 trên pilot.
- Multi-gold tăng `+2.50pp`, nhưng chỉ có 110/1,397 query thuộc nhóm này.
- Training loss giảm mạnh nhưng ranking hầu như không tăng.
- Bottleneck hợp lý tiếp theo là representation/complementarity, không phải tiếp tục đổi loss trên cùng E5 embedding.
- FP16 embeddings bắt buộc được L2-normalize lại sau khi chuyển sang FP32.

EXP-109B gồm hai lớp:

```text
Bounded complementarity screen trên errors/controls
                 ↓ PASS
Full-corpus encoding cho model được chọn
                 ↓
Exact parent retrieval: E5 / alternate encoder / BM25
                 ↓
Corrected weighted-RRF baseline và multi-encoder RRF
                 ↓
Fold-isolated shallow LambdaMART fusion
                 ↓
Fold-0 ambitious gate
                 ↓ PASS + user approval
Full five-fold nested OOF
```

Mục tiêu thực tế:

- Gate để chạy full OOF: Fold-0 Recall@5 `≥0.940` và tăng ít nhất `+1.5pp` so với corrected E5+BM25 baseline.
- Strong outcome: aggregate OOF Recall@5 `≥0.950`.
- Target tags: `PASS_TARGET_096`, `PASS_TARGET_097`.
- Không hứa trước `0.97`; source oracle sẽ cho biết target đó có khả thi với ba encoder hiện tại hay không.

---

## 2. Context bắt buộc cho conversation mới

Conversation triển khai không được dựa vào lịch sử chat. Trước khi sửa code phải đọc đầy đủ và tạo `READING_AUDIT.json` cho:

1. `D:\Study\DSC2026\LegalIR\AGENTS.md`.
2. EXP-109A:
   - `docs/exp109a_softtop5_plan.md`
   - `src/exp109a_softtop5_retrieval.py`
   - `tests/test_exp109a_softtop5_retrieval.py`
   - `results/exp109a_softtop5_retrieval/pilot_screen/outer_fold_0/inner_fold_1/PILOT_REPORT.json`
   - reproduction, NumPy fixture, preflight và smoke reports.
3. EXP-015:
   - `src/exp015_model_screen.py`
   - `src/exp015_stage2_content_ablation.py`
   - `results/exp015_model_screen/summary.md`
   - `results/exp015_model_screen/stage2/summary.md`
   - reports và manifests của E5, Harrier, LAL.
4. EXP-035:
   - source, `REPORT.json`, `evidence_pack.jsonl`.
5. EXP-036:
   - source, `REPORT.json`, `cache/.../universe.jsonl`.
6. EXP-037:
   - toàn bộ `src/exp037_cached_encoder_complementarity.py`
   - fixture, manifest và report hiện có.
7. Dense/sparse Stage 1:
   - EXP-021 dense pipeline và sparse depth/RRF tuning.
   - EXP-034 shallow retrieval.
   - EXP-027 LambdaMART.
   - EXP-104 pre-ranker/fusion.
8. Canonical train data, folds, duplicate mapping, excluded queries và structural corpus manifests.

`READING_AUDIT.json` phải ghi path, SHA-256, nội dung/ràng buộc rút ra và mọi mismatch giữa plan với checkout hiện tại. Mismatch phải dừng để báo, không tự đoán.

### Baselines và số liệu lịch sử cần ghi nhận

- Corpus: 8,507 parent documents, 343,347 source-exact chunks.
- Queries: 7,000; evaluable: 6,991; non-evaluable: 9.
- Label fingerprint: `9bdf9593b61fe3423d1f1a819ac9fb3e8d7225e6003da0afb840c1f5853fd4c9`.
- EXP-015 fixed development fixture, không phải OOF:

| Model | Recall@5 | Recall@20 | Recall@100 | MRR@5 |
|---|---:|---:|---:|---:|
| VietLegal-E5 | 0.9609 | 0.9844 | 0.9922 | 0.8339 |
| VnLegal-LAL | 0.9453 | 0.9844 | 0.9922 | 0.8202 |
| VietLegal-Harrier-0.6B | 0.9297 | 0.9766 | 0.9922 | 0.8401 |
| BGE-M3 dense | 0.8984 | 0.9609 | 0.9766 | 0.7684 |

- Structural `retrieval_text` đã cải thiện Recall@5 so với raw text:
  - E5: `+2.73pp`
  - Harrier: `+5.47pp`
  - LAL: `+7.62pp`
- EXP-037 hiện chỉ có fixture 290 queries:
  - 239 retrieval errors.
  - 51 controls.
  - Candidate set có force-include gold.
  - Vì vậy chỉ đo separability/complementarity, tuyệt đối không gọi là Recall claim.
- EXP-034 official RRF aggregate:
  - Recall@5 `0.9020`
  - Recall@16 `0.9630`
  - Recall@32 `0.9776`
  - Recall@50 `0.9833`
- EXP-109A pilot baseline inner fold:
  - Recall@5 `0.8982`
  - Recall@16 `0.9607`
  - Recall@50 `0.9816`
- EXP-027 `0.920761` là reranked Recall@5 trong historical frozen pool, không phải full-corpus candidate coverage.

---

## 3. Phạm vi và namespace

Tạo namespace mới, không sửa EXP-015/021/027/035/036/037/109A:

```text
src/exp109b_encoder_complementarity.py
tests/test_exp109b_encoder_complementarity.py
cache/exp109b_encoder_complementarity/
results/exp109b_encoder_complementarity/
docs/exp109b_runbook.md
```

Có thể refactor logic bằng cách import helper ổn định, nhưng EXP-109B phải sở hữu manifests, reports, checkpoints và success markers riêng.

Models duy nhất trong EXP-109B:

```text
Baseline: mainguyen9/vietlegal-e5
Candidate 1: mainguyen9/vietlegal-harrier-0.6b
Candidate 2: darklethelong/vnlegal-lal
Sparse source: existing tuned BM25
```

Không dùng:

- BGE-M3 dense.
- BGE-M3 learned-sparse hoặc multi-vector.
- LLM augmentation.
- Query-memory.
- `document_label`.
- Identifier boost.
- EXP-022 ordered append-union.
- Model download hoặc corpus regeneration.
- Public test/submission.

Nếu Harrier và LAL đều fail bounded gate, dừng EXP-109B. BGE-M3 learned-sparse/multi-vector sẽ là EXP-109C riêng.

---

## 4. Model và scoring contract

### VietLegal-E5

```text
backend: sentence_transformers
max_length: 512
query_prefix: "query: "
document_prefix: "passage: "
pooling: model-native SentenceTransformer pooling
dimension: 1024
```

### VietLegal-Harrier-0.6B

```text
backend: sentence_transformers
max_length: 512
query_prefix:
"Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: "
document_prefix: ""
pooling: model-native SentenceTransformer pooling
dimension: 1024
```

### VnLegal-LAL

```text
backend: transformers AutoModel
max_length: 2048
query_prefix:
"Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: "
document_prefix: ""
pooling: last non-padding token
dimension: 1024
```

Phải kiểm tra đúng padding side, attention mask và vị trí last token trên cả left-padding và right-padding fixtures.

### Document representation

Dùng duy nhất structural `retrieval_text`, source-exact, không raw text và không tự tạo capsule.

Parent score chính cho cả ba dense encoders:

\[
S(q,d)=\frac{s_{(1)}(q,d)+s_{(2)}(q,d)}{2}
\]

trong đó `s_(1), s_(2)` là hai cosine chunk scores cao nhất trong toàn bộ parent. Parent chỉ có một chunk thì dùng score đó.

- Primary aggregation: exact `top2_mean`.
- Secondary diagnostics: `max` và normalized LogSumExp.
- Secondary aggregations không được dùng để chọn winner trong EXP-109B.

Embeddings lưu FP16 được chuyển sang FP32 và L2-normalize lại trước cosine scoring. Query và document phải cùng scorer contract.

---

## 5. CLI và orchestration

```powershell
python -u src/exp109b_encoder_complementarity.py audit
python -u src/exp109b_encoder_complementarity.py replay
python -u src/exp109b_encoder_complementarity.py bounded-screen --outer fold_0 --resume
python -u src/exp109b_encoder_complementarity.py preflight --resume
python -u src/exp109b_encoder_complementarity.py encode-selected --outer fold_0 --resume
python -u src/exp109b_encoder_complementarity.py source-audit --outer fold_0 --resume
python -u src/exp109b_encoder_complementarity.py fold0-screen --resume
python -u src/exp109b_encoder_complementarity.py nested-oof --resume
python -u src/exp109b_encoder_complementarity.py overnight-fold0 --resume
python -u src/exp109b_encoder_complementarity.py status
```

`overnight-fold0` chỉ chạy đến Fold-0 report. Không tự động chạy full five-fold.

`nested-oof` phải từ chối chạy nếu:

- Fold-0 ambitious gate chưa pass.
- Người dùng chưa cấp quyền riêng cho full OOF.
- Fingerprint của source/model/scorer đã thay đổi.

---

## 6. Phase 0 — Input audit

Kiểm tra:

- Canonical labels, duplicate aliases và chín non-evaluable queries.
- Fold membership đúng một fold/query.
- 8,507 unique parent IDs.
- 343,347 structural chunks, offsets, parent links và `retrieval_text`.
- E5 full-corpus cache của EXP-109A.
- BM25 raw passage ranking/cache và tuned parent aggregation của EXP-021.
- EXP-035 cohort và EXP-037 bounded fixture.
- Model snapshots tồn tại local; load với `allow_download=False`/`local_files_only=True`.
- Model prefixes, tokenizer budgets, dimensions và pooling đúng model contract.
- Disk space dự kiến cho embeddings, rankings, shards và temporary merge.
- Không có process EXP-109A/109B đang giữ cùng output paths.

Gate:

```text
PASS → Phase 1
FAIL → REJECTED_INPUT_AUDIT
```

Không được sửa manifests cũ để làm gate pass.

---

## 7. Phase 1 — Reproduction gate

### 7.1 EXP-015 replay

Dùng archived embeddings để replay chính xác các metrics EXP-015 của E5, Harrier và LAL:

```text
Recall@5
Recall@20
Recall@100
MRR@5
per-query first-gold ranks
```

Sai số metric tối đa `1e-6`.

Fresh-encode một deterministic mini-fixture rồi so với archived embeddings:

- cosine similarity tương ứng `≥0.9999`;
- dimension, prefix, truncation và pooling giống report cũ;
- ranking ties được xử lý bằng stable parent ID ordering.

Không yêu cầu bitwise equality giữa GPU/dtype paths.

### 7.2 Shared scorer replay

- EXP-109A real-parent NumPy fixture phải pass `max_abs_error ≤1e-6`.
- FP16→FP32 post-normalization phải được test riêng.
- `top2_mean` blockwise scorer phải khớp independent NumPy implementation.
- Parent một chunk, hai chunks và nhiều chunks đều phải có fixture.

Gate:

```text
PASS → Phase 2
FAIL → REJECTED_REPRODUCTION_GATE
```

---

## 8. Phase 2 — Fold-isolated bounded complementarity screen

Không chạy EXP-037 nguyên trạng. Tái sử dụng fixture sau khi kiểm tra fingerprint, nhưng bổ sung logic mới trong EXP-109B.

Với outer Fold 0, chỉ dùng cohort thuộc Folds 1–4 để chọn source. Không dùng một query hoặc label Fold 0 nào.

Candidate document set của mỗi bounded query:

```text
EXP-036 top-96 universe ∪ canonical gold
```

Mỗi parent giữ tối đa tám structural chunks như fixture EXP-037.

Cần ghi rõ trong mọi report:

```json
{
  "bounded_oracle_fixture_not_recall": true,
  "gold_force_included": true,
  "may_not_be_reported_as_full_corpus_recall": true
}
```

### Metrics

So từng candidate model với E5:

- Paired normalized first-gold-rank delta.
- 10,000-sample query bootstrap CI95.
- E5 Top-5 misses được model cứu vào Top-5.
- E5 Top-32 misses được model cứu vào Top-32.
- Gold E5 Top-5 bị model đẩy khỏi Top-5.
- Matched-control Top-5 exit rate.
- Per-fold mean rank delta.
- Single/multi-gold slices.
- EXP-035 error-tag slices.
- Rescue overlap giữa Harrier và LAL.

### Individual model gate

Một model pass khi đồng thời:

1. Bootstrap CI95 lower bound của normalized rank improvement `>0`.
2. Mọi included fold có mean rank delta `≥0`.
3. Có ít nhất 10 E5 Top-5 misses được cứu vào Top-5.
4. Net Top-5 rescues, `rescued - lost`, `≥5`.
5. Số source-generation misses được đưa vào Top-32:
   `≥max(8, ceil(0.06 × source_generation_miss_count))`.
6. Matched-control Top-5 exit rate `≤0.02`.

### Khi cả hai model pass

Với model `m`, định nghĩa:

```text
R_m = tập query E5 miss Top-5 nhưng model m hit Top-5
```

Encode cả Harrier và LAL full corpus chỉ khi mỗi model thỏa ít nhất một điều kiện:

```text
|R_m \ R_other| ≥ 5
hoặc
|R_m \ R_other| / |R_m| ≥ 0.25
```

Nếu không, chỉ chọn một model theo thứ tự:

1. CI lower bound cao hơn.
2. Net Top-5 rescues cao hơn.
3. Source-generation Top-32 additions cao hơn.
4. Control exit thấp hơn.
5. Ít thời gian encoding dự kiến hơn.

Nếu không model nào pass:

```text
REJECTED_BOUNDED_COMPLEMENTARITY
```

Dừng hoàn toàn EXP-109B; không tự chuyển sang BGE-M3.

---

## 9. Phase 3 — GPU/resource preflight

Cho từng model được chọn:

- Forward trên query và documents với batch 8.
- Nếu OOM, giảm tuần tự `8 → 4 → 2 → 1`.
- Không tự đổi dtype/model/token budget.
- VRAM headroom sau peak allocation `≥10%`.
- Đo texts/s riêng cho query và document.
- Ước lượng ETA từ toàn bộ 343,347 chunks.
- Kiểm tra dung lượng cho FP16 shards, merged embeddings và rankings.

Nếu batch 1 vẫn OOM hoặc disk không đủ:

```text
REJECTED_RESOURCE_GATE
```

Preflight chỉ chứng minh feasibility, không phải metric.

---

## 10. Phase 4 — Resumable full-corpus encoding

Encode:

- Toàn bộ 343,347 `retrieval_text` chunks.
- Toàn bộ 7,000 queries.
- Chỉ model được chọn bởi outer-train bounded gate.

Artifacts:

```text
cache/exp109b_encoder_complementarity/embeddings/<model>/chunks/shard-xxxxx.npz
cache/exp109b_encoder_complementarity/embeddings/<model>/queries.npz
cache/exp109b_encoder_complementarity/embeddings/<model>/manifest.json
```

Mỗi shard lưu:

- chunk IDs và parent IDs.
- FP16 embeddings.
- original vector norms.
- token/truncation counts.
- input and model fingerprints.
- shard checksum.

Resume chỉ reuse shard khi toàn bộ fingerprints khớp. Partial/corrupt shard phải rebuild qua owning stage.

Sau merge:

- Không thiếu/thừa/duplicate chunk.
- Query count đúng 7,000.
- NaN/Inf bằng zero.
- Sampled vectors sau normalization có norm trong `[0.99999, 1.00001]`.
- Sampled fresh-encoding parity pass.

Nếu sau Fold-0 gate cần full OOF và một outer-train screen yêu cầu model chưa encode, chỉ lúc đó mới chạy bổ sung model còn thiếu.

---

## 11. Phase 5 — Exact full-corpus source audit

Tạo exact parent rankings cho:

```text
E5
Harrier và/hoặc LAL
BM25
```

Dense source phải chấm tất cả parent bằng `top2_mean`, không lấy K150 historical union làm không gian ứng viên.

BM25 phải lấy từ tuned EXP-021 sparse retrieval và parent aggregation. Không append “E5@100 + novel BM25@50”.

Report curve:

```text
K = {1, 3, 5, 10, 16, 20, 32, 50, 64, 100, 150}
```

Report thêm:

- Standalone metrics mỗi source.
- Gold rank distributions.
- Pairwise Top-K overlap.
- Unique gold contribution mỗi source.
- Single/multi-gold.
- EXP-035 error tags.
- Source membership của mọi gold recovered/lost.
- Per-query best-source oracle, được đánh dấu label-dependent diagnostic.
- Candidate-union coverage theo source depth:
  `{20, 50, 100, 200, 500}`.

### Fusion candidate depth

Với mỗi outer fold, dùng outer-train labels để chọn depth nhỏ nhất trong:

```text
D ∈ {100, 200, 500}
```

thỏa:

```text
union Recall@D ≥ 0.995
và gain khi tăng lên depth kế tiếp < 0.001
```

Nếu không depth nào đạt `0.995`, dùng `D=500` và ghi `CANDIDATE_DEPTH_CEILING_WARNING`.

Không dùng một depth được chọn từ Fold 0 cho các fold khác.

### Full-corpus source viability gate

Trước khi chấm Fold 0, trên Folds 1–4 yêu cầu:

1. Best-source oracle Recall@5 `≥0.950`.
2. Oracle tăng ít nhất `+0.020` so với corrected E5+BM25 baseline.
3. Có ít nhất 20 distinct gold query-document occurrences được alternate encoder đưa vào Top-20 trong khi cả E5 và BM25 xếp dưới Top-50.
4. Candidate union Recall@50 không thấp hơn corrected baseline.

Nếu fail:

```text
REJECTED_FULL_CORPUS_SOURCE_GATE
```

Report phải thêm:

```text
TARGET_097_SOURCE_ORACLE_SUPPORTED
```

chỉ khi best-source oracle Recall@5 `≥0.970`; nếu thấp hơn thì ghi:

```text
TARGET_097_NOT_SUPPORTED_BY_CURRENT_SOURCES
```

Oracle không được dùng trực tiếp để chọn ranking cho held-out queries.

---

## 12. Phase 6 — Corrected weighted-RRF fusion

### Corrected baseline

Dựng lại baseline từ:

```text
exact full-corpus E5 parent ranks
+
tuned BM25 parent ranks
```

Không dùng EXP-022 ordered union và không thay actual score bằng reciprocal của final fused rank.

### Candidate fusion structures

```text
E5 + BM25
E5 + Harrier + BM25
E5 + LAL + BM25
E5 + Harrier + LAL + BM25  # chỉ nếu cả hai được chọn
```

Weighted RRF:

\[
S(d)=\sum_m \frac{w_m}{k+r_m(d)}
\]

Grid:

```text
RRF k ∈ {10, 20, 32, 60, 100}
weights: simplex step 0.10
E5 weight ≥0.30
mỗi included alternate/BM25 weight ≥0.10
sum(weights)=1
```

Missing source rank đóng góp zero.

Với outer Fold 0:

- Tune structure, `k`, weights và candidate depth trên Folds 1–4.
- Khóa config.
- Chấm Fold 0 đúng một lần.

Selection order trên outer-train:

1. Recall@5.
2. Precision@5.
3. Multi-gold Recall@5.
4. MRR@5.
5. Recall@16.
6. Ít source hơn.
7. Gần baseline weights hơn.

---

## 13. Phase 7 — Shallow LambdaMART fusion

LambdaMART chỉ cạnh tranh sau khi source viability gate pass.

Candidate set là union top-D đã chọn trong outer-train. Không sử dụng document ID/label hoặc label-derived inference features.

Allowed features:

- Per-source raw score.
- Per-source rank và reciprocal rank.
- Per-source score z-normalized trong query.
- Margin tới rank 1, rank 5 và rank 10.
- Source-presence mask.
- Số source đồng thuận trong Top-5/10/20.
- Dense parent top1 score, top2 score và top1-top2 gap.
- BM25 parent score/rank.
- Parent chunk count và retrieval-text token length.
- Query token length.

Cấm:

- `document_label`.
- Exact doc ID memorization.
- Nearest labeled-query features.
- Fold/global target statistics.
- Features được build bằng labels của held-out fold.

### Nested training

Với mỗi outer:

1. Outer-heldout bị khóa.
2. Dùng bốn outer-train folds làm inner CV để chọn hyperparameters.
3. Retrain winner trên toàn outer-train.
4. Chấm outer-heldout.

Small fixed grid:

```text
num_leaves ∈ {7, 15}
min_data_in_leaf ∈ {20, 50}
learning_rate ∈ {0.03, 0.05}
num_boost_round ∈ {100, 300}
feature_fraction = 1.0
bagging_fraction = 1.0
deterministic = true
seed = 109
objective = lambdarank
eval_at = [5]
```

Selection order giống weighted RRF. Nếu LambdaMART không thắng RRF trên outer-train, dùng RRF.

---

## 14. Phase 8 — Fold-0 ambitious gate

Report ba hệ thống:

1. Corrected E5+BM25 weighted-RRF baseline.
2. Best multi-encoder weighted RRF.
3. Inner-selected final winner giữa RRF và LambdaMART.

Báo cáo:

- Recall@1/3/5/10/16/20/32/50/64/100/150.
- Precision@5.
- MRR@5.
- Single/multi-gold Recall@5.
- Per-source unique recoveries.
- Gold promotions/losses qua ranh giới Top-5.
- Error-tag breakdown.
- Candidate/oracle ceiling.
- Config được khóa từ Folds 1–4.
- Model/source/depth/fusion selection provenance.

### Ambitious gate

Chỉ được đề nghị chạy full five-fold khi final Fold-0 winner đồng thời:

```text
Recall@5 ≥ 0.940
Recall@5 delta vs corrected baseline ≥ +0.015
Precision@5 không giảm
Multi-gold Recall@5 không giảm
Recall@16 delta ≥ -0.001
Recall@50 delta ≥ -0.001
Net Top-5 gold promotions > 0
```

Nếu tăng `≥0.005` nhưng không đạt ambitious gate:

```text
WEAK_SIGNAL_NO_FULL_OOF
```

Nếu thấp hơn:

```text
REJECTED_FOLD0_GATE
```

Không full OOF chỉ để xác nhận một kết quả quanh `0.92–0.93`.

---

## 15. Phase 9 — Full five-fold nested OOF

Chỉ chạy sau:

- Fold-0 ambitious gate pass.
- Người dùng cấp quyền rõ ràng cho full OOF.

Với từng outer fold:

- Bounded source decision chỉ dùng cohort thuộc outer-train.
- Nếu cần thêm model chưa encode, encode và verify model đó trước.
- Candidate depth chọn trong outer-train.
- RRF weights/structure chọn trong outer-train.
- LambdaMART hyperparameters chọn bằng inner CV trong outer-train.
- Outer-heldout chỉ chấm sau khi mọi quyết định khóa.

Promotion gate aggregate:

```text
Aggregate Recall@5 ≥ 0.940
Aggregate delta vs corrected baseline ≥ +0.015
Bootstrap CI95 lower bound của Recall@5 delta > 0
Ít nhất 4/5 folds có delta ≥ 0
Worst-fold delta ≥ -0.002
Aggregate Precision@5 không giảm
Aggregate multi-gold Recall@5 không giảm
Recall@16 và Recall@50 delta ≥ -0.001
```

Tags:

```text
Recall@5 ≥ .950 → PASS_STRONG_095
Recall@5 ≥ .960 → PASS_TARGET_096
Recall@5 ≥ .970 → PASS_TARGET_097
```

Không tạo public submission tự động.

---

## 16. Logging, resume và artifacts

Detailed logs:

```text
results/exp109b_encoder_complementarity/logs/<run-id>/run.log
results/exp109b_encoder_complementarity/logs/<run-id>/<phase>.log
results/exp109b_encoder_complementarity/logs/<run-id>/<outer>/<phase>.log
```

Terminal chỉ hiện:

- Phase/model/fold hiện tại.
- Progress và throughput.
- Heartbeat mỗi 5 phút.
- GPU/VRAM.
- ETA dựa trên measured throughput.
- Gate result.
- Đường dẫn log/report.

`RUN_STATUS.json`:

```json
{
  "run_id": "...",
  "state": "RUNNING|PASS|REJECTED|FAILED|INTERRUPTED",
  "phase": "...",
  "outer": "...",
  "model": "...",
  "completed": 0,
  "total": 0,
  "throughput": 0.0,
  "eta_seconds": 0,
  "last_heartbeat": "...",
  "input_fingerprint": "...",
  "config_fingerprint": "...",
  "code_fingerprint": "..."
}
```

- Flush logger sau mỗi heartbeat.
- Checkpoint encoding theo shard.
- Checkpoint scoring theo query blocks.
- Ctrl+C/crash phải ghi `INTERRUPTED`, không ghi `_SUCCESS`.
- Gate reject exit code 2.
- `_SUCCESS.json` chỉ được tạo sau khi report đã validate schema và fingerprints.

---

## 17. Tests bắt buộc

### Input và leakage

- Canonical label policy và nine-query exclusion.
- Exactly-one-fold membership.
- Outer fold không xuất hiện trong source selection, depth tuning, RRF tuning hoặc LambdaMART training.
- No label-dependent feature trong held-out inference.

### Encoder/scorer

- Prefix và tokenizer contract từng model.
- LAL last-token pooling với left/right padding.
- FP16→FP32 re-normalization.
- Cosine and `top2_mean` NumPy parity.
- One/two/many-chunk parents.
- Deterministic tie breaking.
- OOM batch fallback.
- Shard resume và corrupt-shard rejection.

### Bounded gate

- Force-included gold fixture luôn mang warning flags.
- Rescue, loss, net rescue và exclusive rescue calculations.
- Bootstrap determinism.
- Conditional one-model/two-model policy.
- Both-fail hard stop.

### Full retrieval/fusion

- Exact full-parent scoring.
- BM25 source không bị biến thành ordered union.
- Candidate depth selection chỉ dùng outer-train.
- Weighted RRF score/reference parity.
- Missing-source handling.
- LambdaMART candidate/features/fold isolation.
- Inner-selected fallback từ LambdaMART về RRF.

### Orchestration

- Missing prerequisite marker hard fail.
- Fingerprint mismatch invalidates stale success.
- Fold-0 rejection blocks `nested-oof`.
- Full OOF requires explicit authorization.
- Logs flush, heartbeat, status and resume behavior.

Regression command:

```powershell
$env:PYTHONPATH='src'
D:\Study\DSC2026\dsc_env\Scripts\python.exe -m pytest `
  tests/test_exp109b_encoder_complementarity.py `
  tests/test_exp109a_softtop5_retrieval.py `
  tests/test_exp035_038_retrieval_closure.py `
  tests/test_exp034_shallow_retrieval.py `
  tests/test_exp027_lambdamart_shortlist.py `
  -q
```

---

## 18. Trình tự triển khai và quyền chạy

Conversation mới phải thực hiện:

1. Đọc toàn bộ reading order và tạo `READING_AUDIT`.
2. Implement namespace EXP-109B.
3. Chạy unit/regression tests.
4. Chạy input audit và scorer/model reproduction.
5. Chạy CPU/small-GPU smoke.
6. Đo ETA thực tế.
7. Dừng và báo cáo trước khi chạy bounded GPU screen hoặc full-corpus encoding, trừ khi người dùng đã cấp quyền chạy rõ ràng trong conversation đó.

Sau khi được cấp quyền:

```text
bounded screen
  → pass thì preflight
  → pass thì encode selected model(s)
  → pass thì source audit
  → pass thì Fold-0 fusion
  → dừng tại Fold-0 report
```

Full five-fold và public submission luôn cần lệnh riêng của người dùng.

---

## 19. Những hướng cố ý để lại cho experiment sau

Nếu EXP-109B fail vì Harrier/LAL không complementary:

- Không thử tiếp dense model ngẫu nhiên.
- Thiết kế EXP-109C cho BGE-M3 learned sparse + multi-vector MaxSim.
- Old BGE-M3 dense result không được dùng để kết luận learned-sparse/multi-vector cũng fail.

Nếu EXP-109B nâng Stage 1 rõ rệt:

- Có thể tái thử training-cardinality-conditioned objective:
  - single-gold dùng decoupled loss;
  - multi-gold dùng SoftTop-5 `alpha=2`.
- Đây là follow-up riêng, không trộn vào EXP-109B vì EXP-109A đã chứng minh loss-only improvement quá nhỏ trên representation cũ.

## 20. Handoff prompt

> Hãy triển khai EXP-109B đúng theo plan này trong `D:\Study\DSC2026\LegalIR`. Không dựa vào lịch sử chat: đọc toàn bộ reading order, tạo `READING_AUDIT.json`, đối chiếu manifests và reports thật trước khi viết code. EXP-109B kiểm tra complementarity của VietLegal-Harrier và VnLegal-LAL so với exact full-corpus VietLegal-E5 + tuned BM25. EXP-037 chỉ là gold-forced bounded separability fixture, không phải recall benchmark; phải bổ sung fold isolation, pairwise rescue overlap, source oracle và full-corpus verification trong namespace mới. Parent dense scorer chính là source-exact all-chunk `top2_mean`; mọi FP16 embedding phải được L2-normalize lại sau FP32 conversion. Corrected baseline phải là inner-tuned weighted RRF từ exact E5 và tuned BM25 rankings, tuyệt đối không dùng EXP-022 ordered union. Nếu cả alternate encoders fail bounded gate thì dừng, không tự chạy BGE-M3. Nếu cả hai pass thì encode cả hai chỉ khi có đủ exclusive rescues theo gate. Implement, test, audit, replay và smoke trước; không download model, không chạy full encoding, full OOF hoặc public submission nếu chưa được người dùng cấp quyền riêng.
