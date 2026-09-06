# EXP-109C — Candidate-Expanded Latent-Condition Late Interaction

## 1. Quyết định nghiên cứu

EXP-109C là bước kế tiếp được đề xuất sau EXP-109B. Nó không thêm một dense
encoder ngẫu nhiên và không quay lại heavy cross-encoder của EXP-107/108.

Giả thuyết chính:

> Candidate coverage hiện đã đủ cao, nhưng LambdaMART của EXP-109B chỉ nhìn
> thấy rank, scalar score và agreement giữa E5/LAL/BM25. Nó không biết token hay
> mảnh luật nào làm một parent phù hợp với từng vế của query. Một late-interaction
> model có thể so khớp từng query token với tất cả token evidence trong parent,
> tạo ra representation giàu hơn mà vẫn index corpus offline. Nếu kết hợp
> representation này với candidate pool mở rộng và một metric adapter nhỏ được
> train fold-isolated, Recall@5 có cơ hội thoát khỏi plateau 0.93.

Kiến trúc tổng quát:

```text
E5 top-D ∪ LAL top-D ∪ BM25 top-D
              ↓  D được khóa trên F1–F4, ưu tiên ceiling
Expanded parent candidate pool
              ↓
Jina-ColBERT-v2 128-d token index trên structural-v3 hiện tại
              ↓
Exact chunk MaxSim + latent two-chunk condition coverage
              ↓
Rich token-interaction features
              ↓
109B scalar features + fold-isolated LambdaMART
              ↓
Optional frozen-embedding low-rank metric adapter
              ↓
Strict F1–F4 gate
              ↓ PASS + explicit authorization
One locked Fold-0 evaluation
```

EXP-109C đặt trần cuối lên trước tốc độ, nhưng mọi bước đắt đều có pilot và gate.
Không LLM, không tạo/annotate dữ liệu mới, không query-memory, không dùng
`document_label`, không đọc public test và không tự tạo submission.

---

## 2. Bằng chứng đã xác minh và cách diễn giải

### 2.1 EXP-109B Fold 0

Nguồn sự thật:

```text
results/exp109b_encoder_complementarity/
  locked_fusion_fold0/FOLD0_LOCKED_FUSION_REPORT.json
```

Kết quả:

| Hệ thống | Recall@1 | Recall@5 | Recall@16 | Recall@50 | MRR@5 |
|---|---:|---:|---:|---:|---:|
| Corrected E5+BM25 | 0.59526 | 0.90248 | 0.96364 | 0.98081 | 0.74180 |
| EXP-109B LambdaMART | 0.69808 | **0.93318** | 0.96906 | 0.98629 | 0.81947 |

Các số liệu quan trọng khác:

- Gain Recall@5: `+3.0699pp`.
- Single-gold Recall@5: `0.95035`.
- Multi-gold Recall@5: `0.73012`, chỉ tăng `+3.5933pp` từ `0.69419`.
- Precision@5: `0.19914`.
- Full union của candidate set hiện tại, nhìn qua cuối ranking, đạt khoảng
  `0.98867` trên Fold 0.
- Status đúng: `WEAK_FOLD0_NO_FULL_OOF_REPRESENTATION_NEXT`.
- Đây là một locked Fold-0 result, không phải full OOF hoặc public score.

Kết luận:

1. Alternate representation và learned fusion là hướng đúng; EXP-109B không
   phải một negative result.
2. Scalar fusion đã lấy được gain lớn nhưng vẫn bỏ lại khoảng `5.55pp` giữa
   Recall@5 và candidate-union ceiling.
3. Với pool hiện tại, để đạt `0.96`, ranker phải thu hồi khoảng 48.3% residual;
   `0.97` cần khoảng 66.4%; `0.98` cần khoảng 84.4%. Vì vậy không được hứa trước
   rằng thêm một feature sẽ tự động đạt 0.97.
4. Multi-gold vẫn là bottleneck lớn nhất. EXP-109C phải có representation cho
   multi-condition/disjoint evidence, không chỉ cải thiện score trung bình.

### 2.2 EXP-109B inner pilot

Strict cross-fit F1–F4, Fold 0 chưa được dùng trong selection:

| Cấu hình | Recall@5 | Delta so với E5+BM25 | Bootstrap CI lower |
|---|---:|---:|---:|
| LAL standalone | 0.90006 | -0.427pp | -1.088pp |
| LAL+BM25 RRF | 0.91290 | +0.857pp | +0.318pp |
| E5+LAL+BM25 RRF | 0.91873 | +1.440pp | +0.971pp |
| LambdaMART top-50/source | **0.92511** | **+2.078pp** | **+1.582pp** |

LambdaMART cải thiện cả bốn folds. Đây là anchor bắt buộc của EXP-109C; không
được so với raw E5 rồi tuyên bố gain.

### 2.3 EXP-109A

SoftTop-5 trên frozen E5 chỉ tạo `+0.4116pp` pilot signal. Loss giảm nhưng
ranking không tăng tương xứng. EXP-109C không tiếp tục search một loss khác trên
cùng single-vector E5 representation.

### 2.4 EXP-013 và EXP-013b

EXP-013 đã implement Jina-ColBERT-v2 selective late interaction:

- Model: `jinaai/jina-colbert-v2`, 559,497,216 parameters.
- Token projection từng được chạy ở 64 dimensions.
- Old corpus index: 435,316 passages, tối đa 48 anchors/passage, int8 row-scale.
- Encoding cũ hoàn tất trong khoảng 9,061 giây trên máy này.
- Exact MaxSim chỉ được phép chạy sau candidate gate.

Nhưng candidate gate cũ fail:

```text
Recall@5  = 0.88793
Recall@64 = 0.97777 < 0.985
```

Do đó exact candidate MaxSim và LambdaMART dùng exact MaxSim của EXP-013 chưa
từng chạy. Không được nói “ColBERT đã thử và fail”. Cái fail là source/candidate
generation cũ, trên corpus fingerprint cũ.

EXP-013b đạt aggregate OOF Recall@5 `0.93050`, nhưng đó là BGE/query-memory/Qwen
cascade trên candidate/corpus cũ; nó không phải exact late interaction của
EXP-109C. Không được reuse query-memory, labels hoặc public-fusion policy từ
EXP-013b.

### 2.5 Tại sao chọn Jina-ColBERT-v2 thay vì BGE-M3 multi-vector

Đây là quyết định theo hardware và artifact hiện có, không phải kết luận rằng
BGE-M3 multi-vector yếu:

- Jina-ColBERT-v2 snapshot và projection head đã có local, code loader đã từng
  chạy thành công, model hỗ trợ multilingual late interaction và 64/128-d
  compact token vectors.
- BGE-M3 dense từng được benchmark không đồng nghĩa BGE-M3 multi-vector đã được
  thử. Tuy nhiên local BGE-M3 cache hiện chỉ có XLM-R backbone; thiếu
  `colbert_linear.pt`, `sparse_linear.pt`, và environment chưa có
  `FlagEmbedding`.
- Official BGE-M3 multi-vector head có dimension 1024. Với corpus hiện tại, một
  token index đủ fidelity sẽ lớn hơn nhiều và không thích hợp với RTX 4050 6GB
  nếu dùng cùng scoring design.
- Jina-ColBERT-v2 128-d là lựa chọn có ceiling/resource ratio tốt hơn cho 109C.

BGE-M3 learned-sparse/multi-vector có thể là EXP-109D riêng. Không cài dependency
hoặc tải head trong EXP-109C trừ khi plan được sửa và người dùng cho phép rõ.

### 2.6 Căn cứ nghiên cứu

- ColBERTv2 mô hình hóa query/document bằng nhiều token vectors và dùng MaxSim,
  gần interaction của cross-encoder hơn single-vector nhưng vẫn index document
  offline: <https://arxiv.org/abs/2112.01488>.
- Jina-ColBERT-v2 là multilingual late-interaction model; paper cho thấy giảm
  128 xuống 64 dimensions chỉ gây trade-off nhỏ, nhưng EXP-109C dùng 128 vì ưu
  tiên ceiling: <https://aclanthology.org/2024.mrl-1.11/>.
- BGE-M3 cũng dùng multi-vector như reranker trên candidate pool thay vì scan
  toàn corpus; paper rerank top-200 trong thí nghiệm: <https://arxiv.org/abs/2402.03216>.

---

## 3. Context bắt buộc cho conversation triển khai

Conversation mới không được dựa vào lịch sử chat. Trước khi sửa code, agent
phải đọc đầy đủ các file sau và tạo `READING_AUDIT.json`.

### 3.1 Chính sách và dữ liệu

1. `AGENTS.md`.
2. `public_test_dataset/train.json`: inspect schema thật, không đoán field.
3. `cache/cv_folds.json`.
4. `cache/final_preprocessed_v2/manifest.json`.
5. Duplicate mapping, exclusions và label-impact artifacts trong
   `cache/final_preprocessed_v2/`.
6. `cache/structural_v3_e5_final_v1/manifest.json` và schema của:
   - `chunks.jsonl`
   - `documents.jsonl`
   - `nodes.jsonl`
   - `doc_to_chunk_ids.json`

Expected current contracts, phải verify lại:

```text
parents:             8,507
chunks:              343,347
queries:             7,000
evaluable:           6,991
non-evaluable:       9
label policy:        canonical_duplicate_alias_drop_empty_passage_v1
label fingerprint:   9bdf9593b61fe3423d1f1a819ac9fb3e8d7225e6003da0afb840c1f5853fd4c9
struct fingerprint:  9743fe70ec4092aca28a9855d0236bdedef41d8a458e39d750062b9ef85a37bc
```

### 3.2 EXP-109B — anchor trực tiếp

Đọc đầy đủ:

```text
docs/EXP-109B_PLAN.md
docs/exp109b_runbook.md
src/exp109b_encoder_complementarity.py
tests/test_exp109b_encoder_complementarity.py
results/exp109b_encoder_complementarity/source_audit/fold_0/SOURCE_AUDIT.json
results/exp109b_encoder_complementarity/cached_fusion_pilot/fold_0/CACHED_FUSION_PILOT.json
results/exp109b_encoder_complementarity/locked_fusion_fold0/FOLD0_LOCKED_FUSION_REPORT.json
```

Audit manifests và `_SUCCESS.json` cho exact cached rankings:

```text
cache/exp109b_encoder_complementarity/rankings/vietlegal_e5/fold_0/
cache/exp109b_encoder_complementarity/rankings/vnlegal_lal/fold_0/
```

BM25 phải lấy từ tuned raw evidence/current contract của EXP-021, không dùng
EXP-022 append-union.

### 3.3 EXP-013/013b — code reuse và historical pitfalls

Đọc đầy đủ:

```text
docs/exp013_runbook.md
src/exp013_core.py
src/exp013_model.py
src/exp013_late_interaction.py
src/exp013_candidates.py
src/exp013_ranker.py
src/exp013_pipeline.py
tests/test_exp013.py
results/exp013_slid/candidate_oracle/candidate_oracle.json
cache/exp013_slid/colbert_leaves/manifest.json
cache/exp013_slid/models/model_report.json
cache/exp013_slid/colbert_leaves/run.log
docs/exp013b_runbook.md
results/exp013b_cascade/candidate_audit/candidate_audit.json
results/exp013b_cascade/oof/oof_report.json
```

Agent phải ghi rõ trong audit:

- Old EXP-013 fingerprint khác current structural fingerprint.
- Old token index chỉ dùng để reproduction/reference, không được dùng để score
  current corpus.
- Old encoder không resume và nối tất cả arrays trong RAM; 109C phải sửa hai
  điểm này trong namespace mới.
- Old exact scorer union tất cả parent tokens và có document-length lottery
  bias; 109C phải có chunk-bounded features và correction diagnostics.
- Không reuse query-memory từ EXP-013/013b.

### 3.4 Các negative results liên quan

Đọc report chính của EXP-035/036/037, EXP-107/108, EXP-109A. Mục đích:

- không gọi bounded force-gold fixture là Recall;
- không dùng evidence selector đã fail gate làm positive truth;
- không lặp SoftTop-5 frozen-E5 search;
- không nạp full article vào 512-token cross-encoder;
- hiểu rõ multi-gold là bottleneck hiện tại.

### 3.5 `READING_AUDIT.json`

Phải chứa cho mỗi input:

```json
{
  "path": "...",
  "exists": true,
  "sha256": "...",
  "schema_version": "...",
  "contract_or_finding": "..."
}
```

Mọi mismatch về count, fingerprint, model snapshot hoặc ranking manifest phải
dừng ở `REJECTED_INPUT_AUDIT`. Không sửa JSON/manifest bằng tay.

---

## 4. Phạm vi, namespace và ownership

Phần “retrieval frozen” trong historical EXP-022/027 handoff của `AGENTS.md`
vẫn áp dụng cho các downstream experiment sở hữu pool đó. EXP-109B và EXP-109C
là research branch mới đã được người dùng yêu cầu rõ để thay đổi Stage 1; chúng
không được ghi đè artifacts của frozen branch.

Tạo experiment riêng:

```text
src/exp109c_latent_condition_late_interaction.py
tests/test_exp109c_latent_condition_late_interaction.py
cache/exp109c_latent_condition_late_interaction/
results/exp109c_latent_condition_late_interaction/
docs/exp109c_runbook.md
```

Không sửa EXP-013, EXP-109A hoặc EXP-109B. Có thể import pure helpers đã test,
nhưng 109C sở hữu schema, manifests, reports, logs, checkpoints và success
markers của nó.

Không được chạy corpus encoding, full scoring, Fold 0, full OOF hoặc download
chỉ vì implementation/tests đã pass. Mỗi action đắt cần authorization riêng.

---

## 5. CLI bắt buộc

```powershell
python -u src/exp109c_latent_condition_late_interaction.py audit
python -u src/exp109c_latent_condition_late_interaction.py reproduce-exp013
python -u src/exp109c_latent_condition_late_interaction.py preflight
python -u src/exp109c_latent_condition_late_interaction.py candidate-ceiling --outer fold_0
python -u src/exp109c_latent_condition_late_interaction.py fidelity-pilot --outer fold_0 --resume
python -u src/exp109c_latent_condition_late_interaction.py encode-corpus --resume
python -u src/exp109c_latent_condition_late_interaction.py encode-queries --resume
python -u src/exp109c_latent_condition_late_interaction.py score-inner --outer fold_0 --resume
python -u src/exp109c_latent_condition_late_interaction.py frozen-inner-screen --outer fold_0
python -u src/exp109c_latent_condition_late_interaction.py train-metric-adapter --outer fold_0 --resume
python -u src/exp109c_latent_condition_late_interaction.py final-inner-gate --outer fold_0
python -u src/exp109c_latent_condition_late_interaction.py locked-fold0 --outer fold_0 --resume
python -u src/exp109c_latent_condition_late_interaction.py nested-oof --resume
python -u src/exp109c_latent_condition_late_interaction.py status
```

`status` chỉ đọc artifact/process state, không load model.

---

## 6. Phase 0 — Input audit và exact reproduction

### 6.1 Audit

Verify toàn bộ Section 3, active worker state, disk, RAM, CUDA, Python package
versions và local model snapshot.

Jina expected local snapshot:

```text
repo_id: jinaai/jina-colbert-v2
snapshot: 4552c4dc1ffd7d7a635b6a41a1077fe9c9cdd974
parameters: 559,497,216
projection tensor: linear.weight, shape [128, 1024]
```

Không download. Nếu snapshot không đủ, dừng và xin phép.

Audit thêm license/model-use eligibility theo luật cuộc thi. Snapshot hiện được
phân phối với license riêng của Jina; việc model đã có local hoặc từng được chạy
không tự động chứng minh submission được phép. Nếu competition terms và model
license không tương thích, dừng trước real scoring/training.

### 6.2 Reproduce EXP-013 primitives

Trên frozen synthetic fixture và 20 real old-corpus pairs:

- Jina query/document marker contract.
- 64-d output khớp old code trong tolerance giải thích được.
- NumPy MaxSim và Torch reference khớp `≤1e-5` FP32.
- int8 dequantized score parity được report riêng.
- Stable tie-break theo `doc_id`.

Không bắt current corpus khớp old ranking.

Gate:

```text
PASS -> Phase 1
FAIL -> REJECTED_REPRODUCTION_GATE
```

---

## 7. Phase 1 — Candidate ceiling expansion

### 7.1 Candidate sources

Chỉ dùng ba source đã có full-corpus exact ranking:

```text
VietLegal-E5
VnLegal-LAL
tuned BM25
```

Không dùng Harrier, query-memory, identifiers hoặc labels làm feature.

### 7.2 Fixed high-ceiling depth và diagnostic curves

Primary candidate contract được khóa trước khi nhìn metric EXP-109C:

```text
D = 200 mỗi source
candidate = unique(E5@200 ∪ LAL@200 ∪ BM25@200)
```

Trên F1–F4, với Fold 0 không tham gia selection, audit thêm hai đường cong
diagnostic:

```text
D ∈ {50, 100, 200} mỗi source
candidate(D) = unique(E5@D ∪ LAL@D ∪ BM25@D)
```

Mỗi D report:

- min/mean/max unique parents/query;
- Recall@pool aggregate và từng inner fold;
- single/multi-gold candidate recall;
- gold absent from all sources;
- marginal rescues từ 50→100 và 100→200;
- projected exact-MaxSim FLOPs/runtime.

Các curves D=50/100 không được dùng để chọn D rồi báo metric trên cùng F1–F4.
Chúng chỉ định lượng chi phí/ceiling. Nếu D=200 không thể chạy vì một hard
resource gate đã đo được, pipeline dừng để xin plan amendment; không tự đổi về
D=50/100. Cách này tránh depth-selection leakage và đúng ưu tiên ceiling của
người dùng.

Gate tối thiểu:

```text
aggregate candidate Recall@pool >= 0.992
each inner fold                    >= 0.990
candidate schema/fingerprint       PASS
```

Tag mạnh:

```text
PASS_CANDIDATE_CEILING_0995 khi aggregate >= 0.995
```

Candidate expansion là bắt buộc vì pool top-50/source hiện tại chỉ cho Fold-0
ceiling khoảng 0.98867; target gần 0.99 không thể được bảo vệ bằng pool đó.

---

## 8. Phase 2 — GPU/resource preflight

Model preflight phải chạy actual forward trên:

- query batch;
- document batch ở selected tokenizer length;
- 128-d projection;
- quantization;
- một exact parent score.

Contract:

```text
model:              jinaai/jina-colbert-v2
dimension:          128 primary
dtype encode:       FP16 CUDA
stored vectors:     int8 symmetric row-scale
stored scales:      FP16
query vectors:      FP16
model parameters:   < 4B
```

Query max length được chọn label-free từ tokenizer distribution:

- audit `{64, 96, 128}`;
- chọn nhỏ nhất cover `>=99.5%` queries;
- mọi truncated query phải được log;
- không dùng metric để chọn query length.

Document max length:

- tokenize toàn structural `retrieval_text` để lấy percentiles;
- chọn cap nhỏ nhất trong `{512, 768, 1024}` cover `>=99.9%` chunks;
- chunks vượt cap phải được report, không silent truncate;
- nếu >0.1% bị truncate, dùng 1024 và report residual.

Resource gate:

- CUDA forward thành công.
- Peak VRAM còn ít nhất 10% headroom.
- Worker private RAM target `<5.5 GiB`.
- System available RAM không xuống dưới `0.75 GiB`; nếu thấp hơn phải checkpoint
  và dừng sạch, không tiếp tục đến OOM.
- Disk estimate gồm index, checkpoints và temporary shards, cần ít nhất 25% free
  headroom sau build.

Không tự fallback về CPU hoặc dimension 64. Nếu 128 không fit, report số đo và
xin duyệt plan amendment.

---

## 9. Phase 3 — Fidelity pilot cho anchor compression

Old EXP-013 giữ tối đa 48 rare tokens/chunk. EXP-109C không mặc định kế thừa
heuristic này.

Primary budget:

```text
maximum anchors/chunk = 96
dimension             = 128
```

Anchor selection label-free:

1. Giữ task/document marker theo model contract.
2. Giữ tokens thuộc structural prefix ở đầu chunk.
3. Giữ digit/citation-bearing tokenizer pieces.
4. Giữ một lượng nhỏ tail tokens để không mù phần cuối.
5. Fill phần còn lại bằng token IDF cao nhất, IDF tính từ current corpus, tie
   theo source position.
6. Không duplicate positions, cuối cùng restore source order.

Pilot dùng ít nhất 256 real query-parent pairs, gồm:

- short/long parents;
- structured/fallback docs;
- single/multi-gold queries;
- positives và high-ranked non-golds;
- không force-gold khi báo ranking metric; force-included rows chỉ dùng parity.

So sánh full-token FP32 reference với compressed int8:

- absolute/relative score error;
- Spearman trên candidates;
- Top-5 agreement;
- positive-vs-hard-negative ordering flips;
- error theo chunk length và parent chunk count.

Gate:

```text
mean absolute MaxSim error <= 0.010
Top-5 agreement            >= 0.97
positive ordering retention >= 0.97
no monotonic error explosion with parent length
```

Nếu fail, tăng anchors lên 128 và chạy lại đúng một lần. Không hạ gate post hoc.
Nếu 128 vẫn fail: `REJECTED_COMPRESSION_FIDELITY_GATE`.

---

## 10. Phase 4 — Resumable current-corpus token index

Không reuse old EXP-013 index vì fingerprint/corpus khác.

### 10.1 Sharded storage

Không `np.concatenate` toàn corpus trong RAM. Mỗi shard tối đa 256 chunks để
bounded-loss checkpoint; receipt vẫn atomic và hash-verified:

```text
cache/exp109c_latent_condition_late_interaction/index/
  manifest.json
  shards/
    shard-00000.vectors.int8.npy
    shard-00000.scales.f16.npy
    shard-00000.passages.jsonl
    shard-00000.json
  _SUCCESS.json
```

Mỗi passage row lưu:

```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "node_id": "...",
  "source_start": 0,
  "source_end": 0,
  "token_start": 0,
  "token_end": 0,
  "original_tokens": 0,
  "retained_tokens": 0,
  "truncated": false
}
```

Resume chỉ ở shard boundary. Partial shard ghi vào temporary path rồi atomic
rename. `_SUCCESS.json` chỉ xuất hiện sau full hash verification.

### 10.2 Process isolation

Model encoder phải được release trước exact scoring. Không cùng lúc giữ:

- Jina backbone;
- full E5/LAL ranking objects;
- toàn token index trong private RAM.

Source rankings được compact trước thành candidate/feature sidecars. Exact
scorer dùng memmap và GPU blocks; không copy toàn index cộng thêm một bản
candidate tensor lớn trên GPU.

Historical timing 435,316 chunks/64-d/48 anchors là khoảng 2.52 giờ. Với
343,347 chunks, 128-d và 96 anchors, plan estimate encoding `2.5–4.5 giờ`;
đây chỉ là estimate, ETA phải được cập nhật từ measured throughput sau 5,000
chunks.

---

## 11. Phase 5 — Exact latent-condition scoring

### 11.1 Token interaction

Với query token `t` và chunk `c`:

\[
m_c(t)=\max_{u\in c} q_t^\top d_u
\]

Chunk score:

\[
s_c=\frac{1}{|q|}\sum_t m_c(t)
\]

Không chọn một evidence capsule trước. Mỗi chunk tạo một vector coverage trên
toàn bộ query tokens.

### 11.2 Latent two-chunk coverage

Chọn `c1` có `s_c` cao nhất. Chọn `c2` để maximize lượng coverage mới:

\[
s_{2}=\frac{1}{|q|}\sum_t \max(m_{c1}(t),m_c(t))
\]

Tie-break deterministic theo:

1. score;
2. ít overlap source offsets hơn;
3. ít retained tokens hơn;
4. `chunk_id`.

Đây là neural analogue của “atomic evidence + multi-condition coverage”:

- query tokens đóng vai conditions ngầm;
- hai chunks có thể rời nhau;
- không cần regex parser;
- không nối text hay đưa LLM-generated evidence vào model;
- vẫn lưu exact chunk IDs/offsets để audit.

### 11.3 Features bắt buộc

Mỗi query-parent có:

```text
li_chunk_top1_mean
li_chunk_top2_mean
li_chunk_top3_mean
li_chunk_top1_top2_gap
li_two_chunk_union_mean
li_two_chunk_incremental_gain
li_full_parent_union_mean              # diagnostic, length-biased
li_idf_weighted_union_mean
li_token_match_min
li_token_match_p10
li_token_match_p25
li_token_match_median
li_token_match_mean
li_token_match_max
li_lower_quartile_mean                 # AND-like condition score
li_selected_chunk_count
li_selected_token_count
li_parent_chunk_count
li_parent_token_count
li_top1_minus_parent_median
li_union_minus_parent_median
li_exact_rank_within_candidate_pool
```

Không tune arbitrary similarity thresholds. Coverage được biểu diễn bằng
quantiles/continuous features và LambdaMART học trong training folds.

### 11.4 Length lottery audit

Report correlation của `li_full_parent_union_mean` và score/rank với:

- parent chunk count;
- parent token count;
- structured vs fallback mode.

Primary features là chunk-bounded/top-two union. Full-parent union chỉ là
diagnostic/feature candidate; nếu permutation importance cho thấy nó tạo
regression ở long parents, remove phải được quyết định trong inner ablation,
không sau Fold 0.

### 11.5 Scoring outputs

Shard theo query, atomic + resumable:

```text
cache/exp109c.../late_scores/outer_fold_0/
  shards/scores-00000.parquet-or-jsonl
  manifest.json
  _SUCCESS.json
```

Không lưu raw query/document text lặp lại. Lưu IDs, scores, selected chunk
provenance và hashes.

---

## 12. Phase 6 — Frozen late-interaction inner screen

Fold 0 tuyệt đối không tham gia selection của EXP-109C. Do Fold 0 đã từng được
đọc ở các experiment trước, report phải gọi nó là “held-out relative to
EXP-109C”, không gọi globally virgin test.

Trên strict cross-fit F1–F4 so sánh:

```text
A. EXP-109B scalar LambdaMART anchor, reproduced
B. Jina exact MaxSim standalone
C. 109B features + raw late score
D. 109B features + full latent-condition feature block
```

Candidate depth được khóa bởi Phase 1. Hyperparameter grid LambdaMART giữ nhỏ:

```text
num_leaves       ∈ {7, 15}
min_data_in_leaf ∈ {50, 100}
learning_rate    ∈ {0.03, 0.05}
rounds           ∈ {200, 400}
```

Tất cả model selection phải nested trong F1–F4. Feature ablation cũng là model
selection và không được nhìn Fold 0.

Report:

- aggregate và từng-fold Recall@1/3/5/10/16/50/pool;
- Precision@5, MRR@5;
- single/multi-gold Recall@5;
- bootstrap paired CI 10,000 samples;
- query wins/losses/ties so với 109B;
- gold moved into/out of Top 5;
- selected one/two-chunk proportions;
- performance theo parent length, query length, parse mode;
- label-dependent oracle giữa anchor và late system, ghi rõ không deployable.

Gate để metric-adapter stage được chạy:

```text
EITHER:
  fused delta Recall@5 >= +0.005
  bootstrap mean > 0
  at least 3/4 folds non-negative
  multi-gold non-decrease
OR:
  anchor-vs-late choice oracle gain >= +0.020
  and no catastrophic fold (< -0.005)
```

Nếu cả hai fail:

```text
REJECTED_FROZEN_LATE_INTERACTION_GATE
```

Không train adapter chỉ vì standalone MaxSim nhìn có vẻ hợp lý.

---

## 13. Phase 7 — Frozen-embedding low-rank metric adapter

Đây là trainable part duy nhất của EXP-109C. Jina backbone và stored token
embeddings vẫn frozen; không LoRA/backprop qua 560M backbone và không re-encode
corpus per fold.

### 13.1 Adapter

Trên 128-d token vector, học residual low-rank transforms:

\[
q'=\operatorname{norm}(q(I+U_qV_q^\top)),\quad
d'=\operatorname{norm}(d(I+U_dV_d^\top))
\]

```text
rank r ∈ {4, 8}
identity initialization
separate query/document transforms
backbone frozen
```

Chỉ hai ranks được phép; chọn nested trong inner train. Không mở rộng grid sau
khi xem validation.

### 13.2 Parent-level weak supervision

Không có gold evidence span. Với mỗi positive parent:

- lấy top-3 chunks theo frozen Jina score làm latent positive bag;
- adapter score tất cả top-3;
- dùng smooth max trong training;
- inference vẫn report hard top1/two-chunk scores.

Không gán một chunk duy nhất thành gold. Không dùng EXP-108 condition parser để
tạo nhãn.

### 13.3 Multi-positive loss

Mỗi positive phải vượt cùng negative set:

\[
L=\frac{1}{|P|}\sum_{p\in P}
-\log\frac{e^{s_p/\tau}}
{e^{s_p/\tau}+\sum_{n\in N}e^{s_n/\tau}}
\]

Tất cả gold parents được giữ; gold không cạnh tranh lẫn nhau. Không reuse
SoftTop-5 objective đã fail ở EXP-109A.

### 13.4 Sáu negatives, rotating curriculum

Giữ giới hạn VRAM sáu negatives/query-step:

1. Hai non-golds cao nhất theo EXP-109B anchor.
2. Hai non-golds rank 6–32, rotate theo epoch.
3. Một non-gold có late token-coverage cao nhưng anchor thấp.
4. Một source-confuser xuất hiện trong top ranks của ít nhất hai sources.

Không gold/duplicate, không ngoài chosen candidate pool. Nếu nhãn contest có
thể incomplete, log high-risk false negatives; không tự đổi label.

### 13.5 Seeds và score ensemble

Predeclare ba seeds:

```text
109, 110, 111
```

Không chọn seed tốt nhất. Final adapter score là mean z-normalized score của ba
seeds, normalization fit trong training fold. Nếu resource pilot cho thấy ba
seeds vượt budget, phải xin amendment trước khi chuyển còn một seed.

### 13.6 Training budget

```text
epochs:               3 maximum
early stopping:       inner validation Recall@5, patience 1
optimizer:            AdamW
learning rate grid:   {0.001, 0.003}
weight decay:         0.0001
temperature grid:     {0.05, 0.10}
checkpoint interval:  every 250 query groups
```

Batch/gradient accumulation được chọn bằng measured preflight. Không đổi loss
hoặc negative policy sau khi xem Fold 0.

---

## 14. Phase 8 — Final strict inner competition

So sánh:

```text
A. EXP-109B reproduced anchor
B. Frozen latent-condition LambdaMART
C. Adapted late score standalone
D. 109B + adapted latent-condition LambdaMART
E. D with a small residual blend to anchor
```

Residual blend grid chỉ:

```text
alpha_late ∈ {0.15, 0.30, 0.45}
```

Tune fold-isolated. Nếu blend không thắng D, chọn D.

Gate trước Fold 0:

```text
aggregate F1–F4 Recall@5             >= 0.945
delta vs 109B inner anchor            >= +0.015
paired bootstrap CI lower             >= +0.008
at least 3/4 folds improve
worst-fold delta                       >= -0.002
multi-gold Recall@5                    >= 0.760
multi-gold delta                       >= +0.030
Precision@5                            non-decrease
MRR@5                                  non-decrease by more than 0.001
Recall@1                               non-decrease by more than 0.002
```

Gate này cố ý tham vọng: Fold 0 đã cho 0.933, nên một inner result chỉ nhích nhẹ
không đủ biện minh cho thêm một held-out evaluation.

Fail:

```text
WEAK_INNER_SIGNAL_NO_FOLD0
```

Giữ artifacts để phân tích, không nới gate post hoc.

---

## 15. Phase 9 — One locked Fold-0 evaluation

Chỉ chạy khi:

- Phase 8 pass;
- config/model/depth/features/adapter ranks/seeds/blend đã khóa;
- user cho phép rõ;
- F0 late scores chưa từng được dùng cho 109C selection.

Training final dùng toàn F1–F4. F0 labels chỉ được load sau predictions đã được
ghi và hash.

Báo cáo:

1. Corrected E5+BM25.
2. EXP-109B locked anchor reproduction.
3. Expanded candidate oracle/ceiling.
4. Frozen Jina standalone.
5. Frozen latent-condition fusion.
6. Adapted late interaction.
7. Final fused winner.
8. Recall@1/3/5/10/16/32/50/pool, Precision@5, MRR@5.
9. Single/multi-gold, wins/losses, source contribution, parent/query length.
10. Gold recovered/lost với exact selected chunk provenance.

Status:

```text
Recall@5 < 0.950      REJECTED_FOLD0_REPRESENTATION
0.950–<0.960          WEAK_FOLD0_NO_FULL_OOF
0.960–<0.970          PASS_FOLD0_STRONG_USER_DECISION
0.970–<0.980          PASS_TARGET_097
>=0.980               PASS_TARGET_098
```

Không tự chạy full OOF ở bất kỳ status nào.

---

## 16. Phase 10 — Full nested OOF, chỉ khi được duyệt

Điều kiện tối thiểu để đề xuất full OOF:

- Fold 0 Recall@5 `>=0.960`;
- delta vs reproduced EXP-109B anchor `>=+0.020`;
- Precision@5 không giảm;
- multi-gold Recall@5 `>=0.78`;
- user cấp quyền riêng.

Mỗi outer fold phải tự:

- select candidate depth trong outer-train;
- train/tune LambdaMART trong inner folds;
- train metric adapters không thấy outer labels;
- fit normalization/blend trong outer-train;
- score outer-heldout một lần.

Promotion gate:

```text
aggregate OOF Recall@5          >= 0.960
at least 4/5 folds improve vs same-fold 109B anchor
worst-fold delta                >= -0.002
aggregate Precision@5          non-decrease
aggregate multi-gold Recall@5  >= 0.78
```

Tags 0.97/0.98 chỉ ghi nếu aggregate OOF thật sự đạt; không suy từ Fold 0.

---

## 17. Logging, resume và resource safety

Logs:

```text
results/exp109c_latent_condition_late_interaction/logs/<run-id>/main.log
results/exp109c_latent_condition_late_interaction/logs/<run-id>/<stage>.log
results/exp109c_latent_condition_late_interaction/logs/<run-id>/<outer>/<stage>.log
```

Terminal chỉ hiện:

- stage/start/end;
- heartbeat mỗi 5 phút;
- progress/throughput/ETA;
- private RAM/system available RAM;
- VRAM peak/current;
- gate result;
- log path.

Detailed records ghi vào `.log`, flush định kỳ. Không giữ toàn log trong stdout.

`RUN_STATUS.json`:

```json
{
  "run_id": "...",
  "state": "RUNNING|PASS|REJECTED|FAILED|INTERRUPTED",
  "stage": "...",
  "outer": "...",
  "completed": 0,
  "total": 0,
  "throughput": 0.0,
  "eta_seconds": 0,
  "private_ram_bytes": 0,
  "available_ram_bytes": 0,
  "vram_bytes": 0,
  "last_heartbeat": "..."
}
```

Ctrl+C/crash phải giữ completed shards/checkpoints. Resume verify hash rồi mới
tiếp tục. Không coi file tồn tại là complete nếu thiếu valid `_SUCCESS.json`.

---

## 18. Tests bắt buộc

### Unit tests

- Jina marker/token/special-token handling.
- 128-d projection shape và L2 normalization.
- NumPy/Torch MaxSim reference.
- int8 row-scale quantization/dequantization.
- IDF anchor selection deterministic và giữ mandatory tokens.
- Query/document truncation counters.
- Chunk top1/top2/two-chunk union math.
- Disjoint synthetic conditions được two-chunk union cover.
- Duplicate/overlap chunk tie-break.
- Parent length lottery diagnostics.
- Candidate union uniqueness/rank provenance.
- D selection policy.
- Low-rank adapter identity initialization cho score parity.
- Multi-positive loss: positives không cạnh tranh nhau.
- Six-negative curriculum không chứa gold/duplicate/out-of-pool.
- Three-seed ensemble deterministic.
- Stable doc-id tie-break.

### Integration tests

- Canonical 6,991/9 label contract.
- Fold 0 labels không được import trong inner selection code path.
- Current structural fingerprint propagation.
- Stale EXP-013 index bị reject.
- Shard atomic write/resume/hash verification.
- Scorer không cần load Jina backbone.
- Process RAM cap/low-memory clean stop.
- EXP-109B anchor reproduction tolerance.
- Lambda feature schema không chứa doc/query IDs hoặc labels.
- Interrupted scoring/training resume.
- Gate pass/reject và authorization barriers.

### Real fixtures

Phải freeze:

- one real query with short parent;
- one long structured parent;
- one fallback parent;
- one multi-gold query;
- one parent cần hai disjoint chunks;
- one tie case;
- one quantization parity case.

Fixture hash nằm trong manifest; không sửa expected output bằng tay sau failure.

---

## 19. Reports bắt buộc

```text
READING_AUDIT.json
REPRODUCTION_REPORT.json
PREFLIGHT.json
CANDIDATE_CEILING_REPORT.json
FIDELITY_PILOT_REPORT.json
INDEX_MANIFEST.json / _SUCCESS.json
INNER_LATE_SCORE_REPORT.json
FROZEN_INNER_SCREEN.json
METRIC_ADAPTER_REPORT.json
FINAL_INNER_GATE.json
FOLD0_REPORT.json                 # chỉ nếu được phép
FULL_OOF_REPORT.json              # chỉ nếu được phép
IMPLEMENTATION_REPORT.md
```

Mỗi report ghi schema version, config, input fingerprints, code SHA, stage
status, warnings và explicit claim boundary.

---

## 20. Stop rules và diễn giải outcome

1. Candidate ceiling fail: vấn đề vẫn ở source coverage; dừng trước Jina index.
2. Compression fidelity fail: không dùng lossy index để kết luận model yếu.
3. Frozen late features fail và choice oracle thấp: Jina representation không
   complement current sources; dừng trước adapter.
4. Adapter overfit inner folds: không lên Fold 0.
5. Fold 0 0.95–0.96: late interaction có giá trị nhưng chưa đủ target; giữ làm
   feature cho nhánh learned sparse/task-specific encoder sau này.
6. Fold 0 >=0.96: mới đáng cân nhắc full OOF.
7. Fold 0 >=0.97 không phải public proof; cần full OOF rồi user quyết định
   submission.

EXP-109C có xác suất cải thiện 109B cao hơn một scalar encoder khác vì nó thêm
loại interaction chưa có. Tuy vậy 0.97 vẫn là stretch target: plan được thiết kế
để có ceiling cao, không phải bảo đảm trước một score.

---

## 21. Thứ tự implementation cho agent mới

1. Đọc Section 3 và emit `READING_AUDIT.json`.
2. Implement schemas, hash/manifests, authorization gates, status/logging.
3. Port/rewrite pure Jina/MaxSim primitives từ EXP-013 trong namespace 109C.
4. Viết unit/reference tests và reproduction.
5. Implement candidate-depth audit; chưa encode model.
6. Implement GPU/resource + fidelity pilot.
7. Implement sharded/resumable index và exact scorer.
8. Implement frozen feature fusion/nested gates.
9. Implement low-rank metric adapter và tests.
10. Run full static/unit/integration suite.
11. Viết `IMPLEMENTATION_REPORT.md` với phần chưa chạy.
12. Dừng và xin phép trước first expensive real run.

Agent không được vừa implement vừa âm thầm chạy corpus encoding.

---

## 22. Handoff prompt copy sang conversation mới

```text
Bạn đang triển khai EXP-109C trong D:\Study\DSC2026\LegalIR.

Đọc toàn bộ docs/EXP-109C_PLAN.md và AGENTS.md trước khi làm gì khác. Sau đó
đọc đầy đủ reading order ở Section 3, inspect schema/manifests/results thật và
emit READING_AUDIT.json. Không dựa vào lịch sử chat, không đoán path/schema.

Implement đúng namespace EXP-109C, theo thứ tự Section 21. Tái sử dụng EXP-013
chỉ như code/reference; không reuse stale old-corpus index, query-memory hay
EXP-013b public fusion. EXP-109B LambdaMART là anchor bắt buộc. Fold 0 không
được dùng trong model/depth/feature selection.

Lặp implement -> test -> inspect artifacts -> repair -> verify cho đến khi
implementation ổn. Ghi progress, tests, mismatches và resource estimates vào
IMPLEMENTATION_REPORT.md. Không download model/dependency, không encode full
corpus, không chạy full scoring, Fold 0, full OOF hay submission nếu chưa có
authorization riêng. Khi implementation và cheap verification hoàn tất, báo
chính xác command đề xuất, ETA, RAM/VRAM/disk estimate và dừng chờ.
```
