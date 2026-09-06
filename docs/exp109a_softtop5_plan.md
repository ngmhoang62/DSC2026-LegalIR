# EXP-109A — Full-Corpus SoftTop-5 Stage-1 Retrieval

> Tài liệu này là handoff tự chứa cho một conversation/agent mới. Agent triển khai **không được dựa vào lịch sử chat**. Phải đọc các file ở Mục 2, kiểm chứng schema và số liệu hiện tại, rồi mới viết code. Nếu checkout mâu thuẫn với plan, ưu tiên artifact/code đã xác minh và ghi rõ sai khác; không tự đoán.

## 1. Mục tiêu và giới hạn của EXP-109A

EXP-109A là một micro-experiment có kiểm soát nhằm trả lời đúng một câu hỏi:

> Khi giữ nguyên VietLegal-E5, corpus/chunk embeddings, document scoring và kiến trúc query projection, một loss tối ưu trực tiếp Top-5 có tăng Recall@5 của **full-corpus Stage 1** so với loss multi-positive hiện tại hay không?

Đây **không** phải EXP-109 hoàn chỉnh và không nhằm tự thân đạt Recall@5 0.97. Nó là gate rẻ trước khi quyết định có đáng fine-tune encoder hoặc xây representation mới ở EXP-109B hay không.

Các ràng buộc:

- Không dùng BGE-M3 trong EXP-109A.
- Không download model/dependency mới.
- Không encode lại corpus hay query.
- Không fine-tune VietLegal-E5 backbone.
- Không dùng document-ID embedding/classification head; scoring phải dựa trên text embeddings để còn tổng quát tới labels chưa thấy trong train folds.
- Không đổi parser, chunking, corpus, BM25 index/ranking method hoặc thêm retrieval channel.
- Không retune document aggregation giữa các loss arms.
- Không LLM, không sinh/annotate dữ liệu mới.
- Không public inference, submission hoặc leaderboard tuning.
- Chưa được chạy full nested OOF/GPU training chỉ vì đã implement xong; phải chờ user cho phép chạy.

Primary metric là macro Recall@5 trên 6,991 evaluable queries. Precision@5 là tie-break/guardrail. Full-corpus dense và inner-tuned E5+BM25 fusion là hai outputs chính. Phải report đường cong `K={1,3,5,10,16,20,32,50,64,100,150}`; K của metric không phải training-pool size.

## 2. Reading order bắt buộc trước khi implement

Agent mới phải đọc **toàn bộ** các file sau, không chỉ grep vài dòng:

1. `D:\Study\DSC2026\LegalIR\AGENTS.md`
2. `D:\Study\DSC2026\LegalIR\src\exp102_mil_nce_retrieval.py`
3. `D:\Study\DSC2026\LegalIR\tests\test_exp102_mil_nce.py`
4. `D:\Study\DSC2026\LegalIR\results\exp102_mil_nce_retrieval\REPORT_full_oof.json`
5. `D:\Study\DSC2026\LegalIR\src\exp021_e5_dense_candidates.py`
6. `D:\Study\DSC2026\LegalIR\src\exp021_aggregation_oof.py`
7. `D:\Study\DSC2026\LegalIR\src\exp021_sparse_depth_tune.py`, `src\exp021_sparse_retrieve.py`, report/cache depth-RRF tương ứng trong `results/exp021_sparse` và `cache/exp021_sparse` để lấy BM25 parent ranking đã tune — không lấy BM25 top-50 đã cắt từ EXP-022.
8. Các report aggregation/dense tương ứng trong `results/exp021_*` và `_SUCCESS.json`.
9. `D:\Study\DSC2026\LegalIR\src\exp022_e5_bm25_union_audit.py`
10. `D:\Study\DSC2026\LegalIR\results\exp022_e5_bm25_union\union_report.json`
11. Ít nhất ba dòng thật trong `cache/exp022_e5_bm25_union/train_oof_candidates.jsonl`, gồm một query có BM25-only candidate. Artifact này chỉ là historical comparator/input audit, **không** là retrieval contract của EXP-109A.
12. `D:\Study\DSC2026\LegalIR\src\exp104_preranker_benchmark.py`
13. `D:\Study\DSC2026\LegalIR\results\exp104_preranker_benchmark\REPORT_benchmark_summary.json`
14. `D:\Study\DSC2026\LegalIR\src\exp034_shallow_retrieval.py` để tái sử dụng pattern checkpoint, manifest, exact ranking và resume — không được đổi EXP-034.
15. `D:\Study\DSC2026\LegalIR\cache\cv_folds.json`, label-impact/exclusion artifacts và manifests của `final_preprocessed_v2`, `structural_v3_e5_final_v1`, `e5_final_v1`.
16. Paper gốc: [Dual-Encoders for Extreme Multi-Label Classification, ICLR 2024](https://openreview.net/forum?id=dNe1T0Ahby), đặc biệt công thức DecoupledSoftmax, SoftTop-k loss, implicit gradient và cách lấy negatives.

Sau khi đọc, tạo `results/exp109a_softtop5_retrieval/input_audit/READING_AUDIT.json` chứa:

- path, SHA-256 và schema/version của từng input;
- shape/dtype của embeddings;
- số chunks, parents, queries, folds;
- 6,991 evaluable và 9 non-evaluable có khớp canonical policy hay không;
- kiểm chứng số parent corpus thật và mọi canonical gold đều map được vào corpus;
- xác định depth/schema/fingerprint của BM25 parent ranking có thể tái tạo từ EXP-021;
- các khác biệt giữa plan và checkout nếu có.

Nếu bất kỳ contract chính nào không khớp, dừng `REJECTED_INPUT_GATE`; không tự sửa dữ liệu.

## 3. Bối cảnh thực nghiệm đã biết — phải kiểm chứng lại

Những con số dưới đây là context để agent biết mình đang so với cái gì, nhưng vẫn phải đọc report gốc:

- EXP-022 fixed union: E5@100 + tối đa 50 BM25 novel/backfill, đúng 150 parents/query; retained candidate Recall@150 khoảng `0.9906591`. Đây chỉ là **historical candidate coverage**. Ordered append-union chưa phải tuned rank fusion và không được dùng làm primary EXP-109A.
- EXP-102 VietLegal-E5 residual query projection:
  - dense OOF Recall@5 `0.9010966`;
  - static RRF OOF Recall@5 `0.9153960`;
  - static RRF Recall@150 `0.9932246`.
- EXP-104 Linear Pointwise: Recall@5 `0.9156821`, Recall@16 `0.9660015`, Recall@50 `0.9853526`, Recall@150 `0.9932246`.
- EXP-027 LambdaMART có verified retained-gold Recall@5 khoảng `0.9207612`, nhưng là reranking trong frozen pool và không được trộn metric scope với EXP-102.
- EXP-102 `compute_mil_nce_loss` đã chấm từng positive riêng với cùng negative set rồi lấy mean. Vì vậy EXP-109A **không được gọi decoupled multi-positive là novelty**.
- Corpus có rất nhiều tail/unseen labels theo outer folds. ID-only classifier có fold ceilings chỉ khoảng 0.69–0.74, nên bị cấm.

Lý do không quay lại BGE-M3 trong experiment này:

- BGE-M3 dense đã thua VietLegal-E5 trong benchmark sau parser/chunking mới.
- Tri-modal BGE-M3 là một representation experiment lớn và confound loss screen.
- Nếu EXP-109A pass, BGE-M3 multi-vector/sparse có thể trở thành EXP-109B riêng, không chen vào A.

## 4. Experimental question và hypotheses

### H0 — Null

SoftTop-5 không cải thiện matched-control Recall@5, hoặc cải thiện do leakage/tuning variance, hoặc đổi lại bằng mất Recall@16/50 và multi-gold.

### H1 — Primary

Trên score vector của toàn bộ parent corpus, SoftTop-5 tạo gradient tập trung vào biên rank 5/6 và tăng nested OOF Recall@5 ít nhất 0.5 percentage point so với matched decoupled control.

### H2 — Secondary

Hybrid loss ổn định hơn pure SoftTop-5: decoupled term giữ global separation, SoftTop-5 term tập trung vào decision boundary Top-5.

## 5. Namespace mới

Không sửa source/cache/results của EXP-021/022/027/034/102/104.

Tạo:

```text
src/exp109a_softtop5_retrieval.py
tests/test_exp109a_softtop5_retrieval.py
cache/exp109a_softtop5_retrieval/
results/exp109a_softtop5_retrieval/
docs/exp109a_runbook.md
```

Nếu file chính quá lớn, được tách thành `src/exp109a/`, nhưng CLI entrypoint trên vẫn phải tồn tại.

Mọi artifact stage phải có:

- schema version;
- config và input fingerprints;
- code SHA-256;
- manifest;
- `_SUCCESS.json` chỉ sau khi artifact hoàn chỉnh;
- atomic write/rename;
- resume không chấp nhận cache có fingerprint khác.

## 6. Shared representation và document score

### 6.1 Frozen inputs

- Query vectors: cached VietLegal-E5 train query embeddings.
- Chunk vectors: cached `e5_final_v1/embeddings.f16.npy`, cast/normalize đúng một lần khi scoring.
- Parent-to-chunk mapping: xây từ `chunk_ids.jsonl` và audit mọi chunk thuộc đúng parent.
- Chỉ train `ResidualProjection` low-rank query-side, khởi tạo identity, cùng dimension/rank với EXP-102 (`1024`, rank `32`) trừ khi audit chứng minh checkout khác.
- Document/chunk embeddings luôn frozen.

### 6.2 Aggregation bị khóa

EXP-109A dùng `top2_mean` cho **tất cả** actual arms:

\[
S(q,d)=\frac{1}{\min(2,|C_d|)}\sum_{c\in Top2(C_d)} \cos(f_\theta(q), e_c)
\]

Đây là policy đã được EXP-021 chọn; không mở lại grid max/top4/logsumexp trong EXP-109A.

Implementation phải:

- dùng tất cả source-exact chunks thuộc parent khi train/evaluate, không chỉ evidence top-4 lưu trong EXP-022;
- giữ gradient qua hai chunk scores thắng;
- deterministic tie-break bằng chunk index;
- xử lý parent chỉ có một chunk;
- không silent truncate/chunk sample.

### 6.3 Legacy bridge control

Vì EXP-102 train bằng normalized LogSumExp nhưng evaluate full-corpus bằng top2 mean, cần một control chẩn đoán riêng:

- `C0_legacy_exp102_replay`: load checkpoints EXP-102, chạy exact evaluator hiện có và phải reproduce report trong tolerance `1e-6` nếu deterministic environment cho phép, tối đa `1e-5` nếu dtype path khác được giải thích.
- C0 không tham gia promotion và không được dùng để chọn hyperparameters.
- Actual Arms A/B/C đều dùng shared `top2_mean`, vì mục tiêu là chỉ so loss trong cùng scorer.

Nếu replay không khớp, dừng `REJECTED_REPRODUCTION_GATE` trước khi train.

## 7. Exact full-corpus training contract

EXP-109A không có output shortlist K150 và không train trên ordered union EXP-022. Với mỗi training query, scorer phải tạo đúng một score cho **mọi parent document trong corpus** (khoảng 8.5K parents; lấy con số chính xác từ manifest).

Điểm quan trọng:

- `k=5` trong SoftTop-5 là decision boundary của objective.
- `K={1,3,5,10,...}` là cutoff lúc evaluation.
- Số labels/documents được chấm lúc train là toàn corpus, không phải một retrieval K.
- EXP-022 chỉ còn là historical comparator và audit nguồn BM25; không giới hạn membership của EXP-109A.

Rules:

- Mọi canonical gold parent phải hiện diện tự nhiên trong full-corpus score vector. Đây không phải “force-insert gold”.
- Mọi positives của query cùng nằm trong vector và không cạnh tranh như negatives trong Arm A.
- Không sample sáu negatives và không lấy top-150 làm denominator.
- Không học embedding theo `doc_id`; parent score chỉ đến từ frozen chunk text embeddings.
- Không xóa query khó hoặc gold đang rank sâu.
- Stable tie-break bằng canonical parent ID chỉ dùng sau score equality, không tham gia gradient.

### 7.1 Exact all-parent top2 scoring

Một batch query cần:

1. project query qua ResidualProjection;
2. chấm cosine với 343K frozen chunks;
3. segment theo parent;
4. giữ top-2 chunk scores/parent;
5. mean top-2 thành vector `[batch, num_parents]`;
6. tính loss trên toàn vector.

Phải implement blockwise/segment-wise để phù hợp RTX 4050 6GB nhưng vẫn giữ autograd tới hai winning chunks. Ưu tiên:

- audit xem chunks của mỗi parent có contiguous trong cache hay không;
- nếu contiguous, dùng parent offsets và batched segment top-k;
- nếu không, tạo read-only parent-major index mapping trong namespace EXP-109A, không duplicate embeddings nếu không cần;
- microbatch query từ 1 trở lên và gradient accumulation để giữ effective batch giống nhau giữa arms;
- chunk-score blocks được merge bằng differentiable top-2, không detach/no-grad.

### 7.2 Feasibility gate, không fallback âm thầm

Preflight phải benchmark forward/backward thật trên full 343K chunks và toàn parent vector với batch sizes `{1,2,4,8}` cho cả Arm A và SoftTop-5. Report VRAM, queries/s và ETA.

Nếu exact all-parent training không fit ngay cả microbatch 1 sau blockwise top-2:

- dừng `REJECTED_FULL_CORPUS_FEASIBILITY_GATE`;
- trình user một plan amendment cho hard-negative approximation;
- không tự quay về K150/K256 rồi giữ tên “full-corpus SoftTop-5”.

Primary evaluation luôn là full-corpus dense ranking và full-corpus dense+BM25 rank fusion.

## 8. Các loss arms

### Arm A — Matched decoupled control

Cho full-corpus parent score vector `s` và tập positives `P`:

\[
L_{dec}= -\frac{1}{|P|}\sum_{p\in P}
\log\frac{\exp(s_p/\tau)}{\exp(s_p/\tau)+\sum_{n\notin P}\exp(s_n/\tau)}
\]

- Mỗi positive đối đầu cùng toàn bộ non-golds.
- Positive khác không nằm trong denominator của positive đang xét.
- `tau=0.05` bị khóa theo EXP-102; không tune trong screen đầu.

### Arm B — Pure SoftTop-5

Với standardized score vector:

\[
x_i=(s_i-\mu_s)/(\sigma_s+10^{-6})
\]

tìm `t` sao cho:

\[
z_i=\sigma(\alpha(x_i+t)),\quad \sum_i z_i=5
\]

và:

\[
L_{top5}= -\frac{1}{|P|}\sum_{p\in P}\log(z_p+10^{-8})
\]

Không cài binary search dưới `no_grad` rồi giả vờ gradient đúng. Phải implement implicit gradient đúng công thức paper hoặc custom `autograd.Function`, sau đó finite-difference gradient check.

Grid nhỏ, chỉ inner-select:

```text
alpha ∈ {1.0, 2.0, 5.0, 10.0}
```

### Arm C — Hybrid

\[
L_{hybrid}=L_{dec}+\lambda L_{top5}
\]

Grid inner-only:

```text
lambda ∈ {0.10, 0.25, 0.50, 1.00}
alpha  ∈ {1.0, 2.0, 5.0}
```

Nếu scale hai terms lệch mạnh, report gradient norm từng term; không tự normalize loss hoặc đổi grid sau khi nhìn outer fold.

## 9. Optimizer/training controls

Tất cả arms phải dùng y hệt:

- cùng folds và training qids;
- seed `2026` và deterministic query order;
- cùng ResidualProjection initialization per comparison;
- cùng optimizer, LR, weight decay, epochs và batch/query accumulation;
- cùng exact full-corpus parent scores và top2 aggregation;
- cùng checkpoint cadence;
- cùng dtype/device;
- cùng evaluation/fusion code.

Default ban đầu, chỉ được thay nếu preflight cho thấy bất khả thi và phải ghi vào plan amendment trước run:

```text
optimizer: AdamW
learning_rate: 2e-4
weight_decay: 0.01
epochs: 4
grad_clip: 1.0
query_batch: chọn lớn nhất qua GPU preflight nhưng khóa chung cho mọi arm
early_stopping: disabled
```

Không cho mỗi arm một epoch budget khác. Lưu:

- model/optimizer/scheduler;
- epoch, batch/query position;
- Python/NumPy/Torch/CUDA RNG states;
- data order;
- config/input/code fingerprints.

Checkpoint tối thiểu cuối mỗi epoch và mỗi 500 query groups. Ctrl+C/crash phải resume được đúng batch boundary.

## 10. Strict nested CV

Folds cố định trong `cache/cv_folds.json`.

Với mỗi outer fold `Fo`:

1. `Fo` bị khóa hoàn toàn cho tới khi config thắng đã chọn.
2. Trên bốn folds còn lại, chạy four-way inner rotation:
   - train 3 folds;
   - validate fold thứ tư;
   - lặp cho cả 4 calibration folds.
3. Aggregate inner metrics theo query count, không average fold means mù.
4. Chọn arm/hyperparameter theo thứ tự:
   1. Recall@5;
   2. multi-gold Recall@5;
   3. Precision@5;
   4. Recall@16;
   5. MRR@5;
   6. nếu hòa trong `1e-4`, ưu tiên A rồi C có lambda nhỏ hơn rồi B.
5. Retrain đúng một winner trên toàn bộ bốn outer-train folds từ initialization mới nhưng cùng seed contract.
6. Chấm outer fold đúng một lần.

Không được:

- chọn alpha/lambda bằng aggregate OOF sau khi đã nhìn outer results;
- dùng Fold 0 như universal calibration cho các outer khác;
- dùng public score để đổi config;
- rerun nhiều seeds rồi chọn seed tốt nhất. Nếu sau promotion muốn robustness seeds, đó là audit hậu nghiệm riêng và không đổi winner.

## 11. Full-corpus retrieval và nested-tuned rank fusion

Không có bước “lọc union E5@100 + BM25@50 rồi mới rank”. Mỗi outer-fold model tạo full dense parent ranking; BM25 cung cấp parent ranking độc lập. Hai rankings được hợp nhất bằng weighted RRF đã chọn hoàn toàn trong inner folds.

### 11.1 Dense full-corpus ranking

1. Project heldout queries.
2. Chấm toàn bộ cached chunks theo batches.
3. Exact top2_mean cho từng parent, không chỉ lấy top-4096 chunks nếu depth audit cho thấy có thể mất parent cần thiết.
4. Sort toàn bộ parents theo dense score, stable tie-break parent ID.
5. Xuất tối thiểu top-150 để report, nhưng evaluator phải biết ranking full corpus.

### 11.2 BM25 source

- Dùng chính BM25 pipeline/tuned parent aggregation của EXP-021 đã được audit.
- Tái tạo parent ranks ở depth cao nhất cache cho phép; không giới hạn cứng 50.
- Không dùng `document_label` slug, không query expansion mới.
- Candidate absent khỏi cached BM25 depth nhận “missing sparse rank”, nhưng vẫn tồn tại qua dense full-corpus ranking.

### 11.3 Nested weighted-RRF tuning

Không khóa cứng `0.65/0.35, k=32`: policy đó là comparator cũ, không phải bằng chứng nested optimum cho model mới.

Grid định trước:

```text
dense_weight ∈ {0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90}
bm25_weight  = 1 - dense_weight
rrf_k        ∈ {10, 20, 32, 60, 100}
```

\[
S_{RRF}(d)=\frac{w_d}{k+r_d(d)}+\frac{1-w_d}{k+r_b(d)}
\]

Source không có rank thì contribution bằng 0; không append theo block và không preserve order của E5@100/BM25@50 union.

Với mỗi outer fold:

- chọn `(dense_weight, rrf_k)` từ bốn-way inner predictions theo Recall@5;
- tie-break bằng Recall@10, Recall@50, Precision@5, rồi ưu tiên weight gần `0.65` và `k=32` hơn;
- khóa policy trước khi outer scoring;
- chạy cùng grid/procedure cho A/B/C;
- report thêm “common fusion policy” lấy từ Arm A và áp cho cả ba để tách hiệu ứng loss khỏi interaction fusion;
- per-arm nested-tuned fusion là primary final-performance result; common-policy fusion là causal ablation.

Regex dynamic routing trong EXP-102 không được dùng làm primary vì đó là hypothesis khác và grid cũ đã nhìn aggregate OOF.

### 11.4 Output cutoffs

Cho dense và fused rankings, bắt buộc report:

```text
K = 1, 3, 5, 10, 16, 20, 32, 50, 64, 100, 150
```

Recall@5 là selection target. Recall@50/100/150 là ceiling/guardrail, không phải lý do chỉ xuất một list K150.

EXP-022 filled-union Recall@150 chỉ là historical reference; không được dùng làm EXP-109A output hoặc promotion baseline.

## 12. Metrics và error slices

Cho dense full-corpus, common-policy RRF và nested-tuned RRF, report:

- macro Recall@1/5/16/32/50/150;
- Precision@5;
- MRR@5;
- single-gold vs multi-gold Recall@5;
- query with 0/1/2+ retained positives;
- label frequency in outer-train: unseen, seen once, seen 2–4, seen ≥5;
- gold movement buckets: `>5→≤5`, `≤5→>5`, unchanged;
- rank-5/rank-6 score margin;
- per-fold delta and aggregate delta;
- number/fraction of queries improved, degraded, tied;
- bootstrap 95% CI for paired Recall@5 delta, resampling queries within folds;
- ceiling gap to the exact candidate contract used.

Multi-label Recall@5 phải dùng `|gold ∩ top5| / |gold|`, không chỉ hit-any-gold.

## 13. Gates tuần tự

### Gate 0 — Static/input

- reading audit hoàn chỉnh;
- 7,000 rows, 6,991 evaluable, 9 non-evaluable;
- five folds cover each query exactly once;
- embeddings/query IDs aligned;
- mọi canonical gold parent tồn tại trong full corpus;
- parent/chunk mapping cover toàn bộ chunks, không duplicate/orphan;
- BM25 ranking depth/schema tái tạo được từ EXP-021;
- manifests/fingerprints pass.

Fail: `REJECTED_INPUT_GATE`.

### Gate 1 — Mathematical correctness

- `sum(SoftTop5(x))` within `1e-5` of 5 for random and adversarial vectors;
- output strictly in `(0,1)`;
- permutation equivariance;
- shift invariance;
- finite gradients for ties/extreme scores;
- analytic/custom gradient agrees finite difference with relative error ≤`1e-3` on float64 fixtures;
- positive below rank 5 receives non-zero corrective gradient;
- no positive is inserted into another positive's negative set in Arm A.

Fail: `REJECTED_LOSS_IMPLEMENTATION_GATE`.

### Gate 2 — Reproduction

- ResidualProjection identity-init test;
- existing EXP-102 checkpoints reproduce exact report within stated tolerance;
- top2 scorer matches an independent NumPy reference on sampled real parents;
- existing untrained/frozen ranking fixture hash matches rerun.

Fail: `REJECTED_REPRODUCTION_GATE`.

### Gate 3 — Cheap smoke

Một small fixture từ train folds, không outer selection:

- 50 optimizer steps/arm;
- loss finite và decreases trên synthetic separable fixture;
- no leakage/gold insertion;
- checkpoint/resume produces same weights/rankings;
- VRAM headroom ≥10%;
- measured throughput/ETA recorded.

Smoke phải gồm full-corpus forward/backward; một toy candidate subset chỉ đủ test toán học, không đủ pass feasibility.

Fail: `REJECTED_PREFLIGHT_GATE`.

### Gate 4 — One-outer nested screen

Chỉ khi user cho phép run, chạy strict nested selection cho outer Fold 0 trước. Tiếp tục full five-fold chỉ nếu Fold 0 winner:

- matched-control delta Recall@5 ≥`+0.003` trên dense full-corpus **và** ≥`+0.002` trên nested-tuned RRF full-corpus;
- multi-gold Recall@5 không giảm;
- Recall@50 giảm không quá `0.001`;
- MRR@5 giảm không quá `0.003`.

Fail: `REJECTED_FOLD0_SCREEN`; không chạy bốn outer còn lại.

Gate này chỉ là resource gate, không phải bằng chứng promotion.

### Gate 5 — Full nested OOF promotion

EXP-109A pass chỉ khi winner aggregate:

- Recall@5 delta ≥`+0.005` so với matched Arm A;
- primary RRF Recall@5 ≥`0.9207`;
- multi-gold Recall@5 delta ≥`+0.020` hoặc ít nhất không giảm nếu baseline sample quá nhỏ; phải report cả CI;
- Recall@50 giảm ≤`0.001`;
- Recall@150 giảm ≤`0.0005`;
- Precision@5 giảm ≤`0.001`;
- MRR@5 giảm ≤`0.005`;
- ít nhất 4/5 outer folds có delta Recall@5 không âm;
- worst-fold delta ≥`-0.002`;
- paired bootstrap 95% CI không chứa một suy giảm lớn hơn `-0.001`.

Pass: `PASS_LOSS_GATE`, đủ cơ sở thiết kế EXP-109B.

Fail: `REJECTED_LOSS_GATE`. Không “cứu” bằng tuning post-hoc, BGE-M3, aggregation routing hoặc encoder fine-tuning trong namespace 109A.

## 14. Ablations bắt buộc nhưng có giới hạn

Chỉ các ablation sau được phép:

1. Arm A vs B vs C.
2. Dense full-corpus vs common-policy RRF vs nested-tuned RRF.
3. Single-gold vs multi-gold.
4. Seen/unseen label frequency.
5. Pure SoftTop-5 alpha và Hybrid alpha/lambda đã khai báo.

Không thêm ablation mới giữa run. Mọi ý tưởng phát sinh phải ghi vào `FOLLOWUPS.md` cho EXP-109B.

## 15. CLI và orchestration

CLI tối thiểu:

```powershell
D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py audit
D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py test-loss
D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py replay-exp102
D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py smoke --resume
D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py nested-screen --outer fold_0 --resume
D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py nested-oof --resume
D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py report
D:\Study\DSC2026\dsc_env\Scripts\python.exe -u src\exp109a_softtop5_retrieval.py status
```

`nested-screen` và `nested-oof` phải từ chối chạy nếu prerequisite gates thiếu `_SUCCESS.json` hoặc fingerprints mismatch.

## 16. Logging

Detailed logs vào:

```text
results/exp109a_softtop5_retrieval/logs/<run-id>/run.log
results/exp109a_softtop5_retrieval/logs/<run-id>/<stage>.log
results/exp109a_softtop5_retrieval/logs/<run-id>/<outer>/<inner>/<arm>.log
```

Terminal chỉ in:

- stage/outer/inner/arm bắt đầu;
- heartbeat mỗi 5 phút hoặc 500 query groups;
- progress, throughput, ETA, VRAM;
- latest train/validation loss;
- gate result;
- log/artifact path.

Flush logger sau mỗi heartbeat. Không dồn toàn bộ pair-level log ra terminal.

`RUN_STATUS.json` tối thiểu có run ID, state, stage, outer, inner, arm, epoch, completed/total, ETA, last heartbeat, config fingerprint và last checkpoint.

## 17. Tests bắt buộc trước khi xin phép run

- input/fold isolation;
- parent/chunk alignment;
- top2 NumPy parity, 1-chunk handling và deterministic ties;
- ResidualProjection identity init;
- Arm A multi-positive denominator correctness;
- SoftTop-5 sum/permutation/shift/gradient tests;
- hybrid gradient-norm logging;
- exact all-parent score-vector membership;
- all canonical positives appear exactly once without ID-specific parameters;
- blockwise top2 gradient parity với non-blockwise toy implementation;
- metric correctness cho multi-gold;
- weighted-RRF reference parity và nested grid isolation;
- nested selection không đọc outer labels;
- checkpoint/resume exactness;
- fingerprint mismatch fail-closed;
- interrupted artifact không có `_SUCCESS.json`;
- CPU synthetic smoke và CUDA preflight khi được phép.

Chạy regression tests liên quan với:

```powershell
$env:PYTHONPATH='src'
D:\Study\DSC2026\dsc_env\Scripts\python.exe -m pytest tests\test_exp109a_softtop5_retrieval.py tests\test_exp102_mil_nce.py -q
```

Không sửa tests cũ chỉ để làm chúng pass.

## 18. Deliverables trước khi chạy tốn thời gian

Conversation mới phải hoàn thành và báo user:

1. source + tests;
2. input/reading audit;
3. mathematical loss report gồm finite-difference gradient;
4. EXP-102 replay report;
5. CPU fixture/smoke report;
6. CUDA preflight + ETA dự kiến cho Fold 0 và full OOF;
7. runbook và exact command;
8. `IMPLEMENTATION_REPORT.md` liệt kê deviations/known risks.

Sau đó **dừng và xin phép user trước khi chạy `nested-screen`**.

## 19. Tiêu chí diễn giải kết quả

- Nếu B/C không thắng A: kết luận loss không phải bottleneck đủ lớn dưới frozen VietLegal-E5 representation; chuyển trọng tâm sang EXP-109B representation/encoder training, không tiếp tục massage loss.
- Nếu dense full-corpus tăng nhưng fused ranking không tăng: SoftTop-5 cải thiện geometry nhưng fusion policy hoặc BM25 interaction đang triệt tiêu gain; xem common-policy và nested-fusion ablation trước khi kết luận.
- Nếu single-gold tăng nhưng multi-gold giảm: SoftTop-5 implementation/objective chưa bảo toàn positive set; không promote.
- Nếu Hybrid thắng Pure: giữ cả separation và boundary pressure là cần thiết.
- Nếu pass +0.5pp: đó là bằng chứng loss có ích, **không** phải bằng chứng sẽ đạt 0.97. Khoảng cách còn lại phải đến từ representation/candidate ranking, encoder fine-tuning hoặc retrieval ensemble được kiểm định riêng.

## 20. Các điều cấm để tránh lặp sai lầm cũ

- Không gọi stored heldout/OOF score là public score.
- Không dùng outer fold để chọn alpha, lambda, epoch, RRF weight hoặc seed.
- Không bỏ query/gold bị miss khỏi denominator metric.
- Không train/evaluate trên E5@100 + BM25@50 ordered union như primary contract.
- Không gọi training score-set size là retrieval K.
- Không đổi aggregation theo query.
- Không thêm ID boost/document_label slug.
- Không dùng pretrained model khác giữa chừng.
- Không dùng EXP-107/108 cross-encoder evidence package trong EXP-109A.
- Không claim XMC label classifier: đây vẫn là text-based dual retrieval, chỉ mượn loss từ XMC.
- Không chạy full five-fold sau khi Fold 0 fail resource gate.

---

## Handoff prompt ngắn cho conversation mới

> Hãy triển khai EXP-109A đúng theo `D:\Study\DSC2026\LegalIR\docs\exp109a_softtop5_plan.md`. Trước tiên đọc toàn bộ reading order và tạo READING_AUDIT; không dựa vào lịch sử chat. EXP-109A screen SoftTop-5 trên frozen VietLegal-E5, exact full-corpus parent scoring, shared top2_mean và low-rank query projection. EXP-022 K150 chỉ là historical comparator, không phải retrieval contract. Final Stage 1 phải report K=1/3/5/10/16/20/32/50/64/100/150 và dùng E5+BM25 weighted RRF được tune strict-nested, không ordered append-union. EXP-102 đã có decoupled multi-positive loss, nên novelty duy nhất là SoftTop-5/hybrid. Implement, test, replay EXP-102, full-corpus smoke và lập ETA; dừng xin phép trước nested Fold-0/GPU run. Không download, encode lại, public inference hay submission.
