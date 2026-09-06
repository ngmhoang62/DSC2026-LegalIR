# EXP-110P — Fold-Safe Semantic Label-Prototype Cache on Colab

## 0. Mục đích của tài liệu

Đây là implementation plan tự chứa đầy đủ context cho một agent ở conversation
khác. Agent triển khai không được dựa vào lịch sử chat. Trước khi viết code,
agent phải đọc toàn bộ reading order ở Mục 5, tạo `READING_AUDIT.json`, rồi mới
được triển khai notebook.

EXP-110P là một nhánh chạy song song trong lúc EXP-109C hoàn thành Jina
late-interaction. Nó không thay thế, không sửa và không đọc artifact chưa hoàn
thành của EXP-109C.

Mục tiêu của EXP-110P là kiểm tra giả thuyết:

> Với một document pháp luật đã từng là gold cho các training query khác, các
> query embeddings của những positive examples đó có tạo thành một prototype/
> exemplar memory đủ mạnh để giúp đưa document ấy vào Top-5 cho một query mới
> hay không?

Đây là specialist channel từ Extreme Multi-Label Classification/few-shot metric
learning, không phải một retriever thay thế toàn bộ E5/LAL/BM25.

Primary target:

- Strict cross-fit F1–F4 Recall@5 tăng ít nhất `+0.005` so với EXP-109B
  LambdaMART anchor.
- Stretch target: `+0.008` hoặc cao hơn.
- Không giảm multi-gold, Precision@5, MRR@5 và Recall@1 ngoài safety margins.
- Fold 0 tuyệt đối không được đọc trước khi configuration lock và explicit user
  authorization.

## 1. Bằng chứng đã biết và claim boundary

Canonical label policy hiện hành:

```text
canonical_duplicate_alias_drop_empty_passage_v1
label_fingerprint = 9bdf9593b61fe3423d1f1a819ac9fb3e8d7225e6003da0afb840c1f5853fd4c9
```

Số liệu đã kiểm tra trực tiếp:

```text
7,000 total queries
6,991 evaluable queries
9 non-evaluable queries
7,626 canonical gold assignments
3,099 / 8,507 documents từng xuất hiện làm gold
1,238 documents xuất hiện từ hai lần trở lên
Fold 0: 1,087 / 1,523 gold occurrences (71.3723%) có label từng xuất hiện
        trong F1–F4
```

Con số `71.3723%` chỉ là seen-label occurrence coverage của Fold 0, không phải
Recall@5 và không được dùng để claim chất lượng.

EXP-109B strict inner anchor trên 5,593 evaluable F1–F4 queries:

```text
Recall@5                  = 0.9251057869956494
Recall@1                  = 0.6753531199713928
MRR@5                     = 0.8044072948328268
Precision@5               = 0.19749687108886105
Single-gold Recall@5      = 0.9429237041351194
Multi-gold Recall@5       = 0.7174585218702866
```

Winner của EXP-109B là `lambdamart_top50_per_source`, sử dụng E5, VnLegal-LAL
và tuned BM25. EXP-110P phải reproduce anchor này trước khi thêm prototype.

EXP-024 đã thử query-memory dạng TF-IDF:

```text
char 3–5 grams: 21 rescued queries / 26 gold occurrences outside EXP-022 pool
word 1–2 grams: 19 rescued queries / 24 gold occurrences
```

Rescues phân bố không ổn định theo fold. Vì vậy EXP-110P không được mô tả là
“query-memory hoàn toàn mới”; novelty thực tế là:

- Semantic E5 query embedding thay TF-IDF lexical similarity.
- Document-level multi-exemplar/centroid/cache-vote features.
- Confidence-gated residual và strict nested fusion với EXP-109B.
- Seen/unseen label audit và candidate-expansion ablation.

EXP-110P không dùng LLM, không sinh query, không thêm annotation và không dùng
dữ liệu ngoài contest.

## 2. Cơ sở phương pháp

Ba ý tưởng được chắt lọc:

1. **Prototypical Networks**: biểu diễn một class bằng mean embedding của support
   examples, rồi dự đoán bằng khoảng cách tới prototype.
   Primary paper: https://papers.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf
2. **Tip-Adapter**: xây key-value cache từ frozen embeddings và labels, retrieve
   support examples rồi residual-combine cache prediction với pretrained prior.
   Primary paper: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950487.pdf
3. **PRIME/XMC label prototypes**: XMC có thể xem là query-to-prototype
   prediction; label prototypes aggregate signals từ related queries.
   Primary paper: https://aclanthology.org/2025.naacl-long.537.pdf

EXP-110P chỉ dùng phần training-free/cache-only phù hợp task hiện tại. Không
triển khai prototype-network training, dynamic-margin loss hay fine-tuning E5;
những phần đó thuộc EXP-110A nếu cache-only screen thất bại.

## 3. Kiến trúc tổng quát

```text
Canonical train labels + fixed folds
                 │
Frozen VietLegal-E5 query embeddings (7,000 × 1,024)
                 │
Strict support bank for each train/validation context
                 │
       ┌─────────┼──────────────┐
       │         │              │
 max exemplar  centroid     soft cache vote
       │         │              │
       └─────────┴──────────────┘
                 │
Prototype features for EXP-109B candidate documents
                 │
       ┌─────────┴──────────────┐
       │                        │
Feature-augmented           confidence-gated
LambdaMART                 protected residual
       │                        │
       └──────── nested selection ────────┘
                 │
Strict cross-fit F1–F4 evaluation
                 │
       PASS → stop and request Fold-0 authorization
       FAIL → preserve diagnostic; no Fold 0
```

## 4. Resource decision: Colab, CPU-first

Current local machine has only khoảng 300 MB available RAM while EXP-109C is
running. Không được chạy EXP-110P locally cùng lúc với Jina, ngoài static tests
nhỏ không đọc large cache.

EXP-110P chạy trên Google Colab với Google Drive mounted. GPU không bắt buộc vì:

- E5 query embeddings đã encode sẵn.
- Không load VietLegal-E5 model.
- Không encode corpus/document.
- Không backpropagation.
- Ma trận lớn nhất inner screen chỉ khoảng `5,600 × 5,600` FP32 (~125 MB).

CPU Colab với NumPy/BLAS là primary path. Nếu Colab có GPU, notebook có thể dùng
GPU chỉ cho block matrix multiplication rồi trả kết quả về CPU/Drive. GPU path
phải parity với NumPy FP32.

GPU budget rule:

```text
Measured projected GPU time ≤ 1 hour → được phép dùng.
1–4 hours                         → báo ETA rồi mới tiếp tục.
> 4 hours hoặc có nguy cơ vượt 5h → STOP_GPU_BUDGET, chuyển CPU.
```

Không có lý do hợp lý để EXP-110P dùng GPU quá 5 giờ. Nếu implementation dự báo
như vậy thì đó là dấu hiệu thiết kế sai hoặc đang encode lại model trái plan.

## 5. Reading order bắt buộc

Agent phải đọc toàn bộ các file sau trước khi implement:

1. `D:\Study\DSC2026\LegalIR\docs\EXP-110P_PLAN.md`
2. `D:\Study\DSC2026\LegalIR\src\exp024_memory_lexical_backoff.py`
3. `D:\Study\DSC2026\LegalIR\results\exp024_memory_lexical\report.json`
4. `D:\Study\DSC2026\LegalIR\src\exp013_candidates.py`
5. `D:\Study\DSC2026\LegalIR\src\exp109b_encoder_complementarity.py`
6. `D:\Study\DSC2026\LegalIR\docs\EXP-109B_PLAN.md`
7. `D:\Study\DSC2026\LegalIR\results\exp109b_encoder_complementarity\cached_fusion_pilot\fold_0\CACHED_FUSION_PILOT.json`
8. `D:\Study\DSC2026\LegalIR\src\exp109c_latent_condition_late_interaction.py`
   - chỉ để reuse canonical label/fold/metric semantics;
   - không đọc late-score results làm feature.
9. `D:\Study\DSC2026\LegalIR\cache\exp021_e5_dense_candidates\query_embeddings\manifest.json`
10. `D:\Study\DSC2026\LegalIR\results\exp108_atomic_condition_reranker\audit-inputs\REPORT.json`

`READING_AUDIT.json` phải ghi path, SHA-256, contract/finding và vai trò của từng
file. Thiếu bất kỳ file bắt buộc nào thì notebook fail-closed.

## 6. Namespace và deliverables

Không sửa EXP-013/014/024/109A/109B/109C.

Tạo:

```text
docs/EXP-110P_PLAN.md
notebooks/exp110p_semantic_label_prototype_colab.ipynb
src/exp110p_prepare_colab_bundle.py
src/exp110p_semantic_label_prototype.py
tests/test_exp110p_semantic_label_prototype.py
results/exp110p_semantic_label_prototype/IMPLEMENTATION_REPORT.md
results/exp110p_semantic_label_prototype/IMPLEMENTATION_REPORT.json
```

Notebook là executable chính và phải tự đủ để chạy trên Colab. Module `.py` là
audit/test mirror của core logic; notebook không được chỉ chứa một cell gọi một
Windows-only script.

Không hard-code `D:\...` bên trong runtime Colab. Windows paths chỉ xuất hiện
trong local bundle-preparation cell/script.

## 7. Google Drive layout

Notebook cell đầu phải mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Một config cell duy nhất:

```python
DRIVE_ROOT = Path('/content/drive/MyDrive/DSC2026/LegalIR/exp110p')
RUN_ID = ...
USE_GPU_FOR_SIMILARITY = False
OUTER = 'fold_0'
```

Notebook phải tự `mkdir(parents=True, exist_ok=True)` cho:

```text
<DRIVE_ROOT>/input/
<DRIVE_ROOT>/code/
<DRIVE_ROOT>/cache/
<DRIVE_ROOT>/cache/similarity/
<DRIVE_ROOT>/cache/prototypes/
<DRIVE_ROOT>/cache/features/
<DRIVE_ROOT>/cache/predictions/
<DRIVE_ROOT>/results/
<DRIVE_ROOT>/results/logs/<RUN_ID>/
<DRIVE_ROOT>/checkpoints/
<DRIVE_ROOT>/exports/
<DRIVE_ROOT>/invalidated/
```

Nếu directory chưa tồn tại, code phải tạo; không yêu cầu user tạo thủ công.

## 8. Colab input bundle

Không upload full corpus, E5 corpus matrix, BM25 SQLite hay Jina index. Chỉ tạo
một compact bundle đủ cho cache-only experiment.

`exp110p_prepare_colab_bundle.py` phải stream dữ liệu và giữ peak private RAM
thấp; không materialize các ranking files hàng trăm MB cùng lúc.

Bundle bắt buộc:

```text
input/train.json
input/cv_folds.json
input/exclusions.json
input/label_impact_report.json
input/e5_query_embeddings/train_queries.f32.npy       # ~28.7 MB
input/e5_query_embeddings/train_query_ids.json
input/e5_query_embeddings/manifest.json
input/exp109b_sources_top50.jsonl
input/exp109b_anchor_inner_predictions.jsonl
input/exp109b_locked_configs.json
input/exp109b_pilot_report.json
input/parent_metadata_minimal.jsonl
input/exp024_report.json
input/INPUT_MANIFEST.json
```

`exp109b_sources_top50.jsonl` mỗi query chứa top-50 của:

```text
vietlegal_e5
vnlegal_lal
bm25 (đã dùng đúng fold-specific tuned parent aggregation)
```

Mỗi candidate source row giữ `doc_id`, rank, raw score và source name. Không
export fold labels vào source rankings.

`exp109b_anchor_inner_predictions.jsonl` chứa strict cross-fit predictions và,
nếu có, LightGBM raw scores cho F1–F4. Nó phải được tạo lại từ locked configs
của verified EXP-109B pilot; không lấy Fold-0 predictions.

`INPUT_MANIFEST.json` ghi:

- schema version;
- canonical label policy/fingerprint;
- folds fingerprint;
- source manifests/fingerprints;
- từng file path tương đối, bytes và SHA-256;
- exporter code SHA-256;
- query/document counts;
- explicit `fold0_predictions_included=false`.

Bundle được copy/upload vào `<DRIVE_ROOT>/input/`. Notebook phải hash-verify toàn
bộ trước khi dùng. Không tự download model hoặc contest data từ Internet.

## 9. Notebook UX, logging và resume

Mỗi phase là một nhóm cells độc lập, có heading Markdown và một orchestrator
cell. Cell output phải in trực tiếp:

- phase bắt đầu/kết thúc;
- completed/total;
- throughput;
- ETA;
- current RAM/available RAM;
- GPU/VRAM nếu được dùng;
- cache/result path;
- gate status.

Đồng thời tee cùng log vào:

```text
<DRIVE_ROOT>/results/logs/<RUN_ID>/run.log
<DRIVE_ROOT>/results/logs/<RUN_ID>/<phase>.log
```

Logger flush mỗi line. Không để toàn bộ logs chỉ nằm trong notebook cell output.

Mỗi phase ghi:

```text
RUN_STATUS.json
phase manifest
_SUCCESS.json chỉ khi hash verify pass
```

Writes phải atomic theo `temporary file → fsync/close → rename`. Với Drive không
hỗ trợ atomic semantics hoàn hảo, ghi receipt SHA-256 sau khi file final đóng và
verify lại trước resume.

Similarity/features cache theo block/shard; disconnect Colab chỉ chạy lại shard
chưa có verified receipt.

## 10. Phase 0 — Implementation/static audit

Trước Colab run:

- Implement core code và notebook.
- Notebook JSON phải parse được bằng `nbformat`.
- Không có output lớn hoặc stale execution state committed trong notebook.
- Local unit tests chỉ dùng synthetic fixtures nhỏ.
- Không load large E5/LAL/BM25 artifacts khi EXP-109C đang chiếm RAM.
- Tạo `IMPLEMENTATION_REPORT.md/json` nêu rõ implemented/missing/not-run.

Không được báo “workflow complete” nếu mới static-test notebook.

## 11. Phase 1 — Colab bootstrap/preflight

Notebook:

1. Mount Drive.
2. Tạo directories.
3. Pin/record versions tương thích:

```text
numpy
scipy
scikit-learn
lightgbm
psutil
torch (chỉ optional GPU similarity)
```

Local reference environment hiện là:

```text
numpy 2.4.4
scikit-learn 1.9.0
lightgbm 4.7.0
scipy 1.18.0
```

Nếu Colab không có đúng versions, agent phải pin hoặc chứng minh ranking parity;
không bypass baseline reproduction vì “version khác”.

Preflight đo:

- CPU count;
- total/available RAM;
- Drive free space;
- GPU availability và model nếu có;
- block matrix multiply benchmark trên 256 queries;
- projected full inner runtime;
- projected peak RAM.

Gate:

```text
available RAM ≥ 4 GiB
Drive free ≥ 5 GiB
projected peak RAM ≤ 70% total
projected CPU runtime ≤ 3 hours
optional GPU runtime ≤ 1 hour, otherwise require user notice
```

Fail → `REJECTED_COLAB_RESOURCE_GATE`.

## 12. Phase 2 — Canonical input and leakage audit

Recompute canonical labels, không tin manifest mù quáng. Bắt buộc reproduce:

```text
7,000 total
6,991 evaluable
9 non-evaluable exact QIDs
7,626 assignments
3,099 observed gold documents
1,238 documents with frequency ≥2
label fingerprint exact match
```

Kiểm tra:

- folds disjoint và cover đúng 7,000 queries;
- embedding IDs cover đúng queries;
- embedding shape `[7000, 1024]`;
- finite, row norms gần 1 sau explicit FP32 renormalization;
- no Fold-0 anchor predictions/source-derived labels in primary inner artifacts;
- candidate doc IDs thuộc known parent universe;
- canonical gold/aliases/exclusions nhất quán.

Gate:

```text
PASS → Phase 3
FAIL → REJECTED_INPUT_OR_LEAKAGE_GATE
```

## 13. Phase 3 — EXP-109B anchor reproduction

Chỉ trên F1–F4, reproduce exact candidate union top-50/source và strict
cross-fit EXP-109B anchor.

Expected metrics:

```text
evaluable queries       5,593
Recall@5                0.9251057869956494
Recall@1                0.6753531199713928
MRR@5                   0.8044072948328268
Precision@5             0.19749687108886105
multi-gold Recall@5     0.7174585218702866
```

Requirements:

- Prediction order identical 100% với exported anchor predictions.
- Candidate set identity/ordering deterministic.
- Metric tolerance `1e-12` từ predictions.
- Nếu LightGBM raw scores khác do version nhưng ranking identical, ghi rõ; raw
  score không được dùng làm cross-environment equality claim.

Fail → `REJECTED_ANCHOR_REPRODUCTION_GATE` trước prototype construction.

## 14. Phase 4 — Similarity cache

Primary inner universe chỉ gồm F1–F4. Fold 0 query vectors không được đưa vào
inner similarity cache.

L2-renormalize frozen E5 embeddings in FP32, rồi compute cosine blocks:

```text
S = Q_inner @ Q_inner.T
shape ≈ 5600 × 5600
dtype FP32
diagonal = -inf khi dùng làm neighbor memory
```

Không bắt buộc giữ cả matrix trong RAM. Preferred implementation:

- block rows 256 hoặc 512;
- write FP32 `.npy` shards/memmap;
- SHA-256 receipt mỗi block;
- resume missing blocks;
- assemble logical matrix bằng mmap.

GPU optional path phải so với NumPy FP32 trên ≥20 blocks:

```text
max absolute error ≤ 1e-5
top-64 neighbor identity/order = 100%
stable tie-break by qid
```

Nếu top-neighbor order khác, dùng CPU FP32 path.

## 15. Phase 5 — Fold-safe support contexts

Không được tạo một global prototype bank rồi dùng cho mọi validation query.

Với final heldout inner fold `H`:

- final train support = ba inner folds còn lại;
- heldout `H` features dùng labels/prototypes chỉ từ final train support;
- training query feature dùng final train support trừ chính query đó.

Khi tune bên trong ba training folds:

- validation fold `V` features dùng chỉ hai train folds;
- train-row feature dùng hai train folds trừ chính query đó;
- tuyệt đối không dùng label của `V` hoặc `H` trong prototype/frequency/threshold.

Mọi prototype feature record phải giữ provenance:

```json
{
  "target_qid": "...",
  "candidate_doc_id": "...",
  "support_fold_names": ["..."],
  "support_query_count": 0,
  "self_excluded": true,
  "heldout_excluded": true,
  "support_fingerprint": "..."
}
```

Không lưu danh sách support QIDs lặp lại trong từng row; lưu fingerprint và một
sidecar mapping fingerprint → support IDs.

## 16. Phase 6 — Prototype/cache representations

Mỗi support query là một key E5 embedding. Mỗi canonical gold document của
query đó là một value/label. Multi-gold query đóng góp vào từng gold label.

Không dùng document ID như numeric/categorical feature; `doc_id` chỉ là lookup
key và deterministic tie-break.

### 16.1 Exemplar features

Cho candidate document `d`, lấy support queries có `d` là gold:

```text
proto_seen
proto_support_count
proto_log_support_count
proto_max_similarity
proto_second_similarity
proto_top2_mean
proto_top3_mean
proto_max_minus_second
proto_max_minus_top3_mean
```

Nếu không seen:

```text
proto_seen = 0
similarity features = fixed missing sentinel
```

Không dùng zero nếu zero có thể bị hiểu là cosine hợp lệ; thêm explicit missing
indicator và thống nhất sentinel.

### 16.2 Centroid features

Document prototype:

```text
c_d = normalize(mean(normalized support query embeddings for d))
proto_centroid_cosine = q · c_d
proto_max_minus_centroid
proto_support_dispersion
```

Dispersion là mean/variance cosine từ exemplars tới centroid, tính support-only.
Không triển khai full covariance hoặc Gaussian classifier trong EXP-110P.

### 16.3 Tip-Adapter-style soft cache vote

Lấy top-N support neighbors của target query, rồi:

```text
affinity_i = exp(beta * (similarity_i - 1))
vote(q,d) = Σ_i affinity_i × 1[d ∈ gold(q_i)]
```

Tạo cả:

- raw vote;
- vote chia `|gold(q_i)|` để multi-label support query không nhân tổng mass;
- frequency-normalized vote chia `support_count(d)^gamma`;
- winning-document margin và normalized vote entropy.

Bounded policy grid:

```text
N     ∈ {16, 32, 64}
beta  ∈ {5, 10, 20}
gamma ∈ {0.0, 0.5, 1.0}
```

Không chạy Cartesian product 27 cấu hình qua mọi downstream arm. Phase 7 strict
nested source screen chọn tối đa một policy/fold; downstream chỉ nhận winner và
fixed exemplar/centroid features.

### 16.4 Query-level confidence features

```text
nearest_support_similarity
nearest_minus_second_support_similarity
top_neighbor_document_agreement
prototype_winner_margin
prototype_vote_entropy
prototype_seen_candidate_fraction
source_agreement_top5 from EXP-109B
anchor rank1/rank5 margin if available
```

Đây là inference-safe features; threshold/model chỉ được fit trong train context.

## 17. Phase 7 — Prototype source screen

Đánh giá riêng, strict nested:

```text
P1: max exemplar
P2: normalized centroid
P3: selected soft cache vote
P4: fixed normalized combination of P1–P3
```

Report K:

```text
1, 3, 5, 10, 16, 20, 50
```

Report trên:

- all evaluable;
- seen-label gold occurrences;
- unseen-label gold occurrences;
- support frequency `1`, `2–3`, `4+`;
- single/multi-gold;
- exact normalized query duplicate;
- E5 nearest similarity bands;
- per fold.

Candidate-expansion diagnostic:

- Top-10 prototype documents ngoài EXP-109B union.
- Novel gold occurrences/queries.
- Per-fold consistency.
- So sánh trực tiếp EXP-024 char/word rescue counts.

Không promotion chỉ vì standalone prototype Recall@5 thấp; specialist value được
đo bằng complementarity/choice oracle và downstream fusion.

## 18. Phase 8 — Downstream arms

Candidate base là unique union top-50/source của E5 + LAL + tuned BM25.

### Arm A — Anchor control

Reproduced EXP-109B LambdaMART, không prototype.

### Arm B — Prototype features in LambdaMART

Append toàn bộ selected exemplar/centroid/cache/confidence features vào existing
EXP-109B source features. Missing features phải explicit.

Hyperparameter search bounded quanh EXP-109B locked configs. Dùng same
LambdaMART objective/metric, deterministic seed và exact group boundaries.

### Arm C — Confidence-gated protected residual

Không cho prototype phá mọi query:

```text
S_final(q,d)
  = z_query(S_anchor(q,d))
  + alpha * g(q) * z_query(S_proto(q,d))
```

Trong đó `g(q)` phụ thuộc vào nearest similarity, prototype margin/entropy,
label-seen availability và baseline confidence. Grid nhỏ, strict nested:

```text
alpha ∈ {0.05, 0.10, 0.20, 0.30}
confidence threshold ∈ train-context quantiles {P50, P70, P85}
```

High-confidence anchor phải có option `g(q)=0`. Không học threshold trên heldout.

### Arm D — Prototype expansion

Chỉ chạy nếu Phase 7 tìm được ≥20 novel gold occurrences tổng và mỗi fold có ít
nhất 2 occurrences:

- thêm tối đa top-10 prototype-only documents;
- source features missing/default;
- prototype features đầy đủ;
- không loại candidate gốc;
- report candidate count/ceiling riêng.

Nếu source gate không đạt, Arm D tự skip; không reject toàn EXP.

### Arm E — Nested arm selection

Trong từng heldout inner fold, chỉ training/tuning folds được chọn giữa B/C/D.
Arm A luôn là fallback. Selection order:

1. Recall@5.
2. Multi-gold Recall@5.
3. Precision@5.
4. MRR@5.
5. Recall@1.
6. Nếu hòa, chọn arm đơn giản hơn: A → C → B → D.

Không chọn một global arm bằng chính aggregate OOF metric rồi claim cùng metric.

## 19. Phase 9 — Strict inner gate

Primary report gồm:

- Anchor và từng arm.
- Nested-selected winner predictions.
- Recall@1/3/5/10/16/20/50.
- Precision@5, MRR@5.
- Single/multi-gold Recall@5.
- Seen/unseen/frequency/duplicate/similarity-band breakdowns.
- Per-fold delta.
- Wins/losses/ties, gold into/out of Top-5.
- Paired bootstrap 10,000 samples.
- Choice-oracle anchor vs prototype arms, đánh dấu label-dependent.
- Prototype coverage và candidate-expansion rescues.

Promotion gate:

```text
aggregate Recall@5 delta vs EXP-109B ≥ +0.005
paired-bootstrap lower 95% CI > 0
at least 3/4 folds improve
worst-fold delta ≥ -0.002
multi-gold Recall@5 delta ≥ 0
Precision@5 non-decrease
MRR@5 delta ≥ -0.001
Recall@1 delta ≥ -0.002
```

Pass:

```text
PASS_EXP110P_STRICT_INNER_GATE
```

Fail:

```text
REJECTED_EXP110P_STRICT_INNER_GATE
```

Alternative diagnostic, không promotion:

```text
choice oracle ≥ +0.015 nhưng fusion fail
→ COMPLEMENTARY_ROUTER_NOT_SOLVED
```

Không nới gate hậu nghiệm. Nếu delta nằm `+0.003..+0.005`, report là weak signal,
không Fold 0.

## 20. Phase 10 — Fold 0, chỉ sau authorization

Notebook phải dừng sau strict inner gate và in rõ:

```text
Fold 0 has not been read.
Explicit user authorization is required.
```

Chỉ khi user cho phép:

- support bank = toàn bộ F1–F4;
- prototype policy/arm/hyperparameters đã lock;
- encode không cần làm lại vì Fold-0 E5 embedding đã có;
- score Fold 0 đúng một lần;
- không retune sau khi nhìn result;
- không public inference/submission.

Fold-0 report phải tách seen/unseen labels vì prototype không thể giúp label chưa
từng xuất hiện nếu không có text-document signal từ anchor.

## 21. Tests bắt buộc

### Data/leakage

- Canonical label counts/fingerprint.
- Fold disjointness.
- Heldout labels absent from support bank.
- Self query absent from training-row memory.
- Fold 0 absent from all inner support/selection artifacts.
- Multi-gold query contributes đúng từng label.
- Canonical aliases do not create duplicate labels.

### Math

- NumPy cosine vs reference.
- Centroid normalization.
- Max/top-k exemplar correctness.
- Tip cache vote vs small dense reference.
- Frequency/cardinality normalization.
- Missing-label sentinel behavior.
- Stable tie-break.
- Query-wise z-score and residual formula.

### Pipeline

- EXP-109B candidate/anchor reproduction.
- Nested context isolation.
- Candidate expansion never removes base candidates.
- LambdaMART group boundaries.
- Nested arm selection does not use heldout metrics for training.
- Bootstrap determinism.
- Gate pass/reject/weak-signal cases.
- Interrupt/resume with hash-verified shards.
- Corrupt receipt/cache rejection.
- Notebook parses and all required cells/tags exist.
- Drive directories auto-create.
- Logger prints to cell and file with flush.

Synthetic tests phải chạy trước Colab real data. Real fixture gồm ít nhất:

- 20 queries;
- seen/unseen labels;
- one/multi-gold;
- singleton/repeated labels;
- exact and near-duplicate queries;
- candidate absent/present cases.

## 22. Expected runtime/storage

Ước lượng sau implementation, phải thay bằng measured preflight:

```text
Input bundle compressed:         khoảng 50–300 MB tùy source export
Inner FP32 similarity cache:     khoảng 125 MB
Features/predictions/results:    dưới 1–2 GB
CPU runtime:                     khoảng 20–90 phút
Optional GPU similarity:         vài phút, chắc chắn dưới 1 giờ nếu đúng design
```

Đây không phải workload 5+ giờ GPU. LightGBM và prototype aggregation chạy CPU.

## 23. Final implementation report và Drive upload checklist

Sau khi viết xong, agent phải trả về user một report rõ ràng gồm:

1. Files đã tạo/sửa.
2. Notebook cell map: cell nào thực hiện phase nào.
3. Checklist Mục 21: PASS/FAIL/NOT RUN từng test.
4. Những phần plan đã implement đầy đủ.
5. Những phần chưa implement hoặc chỉ static-test.
6. Input bundle files cần user upload.
7. Files thực tế đã upload/copy lên Drive.
8. Drive absolute paths, sizes và SHA-256.
9. Measured RAM/CPU/GPU/ETA.
10. Chính xác gate nào đã chạy và scope metric.
11. Fold 0 đã được đọc hay chưa.
12. Resume command/cell nếu Colab disconnect.

Notebook phải tự tạo:

```text
<DRIVE_ROOT>/results/IMPLEMENTATION_REPORT.json
<DRIVE_ROOT>/results/DRIVE_UPLOAD_MANIFEST.json
<DRIVE_ROOT>/results/RUN_REPORT.json              # sau real run
<DRIVE_ROOT>/exports/exp110p_selected_predictions.jsonl
<DRIVE_ROOT>/exports/exp110p_feature_manifest.json
```

`DRIVE_UPLOAD_MANIFEST.json` mỗi file ghi source, destination, bytes, SHA-256,
upload/copy status và verification time. Không được nói “uploaded” nếu chỉ mới
tạo file local.

Notebook cuối cùng phải print một summary ngắn trực tiếp trong cell:

```text
STATUS
anchor Recall@5
winner Recall@5 / delta
multi-gold delta
per-fold deltas
bootstrap CI
seen/unseen breakdown
gate result
Fold-0 seen? true/false
result/log paths on Drive
```

## 24. Stop rules

Fail-closed ngay khi:

- Input/hash/canonical label mismatch.
- EXP-109B anchor không reproduce.
- Fold/self leakage.
- Similarity/top-neighbor parity fail.
- Missing/corrupt Drive cache receipt.
- RAM vượt 70% projected or process risks Colab OOM.

Không tự chạy Fold 0, public inference hoặc submission.

Nếu strict inner fail nhưng choice oracle thấp hơn `+0.015`, đóng EXP-110P và
chuyển sang EXP-110A task-adaptive VietLegal-E5. Nếu oracle cao, chỉ ghi nhận
router opportunity; không sửa gate trong cùng run.