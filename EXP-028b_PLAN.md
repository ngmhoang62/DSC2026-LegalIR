# EXP-028b — benchmark họ pre-ranker trên frozen 24-feature representation

## Tóm tắt

Tạo namespace `exp028b_preranker_benchmark` để kiểm tra giả thuyết: model-family, không chỉ feature/hyperparameter LambdaMART, đang giới hạn Recall@5.

Giữ nguyên EXP-022 gồm 7.000 query × 150 parent IDs, 24 features EXP-027 và K=64. Dùng label policy canonical hiện tại: 6.991 evaluable, 9 non-evaluable, fingerprint `9bdf9593b61fe342...`. EXP-028 cũ chỉ là tham chiếu lịch sử vì dùng denominator khác.

Không chạy cùng lúc với EXP-033 đang active; không dừng hoặc sửa EXP-033. EXP-028b không dùng GPU, không dựng capsule, không fine-tune heavy reranker và không fit single full-train pre-ranker.

## Benchmark và nested selection

- CLI gồm `audit-inputs`, `nested-oof`, `importance`, `report`.
- Audit khóa candidate membership/order, sidecar, 24-column feature schema, folds, canonical labels, preprocessing, Structural-v3 và toàn bộ upstream fingerprints.
- Cả năm họ dùng đúng cùng 24 features:
  1. Pointwise Logistic Regression: fold-fit `StandardScaler`, `class_weight=balanced`, `C ∈ {0.01,0.1,1,10}`.
  2. Pairwise linear logistic: standardized positive-minus-negative vectors, không intercept, cùng C grid. Mỗi positive lấy 32 negatives deterministically: 12 head, 12 ranks 13–64 và 8 seeded tail 65–150; stable backfill, mirrored differences.
  3. LightGBM binary pointwise với fold-local `scale_pos_weight`.
  4. LightGBM LambdaMART `lambdarank`.
  5. LightGBM `rank_xendcg`, đã xác nhận hoạt động với LightGBM 4.7.0.
- Ba họ LightGBM dùng chung grid 24 cấu hình để không thiên vị objective:
  `num_leaves ∈ {15,31,63}`,
  `min_child_samples ∈ {20,50}`,
  `n_estimators ∈ {250,500}`,
  `reg_lambda ∈ {0,2}`,
  learning rate 0.04, deterministic seed 2028.
- Trong mỗi outer fold, từng family/config được chọn bằng bốn inner folds. Chỉ config có Recall@64 ≥0,985 ở mọi inner fold mới eligible; sau đó xếp theo mean Recall@5, worst-inner Recall@5, mean Precision@5, mean Recall@64 và deterministic config order.
- Fit family winner trên toàn outer-train rồi score outer-heldout. Chín non-evaluable queries không tham gia fit/metric nhưng vẫn nhận deterministic ranking để artifact đủ 7.000 query.

## Acceptance và phân tích

- LambdaMART canonical nested-OOF trong chính EXP-028b là anchor; không dùng trực tiếp score retained-gold EXP-028 làm baseline promotion.
- Một family mới chỉ được promote nếu:
  - aggregate Recall@5 tăng ít nhất 0,002;
  - aggregate Precision@5 không giảm;
  - Recall@64 aggregate không thấp hơn LambdaMART và mọi outer fold ≥0,985;
  - ít nhất 4/5 folds có Recall@5 delta không âm;
  - worst-fold Recall@5 delta ≥−0,005.
- Nếu nhiều family pass, chọn Recall@5 cao nhất, rồi Precision@5, worst-fold delta, Recall@64 và CPU scoring latency. Nếu không family nào pass, giữ LambdaMART.
- Báo thêm Recall@16/32/50/64/100/150, paired bootstrap 10.000 mẫu, rank rescue/harm, coefficients/importance, latency, convergence, feature missingness và slices theo gold-count, source availability, rank region và document metadata.
- Mức Recall@5≈0,95 của pipeline bên ngoài chỉ là giả thuyết chưa xác minh; không dùng làm gate cho đến khi có candidate pool, split, labels và feature protocol tương ứng.
- Feature-block ablation sau benchmark chỉ là diagnostic; không dùng lại outer results để promote một feature subset.

## Artifacts, tests và handoff

- Persist fingerprinted `screen_matrix.jsonl`, per-family OOF predictions/scores, selections, metrics, coefficients/importances, error slices, manifests, `_SUCCESS.json` và `REPORT.json`.
- Tests bắt buộc: canonical label counts/fingerprint, 150 unique ordered parent IDs, 24 exact columns, fold-local scaler/class weights/model selection, exclusion of nine empty-label groups, deterministic pair sampling/ties, complete grids, K64 guards, parent-ID outputs và reproducibility.
- Chạy regressions EXP-021/027/028/030 cùng EXP-028b fixtures trước nested OOF.
- Runtime dự kiến sau khi EXP-033 dừng/hoàn tất: audit/tests vài phút; nested CPU khoảng 3–5 giờ. Log phải gọn, theo outer fold, có status/ETA; sửa warning trước long run.
- Nếu promote model mới, lưu OOF ranking để reranker stage tái dựng nested cascade sau này; vẫn hoãn single full-train pre-ranker và test capsules đến inference preparation.
- Sau report, cập nhật `AGENTS.md` chỉ với quyết định bền vững; handover trở lại EXP-033/heavy-reranker work với pre-ranker winner hoặc LambdaMART fallback đã được ghi rõ.
