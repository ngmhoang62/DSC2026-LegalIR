# EXP-106b — Nested K50 Protected Cross-Encoder Residual

This is the implementation record for the approved EXP-106b plan.  It keeps
the EXP-102 K=50 hypothesis, but all candidate generation, calibration and
heavy-reranker evaluation are isolated by outer fold.  It has no Qwen stage,
entity/citation boost, or public-submission stage.

The runner is `src/exp106b_nested_k50_reranker.py`.  Its stages are
`audit-inputs`, `build-nested-candidates`, `build-evidence-sidecars`,
`preflight`, `calibrate-fold`, `train-fold`, `score-fold`, `evaluate-oof`, and
`status`.  Caches and results are exclusively under `exp106b_*` namespaces.

Hard gates are: canonical 6,991/9 labels; exactly 50 unique candidate parents;
aggregate Recall@50 >= .985 and each outer fold >= .982; source-exact,
one-view <=512-token evidence; and, at OOF, Recall@5 >= .940, delta >= .010,
non-decreasing Precision@5, at least four non-negative folds, worst delta >=
-.002, and a positive paired-bootstrap lower 95% bound.  A failure writes a
`REJECTED_*` report and blocks final/public work.
