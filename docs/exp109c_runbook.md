# EXP-109C runbook

EXP-109C là research branch riêng cho latent-condition late interaction. Các
artifact của EXP-013/109A/109B chỉ được đọc làm reference; không bị ghi đè.

Working directory:

```powershell
Set-Location D:\Study\DSC2026\LegalIR
$env:PYTHONPATH = 'src'
$py = 'D:\Study\DSC2026\dsc_env\Scripts\python.exe'
```

## Cheap verification

```powershell
& $py -m py_compile src/exp109c_latent_condition_late_interaction.py
& $py -m pytest tests/test_exp109c_latent_condition_late_interaction.py -q
& $py src/exp109c_latent_condition_late_interaction.py audit
& $py src/exp109c_latent_condition_late_interaction.py reproduce-exp013
& $py src/exp109c_latent_condition_late_interaction.py status
```

`reproduce-exp013` deterministically generates and freezes 20 real pairs from
the EXP-013 query and leaf stores. It records qid/doc/chunk IDs, shapes, FP32
and int8 reference scores, selection ranks and a fixture hash. This is a
reproduction artifact, not new annotation; do not edit it by hand.

Each pair must contain frozen `query_vectors_64`, `document_vectors_64`, and
`expected_maxsim`; `expected_int8_maxsim` is optional but recommended. The
reproduction gate recomputes those values and records the fixture hash.

`audit` không download model và không load GPU. Snapshot Jina bắt buộc phải có
sẵn tại local cache, đúng snapshot `4552c4dc1ffd7d7a635b6a41a1077fe9c9cdd974`
và projection `[128,1024]`.

Before any real Jina scoring or training, record competition/model-license
compatibility in `docs/exp109c_model_use_eligibility.json`. The file must
contain `competition_terms_source`, `model_license_source`, and
`approved_for_competition_use: true`; otherwise the implementation blocks the
real GPU stages.

## Gated stages

Mỗi stage đắt yêu cầu cả `--authorize` hoặc environment variable tương ứng và
success marker của stage trước. Không bỏ qua gate bằng cách đổi depth, giảm
dimension, hoặc fallback CPU.

```powershell
& $py src/exp109c_latent_condition_late_interaction.py candidate-ceiling --outer fold_0
& $py src/exp109c_latent_condition_late_interaction.py preflight --authorize
& $py src/exp109c_latent_condition_late_interaction.py fidelity-pilot --outer fold_0 --resume --authorize
& $py src/exp109c_latent_condition_late_interaction.py encode-corpus --resume --authorize
& $py src/exp109c_latent_condition_late_interaction.py encode-queries --resume --authorize
& $py src/exp109c_latent_condition_late_interaction.py score-inner --outer fold_0 --resume --authorize
& $py src/exp109c_latent_condition_late_interaction.py frozen-inner-screen --outer fold_0
& $py src/exp109c_latent_condition_late_interaction.py train-metric-adapter --outer fold_0 --resume --authorize
& $py src/exp109c_latent_condition_late_interaction.py final-inner-gate --outer fold_0
```

The fidelity stage generates its own 256-pair GPU fixture from the verified
current candidate pool, stratified deterministically over parent length with
positive and high-ranked non-gold rows. It freezes IDs, sampling policy,
fingerprints, full-token FP32 scores and compressed 96/128-anchor scores; the
128 attempt is consumed exactly once only after a 96-anchor failure.

## Adapter implementation boundary

The adapter evaluator validates fold-isolated candidate-pool predictions and
will fail closed unless `adapter_scores` have been materialized for every
candidate. The current background pipeline therefore stops after the frozen
inner screen; it must not claim an adapter PASS, Fold-0, or OOF result until
the adapter-score cache producer is present and has been independently tested.

## Fold 0 and OOF barriers

`locked-fold0` requires a passing final inner gate plus explicit authorization;
`nested-oof` additionally requires a strong, reviewed Fold-0 artifact and a
separate authorization. Passing implementation tests is not authorization.

```powershell
& $py src/exp109c_latent_condition_late_interaction.py locked-fold0 --outer fold_0 --resume --authorize
& $py src/exp109c_latent_condition_late_interaction.py nested-oof --resume --authorize
```

Current implementation stops before corpus encoding, exact scoring, Fold 0 and
full OOF. See `results/exp109c_latent_condition_late_interaction/IMPLEMENTATION_REPORT.md`.

## Frozen-winner Fold-0 track

After `PASS_FROZEN_LATE_INTERACTION_GATE`, the frozen-winner track is the only
authorized path that can evaluate Fold 0 without adapter training.  It never
writes `FINAL_INNER_GATE.json`, `METRIC_ADAPTER_REPORT.json`, a full OOF
artifact, or a public submission.

Its candidate contract is immutable:

```text
VietLegal-E5 top 100 + VnLegal-LAL top 100 + BM25 top 100
-> unique_parent_union
-> no post-union truncation
```

The final pool is therefore variable-sized (the observed F1--F4 pool was
min=123, mean=188.6004, max=266).  `Recall@100` is a ranking cutoff, not a
statement that the candidate pool contains only 100 parents.  Validation must
recompute the D100/source union for every query, reject duplicates or parents
outside that union, and record per-query candidate hashes/counts.
