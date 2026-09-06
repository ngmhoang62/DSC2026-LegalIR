# EXP-112 — Task-Adaptive, Recall-First Legal Retrieval

This English implementation specification supersedes the previous Vietnamese plan.
Authority: the complete user-approved EXP-112 specification in the implementation task.
Workspace: D:\Study\DSC2026\LegalIR
Interpreter: D:\Study\DSC2026\dsc_env\Scripts\python.exe

## 1. Binding objective

Train, evaluate all five outer folds, fit a deployment system, and export a verified
public submission on the local RTX 4050 6GB. Recall 0.98+ is a research ambition,
not a promise. No performance gate may terminate the experiment. Unhelpful
components lose calibration selection; the frozen/no-op path remains eligible.
Correctness, leakage, corruption, invalid scoring, and resource failures require
repair, never fabricated success.

Budget 48 hours, allowance up to 50 hours. A measured forecast beyond 50 hours
requires an extension decision before dispatch. No Colab/Kaggle, automatic model
downloads, additional datasets, LLM augmentation, manual evidence labels, or
leaderboard upload. EXP-110P remains cancelled. Preserve historical artifacts.
Creating code is not completion: COMPLETE_SUBMISSION requires five-fold results,
final fit, actual JSON/ZIP files, and successful reopen validation.

## 2. Evidence and mandatory reading

Read AGENTS.md, scoring.py, train/public JSON, cv_folds.json, final_preprocessed_v2
manifest/exclusions/train_label_impact, structural source schemas, embedding
manifests and pinned model configurations. Record actual paths, SHA256 hashes,
functions inspected, findings and reuse decisions in READING_AUDIT.json before code.

Read historical summaries broadly and reused implementations deeply:
EXP-012b/013b/014 retrieval, evidence, OOF and fusion; EXP-015 model screen;
EXP-016–021 preprocessing/sparse tuning; optimization registry EXP-022–035;
EXP-101/102 projection/MIL; EXP-104/105 ML benchmarks; EXP-106/107 normalization;
EXP-108 evidence/preflight; EXP-109A projection/loss/pilot; EXP-109B features/fusion;
EXP-109C tiers/refinement/dependencies; EXP-111 streaming caches and latest status;
teammate repro README and sparse/fusion/submission code, tracing label dependencies.

Historical scope:
- EXP-013b OOF .930505 and EXP-014 .926862 use older contracts, not reproduction targets.
- EXP-015 was a small 128-query screen, not a universal encoder ranking.
- EXP-104 LR superiority on older features does not prove universal superiority.
- EXP-109A failed a frozen-embedding residual projection hypothesis, not backbone LoRA.
- EXP-109B inner .925106, Fold0 .933178.
- EXP-109C inner .933390, Fold0 .932523, matched-anchor gain .00250.
- EXP-111 source-only F1–F4: V0 .836721, surface .707244, window384 .634668,
  window512 .581417, full-parent .188465, bigram .688971, trigram .717832.
  Union top50/100/200 per source covers .980827/.987618/.991343; not achieved R@5.
  Recheck unfinished frozen_inner_sparse/dense_complement paths rather than invoking blindly.

Do not attribute prior CE failure to truncation alone. Do not call teammate-reported
supervised/local results independently verified label-free public BM25 scores.
Going from .933 to .98 removes about 70% of remaining error; overlapping gains
cannot be added arithmetically.

Research motivation, not imported gain guarantees:
- ADORE/STAR: https://arxiv.org/abs/2104.08051
- LTRe: https://arxiv.org/abs/2010.10469
- RocketQA: https://aclanthology.org/2021.naacl-main.466/
- Native E5 contract: https://huggingface.co/mainguyen9/vietlegal-e5

## 3. Data and evaluation

Revalidate 8507 parents, 343347 chunks, 7000 queries, 6991 canonical evaluable and
nine canonical non-evaluable. Keep original labels for official metrics and existing
canonical mappings for training/historical metrics. Predict every query.

Original-label audit: 20 empty and five duplicate documents excluded; 13 affected
queries; two duplicate-ID and eleven empty-passage gold occurrences. Retained-ID
original-label oracle approximately .998286. Do not invent alias expansion.

Official recall is primary, official precision uses actual returned count. Select
by unrounded official recall, official precision, canonical multi-gold recall,
canonical recall, MRR@5, then simpler/lower-cost recipe. Never select by F1/F2.
Non-evaluable exclusions must not create NaN oracles/bootstrap.

Calibration mapping: F0→F4, F1→F0, F2→F1, F3→F2, F4→F3. Inner training uses the
other three folds. Train inner components, score calibration without its labels,
select whole recipe, write immutable lock, refit on four outer-training folds,
score outer without labels, lock predictions, then evaluate. Continue all folds.
This is development CV with within-run isolation, not historically pristine folds.

Prevent indirect stacking leakage: audit candidate selection, source tuning,
normalization, refinement, teacher scores and evidence. Train ML on frozen-source
features only; adapted query/CE scores are combined afterward, never supplied
in-sample as ML training features.

## 4. Retrieval and frozen feature combination

Required frozen sources: VietLegal-E5, VnLegal-LAL, EXP-021 BM25 V0.
Optional evidence-backed specialist: EXP-111 trigram. Reuse verified v2 shards,
not eight rebuilt indexes. Lock the EXP-111 snapshot before CV.
V0 preserves exact depth1024, parent fusion constant32, head16 implementation.
E5 uses native prefixes, masked mean pooling, normalization, max512.
LAL uses native last-token pooling and query instruction.

C_base = E5@100 ∪ LAL@100 ∪ V0@100.
C_frozen = C_base ∪ trigram@100.
C_adapted = C_frozen ∪ adapted-E5@100.
These are variable-size unique source unions, never 100 total. No evaluation gold
injection. Training gold injection is explicit. Save actual pool sizes and coverage.
Top-five oracle averages min(5, gold intersect pool)/gold_count; it is diagnostic.

Jina, when eligible, uses uniform approximate Config-E medoid features only, scored
on C_base. Old exact refinement depended on a learned anchor. Repair all overwritten
rows with approximate scoring, recompute pool-relative ranks, discard anchor logits
and exact-tier signals. Never retain label-dependent missingness. Verify model/query/
index/scoring fingerprints; no corpus re-encoding. Outside C_base set jina_scored=0.
Disable globally before CV if provenance or measured cost is unacceptable.

At most three blocks:
B0 scalar E5/LAL/BM25 plus label-free metadata;
B1 adds trigram score/rank/normalization/agreement;
B2 adds sanitized Jina.
Preserve actual scores separately from ranks. Dense normalization context top500;
exact-score dense candidates outside stored500, rank censored, never invented.
Sparse unavailable values have explicit indicators. No qid/docid/label frequencies.

For each block compare only:
LambdaMART lambdarank, eval_at5, leaves7, minleaf50, lr.05, rounds300,
feature_fraction1, bagging_fraction1, deterministic, seed112.
LR training-only StandardScaler, C1, balanced, liblinear, max_iter2000, seed112.
No broad model/grid search. Retain equal-weight three-source RRF32 fallback.

## 5. Query-backbone task adaptation

E5 attention query/value LoRA r16 alpha32 dropout.05 throughout backbone.
Record module names, trainable count and actual gradients. Document bank frozen.
Read FP16 bank into FP32 and renormalize. Parent score is mean of highest two
chunk cosines; singleton uses one. No aggregation redesign.

Per microbatch: encode query with gradients; full-corpus GEMM and vectorized top2
parent reduction under no-grad for mining; gather selected top2 chunk vectors;
recompute selected scores with attached query; backprop only through query.
No per-parent GPU calls, no detached learning query, no document optimizer state.

64 unique negatives/query: 16 current hardest, 16 rotated adapted ranks17–100,
16 sparse/LAL disagreement, eight eligible Jina confusers or dense hard negatives,
eight random. Exclude all positives/aliases, deterministic dedup/backfill/rotation.
Use every positive and normalize training weight per query.

tau=.05; independent-positive loss:
mean_p [ log(exp(s_p/tau)+sum_n exp(s_n/tau)) - s_p/tau ].
Other golds do not compete in the denominator.
Drift = 1 - cosine(adapted_query, frozen_query).
Epoch1: Lmulti + .05 drift.
Epoch2: add .25 boundary loss, averaging softplus((s_n-s_p)/tau) over negatives
crossing the GLOBAL full-corpus top5 boundary with each positive, then over positives.
No crossing pair gives zero. Do not double-normalize or average inactive pairs.

AdamW lr5e-5, wd.01, clip1, effective batch16, microbatch4/2/1 by preflight,
maximum two epochs, warmup.1, cosine scheduler, seed112.
FP32 first, checkpointing; FP16/scaler only after measured failure and verified parity.
No silent quantization. Save both epochs; no-adaptation remains eligible.
An epoch1 winner retains the nominal two-epoch scheduler horizon on refit.

Fuse ML and adapted ranks:
(1-beta)/(32+rML) + beta/(32+rAdapted), beta={0,.15,.30,.50,1}.
Missing contribution zero. Compare original frozen anchor, expanded ML beta0,
adapted alone and fixed weights/checkpoints jointly on calibration.
Expanded beta0 is not necessarily the original anchor.

## 6. Optional evidence and CE

One deterministic source-exact package per query/parent. Frozen E5 plus lexical
RRF32 chooses first structural chunk; second disjoint unit only adds an uncovered
high-confidence lexical/numeric condition within budget. Keep governing headings,
real metadata, never slug reconstruction. Evidence precedes low-priority metadata.
Pair including specials <=512; query <=192; longer query explicit head128/tail64
with marker cost adjustment. Metadata <=48; evidence gets remaining tokens.
Long units use recorded source-exact windows. Verify final pair, no silent truncation.
Log clipping, partial units and omitted context. Evidence proxies are diagnostics,
not performance stop gates.

BGE-reranker-v2-m3 absolute logits, not residual deltas.
LoRA r16/alpha32/dropout.05, lr5e-5, wd.01, one epoch, clip1.
All positives plus six negatives: two top, two rotated9–32, one disagreement,
one deep/random. Frozen training pool/evidence independent of supervised adapters.
Loss independent-positive contrastive at logit tau1 plus .1 balanced BCE
(mean positive-class and negative-class mean, query-normalized).
Preflight worst positive count+6. If grouped graph OOM, exact scalar-logit gradient
replay with restored dropout RNG; verify direct/replay gradient parity.

Correct upstream top50 only:
(1-alpha) z(upstream) + alpha z(CE), alpha={0,.1,.25}.
Constant groups z=0. Alpha0 must preserve exact order/ties; tail unchanged.
Optional confidence policy uses normalized margin rank5-rank6, calibration P70:
high confidence alpha0, otherwise selected nonzero alpha. Must beat global policy;
ties choose global. Lock numerical threshold.

## 7. Adaptive output

Always export fixed top5. Optional prefix gap thresholds infinity,6,4,3,2;
standardize using same top50 context; rank1 always retained. Finite threshold only
if calibration official recall>=.98, no fixed-five gold hit removed, precision rises.
Deployment adaptive only if pooled fixed-five official OOF>=.98, every fold loses
zero hits, all five thresholds finite; use maximum threshold (most conservative).
Otherwise recommend fixed five. No post-hoc threshold tuning on outer errors.

## 8. Runtime and profiles

Implement entire graph before long dispatch:
audit → correctness → full throughput preflight → immutable resource lock →
frozen preparation → all five folds → OOF → final lock/fit → public → export verification.
No deferred evaluator stubs.

Benchmark 128 query optimizer updates including mining/backward/checkpoints;
256 retrieval+feature queries; 64 CE groups including worst positive count;
64×50 CE inference including rendering; cold/warm sparse; Jina repair/public
if eligible; model loading/final export. Report wallclock, throughput median/p90,
RAM available/RSS, VRAM allocated/reserved and disk traffic.

Choose globally in order:
P0 two epochs + eligible Jina + inner CE and outer CE refit;
P1 two epochs + eligible Jina + inner CE reused for outer;
P2 two epochs, no Jina, inner CE reused;
P3 two epochs + eligible Jina, no CE;
P4 two epochs, no Jina/CE;
P5 one epoch, no Jina/CE.
Choice by feasibility/provenance, not recall. Forecast 1.25×workload plus recovery
reserve. Include final fits/public costs even if optional component may lose.
Five inner query fits, five outer refits, one final fit maximum.
Do not silently make folds heterogeneous or drop folds to preserve an optional stage.

One GPU worker; sequential model residency; mmap/SQLite byte-bounded caches;
one memory-heavy child per stage; >=1GiB available RAM target and plateau checks;
bounded CPU/BLAS threads. Diagnose slowdown, do not restart blindly.

## 9. Interfaces and artifacts

Entry src/exp112_task_adaptive_retrieval.py; modules src/exp112/.
CLI audit, preflight, prepare-frozen, run-fold --outer fold_i, run-all --resume
--budget-hours 48, evaluate-oof, fit-final, predict-public, verify-submission, status.
run-all must reach submission verification.

Artifacts: READING_AUDIT, INPUT_AUDIT, PREFLIGHT_REPORT, JOB_LEDGER, RESOURCE_LOCK,
FEATURE_SCHEMA, per-stage manifests/success, outer SELECTION_LOCK/prediction locks/
reports, OOF_REPORT, FINAL_CONFIG_LOCK, SUBMISSION_MANIFEST, IMPLEMENTATION_REPORT.
Include data/fold/model/tokenizer/pooling/source/candidate/features/training dependencies,
code/config hashes, artifact hashes, and synthetic/calibration/outer/public scope.
Resume optimizer/scheduler/scaler if any/epoch/query position/order/RNG/accumulation.
Atomic writes; verified hashes; never edit hashes to bypass mismatch.

Detailed flushed UTF8 stage/fold logs, heartbeat at least60s, progress/throughput/
rollingETA/RAM/VRAM. Supervisor checks exit codes/vanished workers, marks stale
RUNNING, preserves diagnostics, verifies real initial progress. Hidden Windows
background process, separate stdout/stderr. PID alone is not proof of work.

Local pinned models only. Audit use permission and parameter limit for ALL active
towers, including frozen experts; sequential residency is not an exemption.
One final refit system, not an unbudgeted five-backbone ensemble.

## 10. Verification and completion

Tests must cover official/canonical labels/metrics; invalid submissions; NaN guards;
isolation including indirect Jina dependencies; identity query encoding; real-parent
NumPy top2 parity; singleton/negative scores/ties; selected-score gradients; actual
backbone gradients and unchanged bank; all-positive negatives; global boundary
normalization; schedule consistency; feature order/missing context/LR scaling;
fusion endpoints; source offsets/pair budgets; CE replay gradients; alpha0/tails;
adaptive prefix/zero-hit-loss; resume/hash/stale worker; bounded RAM;
synthetic five-fold→final→JSON/ZIP even if scores disappoint.
Synthetic fixtures cannot be production results.

Per-fold controls: E5, frozen ML anchor, adapted alone, expanded ML, adapted fusion,
CE if enabled, fixed/adaptive. Metrics R@1/3/5/10/16/20/32/50/64/100/200 where
supported, official/canonical precision, MRR, single/multigold, coverage/oracle,
wins/losses/recovered/lost, pool sizes, runtime/memory. Gold rank buckets1–5/6–10/
11–50/51–100/outside100/outsidepool; drift/train-cal gap, exclusive rescues,
CE regressions, clipping cohorts.

OOF exactly one locked prediction for each7000 queries, all five reports,
mean/std and pooled metrics, paired bootstrap, original/canonical denominators,
matched anchor and explicitly historical comparison rows.

Final recipe is a whole evaluated selection lock, not independently modal fields.
Group identical structural recipes; most frequent, ties lower cost/simpler then
stable fold order. Preserve representative numerical parameters. Never select by
outer performance. Final all-data refit has no independent heldout measurement.

Export submission_recall_first.json, submission.json, submission.zip,
SUBMISSION_MANIFEST.json, optional adaptive diagnostic. JSON maps query ID to
{"answer":[parent IDs]}. Validate exact public qids, one–five unique retained IDs,
no chunk IDs, ZIP member submission.json per verified contest contract, reopen/hash.
No leaderboard upload.

Handoff: files; PASS/FAIL/NOT RUN checklist with evidence; latest111 reuse;110P
excluded; model/parameter audit; one outer dependency graph; measured ledger/profile/
ETA/reserve; actual worker/logs; deviations. Completion report fivefold+aggregate,
whether official .98 reached, helped/bypassed/inconclusive components, actual runtime,
submission absolute paths/hashes and no-upload confirmation.
