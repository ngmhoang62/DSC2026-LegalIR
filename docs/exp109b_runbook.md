# EXP-109B runbook

EXP-109B is an isolated encoder-complementarity experiment. It uses only
VietLegal-E5, VietLegal-Harrier-0.6B, VnLegal-LAL, the frozen structural
`retrieval_text`, and the existing tuned EXP-021 BM25 evidence. It does not
modify earlier experiment namespaces, download models, regenerate the corpus,
use `document_label`, or submit publicly.

Run commands from `D:\Study\DSC2026\LegalIR` with the project environment:

```powershell
$env:PYTHONPATH='src'
$py='D:\Study\DSC2026\dsc_env\Scripts\python.exe'
& $py -u src\exp109b_encoder_complementarity.py audit
& $py -u src\exp109b_encoder_complementarity.py replay
& $py -u src\exp109b_encoder_complementarity.py smoke
& $py -u -m pytest tests\test_exp109b_encoder_complementarity.py -q
```

The audit writes
`results\exp109b_encoder_complementarity\input_audit\READING_AUDIT.json`.
It records the complete reading order, SHA-256 values, schemas, frozen
counts, model snapshots, disk space, and plan/checkout mismatches. A missing
or invalid live corpus/label/model contract produces
`REJECTED_INPUT_AUDIT`; old EXP-035/037 metadata drift is retained as an
explicit mismatch and is never silently treated as an EXP-109B artifact.

`replay` checks archived EXP-015 metrics for all three allowed models at
`1e-6`, verifies the shared FP16-to-FP32/top2 scorer against an independent
NumPy implementation, and records the EXP-109A real-parent parity evidence.
Fresh model encoding is intentionally opt-in:

```powershell
& $py -u src\exp109b_encoder_complementarity.py replay --fresh-models
```

The following phases can allocate GPU memory or encode the corpus. They are
blocked unless the corresponding explicit environment variable is set (or the
hidden `--authorize` switch is used by the operator after granting permission):

```powershell
$env:EXP109B_ALLOW_BOUNDED_GPU='1'
& $py -u src\exp109b_encoder_complementarity.py bounded-screen --outer fold_0 --resume --authorize

$env:EXP109B_ALLOW_PREFLIGHT_GPU='1'
& $py -u src\exp109b_encoder_complementarity.py preflight --outer fold_0 --resume --authorize

$env:EXP109B_ALLOW_FULL_ENCODING='1'
& $py -u src\exp109b_encoder_complementarity.py encode-selected --outer fold_0 --model <selected_model> --resume --authorize

$env:EXP109B_ALLOW_SOURCE_AUDIT='1'
& $py -u src\exp109b_encoder_complementarity.py source-audit --outer fold_0 --resume --authorize

$env:EXP109B_ALLOW_FOLD0='1'
& $py -u src\exp109b_encoder_complementarity.py fold0-screen --resume --authorize
```

The bounded screen is an oracle/separability diagnostic over
`EXP-036 top-96 ∪ canonical gold`, with at most eight structural chunks per
parent. Every report carries all three warning flags:

```json
{
  "bounded_oracle_fixture_not_recall": true,
  "gold_force_included": true,
  "may_not_be_reported_as_full_corpus_recall": true
}
```

Only a model that passes every bounded condition can be selected for full
encoding. If both models pass, both are encoded only when their rescue sets
are complementary under the plan's exclusive-rescue rule. Otherwise the
single model is selected by the locked CI/rescue/source-addition/control/time
tie order. If neither passes, the stage is
`REJECTED_BOUNDED_COMPLEMENTARITY` and no BGE branch is started.

Full encoding is resumable by shard. Shards are reusable only when model,
input, code, scorer, and contract fingerprints match; corrupt or partial
shards are rebuilt by the owning encoding stage. Full-corpus source audit ranks every parent with
`top2_mean`, then tunes candidate depth and weighted RRF on outer-train folds.
It never constructs an EXP-022 ordered append-union.

`overnight-fold0` may be used as a guarded wrapper after the required stages;
it stops at the Fold-0 report. It must not launch full OOF. Full OOF requires
both a passing Fold-0 ambitious gate and a separate explicit authorization:

```powershell
$env:EXP109B_ALLOW_FULL_OOF='1'
& $py -u src\exp109b_encoder_complementarity.py nested-oof --resume --authorize
```

The OOF entrypoint is a complete, guarded nested driver: it prepares each
outer fold only after explicit authorization, reuses only current artifacts,
re-runs fold-isolated RRF/LambdaMART selection, aggregates the five held-out
folds, and writes a success marker only when the aggregate OOF gate passes.
It remains fail-closed if the Fold-0 marker or input fingerprint is stale. It
never creates a public submission. Inspect
`results\exp109b_encoder_complementarity\RUN_STATUS.json` and the stage
reports before interpreting any result.
