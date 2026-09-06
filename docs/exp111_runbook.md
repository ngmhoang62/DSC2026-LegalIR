# EXP-111 runbook

All commands below use `D:\Study\DSC2026\dsc_env\Scripts\python.exe` from
`D:\Study\DSC2026\LegalIR`. Pre-lock stages deserialize labels only for Folds
1--4. They never run the Fold-0 command.

```powershell
$py = 'D:\Study\DSC2026\dsc_env\Scripts\python.exe'
& $py -m unittest tests/test_exp111_multiview_sparse_retrieval.py
& $py -u src/exp111_multiview_sparse_retrieval.py audit --resume
& $py -u src/exp111_multiview_sparse_retrieval.py reproduce --resume
& $py -u src/exp111_multiview_sparse_retrieval.py lexical-audit --resume
& $py -u src/exp111_multiview_sparse_retrieval.py build-index --view v1_surface_structural --resume
& $py -u src/exp111_multiview_sparse_retrieval.py build-index --view v2_windows --resume
& $py -u src/exp111_multiview_sparse_retrieval.py bounded-screen --resume
& $py -u src/exp111_multiview_sparse_retrieval.py full-source-audit --resume
& $py -u src/exp111_multiview_sparse_retrieval.py frozen-inner-sparse --resume
& $py -u src/exp111_multiview_sparse_retrieval.py dense-complement --resume
```

The V2 command creates V2, V3 and V5 in one streamed SQLite build. It writes a
manifest and `_SUCCESS.json` only after the complete transaction is committed.
An interrupted build removes its unfinished database on the next run, rather
than trusting a partial corpus.

`full-source-audit` persists verified 64-query source-score shards; rerunning
it resumes only shards whose index/code/config fingerprint and SHA-256 agree.
`frozen-inner-sparse` cross-fits family RRF and the fixed EXP-109B LightGBM
family over F1--F4. `dense-complement` first proves the exact EXP-109B anchor;
it rejects rather than improvising if the required immutable per-source E5/LAL
cache is absent. `overnight-inner` stops on its first failed gate and never
calls Fold 0.

`scripts/run_exp111_overnight_inner.ps1` is available for a future approved
dispatch only. The former lexical-audit and orchestrator processes were stopped
after the scorer-contract review; do not start this script until the listed
lexical-audit and gate corrections have passed.

`fold0` additionally requires both `--authorize-fold0` and
`EXP111_ALLOW_FOLD0=1`; it is intentionally not an overnight target.
