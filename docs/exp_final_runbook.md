# EXP-112 operating runbook

Read EXP-112_PLAN.md and IMPLEMENTATION_REPORT.md before dispatch. Do not confuse
synthetic tests or preflight with completed CV. No leaderboard upload is performed.

From D:\Study\DSC2026\LegalIR, use ..\dsc_env\Scripts\python.exe and PYTHONPATH=src.

```powershell
$env:PYTHONPATH='src'
& ..\dsc_env\Scripts\python.exe -m pytest tests/test_exp_final_retrieval.py -q
& ..\dsc_env\Scripts\python.exe -u src/exp_final_retrieval.py audit
& ..\dsc_env\Scripts\python.exe -u src/exp_final_retrieval.py preflight
& ..\dsc_env\Scripts\python.exe -u src/exp_final_retrieval.py validate
& ..\dsc_env\Scripts\python.exe -u src/exp_final_retrieval.py public-smoke
& ..\dsc_env\Scripts\python.exe -u src/exp_final_retrieval.py sparse-check
& ..\dsc_env\Scripts\python.exe -u src/exp_final_retrieval.py jina-check
& ..\dsc_env\Scripts\python.exe -u src/exp_final_retrieval.py jina-public-check
& ..\dsc_env\Scripts\python.exe -u src/exp_final_retrieval.py seal
& ..\dsc_env\Scripts\python.exe -u src/exp_final_retrieval.py run-all --resume --budget-hours 48
```

Use scripts/run_exp_final.ps1 for hidden background dispatch only after correctness
and full preflight pass. One GPU worker at a time. All stages auto-resume compatible
checkpoints; --resume is an explicit operator acknowledgement. Do not delete locks
to make changed artifacts appear compatible. A budget over 50h needs user direction.

Inspect results/exp_final_retrieval/logs and RUN_STATUS.json together with
actual process state. Warnings on stderr alone do not indicate failure. Child exit
codes are checked. A stale RUNNING file is not proof that a worker is alive.

Stages run sequentially in child processes to release memory. Weak scores do not
stop five-fold evaluation. Data, numerical and dependency failures do stop for repair.
Submission output is under results/exp_final_retrieval/public/.

The current run is already sealed. Do not rerun preflight against modified code
or overwrite prior locks. Use status and the existing reports; run-all rechecks
the sealed code/input hashes before dispatching child stages.
