# EXP-108 runbook

## Start or resume

```powershell
& ..\dsc_env\Scripts\python.exe -u src\exp108_atomic_condition_reranker.py overnight --resume --run-id <run-id>
```

The orchestrator reuses only stages with a report and `_SUCCESS.json`. A failed,
rejected, or interrupted stage is rebuilt or resumed from its owned checkpoint.

## Status

```powershell
& ..\dsc_env\Scripts\python.exe src\exp108_atomic_condition_reranker.py status
Get-Content results\exp108_atomic_condition_reranker\RUN_STATUS.json
Get-Content results\exp108_atomic_condition_reranker\logs\<run-id>\overnight.log -Tail 30
```

Detailed output is stored in `results/exp108_atomic_condition_reranker/logs`.
The terminal receives only phase transitions and periodic progress/ETA lines.

## Exit semantics

- `0`: completed or stopped at a non-rejected result.
- `1`: implementation/runtime failure; inspect `overnight.log`.
- `2`: an experimental gate rejected continuation.
- `130`: interrupted; training checkpoints are resumable.

No model or dependency download is permitted. EXP-108 never modifies EXP-022,
EXP-106, EXP-106b, EXP-107, or structural-v3 artifacts.
