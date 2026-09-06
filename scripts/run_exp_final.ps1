param([double]$BudgetHours = 48, [switch]$Foreground)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path (Split-Path -Parent $project) 'dsc_env\Scripts\python.exe'
$entry = Join-Path $project 'src\exp_final_retrieval.py'
$logRoot = Join-Path $project 'results\exp_final_retrieval\logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$env:PYTHONPATH = Join-Path $project 'src'
$arguments = @('-u', $entry, 'run-all', '--resume', '--budget-hours', [string]$BudgetHours)
if ($Foreground) { & $python @arguments; exit $LASTEXITCODE }
$worker = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'launcher.stdout.log') -RedirectStandardError (Join-Path $logRoot 'launcher.stderr.log')
Write-Output "EXP-final supervisor PID=$($worker.Id); verify child progress in $logRoot"
