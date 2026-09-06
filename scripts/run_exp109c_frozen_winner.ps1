$ErrorActionPreference = 'Stop'
$root = 'D:\Study\DSC2026\LegalIR'
$python = 'D:\Study\DSC2026\dsc_env\Scripts\python.exe'
$resultRoot = Join-Path $root 'results\exp109c_latent_condition_late_interaction'
$runId = 'frozen-winner-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$logRoot = Join-Path $resultRoot (Join-Path 'logs' $runId)
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$env:PYTHONPATH = 'src'
$env:EXP109C_ALLOW_FOLD0 = '1'

function Invoke-FrozenStage([string]$stage, [string[]]$arguments = @()) {
    $timestamp = (Get-Date).ToUniversalTime().ToString('o')
    "[$timestamp] stage=$stage state=START" | Tee-Object -FilePath (Join-Path $logRoot 'stdout.log') -Append
    # Native-library warnings go to stderr.  Redirect them directly instead
    # of piping through PowerShell: with ErrorActionPreference=Stop, a merged
    # stderr pipeline can otherwise turn a harmless LightGBM warning into a
    # terminating runner exception.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python 'src\exp109c_latent_condition_late_interaction.py' $stage @arguments 1>> (Join-Path $logRoot 'stdout.log') 2>> (Join-Path $logRoot 'stderr.log')
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) { throw "stage=$stage failed exit=$exitCode; inspect stderr.log" }
    "[$((Get-Date).ToUniversalTime().ToString('o'))] stage=$stage state=PASS" | Tee-Object -FilePath (Join-Path $logRoot 'stdout.log') -Append
}

try {
    $lockPath = Join-Path $resultRoot 'FROZEN_WINNER_LOCK.json'
    if (-not (Test-Path -LiteralPath $lockPath)) {
        Invoke-FrozenStage 'frozen-winner-lock' @('--outer', 'fold_0')
    }
    $lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
    if ($lock.status -ne 'PASS_FROZEN_WINNER_LOCK') { throw "frozen winner lock is not PASS: $($lock.status)" }
    Invoke-FrozenStage 'score-fold0-frozen' @('--outer', 'fold_0', '--resume', '--authorize', '--device', 'cuda')
    Invoke-FrozenStage 'train-final-frozen' @('--outer', 'fold_0', '--authorize')
    Invoke-FrozenStage 'evaluate-fold0-frozen' @('--outer', 'fold_0', '--authorize')
    "[$((Get-Date).ToUniversalTime().ToString('o'))] stage=frozen-winner state=COMPLETE" | Add-Content -LiteralPath (Join-Path $logRoot 'stdout.log')
} catch {
    "[$((Get-Date).ToUniversalTime().ToString('o'))] stage=frozen-winner state=FAILED error=$($_.Exception.Message)" | Add-Content -LiteralPath (Join-Path $logRoot 'stdout.log')
    exit 2
}
