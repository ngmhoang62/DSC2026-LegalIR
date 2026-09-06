$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPy = 'D:\Study\DSC2026\dsc_env\Scripts\python.exe'
$env:PYTHONPATH = 'src'
Set-Location $taskRoot

function Invoke-Exp109CStage([string]$stage, [string[]]$stageArgs = @()) {
    Write-Host ("[{0:O}] exp109c stage={1} starting" -f [DateTime]::UtcNow, $stage)
    & $taskPy -u 'src/exp109c_latent_condition_late_interaction.py' $stage '--outer' 'fold_0' @stageArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host ("[{0:O}] exp109c stage={1} complete" -f [DateTime]::UtcNow, $stage)
}

# D100 is a new contract. One receipt-verified cache is expanded from 1024 to
# 2048 to full by query ID; no D200 shard can be read by this contract.
Invoke-Exp109CStage 'candidate-ceiling' @('--authorize')
Invoke-Exp109CStage 'encode-queries' @('--resume', '--authorize', '--device', 'cuda')
$benchmarkPath = Join-Path $taskRoot 'results\exp109c_latent_condition_late_interaction\BATCHED_SCORER_BENCHMARK.json'
$d100Contract = 'd100_batched_chunk_maxsim_masked_fp32_v1'
$reuseBenchmark = $false
if (Test-Path -LiteralPath $benchmarkPath) {
    $benchmark = Get-Content -Raw -LiteralPath $benchmarkPath | ConvertFrom-Json
    $reuseBenchmark = $benchmark.status -eq 'PASS_BATCHED_SCORER_PROMOTION' -and $benchmark.scorer_implementation_contract -eq $d100Contract
}
if ($reuseBenchmark) { Write-Host ("[{0:O}] exp109c D100 benchmark already promoted" -f [DateTime]::UtcNow) }
else { Invoke-Exp109CStage 'benchmark-batched-scorer' @('--authorize', '--device', 'cuda') }
# The 1,024 pilot is immutable.  A corrected, cached 2,048 verdict must not
# be replayed after full-extension receipts exist: those receipts are outside
# the pilot selection by design.  Only evaluate the pilot when no valid
# corrected report is present.
$futilityPath = Join-Path $taskRoot 'results\exp109c_latent_condition_late_interaction\D100_FUTILITY_2048_SCREEN.json'
$reuseFutility = $false
if (Test-Path -LiteralPath $futilityPath) {
    $futility = Get-Content -Raw -LiteralPath $futilityPath | ConvertFrom-Json
    $reuseFutility = $futility.futility.rejection_policy -eq 'all_reject_checks_required' -and $futility.status -in @('AMBIGUOUS_D100_FUTILITY_2048', 'CONTINUE_D100_FUTILITY_2048')
}
if ($reuseFutility) {
    Write-Host ("[{0:O}] exp109c corrected D100 n=2048 futility verdict reused; resuming full cache directly" -f [DateTime]::UtcNow)
} else {
    Invoke-Exp109CStage 'score-inner' @('--resume', '--authorize', '--device', 'cuda', '--query-count', '2048')
    Invoke-Exp109CStage 'd100-futility-screen' @('--query-count', '2048')
    $futility = Get-Content -Raw -LiteralPath $futilityPath | ConvertFrom-Json
    if ($futility.status -like 'REJECTED_D100_FUTILITY_*') {
        Write-Host ("[{0:O}] exp109c D100 futility rejected at n=2048; stopping before full budget" -f [DateTime]::UtcNow)
        exit 0
    }
}
Invoke-Exp109CStage 'score-inner' @('--resume', '--authorize', '--device', 'cuda')
Invoke-Exp109CStage 'frozen-inner-screen'
exit 0
