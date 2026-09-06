param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = 'Stop'
$base = 'D:\Study\DSC2026\LegalIR'
$python = 'D:\Study\DSC2026\dsc_env\Scripts\python.exe'
$orchestratorLog = Join-Path $base 'results\exp111_multiview_sparse\logs\overnight-inner-orchestrator.log'

function Write-StageLog([string]$message) {
    $line = "[$(Get-Date -Format o)] $message"
    $line | Tee-Object -FilePath $orchestratorLog -Append
}

function Invoke-Stage([string[]]$stageArgs) {
    Write-StageLog ("START " + ($stageArgs -join ' '))
    & $python -u (Join-Path $base 'src\exp111_multiview_sparse_retrieval.py') @stageArgs
    if ($LASTEXITCODE -ne 0) {
        Write-StageLog ("STOP exit_code=$LASTEXITCODE stage=" + ($stageArgs -join ' '))
        exit $LASTEXITCODE
    }
    Write-StageLog ("COMPLETE " + ($stageArgs -join ' '))
}

if ($WaitForPid -gt 0) {
    Write-StageLog "WAIT lexical_audit_pid=$WaitForPid"
    try { Wait-Process -Id $WaitForPid -ErrorAction Stop } catch { Write-StageLog "lexical audit PID is already absent; checking emitted gate" }
    $lexical = Join-Path $base 'results\exp111_multiview_sparse\LEXICAL_FAILURE_AUDIT.json'
    if (-not (Test-Path -LiteralPath $lexical)) { Write-StageLog 'STOP lexical audit did not emit artifact'; exit 2 }
}

Set-Location $base
Invoke-Stage @('build-index','--view','v1_surface_structural','--resume')
Invoke-Stage @('build-index','--view','v2_windows','--resume')
Invoke-Stage @('bounded-screen','--resume')
Invoke-Stage @('overnight-inner','--resume')
Write-StageLog 'COMPLETE overnight-inner fold0_called=false'
