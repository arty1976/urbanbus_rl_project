param(
    [string]$DbUrl = 'postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus',
    [int[]]$SnapshotCounts = @(2, 10),
    [int]$WarmRepeats = 2
)

$ErrorActionPreference = 'Stop'

$runner = Join-Path $PSScriptRoot 'run_build_dataset.ps1'
if (-not (Test-Path $runner)) {
    $runner = '.\run_build_dataset.ps1'
}
if (-not (Test-Path $runner)) {
    throw "run_build_dataset.ps1 not found. Put this script in the same folder as the runner or run it from 05_training."
}

function Invoke-Run([int]$n) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $runner --db-url $DbUrl --max-snapshots $n --dry-run
    $exitCode = $LASTEXITCODE
    $sw.Stop()
    [pscustomobject]@{
        snapshots = $n
        elapsed_sec = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        exit_code = $exitCode
        at = (Get-Date)
    }
}

$results = @()

Write-Host "============================================================="
Write-Host " benchmark_gatv2_cold_warm"
Write-Host "   db_url        = $DbUrl"
Write-Host "   snapshot_sets = $($SnapshotCounts -join ', ')"
Write-Host "   warm_repeats  = $WarmRepeats"
Write-Host "============================================================="

foreach ($n in $SnapshotCounts) {
    Write-Host "`n[COLD] snapshots=$n"
    $results += Invoke-Run $n
    for ($i = 1; $i -le $WarmRepeats; $i++) {
        Write-Host "`n[WARM $i/$WarmRepeats] snapshots=$n"
        $results += Invoke-Run $n
    }
}

Write-Host "`n==================== summary ===================="
$results | Format-Table -AutoSize

$outCsv = Join-Path (Get-Location) ('benchmark_gatv2_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.csv')
$results | Export-Csv -Path $outCsv -NoTypeInformation -Encoding UTF8
Write-Host "[CSV] $outCsv"
