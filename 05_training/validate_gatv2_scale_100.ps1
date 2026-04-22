param(
    [string]$DbUrl = "postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus",
    [int[]]$SnapshotCounts = @(10, 100),
    [switch]$DryRun = $true
)

$ErrorActionPreference = "Stop"

$runner = Join-Path $PSScriptRoot "run_build_dataset.ps1"
if (-not (Test-Path $runner)) {
    $runner = ".\run_build_dataset.ps1"
}
if (-not (Test-Path $runner)) {
    throw "run_build_dataset.ps1 not found. Place this script in 05_training or run it from that folder."
}

Write-Host "============================================================="
Write-Host " validate_gatv2_scale_100"
Write-Host "   db_url         = $DbUrl"
Write-Host "   snapshot_sets  = $($SnapshotCounts -join ', ')"
Write-Host "   dry_run        = $DryRun"
Write-Host "============================================================="

$results = @()

foreach ($count in $SnapshotCounts) {
    Write-Host ""
    Write-Host "[RUN] snapshots=$count"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $argsList = @(
        '--db-url', $DbUrl,
        '--max-snapshots', "$count"
    )
    if ($DryRun) { $argsList += '--dry-run' }

    & $runner @argsList
    $exitCode = $LASTEXITCODE
    $sw.Stop()

    $results += [pscustomobject]@{
        snapshots   = $count
        elapsed_sec = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        exit_code   = $exitCode
        at          = Get-Date
    }
}

Write-Host ""
Write-Host "==================== summary ===================="
$results | Format-Table -AutoSize

$csvPath = Join-Path (Get-Location) ("validate_gatv2_scale_100_{0}.csv" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
Write-Host ""
Write-Host "[CSV] $csvPath"
