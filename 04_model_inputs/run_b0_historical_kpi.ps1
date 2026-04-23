param(
    [string]$DbUrl = "",
    [string]$SqlPath = ".\create_b0_historical_kpi.sql",
    [string]$LogRoot = ".\logs",
    [string]$PsqlPath = "",
    [switch]$NoPause
)

Set-StrictMode -Version Latest
$prevErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"

try {
    $scriptDir = $null

    if ($PSScriptRoot -and $PSScriptRoot.Trim().Length -gt 0) {
        $scriptDir = $PSScriptRoot
    }
    else {
        $scriptDir = (Get-Location).Path
    }

    if (-not [System.IO.Path]::IsPathRooted($SqlPath)) {
        $SqlPath = Join-Path $scriptDir $SqlPath
    }

    if (-not [System.IO.Path]::IsPathRooted($LogRoot)) {
        $LogRoot = Join-Path $scriptDir $LogRoot
    }

    if (-not (Test-Path -LiteralPath $SqlPath)) {
        throw "SQL file not found: $SqlPath"
    }

    if (-not (Test-Path -LiteralPath $LogRoot)) {
        New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
    }

    if (-not $PsqlPath -or $PsqlPath.Trim().Length -eq 0) {
        $cmd = Get-Command psql -ErrorAction SilentlyContinue
        if ($cmd) {
            $PsqlPath = $cmd.Source
        }
    }

    if (-not $PsqlPath -or $PsqlPath.Trim().Length -eq 0) {
        $commonRoots = @(
            "C:\Program Files\PostgreSQL",
            "C:\Program Files (x86)\PostgreSQL"
        )

        foreach ($root in $commonRoots) {
            if (Test-Path $root) {
                $found = Get-ChildItem $root -Recurse -Filter psql.exe -ErrorAction SilentlyContinue |
                    Sort-Object FullName -Descending |
                    Select-Object -First 1
                if ($found) {
                    $PsqlPath = $found.FullName
                    break
                }
            }
        }
    }

    if (-not $PsqlPath -or $PsqlPath.Trim().Length -eq 0) {
        throw "psql.exe not found. Install PostgreSQL client or pass -PsqlPath `"C:\Program Files\PostgreSQL\18\bin\psql.exe`""
    }

    if (-not (Test-Path -LiteralPath $PsqlPath)) {
        throw "psql.exe path not found: $PsqlPath"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDir = Join-Path $LogRoot ("b0_historical_kpi_" + $ts)
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    $stdoutLog = Join-Path $runDir "stdout.log"
    $stderrLog = Join-Path $runDir "stderr.log"

    Write-Host "[INFO] SQL path  : $SqlPath"
    Write-Host "[INFO] Log dir   : $runDir"
    Write-Host "[INFO] psql path : $PsqlPath"

    $psqlArgs = @(
        "-v", "ON_ERROR_STOP=1",
        "-f", $SqlPath
    )

    if ($DbUrl -and $DbUrl.Trim().Length -gt 0) {
        $psqlArgs = @(
            $DbUrl
        ) + $psqlArgs
    }

    Write-Host "[INFO] Running psql..."
    & $PsqlPath @psqlArgs 1> $stdoutLog 2> $stderrLog
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "[FAIL] psql exited with code $exitCode"
        Write-Host "[FAIL] Check logs:"
        Write-Host "       $stdoutLog"
        Write-Host "       $stderrLog"
        exit $exitCode
    }

    Write-Host "[OK] SQL completed successfully."
    Write-Host "[OK] Logs:"
    Write-Host "     $stdoutLog"
    Write-Host "     $stderrLog"    Write-Host "[INFO] Previewing result rows..."

    $previewArgs = @(
        "-t",
        "-A",
        "-F", "|",
        "-c", "SELECT COUNT(*) AS row_count, MIN(state_ts) AS min_state_ts, MAX(state_ts) AS max_state_ts, COUNT(*) FILTER (WHERE avg_wait_seconds IS NOT NULL) AS wait_filled_rows, COUNT(*) FILTER (WHERE cv_headway IS NULL) AS cv_headway_null_rows, COUNT(*) FILTER (WHERE intervention_rate = 0.0) AS zero_intervention_rows FROM public.baseline_b0_historical_kpi_by_window;"
    )

    if ($DbUrl -and $DbUrl.Trim().Length -gt 0) {
        $previewArgs = @($DbUrl) + $previewArgs
    }

    & $PsqlPath @previewArgs
    $previewExit = $LASTEXITCODE

    if ($previewExit -ne 0) {
        Write-Host "[WARN] Preview query failed with code $previewExit"
    }
    else {
        Write-Host "[OK] Preview query completed."
    }

    Write-Host "[DONE] B0 historical KPI build finished."
}
finally {
    $ErrorActionPreference = $prevErrorAction
    if (-not $NoPause) {
        Write-Host ""
        Read-Host "Press Enter to exit"
    }
}

