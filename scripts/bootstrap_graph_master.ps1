# ============================================================
# bootstrap_graph_master.ps1
# ============================================================

$DbName = "urbanbus"
$ResultFile = "bootstrap_execution_result.txt"
$DbUser = "postgres"

# 1. Discover psql.exe
$PsqlPath = "psql"
if (!(Get-Command $PsqlPath -ErrorAction SilentlyContinue)) {
    $CommonPaths = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($CommonPaths) { $PsqlPath = $CommonPaths[0].FullName }
    else { Write-Error "psql.exe not found"; exit 1 }
}

# 2. Get Password Securely
Write-Host "`n--- Integrated Graph Master Bootstrap ---" -ForegroundColor Cyan
$pass = Read-Host "Enter Password for user $DbUser" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($pass)
$env:PGPASSWORD = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

function Run-SqlGate {
    param([string]$SqlFile, [string]$PhaseName)
    Write-Host "Checking Gate: $PhaseName ... " -NoNewline
    $res = & $PsqlPath -X -U $DbUser -d $DbName -t -A -f $SqlFile 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Error "Validation failed at $PhaseName : $res"
        return $false
    }
    Write-Host " PASSED" -ForegroundColor Green
    return $true
}

function Run-SqlJob {
    param([string]$SqlFile, [string]$JobName)
    Write-Host "Running: $JobName ... " -NoNewline
    $res = & $PsqlPath -X -U $DbUser -d $DbName -f $SqlFile 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Host " ERROR" -ForegroundColor Red
        $res | Out-File -FilePath $ResultFile -Append -Encoding utf8
        return $false
    }
    Write-Host " DONE" -ForegroundColor Green
    return $true
}

try {
    "=== Bootstrap Started ($(Get-Date)) ===" | Out-File -FilePath $ResultFile -Encoding utf8

    # --- Phase 0: Base Schema ---
    if (!(Run-SqlJob "02_ingest_jobs/create_fact_stop_usage_hourly.sql" "Phase 0 - Fact DDL")) { exit 1 }

    # --- Phase 1: Master Schema ---
    if (!(Run-SqlJob "02_ingest_jobs/create_graph_node_master.sql" "Phase 1 - Node DDL")) { exit 1 }
    if (!(Run-SqlJob "02_ingest_jobs/create_graph_edge_master.sql" "Phase 1 - Edge DDL")) { exit 1 }
    if (!(Run-SqlJob "02_ingest_jobs/create_stop_link_mapping_master.sql" "Phase 1 - Mapping DDL")) { exit 1 }
    if (!(Run-SqlJob "02_ingest_jobs/create_route_link_graph_edge_view_and_load.sql" "Phase 1 - Link Load")) { exit 1 }
    if (!(Run-SqlGate "03_validation_queries/graph_master_readiness.sql" "Phase 1 Gate")) { exit 1 }

    # --- Phase 2: Base Mapping ---
    if (!(Run-SqlJob "02_ingest_jobs/prepare_source_spatial_columns.sql" "Phase 2 - Spatial Prep")) { exit 1 }
    if (!(Run-SqlJob "02_ingest_jobs/load_graph_master_phase_2.sql" "Phase 2 - Stop Mapping")) { exit 1 }
    if (!(Run-SqlGate "03_validation_queries/graph_mapping_quality_check.sql" "Phase 2 Gate")) { exit 1 }

    # --- Phase 3: Refinement ---
    if (!(Run-SqlJob "02_ingest_jobs/refine_stop_link_mapping_phase_3.sql" "Phase 3 - Refinement")) { exit 1 }
    if (!(Run-SqlGate "03_validation_queries/stop_link_mapping_phase_3_quality_check.sql" "Phase 3 Gate")) { exit 1 }

    # --- Phase 4 Gate: Fact Data Check ---
    Write-Host "Checking Fact Data for Phase 4 ... " -NoNewline
    $rowCount = & $PsqlPath -X -U $DbUser -d $DbName -t -A -f "03_validation_queries/check_fact_data_presence.sql"
    if ([int]$rowCount -eq 0) {
        Write-Host " HOLD (Zero Rows)" -ForegroundColor Yellow
        "Phase 4 HOLD: No data in fact_stop_usage_hourly. Skipping load." | Out-File -FilePath $ResultFile -Append -Encoding utf8
    } else {
        Write-Host " PROCEED ($rowCount rows)" -ForegroundColor Green
        if (!(Run-SqlJob "02_ingest_jobs/create_graph_state_timeslice.sql" "Phase 4 - State DDL")) { exit 1 }
        if (!(Run-SqlJob "02_ingest_jobs/load_graph_state_timeslice_initial.sql" "Phase 4 - State Load")) { exit 1 }
        if (!(Run-SqlGate "03_validation_queries/graph_state_timeslice_readiness.sql" "Phase 4 Gate")) { exit 1 }
    }

    Write-Host "`nBootstrap Process Finished successfully." -ForegroundColor Cyan
} finally {
    $env:PGPASSWORD = $null
}
