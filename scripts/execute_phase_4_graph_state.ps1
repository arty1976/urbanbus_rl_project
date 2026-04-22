# ============================================================
# execute_phase_4_graph_state.ps1 (Connection Check Version)
# ============================================================

$ResultFile = "phase_4_execution_result.txt"
$SqlFiles = @(
    "02_ingest_jobs/create_graph_state_timeslice.sql",
    "02_ingest_jobs/load_graph_state_timeslice_initial.sql",
    "03_validation_queries/graph_state_timeslice_readiness.sql"
)

# 1. Discover psql.exe
$PsqlPath = "psql"
if (!(Get-Command $PsqlPath -ErrorAction SilentlyContinue)) {
    $CommonPaths = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($CommonPaths) { $PsqlPath = $CommonPaths[0].FullName }
    else { Write-Error "psql.exe not found"; exit 1 }
}

# 2. Get Password and Set Env Var
try {
    Write-Host "Please enter your PostgreSQL password." -ForegroundColor Cyan
    $SecurePass = Read-Host "Password for user postgres" -AsSecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecurePass)
    $PlainPass = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    $env:PGPASSWORD = $PlainPass
} catch {
    Write-Error "Failed to secure password input."
    exit 1
}

# 3. Pre-flight Connection Check
$DbName = "urbanbus"
Write-Host "Verifying connection to database [$DbName]..." -NoNewline
$test = & $PsqlPath -X -U postgres -d $DbName -c "SELECT 1" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Error "Connection failed. Please check if database [$DbName] exists and your password is correct. Error: $test"
    $env:PGPASSWORD = $null
    exit 1
}
Write-Host " SUCCESS" -ForegroundColor Green

# 4. Initialize Result File
"=== Phase 4 Execution Log ($(Get-Date)) ===" | Out-File -FilePath $ResultFile -Encoding utf8

# 5. Execute all SQL files in a single session
Write-Host "Running Phase 4 jobs in [$DbName]... " -NoNewline
$PsqlArgs = @("-X", "-U", "postgres", "-d", $DbName)
foreach ($file in $SqlFiles) {
    if (Test-Path $file) { $PsqlArgs += "-f"; $PsqlArgs += $file }
}

try {
    & $PsqlPath $PsqlArgs 2>&1 | Out-File -FilePath $ResultFile -Append -Encoding utf8
    Write-Host "DONE" -ForegroundColor Green
} catch {
    "Fatal error during execution: $_" | Out-File -FilePath $ResultFile -Append -Encoding utf8
    Write-Host " ERROR" -ForegroundColor Red
} finally {
    $env:PGPASSWORD = $null
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)
}

Write-Host "`nExecution results saved to $ResultFile"
