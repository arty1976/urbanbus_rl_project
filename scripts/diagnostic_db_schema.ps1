# ============================================================
# diagnostic_db_schema.ps1
# ============================================================

$DbName = "urbanbus"
$ResultFile = "db_schema_diagnostic.txt"

# 1. Discover psql.exe
$PsqlPath = "psql"
if (!(Get-Command $PsqlPath -ErrorAction SilentlyContinue)) {
    $CommonPaths = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($CommonPaths) { $PsqlPath = $CommonPaths[0].FullName }
}

# 2. Get Password and Verify
$pass = Read-Host "Enter Password for user postgres" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($pass)
$PlainPass = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
$env:PGPASSWORD = $PlainPass

try {
    Write-Host "Diagnosing [$DbName]..."
    # List tables across all schemas
    $query = "SELECT schemaname, tablename FROM pg_catalog.pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema');"
    & $PsqlPath -X -U postgres -d $DbName -c $query 2>&1 | Out-File -FilePath $ResultFile -Encoding utf8
    Write-Host "Diagnostic results saved to $ResultFile"
} finally {
    $env:PGPASSWORD = $null
}
