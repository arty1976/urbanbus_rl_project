# run_validation.ps1
$psqlPath = "C:\Program Files\PostgreSQL\18\bin\psql.exe"
if (-not (Test-Path $psqlPath)) {
    # Try default psql in path
    $psqlPath = "psql"
}

$outputFile = "D:\urbanbus_rl_project\validation_output.txt"
$sqlFile = "D:\urbanbus_rl_project\03_validation_queries\route_link_promotion_readiness.sql"

Write-Host "Running validation SQL..."
& $psqlPath -X -h localhost -p 5432 -U postgres -d urbanbus -f $sqlFile > $outputFile 2>&1

Write-Host "Validation completed. Output saved to $outputFile"
