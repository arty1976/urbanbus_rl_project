$secrets = Get-Content -Raw "configs\secrets.json" | ConvertFrom-Json
$env:PGPASSWORD = $secrets.postgres.password
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -X -h localhost -p 5432 -U postgres -d urbanbus -f scripts\temp_queries.sql > scripts\temp_queries_output.txt 2>&1
Write-Host '실행 완료'
