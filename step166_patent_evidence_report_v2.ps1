$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$Reporter = ".\05_training\patent_evidence\patent_evidence_report_v2_step166.py"
$Validator = ".\05_training\patent_evidence\validate_patent_evidence_report_v2_step166.py"
$Test = ".\05_training\patent_evidence\test_patent_evidence_report_v2_step166.py"

Write-Host "[INFO] Step 166 Patent Evidence Report v2 should be copied into:"
Write-Host "       $ProjectRoot\05_training\patent_evidence"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\patent_evidence_report_v2_schema_step166.md"
Write-Host "  05_training\patent_evidence\patent_evidence_report_v2_step166.py"
Write-Host "  05_training\patent_evidence\validate_patent_evidence_report_v2_step166.py"
Write-Host "  05_training\patent_evidence\test_patent_evidence_report_v2_step166.py"
Write-Host ""

foreach ($p in @($Reporter, $Validator, $Test)) {
    if (-not (Test-Path $p)) {
        throw "[STOP] Required file not found: $p"
    }
}

$OutRoot = "artifacts\patent_evidence\patent_evidence_report_v2_step166_selftest"
& $py $Reporter `
    --mode sample `
    --output-root $OutRoot `
    --top-k-edges 10
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 166 report v2 sample run failed"
}

$Manifest = Join-Path $OutRoot "report_v2\patent_evidence_report_v2_manifest.json"
if (-not (Test-Path $Manifest)) {
    throw "[FAIL] Step 166 manifest not found: $Manifest"
}

& $py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 166 validator failed"
}

& $py $Test
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 166 unittest failed"
}

Write-Host "[OK] Step 166 Patent Evidence Report v2 self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot\report_v2"
Write-Host "[DONE] Step 166 complete. Patent evidence report v2 generated; non-claim guards remain locked."
