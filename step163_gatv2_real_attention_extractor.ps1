$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$ToolDir = Join-Path $ProjectRoot "05_training\patent_evidence"
$Extractor = Join-Path $ToolDir "gatv2_real_attention_extractor_step163.py"
$Validator = Join-Path $ToolDir "validate_gatv2_real_attention_extractor_step163.py"
$TestFile = Join-Path $ToolDir "test_gatv2_real_attention_extractor_step163.py"
$Schema = Join-Path $ToolDir "gatv2_real_attention_extractor_schema_step163.md"

Write-Host "[INFO] Step 163 GATv2 Real Attention Extractor should be copied into:"
Write-Host "       $ToolDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\gatv2_real_attention_extractor_schema_step163.md"
Write-Host "  05_training\patent_evidence\gatv2_real_attention_extractor_step163.py"
Write-Host "  05_training\patent_evidence\validate_gatv2_real_attention_extractor_step163.py"
Write-Host "  05_training\patent_evidence\test_gatv2_real_attention_extractor_step163.py"
Write-Host ""

foreach ($p in @($Extractor, $Validator, $TestFile, $Schema)) {
    if (-not (Test-Path $p)) {
        throw "[FAIL] Missing expected Step 163 file: $p"
    }
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$OutRoot = "artifacts\patent_evidence\gatv2_real_attention_extractor_step163_selftest"
if (Test-Path $OutRoot) {
    Remove-Item -Recurse -Force $OutRoot
}

& $py $Extractor `
  --mode sample `
  --output-root $OutRoot `
  --condition-id A `
  --seed 1 `
  --file-format csv `
  --top-k-per-attempt 12

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 163 extractor sample run failed"
}

$Manifest = Join-Path $OutRoot "gatv2_real_attention_extractor_manifest.json"
& $py $Validator --manifest $Manifest --require-real-gatv2conv
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 163 validation failed"
}

& $py $TestFile
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 163 unittest failed"
}

Write-Host "[OK] Step 163 GATv2 Real Attention Extractor self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot"
Write-Host "[DONE] Step 163 complete. Non-claim guards remain locked."
