$ErrorActionPreference = "Stop"

Set-Location "C:\Users\ryujo\urbanbus_rl_project"

$ValidatorPath = ".\05_training\rewards\validate_reward_ablation_runner_dry_run_plan_step119.py"
$TestPath = ".\05_training\rewards\test_validate_reward_ablation_runner_dry_run_plan_step119.py"

$text = Get-Content $ValidatorPath -Raw -Encoding UTF8

# Add JSON-safe default helper if missing.
if ($text -notlike "*def json_default_for_report*") {
    $helper = @'
def json_default_for_report(obj):
    """Make validation reports JSON-serializable.

    The validator may place sets in expected/actual check payloads.
    JSON cannot serialize Python set objects, so convert them to sorted lists.
    Also handle a few scalar-like objects defensively.
    """
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


'@

    $marker = 'def dump_json(path: Path, payload: Dict[str, Any]) -> None:'
    if (-not $text.Contains($marker)) {
        throw "[STOP] dump_json function marker not found in validator"
    }
    $text = $text.Replace($marker, $helper + $marker)
}

# Patch json.dump to use the helper.
$text = $text.Replace(
    'json.dump(payload, f, ensure_ascii=False, indent=2)',
    'json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default_for_report)'
)

Set-Content -Path $ValidatorPath -Value $text -Encoding UTF8
Write-Host "[OK] patched Step 119 validator JSON serialization for set objects"

# Optional cleanup: replace deprecated datetime.utcnow() in the test file.
if (Test-Path $TestPath) {
    $test = Get-Content $TestPath -Raw -Encoding UTF8
    if ($test -like "*datetime.utcnow()*") {
        if ($test -notlike "*timezone*") {
            $test = $test.Replace('from datetime import datetime', 'from datetime import datetime, timezone')
        }
        $test = $test.Replace('datetime.utcnow()', 'datetime.now(timezone.utc)')
        Set-Content -Path $TestPath -Value $test -Encoding UTF8
        Write-Host "[OK] patched Step 119 test datetime UTC deprecation warning"
    }
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 119 reward ablation runner dry-run plan self-test failed after JSON serialization patch"
}

Write-Host "[DONE] Step 119 JSON serialization patch and self-test PASS"
