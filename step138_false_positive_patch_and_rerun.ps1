$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$PyMain = ".\05_training\rewards\update_project_log_reward_execution_gates_step138.py"
$PyValidate = ".\05_training\rewards\validate_reward_execution_gate_project_log_update_step138.py"
$PyTest = ".\05_training\rewards\test_reward_execution_gate_project_log_update_step138.py"

if (-not (Test-Path $PyMain)) {
    throw "[STOP] Step 138 main python file not found: $PyMain. Run step138 script once first."
}
if (-not (Test-Path $PyValidate)) {
    throw "[STOP] Step 138 validator python file not found: $PyValidate"
}
if (-not (Test-Path $PyTest)) {
    throw "[STOP] Step 138 test python file not found: $PyTest"
}

$PatchPy = @'
from __future__ import annotations

from pathlib import Path


path = Path("05_training/rewards/update_project_log_reward_execution_gates_step138.py")
text = path.read_text(encoding="utf-8-sig")

old = """def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    violations: List[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                violations.append(f"forbidden_true: {path}")
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))

    return violations
"""

new = """def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    violations: List[str] = []

    # Step 138 patch:
    # Step 137 static_review.required_source_tokens is a token-presence map.
    # A value True there means "the source code contains this safety token",
    # not "the experiment flag is true". Therefore it must not be treated as
    # a forbidden execution/result/claim flag.
    ignored_token_presence_maps = {
        "required_source_tokens",
    }

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)

            if key in ignored_token_presence_maps:
                continue

            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                violations.append(f"forbidden_true: {path}")

            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))

    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                violations.extend(recursive_forbidden_true(value, path))

    return violations
"""

if old not in text:
    if "ignored_token_presence_maps" in text:
        print("[OK] Step 138 false-positive patch already applied")
    else:
        raise RuntimeError("target recursive_forbidden_true block not found")
else:
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print("[OK] Step 138 false-positive patch applied")
'@

$PatchPath = ".\artifacts\rewards\step138_false_positive_patch.py"
New-Item -ItemType Directory -Force -Path ".\artifacts\rewards" | Out-Null
Set-Content -Path $PatchPath -Value $PatchPy -Encoding UTF8

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $PatchPath
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 138 false-positive patch failed"
}

$OutputRoot = "artifacts/rewards/reward_execution_gate_project_log_update_step138"
$Manifest = Join-Path $ProjectRoot "$OutputRoot\reward_execution_gate_project_log_update_step138_manifest.json"

& $py $PyMain `
  --project-root "." `
  --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 138 project log update failed after patch"
}

& $py $PyValidate --manifest $Manifest --project-log "project_log.md"
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 138 manifest/log validation failed after patch"
}

& $py $PyTest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 138 self-test failed after patch"
}

Write-Host "[DONE] Step 138 false-positive patch and project log update PASS."
