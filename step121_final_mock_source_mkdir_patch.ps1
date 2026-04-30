$ErrorActionPreference = "Stop"

Set-Location "C:\Users\ryujo\urbanbus_rl_project"

$PatchPy = @'
from pathlib import Path
import re

path = Path("05_training/rewards/test_validate_reward_ablation_execution_manifest_step121.py")
text = path.read_text(encoding="utf-8-sig")

# Robust patch strategy:
# 1) Try to insert immediately after the write_mock_step119_plan(...) function signature.
# 2) If the signature shape is unusual, insert immediately before the csv_path assignment.
mkdir_line = "    source_root.mkdir(parents=True, exist_ok=True)\n"

if "source_root.mkdir(parents=True, exist_ok=True)" in text:
    path.write_text(text, encoding="utf-8")
    print("[OK] source_root.mkdir already present")
    raise SystemExit(0)

pattern = r"(def\s+write_mock_step119_plan\s*\([^\)]*\)\s*(?:->\s*[^:]+)?\s*:\s*\r?\n)"
match = re.search(pattern, text)
if match:
    insert_at = match.end(1)
    text = text[:insert_at] + mkdir_line + text[insert_at:]
    path.write_text(text, encoding="utf-8")
    print("[OK] inserted source_root.mkdir after write_mock_step119_plan signature")
    raise SystemExit(0)

# Fallback: insert before the first csv_path assignment that uses source_root.
fallback_patterns = [
    '    csv_path = source_root / "reward_ablation_runner_dry_run_plan.csv"',
    "    csv_path = source_root / 'reward_ablation_runner_dry_run_plan.csv'",
]
for fp in fallback_patterns:
    pos = text.find(fp)
    if pos >= 0:
        text = text[:pos] + mkdir_line + text[pos:]
        path.write_text(text, encoding="utf-8")
        print("[OK] inserted source_root.mkdir before csv_path assignment")
        raise SystemExit(0)

# Last diagnostic: show nearby function names for debugging.
funcs = re.findall(r"^def\s+([A-Za-z0-9_]+)\s*\(", text, flags=re.MULTILINE)
raise RuntimeError(
    "Could not patch Step 121 test file. Found functions=" + repr(funcs[:20])
)
'@

$PatchPath = ".\05_training\rewards\_patch_step121_mock_source_mkdir_final.py"
$PatchPy | Set-Content -Path $PatchPath -Encoding UTF8

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $PatchPath
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 121 final mkdir patch failed"
}

$TestPath = ".\05_training\rewards\test_validate_reward_ablation_execution_manifest_step121.py"
& $py $TestPath
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 121 self-test failed after final mkdir patch"
}

Write-Host "[DONE] Step 121 final mkdir patch and self-test PASS"
