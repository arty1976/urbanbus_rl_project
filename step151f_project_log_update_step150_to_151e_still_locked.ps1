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

$TestPath = ".\05_training\rewards\test_project_log_update_step151f.py"
$GenPath = ".\05_training\rewards\project_log_update_step151f.py"
$ValPath = ".\05_training\rewards\validate_project_log_update_step151f.py"

& $py $TestPath
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-F self-test failed"
}

& $py $GenPath `
  --project-root "." `
  --project-log ".\project_log.md" `
  --step150-manifest ".\artifacts\rewards\actual_reward_ablation_operator_release_checklist_step150\actual_reward_ablation_operator_release_checklist_step150_manifest.json" `
  --step151a-manifest ".\artifacts\rewards\explicit_operator_release_manifest_draft_step151a\explicit_operator_release_manifest_draft_step151a_manifest.json" `
  --step151b-manifest ".\artifacts\rewards\h200_expected_preflight_rerun_checklist_step151b\h200_expected_preflight_rerun_checklist_step151b_manifest.json" `
  --step151c-manifest ".\artifacts\rewards\h200_actual_preflight_rerun_result_intake_placeholder_step151c\h200_actual_preflight_rerun_result_intake_placeholder_step151c_manifest.json" `
  --step151d-manifest ".\artifacts\rewards\operator_approval_decision_draft_step151d\operator_approval_decision_draft_step151d_manifest.json" `
  --step151e-manifest ".\artifacts\rewards\h200_handoff_packet_index_step151e\h200_handoff_packet_index_step151e_manifest.json" `
  --output-root ".\artifacts\rewards\project_log_update_step151f"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-F project log update generation failed"
}

$Step151FManifest = ".\artifacts\rewards\project_log_update_step151f\project_log_update_step151f_manifest.json"

& $py $ValPath --manifest $Step151FManifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-F manifest validation failed"
}

Write-Host "[DONE] Step 151-F project log update complete."
Write-Host "[OK] manifest: $Step151FManifest"
