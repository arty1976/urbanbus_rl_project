$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardDir | Out-Null

# ---------------------------------------------------------------------
# Step 148 main python
# ---------------------------------------------------------------------
$MainPy = @'
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


STEP_ID = 148
ARTIFACT_VERSION = "h200_receive_side_preflight_gate_step148_v1"

REQUIRED_SOURCE_FILES = [
    "05_training/rewards/final_reward_spec_step111.md",
    "05_training/rewards/final_reward_spec_step111.json",
    "05_training/rewards/a_family_72run_release_matrix_extension_draft_step143.md",
    "05_training/rewards/a_family_72run_release_matrix_extension_draft_step143.py",
    "05_training/rewards/validate_a_family_72run_release_matrix_extension_draft_step143.py",
    "05_training/rewards/test_a_family_72run_release_matrix_extension_draft_step143.py",
    "05_training/rewards/h200_execution_package_boundary_manifest_step144.md",
    "05_training/rewards/h200_execution_package_boundary_manifest_step144.py",
    "05_training/rewards/validate_h200_execution_package_boundary_manifest_step144.py",
    "05_training/rewards/test_h200_execution_package_boundary_manifest_step144.py",
    "05_training/rewards/h200_transfer_package_export_manifest_step145.md",
    "05_training/rewards/h200_transfer_package_export_manifest_step145.py",
    "05_training/rewards/validate_h200_transfer_package_export_manifest_step145.py",
    "05_training/rewards/test_h200_transfer_package_export_manifest_step145.py",
    "05_training/rewards/h200_transfer_package_integrity_verifier_step146.md",
    "05_training/rewards/h200_transfer_package_integrity_verifier_step146.py",
    "05_training/rewards/validate_h200_transfer_package_integrity_verifier_step146.py",
    "05_training/rewards/test_h200_transfer_package_integrity_verifier_step146.py",
    "05_training/rewards/h200_receive_side_transfer_runbook_step147.md",
    "05_training/rewards/h200_receive_side_transfer_runbook_step147.py",
    "05_training/rewards/validate_h200_receive_side_transfer_runbook_step147.py",
    "05_training/rewards/test_h200_receive_side_transfer_runbook_step147.py",
    "step147_h200_receive_side_transfer_runbook.ps1",
]

LOCAL_ONLY_PATTERNS = [
    "artifacts/rewards/**",
    "05_training/rewards/*.latest.json",
    "artifacts/baseline_v1/**",
    "1000005000/**",
    "project_files/**",
]

H200_RECEIVE_SIDE_CHECKLIST = [
    "clone_or_copy_package_to_h200_project_root",
    "verify_git_commit_matches_transfer_manifest",
    "run_step146_integrity_verifier_on_h200",
    "confirm_python_version",
    "confirm_torch_import",
    "confirm_cuda_visible",
    "confirm_gpu_count",
    "confirm_checkpoint_output_root_exists_or_can_be_created",
    "confirm_artifact_output_root_exists_or_can_be_created",
    "confirm_no_latest_json_used_as_source_of_truth",
    "confirm_actual_execution_release_flag_is_false",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(project_root: Path, args: List[str]) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(project_root),
            stderr=subprocess.STDOUT,
            text=True,
        )
        return out.strip()
    except Exception as exc:
        return f"GIT_UNAVAILABLE: {exc}"


def collect_git_metadata(project_root: Path) -> Dict[str, Any]:
    return {
        "branch": run_git(project_root, ["branch", "--show-current"]),
        "commit": run_git(project_root, ["rev-parse", "HEAD"]),
        "status_short": run_git(project_root, ["status", "--short"]),
        "remote_origin": run_git(project_root, ["remote", "get-url", "origin"]),
    }


def classify_missing_required(project_root: Path) -> List[str]:
    missing = []
    for rel in REQUIRED_SOURCE_FILES:
        if not (project_root / rel).exists():
            missing.append(rel)
    return missing


def build_file_inventory(project_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rel in REQUIRED_SOURCE_FILES:
        path = project_root / rel
        row: Dict[str, Any] = {
            "relative_path": rel,
            "exists": path.exists(),
            "size_bytes": None,
            "sha256": None,
        }
        if path.exists() and path.is_file():
            row["size_bytes"] = path.stat().st_size
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return rows


def build_manifest(
    project_root: Path,
    output_root: Path,
    h200_project_root: str,
    require_clean_worktree: bool = False,
) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    hard_failures: List[str] = []
    warnings: List[str] = []

    if not project_root.exists():
        hard_failures.append(f"project_root_not_found: {project_root}")

    missing_required = classify_missing_required(project_root)
    if missing_required:
        hard_failures.append(
            "missing_required_source_files: " + ", ".join(missing_required)
        )

    git_meta = collect_git_metadata(project_root)
    if str(git_meta.get("commit", "")).startswith("GIT_UNAVAILABLE"):
        warnings.append("git_metadata_unavailable")

    status_short = str(git_meta.get("status_short", ""))
    if status_short and not status_short.startswith("GIT_UNAVAILABLE"):
        if require_clean_worktree:
            hard_failures.append("working_tree_not_clean")
        else:
            warnings.append("working_tree_not_clean_but_not_blocking_for_step148")

    if "\\" in h200_project_root:
        hard_failures.append("h200_project_root_must_be_linux_style_path")

    if not h200_project_root.startswith("/"):
        hard_failures.append("h200_project_root_must_be_absolute_linux_path")

    inventory = build_file_inventory(project_root)

    gate_status = (
        "READY_FOR_H200_RECEIVE_SIDE_PREFLIGHT_REVIEW"
        if not hard_failures
        else "BLOCKED"
    )

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "step_id": STEP_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "h200_project_root": h200_project_root,
        "gate_status": gate_status,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "git_metadata": git_meta,
        "required_source_file_count": len(REQUIRED_SOURCE_FILES),
        "missing_required_source_files": missing_required,
        "file_inventory": inventory,
        "local_only_patterns": LOCAL_ONLY_PATTERNS,
        "receive_side_checklist": H200_RECEIVE_SIDE_CHECKLIST,
        "receive_side_expected_commands": [
            "git status --short",
            "git rev-parse HEAD",
            "python 05_training/rewards/h200_transfer_package_integrity_verifier_step146.py --help",
            "python 05_training/rewards/h200_receive_side_preflight_gate_step148.py --help",
        ],
        "operator_handoff_policy": {
            "source_of_truth": [
                "tracked_source_scripts",
                "tracked_validation_scripts",
                "tracked_markdown_specs",
                "tracked_json_specs",
            ],
            "not_source_of_truth": [
                "latest_json_pointer_files",
                "local_artifacts_recreated_from_scripts",
                "windows_absolute_paths",
                "manual_console_claims",
            ],
        },
        "execution_locks": {
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "train_allowed": False,
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
        "next_step_recommendation": {
            "step": 149,
            "title": "H200 environment preflight result manifest",
            "meaning": (
                "Run this receive-side checklist on the H200 server and save "
                "the actual environment result without releasing training."
            ),
        },
    }

    manifest_path = output_root / "h200_receive_side_preflight_gate_step148_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/h200_receive_side_preflight_gate_step148",
    )
    parser.add_argument(
        "--h200-project-root",
        default="/workspace/urbanbus_rl_project",
    )
    parser.add_argument("--require-clean-worktree", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(
        project_root=Path(args.project_root),
        output_root=Path(args.output_root),
        h200_project_root=args.h200_project_root,
        require_clean_worktree=bool(args.require_clean_worktree),
    )

    print("[OK] Step 148 H200 receive-side preflight gate completed")
    print(f"[OK] gate_status     : {manifest['gate_status']}")
    print(f"[OK] hard_failures   : {len(manifest['hard_failures'])}")
    print(f"[OK] warnings        : {len(manifest['warnings'])}")
    print(f"[OK] h200_root       : {manifest['h200_project_root']}")
    print(f"[OK] manifest        : {manifest['manifest_path']}")
    print(f"[OK] actual_execution_allowed : {manifest['execution_locks']['actual_execution_allowed']}")
    print(f"[OK] train_allowed            : {manifest['execution_locks']['train_allowed']}")

    if manifest["hard_failures"]:
        for failure in manifest["hard_failures"]:
            print(f"[BLOCKED] {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
'@

$MainPy | Set-Content -Path "$RewardDir\h200_receive_side_preflight_gate_step148.py" -Encoding UTF8

# ---------------------------------------------------------------------
# Step 148 validator
# ---------------------------------------------------------------------
$ValidatePy = @'
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FALSE_LOCKS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "train_allowed",
    "actual_results",
    "winner_selected",
    "trainable_reward_promoted",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def validate_manifest(path: Path) -> None:
    payload = load_json(path)

    if payload.get("artifact_version") != "h200_receive_side_preflight_gate_step148_v1":
        raise RuntimeError("artifact_version mismatch")

    if int(payload.get("step_id", -1)) != 148:
        raise RuntimeError("step_id must be 148")

    if payload.get("gate_status") != "READY_FOR_H200_RECEIVE_SIDE_PREFLIGHT_REVIEW":
        raise RuntimeError(f"gate_status is not ready: {payload.get('gate_status')}")

    if payload.get("hard_failures"):
        raise RuntimeError(f"hard_failures found: {payload.get('hard_failures')}")

    locks = payload.get("execution_locks", {})
    for key in REQUIRED_FALSE_LOCKS:
        if locks.get(key) is not False:
            raise RuntimeError(f"execution lock must remain false: {key}")

    h200_root = str(payload.get("h200_project_root", ""))
    if "\\" in h200_root or not h200_root.startswith("/"):
        raise RuntimeError("h200_project_root must be an absolute Linux-style path")

    inventory = payload.get("file_inventory", [])
    if not inventory:
        raise RuntimeError("file_inventory is empty")

    missing = [x["relative_path"] for x in inventory if not x.get("exists")]
    if missing:
        raise RuntimeError(f"missing inventory files: {missing}")

    print("[OK] Step 148 manifest validation PASS")
    print(f"[OK] manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    validate_manifest(Path(args.manifest))


if __name__ == "__main__":
    main()
'@

$ValidatePy | Set-Content -Path "$RewardDir\validate_h200_receive_side_preflight_gate_step148.py" -Encoding UTF8

# ---------------------------------------------------------------------
# Step 148 tests
# ---------------------------------------------------------------------
$TestPy = @'
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h200_receive_side_preflight_gate_step148 as step148


def make_required_files(root: Path) -> None:
    for rel in step148.REQUIRED_SOURCE_FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"dummy file for {rel}\n", encoding="utf-8")


def test_ready_manifest_with_all_required_files() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_required_files(root)
        manifest = step148.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step148",
            h200_project_root="/workspace/urbanbus_rl_project",
        )

        assert manifest["gate_status"] == "READY_FOR_H200_RECEIVE_SIDE_PREFLIGHT_REVIEW"
        assert manifest["hard_failures"] == []
        assert manifest["execution_locks"]["actual_execution_allowed"] is False
        assert manifest["execution_locks"]["train_allowed"] is False
        assert manifest["execution_locks"]["paper_level_claim_allowed"] is False
        assert manifest["required_source_file_count"] == len(step148.REQUIRED_SOURCE_FILES)

        manifest_path = Path(manifest["manifest_path"])
        assert manifest_path.exists()
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert saved["step_id"] == 148


def test_missing_required_file_blocks_gate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_required_files(root)
        missing_path = root / step148.REQUIRED_SOURCE_FILES[0]
        missing_path.unlink()

        manifest = step148.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step148",
            h200_project_root="/workspace/urbanbus_rl_project",
        )

        assert manifest["gate_status"] == "BLOCKED"
        assert manifest["missing_required_source_files"]
        assert any("missing_required_source_files" in x for x in manifest["hard_failures"])


def test_windows_style_h200_path_blocks_gate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_required_files(root)

        manifest = step148.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step148",
            h200_project_root="C:\\workspace\\urbanbus_rl_project",
        )

        assert manifest["gate_status"] == "BLOCKED"
        assert any("linux_style" in x for x in manifest["hard_failures"])


def main() -> None:
    test_ready_manifest_with_all_required_files()
    test_missing_required_file_blocks_gate()
    test_windows_style_h200_path_blocks_gate()
    print("[OK] Step 148 H200 receive-side preflight gate self-test PASS")


if __name__ == "__main__":
    main()
'@

$TestPy | Set-Content -Path "$RewardDir\test_h200_receive_side_preflight_gate_step148.py" -Encoding UTF8

# ---------------------------------------------------------------------
# Step 148 markdown
# ---------------------------------------------------------------------
$Md = @'
# Step 148 — H200 receive-side preflight / operator handoff gate

## Purpose

Step 148 defines the receive-side preflight gate for the H200 server.

This step does not run training.  
This step does not release actual execution.  
This step only verifies that the H200-side operator has the required source files, validation scripts, transfer integrity tools, and execution locks before any real run is attempted.

## Scope

Step 148 checks:

1. Required Step 111 / Step 143 / Step 144 / Step 145 / Step 146 / Step 147 files exist.
2. Git metadata can be recorded.
3. H200 project root is represented as a Linux-style absolute path.
4. Receive-side checklist is written into a manifest.
5. Actual execution flags remain locked.
6. Training remains disallowed.
7. Paper-level and causal performance claims remain disallowed.

## Required source-of-truth inputs

The receive side must use tracked source files and validation scripts as source of truth.

Do not use:

- `*.latest.json` pointer files
- local artifacts as primary source
- Windows absolute paths
- manual console claims

## Execution locks

The following must remain false in this step:

- `actual_execution_allowed`
- `actual_execution_released`
- `train_allowed`
- `actual_results`
- `winner_selected`
- `trainable_reward_promoted`
- `paper_level_claim_allowed`
- `causal_performance_claim_allowed`

## Expected status

Successful Step 148 status:

```text
READY_FOR_H200_RECEIVE_SIDE_PREFLIGHT_REVIEW
```

This means the package is ready to be checked on H200, not that training is allowed.

## Next step

Recommended next step:

```text
Step 149 — H200 environment preflight result manifest
```

Step 149 should be run on the H200 server and should record actual environment facts such as Python version, PyTorch import, CUDA visibility, GPU count, disk paths, and Step 146 integrity verification result.
'@

$Md | Set-Content -Path "$RewardDir\h200_receive_side_preflight_gate_step148.md" -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 148
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$OutputRoot = ".\artifacts\rewards\h200_receive_side_preflight_gate_step148"
$Manifest = Join-Path $OutputRoot "h200_receive_side_preflight_gate_step148_manifest.json"

& $py "$RewardDir\test_h200_receive_side_preflight_gate_step148.py"
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 148 self-test failed"
}

& $py "$RewardDir\h200_receive_side_preflight_gate_step148.py" `
  --project-root "." `
  --output-root $OutputRoot `
  --h200-project-root "/workspace/urbanbus_rl_project"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 148 preflight gate failed"
}

& $py "$RewardDir\validate_h200_receive_side_preflight_gate_step148.py" `
  --manifest $Manifest

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 148 manifest validation failed"
}

Write-Host "[DONE] Step 148 H200 receive-side preflight gate complete."
