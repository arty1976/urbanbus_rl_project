from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "05_training" / "policies" / "generate_h200_transfer_package_manifest.py"
DOC = ROOT / "05_training" / "policies" / "H200_transfer_package_manifest.md"

REQUIRED_SOURCE_FILES = [
    "05_training/train_mappo_actual.py",
    "05_training/mappo_runner.py",
    "05_training/simulator_adapter_interface.py",
    "05_training/adapters/historical_replay_adapter.py",
    "05_training/policies/validate_mappo_checkpoint.py",
    "05_training/policies/h200_actual_checkpoint_preflight_checklist.py",
    "05_training/policies/a_family_mappo_actual_execution_plan.py",
    "05_training/policies/run_a_family_mappo_actual_matrix_from_plan.py",
    "05_training/policies/run_h200_actual_checkpoint_arrival_workflow.py",
    "05_training/policies/generate_h200_environment_report.py",
    "05_training/policies/validate_h200_actual_checkpoint_folder.py",
    "05_training/policies/run_h200_training_local_validation_bundle.py",
    "05_training/policies/generate_h200_transfer_package_manifest.py",
    "05_training/policies/H200_server_transfer_checklist.md",
    "05_training/policies/H200_mappo_training_runbook.md",
    "05_training/policies/H200_actual_checkpoint_folder_contract.md",
    "05_training/policies/H200_training_local_validation_bundle.md",
    "05_training/policies/MAPPO_actual_reproducibility_note.md",
]

REQUIRED_ARTIFACTS = [
    "artifacts/baseline_v1/baseline_contract.json",
    "artifacts/experiment_A_v1/experiment_A_contract.json",
    "artifacts/baseline_v1/B1_noop/scenario_index.parquet",
    "artifacts/baseline_v1/B2_rulebased/scenario_index.parquet",
]


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_fake_project(tmp: Path, with_artifacts: bool) -> Path:
    root = tmp / "fake_project"
    root.mkdir(parents=True, exist_ok=True)

    for rel in REQUIRED_SOURCE_FILES:
        touch(root / rel, "placeholder\n")

    if with_artifacts:
        for rel in REQUIRED_ARTIFACTS:
            touch(root / rel, "placeholder\n")

    return root


def run_generator(project_root: Path, output_path: Path, *args: str, expect_ok: bool) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--project-root",
        str(project_root),
        "--output-path",
        str(output_path),
        *args,
    ]

    completed = subprocess.run(cmd, text=True, capture_output=True)

    if expect_ok and completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError(f"expected PASS but got returncode={completed.returncode}")

    if (not expect_ok) and completed.returncode == 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("expected BLOCKED but command passed")

    return completed


def test_doc_contains_contract() -> None:
    if not DOC.exists():
        raise AssertionError(f"doc not found: {DOC}")

    text = DOC.read_text(encoding="utf-8-sig")
    required = [
        "Step 74",
        "H200 transfer package manifest",
        "PLANNING_MANIFEST_READY",
        "TRANSFER_READY",
        "strict_artifacts",
        "baseline_contract.json",
        "experiment_A_contract.json",
        "generate_h200_environment_report.py",
        "run_h200_training_local_validation_bundle.py",
        "train_mappo_actual.py",
        "validate_mappo_checkpoint.py",
        "run_h200_actual_checkpoint_arrival_workflow.py",
        "best_mappo.pt",
        "Do not return only",
        "Causal performance claims require Phase 2 causal simulator evaluation.",
    ]

    missing = [x for x in required if x not in text]
    if missing:
        raise AssertionError(f"doc missing required phrases: {missing}")


def test_planning_manifest_allows_missing_artifacts(tmp: Path) -> None:
    project_root = make_fake_project(tmp, with_artifacts=False)
    out = tmp / "planning_manifest.json"

    completed = run_generator(project_root, out, expect_ok=True)
    assert "[OK] Step 74" in completed.stdout

    manifest = read_json(out)
    assert manifest["transfer_manifest_status"] == "PLANNING_MANIFEST_READY"
    assert manifest["strict_artifacts"] is False
    assert manifest["source_file_check"]["all_present"] is True
    assert manifest["required_artifact_check"]["all_present"] is False
    assert manifest["claim_boundary"]["trained_h200_checkpoint_exists"] is False
    assert manifest["claim_boundary"]["causal_performance_claim_allowed"] is False


def test_strict_artifacts_blocks_when_artifacts_missing(tmp: Path) -> None:
    project_root = make_fake_project(tmp, with_artifacts=False)
    out = tmp / "strict_blocked_manifest.json"

    completed = run_generator(project_root, out, "--strict-artifacts", expect_ok=False)
    assert "missing required artifacts" in completed.stderr

    manifest = read_json(out)
    assert manifest["transfer_manifest_status"] == "BLOCKED"


def test_strict_artifacts_passes_when_artifacts_exist(tmp: Path) -> None:
    project_root = make_fake_project(tmp, with_artifacts=True)
    out = tmp / "strict_ready_manifest.json"

    completed = run_generator(project_root, out, "--strict-artifacts", expect_ok=True)
    assert "[OK] Step 74" in completed.stdout

    manifest = read_json(out)
    assert manifest["transfer_manifest_status"] == "TRANSFER_READY"
    assert manifest["strict_artifacts"] is True
    assert manifest["source_file_check"]["all_present"] is True
    assert manifest["required_artifact_check"]["all_present"] is True
    assert len(manifest["h200_command_pack"]) >= 8


def test_missing_source_blocks_even_in_planning_mode(tmp: Path) -> None:
    project_root = make_fake_project(tmp, with_artifacts=True)
    (project_root / "05_training" / "train_mappo_actual.py").unlink()

    out = tmp / "missing_source_manifest.json"
    completed = run_generator(project_root, out, expect_ok=False)
    assert "missing required source files" in completed.stderr

    manifest = read_json(out)
    assert manifest["transfer_manifest_status"] == "BLOCKED"


def main() -> None:
    if not GENERATOR.exists():
        raise AssertionError(f"generator not found: {GENERATOR}")

    test_doc_contains_contract()

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_planning_manifest_allows_missing_artifacts(tmp)
        test_strict_artifacts_blocks_when_artifacts_missing(tmp)
        test_strict_artifacts_passes_when_artifacts_exist(tmp)
        test_missing_source_blocks_even_in_planning_mode(tmp)

    print("[OK] Step 74 H200 transfer package manifest self-test PASS")


if __name__ == "__main__":
    main()
