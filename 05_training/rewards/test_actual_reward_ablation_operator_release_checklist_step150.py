from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


CONDITIONS = ["A", "A90", "A80", "A70"]
REWARDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
SEEDS = [1, 2, 3]


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cmd, cwd: Path, expect_ok: bool = True) -> subprocess.CompletedProcess:
    cp = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if expect_ok and cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command failed: {cmd}")
    if (not expect_ok) and cp.returncode == 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command unexpectedly passed: {cmd}")
    return cp


def matrix_manifest() -> dict:
    runs = []
    run_id = 0
    for condition in CONDITIONS:
        for reward_id in REWARDS:
            for seed in SEEDS:
                run_id += 1
                runs.append(
                    {
                        "run_id": f"step143_{run_id:03d}",
                        "condition_id": condition,
                        "reward_id": reward_id,
                        "seed": seed,
                    }
                )
    return {
        "step": 143,
        "audit_status": "PASS",
        "matrix_status": "READY",
        "runs": runs,
    }


def baseline_manifest() -> dict:
    return {
        "step": 141,
        "audit_status": "PASS",
        "baseline_reference_status": "READY",
        "baselines": [
            {"baseline_id": "B0R"},
            {"baseline_id": "B1"},
            {"baseline_id": "B2"},
        ],
    }


def generic_manifest(step: int, status_key: str = "audit_status") -> dict:
    return {
        "step": step,
        status_key: "PASS",
        "status": "READY",
        "actual_execution_allowed": False,
        "train_allowed": False,
    }


def step149_manifest() -> dict:
    return {
        "step": 149,
        "audit_status": "PASS",
        "environment_status": "LOCAL_ENVIRONMENT_PREFLIGHT_RECORDED_H200_NOT_ASSERTED_ACTUAL_STILL_LOCKED",
        "expect_h200": False,
        "torch_import_ok": True,
        "cuda_available": False,
        "gpu_count": 0,
        "hard_failures": 0,
        "warnings": 3,
        "actual_execution_allowed": False,
        "train_allowed": False,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reward_dir = project_root / "05_training" / "rewards"
    generator = reward_dir / "actual_reward_ablation_operator_release_checklist_step150.py"
    validator = reward_dir / "validate_actual_reward_ablation_operator_release_checklist_step150.py"

    root = project_root / "artifacts" / "rewards" / "actual_reward_ablation_operator_release_checklist_step150_selftest"
    input_root = root / "input"
    output_root = root / "positive_output"
    negative_output = root / "negative_output"

    step143 = input_root / "step143_matrix.json"
    step141 = input_root / "step141_baseline.json"
    step144 = input_root / "step144_boundary.json"
    step145 = input_root / "step145_export.json"
    step146 = input_root / "step146_integrity.json"
    step147 = input_root / "step147_runbook.json"
    step148 = input_root / "step148_gate.json"
    step149 = input_root / "step149_environment.json"

    dump_json(step143, matrix_manifest())
    dump_json(step141, baseline_manifest())
    dump_json(step144, generic_manifest(144))
    dump_json(step145, generic_manifest(145))
    dump_json(step146, generic_manifest(146))
    dump_json(step147, generic_manifest(147))
    dump_json(step148, generic_manifest(148, status_key="gate_status"))
    dump_json(step149, step149_manifest())

    run(
        [
            sys.executable,
            str(generator),
            "--project-root",
            str(project_root),
            "--output-root",
            str(output_root),
            "--step143-matrix-manifest",
            str(step143),
            "--step141-baseline-manifest",
            str(step141),
            "--step144-manifest",
            str(step144),
            "--step145-manifest",
            str(step145),
            "--step146-manifest",
            str(step146),
            "--step147-manifest",
            str(step147),
            "--step148-manifest",
            str(step148),
            "--step149-manifest",
            str(step149),
        ],
        cwd=project_root,
        expect_ok=True,
    )

    manifest = output_root / "actual_reward_ablation_operator_release_checklist_step150_manifest.json"
    run([sys.executable, str(validator), "--manifest", str(manifest)], cwd=project_root, expect_ok=True)

    bad_matrix = matrix_manifest()
    bad_matrix["runs"] = bad_matrix["runs"][:-1]
    bad_step143 = input_root / "step143_bad_matrix.json"
    dump_json(bad_step143, bad_matrix)

    run(
        [
            sys.executable,
            str(generator),
            "--project-root",
            str(project_root),
            "--output-root",
            str(negative_output),
            "--step143-matrix-manifest",
            str(bad_step143),
            "--step141-baseline-manifest",
            str(step141),
            "--step144-manifest",
            str(step144),
            "--step145-manifest",
            str(step145),
            "--step146-manifest",
            str(step146),
            "--step147-manifest",
            str(step147),
            "--step148-manifest",
            str(step148),
            "--step149-manifest",
            str(step149),
        ],
        cwd=project_root,
        expect_ok=False,
    )

    print("[OK] Step 150 actual reward ablation operator release checklist self-test PASS")
    print(f"[OK] selftest_root: {root}")


if __name__ == "__main__":
    main()
