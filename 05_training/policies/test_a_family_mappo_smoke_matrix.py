from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def append_training_dir_to_path() -> Path:
    here = Path(__file__).resolve()
    training_dir = here.parents[1]
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


def run_cmd(cmd, expect_success: bool):
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(" ".join(str(x) for x in cmd))
    print(proc.stdout)

    if expect_success and proc.returncode != 0:
        raise RuntimeError(f"expected success but failed: returncode={proc.returncode}")

    if (not expect_success) and proc.returncode == 0:
        raise RuntimeError("expected failure but command succeeded")

    return proc


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    import pandas as pd

    runner = training_dir / "run_a_family_mappo_smoke_matrix.py"
    out_dir = project_root / "artifacts" / "experiment_A_v1" / "a_family_mappo_smoke_matrix_selftest"

    run_cmd(
        [
            sys.executable,
            str(runner),
            "--conditions",
            "A,A90,A80,A70",
            "--seeds",
            "56",
            "--device",
            "cpu",
            "--output-root",
            str(out_dir),
        ],
        expect_success=True,
    )

    status_path = out_dir / "status.json"
    manifest_path = out_dir / "matrix_manifest.json"
    canonical_window = out_dir / "canonical_eval" / "kpi_by_window.parquet"
    canonical_manifest = out_dir / "canonical_eval" / "aggregation_manifest.json"
    metadata_report = out_dir / "reports" / "canonical_kpi_by_window_policy_metadata_validation.json"

    if not status_path.exists():
        raise RuntimeError(f"status not found: {status_path}")
    if not manifest_path.exists():
        raise RuntimeError(f"manifest not found: {manifest_path}")
    if not canonical_window.exists():
        raise RuntimeError(f"canonical kpi_by_window not found: {canonical_window}")
    if not canonical_manifest.exists():
        raise RuntimeError(f"canonical manifest not found: {canonical_manifest}")
    if not metadata_report.exists():
        raise RuntimeError(f"post-canonical metadata validation report not found: {metadata_report}")

    status = read_json(status_path)
    manifest = read_json(manifest_path)
    canonical_agg_manifest = read_json(canonical_manifest)
    metadata_payload = read_json(metadata_report)

    if status.get("status") != "PASS":
        raise RuntimeError(f"matrix status must be PASS: {status}")

    if int(status.get("run_count", 0)) != 4:
        raise RuntimeError(f"run_count must be 4 for one-seed A-family matrix: {status}")

    if metadata_payload.get("valid") is not True:
        raise RuntimeError(f"post-canonical metadata validation must be valid=true: {metadata_payload}")

    df = pd.read_parquet(canonical_window)

    expected_conditions = ["A", "A70", "A80", "A90"]
    observed_conditions = sorted(df["condition_id"].astype(str).str.upper().unique().tolist())
    if observed_conditions != expected_conditions:
        raise RuntimeError(f"condition_id mismatch: expected={expected_conditions}, got={observed_conditions}")

    if len(df) != 4:
        raise RuntimeError(f"canonical row count must be 4, got {len(df)}")

    if set(df["policy_source"].astype(str).tolist()) != {"mappo_policy"}:
        raise RuntimeError("all rows must have policy_source=mappo_policy")

    if not bool((df["checkpoint_loaded"].astype(bool) == True).all()):
        raise RuntimeError("all rows must have checkpoint_loaded=true")

    if not bool((df["checkpoint_validator_ran"].astype(bool) == True).all()):
        raise RuntimeError("all rows must have checkpoint_validator_ran=true")

    if not bool((df["mock_action_used"].astype(bool) == False).all()):
        raise RuntimeError("all rows must have mock_action_used=false")

    if not bool((df["placeholder_fallback_used"].astype(bool) == False).all()):
        raise RuntimeError("all rows must have placeholder_fallback_used=false")

    if not bool((df["actual_policy_claim_ready"].astype(bool) == False).all()):
        raise RuntimeError("mappo_smoke matrix must not be actual claim-ready")

    if not bool((df["causal_policy_claim_ready"].astype(bool) == False).all()):
        raise RuntimeError("mappo_smoke matrix must not be causal claim-ready")

    if bool(canonical_agg_manifest.get("causal_comparison_allowed", True)):
        raise RuntimeError("mappo_smoke matrix canonical output must not allow causal comparison")

    run_cmd(
        [
            sys.executable,
            str(runner),
            "--conditions",
            "A,B2",
            "--seeds",
            "56",
            "--device",
            "cpu",
            "--output-root",
            str(out_dir / "expected_fail_bad_condition"),
        ],
        expect_success=False,
    )

    report = {
        "status": "PASS",
        "step": 56,
        "runner": str(runner),
        "output_root": str(out_dir),
        "status_json": str(status_path),
        "manifest": str(manifest_path),
        "canonical_window": str(canonical_window),
        "metadata_report": str(metadata_report),
        "conditions": observed_conditions,
        "row_count": int(len(df)),
        "policy_source_counts": df["policy_source"].astype(str).value_counts().to_dict(),
        "source_mode_counts": df["source_mode"].astype(str).value_counts().to_dict(),
        "note": (
            "Step 56 self-test validates A/A90/A80/A70 mappo_smoke matrix path. "
            "Smoke artifacts are not for performance claims."
        ),
    }

    out_report = out_dir / "step56_a_family_mappo_smoke_matrix_selftest_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] Step 56 A-family mappo_smoke matrix self-test PASS")
    print(f"[OK] report: {out_report}")


if __name__ == "__main__":
    main()
