from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
CONDITIONS = ["A", "A90", "A80", "A70"]
SEEDS = [1, 2, 3]


def unique_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def load_json(path: Path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_mock_step122_noop_output(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "reward_ablation_sandbox_noop_results.csv"
    manifest_path = root / "reward_ablation_sandbox_noop_runner_guard_step122_manifest.json"

    fieldnames = [
        "run_id",
        "candidate_id",
        "condition_id",
        "seed",
        "noop_status",
        "execute_allowed",
        "actual_training_allowed",
        "train_with_this_reward_allowed",
        "actual_results",
        "winner_selected",
        "command_hash",
        "planned_output_root",
    ]

    rows = []
    for candidate in CANDIDATES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_id = f"{candidate}_{condition}_seed{seed:03d}"
                rows.append({
                    "run_id": run_id,
                    "candidate_id": candidate,
                    "condition_id": condition,
                    "seed": seed,
                    "noop_status": "NOOP_RECORDED_NOT_EXECUTED",
                    "execute_allowed": False,
                    "actual_training_allowed": False,
                    "train_with_this_reward_allowed": False,
                    "actual_results": False,
                    "winner_selected": False,
                    "command_hash": f"hash_{run_id}",
                    "planned_output_root": str(root / "planned" / run_id),
                })

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "artifact_version": "mock_step122_noop_runner_manifest_v1",
        "audit_status": "PASS",
        "row_count": 72,
        "actual_results": False,
        "winner_selected": False,
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "output_files": {
            "noop_results_csv": str(csv_path)
        }
    }
    dump_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tag = unique_tag()
    source_root = project_root / "artifacts" / "rewards" / f"step123_mock_step122_noop_source_{tag}"
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_noop_ingestion_step123_selftest_{tag}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_noop_ingestion_step123_validation_{tag}"

    ingest = project_root / "05_training" / "rewards" / "reward_ablation_noop_result_ingestion_guard_step123.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_noop_result_ingestion_guard_step123.py"

    noop_manifest = write_mock_step122_noop_output(source_root)

    cmd = [
        sys.executable,
        str(ingest),
        "--noop-manifest",
        str(noop_manifest),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 123 ingestion script failed")

    ingestion_manifest = output_root / "reward_ablation_noop_result_ingestion_guard_step123_manifest.json"
    if not ingestion_manifest.exists():
        raise SystemExit("[FAIL] ingestion manifest missing")

    vcmd = [
        sys.executable,
        str(validator),
        "--ingestion-manifest",
        str(ingestion_manifest),
        "--output-root",
        str(validation_root),
    ]
    vresult = subprocess.run(vcmd, cwd=project_root, text=True, capture_output=True)
    print(vresult.stdout)
    if vresult.returncode != 0:
        print(vresult.stderr)
        raise SystemExit("[FAIL] Step 123 validator failed")

    report = load_json(validation_root / "reward_ablation_noop_result_ingestion_guard_step123_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_NOOP_RESULT_INGESTION_GUARD_NOT_ACTUAL_RESULTS",
        "next_status": "READY_FOR_STEP124_REWARD_ABLATION_SELECTION_CRITERIA_GATE",
        "row_count": 72,
        "actual_results": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test: a corrupted actual_results=true row must fail ingestion validation.
    bad_root = project_root / "artifacts" / "rewards" / f"step123_bad_step122_noop_source_{tag}"
    bad_manifest = write_mock_step122_noop_output(bad_root)
    bad_payload = load_json(bad_manifest)
    bad_csv = Path(bad_payload["output_files"]["noop_results_csv"])
    with open(bad_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["actual_results"] = "True"
    with open(bad_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bad_output = project_root / "artifacts" / "rewards" / f"reward_ablation_noop_ingestion_step123_bad_case_{tag}"
    bad_cmd = [
        sys.executable,
        str(ingest),
        "--noop-manifest",
        str(bad_manifest),
        "--output-root",
        str(bad_output),
    ]
    bad_result = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] corrupted actual_results row unexpectedly passed ingestion")

    print("[OK] Step 123 reward ablation no-op result ingestion guard self-test PASS")
    print("[DONE] Step 123 reward ablation no-op result ingestion guard complete.")


if __name__ == "__main__":
    main()
