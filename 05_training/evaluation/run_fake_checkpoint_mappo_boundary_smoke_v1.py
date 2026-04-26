"""
run_fake_checkpoint_mappo_boundary_smoke_v1.py
==============================================

Step 34 smoke test for fake-checkpoint MAPPO boundary.

Purpose
-------
This smoke test creates a fake checkpoint file and verifies that:

1. run_causal_rollout.py --a-family-bridge accepts --policy-kind mappo
   when --require-existing-checkpoint is used and the checkpoint exists.
2. A/A90/A80/A70 rollout rows use policy_source=mappo_policy.
3. source_mode is causal_*_mappo_policy_v1.
4. placeholder_policy is not mixed into actual MAPPO boundary rows.
5. qwen_trigger_rate remains 0.0.
6. canonical_kpi_aggregator.py can consume the generated window_rollup files.

Important
---------
This is not a performance claim.
The fake checkpoint is not a trained neural MAPPO model.
It only validates the policy boundary and artifact path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
A_FAMILY = ["A", "A90", "A80", "A70"]


def run_cmd(args: List[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        print("[CMD FAILED]", " ".join(args))
        print("[STDOUT]")
        print(result.stdout)
        print("[STDERR]")
        print(result.stderr)
        raise RuntimeError(f"command failed with returncode={result.returncode}")

    return result


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_fake_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "fake_mappo_checkpoint_for_boundary_smoke_only",
        "trained_model": False,
        "performance_claim_allowed": False,
        "purpose": "Validate checkpoint existence boundary; do not use for paper results.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_mappo_boundary_rollout(
    *,
    scenario_index: Path,
    output_root: Path,
    checkpoint_path: Path,
    limit: int,
    seeds: str,
    write_parquet: bool,
) -> Dict[str, Any]:
    cmd = [
        PY,
        "05_training/run_causal_rollout.py",
        "--a-family-bridge",
        "--scenario-index",
        str(scenario_index),
        "--output-root",
        str(output_root),
        "--limit",
        str(limit),
        "--conditions",
        "A,A90,A80,A70",
        "--seeds",
        str(seeds),
        "--policy-kind",
        "mappo",
        "--checkpoint-path",
        str(checkpoint_path),
        "--require-existing-checkpoint",
    ]

    if write_parquet:
        cmd.append("--write-parquet")

    result = run_cmd(cmd)

    entry_manifest = output_root / "entry_manifest.json"
    writer_manifest = output_root / "manifest.json"
    summary_json = output_root / "summary.json"
    rows_jsonl = output_root / "combined_a_family_window_rollup.jsonl"

    for path in [entry_manifest, writer_manifest, summary_json, rows_jsonl]:
        if not path.exists():
            raise RuntimeError(f"expected rollout artifact missing: {path}")

    summary = load_json(summary_json)
    if not summary.get("passed", False):
        raise RuntimeError(f"rollout summary failed: {summary}")

    return {
        "stdout": result.stdout,
        "entry_manifest": str(entry_manifest),
        "writer_manifest": str(writer_manifest),
        "summary_json": str(summary_json),
        "rows_jsonl": str(rows_jsonl),
        "summary": summary,
    }


def validate_mappo_boundary_rows(rows_jsonl: Path, checkpoint_path: Path) -> Dict[str, Any]:
    rows = read_jsonl(rows_jsonl)

    if not rows:
        raise RuntimeError(f"no rows found: {rows_jsonl}")

    violations: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        condition_id = str(row.get("condition_id"))
        policy_source = str(row.get("policy_source"))
        source_mode = str(row.get("source_mode"))
        qwen_trigger_rate = float(row.get("qwen_trigger_rate", 0.0))
        policy_checkpoint_path = str(row.get("policy_checkpoint_path"))

        problems = []

        if condition_id not in A_FAMILY:
            problems.append(f"unexpected condition_id={condition_id}")

        if policy_source != "mappo_policy":
            problems.append(f"policy_source must be mappo_policy, got {policy_source}")

        if "placeholder" in policy_source.lower():
            problems.append("placeholder appeared in policy_source")

        if not source_mode.startswith("causal_"):
            problems.append(f"source_mode must start with causal_, got {source_mode}")

        if "_mappo_policy_v1" not in source_mode:
            problems.append(f"source_mode must contain _mappo_policy_v1, got {source_mode}")

        if "stub" in source_mode.lower() or "placeholder" in source_mode.lower() or "smoke" in source_mode.lower():
            problems.append(f"source_mode contains placeholder/smoke marker: {source_mode}")

        if abs(qwen_trigger_rate) > 1e-12:
            problems.append(f"qwen_trigger_rate must be 0.0, got {qwen_trigger_rate}")

        if Path(policy_checkpoint_path).as_posix() != checkpoint_path.as_posix():
            problems.append(
                f"policy_checkpoint_path mismatch: got {policy_checkpoint_path}, expected {checkpoint_path.as_posix()}"
            )

        if problems:
            violations.append(
                {
                    "row_index": idx,
                    "condition_id": condition_id,
                    "problems": problems,
                }
            )

    condition_ids = sorted(set(str(row.get("condition_id")) for row in rows))
    source_modes = sorted(set(str(row.get("source_mode")) for row in rows))
    policy_sources = sorted(set(str(row.get("policy_source")) for row in rows))

    summary = {
        "row_count": len(rows),
        "condition_ids": condition_ids,
        "source_modes": source_modes,
        "policy_sources": policy_sources,
        "violation_count": len(violations),
        "violations": violations,
        "passed": len(violations) == 0,
    }

    if violations:
        raise RuntimeError(f"MAPPO boundary row validation failed: {summary}")

    return summary


def run_canonical_for_condition(
    *,
    condition_id: str,
    rollout_root: Path,
    contract: Path,
    scenario_index: Path,
    smoke: bool,
) -> Dict[str, Any]:
    input_root = rollout_root / condition_id
    output_root = input_root / "canonical_eval"

    cmd = [
        PY,
        "05_training/evaluation/canonical_kpi_aggregator.py",
        "--mode",
        "official_rollup",
        "--contract",
        str(contract),
        "--input-root",
        str(input_root),
        "--scenario-index",
        str(scenario_index),
        "--output-root",
        str(output_root),
    ]

    if smoke:
        cmd.append("--smoke")

    run_cmd(cmd)

    expected = {
        "validation": output_root / "official_rollup_input_validation.json",
        "kpi_by_window": output_root / "kpi_by_window.parquet",
        "kpi_by_seed": output_root / "kpi_by_seed.parquet",
        "kpi_by_time_band": output_root / "kpi_by_time_band.parquet",
        "kpi_overall": output_root / "kpi_overall.json",
        "manifest": output_root / "aggregation_manifest.json",
    }

    missing = [str(path) for path in expected.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"canonical outputs missing for {condition_id}: {missing}")

    manifest = load_json(expected["manifest"])
    overall = load_json(expected["kpi_overall"])

    if not bool(manifest.get("causal_comparison_allowed", False)):
        raise RuntimeError(f"expected causal_comparison_allowed=True for fake mappo boundary: {condition_id}")

    return {
        "condition_id": condition_id,
        "output_root": str(output_root),
        "files": {key: str(value) for key, value in expected.items()},
        "row_counts": manifest.get("row_counts", {}),
        "overall_n_windows_total": overall.get("n_windows_total"),
        "causal_comparison_allowed": manifest.get("causal_comparison_allowed"),
        "passed": True,
    }


def run_step34(
    *,
    scenario_index: Path,
    contract: Path,
    output_root: Path,
    limit: int,
    seeds: str,
    write_parquet: bool,
    smoke: bool,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    fake_checkpoint = create_fake_checkpoint(
        output_root / "mock_checkpoints" / "fake_mappo_boundary.pt"
    )

    rollout_root = output_root / "rollout"

    rollout = run_mappo_boundary_rollout(
        scenario_index=scenario_index,
        output_root=rollout_root,
        checkpoint_path=fake_checkpoint,
        limit=limit,
        seeds=seeds,
        write_parquet=write_parquet,
    )

    boundary_summary = validate_mappo_boundary_rows(
        rows_jsonl=Path(rollout["rows_jsonl"]),
        checkpoint_path=fake_checkpoint,
    )

    canonical_results: List[Dict[str, Any]] = []

    for condition_id in A_FAMILY:
        canonical_results.append(
            run_canonical_for_condition(
                condition_id=condition_id,
                rollout_root=rollout_root,
                contract=contract,
                scenario_index=scenario_index,
                smoke=smoke,
            )
        )

    manifest = {
        "artifact_version": "fake_checkpoint_mappo_boundary_smoke_v1",
        "important": "This is not a performance claim. Fake checkpoint is for boundary validation only.",
        "fake_checkpoint": str(fake_checkpoint),
        "scenario_index": str(scenario_index),
        "contract": str(contract),
        "output_root": str(output_root),
        "limit": limit,
        "seeds": seeds,
        "write_parquet": write_parquet,
        "smoke": smoke,
        "rollout": rollout,
        "boundary_summary": boundary_summary,
        "canonical_results": canonical_results,
        "passed": bool(boundary_summary["passed"]) and all(r["passed"] for r in canonical_results),
    }

    write_json(output_root / "step34_manifest.json", manifest)

    if not manifest["passed"]:
        raise RuntimeError(f"Step 34 failed: {manifest}")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario-index",
        default="artifacts/baseline_v1/B1_noop/scenario_index.parquet",
    )
    parser.add_argument(
        "--contract",
        default="artifacts/baseline_v1/baseline_contract.json",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/step34_fake_checkpoint_mappo_boundary_smoke",
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()

    manifest = run_step34(
        scenario_index=Path(args.scenario_index),
        contract=Path(args.contract),
        output_root=Path(args.output_root),
        limit=int(args.limit),
        seeds=str(args.seeds),
        write_parquet=bool(args.write_parquet),
        smoke=bool(args.smoke),
    )

    print("[OK] Step 34 fake-checkpoint MAPPO boundary smoke completed")
    print(f"[OK] output_root : {args.output_root}")
    print(f"[OK] manifest    : {Path(args.output_root) / 'step34_manifest.json'}")
    print(f"[OK] fake_ckpt   : {manifest['fake_checkpoint']}")
    print(f"[OK] passed      : {manifest['passed']}")

    if args.self_test:
        print("[OK] run_fake_checkpoint_mappo_boundary_smoke_v1 self-test passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
