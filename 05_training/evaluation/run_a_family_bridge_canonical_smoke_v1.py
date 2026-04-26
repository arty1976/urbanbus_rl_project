"""
run_a_family_bridge_canonical_smoke_v1.py
=========================================

Step 30 end-to-end smoke:
run_causal_rollout.py --a-family-bridge
  -> A/A90/A80/A70 window_rollup files
  -> canonical_kpi_aggregator.py official_rollup
  -> condition-level canonical_eval outputs

This is not a performance claim.
It validates that the bridge-generated A-family rollouts can enter the
canonical KPI aggregation path.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def run_bridge_rollout(
    *,
    scenario_index: Path,
    output_root: Path,
    limit: int,
    seeds: str,
    policy_kind: str,
    checkpoint_path: str,
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
        str(policy_kind),
    ]

    if policy_kind == "mappo":
        cmd.extend(["--checkpoint-path", str(checkpoint_path)])

    if write_parquet:
        cmd.append("--write-parquet")

    run_cmd(cmd)

    entry_manifest = output_root / "entry_manifest.json"
    writer_manifest = output_root / "manifest.json"
    summary_json = output_root / "summary.json"

    for path in [entry_manifest, writer_manifest, summary_json]:
        if not path.exists():
            raise RuntimeError(f"expected rollout artifact missing: {path}")

    summary = load_json(summary_json)
    if not summary.get("passed", False):
        raise RuntimeError(f"bridge rollout summary failed: {summary}")

    return {
        "entry_manifest": str(entry_manifest),
        "writer_manifest": str(writer_manifest),
        "summary_json": str(summary_json),
        "summary": summary,
    }


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

    if not input_root.exists():
        raise RuntimeError(f"input_root for condition missing: {input_root}")

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

    return {
        "condition_id": condition_id,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "files": {key: str(value) for key, value in expected.items()},
        "manifest": manifest,
        "overall": overall,
        "passed": True,
    }


def summarize_condition_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "condition_count": len(results),
        "conditions": [],
        "passed": True,
    }

    for result in results:
        manifest = result.get("manifest", {})
        overall = result.get("overall", {})

        out["conditions"].append(
            {
                "condition_id": result["condition_id"],
                "output_root": result["output_root"],
                "row_counts": manifest.get("row_counts", {}),
                "causal_comparison_allowed": manifest.get("causal_comparison_allowed"),
                "overall_n_windows_total": overall.get("n_windows_total"),
                "overall_condition_ids": overall.get("condition_ids"),
                "passed": result.get("passed", False),
            }
        )

        if not result.get("passed", False):
            out["passed"] = False

    return out


def run_step30(
    *,
    scenario_index: Path,
    contract: Path,
    output_root: Path,
    limit: int,
    seeds: str,
    policy_kind: str,
    checkpoint_path: str,
    write_parquet: bool,
    smoke: bool,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    rollout_root = output_root / "rollout"

    bridge = run_bridge_rollout(
        scenario_index=scenario_index,
        output_root=rollout_root,
        limit=limit,
        seeds=seeds,
        policy_kind=policy_kind,
        checkpoint_path=checkpoint_path,
        write_parquet=write_parquet,
    )

    condition_results: List[Dict[str, Any]] = []

    for condition_id in A_FAMILY:
        condition_results.append(
            run_canonical_for_condition(
                condition_id=condition_id,
                rollout_root=rollout_root,
                contract=contract,
                scenario_index=scenario_index,
                smoke=smoke,
            )
        )

    condition_summary = summarize_condition_results(condition_results)

    manifest = {
        "artifact_version": "a_family_bridge_canonical_smoke_v1",
        "scenario_index": str(scenario_index),
        "contract": str(contract),
        "output_root": str(output_root),
        "rollout_root": str(rollout_root),
        "limit": limit,
        "seeds": seeds,
        "policy_kind": policy_kind,
        "checkpoint_path": None if policy_kind == "placeholder" else checkpoint_path,
        "write_parquet": write_parquet,
        "smoke": smoke,
        "bridge": bridge,
        "condition_summary": condition_summary,
        "passed": bool(condition_summary["passed"]),
    }

    write_json(output_root / "step30_manifest.json", manifest)

    if not manifest["passed"]:
        raise RuntimeError(f"Step 30 failed: {manifest}")

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
        default="artifacts/step30_a_family_bridge_canonical_smoke",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--policy-kind",
        choices=["placeholder", "mappo"],
        default="placeholder",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/experiment_A_v1/checkpoints/best.pt",
    )
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()

    manifest = run_step30(
        scenario_index=Path(args.scenario_index),
        contract=Path(args.contract),
        output_root=Path(args.output_root),
        limit=int(args.limit),
        seeds=str(args.seeds),
        policy_kind=str(args.policy_kind),
        checkpoint_path=str(args.checkpoint_path),
        write_parquet=bool(args.write_parquet),
        smoke=bool(args.smoke),
    )

    print("[OK] Step 30 A-family bridge canonical smoke completed")
    print(f"[OK] output_root : {args.output_root}")
    print(f"[OK] manifest    : {Path(args.output_root) / 'step30_manifest.json'}")
    print(f"[OK] passed      : {manifest['passed']}")

    if args.self_test:
        print("[OK] run_a_family_bridge_canonical_smoke_v1 self-test passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
