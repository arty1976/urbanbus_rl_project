"""
run_policy_interface_canonical_smoke_v1.py
==========================================

Step 38 canonical KPI smoke for policy-interface rollout rows.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.

Purpose
-------
This smoke test verifies that Step 37 policy-interface rollout rows can enter
the canonical KPI aggregation path.

Flow
----
run_a_family_policy_interface_scenario_writer_v1.py
-> A/A90/A80/A70 window_rollup files with action fields
-> canonical_kpi_aggregator.py official_rollup
-> condition-level canonical_eval outputs

Important
---------
This is not a performance claim.
Step 37 still uses conservative mock policy actions.
The purpose is pipeline compatibility and schema safety only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
A_FAMILY = ["A", "A90", "A80", "A70"]

REQUIRED_ACTION_FIELDS = [
    "action_version",
    "dispatch_delta",
    "hold_seconds",
    "skip_stop_flag",
    "target_headway_ratio",
    "policy_debug",
    "policy_action_debug",
    "policy_interface_version",
    "performance_claim_allowed",
]


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def parse_policy_action_debug(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)

    try:
        return json.loads(str(value))
    except Exception:
        return {}


def validate_action_fields(rows_jsonl: Path) -> Dict[str, Any]:
    rows = read_jsonl(rows_jsonl)

    if not rows:
        raise RuntimeError(f"empty rollout rows: {rows_jsonl}")

    violations: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        problems: List[str] = []

        condition_id = str(row.get("condition_id"))
        policy_source = str(row.get("policy_source"))
        source_mode = str(row.get("source_mode"))
        qwen_trigger_rate = float(row.get("qwen_trigger_rate", 0.0))

        for field in REQUIRED_ACTION_FIELDS:
            if field not in row:
                problems.append(f"missing action/interface field: {field}")

        if condition_id not in A_FAMILY:
            problems.append(f"unexpected condition_id={condition_id}")

        if policy_source != "mappo_policy":
            problems.append(f"policy_source must be mappo_policy, got {policy_source}")

        if not source_mode.startswith("causal_"):
            problems.append(f"source_mode must start with causal_, got {source_mode}")

        if "_mappo_policy_v1" not in source_mode:
            problems.append(f"source_mode must contain _mappo_policy_v1, got {source_mode}")

        lower_mode = source_mode.lower()
        if "placeholder" in lower_mode or "stub" in lower_mode or "smoke" in lower_mode:
            problems.append(f"source_mode contains placeholder/stub/smoke marker: {source_mode}")

        if abs(qwen_trigger_rate) > 1e-12:
            problems.append(f"qwen_trigger_rate must be 0.0, got {qwen_trigger_rate}")

        if bool(row.get("performance_claim_allowed")) is not False:
            problems.append("performance_claim_allowed must be False for Step 38 mock-action rows")

        policy_debug = parse_policy_action_debug(row.get("policy_action_debug"))
        if policy_debug.get("mock_action") is not True:
            problems.append("policy_action_debug.mock_action must be True")

        if policy_debug.get("performance_claim_allowed") is not False:
            problems.append("policy_action_debug.performance_claim_allowed must be False")

        if problems:
            violations.append(
                {
                    "row_index": idx,
                    "condition_id": condition_id,
                    "window_id": row.get("window_id"),
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
        raise RuntimeError(f"policy interface action field validation failed: {summary}")

    return summary


def run_policy_interface_writer(
    *,
    scenario_index: Path,
    output_root: Path,
    limit: int,
    seeds: str,
    write_parquet: bool,
) -> Dict[str, Any]:
    cmd = [
        PY,
        "05_training/rollouts/run_a_family_policy_interface_scenario_writer_v1.py",
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
    ]

    if write_parquet:
        cmd.append("--write-parquet")

    run_cmd(cmd)

    manifest_path = output_root / "manifest.json"
    action_summary_path = output_root / "action_summary.json"
    rows_jsonl = output_root / "combined_policy_interface_window_rollup.jsonl"

    for path in [manifest_path, action_summary_path, rows_jsonl]:
        if not path.exists():
            raise RuntimeError(f"expected writer output missing: {path}")

    manifest = load_json(manifest_path)

    if not manifest.get("passed", False):
        raise RuntimeError(f"Step 37 writer manifest failed: {manifest}")

    action_summary = load_json(action_summary_path)

    if not action_summary.get("passed", False):
        raise RuntimeError(f"Step 37 writer action summary failed: {action_summary}")

    field_summary = validate_action_fields(rows_jsonl)

    return {
        "manifest_path": str(manifest_path),
        "action_summary_path": str(action_summary_path),
        "rows_jsonl": str(rows_jsonl),
        "manifest": manifest,
        "action_summary": action_summary,
        "field_summary": field_summary,
        "passed": True,
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
        raise RuntimeError(f"condition input root missing: {input_root}")

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
        "causal_comparison_allowed": manifest.get("causal_comparison_allowed"),
        "passed": True,
    }


def summarize_canonical_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    conditions = []

    for result in results:
        manifest = result.get("manifest", {})
        overall = result.get("overall", {})

        conditions.append(
            {
                "condition_id": result.get("condition_id"),
                "output_root": result.get("output_root"),
                "row_counts": manifest.get("row_counts", {}),
                "causal_comparison_allowed": manifest.get("causal_comparison_allowed"),
                "overall_n_windows_total": overall.get("n_windows_total"),
                "overall_condition_ids": overall.get("condition_ids"),
                "passed": result.get("passed", False),
            }
        )

    return {
        "condition_count": len(conditions),
        "conditions": conditions,
        "passed": all(item["passed"] for item in conditions),
    }


def run_step38(
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

    rollout_root = output_root / "rollout"

    writer = run_policy_interface_writer(
        scenario_index=scenario_index,
        output_root=rollout_root,
        limit=limit,
        seeds=seeds,
        write_parquet=write_parquet,
    )

    canonical_results = []

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

    canonical_summary = summarize_canonical_results(canonical_results)

    manifest = {
        "artifact_version": "policy_interface_canonical_smoke_v1",
        "important": "This is not a performance claim. Step 38 validates mock-action rollout compatibility with canonical KPI aggregation.",
        "scenario_index": str(scenario_index),
        "contract": str(contract),
        "output_root": str(output_root),
        "limit": limit,
        "seeds": seeds,
        "write_parquet": write_parquet,
        "smoke": smoke,
        "writer": writer,
        "canonical_summary": canonical_summary,
        "canonical_results": canonical_results,
        "passed": bool(writer["passed"]) and bool(canonical_summary["passed"]),
    }

    write_json(output_root / "step38_manifest.json", manifest)

    if not manifest["passed"]:
        raise RuntimeError(f"Step 38 failed: {manifest}")

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
        default="artifacts/step38_policy_interface_canonical_smoke",
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()

    manifest = run_step38(
        scenario_index=Path(args.scenario_index),
        contract=Path(args.contract),
        output_root=Path(args.output_root),
        limit=int(args.limit),
        seeds=str(args.seeds),
        write_parquet=bool(args.write_parquet),
        smoke=bool(args.smoke),
    )

    print("[OK] Step 38 policy-interface canonical KPI smoke completed")
    print(f"[OK] output_root : {args.output_root}")
    print(f"[OK] manifest    : {Path(args.output_root) / 'step38_manifest.json'}")
    print(f"[OK] passed      : {manifest['passed']}")

    if args.self_test:
        print("[OK] run_policy_interface_canonical_smoke_v1 self-test passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
