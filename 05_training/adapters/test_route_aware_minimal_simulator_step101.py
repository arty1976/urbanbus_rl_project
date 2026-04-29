#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-test for Step 101 route-aware minimal simulator scaffold."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("route_aware_minimal_simulator_step101.py")
if not MODULE_PATH.exists():
    # When copied into the project tree, this test sits beside the module.
    MODULE_PATH = Path("05_training/adapters/route_aware_minimal_simulator_step101.py")


def load_module():
    spec = importlib.util.spec_from_file_location("route_aware_minimal_simulator_step101", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_getbs02_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"route_id": "R1", "direction_id": "0", "stop_id": "S1", "ordered_stop_sequence": 1, "route_no": "101", "stop_name": "Alpha"},
        {"route_id": "R1", "direction_id": "0", "stop_id": "S2", "ordered_stop_sequence": 2, "route_no": "101", "stop_name": "Beta"},
        {"route_id": "R1", "direction_id": "0", "stop_id": "S3", "ordered_stop_sequence": 3, "route_no": "101", "stop_name": "Gamma"},
        {"route_id": "R1", "direction_id": "0", "stop_id": "S4", "ordered_stop_sequence": 4, "route_no": "101", "stop_name": "Delta"},
        {"route_id": "R2", "direction_id": "1", "stop_id": "T1", "ordered_stop_sequence": 1, "route_no": "202", "stop_name": "One"},
        {"route_id": "R2", "direction_id": "1", "stop_id": "T2", "ordered_stop_sequence": 2, "route_no": "202", "stop_name": "Two"},
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        contract_path = root / "contract.json"
        csv_path = root / "getbs02_route_stop_sequence_normalized.csv"
        out_root = root / "out"

        write_json(contract_path, {
            "artifact_version": "causal_simulator_v2_contract_step100_v1",
            "contract_status": "READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD",
            "claim_guards": {
                "paper_level_claim_allowed": False,
                "causal_performance_claim_allowed": False,
            },
            "blocked_actual_observed_fields": {
                "actual_headway": "not_observed",
                "actual_arrival_departure_time": "not_observed",
                "actual_dwell": "not_observed",
            },
        })
        write_getbs02_csv(csv_path)

        # Direct function/class path.
        records = module.read_route_sequence_csv(csv_path)
        table = module.build_route_table(records)
        assert ("R1", "0") in table
        sim = module.RouteAwareMinimalSimulatorStep101(table, num_agents=2, max_steps=5)
        obs = sim.reset(seed=1, scenario_config={"route_id": "R1", "direction_id": "0"})
        assert obs["route_id"] == "R1"
        assert obs["claim_guards"]["paper_level_claim_allowed"] is False
        result = sim.step({0: 1, 1: "advance"})
        assert result.obs["agents"][0]["stop_order"] == 2
        assert result.info["actual_policy_claim_ready"] is False
        assert result.info["causal_performance_claim_allowed"] is False
        assert result.info["spacing_proxy"]["not_actual_headway"] is True

        manifest = module.run_step101(
            module.Step101Paths(contract_json=contract_path, route_sequence_csv=csv_path, output_root=out_root),
            num_agents=2,
            max_steps=5,
            seed=101,
            route_id="R1",
            direction_id="0",
        )
        assert manifest["audit_status"] == "PASS"
        assert manifest["scaffold_status"] == "READY_FOR_STEP102_ROLLOUT_WRITER_SCAFFOLD"
        assert manifest["route_sequence_readiness_summary"]["eligible_route_direction_count"] == 2
        assert manifest["smoke_rollout"]["trace_rows"] > 0
        assert manifest["claim_guards"]["actual_headway_observed"] is False
        assert (out_root / "route_aware_minimal_simulator_manifest.json").exists()
        assert (out_root / "route_aware_minimal_simulator_report.md").exists()
        assert (out_root / "route_sequence_readiness_step101.csv").exists()
        assert (out_root / "smoke_rollout_trace_step101.csv").exists()

        # CLI path.
        cli_out = root / "cli_out"
        completed = subprocess.run(
            [
                sys.executable,
                str(MODULE_PATH),
                "--contract-json", str(contract_path),
                "--route-sequence-csv", str(csv_path),
                "--output-root", str(cli_out),
                "--route-id", "R1",
                "--direction-id", "0",
                "--num-agents", "2",
                "--max-steps", "5",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        assert "[OK] Step 101 route-aware minimal simulator scaffold completed" in completed.stdout

    print("[OK] Step 101 route-aware minimal simulator scaffold self-test PASS")


if __name__ == "__main__":
    main()
