#!/usr/bin/env python3
"""
Self-test for Step 105 route-aware v2 pipeline readiness gate.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from validate_route_aware_v2_pipeline_step105 import validate_pipeline


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else ["id"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")


def build_synthetic_project(root: Path) -> None:
    audit_root = root / "artifacts" / "daegu_bis_api_audit"

    write_json(
        audit_root / "causal_simulator_v2_contract_step100" / "causal_simulator_v2_contract.json",
        {
            "contract_status": "READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD",
            "field_classification": {
                "actual_headway": "not_observed",
                "actual_arrival_departure_time": "not_observed",
                "actual_dwell": "not_observed",
            },
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    )

    write_json(
        audit_root / "route_aware_minimal_simulator_step101" / "route_aware_minimal_simulator_manifest.json",
        {
            "scaffold_status": "READY_FOR_STEP102_ROLLOUT_WRITER_SCAFFOLD",
            "trace_rows": 24,
        },
    )

    s102 = audit_root / "route_aware_rollout_writer_step102"
    write_json(
        s102 / "route_aware_rollout_writer_manifest.json",
        {
            "rollout_status": "READY_FOR_STEP103_CANONICAL_KPI_SCAFFOLD",
            "parquet_ready": True,
        },
    )
    write_csv(s102 / "raw_events.csv", [{"a": 1}, {"a": 2}])
    write_csv(s102 / "window_rollup.csv", [{"condition_id": "A", "seed": 1, "window_id": "w1"}])

    s103 = audit_root / "route_aware_canonical_kpi_step103"
    write_json(
        s103 / "route_aware_canonical_kpi_step103_manifest.json",
        {
            "integration_status": "READY_FOR_STEP104_PROJECT_LOG_RUNBOOK_UPDATE",
            "causal_allowed": False,
            "kpi_by_window": 24,
            "kpi_by_seed": 12,
            "kpi_by_time_band": 12,
        },
    )
    canonical = s103 / "canonical_eval"
    canonical.mkdir(parents=True, exist_ok=True)
    # Empty parquet placeholders are enough for existence checks in a minimal env.
    (canonical / "kpi_by_window.parquet").write_bytes(b"placeholder")
    (canonical / "kpi_by_seed.parquet").write_bytes(b"placeholder")
    (canonical / "kpi_by_time_band.parquet").write_bytes(b"placeholder")
    write_json(canonical / "kpi_overall.json", {"causal_comparison_allowed": False})

    # Step 104 docs
    (root / "project_log.md").write_text("# log\n", encoding="utf-8")
    runbook = root / "05_training" / "adapters" / "route_aware_v2_pipeline_runbook_step104.md"
    runbook.parent.mkdir(parents=True, exist_ok=True)
    runbook.write_text("# runbook\n", encoding="utf-8")
    write_json(
        audit_root / "project_log_runbook_step104" / "project_log_runbook_step104_manifest.json",
        {"status": "PASS"},
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        build_synthetic_project(root)

        payload = validate_pipeline(
            project_root=root,
            output_dir=Path("artifacts/daegu_bis_api_audit/route_aware_v2_pipeline_readiness_step105"),
            require_step104=True,
        )

        assert payload["audit_status"] == "PASS", payload
        assert payload["readiness_status"] == "READY_FOR_STEP106_MINIMAL_QUEUE_DEMAND_SCAFFOLD", payload
        assert payload["hard_failure_count"] == 0, payload
        assert payload["claim_guards"]["actual_headway"] == "not_observed"
        assert payload["claim_guards"]["causal_performance_claim_allowed"] is False

        out = root / "artifacts" / "daegu_bis_api_audit" / "route_aware_v2_pipeline_readiness_step105"
        assert (out / "route_aware_v2_pipeline_readiness_step105.json").exists()
        assert (out / "route_aware_v2_pipeline_readiness_step105.md").exists()
        assert (out / "route_aware_v2_pipeline_readiness_matrix_step105.csv").exists()

    print("[OK] Step 105 route-aware v2 pipeline readiness self-test PASS")


if __name__ == "__main__":
    main()
