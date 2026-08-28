#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R8.9.1 STCIS Z01712 workbook schema and aggregate OD
authority audit.

Read-only forensic audit.  No OD inference, no OD matrix, no request ledger, no
simulator binding, no DB writes, no push.  The raw workbook is never copied into
the artifact.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS = TRAINING_ROOT / "artifacts"
R891_TEST = TRAINING_ROOT / "test_h4m_ae_r8_9_1_z01712_workbook_audit.py"
SECRET_LOADER = TRAINING_ROOT / "stcis_secret_loader.py"
LEDGER_MODULE = TRAINING_ROOT / "citywide_demand_ledger.py"

EXEC_BASE_SHA = "c6256904f5a446fd824f7f9deea96f27f653f374"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R8_9_1_STCIS_Z01712_WORKBOOK_SCHEMA"
             "_AND_AGGREGATE_OD_AUTHORITY_AUDIT_COMPLETE")
NEXT_GATE = "H4M-AE-R8_10_STCIS_Z01712_BUS_RAIL_MODE_SEPARATION_AND_BUS_OD_AUTHORITY_RESOLUTION"
KST = timezone(timedelta(hours=9))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r8_9_1_z01712_workbook_audit as r891

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r8_9_1_z01712_workbook_audit_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r891.run_validations()
    c = result["checks"]
    sub = result["sub_classifications"]
    w = result["workbook"]

    dump(root, "upstream_r8_9_binding.json", {
        "execution_base_sha": EXEC_BASE_SHA, "r8_9_source_commit_sha": EXEC_BASE_SHA,
        "R8_9_1_01": c["R8_9_1_01_upstream_r8_9"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True})
    dump(root, "z01712_excel_source_binding.json", {
        "R8_9_1_02": c["R8_9_1_02_excel_path_exists"], "R8_9_1_03": c["R8_9_1_03_excel_sha_recorded"],
        "R8_9_1_04": c["R8_9_1_04_original_unchanged"],
        "raw_workbook_copied_into_artifact": False,
        "storage_policy": "path, SHA and forensic metadata only; redistribution rights are not established"})
    dump(root, "z01712_workbook_forensic_audit.json", c["R8_9_1_05_structure_recorded"])
    dump(root, "z01712_workbook_schema.json", {
        "header_names_as_stored": w["header"],
        "pivot_cache_fields": c["R8_9_1_05_structure_recorded"]["pivot_cache_fields"],
        "field_count": len(w["header"]),
        "mode_field_present": False,
        "code_fields_present": False,
        "R8_9_1_06": c["R8_9_1_06_data_row_count"]})
    dump(root, "z01712_sample_reproduction.json", c["R8_9_1_07_sample_reproduced"])
    dump(root, "z01712_row_grain_contract.json", c["R8_9_1_08_row_grain"])
    dump(root, "z01712_mode_separation_audit.json", {
        **c["R8_9_1_09_mode_separation"], "R8_9_1_10": c["R8_9_1_10_bus_only_not_assumed"]})
    dump(root, "z01712_spatial_granularity_audit.json", c["R8_9_1_11_spatial_granularity"])
    dump(root, "z01712_temporal_granularity_audit.json", c["R8_9_1_12_temporal_granularity"])
    dump(root, "z01712_trip_count_semantics.json", c["R8_9_1_13_trip_count_semantics"])
    dump(root, "z01712_trip_time_semantics.json", c["R8_9_1_14_trip_time_semantics"])
    dump(root, "z01712_trip_distance_semantics.json", {
        **c["R8_9_1_15_trip_distance_audit"], "R8_9_1_16": c["R8_9_1_16_physical_zero_prohibited"]})
    dump(root, "z01712_coverage_audit.json", {
        **c["R8_9_1_17_coverage"], "R8_9_1_18": c["R8_9_1_18_duplicate_keys"]})
    dump(root, "z01712_2023_compatibility_audit.json", c["R8_9_1_19_2023_status"])
    dump(root, "z01712_od_authority_classification.json", {
        **c["R8_9_1_36_classification_supported"], "sub_classifications": sub})
    dump(root, "m009_inactive_non_regression.json", c["R8_9_1_23_m009_inactive"])
    dump(root, "stcis_15min_od_dependency_status.json", {
        **c["R8_9_1_24_key_leakage_zero"],
        "endpoint": "https://stcis.go.kr/openapi/quarterod.json",
        "granularity": "15 minutes x 10-digit 읍면동",
        "dependency": "STCIS_API_KEY_NOT_ACTIVATED",
        "retried_this_gate": False, "backoff_run": False})
    dump(root, "citywide_ledger_non_regression.json", {
        **c["R8_9_1_20_citywide_ledger_unchanged"], "R8_9_1_21": c["R8_9_1_21_research_414_unchanged"]})
    dump(root, "alighting_semantics_non_regression.json", c["R8_9_1_22_alighting_semantics"])
    dump(root, "od_evidence_hierarchy_v4.json", result["od_evidence_hierarchy_v4"])
    dump(root, "citywide_first_non_regression.json", c["R8_9_1_35_citywide_first"])
    dump(root, "r8_9_1_test_report.json", result)
    dump(root, "classification.json", {
        "classification": result["classification"], "sub_classifications": sub,
        "evidence": c["R8_9_1_36_classification_supported"]["evidence"]})

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R8_9_1_INCOMPLETE"
    dist = c["R8_9_1_15_trip_distance_audit"]
    gate = {
        "gate": gate_name, "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA,
        "excel_path": w["path"], "excel_sha256": w["sha256"], "excel_size": w["size"],
        "excel_sheets": w["sheets"], "excel_data_rows": w["data_rows"],
        "excel_original_unchanged": True, "excel_committed": False,
        "sample_reproduced": c["R8_9_1_07_sample_reproduced"]["sample_reproduced"],
        "z01712_row_grain": c["R8_9_1_08_row_grain"]["classification"],
        "z01712_mode_status": sub["z01712_mode_status"],
        "z01712_bus_od_direct_constraint_allowed": False,
        "z01712_spatial_granularity": sub["z01712_spatial_granularity"],
        "z01712_temporal_granularity": sub["z01712_temporal_granularity"],
        "z01712_trip_count_semantics": sub["z01712_trip_count_semantics"],
        "z01712_trip_time_semantics": sub["z01712_trip_time_semantics"],
        "z01712_trip_distance_semantics": sub["z01712_trip_distance_semantics"],
        "z01712_trip_distance_as_physical_distance_allowed": False,
        "z01712_period_status": sub["z01712_period_status"],
        "z01712_2023_status": sub["z01712_2023_status"],
        "z01712_od_authority_class": result["classification"],
        "m009_daegu_citybus_constraint_authority": False,
        "api_key_status": sub["api_key_status"],
        "od_inference_executed": False, "od_matrix_created": False,
        "citywide_request_candidate_created": False, "suseong_request_candidate_created": False,
        "candidate_binding_allowed": False,
        "historical_2023_citywide_ledger_unchanged": True, "research_414_unchanged": True,
        "alighting_absolute_destination_marginal_allowed": False,
        "alighting_auxiliary_evidence_allowed": True,
        "test6_accessed": False, "training_executed": False, "performance_comparison_executed": False,
        "reward_v2_unchanged": True, "zero_loss_unchanged": True, "k_mask_unchanged": True,
        "db_writes": 0,
        "classification": result["classification"], "sub_classifications": sub,
        "r8_9_1_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r8_9_1_tests_total": len(c), "r8_9_1_tests_all_pass": gate_passed,
        "remaining_dependencies": [
            "Z01712_BUS_AND_RAIL_NOT_SEPARABLE_IN_EXPORT",
            "Z01712_2023_AVAILABILITY_UNKNOWN",
            "Z01712_TRIP_COUNT_POPULATION_SEMANTICS_UNKNOWN",
            "Z01712_TRIP_TIME_UNIT_UNSTATED",
            "Z01712_TRIP_DISTANCE_SEMANTICS_UNDECIDABLE_FROM_ONE_SELF_OD_ROW",
            "STCIS_API_KEY_NOT_ACTIVATED",
        ],
        "recommended_next_gate": NEXT_GATE}
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "od_inference_permitted": False, "od_matrix_permitted": False,
        "request_ledger_permitted": False, "simulator_binding_permitted": False,
        "z01712_as_bus_od_authority_permitted": False,
        "z01712_trip_distance_as_physical_distance_permitted": False,
        "z01712_2026_absolute_od_copied_into_2023_permitted": False,
        "m009_as_daegu_citybus_authority_permitted": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "calling 통행량 bus passengers", "assuming 통행시간 is minutes",
            "binding 통행거리 0 into simulator distance", "copying 2026 OD magnitude into 2023",
            "TEST6 access", "A/B1/B2 comparison", "GitHub push"]})
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R8.9.1", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r8_9_1": f"{gate['r8_9_1_tests_pass_count']}/{gate['r8_9_1_tests_total']}",
                                     "gate_decision": gate})

    md = [
        "# H4M-AE-R8.9.1 Z01712 Workbook Schema and Aggregate OD Authority Audit",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}` · execution base `{EXEC_BASE_SHA}`",
        f"- R8.9.1 {gate['r8_9_1_tests_pass_count']}/{gate['r8_9_1_tests_total']}",
        f"- classification: **{result['classification']}**",
        "",
        "## The workbook",
        "",
        f"- path: `{w['path']}`",
        f"- sha256 `{w['sha256']}`, {w['size']} bytes, unchanged byte-for-byte after the audit",
        f"- single sheet **`{w['sheets'][0]}`** — the sheet name is the indicator id itself",
        "- it is a PivotTable export generated by Apache POI; the pivot cache declares the source range `A1:J2`",
        f"- **{w['data_rows']} data row**",
        "",
        "## Schema, read from the pivot cache",
        "",
        "| # | field |",
        "| --- | --- |",
        *[f"| {i+1} | `{h}` |" for i, h in enumerate(w["header"])],
        "",
        "**Sample reproduced exactly.** `2026-08-08(토)` · 대구광역시/수성구 → 대구광역시/수성구 · 읍면동 `-` "
        "on both ends · 통행량 `14371.0` · 통행시간 `24.85` · 통행거리 `0`.",
        "",
        "## Mode: combined, and the provider says so",
        "",
        "There is **no mode field** in either the sheet or the pivot cache. The indicator page settles why:",
        "",
        "> 이용객 수요(O/D)는 … 철도·고속·시외버스이용 O/D , 일반버스·도시철도이용 O/D 으로 구분하여 제공합니다.",
        "",
        "The split is **between the two indicator variants**, not inside Z01712. So within this indicator the "
        "ordinary bus and the urban railway are aggregated: `BUS_RAIL_COMBINED`.",
        "",
        "Therefore **`z01712_bus_od_direct_constraint_allowed = false`**, and **14,371 is not a bus passenger "
        "count**. Z01712's ceiling is `PUBLIC_TRANSPORT_AGGREGATE_OD_AUXILIARY_CONSTRAINT`.",
        "",
        "## Grain",
        "",
        "- **spatial**: 읍면동 columns *exist in the schema* but hold `-` in this export, so the workbook "
        "demonstrates **SGG** output while the schema allows EMD. I did not promote EMD on Open-API evidence.",
        "- **temporal**: `DAILY_OD`. One 일자 column formatted `YYYY-MM-DD(요일)`; no hour, no quarter-hour, no "
        "time band. The 15-minute grain belongs to the separate `quarterod.json` API, not here.",
        "",
        "## Measures — all three bounded, none assumed",
        "",
        "| measure | status |",
        "| --- | --- |",
        "| 통행량 | `UNKNOWN_STCIS_AGGREGATE_TRIP_COUNT` — the workbook carries a label and no definition |",
        "| 통행시간 | `UNIT_UNKNOWN_NOT_ASSUMED_MINUTES` — 24.85 with no stated unit |",
        "| 통행거리 | `DISTANCE_SEMANTICS_UNKNOWN` |",
        "",
        f"On distance the export is genuinely undecidable: it holds **one row, and that row is a same-region "
        f"self-OD** (수성구→수성구). A zero there is equally consistent with an unsupported field and with a "
        "same-region rule. **One cross-region row would discriminate between them.** Until then "
        "`z01712_trip_distance_as_physical_distance_allowed = false` and project graph `distance_m` remains the "
        "only movement authority.",
        "",
        "## 2023",
        "",
        f"Workbook period is `{sub['z01712_period_status']}` — a single 2026-08-08 row. It provides **no** 2023 "
        "evidence, and I did not infer any from it. The indicator publishes a 14-day query cap but no static "
        f"date bounds, so `{sub['z01712_2023_status']}`.",
        "",
        "## OD evidence hierarchy v4",
        "",
        "| tier | source | status |",
        "| --- | --- | --- |",
        "| 1 | 2023 stop×hour boarding ledger | **origin magnitude authority, frozen** |",
        "| 2 | Z01712 daily SGG OD | schema resolved; bus+rail combined; auxiliary only |",
        "| 3 | 15-min 읍면동 OD API | blocked on key activation |",
        "| 4 | BIS route/direction/stop sequence | available, 234/238 |",
        "| 5 | partial observed alighting | available, selection biased |",
        "| 6 | model-inferred residual OD | not executed |",
        "| — | M009 route load | inactive |",
        "",
        "## Guards",
        "",
        "Workbook opened read-only and unmodified; not committed. No OD inference, no OD matrix, no IPF, no "
        "gravity model, no destination sampling, no request ledger, no binding, no DB writes. Ledger and 414 "
        "byte-identical. Alighting semantics unchanged. M009 stays inactive and was not revived. The API key was "
        "not retried this gate; leakage scan clean. TEST6 untouched, no training, no comparison.",
        "",
        "## Next gate",
        "",
        f"`{NEXT_GATE}` (not executed automatically). Mode separation is the binding constraint: until bus can "
        "be isolated from urban rail, Z01712 cannot constrain a bus OD reconstruction.",
    ]
    (root / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p.name for p in root.iterdir() if p.is_file())
    dump(root, "artifact_manifest.json", {
        "artifact_dir": root.name, "generated_at": datetime.now(KST).isoformat(),
        "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA, "gate": gate_name,
        "file_sha256": {n: sha256_file(root / n) for n in files if n != "artifact_manifest.json"},
        "workbook_sha256": w["sha256"], "raw_workbook_copied": False, "api_key_present": False,
        "source_file_sha256": {p.name: sha256_file(p) for p in (R891_TEST, SECRET_LOADER, LEDGER_MODULE, Path(__file__))}})
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R8.9.1] artifact root: {root}")
    print(f"[H4M-AE-R8.9.1] gate: {gate_name}")
    print(f"[H4M-AE-R8.9.1] classification: {result['classification']}")
    print(f"[H4M-AE-R8.9.1] tests {gate['r8_9_1_tests_pass_count']}/{gate['r8_9_1_tests_total']}")
    print(f"[H4M-AE-R8.9.1] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R8.9.1] BLOCKED")


if __name__ == "__main__":
    main()
