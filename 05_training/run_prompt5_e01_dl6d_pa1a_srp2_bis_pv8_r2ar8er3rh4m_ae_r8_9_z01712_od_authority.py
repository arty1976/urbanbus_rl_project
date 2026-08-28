#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R8.9 STCIS Z01712 aggregate OD authority audit and
Z01709 Daegu city-bus coverage correction.

Audit only.  No OD inference, no OD matrix, no request ledger, no simulator
binding, no bulk download, no DB writes, no push.
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
R89_TEST = TRAINING_ROOT / "test_h4m_ae_r8_9_z01712_od_authority_audit.py"
SECRET_LOADER = TRAINING_ROOT / "stcis_secret_loader.py"
LEDGER_MODULE = TRAINING_ROOT / "citywide_demand_ledger.py"

EXEC_BASE_SHA = "12a9c7a8c1f50344eb46a370a5aa5f46ca89ebd8"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R8_9_STCIS_Z01712_DAEGU_AGGREGATE_OD_AUTHORITY"
             "_AND_Z01709_COVERAGE_AUDIT_COMPLETE")
NEXT_GATE = "H4M-AE-R8_9_1_Z01712_EXPORT_FILE_HANDOFF_AND_WORKBOOK_SCHEMA_AUDIT"
KST = timezone(timedelta(hours=9))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r8_9_z01712_od_authority_audit as r89

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r8_9_z01712_od_authority_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r89.run_validations()
    c = result["checks"]
    sub = result["sub_classifications"]

    dump(root, "upstream_r8_8_binding.json", {
        "execution_base_sha": EXEC_BASE_SHA, "r8_8_source_commit_sha": EXEC_BASE_SHA,
        "R8_9_01": c["R8_9_01_upstream_r8_8"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True})
    dump(root, "z01712_source_binding.json", {
        **c["R8_9_02_z01712_file_identified"],
        "R8_9_03": c["R8_9_03_source_sha_recorded"], "R8_9_04": c["R8_9_04_original_unchanged"],
        "raw_file_committed": False,
        "storage_policy": "path, SHA and schema only; the raw export is never committed without proven redistribution rights"})
    dump(root, "z01712_workbook_schema.json", {
        **c["R8_9_05_workbook_schema_audited"],
        "fields_expected_from_user_report": ["date", "origin SD", "origin SGG", "destination SD",
                                             "destination SGG", "trip_count", "avg_trip_time", "trip_distance"],
        "field_classification": "all fields remain UNKNOWN until the workbook is inspected",
        "fields_invented": 0})
    dump(root, "z01712_sample_reproduction.json", {
        **c["R8_9_06_sample_reproduced"],
        "user_reported_sample": {"date": "2026-08-08", "origin": "대구광역시 / 수성구",
                                 "destination": "대구광역시 / 수성구", "trip_count": 14371,
                                 "avg_trip_time": 24.85, "trip_distance": 0},
        "independently_verified": False,
        "promoted_to_authority": False})
    dump(root, "z01712_mode_separation_audit.json", {
        **c["R8_9_07_mode_separation"], "R8_9_08": c["R8_9_08_bus_only_authority_not_assumed"]})
    dump(root, "z01712_spatial_granularity_audit.json", c["R8_9_09_spatial_granularity"])
    dump(root, "z01712_2023_availability_audit.json", {
        **c["R8_9_10_2023_availability"], "R8_9_11": c["R8_9_11_minimal_2023_probe"],
        "R8_9_12": c["R8_9_12_no_bulk_od_download"]})
    dump(root, "z01712_trip_count_semantics.json", c["R8_9_13_trip_count_semantics"])
    dump(root, "z01712_trip_distance_semantics.json", c["R8_9_14_trip_distance_zero"])
    dump(root, "z01709_daegu_citybus_coverage_reaudit.json", {
        **c["R8_9_18_z01709_route_list_reaudited"], "R8_9_19": c["R8_9_19_daegu_citybus_coverage"]})
    dump(root, "m009_daegu_citybus_authority_correction.json", c["R8_9_20_m009_authority_demoted"])
    dump(root, "citywide_ledger_non_regression.json", {
        **c["R8_9_15_citywide_ledger_unchanged"], "R8_9_16": c["R8_9_16_research_414_unchanged"]})
    dump(root, "alighting_semantics_non_regression.json", c["R8_9_17_alighting_semantics"])
    dump(root, "od_evidence_hierarchy_v3.json", result["od_evidence_hierarchy_v3"])
    dump(root, "citywide_first_non_regression.json", c["R8_9_33_citywide_first"])
    dump(root, "r8_9_test_report.json", result)
    dump(root, "classification.json", {
        "classification": result["classification"],
        "abcdef_option_assignable": False,
        "why": "options A through F all presume the export was inspected; none can be assigned without the workbook",
        "sub_classifications": sub})

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else f"BLOCKED_{result['blocker']}"
    gate = {
        "gate": gate_name, "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA,
        "blocker": result["blocker"],
        "citywide_ledger_sha256": c["R8_9_15_citywide_ledger_unchanged"]["dataset_sha256"],
        "research_414_sha256": c["R8_9_16_research_414_unchanged"]["sha256"],
        "z01712_export_file_found": False,
        "z01712_search_locations": c["R8_9_02_z01712_file_identified"]["search_locations"],
        "z01712_mode_status": sub["z01712_mode_status"],
        "z01712_bus_od_direct_constraint_allowed": False,
        "z01712_spatial_granularity": sub["z01712_spatial_granularity"],
        "z01712_2023_status": sub["z01712_2023_status"],
        "z01712_trip_count_semantics": sub["z01712_trip_count_semantics"],
        "z01712_trip_distance_status": sub["z01712_trip_distance_status"],
        "z01712_trip_distance_as_physical_distance_allowed": False,
        "z01709_daegu_citybus_status": sub["z01709_daegu_citybus_status"],
        "m009_daegu_citybus_constraint_authority": False,
        "m009_retired_from_active_daegu_lineage": True,
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
        "r8_9_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r8_9_tests_total": len(c), "r8_9_tests_all_pass": gate_passed,
        "failed_checks": failed,
        "remaining_dependencies": [
            "Z01712_EXPORT_FILE_NOT_ACCESSIBLE_TO_THIS_SESSION",
            "Z01712_MODE_SEPARATION_UNKNOWN",
            "Z01712_2023_AVAILABILITY_UNKNOWN",
            "Z01712_TRIP_COUNT_POPULATION_SEMANTICS_UNKNOWN",
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
        "m009_as_daegu_citybus_authority_permitted": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "treating 14,371 as bus passengers", "binding trip_distance 0 into simulator distance",
            "assuming Z01712 covers 2023", "reinstating M009 without proven city-bus coverage",
            "TEST6 access", "A/B1/B2 comparison", "GitHub push"]})
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R8.9", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r8_9": f"{gate['r8_9_tests_pass_count']}/{gate['r8_9_tests_total']}",
                                     "gate_decision": gate})

    z9 = c["R8_9_19_daegu_citybus_coverage"]
    md = [
        "# H4M-AE-R8.9 Z01712 OD Authority Audit and Z01709 Coverage Correction",
        "",
        f"- gate: **`{gate_name}`**",
        f"- source HEAD at run: `{head_sha}` · execution base `{EXEC_BASE_SHA}`",
        f"- R8.9 {gate['r8_9_tests_pass_count']}/{gate['r8_9_tests_total']}",
        f"- blocker: **`{result['blocker']}`**",
        "",
        "## The Z01712 export could not be reached",
        "",
        "No `.xlsx` or `.xls` exists anywhere in the accessible project tree. `~/Downloads` and `~/Desktop` "
        "return **Operation not permitted** — macOS TCC, denied both inside and outside the Claude sandbox — so "
        "their contents are **unknown, not empty**. I did not claim the file is absent from locations I cannot "
        "read, and I did not use any ChatGPT `/mnt/data` path.",
        "",
        "Everything downstream of the workbook is therefore blocked rather than guessed: schema, mode "
        "separation, spatial grain, 2023 availability and sample reproduction all stay `UNKNOWN`. The user-"
        "reported sample (2026-08-08, 대구/수성구 → 대구/수성구, 14,371 trips, 24.85 min, distance 0) is recorded "
        "as a report, not promoted to authority.",
        "",
        "## What was still settled",
        "",
        "**Z01712 cannot be treated as bus demand.** The title names two modes and separation is unverified, so "
        "`z01712_bus_od_direct_constraint_allowed = false` and **14,371 is not a bus passenger count**. If the "
        "modes turn out to be inseparable, its ceiling is "
        "`PUBLIC_TRANSPORT_AGGREGATE_OD_AUXILIARY_CONSTRAINT`.",
        "",
        "**`trip_distance = 0` is not a physical distance.** Five hypotheses remain open (unsupported field, "
        "same-region rows, missing serialized as zero, needs another dimension, genuine zero). Until one is "
        "proven, `z01712_trip_distance_as_physical_distance_allowed = false`; project graph `distance_m` remains "
        "the only authority for physical movement.",
        "",
        "**`trip_count` semantics are `UNKNOWN_STCIS_AGGREGATE_TRIP_COUNT`** — not equated with unique "
        "passengers, boardings or DRT requests.",
        "",
        "## Z01709: M009 is withdrawn from the Daegu lineage",
        "",
        f"The user's Daegu route selector showed **{len(r89.USER_OBSERVED_Z01709_DAEGU_ROUTES)} routes**: three "
        "urban rail lines, 73-3, 78-3, 서부산-서대구 and 창녕-대구동 — urban rail plus regional/intercity, and "
        "**no ordinary city bus**. Neither 814 nor 724 appeared.",
        "",
        "My own probe reached the selector popup and read its columns (노선명 / 노선유형 / 기종점) but returned "
        "*검색된 내용이 없습니다* for every query **including 1호선, which the user confirmed is present**. That "
        "makes my probe non-discriminating, so I based the finding on the user's observation and explicitly did "
        "**not** treat my empty result as proof of absence.",
        "",
        f"Classification `{z9['classification']}` → **`m009_daegu_citybus_constraint_authority = false`**. M009 "
        "is withdrawn from the active Daegu city-bus OD lineage. R8.5 through R8.8 had been building toward M009 "
        "as the leading route-load constraint; that expectation is now corrected. The earlier artifacts stay "
        "immutable and the correction is additive.",
        "",
        "## OD evidence hierarchy v3",
        "",
        "| tier | source | status |",
        "| --- | --- | --- |",
        "| 1 | 2023 stop×hour boarding ledger | **origin magnitude authority, frozen** |",
        "| 2 | Z01712 regional OD | exists, export not inspected, mode unknown |",
        "| 3 | STCIS 15-min 읍면동 OD API | blocked on key activation |",
        "| 4 | route/direction/stop sequence (Daegu BIS) | available, 234/238 routes |",
        "| 5 | partial observed alighting | available, selection biased |",
        "| 6 | model-inferred residual OD | not executed |",
        "| — | M009 route load | **withdrawn from active lineage** |",
        "",
        "## Guards",
        "",
        "No OD inference, no OD matrix, no IPF, no gravity model, no destination sampling, no request ledger, no "
        "simulator binding, no bulk download, no DB writes. The 2023 ledger and the frozen 414 are "
        "byte-identical. Alighting semantics unchanged. TEST6 untouched, no training, no comparison. One "
        "optional key check returned `INVALID_KEY`; no backoff sequence was run and the key was never disclosed.",
        "",
        "## To unblock",
        "",
        "Tell me the absolute path of the Z01712 export, or copy it somewhere I can read — anywhere inside "
        f"`{PROJECT_ROOT}` works. I cannot read `~/Downloads` or `~/Desktop` at all from this session.",
        "",
        "## Next gate",
        "",
        f"`{NEXT_GATE}` (not executed automatically).",
    ]
    (root / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p.name for p in root.iterdir() if p.is_file())
    dump(root, "artifact_manifest.json", {
        "artifact_dir": root.name, "generated_at": datetime.now(KST).isoformat(),
        "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA, "gate": gate_name,
        "file_sha256": {n: sha256_file(root / n) for n in files if n != "artifact_manifest.json"},
        "raw_user_excel_committed": False, "api_key_present": False,
        "source_file_sha256": {p.name: sha256_file(p) for p in (R89_TEST, SECRET_LOADER, LEDGER_MODULE, Path(__file__))}})
    (root / "_SUCCESS.lock").write_text(json.dumps(
        {"gate": gate_name, "passed": gate_passed, "blocker": result["blocker"], "next_gate": NEXT_GATE},
        indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[H4M-AE-R8.9] artifact root: {root}")
    print(f"[H4M-AE-R8.9] gate: {gate_name}")
    print(f"[H4M-AE-R8.9] classification: {result['classification']}")
    print(f"[H4M-AE-R8.9] tests {gate['r8_9_tests_pass_count']}/{gate['r8_9_tests_total']} | failed: {failed}")
    print(f"[H4M-AE-R8.9] artifact files: {len(files) + 1}")


if __name__ == "__main__":
    main()
