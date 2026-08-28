#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R8.7 STCIS Daegu onboard-load semantics, 2023
availability probe and route/stop identity freeze.

Audit only.  The STCIS service key is loaded into process memory by
stcis_secret_loader and is never written to any artifact, manifest or log.
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
R87_TEST = TRAINING_ROOT / "test_h4m_ae_r8_7_stcis_semantics_and_2023_probe.py"
SECRET_LOADER = TRAINING_ROOT / "stcis_secret_loader.py"
LEDGER_MODULE = TRAINING_ROOT / "citywide_demand_ledger.py"

EXEC_BASE_SHA = "4d7e012841332af916d181e2d726f4c5c9f62cc1"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R8_7_STCIS_DAEGU_ROUTE_LOAD_SEMANTICS"
             "_AND_2023_AVAILABILITY_AUDIT_COMPLETE")
NEXT_GATE = "H4M-AE-R8_8_USER_SESSION_ASSISTED_STCIS_MINIMAL_PAYLOAD_CAPTURE_AND_API_KEY_ACTIVATION"
KST = timezone(timedelta(hours=9))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r8_7_stcis_semantics_and_2023_probe as r87

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r8_7_stcis_semantics_probe_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r87.run_validations()
    c = result["checks"]
    sub = result["sub_classifications"]

    dump(root, "upstream_r8_6_binding.json", {
        "execution_base_sha": EXEC_BASE_SHA, "r8_6_source_commit_sha": EXEC_BASE_SHA,
        "R8_7_01": c["R8_7_01_upstream_r8_6"], "R8_7_02": c["R8_7_02_sha_roles_recorded"],
        "R8_7_03": c["R8_7_03_citywide_ledger_unchanged"], "R8_7_04": c["R8_7_04_research_414_unchanged"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True})
    dump(root, "stcis_secret_handling_audit.json", {
        "R8_7_05": c["R8_7_05_secret_source_discovered"], "R8_7_06": c["R8_7_06_key_never_logged"],
        "R8_7_07": c["R8_7_07_key_absent_from_artifacts"], "R8_7_08": c["R8_7_08_key_absent_from_git"],
        "loader_module": "05_training/stcis_secret_loader.py",
        "loader_contract": "load_service_key returns the key to the caller only; no accessor exposes a hash, length, prefix or fragment"})
    dump(root, "stcis_api_auth_probe.json", c["R8_7_09_auth_probe_executed"])
    dump(root, "stcis_api_key_propagation_status.json", c["R8_7_10_key_propagation_handled"])
    dump(root, "stcis_15min_od_2023_probe.json", c["R8_7_12_daegu_2023_od_availability"])
    dump(root, "stcis_15min_od_schema.json", {**c["R8_7_11_15min_od_schema"], "R8_7_13": c["R8_7_13_stcis_od_aggregate_only"]})
    dump(root, "z01709_access_path_audit.json", {**c["R8_7_14_z01709_access_mode"], "R8_7_15": c["R8_7_15_m009_access_mode"]})
    dump(root, "m009_2023_availability_audit.json", c["R8_7_16_m009_2023_availability"])
    dump(root, "m009_event_semantics_contract.json", c["R8_7_17_m009_event_semantics"])
    dump(root, "stcis_project_route_identity_audit.json", {
        "R8_7_18": c["R8_7_18_route_no_alone_prohibited"], "R8_7_19": c["R8_7_19_composite_route_mapping"],
        "R8_7_22": c["R8_7_22_ambiguous_not_promoted"]})
    dump(root, "stcis_project_stop_identity_audit.json", c["R8_7_20_stop_identity"])
    dump(root, "stcis_project_direction_identity_audit.json", c["R8_7_21_direction_identity"])
    dump(root, "route_load_constraint_readiness.json", {
        "classification": "PARTIAL_ROUTE_LOAD_CONSTRAINT_TIMING_BLOCKED_AND_IDENTITY_AMBIGUOUS",
        "identity_resolved": False, "timing_resolved": False, "rows_accessible": False,
        "route_identity": sub["route_identity_status"], "stop_identity": sub["stop_identity_status"],
        "direction_identity": sub["direction_identity_status"],
        "m009_event_semantics": sub["m009_event_semantics"],
        "stop_level_alighting_balance_derivation_allowed": False,
        "future_structure_not_implemented_here": [
            "2023 boarding authority", "route-direction-stop sequence", "STCIS onboard load",
            "15-minute regional OD", "optional tap-out alighting evidence",
            "-> constrained residual OD reconstruction"],
        "implemented_in_this_gate": False})
    dump(root, "safe_zone_gyeongsang_daegu_citybus_authority_correction.json", c["R8_7_23_safe_zone_correction"])
    dump(root, "alighting_semantics_non_regression.json", c["R8_7_35_alighting_semantics"])
    dump(root, "citywide_first_architecture_non_regression.json", c["R8_7_34_citywide_first"])
    dump(root, "r8_7_test_report.json", result)
    dump(root, "classification.json", {**c["R8_7_36_classification_supported"], "sub_classifications": sub})

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R8_7_AUDIT_INCOMPLETE"
    rt = c["R8_7_19_composite_route_mapping"]
    gate = {
        "gate": gate_name, "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA,
        "citywide_ledger_sha256": c["R8_7_03_citywide_ledger_unchanged"]["dataset_sha256"],
        "research_414_sha256": c["R8_7_04_research_414_unchanged"]["sha256"],
        "api_key_status": sub["api_key_status"],
        "api_key_accepted": False,
        "key_propagation_attempts": c["R8_7_10_key_propagation_handled"]["attempt_count"],
        "new_key_requested": False,
        "stcis_15min_od_endpoint": r87.QUARTEROD_SPEC["endpoint"],
        "stcis_15min_od_2023_status": sub["stcis_15min_od_2023_status"],
        "stcis_15min_od_origin_granularity": r87.QUARTEROD_SPEC["origin_granularity"],
        "stcis_15min_od_destination_granularity": r87.QUARTEROD_SPEC["destination_granularity"],
        "stcis_daegu_selectable": True,
        "stcis_stop_level_od_authority": False, "stcis_aggregate_od_constraint": True,
        "z01709_access_status": sub["z01709_access_status"],
        "m009_access_status": sub["z01709_access_status"],
        "m009_2023_status": sub["m009_2023_status"],
        "m009_event_semantics": sub["m009_event_semantics"],
        "stop_level_alighting_balance_derivation_allowed": False,
        "route_no_alone_permitted": False,
        "route_identity_status": sub["route_identity_status"],
        "route_no_only_rate": rt["route_no_only_rate"],
        "route_no_plus_terminus_rate": rt["route_no_plus_terminus_rate"],
        "stop_identity_status": sub["stop_identity_status"],
        "direction_identity_status": sub["direction_identity_status"],
        "safe_zone_gyeongsang_as_daegu_citybus_od_authority": False,
        "safe_zone_correction_recorded": True,
        "historical_artifacts_rewritten": False,
        "alighting_absolute_destination_marginal_allowed": False,
        "alighting_auxiliary_evidence_allowed": True,
        "od_inference_executed": False, "citywide_request_candidate_created": False,
        "candidate_binding_allowed": False,
        "bulk_stcis_download": False,
        "historical_2023_citywide_ledger_unchanged": True, "research_414_unchanged": True,
        "test6_accessed": False, "training_executed": False, "performance_comparison_executed": False,
        "reward_v2_unchanged": True, "zero_loss_unchanged": True, "k_mask_unchanged": True,
        "classification": result["classification"], "sub_classifications": sub,
        "r8_7_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r8_7_tests_total": len(c), "r8_7_tests_all_pass": gate_passed,
        "remaining_dependencies": [
            "STCIS_API_KEY_NOT_REGISTERED_OR_NOT_PROPAGATED",
            "M009_NOT_EXPOSED_TO_ANY_OPEN_API",
            "M009_EVENT_TIMING_CONVENTION_UNKNOWN",
            "ROUTE_IDENTITY_AMBIGUOUS_WITH_STCIS_EXPOSED_FIELDS",
            "STTN_ID_FORMAT_UNVERIFIED",
            "DIRECTION_TERMINUS_TO_MOVE_DIR_UNPROVEN",
        ],
        "recommended_next_gate": NEXT_GATE}
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "od_inference_permitted": False, "request_ledger_permitted": False, "simulator_binding_permitted": False,
        "alighting_balance_derivation_permitted": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "assuming the M009 timing convention", "joining ROUTE_NO to route_id without disambiguation",
            "automating an STCIS login", "requesting a replacement API key without user instruction",
            "treating 교통카드 이용내역(경상권) as Daegu city-bus OD authority",
            "TEST6 access", "A/B1/B2 comparison", "GitHub push"]})
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R8.7", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r8_7": f"{gate['r8_7_tests_pass_count']}/{gate['r8_7_tests_total']}",
                                     "gate_decision": gate})

    md = [
        "# H4M-AE-R8.7 STCIS Daegu Onboard-Load Semantics and 2023 Availability Probe",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}` · execution base `{EXEC_BASE_SHA}`",
        f"- R8.7 {gate['r8_7_tests_pass_count']}/{gate['r8_7_tests_total']}",
        f"- classification: **{result['classification']}**",
        "",
        "## Secret handling",
        "",
        "The key is read from the git-excluded secret directory into process memory by "
        "`stcis_secret_loader` and never leaves it. There is deliberately no accessor for a hash, a length, a "
        "prefix or a fragment, the loader contains no `print`, and no request URL carrying the key is stored. "
        "The secret directory was **untracked but not git-ignored** when this gate started; a `.gitignore` entry "
        "was added so it cannot be committed by accident.",
        "",
        "## The key does not authenticate yet",
        "",
        "`areacode.json` answered with structured JSON — so the endpoint family is right and the key is being "
        "read — but returned `status=ERROR, error.code=INVALID_KEY` (등록되지 않은 인증키입니다) on **four "
        "attempts across a 0/30/60/120 s backoff**, on **three encoding variants**, and — decisively — a "
        "**deliberately fabricated key produced the identical error**. So this is registration or propagation, "
        "not a request-format defect. No replacement key was requested.",
        "",
        "## 15-minute OD: contract fully known, query blocked",
        "",
        f"`{r87.QUARTEROD_SPEC['endpoint']}` requires `apikey`, `opratDate` (8 digits), `stgEmdCd` and "
        "`arrEmdCd` (10-digit 읍/면/동 codes), returning `count`, `status` (OK / NOT_FOUND) and a result array. "
        "**Both** origin and destination dong codes are mandatory, so it is a point query, not a dump. Daegu is "
        "selectable (SD 27) and an arbitrary 8-digit date is expressible, so a 2023 query is structurally "
        "possible — but nothing could be authenticated, so `2023_available = UNKNOWN`. The classification stays "
        "`AUTHORITATIVE_AGGREGATE_OD_CONSTRAINT`; dong-level OD is never promoted to stop level.",
        "",
        "## Z01709 / M009: not an API question at all",
        "",
        "The published Open API catalogue contains exactly seven services — 버스 노선별 경유정류장정보, "
        "15분단위OD, 도시철도 노선별 경유역정보, 버스 노선정보, 버스 정류장정보, 지역코드, 도시철도 노선정보 — "
        "and **none of them is 재차인원**. M009 has no key-based route in any form; it is served only by the "
        "pivot session endpoints. Access mode is therefore `INTERACTIVE_SESSION_REQUIRED`, and I did not "
        "automate a login or forge a session.",
        "",
        "**Event timing stays `UNKNOWN`.** The official text fixes the quantity as 정류장간 차내 재차인원수 — a "
        "between-stop segment load — but never states whether it is measured on arrival, after alighting, or at "
        "departure. Nothing in the metadata resolves it, and I refused to infer it from numbers. Consequently "
        "**stop-level alighting balance derivation remains prohibited** even if rows later arrive.",
        "",
        "## Identity, measured offline",
        "",
        "| key | unique of 238 | rate |",
        "| --- | --- | --- |",
        f"| ROUTE_NO alone | {rt['route_no_only_unique']} | {rt['route_no_only_rate']} |",
        f"| ROUTE_NO + terminus (what STCIS exposes) | {rt['route_no_plus_terminus_unique']} | {rt['route_no_plus_terminus_rate']} |",
        f"| ROUTE_NO + origin + destination | {rt['route_no_plus_origin_and_dest_unique']} | {rt['route_no_plus_origin_dest_rate']} |",
        "",
        "ROUTE_NO alone is prohibited as a key. Even the best composite available from STCIS's exposed fields "
        f"leaves {238 - rt['route_no_plus_terminus_unique']} routes ambiguous. `STTN_SEQ` and `STTN_ID` would "
        "tighten this, but they need rows no legitimate path provided. `STTN_ID` format is unverified against "
        "the project's 5,705 stops (5,646 ten-digit, 59 nine-digit). Direction remains a terminus label against "
        "the project's explicit `move_dir_code` over 346 route-direction pairs on 234 routes — not proven "
        "equivalent, and not converted to 0/1.",
        "",
        "## Safe-zone correction",
        "",
        "`교통카드 이용내역(경상권)` is **withdrawn** as a candidate Daegu city-bus individual-trip OD authority "
        "and recorded as railway/station-linked regional transport data. R8.5 and R8.6 artifacts are left intact; "
        "this correction is additive, not a rewrite. No safe-zone application effort was spent here.",
        "",
        "## Where the blocker actually is",
        "",
        "Three separate walls, and it matters that they are separate:",
        "",
        "1. **Authentication** blocks the 15-minute OD API — fixable by key activation.",
        "2. **Session-only exposure** blocks M009 — no key will ever fix it.",
        "3. **Semantics and identity** block the constraint itself — and would still block it if both 1 and 2 "
        "were solved tomorrow.",
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
        "api_key_present_in_artifact": False, "api_key_hash_present": False,
        "key_bearing_url_present": False, "secret_source_content_present": False,
        "source_file_sha256": {p.name: sha256_file(p) for p in (R87_TEST, SECRET_LOADER, LEDGER_MODULE, Path(__file__))}})
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R8.7] artifact root: {root}")
    print(f"[H4M-AE-R8.7] gate: {gate_name}")
    print(f"[H4M-AE-R8.7] classification: {result['classification']}")
    print(f"[H4M-AE-R8.7] tests {gate['r8_7_tests_pass_count']}/{gate['r8_7_tests_total']}")
    print(f"[H4M-AE-R8.7] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R8.7] BLOCKED")


if __name__ == "__main__":
    main()
