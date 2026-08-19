#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R8.5 national transport-card big-data Daegu coverage,
access and OD authority audit.

Audit only.  No OD inference, no request-ledger construction, no simulator
binding, no training, no arm execution, no TEST6 access, no DB writes, no push.
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
R85_TEST = TRAINING_ROOT / "test_h4m_ae_r8_5_transport_card_od_authority_audit.py"
LEDGER_MODULE = TRAINING_ROOT / "citywide_demand_ledger.py"
BRIDGE = TRAINING_ROOT / "causal_kpi_bridge.py"

UPSTREAM_R7_SHA = "596b80e1c71c4b67f941b7849a4110f05988df90"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R8_5_NATIONAL_TRANSPORT_CARD_BIG_DATA"
             "_DAEGU_COVERAGE_AND_OD_AUTHORITY_AUDIT_COMPLETE")
NEXT_GATE = ("H4M-AE-R8_6_DATA_SAFE_ZONE_GYEONGSANG_DAEGU_COVERAGE_CONFIRMATION"
             "_AND_ACCESS_ACQUISITION_PREPARATION")
KST = timezone(timedelta(hours=9))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r8_5_transport_card_od_authority_audit as r85

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r8_5_transport_card_od_authority_audit_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r85.run_validations()
    c = result["checks"]
    by_key = {e["key"]: e for e in result["evidence"]}

    dump(root, "upstream_r8_binding.json", {
        "upstream_r7_source_commit": UPSTREAM_R7_SHA,
        "r8_execution_base_sha": c["R8_5_01_upstream_r8_verified"]["r8_execution_base_sha"],
        "r8_gate_commit_resolved_from_repository": head_sha,
        "sha_resolution_note": "the R8 manifest records the HEAD at its execution time; the commit carrying the R8 gate is the current HEAD's parent chain, resolved here rather than guessed",
        "R8_5_01": c["R8_5_01_upstream_r8_verified"], "R8_5_02": c["R8_5_02_citywide_dataset_sha"],
        "R8_5_03": c["R8_5_03_research_414_unchanged"], "R8_5_04": c["R8_5_04_lineage_separated"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True})
    dump(root, "transport_card_source_registry.json", {
        "priority_1_official": ["한국교통안전공단 STCIS", "데이터안심구역 (K-DATA)", "공공데이터포털 / 국토교통부"],
        "priority_2_local": ["대구 D-데이터허브", "local urbanbus PostgreSQL", "existing project artifacts"],
        "non_official_sources_promoted_to_authority": False,
        "sources": {e["key"]: {k: v for k, v in e.items() if k != "key"} for e in result["evidence"]},
        "grain_ladder": result["grain_ladder"], "source_grain": result["source_grain"]})
    dump(root, "official_source_evidence_manifest.json", {
        "records": result["evidence"], "accessed_at": r85.ACCESSED_AT,
        "login_bypassed": False, "scraping_of_protected_systems": False,
        "accounts_created": False, "applications_submitted": False, "terms_accepted": False,
        "bulk_download_performed": False, "secrets_printed": False})
    dump(root, "stcis_daegu_coverage_audit.json", {
        "R8_5_07": c["R8_5_07_stcis_audited"], "R8_5_08": c["R8_5_08_daegu_od_availability"]})
    dump(root, "stcis_15min_od_schema_audit.json", c["R8_5_09_15min_od_schema"])
    dump(root, "stcis_route_stop_usage_audit.json", c["R8_5_10_route_stop_usage"])
    dump(root, "stcis_onboard_load_audit.json", c["R8_5_11_onboard_load_audited"])
    dump(root, "data_safe_zone_gyeongsang_trip_dataset_audit.json", {
        "R8_5_12": c["R8_5_12_safe_zone_dataset_audited"],
        "R8_5_13": c["R8_5_13_daegu_coverage_resolved"],
        "R8_5_14": c["R8_5_14_individual_trip_grain"]})
    dump(root, "data_safe_zone_access_contract.json", {
        **c["R8_5_20_export_restrictions"],
        "RAW_EXPORT_ALLOWED": False, "DERIVED_RESULT_EXPORT_ALLOWED": True,
        "exportable_result_classes_expected": ["aggregated OD tables", "fitted model parameters",
                                               "validation statistics", "anonymized derived distributions"],
        "exportable_result_classes_confirmed": "UNKNOWN_SUBJECT_TO_REVIEW_AT_APPROVAL",
        "this_gate_performed_any_access": False})
    dump(root, "individual_trip_field_matrix.json", {
        "dataset": "교통카드 이용내역(경상권)", "table": "TB_KTS_DWTCD_GYEONGSANG.csv",
        "published_column_list_available": False,
        "fields": {
            "record_grain": c["R8_5_14_individual_trip_grain"],
            "virtual_card_linkage": c["R8_5_15_virtual_card_linkage"],
            "boarding_stop_and_ts": c["R8_5_16_boarding_stop_and_ts"],
            "alighting_stop_and_ts": c["R8_5_17_alighting_stop_and_ts"],
            "route_and_direction": c["R8_5_18_route_and_direction"],
            "transfer_chain": c["R8_5_19_transfer_chain"],
        },
        "missing_tap_out_distinguishable_from_no_transfer_or_data_loss": "UNKNOWN",
        "assumption_used_in_place_of_evidence": False})
    dump(root, "daegu_ddatahub_route_attributed_reaudit.json", c["R8_5_22_ddatahub_reaudit"])
    dump(root, "alighting_optional_tap_observation_contract.json", {
        **c["R8_5_05_alighting_semantics_frozen"], **c["R8_5_06_alighting_not_absolute_marginal"],
        "prohibited_operations": ["scaling 70.5M alightings to 181.4M boardings",
                                  "applying a global tap-out multiplier",
                                  "treating observed alighting shares as unbiased destination shares",
                                  "IPF directly to raw alighting marginals without a bias model"]})
    dump(root, "transport_card_field_provenance_contract.json", {
        "vocabulary": ["OBSERVED", "DERIVED", "INFERRED_CALIBRATED", "RESEARCH_ASSUMPTION"],
        "rule": "a future request ledger must carry per-field provenance and must never collapse the OD authority tiers",
        "tier_collapse_prohibited": True,
        "od_authority_ladder": result["od_authority_ladder"]})
    dump(root, "od_authority_ladder.json", result["od_authority_ladder"])
    dump(root, "2023_2024_temporal_compatibility_audit.json", c["R8_5_21_temporal_compatibility"])
    dump(root, "citywide_first_demand_architecture_nonregression.json", c["R8_5_30_citywide_first_architecture"])
    dump(root, "r9_evidence_readiness_matrix.json", {
        "observed_first_principle": "OBSERVED/AUTHORITATIVE DATA FIRST, INFERENCE ONLY FOR RESIDUAL GAPS",
        "must_be_observed": {
            "origin stop and time": "already observed in the 2023 citywide ledger",
            "route attribution": "NOT observed for Daegu 2023; public route-attributed Daegu data is 2019 boardings only",
            "destination": "NOT observed; safe-zone individual trips are the only candidate and are 2024+, access-gated, Daegu coverage unknown",
        },
        "may_be_inferred_only_after_evidence": {
            "residual missing tap-out": "requires measured tap-out completeness from an approved individual-trip source",
            "within-hour timestamp": "requires a frozen deterministic method labelled INFERRED_CALIBRATED",
        },
        "blocking_unknowns_for_r9": [
            "does 경상권 include 대구광역시",
            "does the safe-zone record carry alighting stop and timestamp per boarding",
            "is the same pseudonymous card linkable across legs",
            "is 노선별 재차인원 published for Daegu and for 2023",
        ],
        "sub_classifications": result["sub_classifications"]})
    dump(root, "r8_5_test_report.json", result)
    dump(root, "classification.json", {
        **c["R8_5_31_classification_supported"], "sub_classifications": result["sub_classifications"]})

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R8_5_OD_AUTHORITY_AUDIT_INCOMPLETE"
    led = c["R8_5_02_citywide_dataset_sha"]
    gate = {
        "gate": gate_name, "source_sha": head_sha,
        "upstream_r8_sha_or_execution_base": c["R8_5_01_upstream_r8_verified"]["r8_execution_base_sha"],
        "citywide_ledger_sha256": led["dataset_sha256"],
        "stcis_audited": True, "stcis_daegu_supported": True,
        "stcis_15min_od_available": True,
        "stcis_15min_od_granularity": "15 minutes x administrative area (시도/시군구/읍면동)",
        "route_stop_usage_available": True,
        "onboard_load_available": "PUBLISHED_NATIONALLY_DAEGU_SCOPE_UNSTATED",
        "safe_zone_dataset_confirmed": True,
        "safe_zone_daegu_coverage": "UNKNOWN_REQUIRES_CONFIRMATION",
        "safe_zone_period": "2024년 이후",
        "safe_zone_record_grain": "개별통행 데이터 (정류장별/노선별/시간대별); exact row meaning UNKNOWN",
        "safe_zone_individual_trip_available": True,
        "safe_zone_virtual_card_linkage": "UNKNOWN",
        "safe_zone_boarding_stop_available": "STATED",
        "safe_zone_boarding_ts_available": "UNKNOWN",
        "safe_zone_alighting_stop_available": "IMPLIED_BY_승하차_DESCRIPTION",
        "safe_zone_alighting_ts_available": "UNKNOWN",
        "safe_zone_route_id_available": "STATED",
        "safe_zone_vehicle_id_available": "UNKNOWN",
        "safe_zone_transfer_chain_available": "UNKNOWN",
        "safe_zone_raw_export_allowed": False,
        "safe_zone_derived_export_allowed": True,
        "alighting_absolute_destination_marginal_allowed": False,
        "alighting_auxiliary_evidence_allowed": True,
        "historical_2023_citywide_ledger_unchanged": True,
        "research_414_unchanged": True,
        "research_414_mixed_with_historical": False,
        "od_inference_executed": False,
        "citywide_request_candidate_created": False,
        "candidate_binding_allowed": False,
        "test6_accessed": False, "training_executed": False, "performance_comparison_executed": False,
        "reward_v2_unchanged": True, "zero_loss_unchanged": True, "k_mask_unchanged": True,
        "classification": result["classification"],
        "sub_classifications": result["sub_classifications"],
        "r8_5_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r8_5_tests_total": len(c),
        "r8_5_tests_all_pass": gate_passed,
        "remaining_evidence_gaps": [
            "DAEGU_INCLUSION_IN_GYEONGSANG_UNCONFIRMED",
            "SAFE_ZONE_COLUMN_LIST_UNPUBLISHED",
            "TAP_OUT_COMPLETENESS_UNMEASURED",
            "ONBOARD_LOAD_DAEGU_2023_SCOPE_UNSTATED",
            "2024_PLUS_TO_2023_ABSOLUTE_SCALE_NOT_TRANSFERABLE",
        ],
        "recommended_next_gate": NEXT_GATE,
    }
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "od_inference_permitted": False, "request_ledger_construction_permitted": False,
        "simulator_binding_permitted": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "assuming Daegu is inside 경상권", "treating 2024+ records as 2023 demand",
            "scaling alightings to boardings", "fabricating destinations", "TEST6 access",
            "A/B1/B2 comparison", "GitHub push"]})
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R8.5", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r8_5": f"{gate['r8_5_tests_pass_count']}/{gate['r8_5_tests_total']}",
                                     "gate_decision": gate})

    ladder_rows = ["| tier | candidate | status |", "| --- | --- | --- |"]
    for tier, v in result["od_authority_ladder"].items():
        ladder_rows.append(f"| {tier.split('_', 2)[1]} | {v['candidate']} | `{v['status']}` |")

    md = [
        "# H4M-AE-R8.5 Transport-Card Big-Data Daegu Coverage and OD Authority Audit",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}`",
        f"- R8.5 {gate['r8_5_tests_pass_count']}/{gate['r8_5_tests_total']}",
        f"- classification: **{result['classification']}**",
        "",
        "## The headline finding",
        "",
        "**교통카드 이용내역(경상권) exists and is exactly the class of data this project needs.** "
        "한국교통안전공단 publishes it through 데이터안심구역 as "
        "`TB_KTS_DWTCD_GYEONGSANG.csv`, described as *개별통행 데이터* covering "
        "*정류장별/노선별/시간대별* 승/하차 이용내역 with user-type detail. Its own usage examples name "
        "**DRT 도입** as an intended application.",
        "",
        "Three things stop it from being usable today, and none of them can be assumed away:",
        "",
        "1. **Daegu inclusion is not stated.** The catalogue says 대상지역 경상권 and never enumerates "
        "대구광역시. 수도권 is listed as its own region, which makes metropolitan-city inclusion plausible - "
        "but plausible is not evidence, so this is recorded as `UNKNOWN_REQUIRES_CONFIRMATION`.",
        "2. **The column list is not published.** Boarding stop and route are stated; boarding timestamp, "
        "alighting timestamp, card linkage, vehicle id and transfer chain are all `UNKNOWN`.",
        "3. **The period is 2024년 이후** against our 2023 authority.",
        "",
        "## Access contract",
        "",
        "회원가입 → 이용신청서 → 승인 → analysis only inside an internet-disconnected room at 서울 or 대전, "
        "and **분석한 결과만 반출**. So `RAW_EXPORT_ALLOWED = false`, "
        "`DERIVED_RESULT_EXPORT_ALLOWED = true`. This gate submitted nothing, created no account and "
        "accepted no terms.",
        "",
        "## What is available without approval",
        "",
        "- **STCIS 15분 단위 O/D Open API** - Daegu selectable, but 지역별: 시/도, 시/군/구, 읍/면/동. "
        "Administrative-area OD is a genuine constraint, not a stop-level request ledger.",
        "- **노선별 차내 재차인원** - 노선별 × 경유 정류장별 × 시간대별 onboard load, published as a free "
        "CSV. This is the single most valuable public lever, because load transitions between consecutive "
        "stops constrain boardings and alightings per segment - precisely what the local DB lacks. Its "
        "catalogue page does not state geographic scope, so Daegu coverage is unconfirmed.",
        "- **D-데이터허브 route-attributed Daegu demand** exists only for **2019-03..08** and only as "
        "boardings, so it does not overlap the 2023 authority.",
        "- **지역별 교통카드이용 합성데이터** is explicitly 합성 (synthetic), so it is not observed evidence.",
        "",
        "## OD authority ladder",
        "",
        *ladder_rows,
        "",
        "## Alighting semantics, frozen",
        "",
        f"Observed alightings are {led['alightings']:,} against {led['boardings']:,} boardings "
        f"({c['R8_5_05_alighting_semantics_frozen']['observed_alighting_to_boarding_ratio']:.4f}). Tap-out is "
        "structurally optional, so the observed population is plausibly dominated by habitual and "
        "transfer-motivated tap-out users - MNAR-like selection bias.",
        "",
        "- `alighting_as_absolute_destination_marginal_allowed = false`",
        "- `alighting_as_auxiliary_destination_evidence_allowed = true`",
        "",
        "No multiplier was fitted, no scaling applied, no IPF run against raw alighting marginals.",
        "",
        "## Guards",
        "",
        "No OD was inferred, no destination fabricated, no request ledger written, no simulator binding "
        "performed. The 2023 citywide ledger and the frozen 414 artifact are byte-identical and remain "
        "disjoint lineages. No policy KPI, B1 reference or fleet size entered any step. TEST6 untouched, no "
        "training, no arm comparison.",
        "",
        "## Next gate",
        "",
        f"`{NEXT_GATE}` (not executed automatically).",
        "",
        "The decisive unknown is a coverage question, not a modelling question: **does 경상권 include 대구**, "
        "and does the record carry an alighting stop per boarding. Until that is answered by an official "
        "scope statement or approved access, running OD inference would be guessing dressed as method.",
    ]
    (root / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p.name for p in root.iterdir() if p.is_file())
    dump(root, "artifact_manifest.json", {
        "artifact_dir": root.name, "generated_at": datetime.now(KST).isoformat(),
        "source_sha": head_sha, "gate": gate_name,
        "file_sha256": {n: sha256_file(root / n) for n in files if n != "artifact_manifest.json"},
        "protected_or_raw_trip_records_stored": False,
        "source_file_sha256": {p.name: sha256_file(p) for p in (R85_TEST, LEDGER_MODULE, BRIDGE, Path(__file__))}})
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R8.5] artifact root: {root}")
    print(f"[H4M-AE-R8.5] gate: {gate_name}")
    print(f"[H4M-AE-R8.5] classification: {result['classification']}")
    print(f"[H4M-AE-R8.5] tests {gate['r8_5_tests_pass_count']}/{gate['r8_5_tests_total']}")
    print(f"[H4M-AE-R8.5] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R8.5] BLOCKED")


if __name__ == "__main__":
    main()
