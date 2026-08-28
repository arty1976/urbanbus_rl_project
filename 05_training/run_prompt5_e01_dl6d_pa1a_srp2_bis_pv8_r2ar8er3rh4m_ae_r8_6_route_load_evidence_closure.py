#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R8.6 public route-load evidence closure and data
safe zone Daegu access preparation.

Evidence closure only.  No OD inference, no request ledger, no simulator binding,
no training, no arm execution, no TEST6 access, no DB writes, no application
submitted, no push.
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
R86_TEST = TRAINING_ROOT / "test_h4m_ae_r8_6_route_load_evidence_closure.py"
LEDGER_MODULE = TRAINING_ROOT / "citywide_demand_ledger.py"

EXEC_BASE_SHA = "4215bdd4e0a33b57886b726e1ec20a9d9932906e"
R85_SOURCE_SHA = "4215bdd4e0a33b57886b726e1ec20a9d9932906e"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R8_6_PUBLIC_ROUTE_LOAD_EVIDENCE_CLOSURE"
             "_AND_DATA_SAFE_ZONE_DAEGU_ACCESS_PREPARATION_COMPLETE")
NEXT_GATE = ("H4M-AE-R8_7_DATA_SAFE_ZONE_ACCESS_REQUEST_PACKET_FINALIZATION"
             "_AND_PUBLIC_ROUTE_LOAD_CONSTRAINT_FREEZE")
KST = timezone(timedelta(hours=9))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r8_6_route_load_evidence_closure as r86

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r8_6_route_load_evidence_closure_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r86.run_validations()
    c = result["checks"]
    sub = result["sub_classifications"]

    dump(root, "upstream_r8_5_binding.json", {
        "execution_base_sha": EXEC_BASE_SHA, "r8_5_source_commit_sha": R85_SOURCE_SHA,
        "R8_6_01": c["R8_6_01_upstream_r8_5_verified"], "R8_6_02": c["R8_6_02_sha_roles_separated"],
        "R8_6_03": c["R8_6_03_citywide_ledger_unchanged"], "R8_6_04": c["R8_6_04_research_414_unchanged"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True})
    dump(root, "public_route_load_source_binding.json", {
        "indicator": result["stcis_indicator"], "catalogue": c["R8_6_05_actual_payload_inspected"]["catalogue"],
        "R8_6_05": c["R8_6_05_actual_payload_inspected"], "R8_6_07": c["R8_6_07_period_recorded"],
        "reproducible_acquisition_instructions": [
            "open https://stcis.go.kr/pivotIndi/wpsPivotIndicator.do?siteGb=P&indiClss=IC03&indiSel=IC0303",
            "select 시/도 = 대구광역시 (SD_CD 27)",
            "set a date range within the 14-day cap (5 days in the restricted mode)",
            "select 노선 and 정류장 dimensions plus 시간대",
            "export via the Excel export action; the metadata contract is served by indiInfoAjax.do",
        ],
        "payload_cached_in_artifact": False,
        "cache_reason": "no license statement authorises redistributing the indicator payload, so only the schema, identity and acquisition path are stored"})
    dump(root, "public_route_load_schema.json", {
        "dimensions": result["stcis_dimensions"], "measures": result["stcis_measures"],
        "R8_6_06": c["R8_6_06_schema_recorded"],
        "R8_6_10": c["R8_6_10_route_identifier_semantics"],
        "R8_6_11": c["R8_6_11_stop_identifier_semantics"],
        "R8_6_12": c["R8_6_12_direction_semantics"],
        "R8_6_13": c["R8_6_13_time_granularity"]})
    dump(root, "public_route_load_daegu_coverage.json", {
        "R8_6_08": c["R8_6_08_daegu_coverage"], "R8_6_09": c["R8_6_09_daegu_row_count"]})
    dump(root, "public_route_load_data_quality_audit.json", {
        "row_payload_obtained": False,
        "null_rates": None, "duplicate_key_rate": None, "outliers": None,
        "status": "NOT_COMPUTABLE_WITHOUT_ROW_PAYLOAD",
        "fabricated_quality_metrics": False})
    dump(root, "route_id_mapping_audit.json", c["R8_6_15_route_join_coverage"])
    dump(root, "stop_id_mapping_audit.json", c["R8_6_16_stop_join_coverage"])
    dump(root, "direction_mapping_audit.json", {
        **c["R8_6_12_direction_semantics"], "R8_6_18": c["R8_6_18_no_silent_fuzzy_mapping"]})
    dump(root, "stop_sequence_consistency_audit.json", c["R8_6_17_sequence_consistency"])
    dump(root, "onboard_load_event_semantics.json", c["R8_6_14_onboard_event_timing"])
    dump(root, "onboard_load_od_constraint_feasibility.json", c["R8_6_19_od_constraint_strength"])
    dump(root, "onboard_load_2023_compatibility.json", c["R8_6_20_2023_compatibility"])
    dump(root, "safe_zone_gyeongsang_daegu_coverage_audit.json", c["R8_6_21_safe_zone_daegu_fresh_audit"])
    dump(root, "safe_zone_individual_trip_field_matrix.json", c["R8_6_22_safe_zone_field_matrix"])
    dump(root, "safe_zone_row_grain_contract.json", c["R8_6_23_row_grain"])
    dump(root, "safe_zone_tap_out_missingness_contract.json", c["R8_6_24_tap_out_missingness"])
    dump(root, "safe_zone_access_export_contract.json", {
        **c["R8_6_25_access_export_documented"], "R8_6_26": c["R8_6_26_no_application_performed"]})
    dump(root, "synthetic_transport_card_data_audit.json", c["R8_6_28_synthetic_dataset_audited"])
    dump(root, "stcis_aggregate_od_freeze.json", c["R8_6_29_stcis_remains_aggregate_od"])
    dump(root, "od_evidence_hierarchy_v2.json", {
        "principle": "the OD builder must always consume the strongest available tier first",
        "tier_1_safe_zone_individual_or_partial_trip": {
            "status": sub["safe_zone_daegu_status"], "period": "2024년 이후",
            "blocking": "Daegu inclusion unconfirmed and access unapproved"},
        "tier_2_public_route_stop_time_onboard_load": {
            "status": sub["public_route_load_status"], "daegu": sub["public_route_load_daegu_status"],
            "constraint_class": c["R8_6_19_od_constraint_strength"]["classification"]},
        "tier_3_stcis_15min_regional_od": {
            "status": "AVAILABLE_AGGREGATE_ONLY",
            "granularity": c["R8_6_29_stcis_remains_aggregate_od"]["daegu_administrative_granularity"]},
        "tier_4_2023_citywide_stop_hour_boarding_authority": {
            "status": "BUILT_AND_FROZEN",
            "dataset_sha256": c["R8_6_03_citywide_ledger_unchanged"]["dataset_sha256"]},
        "tier_5_optional_tap_out_alighting_evidence": {
            "status": "AVAILABLE_SELECTION_BIASED",
            "observed_ratio": c["R8_6_30_alighting_not_promoted"]["observed_ratio"]},
        "tier_6_model_inferred_residual_od": {"status": "NOT_EXECUTED"},
        "tier_collapse_prohibited": True})
    dump(root, "alighting_optional_tap_non_regression.json", c["R8_6_30_alighting_not_promoted"])
    dump(root, "citywide_first_architecture_non_regression.json", c["R8_6_37_citywide_first_architecture"])
    dump(root, "r8_6_test_report.json", result)
    dump(root, "classification.json", {**c["R8_6_38_classification_supported"], "sub_classifications": sub})

    prep = [
        "# Data Safe Zone Access Preparation Package",
        "",
        "**Status: PREPARED, NOT SUBMITTED.** No account was created, no login performed, no application "
        "submitted, no terms accepted, no visit reserved and no personal information provided. Everything below "
        "is material for the user to review and, if they choose, submit themselves.",
        "",
        "## Target",
        "",
        "| item | value |",
        "| --- | --- |",
        "| organization | 한국데이터산업진흥원 (K-DATA) 데이터안심구역, data provided by 한국교통안전공단 |",
        "| portal | https://dsz.kdata.or.kr |",
        "| dataset | 교통카드 이용내역(경상권) — `TB_KTS_DWTCD_GYEONGSANG.csv` |",
        "| companion datasets | 노선 정보, 노선정류장 정보, 정류장 정보 (`TB_KTS_STTN.csv`) |",
        "| period | 2024년 이후 |",
        "| analysis sites | 서울, 대전 |",
        "",
        "## The question to resolve before anything else",
        "",
        "**Does 경상권 include 대구광역시?** No public page states this. It is the single blocking unknown, and "
        "it should be asked of the provider directly before any application effort is spent. A second question "
        "is worth asking in the same message: **does each boarding record carry an alighting stop and "
        "timestamp**, and can a missing tap-out be distinguished from a transfer or from data loss.",
        "",
        "## Access steps as published",
        "",
        "1. 회원가입 on the 데이터안심구역 portal",
        "2. 이용신청서 submission with research purpose and analysis plan",
        "3. 승인 by the reviewing body",
        "4. On-site analysis in an internet-disconnected room at 서울 or 대전",
        "5. 분석한 결과만 반출 — only analysis results leave the room",
        "",
        "## Export constraint that shapes the research design",
        "",
        "`RAW_EXPORT_ALLOWED = false`, `DERIVED_RESULT_EXPORT_ALLOWED = true`. The simulator therefore cannot "
        "consume individual trip records directly. What can leave is a **derived OD structure**: aggregated OD "
        "tables, fitted tap-out missingness parameters, validation statistics and anonymized distributions. The "
        "OD builder must be designed around exporting parameters, not records.",
        "",
        "## Draft research purpose (for the user to review, edit and submit)",
        "",
        "> 본 연구는 대구광역시 대중교통 수요를 재구성하여 수요응답형 교통(DRT) 도입 효과를 검증하기 위한 "
        "> 시뮬레이션 연구입니다. GATv2 그래프 신경망과 MAPPO 다중에이전트 강화학습을 이용해 정류장·노선·시간대 "
        "> 단위 수요를 반영한 차량 운행 정책을 학습·평가합니다.",
        "> ",
        "> 교통카드 이용내역 자료는 다음 두 가지 목적에 한해 사용합니다. 첫째, 승차 기록만으로는 알 수 없는 "
        "> 목적지(OD) 구조를 파악하는 것입니다. 둘째, 하차 태그가 구조적으로 선택적이기 때문에 발생하는 "
        "> 하차 결측(tap-out missingness)의 성격과 크기를 추정하는 것입니다.",
        "> ",
        "> 반출을 요청하는 결과물은 개인 식별이 불가능한 집계 결과에 한합니다. 구체적으로 정류장군·시간대 단위 "
        "> 집계 OD 표, 하차 결측 보정 모형의 추정 파라미터, 검증 통계량, 익명화된 분포 요약입니다. 개별 통행 "
        "> 원자료나 가상카드 단위 기록의 반출은 요청하지 않으며, 개인 재식별을 시도하지 않습니다.",
        "",
        "## What must NOT be claimed in the application",
        "",
        "- that Daegu is already known to be included",
        "- that 2024+ records represent 2023 demand magnitude",
        "- that raw trip records are needed",
    ]
    (root / "safe_zone_access_preparation.md").write_text("\n".join(prep) + "\n", encoding="utf-8")

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R8_6_EVIDENCE_CLOSURE_INCOMPLETE"
    led = c["R8_6_03_citywide_ledger_unchanged"]
    join = c["R8_6_15_route_join_coverage"]
    gate = {
        "gate": gate_name, "source_sha": head_sha,
        "execution_base_sha": EXEC_BASE_SHA, "r8_5_source_commit_sha": R85_SOURCE_SHA,
        "citywide_ledger_sha256": led["dataset_sha256"],
        "research_414_sha256": c["R8_6_04_research_414_unchanged"]["sha256"],
        "public_route_load_dataset_confirmed": True,
        "public_route_load_provider": "국토교통부 / 한국교통안전공단 STCIS (indicator Z01709)",
        "public_route_load_period": "UNKNOWN_NOT_PUBLISHED",
        "public_route_load_row_count": None,
        "public_route_load_daegu_status": sub["public_route_load_daegu_status"],
        "public_route_load_daegu_row_count": None,
        "public_route_id_available": "ROUTE_NO (route number, not route_id)",
        "public_stop_id_available": "STTN_ID",
        "public_direction_available": "STG_ARR_NMA terminus label only",
        "public_stop_sequence_available": "STTN_SEQ",
        "public_time_granularity": "TZON time band within OPRAT_DATE",
        "public_onboard_count_available": "M009 재차인원 (bus only)",
        "public_onboard_event_semantics": c["R8_6_14_onboard_event_timing"]["classification"],
        "project_route_mapping_rate": join["route_mapping_rate_unambiguous"],
        "project_stop_mapping_rate": None,
        "direction_mapping_rate": None,
        "sequence_consistency_rate": None,
        "onboard_load_od_constraint_class": c["R8_6_19_od_constraint_strength"]["classification"],
        "onboard_load_2023_compatibility": c["R8_6_20_2023_compatibility"]["classification"],
        "safe_zone_dataset_confirmed": True,
        "safe_zone_daegu_status": sub["safe_zone_daegu_status"],
        "safe_zone_period": "2024년 이후",
        "safe_zone_row_grain": "UNKNOWN",
        "safe_zone_schema_publicly_resolved": "PARTIAL",
        "safe_zone_virtual_card_status": result["safe_zone_field_matrix"]["virtual_card_id"],
        "safe_zone_boarding_stop_status": result["safe_zone_field_matrix"]["boarding_stop_id"],
        "safe_zone_boarding_ts_status": result["safe_zone_field_matrix"]["boarding_ts"],
        "safe_zone_alighting_stop_status": result["safe_zone_field_matrix"]["alighting_stop_id"],
        "safe_zone_alighting_ts_status": result["safe_zone_field_matrix"]["alighting_ts"],
        "safe_zone_route_id_status": result["safe_zone_field_matrix"]["route_id"],
        "safe_zone_vehicle_id_status": result["safe_zone_field_matrix"]["vehicle_id"],
        "safe_zone_transfer_status": result["safe_zone_field_matrix"]["transfer_flag"],
        "tap_out_missingness_distinguishable": c["R8_6_24_tap_out_missingness"]["distinguishable"],
        "safe_zone_raw_export_allowed": False,
        "safe_zone_derived_export_allowed": True,
        "safe_zone_application_submitted": False,
        "synthetic_transport_card_data_status": sub["synthetic_data_status"],
        "stcis_stop_level_od_authority": False,
        "stcis_aggregate_od_constraint": True,
        "alighting_absolute_destination_marginal_allowed": False,
        "alighting_auxiliary_evidence_allowed": True,
        "historical_2023_citywide_ledger_unchanged": True,
        "research_414_unchanged": True,
        "od_inference_executed": False,
        "citywide_request_candidate_created": False,
        "candidate_binding_allowed": False,
        "test6_accessed": False, "training_executed": False, "performance_comparison_executed": False,
        "reward_v2_unchanged": True, "zero_loss_unchanged": True, "k_mask_unchanged": True,
        "classification": result["classification"], "sub_classifications": sub,
        "r8_6_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r8_6_tests_total": len(c), "r8_6_tests_all_pass": gate_passed,
        "remaining_dependencies": [
            "SAFE_ZONE_DAEGU_INCLUSION_UNCONFIRMED",
            "STCIS_ROW_PAYLOAD_REQUIRES_INTERACTIVE_SESSION",
            "STCIS_COVERAGE_PERIOD_UNPUBLISHED",
            "ONBOARD_COUNT_TIMING_CONVENTION_UNKNOWN",
            "ROUTE_NO_TO_ROUTE_ID_AMBIGUOUS_FOR_125_OF_238",
            "STTN_ID_FORMAT_UNVERIFIED",
        ],
        "recommended_next_gate": NEXT_GATE}
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "od_inference_permitted": False, "request_ledger_construction_permitted": False,
        "simulator_binding_permitted": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "assuming Daegu is inside 경상권", "assuming the onboard-count timing convention",
            "joining ROUTE_NO to route_id without disambiguation", "scaling alightings to boardings",
            "fabricating destinations", "TEST6 access", "A/B1/B2 comparison", "GitHub push"]})
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R8.6", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r8_6": f"{gate['r8_6_tests_pass_count']}/{gate['r8_6_tests_total']}",
                                     "gate_decision": gate})

    dim_rows = ["| dimension | column | displayed |", "| --- | --- | --- |"]
    for d in result["stcis_dimensions"][:8]:
        dim_rows.append(f"| {d['name']} | `{d['column']}` | {d['displayed']} |")

    md = [
        "# H4M-AE-R8.6 Public Route-Load Evidence Closure and Safe Zone Access Preparation",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}`",
        f"- execution base `{EXEC_BASE_SHA}` (R8.5 source commit, recorded separately from this gate's commit)",
        f"- R8.6 {gate['r8_6_tests_pass_count']}/{gate['r8_6_tests_total']}",
        f"- classification: **{result['classification']}**",
        "",
        "## The public dataset is not a file",
        "",
        "`국토교통부_노선별 재차인원 현황` on data.go.kr carries **전체 행 = 1** and 제공형태 = "
        "*기관자체에서 다운로드(제공데이터URL기재)*. It is a pointer to an STCIS indicator, not a downloadable "
        "CSV. Treating the catalogue entry as the dataset would have been exactly the title-as-schema error the "
        "gate forbids.",
        "",
        "## What the authoritative endpoint actually returns",
        "",
        "Reading `indiInfoAjax.do` for `indiClss=IC03&indiSel=IC0303` gives indicator **`Z01709` "
        "노선·정류장 지표(노선별 차내 재차인원)** with this dimension contract:",
        "",
        *dim_rows,
        "",
        "Measure: **`M009` 재차인원**, and the page states *노선별 차내 재차인원은 **버스만** 제공합니다* — "
        "bus-only, which is exactly the study scope.",
        "",
        f"**Daegu is confirmed selectable**: the live markup lists `<option value=\"27\">대구광역시</option>`. "
        "Query windows are capped at 14 days (5 in the restricted mode).",
        "",
        "## What could not be obtained",
        "",
        "The row payload needs an interactive STCIS session; the list endpoints return the HTML shell to a bare "
        "request. I recorded that as `ACCESS_REQUIRED_INTERACTIVE_SESSION` and stopped, rather than forging a "
        "session. Consequently the Daegu row count, null rates, duplicate rates and the **coverage period** are "
        "all unresolved, and no quality metric was fabricated to fill the gap.",
        "",
        "## Joinability, measured against the project authority",
        "",
        f"- STCIS keys routes by **`ROUTE_NO`**, a route *number*. The project authority keys on `route_id`.",
        f"- Of 238 project routes, only **{join['route_ids_unambiguous_by_route_no']}** have a route_no unique "
        f"enough to join deterministically; **{join['route_ids_ambiguous_by_route_no']}** collide "
        f"(rate {join['route_mapping_rate_unambiguous']}). No fuzzy match was accepted.",
        "- Direction: STCIS offers only a terminus label `STG_ARR_NMA`; the project has explicit `move_dir_code` "
        f"0/1 over {c['R8_6_17_sequence_consistency']['routes_with_sequence']} of 238 routes. Identity must be "
        "resolved, not assumed.",
        "- `STTN_ID` format is unverified against the project's 9/10-digit `stop_id` because no sample row exists.",
        "",
        "## OD constraint strength",
        "",
        f"**{c['R8_6_19_od_constraint_strength']['classification']}** — not strong, and the reasons are concrete: "
        "the onboard-count timing convention (before/after stop service, arrival/departure) is unpublished so the "
        "balance equation has no defined endpoint; there is no explicit direction id; the route join is ambiguous "
        "for the majority; and nothing was verified numerically.",
        "",
        "## Safe zone",
        "",
        "Re-audited and unchanged: **Daegu inclusion in 경상권 is still not stated anywhere public.** Route, "
        "stop, mode and user type are confirmed present; boarding timestamp, alighting stop, alighting timestamp, "
        "card linkage, vehicle id and transfer chain are all `UNKNOWN_PENDING_ACCESS`. Row grain is `UNKNOWN` — "
        "the description says 개별통행 but nothing proves whether a row is a boarding transaction or a completed "
        "trip. The five tap-out states (observed / no record / final destination without tap-out / transfer / "
        "data loss) cannot currently be separated, so missing-tap-out recovery is **not** observable and any "
        "treatment of it stays `INFERRED_CALIBRATED`.",
        "",
        "An access package is prepared in `safe_zone_access_preparation.md` — **not submitted**. No account, no "
        "login, no terms accepted. Its first recommendation is to ask the provider the coverage question directly "
        "before spending application effort.",
        "",
        "## Guards",
        "",
        "No OD inferred, no request ledger, no simulator binding. Alighting stays "
        f"`{c['R8_6_30_alighting_not_promoted']['observed_ratio']}` of boardings and is auxiliary evidence only. "
        "STCIS stays aggregate-OD-only. The 2023 ledger and the frozen 414 are byte-identical. TEST6 untouched, "
        "no training, no comparison.",
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
        "protected_individual_trip_data_stored": False,
        "public_payload_cached": False,
        "source_file_sha256": {p.name: sha256_file(p) for p in (R86_TEST, LEDGER_MODULE, Path(__file__))}})
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R8.6] artifact root: {root}")
    print(f"[H4M-AE-R8.6] gate: {gate_name}")
    print(f"[H4M-AE-R8.6] classification: {result['classification']}")
    print(f"[H4M-AE-R8.6] tests {gate['r8_6_tests_pass_count']}/{gate['r8_6_tests_total']}")
    print(f"[H4M-AE-R8.6] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R8.6] BLOCKED")


if __name__ == "__main__":
    main()
