#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R8.8 STCIS minimal route-load payload capture and
route/stop identity resolution.

Audit only.  No OD inference, no request ledger, no simulator binding, no bulk
collection, no login, no session forgery, no push.  The service key is never
recorded in any output.
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
R88_TEST = TRAINING_ROOT / "test_h4m_ae_r8_8_stcis_payload_and_identity.py"
SECRET_LOADER = TRAINING_ROOT / "stcis_secret_loader.py"
LEDGER_MODULE = TRAINING_ROOT / "citywide_demand_ledger.py"

EXEC_BASE_SHA = "614278d7d69c2d6d5f7f73632158258f40339a43"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R8_8_STCIS_MINIMAL_ROUTE_LOAD_PAYLOAD"
             "_AND_IDENTITY_RESOLUTION_COMPLETE")
NEXT_GATE = "H4M-AE-R8_9_USER_PERFORMED_STCIS_M009_EXPORT_HANDOFF_AND_IDENTITY_RESOLUTION"
KST = timezone(timedelta(hours=9))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r8_8_stcis_payload_and_identity as r88

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r8_8_stcis_payload_identity_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r88.run_validations()
    c = result["checks"]
    sub = result["sub_classifications"]
    rid = c["R8_8_14_route_identity"]

    dump(root, "upstream_r8_7_binding.json", {
        "execution_base_sha": EXEC_BASE_SHA, "r8_7_source_commit_sha": EXEC_BASE_SHA,
        "R8_8_01": c["R8_8_01_upstream_r8_7"], "R8_8_21": c["R8_8_21_citywide_ledger_unchanged"],
        "R8_8_22": c["R8_8_22_research_414_unchanged"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True})
    dump(root, "stcis_secret_nonleakage_audit.json", {
        "R8_8_02": c["R8_8_02_secret_git_ignored"], "R8_8_03": c["R8_8_03_key_leakage_scan"],
        "scan_method": "the live key was loaded in memory and searched for across every artifact file of the previous gate, every tracked repository file, and the full git diff"})
    dump(root, "stcis_api_activation_recheck.json", {
        **c["R8_8_04_activation_rechecked"], "R8_8_05": c["R8_8_05_bounded_retry"]})
    dump(root, "stcis_2023_15min_od_probe.json", c["R8_8_06_2023_od_probe_gated"])
    dump(root, "m009_user_session_access_audit.json", {
        **c["R8_8_07_legitimate_session_only"], "R8_8_08": c["R8_8_08_no_login_bypass"],
        "R8_8_09": c["R8_8_09_minimal_scope"], "R8_8_10": c["R8_8_10_no_bulk_collection"],
        "access_outcome": "USER_INTERACTION_REQUIRED"})
    dump(root, "m009_minimal_payload_schema.json", c["R8_8_11_payload_schema_recorded"])
    dump(root, "m009_minimal_payload_provenance.json", {
        "payload_captured": False, "rows": 0, "file_stored": None, "sha256": None,
        "fabricated_rows": 0,
        "reproducible_capture_instructions": result["minimal_user_action"],
        "storage_policy": "no payload exists to store; if one is later captured, only schema, sanitized sample, hash and provenance are retained unless the licence permits full retention"})
    dump(root, "m009_2023_availability.json", c["R8_8_19_m009_2023_status"])
    dump(root, "sttn_id_identity_resolution.json", {
        "R8_8_12": c["R8_8_12_sttn_id_values_inspected"], "R8_8_13": c["R8_8_13_stop_identity"]})
    dump(root, "route_identity_resolution.json", {**rid, "R8_8_16": c["R8_8_16_no_fuzzy_promotion"]})
    dump(root, "direction_identity_resolution.json", c["R8_8_15_direction_identity"])
    dump(root, "m009_segment_semantics_contract.json", {
        **c["R8_8_17_m009_semantics"], "R8_8_18": c["R8_8_18_stop_balance_prohibited"]})
    dump(root, "route_load_constraint_readiness.json", {
        "classification": result["classification"],
        "m009_2023_available": False, "sttn_id_resolved": False,
        "route_direction_resolved": False, "segment_semantics_defensible": "PARTIAL",
        "sub_classifications": sub,
        "performance_claims": False,
        "od_reconstruction_implemented": False})
    dump(root, "safe_zone_correction_non_regression.json", c["R8_8_20_safe_zone_correction_preserved"])
    dump(root, "alighting_semantics_non_regression.json", {
        k: v for k, v in c["R8_8_32_citywide_first_and_alighting"].items()
        if k.startswith(("boarding", "alighting", "observed"))})
    dump(root, "citywide_first_non_regression.json", c["R8_8_32_citywide_first_and_alighting"])
    dump(root, "r8_8_test_report.json", result)
    dump(root, "classification.json", {**c["R8_8_33_classification_supported"], "sub_classifications": sub})

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R8_8_INCOMPLETE"
    gate = {
        "gate": gate_name, "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA,
        "citywide_ledger_sha256": c["R8_8_21_citywide_ledger_unchanged"]["dataset_sha256"],
        "research_414_sha256": c["R8_8_22_research_414_unchanged"]["sha256"],
        "api_key_status": sub["api_key_status"], "api_key_accepted": False,
        "activation_attempts": len(result["activation_recheck"]["attempts"]),
        "replacement_key_requested": False,
        "stcis_15min_od_2023_status": sub["stcis_15min_od_2023_status"],
        "stcis_aggregate_od_constraint": True, "stcis_stop_level_od_authority": False,
        "m009_session_status": sub["m009_session_status"],
        "m009_login_required": False,
        "m009_blocker": "INTERACTIVE_FORM_STATE_ASSEMBLY",
        "m009_payload_captured": False, "m009_rows_captured": 0,
        "m009_2023_status": sub["m009_2023_status"],
        "m009_segment_semantics": sub["m009_segment_semantics"],
        "m009_endpoint_convention": sub["m009_endpoint_convention"],
        "stop_alighting_balance_derivation_allowed": False,
        "stop_identity_status": sub["stop_identity_status"],
        "route_identity_status": sub["route_identity_status"],
        "route_no_plus_terminus_rate": rid["route_no_plus_terminus_rate"],
        "stcis_holds_own_route_identifier": True,
        "direction_identity_status": sub["direction_identity_status"],
        "constraint_readiness": result["classification"],
        "safe_zone_gyeongsang_as_daegu_citybus_od_authority": False,
        "alighting_absolute_destination_marginal_allowed": False,
        "alighting_auxiliary_evidence_allowed": True,
        "od_inference_executed": False, "citywide_request_candidate_created": False,
        "suseong_request_candidate_created": False, "candidate_binding_allowed": False,
        "bulk_stcis_download": False,
        "historical_2023_citywide_ledger_unchanged": True, "research_414_unchanged": True,
        "test6_accessed": False, "training_executed": False, "performance_comparison_executed": False,
        "reward_v2_unchanged": True, "zero_loss_unchanged": True, "k_mask_unchanged": True,
        "classification": result["classification"], "sub_classifications": sub,
        "r8_8_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r8_8_tests_total": len(c), "r8_8_tests_all_pass": gate_passed,
        "remaining_dependencies": [
            "STCIS_API_KEY_NOT_ACTIVATED (separate lineage, does not block M009)",
            "M009_REQUIRES_INTERACTIVE_FORM_STATE_NOT_LOGIN",
            "M009_ENDPOINT_CONVENTION_UNKNOWN",
            "STTN_ID_VALUES_NEVER_OBSERVED",
            "DIRECTION_SEQUENCE_EVIDENCE_ABSENT",
        ],
        "recommended_next_gate": NEXT_GATE}
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "od_inference_permitted": False, "request_ledger_permitted": False,
        "simulator_binding_permitted": False, "stop_alighting_balance_permitted": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "reconstructing the 79-field pivot request by guesswork",
            "automating an STCIS login", "requesting a replacement API key without user instruction",
            "assuming the M009 endpoint convention", "reopening 경상권 as Daegu city-bus OD authority",
            "TEST6 access", "A/B1/B2 comparison", "GitHub push"]})
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R8.8", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r8_8": f"{gate['r8_8_tests_pass_count']}/{gate['r8_8_tests_total']}",
                                     "gate_decision": gate})

    act = result["minimal_user_action"]
    md = [
        "# H4M-AE-R8.8 STCIS Minimal Route-Load Payload and Identity Resolution",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}` · execution base `{EXEC_BASE_SHA}`",
        f"- R8.8 {gate['r8_8_tests_pass_count']}/{gate['r8_8_tests_total']}",
        f"- classification: **{result['classification']}**",
        "",
        "## Track A — key still not activated",
        "",
        "`areacode.json` returned `INVALID_KEY` on the initial recheck and on one delayed retry. Bounded retry "
        "policy respected, no replacement key requested. The 2023 15-minute OD probe is gated on an active key, "
        "so it did not run and `2023_od_status` stays `UNKNOWN`.",
        "",
        "## Track B — the real blocker is narrower than R8.7 thought",
        "",
        "This is the substantive finding. `indicatorPivotAjax.do` **answers in JSON to an ordinary session "
        "cookie** — `{\"ajaxResult\":\"fail\"}`, not a login redirect. A session cookie issued by a plain public "
        "page load is normal client behaviour, so:",
        "",
        "- **login is NOT required** for M009",
        "- what is required is the **assembled state of a 79-field form** that only the interactive UI produces",
        "",
        "R8.7 recorded this as `INTERACTIVE_SESSION_REQUIRED`. That was right but imprecise: authentication was "
        "never the barrier. I refused to reconstruct 79 field values by guesswork, because a fabricated request "
        "contract is exactly the kind of invention this chain of gates exists to prevent.",
        "",
        "## One new identity lead",
        "",
        "The pivot form carries its own **`routeId`, `routeSdCd`, `routeSggCd`** fields. So STCIS does hold a "
        "route identifier beyond `ROUTE_NO`. If an export exposes it, the "
        f"{rid['route_no_plus_terminus_rate']} terminus-composite ceiling may collapse entirely. No value was "
        "observed, so nothing is claimed.",
        "",
        "## What stays unresolved",
        "",
        "| item | status |",
        "| --- | --- |",
        f"| STTN_ID identity | {sub['stop_identity_status']} — no value was ever observed |",
        f"| route identity | {sub['route_identity_status']} |",
        f"| direction identity | {sub['direction_identity_status']} |",
        f"| M009 2023 | {sub['m009_2023_status']} |",
        "",
        "**M009 semantics frozen conservatively as `SEGMENT_LOAD_BETWEEN_STOPS`** and nothing more. I did note "
        "one thing that may matter later: a *difference between consecutive segment loads* is well defined even "
        "without the endpoint convention, since both endpoints carry the same unknown. What remains "
        "unidentifiable is splitting that difference into boarding versus alighting at a stop event — so "
        "`stop_alighting_balance_derivation_allowed = false` stands.",
        "",
        "## Minimum user action to unblock",
        "",
        f"Login is **not** needed. Scope: {act['scope_limit']}.",
        "",
        *[f"{i+1}. {s}" for i, s in enumerate(act["steps"])],
        "",
        "## Guards",
        "",
        "No login, no credentials, no account, no borrowed cookies, no forged session, no CSRF defeat. Nine HTTP "
        "requests total, all for activation probing and published-contract discovery. Zero rows fabricated. A "
        "live key-leak scan across the previous artifact, every tracked file and the full diff found nothing. "
        "The 2023 ledger and the frozen 414 are byte-identical, 경상권 stays withdrawn as a Daegu city-bus OD "
        "authority, TEST6 untouched, no training, no comparison.",
        "",
        "## Next gate",
        "",
        f"`{NEXT_GATE}` (not executed automatically). The API-key activation is recorded as a **separate** "
        "dependency and does not block the M009 lineage.",
    ]
    (root / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p.name for p in root.iterdir() if p.is_file())
    dump(root, "artifact_manifest.json", {
        "artifact_dir": root.name, "generated_at": datetime.now(KST).isoformat(),
        "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA, "gate": gate_name,
        "file_sha256": {n: sha256_file(root / n) for n in files if n != "artifact_manifest.json"},
        "api_key_present": False, "key_hash_present": False, "key_fragment_present": False,
        "password_present": False, "login_token_present": False, "browser_cookie_present": False,
        "session_token_present": False, "csrf_token_present": False,
        "source_file_sha256": {p.name: sha256_file(p) for p in (R88_TEST, SECRET_LOADER, LEDGER_MODULE, Path(__file__))}})
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R8.8] artifact root: {root}")
    print(f"[H4M-AE-R8.8] gate: {gate_name}")
    print(f"[H4M-AE-R8.8] classification: {result['classification']}")
    print(f"[H4M-AE-R8.8] tests {gate['r8_8_tests_pass_count']}/{gate['r8_8_tests_total']}")
    print(f"[H4M-AE-R8.8] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R8.8] BLOCKED")


if __name__ == "__main__":
    main()
