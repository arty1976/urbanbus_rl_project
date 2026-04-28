from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_DIR = PROJECT_ROOT / "05_training" / "adapters"

AUDIT_MD = ADAPTERS_DIR / "tensor_signal_data_availability_audit.md"
REQ_JSON = ADAPTERS_DIR / "tensor_signal_data_requirements_v2.json"
INSPECTOR = ADAPTERS_DIR / "inspect_tensor_signal_available_fields.py"


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_files_exist() -> None:
    assert_true(AUDIT_MD.exists(), f"missing: {AUDIT_MD}")
    assert_true(REQ_JSON.exists(), f"missing: {REQ_JSON}")
    assert_true(INSPECTOR.exists(), f"missing: {INSPECTOR}")


def test_requirements_json_contract() -> None:
    payload = json.loads(REQ_JSON.read_text(encoding="utf-8-sig"))
    assert_true(payload["artifact_version"] == "tensor_signal_data_requirements_v2", "bad artifact_version")
    assert_true(payload["step"] == 97, "bad step")

    source_classes = payload.get("source_classes", {})
    for key in [
        "tensor_db_direct",
        "signal_csv_direct",
        "tensor_db_plus_signal_csv_derived",
        "proxy_only",
        "currently_unavailable",
    ]:
        assert_true(key in source_classes, f"missing source class: {key}")

    unavailable = source_classes["currently_unavailable"]["examples"]
    for forbidden in [
        "red_light_delay_seconds",
        "green_time_seconds",
        "cycle_length_seconds",
        "phase_sequence",
        "signal_offset_seconds",
    ]:
        assert_true(forbidden in unavailable, f"missing unavailable field: {forbidden}")

    rules = payload.get("causal_simulator_v2_admission_rules", {})
    assert_true("currently_unavailable" in rules.get("reject", []), "currently_unavailable must be rejected")


def test_audit_md_keywords() -> None:
    text = AUDIT_MD.read_text(encoding="utf-8-sig")
    required_phrases = [
        "toy causal smoke validation",
        "trained_model",
        "performance_claim_allowed",
        "signal_count_250m",
        "nearest_signal_distance_m",
        "red-light delay",
        "green time",
        "cycle length",
        "Step 98",
        "static signal infrastructure"
    ]
    for phrase in required_phrases:
        assert_true(phrase in text, f"missing phrase in audit md: {phrase}")


def test_inspector_runs_without_inputs() -> None:
    out_json = PROJECT_ROOT / "artifacts" / "step97_selftest" / "audit_report_no_inputs.json"
    if out_json.exists():
        out_json.unlink()

    cmd = [
        sys.executable,
        str(INSPECTOR),
        "--output-json",
        str(out_json),
    ]
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)

    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("inspector failed without optional inputs")

    assert_true(out_json.exists(), "inspector did not write output json")
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert_true(payload["step"] == 97, "bad report step")
    assert_true(payload["claim_guardrails"]["performance_claim_allowed"] is False, "claim guardrail broken")

    cls = payload["availability_classification"]
    assert_true("red_light_delay_seconds" in cls["currently_unavailable"], "red_light_delay must be unavailable")
    assert_true(cls["causal_simulator_v2_admission"]["not_allowed_now"], "not_allowed_now must not be empty")


def main() -> None:
    test_files_exist()
    test_requirements_json_contract()
    test_audit_md_keywords()
    test_inspector_runs_without_inputs()
    print("[OK] Step 97 tensor + signal data availability audit self-test PASS")


if __name__ == "__main__":
    main()
