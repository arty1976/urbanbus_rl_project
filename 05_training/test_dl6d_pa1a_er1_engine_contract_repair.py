from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
MODULE_PATH = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_pa1a_er1_engine_contract_repair.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dl6d_pa1a_er1", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dl6d_pa1a_er1"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_upstream_manifest_defect_is_detected():
    module = load_module()
    defect = module.upstream_manifest_defect_record()
    assert defect["defect_detected"] is True
    assert defect["defect_class"] == "STALE_STAGE_LOCK_METADATA_AFTER_RETRY"
    assert defect["upstream_artifact_content_usable"] is True
    assert defect["upstream_artifact_manifest_clean"] is False
    assert defect["actual_terminal_gate"] == "BLOCKED_SUSEONG_DL6D_PA1A_ENGINE_PARTIAL_REPAIR_REQUIRED"


def test_repair_plan_freezes_no_historical_execution():
    module = load_module()
    plan = module.repair_plan_contract({"defect_detected": True})
    assert plan["implementation_performed_in_reconcile"] is False
    assert plan["historical_branch_execution_allowed"] is False
    assert plan["validation_execution_allowed"] is False
    assert plan["action_mapping"] == {"H": 0, "S": 1, "K": 3}
    assert plan["legacy_action_2_accepted"] is False
    assert plan["thirty_minute_total_steps"] == 30


def test_reconcile_artifact_root_must_be_absolute(tmp_path):
    module = load_module()
    try:
        module.validate_artifact_root(Path("relative/path"), "reconcile")
    except ValueError as exc:
        assert "--artifact-root must be an absolute path" in str(exc)
    else:
        raise AssertionError("relative artifact root must fail")

    root = tmp_path / "er1"
    resolved = module.validate_artifact_root(root, "reconcile")
    assert resolved == root
    assert root.exists()


def test_manifest_terminal_lock_protocol(tmp_path):
    module = load_module()
    root = tmp_path / "er1_manifest"
    root.mkdir()
    writer = module.Writer(root)
    for rel_path in module.RECONCILE_PAYLOADS:
        writer.text(rel_path, f"{rel_path}\n")
    gate = {"mode": "reconcile", "gate": module.PASS_RECONCILE, "gate_passed": True}
    manifest = module.write_manifest(writer)
    lock = module.write_terminal_lock(writer, "_RECONCILE_COMPLETE.lock", gate)
    verification = module.verify_manifest_and_lock(root, "_RECONCILE_COMPLETE.lock")
    manifest_payload = json.loads((root / "artifact_manifest.json").read_text())
    assert manifest["missing_payload_count"] == 0
    assert lock["manifest_relative_path"] == "artifact_manifest.json"
    assert verification["manifest_hash_ok"] is True
    assert verification["manifest_size_ok"] is True
    assert verification["terminal_lock_listed_inside_manifest"] is False
    assert all(item["relative_path"] not in {"artifact_manifest.json", "_RECONCILE_COMPLETE.lock"} for item in manifest_payload["files"])


def test_new_dynamics_modules_are_importable():
    module = load_module()
    static = module.static_implementation_audit()
    assert static["python_syntax_valid"] is True
    assert static["all_modules_importable"] is True
    proxy = module.scan_proxy_dependency()
    assert proxy["r1_proxy_import_count"] == 0
    assert proxy["thirty_minute_audit_reference_count"] == 0
    assert proxy["proxy_reward_literal_reuse_count"] == 0


def test_action_adapter_blocks_legacy_action_2():
    sys.path.insert(0, str(PROJECT_ROOT / "05_training"))
    from simulator.dynamics_multiagent_orchestrator import (
        DynamicsBranchAction,
        UnsupportedDynamicsActionError,
        adapt_branch_action,
        reject_engine_action,
    )

    assert adapt_branch_action(DynamicsBranchAction.HOLD_CURRENT_POSITION) == 0
    assert adapt_branch_action(DynamicsBranchAction.SERVE_AND_MOVE_TO_NEXT_STOP) == 1
    assert adapt_branch_action(DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP) == 3
    try:
        reject_engine_action(2)
    except UnsupportedDynamicsActionError:
        pass
    else:
        raise AssertionError("legacy engine action 2 must be rejected")


def test_replay_event_order_and_p95_no_fallback():
    sys.path.insert(0, str(PROJECT_ROOT / "05_training"))
    from simulator.dynamics_horizon_aggregator import aggregate_horizon_kpis
    from simulator.dynamics_replay_contract import ReplayEvent, ReplayEventType, ReplayFrame

    later = ReplayEvent(
        event_id="b",
        event_timestamp_seconds=10,
        event_type=ReplayEventType.PASSENGER_ARRIVAL,
        stop_id="S1",
        passenger_id="P2",
    )
    earlier = ReplayEvent(
        event_id="a",
        event_timestamp_seconds=5,
        event_type=ReplayEventType.PASSENGER_ARRIVAL,
        stop_id="S1",
        passenger_id="P1",
    )
    frame = ReplayFrame(step_index=0, frame_start_seconds=0, frame_end_seconds=60, events=(later, earlier))
    assert frame.canonical_event_order == ("a", "b")
    kpis = aggregate_horizon_kpis([], [])
    assert kpis["p95_wait_seconds"]["value"] is None
    assert kpis["p95_wait_seconds"]["status"] == "UNAVAILABLE_MISSING_REQUIRED_DATA"
    assert kpis["energy_metric_status"] == "FORMULA_NOT_SELECTED"
    assert kpis["new_reward_formula_created"] is False


def test_implement_manifest_protocol_excludes_stage_locks(tmp_path):
    module = load_module()
    root = tmp_path / "implement_manifest"
    root.mkdir()
    writer = module.Writer(root)
    payloads = [*module.RECONCILE_PAYLOADS, *module.IMPLEMENT_NEW_PAYLOADS]
    for rel_path in payloads:
        if rel_path in {"artifact_manifest_implement.json", "_RECONCILE_COMPLETE.lock", "_IMPLEMENT_COMPLETE.lock"}:
            continue
        writer.text(rel_path, f"{rel_path}\n")
    gate = {"mode": "implement", "gate": module.PASS_IMPLEMENT, "gate_passed": True}
    manifest = module.write_named_manifest(writer, "artifact_manifest_implement.json", payloads, "TEST_IMPLEMENT")
    module.write_terminal_lock_for_manifest(writer, "_IMPLEMENT_COMPLETE.lock", "artifact_manifest_implement.json", gate)
    verification = module.verify_manifest_and_lock(root, "_IMPLEMENT_COMPLETE.lock")
    manifest_payload = json.loads((root / "artifact_manifest_implement.json").read_text())
    listed = {item["relative_path"] for item in manifest_payload["files"]}
    assert manifest["missing_payload_count"] == 0
    assert "artifact_manifest_implement.json" not in listed
    assert "_RECONCILE_COMPLETE.lock" not in listed
    assert "_IMPLEMENT_COMPLETE.lock" not in listed
    assert verification["manifest_hash_ok"] is True
    assert verification["terminal_lock_listed_inside_manifest"] is False
