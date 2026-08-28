from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
MODULE_PATH = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r3_r1_split_inventory.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dl6d_r3_r1_i0", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dl6d_r3_r1_i0"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_capacity_thresholds_follow_revised_patch_contract():
    module = load_module()
    payload, table = module.capacity_thresholds(526)
    by_target = dict(zip(table["target_harmful_count"], table["required_skip_valid_count"]))
    assert by_target[26] == 169
    assert by_target[40] == 260
    assert by_target[50] == 325
    assert by_target[60] == 390
    assert by_target[86] == 558
    assert payload["validation_capacity_26_plausible"] is True
    assert payload["validation_capacity_40_plausible"] is True
    assert payload["validation_capacity_50_plausible"] is True
    assert payload["validation_capacity_86_plausible"] is False
    assert payload["validation_capacity_highest_tier"] == "CAPACITY_50_PLAUSIBLE"
    assert payload["validation_capacity_is_gate"] is False


def test_validation_forbidden_column_audit_detects_outcome_terms():
    module = load_module()
    import pandas as pd

    frame = pd.DataFrame({"inventory_row_id": ["a"], "skip_valid": [True], "skip_minus_serve_reward": [0.1]})
    audit = module.forbidden_column_audit(frame)
    assert audit["validation_outcome_leakage_detected"] is True
    assert audit["forbidden_column_count"] == 1


def test_inventory_row_id_is_split_sensitive():
    module = load_module()
    payload_train = {
        "split": "train",
        "snapshot_id": 1,
        "state_ts": "2022-12-31T20:00:00+00:00",
        "agent_id": 3,
        "action_contract_version": module.ACTION_CONTRACT_VERSION,
        "observation_contract_version": module.OBSERVATION_CONTRACT_VERSION,
    }
    payload_validation = dict(payload_train)
    payload_validation["split"] = "validation"
    assert module.stable_hash(payload_train) != module.stable_hash(payload_validation)


def test_dl6c_legacy_stop_rows_fail_closed_without_k4_state_machine():
    module = load_module()
    dl6c = module.import_dl6c()
    count = 0
    total = 0
    for snapshot_id in range(6017, 6571):
        wid = module.window_id("test", snapshot_id)
        for agent_id in range(8):
            case_id = dl6c.stable_int("dl6c", wid, agent_id, modulo=8)
            routes, vehicle, _scenario = dl6c.route_for_case(f"R{agent_id}", case_id)
            vehicle.agent_id = agent_id
            mask = dl6c.engine.build_distinct_three_action_mask(vehicle, routes)
            count += int(mask["skip_valid"])
            total += 1
    assert total == 4432
    assert count == 0
