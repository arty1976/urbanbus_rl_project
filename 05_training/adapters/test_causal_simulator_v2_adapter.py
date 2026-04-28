
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from causal_simulator_v2_adapter import CausalSimulatorV2Adapter, FORBIDDEN_DYNAMIC_SIGNAL_FIELDS


ROOT = Path(__file__).resolve().parents[2]


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def make_fixture(root: Path):
    signal_dir = root / "signal_features_v2"
    contract_dir = root / "causal_simulator_v2_contract"
    signal_dir.mkdir(parents=True, exist_ok=True)
    contract_dir.mkdir(parents=True, exist_ok=True)

    node_df = pd.DataFrame([
        {"node_uid":"STOP:1","node_index":0,"signal_count_250m":2,"nearest_signal_distance_m":12.0,"pedestrian_signal_count_250m":1,"blink_signal_ratio_250m":0.5,"controlled_signal_ratio_250m":1.0,"signal_delay_risk_proxy":1.2,"intersection_complexity_proxy":2.3},
        {"node_uid":"STOP:2","node_index":1,"signal_count_250m":1,"nearest_signal_distance_m":90.0,"pedestrian_signal_count_250m":0,"blink_signal_ratio_250m":0.0,"controlled_signal_ratio_250m":1.0,"signal_delay_risk_proxy":0.8,"intersection_complexity_proxy":1.1},
        {"node_uid":"STOP:3","node_index":2,"signal_count_250m":4,"nearest_signal_distance_m":5.0,"pedestrian_signal_count_250m":2,"blink_signal_ratio_250m":0.25,"controlled_signal_ratio_250m":1.0,"signal_delay_risk_proxy":1.8,"intersection_complexity_proxy":3.2},
    ])
    edge_df = pd.DataFrame([
        {"src_idx":0,"dst_idx":1,"distance_m":120.0,"edge_signal_count":3,"edge_signal_density_per_km":25.0,"edge_nearest_signal_distance_m":12.0,"edge_control_complexity_proxy":1.7},
        {"src_idx":1,"dst_idx":2,"distance_m":200.0,"edge_signal_count":5,"edge_signal_density_per_km":25.0,"edge_nearest_signal_distance_m":5.0,"edge_control_complexity_proxy":2.4},
    ])

    node_path = signal_dir / "node_signal_features.parquet"
    edge_path = signal_dir / "edge_signal_features.parquet"
    contract_path = contract_dir / "causal_simulator_v2_input_contract.json"
    node_df.to_parquet(node_path, index=False)
    edge_df.to_parquet(edge_path, index=False)

    contract = {
        "artifact_version": "causal_simulator_v2_input_contract_draft_step100",
        "step": 100,
        "feature_classes": {"dynamic_signal_phase_features": {"admission": "rejected", "columns": list(FORBIDDEN_DYNAMIC_SIGNAL_FIELDS)}},
        "claim_guardrails": {
            "trained_model": False,
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "dynamic_signal_phase_claim_allowed": False,
            "red_light_delay_claim_allowed": False,
            "green_time_claim_allowed": False,
            "cycle_length_claim_allowed": False,
        },
    }
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    return contract_path, node_path, edge_path


def test_reset_step_graph() -> None:
    contract, node, edge = make_fixture(ROOT / "artifacts" / "step101_selftest")
    ad = CausalSimulatorV2Adapter(contract, node, edge, num_agents=2, episode_steps=3, seed=123)
    obs0 = ad.reset(seed=123)
    assert_true(tuple(obs0["actor_obs"].shape) == (2, 9), "actor_obs shape mismatch")
    assert_true(tuple(obs0["critic_obs"].shape) == (1, 8), "critic_obs shape mismatch")
    assert_true(tuple(obs0["edge_index"].shape) == (2, 2), "edge_index shape mismatch")
    assert_true(tuple(obs0["edge_attr"].shape) == (2, 5), "edge_attr shape mismatch")
    p0 = float(obs0["actor_obs"][0, 0])
    sr = ad.step({0: 2, 1: 0})
    p1 = float(sr.obs["actor_obs"][0, 0])
    assert_true(p0 != p1, "state should change")
    assert_true(sr.info["performance_claim_allowed"] is False, "claim guardrail broken")
    assert_true(sr.info["dynamic_signal_phase_claim_allowed"] is False, "phase guardrail broken")
    gs = ad.get_graph_skeleton()
    assert_true(gs.num_nodes == 3, "bad num_nodes")
    assert_true(gs.num_edges == 2, "bad num_edges")
    kpis = ad.compute_kpis([sr])
    assert_true(kpis["static_signal_context_loaded"] == 1.0, "missing signal KPI")
    ad.close()


def test_reject_bad_contract() -> None:
    contract, node, edge = make_fixture(ROOT / "artifacts" / "step101_bad_contract")
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["claim_guardrails"]["dynamic_signal_phase_claim_allowed"] = True
    contract.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        CausalSimulatorV2Adapter(contract, node, edge, num_agents=2)
    except RuntimeError as exc:
        assert_true("dynamic_signal_phase_claim_allowed" in str(exc), "wrong error")
        return
    raise AssertionError("bad contract was not rejected")


def test_reject_forbidden_column() -> None:
    contract, node, edge = make_fixture(ROOT / "artifacts" / "step101_bad_column")
    df = pd.read_parquet(node)
    df["red_light_delay_seconds"] = 1.0
    df.to_parquet(node, index=False)
    try:
        CausalSimulatorV2Adapter(contract, node, edge, num_agents=2)
    except RuntimeError as exc:
        assert_true("forbidden dynamic signal fields" in str(exc), "wrong error")
        return
    raise AssertionError("forbidden column was not rejected")


def main() -> None:
    test_reset_step_graph()
    test_reject_bad_contract()
    test_reject_forbidden_column()
    print("[OK] Step 101 CausalSimulatorAdapter v2 scaffold self-test PASS")


if __name__ == "__main__":
    main()
