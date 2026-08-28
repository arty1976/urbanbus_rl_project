from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
MODULE_PATH = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r2_combined_retraining_readiness_reaudit.py"
R1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r1_observation_contract_repair_20260802_011348"


def load_module():
    spec = importlib.util.spec_from_file_location("dl6d_r2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dl6d_r2_upstream_reconciliation_and_source_drift_clean():
    module = load_module()
    upstream, manifests = module.upstream_validation()
    drift = module.source_drift_audit()
    assert upstream["upstream_integrity_passed"] is True
    assert manifests["upstream_manifest_reconciliation_passed"] is True
    assert drift["source_drift_detected"] is False


def test_dl6d_r2_exclusive_30m_classification_reconciles_to_558():
    module = load_module()
    rewards = pd.read_parquet(R1 / "thirty_minute_skip_reward_comparison.parquet")
    kpis = pd.read_parquet(R1 / "thirty_minute_skip_kpi_delta.parquet")
    primary, flags, summary = module.build_corrected_classification(kpis, rewards)
    assert len(primary) == 558
    assert len(flags) == 558
    assert summary["primary_class_count_sum"] == 558
    assert abs(summary["primary_class_rate_sum"] - 1.0) < 1e-12
    assert summary["exclusive_primary_class_valid"] is True


def test_dl6d_r2_reward_service_alignment_blocks_positive_service_harm():
    module = load_module()
    rewards = pd.read_parquet(R1 / "thirty_minute_skip_reward_comparison.parquet")
    kpis = pd.read_parquet(R1 / "thirty_minute_skip_kpi_delta.parquet")
    primary, flags, _summary = module.build_corrected_classification(kpis, rewards)
    _matrix, _align, service, _network = module.reward_kpi_alignment(primary, flags)
    assert service["direct_service_harm_count"] == 120
    assert service["positive_reward_with_direct_service_harm_count"] == 86
    assert service["reward_service_alignment_valid"] is False


def test_dl6d_r2_temporal_credit_keeps_30m_signal_available():
    module = load_module()
    temporal, horizon, discount, gae, normalization = module.temporal_credit_audits()
    assert temporal["decision_interval_minutes"] == 1
    assert horizon["effective_rollout_horizon_minutes"] >= 30
    assert discount["discount_weight_at_30m"] > 0.1
    assert gae["truncation_bootstrap_valid"] is True
    assert normalization["network_harm_signal_erased"] is False


def test_dl6d_r2_fresh_training_guards_prevent_legacy_reuse():
    module = load_module()
    fresh = module.fresh_initialization_contract()
    training, mutation, external = module.guard_payloads()
    assert fresh["full_fresh_initialization"] is True
    assert fresh["legacy_checkpoint_reuse_authorized"] is False
    assert training["training_run_count"] == 0
    assert mutation["parameter_mutation_count"] == 0
    assert external["api_call_count"] == 0
