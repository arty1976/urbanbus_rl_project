"""Focused pure-function guards for the BT8-R15 repair-selection audit.

The R15 executor is intentionally a frozen-support selection audit.  These
tests exercise its identity-keyed stochastic sampler and eligibility rules
without loading checkpoints, accessing MPS, or authorizing any execution.
"""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r15_minimal_exploration_deadlock_repair_selection_audit as R15  # noqa: E402


def _action(*, kind: str, agent_id: str | None, candidate_id: str | None,
            probability: float, identity_digest: str) -> dict[str, object]:
    return {
        "kind": kind,
        "agent_id": agent_id,
        "candidate_id": candidate_id,
        "policy_probability": probability,
        "identity_digest": identity_digest,
        "score": 0.0,
    }


def _distribution() -> list[dict[str, object]]:
    return [
        _action(kind="CANDIDATE", agent_id="agent-02", candidate_id="candidate-b",
                probability=0.20, identity_digest="candidate-b-digest"),
        _action(kind="NO_ASSIGN", agent_id=None, candidate_id=None,
                probability=0.50, identity_digest="no-assign-digest"),
        _action(kind="CANDIDATE", agent_id="agent-01", candidate_id="candidate-a",
                probability=0.30, identity_digest="candidate-a-digest"),
    ]


def _e0_control() -> dict[str, object]:
    return {
        "bd_control_exact": True,
        "same_seed_replay_mismatch_count": 0,
        "ac_control_mismatch_count": 0,
        "ac_e1_triggered_count": 0,
    }


def _zero_integrity_counters() -> dict[str, int]:
    return {"nan_or_inf": 0, "illegal_selection": 0, "zero_loss_violation": 0}


def _band_row(*, cell_id: str, time_band: str, candidate: bool) -> dict[str, object]:
    return {
        "repair_level": "E1",
        "cell_id": cell_id,
        "time_band": time_band,
        "sampled_action_family": "CANDIDATE" if candidate else "NO_ASSIGN",
        "non_no_assign_probability_mass": 0.25,
    }


def _parquet_row(*, probe_seed: int) -> dict[str, object]:
    row: dict[str, object] = {column: f"value:{column}" for column in R15.PROBE_COLUMNS}
    row.update({
        "support_size": 3,
        "legal_candidate_count": 2,
        "probe_seed": probe_seed,
        "training_only_triggered": True,
        "sampled_is_no_assign": False,
        "legal": True,
        "zero_loss_feasible": True,
        "policy_probability": 0.25,
        "selection_probability_if_modified": 0.50,
        "no_assign_probability": 0.50,
        "non_no_assign_probability_mass": 0.50,
        "finite": True,
        "candidate_order_identity_equal": True,
        "agent_order_identity_equal": True,
        "combined_order_identity_equal": True,
    })
    return row


def test_semantic_exponential_race_is_reproducible_and_order_independent() -> None:
    actions = _distribution()
    kwargs = {
        "snapshot_digest": "preserved-snapshot-digest",
        "probe_seed": 17,
        "contract_id": R15.R15_E1_CONTRACT_ID,
        "repair_level": "E1",
    }

    first = R15.categorical_exponential_race(actions=actions, **kwargs)
    replay = R15.categorical_exponential_race(actions=deepcopy(actions), **kwargs)
    candidate_permuted = R15.categorical_exponential_race(actions=list(reversed(actions)), **kwargs)
    agent_permuted = R15.categorical_exponential_race(actions=[actions[2], actions[0], actions[1]], **kwargs)

    expected = R15.action_identity(first)
    assert R15.action_identity(replay) == expected
    assert R15.action_identity(candidate_permuted) == expected
    assert R15.action_identity(agent_permuted) == expected
    assert R15.canonical_rng_keyset_sha(actions, **kwargs) == R15.canonical_rng_keyset_sha(list(reversed(actions)), **kwargs)


def test_e1_is_training_only_and_preserves_frozen_inference_distribution() -> None:
    distribution = _distribution()
    frozen_before = deepcopy(distribution)
    deterministic = next(action for action in distribution if action["kind"] == "NO_ASSIGN")

    inference, inference_triggered, inference_mode, inference_modified = R15.e1_selection(
        distribution=distribution, deterministic=deterministic, snapshot_digest="snapshot", probe_seed=3,
        training_mode=False,
    )
    training, training_triggered, training_mode, training_modified = R15.e1_selection(
        distribution=distribution, deterministic=deterministic, snapshot_digest="snapshot", probe_seed=3,
        training_mode=True,
    )

    assert R15.action_identity(inference) == R15.action_identity(deterministic)
    assert inference_triggered is False
    assert inference_mode == "DETERMINISTIC_EXPLOIT_NO_E1_TRIGGER"
    assert inference_modified is None
    assert training_triggered is True
    assert training_mode == "FROZEN_MASKED_CATEGORICAL"
    assert training_modified is None
    assert R15.action_identity(training) in {R15.action_identity(action) for action in distribution}
    assert distribution == frozen_before

    candidate_deterministic = next(action for action in distribution if action["kind"] == "CANDIDATE")
    unchanged, triggered, mode, modified = R15.e1_selection(
        distribution=distribution, deterministic=candidate_deterministic, snapshot_digest="snapshot", probe_seed=3,
        training_mode=True,
    )
    assert R15.action_identity(unchanged) == R15.action_identity(candidate_deterministic)
    assert triggered is False
    assert mode == "DETERMINISTIC_EXPLOIT_NO_E1_TRIGGER"
    assert modified is None


def test_e2_only_rescues_training_no_assign_deadlocks_and_keeps_no_assign_in_support() -> None:
    distribution = _distribution()
    deterministic_no_assign = next(action for action in distribution if action["kind"] == "NO_ASSIGN")

    selected, triggered, mode, conditional_probability = R15.e2_selection(
        distribution=distribution, deterministic=deterministic_no_assign, snapshot_digest="snapshot", probe_seed=11,
        training_mode=True,
    )
    candidate_mass = sum(float(action["policy_probability"]) for action in distribution if action["kind"] == "CANDIDATE")
    assert triggered is True
    assert mode == "TRAINING_ONLY_CONDITIONAL_NON_NO_ASSIGN"
    assert R15.action_family(selected) == "CANDIDATE"
    assert conditional_probability == float(selected["policy_probability"]) / candidate_mass
    assert any(action["kind"] == "NO_ASSIGN" for action in distribution)

    inference, triggered_inference, mode_inference, modified_inference = R15.e2_selection(
        distribution=distribution, deterministic=deterministic_no_assign, snapshot_digest="snapshot", probe_seed=11,
        training_mode=False,
    )
    assert R15.action_identity(inference) == R15.action_identity(deterministic_no_assign)
    assert triggered_inference is False
    assert mode_inference == "DETERMINISTIC_EXPLOIT_NO_E2_TRIGGER"
    assert modified_inference is None

    candidate_deterministic = next(action for action in distribution if action["kind"] == "CANDIDATE")
    unchanged, triggered_candidate, _, _ = R15.e2_selection(
        distribution=distribution, deterministic=candidate_deterministic, snapshot_digest="snapshot", probe_seed=11,
        training_mode=True,
    )
    assert R15.action_identity(unchanged) == R15.action_identity(candidate_deterministic)
    assert triggered_candidate is False


def test_e1_pass_requires_positive_candidate_exposure_for_every_bd_cell_and_time_band() -> None:
    complete = [
        _band_row(cell_id=cell_id, time_band=time_band, candidate=True)
        for cell_id in ("BD-R1", "BD-R2")
        for time_band in R15.TIME_BANDS
    ]
    order = {
        "candidate_order_failures": 0,
        "agent_order_failures": 0,
        "combined_order_failures": 0,
    }
    assert R15.e1_pass(e0=_e0_control(), rows=complete, order=order, counters_value=_zero_integrity_counters()) is True

    missing_exposure = deepcopy(complete)
    missing_exposure[-1]["sampled_action_family"] = "NO_ASSIGN"
    assert R15.e1_pass(e0=_e0_control(), rows=missing_exposure, order=order,
                       counters_value=_zero_integrity_counters()) is False

    incomplete_bands = complete[:-1]
    assert R15.e1_pass(e0=_e0_control(), rows=incomplete_bands, order=order,
                       counters_value=_zero_integrity_counters()) is False


def test_probe_parquet_roundtrips_exact_normalized_rows(tmp_path: Path) -> None:
    path = tmp_path / "exploration_probe_results.parquet"
    rows = [_parquet_row(probe_seed=7), _parquet_row(probe_seed=2)]
    result = R15.write_probe_parquet(path, rows)

    assert path.is_file()
    assert path.read_bytes()[:4] == path.read_bytes()[-4:] == b"PAR1"
    assert result["row_count"] == 2
    assert result["roundtrip_passed"] is True
    assert result["columns"] == R15.PROBE_COLUMNS
    replay = R15.normalize_parquet_rows(pd.read_parquet(path, engine="pyarrow").to_dict("records"))
    assert replay == sorted(R15.normalize_parquet_rows(rows), key=lambda row: (
        str(row["repair_level"]), str(row["cell_id"]), str(row["snapshot_digest"]), int(row["probe_seed"]),
    ))


def test_source_and_counters_guard_audit_only_execution() -> None:
    assert R15.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_r15_minimal_exploration_deadlock_repair_selection_audit.py",
        "05_training/test_h4m_ae_ls3_bt8_r15_minimal_exploration_deadlock_repair_selection_audit.py",
    }
    assert R15.PASS_GATE == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R15_MINIMAL_EXPLORATION_DEADLOCK_REPAIR_SELECTION_AUDIT_COMPLETE"
    assert all(value is False for value in R15.EXECUTION_LOCKS.values())
    value = R15.counters()
    for key in (
        "training", "optimizer_creation", "optimizer_step", "backward", "causal_rollout",
        "simulator_execution", "candidate_generation", "candidate_regeneration", "local_search_rerun",
        "zero_loss_reevaluation", "reward_settlement", "parameter_mutation", "checkpoint_write",
        "checkpoint_mutation", "test6_access", "github_push",
    ):
        assert value[key] == 0

    source = (ROOT / "run_h4m_ae_ls3_bt8_r15_minimal_exploration_deadlock_repair_selection_audit.py").read_text(encoding="utf-8")
    for forbidden in (
        "optimizer.step(", ".backward(", "torch.optim.", "require_capability(",
        "candidate_plan_authoritative", "compute_reward_v2(",
    ):
        assert forbidden not in source
