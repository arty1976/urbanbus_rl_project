from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = ROOT / "05_training" / "policies" / "H200_actual_mappo_final_runbook.md"

REQUIRED_FILES = [
    ROOT / "05_training" / "policies" / "h200_actual_checkpoint_preflight_checklist.py",
    ROOT / "05_training" / "policies" / "a_family_mappo_actual_execution_plan.py",
    ROOT / "05_training" / "policies" / "run_a_family_mappo_actual_matrix_from_plan.py",
    ROOT / "05_training" / "policies" / "validate_window_rollup_policy_metadata.py",
    ROOT / "05_training" / "evaluation" / "canonical_kpi_aggregator.py",
]

REQUIRED_PHRASES = [
    "Step 61",
    "H200 actual MAPPO final runbook",
    "READY_TO_EXECUTE",
    "DRY_RUN_VALIDATED",
    "run_count = 12",
    "command_count = 48",
    "rollout_command",
    "metadata_validation_command",
    "canonical_command",
    "post_canonical_validation_command",
    "trained_model = true",
    "performance_claim_allowed = true",
    "qwen_train = false",
    "qwen_inference = false",
    "qwen_trigger_rate = 0.0",
    "reward_version = mappo_reward_v1",
    "energy_proxy_model_version = daegu_energy_proxy_v1",
    "k_dist_kwh_per_m = 0.0012",
    "k_acc_kwh_per_event = 0.1800",
    "k_idle_kwh_per_sec = 0.0080",
    "HistoricalReplayAdapter",
    "causal performance claim allowed = false",
    "--dry-run",
    "--execute",
]

FORBIDDEN_CLAIMS = [
    "causally outperformed baselines",
    "causal superiority proven",
]


def main() -> None:
    if not RUNBOOK.exists():
        raise AssertionError(f"runbook not found: {RUNBOOK}")

    text = RUNBOOK.read_text(encoding="utf-8-sig")

    missing_files = [str(p) for p in REQUIRED_FILES if not p.exists()]
    if missing_files:
        raise AssertionError(f"required prior files missing: {missing_files}")

    missing_phrases = [p for p in REQUIRED_PHRASES if p not in text]
    if missing_phrases:
        raise AssertionError(f"runbook missing required phrases: {missing_phrases}")

    forbidden_present = [p for p in FORBIDDEN_CLAIMS if p in text and "Do not write" not in text]
    if forbidden_present:
        raise AssertionError(f"runbook contains unsupported causal claims: {forbidden_present}")

    order = [
        text.index("rollout_command"),
        text.index("metadata_validation_command"),
        text.index("canonical_command"),
        text.index("post_canonical_validation_command"),
    ]
    if order != sorted(order):
        raise AssertionError("command order is not documented in the required order")

    if text.count("causal") < 8:
        raise AssertionError("runbook does not emphasize causal claim boundaries enough")

    if text.count("BLOCKED") < 2:
        raise AssertionError("runbook does not document blocked/stop behavior enough")

    print("[OK] Step 61 H200 actual MAPPO final runbook self-test PASS")


if __name__ == "__main__":
    main()
