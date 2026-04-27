from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "05_training" / "policies" / "H200_mappo_training_runbook.md"

REQUIRED_PHRASES = [
    "Step 66",
    "H200 MAPPO training runbook draft",
    "best_mappo.pt",
    "condition_id = A",
    "qwen_train = false",
    "qwen_inference = false",
    "qwen_trigger_rate = 0.0",
    "reward_version = mappo_reward_v1",
    "energy_proxy_model_version = daegu_energy_proxy_v1",
    "k_dist_kwh_per_m = 0.0012",
    "k_acc_kwh_per_event = 0.1800",
    "k_idle_kwh_per_sec = 0.0080",
    "trained_model = true",
    "performance_claim_allowed = true",
    "validate_mappo_checkpoint.py",
    "h200_actual_checkpoint_preflight_checklist.py",
    "run_h200_actual_checkpoint_arrival_workflow.py",
    "READY_FOR_MANUAL_EXECUTE",
    "CUDA",
    "git commit hash pinned",
    "Causal superiority requires Phase 2 causal simulator evaluation.",
]

FORBIDDEN_UNQUALIFIED_CLAIMS = [
    "proves causal superiority over baselines.",
]


def main() -> None:
    if not DOC.exists():
        raise AssertionError(f"doc not found: {DOC}")

    text = DOC.read_text(encoding="utf-8-sig")

    missing = [p for p in REQUIRED_PHRASES if p not in text]
    if missing:
        raise AssertionError(f"missing required phrases: {missing}")

    for phrase in FORBIDDEN_UNQUALIFIED_CLAIMS:
        idx = text.find(phrase)
        if idx >= 0:
            prefix = text[max(0, idx - 300):idx]
            if "Not allowed" not in prefix:
                raise AssertionError(f"unqualified forbidden claim found: {phrase}")

    required_order = [
        "Required repository state on H200",
        "Required environment report",
        "Python environment setup",
        "Required local validation on H200 before training",
        "Training command template",
        "Immediate post-training validation",
        "Arrival workflow after transfer",
    ]

    positions = [text.index(p) for p in required_order]
    if positions != sorted(positions):
        raise AssertionError("runbook sections are not in the expected order")

    print("[OK] Step 66 H200 MAPPO training runbook self-test PASS")


if __name__ == "__main__":
    main()
