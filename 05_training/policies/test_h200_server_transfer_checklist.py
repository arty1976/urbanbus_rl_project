from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "05_training" / "policies" / "H200_server_transfer_checklist.md"

REQUIRED_PHRASES = [
    "Step 72",
    "H200 server transfer checklist",
    "PINNED_COMMIT_HASH",
    "git rev-parse HEAD",
    "git checkout <PINNED_COMMIT_HASH>",
    "generate_h200_environment_report.py",
    "--require-cuda",
    "--require-h200-name",
    "--require-clean-git",
    "run_h200_training_local_validation_bundle.py",
    "bundle_status = EXECUTION_VALIDATED",
    "train_mappo_actual.py",
    "--dry-run-contract",
    "validate_mappo_checkpoint.py",
    "h200_actual_checkpoint_preflight_checklist.py",
    "validate_h200_actual_checkpoint_folder.py",
    "run_h200_actual_checkpoint_arrival_workflow.py",
    "best_mappo.pt",
    "training_config.json",
    "checkpoint_manifest.json",
    "environment_report.json",
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
    "READY_FOR_MANUAL_EXECUTE",
    "Causal performance claims require Phase 2 causal simulator evaluation.",
]

SECTION_ORDER = [
    "Commit pinning before transfer",
    "Recommended H200 checkout",
    "Minimum source files that must exist on H200",
    "Minimum artifacts that must be available on H200",
    "H200 environment report command",
    "Local validation bundle command on H200",
    "Training CLI dry-run contract on H200",
    "Post-training validation on H200",
    "Required return package from H200 to notebook",
    "Notebook arrival workflow after return",
    "Claim boundary",
    "Stop conditions",
]


def main() -> None:
    if not DOC.exists():
        raise AssertionError(f"doc not found: {DOC}")

    text = DOC.read_text(encoding="utf-8-sig")

    missing = [p for p in REQUIRED_PHRASES if p not in text]
    if missing:
        raise AssertionError(f"missing required phrases: {missing}")

    positions = [text.index(p) for p in SECTION_ORDER]
    if positions != sorted(positions):
        raise AssertionError("checklist sections are not in the expected order")

    forbidden = "causally outperformed baselines"
    idx = text.find(forbidden)
    if idx >= 0:
        prefix = text[max(0, idx - 200):idx]
        if "Not allowed" not in prefix:
            raise AssertionError("unqualified causal superiority phrase found")

    print("[OK] Step 72 H200 server transfer checklist self-test PASS")


if __name__ == "__main__":
    main()
