from __future__ import annotations

from pathlib import Path


REPORT = Path("05_training/evaluation/phase2_toy_causal_smoke_pipeline_status.md")

REQUIRED_TEXT = [
    "Phase 2 Toy Causal Smoke Pipeline Status Report",
    "Steps 77 through 94",
    "CausalSimulatorAdapter",
    "12-KPI reward_v1",
    "toy MAPPO smoke training",
    "smoke checkpoint",
    "A-family smoke matrix evaluation",
    "trained_model = false",
    "performance_claim_allowed = false",
    "smoke_training_only = true",
    "smoke_evaluation_only = true",
    "matrix_smoke_evaluation_only = true",
    "MAPPO policy convergence",
    "Real operational improvement in Daegu",
    "not ready for performance claims",
    "Step 96: realism gap analysis",
]


def main() -> None:
    if not REPORT.exists():
        raise SystemExit(f"[FAIL] missing report: {REPORT}")

    text = REPORT.read_text(encoding="utf-8-sig")
    missing = [item for item in REQUIRED_TEXT if item not in text]
    if missing:
        raise SystemExit(f"[FAIL] status report missing required text: {missing}")

    print("[OK] Step 95 status report self-test PASS")
    print(f"[OK] validated: {REPORT}")


if __name__ == "__main__":
    main()
