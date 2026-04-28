from __future__ import annotations

from pathlib import Path


REPORT = Path("05_training/adapters/toy_causal_simulator_realism_gap_analysis.md")

REQUIRED_TEXT = [
    "Toy Causal Simulator Realism Gap Analysis",
    "causal simulator v2 design",
    "Network scale",
    "Demand Generation",
    "Passenger Queue and Service Process",
    "Bus Movement Dynamics",
    "Action Feasibility",
    "Capacity and Crowding",
    "Timetable and Dispatch Schedule",
    "Traffic and Signal Delay",
    "Calibration and Validation",
    "Reward-Hacking Risk",
    "Minimal Causal Simulator v2 Requirements",
    "Data Needed for v2",
    "Step 97 ??Causal simulator v2 contract",
    "real operational performance claims",
]


def main() -> None:
    if not REPORT.exists():
        raise SystemExit(f"[FAIL] missing report: {REPORT}")

    text = REPORT.read_text(encoding="utf-8-sig")
    missing = [item for item in REQUIRED_TEXT if item not in text]
    if missing:
        raise SystemExit(f"[FAIL] realism gap analysis missing required text: {missing}")

    print("[OK] Step 96 realism gap analysis self-test PASS")
    print(f"[OK] validated: {REPORT}")


if __name__ == "__main__":
    main()
