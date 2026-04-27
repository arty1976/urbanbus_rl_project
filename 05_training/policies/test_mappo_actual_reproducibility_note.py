from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "05_training" / "policies" / "MAPPO_actual_reproducibility_note.md"

REQUIRED_PHRASES = [
    "MAPPO actual execution reproducibility note",
    "Step 58",
    "Step 59",
    "Step 60",
    "Step 61",
    "Step 62",
    "trained_model = true",
    "performance_claim_allowed = true",
    "qwen_trigger_rate = 0.0",
    "reward_version = mappo_reward_v1",
    "energy_proxy_model_version = daegu_energy_proxy_v1",
    "READY_FOR_MANUAL_EXECUTE",
    "READY_TO_EXECUTE",
    "DRY_RUN_VALIDATED",
    "strict OR semantics",
    "HistoricalReplayAdapter",
    "non-causal",
    "causal performance claims require",
]

FORBIDDEN_UNQUALIFIED_CLAIMS = [
    "causally outperformed baselines.",
    "proves causal performance improvement.",
]


def main() -> None:
    if not DOC.exists():
        raise AssertionError(f"doc not found: {DOC}")

    text = DOC.read_text(encoding="utf-8-sig")

    missing = [p for p in REQUIRED_PHRASES if p not in text]
    if missing:
        raise AssertionError(f"missing required phrases: {missing}")

    # These phrases may appear only inside the "Not allowed" section.
    for phrase in FORBIDDEN_UNQUALIFIED_CLAIMS:
        idx = text.find(phrase)
        if idx >= 0:
            prefix = text[max(0, idx - 300):idx]
            if "Not allowed" not in prefix:
                raise AssertionError(f"forbidden unqualified causal claim found: {phrase}")

    order = [
        text.index("Step 58"),
        text.index("Step 59"),
        text.index("Step 60"),
        text.index("Step 61"),
        text.index("Step 62"),
    ]
    if order != sorted(order):
        raise AssertionError("Step order is not documented correctly")

    print("[OK] Step 65 MAPPO actual reproducibility note self-test PASS")


if __name__ == "__main__":
    main()
