from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()

TARGET_SOURCE = PROJECT_ROOT / "05_training" / "run_b2_rulebased_rollout.py"
TARGET_POLICY_CONFIG = PROJECT_ROOT / "artifacts" / "baseline_v1" / "B2_rulebased" / "policy_config.json"

CALIBRATION = {
    "target_headway_seconds": 600,
    "low_headway_threshold_seconds": 520,
    "high_headway_threshold_seconds": 780,
    "max_hold_seconds": 120,
    "allow_skip": True,
    "night_intervention_budget": 1,
    "calibration_version": "b2_rule_trigger_calibration_v1",
    "calibration_reason": (
        "Fix no-op leakage: previous low=360/high=900 thresholds did not intersect "
        "the proxy headway distribution enough to produce B2 interventions."
    ),
}


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak_b2_rule_calibration_v1")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
    return bak


def replace_number_patterns(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    patterns = [
        (
            "low_headway_threshold_seconds",
            [
                (r'("low_headway_threshold_seconds"\s*:\s*)\d+', r'\g<1>520'),
                (r"('low_headway_threshold_seconds'\s*:\s*)\d+", r"\g<1>520"),
                (r"(low_headway_threshold_seconds\s*=\s*)\d+", r"\g<1>520"),
                (r"(LOW_HEADWAY_THRESHOLD_SECONDS\s*=\s*)\d+", r"\g<1>520"),
                (r'(\.get\(\s*"low_headway_threshold_seconds"\s*,\s*)\d+(\s*\))', r"\g<1>520\2"),
                (r"(\.get\(\s*'low_headway_threshold_seconds'\s*,\s*)\d+(\s*\))", r"\g<1>520\2"),
            ],
        ),
        (
            "high_headway_threshold_seconds",
            [
                (r'("high_headway_threshold_seconds"\s*:\s*)\d+', r'\g<1>780'),
                (r"('high_headway_threshold_seconds'\s*:\s*)\d+", r"\g<1>780"),
                (r"(high_headway_threshold_seconds\s*=\s*)\d+", r"\g<1>780"),
                (r"(HIGH_HEADWAY_THRESHOLD_SECONDS\s*=\s*)\d+", r"\g<1>780"),
                (r'(\.get\(\s*"high_headway_threshold_seconds"\s*,\s*)\d+(\s*\))', r"\g<1>780\2"),
                (r"(\.get\(\s*'high_headway_threshold_seconds'\s*,\s*)\d+(\s*\))", r"\g<1>780\2"),
            ],
        ),
        (
            "night_intervention_budget",
            [
                (r'("night_intervention_budget"\s*:\s*)\d+', r'\g<1>1'),
                (r"('night_intervention_budget'\s*:\s*)\d+", r"\g<1>1"),
                (r"(night_intervention_budget\s*=\s*)\d+", r"\g<1>1"),
                (r"(NIGHT_INTERVENTION_BUDGET\s*=\s*)\d+", r"\g<1>1"),
                (r'(\.get\(\s*"night_intervention_budget"\s*,\s*)\d+(\s*\))', r"\g<1>1\2"),
                (r"(\.get\(\s*'night_intervention_budget'\s*,\s*)\d+(\s*\))", r"\g<1>1\2"),
            ],
        ),
    ]

    for key, reps in patterns:
        total = 0
        for pattern, repl in reps:
            text, n = re.subn(pattern, repl, text)
            total += n
        counts[key] = total

    return text, counts


def patch_source() -> dict[str, Any]:
    if not TARGET_SOURCE.exists():
        raise SystemExit(f"[STOP] source file not found: {TARGET_SOURCE}")

    backup_path = backup(TARGET_SOURCE)

    text = TARGET_SOURCE.read_text(encoding="utf-8-sig")
    patched, counts = replace_number_patterns(text)

    if all(v == 0 for v in counts.values()):
        raise SystemExit(
            "[STOP] no threshold/budget pattern was patched. "
            "Open run_b2_rulebased_rollout.py and check exact variable names."
        )

    if "b2_rule_trigger_calibration_v1" not in patched:
        marker = "\n# B2 rule trigger calibration v1: low=520, high=780, night_budget=1\n"
        patched = marker + patched

    TARGET_SOURCE.write_text(patched, encoding="utf-8")

    return {
        "source": str(TARGET_SOURCE),
        "backup": str(backup_path),
        "replacement_counts": counts,
    }


def update_dict_recursive(obj: Any) -> bool:
    changed = False

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in CALIBRATION:
                if obj[key] != CALIBRATION[key]:
                    obj[key] = CALIBRATION[key]
                    changed = True
            changed = update_dict_recursive(value) or changed

        # Make sure a clear top-level rule params block exists if this is root-ish.
        if "b2_rule_calibration" not in obj:
            obj["b2_rule_calibration"] = CALIBRATION.copy()
            changed = True

    elif isinstance(obj, list):
        for item in obj:
            changed = update_dict_recursive(item) or changed

    return changed


def patch_policy_config() -> dict[str, Any]:
    if not TARGET_POLICY_CONFIG.exists():
        return {
            "policy_config": str(TARGET_POLICY_CONFIG),
            "exists": False,
            "patched": False,
        }

    backup_path = backup(TARGET_POLICY_CONFIG)

    with open(TARGET_POLICY_CONFIG, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    changed = update_dict_recursive(payload)

    with open(TARGET_POLICY_CONFIG, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "policy_config": str(TARGET_POLICY_CONFIG),
        "exists": True,
        "patched": bool(changed),
        "backup": str(backup_path),
    }


def main() -> None:
    report = {
        "step": "B2-Rule-Calibration",
        "calibration": CALIBRATION,
        "source_patch": patch_source(),
        "policy_config_patch": patch_policy_config(),
        "claim_guards": {
            "strict_canonical": False,
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
        },
    }

    out_dir = PROJECT_ROOT / "artifacts" / "baseline_v1" / "B2_rulebased_calibration_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "b2_rule_calibration_patch_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] B2 rule calibration patch completed")
    print("[OK] source:", report["source_patch"]["source"])
    print("[OK] replacement_counts:", report["source_patch"]["replacement_counts"])
    print("[OK] policy_config_exists:", report["policy_config_patch"]["exists"])
    print("[OK] report:", out_path)


if __name__ == "__main__":
    main()
