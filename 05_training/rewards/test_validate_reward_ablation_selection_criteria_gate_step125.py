from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def load_json(path: Path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f: return json.load(f)
        except UnicodeDecodeError: continue
    raise RuntimeError(f"failed to read json: {path}")

def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)

def suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    materializer = root / "05_training" / "rewards" / "reward_ablation_selection_criteria_gate_step125.py"
    validator = root / "05_training" / "rewards" / "validate_reward_ablation_selection_criteria_gate_step125.py"
    spec = root / "05_training" / "rewards" / "reward_ablation_selection_criteria_gate_step125.json"
    s = suffix()
    output_root = root / "artifacts" / "rewards" / f"reward_ablation_selection_criteria_gate_step125_selftest_{s}"
    validation_root = root / "artifacts" / "rewards" / f"reward_ablation_selection_criteria_gate_step125_validation_{s}"
    res = subprocess.run([sys.executable, str(materializer), "--spec", str(spec), "--output-root", str(output_root)], cwd=root, text=True, capture_output=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr); raise SystemExit("[FAIL] Step 125 materializer failed")
    manifest = output_root / "reward_ablation_selection_criteria_gate_manifest_step125.json"
    val = subprocess.run([sys.executable, str(validator), "--manifest", str(manifest), "--output-root", str(validation_root)], cwd=root, text=True, capture_output=True)
    print(val.stdout)
    if val.returncode != 0:
        print(val.stderr); raise SystemExit("[FAIL] Step 125 validator failed")
    report = load_json(validation_root / "reward_ablation_selection_criteria_gate_step125_validation_report.json")
    expected = {"audit_status": "PASS", "gate_status": "PASS_SELECTION_CRITERIA_DEFINED_NO_WINNER_SELECTED", "next_status": "READY_FOR_STEP126_REWARD_ABLATION_ACTUAL_RESULT_INGESTION_PREFLIGHT", "candidate_count": 6, "actual_results": False, "winner_selected": False, "train_with_this_reward_allowed": False, "failure_count": 0}
    for k, v in expected.items():
        if report.get(k) != v: raise SystemExit(f"[FAIL] {k}: expected={v!r}, actual={report.get(k)!r}")
    bad = load_json(manifest)
    bad["winner_selected"] = True
    bad["source_spec_snapshot"]["guard_flags"]["winner_selected"] = True
    bad_path = output_root / "bad_winner_selected_manifest_step125.json"
    dump_json(bad_path, bad)
    bad_root = root / "artifacts" / "rewards" / f"reward_ablation_selection_criteria_gate_step125_bad_validation_{s}"
    bad_res = subprocess.run([sys.executable, str(validator), "--manifest", str(bad_path), "--output-root", str(bad_root)], cwd=root, text=True, capture_output=True)
    if bad_res.returncode == 0:
        print(bad_res.stdout); raise SystemExit("[FAIL] bad winner_selected manifest unexpectedly passed")
    bad_report = load_json(bad_root / "reward_ablation_selection_criteria_gate_step125_validation_report.json")
    failed = {c["check_id"] for c in bad_report.get("checks", []) if not c.get("passed", False)}
    if "manifest_guard_winner_selected" not in failed: raise SystemExit("[FAIL] winner_selected guard did not fail")
    print("[OK] Step 125 reward ablation selection criteria gate self-test PASS")
    print("[DONE] Step 125 reward ablation selection criteria gate complete.")
if __name__ == "__main__": main()
