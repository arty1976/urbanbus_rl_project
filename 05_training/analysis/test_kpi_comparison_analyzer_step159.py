from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List


def run(cmd: List[str], cwd: Path, expect_ok: bool = True) -> subprocess.CompletedProcess:
    cp = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    if expect_ok and cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command failed (expected ok): {cmd}")
    if (not expect_ok) and cp.returncode == 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command unexpectedly passed (expected fail): {cmd}")
    return cp


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def read_csv_dicts(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    analysis_dir = project_root / "05_training" / "analysis"
    gen = analysis_dir / "kpi_comparison_analyzer_step159.py"
    val = analysis_dir / "validate_kpi_comparison_analyzer_step159.py"

    selftest_root = project_root / "artifacts" / "analysis" / "kpi_comparison_analyzer_step159_selftest"
    selftest_root.mkdir(parents=True, exist_ok=True)

    # ---- Test 1: sample-mode end-to-end PASS ------------------------------
    out1 = selftest_root / "sample_ok"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--output-root",
            str(out1),
            "--sample-mode",
            "--baseline-condition",
            "B0",
        ],
        cwd=project_root,
        expect_ok=True,
    )
    manifest1 = out1 / "kpi_comparison_manifest_step159.json"
    run([sys.executable, str(val), "--manifest", str(manifest1)], cwd=project_root, expect_ok=True)

    # ---- Test 2: tampered guard with train_allowed=true is rejected -------
    guard1 = out1 / "claim_guard_status_step159.json"
    tamper_dir2 = selftest_root / "tamper_train_allowed"
    tamper_dir2.mkdir(parents=True, exist_ok=True)
    tampered_guard2 = tamper_dir2 / "claim_guard_status_step159.json"
    payload2 = json.loads(guard1.read_text(encoding="utf-8"))
    payload2["train_allowed"] = True
    tampered_guard2.write_text(json.dumps(payload2, ensure_ascii=False, indent=2), encoding="utf-8")
    run(
        [
            sys.executable,
            str(val),
            "--manifest",
            str(manifest1),
            "--guard",
            str(tampered_guard2),
        ],
        cwd=project_root,
        expect_ok=False,
    )

    # ---- Test 3: tampered guard with winner_selected=true is rejected -----
    tamper_dir3 = selftest_root / "tamper_winner_selected"
    tamper_dir3.mkdir(parents=True, exist_ok=True)
    tampered_guard3 = tamper_dir3 / "claim_guard_status_step159.json"
    payload3 = json.loads(guard1.read_text(encoding="utf-8"))
    payload3["winner_selected"] = True
    tampered_guard3.write_text(json.dumps(payload3, ensure_ascii=False, indent=2), encoding="utf-8")
    run(
        [
            sys.executable,
            str(val),
            "--manifest",
            str(manifest1),
            "--guard",
            str(tampered_guard3),
        ],
        cwd=project_root,
        expect_ok=False,
    )

    # ---- Test 4: strict mode with missing KPIs is rejected ----------------
    partial_csv = selftest_root / "partial_kpis.csv"
    header = [
        "condition_id",
        "seed",
        "time_band",
        "window_count",
        "cv_headway",
        "avg_wait_seconds",
        "bunching_rate",
        "on_time_rate",
        "energy_proxy",
        "passenger_service_rate",
    ]
    rows: List[List[object]] = []
    for cond in ["B0", "A"]:
        for seed in [1, 2]:
            for band in ["AM_PEAK", "PM_PEAK"]:
                rows.append([cond, seed, band, 96, 0.4, 320.0, 0.18, 0.78, 1500.0, 0.93])
    write_csv(partial_csv, header, rows)

    out4 = selftest_root / "strict_missing_fail"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--output-root",
            str(out4),
            "--input-csv",
            str(partial_csv),
            "--baseline-condition",
            "B0",
            "--strict",
        ],
        cwd=project_root,
        expect_ok=False,
    )

    # Same file should pass without --strict (has 6 KPIs, meets minimum).
    out4b = selftest_root / "non_strict_partial_ok"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--output-root",
            str(out4b),
            "--input-csv",
            str(partial_csv),
            "--baseline-condition",
            "B0",
        ],
        cwd=project_root,
        expect_ok=True,
    )

    # ---- Test 5: missing baseline condition is rejected -------------------
    no_baseline_csv = selftest_root / "no_baseline.csv"
    rows5: List[List[object]] = []
    for cond in ["B1", "B2", "A"]:
        for seed in [1, 2]:
            for band in ["AM_PEAK", "PM_PEAK"]:
                rows5.append([cond, seed, band, 96, 0.4, 320.0, 0.18, 0.78, 1500.0, 0.93])
    write_csv(no_baseline_csv, header, rows5)

    out5 = selftest_root / "missing_baseline_fail"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--output-root",
            str(out5),
            "--input-csv",
            str(no_baseline_csv),
            "--baseline-condition",
            "B0",
        ],
        cwd=project_root,
        expect_ok=False,
    )

    # ---- Test 6: baseline mean = 0 yields null pct_delta without crash ----
    zero_baseline_csv = selftest_root / "zero_baseline.csv"
    rows6: List[List[object]] = []
    for cond in ["B0", "A"]:
        for seed in [1, 2]:
            for band in ["AM_PEAK", "PM_PEAK"]:
                cv = 0.4
                wait = 320.0
                bunch = 0.18
                on_time = 0.78
                energy = 1500.0
                svc = 0.93
                # intervention_rate stays at 0 for B0; condition A has 0.10
                interv = 0.0 if cond == "B0" else 0.10
                rows6.append([cond, seed, band, 96, cv, wait, bunch, on_time, energy, svc, interv])
    header6 = header + ["intervention_rate"]
    write_csv(zero_baseline_csv, header6, rows6)

    out6 = selftest_root / "zero_baseline_ok"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--output-root",
            str(out6),
            "--input-csv",
            str(zero_baseline_csv),
            "--baseline-condition",
            "B0",
        ],
        cwd=project_root,
        expect_ok=True,
    )

    delta_by_cond = out6 / "kpi_delta_by_condition_step159.csv"
    rows_out = read_csv_dicts(delta_by_cond)
    interv_rows = [
        r for r in rows_out
        if r["kpi"] == "intervention_rate" and r["condition_id"] == "A"
    ]
    if not interv_rows:
        raise RuntimeError("expected intervention_rate row for condition A")
    pct_value = interv_rows[0]["pct_delta"]
    if pct_value not in ("", None):
        raise RuntimeError(
            f"expected null pct_delta when baseline_mean=0; got {pct_value!r}"
        )

    print("[OK] Step 159 KPI comparison analyzer self-test PASS")
    print(f"[OK] selftest_root: {selftest_root}")


if __name__ == "__main__":
    main()
