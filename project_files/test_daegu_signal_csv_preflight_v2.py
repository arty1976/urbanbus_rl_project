from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_DIR = PROJECT_ROOT / "05_training" / "adapters"

PREFLIGHT = ADAPTERS_DIR / "run_daegu_signal_csv_preflight_v2.py"
PATCHER = ADAPTERS_DIR / "patch_signal_feature_builder_for_daegu_csv_v2.py"
BUILDER = ADAPTERS_DIR / "build_signal_features_for_causal_simulator_v2.py"


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def make_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "toy_daegu_signal.csv"
    df = pd.DataFrame([
        {
            "시도명": "대구광역시",
            "시군구명": "북구",
            "소재지도로명주소": "대구광역시 북구 호암로 20",
            "위도": 35.882663,
            "경도": 128.592025,
            "신호등관리번호": 116,
            "신호등구분": 6,
            "신호제어방식": 3,
            "신호시간결정방식": 1,
            "점멸등운영여부": "Y",
            "점멸등운영시작시각": "00:00",
            "점멸등운영종료시각": "23:59",
            "보행자작동신호기유무": "N",
            "잔여시간표시기유무": "N",
            "시각장애인용음향신호기유무": "N",
            "신호등화순서": "황색등화",
            "신호등화시간": 1,
            "데이터기준일자": "2025-12-31",
        },
        {
            "시도명": "대구광역시",
            "시군구명": "수성구",
            "소재지도로명주소": "대구광역시 수성구 달구벌대로",
            "위도": 35.870000,
            "경도": 128.620000,
            "신호등관리번호": 117,
            "신호등구분": 1,
            "신호제어방식": 1,
            "신호시간결정방식": 2,
            "점멸등운영여부": "N",
            "점멸등운영시작시각": "",
            "점멸등운영종료시각": "",
            "보행자작동신호기유무": "Y",
            "잔여시간표시기유무": "Y",
            "시각장애인용음향신호기유무": "Y",
            "신호등화순서": "녹색등화",
            "신호등화시간": 1,
            "데이터기준일자": "2025-12-31",
        },
    ])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def test_files_exist() -> None:
    assert_true(PREFLIGHT.exists(), f"missing: {PREFLIGHT}")
    assert_true(PATCHER.exists(), f"missing: {PATCHER}")
    assert_true(BUILDER.exists(), f"missing: {BUILDER}")


def test_preflight_fixture() -> None:
    fixture_root = PROJECT_ROOT / "artifacts" / "step99_selftest"
    csv_path = make_fixture(fixture_root)
    out_dir = fixture_root / "preflight"

    cmd = [
        sys.executable,
        str(PREFLIGHT),
        "--signal-csv",
        str(csv_path),
        "--output-dir",
        str(out_dir),
    ]
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("preflight fixture failed")

    report_path = out_dir / "daegu_signal_csv_preflight_report.json"
    assert_true(report_path.exists(), "missing preflight report json")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert_true(report["step"] == 99, "bad step")
    assert_true(report["step_99_ready"] is True, "fixture should be ready")
    assert_true(report["coordinate_summary"]["valid_coordinate_rows"] == 2, "bad coordinate count")
    assert_true("신호등화시간" in report["dynamic_quarantine"]["columns_present"], "dynamic-looking column should be quarantined")
    assert_true(report["claim_guardrails"]["red_light_delay_claim_allowed"] is False, "red-light guardrail broken")


def test_patcher_and_step98_builder_still_compile() -> None:
    completed = subprocess.run(
        [sys.executable, str(PATCHER)],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("builder patcher failed")

    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(BUILDER)],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("patched builder compile failed")


def main() -> None:
    test_files_exist()
    test_preflight_fixture()
    test_patcher_and_step98_builder_still_compile()
    print("[OK] Step 99 Daegu signal CSV preflight self-test PASS")


if __name__ == "__main__":
    main()
