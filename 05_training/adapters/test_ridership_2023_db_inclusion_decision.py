from future import annotations

from pathlib import Path

PROJECT_ROOT = Path(file).resolve().parents[2]
PROJECT_LOG = PROJECT_ROOT / "project_log.md"
DECISION_DOC = PROJECT_ROOT / "05_training" / "adapters" / "ridership_2023_db_inclusion_decision.md"

def assert_true(cond: bool, msg: str) -> None:
if not cond:
raise AssertionError(msg)

def test_decision_doc() -> None:
assert_true(DECISION_DOC.exists(), f"missing decision doc: {DECISION_DOC}")
text = DECISION_DOC.read_text(encoding="utf-8-sig")

required = [
    "Step 105",
    "public.stg_daegu_stop_usage_2023",
    "row_count = 2,102,115",
    "boarding_total = 181,556,972",
    "alighting_total = 70,567,680",
    "do_not_reimport",
    "tensor_rebuild_required_now = false",
    "archive_for_future_cross_year_validation",
    "do_not_merge_into_current_2023_tensor_db",
]

for phrase in required:
    assert_true(phrase in text, f"missing decision doc phrase: {phrase}")

def test_project_log() -> None:
assert_true(PROJECT_LOG.exists(), f"missing project log: {PROJECT_LOG}")
text = PROJECT_LOG.read_text(encoding="utf-8-sig")

required = [
    "STEP_105_RIDERSHIP_2023_DB_INCLUSION_DECISION_START",
    "2023 Ridership DB Inclusion Decision",
    "public.stg_daegu_stop_usage_2023",
    "row_count = 2,102,115",
    "boarding_total = 181,556,972",
    "alighting_total = 70,567,680",
    "already_loaded_in_db = true",
    "do_not_reimport = true",
    "tensor_rebuild_required_now = false",
    "performance_claim_allowed = false",
]

for phrase in required:
    assert_true(phrase in text, f"missing project log phrase: {phrase}")

def main() -> None:
test_decision_doc()
test_project_log()
print("[OK] Step 105 ridership DB inclusion decision self-test PASS")

if name == "main":
main()
