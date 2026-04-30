from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT / "05_training" / "adapters"
DOC_PATH = ADAPTER_DIR / "tensor_db_data_availability_audit.md"
REQ_PATH = ADAPTER_DIR / "tensor_db_data_requirements_v2.json"
INSPECTOR_PATH = ADAPTER_DIR / "inspect_tensor_db_available_fields.py"


def load_module():
    spec = importlib.util.spec_from_file_location("inspect_tensor_db_available_fields", INSPECTOR_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to import inspector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing required text: {needle}")


def main() -> None:
    for path in [DOC_PATH, REQ_PATH, INSPECTOR_PATH]:
        if not path.exists():
            raise AssertionError(f"missing required file: {path}")

    doc = DOC_PATH.read_text(encoding="utf-8-sig")
    for needle in [
        "Purpose",
        "Classification rules",
        "Candidate v2 simulator data requirements",
        "Tensor DB/source relation inspection scope",
        "Availability matrix",
        "What can be used in v2 required contract",
        "What can be used only as optional/proxy",
        "What must remain missing/assumed",
        "Paper-claim guardrails",
        "How Step 97 constrains Step 98",
        "How Step 97 leads to Step 99 missing data acquisition plan",
        "Local execution command for actual DB inspection",
        "2023 CSV 재적재 금지",
        "기존 tensor DB 덮어쓰기 금지",
    ]:
        assert_contains(doc, needle)

    prohibited_phrases = [
        "performance improvement confirmed",
        "causal performance confirmed",
        "fleet reduction benefit confirmed",
        "paper-level performance claim allowed",
    ]
    lowered_doc = doc.lower()
    for phrase in prohibited_phrases:
        if phrase in lowered_doc:
            raise AssertionError(f"prohibited claim phrase found: {phrase}")

    req = json.loads(REQ_PATH.read_text(encoding="utf-8-sig"))
    classes = set(req.get("classification_classes", []))
    if classes != {"observed", "derived", "proxy", "assumed", "missing"}:
        raise AssertionError(f"unexpected classes: {classes}")

    fields = req.get("candidate_fields", [])
    if len(fields) != 45:
        raise AssertionError(f"expected 45 candidate fields, got {len(fields)}")

    for field in fields:
        cls = field["availability_class"]
        if cls not in classes:
            raise AssertionError(f"bad availability class: {field}")
        if field.get("is_allowed_in_v2_required_contract") and cls not in {"observed", "derived"}:
            raise AssertionError(f"non-observed/derived required candidate: {field['field_name']}")
        if cls in {"missing", "assumed"}:
            if field.get("paper_level_claim_allowed") is not False:
                raise AssertionError(f"missing/assumed must block paper claim: {field['field_name']}")
            if field.get("causal_performance_claim_allowed") is not False:
                raise AssertionError(f"missing/assumed must block causal claim: {field['field_name']}")
        if cls == "proxy":
            if field.get("paper_level_claim_allowed") is not False:
                raise AssertionError(f"proxy must block paper claim: {field['field_name']}")
            if field.get("observed_kpi") is not False:
                raise AssertionError(f"proxy must not be observed KPI: {field['field_name']}")

    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        report = mod.run_self_test(output_root=out)
        if not (out / "tensor_db_available_fields_report.json").exists():
            raise AssertionError("self-test JSON report not created")
        if not (out / "tensor_db_available_fields_report.md").exists():
            raise AssertionError("self-test markdown report not created")
        if len(report.get("field_evaluations", [])) != 45:
            raise AssertionError("self-test did not evaluate all fields")

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, str(INSPECTOR_PATH), "--self-test", "--output-root", tmp],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(f"inspector CLI self-test failed\nSTDOUT={result.stdout}\nSTDERR={result.stderr}")
        if "self-test PASS" not in result.stdout:
            raise AssertionError(f"expected PASS output, got: {result.stdout}")

    print("[OK] Step 97 tensor DB data availability audit self-test PASS")


if __name__ == "__main__":
    main()
