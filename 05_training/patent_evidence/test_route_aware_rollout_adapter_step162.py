from __future__ import annotations

import importlib.util
import json
import shutil
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ADAPTER_PATH = HERE / "route_aware_rollout_adapter_step162.py"
VALIDATOR_PATH = HERE / "validate_route_aware_rollout_adapter_step162.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


adapter = load_module(ADAPTER_PATH, "route_aware_rollout_adapter_step162")
validator = load_module(VALIDATOR_PATH, "validate_route_aware_rollout_adapter_step162")


class Step162RouteAwareRolloutAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("artifacts/patent_evidence/step162_unittest")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def test_sample_adapter_and_validator_pass(self) -> None:
        out = self.root / "sample"
        manifest = adapter.run_adapter(
            mode="sample",
            output_root=out,
            raw_events_path="",
            window_rollup_path="",
            source_manifest_path="",
            condition_id="A",
            seed=1,
            file_format="csv",
            max_events=10,
        )
        manifest_path = Path(manifest["output_files"]["adapter_manifest"])
        validated = validator.validate_manifest(manifest_path)
        self.assertEqual(validated["audit_status"], "PASS")
        self.assertGreater(validated["summary"]["normalized_event_count"], 0)
        self.assertFalse(validated["paper_level_claim_allowed"])
        self.assertFalse(validated["causal_performance_claim_allowed"])

    def test_validator_blocks_mutated_claim_flag(self) -> None:
        out = self.root / "mutated"
        manifest = adapter.run_adapter(
            mode="sample",
            output_root=out,
            raw_events_path="",
            window_rollup_path="",
            source_manifest_path="",
            condition_id="A",
            seed=1,
            file_format="csv",
            max_events=5,
        )
        manifest_path = Path(manifest["output_files"]["adapter_manifest"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["paper_level_claim_allowed"] = True
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            validator.validate_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
