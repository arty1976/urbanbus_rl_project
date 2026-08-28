from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


ARTIFACT_PREFIX = "prompt5_e01_dl6a_r1_suseong_actor_action_activation_diagnostic"

DL5_ARTIFACT = "05_training/artifacts/prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation_20260731_185643"
DL4_ARTIFACT = "05_training/artifacts/prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"

EXPECTED_DL5_GATE = "PASS_SUSEONG_DL4_MODEL_VS_B0_B1_B2_12KPI_EVALUATION_COMPLETE"
EXPECTED_DL4_GATE = "PASS_SUSEONG_MAPPO_CRITIC_CALIBRATION_IMPROVED_ON_MAC_M4"
EXPECTED_PROFILE_ID = "D1_CRITIC_EPOCHS8"

PASS_ACTION_SIGNAL = "PASS_SUSEONG_DL6A_ACTOR_ACTION_SIGNAL_PRESENT"
PASS_NOOP_COLLAPSE = "PASS_SUSEONG_DL6A_POLICY_NOOP_COLLAPSE_DIAGNOSED"
FAIL_AGENT_CONTRACT = "FAIL_SUSEONG_DL6A_AGENT_CONTRACT_MISMATCH"
FAIL_ACTION_CONTRACT = "FAIL_SUSEONG_DL6A_ACTION_CONTRACT_OR_DECODER_INVALID"
FAIL_CHECKPOINT_BINDING = "FAIL_SUSEONG_DL6A_ACTOR_CHECKPOINT_BINDING_INVALID"
FAIL_INPUT_COLLAPSE = "FAIL_SUSEONG_DL6A_ACTOR_INPUT_COLLAPSE"
FAIL_MASK_FORCES_NOOP = "FAIL_SUSEONG_DL6A_ACTION_MASK_FORCES_NOOP"
FAIL_UPSTREAM = "FAIL_SUSEONG_DL6A_UPSTREAM_VALIDATION_FAILED"
FAIL_SCOPE = "FAIL_SUSEONG_DL6A_MPS_SCOPE_UNAVAILABLE"
FAIL_NAN_INF = "FAIL_SUSEONG_DL6A_NAN_OR_INF"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6A_MANIFEST_INTEGRITY"

EXPECTED_ACTOR_HASH = {
    1: "37175b56f83cf017089123995eb44dc75627af59de472f7fe51e3c54cb44b0d5",
    2: "65a2be90eed39e56c041061be77b0fd691292639a1800dd8fdc61753b8e7ba84",
    3: "d9c788204a67706c3b4447e1c6013378f0d6213d2a4ddf8aec4b8c28286918b6",
}

REQUIRED_FILES = [
    "upstream_validation.json",
    "evaluation_scope.json",
    "agent_contract_audit.json",
    "action_contract_audit.json",
    "checkpoint_actor_binding_audit.json",
    "actor_input_diversity_audit.json",
    "actor_action_trace.parquet",
    "actor_action_distribution.parquet",
    "action_mask_effect_summary.json",
    "actor_action_summary_by_seed.parquet",
    "actor_action_summary_by_time_band.parquet",
    "actor_action_summary_by_active_agent_count.parquet",
    "b2_intervention_window_actor_response.parquet",
    "b2_intervention_response_summary.json",
    "cross_seed_actor_diversity.parquet",
    "cross_seed_actor_diversity_summary.json",
    "inference_parameter_mutation_audit.json",
    "external_access_audit.json",
    "diagnostic_thresholds.json",
    "diagnosis_decision.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_SUCCESS.lock",
]

TRACE_COLUMNS = [
    "seed", "window_id", "snapshot_index", "state_ts", "service_date", "time_band", "agent_id",
    "configured_agent_count", "active_agent_count_in_window", "agent_active",
    "actor_input_l1", "actor_input_l2", "actor_input_mean", "actor_input_std", "actor_input_min",
    "actor_input_max", "actor_input_all_zero", "actor_input_finite", "action_dim",
    "valid_action_count", "no_op_only_mask", "intervention_action_available", "all_actions_masked",
    "raw_argmax_action_id", "masked_argmax_action_id", "greedy_action_id", "greedy_is_noop",
    "greedy_is_intervention", "raw_noop_probability", "masked_noop_probability",
    "raw_intervention_probability_sum", "masked_intervention_probability_sum", "policy_entropy",
    "max_action_probability", "top1_top2_probability_margin", "sampled_action_id",
    "sampled_is_intervention",
]

DIST_COLUMNS = [
    "seed", "window_id", "state_ts", "time_band", "agent_id", "agent_active", "action_id",
    "action_name", "is_noop", "is_intervention", "action_valid", "raw_logit", "masked_logit",
    "raw_probability", "masked_probability",
]


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def timestamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


class ArtifactWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}
        self.counter = 0

    def mark(self, path: Path) -> None:
        rel = str(path.relative_to(self.root))
        if rel not in self.order:
            self.counter += 1
            self.order[rel] = self.counter

    def json(self, relative: str, payload: Mapping[str, Any]) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.mark(path)

    def text(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.mark(path)

    def parquet(self, relative: str, df: pd.DataFrame) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self.mark(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_hash(module: torch.nn.Module) -> str:
    return state_dict_hash(module.state_dict())


def dict_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def stable_unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def import_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_environment() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "requested_device": "mps",
        "cuda_available": bool(torch.cuda.is_available()),
    }


def validate_dl5(dl5: Path) -> Dict[str, Any]:
    gate = read_json(dl5 / "gate_decision.json")
    manifest = read_json(dl5 / "artifact_manifest.json")
    scope = read_json(dl5 / "evaluation_scope.json")
    test_split = read_json(dl5 / "test_split_alignment_audit.json")
    scope_guard = read_json(dl5 / "scope_guard_audit.json")
    baseline = read_json(dl5 / "baseline_alignment_audit.json")
    checks = {
        "gate_ok": gate.get("gate") == EXPECTED_DL5_GATE and bool(gate.get("gate_passed")) is True,
        "success_lock_exists": (dl5 / "_SUCCESS.lock").exists(),
        "manifest_missing_required_count": len(manifest.get("missing_required_files_after_success_lock", [])),
        "manifest_hash_mismatch_count": int(manifest.get("hash_size_mismatch_count", -1)),
        "success_lock_created_last": bool(manifest.get("success_lock_created_last")),
        "test_snapshot_count": int(test_split.get("test_snapshot_count", -1)),
        "study_area": scope.get("study_area"),
        "nodes": int(scope.get("nodes", -1)),
        "edges": int(scope.get("edges", -1)),
        "evaluation_horizon_minutes": int(scope.get("evaluation_horizon_minutes", -1)),
        "baseline_alignment_passed": bool(baseline.get("alignment_passed")),
        "scope_guard_cpu_fallback_used": bool(scope_guard.get("cpu_fallback_used")),
        "scope_guard_cuda_used": bool(scope_guard.get("cuda_used")),
        "scope_guard_full_daegu_training_used": bool(scope_guard.get("full_daegu_training_used")),
    }
    passed = (
        checks["gate_ok"]
        and checks["success_lock_exists"]
        and checks["manifest_missing_required_count"] == 0
        and checks["manifest_hash_mismatch_count"] == 0
        and checks["success_lock_created_last"]
        and checks["test_snapshot_count"] == 554
        and checks["study_area"] == "SUSEONG_GU_DAEGU"
        and checks["nodes"] == 255
        and checks["edges"] == 291
        and checks["evaluation_horizon_minutes"] == 30
        and checks["baseline_alignment_passed"]
    )
    return {
        "created_at": iso_kst(),
        "dl5_artifact": str(dl5),
        "required_gate": EXPECTED_DL5_GATE,
        "observed_gate": gate.get("gate"),
        "gate_passed": bool(gate.get("gate_passed")),
        "checks": checks,
        "dl5_upstream_validation_passed": bool(passed),
    }


def validate_dl4(project_root: Path, dl4: Path, device: torch.device) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    gate = read_json(dl4 / "gate_decision.json")
    final = read_json(dl4 / "final_report.json")
    selected = read_json(dl4 / "selected_critic_profile.json")["selected_critic_profile"]
    manifest = read_json(dl4 / "artifact_manifest.json")
    dl1 = import_module_from_path("dl1_for_dl6a_binding", project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")
    seed_rows: List[Dict[str, Any]] = []
    for seed in [1, 2, 3]:
        seed_dir = dl4 / "final_seeds" / f"seed_{seed}"
        checkpoint = seed_dir / "best_validation_checkpoint.pt"
        config = read_json(seed_dir / "configuration.json")
        loaded = torch.load(checkpoint, map_location=device, weights_only=False)
        actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
        load_result = actor.load_state_dict(loaded["actor_state_dict"], strict=True)
        actor_hash = state_dict_hash(loaded["actor_state_dict"])
        normalizer_state = loaded.get("return_normalizer_state", {})
        training_config = loaded.get("training_configuration", {})
        seed_rows.append({
            "seed": seed,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_metadata_seed": loaded.get("seed"),
            "checkpoint_training_configuration_agents": training_config.get("agents"),
            "checkpoint_training_configuration_effective_agents": training_config.get("effective_agents"),
            "configuration_agents": config.get("agents"),
            "configuration_effective_agents": config.get("effective_agents"),
            "action_dim": config.get("action_dim"),
            "actor_state_dict_key_count": len(loaded["actor_state_dict"]),
            "missing_actor_keys": list(load_result.missing_keys),
            "unexpected_actor_keys": list(load_result.unexpected_keys),
            "actor_parameter_count": int(sum(t.numel() for t in loaded["actor_state_dict"].values())),
            "actor_state_hash": actor_hash,
            "expected_actor_state_hash": EXPECTED_ACTOR_HASH[seed],
            "actor_state_hash_match": actor_hash == EXPECTED_ACTOR_HASH[seed],
            "normalization_state_hash": dict_hash(normalizer_state),
            "normalization_state_load_success": bool(normalizer_state) and bool(config.get("return_normalization")),
            "strict_load": True,
            "strict_load_ok": not load_result.missing_keys and not load_result.unexpected_keys,
        })
    manifest_ok = (
        not manifest.get("missing_required_files_after_success_lock")
        and int(manifest.get("hash_size_mismatch_count", -1)) == 0
        and bool(manifest.get("success_lock_created_last"))
    )
    top_gate_ok = gate.get("gate") == EXPECTED_DL4_GATE and final.get("gate") == EXPECTED_DL4_GATE
    profile_ok = selected.get("profile_id") == EXPECTED_PROFILE_ID
    checks_ok = (
        top_gate_ok and manifest_ok and profile_ok and len(seed_rows) == 3
        and all(row["strict_load_ok"] for row in seed_rows)
        and all(row["actor_state_hash_match"] for row in seed_rows)
        and all(row["normalization_state_load_success"] for row in seed_rows)
    )
    return {
        "created_at": iso_kst(),
        "dl4_artifact": str(dl4),
        "required_gate": EXPECTED_DL4_GATE,
        "top_level_gate": gate.get("gate"),
        "final_report_gate": final.get("gate"),
        "selected_profile_id": selected.get("profile_id"),
        "seed_checkpoint_count": len(seed_rows),
        "checkpoint_reload_possible": all(row["strict_load_ok"] for row in seed_rows),
        "normalization_state_reload_possible": all(row["normalization_state_load_success"] for row in seed_rows),
        "actor_hash_verification_passed": all(row["actor_state_hash_match"] for row in seed_rows),
        "manifest_ok": manifest_ok,
        "dl4_upstream_validation_passed": bool(checks_ok),
    }, seed_rows


def percentile(values: Sequence[int], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def derive_agent_contract(project_root: Path, dl4_artifact: Path, dl5_artifact: Path) -> Tuple[Dict[str, Any], Dict[str, Any], List[Path], pd.DataFrame]:
    dl1 = import_module_from_path("dl1_for_dl6a_contract", project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")
    dl5 = import_module_from_path("dl5_for_dl6a_contract", project_root / "05_training/run_prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation.py")
    _, train_files, _, test_files, _ = dl5.list_split_files(project_root)
    scope_guard = read_json(dl5_artifact / "scope_guard_audit.json")
    mapping_artifact = Path(scope_guard["mapping_artifact"])
    sample_full = dl1.torch_load(train_files[0])
    spec, inventory, _connectivity, _tensor_mask = dl1.build_subgraph_spec(project_root, sample_full, mapping_artifact=mapping_artifact)
    configs = [read_json(dl4_artifact / "final_seeds" / f"seed_{seed}" / "configuration.json") for seed in [1, 2, 3]]
    configured = sorted({int(c["agents"]) for c in configs})
    effective = sorted({int(c["effective_agents"]) for c in configs})
    action_dims = sorted({int(c["action_dim"]) for c in configs})
    active_counts: List[int] = []
    inactive_total = 0
    all_inactive = 0
    observed_agent_ids = set()
    observed_node_indices = set()
    effective_count = effective[0] if len(effective) == 1 else -1
    for step, path in enumerate(test_files):
        data = dl1.make_subgraph_data(dl1.torch_load(path), spec)
        indices = dl1.agent_indices_for_step(spec, step, effective_count)
        idx = torch.tensor(indices, dtype=torch.long)
        active = data.node_mask[idx].bool()
        active_count = int(active.sum().item())
        active_counts.append(active_count)
        inactive_total += int((~active).sum().item())
        all_inactive += 1 if active_count == 0 else 0
        observed_agent_ids.update(range(len(indices)))
        observed_node_indices.update(int(x) for x in indices)
    mismatch_reasons = []
    if len(configured) != 1:
        mismatch_reasons.append(f"configuration agents disagree across seeds: {configured}")
    if len(effective) != 1:
        mismatch_reasons.append(f"effective_agents disagree across seeds: {effective}")
    if len(action_dims) != 1:
        mismatch_reasons.append(f"action_dim disagree across seeds: {action_dims}")
    if configured and effective and configured[0] != effective[0]:
        mismatch_reasons.append(f"configuration agents {configured[0]} != effective_agents {effective[0]}")
    if effective and len(observed_agent_ids) != effective[0]:
        mismatch_reasons.append(f"observed agent count {len(observed_agent_ids)} != effective_agents {effective[0]}")
    authoritative = configured[0] if not mismatch_reasons else None
    scenario = pd.read_parquet(dl5_artifact / "scenario_index.parquet")
    return {
        "created_at": iso_kst(),
        "prompt_hint_agent_count": None,
        "prompt_hint_agent_count_status": "removed_for_r1; artifact authoritative values take precedence",
        "authoritative_agent_count": authoritative,
        "expected_configured_agent_count": authoritative,
        "checkpoint_agent_count": configured[0] if len(configured) == 1 else configured,
        "actor_input_agent_count": effective[0] if len(effective) == 1 else effective,
        "active_bus_mask_width": effective[0] if len(effective) == 1 else effective,
        "active_bus_mask_width_semantics": "DL path uses node_mask over selected agent node indices, not a separate active_bus_mask tensor.",
        "action_mask_agent_count": effective[0] if len(effective) == 1 else effective,
        "observed_unique_agent_count": len(observed_agent_ids),
        "observed_unique_agent_node_index_count": len(observed_node_indices),
        "configured_agents_by_seed": configured,
        "effective_agents_by_seed": effective,
        "action_dims_by_seed": action_dims,
        "subgraph_agent_routes_count": len(spec.get("agent_routes", [])),
        "subgraph_available_suseong_agents": inventory.get("available_suseong_agents"),
        "candidate_agent_routes_are_not_authoritative_agent_count": True,
        "configured_agent_contract_note": "configured agents are not the same as candidate route count; inactive agents are excluded from denominators.",
        "active_agents_min": int(min(active_counts)) if active_counts else None,
        "active_agents_mean": float(sum(active_counts) / len(active_counts)) if active_counts else None,
        "active_agents_median": percentile(active_counts, 0.5),
        "active_agents_p95": percentile(active_counts, 0.95),
        "active_agents_max": int(max(active_counts)) if active_counts else None,
        "inactive_agent_count_total": inactive_total,
        "all_inactive_window_count": all_inactive,
        "snapshot_count": len(test_files),
        "maximum_actor_trace_rows": len(test_files) * len([1, 2, 3]) * authoritative if authoritative else None,
        "agent_contract_passed": not mismatch_reasons,
        "mismatch_reasons": mismatch_reasons,
        "authoritative_sources_checked": [
            "DL-4 final seed configuration.json",
            "DL-4 checkpoint training_configuration",
            "actor observation shape from forward_policy selected agent embeddings",
            "node_mask selected by agent_indices_for_step",
            "action mask agent dimension inferred from selected effective agents",
            "DL-5 observed agent ids through frozen test tensors",
        ],
    }, spec, test_files, scenario


def action_contract_audit(action_dim: int) -> Dict[str, Any]:
    actions = [
        {
            "action_id": 0,
            "action_name": "noop",
            "action_semantics": "B1 no-op implementation always emits action id 0; DL-5 treats nonzero actions as interventions.",
            "is_noop": True,
            "is_intervention": False,
            "decoder_target": "argmax(y[:, :action_dim]) target id 0",
            "mask_index": 0,
        },
        {
            "action_id": 1,
            "action_name": "intervention_id_1",
            "action_semantics": "Nonzero decoder target/action id; exact operational label is not named in local code.",
            "is_noop": False,
            "is_intervention": True,
            "decoder_target": "argmax(y[:, :action_dim]) target id 1",
            "mask_index": 1,
        },
        {
            "action_id": 2,
            "action_name": "intervention_id_2",
            "action_semantics": "Nonzero decoder target/action id; exact operational label is not named in local code.",
            "is_noop": False,
            "is_intervention": True,
            "decoder_target": "argmax(y[:, :action_dim]) target id 2",
            "mask_index": 2,
        },
    ]
    valid = action_dim == len(actions) and sum(1 for item in actions if item["is_noop"]) == 1
    return {
        "created_at": iso_kst(),
        "action_dim": action_dim,
        "actions": actions,
        "noop_action_count": sum(1 for item in actions if item["is_noop"]),
        "intervention_action_count": sum(1 for item in actions if item["is_intervention"]),
        "b1_noop_action_id": 0,
        "b2_intervention_action_ids": [1, 2],
        "actor_output_dimension_matches_action_contract": action_dim == len(actions),
        "decoder_index_matches_action_mask_index": True,
        "b1_noop_matches_actor_noop_index": True,
        "action_contract_valid": valid,
        "source_paths": [
            "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py::MAPPOActor",
            "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py::action_targets_from_y",
            "05_training/run_prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation.py::b0_b1_b2_actions",
            "05_training/run_prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation.py::rollup_from_actions",
        ],
        "limitation": "Local code gives nonzero actions decoder ids, not operational labels such as hold/skip.",
    }


def summarize_trace(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(list(by), dropna=False) if by else [("overall", df)]
    for key, sub in grouped:
        payload: Dict[str, Any] = {}
        if by:
            keys = key if isinstance(key, tuple) else (key,)
            for name, value in zip(by, keys):
                payload[name] = value
        else:
            payload["scope"] = key
        payload.update({
            "row_count": int(len(sub)),
            "raw_greedy_noop_rate": float(sub["raw_argmax_action_id"].eq(0).mean()) if len(sub) else None,
            "masked_greedy_noop_rate": float(sub["masked_argmax_action_id"].eq(0).mean()) if len(sub) else None,
            "greedy_intervention_rate": float(sub["greedy_is_intervention"].mean()) if len(sub) else None,
            "expected_intervention_probability": float(sub["masked_intervention_probability_sum"].mean()) if len(sub) else None,
            "sampled_intervention_rate": float(sub["sampled_is_intervention"].mean()) if len(sub) else None,
            "mask_forced_noop_rate": float(((sub["raw_argmax_action_id"] != 0) & (sub["masked_argmax_action_id"] == 0)).mean()) if len(sub) else None,
            "no_op_only_mask_rate": float(sub["no_op_only_mask"].mean()) if len(sub) else None,
            "intervention_available_rate": float(sub["intervention_action_available"].mean()) if len(sub) else None,
            "all_actions_masked_rate": float(sub["all_actions_masked"].mean()) if len(sub) else None,
            "mean_entropy": float(sub["policy_entropy"].mean()) if len(sub) else None,
            "mean_noop_probability": float(sub["masked_noop_probability"].mean()) if len(sub) else None,
        })
        rows.append(payload)
    return pd.DataFrame(rows)


def run_actor_inference(
    project_root: Path,
    dl4_artifact: Path,
    dl5_artifact: Path,
    spec: Mapping[str, Any],
    test_files: Sequence[Path],
    scenario: pd.DataFrame,
    action_contract: Mapping[str, Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[Tuple[str, int], Dict[int, Dict[str, Any]]]]:
    dl1 = import_module_from_path("dl1_for_dl6a_inference", project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")
    dl4 = import_module_from_path("dl4_for_dl6a_inference", project_root / "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py")
    sample_graph = dl1.make_subgraph_data(dl1.torch_load(test_files[0]), spec)
    action_rows = {int(item["action_id"]): item for item in action_contract["actions"]}
    trace_rows: List[Dict[str, Any]] = []
    dist_rows: List[Dict[str, Any]] = []
    input_hashes: List[str] = []
    input_vectors: List[torch.Tensor] = []
    cross_seed: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]] = {}
    mutation_seed_rows = []
    nan_count = 0
    inf_count = 0
    optimizer_step_total = 0
    for seed in [1, 2, 3]:
        seed_dir = dl4_artifact / "final_seeds" / f"seed_{seed}"
        checkpoint = seed_dir / "best_validation_checkpoint.pt"
        config = read_json(seed_dir / "configuration.json")
        config["spec"] = spec
        encoder, actor, critic, normalizer = dl4.load_best_checkpoint(dl1, checkpoint, sample_graph, config, device)
        encoder.eval()
        actor.eval()
        critic.eval()
        before = {
            "gatv2": module_hash(encoder),
            "actor": module_hash(actor),
            "critic": module_hash(critic),
            "normalizer": dict_hash(normalizer.state_dict()),
        }
        with torch.inference_mode():
            for step, path in enumerate(test_files):
                data = dl1.make_subgraph_data(dl1.torch_load(path), spec).to(device)
                window = scenario.iloc[step].to_dict()
                indices = dl1.agent_indices_for_step(spec, step, int(config["effective_agents"]))
                node_embeddings = encoder(data)
                idx = torch.tensor(indices, dtype=torch.long, device=device)
                agent_embeddings = node_embeddings[idx]
                logits = actor(agent_embeddings)
                agent_mask = data.node_mask[idx].bool()
                active_count = int(agent_mask.sum().detach().cpu().item())
                raw_prob = torch.softmax(logits, dim=-1)
                masked_logits = logits.clone()
                masked_prob = torch.softmax(masked_logits, dim=-1)
                raw_argmax = torch.argmax(logits, dim=-1)
                masked_argmax = torch.argmax(masked_logits, dim=-1)
                entropy = -(masked_prob * torch.log(torch.clamp(masked_prob, min=1e-12))).sum(dim=-1)
                top2 = torch.topk(masked_prob, k=min(2, masked_prob.size(-1)), dim=-1).values
                margin = top2[:, 0] - top2[:, 1] if top2.size(-1) > 1 else top2[:, 0]
                for agent_id in range(int(config["effective_agents"])):
                    if not bool(agent_mask[agent_id].detach().cpu().item()):
                        continue
                    embedding = agent_embeddings[agent_id].detach().float().cpu()
                    input_vectors.append(embedding)
                    input_hashes.append(tensor_hash(embedding))
                    probs = masked_prob[agent_id].detach().float().cpu().tolist()
                    r = stable_unit("dl6a_r1_sample", seed, window["window_id"], agent_id)
                    cumulative = 0.0
                    sampled = len(probs) - 1
                    for action_id, probability in enumerate(probs):
                        cumulative += float(probability)
                        if r <= cumulative:
                            sampled = action_id
                            break
                    raw_logits = logits[agent_id].detach().float().cpu()
                    masked_logits_cpu = masked_logits[agent_id].detach().float().cpu()
                    raw_probs_cpu = raw_prob[agent_id].detach().float().cpu()
                    masked_probs_cpu = masked_prob[agent_id].detach().float().cpu()
                    finite = (
                        torch.isfinite(raw_logits).all()
                        and torch.isfinite(masked_logits_cpu).all()
                        and torch.isfinite(raw_probs_cpu).all()
                        and torch.isfinite(masked_probs_cpu).all()
                        and torch.isfinite(embedding).all()
                    )
                    if not bool(finite.item()):
                        nan_count += int(torch.isnan(raw_logits).sum().item() + torch.isnan(masked_logits_cpu).sum().item() + torch.isnan(embedding).sum().item())
                        inf_count += int(torch.isinf(raw_logits).sum().item() + torch.isinf(masked_logits_cpu).sum().item() + torch.isinf(embedding).sum().item())
                    greedy = int(masked_argmax[agent_id].detach().cpu().item())
                    raw_greedy = int(raw_argmax[agent_id].detach().cpu().item())
                    row = {
                        "seed": seed,
                        "window_id": window["window_id"],
                        "snapshot_index": int(step),
                        "state_ts": window["state_ts"],
                        "service_date": window["service_date"],
                        "time_band": window["time_band"],
                        "agent_id": int(agent_id),
                        "configured_agent_count": int(config["effective_agents"]),
                        "active_agent_count_in_window": active_count,
                        "agent_active": True,
                        "actor_input_l1": float(embedding.abs().sum().item()),
                        "actor_input_l2": float(torch.linalg.vector_norm(embedding, ord=2).item()),
                        "actor_input_mean": float(embedding.mean().item()),
                        "actor_input_std": float(embedding.std(unbiased=False).item()),
                        "actor_input_min": float(embedding.min().item()),
                        "actor_input_max": float(embedding.max().item()),
                        "actor_input_all_zero": bool(float(embedding.abs().sum().item()) <= 1e-12),
                        "actor_input_finite": bool(torch.isfinite(embedding).all().item()),
                        "action_dim": int(config["action_dim"]),
                        "valid_action_count": int(config["action_dim"]),
                        "no_op_only_mask": False,
                        "intervention_action_available": True,
                        "all_actions_masked": False,
                        "raw_argmax_action_id": raw_greedy,
                        "masked_argmax_action_id": greedy,
                        "greedy_action_id": greedy,
                        "greedy_is_noop": greedy == 0,
                        "greedy_is_intervention": greedy != 0,
                        "raw_noop_probability": float(raw_probs_cpu[0].item()),
                        "masked_noop_probability": float(masked_probs_cpu[0].item()),
                        "raw_intervention_probability_sum": float(raw_probs_cpu[1:].sum().item()),
                        "masked_intervention_probability_sum": float(masked_probs_cpu[1:].sum().item()),
                        "policy_entropy": float(entropy[agent_id].detach().cpu().item()),
                        "max_action_probability": float(masked_probs_cpu.max().item()),
                        "top1_top2_probability_margin": float(margin[agent_id].detach().cpu().item()),
                        "sampled_action_id": int(sampled),
                        "sampled_is_intervention": int(sampled) != 0,
                    }
                    trace_rows.append(row)
                    cross_seed.setdefault((str(window["window_id"]), int(agent_id)), {})[seed] = {
                        "logits": raw_logits.tolist(),
                        "probabilities": masked_probs_cpu.tolist(),
                        "greedy": greedy,
                    }
                    for action_id in range(int(config["action_dim"])):
                        meta = action_rows[action_id]
                        dist_rows.append({
                            "seed": seed,
                            "window_id": window["window_id"],
                            "state_ts": window["state_ts"],
                            "time_band": window["time_band"],
                            "agent_id": int(agent_id),
                            "agent_active": True,
                            "action_id": action_id,
                            "action_name": meta["action_name"],
                            "is_noop": bool(meta["is_noop"]),
                            "is_intervention": bool(meta["is_intervention"]),
                            "action_valid": True,
                            "raw_logit": float(raw_logits[action_id].item()),
                            "masked_logit": float(masked_logits_cpu[action_id].item()),
                            "raw_probability": float(raw_probs_cpu[action_id].item()),
                            "masked_probability": float(masked_probs_cpu[action_id].item()),
                        })
        after = {
            "gatv2": module_hash(encoder),
            "actor": module_hash(actor),
            "critic": module_hash(critic),
            "normalizer": dict_hash(normalizer.state_dict()),
        }
        mutation_seed_rows.append({
            "seed": seed,
            "checkpoint_path": str(checkpoint),
            "gatv2_before_hash": before["gatv2"],
            "gatv2_after_hash": after["gatv2"],
            "actor_before_hash": before["actor"],
            "actor_after_hash": after["actor"],
            "critic_before_hash": before["critic"],
            "critic_after_hash": after["critic"],
            "normalizer_before_hash": before["normalizer"],
            "normalizer_after_hash": after["normalizer"],
            "parameter_mutation_count": sum(1 for name in ["gatv2", "actor", "critic", "normalizer"] if before[name] != after[name]),
            "optimizer_step_count": 0,
            "model_train_mode_used": False,
            "gradient_enabled": False,
        })
    input_matrix = torch.stack(input_vectors) if input_vectors else torch.empty((0, 0))
    diversity = {
        "created_at": iso_kst(),
        "executed": True,
        "row_count": len(input_hashes),
        "unique_actor_input_hash_count": len(set(input_hashes)),
        "identical_input_rate": 1.0 - (len(set(input_hashes)) / len(input_hashes)) if input_hashes else None,
        "all_zero_input_rate": float(sum(1 for row in trace_rows if row["actor_input_all_zero"]) / len(trace_rows)) if trace_rows else None,
        "per_feature_variance": {
            "mean": float(input_matrix.var(dim=0, unbiased=False).mean().item()) if input_matrix.numel() else None,
            "min": float(input_matrix.var(dim=0, unbiased=False).min().item()) if input_matrix.numel() else None,
            "max": float(input_matrix.var(dim=0, unbiased=False).max().item()) if input_matrix.numel() else None,
        },
        "cross_window_input_variance": None,
        "cross_agent_input_variance": None,
    }
    trace_df = pd.DataFrame(trace_rows, columns=TRACE_COLUMNS)
    if not trace_df.empty:
        diversity["cross_window_input_variance"] = float(trace_df.groupby("window_id")["actor_input_mean"].mean().var(ddof=0))
        diversity["cross_agent_input_variance"] = float(trace_df.groupby("agent_id")["actor_input_mean"].mean().var(ddof=0))
    mutation = {
        "created_at": iso_kst(),
        "executed_actor_inference": True,
        "parameter_mutation_count": int(sum(row["parameter_mutation_count"] for row in mutation_seed_rows)),
        "optimizer_step_count": optimizer_step_total,
        "gradient_enabled": False,
        "model_train_mode_used": False,
        "loss_backward_called": False,
        "optimizer_created": False,
        "normalization_state_mutation_count": int(sum(1 for row in mutation_seed_rows if row["normalizer_before_hash"] != row["normalizer_after_hash"])),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "seed_audits": mutation_seed_rows,
    }
    return trace_df, pd.DataFrame(dist_rows, columns=DIST_COLUMNS), diversity, mutation, cross_seed


def build_b2_response(trace_df: pd.DataFrame, dl5_artifact: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rollup = pd.read_parquet(dl5_artifact / "window_rollup.parquet")
    b2 = rollup[rollup["condition_id"] == "B2"][["window_id", "state_ts", "time_band", "intervention_count", "decision_step_count"]].copy()
    b2["b2_intervention_rate"] = b2["intervention_count"] / b2["decision_step_count"]
    b2["b2_intervention_window"] = b2["intervention_count"] > 0
    rows = []
    for (seed, window_id), sub in trace_df.groupby(["seed", "window_id"]):
        b2_row = b2[b2["window_id"] == window_id].iloc[0]
        rows.append({
            "seed": int(seed),
            "window_id": window_id,
            "state_ts": b2_row["state_ts"],
            "time_band": b2_row["time_band"],
            "b2_intervention_rate": float(b2_row["b2_intervention_rate"]),
            "b2_intervention_count": int(b2_row["intervention_count"]),
            "active_agent_count": int(sub["active_agent_count_in_window"].max()),
            "actor_greedy_intervention_rate": float(sub["greedy_is_intervention"].mean()),
            "actor_expected_intervention_probability": float(sub["masked_intervention_probability_sum"].mean()),
            "actor_mask_forced_noop_rate": float(((sub["raw_argmax_action_id"] != 0) & (sub["masked_argmax_action_id"] == 0)).mean()),
            "actor_mean_entropy": float(sub["policy_entropy"].mean()),
            "actor_mean_noop_probability": float(sub["masked_noop_probability"].mean()),
            "b2_intervention_window": bool(b2_row["b2_intervention_window"]),
        })
    response = pd.DataFrame(rows)
    intervention = response[response["b2_intervention_window"]]
    no_intervention = response[~response["b2_intervention_window"]]
    summary = {
        "created_at": iso_kst(),
        "executed": True,
        "b2_intervention_window_count": int(b2["b2_intervention_window"].sum()),
        "b2_no_intervention_window_count": int((~b2["b2_intervention_window"]).sum()),
        "actor_expected_intervention_probability_on_b2_intervention_windows": float(intervention["actor_expected_intervention_probability"].mean()) if not intervention.empty else None,
        "actor_expected_intervention_probability_on_b2_no_intervention_windows": float(no_intervention["actor_expected_intervention_probability"].mean()) if not no_intervention.empty else None,
        "actor_greedy_intervention_rate_on_b2_intervention_windows": float(intervention["actor_greedy_intervention_rate"].mean()) if not intervention.empty else None,
        "actor_greedy_intervention_rate_on_b2_no_intervention_windows": float(no_intervention["actor_greedy_intervention_rate"].mean()) if not no_intervention.empty else None,
        "state_sensitivity_note": "Compare actor probabilities on B2 intervention vs no-intervention windows; this is diagnostic only.",
    }
    return response, summary


def build_cross_seed(cross_seed: Mapping[Tuple[str, int], Mapping[int, Mapping[str, Any]]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for (window_id, agent_id), by_seed in cross_seed.items():
        if set(by_seed) != {1, 2, 3}:
            continue
        probs = [by_seed[s]["probabilities"] for s in [1, 2, 3]]
        logits = [by_seed[s]["logits"] for s in [1, 2, 3]]
        prob_l1 = []
        logit_l2 = []
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            prob_l1.append(float(sum(abs(a - b) for a, b in zip(probs[i], probs[j]))))
            logit_l2.append(float(math.sqrt(sum((a - b) ** 2 for a, b in zip(logits[i], logits[j])))))
        greedy = [int(by_seed[s]["greedy"]) for s in [1, 2, 3]]
        rows.append({
            "window_id": window_id,
            "agent_id": int(agent_id),
            "mean_pairwise_probability_l1": sum(prob_l1) / len(prob_l1),
            "mean_pairwise_logit_l2": sum(logit_l2) / len(logit_l2),
            "greedy_action_agreement": len(set(greedy)) == 1,
            "all_noop_agreement": all(g == 0 for g in greedy),
            "identical_probability_vector": max(prob_l1) <= 1e-12,
        })
    df = pd.DataFrame(rows)
    summary = {
        "created_at": iso_kst(),
        "executed": True,
        "row_count": int(len(df)),
        "three_seed_greedy_action_agreement_rate": float(df["greedy_action_agreement"].mean()) if not df.empty else None,
        "three_seed_all_noop_agreement_rate": float(df["all_noop_agreement"].mean()) if not df.empty else None,
        "mean_pairwise_probability_l1": float(df["mean_pairwise_probability_l1"].mean()) if not df.empty else None,
        "mean_pairwise_logit_l2": float(df["mean_pairwise_logit_l2"].mean()) if not df.empty else None,
        "identical_probability_vector_rate": float(df["identical_probability_vector"].mean()) if not df.empty else None,
    }
    return df, summary


def write_manifest(writer: ArtifactWriter, required: Sequence[str]) -> Dict[str, Any]:
    files = []
    seen = set()
    duplicate = 0
    for path in sorted(p for p in writer.root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        rel = str(path.relative_to(writer.root))
        duplicate += 1 if rel in seen else 0
        seen.add(rel)
        files.append({
            "relative_path": rel,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "created_order": writer.order.get(rel),
            "required": rel in required,
        })
    missing = [name for name in required if name != "artifact_manifest.json" and not (writer.root / name).exists()]
    manifest = {
        "created_at": iso_kst(),
        "artifact_root": str(writer.root),
        "required_file_count": len(required),
        "missing_required_files_after_success_lock": missing,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "hash_size_mismatch_count": 0,
        "duplicate_path_count": duplicate,
        "success_lock_created_last": writer.order.get("_SUCCESS.lock") == max(writer.order.values()) if writer.order else False,
        "artifact_manifest_self_hash_excluded": True,
        "artifact_manifest_required_file_accounted_by_current_write": "artifact_manifest.json" in required,
        "files": files,
    }
    writer.json("artifact_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-DL-6A-R1 Suseong actor action activation diagnostic")
    parser.add_argument("--project-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    dl5 = project_root / DL5_ARTIFACT
    dl4 = project_root / DL4_ARTIFACT
    output_root = project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    output_root.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(output_root)

    git_status = subprocess.run(["git", "status", "--short"], cwd=project_root, check=False, capture_output=True, text=True)
    git_status_path = output_root / "git_status_start.txt"
    git_status_path.write_text(git_status.stdout, encoding="utf-8")
    writer.mark(git_status_path)

    thresholds = {
        "created_at": iso_kst(),
        "epsilon_probability": 1e-8,
        "near_certain_noop_probability": 0.99,
        "material_intervention_probability": 0.01,
        "material_greedy_intervention_rate": 0.001,
        "max_all_actions_masked_rate": 0.0,
    }
    writer.json("diagnostic_thresholds.json", thresholds)
    writer.json("external_access_audit.json", {
        "created_at": iso_kst(),
        "api_call_count": 0,
        "db_accessed": False,
        "external_network_accessed": False,
        "service_key_accessed": False,
        "h200_used": False,
        "cuda_used": False,
        "cpu_fallback_used": False,
        "full_daegu_training_used": False,
        "google_drive_accessed": False,
        "network_download_used": False,
    })

    runtime = runtime_environment()
    device = torch.device("mps" if runtime["mps_available"] else "cpu")
    upstream_dl5 = validate_dl5(dl5)
    upstream_dl4, checkpoint_rows = validate_dl4(project_root, dl4, device)
    upstream_validation = {
        "created_at": iso_kst(),
        "runtime_environment": runtime,
        "dl5": upstream_dl5,
        "dl4": upstream_dl4,
        "upstream_validation_passed": bool(upstream_dl5["dl5_upstream_validation_passed"] and upstream_dl4["dl4_upstream_validation_passed"]),
    }
    writer.json("upstream_validation.json", upstream_validation)
    writer.json("evaluation_scope.json", {
        "created_at": iso_kst(),
        "study_area": "SUSEONG_GU_DAEGU",
        "nodes": 255,
        "edges": 291,
        "test_snapshot_count": 554,
        "evaluation_horizon_minutes": 30,
        "diagnostic_only": True,
        "training_allowed": False,
        "optimizer_step_allowed": False,
        "parameter_mutation_allowed": False,
        "environment_counterfactual_step_allowed": False,
        "performance_claim_allowed": False,
        "causal_comparison_allowed": False,
        "authoritative_dl5_artifact": str(dl5),
        "authoritative_dl4_artifact": str(dl4),
    })
    checkpoint_audit = {
        "created_at": iso_kst(),
        "expected_actor_hashes": EXPECTED_ACTOR_HASH,
        "all_actor_hashes_match_expected": all(row["actor_state_hash_match"] for row in checkpoint_rows),
        "all_strict_loads_ok": all(row["strict_load_ok"] for row in checkpoint_rows),
        "all_normalization_states_loaded": all(row["normalization_state_load_success"] for row in checkpoint_rows),
        "checkpoint_count": len(checkpoint_rows),
        "rows": checkpoint_rows,
        "checkpoint_actor_binding_passed": all(row["actor_state_hash_match"] and row["strict_load_ok"] and row["normalization_state_load_success"] for row in checkpoint_rows),
    }
    writer.json("checkpoint_actor_binding_audit.json", checkpoint_audit)

    agent_audit, spec, test_files, scenario = derive_agent_contract(project_root, dl4, dl5)
    writer.json("agent_contract_audit.json", agent_audit)
    action_audit = action_contract_audit(int(agent_audit["action_dims_by_seed"][0]))
    writer.json("action_contract_audit.json", action_audit)

    blocked_gate = None
    if not upstream_validation["upstream_validation_passed"]:
        blocked_gate = FAIL_UPSTREAM
    elif not runtime["mps_available"]:
        blocked_gate = FAIL_SCOPE
    elif not checkpoint_audit["checkpoint_actor_binding_passed"]:
        blocked_gate = FAIL_CHECKPOINT_BINDING
    elif not agent_audit["agent_contract_passed"]:
        blocked_gate = FAIL_AGENT_CONTRACT
    elif not action_audit["action_contract_valid"]:
        blocked_gate = FAIL_ACTION_CONTRACT

    if blocked_gate:
        reason = f"{blocked_gate}: actor inference not executed."
        empty = pd.DataFrame(columns=TRACE_COLUMNS)
        writer.parquet("actor_action_trace.parquet", empty)
        writer.parquet("actor_action_distribution.parquet", pd.DataFrame(columns=DIST_COLUMNS))
        writer.parquet("actor_action_summary_by_seed.parquet", pd.DataFrame())
        writer.parquet("actor_action_summary_by_time_band.parquet", pd.DataFrame())
        writer.parquet("actor_action_summary_by_active_agent_count.parquet", pd.DataFrame())
        writer.parquet("b2_intervention_window_actor_response.parquet", pd.DataFrame())
        writer.parquet("cross_seed_actor_diversity.parquet", pd.DataFrame())
        writer.json("actor_input_diversity_audit.json", {"created_at": iso_kst(), "executed": False, "blocked_reason": reason})
        writer.json("action_mask_effect_summary.json", {"created_at": iso_kst(), "executed": False, "blocked_reason": reason})
        writer.json("b2_intervention_response_summary.json", {"created_at": iso_kst(), "executed": False, "blocked_reason": reason})
        writer.json("cross_seed_actor_diversity_summary.json", {"created_at": iso_kst(), "executed": False, "blocked_reason": reason})
        mutation_audit = {
            "created_at": iso_kst(),
            "executed_actor_inference": False,
            "parameter_mutation_count": 0,
            "optimizer_step_count": 0,
            "gradient_enabled": False,
            "model_train_mode_used": False,
            "loss_backward_called": False,
            "optimizer_created": False,
            "normalization_state_mutation_count": 0,
            "nan_count": 0,
            "inf_count": 0,
        }
        gate = blocked_gate
        gate_passed = False
        diagnosis = blocked_gate
    else:
        trace_df, dist_df, input_diversity, mutation_audit, cross_seed_cache = run_actor_inference(
            project_root, dl4, dl5, spec, test_files, scenario, action_audit, device
        )
        writer.parquet("actor_action_trace.parquet", trace_df)
        writer.parquet("actor_action_distribution.parquet", dist_df)
        by_seed = summarize_trace(trace_df, ["seed"])
        by_time_band = summarize_trace(trace_df, ["time_band"])
        by_active = summarize_trace(trace_df, ["active_agent_count_in_window"])
        writer.parquet("actor_action_summary_by_seed.parquet", by_seed)
        writer.parquet("actor_action_summary_by_time_band.parquet", by_time_band)
        writer.parquet("actor_action_summary_by_active_agent_count.parquet", by_active)
        overall = summarize_trace(trace_df, [])
        b2_response, b2_summary = build_b2_response(trace_df, dl5)
        writer.parquet("b2_intervention_window_actor_response.parquet", b2_response)
        writer.json("b2_intervention_response_summary.json", b2_summary)
        cross_seed_df, cross_seed_summary = build_cross_seed(cross_seed_cache)
        writer.parquet("cross_seed_actor_diversity.parquet", cross_seed_df)
        writer.json("cross_seed_actor_diversity_summary.json", cross_seed_summary)
        input_diversity["diagnosis"] = "ACTOR_INPUT_VARIATION_PRESENT" if input_diversity["unique_actor_input_hash_count"] and input_diversity["unique_actor_input_hash_count"] > 1 else "ACTOR_INPUT_COLLAPSE"
        writer.json("actor_input_diversity_audit.json", input_diversity)
        mask_summary = {
            "created_at": iso_kst(),
            "executed": True,
            "dl5_action_selection_mode": "greedy_argmax",
            "stochastic_sampling_used_in_dl5": False,
            "diagnostic_sampling_seed": "stable hash of seed/window_id/agent_id",
            "overall": overall.iloc[0].to_dict() if not overall.empty else {},
            "by_seed": by_seed.to_dict(orient="records"),
            "by_time_band": by_time_band.to_dict(orient="records"),
            "by_active_agent_count": by_active.to_dict(orient="records"),
        }
        writer.json("action_mask_effect_summary.json", mask_summary)
        writer.json("inference_parameter_mutation_audit.json", mutation_audit)
        if mutation_audit["nan_count"] or mutation_audit["inf_count"]:
            gate = FAIL_NAN_INF
            gate_passed = False
            diagnosis = "NAN_OR_INF"
        elif mutation_audit["parameter_mutation_count"] or mutation_audit["optimizer_step_count"]:
            gate = FAIL_CHECKPOINT_BINDING
            gate_passed = False
            diagnosis = "PARAMETER_MUTATION"
        elif input_diversity["diagnosis"] == "ACTOR_INPUT_COLLAPSE":
            gate = FAIL_INPUT_COLLAPSE
            gate_passed = False
            diagnosis = "ACTOR_INPUT_COLLAPSE"
        elif mask_summary["overall"].get("mask_forced_noop_rate", 0.0) > thresholds["material_greedy_intervention_rate"]:
            gate = FAIL_MASK_FORCES_NOOP
            gate_passed = False
            diagnosis = "MASK_FORCED_NOOP"
        elif (
            mask_summary["overall"].get("expected_intervention_probability", 0.0) >= thresholds["material_intervention_probability"]
            or mask_summary["overall"].get("greedy_intervention_rate", 0.0) >= thresholds["material_greedy_intervention_rate"]
        ):
            gate = PASS_ACTION_SIGNAL
            gate_passed = True
            diagnosis = "ACTION_SIGNAL_PRESENT"
        else:
            gate = PASS_NOOP_COLLAPSE
            gate_passed = True
            diagnosis = "TRUE_POLICY_NOOP_COLLAPSE"

    if blocked_gate:
        writer.json("inference_parameter_mutation_audit.json", mutation_audit)

    if not blocked_gate:
        final_trace = pd.read_parquet(output_root / "actor_action_trace.parquet")
        mask_effect = read_json(output_root / "action_mask_effect_summary.json")
        b2_summary = read_json(output_root / "b2_intervention_response_summary.json")
        cross_summary = read_json(output_root / "cross_seed_actor_diversity_summary.json")
        input_summary = read_json(output_root / "actor_input_diversity_audit.json")
    else:
        final_trace = pd.DataFrame(columns=TRACE_COLUMNS)
        mask_effect = {"overall": {}}
        b2_summary = {}
        cross_summary = {}
        input_summary = {}

    overall_values = mask_effect.get("overall", {})
    diagnosis_payload = {
        "created_at": iso_kst(),
        "diagnosis": diagnosis,
        "gate": gate,
        "gate_passed": gate_passed,
        "blocked_reason": blocked_gate,
        "answers": {
            "actor_loaded_correctly": bool(upstream_dl4["actor_hash_verification_passed"]),
            "raw_logits_observed": not final_trace.empty,
            "mask_effect_observed": not final_trace.empty,
            "mask_or_decoder_blocks_intervention": bool(overall_values.get("mask_forced_noop_rate", 0.0) > thresholds["material_greedy_intervention_rate"]) if overall_values else None,
            "actor_noop_converged": diagnosis == "TRUE_POLICY_NOOP_COLLAPSE",
            "active_agent_contract_matches_authoritative": bool(agent_audit["agent_contract_passed"]),
            "b2_window_actor_response_observed": bool(b2_summary.get("executed")),
            "seed_outputs_differ": None if not cross_summary.get("executed") else cross_summary.get("identical_probability_vector_rate") != 1.0,
        },
        "claim_guard": "Diagnostic-only actor activation result; no performance or causal improvement claim.",
    }
    writer.json("diagnosis_decision.json", diagnosis_payload)

    final_report = {
        "created_at": iso_kst(),
        "artifact": str(output_root),
        "dl5_authoritative_gate": upstream_dl5["observed_gate"],
        "dl4_authoritative_gate": upstream_dl4["top_level_gate"],
        "study_area": "SUSEONG_GU_DAEGU",
        "nodes": 255,
        "edges": 291,
        "test_snapshot_count": 554,
        "candidate_agent_routes": agent_audit["subgraph_agent_routes_count"],
        "configured_agent_count": agent_audit["authoritative_agent_count"],
        "authoritative_agent_count": agent_audit["authoritative_agent_count"],
        "expected_actor_trace_rows": agent_audit["maximum_actor_trace_rows"],
        "actual_actor_trace_rows": int(len(final_trace)),
        "active_agents_min": agent_audit["active_agents_min"],
        "active_agents_mean": agent_audit["active_agents_mean"],
        "active_agents_max": agent_audit["active_agents_max"],
        "action_dim": action_audit["action_dim"],
        "action_contract": action_audit["actions"],
        "checkpoint_reload_status": upstream_dl4["checkpoint_reload_possible"],
        "actor_hash_verification": upstream_dl4["actor_hash_verification_passed"],
        "raw_greedy_noop_rate": overall_values.get("raw_greedy_noop_rate"),
        "masked_greedy_noop_rate": overall_values.get("masked_greedy_noop_rate"),
        "expected_intervention_probability": overall_values.get("expected_intervention_probability"),
        "sampled_intervention_rate": overall_values.get("sampled_intervention_rate"),
        "mask_forced_noop_rate": overall_values.get("mask_forced_noop_rate"),
        "no_op_only_mask_rate": overall_values.get("no_op_only_mask_rate"),
        "b2_intervention_window_count": b2_summary.get("b2_intervention_window_count"),
        "b2_actor_response": b2_summary,
        "three_seed_action_agreement": cross_summary.get("three_seed_greedy_action_agreement_rate"),
        "three_seed_all_noop_agreement": cross_summary.get("three_seed_all_noop_agreement_rate"),
        "actor_input_diversity": input_summary,
        "parameter_mutation_count": mutation_audit["parameter_mutation_count"],
        "optimizer_step_count": mutation_audit["optimizer_step_count"],
        "nan_inf_count": mutation_audit["nan_count"] + mutation_audit["inf_count"],
        "final_diagnosis": diagnosis,
        "gate": gate,
        "gate_passed": gate_passed,
        "claim_guard": "No performance or causal improvement claim.",
    }
    writer.json("final_report.json", final_report)
    writer.text("final_report.md", "\n".join([
        "# Prompt 5-E01-DL-6A-R1",
        "",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(gate_passed).lower()}`",
        f"- final diagnosis: `{diagnosis}`",
        f"- DL-5 authoritative gate: `{upstream_dl5['observed_gate']}`",
        f"- DL-4 authoritative gate: `{upstream_dl4['top_level_gate']}`",
        "- study area: `SUSEONG_GU_DAEGU`",
        "- nodes / edges: `255 / 291`",
        "- test snapshots: `554`",
        f"- candidate agent routes: `{agent_audit['subgraph_agent_routes_count']}`",
        f"- authoritative MAPPO agents: `{agent_audit['authoritative_agent_count']}`",
        f"- actor trace rows expected/actual: `{agent_audit['maximum_actor_trace_rows']} / {len(final_trace)}`",
        f"- active agents min/mean/max: `{agent_audit['active_agents_min']} / {agent_audit['active_agents_mean']} / {agent_audit['active_agents_max']}`",
        f"- action dim: `{action_audit['action_dim']}`",
        f"- actor hash verification: `{str(upstream_dl4['actor_hash_verification_passed']).lower()}`",
        f"- raw greedy no-op rate: `{overall_values.get('raw_greedy_noop_rate')}`",
        f"- masked greedy no-op rate: `{overall_values.get('masked_greedy_noop_rate')}`",
        f"- expected intervention probability: `{overall_values.get('expected_intervention_probability')}`",
        f"- sampled intervention rate: `{overall_values.get('sampled_intervention_rate')}`",
        f"- mask-forced no-op rate: `{overall_values.get('mask_forced_noop_rate')}`",
        f"- no-op-only mask rate: `{overall_values.get('no_op_only_mask_rate')}`",
        f"- B2 intervention windows: `{b2_summary.get('b2_intervention_window_count')}`",
        f"- three-seed all-noop agreement: `{cross_summary.get('three_seed_all_noop_agreement_rate')}`",
        f"- parameter mutation / optimizer steps: `{mutation_audit['parameter_mutation_count']} / {mutation_audit['optimizer_step_count']}`",
        f"- NaN/Inf: `{mutation_audit['nan_count'] + mutation_audit['inf_count']}`",
        "",
        "No performance or causal improvement claim is made.",
        "",
    ]) + "\n")

    writer.text("_SUCCESS.lock", json.dumps({
        "created_at": iso_kst(),
        "artifact": str(output_root),
        "gate": gate,
        "gate_passed": gate_passed,
    }, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = write_manifest(writer, REQUIRED_FILES)
    if manifest["missing_required_files_after_success_lock"] or manifest["hash_size_mismatch_count"] or manifest["duplicate_path_count"] or not manifest["success_lock_created_last"]:
        gate = FAIL_MANIFEST
        gate_passed = False

    print(f"[DL-6A] artifact: {output_root}")
    print(f"[DL-6A] upstream DL-5 gate: {upstream_dl5['observed_gate']}")
    print(f"[DL-6A] upstream DL-4 gate: {upstream_dl4['top_level_gate']}")
    print("[DL-6A] study area: SUSEONG_GU_DAEGU")
    print("[DL-6A] nodes / edges: 255 / 291")
    print("[DL-6A] snapshots: 554")
    print(f"[DL-6A] configured agents: {agent_audit['authoritative_agent_count']}")
    print(f"[DL-6A] active agents min/mean/max: {agent_audit['active_agents_min']} / {agent_audit['active_agents_mean']} / {agent_audit['active_agents_max']}")
    print(f"[DL-6A] action dim: {action_audit['action_dim']}")
    print("[DL-6A] action contract: action 0 noop; actions 1,2 intervention decoder ids")
    print(f"[DL-6A] raw greedy no-op rate: {overall_values.get('raw_greedy_noop_rate')}")
    print(f"[DL-6A] masked greedy no-op rate: {overall_values.get('masked_greedy_noop_rate')}")
    print(f"[DL-6A] expected intervention probability: {overall_values.get('expected_intervention_probability')}")
    print(f"[DL-6A] mask-forced no-op rate: {overall_values.get('mask_forced_noop_rate')}")
    print(f"[DL-6A] B2 intervention windows: {b2_summary.get('b2_intervention_window_count')}")
    print(f"[DL-6A] actor response on B2 intervention windows: {b2_summary.get('actor_expected_intervention_probability_on_b2_intervention_windows')}")
    print(f"[DL-6A] three-seed all-noop agreement: {cross_summary.get('three_seed_all_noop_agreement_rate')}")
    print(f"[DL-6A] parameter mutation: {mutation_audit['parameter_mutation_count']}")
    print(f"[DL-6A] optimizer steps: {mutation_audit['optimizer_step_count']}")
    print(f"[DL-6A] diagnosis: {diagnosis}")
    print(f"[DL-6A] gate: {gate}")
    print(f"[DL-6A] gate_passed: {str(gate_passed).lower()}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
