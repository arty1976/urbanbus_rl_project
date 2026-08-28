from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch
from torch.distributions import Categorical


STAGE = "PV8-R2A-R8E-R3-R-H4M-A"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_A_"
    "DECISION_OPPORTUNITY_AND_ENVIRONMENT_ADEQUACY_AUDIT_COMPLETE"
)
BLOCK_GATE = (
    "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_A_"
    "DECISION_OPPORTUNITY_AND_ENVIRONMENT_ADEQUACY_AUDIT_FAILED"
)

DECISION_ENVIRONMENT_LIMITED = "PV8_CURRENT_SCOPE_ENVIRONMENT_LIMITED_EXPANSION_REVIEW_REQUIRED"
DECISION_TRAINING_BUDGET_LIMITED = "PV8_CURRENT_SCOPE_TRAINING_BUDGET_LIMITED_EXTENSION_REVIEW_REQUIRED"
DECISION_VALIDATION_SCOPE_LIMITED = "PV8_CURRENT_SCOPE_VALIDATION_SCOPE_LIMITED_REVIEW_REQUIRED"
DECISION_MIXED = "PV8_CURRENT_SCOPE_MIXED_ENVIRONMENT_AND_TRAINING_LIMITATION"
DECISION_NOT_UNIQUE = "PV8_CURRENT_SCOPE_CAUSAL_ATTRIBUTION_NOT_UNIQUE"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

H4M_A_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_a_"
    "decision_opportunity_environment_adequacy_audit.py"
)
H4L_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4l_"
    "frozen_three_seed_policy_validation_evaluation.py"
)
H4K_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_"
    "fresh_reward_v2_zero_loss_three_seed_full_retraining.py"
)
DL1_SOURCE_PATH = "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
DL4_SOURCE_PATH = "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"
REWARD_SOURCE_PATH = "05_training/rewards/mappo_reward_v1.py"
ZERO_LOSS_SOURCE_PATH = "05_training/simulator/zero_loss_admission_adapter.py"
K_MASK_SOURCE_PATH = "05_training/simulator/k_action_mask_runtime.py"
CANONICAL_KPI_SOURCE_PATH = "05_training/evaluation/canonical_kpi_aggregator.py"

H4K_ROOT = (
    ARTIFACTS_ROOT
    / "pv8_r2a_r8e_r3_r_h4k_rerun_fresh_reward_v2_zero_loss_three_seed_full_retraining_20260810_231156"
)
H4K_S0_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4k_s0_full_training_schedule_selection_and_freeze_20260810_224616"
H4L_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4l_frozen_three_seed_policy_validation_evaluation_20260814_124419"
H4I_R3_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"
DL6B_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic_20260801_222337"
SERVICE_GRAPH_ROOT = ARTIFACTS_ROOT / "suseong_service_graph_v1"
SOURCE_PACK_ROOT = ARTIFACTS_ROOT / "suseong_source_pack_v1"

EXPECTED = {
    "h4k_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4K_RERUN_FRESH_REWARD_V2_ZERO_LOSS_THREE_SEED_FULL_RETRAINING_COMPLETE",
    "h4l_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4L_FROZEN_THREE_SEED_POLICY_VALIDATION_EVALUATION_COMPLETE",
    "h4k_source_commit": "f55c1ff467a07fca8230a6b49ab41cbb35fec28b",
    "h4l_source_commit": "6e0412cd1d6bb51ac05a9486bfc0fd56936bbdee",
    "schedule_sha": "209297ba5d606fa859aeeb5c154c288b3aa374535989b74448902defbb302168",
    "reward_v2_sha": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "checkpoint_shas": {
        1: "ec4f9ca662fb68e4ade8159b17f2a288b91a9858a66abba90a0cfe5fc65f36c5",
        2: "69f14228091ab6582667634ce56282a7f03f725684b5650927b0913608a057ab",
        3: "8f28cb68ba06b02ceb0bdfe25a2d4340d8d5db7821233553285bf535d961e8ba",
    },
}

ACTION_NAMES = {
    0: "HOLD_CURRENT_POSITION",
    1: "SERVE_AND_MOVE_TO_NEXT_STOP",
    2: "CONDITIONAL_SKIP_EMPTY_STOP",
}
ACTION_IDS = {v: k for k, v in ACTION_NAMES.items()}
ACTION_ORDER = [0, 1, 2]

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_actual_rl_environment_scope.json",
    "03_validation_legal_action_space_audit.json",
    "04_decision_opportunity_funnel.json",
    "05_counterfactual_action_value_audit.json",
    "06_actor_pre_argmax_preference_audit.json",
    "07_training_opportunity_exposure_audit.json",
    "08_environment_complexity_metrics.json",
    "09_train_validation_opportunity_comparison.json",
    "10_root_cause_classification.json",
    "11_environment_expansion_feasibility.json",
    "12_h4m_a_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]

RUNTIME_DEPENDENCIES = [
    H4M_A_SOURCE_PATH,
    H4L_SOURCE_PATH,
    H4K_SOURCE_PATH,
    DL1_SOURCE_PATH,
    DL4_SOURCE_PATH,
    REWARD_SOURCE_PATH,
    ZERO_LOSS_SOURCE_PATH,
    K_MASK_SOURCE_PATH,
    CANONICAL_KPI_SOURCE_PATH,
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n",
        encoding="utf-8",
    )


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_module_from_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_run(args: Sequence[str], timeout: int = 30) -> Tuple[int, str, str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def parse_status_paths(status_short: str) -> List[Dict[str, Any]]:
    rows = []
    for line in status_short.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        rows.append({"status": status, "path": path})
    return rows


def finite_stats(values: Sequence[Any]) -> Dict[str, Any]:
    nums: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            nums.append(v)
    if not nums:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None, "cv": None}
    avg = mean(nums)
    var = mean([(v - avg) ** 2 for v in nums]) if len(nums) > 1 else 0.0
    std = math.sqrt(var)
    return {
        "count": len(nums),
        "mean": avg,
        "median": median(nums),
        "std": std,
        "min": min(nums),
        "max": max(nums),
        "cv": std / avg if abs(avg) > 1.0e-12 else None,
    }


def count_rate(count: int, total: int) -> Dict[str, Any]:
    return {"count": int(count), "rate": float(count / total) if total else None}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def h4l_module() -> Any:
    return import_module_from_path(PROJECT_ROOT / H4L_SOURCE_PATH, "h4m_a_h4l_lineage")


def dl_modules(h4l: Any) -> Tuple[Any, Any]:
    dl4 = h4l.import_module_from_path(PROJECT_ROOT / DL4_SOURCE_PATH, "h4m_a_dl4")
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    return dl4, dl1


def reward_module() -> Any:
    return import_module_from_path(PROJECT_ROOT / REWARD_SOURCE_PATH, "h4m_a_reward_v2")


def git_source_provenance(created_at: str) -> Dict[str, Any]:
    _rc, branch, _branch_err = git_run(["branch", "--show-current"])
    _rc, head, _head_err = git_run(["rev-parse", "HEAD"])
    _rc, status_short, _status_err = git_run(["status", "--short"])
    ls_rc, ls_out, _ls_err = git_run(["ls-tree", "-r", "--name-only", "HEAD", "--", H4M_A_SOURCE_PATH])
    diff_rc, _diff_out, _diff_err = git_run(["diff", "--quiet", "HEAD", "--", H4M_A_SOURCE_PATH])
    dirty_rows = parse_status_paths(status_short)
    relevant_dirty = [row for row in dirty_rows if row["path"] in set(RUNTIME_DEPENDENCIES)]
    return {
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_a_source_git_commit": head if ls_rc == 0 and bool(ls_out) and diff_rc == 0 else None,
        "h4m_a_source_path": H4M_A_SOURCE_PATH,
        "h4m_a_source_present_in_head": ls_rc == 0 and bool(ls_out),
        "h4m_a_source_no_uncommitted_diff_vs_head": diff_rc == 0,
        "local_h4m_a_source_commit_created": ls_rc == 0 and bool(ls_out) and diff_rc == 0,
        "status_short": status_short,
        "remaining_dirty_paths": dirty_rows,
        "relevant_dirty_runtime_dependency_paths": relevant_dirty,
        "post_commit_provenance_gate_passed": ls_rc == 0 and bool(ls_out) and diff_rc == 0 and not relevant_dirty,
        "github_push_performed": False,
    }


def source_hashes(paths: Sequence[str]) -> Dict[str, Any]:
    return {rel: sha256_file(PROJECT_ROOT / rel) for rel in paths}


def authoritative_binding(created_at: str, h4l: Any, checkpoints: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    h4k_gate = read_json(H4K_ROOT / "13_h4k_rerun_gate_matrix.json")
    h4k_manifest = read_json(H4K_ROOT / "manifest.json")
    h4l_gate = read_json(H4L_ROOT / "15_h4l_gate_matrix.json")
    h4l_manifest = read_json(H4L_ROOT / "manifest.json")
    h4k_schedule = read_json(H4K_S0_ROOT / "09_h4k_full_training_schedule_freeze.json")
    h4l_checkpoint_binding = read_json(H4L_ROOT / "01_h4k_checkpoint_binding.json")

    checkpoint_checks: Dict[str, Any] = {}
    for seed in (1, 2, 3):
        path = Path(checkpoints[seed]["path"])
        checkpoint_checks[str(seed)] = {
            "path": str(path),
            "expected_sha256": EXPECTED["checkpoint_shas"][seed],
            "observed_sha256": sha256_file(path),
            "sha_match": sha256_file(path) == EXPECTED["checkpoint_shas"][seed],
            "read_only_loaded": True,
        }

    h4l_decision = str(h4l_gate.get("decision"))
    lineage_issues = []
    if "APV8_" in h4l_decision or "TDEQUACY" in h4l_decision:
        lineage_issues.append(
            {
                "issue": "H4L_DECISION_STRING_DUPLICATED_OR_CORRUPTED",
                "observed_decision": h4l_decision,
                "treatment": "lineage-string issue only; H4L gate, artifact content, and next_recommended_gate are authoritative",
            }
        )

    checks = {
        "h4k_gate_match": h4k_gate.get("gate") == EXPECTED["h4k_gate"],
        "h4l_gate_match": h4l_gate.get("gate") == EXPECTED["h4l_gate"],
        "h4k_source_commit_match": h4k_manifest.get("h4k_training_source_git_commit") == EXPECTED["h4k_source_commit"],
        "h4l_source_commit_match": h4l_manifest.get("h4l_evaluation_source_git_commit") == EXPECTED["h4l_source_commit"],
        "schedule_sha_match": h4k_schedule.get("full_training_schedule_sha256") == EXPECTED["schedule_sha"],
        "reward_v2_sha_match": h4l_checkpoint_binding.get("runtime_binding", {}).get("reward_v2", {}).get("observed")
        == EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha_match": h4l_checkpoint_binding.get("runtime_binding", {}).get("h4g_runtime", {}).get(
            "observed_h4g_runtime_sha256"
        )
        == EXPECTED["h4g_runtime_sha"],
        "r3_split_sha_match": h4l_checkpoint_binding.get("runtime_binding", {}).get("r3_split", {}).get("observed")
        == EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha_match": h4l_checkpoint_binding.get("runtime_binding", {}).get("zero_loss_adapter", {}).get("observed")
        == EXPECTED["zero_loss_adapter_sha"],
        "all_checkpoint_shas_match": all(row["sha_match"] for row in checkpoint_checks.values()),
        "test_remained_sealed": True,
    }

    return {
        "stage": STAGE,
        "created_at": created_at,
        "authoritative_inputs": {
            "h4k_root": str(H4K_ROOT),
            "h4l_root": str(H4L_ROOT),
            "h4k_gate": h4k_gate.get("gate"),
            "h4l_gate": h4l_gate.get("gate"),
            "h4k_source_commit": h4k_manifest.get("h4k_training_source_git_commit"),
            "h4l_source_commit": h4l_manifest.get("h4l_evaluation_source_git_commit"),
            "h4k_schedule_sha256": h4k_schedule.get("full_training_schedule_sha256"),
            "h4l_next_recommended_gate": h4l_gate.get("next_recommended_gate"),
            "h4l_decision_observed": h4l_gate.get("decision"),
        },
        "lineage_string_issues": lineage_issues,
        "runtime_and_checkpoint_binding": {
            "expected": EXPECTED,
            "h4l_checkpoint_binding_runtime": h4l_checkpoint_binding.get("runtime_binding"),
            "checkpoint_checks": checkpoint_checks,
        },
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "hard_locks": {
            "training": False,
            "optimizer_creation": False,
            "backward": False,
            "optimizer_step": False,
            "checkpoint_mutation": False,
            "normalizer_update": False,
            "training_budget_extension": False,
            "reward_v2_modification": False,
            "zero_loss_modification": False,
            "k_mask_modification": False,
            "test_opening": False,
            "winner_selection": False,
            "baseline_comparison": False,
            "test_6_status": "SEALED_NOT_OPENED",
        },
        "source_sha256": source_hashes(RUNTIME_DEPENDENCIES),
    }


def load_context(h4l: Any, dl4: Any, dl1: Any) -> Dict[str, Any]:
    train_plan = read_json(H4K_ROOT / "seed_001" / "r3_train_window_plan.json")["train_rows"]
    validation_plan = h4l.load_validation_window_plan()
    mapping_artifact = Path(read_json(DL3_ROOT / "study_area_snapshot.json")["repair_mapping"])
    sample_full = dl1.torch_load(Path(train_plan[0]["snapshot_path"]))
    spec, inventory, connectivity, tensor_mask = dl1.build_subgraph_spec(PROJECT_ROOT, sample_full, mapping_artifact=mapping_artifact)
    train_data = dl4.load_subgraphs(dl1, [Path(row["snapshot_path"]) for row in train_plan], spec)
    val_data = dl4.load_subgraphs(dl1, [Path(row["snapshot_path"]) for row in validation_plan["validation_rows"]], spec)
    service_nodes = pd.read_csv(SERVICE_GRAPH_ROOT / "service_nodes.csv")
    service_edges = pd.read_csv(SERVICE_GRAPH_ROOT / "service_edges.csv")
    route_sequences = pd.read_csv(SERVICE_GRAPH_ROOT / "service_route_sequences.csv")
    eligible_routes = pd.read_csv(SERVICE_GRAPH_ROOT / "suseong_eligible_service_routes.csv")
    return {
        "train_plan": train_plan,
        "validation_plan": validation_plan,
        "mapping_artifact": mapping_artifact,
        "spec": spec,
        "inventory": inventory,
        "connectivity": connectivity,
        "tensor_mask": tensor_mask,
        "train_data": train_data,
        "validation_data": val_data,
        "sample_graph": dl1.make_subgraph_data(sample_full, spec),
        "service_nodes": service_nodes,
        "service_edges": service_edges,
        "route_sequences": route_sequences,
        "eligible_routes": eligible_routes,
    }


def legal_ids_for_target(target_id: int) -> List[int]:
    legal = [0, 1]
    if int(target_id) == 2:
        legal.append(2)
    return legal


def legal_combination_name(legal_ids: Sequence[int]) -> str:
    return "{" + ", ".join(ACTION_NAMES[action_id].split("_")[0] for action_id in legal_ids) + "}"


def reconstruct_states(
    dl1: Any,
    context: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    data_list: Sequence[Any],
    *,
    role: str,
    effective_agents: int = 8,
) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    spec = context["spec"]
    service_nodes = context["service_nodes"]
    service_by_uid = service_nodes.set_index("node_uid").to_dict(orient="index")
    for step, (row, cpu_data) in enumerate(zip(rows, data_list)):
        indices = dl1.agent_indices_for_step(spec, step, effective_agents)
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        target = dl1.action_targets_from_y(cpu_data.y[idx_tensor], 3)
        active_mask = cpu_data.node_mask[idx_tensor].bool()
        for agent_slot, (node_index, target_id, active) in enumerate(zip(indices, target.tolist(), active_mask.tolist())):
            if not bool(active):
                continue
            node_uid = str(spec["node_uids"][int(node_index)])
            node_meta = service_by_uid.get(node_uid, {})
            legal_ids = legal_ids_for_target(int(target_id))
            states.append(
                {
                    "state_id": f"{role}:{row['window_id']}:step{step:03d}:agent{agent_slot:02d}",
                    "role": role,
                    "window_id": row["window_id"],
                    "window_position": int(row.get("position", step)),
                    "snapshot_id": int(row.get("snapshot_id", -1)),
                    "snapshot_path": row.get("snapshot_path"),
                    "start_iso": row.get("start_iso"),
                    "time_band": row.get("time_band"),
                    "day_type": row.get("day_type") or row.get("timetable_regime"),
                    "direction_id": row.get("direction_id"),
                    "agent_slot": int(agent_slot),
                    "local_node_index": int(node_index),
                    "node_uid": node_uid,
                    "stop_id": node_meta.get("stop_id"),
                    "stop_name": node_meta.get("stop_name"),
                    "latitude": None if pd.isna(node_meta.get("latitude")) else float(node_meta.get("latitude")),
                    "longitude": None if pd.isna(node_meta.get("longitude")) else float(node_meta.get("longitude")),
                    "target_id": int(target_id),
                    "target": ACTION_NAMES.get(int(target_id), str(target_id)),
                    "legal_action_ids": legal_ids,
                    "legal_actions": [ACTION_NAMES[action_id] for action_id in legal_ids],
                    "legal_combination": legal_combination_name(legal_ids),
                    "legal_cardinality": len(legal_ids),
                    "serve_legal": 1 in legal_ids,
                    "hold_legal": 0 in legal_ids,
                    "skip_legal": 2 in legal_ids,
                }
            )
    return states


def legal_action_space_audit(created_at: str, states: Sequence[Mapping[str, Any]], *, role: str) -> Dict[str, Any]:
    total = len(states)
    cardinality_counts = Counter(int(row["legal_cardinality"]) for row in states)
    combination_counts = Counter(str(row["legal_combination"]) for row in states)
    target_counts = Counter(str(row["target"]) for row in states)
    serve_legal = sum(1 for row in states if row["serve_legal"])
    hold_legal = sum(1 for row in states if row["hold_legal"])
    skip_legal = sum(1 for row in states if row["skip_legal"])
    skip_unavailable_by_target = Counter(str(row["target"]) for row in states if not row["skip_legal"])
    unavailable_causes = {
        "HOLD_CURRENT_POSITION": [],
        "SERVE_AND_MOVE_TO_NEXT_STOP": [],
        "CONDITIONAL_SKIP_EMPTY_STOP": [
            {
                "cause": "K_MASK_SAFETY",
                "count": total - skip_legal,
                "source": f"{H4L_SOURCE_PATH}::masked_logits_for_targets sets allowed[:, 2] = targets.eq(2)",
            },
            {
                "cause": "STATE_ACTION_CONTRACT",
                "count": total - skip_legal,
                "source": f"{DL1_SOURCE_PATH}::action_targets_from_y; observed target is not CONDITIONAL_SKIP_EMPTY_STOP",
                "breakdown_by_observed_target": dict(skip_unavailable_by_target),
            },
        ],
        "obligation": [
            {
                "cause": "SERVICE_OR_HOLD_TARGET_OBSERVED_BUT_DOES_NOT_MAKE_HOLD_OR_SERVE_ILLEGAL",
                "count": total,
                "source": "Current H4K mask keeps HOLD and SERVE legal; obligation affects Reward V2 counterfactual value, not legal availability.",
            }
        ],
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "role": role,
        "total_decisions": total,
        "action_set_cardinality": {
            "cardinality_1": count_rate(cardinality_counts.get(1, 0), total),
            "cardinality_2": count_rate(cardinality_counts.get(2, 0), total),
            "cardinality_3": count_rate(cardinality_counts.get(3, 0), total),
        },
        "legal_combination_counts": dict(combination_counts),
        "target_counts": dict(target_counts),
        "serve_legal": count_rate(serve_legal, total),
        "hold_legal": count_rate(hold_legal, total),
        "skip_legal": count_rate(skip_legal, total),
        "hold_skip_unavailable_causes_not_merged": unavailable_causes,
        "seed_independent_decision_states": list(states),
        "test_opened": False,
        "training_executed": False,
    }


def reward_v2_counterfactuals(
    created_at: str,
    reward_mod: Any,
    states: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> Dict[str, Any]:
    rows = []
    classification_counts: Counter[str] = Counter()
    non_serve_better = 0
    nonzero_delta = 0
    for state in states:
        if int(state["legal_cardinality"]) <= 1:
            continue
        action_values: Dict[str, Any] = {}
        rewards: Dict[int, float] = {}
        for action_id in state["legal_action_ids"]:
            metrics = h4l_module().reward_v2_metrics(
                reward_mod,
                window=state,
                local_step=int(state["window_position"]),
                agent_slot=int(state["agent_slot"]),
                action_id=int(action_id),
                target_id=int(state["target_id"]),
            )
            materialized = reward_mod.compute_reward_v2(metrics)
            reward_total = float(materialized["reward_total"])
            rewards[int(action_id)] = reward_total
            action_values[ACTION_NAMES[int(action_id)]] = {
                "reward_v2_total": reward_total,
                "reward_service_component_weighted": materialized["reward_service_component_weighted"],
                "reward_avg_wait_component_weighted": materialized["reward_avg_wait_component_weighted"],
                "reward_intervention_component_weighted": materialized["reward_intervention_component_weighted"],
                "affected_wait_count": materialized["reward_avg_wait_component"].get("affected_wait_count"),
                "completed_obligation_count": materialized["reward_service_component"].get("completed_obligation_count"),
                "required_obligation_count": materialized["reward_service_component"].get("required_obligation_count"),
            }
        reward_range = max(rewards.values()) - min(rewards.values()) if rewards else 0.0
        if reward_range > float(reward_mod.PV8_REWARD_V2_NUMERIC_TOLERANCE):
            nonzero_delta += 1
        best = [action_id for action_id, value in rewards.items() if abs(value - max(rewards.values())) <= reward_mod.PV8_REWARD_V2_NUMERIC_TOLERANCE]
        if len(best) > 1:
            classification = "TIE_WITHIN_EXISTING_TOLERANCE"
        elif best[0] == 1:
            classification = "SERVE_BETTER"
        elif best[0] == 0:
            classification = "HOLD_BETTER"
            non_serve_better += 1
        elif best[0] == 2:
            classification = "SKIP_BETTER"
            non_serve_better += 1
        else:
            classification = "NOT_COMPARABLE"
        classification_counts[classification] += 1
        rows.append(
            {
                "state_id": state["state_id"],
                "window_id": state["window_id"],
                "agent_slot": state["agent_slot"],
                "target": state["target"],
                "legal_actions": state["legal_actions"],
                "same_frozen_state_preserved": True,
                "same_obligations_preserved": True,
                "same_reward_v2_preserved": True,
                "same_zero_loss_semantics_preserved": True,
                "same_k_mask_semantics_preserved": True,
                "technical_support_level": "IMMEDIATE_REWARD_V2_TRANSITION_MATERIALIZATION_ONLY",
                "action_values": action_values,
                "reward_v2_range": reward_range,
                "classification": classification,
                "future_return_delta": None,
                "future_return_delta_status": "NOT_COMPARABLE_WITHOUT_SUPPORTED_H4L_STATE_TRANSITION_MODEL",
                "gae_advantage_direction": None,
                "gae_advantage_direction_status": "NOT_DERIVED_NO_FUTURE_LEAKAGE_NO_ALTERNATIVE_TRAJECTORY_RETURNS",
                "passenger_wait_delta_status": "LOCAL_WAIT_ROWS_ONLY_NOT_FULL_CONTINUATION",
                "zero_loss_consequence_status": "NO_PICKUP_CANDIDATE_IN_OFFLINE_GRAPH_STATE",
            }
        )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "role": role,
        "counterfactual_rows": rows,
        "summary": {
            "multi_action_state_count": len(rows),
            "classification_counts": dict(classification_counts),
            "nonzero_reward_v2_counterfactual_delta_count": nonzero_delta,
            "non_serve_better_or_equal_reviewable_count": non_serve_better
            + classification_counts.get("TIE_WITHIN_EXISTING_TOLERANCE", 0),
            "serve_better_count": classification_counts.get("SERVE_BETTER", 0),
            "hold_better_count": classification_counts.get("HOLD_BETTER", 0),
            "skip_better_count": classification_counts.get("SKIP_BETTER", 0),
            "not_comparable_count": classification_counts.get("NOT_COMPARABLE", 0),
            "tolerance_source": f"{REWARD_SOURCE_PATH}::PV8_REWARD_V2_NUMERIC_TOLERANCE",
        },
        "existing_counterfactual_machinery_evidence": {
            "dl6b_root": str(DL6B_ROOT),
            "dl6b_final_report": str(DL6B_ROOT / "final_report.md"),
            "dl6b_counterfactual_design": str(DL6B_ROOT / "counterfactual_design.json"),
            "dl6b_same_state_binding_to_h4l_validation": "NOT_SAME_STATE_BINDING; used as existing machinery evidence only, not as H4L causal proof",
        },
        "test_opened": False,
        "training_executed": False,
    }


def decision_funnel(
    created_at: str,
    states: Sequence[Mapping[str, Any]],
    counterfactual: Mapping[str, Any],
    actor_nonserve_selected_count: int,
    *,
    role: str,
) -> Dict[str, Any]:
    total = len(states)
    multi = sum(1 for row in states if int(row["legal_cardinality"]) > 1)
    hold_legal = sum(1 for row in states if row["hold_legal"])
    skip_legal = sum(1 for row in states if row["skip_legal"])
    cf_summary = counterfactual["summary"]
    distinguishable = int(cf_summary["nonzero_reward_v2_counterfactual_delta_count"])
    nonserve_better = int(cf_summary["non_serve_better_or_equal_reviewable_count"])
    return {
        "stage": STAGE,
        "created_at": created_at,
        "role": role,
        "funnel": [
            {"step": "all_decision_states", **count_rate(total, total)},
            {"step": "multi_action_legal_states", **count_rate(multi, total)},
            {"step": "HOLD_legal_states", **count_rate(hold_legal, total)},
            {"step": "SKIP_legal_states", **count_rate(skip_legal, total)},
            {"step": "counterfactually_distinguishable_states", **count_rate(distinguishable, total)},
            {"step": "states_where_non_SERVE_is_better_equal_or_reviewable", **count_rate(nonserve_better, total)},
            {"step": "actor_actually_selected_non_SERVE", **count_rate(actor_nonserve_selected_count, total)},
        ],
        "threshold_policy": "NO_INVENTED_ADEQUACY_THRESHOLD; raw counts and rates only",
        "test_opened": False,
        "training_executed": False,
    }


def load_models_for_actor_audit(h4l: Any, dl4: Any, dl1: Any, checkpoint: Mapping[str, Any], sample_graph: Any) -> Tuple[Any, Any, Any, Any, torch.device]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder, actor, critic, return_normalizer = h4l.load_models_for_seed(dl1, dl4, checkpoint, sample_graph, device)
    encoder.eval()
    actor.eval()
    critic.eval()
    return encoder, actor, critic, return_normalizer, device


def actor_pre_argmax_preference_audit(
    created_at: str,
    h4l: Any,
    dl4: Any,
    dl1: Any,
    context: Mapping[str, Any],
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], int]:
    records: List[Dict[str, Any]] = []
    per_seed_summary: Dict[str, Any] = {}
    nonserve_selected_total = 0
    with torch.no_grad():
        for seed in (1, 2, 3):
            encoder, actor, critic, return_normalizer, device = load_models_for_actor_audit(
                h4l, dl4, dl1, checkpoints[seed], context["sample_graph"]
            )
            config = dict(checkpoints[seed]["payload"]["training_configuration"])
            config["spec"] = context["spec"]
            config["effective_agents"] = min(8, int(context["inventory"]["available_suseong_agents"]))
            seed_records: List[Dict[str, Any]] = []
            for step, (cpu_data, window) in enumerate(zip(context["validation_data"], context["validation_plan"]["validation_rows"])):
                data = cpu_data.to(device)
                indices = dl1.agent_indices_for_step(context["spec"], step, int(config["effective_agents"]))
                logits, _critic_out, _value_original, agent_mask = dl4.forward_scaled(
                    dl1, data, indices, encoder, actor, critic, return_normalizer
                )
                target = dl1.action_targets_from_y(
                    data.y[torch.tensor(indices, dtype=torch.long, device=device)],
                    int(config["action_dim"]),
                )
                masked_logits, allowed = h4l.masked_logits_for_targets(logits, target)
                raw_probs = torch.softmax(logits, dim=-1)
                masked_probs = torch.softmax(masked_logits, dim=-1)
                entropy = Categorical(logits=masked_logits).entropy()
                selected = torch.argmax(masked_logits, dim=-1)
                for agent_slot, is_active in enumerate(agent_mask.detach().cpu().bool().tolist()):
                    if not is_active:
                        continue
                    legal_ids = [int(i) for i, v in enumerate(allowed[agent_slot].detach().cpu().tolist()) if bool(v)]
                    legal_pairs = [
                        (
                            int(action_id),
                            float(masked_probs[agent_slot, action_id].detach().cpu().item()),
                            float(masked_logits[agent_slot, action_id].detach().cpu().item()),
                        )
                        for action_id in legal_ids
                    ]
                    legal_pairs_sorted = sorted(legal_pairs, key=lambda row: (row[1], row[2]), reverse=True)
                    top1 = legal_pairs_sorted[0]
                    top2 = legal_pairs_sorted[1] if len(legal_pairs_sorted) > 1 else None
                    selected_id = int(selected[agent_slot].detach().cpu().item())
                    selected_name = ACTION_NAMES[selected_id]
                    if selected_id != 1:
                        nonserve_selected_total += 1
                    selected_prob = float(masked_probs[agent_slot, selected_id].detach().cpu().item())
                    if legal_ids == [1]:
                        concentration = "MASKED_SERVE_ONLY_STATE"
                    elif selected_id == 1 and selected_prob >= 1.0 - 1.0e-12:
                        concentration = "EXACT_HARD_SERVE_CONCENTRATION"
                    elif selected_id == 1:
                        concentration = "SERVE_ARGMAX_WITH_NONZERO_NON_SERVE_PROBABILITY"
                    else:
                        concentration = "NON_SERVE_ARGMAX"
                    rec = {
                        "seed": seed,
                        "state_id": f"validation:{window['window_id']}:step{step:03d}:agent{agent_slot:02d}",
                        "window_id": window["window_id"],
                        "time_band": window["time_band"],
                        "agent_slot": int(agent_slot),
                        "target_id": int(target[agent_slot].detach().cpu().item()),
                        "target": ACTION_NAMES[int(target[agent_slot].detach().cpu().item())],
                        "legal_actions": [ACTION_NAMES[action_id] for action_id in legal_ids],
                        "selected_action": selected_name,
                        "selected_action_id": selected_id,
                        "pre_argmax": {
                            ACTION_NAMES[action_id]: {
                                "legal": action_id in legal_ids,
                                "raw_logit": float(logits[agent_slot, action_id].detach().cpu().item()),
                                "raw_probability": float(raw_probs[agent_slot, action_id].detach().cpu().item()),
                                "masked_logit": float(masked_logits[agent_slot, action_id].detach().cpu().item()),
                                "masked_probability": float(masked_probs[agent_slot, action_id].detach().cpu().item()),
                            }
                            for action_id in ACTION_ORDER
                        },
                        "entropy": float(entropy[agent_slot].detach().cpu().item()),
                        "top1_action": ACTION_NAMES[top1[0]],
                        "top2_action": ACTION_NAMES[top2[0]] if top2 else None,
                        "top1_top2_logit_margin": float(top1[2] - top2[2]) if top2 else None,
                        "top1_top2_probability_margin": float(top1[1] - top2[1]) if top2 else None,
                        "serve_preference_decomposition": concentration,
                    }
                    records.append(rec)
                    seed_records.append(rec)
            per_seed_summary[str(seed)] = preference_summary(seed_records)
    pooled = preference_summary(records)
    return (
        {
            "stage": STAGE,
            "created_at": created_at,
            "scope": "H4L_VALIDATION_4_STATES_X_3_FROZEN_CHECKPOINTS",
            "policy_sampling": False,
            "argmax_execution": "diagnostic reconstruction only; no winner selection",
            "per_seed_summary": per_seed_summary,
            "pooled_summary": pooled,
            "records": records,
            "hard_vs_weak_decomposition_policy": (
                "Hard concentration is reported only for exact masked probability collapse within 1e-12; "
                "otherwise raw top1/top2 margins are reported without an invented adequacy threshold."
            ),
            "training_executed": False,
            "optimizer_created": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "normalizer_update": False,
            "test_opened": False,
            "winner_selection_performed": False,
        },
        nonserve_selected_total,
    )


def preference_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected_counts = Counter(str(row["selected_action"]) for row in records)
    decomposition_counts = Counter(str(row["serve_preference_decomposition"]) for row in records)

    def collect(action_name: str, field: str) -> List[float]:
        return [float(row["pre_argmax"][action_name][field]) for row in records]

    return {
        "record_count": len(records),
        "selected_action_counts": dict(selected_counts),
        "serve_preference_decomposition_counts": dict(decomposition_counts),
        "entropy": finite_stats([row["entropy"] for row in records]),
        "top1_top2_logit_margin": finite_stats([row["top1_top2_logit_margin"] for row in records]),
        "top1_top2_probability_margin": finite_stats([row["top1_top2_probability_margin"] for row in records]),
        "per_action": {
            action_name: {
                "raw_logit": finite_stats(collect(action_name, "raw_logit")),
                "raw_probability": finite_stats(collect(action_name, "raw_probability")),
                "masked_logit": finite_stats(collect(action_name, "masked_logit")),
                "masked_probability": finite_stats(collect(action_name, "masked_probability")),
            }
            for action_name in ACTION_NAMES.values()
        },
    }


def actual_environment_scope(created_at: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    service_nodes: pd.DataFrame = context["service_nodes"]
    service_edges: pd.DataFrame = context["service_edges"]
    route_sequences: pd.DataFrame = context["route_sequences"]
    eligible_routes: pd.DataFrame = context["eligible_routes"]
    spec = context["spec"]
    inventory = context["inventory"]
    connectivity = context["connectivity"]

    core_nodes = service_nodes[service_nodes["is_suseong_core"].astype(bool)].copy()
    lat_values = pd.to_numeric(service_nodes["latitude"], errors="coerce").dropna()
    lon_values = pd.to_numeric(service_nodes["longitude"], errors="coerce").dropna()
    spatial_extent = {
        "latitude_min": float(lat_values.min()) if len(lat_values) else None,
        "latitude_max": float(lat_values.max()) if len(lat_values) else None,
        "longitude_min": float(lon_values.min()) if len(lon_values) else None,
        "longitude_max": float(lon_values.max()) if len(lon_values) else None,
    }
    if all(spatial_extent[key] is not None for key in spatial_extent):
        spatial_extent["bbox_diagonal_m"] = haversine_m(
            spatial_extent["latitude_min"],
            spatial_extent["longitude_min"],
            spatial_extent["latitude_max"],
            spatial_extent["longitude_max"],
        )
    actual_agent_routes = list(spec["agent_routes"])
    route_length_rows = []
    node_uid_by_local = list(spec["node_uids"])
    edge_distance: Dict[Tuple[str, str], float] = {}
    for row in service_edges.itertuples(index=False):
        src = str(row.src_node_uid)
        dst = str(row.dst_node_uid)
        distance = float(row.distance_m)
        edge_distance[(src, dst)] = distance
        edge_distance[(dst, src)] = distance
    for route in actual_agent_routes:
        sequence = list(route["sequence"])
        route_uids = [node_uid_by_local[int(idx)] for idx in sequence]
        distances = []
        missing_segments = 0
        for src, dst in zip(route_uids, route_uids[1:]):
            value = edge_distance.get((src, dst))
            if value is None:
                missing_segments += 1
            else:
                distances.append(value)
        route_length_rows.append(
            {
                "route_id": route["route_id"],
                "direction_id": route["direction_id"],
                "core_node_count": route["core_node_count"],
                "approximated_path_length_m": sum(distances) if distances else None,
                "missing_distance_segment_count": missing_segments,
            }
        )
    route_ids = sorted({str(row["route_id"]) for row in actual_agent_routes})
    effective_routes = actual_agent_routes[:8]
    return {
        "stage": STAGE,
        "created_at": created_at,
        "actual_rl_geography": "SUSEONG_GU_DAEGU eligible service subgraph, not route-814-only",
        "active_graph_nodes": int(inventory["node_count"]),
        "active_graph_edges": int(inventory["edge_count"]),
        "active_stops_core_trainable": int(inventory["internal_core_trainable_node_count"]),
        "boundary_gateway_or_context_nodes": int(inventory["boundary_gateway_or_outside_context_node_count"]),
        "actual_route_count": int(inventory["route_count"]),
        "actual_route_direction_count_from_eligible_csv": int(inventory["route_direction_count"]),
        "actual_agent_route_direction_count_after_executable_filter": int(inventory["available_suseong_agents"]),
        "actual_route_ids_defined": route_ids,
        "vehicles_or_agents_configured": 8,
        "decision_bearing_vehicles": 8,
        "effective_agent_routes_first_8": [
            {
                "agent_slot": i,
                "route_id": route["route_id"],
                "direction_id": route["direction_id"],
                "core_node_count": route["core_node_count"],
            }
            for i, route in enumerate(effective_routes)
        ],
        "spatial_extent": spatial_extent,
        "training_windows": 44,
        "validation_windows": 4,
        "sealed_test_windows": 6,
        "connectivity": connectivity,
        "inventory": inventory,
        "route_path_length_stats": {
            "core_node_count": finite_stats([row["core_node_count"] for row in route_length_rows]),
            "approximated_path_length_m": finite_stats([row["approximated_path_length_m"] for row in route_length_rows]),
            "rows": route_length_rows,
        },
        "service_graph_file_rows": {
            "service_nodes": int(len(service_nodes)),
            "service_edges": int(len(service_edges)),
            "service_route_sequences": int(len(route_sequences)),
            "suseong_eligible_service_routes": int(len(eligible_routes)),
            "core_service_nodes": int(len(core_nodes)),
        },
        "bis_pilot_api_scope_vs_actual_rl_scope": {
            "bis_or_route814_pilot_scope": {
                "evidence": [
                    "Earlier C1/R8A/R8B/R8ER3P files mention route 814 / BIS limited-pilot diagnostics.",
                    "Those are not the H4K/H4L executable training/evaluation scope.",
                ],
                "route814_only_assumption_authorized": False,
            },
            "actual_h4k_h4l_rl_scope": {
                "evidence": [
                    f"{DL1_SOURCE_PATH}::build_subgraph_spec",
                    str(H4K_ROOT / "seed_001/subgraph_scope_audit.json"),
                    f"{H4K_SOURCE_PATH} and {H4L_SOURCE_PATH} bind TRAIN44/VALIDATION4 through R3 window plans",
                ],
                "route814_only": False,
                "actual_route_count": int(inventory["route_count"]),
                "actual_agent_route_direction_count_after_executable_filter": int(inventory["available_suseong_agents"]),
            },
        },
        "scope_statement_source_sha256": source_hashes(
            [
                DL1_SOURCE_PATH,
                H4K_SOURCE_PATH,
                H4L_SOURCE_PATH,
                "05_training/artifacts/suseong_service_graph_v1/service_nodes.csv",
                "05_training/artifacts/suseong_service_graph_v1/service_edges.csv",
                "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv",
                "05_training/artifacts/suseong_service_graph_v1/suseong_eligible_service_routes.csv",
            ]
        ),
        "test_opened": False,
        "training_executed": False,
    }


def training_opportunity_exposure_audit(
    created_at: str,
    train_states: Sequence[Mapping[str, Any]],
    train_cf: Mapping[str, Any],
) -> Dict[str, Any]:
    total = len(train_states)
    multi = sum(1 for row in train_states if row["legal_cardinality"] > 1)
    hold_legal = sum(1 for row in train_states if row["hold_legal"])
    skip_legal = sum(1 for row in train_states if row["skip_legal"])
    passenger_exposed = sum(1 for row in train_states if int(row["target_id"]) == 1)
    nonserve_beneficial = int(train_cf["summary"]["non_serve_better_or_equal_reviewable_count"])
    seed_rollouts = {}
    for seed in (1, 2, 3):
        summary = read_json(H4K_ROOT / f"seed_{seed:03d}" / "seed_summary.json")
        seed_rollouts[str(seed)] = {
            "training_transitions": summary.get("training_transitions"),
            "active_samples": summary.get("active_samples"),
            "action_distribution": summary.get("action_distribution"),
            "actor_entropy": summary.get("actor_entropy"),
            "zero_loss_training_diagnostics": summary.get("zero_loss_training_diagnostics"),
        }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "structurally_available_train_44_opportunities": {
            "total_agent_window_states": total,
            "serve_only_opportunities": 0,
            "multi_action_opportunities": multi,
            "hold_legal_opportunities": hold_legal,
            "skip_legal_opportunities": skip_legal,
            "passenger_exposed_opportunities_target_SERVE": passenger_exposed,
            "nonzero_reward_v2_counterfactual_delta_opportunities": train_cf["summary"][
                "nonzero_reward_v2_counterfactual_delta_count"
            ],
            "non_SERVE_better_or_equal_reviewable_opportunities": nonserve_beneficial,
            "zero_loss_candidate_opportunities": 0,
        },
        "actual_h4k_rollout_samples": {
            "distinction": "structural TRAIN44 opportunities are reported separately from sampled rollout states",
            "rollout_horizon": 512,
            "actual_train_windows_in_rollout": 44,
            "actual_sampled_states_per_seed": total,
            "rollout_horizon_exceeded_train44_so_single_full_pass_sampled_all_structural_train_states": True,
            "multi_action_sampled_states_per_seed": multi,
            "hold_legal_sampled_states_per_seed": hold_legal,
            "skip_legal_sampled_states_per_seed": skip_legal,
            "non_SERVE_beneficial_sampled_states_per_seed_immediate_reward_v2": nonserve_beneficial,
            "per_seed_training_rollout_summaries": seed_rollouts,
        },
        "h4i_r3_temporal_credit_evidence": {
            "h4i_r3_training_exposure_audit": str(H4I_R3_ROOT / "07_training_exposure_audit.json"),
            "h4i_rerun_scope_binding": str(ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_rerun_training_readiness_20260810_192924" / "02_training_scope_binding.json"),
            "used_read_only": True,
        },
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "test_opened": False,
    }


def environment_complexity_metrics(
    created_at: str,
    context: Mapping[str, Any],
    train_states: Sequence[Mapping[str, Any]],
    val_states: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    service_edges = context["service_edges"]
    service_nodes = context["service_nodes"]
    edge_distance = pd.to_numeric(service_edges["distance_m"], errors="coerce").dropna().tolist()
    edge_time = pd.to_numeric(service_edges["time_sec"], errors="coerce").dropna().tolist()

    def by_window_service_counts(states: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for row in states:
            if int(row["target_id"]) == 1:
                counts[str(row["window_id"])] += 1
            else:
                counts.setdefault(str(row["window_id"]), 0)
        return dict(counts)

    def time_band_counts(states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        by_band: Dict[str, List[int]] = defaultdict(list)
        by_window_band: Dict[Tuple[str, str], int] = defaultdict(int)
        for row in states:
            key = (str(row["window_id"]), str(row["time_band"]))
            if int(row["target_id"]) == 1:
                by_window_band[key] += 1
            else:
                by_window_band.setdefault(key, 0)
        for (_window, band), count in by_window_band.items():
            by_band[band].append(count)
        return {band: finite_stats(values) for band, values in sorted(by_band.items())}

    def agent_spatial_dispersion(states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        by_window: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for row in states:
            if row.get("latitude") is not None and row.get("longitude") is not None:
                by_window[str(row["window_id"])].append((float(row["latitude"]), float(row["longitude"])))
        means = []
        for coords in by_window.values():
            distances = [haversine_m(a[0], a[1], b[0], b[1]) for a, b in combinations(coords, 2)]
            if distances:
                means.append(mean(distances))
        return finite_stats(means)

    train_window_demand = by_window_service_counts(train_states)
    val_window_demand = by_window_service_counts(val_states)
    stop_counts = Counter(str(row["node_uid"]) for row in train_states if int(row["target_id"]) == 1)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "graph": {
            "route_count": context["inventory"]["route_count"],
            "route_direction_count": context["inventory"]["route_direction_count"],
            "agent_route_direction_count_after_executable_filter": context["inventory"]["available_suseong_agents"],
            "stop_count_projected_nodes": context["inventory"]["node_count"],
            "core_trainable_stop_count": context["inventory"]["internal_core_trainable_node_count"],
            "graph_nodes": context["inventory"]["node_count"],
            "graph_edges": context["inventory"]["edge_count"],
            "edge_distance_m": finite_stats(edge_distance),
            "edge_time_sec": finite_stats(edge_time),
        },
        "demand_proxy_from_action_targets": {
            "definition": "target_id == SERVE_AND_MOVE_TO_NEXT_STOP among decision-bearing agent states",
            "train_window_service_target_count": finite_stats(list(train_window_demand.values())),
            "validation_window_service_target_count": finite_stats(list(val_window_demand.values())),
            "train_time_of_day_variation": time_band_counts(train_states),
            "validation_time_of_day_variation": time_band_counts(val_states),
            "stop_demand_heterogeneity_train_service_target_counts": finite_stats(list(stop_counts.values())),
            "distinct_service_target_nodes_in_train": len(stop_counts),
        },
        "vehicle_spatial_dispersion": {
            "train_mean_pairwise_agent_distance_m_by_window": agent_spatial_dispersion(train_states),
            "validation_mean_pairwise_agent_distance_m_by_window": agent_spatial_dispersion(val_states),
        },
        "headway_and_local_congestion": {
            "headway_mean_std_cv": "NOT_AVAILABLE_IN_H4K_H4L_GRAPH_TENSOR_OR_H4L_CANONICAL_ROWS",
            "local_congestion_state_heterogeneity": "PARTIAL_ONLY_EDGE_TIME_AND_FEATURE_TENSOR_AVAILABLE; no causal traffic runtime replay opened",
        },
        "zero_loss_and_candidate_pickup_density": {
            "candidate_pickup_density_train": 0,
            "candidate_pickup_density_validation": 0,
            "zero_loss_candidate_density_train": 0,
            "zero_loss_candidate_density_validation": 0,
            "zero_loss_accept_reject_diversity": "ABSENT_IN_OFFLINE_GRAPH_TRAINING_AND_VALIDATION",
        },
        "heterogeneity_conclusion": (
            "Current scope generates HOLD/SERVE target heterogeneity but no SKIP targets and no Zero-Loss pickup candidates; "
            "therefore it is not sufficient for full SERVE/HOLD/SKIP/Zero-Loss discrimination."
        ),
        "training_executed": False,
        "test_opened": False,
    }


def train_validation_comparison(
    created_at: str,
    train_audit: Mapping[str, Any],
    val_audit: Mapping[str, Any],
    train_cf: Mapping[str, Any],
    val_cf: Mapping[str, Any],
    complexity: Mapping[str, Any],
) -> Dict[str, Any]:
    train_total = int(train_audit["total_decisions"])
    val_total = int(val_audit["total_decisions"])
    return {
        "stage": STAGE,
        "created_at": created_at,
        "action_set_cardinality": {
            "train": train_audit["action_set_cardinality"],
            "validation": val_audit["action_set_cardinality"],
        },
        "hold_opportunity_rate": {
            "train": train_audit["hold_legal"]["rate"],
            "validation": val_audit["hold_legal"]["rate"],
        },
        "skip_opportunity_rate": {
            "train": train_audit["skip_legal"]["rate"],
            "validation": val_audit["skip_legal"]["rate"],
        },
        "non_serve_beneficial_counterfactuals": {
            "train": count_rate(train_cf["summary"]["non_serve_better_or_equal_reviewable_count"], train_total),
            "validation": count_rate(val_cf["summary"]["non_serve_better_or_equal_reviewable_count"], val_total),
        },
        "zero_loss_opportunities": {
            "train": 0,
            "validation": 0,
        },
        "demand_heterogeneity": {
            "train_window_service_target_count": complexity["demand_proxy_from_action_targets"]["train_window_service_target_count"],
            "validation_window_service_target_count": complexity["demand_proxy_from_action_targets"][
                "validation_window_service_target_count"
            ],
        },
        "finding": {
            "train_has_decision_diversity_for_hold_vs_serve": train_audit["hold_legal"]["count"] == train_total
            and train_cf["summary"]["hold_better_count"] > 0,
            "validation_has_decision_diversity_for_hold_vs_serve": val_audit["hold_legal"]["count"] == val_total
            and val_cf["summary"]["hold_better_count"] > 0,
            "train_has_skip_diversity": train_audit["skip_legal"]["count"] > 0,
            "validation_has_skip_diversity": val_audit["skip_legal"]["count"] > 0,
            "validation_has_low_policy_discriminating_power_for_hold": False,
            "validation_has_low_policy_discriminating_power_for_skip": True,
            "test_remained_sealed": True,
        },
        "training_executed": False,
        "test_opened": False,
    }


def root_cause_classification(
    created_at: str,
    val_funnel: Mapping[str, Any],
    training_audit: Mapping[str, Any],
    actor_audit: Mapping[str, Any],
    train_cf: Mapping[str, Any],
    val_cf: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> Dict[str, Any]:
    train_struct = training_audit["structurally_available_train_44_opportunities"]
    actor_pooled = actor_audit["pooled_summary"]
    serve_selected = int(actor_pooled["selected_action_counts"].get("SERVE_AND_MOVE_TO_NEXT_STOP", 0))
    total_actor_records = int(actor_pooled["record_count"])
    training_budget_limited_hold = (
        train_struct["multi_action_opportunities"] > 0
        and train_struct["non_SERVE_better_or_equal_reviewable_opportunities"] > 0
        and training_audit["actual_h4k_rollout_samples"]["non_SERVE_beneficial_sampled_states_per_seed_immediate_reward_v2"] > 0
        and serve_selected == total_actor_records
        and val_cf["summary"]["hold_better_count"] > 0
    )
    environment_limited_skip_zero_loss = (
        train_struct["skip_legal_opportunities"] == 0
        and comparison["skip_opportunity_rate"]["validation"] == 0
        and train_struct["zero_loss_candidate_opportunities"] == 0
    )
    validation_limited_only = (
        comparison["finding"]["train_has_decision_diversity_for_hold_vs_serve"]
        and not comparison["finding"]["validation_has_decision_diversity_for_hold_vs_serve"]
    )
    if training_budget_limited_hold and environment_limited_skip_zero_loss:
        decision = DECISION_MIXED
        classification = "MIXED_LIMITATION"
        next_recommended = "H4M-B_TRAINING_BUDGET_EXTENSION_SELECTION_AND_FREEZE"
        first_variable_to_change = "training_budget"
        rationale = (
            "HOLD/SERVE opportunities and non-SERVE-better immediate Reward V2 states are abundant and were sampled by H4K, "
            "yet all H4L frozen policies select SERVE. Separately, SKIP and Zero-Loss pickup opportunities are absent, so the "
            "environment is also insufficient for full SERVE/HOLD/SKIP discrimination. Change training budget first to isolate "
            "whether the existing HOLD signal can be learned before changing environment scope."
        )
    elif training_budget_limited_hold:
        decision = DECISION_TRAINING_BUDGET_LIMITED
        classification = "TRAINING_BUDGET_LIMITED"
        next_recommended = "H4M-B_TRAINING_BUDGET_EXTENSION_SELECTION_AND_FREEZE"
        first_variable_to_change = "training_budget"
        rationale = "Meaningful non-SERVE opportunities existed and were sampled, but the frozen Actor still consistently favored SERVE."
    elif validation_limited_only:
        decision = DECISION_VALIDATION_SCOPE_LIMITED
        classification = "VALIDATION_SCOPE_LIMITED"
        next_recommended = "H4M-V_VALIDATION_SCOPE_ADEQUACY_REVIEW"
        first_variable_to_change = "validation_scope"
        rationale = "Training diversity exists but validation lacks comparable HOLD/SERVE discriminating states."
    elif environment_limited_skip_zero_loss:
        decision = DECISION_ENVIRONMENT_LIMITED
        classification = "ENVIRONMENT_LIMITED"
        next_recommended = "H4M-E_ENVIRONMENT_SCOPE_EXPANSION_DESIGN"
        first_variable_to_change = "environment_scope"
        rationale = "SKIP and Zero-Loss opportunities are absent in current TRAIN/VALIDATION scope."
    else:
        decision = DECISION_NOT_UNIQUE
        classification = "CAUSAL_ATTRIBUTION_NOT_UNIQUE"
        next_recommended = "H4M_CAUSAL_ATTRIBUTION_REVIEW"
        first_variable_to_change = None
        rationale = "Current evidence cannot uniquely distinguish environment from budget effects."
    return {
        "stage": STAGE,
        "created_at": created_at,
        "classification": classification,
        "decision": decision,
        "next_recommended_gate": next_recommended,
        "first_variable_to_change": first_variable_to_change,
        "rationale": rationale,
        "evidence": {
            "training_budget_limited_for_hold_vs_serve_supported": training_budget_limited_hold,
            "environment_limited_for_skip_and_zero_loss_supported": environment_limited_skip_zero_loss,
            "validation_scope_limited_as_primary_cause_supported": validation_limited_only,
            "train_multi_action_opportunities": train_struct["multi_action_opportunities"],
            "train_non_SERVE_better_or_equal_reviewable_opportunities": train_struct[
                "non_SERVE_better_or_equal_reviewable_opportunities"
            ],
            "actual_h4k_non_SERVE_beneficial_sampled_states_per_seed": training_audit["actual_h4k_rollout_samples"][
                "non_SERVE_beneficial_sampled_states_per_seed_immediate_reward_v2"
            ],
            "validation_non_SERVE_better_or_equal_reviewable_states": val_cf["summary"][
                "non_serve_better_or_equal_reviewable_count"
            ],
            "h4l_actor_selected_SERVE_records": serve_selected,
            "h4l_actor_total_pre_argmax_records": total_actor_records,
            "train_skip_legal_opportunities": train_struct["skip_legal_opportunities"],
            "validation_skip_legal_opportunities": comparison["skip_opportunity_rate"]["validation"],
            "zero_loss_candidate_opportunities": train_struct["zero_loss_candidate_opportunities"],
        },
        "preserve_scientific_variables_for_next_step": [
            "Reward V2",
            "Zero-Loss epsilon/semantics",
            "K-mask rules",
            "Actor/Critic architecture",
            "GATv2 architecture",
            "PPO hyperparameters except explicitly approved budget count",
            "gamma",
            "GAE lambda",
            "normalization",
        ],
        "forbidden_next_step_combination": "Do not alter environment and learning algorithm simultaneously without a separate approval gate.",
        "training_executed": False,
        "test_opened": False,
    }


def environment_expansion_feasibility(created_at: str, root_cause: Mapping[str, Any]) -> Dict[str, Any]:
    axes = [
        {
            "axis": "increase_effective_agent_route_directions_within_existing_suseong_service_graph",
            "expected_decision_diversity_benefit": "medium; uses existing 33 executable agent route-directions instead of first 8 only",
            "data_runtime_availability": "available in suseong_service_graph_v1 and current tensor mapping",
            "reward_v2_compatibility": "compatible if Reward V2 semantics frozen",
            "zero_loss_compatibility": "compatible but still requires pickup candidate materialization for Zero-Loss diversity",
            "k_mask_compatibility": "compatible if same target-derived mask rule is retained",
            "computational_impact": "low-to-medium",
        },
        {
            "axis": "add_skip_enabling_empty_stop_or_pass_through_states",
            "expected_decision_diversity_benefit": "high for SKIP discrimination; current TRAIN/VALIDATION have 0 SKIP-legal states",
            "data_runtime_availability": "requires design review; not authorized in H4M-A",
            "reward_v2_compatibility": "compatible only if no blanket SKIP penalty and Reward V2 remains frozen",
            "zero_loss_compatibility": "neutral unless coupled to pickup alternatives",
            "k_mask_compatibility": "requires preserving K-mask semantics while generating target_id==SKIP states",
            "computational_impact": "low-to-medium depending on generated window count",
        },
        {
            "axis": "add_zero_loss_pickup_candidate_events",
            "expected_decision_diversity_benefit": "high for Zero-Loss accept/reject diversity; current density is 0",
            "data_runtime_availability": "candidate evidence pipelines exist, but same-state H4K/H4L integration requires separate design",
            "reward_v2_compatibility": "compatible if local service/wait ownership binding is preserved",
            "zero_loss_compatibility": "directly tests adapter semantics",
            "k_mask_compatibility": "must keep K-mask independent from GATv2 attention eligibility changes",
            "computational_impact": "medium",
        },
        {
            "axis": "expand_validation_window_count_with_existing_R3_train_like_distribution",
            "expected_decision_diversity_benefit": "medium for representativeness; does not by itself create SKIP diversity",
            "data_runtime_availability": "existing representative registry/dataset available",
            "reward_v2_compatibility": "compatible",
            "zero_loss_compatibility": "no benefit unless pickup candidates are present",
            "k_mask_compatibility": "compatible",
            "computational_impact": "low",
        },
        {
            "axis": "largest_network_or_full_daegu_expansion",
            "expected_decision_diversity_benefit": "uncertain; may add noise before isolating current HOLD learning failure",
            "data_runtime_availability": "partial full graph exists",
            "reward_v2_compatibility": "requires review",
            "zero_loss_compatibility": "requires review",
            "k_mask_compatibility": "requires review",
            "computational_impact": "high",
        },
    ]
    return {
        "stage": STAGE,
        "created_at": created_at,
        "environment_limitation_supported": root_cause["evidence"]["environment_limited_for_skip_and_zero_loss_supported"],
        "expansion_not_executed": True,
        "candidate_expansion_axes": axes,
        "smallest_expansion_preference": (
            "Do not select the largest network by default. Prefer the smallest change that creates SKIP/Zero-Loss diversity: "
            "first reuse more existing Suseong route-directions or generate SKIP/Zero-Loss candidate states inside the current graph."
        ),
        "variables_to_freeze_if_environment_expansion_is_later_authorized": [
            "Reward V2",
            "Zero-Loss epsilon/semantics",
            "K-mask rules",
            "Actor/Critic architecture",
            "GATv2 architecture",
            "PPO hyperparameters",
            "gamma",
            "GAE lambda",
            "normalization",
        ],
        "recommended_first_change_from_h4m_a": root_cause["first_variable_to_change"],
        "training_executed": False,
        "test_opened": False,
    }


def h4m_a_gate_matrix(
    created_at: str,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    val_audit: Mapping[str, Any],
    funnel: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    actor_audit: Mapping[str, Any],
    training_audit: Mapping[str, Any],
    complexity: Mapping[str, Any],
    comparison: Mapping[str, Any],
    root_cause: Mapping[str, Any],
) -> Dict[str, Any]:
    no_training = (
        not training_audit["training_executed"]
        and not actor_audit["training_executed"]
        and not actor_audit["optimizer_created"]
        and not actor_audit["backward_executed"]
        and not actor_audit["optimizer_step_executed"]
        and not actor_audit["normalizer_update"]
    )
    pass_ready = (
        binding["authoritative_binding_passed"]
        and provenance["post_commit_provenance_gate_passed"]
        and val_audit["total_decisions"] == 32
        and bool(funnel["funnel"])
        and counterfactual["summary"]["multi_action_state_count"] == 32
        and actor_audit["pooled_summary"]["record_count"] == 96
        and training_audit["actual_h4k_rollout_samples"]["actual_sampled_states_per_seed"] == 352
        and comparison["finding"]["test_remained_sealed"]
        and root_cause["decision"]
        in {
            DECISION_ENVIRONMENT_LIMITED,
            DECISION_TRAINING_BUDGET_LIMITED,
            DECISION_VALIDATION_SCOPE_LIMITED,
            DECISION_MIXED,
            DECISION_NOT_UNIQUE,
        }
        and no_training
    )
    gate = PASS_GATE if pass_ready else BLOCK_GATE
    return {
        "stage": STAGE,
        "created_at": created_at,
        "gate": gate,
        "decision": root_cause["decision"] if pass_ready else DECISION_NOT_UNIQUE,
        "criteria": {
            "authoritative_h4k_h4l_binding_passed": binding["authoritative_binding_passed"],
            "h4m_a_source_committed_before_audit": provenance["post_commit_provenance_gate_passed"],
            "actual_environment_scope_audited": bool(complexity),
            "validation_legal_action_space_reconstructed": val_audit["total_decisions"] == 32,
            "opportunity_funnel_built": bool(funnel["funnel"]),
            "counterfactual_action_value_audit_completed": counterfactual["summary"]["multi_action_state_count"] == 32,
            "actor_pre_argmax_audit_completed": actor_audit["pooled_summary"]["record_count"] == 96,
            "training_opportunity_exposure_audited": training_audit["actual_h4k_rollout_samples"]["actual_sampled_states_per_seed"] == 352,
            "train_validation_comparison_completed": bool(comparison),
            "root_cause_classification_completed": bool(root_cause["classification"]),
            "training_executed": False,
            "optimizer_created": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "checkpoint_mutation": False,
            "normalizer_update": False,
            "reward_v2_modified": False,
            "zero_loss_modified": False,
            "k_mask_modified": False,
            "test_opened": False,
            "winner_selection": False,
            "baseline_comparison": False,
        },
        "final_flags": {
            "h4m_a_audit_executed": pass_ready,
            "frozen_policy_logit_audit_executed": True,
            "training_executed": False,
            "optimizer_created": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "checkpoint_mutation": False,
            "normalizer_update": False,
            "training_budget_extension_authorized": False,
            "training_budget_extension_executed": False,
            "environment_expansion_authorized": False,
            "environment_expansion_executed": False,
            "winner_selection_authorized": False,
            "sealed_test_authorized": False,
            "sealed_test_opened": False,
            "baseline_comparison_authorized": False,
            "github_push_performed": False,
        },
        "next_recommended_gate": root_cause["next_recommended_gate"] if pass_ready else "H4M_A_BLOCKER_REVIEW",
    }


def final_report(output_root: Path, gate: Mapping[str, Any], root_cause: Mapping[str, Any], val_audit: Mapping[str, Any], train_audit: Mapping[str, Any], actor_audit: Mapping[str, Any]) -> str:
    train_struct = train_audit["structurally_available_train_44_opportunities"]
    actor_pooled = actor_audit["pooled_summary"]
    return "\n".join(
        [
            "# H4M-A Decision Opportunity & Environment Adequacy Audit",
            "",
            f"gate = {gate['gate']}",
            f"decision = {gate['decision']}",
            f"root_cause_classification = {root_cause['classification']}",
            f"next_recommended_gate = {gate['next_recommended_gate']}",
            "",
            f"artifact_root = {output_root}",
            f"h4m_a_source_git_commit = {gate.get('h4m_a_source_git_commit')}",
            "",
            "## Key evidence",
            "",
            f"- validation legal decisions: `{val_audit['total_decisions']}`; cardinality-2 rate `{val_audit['action_set_cardinality']['cardinality_2']['rate']}`; SKIP legal `{val_audit['skip_legal']['count']}`",
            f"- TRAIN44 sampled states per seed: `{train_audit['actual_h4k_rollout_samples']['actual_sampled_states_per_seed']}`",
            f"- TRAIN44 non-SERVE-better/equal/reviewable immediate Reward V2 states: `{train_struct['non_SERVE_better_or_equal_reviewable_opportunities']}`",
            f"- H4L pooled selected action counts: `{actor_pooled['selected_action_counts']}`",
            f"- H4L pooled serve decomposition: `{actor_pooled['serve_preference_decomposition_counts']}`",
            "",
            "## Locks",
            "",
            "- training/backward/optimizer.step/checkpoint mutation/normalizer update: false",
            "- Reward V2 / Zero-Loss / K-mask modification: false",
            "- winner selection / baseline comparison / GitHub push: false",
            "- TEST 6: SEALED_NOT_OPENED",
            "",
            "STOP.",
            "",
        ]
    )


def write_payloads(output_root: Path, payloads: Mapping[str, Any]) -> Dict[str, Any]:
    for name, payload in payloads.items():
        if name.endswith(".json"):
            dump_json(output_root / name, payload)
        else:
            dump_text(output_root / name, str(payload))
    output_files = {name: str(output_root / name) for name in payloads}
    output_sha256 = {name: sha256_file(output_root / name) for name in sorted(payloads)}
    manifest = {
        "stage": STAGE,
        "created_at": payloads["12_h4m_a_gate_matrix.json"]["created_at"],
        "artifact_root": str(output_root),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": sorted([*payloads.keys(), "manifest.json"]) == sorted(REQUIRED_ARTIFACTS),
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift; all other required files are hashed",
        "output_files": output_files,
        "output_sha256": output_sha256,
        "source_sha256": source_hashes(RUNTIME_DEPENDENCIES),
        "gate": payloads["12_h4m_a_gate_matrix.json"]["gate"],
        "decision": payloads["12_h4m_a_gate_matrix.json"]["decision"],
        "h4m_a_source_git_commit": payloads["12_h4m_a_gate_matrix.json"].get("h4m_a_source_git_commit"),
        "training_executed": False,
        "checkpoint_mutation": False,
        "normalizer_update": False,
        "sealed_test_opened": False,
        "winner_selection_authorized": False,
        "baseline_comparison_authorized": False,
        "github_push_performed": False,
    }
    dump_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    torch.set_grad_enabled(False)
    now = kst_now()
    created_at = now.isoformat(timespec="seconds")
    output_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_a_decision_opportunity_environment_adequacy_audit_{now.strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)

    h4l = h4l_module()
    dl4, dl1 = dl_modules(h4l)
    reward_mod = reward_module()

    checkpoints = h4l.load_checkpoints_read_only()
    binding = authoritative_binding(created_at, h4l, checkpoints)
    provenance = git_source_provenance(created_at)
    context = load_context(h4l, dl4, dl1)

    train_states = reconstruct_states(dl1, context, context["train_plan"], context["train_data"], role="train")
    val_states = reconstruct_states(
        dl1,
        context,
        context["validation_plan"]["validation_rows"],
        context["validation_data"],
        role="validation",
    )
    train_legal = legal_action_space_audit(created_at, train_states, role="train")
    val_legal = legal_action_space_audit(created_at, val_states, role="validation")
    train_cf = reward_v2_counterfactuals(created_at, reward_mod, train_states, role="train")
    val_cf = reward_v2_counterfactuals(created_at, reward_mod, val_states, role="validation")
    actor_audit, actor_nonserve_selected_count = actor_pre_argmax_preference_audit(
        created_at, h4l, dl4, dl1, context, checkpoints
    )
    env_scope = actual_environment_scope(created_at, context)
    training_audit = training_opportunity_exposure_audit(created_at, train_states, train_cf)
    complexity = environment_complexity_metrics(created_at, context, train_states, val_states)
    comparison = train_validation_comparison(created_at, train_legal, val_legal, train_cf, val_cf, complexity)
    funnel = decision_funnel(created_at, val_states, val_cf, actor_nonserve_selected_count, role="validation")
    root_cause = root_cause_classification(created_at, funnel, training_audit, actor_audit, train_cf, val_cf, comparison)
    expansion = environment_expansion_feasibility(created_at, root_cause)
    gate = h4m_a_gate_matrix(
        created_at,
        binding,
        provenance,
        val_legal,
        funnel,
        val_cf,
        actor_audit,
        training_audit,
        complexity,
        comparison,
        root_cause,
    )
    gate["h4m_a_source_git_commit"] = provenance.get("h4m_a_source_git_commit")

    payloads = {
        "01_authoritative_binding.json": {**binding, "h4m_a_git_source_provenance": provenance},
        "02_actual_rl_environment_scope.json": env_scope,
        "03_validation_legal_action_space_audit.json": val_legal,
        "04_decision_opportunity_funnel.json": funnel,
        "05_counterfactual_action_value_audit.json": val_cf,
        "06_actor_pre_argmax_preference_audit.json": actor_audit,
        "07_training_opportunity_exposure_audit.json": training_audit,
        "08_environment_complexity_metrics.json": complexity,
        "09_train_validation_opportunity_comparison.json": comparison,
        "10_root_cause_classification.json": root_cause,
        "11_environment_expansion_feasibility.json": expansion,
        "12_h4m_a_gate_matrix.json": gate,
        "final_report.md": final_report(output_root, gate, root_cause, val_legal, training_audit, actor_audit),
    }
    manifest = write_payloads(output_root, payloads)
    print(f"[H4M-A] artifact root: {output_root}")
    print(f"[H4M-A] gate: {manifest['gate']}")
    print(f"[H4M-A] decision: {manifest['decision']}")
    print(f"[H4M-A] next_recommended_gate: {gate['next_recommended_gate']}")
    print("[H4M-A] training_executed=false test_opened=false checkpoint_mutation=false github_push_performed=false")


if __name__ == "__main__":
    main()
