from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional


A_FAMILY_CONDITIONS = ["A", "A90", "A80", "A70"]

POLICY_SOURCE_MODE_MOCK_SMOKE = "mock_smoke"
POLICY_SOURCE_MODE_MAPPO_SMOKE = "mappo_smoke"
POLICY_SOURCE_MODE_MAPPO_ACTUAL = "mappo_actual"

ALLOWED_POLICY_SOURCE_MODES = [
    POLICY_SOURCE_MODE_MOCK_SMOKE,
    POLICY_SOURCE_MODE_MAPPO_SMOKE,
    POLICY_SOURCE_MODE_MAPPO_ACTUAL,
]

INTEGRATION_CONTRACT_VERSION = "a_family_rollout_policy_integration_v1"


class AFamilyPolicyIntegrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AFamilyPolicyIntegrationConfig:
    condition_id: str
    policy_source_mode: str
    checkpoint_path: str = ""
    checkpoint_validation_mode: str = ""
    causal_simulator: bool = False
    qwen_train: bool = False
    qwen_inference: bool = False
    qwen_trigger_rate: float = 0.0
    allow_mock: bool = False


@dataclass(frozen=True)
class AFamilyPolicyIntegrationPlan:
    integration_contract_version: str
    condition_id: str
    policy_source_mode: str
    checkpoint_required: bool
    checkpoint_validation_mode: str
    use_mappo_policy_action_source: bool
    allow_mock_action: bool
    expected_policy_source: str
    expected_source_mode: str
    actual_policy_claim_possible: bool
    causal_policy_claim_possible: bool
    qwen_train: bool
    qwen_inference: bool
    qwen_trigger_rate: float
    required_metadata_fields: List[str]
    notes: List[str]


def normalize_condition_id(condition_id: str) -> str:
    cid = str(condition_id).strip().upper()
    if cid not in A_FAMILY_CONDITIONS:
        raise AFamilyPolicyIntegrationError(
            f"unsupported A-family condition_id: {condition_id}. "
            f"allowed={A_FAMILY_CONDITIONS}"
        )
    return cid


def normalize_policy_source_mode(policy_source_mode: str) -> str:
    mode = str(policy_source_mode).strip().lower()
    if mode not in ALLOWED_POLICY_SOURCE_MODES:
        raise AFamilyPolicyIntegrationError(
            f"unsupported policy_source_mode: {policy_source_mode}. "
            f"allowed={ALLOWED_POLICY_SOURCE_MODES}"
        )
    return mode


def validate_qwen_disabled(
    *,
    qwen_train: bool,
    qwen_inference: bool,
    qwen_trigger_rate: float,
) -> None:
    if bool(qwen_train):
        raise AFamilyPolicyIntegrationError("A-family MAPPO path requires qwen_train=false")
    if bool(qwen_inference):
        raise AFamilyPolicyIntegrationError("A-family MAPPO path requires qwen_inference=false")
    if float(qwen_trigger_rate) != 0.0:
        raise AFamilyPolicyIntegrationError("A-family MAPPO path requires qwen_trigger_rate=0.0")


def source_mode_for_plan(
    *,
    condition_id: str,
    policy_source_mode: str,
    causal_simulator: bool,
) -> str:
    cid = normalize_condition_id(condition_id)
    mode = normalize_policy_source_mode(policy_source_mode)

    if mode == POLICY_SOURCE_MODE_MOCK_SMOKE:
        return f"noncausal_{cid}_mock_policy_smoke_v1"

    if mode == POLICY_SOURCE_MODE_MAPPO_SMOKE:
        return f"noncausal_{cid}_mappo_policy_smoke_v1"

    if causal_simulator:
        return f"causal_{cid}_mappo_policy_v1"

    return f"noncausal_{cid}_mappo_policy_actual_v1"


def checkpoint_validation_mode_for_policy_source_mode(policy_source_mode: str) -> str:
    mode = normalize_policy_source_mode(policy_source_mode)

    if mode == POLICY_SOURCE_MODE_MAPPO_ACTUAL:
        return "actual"

    if mode == POLICY_SOURCE_MODE_MAPPO_SMOKE:
        return "smoke"

    return ""


def required_policy_metadata_fields() -> List[str]:
    from policies.policy_source_metadata import policy_metadata_columns

    return policy_metadata_columns()


def build_a_family_policy_integration_plan(
    config: AFamilyPolicyIntegrationConfig,
) -> AFamilyPolicyIntegrationPlan:
    cid = normalize_condition_id(config.condition_id)
    mode = normalize_policy_source_mode(config.policy_source_mode)

    validate_qwen_disabled(
        qwen_train=bool(config.qwen_train),
        qwen_inference=bool(config.qwen_inference),
        qwen_trigger_rate=float(config.qwen_trigger_rate),
    )

    if mode == POLICY_SOURCE_MODE_MOCK_SMOKE and not bool(config.allow_mock):
        raise AFamilyPolicyIntegrationError(
            "mock_smoke requires allow_mock=true so that accidental mock usage is explicit"
        )

    checkpoint_required = mode in {
        POLICY_SOURCE_MODE_MAPPO_SMOKE,
        POLICY_SOURCE_MODE_MAPPO_ACTUAL,
    }

    if checkpoint_required and not str(config.checkpoint_path or "").strip():
        raise AFamilyPolicyIntegrationError(
            f"{mode} requires a non-empty checkpoint_path"
        )

    expected_validation_mode = checkpoint_validation_mode_for_policy_source_mode(mode)
    requested_validation_mode = str(config.checkpoint_validation_mode or expected_validation_mode).strip().lower()

    if expected_validation_mode and requested_validation_mode != expected_validation_mode:
        raise AFamilyPolicyIntegrationError(
            f"{mode} requires checkpoint_validation_mode={expected_validation_mode}, "
            f"got {requested_validation_mode}"
        )

    if mode == POLICY_SOURCE_MODE_MOCK_SMOKE and requested_validation_mode:
        raise AFamilyPolicyIntegrationError(
            "mock_smoke must not set checkpoint_validation_mode"
        )

    use_mappo = mode in {
        POLICY_SOURCE_MODE_MAPPO_SMOKE,
        POLICY_SOURCE_MODE_MAPPO_ACTUAL,
    }

    expected_policy_source = "mappo_policy" if use_mappo else "mock_policy"
    expected_source_mode = source_mode_for_plan(
        condition_id=cid,
        policy_source_mode=mode,
        causal_simulator=bool(config.causal_simulator),
    )

    actual_policy_claim_possible = (
        mode == POLICY_SOURCE_MODE_MAPPO_ACTUAL
    )

    causal_policy_claim_possible = (
        mode == POLICY_SOURCE_MODE_MAPPO_ACTUAL
        and bool(config.causal_simulator)
    )

    notes: List[str] = []

    if mode == POLICY_SOURCE_MODE_MOCK_SMOKE:
        notes.append("mock_smoke is for development only and cannot support performance claims")

    if mode == POLICY_SOURCE_MODE_MAPPO_SMOKE:
        notes.append("mappo_smoke verifies strict neural action path but cannot support actual claims")

    if mode == POLICY_SOURCE_MODE_MAPPO_ACTUAL:
        notes.append("mappo_actual requires H200-trained checkpoint and validator actual mode pass")

    if not bool(config.causal_simulator):
        notes.append("noncausal replay output cannot support causal performance claims")

    return AFamilyPolicyIntegrationPlan(
        integration_contract_version=INTEGRATION_CONTRACT_VERSION,
        condition_id=cid,
        policy_source_mode=mode,
        checkpoint_required=checkpoint_required,
        checkpoint_validation_mode=requested_validation_mode,
        use_mappo_policy_action_source=use_mappo,
        allow_mock_action=bool(config.allow_mock),
        expected_policy_source=expected_policy_source,
        expected_source_mode=expected_source_mode,
        actual_policy_claim_possible=bool(actual_policy_claim_possible),
        causal_policy_claim_possible=bool(causal_policy_claim_possible),
        qwen_train=False,
        qwen_inference=False,
        qwen_trigger_rate=0.0,
        required_metadata_fields=required_policy_metadata_fields(),
        notes=notes,
    )


def plan_to_dict(plan: AFamilyPolicyIntegrationPlan) -> Dict[str, Any]:
    return asdict(plan)


def build_plan_from_dict(payload: Mapping[str, Any]) -> Dict[str, Any]:
    config = AFamilyPolicyIntegrationConfig(
        condition_id=str(payload.get("condition_id", "")),
        policy_source_mode=str(payload.get("policy_source_mode", "")),
        checkpoint_path=str(payload.get("checkpoint_path", "")),
        checkpoint_validation_mode=str(payload.get("checkpoint_validation_mode", "")),
        causal_simulator=bool(payload.get("causal_simulator", False)),
        qwen_train=bool(payload.get("qwen_train", False)),
        qwen_inference=bool(payload.get("qwen_inference", False)),
        qwen_trigger_rate=float(payload.get("qwen_trigger_rate", 0.0)),
        allow_mock=bool(payload.get("allow_mock", False)),
    )
    return plan_to_dict(build_a_family_policy_integration_plan(config))


def build_matrix_preview(
    *,
    checkpoint_path: str = "REQUIRED_CHECKPOINT_PATH",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for cid in A_FAMILY_CONDITIONS:
        for mode in ALLOWED_POLICY_SOURCE_MODES:
            payload: Dict[str, Any] = {
                "condition_id": cid,
                "policy_source_mode": mode,
                "checkpoint_path": checkpoint_path if mode != POLICY_SOURCE_MODE_MOCK_SMOKE else "",
                "checkpoint_validation_mode": checkpoint_validation_mode_for_policy_source_mode(mode),
                "causal_simulator": False,
                "allow_mock": mode == POLICY_SOURCE_MODE_MOCK_SMOKE,
            }

            try:
                rows.append(build_plan_from_dict(payload))
            except Exception as exc:
                rows.append(
                    {
                        "condition_id": cid,
                        "policy_source_mode": mode,
                        "valid": False,
                        "error": str(exc),
                    }
                )

    return rows
