from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional


POLICY_METADATA_VERSION = "policy_source_metadata_v1"

EXPECTED_POLICY_SOURCE = "mappo_policy"
EXPECTED_ACTION_SOURCE_VERSION = "mappo_policy_action_source_v1"
EXPECTED_REWARD_VERSION = "mappo_reward_v1"
EXPECTED_ENERGY_PROXY_MODEL_VERSION = "daegu_energy_proxy_v1"

EXPECTED_K_DIST_KWH_PER_M = 0.0012
EXPECTED_K_ACC_KWH_PER_EVENT = 0.1800
EXPECTED_K_IDLE_KWH_PER_SEC = 0.0080

REQUIRED_POLICY_METADATA_FIELDS = [
    "policy_metadata_version",
    "condition_id",
    "source_mode",
    "policy_source",
    "policy_action_source_version",
    "policy_action_source_mode",
    "checkpoint_path",
    "checkpoint_validation_mode",
    "checkpoint_validator_ran",
    "checkpoint_loaded",
    "trained_model",
    "performance_claim_allowed",
    "placeholder_fallback_used",
    "mock_action_used",
    "qwen_train",
    "qwen_inference",
    "qwen_trigger_rate",
    "reward_version",
    "energy_proxy_model_version",
    "k_dist_kwh_per_m",
    "k_acc_kwh_per_event",
    "k_idle_kwh_per_sec",
    "actual_policy_claim_ready",
    "causal_policy_claim_ready",
]


class PolicySourceMetadataError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicySourceMetadataValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    normalized: Dict[str, Any]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False

    return bool(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _float_equal(a: Any, b: Any, tol: float = 1.0e-12) -> bool:
    return abs(_as_float(a) - _as_float(b)) <= tol


def _energy_constants_from_action_metadata(action_metadata: Mapping[str, Any]) -> Dict[str, Any]:
    constants = action_metadata.get("energy_proxy_constants", {})
    if constants is None:
        constants = {}

    return {
        "k_dist_kwh_per_m": constants.get(
            "k_dist_kwh_per_m",
            action_metadata.get("k_dist_kwh_per_m", EXPECTED_K_DIST_KWH_PER_M),
        ),
        "k_acc_kwh_per_event": constants.get(
            "k_acc_kwh_per_event",
            action_metadata.get("k_acc_kwh_per_event", EXPECTED_K_ACC_KWH_PER_EVENT),
        ),
        "k_idle_kwh_per_sec": constants.get(
            "k_idle_kwh_per_sec",
            action_metadata.get("k_idle_kwh_per_sec", EXPECTED_K_IDLE_KWH_PER_SEC),
        ),
    }


def derive_source_mode(
    *,
    condition_id: str,
    validation_mode: str,
    trained_model: bool,
    performance_claim_allowed: bool,
    causal_simulator: bool = False,
) -> str:
    """
    Derive source_mode for rollout rows.

    Important:
    - causal_* source_mode is allowed only when causal_simulator=True and
      actual trained checkpoint conditions are satisfied.
    - smoke/noncausal source_mode must never be used for performance claims.
    """

    cid = str(condition_id).strip().upper()
    mode = str(validation_mode).strip().lower()

    if (
        causal_simulator
        and mode == "actual"
        and bool(trained_model)
        and bool(performance_claim_allowed)
    ):
        return f"causal_{cid}_mappo_policy_v1"

    if mode == "actual":
        return f"noncausal_{cid}_mappo_policy_actual_v1"

    return f"noncausal_{cid}_mappo_policy_smoke_v1"


def build_rollout_policy_metadata(
    action_metadata: Mapping[str, Any],
    *,
    condition_id: str = "A",
    causal_simulator: bool = False,
    source_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert MAPPOPolicyActionSource metadata into rollout-row-safe metadata.

    This helper is intended for A/A90/A80/A70 rollout writers. It does not
    perform simulator stepping. It only normalizes policy provenance fields.
    """

    cid = str(condition_id).strip().upper()

    validation_mode = str(
        action_metadata.get("checkpoint_validation_mode", "actual")
    ).strip().lower()

    trained_model = _as_bool(action_metadata.get("trained_model", False))
    performance_claim_allowed = _as_bool(
        action_metadata.get("performance_claim_allowed", False)
    )

    if source_mode is None:
        source_mode = derive_source_mode(
            condition_id=cid,
            validation_mode=validation_mode,
            trained_model=trained_model,
            performance_claim_allowed=performance_claim_allowed,
            causal_simulator=bool(causal_simulator),
        )

    constants = _energy_constants_from_action_metadata(action_metadata)

    normalized = {
        "policy_metadata_version": POLICY_METADATA_VERSION,
        "condition_id": cid,
        "source_mode": str(source_mode),
        "policy_source": str(action_metadata.get("policy_source", "")),
        "policy_action_source_version": str(
            action_metadata.get("policy_action_source_version", "")
        ),
        "policy_action_source_mode": str(
            action_metadata.get("policy_action_source_mode", "")
        ),
        "checkpoint_path": str(action_metadata.get("checkpoint_path", "")),
        "checkpoint_validation_mode": validation_mode,
        "checkpoint_validator_ran": _as_bool(
            action_metadata.get("checkpoint_validator_ran", False)
        ),
        "checkpoint_loaded": _as_bool(action_metadata.get("checkpoint_loaded", False)),
        "trained_model": trained_model,
        "performance_claim_allowed": performance_claim_allowed,
        "placeholder_fallback_used": _as_bool(
            action_metadata.get("placeholder_fallback_used", True)
        ),
        "mock_action_used": _as_bool(action_metadata.get("mock_action_used", True)),
        "qwen_train": _as_bool(action_metadata.get("qwen_train", True)),
        "qwen_inference": _as_bool(action_metadata.get("qwen_inference", True)),
        "qwen_trigger_rate": _as_float(action_metadata.get("qwen_trigger_rate", 1.0)),
        "reward_version": str(action_metadata.get("reward_version", "")),
        "energy_proxy_model_version": str(
            action_metadata.get("energy_proxy_model_version", "")
        ),
        "k_dist_kwh_per_m": _as_float(constants["k_dist_kwh_per_m"]),
        "k_acc_kwh_per_event": _as_float(constants["k_acc_kwh_per_event"]),
        "k_idle_kwh_per_sec": _as_float(constants["k_idle_kwh_per_sec"]),
    }

    actual_policy_claim_ready = (
        normalized["policy_source"] == EXPECTED_POLICY_SOURCE
        and normalized["policy_action_source_version"] == EXPECTED_ACTION_SOURCE_VERSION
        and normalized["checkpoint_validator_ran"] is True
        and normalized["checkpoint_loaded"] is True
        and normalized["trained_model"] is True
        and normalized["performance_claim_allowed"] is True
        and normalized["placeholder_fallback_used"] is False
        and normalized["mock_action_used"] is False
        and normalized["qwen_train"] is False
        and normalized["qwen_inference"] is False
        and float(normalized["qwen_trigger_rate"]) == 0.0
        and normalized["reward_version"] == EXPECTED_REWARD_VERSION
        and normalized["energy_proxy_model_version"] == EXPECTED_ENERGY_PROXY_MODEL_VERSION
        and _float_equal(normalized["k_dist_kwh_per_m"], EXPECTED_K_DIST_KWH_PER_M)
        and _float_equal(normalized["k_acc_kwh_per_event"], EXPECTED_K_ACC_KWH_PER_EVENT)
        and _float_equal(normalized["k_idle_kwh_per_sec"], EXPECTED_K_IDLE_KWH_PER_SEC)
    )

    causal_policy_claim_ready = (
        bool(actual_policy_claim_ready)
        and bool(causal_simulator)
        and str(normalized["source_mode"]).startswith("causal_")
    )

    normalized["actual_policy_claim_ready"] = bool(actual_policy_claim_ready)
    normalized["causal_policy_claim_ready"] = bool(causal_policy_claim_ready)

    return normalized


def validate_rollout_policy_metadata(
    metadata: Mapping[str, Any],
    *,
    require_actual_ready: bool = False,
    require_causal_ready: bool = False,
) -> PolicySourceMetadataValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    normalized = dict(metadata)

    missing = [k for k in REQUIRED_POLICY_METADATA_FIELDS if k not in normalized]
    if missing:
        errors.append(f"missing policy metadata fields: {missing}")

    if normalized.get("policy_metadata_version") != POLICY_METADATA_VERSION:
        errors.append(
            f"policy_metadata_version mismatch: expected {POLICY_METADATA_VERSION}, "
            f"got {normalized.get('policy_metadata_version')}"
        )

    if normalized.get("policy_source") != EXPECTED_POLICY_SOURCE:
        errors.append(
            f"policy_source mismatch: expected {EXPECTED_POLICY_SOURCE}, got {normalized.get('policy_source')}"
        )

    if normalized.get("policy_action_source_version") != EXPECTED_ACTION_SOURCE_VERSION:
        errors.append(
            "policy_action_source_version mismatch: "
            f"expected {EXPECTED_ACTION_SOURCE_VERSION}, got {normalized.get('policy_action_source_version')}"
        )

    if _as_bool(normalized.get("checkpoint_validator_ran", False)) is not True:
        errors.append("checkpoint_validator_ran must be true")

    if _as_bool(normalized.get("checkpoint_loaded", False)) is not True:
        errors.append("checkpoint_loaded must be true")

    if _as_bool(normalized.get("placeholder_fallback_used", True)) is not False:
        errors.append("placeholder_fallback_used must be false")

    if _as_bool(normalized.get("mock_action_used", True)) is not False:
        errors.append("mock_action_used must be false")

    if _as_bool(normalized.get("qwen_train", True)) is not False:
        errors.append("qwen_train must be false")

    if _as_bool(normalized.get("qwen_inference", True)) is not False:
        errors.append("qwen_inference must be false")

    if _as_float(normalized.get("qwen_trigger_rate", 1.0)) != 0.0:
        errors.append("qwen_trigger_rate must be 0.0")

    if normalized.get("reward_version") != EXPECTED_REWARD_VERSION:
        errors.append(
            f"reward_version mismatch: expected {EXPECTED_REWARD_VERSION}, got {normalized.get('reward_version')}"
        )

    if normalized.get("energy_proxy_model_version") != EXPECTED_ENERGY_PROXY_MODEL_VERSION:
        errors.append(
            "energy_proxy_model_version mismatch: "
            f"expected {EXPECTED_ENERGY_PROXY_MODEL_VERSION}, got {normalized.get('energy_proxy_model_version')}"
        )

    if not _float_equal(normalized.get("k_dist_kwh_per_m"), EXPECTED_K_DIST_KWH_PER_M):
        errors.append("k_dist_kwh_per_m mismatch")

    if not _float_equal(normalized.get("k_acc_kwh_per_event"), EXPECTED_K_ACC_KWH_PER_EVENT):
        errors.append("k_acc_kwh_per_event mismatch")

    if not _float_equal(normalized.get("k_idle_kwh_per_sec"), EXPECTED_K_IDLE_KWH_PER_SEC):
        errors.append("k_idle_kwh_per_sec mismatch")

    actual_ready = _as_bool(normalized.get("actual_policy_claim_ready", False))
    causal_ready = _as_bool(normalized.get("causal_policy_claim_ready", False))

    if require_actual_ready and not actual_ready:
        errors.append("actual_policy_claim_ready is required but false")

    if require_causal_ready and not causal_ready:
        errors.append("causal_policy_claim_ready is required but false")

    if not actual_ready:
        warnings.append(
            "actual_policy_claim_ready=false; this metadata is valid for smoke/non-claim flow only"
        )

    if str(normalized.get("source_mode", "")).startswith("causal_") and not causal_ready:
        errors.append("source_mode is causal_* but causal_policy_claim_ready is false")

    return PolicySourceMetadataValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        normalized=normalized,
    )


def validate_policy_metadata_records(
    records: Iterable[Mapping[str, Any]],
    *,
    require_actual_ready: bool = False,
    require_causal_ready: bool = False,
) -> Dict[str, Any]:
    results = [
        validate_rollout_policy_metadata(
            row,
            require_actual_ready=require_actual_ready,
            require_causal_ready=require_causal_ready,
        )
        for row in records
    ]

    errors = []
    warnings = []

    for idx, result in enumerate(results):
        for err in result.errors:
            errors.append({"row_index": idx, "error": err})
        for warning in result.warnings:
            warnings.append({"row_index": idx, "warning": warning})

    return {
        "valid": len(errors) == 0,
        "record_count": len(results),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def policy_metadata_columns() -> List[str]:
    return list(REQUIRED_POLICY_METADATA_FIELDS)
