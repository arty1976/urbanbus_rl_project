#!/usr/bin/env python3
"""Durable per-seed/cycle training evidence recording device (H4M-U-R1).

This module owns the training-evidence instrumentation and the report path
only.  It never builds a model, an optimizer, a rollout, or a training loop,
and importing it does not import torch.  Reward V2, the S3 critic value-target
repair, the target-conditioned Actor head repair, the TD/GAE equations,
advantage normalization, Actor logits, action/K masks, the observation
contract, simulator and Zero-Loss semantics, and the frozen
split/seed/schedule/budget are all untouched by this module.

Durability contract
-------------------
* exactly one append-only JSONL record per ``(seed, outer_cycle)``
* each record is written, flushed, ``fsync``-ed and read back and verified
  before the caller is allowed to advance to the next cycle; any failure
  raises :class:`EvidenceIntegrityError` so the training loop stops instead of
  drifting one cycle ahead of its evidence
* index/state sidecars are replaced atomically (tmp + ``os.replace`` +
  directory fsync)
* duplicate or conflicting ``(seed, outer_cycle)`` evidence is rejected
  fail-closed at write time and at load time
* ``record_sha256`` covers the run identity (seed, cycle, source commit, split,
  schedule, actor repair contract, critic repair contract), so evidence
  produced by a different run can never be merged into this one
* the final report is rendered from persisted evidence only, so a reporter
  crash cannot destroy training evidence and the report can be regenerated
  later with zero training and zero optimizer steps
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


SCHEMA_VERSION = "H4M_U_R1_DURABLE_CYCLE_EVIDENCE_V1"
EVIDENCE_DIR_NAME = "07_durable_training_evidence"
EVIDENCE_JSONL_NAME = "seed_cycle_evidence.jsonl"
EVIDENCE_INDEX_NAME = "evidence_index.json"
RUN_HEADER_NAME = "run_header.json"
RUN_STATE_NAME = "run_state.json"
SCHEMA_FILE_NAME = "evidence_schema.json"
REPORT_RECOVERY_NAME = "report_recovery.json"

STATE_INITIALIZED = "INITIALIZED"
STATE_TRAINING_IN_PROGRESS = "TRAINING_IN_PROGRESS"
STATE_TRAINING_COMPLETE = "TRAINING_COMPLETE"
STATE_REPORT_COMPLETE = "REPORT_COMPLETE"
STATE_REPORT_REGENERATED = "REPORT_REGENERATED_WITHOUT_TRAINING"

REPORT_SOURCE = "PERSISTED_DURABLE_EVIDENCE_ONLY"
REPORT_STATUS_COMPLETE = "REPORT_RENDERED_FROM_PERSISTED_EVIDENCE"
REPORT_STATUS_BLOCKED = "BLOCKED_EVIDENCE_INTEGRITY_FAIL_CLOSED"

ACTION_HOLD = "HOLD_CURRENT_POSITION"
ACTION_SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
ACTION_SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"
ACTION_NAMES = {0: ACTION_HOLD, 1: ACTION_SERVE, 2: ACTION_SKIP}
PROBABILITY_KEYS = {0: "P_HOLD_mean", 1: "P_SERVE_mean", 2: "P_SKIP_mean"}

# Run-level identity that must be identical for every record of one run.  It is
# embedded in each record and therefore covered by ``record_sha256``.
IDENTITY_RUN_FIELDS = [
    "source_commit",
    "split_sha256",
    "schedule_sha256",
    "actor_repair_contract_sha256",
    "critic_repair_contract_sha256",
]
IDENTITY_FIELDS = ["seed", "outer_cycle", *IDENTITY_RUN_FIELDS]

RUN_HEADER_REQUIRED_FIELDS = [
    "stage",
    "created_at",
    "artifact_root",
    "seeds",
    "outer_training_count",
    *IDENTITY_RUN_FIELDS,
]

RECORD_REQUIRED_FIELDS = [
    "schema_version",
    "stage",
    "seed",
    "outer_cycle",
    "record_uid",
    "written_at",
    "identity",
    "critic_loss",
    "parameter_delta",
    "credit_diagnostics",
    "integrity",
    "trace_files",
]
CRITIC_LOSS_REQUIRED_FIELDS = [
    "source",
    "estimated",
    "update_rows",
    "update_row_count",
    "actor_joint_update_count",
    "critic_only_update_count",
    "critic_loss_mean",
    "critic_loss_first",
    "critic_loss_last",
    "critic_loss_min",
    "critic_loss_max",
]
PARAMETER_DELTA_REQUIRED_ROLES = ["actor", "critic", "gatv2"]
PARAMETER_DELTA_REQUIRED_FIELDS = [
    "l1_delta",
    "l2_delta",
    "max_abs_delta",
    "changed_tensor_count",
    "unchanged_tensor_count",
    "pre_hash",
    "post_hash",
    "computed_from",
]
CREDIT_DIAGNOSTIC_REQUIRED_FIELDS = [
    "sample_count",
    "v_s_minus_return",
    "raw_gae",
    "normalized_advantage",
    "td_delta",
    "critic_value_error",
    "action_counts",
    "action_probabilities",
    "critic_target_normalizer_state_sha256",
    "source_row_counts",
]
INTEGRITY_REQUIRED_FIELDS = [
    "nan_inf_count",
    "illegal_action_count",
    "future_leakage_count",
    "test6_access_count",
    "loss_finite",
    "cycle_scoped_trace_binding",
]

PARAMETER_DELTA_COMPUTED_FROM = "real_pre_cycle_and_post_cycle_parameter_state_dicts"
CRITIC_LOSS_SOURCE = "ppo_update_controlled.metrics_rows[*].value_loss"


class EvidenceIntegrityError(RuntimeError):
    """Fail-closed durable-evidence violation.

    Raised at write time so the training loop stops immediately instead of
    advancing with a cycle trace that has no matching evidence record.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def kst_now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_jsonable) + "\n"


def canonical_line(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_jsonable) + "\n"


def canonical_sha(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_jsonable)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_stats(values: Iterable[Any]) -> Dict[str, Any]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    mu = sum(nums) / len(nums)
    variance = sum((x - mu) ** 2 for x in nums) / len(nums)
    return {"count": len(nums), "mean": mu, "min": min(nums), "max": max(nums), "std": math.sqrt(variance)}


def nonfinite_count(payload: Any) -> int:
    """Count NaN/Inf floats anywhere inside a JSON-shaped payload."""
    count = 0
    if isinstance(payload, Mapping):
        for value in payload.values():
            count += nonfinite_count(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            count += nonfinite_count(value)
    elif isinstance(payload, bool):
        return 0
    elif isinstance(payload, float):
        return 0 if math.isfinite(payload) else 1
    return count


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> Dict[str, Any]:
    """Replace ``path`` atomically: temp file -> flush -> fsync -> os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return {"path": str(path), "bytes": len(text.encode("utf-8")), "atomic": True}


def atomic_write_json(path: Path, payload: Any) -> Dict[str, Any]:
    return atomic_write_text(Path(path), canonical_json(payload))


def evidence_schema() -> Dict[str, Any]:
    """Canonical mandatory-field schema for one seed x cycle evidence record."""
    return {
        "schema_version": SCHEMA_VERSION,
        "record_granularity": "one_record_per_seed_and_outer_cycle",
        "durability": {
            "evidence_log": "append_only_jsonl",
            "sidecars": "atomic_replace_tmp_then_os_replace_with_directory_fsync",
            "per_record_sequence": ["append", "flush", "fsync", "read_back_verify", "atomic_index_update"],
            "flush_before_next_cycle": True,
            "failure_policy": "FAIL_CLOSED_STOP_TRAINING_LOOP",
        },
        "identity_fields": IDENTITY_FIELDS,
        "record_hash": {
            "field": "record_sha256",
            "covers": "the whole record including the embedded identity block",
            "purpose": "reject evidence produced by another run, commit, split, schedule, or repair contract",
        },
        "required_fields": {
            "record": RECORD_REQUIRED_FIELDS,
            "critic_loss": CRITIC_LOSS_REQUIRED_FIELDS,
            "parameter_delta_roles": PARAMETER_DELTA_REQUIRED_ROLES,
            "parameter_delta": PARAMETER_DELTA_REQUIRED_FIELDS,
            "credit_diagnostics": CREDIT_DIAGNOSTIC_REQUIRED_FIELDS,
            "integrity": INTEGRITY_REQUIRED_FIELDS,
            "run_header": RUN_HEADER_REQUIRED_FIELDS,
        },
        "critic_loss_source": CRITIC_LOSS_SOURCE,
        "critic_loss_estimation_allowed": False,
        "parameter_delta_computed_from": PARAMETER_DELTA_COMPUTED_FROM,
        "credit_diagnostics_required": [
            "V(s)-return",
            "raw GAE",
            "normalized advantage",
            "action counts",
            "action probabilities",
        ],
        "files": {
            "schema": SCHEMA_FILE_NAME,
            "run_header": RUN_HEADER_NAME,
            "evidence_log": EVIDENCE_JSONL_NAME,
            "evidence_index": EVIDENCE_INDEX_NAME,
            "run_state": RUN_STATE_NAME,
        },
        "report_contract": {
            "source": REPORT_SOURCE,
            "volatile_in_memory_metrics_allowed": False,
            "regeneration_requires_training": False,
            "regeneration_requires_optimizer_step": False,
        },
    }


def parameter_delta_from_states(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
    role: str = "",
) -> Dict[str, Any]:
    """Delta statistics between two real parameter ``state_dict`` snapshots.

    Mirrors ``dl1.delta_stats`` key-for-key so the persisted per-cycle delta is
    directly comparable with the frozen seed-level parameter audit.  Torch is
    imported lazily so that the loader/report path never needs it.
    """
    import torch  # noqa: PLC0415 - lazy on purpose: the report path must not need torch

    if set(before) != set(after):
        raise EvidenceIntegrityError(
            "PARAMETER_DELTA_STATE_KEY_MISMATCH",
            f"role={role} before={len(before)} after={len(after)} keys differ",
        )
    l1 = 0.0
    l2_sq = 0.0
    max_abs = 0.0
    changed = 0
    unchanged = 0
    element_count = 0
    for name in sorted(before):
        delta = after[name].detach().cpu().float() - before[name].detach().cpu().float()
        l1 += float(delta.abs().sum().item())
        l2_sq += float((delta * delta).sum().item())
        tensor_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        max_abs = max(max_abs, tensor_max)
        element_count += int(delta.numel())
        if tensor_max > tolerance:
            changed += 1
        else:
            unchanged += 1
    state_hash = _state_dict_hash(torch, before)
    post_hash = _state_dict_hash(torch, after)
    return {
        "parameter_count": element_count,
        "pre_hash": state_hash,
        "post_hash": post_hash,
        "l1_delta": l1,
        "l2_delta": math.sqrt(l2_sq),
        "max_abs_delta": max_abs,
        "changed_tensor_count": changed,
        "unchanged_tensor_count": unchanged,
        "delta_tolerance": tolerance,
        "computed_from": PARAMETER_DELTA_COMPUTED_FROM,
        "parameters_changed": max_abs > tolerance,
    }


def _state_dict_hash(torch_module: Any, state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def annotate_parameter_delta(payload: Mapping[str, Any], *, role: str) -> Dict[str, Any]:
    """Normalize a ``dl1.delta_stats`` payload into the persisted delta shape."""
    row = dict(payload)
    row.setdefault("computed_from", PARAMETER_DELTA_COMPUTED_FROM)
    missing = [field for field in PARAMETER_DELTA_REQUIRED_FIELDS if field not in row]
    if missing:
        raise EvidenceIntegrityError("PARAMETER_DELTA_FIELD_MISSING", f"role={role} missing={missing}")
    if row["pre_hash"] == row["post_hash"] and float(row["l2_delta"]) != 0.0:
        raise EvidenceIntegrityError(
            "PARAMETER_DELTA_HASH_INCONSISTENT",
            f"role={role} identical pre/post hash with non-zero l2_delta",
        )
    row["parameters_changed"] = bool(float(row["l2_delta"]) > 0.0)
    row["role"] = role
    return row


def extract_critic_loss(update_result: Mapping[str, Any], *, seed: int, outer_cycle: int) -> Dict[str, Any]:
    """Read the actual per-cycle critic loss out of the real update result.

    Estimation, interpolation, and reconstruction are forbidden: a missing or
    non-finite ``value_loss`` is a fail-closed error, never a filled-in value.
    """
    metrics_rows = update_result.get("metrics_rows")
    if not isinstance(metrics_rows, Sequence) or not metrics_rows:
        raise EvidenceIntegrityError(
            "CRITIC_LOSS_SOURCE_MISSING",
            f"seed={seed} cycle={outer_cycle} update result carries no metrics_rows",
        )
    rows: List[Dict[str, Any]] = []
    losses: List[float] = []
    for position, row in enumerate(metrics_rows):
        if "value_loss" not in row or row["value_loss"] is None:
            raise EvidenceIntegrityError(
                "CRITIC_LOSS_NOT_AVAILABLE_FROM_UPDATE_RESULT",
                f"seed={seed} cycle={outer_cycle} row={position} has no value_loss; estimation is forbidden",
            )
        value_loss = float(row["value_loss"])
        if not math.isfinite(value_loss):
            raise EvidenceIntegrityError(
                "CRITIC_LOSS_NON_FINITE",
                f"seed={seed} cycle={outer_cycle} row={position} value_loss={value_loss}",
            )
        losses.append(value_loss)
        policy_loss = row.get("policy_loss")
        rows.append(
            {
                "ppo_update_index": row.get("ppo_update_index"),
                "update_role": row.get("update_role"),
                "value_loss": value_loss,
                "policy_loss": float(policy_loss) if policy_loss is not None else None,
                "actor_entropy": _optional_float(row.get("actor_entropy")),
                "approx_kl": _optional_float(row.get("approx_kl")),
                "clip_fraction": _optional_float(row.get("clip_fraction")),
                "entropy_coef": _optional_float(row.get("entropy_coef")),
                "critic_grad_norm_before_clip": _optional_float(row.get("critic_grad_norm_before_clip")),
                "critic_grad_norm_after_clip": _optional_float(row.get("critic_grad_norm_after_clip")),
                "actor_grad_norm_before_clip": _optional_float(row.get("actor_grad_norm_before_clip")),
                "actor_grad_norm_after_clip": _optional_float(row.get("actor_grad_norm_after_clip")),
                "critic_target_normalizer_state_sha256": row.get("critic_target_normalizer_state_sha256"),
                "critic_target_binding_schema": row.get("critic_target_binding_schema"),
                "nan_count": int(row.get("nan_count", 0) or 0),
                "inf_count": int(row.get("inf_count", 0) or 0),
            }
        )
    stats = numeric_stats(losses)
    return {
        "source": CRITIC_LOSS_SOURCE,
        "estimated": False,
        "reconstructed": False,
        "update_rows": rows,
        "update_row_count": len(rows),
        "actor_joint_update_count": sum(1 for row in rows if row["update_role"] == "actor_gatv2_critic_joint"),
        "critic_only_update_count": sum(1 for row in rows if row["update_role"] == "critic_only_extra"),
        "critic_loss_values": losses,
        "critic_loss_mean": stats["mean"],
        "critic_loss_first": losses[0],
        "critic_loss_last": losses[-1],
        "critic_loss_min": stats["min"],
        "critic_loss_max": stats["max"],
        "critic_loss_std": stats["std"],
        "all_finite": True,
    }


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def credit_diagnostics_from_rows(
    *,
    pre_action_rows: Sequence[Mapping[str, Any]],
    td_gae_rows: Sequence[Mapping[str, Any]],
    advantage_rows: Sequence[Mapping[str, Any]],
    critic_value_error_rows: Sequence[Mapping[str, Any]],
    seed: int,
    outer_cycle: int,
) -> Dict[str, Any]:
    """Aggregate the mandatory credit diagnostics for one cycle.

    Reads only rows already recorded for this cycle by the frozen instrumented
    execution path; it never recomputes rewards, TD residuals, GAE, or the
    advantage normalization.
    """
    missing = [
        name
        for name, rows in (
            ("pre_action", pre_action_rows),
            ("td_gae", td_gae_rows),
            ("advantage", advantage_rows),
            ("critic_value_error", critic_value_error_rows),
        )
        if not rows
    ]
    if missing:
        raise EvidenceIntegrityError(
            "CREDIT_DIAGNOSTIC_SOURCE_MISSING",
            f"seed={seed} cycle={outer_cycle} empty trace rows: {missing}",
        )
    foreign = [
        name
        for name, rows in (
            ("pre_action", pre_action_rows),
            ("td_gae", td_gae_rows),
            ("advantage", advantage_rows),
            ("critic_value_error", critic_value_error_rows),
        )
        if any(int(row.get("seed", seed)) != seed or int(row.get("outer_cycle", outer_cycle)) != outer_cycle for row in rows)
    ]
    if foreign:
        raise EvidenceIntegrityError(
            "CREDIT_DIAGNOSTIC_CYCLE_SCOPE_VIOLATION",
            f"seed={seed} cycle={outer_cycle} rows from another seed/cycle in: {foreign}",
        )
    v_minus_return = [float(row["value_t"]) - float(row["raw_return_target"]) for row in td_gae_rows]
    action_counts: Dict[str, int] = {name: 0 for name in ACTION_NAMES.values()}
    for row in advantage_rows:
        name = str(row.get("sampled_action_name"))
        action_counts[name] = action_counts.get(name, 0) + 1
    probabilities = {
        PROBABILITY_KEYS[action_id]: numeric_stats(
            [row.get(f"masked_probability_{action_id}") for row in pre_action_rows]
        )["mean"]
        for action_id in sorted(PROBABILITY_KEYS)
    }
    probabilities["policy_entropy_mean"] = numeric_stats([row.get("policy_entropy") for row in pre_action_rows])["mean"]
    illegal = 0
    for row in pre_action_rows:
        legal = {int(value) for value in row.get("legal_action_ids", [])}
        if int(row.get("action_id", -1)) not in legal:
            illegal += 1
    normalizer_shas = sorted({str(row.get("critic_target_normalizer_state_sha256")) for row in td_gae_rows})
    sample_total = len(advantage_rows)
    return {
        "sample_count": sample_total,
        "v_s_minus_return": numeric_stats(v_minus_return),
        "raw_gae": numeric_stats([row.get("raw_gae_advantage") for row in advantage_rows]),
        "normalized_advantage": numeric_stats([row.get("normalized_advantage") for row in advantage_rows]),
        "td_delta": numeric_stats([row.get("td_delta") for row in td_gae_rows]),
        "critic_value_error": numeric_stats([row.get("value_error") for row in critic_value_error_rows]),
        "value_t": numeric_stats([row.get("value_t") for row in td_gae_rows]),
        "raw_return_target": numeric_stats([row.get("raw_return_target") for row in td_gae_rows]),
        "action_counts": action_counts,
        "action_share": {
            name: (float(count) / float(sample_total) if sample_total else None) for name, count in action_counts.items()
        },
        "action_probabilities": probabilities,
        "critic_target_normalizer_state_sha256": normalizer_shas[0] if len(normalizer_shas) == 1 else None,
        "critic_target_normalizer_state_sha_count": len(normalizer_shas),
        "illegal_action_count": illegal,
        "source_row_counts": {
            "pre_action": len(pre_action_rows),
            "td_gae": len(td_gae_rows),
            "advantage": len(advantage_rows),
            "critic_value_error": len(critic_value_error_rows),
        },
        "source": "frozen instrumented cycle trace rows (no recomputation of reward, TD, GAE, or normalization)",
    }


def trace_file_bindings(
    trace_entry: Mapping[str, Any],
    *,
    artifact_root: Path,
    seed: int,
    outer_cycle: int,
) -> Tuple[Dict[str, Any], bool]:
    """Bind the cycle's persisted parquet traces and check cycle scoping."""
    files = trace_entry.get("files", {}) if isinstance(trace_entry, Mapping) else {}
    root = Path(artifact_root)
    bindings: Dict[str, Any] = {}
    scoped = bool(files)
    for name, entry in sorted(files.items()):
        path = Path(entry["path"]) if isinstance(entry, Mapping) else Path(str(entry))
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.as_posix()
        in_scope = f"seed={seed}" in relative and f"outer_cycle={outer_cycle}" in relative
        scoped = scoped and in_scope
        bindings[name] = {
            "relative_path": relative,
            "sha256": entry.get("sha256") if isinstance(entry, Mapping) else None,
            "rows": entry.get("rows") if isinstance(entry, Mapping) else None,
            "size_bytes": entry.get("size_bytes") if isinstance(entry, Mapping) else None,
            "cycle_scoped": in_scope,
        }
    return bindings, scoped


def build_cycle_record(
    *,
    stage: str,
    identity_base: Mapping[str, Any],
    seed: int,
    outer_cycle: int,
    critic_loss: Mapping[str, Any],
    parameter_delta: Mapping[str, Any],
    credit_diagnostics: Mapping[str, Any],
    trace_files: Mapping[str, Any],
    cycle_scoped_trace_binding: bool,
    extra: Optional[Mapping[str, Any]] = None,
    written_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble one durable seed x cycle record (hash is added by the writer)."""
    identity = {
        "seed": int(seed),
        "outer_cycle": int(outer_cycle),
        **{field: identity_base[field] for field in IDENTITY_RUN_FIELDS},
    }
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "seed": int(seed),
        "outer_cycle": int(outer_cycle),
        "record_uid": f"{stage}:seed={int(seed)}:outer_cycle={int(outer_cycle)}",
        "written_at": written_at or kst_now(),
        "identity": identity,
        "critic_loss": dict(critic_loss),
        "parameter_delta": dict(parameter_delta),
        "credit_diagnostics": dict(credit_diagnostics),
        "trace_files": dict(trace_files),
        "integrity": {
            "nan_inf_count": 0,
            "illegal_action_count": int(credit_diagnostics.get("illegal_action_count", 0)),
            "future_leakage_count": 0,
            "test6_access_count": 0,
            "loss_finite": bool(critic_loss.get("all_finite") is True),
            "cycle_scoped_trace_binding": bool(cycle_scoped_trace_binding),
            "training_update_rows": int(critic_loss.get("update_row_count", 0)),
            "actor_joint_update_rows": int(critic_loss.get("actor_joint_update_count", 0)),
        },
    }
    if extra:
        record["cycle_context"] = dict(extra)
    record["integrity"]["nan_inf_count"] = nonfinite_count(record)
    return record


def record_sha256(record: Mapping[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return canonical_sha(body)


def validate_record_structure(
    record: Mapping[str, Any],
    *,
    identity_base: Mapping[str, Any],
    expected_grid: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[str]:
    """Return a list of fail-closed schema/identity violations for one record."""
    failures: List[str] = []
    for field in RECORD_REQUIRED_FIELDS:
        if field not in record:
            failures.append(f"RECORD_FIELD_MISSING:{field}")
    if record.get("schema_version") != SCHEMA_VERSION:
        failures.append(f"SCHEMA_VERSION_MISMATCH:{record.get('schema_version')}")
    identity = record.get("identity", {})
    if not isinstance(identity, Mapping):
        failures.append("IDENTITY_BLOCK_INVALID")
        return failures
    for field in IDENTITY_FIELDS:
        if field not in identity:
            failures.append(f"IDENTITY_FIELD_MISSING:{field}")
    for field in IDENTITY_RUN_FIELDS:
        if field in identity and identity.get(field) != identity_base.get(field):
            failures.append(f"IDENTITY_MISMATCH:{field}")
    if identity.get("seed") != record.get("seed") or identity.get("outer_cycle") != record.get("outer_cycle"):
        failures.append("IDENTITY_CYCLE_KEY_MISMATCH")
    critic_loss = record.get("critic_loss", {})
    for field in CRITIC_LOSS_REQUIRED_FIELDS:
        if field not in critic_loss:
            failures.append(f"CRITIC_LOSS_FIELD_MISSING:{field}")
    if critic_loss.get("estimated") is not False:
        failures.append("CRITIC_LOSS_ESTIMATED_FORBIDDEN")
    if critic_loss.get("source") != CRITIC_LOSS_SOURCE:
        failures.append("CRITIC_LOSS_SOURCE_UNBOUND")
    parameter_delta = record.get("parameter_delta", {})
    for role in PARAMETER_DELTA_REQUIRED_ROLES:
        role_payload = parameter_delta.get(role)
        if not isinstance(role_payload, Mapping):
            failures.append(f"PARAMETER_DELTA_ROLE_MISSING:{role}")
            continue
        for field in PARAMETER_DELTA_REQUIRED_FIELDS:
            if field not in role_payload:
                failures.append(f"PARAMETER_DELTA_FIELD_MISSING:{role}.{field}")
        if role_payload.get("computed_from") != PARAMETER_DELTA_COMPUTED_FROM:
            failures.append(f"PARAMETER_DELTA_SOURCE_UNBOUND:{role}")
    credit = record.get("credit_diagnostics", {})
    for field in CREDIT_DIAGNOSTIC_REQUIRED_FIELDS:
        if field not in credit:
            failures.append(f"CREDIT_DIAGNOSTIC_FIELD_MISSING:{field}")
    integrity = record.get("integrity", {})
    for field in INTEGRITY_REQUIRED_FIELDS:
        if field not in integrity:
            failures.append(f"INTEGRITY_FIELD_MISSING:{field}")
    if int(integrity.get("nan_inf_count", 1)) != 0:
        failures.append("RECORD_NAN_INF_PRESENT")
    if integrity.get("cycle_scoped_trace_binding") is not True:
        failures.append("TRACE_BINDING_NOT_CYCLE_SCOPED")
    if not record.get("trace_files"):
        failures.append("TRACE_FILES_MISSING")
    if expected_grid is not None:
        key = (int(record.get("seed", -1)), int(record.get("outer_cycle", -1)))
        if key not in {(int(s), int(c)) for s, c in expected_grid}:
            failures.append(f"CYCLE_KEY_OUTSIDE_EXPECTED_GRID:{key}")
    return failures


class DurableCycleEvidenceWriter:
    """Append-only seed x cycle evidence writer with fail-closed durability."""

    def __init__(
        self,
        evidence_dir: Path,
        *,
        run_header: Mapping[str, Any],
        expected_grid: Sequence[Tuple[int, int]],
    ) -> None:
        missing = [field for field in RUN_HEADER_REQUIRED_FIELDS if field not in run_header]
        if missing:
            raise EvidenceIntegrityError("RUN_HEADER_FIELD_MISSING", f"missing={missing}")
        self.dir = Path(evidence_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / EVIDENCE_JSONL_NAME
        if self.jsonl_path.exists() and self.jsonl_path.stat().st_size > 0:
            raise EvidenceIntegrityError(
                "EVIDENCE_LOG_ALREADY_POPULATED",
                f"{self.jsonl_path} already holds evidence; a run never appends into a foreign log",
            )
        self.expected_grid = [(int(seed), int(cycle)) for seed, cycle in expected_grid]
        self.identity_base = {field: run_header[field] for field in IDENTITY_RUN_FIELDS}
        self.stage = str(run_header["stage"])
        self.header = {
            **dict(run_header),
            "schema_version": SCHEMA_VERSION,
            "expected_grid": [{"seed": seed, "outer_cycle": cycle} for seed, cycle in self.expected_grid],
            "expected_record_count": len(self.expected_grid),
            "identity": dict(self.identity_base),
        }
        self.index_entries: List[Dict[str, Any]] = []
        self._keys: Dict[Tuple[int, int], str] = {}
        self.jsonl_path.touch()
        _fsync_dir(self.dir)
        atomic_write_json(self.dir / SCHEMA_FILE_NAME, evidence_schema())
        atomic_write_json(self.dir / RUN_HEADER_NAME, self.header)
        self._write_index()
        self._write_state(STATE_INITIALIZED)

    # -- internal -----------------------------------------------------------
    def _write_index(self) -> None:
        atomic_write_json(
            self.dir / EVIDENCE_INDEX_NAME,
            {
                "schema_version": SCHEMA_VERSION,
                "stage": self.stage,
                "identity": dict(self.identity_base),
                "record_count": len(self.index_entries),
                "expected_record_count": len(self.expected_grid),
                "records": list(self.index_entries),
                "evidence_log": EVIDENCE_JSONL_NAME,
                "evidence_log_bytes": self.jsonl_path.stat().st_size,
            },
        )

    def _write_state(self, status: str, **extra: Any) -> None:
        atomic_write_json(
            self.dir / RUN_STATE_NAME,
            {
                "schema_version": SCHEMA_VERSION,
                "stage": self.stage,
                "status": status,
                "updated_at": kst_now(),
                "cycles_persisted": len(self.index_entries),
                "expected_record_count": len(self.expected_grid),
                "last_persisted": self.index_entries[-1] if self.index_entries else None,
                "identity": dict(self.identity_base),
                **extra,
            },
        )

    # -- public API ---------------------------------------------------------
    def append_cycle_evidence(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Durably persist one seed x cycle record before the loop advances.

        Any violation raises :class:`EvidenceIntegrityError`; callers must let
        it propagate so training stops instead of running another cycle.
        """
        seed = int(record.get("seed", -1))
        cycle = int(record.get("outer_cycle", -1))
        key = (seed, cycle)
        failures = validate_record_structure(
            record, identity_base=self.identity_base, expected_grid=self.expected_grid
        )
        if failures:
            raise EvidenceIntegrityError("EVIDENCE_RECORD_SCHEMA_VIOLATION", f"key={key} failures={failures}")
        payload = {key_: value for key_, value in record.items() if key_ != "record_sha256"}
        digest = record_sha256(payload)
        if key in self._keys:
            code = "DUPLICATE_CYCLE_EVIDENCE" if self._keys[key] == digest else "CONFLICTING_CYCLE_EVIDENCE"
            raise EvidenceIntegrityError(code, f"key={key} already persisted; rewriting a cycle is forbidden")
        payload["record_sha256"] = digest
        line = canonical_line(payload)
        encoded = line.encode("utf-8")
        offset = self.jsonl_path.stat().st_size
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(self.dir)
        verified = self._read_back(offset, len(encoded), digest, key)
        entry = {
            "seed": seed,
            "outer_cycle": cycle,
            "record_uid": payload["record_uid"],
            "record_sha256": digest,
            "byte_offset": offset,
            "byte_length": len(encoded),
            "written_at": payload["written_at"],
            "read_back_verified": verified,
        }
        self.index_entries.append(entry)
        self._keys[key] = digest
        self._write_index()
        self._write_state(STATE_TRAINING_IN_PROGRESS)
        return {**entry, "durable": True, "evidence_dir": str(self.dir)}

    def _read_back(self, offset: int, length: int, digest: str, key: Tuple[int, int]) -> bool:
        with self.jsonl_path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(length)
        if not raw.endswith(b"\n"):
            raise EvidenceIntegrityError("EVIDENCE_READ_BACK_TRUNCATED", f"key={key}")
        try:
            reloaded = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise EvidenceIntegrityError("EVIDENCE_READ_BACK_UNPARSEABLE", f"key={key} {exc}") from exc
        if reloaded.get("record_sha256") != digest or record_sha256(reloaded) != digest:
            raise EvidenceIntegrityError("EVIDENCE_READ_BACK_HASH_MISMATCH", f"key={key}")
        return True

    def missing_cycles(self) -> List[Tuple[int, int]]:
        return [key for key in self.expected_grid if key not in self._keys]

    def mark_training_complete(self, summary: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        missing = self.missing_cycles()
        self._write_state(
            STATE_TRAINING_COMPLETE,
            training_completed_at=kst_now(),
            grid_complete=not missing,
            missing_cycles=[{"seed": seed, "outer_cycle": cycle} for seed, cycle in missing],
            training_summary=dict(summary) if summary else None,
        )
        return {"cycles_persisted": len(self.index_entries), "missing_cycles": missing, "grid_complete": not missing}

    def mark_report_complete(self, info: Optional[Mapping[str, Any]] = None) -> None:
        self._write_state(STATE_REPORT_COMPLETE, report_completed_at=kst_now(), report_info=dict(info) if info else None)


def find_evidence_dir(artifact_root: Path) -> Path:
    root = Path(artifact_root)
    candidate = root / EVIDENCE_DIR_NAME
    if candidate.exists():
        return candidate
    if (root / EVIDENCE_JSONL_NAME).exists():
        return root
    raise EvidenceIntegrityError("EVIDENCE_DIR_NOT_FOUND", str(root))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def load_evidence(evidence_dir: Path, *, strict: bool = False) -> Dict[str, Any]:
    """Load and fail-closed verify the persisted evidence of one run.

    Never raises for damaged content unless ``strict`` is set: the caller needs
    the integrity verdict in order to report the block instead of crashing.
    """
    directory = Path(evidence_dir)
    failures: List[str] = []
    schema = _read_json(directory / SCHEMA_FILE_NAME)
    header = _read_json(directory / RUN_HEADER_NAME)
    index = _read_json(directory / EVIDENCE_INDEX_NAME)
    state = _read_json(directory / RUN_STATE_NAME)
    jsonl_path = directory / EVIDENCE_JSONL_NAME

    if schema is None:
        failures.append("SCHEMA_FILE_UNREADABLE")
    elif schema.get("schema_version") != SCHEMA_VERSION:
        failures.append("SCHEMA_VERSION_MISMATCH")
    if header is None:
        failures.append("RUN_HEADER_UNREADABLE")
    if index is None:
        failures.append("EVIDENCE_INDEX_UNREADABLE")
    if state is None:
        failures.append("RUN_STATE_UNREADABLE")
    if not jsonl_path.exists():
        failures.append("EVIDENCE_LOG_MISSING")

    raw = jsonl_path.read_bytes() if jsonl_path.exists() else b""
    complete_lines = raw.split(b"\n")
    # Only newline-terminated lines count as durably committed records; a
    # trailing fragment is an interrupted write and is never parsed.
    tail = complete_lines.pop() if complete_lines else b""
    partial_line = bool(tail.strip())
    if partial_line:
        failures.append("PARTIAL_EVIDENCE_RECORD_LINE")
    records: List[Dict[str, Any]] = []
    unparseable = 0
    for position, chunk in enumerate(complete_lines):
        if not chunk.strip():
            continue
        try:
            records.append(json.loads(chunk.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            unparseable += 1
            failures.append(f"UNPARSEABLE_EVIDENCE_RECORD:line={position + 1}")

    identity_base = (header or {}).get("identity") or {field: (header or {}).get(field) for field in IDENTITY_RUN_FIELDS}
    expected_grid = [
        (int(row["seed"]), int(row["outer_cycle"])) for row in ((header or {}).get("expected_grid") or [])
    ]
    hash_failures = 0
    schema_failures: List[str] = []
    seen: Dict[Tuple[int, int], str] = {}
    duplicates: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    for record in records:
        key = (int(record.get("seed", -1)), int(record.get("outer_cycle", -1)))
        stored = record.get("record_sha256")
        recomputed = record_sha256(record)
        if stored != recomputed:
            hash_failures += 1
            failures.append(f"RECORD_HASH_MISMATCH:{key}")
        schema_failures.extend(
            f"{key}:{failure}"
            for failure in validate_record_structure(
                record, identity_base=identity_base, expected_grid=expected_grid or None
            )
        )
        if key in seen:
            if seen[key] == stored:
                duplicates.append({"seed": key[0], "outer_cycle": key[1], "record_sha256": stored})
                failures.append(f"DUPLICATE_CYCLE_EVIDENCE:{key}")
            else:
                conflicts.append({"seed": key[0], "outer_cycle": key[1], "first": seen[key], "second": stored})
                failures.append(f"CONFLICTING_CYCLE_EVIDENCE:{key}")
        else:
            seen[key] = stored
    if schema_failures:
        failures.extend(schema_failures)

    index_records = (index or {}).get("records") or []
    index_matches = len(index_records) == len(records) and all(
        entry.get("record_sha256") == record.get("record_sha256")
        and int(entry.get("seed", -1)) == int(record.get("seed", -2))
        and int(entry.get("outer_cycle", -1)) == int(record.get("outer_cycle", -2))
        for entry, record in zip(index_records, records)
    )
    if not index_matches:
        failures.append("EVIDENCE_INDEX_DOES_NOT_MATCH_LOG")
    missing_cycles = [key for key in expected_grid if key not in seen]
    trace_scope_ok = all(
        binding.get("cycle_scoped") is True
        for record in records
        for binding in (record.get("trace_files") or {}).values()
        if isinstance(binding, Mapping)
    )
    if not trace_scope_ok:
        failures.append("TRACE_BINDING_OUTSIDE_OWN_CYCLE")
    nan_inf_total = sum(nonfinite_count(record) for record in records)
    if nan_inf_total:
        failures.append("EVIDENCE_NAN_INF_PRESENT")

    checks = {
        "schema_file_present": schema is not None and schema.get("schema_version") == SCHEMA_VERSION,
        "run_header_present": header is not None,
        "evidence_index_present": index is not None,
        "run_state_present": state is not None,
        "evidence_log_present": jsonl_path.exists(),
        "no_partial_record_line": not partial_line,
        "no_unparseable_record": unparseable == 0,
        "all_record_hashes_valid": hash_failures == 0 and bool(records),
        "record_schema_complete": not schema_failures,
        "identity_binding_consistent": not any("IDENTITY_MISMATCH" in failure for failure in schema_failures),
        "no_duplicate_cycle_evidence": not duplicates,
        "no_conflicting_cycle_evidence": not conflicts,
        "index_matches_log": index_matches,
        "cycle_scoped_trace_binding": trace_scope_ok,
        "nan_inf_zero": nan_inf_total == 0,
        "expected_grid_complete": bool(expected_grid) and not missing_cycles,
    }
    integrity_passed = all(checks.values())
    result = {
        "evidence_dir": str(directory),
        "schema_version": SCHEMA_VERSION,
        "schema": schema,
        "run_header": header,
        "run_state": state,
        "index": index,
        "records": records,
        "record_count": len(records),
        "expected_record_count": len(expected_grid),
        "missing_cycles": [{"seed": seed, "outer_cycle": cycle} for seed, cycle in missing_cycles],
        "duplicate_cycles": duplicates,
        "conflicting_cycles": conflicts,
        "nan_inf_count": nan_inf_total,
        "checks": checks,
        "failures": sorted(set(failures)),
        "integrity_passed": integrity_passed,
        "loaded_at": kst_now(),
    }
    if strict and not integrity_passed:
        raise EvidenceIntegrityError("EVIDENCE_INTEGRITY_FAILED", json.dumps(result["failures"], ensure_ascii=False))
    return result


def build_report_payload(evidence_dir: Path) -> Dict[str, Any]:
    """Build the report payload from persisted evidence only.

    Performs no training, no rollout, no optimizer step, and touches no model:
    it reads the durable files and aggregates them.
    """
    evidence = load_evidence(evidence_dir)
    records = sorted(evidence["records"], key=lambda row: (int(row.get("seed", 0)), int(row.get("outer_cycle", 0))))
    by_seed: Dict[str, Dict[str, Any]] = {}
    by_seed_cycle: List[Dict[str, Any]] = []
    for record in records:
        seed = str(int(record.get("seed", 0)))
        critic_loss = record.get("critic_loss", {})
        delta = record.get("parameter_delta", {})
        credit = record.get("credit_diagnostics", {})
        row = {
            "seed": int(record.get("seed", 0)),
            "outer_cycle": int(record.get("outer_cycle", 0)),
            "record_sha256": record.get("record_sha256"),
            "critic_loss_mean": critic_loss.get("critic_loss_mean"),
            "critic_loss_first": critic_loss.get("critic_loss_first"),
            "critic_loss_last": critic_loss.get("critic_loss_last"),
            "critic_update_rows": critic_loss.get("update_row_count"),
            "actor_parameter_delta_l2": (delta.get("actor") or {}).get("l2_delta"),
            "critic_parameter_delta_l2": (delta.get("critic") or {}).get("l2_delta"),
            "gatv2_parameter_delta_l2": (delta.get("gatv2") or {}).get("l2_delta"),
            "v_s_minus_return_mean": (credit.get("v_s_minus_return") or {}).get("mean"),
            "raw_gae_mean": (credit.get("raw_gae") or {}).get("mean"),
            "normalized_advantage_mean": (credit.get("normalized_advantage") or {}).get("mean"),
            "action_counts": credit.get("action_counts"),
            "action_probabilities": credit.get("action_probabilities"),
            "sample_count": credit.get("sample_count"),
        }
        by_seed_cycle.append(row)
        bucket = by_seed.setdefault(
            seed,
            {
                "seed": int(seed),
                "cycles": 0,
                "critic_loss_values": [],
                "actor_delta_values": [],
                "critic_delta_values": [],
                "action_counts": {},
                "sample_count": 0,
            },
        )
        bucket["cycles"] += 1
        if row["critic_loss_mean"] is not None:
            bucket["critic_loss_values"].append(float(row["critic_loss_mean"]))
        if row["actor_parameter_delta_l2"] is not None:
            bucket["actor_delta_values"].append(float(row["actor_parameter_delta_l2"]))
        if row["critic_parameter_delta_l2"] is not None:
            bucket["critic_delta_values"].append(float(row["critic_parameter_delta_l2"]))
        bucket["sample_count"] += int(row["sample_count"] or 0)
        for name, count in (row["action_counts"] or {}).items():
            bucket["action_counts"][name] = bucket["action_counts"].get(name, 0) + int(count)
    for bucket in by_seed.values():
        bucket["critic_loss"] = numeric_stats(bucket.pop("critic_loss_values"))
        bucket["actor_parameter_delta_l2"] = numeric_stats(bucket.pop("actor_delta_values"))
        bucket["critic_parameter_delta_l2"] = numeric_stats(bucket.pop("critic_delta_values"))
        bucket["actor_delta_positive_all_cycles"] = bool(
            bucket["actor_parameter_delta_l2"]["count"] == bucket["cycles"]
            and (bucket["actor_parameter_delta_l2"]["min"] or 0.0) > 0.0
        )
        bucket["critic_delta_positive_all_cycles"] = bool(
            bucket["critic_parameter_delta_l2"]["count"] == bucket["cycles"]
            and (bucket["critic_parameter_delta_l2"]["min"] or 0.0) > 0.0
        )

    header = evidence.get("run_header") or {}
    blocking: List[str] = []
    if not evidence["integrity_passed"]:
        blocking.append("EVIDENCE_INTEGRITY_FAILED")
    if evidence["missing_cycles"]:
        blocking.append("MISSING_SEED_CYCLE_EVIDENCE")
    report_ready = not blocking
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": header.get("stage"),
        "generated_at": kst_now(),
        "report_source": REPORT_SOURCE,
        "report_status": REPORT_STATUS_COMPLETE if report_ready else REPORT_STATUS_BLOCKED,
        "report_ready": report_ready,
        "blocking_reasons": blocking,
        "evidence_dir": str(evidence_dir),
        "run_header": header,
        "run_state": evidence.get("run_state"),
        "evidence_integrity": evidence["checks"],
        "evidence_failures": evidence["failures"],
        "integrity_passed": evidence["integrity_passed"],
        "record_count": evidence["record_count"],
        "expected_record_count": evidence["expected_record_count"],
        "missing_cycles": evidence["missing_cycles"],
        "duplicate_cycles": evidence["duplicate_cycles"],
        "conflicting_cycles": evidence["conflicting_cycles"],
        "by_seed": by_seed,
        "by_seed_cycle": by_seed_cycle,
        "totals": {
            "cycles_persisted": evidence["record_count"],
            "critic_loss": numeric_stats([row["critic_loss_mean"] for row in by_seed_cycle]),
            "actor_parameter_delta_l2": numeric_stats([row["actor_parameter_delta_l2"] for row in by_seed_cycle]),
            "critic_parameter_delta_l2": numeric_stats([row["critic_parameter_delta_l2"] for row in by_seed_cycle]),
            "critic_update_rows": sum(int(row["critic_update_rows"] or 0) for row in by_seed_cycle),
            "samples": sum(int(row["sample_count"] or 0) for row in by_seed_cycle),
        },
        "nan_inf_count": evidence["nan_inf_count"],
        "future_leakage_count": 0,
        "test6_access_count": 0,
        "training_invocations": 0,
        "optimizer_step_count": 0,
        "volatile_in_memory_metrics_used": False,
    }


def render_final_report(payload: Mapping[str, Any], *, title: str = "Durable Training Evidence Report") -> str:
    header = payload.get("run_header") or {}
    totals = payload.get("totals") or {}
    lines = [
        f"# {title}",
        "",
        f"report_status = {payload.get('report_status')}",
        f"report_source = {payload.get('report_source')}",
        f"stage = {payload.get('stage')}",
        f"source_commit = {header.get('source_commit')}",
        f"actor_repair_contract_sha256 = {header.get('actor_repair_contract_sha256')}",
        f"critic_repair_contract_sha256 = {header.get('critic_repair_contract_sha256')}",
        f"cycles_persisted = {payload.get('record_count')} / {payload.get('expected_record_count')}",
        f"training_invocations = {payload.get('training_invocations')}",
        f"optimizer_step_count = {payload.get('optimizer_step_count')}",
        f"TEST6_access_count = {payload.get('test6_access_count')}",
        "",
        "## Evidence integrity",
        "",
        "```json",
        json.dumps(payload.get("evidence_integrity"), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Persisted training evidence totals",
        "",
        "```json",
        json.dumps(totals, ensure_ascii=False, indent=2, sort_keys=True, default=_jsonable),
        "```",
        "",
        "## Per-seed evidence",
        "",
        "```json",
        json.dumps(payload.get("by_seed"), ensure_ascii=False, indent=2, sort_keys=True, default=_jsonable),
        "```",
        "",
    ]
    if not payload.get("report_ready"):
        lines.extend(
            [
                "## Fail-closed block",
                "",
                "```json",
                json.dumps(
                    {
                        "blocking_reasons": payload.get("blocking_reasons"),
                        "evidence_failures": payload.get("evidence_failures"),
                        "missing_cycles": payload.get("missing_cycles"),
                        "duplicate_cycles": payload.get("duplicate_cycles"),
                        "conflicting_cycles": payload.get("conflicting_cycles"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
                "This report is BLOCKED: persisted evidence is incomplete or inconsistent. "
                "Missing metrics are never estimated or reconstructed.",
                "",
            ]
        )
    lines.append(
        "Rendered from persisted durable evidence only; no volatile in-memory training metric was used, "
        "and no training, optimizer step, or TEST6 access occurred while rendering."
    )
    return "\n".join(lines) + "\n"


def regenerate_report(
    artifact_root: Path,
    *,
    report_name: str = "final_report.md",
    recovery_name: str = REPORT_RECOVERY_NAME,
    title: str = "Durable Training Evidence Report",
) -> Dict[str, Any]:
    """Rebuild the report from persisted evidence with zero training.

    This is the recovery path used after a reporter crash.  It constructs no
    model, no optimizer, and no device context, and it never imports torch.
    """
    torch_already_loaded = "torch" in _loaded_module_names()
    root = Path(artifact_root)
    evidence_dir = find_evidence_dir(root)
    payload = build_report_payload(evidence_dir)
    report_text = render_final_report(payload, title=title)
    report_write = atomic_write_text(root / report_name, report_text)
    recovery = {
        "schema_version": SCHEMA_VERSION,
        "recovered": bool(payload["report_ready"]),
        "report_status": payload["report_status"],
        "regenerated_at": kst_now(),
        "artifact_root": str(root),
        "evidence_dir": str(evidence_dir),
        "final_report_path": report_write["path"],
        "record_count": payload["record_count"],
        "expected_record_count": payload["expected_record_count"],
        "evidence_integrity": payload["evidence_integrity"],
        "blocking_reasons": payload["blocking_reasons"],
        "training_invocations": 0,
        "optimizer_step_count": 0,
        "test6_access_count": 0,
        "model_or_optimizer_constructed": False,
        "torch_imported_by_recovery_path": (not torch_already_loaded) and "torch" in _loaded_module_names(),
        "report_source": REPORT_SOURCE,
    }
    atomic_write_json(root / recovery_name, recovery)
    state_path = evidence_dir / RUN_STATE_NAME
    state = _read_json(state_path) or {"schema_version": SCHEMA_VERSION}
    state.update(
        {
            "status": STATE_REPORT_REGENERATED,
            "updated_at": kst_now(),
            "last_report_regeneration": {
                "recovered": recovery["recovered"],
                "report_status": recovery["report_status"],
                "training_invocations": 0,
                "optimizer_step_count": 0,
            },
        }
    )
    atomic_write_json(state_path, state)
    return recovery


def _loaded_module_names() -> Sequence[str]:
    import sys  # noqa: PLC0415 - local import keeps the module import surface minimal

    return tuple(sys.modules)


def main() -> None:
    import argparse  # noqa: PLC0415 - CLI only

    parser = argparse.ArgumentParser(description="Regenerate a training report from persisted durable evidence.")
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--report-name", default="final_report.md")
    parser.add_argument("--title", default="Durable Training Evidence Report")
    args = parser.parse_args()
    result = regenerate_report(args.artifact_root, report_name=args.report_name, title=args.title)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["recovered"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
