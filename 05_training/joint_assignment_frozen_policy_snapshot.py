"""Lossless, fail-closed frozen-policy replay evidence for Joint Assignment.

This is evidence infrastructure only.  It deliberately knows nothing about
Local Search, Zero-Loss evaluation, simulator state, rewards, PPO, or an
optimizer.  A caller hands it the *already-finalized* actor inputs immediately
before the Joint Actor forward call; replay consumes only those persisted
inputs and a specifically bound checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

import torch


SNAPSHOT_SCHEMA_VERSION = "LS3_BT7_FROZEN_POLICY_SNAPSHOT_V1"
COLLECTION_SCHEMA_VERSION = "LS3_BT7_FROZEN_POLICY_SNAPSHOT_COLLECTION_V1"
FEATURE_CONTRACT_ID = "LS3_JOINT_ACTOR_INPUT_FEATURE_CONTRACT_V1"
TENSOR_FIELDS: Tuple[str, ...] = (
    "global_feats", "demand_feats", "agent_feats", "agent_mask",
    "candidate_feats", "pair_agent_index", "safe_mask",
)


class FrozenPolicySnapshotError(RuntimeError):
    """A replay evidence mismatch must stop before any policy forward."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class FrozenReplayOutput:
    """The policy output over the historical, persisted valid support only."""

    pair_logits: torch.Tensor
    no_assign_logit: torch.Tensor
    probabilities: torch.Tensor
    argmax_index: int
    selected_is_no_assign: bool
    selected_pair: Tuple[str, str] | None
    snapshot_digest: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def actor_config(*, global_dim: int, demand_dim: int, agent_dim: int,
                 candidate_dim: int, hidden: int = 128, heads: int = 4,
                 actor_module_sha256: str, actor_head_id: str,
                 actor_head_version: str) -> Dict[str, Any]:
    """The architecture/config binding needed to instantiate a replay actor."""
    return {
        "actor_class": "MultiAgentCandidateAssignmentHead",
        "actor_module_sha256": str(actor_module_sha256),
        "actor_head_id": str(actor_head_id),
        "actor_head_version": str(actor_head_version),
        "global_dim": int(global_dim), "demand_dim": int(demand_dim),
        "agent_dim": int(agent_dim), "candidate_dim": int(candidate_dim),
        "hidden": int(hidden), "heads": int(heads),
    }


def actor_config_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(config))


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise FrozenPolicySnapshotError("SNAPSHOT_TENSOR_NOT_A_TENSOR")
    # clone is required even for a CPU input: snapshots must not alias mutable
    # rollout tensors.  It preserves dtype and logical tensor shape exactly.
    return tensor.detach().to(device="cpu").clone(memory_format=torch.contiguous_format).contiguous()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = _cpu_clone(tensor)
    # uint8 exposes the exact logical tensor representation after the canonical
    # contiguous clone; no text conversion, rounding, or re-normalization occurs.
    return value.view(torch.uint8).numpy().tobytes()


def _tensor_descriptor(tensor: torch.Tensor) -> Dict[str, Any]:
    value = _cpu_clone(tensor)
    return {"dtype": str(value.dtype), "shape": list(value.shape), "numel": int(value.numel())}


def _tensor_digest(metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"LS3_FROZEN_POLICY_SNAPSHOT_DIGEST_V1\0")
    digest.update(canonical_json(dict(metadata)))
    for name in TENSOR_FIELDS:
        if name not in tensors:
            raise FrozenPolicySnapshotError("SNAPSHOT_REQUIRED_TENSOR_MISSING", name)
        tensor = tensors[name]
        descriptor = {"name": name, **_tensor_descriptor(tensor)}
        raw = _tensor_bytes(tensor)
        digest.update(canonical_json(descriptor))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise FrozenPolicySnapshotError(code, detail)


def _normal_pair_ids(candidate_ids: Sequence[Sequence[str] | Mapping[str, str]]) -> list[Dict[str, str]]:
    normalized: list[Dict[str, str]] = []
    for row in candidate_ids:
        if isinstance(row, Mapping):
            agent_id, candidate_id = row.get("agent_id"), row.get("candidate_id")
        else:
            _require(len(row) == 2, "SNAPSHOT_CANDIDATE_ID_INVALID")
            agent_id, candidate_id = row
        _require(bool(agent_id) and bool(candidate_id), "SNAPSHOT_CANDIDATE_ID_INVALID")
        normalized.append({"agent_id": str(agent_id), "candidate_id": str(candidate_id)})
    pairs = [(row["agent_id"], row["candidate_id"]) for row in normalized]
    _require(len(pairs) == len(set(pairs)), "SNAPSHOT_DUPLICATE_CANDIDATE_IDENTITY")
    return normalized


def _validate_tensors(tensors: Mapping[str, torch.Tensor], *, agent_ids: Sequence[str],
                      candidate_ids: Sequence[Mapping[str, str]], selectable_pair_count: int) -> None:
    for name in TENSOR_FIELDS:
        _require(name in tensors, "SNAPSHOT_REQUIRED_TENSOR_MISSING", name)
        _require(isinstance(tensors[name], torch.Tensor), "SNAPSHOT_TENSOR_NOT_A_TENSOR", name)
    glob, demand = tensors["global_feats"], tensors["demand_feats"]
    agents, agent_mask = tensors["agent_feats"], tensors["agent_mask"]
    candidates, pair_index, safe_mask = (tensors["candidate_feats"], tensors["pair_agent_index"],
                                          tensors["safe_mask"])
    _require(glob.ndim == 2 and demand.ndim == 2 and glob.shape[0] == demand.shape[0] == 1,
             "SNAPSHOT_GLOBAL_OR_DEMAND_SHAPE_INVALID")
    _require(agents.ndim == 3 and agent_mask.ndim == 2 and agents.shape[:2] == agent_mask.shape,
             "SNAPSHOT_AGENT_SHAPE_INVALID")
    _require(candidates.ndim == 3 and candidates.shape[0] == 1, "SNAPSHOT_CANDIDATE_SHAPE_INVALID")
    _require(pair_index.ndim == 2 and safe_mask.ndim == 2 and pair_index.shape == safe_mask.shape
             and pair_index.shape[:2] == candidates.shape[:2], "SNAPSHOT_PAIR_MAPPING_SHAPE_INVALID")
    _require(agent_mask.dtype == torch.bool, "SNAPSHOT_AGENT_MASK_DTYPE_INVALID")
    _require(safe_mask.dtype == torch.bool, "SNAPSHOT_SAFE_MASK_DTYPE_INVALID")
    _require(pair_index.dtype == torch.long, "SNAPSHOT_PAIR_AGENT_INDEX_DTYPE_INVALID")
    _require(agents.dtype.is_floating_point and candidates.dtype.is_floating_point
             and glob.dtype.is_floating_point and demand.dtype.is_floating_point,
             "SNAPSHOT_FEATURE_DTYPE_INVALID")
    _require(agents.shape[1] == len(agent_ids) and bool(agent_mask[0].all().item()),
             "SNAPSHOT_AGENT_ID_OR_MASK_MISMATCH")
    _require(len(set(str(value) for value in agent_ids)) == len(agent_ids), "SNAPSHOT_DUPLICATE_AGENT_ID")
    _require(selectable_pair_count == len(candidate_ids), "SNAPSHOT_SELECTABLE_PAIR_COUNT_MISMATCH")
    _require(0 <= selectable_pair_count <= candidates.shape[1], "SNAPSHOT_SELECTABLE_PAIR_COUNT_INVALID")
    # The actor path intentionally crops to the first selectable pairs.  Any
    # later tensor slots are immutable storage-only masked padding, not actions.
    _require(bool(safe_mask[0, :selectable_pair_count].all().item()), "SNAPSHOT_SELECTABLE_MASK_INVALID")
    _require(not bool(safe_mask[0, selectable_pair_count:].any().item()), "SNAPSHOT_PADDING_MASK_INVALID")
    for slot, pair in enumerate(candidate_ids):
        _require(pair["agent_id"] in agent_ids, "SNAPSHOT_CANDIDATE_AGENT_UNKNOWN", str(slot))
        expected = list(agent_ids).index(pair["agent_id"])
        _require(int(pair_index[0, slot].item()) == expected, "SNAPSHOT_PAIR_AGENT_MAPPING_MISMATCH", str(slot))


def _validate_actor_dimensions(tensors: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> None:
    for key in ("global_dim", "demand_dim", "agent_dim", "candidate_dim"):
        _require(key in config, "SNAPSHOT_ACTOR_CONFIG_INCOMPLETE", key)
    _require(int(tensors["global_feats"].shape[-1]) == int(config["global_dim"]),
             "SNAPSHOT_GLOBAL_FEATURE_DIMENSION_MISMATCH")
    _require(int(tensors["demand_feats"].shape[-1]) == int(config["demand_dim"]),
             "SNAPSHOT_DEMAND_FEATURE_DIMENSION_MISMATCH")
    _require(int(tensors["agent_feats"].shape[-1]) == int(config["agent_dim"]),
             "SNAPSHOT_AGENT_FEATURE_DIMENSION_MISMATCH")
    _require(int(tensors["candidate_feats"].shape[-1]) == int(config["candidate_dim"]),
             "SNAPSHOT_CANDIDATE_FEATURE_DIMENSION_MISMATCH")


def capture_actor_input(*, decision_id: str, window_id: str, seed: int, decision_index: int,
                        time_band: str, actor_inputs: Mapping[str, torch.Tensor],
                        agent_ids: Sequence[str], candidate_ids: Sequence[Sequence[str] | Mapping[str, str]],
                        selectable_pair_count: int, candidate_support_digest: str,
                        no_assign_option: str, actor_config_value: Mapping[str, Any],
                        feature_contract: Mapping[str, Any], frozen_authority_hashes: Mapping[str, str],
                        source_commit: str, captured_device: str) -> Dict[str, Any]:
    """Copy the exact post-filter, pre-forward actor boundary without mutation."""
    _require(bool(decision_id) and bool(window_id) and bool(time_band), "SNAPSHOT_DECISION_IDENTITY_INVALID")
    _require(bool(candidate_support_digest), "SNAPSHOT_SUPPORT_DIGEST_MISSING")
    _require(no_assign_option == "NO_ASSIGN_KEEP_CURRENT_PLANS", "SNAPSHOT_NO_ASSIGN_MISSING")
    normalized_pairs = _normal_pair_ids(candidate_ids)
    tensors = {name: _cpu_clone(actor_inputs[name]) for name in TENSOR_FIELDS
               if name in actor_inputs}
    _validate_tensors(tensors, agent_ids=agent_ids, candidate_ids=normalized_pairs,
                      selectable_pair_count=int(selectable_pair_count))
    _validate_actor_dimensions(tensors, actor_config_value)
    metadata = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "decision_id": str(decision_id), "window_id": str(window_id), "seed": int(seed),
        "decision_index": int(decision_index), "time_band": str(time_band),
        "agent_ids": [str(value) for value in agent_ids],
        "candidate_ids": normalized_pairs,
        "candidate_order": normalized_pairs,
        "selectable_pair_count": int(selectable_pair_count),
        "actor_tensor_pair_count": int(tensors["candidate_feats"].shape[1]),
        "storage_padding_pair_count": int(tensors["candidate_feats"].shape[1]) - int(selectable_pair_count),
        "no_assign_option": no_assign_option,
        "no_assign_index": int(selectable_pair_count),
        "candidate_support_digest": str(candidate_support_digest),
        "feature_contract": dict(feature_contract),
        "feature_contract_id": str(feature_contract.get("feature_contract_id", FEATURE_CONTRACT_ID)),
        "actor_config": dict(actor_config_value),
        "actor_config_sha256": actor_config_sha256(actor_config_value),
        "frozen_authority_hashes": dict(sorted(frozen_authority_hashes.items())),
        "source_commit": str(source_commit),
        "captured_device": str(captured_device),
        "tensor_descriptors": {name: _tensor_descriptor(tensors[name]) for name in TENSOR_FIELDS},
    }
    digest = _tensor_digest(metadata, tensors)
    return {"metadata": metadata, "tensors": tensors, "snapshot_digest": digest}


def _validate_payload(payload: Mapping[str, Any]) -> None:
    _require(set(payload) == {"metadata", "tensors", "snapshot_digest"}, "SNAPSHOT_PAYLOAD_KEYS_INVALID")
    metadata, tensors = payload["metadata"], payload["tensors"]
    _require(isinstance(metadata, Mapping) and isinstance(tensors, Mapping), "SNAPSHOT_PAYLOAD_TYPE_INVALID")
    _require(metadata.get("snapshot_schema_version") == SNAPSHOT_SCHEMA_VERSION, "SNAPSHOT_SCHEMA_VERSION_MISMATCH")
    _require(metadata.get("no_assign_option") == "NO_ASSIGN_KEEP_CURRENT_PLANS", "SNAPSHOT_NO_ASSIGN_MISSING")
    _require(metadata.get("no_assign_index") == metadata.get("selectable_pair_count"), "SNAPSHOT_NO_ASSIGN_INDEX_INVALID")
    candidate_ids = _normal_pair_ids(metadata.get("candidate_ids", []))
    _require(candidate_ids == metadata.get("candidate_order"), "SNAPSHOT_CANDIDATE_ORDER_METADATA_MISMATCH")
    _validate_tensors(tensors, agent_ids=metadata.get("agent_ids", []), candidate_ids=candidate_ids,
                      selectable_pair_count=int(metadata.get("selectable_pair_count", -1)))
    _validate_actor_dimensions(tensors, metadata.get("actor_config", {}))
    expected = _tensor_digest(metadata, tensors)
    _require(payload.get("snapshot_digest") == expected, "SNAPSHOT_DIGEST_MISMATCH")


def write_snapshot(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Write a device-neutral tensor binary and a readable, signed manifest."""
    _validate_payload(payload)
    path.mkdir(parents=True, exist_ok=False)
    binary = path / "actor_input_snapshot.pt"
    # torch's tensor archive preserves dtype and shape; tensors were copied to
    # CPU first, so the stored format has no device-local MPS dependency.
    torch.save({"tensors": payload["tensors"]}, binary)
    manifest = {"metadata": payload["metadata"], "snapshot_digest": payload["snapshot_digest"],
                "tensor_binary": binary.name, "tensor_binary_sha256": sha256_file(binary),
                "serialization_format": "torch.save binary tensor archive (CPU tensors)",
                "device_neutral": True, "lossy_transformations": []}
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (path / "snapshot_manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def load_snapshot(path: Path) -> Dict[str, Any]:
    manifest_path = path / "snapshot_manifest.json"
    _require(manifest_path.is_file(), "SNAPSHOT_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplied_manifest_digest = manifest.pop("manifest_sha256", None)
    _require(supplied_manifest_digest == canonical_sha256(manifest), "SNAPSHOT_MANIFEST_DIGEST_MISMATCH")
    binary = path / str(manifest.get("tensor_binary", ""))
    _require(binary.is_file(), "SNAPSHOT_TENSOR_BINARY_MISSING")
    _require(manifest.get("tensor_binary_sha256") == sha256_file(binary), "SNAPSHOT_TENSOR_BINARY_DIGEST_MISMATCH")
    stored = torch.load(binary, map_location="cpu", weights_only=False)
    _require(isinstance(stored, Mapping) and isinstance(stored.get("tensors"), Mapping), "SNAPSHOT_TENSOR_BINARY_INVALID")
    payload = {"metadata": manifest.get("metadata"), "tensors": stored["tensors"],
               "snapshot_digest": manifest.get("snapshot_digest")}
    _validate_payload(payload)
    return payload


class SnapshotCollectionWriter:
    """Append-only per-decision snapshots, finalized against one checkpoint."""

    def __init__(self, *, root: Path, collection_binding: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.collection_binding = dict(collection_binding)
        _require(self.collection_binding.get("snapshot_schema_version") == SNAPSHOT_SCHEMA_VERSION,
                 "SNAPSHOT_COLLECTION_SCHEMA_BINDING_INVALID")
        _require(bool(self.collection_binding.get("actor_config_sha256")), "SNAPSHOT_COLLECTION_ACTOR_BINDING_MISSING")
        self.entries: list[Dict[str, Any]] = []
        self._identities: set[Tuple[str, int, int]] = set()
        self._finalized = False

    def capture(self, **kwargs: Any) -> Dict[str, Any]:
        _require(not self._finalized, "SNAPSHOT_COLLECTION_ALREADY_FINALIZED")
        payload = capture_actor_input(**kwargs)
        metadata = payload["metadata"]
        identity = (str(metadata["decision_id"]), int(metadata["seed"]), int(metadata["decision_index"]))
        _require(identity not in self._identities, "SNAPSHOT_DUPLICATE_DECISION_IDENTITY", repr(identity))
        _require(metadata["actor_config_sha256"] == self.collection_binding["actor_config_sha256"],
                 "SNAPSHOT_COLLECTION_ACTOR_CONFIG_MISMATCH")
        _require(metadata["feature_contract"] == self.collection_binding["feature_contract"],
                 "SNAPSHOT_COLLECTION_FEATURE_CONTRACT_MISMATCH")
        _require(metadata["frozen_authority_hashes"] == self.collection_binding["frozen_authority_hashes"],
                 "SNAPSHOT_COLLECTION_FROZEN_AUTHORITY_MISMATCH")
        directory = self.root / "snapshots" / payload["snapshot_digest"]
        manifest = write_snapshot(directory, payload)
        self._identities.add(identity)
        self.entries.append({"decision_identity": {"decision_id": identity[0], "seed": identity[1],
                                                     "decision_index": identity[2]},
                             "snapshot_digest": payload["snapshot_digest"],
                             "relative_path": directory.relative_to(self.root).as_posix(),
                             "snapshot_manifest_sha256": manifest["manifest_sha256"]})
        return {"snapshot_digest": payload["snapshot_digest"], "relative_path": directory.relative_to(self.root).as_posix()}

    def finalize(self, *, checkpoint_path: Path) -> Dict[str, Any]:
        _require(not self._finalized, "SNAPSHOT_COLLECTION_ALREADY_FINALIZED")
        checkpoint_path = Path(checkpoint_path)
        _require(checkpoint_path.is_file(), "SNAPSHOT_CHECKPOINT_MISSING")
        _require(bool(self.entries), "SNAPSHOT_COLLECTION_EMPTY")
        collection = {"collection_schema_version": COLLECTION_SCHEMA_VERSION,
                      **self.collection_binding,
                      "checkpoint_file": checkpoint_path.name,
                      "checkpoint_sha256": sha256_file(checkpoint_path),
                      "entries": list(self.entries),
                      "snapshot_count": len(self.entries),
                      "replay_rule": "snapshot + bound checkpoint only; no Local Search, Zero-Loss, feature recomputation, mask reconstruction, or sorting"}
        collection["collection_digest"] = canonical_sha256(collection)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "collection_manifest.json").write_bytes(canonical_json(collection) + b"\n")
        self._finalized = True
        return collection


def load_collection_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = payload.pop("collection_digest", None)
    _require(supplied == canonical_sha256(payload), "SNAPSHOT_COLLECTION_DIGEST_MISMATCH")
    _require(payload.get("collection_schema_version") == COLLECTION_SCHEMA_VERSION,
             "SNAPSHOT_COLLECTION_SCHEMA_VERSION_MISMATCH")
    _require(payload.get("snapshot_schema_version") == SNAPSHOT_SCHEMA_VERSION,
             "SNAPSHOT_COLLECTION_SNAPSHOT_SCHEMA_MISMATCH")
    entries = payload.get("entries", [])
    identities = [(row["decision_identity"]["decision_id"], row["decision_identity"]["seed"],
                   row["decision_identity"]["decision_index"]) for row in entries]
    _require(len(identities) == len(set(identities)), "SNAPSHOT_DUPLICATE_DECISION_IDENTITY")
    payload["collection_digest"] = supplied
    return payload


def replay_snapshot(*, snapshot_root: Path, collection_manifest_path: Path, checkpoint_path: Path,
                    actor: torch.nn.Module, actor_config_value: Mapping[str, Any],
                    feature_contract: Mapping[str, Any], frozen_authority_hashes: Mapping[str, str],
                    expected_device: str) -> FrozenReplayOutput:
    """Pure replay: persistently captured input plus its bound final checkpoint."""
    collection = load_collection_manifest(collection_manifest_path)
    _require(collection.get("actor_config_sha256") == actor_config_sha256(actor_config_value),
             "SNAPSHOT_REPLAY_ACTOR_CONFIG_MISMATCH")
    _require(collection.get("feature_contract") == dict(feature_contract),
             "SNAPSHOT_REPLAY_FEATURE_CONTRACT_MISMATCH")
    _require(collection.get("frozen_authority_hashes") == dict(sorted(frozen_authority_hashes.items())),
             "SNAPSHOT_REPLAY_FROZEN_AUTHORITY_MISMATCH")
    _require(collection.get("captured_device") == str(expected_device), "SNAPSHOT_REPLAY_DEVICE_MISMATCH")
    checkpoint_path = Path(checkpoint_path)
    _require(checkpoint_path.is_file(), "SNAPSHOT_REPLAY_CHECKPOINT_MISSING")
    _require(collection.get("checkpoint_sha256") == sha256_file(checkpoint_path),
             "SNAPSHOT_REPLAY_CHECKPOINT_MISMATCH")
    payload = load_snapshot(snapshot_root)
    _require(payload["metadata"].get("actor_config_sha256") == collection.get("actor_config_sha256"),
             "SNAPSHOT_REPLAY_SNAPSHOT_ACTOR_BINDING_MISMATCH")
    _require(payload["metadata"].get("feature_contract") == collection.get("feature_contract"),
             "SNAPSHOT_REPLAY_SNAPSHOT_FEATURE_CONTRACT_MISMATCH")
    _require(payload["metadata"].get("frozen_authority_hashes") == collection.get("frozen_authority_hashes"),
             "SNAPSHOT_REPLAY_SNAPSHOT_FROZEN_AUTHORITY_MISMATCH")
    _require(any(row.get("snapshot_digest") == payload["snapshot_digest"] for row in collection.get("entries", [])),
             "SNAPSHOT_REPLAY_SNAPSHOT_NOT_IN_COLLECTION")
    checkpoint = torch.load(checkpoint_path, map_location=expected_device, weights_only=False)
    _require(isinstance(checkpoint, Mapping) and "actor" in checkpoint, "SNAPSHOT_REPLAY_ACTOR_STATE_MISSING")
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.to(expected_device)
    actor.eval()
    tensors = {name: value.to(expected_device) for name, value in payload["tensors"].items()}
    metadata = payload["metadata"]
    selectable = int(metadata["selectable_pair_count"])
    with torch.no_grad():
        pair_logits, no_assign_logit = actor(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"],
                                             agent_feats=tensors["agent_feats"], agent_mask=tensors["agent_mask"],
                                             candidate_feats=tensors["candidate_feats"], pair_agent_index=tensors["pair_agent_index"],
                                             safe_mask=tensors["safe_mask"])
        pair_logits = pair_logits[:, :selectable]
        safe_mask = tensors["safe_mask"][:, :selectable]
        full = torch.cat([pair_logits, no_assign_logit], dim=-1)
        probabilities = torch.softmax(full, dim=-1)
        argmax_index = int(torch.argmax(probabilities[0]).item())
    candidate_ids = metadata["candidate_ids"]
    return FrozenReplayOutput(pair_logits=pair_logits.detach(), no_assign_logit=no_assign_logit.detach(),
                              probabilities=probabilities.detach(), argmax_index=argmax_index,
                              selected_is_no_assign=argmax_index == selectable,
                              selected_pair=None if argmax_index == selectable else
                              (candidate_ids[argmax_index]["agent_id"], candidate_ids[argmax_index]["candidate_id"]),
                              snapshot_digest=str(payload["snapshot_digest"]))
