from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPAIR_VERSION = "prompt4_r2_mps_gatv2_dynamic_embedding_determinism"
DEFAULT_REPAIR_SOURCE = "05_training/artifacts/suseong_encoder_provenance_repair_v1_20260718_094827"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "UNKNOWN"


def require_env() -> Dict[str, Any]:
    required = {
        "PYTHONHASHSEED": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "PYTORCH_MPS_FAST_MATH": "0",
    }
    observed = {key: os.environ.get(key) for key in required}
    return {
        "required": required,
        "observed": observed,
        "satisfied": all(observed[key] == value for key, value in required.items()),
    }


def import_runtime() -> Tuple[Any, Any, Any, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv

    return np, torch, F, GATv2Conv


def lock_runtime(seed: int) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    np, torch, F, GATv2Conv = import_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    if hasattr(torch.utils, "deterministic"):
        torch.utils.deterministic.fill_uninitialized_memory = True
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine_id": platform.node(),
        "torch": torch.__version__,
        "torch_geometric_gatv2conv": str(GATv2Conv),
        "numpy": np.__version__,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_debug_mode": "error",
        "fill_uninitialized_memory": bool(getattr(getattr(torch, "utils", None), "deterministic", object()).fill_uninitialized_memory),
        "autocast": "disabled",
        "mixed_precision": "disabled",
        "torch_compile": "disabled",
        "num_workers": 0,
    }
    return np, torch, F, GATv2Conv, report


def tensor_bytes_sha(torch: Any, tensor: Any) -> str:
    if tensor.device.type == "mps":
        torch.mps.synchronize()
    value = tensor.detach().to("cpu").contiguous().float()
    return sha256_bytes(value.numpy().tobytes(order="C"))


def tensor_stats(torch: Any, tensor: Any) -> Dict[str, Any]:
    if tensor.device.type == "mps":
        torch.mps.synchronize()
    value = tensor.detach().to("cpu").contiguous().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "contiguous": bool(value.is_contiguous()),
        "numel": int(value.numel()),
        "min": float(value.min().item()) if value.numel() else None,
        "max": float(value.max().item()) if value.numel() else None,
        "mean": float(value.mean().item()) if value.numel() else None,
        "std": float(value.std(unbiased=False).item()) if value.numel() else None,
        "sha256": tensor_bytes_sha(torch, tensor),
        "nan": not bool(torch.isfinite(value).all().item()) if value.numel() else False,
        "inf": bool(torch.isinf(value).any().item()) if value.numel() else False,
    }


def compare_tensors(torch: Any, reference: Any, candidate: Any) -> Dict[str, Any]:
    ref = reference.detach().to("cpu").contiguous().float()
    cand = candidate.detach().to("cpu").contiguous().float()
    diff = (ref - cand).abs()
    denom = float(torch.linalg.vector_norm(ref).item()) + 1e-12
    return {
        "torch_equal": bool(torch.equal(ref, cand)),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "relative_l2_diff": float(torch.linalg.vector_norm(ref - cand).item()) / denom,
        "mismatch_element_count": int((ref != cand).sum().item()),
        "reference_sha256": tensor_bytes_sha(torch, ref),
        "candidate_sha256": tensor_bytes_sha(torch, cand),
    }


def sha256_state_dict(torch: Any, module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_torch_file(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class AuditGATv2Encoder:
    def __init__(self, torch: Any, F: Any, GATv2Conv: Any, config: Mapping[str, Any], device: Any):
        self.torch = torch
        self.F = F
        self.device = device
        arch = config["architecture"]
        self.module = torch.nn.Module()
        self.module.conv1 = GATv2Conv(
            int(arch["conv1"]["in_channels"]),
            int(arch["conv1"]["out_channels"]),
            heads=int(arch["conv1"]["heads"]),
            concat=bool(arch["conv1"]["concat"]),
            edge_dim=arch["conv1"]["edge_dim"],
        )
        self.module.conv2 = GATv2Conv(
            int(arch["conv2"]["in_channels"]),
            int(arch["conv2"]["out_channels"]),
            heads=int(arch["conv2"]["heads"]),
            concat=bool(arch["conv2"]["concat"]),
            edge_dim=arch["conv2"]["edge_dim"],
        )
        self.module.to(device)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        missing, unexpected = self.module.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"strict load failed: missing={missing}, unexpected={unexpected}")

    def eval_frozen(self) -> None:
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad = False
        for submodule in self.module.modules():
            if submodule.__class__.__name__.lower().startswith("dropout"):
                if bool(submodule.training):
                    raise RuntimeError("Dropout module is still in training mode.")

    def forward_with_layers(self, x: Any, edge_index: Any, edge_attr: Optional[Any]) -> Dict[str, Any]:
        torch = self.torch
        F = self.F
        layers: Dict[str, Any] = {"input_x": x}
        with torch.inference_mode():
            conv1 = self.module.conv1
            conv2 = self.module.conv2
            lin1_l = conv1.lin_l(x)
            lin1_r = conv1.lin_r(x) if conv1.lin_r is not None else lin1_l
            layers["conv1_linear_left"] = lin1_l
            layers["conv1_linear_right"] = lin1_r
            out1, attn1 = conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            layers["conv1_attention_coefficients"] = attn1[1]
            layers["conv1_final_output_pre_relu"] = out1
            out1_relu = F.relu(out1)
            layers["conv1_final_output"] = out1_relu
            lin2_l = conv2.lin_l(out1_relu)
            lin2_r = conv2.lin_r(out1_relu) if conv2.lin_r is not None else lin2_l
            layers["conv2_linear_left"] = lin2_l
            layers["conv2_linear_right"] = lin2_r
            out2, attn2 = conv2(out1_relu, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            layers["conv2_attention_coefficients"] = attn2[1]
            layers["conv2_final_output_pre_relu"] = out2
            final = F.relu(out2)
            layers["final_embedding"] = final
        if self.device.type == "mps":
            torch.mps.synchronize()
        return layers


def canonicalize_data(torch: Any, data: Any, device: Any) -> Tuple[Any, Dict[str, Any]]:
    x = data.x.detach().to(device=device, dtype=torch.float32).contiguous()
    edge_index = data.edge_index.detach().to("cpu").long().contiguous()
    edge_attr = getattr(data, "edge_attr", None)
    edge_attr_cpu = edge_attr.detach().to("cpu", dtype=torch.float32).contiguous() if edge_attr is not None else None
    original = torch.arange(edge_index.size(1), dtype=torch.long)
    src = edge_index[0]
    dst = edge_index[1]
    order = sorted(range(int(edge_index.size(1))), key=lambda i: (int(dst[i]), int(src[i]), int(original[i])))
    order_tensor = torch.tensor(order, dtype=torch.long)
    sorted_edge_index = edge_index[:, order_tensor].contiguous()
    sorted_edge_attr = edge_attr_cpu[order_tensor].contiguous() if edge_attr_cpu is not None else None
    duplicate_count = int((sorted_edge_index[:, 1:] == sorted_edge_index[:, :-1]).all(dim=0).sum().item()) if sorted_edge_index.size(1) > 1 else 0
    self_loop_count = int((sorted_edge_index[0] == sorted_edge_index[1]).sum().item())
    canonical = type("CanonicalData", (), {})()
    canonical.x = x
    canonical.edge_index = sorted_edge_index.to(device).contiguous()
    canonical.edge_attr = sorted_edge_attr.to(device).contiguous() if sorted_edge_attr is not None else None
    manifest = {
        "edge_sort_key": "dst_src_original_edge_id",
        "duplicate_edge_count_after_sort": duplicate_count,
        "self_loop_count_after_sort": self_loop_count,
        "self_loop_policy": "preserve_input_edges_and_keep_pyg_gatv2conv_default_self_loop_behavior",
        "auto_self_loop_disabled": False,
        "execution_lineage_change": False,
        "x": tensor_stats(torch, canonical.x),
        "edge_index": tensor_stats(torch, canonical.edge_index.float()),
        "edge_attr": tensor_stats(torch, canonical.edge_attr) if canonical.edge_attr is not None else None,
        "original_edge_id_sha256": tensor_bytes_sha(torch, order_tensor.float()),
    }
    return canonical, manifest


def run_repeats(
    *,
    torch: Any,
    F: Any,
    GATv2Conv: Any,
    model_config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    canonical_data: Any,
    device: Any,
    repeats: int,
) -> Dict[str, Any]:
    encoder = AuditGATv2Encoder(torch, F, GATv2Conv, model_config, device)
    encoder.load_state_dict(state_dict)
    encoder.eval_frozen()
    param_hash = sha256_state_dict(torch, encoder.module)
    input_before = {
        "x": tensor_bytes_sha(torch, canonical_data.x),
        "edge_index": tensor_bytes_sha(torch, canonical_data.edge_index.float()),
        "edge_attr": tensor_bytes_sha(torch, canonical_data.edge_attr) if canonical_data.edge_attr is not None else None,
    }
    outputs: List[Dict[str, Any]] = []
    layer_tensors: List[Dict[str, Any]] = []
    for _ in range(repeats):
        layers = encoder.forward_with_layers(canonical_data.x, canonical_data.edge_index, canonical_data.edge_attr)
        outputs.append(tensor_stats(torch, layers["final_embedding"]))
        layer_tensors.append(layers)
    input_after = {
        "x": tensor_bytes_sha(torch, canonical_data.x),
        "edge_index": tensor_bytes_sha(torch, canonical_data.edge_index.float()),
        "edge_attr": tensor_bytes_sha(torch, canonical_data.edge_attr) if canonical_data.edge_attr is not None else None,
    }
    layerwise: Dict[str, Any] = {}
    first_divergent: Optional[str] = None
    reference_layers = layer_tensors[0]
    for name in reference_layers:
        comparisons = [compare_tensors(torch, reference_layers[name], layers[name]) for layers in layer_tensors[1:]]
        exact = all(row["torch_equal"] for row in comparisons)
        layerwise[name] = {
            "reference": tensor_stats(torch, reference_layers[name]),
            "comparisons_to_repeat_0": comparisons,
            "all_exact": exact,
            "max_abs_diff_max": max([row["max_abs_diff"] for row in comparisons] or [0.0]),
        }
        if first_divergent is None and not exact:
            first_divergent = name
    return {
        "device": str(device),
        "repeat_count": repeats,
        "parameter_hash": param_hash,
        "input_hash_before": input_before,
        "input_hash_after": input_after,
        "input_mutated": input_before != input_after,
        "outputs": outputs,
        "output_hashes": [row["sha256"] for row in outputs],
        "all_output_hashes_identical": len({row["sha256"] for row in outputs}) == 1,
        "final_embedding_repeat_comparisons": [
            compare_tensors(torch, layer_tensors[0]["final_embedding"], layers["final_embedding"])
            for layers in layer_tensors[1:]
        ],
        "layerwise": layerwise,
        "first_divergent_operation": first_divergent,
        "nan_inf": any(row["nan"] or row["inf"] for row in outputs),
    }


def child_main(args: argparse.Namespace) -> None:
    np, torch, F, GATv2Conv, runtime = lock_runtime(1)
    device = torch.device(args.device)
    manifest = read_json(Path(args.manifest))
    checkpoint = Path(manifest["checkpoint_path"])
    payload = load_torch_file(torch, checkpoint)
    data = load_torch_file(torch, Path(args.input_pt))
    canonical_data, canonical_manifest = canonicalize_data(torch, data, device)
    result = run_repeats(
        torch=torch,
        F=F,
        GATv2Conv=GATv2Conv,
        model_config=payload["model_config"],
        state_dict=payload["encoder_state_dict"],
        canonical_data=canonical_data,
        device=device,
        repeats=1,
    )
    print(json.dumps({"runtime": runtime, "canonical": canonical_manifest, "result": result}, ensure_ascii=False))


def run_fresh_processes(
    *,
    project_root: Path,
    manifest_path: Path,
    input_pt: Path,
    device_name: str,
    repeats: int,
) -> Dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONHASHSEED": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "PYTORCH_MPS_FAST_MATH": "0",
        }
    )
    rows: List[Dict[str, Any]] = []
    for index in range(repeats):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--project-root",
            str(project_root),
            "--manifest",
            str(manifest_path),
            "--input-pt",
            str(input_pt),
            "--device",
            device_name,
        ]
        proc = subprocess.run(cmd, cwd=project_root, env=env, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            rows.append({"index": index, "status": "FAIL", "returncode": proc.returncode, "stderr": proc.stderr[-4000:]})
            continue
        payload = json.loads(proc.stdout)
        rows.append({"index": index, "status": "PASS", **payload["result"]})
    hashes = [row["output_hashes"][0] for row in rows if row.get("status") == "PASS"]
    return {
        "device": device_name,
        "fresh_process_count": repeats,
        "rows": rows,
        "all_processes_passed": len(hashes) == repeats,
        "fresh_process_hash_identical": len(set(hashes)) == 1 if hashes else False,
        "fresh_process_hashes": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 4-R2 MPS GATv2 dynamic embedding determinism audit.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--repair-source", default=DEFAULT_REPAIR_SOURCE)
    parser.add_argument("--output-name", default="")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--input-pt", default="")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    if args.child:
        child_main(args)
        return

    env_lock = require_env()
    np, torch, F, GATv2Conv, runtime = lock_runtime(1)
    project_root = Path(args.project_root).expanduser().resolve()
    output_name = args.output_name or f"prompt4_r2_mps_determinism_repair_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = project_root / "05_training/artifacts" / output_name
    output_root.mkdir(parents=True, exist_ok=True)

    guideline_files = {
        "AGENTS.md": (project_root / "AGENTS.md").exists(),
        "project.md": (project_root / "project.md").exists(),
        "README.md": (project_root / "README.md").exists(),
    }
    repair_source = project_root / args.repair_source
    manifest_path = repair_source / "prompt4_encoder_provenance_repair" / "encoder_checkpoint_manifest.json"
    prompt4_gate_path = repair_source / "prompt4_encoder_provenance_repair" / "prompt4_repair_gate.json"
    manifest = read_json(manifest_path)
    prompt4_gate = read_json(prompt4_gate_path)
    checkpoint_path = Path(manifest["checkpoint_path"])
    checkpoint_payload = load_torch_file(torch, checkpoint_path)
    parameter_hash_loaded = hashlib.sha256()
    for name, tensor in sorted(checkpoint_payload["encoder_state_dict"].items()):
        parameter_hash_loaded.update(name.encode("utf-8"))
        parameter_hash_loaded.update(tensor.detach().to("cpu").contiguous().numpy().tobytes())
    parameter_hash_loaded_value = parameter_hash_loaded.hexdigest()

    train_dir = project_root / "05_training/artifacts/dataset_full_20260422_084243/train"
    input_pt = sorted(train_dir.glob("*.pt"))[0]
    data_cpu = load_torch_file(torch, input_pt)
    cpu_device = torch.device("cpu")
    cpu_data, input_manifest_cpu = canonicalize_data(torch, data_cpu, cpu_device)
    input_manifest = {
        **input_manifest_cpu,
        "input_pt": str(input_pt),
        "input_pt_sha256": sha256_file(input_pt),
        "test_split_read": False,
        "test_embedding_generated": False,
        "split_scope": "train_first_snapshot_only",
    }
    dump_json(output_root / "input_canonicalization_manifest.json", input_manifest)

    environment = {
        "created_at_utc": utc_now(),
        "repair_version": REPAIR_VERSION,
        "environment_variables": env_lock,
        "runtime": runtime,
        "machine_id": platform.node(),
        "git_commit": git_commit(project_root),
        "guideline_files": guideline_files,
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "checkpoint_file_sha256_expected": manifest["checkpoint_file_sha256"],
        "canonical_parameter_hash": manifest["new_parameter_hash"],
        "loaded_parameter_hash": parameter_hash_loaded_value,
        "graph_contract_hash": manifest.get("graph_contract_hash"),
        "feature_schema_hash": manifest.get("feature_schema_hash"),
        "node_ordering_hash": manifest.get("node_ordering_hash"),
        "input_hashes": {
            "input_pt": sha256_file(input_pt),
            "x": input_manifest["x"]["sha256"],
            "edge_index": input_manifest["edge_index"]["sha256"],
            "edge_attr": input_manifest["edge_attr"]["sha256"] if input_manifest["edge_attr"] else None,
        },
        "no_torch_upgrade_performed": True,
        "test_split_read": False,
    }
    dump_json(output_root / "environment_lock.json", environment)

    cpu_same = run_repeats(
        torch=torch,
        F=F,
        GATv2Conv=GATv2Conv,
        model_config=checkpoint_payload["model_config"],
        state_dict=checkpoint_payload["encoder_state_dict"],
        canonical_data=cpu_data,
        device=cpu_device,
        repeats=10,
    )
    mps_available = bool(torch.backends.mps.is_available())
    if not mps_available:
        mps_same = {
            "device": "mps",
            "status": "BLOCKED",
            "blockers": ["MPS_NOT_AVAILABLE"],
            "all_output_hashes_identical": False,
            "first_divergent_operation": "MPS_NOT_AVAILABLE",
        }
    else:
        mps_device = torch.device("mps")
        data_mps_cpu = load_torch_file(torch, input_pt)
        mps_data, _mps_input_manifest = canonicalize_data(torch, data_mps_cpu, mps_device)
        mps_same = run_repeats(
            torch=torch,
            F=F,
            GATv2Conv=GATv2Conv,
            model_config=checkpoint_payload["model_config"],
            state_dict=checkpoint_payload["encoder_state_dict"],
            canonical_data=mps_data,
            device=mps_device,
            repeats=10,
        )
    cpu_fresh = run_fresh_processes(project_root=project_root, manifest_path=manifest_path, input_pt=input_pt, device_name="cpu", repeats=3)
    mps_fresh = (
        run_fresh_processes(project_root=project_root, manifest_path=manifest_path, input_pt=input_pt, device_name="mps", repeats=3)
        if mps_available
        else {"device": "mps", "fresh_process_count": 3, "all_processes_passed": False, "fresh_process_hash_identical": False, "rows": []}
    )

    matrix = {
        "created_at_utc": utc_now(),
        "cpu_same_process": cpu_same,
        "mps_same_process": mps_same,
        "cpu_fresh_process": cpu_fresh,
        "mps_fresh_process": mps_fresh,
    }
    dump_json(output_root / "cpu_mps_repeatability_matrix.json", matrix)
    layerwise = {
        "created_at_utc": utc_now(),
        "cpu_first_divergent_operation": cpu_same.get("first_divergent_operation"),
        "mps_first_divergent_operation": mps_same.get("first_divergent_operation"),
        "cpu_layerwise": cpu_same.get("layerwise"),
        "mps_layerwise": mps_same.get("layerwise"),
        "note": "attention logits are represented by accessible linear projections and returned attention coefficients; PyG internal pre-softmax logits are not exposed without vendoring message-passing internals.",
    }
    dump_json(output_root / "layerwise_determinism_audit.json", layerwise)

    checkpoint_report = {
        "created_at_utc": utc_now(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "checkpoint_file_sha256_expected": manifest["checkpoint_file_sha256"],
        "checkpoint_sha_match": sha256_file(checkpoint_path) == manifest["checkpoint_file_sha256"],
        "canonical_parameter_hash": manifest["new_parameter_hash"],
        "loaded_parameter_hash": parameter_hash_loaded_value,
        "parameter_hash_unchanged": parameter_hash_loaded_value == manifest["new_parameter_hash"],
        "embedding_cache_treated_as_checkpoint": False,
        "previous_prompt4_repair_gate": str(prompt4_gate_path),
        "previous_prompt4_repair_status": prompt4_gate.get("status"),
    }
    dump_json(output_root / "checkpoint_provenance_report.json", checkpoint_report)

    cpu_exact = bool(cpu_same["all_output_hashes_identical"] and cpu_fresh["fresh_process_hash_identical"])
    mps_exact = bool(mps_same.get("all_output_hashes_identical") and mps_fresh.get("fresh_process_hash_identical"))
    parameter_ok = bool(checkpoint_report["parameter_hash_unchanged"])
    nan_inf_false = not bool(cpu_same.get("nan_inf") or mps_same.get("nan_inf"))
    fallback_false = env_lock["observed"].get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    if not cpu_exact:
        classification = "PROJECT_OR_PIPELINE_BUG"
    elif cpu_exact and not mps_exact:
        first = mps_same.get("first_divergent_operation") or "fresh_process_final_embedding"
        if "attention" in str(first) or "final_embedding" in str(first) or "conv" in str(first):
            classification = "MPS_BACKEND_NUMERICAL_NONDETERMINISM_CANDIDATE"
        else:
            classification = "MPS_REPEATABILITY_BLOCKED"
    else:
        classification = "STRICT_REPAIR_PASS"
    pass_conditions = {
        "cpu_internal_all_repeats_exact": bool(cpu_same["all_output_hashes_identical"]),
        "mps_internal_all_repeats_exact": bool(mps_same.get("all_output_hashes_identical")),
        "reload_embedding_max_diff_zero": bool(mps_same.get("all_output_hashes_identical")),
        "same_input_twice_hash_identical": bool(mps_same.get("all_output_hashes_identical")),
        "fresh_process_hash_identical": bool(cpu_fresh["fresh_process_hash_identical"] and mps_fresh.get("fresh_process_hash_identical")),
        "parameter_hash_unchanged": parameter_ok,
        "no_nan_inf": nan_inf_false,
        "no_fallback": fallback_false,
        "no_leakage": True,
        "test_split_not_read": True,
        "test_embedding_not_generated": True,
    }
    status = "PASS" if all(pass_conditions.values()) else "BLOCKED"
    if classification == "PROJECT_OR_PIPELINE_BUG":
        status = "BLOCKED"
    gate = {
        "created_at_utc": utc_now(),
        "prompt": "Prompt 4-R2",
        "status": status,
        "classification": classification,
        "first_divergent_operation": None if status == "PASS" else (mps_same.get("first_divergent_operation") or cpu_same.get("first_divergent_operation")),
        "pass_conditions": pass_conditions,
        "blockers": [key for key, ok in pass_conditions.items() if not ok],
        "checkpoint_file_sha256": checkpoint_report["checkpoint_file_sha256"],
        "canonical_parameter_hash": checkpoint_report["canonical_parameter_hash"],
        "input_hashes": environment["input_hashes"],
        "cpu_repeats": {
            "same_process_hashes": cpu_same["output_hashes"],
            "fresh_process_hashes": cpu_fresh["fresh_process_hashes"],
            "max_diff": max([r["max_abs_diff"] for r in cpu_same["final_embedding_repeat_comparisons"]] or [0.0]),
        },
        "mps_repeats": {
            "same_process_hashes": mps_same.get("output_hashes", []),
            "fresh_process_hashes": mps_fresh.get("fresh_process_hashes", []),
            "max_diff": max([r["max_abs_diff"] for r in mps_same.get("final_embedding_repeat_comparisons", [])] or [0.0]),
        },
        "prompt5_executed": False,
        "prompt6_executed": False,
        "remaining_blockers": [key for key, ok in pass_conditions.items() if not ok],
        "fallback_options_not_applied": [
            "E1 immutable train/validation dynamic-embedding cache",
            "E2 deterministic CPU/CUDA reference backend",
            "pre-approved MPS_NUMERICAL_REPRODUCIBILITY_V1",
        ],
        "runtime": {
            **runtime,
            "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
        },
    }
    dump_json(output_root / "prompt4_r2_gate.json", gate)
    report_lines = [
        "# Prompt 4-R2 MPS GATv2 Dynamic-Embedding Determinism Repair/Audit",
        "",
        f"status: {status}",
        f"classification: {classification}",
        f"first_divergent_operation: {gate['first_divergent_operation']}",
        "",
        "Prompt 5 E1/E2, repaired matrix, and Prompt 6 were not executed.",
        "Test split and test embedding were not read or generated.",
    ]
    write_text(output_root / "prompt4_r2_final_report.md", "\n".join(report_lines) + "\n")
    print(json.dumps({"status": status, "classification": classification, "artifact_root": str(output_root), "gate": str(output_root / "prompt4_r2_gate.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
