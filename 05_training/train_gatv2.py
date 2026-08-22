import os
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

# --- H4M-AE-R9.8 LS3-BT3 fail-closed training authorization -------------------------
import sys as _authz_sys
from pathlib import Path as _AuthzPath

for _authz_dir in (_AuthzPath(__file__).resolve().parent, _AuthzPath(__file__).resolve().parent.parent):
    if (_authz_dir / "simulator_authorization.py").exists():
        if str(_authz_dir) not in _authz_sys.path:
            _authz_sys.path.insert(0, str(_authz_dir))
        break
import simulator_authorization as _authz  # noqa: E402
# -----------------------------------------------------------------------------------


DATASET_DIR = r"C:\Users\ryujo\urbanbus_rl_project\05_training\artifacts\dataset_full_20260422_084243\train"
DEVICE = "cpu"
EPOCHS = 3
BATCH_SIZE = 1
MAX_FILES = 128


class NodeLevelGATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=2,
            concat=True,
            edge_dim=edge_dim,
        )
        self.conv2 = GATv2Conv(
            hidden_channels * 2,
            hidden_channels,
            heads=1,
            concat=True,
            edge_dim=edge_dim,
        )
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        out = self.lin(x)   # [total_nodes, out_channels]
        return out


def main():
    _authz.require_capability("training", site="train_gatv2.py::main")
    dataset_dir = Path(DATASET_DIR)

    files = sorted(glob.glob(str(dataset_dir / "*.pt")))
    print(f"search_root={dataset_dir}")
    print(f"found_pt_files={len(files)}")

    if not files:
        raise FileNotFoundError(f"No .pt files found in {dataset_dir}")

    files = files[:MAX_FILES]
    print(f"num_files_for_smoke_test={len(files)}")
    print(f"first_file={files[0]}")
    print(f"last_file={files[-1]}")

    data_list = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    sample = data_list[0]

    in_channels = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1) if hasattr(sample, "edge_attr") and sample.edge_attr is not None else None

    if not hasattr(sample, "y") or sample.y is None:
        raise ValueError("Sample data has no y")

    if sample.y.ndim != 2:
        raise ValueError(f"Expected node-level y with ndim=2, but got shape={tuple(sample.y.shape)}")

    out_channels = sample.y.size(1)

    print(f"in_channels={in_channels}, edge_dim={edge_dim}, out_channels={out_channels}")
    print(f"sample_x_shape={tuple(sample.x.shape)}")
    print(f"sample_y_shape={tuple(sample.y.shape)}")

    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = NodeLevelGATv2(
        in_channels=in_channels,
        hidden_channels=32,
        out_channels=out_channels,
        edge_dim=edge_dim,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        step_count = 0

        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            pred = model(batch)          # [total_nodes, 3]
            target = batch.y.float()     # [total_nodes, 3]

            if pred.shape != target.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}"
                )

            # train_mask가 있으면 train 노드만 손실 계산
            if hasattr(batch, "train_mask") and batch.train_mask is not None:
                mask = batch.train_mask
                if mask.dtype != torch.bool:
                    mask = mask.bool()
                pred = pred[mask]
                target = target[mask]

            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            step_count += 1
            print(
                f"epoch={epoch + 1} step={step_count} "
                f"pred_shape={tuple(pred.shape)} target_shape={tuple(target.shape)} "
                f"loss={loss.item():.6f}"
            )

        print(f"epoch={epoch + 1} avg_loss={total_loss / max(step_count, 1):.6f}")

    print("SMOKE TEST PASS")


if __name__ == "__main__":
    main()
