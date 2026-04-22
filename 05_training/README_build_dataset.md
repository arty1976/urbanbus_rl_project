# 05_training dataset builder files

## Included
- `build_gatv2_dataset.py` — canonical GATv2 dataset builder
- `run_build_dataset.ps1` — PowerShell runner with log directory creation

## Smoke test
```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project\05_training"
.\run_build_dataset.ps1 --max-snapshots 8 --dry-run
```

## Full build
```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project\05_training"
.\run_build_dataset.ps1
```

## Notes
- Prefers materialized tables if available.
- Keeps the full graph and uses `node_mask` for supervised labels.
- Uses chronological split only.
