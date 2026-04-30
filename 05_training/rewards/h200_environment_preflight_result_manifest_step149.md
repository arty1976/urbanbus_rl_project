# Step 149 H200 environment preflight result manifest

This step records environment facts before any H200 actual execution.

It does not run training and does not release actual reward ablation.

## Status

- audit_status: `PASS`
- environment_status: `LOCAL_ENVIRONMENT_PREFLIGHT_RECORDED_H200_NOT_ASSERTED_ACTUAL_STILL_LOCKED`
- expect_h200: `False`
- project_root: `C:\Users\ryujo\urbanbus_rl_project`
- actual_execution_allowed: `False`
- train_allowed: `False`

## Python / platform

- python_executable: `C:\Users\ryujo\urbanbus_rl_project\05_training\.venv\Scripts\python.exe`
- python_version: `3.12.0 (tags/v3.12.0:0fb18b0, Oct  2 2023, 13:03:39) [MSC v.1935 64 bit (AMD64)]`
- platform: `Windows-11-10.0.26200-SP0`

## PyTorch / CUDA

- torch_import_ok: `True`
- torch_version: `2.5.1+cpu`
- cuda_available: `False`
- cuda_version: ``
- gpu_count: `0`
- gpu_names: `[]`

## Guard

- PASS in local mode only means the environment facts were recorded.
- Use `--expect-h200` on H200 to require CUDA/GPU availability.
- Actual execution remains locked until a later explicit release manifest.
