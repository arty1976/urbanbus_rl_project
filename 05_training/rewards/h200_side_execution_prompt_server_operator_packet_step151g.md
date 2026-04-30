# Step 151-G - H200-side execution prompt / server operator packet, still locked

This step creates a server-operator packet for the real H200-side Step 149 rerun.
It does not run H200 preflight, does not record operator approval, and does not release actual reward ablation execution.

Required H200 command:

```bash
python 05_training/rewards/h200_environment_preflight_result_manifest_step149.py \
  --project-root /workspace/urbanbus_rl_project \
  --expect-h200 \
  --min-gpu-count 1 \
  --output-root artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual
```
