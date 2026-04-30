# Step 145 H200 transfer package export manifest

This step exports the file-level manifest for the H200 transfer package.

It does not run H200 jobs and does not release actual execution.

## Status

- audit_status: `PASS`
- export_status: `H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_READY_ACTUAL_STILL_LOCKED`
- transfer_file_count: `10`
- planned_run_count: `72`
- actual_execution_allowed: `False`
- actual_execution_released: `False`
- actual_results: `False`
- winner_selected: `False`
- paper_level_claim_allowed: `False`

## Transfer files

| # | relative_path | size_bytes | sha256 |
|---:|---|---:|---|
| 1 | `05_training/rewards/final_reward_spec_step111.md` | 4813 | `e36cb819419d...` |
| 2 | `05_training/rewards/final_reward_spec_step111.json` | 5743 | `22a87b82f325...` |
| 3 | `05_training/rewards/a_family_72run_release_matrix_extension_draft_step143.md` | 855 | `68330a9ca524...` |
| 4 | `05_training/rewards/a_family_72run_release_matrix_extension_draft_step143.py` | 22120 | `d2da26bc060f...` |
| 5 | `05_training/rewards/validate_a_family_72run_release_matrix_extension_draft_step143.py` | 6830 | `3c219b546945...` |
| 6 | `05_training/rewards/test_a_family_72run_release_matrix_extension_draft_step143.py` | 3968 | `0766f69bc84f...` |
| 7 | `05_training/rewards/h200_execution_package_boundary_manifest_step144.md` | 936 | `fa11c8b69eed...` |
| 8 | `05_training/rewards/h200_execution_package_boundary_manifest_step144.py` | 21329 | `c0dab10f439e...` |
| 9 | `05_training/rewards/validate_h200_execution_package_boundary_manifest_step144.py` | 5680 | `f271bc0c47c0...` |
| 10 | `05_training/rewards/test_h200_execution_package_boundary_manifest_step144.py` | 4020 | `99637a485db9...` |

## Guard

- `artifacts/**`, `*.latest.json`, baseline artifact folders, API sampling folders, and temporary project_files are not transfer source-of-truth targets.
- Step 143 remains the 72-run matrix source of truth for H200 planning.
- Actual execution remains locked until a later explicit operator release.
