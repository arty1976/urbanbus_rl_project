# Route Link Promotion Final Report (Dedup Remediation)

- **Execution Date**: 2026-04-10
- **Target Table**: `public.route_link_sequence`
- **Source View**: `public.vw_stg_daegu_route_links_api_dedup` (Remediated)
- **Status**: **승인 (APPROVED)** ✅

---

## 1. Summary of Results

| Indicator | Expected | Actual | Status | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Eligible Routes** | 234 | 234 | **PASS** | Source registry matched. |
| **Promoted Routes** | 234 | 234 | **PASS** | Excluded legacy route 1000. |
| **Staging Duplicates** | 0 | 0 | **PASS** | Resolved by Dedup View. |
| **Promoted Duplicates** | 0 | 0 | **PASS** | Unique constraint verified. |
| **Count Mismatch** | 0 | 0 | **PASS** | All records from view promoted. |
| **Continuity Breaks** | 0 | 0 | **PASS** | Sequence is gapless. |

---

## 2. Detailed Findings

### [PASS] staging_dup_natural_key
- **Count**: 0 rows
- **Comment**: Deduplication view correctly prioritizes the latest `loaded_at` record for each (route, direction, sequence) key.

### [PASS] count_mismatch_by_route_dir
- **Count**: 0 rows
- **Comment**: Total of 39,247 rows promoted, matching the unique keys in the staging layer.

### [PASS] unexpected_route_in_promoted
- **Count**: 0 rows
- **Comment**: Legacy sample `route_id = '1000'` has been purged from the analytical layer.

---

## 3. Final Decision: 승인 (APPROVED)

The remediation plan has been successfully executed.
1. Legacy contamination was purged.
2. Deduplication view was established and validated.
3. Patch SQL files were implemented to prevent future regression.

---

## 4. Next Steps
- Implement automated cleanup of `1000` from staging (Optional).
- Standardize the `vw_..._dedup` pattern for other staging tables if similar duplicate risks exist.
