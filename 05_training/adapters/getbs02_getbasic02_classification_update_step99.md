# Step 99-A/B Classification Update — Daegu BIS API Acquisition Audit

Artifact version: `daegu_bis_api_audit_step99_ab_classification_update_v1`  
Status: `APPROVED_FOR_NEXT_STEP`  
Scope: API acquisition feasibility and artifact-only collection audit  
DB write: forbidden and not performed  
Tensor DB overwrite: forbidden and not performed  
Paper-level performance claim: not allowed  
Causal performance claim: not allowed  
Fleet reduction claim: not allowed

---

## 1. Purpose

This document updates the Step 97 tensor DB data availability and generability classification after Step 99-A `/getBs02` and Step 99-B `/getBasic02` audits.

Step 97 originally classified `route_id`, `direction_id`, and `ordered_stop_sequence` as missing for causal simulator v2. Step 99-A/B shows that these fields can be recovered from Daegu BIS API artifacts for the main route population, with four known exceptions.

---

## 2. Step 99-A `/getBs02` result

Final status: `PASS_WITH_KNOWN_EXCEPTIONS`

Observed results:

| Metric | Value |
|---|---:|
| attempted routes | 238 |
| successful routes | 234 |
| exception routes | 4 |
| normalized route-stop rows | 20,508 |
| ordered_stop_sequence success among successful routes | 234 / 234 |

Known exception routes:

| route_id | route_no | route_type | status |
|---|---|---|---|
| 4040006020 | 달성6 | 지선 | AUTH_ERROR |
| 4040006021 | 달성6 | 지선 | AUTH_ERROR |
| 4040007001 | 달성7 | 지선 | AUTH_ERROR |
| 4040007009 | 달성7 | 지선 | AUTH_ERROR |

Detected schema fields:

| Normalized field | API evidence |
|---|---|
| route_id | request routeId attached to each normalized row |
| direction_id | `moveDir` |
| stop_id | `bsId` |
| stop_name | `bsNm` |
| stop_order | `seq` |

Interpretation:

`/getBs02` can provide route-aware stop sequences for the main route population. The correct endpoint is `https://apis.data.go.kr/6270000/dbmsapi02/getBs02`, not `dbmsapi01/getBs02`.

---

## 3. Step 99-B `/getBasic02` result

Final status: `PASS`

Observed results:

| Metric | Value |
|---|---:|
| candidate response rows | 25,151 |
| normalized master rows | 522 |
| unique API route_id count | 522 |
| project DB route_id count | 238 |
| matched project route_id count | 234 |
| API-side coverage | 234 / 522 = 44.8276% |
| project-route coverage | 234 / 238 = 98.32% |

Detected schema fields:

| Field | Status | Notes |
|---|---|---|
| route_id | observed_candidate | detected in `/getBasic02` response |
| route_no | observed_candidate | route number/name-like field detected |
| route_type | missing_or_unparsed in `/getBasic02` | already present in `public.stg_daegu_routes.route_type` |
| direction_id | missing_or_unparsed in `/getBasic02` | available from `/getBs02.moveDir` |
| origin_destination_stop | observed_candidate | origin/destination stop fields detected |

Interpretation:

`/getBasic02` is useful as a master route snapshot source. For causal simulator v2, `/getBs02` remains the stronger evidence source for direction-aware ordered stop sequences.

---

## 4. Step 97 classification update

| Field | Step 97 previous status | Updated status | Evidence |
|---|---|---|---|
| route_id | missing | observed_candidate_full_collection_234_of_238_routes | `/getBs02` and `/getBasic02` route IDs matched project route population except four known exceptions |
| direction_id | missing | observed_candidate_full_collection_from_getBs02_234_of_238_routes | `/getBs02.moveDir` |
| ordered_stop_sequence | missing | observed_candidate_full_collection_234_of_238_routes | `/getBs02.bsId + seq` |
| stop_id | observed/graph-available | observed_candidate_api_confirmed | `/getBs02.bsId`; sample graph coverage reached 100% against `public.gatv2_node_master_active.node_uid` |
| stop_name | partially available | observed_candidate_api_confirmed | `/getBs02.bsNm` |
| route_no | available from route catalog | observed_candidate_from_getBasic02 | `/getBasic02` route number/name-like field |
| route_type | observed from existing DB | observed_from_existing_stg_daegu_routes | `public.stg_daegu_routes.route_type`; not from `/getBasic02` |
| origin_destination_stop | available from route catalog | observed_candidate_from_getBasic02 | `/getBasic02` origin/destination stop fields |

Do not upgrade these fields to `complete_observed_100pct` until the four known exception routes are resolved or explicitly excluded from causal simulator v2 scope.

---

## 5. Implication for Causal Simulator v2

The route-aware simulator blocker is removed for the main route population.

The v2 input contract can now include:

```text
route_id
+ direction_id
+ stop_order
+ stop_id
+ node_uid / node_index
```

Recommended contract status:

```text
route_aware_mode = required_candidate_pending_exception_policy
coverage_scope = 234_of_238_routes
exception_policy_required = true
```

The simulator should support one of these exception policies:

1. Exclude the four known exception routes from route-aware v2 scope.
2. Backfill them from a trusted static route catalog if later available.
3. Treat them as non-route-aware fallback routes with explicit metadata.

---

## 6. Guardrail record

The following actions were not performed:

- DB writes
- tensor DB overwrite
- 2023 CSV reload
- performance claim generation
- fleet reduction claim generation

The `/getBs02` full collection was artifact-only: raw snapshots, manifest, normalized CSV, route summary, and report files.

---

## 7. Next step

Proceed to Step 99-C `/getPos02` live bus position sampling plan.

Because `/getPos02` is expected to be a live vehicle-position API, it should be sampled during bus operating hours. Empty responses during off-hours must be classified as `no_live_vehicle_or_off_hours`, not as API failure.
