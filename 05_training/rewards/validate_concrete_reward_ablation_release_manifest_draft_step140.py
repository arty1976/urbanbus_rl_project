from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any,Dict,List
VERSION="concrete_reward_ablation_release_manifest_draft_step140_v1"
MATRIX={"conditions":["A"],"reward_ids":["R0","R1","R2","R3","R4","R5"],"seeds":[1,2,3],"planned_run_count":18}
FALSE_KEYS=["actual_execution_allowed","actual_execution_released","actual_executed","actual_results","reward_result_written","winner_selected","trainable_reward_promoted","train_with_this_reward_allowed","actual_training_allowed","final_reward_design_claim_allowed","best_reward_claim_allowed","paper_level_claim_allowed","causal_performance_claim_allowed"]
def load(p:Path)->Dict[str,Any]:
    for e in ("utf-8-sig","utf-8"):
        try: return json.loads(p.read_text(encoding=e))
        except Exception: pass
    raise RuntimeError(f"failed_to_read_json: {p}")
def validate_draft(d:Dict[str,Any])->List[str]:
    err=[]
    if d.get("release_manifest_version")!="actual_reward_ablation_release_manifest_v1": err.append("draft_release_manifest_version_mismatch")
    if d.get("requested_matrix")!=MATRIX: err.append("draft_requested_matrix_mismatch")
    dec=d.get("release_decision",{})
    if dec.get("actual_execution_allowed") or dec.get("actual_execution_released") or dec.get("release_ready"): err.append("draft_must_not_unlock_execution")
    if dec.get("release_validation_status") not in ("BLOCKED_PENDING_OPERATOR_APPROVAL","DRAFT_READY_FOR_SEPARATE_MANUAL_RELEASE_REVIEW"): err.append("invalid_release_validation_status")
    for k,v in d.get("non_claim_guards",{}).items():
        if bool(v): err.append(f"draft_non_claim_guard_true:{k}")
    return err
def validate_payload(p:Dict[str,Any],d:Dict[str,Any],require_pass=True)->Dict[str,Any]:
    err=[]
    if p.get("artifact_version")!=VERSION: err.append("artifact_version_mismatch")
    if p.get("step")!=140: err.append("step_must_be_140")
    if p.get("audit_status") not in ("PASS","BLOCKED"): err.append("invalid_audit_status")
    if require_pass and p.get("audit_status")!="PASS": err.append("audit_status_not_pass")
    if require_pass and p.get("draft_status")!="CONCRETE_RELEASE_MANIFEST_DRAFT_CREATED_ACTUAL_STILL_LOCKED": err.append("draft_status_not_created")
    for k in FALSE_KEYS:
        if bool(p.get(k,False)): err.append(f"forbidden_true_payload_key:{k}")
    err += validate_draft(d)
    return {"validation_status":"PASS" if not err else "FAIL","errors":err}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--draft",required=True); ap.add_argument("--allow-blocked",action="store_true"); a=ap.parse_args()
    r=validate_payload(load(Path(a.manifest)),load(Path(a.draft)),require_pass=not a.allow_blocked)
    print(f"[OK] validation_status: {r['validation_status']}")
    if r["errors"]:
        [print(f"[FAIL] {e}") for e in r["errors"]]; raise SystemExit(2)
    print("[OK] Step 140 concrete release manifest draft validation PASS")
if __name__=="__main__": main()
