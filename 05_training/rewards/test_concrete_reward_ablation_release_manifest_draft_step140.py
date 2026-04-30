from __future__ import annotations
from validate_concrete_reward_ablation_release_manifest_draft_step140 import validate_payload

def sample():
    p={"artifact_version":"concrete_reward_ablation_release_manifest_draft_step140_v1","step":140,"audit_status":"PASS","draft_status":"CONCRETE_RELEASE_MANIFEST_DRAFT_CREATED_ACTUAL_STILL_LOCKED","actual_execution_allowed":False,"actual_execution_released":False,"actual_executed":False,"actual_results":False,"reward_result_written":False,"winner_selected":False,"trainable_reward_promoted":False,"train_with_this_reward_allowed":False,"actual_training_allowed":False,"final_reward_design_claim_allowed":False,"best_reward_claim_allowed":False,"paper_level_claim_allowed":False,"causal_performance_claim_allowed":False}
    d={"release_manifest_version":"actual_reward_ablation_release_manifest_v1","requested_matrix":{"conditions":["A"],"reward_ids":["R0","R1","R2","R3","R4","R5"],"seeds":[1,2,3],"planned_run_count":18},"release_decision":{"actual_execution_allowed":False,"actual_execution_released":False,"release_ready":False,"release_validation_status":"BLOCKED_PENDING_OPERATOR_APPROVAL"},"non_claim_guards":{"actual_results":False,"winner_selected":False,"trainable_reward_promoted":False,"train_with_this_reward_allowed":False,"actual_training_allowed":False,"paper_level_claim_allowed":False,"causal_performance_claim_allowed":False}}
    return p,d

def main():
    p,d=sample(); assert validate_payload(p,d)["validation_status"]=="PASS"
    p,d=sample(); p["actual_execution_allowed"]=True; assert validate_payload(p,d)["validation_status"]=="FAIL"
    p,d=sample(); d["release_decision"]["release_ready"]=True; assert validate_payload(p,d)["validation_status"]=="FAIL"
    p,d=sample(); d["requested_matrix"]["planned_run_count"]=17; assert validate_payload(p,d)["validation_status"]=="FAIL"
    print("[OK] Step 140 concrete release manifest draft self-test PASS")
if __name__=="__main__": main()
