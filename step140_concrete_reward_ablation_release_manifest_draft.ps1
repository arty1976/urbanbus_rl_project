$ErrorActionPreference = "Stop"
$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) { throw "[STOP] Project root not found: $ProjectRoot" }
Set-Location $ProjectRoot
$RewardsDir = Join-Path $ProjectRoot "05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

$PyMain = Join-Path $RewardsDir "generate_concrete_reward_ablation_release_manifest_draft_step140.py"
$PyValidate = Join-Path $RewardsDir "validate_concrete_reward_ablation_release_manifest_draft_step140.py"
$PyTest = Join-Path $RewardsDir "test_concrete_reward_ablation_release_manifest_draft_step140.py"
$MdDoc = Join-Path $RewardsDir "concrete_reward_ablation_release_manifest_draft_step140.md"

$MainCode = @'
from __future__ import annotations
import argparse, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

STEP=140
VERSION="concrete_reward_ablation_release_manifest_draft_step140_v1"
STEP139_POINTER="05_training/rewards/reward_ablation_actual_execution_release_manifest_design_step139.latest.json"
MATRIX={"conditions":["A"],"reward_ids":["R0","R1","R2","R3","R4","R5"],"seeds":[1,2,3],"planned_run_count":18}
REQUIRED_PHRASE="I APPROVE ACTUAL REWARD ABLATION EXECUTION FOR A_R0_TO_R5_SEEDS_1_TO_3"
FALSE_GUARDS=["actual_execution_allowed","actual_execution_released","actual_executed","actual_results","reward_result_written","winner_selected","trainable_reward_promoted","train_with_this_reward_allowed","actual_training_allowed","final_reward_design_claim_allowed","best_reward_claim_allowed","paper_level_claim_allowed","causal_performance_claim_allowed"]

def now(): return datetime.now(timezone.utc).isoformat()
def read_json(p:Path)->Dict[str,Any]:
    for enc in ("utf-8-sig","utf-8"):
        try: return json.loads(p.read_text(encoding=enc))
        except Exception: pass
    raise RuntimeError(f"failed_to_read_json: {p}")
def write_json(p:Path,d:Dict[str,Any]):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8")
def write_text(p:Path,s:str):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s,encoding="utf-8")
def res(root:Path,raw:str)->Path:
    p=Path(str(raw)); return p if p.is_absolute() else root/p

def find_step139(root:Path):
    warnings=[]; ptr=root/STEP139_POINTER
    if ptr.exists():
        try:
            pp=read_json(ptr); raw=pp.get("manifest_path") or pp.get("path")
            if raw and res(root,raw).exists(): return res(root,raw), warnings
            warnings.append("step139_pointer_manifest_missing_or_empty")
        except Exception as e: warnings.append(f"step139_pointer_unreadable: {e}")
    else: warnings.append("step139_pointer_missing")
    c=sorted([p for p in (root/"artifacts"/"rewards").glob("**/*step139*manifest*.json") if p.is_file()], key=lambda p:p.stat().st_mtime, reverse=True) if (root/"artifacts"/"rewards").exists() else []
    if c:
        warnings.append("using_step139_fallback_manifest_search"); return c[0], warnings
    return None, warnings

def check_false_recursive(obj:Any,prefix=""):
    bad=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            path=f"{prefix}.{k}" if prefix else str(k)
            if k=="required_source_tokens": continue
            if k in FALSE_GUARDS and bool(v): bad.append(f"forbidden_true: {path}")
            if isinstance(v,(dict,list)): bad += check_false_recursive(v,path)
    elif isinstance(obj,list):
        for i,v in enumerate(obj):
            if isinstance(v,(dict,list)): bad += check_false_recursive(v,f"{prefix}[{i}]")
    return bad

def load_template(root:Path):
    warnings=[]; violations=[]
    mpath, w = find_step139(root); warnings += w
    if not mpath or not mpath.exists(): return Path(""),Path(""),{}, {}, warnings, ["step139_manifest_missing"]
    m=read_json(mpath)
    if m.get("step")!=139: violations.append(f"step139_step_mismatch: {m.get('step')}")
    if m.get("audit_status")!="PASS": violations.append(f"step139_audit_not_pass: {m.get('audit_status')}")
    if m.get("design_status")!="RELEASE_MANIFEST_SCHEMA_DESIGNED_ACTUAL_STILL_LOCKED": violations.append("step139_design_status_invalid")
    violations += check_false_recursive(m)
    tpath=res(root,m.get("release_manifest_template_path") or m.get("template_path") or "")
    if not tpath.exists(): return mpath,tpath,m,{},warnings,violations+["step139_template_missing"]
    t=read_json(tpath)
    if t.get("release_manifest_version")!="actual_reward_ablation_release_manifest_v1": violations.append("template_version_mismatch")
    if t.get("requested_matrix")!=MATRIX: violations.append("template_matrix_mismatch")
    dec=t.get("release_decision",{})
    if dec.get("actual_execution_allowed") or dec.get("actual_execution_released") or dec.get("release_ready"): violations.append("template_unlocks_execution")
    return mpath,tpath,m,t,warnings,violations

def build_md(payload):
    return "\n".join(["# Step 140 concrete reward ablation release manifest draft generator","","This step creates a concrete release manifest draft from Step 139.","It does not unlock actual execution.","",f"- audit_status: `{payload['audit_status']}`",f"- draft_status: `{payload['draft_status']}`",f"- release_validation_status: `{payload['release_validation_status']}`",f"- actual_execution_allowed: `{payload['actual_execution_allowed']}`",f"- actual_results: `{payload['actual_results']}`",f"- winner_selected: `{payload['winner_selected']}`",""])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",default="."); ap.add_argument("--output-root",default="artifacts/rewards/concrete_reward_ablation_release_manifest_draft_step140"); ap.add_argument("--requested-by",default=""); ap.add_argument("--reviewed-by",default=""); ap.add_argument("--approved-by",default=""); ap.add_argument("--operator-approval-phrase",default=""); ap.add_argument("--result-output-root",default="artifacts/rewards/actual_reward_ablation_results")
    a=ap.parse_args(); root=Path(a.project_root).resolve(); out=(root/a.output_root).resolve()
    mpath,tpath,m,t,warnings,violations=load_template(root)
    phrase_matches=a.operator_approval_phrase.strip()==REQUIRED_PHRASE
    release_blocks=[]
    if not a.operator_approval_phrase.strip(): release_blocks.append("operator_approval_phrase_missing")
    elif not phrase_matches: release_blocks.append("operator_approval_phrase_mismatch")
    if not a.requested_by.strip(): release_blocks.append("requested_by_missing")
    if not a.reviewed_by.strip(): release_blocks.append("reviewed_by_missing")
    if not a.approved_by.strip(): release_blocks.append("approved_by_missing")
    if not a.result_output_root.strip(): release_blocks.append("result_output_root_missing")
    rel_status="DRAFT_READY_FOR_SEPARATE_MANUAL_RELEASE_REVIEW" if not release_blocks else "BLOCKED_PENDING_OPERATOR_APPROVAL"
    draft={"release_manifest_version":"actual_reward_ablation_release_manifest_v1","draft_artifact_version":VERSION,"created_from_step":140,"created_at_utc":now(),"purpose":"Concrete release manifest draft only. This file does not release actual execution.","operator_fields":{"requested_by":a.requested_by,"reviewed_by":a.reviewed_by,"approved_by":a.approved_by,"approval_timestamp_utc":"","operator_approval_phrase":a.operator_approval_phrase},"required_operator_approval_phrase":REQUIRED_PHRASE,"operator_approval_phrase_matches":phrase_matches,"operator_fields_complete":bool(a.requested_by.strip() and a.reviewed_by.strip() and a.approved_by.strip() and a.operator_approval_phrase.strip()),"requested_matrix":MATRIX,"result_output_root":a.result_output_root,"source_step139":{"manifest_path":str(mpath),"template_path":str(tpath),"audit_status":m.get("audit_status"),"design_status":m.get("design_status")},"upstream_manifest_paths":t.get("upstream_manifest_paths",{}),"release_decision":{"actual_execution_allowed":False,"actual_execution_released":False,"release_ready":False,"release_validation_status":rel_status,"release_blocking_reasons":release_blocks,"next_required_step":"Step 141 explicit release validator"},"non_claim_guards":{"actual_results":False,"winner_selected":False,"trainable_reward_promoted":False,"train_with_this_reward_allowed":False,"actual_training_allowed":False,"paper_level_claim_allowed":False,"causal_performance_claim_allowed":False}}
    audit="PASS" if not violations else "BLOCKED"; dstatus="CONCRETE_RELEASE_MANIFEST_DRAFT_CREATED_ACTUAL_STILL_LOCKED" if audit=="PASS" else "CONCRETE_RELEASE_MANIFEST_DRAFT_BLOCKED"
    draft_path=out/"concrete_actual_reward_ablation_release_manifest_draft_step140.json"; man_path=out/"concrete_reward_ablation_release_manifest_draft_step140_manifest.json"; md_path=out/"concrete_reward_ablation_release_manifest_draft_step140.md"
    payload={"artifact_version":VERSION,"step":140,"created_at_utc":now(),"project_root":str(root),"output_root":str(out),"python_executable":sys.executable,"python_version":sys.version.replace("\n"," "),"platform":platform.platform(),"audit_status":audit,"draft_status":dstatus,"release_validation_status":rel_status,"blocking_reasons":violations,"warnings":warnings,"step139_manifest_path":str(mpath),"step139_template_path":str(tpath),"concrete_draft_path":str(draft_path),"requested_matrix":MATRIX,"actual_execution_allowed":False,"actual_execution_released":False,"actual_executed":False,"actual_results":False,"reward_result_written":False,"winner_selected":False,"trainable_reward_promoted":False,"train_with_this_reward_allowed":False,"actual_training_allowed":False,"final_reward_design_claim_allowed":False,"best_reward_claim_allowed":False,"paper_level_claim_allowed":False,"causal_performance_claim_allowed":False,"scope_note":"Step 140 creates a concrete release manifest draft. It does not execute actual reward ablation."}
    write_json(draft_path,draft); write_json(man_path,payload); write_text(md_path,build_md(payload)); write_text(root/"05_training"/"rewards"/"concrete_reward_ablation_release_manifest_draft_step140.md",build_md(payload)); write_json(root/"05_training"/"rewards"/"concrete_reward_ablation_release_manifest_draft_step140.latest.json",{"manifest_path":str(man_path),"concrete_draft_path":str(draft_path),"audit_status":audit,"draft_status":dstatus,"release_validation_status":rel_status,"actual_execution_allowed":False,"actual_execution_released":False,"actual_results":False,"created_at_utc":payload["created_at_utc"]})
    print("[OK] Step 140 concrete reward ablation release manifest draft generator completed"); print(f"[OK] audit_status              : {audit}"); print(f"[OK] draft_status              : {dstatus}"); print(f"[OK] release_validation_status : {rel_status}"); print("[OK] actual_execution_allowed  : False"); print("[OK] actual_execution_released : False"); print("[OK] actual_executed           : False"); print("[OK] actual_results            : False"); print("[OK] winner_selected           : False"); print("[OK] training_allowed          : False"); print("[OK] paper_claim               : False"); print(f"[OK] concrete_draft            : {draft_path}")
    for w in warnings: print(f"[WARN] {w}")
    if audit!="PASS":
        print("[BLOCKED] Step 140 draft generator failed"); [print(f"[BLOCKED] {r}") for r in violations]; raise SystemExit(2)
if __name__=="__main__": main()
'@

$ValidateCode = @'
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
'@

$TestCode = @'
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
'@

$DocText = @'
# Step 140 concrete reward ablation release manifest draft generator

Step 140 creates a concrete release manifest draft from the Step 139 template.

This step does not unlock actual execution.

Default status:
- release_validation_status = BLOCKED_PENDING_OPERATOR_APPROVAL
- actual_execution_allowed = false
- actual_execution_released = false
- actual_executed = false
- actual_results = false
- winner_selected = false
- train_with_this_reward_allowed = false
- paper_level_claim_allowed = false
- causal_performance_claim_allowed = false
'@

Set-Content -Path $PyMain -Value $MainCode -Encoding UTF8
Set-Content -Path $PyValidate -Value $ValidateCode -Encoding UTF8
Set-Content -Path $PyTest -Value $TestCode -Encoding UTF8
Set-Content -Path $MdDoc -Value $DocText -Encoding UTF8

if (Test-Path ".\05_training\.venv\Scripts\python.exe") { $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path } else { $py = "python" }
$OutputRoot = "artifacts/rewards/concrete_reward_ablation_release_manifest_draft_step140"
$Manifest = Join-Path $ProjectRoot "$OutputRoot\concrete_reward_ablation_release_manifest_draft_step140_manifest.json"
$Draft = Join-Path $ProjectRoot "$OutputRoot\concrete_actual_reward_ablation_release_manifest_draft_step140.json"

& $py $PyMain --project-root "." --output-root $OutputRoot --requested-by "" --reviewed-by "" --approved-by "" --operator-approval-phrase "" --result-output-root "artifacts/rewards/actual_reward_ablation_results"
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 140 concrete release manifest draft generator failed" }
& $py $PyValidate --manifest $Manifest --draft $Draft
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 140 manifest/draft validation failed" }
& $py $PyTest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 140 self-test failed" }
Write-Host "[DONE] Step 140 concrete reward ablation release manifest draft generator complete."

