param(
    [string]$BaseUrl = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02",
    [string]$RouteParam = "routeId",
    [string[]]$RouteId = @(),
    [int]$MaxRoutes = 3,
    [int]$Samples = 3,
    [int]$IntervalSec = 300,
    [string]$OutputRoot = ".\artifacts\daegu_bis_api_audit\getpos02_repeated_sampling",
    [string[]]$ExtraParam = @(),
    [string]$ServiceKey = ""
)

$ErrorActionPreference = "Stop"

$AuditScript = ".\05_training\adapters\inspect_getpos02_live_position_sampling.py"
if (-not (Test-Path $AuditScript)) {
    throw "[STOP] audit script not found: $AuditScript"
}

if (-not $ServiceKey) {
    if ($env:DAEGU_BIS_SERVICE_KEY) {
        $ServiceKey = $env:DAEGU_BIS_SERVICE_KEY
    } elseif ($env:DATAGO_SERVICE_KEY) {
        $ServiceKey = $env:DATAGO_SERVICE_KEY
    }
}

if (-not $ServiceKey) {
    throw "[STOP] service key not found. Set DAEGU_BIS_SERVICE_KEY or DATAGO_SERVICE_KEY."
}

if (-not $RouteId -or $RouteId.Count -eq 0) {
    throw "[STOP] at least one -RouteId is required."
}

# Normalize RouteId input robustly.
# Accept both:
#   -RouteId "1000005000","1000001000","3000655000"
# and:
#   -RouteId "1000005000,1000001000,3000655000"
# The API accepts only one routeId per request.
$RouteIdExpanded = @()
foreach ($ridItem in @($RouteId)) {
    foreach ($part in ([string]$ridItem).Split(",")) {
        $clean = $part.Trim().Trim('"').Trim("'")
        if ($clean) {
            $RouteIdExpanded += $clean
        }
    }
}
$RouteId = @($RouteIdExpanded | Select-Object -Unique)

if (-not $RouteId -or $RouteId.Count -eq 0) {
    throw "[STOP] no valid RouteId values after normalization."
}

if ($MaxRoutes -gt 0 -and $RouteId.Count -gt $MaxRoutes) {
    $RouteId = @($RouteId[0..($MaxRoutes - 1)])
}

if ($Samples -lt 1) {
    throw "[STOP] Samples must be >= 1."
}

$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $OutputRoot $RunId
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$ManifestRows = @()

Write-Host "[INFO] Step 99-C2 /getPos02 repeated sampling"
Write-Host "[INFO] BaseUrl     : $BaseUrl"
Write-Host "[INFO] RouteParam  : $RouteParam"
Write-Host "[INFO] RouteId     : $($RouteId -join ',')"
Write-Host "[INFO] Samples     : $Samples"
Write-Host "[INFO] IntervalSec : $IntervalSec"
Write-Host "[INFO] RunDir      : $RunDir"
Write-Host "[INFO] ServiceKey  : <REDACTED>"

for ($i = 1; $i -le $Samples; $i++) {
    $SampleTag = "{0:D3}" -f $i
    $SampleDir = Join-Path $RunDir ("sample_" + $SampleTag)
    New-Item -ItemType Directory -Force -Path $SampleDir | Out-Null

    $StartedAt = Get-Date -Format "o"

    Write-Host ""
    Write-Host "[INFO] sample $i / $Samples"
    Write-Host "[INFO] sample_dir: $SampleDir"

    $ArgsList = @(
        $AuditScript,
        "--base-url", $BaseUrl,
        "--route-param", $RouteParam,
        "--max-routes", "$MaxRoutes",
        "--output-dir", $SampleDir,
        "--service-key", $ServiceKey
    )

    foreach ($rid in @($RouteId)) {
        $ArgsList += @("--route-id", "$rid")
    }

    foreach ($ep in @($ExtraParam)) {
        if ($ep) {
            $ArgsList += @("--extra-param", "$ep")
        }
    }

    python @ArgsList

    $ExitCode = $LASTEXITCODE
    $ReportJson = Join-Path $SampleDir "getpos02_live_position_sampling_report.json"

    $AuditStatus = "SCRIPT_FAILED"
    $CandidateRows = 0
    $NormalizedRows = 0
    $LivePossible = $false

    if ($ExitCode -eq 0 -and (Test-Path $ReportJson)) {
        $Report = Get-Content $ReportJson -Raw -Encoding UTF8 | ConvertFrom-Json
        $AuditStatus = [string]$Report.audit_status
        $CandidateRows = [int]$Report.schema_summary.total_candidate_rows
        $NormalizedRows = [int]$Report.schema_summary.total_normalized_rows
        $LivePossible = [bool]$Report.schema_summary.live_position_possible_any
    }

    $EndedAt = Get-Date -Format "o"

    $ManifestRows += [PSCustomObject]@{
        sample_index = $i
        sample_tag = $SampleTag
        started_at = $StartedAt
        ended_at = $EndedAt
        audit_status = $AuditStatus
        candidate_rows = $CandidateRows
        normalized_rows = $NormalizedRows
        live_position_possible = $LivePossible
        sample_dir = $SampleDir
        report_json = $ReportJson
    }

    Write-Host "[OK] sample_status : $AuditStatus"
    Write-Host "[OK] candidate_rows: $CandidateRows"
    Write-Host "[OK] normalized    : $NormalizedRows"
    Write-Host "[OK] live_possible : $LivePossible"

    if ($i -lt $Samples) {
        Write-Host "[INFO] sleeping $IntervalSec seconds before next sample..."
        Start-Sleep -Seconds $IntervalSec
    }
}

$ManifestPath = Join-Path $RunDir "step99c2_repeated_sampling_manifest.csv"
$ReportMdPath = Join-Path $RunDir "step99c2_repeated_sampling_report.md"
$ReportJsonPath = Join-Path $RunDir "step99c2_repeated_sampling_report.json"

$ManifestRows | Export-Csv -Path $ManifestPath -NoTypeInformation -Encoding UTF8

$TotalCandidate = ($ManifestRows | Measure-Object -Property candidate_rows -Sum).Sum
$TotalNormalized = ($ManifestRows | Measure-Object -Property normalized_rows -Sum).Sum
$PassCount = @($ManifestRows | Where-Object { $_.audit_status -eq "PASS" }).Count
$ReviewCount = @($ManifestRows | Where-Object { $_.audit_status -ne "PASS" }).Count

$Summary = [PSCustomObject]@{
    artifact_version = "daegu_bis_api_audit_step99c2_getpos02_repeated_sampling_v1"
    created_at = Get-Date -Format "o"
    run_id = $RunId
    run_dir = $RunDir
    routes_requested = $RouteId
    samples = $Samples
    interval_sec = $IntervalSec
    total_candidate_rows = $TotalCandidate
    total_normalized_rows = $TotalNormalized
    pass_sample_count = $PassCount
    review_required_sample_count = $ReviewCount
    paper_level_claim_allowed = $false
    causal_performance_claim_allowed = $false
    vehicle_trajectory_claim_allowed = $false
    manifest_csv = $ManifestPath
}

$Summary | ConvertTo-Json -Depth 20 | Set-Content -Path $ReportJsonPath -Encoding UTF8

$Lines = @()
$Lines += "# Step 99-C2 - getPos02 Repeated Live Position Sampling"
$Lines += ""
$Lines += "- artifact_version: daegu_bis_api_audit_step99c2_getpos02_repeated_sampling_v1"
$Lines += "- created_at: $($Summary.created_at)"
$Lines += "- run_id: $RunId"
$Lines += "- routes_requested: $($RouteId -join ', ')"
$Lines += "- samples: $Samples"
$Lines += "- interval_sec: $IntervalSec"
$Lines += "- total_candidate_rows: $TotalCandidate"
$Lines += "- total_normalized_rows: $TotalNormalized"
$Lines += "- pass_sample_count: $PassCount"
$Lines += "- review_required_sample_count: $ReviewCount"
$Lines += "- paper_level_claim_allowed: False"
$Lines += "- causal_performance_claim_allowed: False"
$Lines += "- vehicle_trajectory_claim_allowed: False"
$Lines += ""
$Lines += "## Interpretation"
$Lines += ""
$Lines += "This run is a repeated live-position sampling audit. It can support a vehicle trajectory feasibility assessment only after checking whether the same vehicle identifier appears across multiple samples with changing sequence or position."
$Lines += ""
$Lines += "A single sample confirms live position snapshot availability, not full vehicle trajectory, actual headway, dwell time, or arrival/departure time."
$Lines += ""
$Lines += "## Guardrails"
$Lines += ""
$Lines += "- DB write: forbidden and not performed"
$Lines += "- tensor DB overwrite: forbidden and not performed"
$Lines += "- API scope: small repeated sampling only"
$Lines += "- paper-level claim: forbidden"

$Lines | Set-Content -Path $ReportMdPath -Encoding UTF8

Write-Host ""
Write-Host "[OK] Step 99-C2 repeated sampling complete"
Write-Host "[OK] run_dir      : $RunDir"
Write-Host "[OK] manifest     : $ManifestPath"
Write-Host "[OK] report_json  : $ReportJsonPath"
Write-Host "[OK] report_md    : $ReportMdPath"
Write-Host "[OK] total rows   : $TotalNormalized"
