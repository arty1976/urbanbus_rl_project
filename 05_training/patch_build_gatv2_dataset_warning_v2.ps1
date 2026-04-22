param(
    [string]$TargetFile = 'C:\Users\ryujo\urbanbus_rl_project\05_training\build_gatv2_dataset.py'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $TargetFile)) {
    throw "Target file not found: $TargetFile"
}

$backup = "$TargetFile.bak_warning_v2"
Copy-Item $TargetFile $backup -Force

$content = Get-Content $TargetFile -Raw -Encoding UTF8
$old = 'x[node_indices] = torch.from_numpy(x_vals.copy())'
$new = 'x[node_indices] = torch.tensor(np.array(x_vals, copy=True), dtype=torch.float32)'

if ($content -notmatch [regex]::Escape($old)) {
    Write-Host "Target line not found. Backup created at: $backup"
    exit 1
}

$content = $content -replace [regex]::Escape($old), [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $new }
Set-Content -Path $TargetFile -Value $content -Encoding UTF8

Write-Host "[OK] patched warning line"
Write-Host "  file   = $TargetFile"
Write-Host "  backup = $backup"
