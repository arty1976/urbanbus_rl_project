param(
    [string]$TargetFile = "C:\Users\ryujo\urbanbus_rl_project\05_training\build_gatv2_dataset.py"
)

if (-not (Test-Path $TargetFile)) {
    Write-Error "Target file not found: $TargetFile"
    exit 1
}

$backup = "$TargetFile.bak_warning_fix"
Copy-Item $TargetFile $backup -Force

$content = Get-Content $TargetFile -Raw -Encoding UTF8
$old = 'torch.from_numpy(x_vals)'
$new = 'torch.from_numpy(x_vals.copy())'

if ($content -notmatch [regex]::Escape($old)) {
    Write-Host "No exact match found for patch target. Backup created at: $backup"
    exit 2
}

$content = $content.Replace($old, $new)
Set-Content -Path $TargetFile -Value $content -Encoding UTF8

Write-Host "OK: patched warning source"
Write-Host "backup: $backup"
Write-Host "replacement: $old -> $new"
