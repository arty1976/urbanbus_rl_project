param(
    [string]$TargetFile = ".\build_gatv2_dataset.py"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $TargetFile)) {
    throw "Target file not found: $TargetFile"
}

$backup = "$TargetFile.bak_warning_v4"
Copy-Item $TargetFile $backup -Force

$text = Get-Content $TargetFile -Raw -Encoding UTF8
$original = $text

$patterns = @(
    'x\[node_indices\]\s*=\s*torch\.from_numpy\(x_vals\.copy\(\)\)',
    'x\[node_indices\]\s*=\s*torch\.from_numpy\(x_vals\)',
    'x\[node_indices\]\s*=\s*torch\.tensor\(np\.array\(x_vals,\s*copy=True\),\s*dtype=torch\.float32\)',
    'x\[node_indices\]\s*=\s*torch\.tensor\(x_vals,\s*dtype=torch\.float32\)'
)

$replacement = 'x[node_indices] = torch.tensor(x_vals.tolist(), dtype=torch.float32)'

$changed = $false
foreach ($pattern in $patterns) {
    $newText = [regex]::Replace($text, $pattern, $replacement, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if ($newText -ne $text) {
        $text = $newText
        $changed = $true
    }
}

if (-not $changed) {
    throw "No target assignment pattern found in $TargetFile"
}

Set-Content -Path $TargetFile -Value $text -Encoding UTF8

Write-Host "[DONE] patched: $TargetFile"
Write-Host "[BACKUP] $backup"
Write-Host "[LINE] $replacement"
