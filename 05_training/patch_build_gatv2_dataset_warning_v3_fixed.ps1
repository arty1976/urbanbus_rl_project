
param(
    [string]$TargetFile = ".\build_gatv2_dataset.py",
    [string]$BackupFile = ".\build_gatv2_dataset.py.bak_warning_v3"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $TargetFile)) {
    throw "Target file not found: $TargetFile"
}

# Restore from backup first if available, so we start from a known-good state
if (Test-Path $BackupFile) {
    Copy-Item $BackupFile $TargetFile -Force
    Write-Host "[RESTORE] restored from backup: $BackupFile"
} else {
    Write-Host "[WARN] backup not found, patching current file in place"
}

$content = Get-Content $TargetFile -Raw -Encoding UTF8

# Remove bad accidental lines from previous patch attempts
$content = $content -replace '(?m)^[ \t]*numpy as np\s*[\r]?\n', ''
$content = $content -replace '(?m)^[ \t]+import numpy as np\s*[\r]?\n', ''

# Ensure a proper top-level import exists
if ($content -notmatch '(?m)^import numpy as np\s*$' -and $content -notmatch '(?m)^from numpy import ') {
    $lines = $content -split "`r?`n"
    $insertAt = 0
    for ($i = 0; $i -lt $lines.Length; $i++) {
        if ($lines[$i] -match '^(from\s+\S+\s+import\s+.+|import\s+.+)$') {
            $insertAt = $i + 1
            continue
        }
        break
    }
    $before = @()
    if ($insertAt -gt 0) { $before = $lines[0..($insertAt-1)] }
    $after = @()
    if ($insertAt -lt $lines.Length) { $after = $lines[$insertAt..($lines.Length-1)] }
    $content = (($before + 'import numpy as np' + $after) -join "`r`n")
    Write-Host "[PATCH] inserted: import numpy as np"
} else {
    Write-Host "[SKIP] numpy import already present"
}

# Replace the warning-prone assignment with a safer copy-based tensor construction
$patterns = @(
    'x\[node_indices\]\s*=\s*torch\.from_numpy\(x_vals\.copy\(\)\)',
    'x\[node_indices\]\s*=\s*torch\.from_numpy\(x_vals\)',
    'x\[node_indices\]\s*=\s*torch\.tensor\(np\.array\(x_vals,\s*copy=True\),\s*dtype=torch\.float32\)'
)

$replacement = 'x[node_indices] = torch.tensor(np.array(x_vals, copy=True), dtype=torch.float32)'

$changed = $false
foreach ($p in $patterns[0..1]) {
    $newContent = [regex]::Replace($content, $p, $replacement)
    if ($newContent -ne $content) {
        $content = $newContent
        $changed = $true
    }
}

Set-Content $TargetFile -Value $content -Encoding UTF8

# Show the final target line for verification
$line = Select-String -Path $TargetFile -Pattern 'x\[node_indices\]\s*=' | Select-Object -First 1
Write-Host "[DONE] patched: $TargetFile"
if (Test-Path $BackupFile) { Write-Host "[BACKUP] $BackupFile" }
if ($line) {
    Write-Host "[LINE] $($line.Line.Trim())"
}
