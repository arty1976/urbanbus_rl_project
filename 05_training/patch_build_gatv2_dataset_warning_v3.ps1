param(
    [string]$TargetFile = ".\build_gatv2_dataset.py"
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $TargetFile)) {
    throw "Target file not found: $TargetFile"
}

$backup = "$TargetFile.bak_warning_v3"
Copy-Item $TargetFile $backup -Force

$content = Get-Content $TargetFile -Raw -Encoding UTF8

$patterns = @(
    'x\[node_indices\]\s*=\s*torch\.from_numpy\(x_vals\.copy\(\)\)',
    'x\[node_indices\]\s*=\s*torch\.from_numpy\(x_vals\)',
    'x\[node_indices\]\s*=\s*torch\.as_tensor\(x_vals.*?\)'
)

$replacement = 'x[node_indices] = torch.tensor(np.array(x_vals, copy=True), dtype=torch.float32)'
$changed = $false

foreach ($pattern in $patterns) {
    $newContent = [regex]::Replace($content, $pattern, $replacement)
    if ($newContent -ne $content) {
        $content = $newContent
        $changed = $true
        break
    }
}

if ($content -notmatch 'import numpy as np') {
    if ($content -match 'import numpy\s*$') {
        $content = $content -replace 'import numpy\s*$', 'import numpy`r`nimport numpy as np'
    } elseif ($content -match 'import numpy as numpy') {
        $content = $content -replace 'import numpy as numpy', 'import numpy as numpy`r`nimport numpy as np'
    } else {
        # Insert after the first import block if possible
        $content = [regex]::Replace($content, '(^((from|import)\s+.+\r?\n)+)', "$1import numpy as np`r`n", 1)
        if ($content -notmatch 'import numpy as np') {
            $content = "import numpy as np`r`n" + $content
        }
    }
    $changed = $true
}

Set-Content -Path $TargetFile -Value $content -Encoding UTF8

Write-Host "[DONE] patched: $TargetFile"
Write-Host "[BACKUP] $backup"
if (-not $changed) {
    Write-Host "[WARN] No matching warning line pattern was found. numpy alias check may still have updated the file."
}

# Show the final assignment line for verification
Select-String -Path $TargetFile -Pattern 'x\[node_indices\].*torch\.' | ForEach-Object {
    Write-Host "[LINE] $($_.Line.Trim())"
}
