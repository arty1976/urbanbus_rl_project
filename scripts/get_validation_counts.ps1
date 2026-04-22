$file = "d:\urbanbus_rl_project\validation_output_utf8.txt"
$lines = Get-Content $file -Encoding UTF8

$results = @()
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^\((\d+) rows?\)') {
        $rowCount = $matches[1]

        # Look for the first data line to get check_name value
        $checkName = ""
        for ($k = $i - 1; $k -ge [Math]::Max(0, $i - 5); $k--) {
            if ($lines[$k] -match '^\s+(\w+)\s*\|') {
                $checkName = $matches[1]
                break
            }
        }
        $results += [PSCustomObject]@{
            LineNum    = $i + 1
            RowCount   = [int]$rowCount
            CheckName  = $checkName
        }
    }
}

$results | Format-Table -AutoSize
