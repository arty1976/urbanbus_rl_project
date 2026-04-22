# get_project_diff.ps1
Write-Host "Getting project differences..."
$outputFile = "project_diff_report.txt"

"=== GIT STATUS ===" | Out-File -FilePath $outputFile -Encoding utf8
git status | Out-File -FilePath $outputFile -Encoding utf8 -Append

"=== GIT DIFF (Uncommitted Changes) ===" | Out-File -FilePath $outputFile -Encoding utf8 -Append
git diff | Out-File -FilePath $outputFile -Encoding utf8 -Append

"=== GIT DIFF (Latest Commit vs Previous Commit) ===" | Out-File -FilePath $outputFile -Encoding utf8 -Append
git diff HEAD~1 HEAD | Out-File -FilePath $outputFile -Encoding utf8 -Append

Write-Host "Report generated at: $outputFile"
