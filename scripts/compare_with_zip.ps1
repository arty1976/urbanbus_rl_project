# compare_with_zip.ps1
$zipPath = "D:\urbanbus_rl_project (2).zip"
$extractPath = "D:\urbanbus_rl_project_temp_compare"
$currentProject = "D:\urbanbus_rl_project"
$reportFile = "D:\urbanbus_rl_project\project_diff_report.txt"

Write-Host "1/4. Cleaning up previous extractions if any..."
if (Test-Path $extractPath) {
    Remove-Item -Path $extractPath -Recurse -Force
}

Write-Host "2/4. Extracting the zip file..."
Expand-Archive -LiteralPath $zipPath -DestinationPath $extractPath -Force

# 압축 해제된 폴더 내부에 'urbanbus_rl_project' 폴더가 한번 더 감싸져 있는지 확인
$targetComparePath = $extractPath
$subDirs = Get-ChildItem -Path $extractPath -Directory
if ($subDirs.Count -eq 1) {
    $targetComparePath = $subDirs[0].FullName
}

Write-Host "3/4. Generating git diff report (This compares the Extracted Zip with the Current Folder)..."
# 비교시 project_diff_report.txt 자기 자신과 artifacts 폴더, 로그 폴더 등 일부 캐시 데이터는 무시하는 것이 좋습니다.
# 여기서는 가장 간단하게 전체를 비교합니다.
cmd.exe /c "git diff --no-index `"$targetComparePath`" `"$currentProject`" > `"$reportFile`""

Write-Host "4/4. Cleaning up temporary folder..."
if (Test-Path $extractPath) {
    Remove-Item -Path $extractPath -Recurse -Force
}

Write-Host "Done! Report generated at: $reportFile"
