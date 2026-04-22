# convert_encoding.ps1
$inputFile = "D:\urbanbus_rl_project\validation_output.txt"
$outputFile = "D:\urbanbus_rl_project\validation_output_utf8.txt"

if (Test-Path $inputFile) {
    Get-Content $inputFile | Set-Content -Path $outputFile -Encoding utf8
    Write-Host "Converted $inputFile to $outputFile (UTF-8)"
} else {
    Write-Error "$inputFile not found."
}
