$ErrorActionPreference = 'Stop'

function Show-Usage {
    Write-Host @"
Usage:
  .\run_build_dataset.ps1 -MaxSnapshots 1 -DryRun
  .\run_build_dataset.ps1 --max-snapshots 1 --dry-run
  .\run_build_dataset.ps1 --db-url "postgresql://..." --out-dir "C:\path\to\out"
  .\run_build_dataset.ps1 --db-driver pg8000

Supported options:
  -DbUrl | --db-url <value>
  -OutDir | --out-dir <value>
  -MaxSnapshots | --max-snapshots <int>
  -DryRun | --dry-run
  -NoUseMaterialized | --no-use-materialized
  --use-materialized
  --db-driver <pg8000|psycopg|psycopg2>
  -Help | --help
"@
}

function Normalize-OptionName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Token
    )

    return (($Token -replace '^[/-]+', '') -replace '[-_\s]', '').ToLowerInvariant()
}

function Require-NextValue {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Tokens,
        [Parameter(Mandatory = $true)]
        [int]$Index,
        [Parameter(Mandatory = $true)]
        [string]$OptionName
    )

    if (($Index + 1) -ge $Tokens.Count) {
        throw "Option '$OptionName' requires a value."
    }

    return $Tokens[$Index + 1]
}

function Normalize-DbUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value,
        [Parameter(Mandatory = $true)]
        [string]$Driver
    )

    $Result = $Value.Trim()
    if ($Result -eq '') { return $Result }

    # psycopg2/libpq 관련 Windows 인코딩 문제를 피하기 위해 기본 host는 127.0.0.1 사용
    $Result = [regex]::Replace($Result, '(?i)@localhost(?=[:/]|$)', '@127.0.0.1')

    switch ($Driver) {
        'pg8000' {
            $Result = $Result -replace '^(?i)postgresql\+psycopg2://', 'postgresql+pg8000://'
            $Result = $Result -replace '^(?i)postgresql\+psycopg://', 'postgresql+pg8000://'
            $Result = $Result -replace '^(?i)postgresql://', 'postgresql+pg8000://'
            # pg8000에는 psycopg/libpq 전용 client_encoding 파라미터를 붙이지 않음
            $Result = $Result -replace '([?&])client_encoding=[^&]*', '$1'
            $Result = $Result -replace '[?&]$', ''
            $Result = $Result -replace '\?&', '?'
        }
        'psycopg' {
            $Result = $Result -replace '^(?i)postgresql\+psycopg2://', 'postgresql+psycopg://'
            $Result = $Result -replace '^(?i)postgresql\+pg8000://', 'postgresql+psycopg://'
            $Result = $Result -replace '^(?i)postgresql://', 'postgresql+psycopg://'
        }
        default {
            if ($Result -notmatch '(?i)^postgresql\+psycopg2://') {
                $Result = $Result -replace '^(?i)postgresql\+psycopg://', 'postgresql+psycopg2://'
                $Result = $Result -replace '^(?i)postgresql\+pg8000://', 'postgresql+psycopg2://'
                $Result = $Result -replace '^(?i)postgresql://', 'postgresql+psycopg2://'
            }
            if ($Result -notmatch '(?i)([?&])client_encoding=') {
                if ($Result.Contains('?')) {
                    $Result += '&client_encoding=utf8'
                } else {
                    $Result += '?client_encoding=utf8'
                }
            }
        }
    }

    return $Result
}

$DbUrl = ''
$OutDir = ''
$MaxSnapshots = 0
$DryRun = $false
$NoUseMaterialized = $false
$ShowHelp = $false
$DbDriver = 'pg8000'
$UnknownArgs = New-Object System.Collections.Generic.List[string]

$Tokens = @($args)
$Index = 0

while ($Index -lt $Tokens.Count) {
    $Token = [string]$Tokens[$Index]

    if ([string]::IsNullOrWhiteSpace($Token)) {
        $Index++
        continue
    }

    $NamePart = $Token
    $InlineValue = $null

    if ($Token -match '^(--?|/)([^=]+)=(.*)$') {
        $NamePart = $Matches[1] + $Matches[2]
        $InlineValue = $Matches[3]
    }

    $Normalized = Normalize-OptionName -Token $NamePart

    switch ($Normalized) {
        'dburl' {
            $DbUrl = if ($null -ne $InlineValue) { $InlineValue } else { Require-NextValue -Tokens $Tokens -Index $Index -OptionName $Token }
            if ($null -eq $InlineValue) { $Index++ }
        }
        'outdir' {
            $OutDir = if ($null -ne $InlineValue) { $InlineValue } else { Require-NextValue -Tokens $Tokens -Index $Index -OptionName $Token }
            if ($null -eq $InlineValue) { $Index++ }
        }
        'maxsnapshots' {
            $RawValue = if ($null -ne $InlineValue) { $InlineValue } else { Require-NextValue -Tokens $Tokens -Index $Index -OptionName $Token }
            if ($null -eq $InlineValue) { $Index++ }

            $ParsedValue = 0
            if (-not [int]::TryParse($RawValue, [ref]$ParsedValue)) {
                throw "Option '$Token' requires an integer value, but received '$RawValue'."
            }
            $MaxSnapshots = $ParsedValue
        }
        'dryrun' {
            $DryRun = $true
        }
        'nousematerialized' {
            $NoUseMaterialized = $true
        }
        'usematerialized' {
            $NoUseMaterialized = $false
        }
        'dbdriver' {
            $DbDriver = if ($null -ne $InlineValue) { $InlineValue } else { Require-NextValue -Tokens $Tokens -Index $Index -OptionName $Token }
            if ($null -eq $InlineValue) { $Index++ }
            $DbDriver = $DbDriver.Trim().ToLowerInvariant()
            if (@('pg8000','psycopg','psycopg2') -notcontains $DbDriver) {
                throw "Option '$Token' must be one of: pg8000, psycopg, psycopg2. Received '$DbDriver'."
            }
        }
        'help' {
            $ShowHelp = $true
        }
        'h' {
            $ShowHelp = $true
        }
        default {
            $UnknownArgs.Add($Token) | Out-Null
        }
    }

    $Index++
}

if ($ShowHelp) {
    Show-Usage
    exit 0
}

if ($UnknownArgs.Count -gt 0) {
    Write-Host ''
    Write-Host '[ERROR] Unknown arguments detected:'
    foreach ($UnknownArg in $UnknownArgs) {
        Write-Host "  - $UnknownArg"
    }
    Write-Host ''
    Show-Usage
    exit 2
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = 'python'
$PythonScript = Join-Path $ScriptDir 'build_gatv2_dataset.py'
$LogStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogDir = Join-Path $ScriptDir ('logs_build_' + $LogStamp)
$LogFile = Join-Path $LogDir 'build.log'
$ConsoleLogFile = Join-Path $LogDir 'runner_console.log'

if (-not (Test-Path -LiteralPath $PythonScript)) {
    throw "Python dataset builder not found: $PythonScript"
}

$PythonCommand = Get-Command $PythonExe -ErrorAction SilentlyContinue
if ($null -eq $PythonCommand) {
    throw "Python executable not found in PATH. Install Python or adjust `$PythonExe in this script."
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# libpq가 읽는 일부 환경값이 로컬 인코딩 문제를 일으킬 수 있어 최소화
$env:PGCLIENTENCODING = 'UTF8'
$env:PYTHONUTF8 = '1'
foreach ($Name in @('PGSERVICE', 'PGSERVICEFILE', 'PGPASSFILE', 'PGSYSCONFDIR')) {
    if (Test-Path Env:$Name) {
        Remove-Item Env:$Name -ErrorAction SilentlyContinue
    }
}

$ArgsList = @(
    $PythonScript,
    '--log-dir', $LogDir
)

if ($DbUrl -eq '') {
    switch ($DbDriver) {
        'pg8000' { $DbUrl = 'postgresql+pg8000://postgres:password@127.0.0.1:5432/urbanbus' }
        'psycopg' { $DbUrl = 'postgresql+psycopg://postgres:password@127.0.0.1:5432/urbanbus' }
        default { $DbUrl = 'postgresql+psycopg2://postgres:password@127.0.0.1:5432/urbanbus?client_encoding=utf8' }
    }
} else {
    $DbUrl = Normalize-DbUrl -Value $DbUrl -Driver $DbDriver
}

$ArgsList += @('--db-url', $DbUrl)
if ($OutDir -ne '') {
    $ArgsList += @('--out-dir', $OutDir)
}
if ($MaxSnapshots -gt 0) {
    $ArgsList += @('--max-snapshots', "$MaxSnapshots")
}
if ($DryRun) {
    $ArgsList += '--dry-run'
}
if ($NoUseMaterialized) {
    $ArgsList += '--no-use-materialized'
} else {
    $ArgsList += '--use-materialized'
}

$RenderedArgs = $ArgsList | ForEach-Object {
    if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
}

Write-Host '============================================================='
Write-Host ' build_gatv2_dataset runner'
Write-Host "   python         = $($PythonCommand.Source)"
Write-Host "   script         = $PythonScript"
Write-Host "   max snapshots  = $MaxSnapshots"
Write-Host "   dry run        = $DryRun"
Write-Host "   db driver      = $DbDriver"
Write-Host "   db url         = $DbUrl"
Write-Host "   use materialized = $(-not $NoUseMaterialized)"
Write-Host "   log dir        = $LogDir"
Write-Host "   command        = $PythonExe $($RenderedArgs -join ' ')"
Write-Host '============================================================='

$Start = Get-Date
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $PythonExe @ArgsList 2>&1 | Tee-Object -FilePath $ConsoleLogFile
    $ExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}
$Elapsed = (Get-Date) - $Start

Write-Host ''
if ($ExitCode -eq 0) {
    Write-Host "[DONE] dataset build completed in $([math]::Round($Elapsed.TotalSeconds, 1))s"
    Write-Host " build log: $LogFile"
    Write-Host " runner log: $ConsoleLogFile"
    exit 0
}

Write-Host "[FAIL] dataset build failed in $([math]::Round($Elapsed.TotalSeconds, 1))s"
Write-Host " build log: $LogFile"
Write-Host " runner log: $ConsoleLogFile"
exit $ExitCode
