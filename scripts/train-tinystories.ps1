[CmdletBinding()]
param(
    [ValidateSet('en', 'fr')]
    [string]$Language = 'en',

    [string]$Config = 'configs/mini-17m-tinystories-512.yaml',
    [int]$Updates = 1000,
    [int]$EvalEvery = 50,
    [int]$CheckpointEvery = 250,
    [int]$ShuffleBuffer = 32,
    [int]$MaxValidationBatches = 20,
    [int]$SampleTokens = 32,

# Prepare corpora automatically when they do not exist.
    [switch]$PrepareData,

# English: download only the first complete stories up to these caps.
    [int]$EnglishTrainMB = 2,
    [int]$EnglishValidationMB = 1,

# French: btecsec/tinystories-fr currently exposes 20k translated stories.
# We fetch a deterministic prefix then split it with seed 42.
    [int]$FrenchTrainStories = 2500,
    [int]$FrenchValidationStories = 500,

    [switch]$RebuildTokenizer,
    [switch]$SkipTests,

# PR12 BPE is pedagogical and rescans the corpus for every merge.
# This switch acknowledges that a large vocab/corpus may be very slow.
    [switch]$AllowSlowTokenizer
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http

function Fail([string]$Message) {
    throw "[lxp-mini-1xm] $Message"
}

function Ensure-Directory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Get-RepoRoot {
    # Recommended location is <repo>/scripts/train-tinystories.ps1.
    $candidateFromScripts = Split-Path -Parent $PSScriptRoot
    if (Test-Path -LiteralPath (Join-Path $candidateFromScripts 'settings.gradle.kts')) {
        return (Resolve-Path $candidateFromScripts).Path
    }

    if (Test-Path -LiteralPath (Join-Path (Get-Location) 'settings.gradle.kts')) {
        return (Resolve-Path (Get-Location)).Path
    }

    Fail 'Run this script from the repository root, or place it in <repo>/scripts/.'
}

function ConvertTo-GradleApplicationArg([string]$Value) {
    # Gradle's application plugin receives every application argument through one
    # --args=<string> option. On Windows, gradlew.bat is reparsed by cmd.exe, so
    # embedding double quotes inside that string breaks values containing spaces
    # (for example: --prompt "Once upon a time").
    #
    # Keep the outer native argument intact and use single quotes *inside* the
    # Gradle --args value. Gradle then parses the quoted application argument.
    if ($Value -match "[\s']") {
        if ($Value.Contains("'")) {
            Fail "Application argument contains an unsupported apostrophe: $Value"
        }
        return "'$Value'"
    }
    return $Value
}

function Invoke-Gradle([string[]]$AppArgs) {
    $cliArgs = ($AppArgs | ForEach-Object { ConvertTo-GradleApplicationArg $_ }) -join ' '
    Write-Host ''
    Write-Host ">>> lxp-mini-1xm $cliArgs" -ForegroundColor Cyan

    # IMPORTANT: --args=... must be ONE native argument to gradlew.bat.
    & $script:GradleWrapper 'run' "--args=$cliArgs"
    if ($LASTEXITCODE -ne 0) {
        Fail "Gradle command failed with exit code $LASTEXITCODE"
    }
}

function Download-TextPrefixAtStoryBoundary(
        [string]$Url,
        [string]$Destination,
        [long]$MaximumBytes
) {
    Write-Host "Downloading up to $([Math]::Round($MaximumBytes / 1MB, 1)) MB from:`n  $Url"

    $handler = [System.Net.Http.HttpClientHandler]::new()
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromMinutes(30)

    try {
        $response = $client.GetAsync(
                $Url,
                [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead
        ).GetAwaiter().GetResult()
        $response.EnsureSuccessStatusCode() | Out-Null

        $stream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        try {
            $memory = [System.IO.MemoryStream]::new()
            try {
                $buffer = New-Object byte[] (1024 * 1024)
                while ($memory.Length -lt $MaximumBytes) {
                    $remaining = $MaximumBytes - $memory.Length
                    $wanted = [Math]::Min($buffer.Length, $remaining)
                    $read = $stream.Read($buffer, 0, [int]$wanted)
                    if ($read -le 0) { break }
                    $memory.Write($buffer, 0, $read)
                }

                $text = [System.Text.Encoding]::UTF8.GetString($memory.ToArray())
                $separator = '<|endoftext|>'
                $firstBoundary = $text.IndexOf($separator, [StringComparison]::Ordinal)
                $lastBoundary = $text.LastIndexOf($separator, [StringComparison]::Ordinal)
                if ($firstBoundary -lt 0 -or $lastBoundary -le $firstBoundary) {
                    Fail "Not enough complete TinyStories document boundaries were found in the downloaded prefix."
                }

                # TinyStories raw files can contain a partial snippet before the first marker.
                # Keep only complete stories between the first and last marker.
                $contentStart = $firstBoundary + $separator.Length
                $text = $text.Substring($contentStart, $lastBoundary - $contentStart).Trim()
                $text = $text + "`n$separator`n"
                [System.IO.File]::WriteAllText(
                        $Destination,
                        $text,
                        [System.Text.UTF8Encoding]::new($false)
                )
            }
            finally {
                $memory.Dispose()
            }
        }
        finally {
            $stream.Dispose()
            $response.Dispose()
        }
    }
    finally {
        $client.Dispose()
        $handler.Dispose()
    }
}

function Prepare-EnglishCorpus([string]$TrainPath, [string]$ValidationPath) {
    $trainUrl = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt'
    $validationUrl = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt'

    Download-TextPrefixAtStoryBoundary $trainUrl $TrainPath ([long]$EnglishTrainMB * 1MB)
    Download-TextPrefixAtStoryBoundary $validationUrl $ValidationPath ([long]$EnglishValidationMB * 1MB)
}

function Get-HuggingFaceFrenchStories([int]$Count) {
    $dataset = [Uri]::EscapeDataString('btecsec/tinystories-fr')
    $batchSize = 100
    $stories = [System.Collections.Generic.List[string]]::new()

    for ($offset = 0; $offset -lt $Count; $offset += $batchSize) {
        $length = [Math]::Min($batchSize, $Count - $offset)
        $uri = "https://datasets-server.huggingface.co/rows?dataset=$dataset&config=default&split=train&offset=$offset&length=$length"
        Write-Progress -Activity 'Downloading French TinyStories' -Status "$offset / $Count" -PercentComplete (($offset / [double]$Count) * 100)
        $result = Invoke-RestMethod -Uri $uri -Method Get

        foreach ($entry in $result.rows) {
            $text = [string]$entry.row.text
            if (-not [string]::IsNullOrWhiteSpace($text)) {
                $stories.Add($text.Trim())
            }
        }
    }

    Write-Progress -Activity 'Downloading French TinyStories' -Completed
    return $stories
}

function Prepare-FrenchCorpus([string]$TrainPath, [string]$ValidationPath) {
    $total = $FrenchTrainStories + $FrenchValidationStories
    if ($total -gt 20000) {
        Fail 'btecsec/tinystories-fr contains 20,000 rows; TrainStories + ValidationStories must be <= 20,000.'
    }

    $stories = Get-HuggingFaceFrenchStories $total
    if ($stories.Count -lt $total) {
        Fail "Expected $total French stories but received only $($stories.Count)."
    }

    # Deterministic Fisher-Yates shuffle, seed 42, before the split.
    $rng = [System.Random]::new(42)
    for ($i = $stories.Count - 1; $i -gt 0; $i--) {
        $j = $rng.Next($i + 1)
        $tmp = $stories[$i]
        $stories[$i] = $stories[$j]
        $stories[$j] = $tmp
    }

    $separator = "`n<|endoftext|>`n"
    $train = ($stories.GetRange(0, $FrenchTrainStories) -join $separator) + $separator
    $validation = ($stories.GetRange($FrenchTrainStories, $FrenchValidationStories) -join $separator) + $separator

    [System.IO.File]::WriteAllText($TrainPath, $train, [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::WriteAllText($ValidationPath, $validation, [System.Text.UTF8Encoding]::new($false))
}

$RepoRoot = Get-RepoRoot
Set-Location $RepoRoot

$script:GradleWrapper = if ($env:OS -eq 'Windows_NT') {
    Join-Path $RepoRoot 'gradlew.bat'
} else {
    Join-Path $RepoRoot 'gradlew'
}

if (-not (Test-Path -LiteralPath $script:GradleWrapper)) {
    Fail "Gradle wrapper not found: $script:GradleWrapper"
}
if (-not (Test-Path -LiteralPath $Config)) {
    Fail "Config not found: $Config"
}

# Read vocabSize from the selected YAML so tokenizer and model stay aligned.
$configText = Get-Content -LiteralPath $Config -Raw
$vocabMatch = [regex]::Match($configText, '(?m)^\s*vocabSize:\s*(\d+)\s*$')
if (-not $vocabMatch.Success) {
    Fail "Could not read model.vocabSize from $Config"
}
$vocabSize = [int]$vocabMatch.Groups[1].Value

$slug = if ($Language -eq 'en') { 'tinystories-en' } else { 'tinystories-fr' }
$dataDir = Join-Path $RepoRoot "data/prepared/$slug"
$artifactDir = Join-Path $RepoRoot "artifacts/tokenizers/$slug"
$runsDir = Join-Path $RepoRoot "runs/$slug"
Ensure-Directory $dataDir
Ensure-Directory $artifactDir
Ensure-Directory $runsDir

$trainCorpus = Join-Path $dataDir 'train.txt'
$validationCorpus = Join-Path $dataDir 'validation.txt'
$tokenizer = Join-Path $artifactDir "bpe-$vocabSize.json"
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$runDir = Join-Path $runsDir "$timestamp-$vocabSize"

if ($PrepareData -or -not (Test-Path -LiteralPath $trainCorpus) -or -not (Test-Path -LiteralPath $validationCorpus)) {
    Write-Host "Preparing $Language corpus..." -ForegroundColor Yellow
    if ($Language -eq 'en') {
        Prepare-EnglishCorpus $trainCorpus $validationCorpus
    } else {
        Prepare-FrenchCorpus $trainCorpus $validationCorpus
    }
}

if (-not (Test-Path -LiteralPath $trainCorpus)) { Fail "Train corpus missing: $trainCorpus" }
if (-not (Test-Path -LiteralPath $validationCorpus)) { Fail "Validation corpus missing: $validationCorpus" }

$trainBytes = (Get-Item -LiteralPath $trainCorpus).Length
$validationBytes = (Get-Item -LiteralPath $validationCorpus).Length

Write-Host ''
Write-Host '=== LXP Mini TinyStories training ===' -ForegroundColor Green
Write-Host "Language:             $Language"
Write-Host "Config:               $Config"
Write-Host "Vocabulary:           $vocabSize"
Write-Host "Train corpus:         $trainCorpus ($([Math]::Round($trainBytes / 1MB, 2)) MB)"
Write-Host "Validation corpus:    $validationCorpus ($([Math]::Round($validationBytes / 1MB, 2)) MB)"
Write-Host "Tokenizer:            $tokenizer"
Write-Host "Run directory:        $runDir"
Write-Host "Optimizer updates:    $Updates"
Write-Host "Validation max batch: $MaxValidationBatches"

# Important PR12 limitation: BpeTokenizerTrainer repeatedly scans the full token array.
# The guard is intentionally conservative. It protects against accidentally launching
# an 8192-vocab BPE build on tens/hundreds of MB and assuming the script is frozen.
if (-not (Test-Path -LiteralPath $tokenizer) -or $RebuildTokenizer) {
    if ($vocabSize -gt 1024 -and $trainBytes -gt 2MB -and -not $AllowSlowTokenizer) {
        Fail @"
Current PR12 BPE is pedagogical and rescans the corpus for each merge.
You selected vocabSize=$vocabSize on a $([Math]::Round($trainBytes / 1MB, 2)) MB corpus.
That can be extremely slow. Either:
  1) optimize the BPE trainer before the long 17M run (recommended),
  2) use a smaller config/corpus for a lab run, or
  3) rerun with -AllowSlowTokenizer to acknowledge the cost.
"@
    }
}

if (-not $SkipTests) {
    Write-Host ''
    Write-Host '>>> Running tests' -ForegroundColor Cyan
    & $script:GradleWrapper 'test'
    if ($LASTEXITCODE -ne 0) { Fail "Tests failed with exit code $LASTEXITCODE" }
}

Invoke-Gradle @('model', 'info', '--config', $Config)

if ($RebuildTokenizer -and (Test-Path -LiteralPath $tokenizer)) {
    Remove-Item -LiteralPath $tokenizer -Force
}

if (-not (Test-Path -LiteralPath $tokenizer)) {
    Invoke-Gradle @(
        'tokenizer', 'bpe', 'train',
        '--input', $trainCorpus,
        '--vocab-size', [string]$vocabSize,
        '--output', $tokenizer
    )
} else {
    Write-Host "Tokenizer already exists; reusing $tokenizer" -ForegroundColor DarkGray
}

$prompts = if ($Language -eq 'en') {
    @('Once upon a time', 'Lily', 'The little dog')
} else {
    @('Il était une fois', 'Lina', 'Le petit chien')
}

$trainAppArgs = [System.Collections.Generic.List[string]]::new()
@(
    'train', 'corpus',
    '--config', $Config,
    '--tokenizer', $tokenizer,
    '--train-corpus', $trainCorpus,
    '--validation-corpus', $validationCorpus,
    '--run-dir', $runDir,
    '--updates', [string]$Updates,
    '--eval-every', [string]$EvalEvery,
    '--checkpoint-every', [string]$CheckpointEvery,
    '--shuffle-buffer', [string]$ShuffleBuffer,
    '--max-validation-batches', [string]$MaxValidationBatches
) | ForEach-Object { $trainAppArgs.Add($_) }

foreach ($prompt in $prompts) {
    $trainAppArgs.Add('--prompt')
    $trainAppArgs.Add($prompt)
}

$trainAppArgs.Add('--sample-tokens')
$trainAppArgs.Add([string]$SampleTokens)

Invoke-Gradle $trainAppArgs.ToArray()

Invoke-Gradle @(
    'evaluate',
    '--run-dir', $runDir,
    '--validation-corpus', $validationCorpus,
    '--max-batches', [string]$MaxValidationBatches
)

Write-Host ''
Write-Host 'Training/evaluation completed.' -ForegroundColor Green
Write-Host "Run: $runDir"
Write-Host 'Useful files:'
Write-Host "  $runDir/config.yaml"
Write-Host "  $runDir/tokenizer.json"
Write-Host "  $runDir/metrics.jsonl"
Write-Host "  $runDir/checkpoints/"
Write-Host "  $runDir/samples/"