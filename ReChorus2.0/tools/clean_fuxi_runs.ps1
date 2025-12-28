Param(
  [string]$ReChorusRoot = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path,
  [string]$Dataset = "MovieLens_100K",
  [switch]$AllDatasets
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveRoot = Join-Path $ReChorusRoot (Join-Path "results" (Join-Path "archive" $stamp))
New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null

function Move-IfExists {
  Param([string]$Src, [string]$DstDir)
  if (Test-Path $Src) {
    New-Item -ItemType Directory -Force -Path $DstDir | Out-Null
    Move-Item -Force $Src $DstDir
  }
}

function Move-DirContents {
  Param([string]$Dir, [string]$DstDir)
  if (-not (Test-Path $Dir)) { return }
  $items = Get-ChildItem -LiteralPath $Dir -Force -ErrorAction SilentlyContinue
  if ($null -eq $items -or $items.Count -eq 0) { return }
  New-Item -ItemType Directory -Force -Path $DstDir | Out-Null
  foreach ($it in $items) {
    Move-Item -Force $it.FullName $DstDir
  }
}

# 1) Archive results tables in ReChorus2.0/results
Move-IfExists (Join-Path $ReChorusRoot (Join-Path "results" "summary_runs.csv")) $archiveRoot
Move-IfExists (Join-Path $ReChorusRoot (Join-Path "results" "summary_agg.csv")) $archiveRoot
Move-IfExists (Join-Path $ReChorusRoot (Join-Path "results" "fuxi_sweep_MovieLens_100K.csv")) $archiveRoot

# 2) Archive FuXiAlpha logs & models (both ReChorus2.0/* and parent */*)
$parent = Resolve-Path (Join-Path $ReChorusRoot "..")
$targets = @(
  (Join-Path $ReChorusRoot "log") + "\\FuXiAlpha",
  (Join-Path $ReChorusRoot "model") + "\\FuXiAlpha",
  (Join-Path $parent "log") + "\\FuXiAlpha",
  (Join-Path $parent "model") + "\\FuXiAlpha"
)

foreach ($t in $targets) {
  if (-not (Test-Path $t)) { continue }
  $dst = Join-Path $archiveRoot (Split-Path $t -Leaf)
  if ($AllDatasets) {
    Move-DirContents -Dir $t -DstDir $dst
  } else {
    # move only files matching dataset in name
    $items = Get-ChildItem -LiteralPath $t -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -match $Dataset }
    if ($null -eq $items -or $items.Count -eq 0) { continue }
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    foreach ($it in $items) {
      Move-Item -Force $it.FullName $dst
    }
  }
}

Write-Host "[CLEAN] Archived old FuXiAlpha runs/results to:" $archiveRoot
Write-Host "- Tip: use -AllDatasets to wipe FuXiAlpha across all datasets."
