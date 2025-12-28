param(
	[switch]$DeleteRecCsv = $true,
	[switch]$DeleteSeqReaderPickle = $false,
	[switch]$DeleteModelCheckpoints = $false,
	[switch]$DeleteArchivedResults = $false,
	[int]$KeepLatestArchives = 2,
	[switch]$DryRun = $false
)

$ErrorActionPreference = 'Stop'

function Get-RepoRoot {
	# This script lives in ReChorus2.0/tools. Repo root is two levels up.
	# $PSScriptRoot is reliable for scripts executed via -File.
	$here = $PSScriptRoot
	return (Resolve-Path (Join-Path $here '..\..')).Path
}

function Get-TotalBytes($files) {
	$sum = 0
	foreach ($f in @($files)) {
		if ($null -ne $f -and $null -ne $f.Length) { $sum += [int64]$f.Length }
	}
	return [int64]$sum
}

function Format-Bytes([long]$bytes) {
	if ($bytes -ge 1GB) { return ('{0:N2} GB' -f ($bytes / 1GB)) }
	if ($bytes -ge 1MB) { return ('{0:N2} MB' -f ($bytes / 1MB)) }
	if ($bytes -ge 1KB) { return ('{0:N2} KB' -f ($bytes / 1KB)) }
	return "$bytes B"
}

$repoRoot = Get-RepoRoot
$rechorusRoot = (Resolve-Path (Join-Path $repoRoot 'ReChorus2.0')).Path

Write-Host "RepoRoot: $repoRoot"
Write-Host "ReChorus : $rechorusRoot"

$targets = @()

if ($DeleteRecCsv) {
	$targets += @{ Name = 'rec-*.csv'; Paths = @(
		(Join-Path $repoRoot 'log'),
		(Join-Path $rechorusRoot 'log')
	) ; Filter = 'rec-*.csv' ; Recurse = $true }
}

if ($DeleteSeqReaderPickle) {
	$targets += @{ Name = 'SeqReader_*.pkl'; Paths = @(
		(Join-Path $rechorusRoot 'data')
	) ; Filter = 'SeqReader_*.pkl' ; Recurse = $true }
}

if ($DeleteModelCheckpoints) {
	$targets += @{ Name = '*.pt'; Paths = @(
		(Join-Path $repoRoot 'model'),
		(Join-Path $rechorusRoot 'model')
	) ; Filter = '*.pt' ; Recurse = $true }
}

if ($DeleteArchivedResults) {
	$targets += @{ Name = 'results/archive/*'; Paths = @(
		(Join-Path $rechorusRoot 'results\archive')
	) ; Filter = '*' ; Recurse = $false }
}

$allFiles = New-Object System.Collections.Generic.List[System.IO.FileInfo]
$allDirsToDelete = @()

foreach ($t in $targets) {
	foreach ($p in $t.Paths) {
		if (-not (Test-Path $p)) { continue }
		if ($t.Name -eq 'results/archive/*') {
			# Keep latest archives by folder name
			$dirs = Get-ChildItem -LiteralPath $p -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
			$toDelete = $dirs | Select-Object -Skip $KeepLatestArchives
			$allDirsToDelete += $toDelete
			continue
		}

		$files = Get-ChildItem -LiteralPath $p -Filter $t.Filter -File -Recurse:$t.Recurse -ErrorAction SilentlyContinue
		foreach ($f in $files) { $allFiles.Add($f) }
	}
}

# De-dup by full path
$uniqueFiles = $allFiles | Sort-Object FullName -Unique
$bytes = Get-TotalBytes $uniqueFiles

Write-Host "Will delete files: $($uniqueFiles.Count)  (" -NoNewline
Write-Host (Format-Bytes $bytes) -NoNewline
Write-Host ")"
if ($allDirsToDelete.Count -gt 0) {
	Write-Host "Will delete archive dirs: $($allDirsToDelete.Count) (keeping latest $KeepLatestArchives)"
}

if ($DryRun) {
	Write-Host "DryRun=ON. Nothing deleted."
	return
}

foreach ($f in $uniqueFiles) {
	Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
}

foreach ($d in ($allDirsToDelete | Sort-Object FullName -Unique)) {
	Remove-Item -LiteralPath $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Done. Deleted files: $($uniqueFiles.Count). Freed: $(Format-Bytes $bytes)"
