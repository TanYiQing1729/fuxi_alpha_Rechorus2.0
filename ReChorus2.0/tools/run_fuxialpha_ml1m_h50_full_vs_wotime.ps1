param(
  [int]$Epoch = 50,
  [int]$EarlyStop = 5,
  [int]$HistoryMax = 50,
  [string]$DataAppendix = "_ep50_h50_full_vs_wotime",
  [int[]]$Seeds = @(0, 1, 2)
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (!(Test-Path $py)) {
  throw "Python not found at: $py"
}

$dataset = "MovieLens_1M"

Write-Host "[INFO] root=$root"
Write-Host "[INFO] python=$py"
Write-Host "[INFO] dataset=$dataset appendix=$DataAppendix epoch=$Epoch early_stop=$EarlyStop history_max=$HistoryMax"

# Clean previous artifacts for this appendix
$logGlob = Join-Path $root "log\FuXiAlpha\*${dataset}${DataAppendix}*"
if (Test-Path $logGlob) {
  Remove-Item -LiteralPath $logGlob -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host "[CLEAN] log removed: $logGlob"
}

$modelGlob = Join-Path $root "model\FuXiAlpha\*${dataset}${DataAppendix}*"
if (Test-Path $modelGlob) {
  Remove-Item -LiteralPath $modelGlob -Force -ErrorAction SilentlyContinue
  Write-Host "[CLEAN] model removed: $modelGlob"
}

$cachePath = Join-Path $root "data\${dataset}\SeqReader${DataAppendix}.pkl"
if (Test-Path $cachePath) {
  Remove-Item -LiteralPath $cachePath -Force -ErrorAction SilentlyContinue
  Write-Host "[CLEAN] cache removed: $cachePath"
}

function Run-One {
  param(
    [int]$Seed,
    [string]$Tag,
    [int]$UsePos,
    [int]$UseTime
  )

  Write-Host "\n[RUN] seed=$Seed tag=$Tag pos=$UsePos time=$UseTime" -ForegroundColor Cyan

  $args = @(
    "-u", "src\\main.py",
    "--model_name", "FuXiAlpha",
    "--dataset", $dataset,
    "--data_appendix", $DataAppendix,
    "--random_seed", "$Seed",
    "--epoch", "$Epoch",
    "--early_stop", "$EarlyStop",
    "--history_max", "$HistoryMax",
    "--lr", "0.001",
    "--l2", "0",
    "--emb_size", "64",
    "--num_layers", "1",
    "--num_heads", "4",
    "--time_buckets", "32",
    "--time_bucket_max", "1000000",
    "--ff_hidden_size", "256",
    "--attn_dropout", "0.0",
    "--fuxi_use_pos", "$UsePos",
    "--fuxi_use_time", "$UseTime",
    "--fuxi_use_latent", "1",
    "--fuxi_single_stage_ffn", "0"
  )

  & $py @args
  if ($LASTEXITCODE -ne 0) {
    throw "Run failed: seed=$Seed tag=$Tag"
  }
}

foreach ($seed in $Seeds) {
  Run-One -Seed $seed -Tag "full" -UsePos 1 -UseTime 1
  Run-One -Seed $seed -Tag "wotime" -UsePos 1 -UseTime 0
}

Write-Host "\n[SUM] summarize_results.py" -ForegroundColor Yellow
& $py -u "tools\summarize_results.py" --rechorus_root $root
if ($LASTEXITCODE -ne 0) { throw "summarize_results.py failed" }

Write-Host "\n[PLOT] plot_results.py" -ForegroundColor Yellow
$datasetName = "$dataset$DataAppendix"
& $py -u "tools\plot_results.py" --rechorus_root $root --datasets $datasetName --models "FuXiAlpha" --split test
if ($LASTEXITCODE -ne 0) { throw "plot_results.py failed" }

Write-Host "\n[DONE] $datasetName full vs wotime finished." -ForegroundColor Green
