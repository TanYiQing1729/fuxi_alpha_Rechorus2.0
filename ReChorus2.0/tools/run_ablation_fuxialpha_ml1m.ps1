param(
  [int]$Epoch = 50,
  [int]$EarlyStop = 5,
  [string]$DataAppendix = "_ep50_abla",
  [int[]]$Seeds = @(0, 1, 2)
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (!(Test-Path $py)) {
  throw "Python not found at: $py"
}

Write-Host "[INFO] root=$root"
Write-Host "[INFO] python=$py"
Write-Host "[INFO] dataset=MovieLens_1M appendix=$DataAppendix epoch=$Epoch early_stop=$EarlyStop"

# Clean previous ablation artifacts
$logGlob = Join-Path $root "log\FuXiAlpha\*MovieLens_1M${DataAppendix}*"
if (Test-Path $logGlob) {
  Remove-Item -LiteralPath $logGlob -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host "[CLEAN] log removed: $logGlob"
}

$modelGlob = Join-Path $root "model\FuXiAlpha\*MovieLens_1M${DataAppendix}*"
if (Test-Path $modelGlob) {
  Remove-Item -LiteralPath $modelGlob -Force -ErrorAction SilentlyContinue
  Write-Host "[CLEAN] model removed: $modelGlob"
}

$cachePath = Join-Path $root "data\MovieLens_1M\SeqReader${DataAppendix}.pkl"
if (Test-Path $cachePath) {
  Remove-Item -LiteralPath $cachePath -Force -ErrorAction SilentlyContinue
  Write-Host "[CLEAN] cache removed: $cachePath"
}

function Run-One {
  param(
    [int]$Seed,
    [string]$Tag,
    [int]$UsePos,
    [int]$UseTime,
    [int]$SingleStageFfn
  )

  Write-Host "\n[ABLA] seed=$Seed tag=$Tag pos=$UsePos time=$UseTime ssffn=$SingleStageFfn" -ForegroundColor Cyan

  $args = @(
    "-u", "src\\main.py",
    "--model_name", "FuXiAlpha",
    "--dataset", "MovieLens_1M",
    "--data_appendix", $DataAppendix,
    "--random_seed", "$Seed",
    "--epoch", "$Epoch",
    "--early_stop", "$EarlyStop",
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
    "--fuxi_single_stage_ffn", "$SingleStageFfn"
  )

  & $py @args

  if ($LASTEXITCODE -ne 0) {
    throw "Run failed: seed=$Seed tag=$Tag"
  }
}

$variants = @(
  @{ tag = "full";   pos = 1; time = 1; ssffn = 0 },
  @{ tag = "wotime"; pos = 1; time = 0; ssffn = 0 },
  @{ tag = "wopos";  pos = 0; time = 1; ssffn = 0 },
  @{ tag = "ssffn1"; pos = 1; time = 1; ssffn = 1 }
)

foreach ($seed in $Seeds) {
  foreach ($v in $variants) {
    Run-One -Seed $seed -Tag $v.tag -UsePos $v.pos -UseTime $v.time -SingleStageFfn $v.ssffn
  }
}

Write-Host "\n[SUM] summarize_results.py" -ForegroundColor Yellow
& $py -u "tools\summarize_results.py" --rechorus_root $root
if ($LASTEXITCODE -ne 0) { throw "summarize_results.py failed" }

Write-Host "\n[PLOT] plot_results.py" -ForegroundColor Yellow
$datasetName = "MovieLens_1M$DataAppendix"
& $py -u "tools\plot_results.py" --rechorus_root $root --datasets $datasetName --models "FuXiAlpha" --split dev
if ($LASTEXITCODE -ne 0) { throw "plot_results.py (dev) failed" }
& $py -u "tools\plot_results.py" --rechorus_root $root --datasets $datasetName --models "FuXiAlpha" --split test
if ($LASTEXITCODE -ne 0) { throw "plot_results.py failed" }

Write-Host "\n[DONE] MovieLens_1M ablation finished." -ForegroundColor Green
