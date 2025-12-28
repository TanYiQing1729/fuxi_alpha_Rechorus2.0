Param(
  [string]$Python = "python",
  [string]$ReChorusRoot = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path,
  [string]$Gpu = "",
  [string]$Dataset = "MovieLens_100K",
  [int[]]$Seeds = @(0,1,2),
  [double[]]$Lrs = @(0.001, 0.0005),
  [double[]]$AttnDropouts = @(0.0, 0.1, 0.2),
  [int]$HistoryMax = 20,
  [string]$Appendix = "fuxi_sweep",
  [int]$Epoch = 25,
  [int]$EarlyStop = 5,
  [int]$BatchSize = 256,
  [int]$EvalBatchSize = 256,
  [int]$NumWorkers = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dataset {
  Param([string]$Dataset)

  $dsDir = Join-Path $ReChorusRoot (Join-Path "data" $Dataset)
  $trainPath = Join-Path $dsDir "train.csv"
  $devPath = Join-Path $dsDir "dev.csv"
  $testPath = Join-Path $dsDir "test.csv"

  if ((Test-Path $trainPath) -and (Test-Path $devPath) -and (Test-Path $testPath)) {
    return
  }

  $builder = Join-Path $dsDir "build_topk.py"
  if (Test-Path $builder) {
    Write-Host "[DATA] Missing csv for $Dataset. Running: data/$Dataset/build_topk.py"
    & $Python $builder
    if ($LASTEXITCODE -ne 0) {
      throw "Dataset build failed: $Dataset"
    }
  }

  if (-not ((Test-Path $trainPath) -and (Test-Path $devPath) -and (Test-Path $testPath))) {
    throw "Dataset not ready (need train/dev/test.csv): $Dataset"
  }
}

function Run-One {
  Param(
    [double]$Lr,
    [double]$AttnDropout,
    [int]$Seed
  )

  $cmdArgs = @(
    "src/main.py",
    "--model_name", "FuXiAlpha",
    "--dataset", $Dataset,
    "--data_appendix", $Appendix,
    "--random_seed", $Seed,
    "--epoch", $Epoch,
    "--early_stop", $EarlyStop,
    "--batch_size", $BatchSize,
    "--eval_batch_size", $EvalBatchSize,
    "--num_workers", $NumWorkers,
    "--history_max", $HistoryMax,
    "--topk", "5,10,20",
    "--metric", "NDCG,HR",
    "--lr", ([string]$Lr),
    "--attn_dropout", ([string]$AttnDropout),
    "--fuxi_use_pos", "1",
    "--fuxi_use_time", "1",
    "--fuxi_use_latent", "1",
    "--fuxi_single_stage_ffn", "0"
  )

  if ($Gpu -ne "") {
    $cmdArgs += @("--gpu", $Gpu)
  }

  Write-Host ("\n[RUN] FuXiAlpha  dataset={0}  seed={1}  lr={2}  attn_dropout={3}  epoch={4}  early_stop={5}  appendix={6}" -f $Dataset, $Seed, $Lr, $AttnDropout, $Epoch, $EarlyStop, $Appendix)
  & $Python @cmdArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Run failed: dataset=$Dataset seed=$Seed lr=$Lr attn_dropout=$AttnDropout"
  }
}

Push-Location $ReChorusRoot
try {
  Ensure-Dataset -Dataset $Dataset

  foreach ($seed in $Seeds) {
    foreach ($lr in $Lrs) {
      foreach ($ad in $AttnDropouts) {
        Run-One -Lr $lr -AttnDropout $ad -Seed $seed
      }
    }
  }

  Write-Host "\n[POST] Summarize sweep (FuXiAlpha only) -> results/fuxi_sweep_*.csv"
  & $Python "tools/summarize_fuxi_sweep.py" --rechorus_root "." --dataset $Dataset --appendix $Appendix
  if ($LASTEXITCODE -ne 0) { throw "summarize_fuxi_sweep.py failed" }

  Write-Host "\n[POST] Summarize all logs -> results/summary_*.csv"
  & $Python "tools/summarize_results.py" --rechorus_root "."
  if ($LASTEXITCODE -ne 0) { throw "summarize_results.py failed" }

  Write-Host "\n[POST] Plot baseline vs FuXiAlpha (test split, HR@10 & NDCG@10) -> figures/*.png"
  $plotDataset = $Dataset
  if ($Appendix -ne "") {
    $plotDataset = ($Dataset + $Appendix)
  }
  & $Python "tools/plot_results.py" --rechorus_root "." --split "test" --datasets $plotDataset --models "SASRec" "GRU4Rec" "TiSASRec" "FuXiAlpha" --metric "HR@10" "NDCG@10"
  if ($LASTEXITCODE -ne 0) { throw "plot_results.py failed" }

  Write-Host "\n[DONE] FuXiAlpha sweep finished."
  Write-Host "- results/fuxi_sweep_${Dataset}.csv"
  Write-Host "- results/summary_runs.csv"
  Write-Host "- results/summary_agg.csv"
}
finally {
  Pop-Location
}
