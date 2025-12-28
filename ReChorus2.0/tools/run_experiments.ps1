Param(
  [string]$Python = "python",
  [string]$ReChorusRoot = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path,
  [string]$Gpu = "",
  [string]$DataAppendix = "",
  [int[]]$Seeds = @(0,1,2),
  [string[]]$Datasets = @("Grocery_and_Gourmet_Food","MovieLens_1M","MovieLens_100K"),
  [string[]]$Baselines = @("SASRec","GRU4Rec","TiSASRec"),
  [int]$Epoch = 15,
  [int]$EarlyStop = 5,
  [int]$BatchSize = 256,
  [int]$EvalBatchSize = 256,
  [int]$NumWorkers = 0,
  [int]$HistoryMax = 20,
  [string]$FuXiLr = "",
  [string]$FuXiAttnDropout = "",
  [switch]$RunAblations,
  [switch]$Smoke
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Smoke) {
  $Seeds = @(0)
  $Epoch = 1
  $EarlyStop = 1
}

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
    [string]$ModelName,
    [string]$Dataset,
    [int]$Seed,
    [string[]]$ExtraArgs
  )

  $cmdArgs = @(
    "src/main.py",
    "--model_name", $ModelName,
    "--dataset", $Dataset,
    "--random_seed", $Seed,
    "--epoch", $Epoch,
    "--early_stop", $EarlyStop,
    "--batch_size", $BatchSize,
    "--eval_batch_size", $EvalBatchSize,
    "--num_workers", $NumWorkers,
    "--history_max", $HistoryMax,
    "--topk", "5,10,20",
    "--metric", "NDCG,HR"
  )

  if ($DataAppendix -ne "") {
    $cmdArgs += @("--data_appendix", $DataAppendix)
  }

  # Only include --gpu when the value is non-empty. Passing an empty string
  # does not survive argument splitting on Windows and will break argparse.
  if ($Gpu -ne "") {
    $cmdArgs += @("--gpu", $Gpu)
  }

  if ($null -ne $ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $cmdArgs += $ExtraArgs
  }

  Write-Host ("\n[RUN] {0}  dataset={1}  seed={2}  epoch={3}" -f $ModelName, $Dataset, $Seed, $Epoch)
  & $Python @cmdArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Run failed: model=$ModelName dataset=$Dataset seed=$Seed"
  }
}

Push-Location $ReChorusRoot
try {
  foreach ($dataset in $Datasets) {
    Ensure-Dataset -Dataset $dataset
    foreach ($seed in $Seeds) {
      foreach ($m in $Baselines) {
        Run-One -ModelName $m -Dataset $dataset -Seed $seed -ExtraArgs @()
      }

      if (-not $RunAblations) {
        $fuxiArgs = @(
          "--fuxi_use_pos", "1",
          "--fuxi_use_time", "1",
          "--fuxi_use_latent", "1",
          "--fuxi_single_stage_ffn", "0"
        )

        if ($FuXiLr -ne "") {
          $fuxiArgs += @("--lr", $FuXiLr)
        }
        if ($FuXiAttnDropout -ne "") {
          $fuxiArgs += @("--attn_dropout", $FuXiAttnDropout)
        }

        Run-One -ModelName "FuXiAlpha" -Dataset $dataset -Seed $seed -ExtraArgs $fuxiArgs
      } else {
        $ablations = @(
          @{ name = "full"; args = @("--fuxi_use_pos","1","--fuxi_use_time","1","--fuxi_use_latent","1","--fuxi_single_stage_ffn","0") },
          @{ name = "wotime"; args = @("--fuxi_use_pos","1","--fuxi_use_time","0","--fuxi_use_latent","1","--fuxi_single_stage_ffn","0") },
          @{ name = "wopos"; args = @("--fuxi_use_pos","0","--fuxi_use_time","1","--fuxi_use_latent","1","--fuxi_single_stage_ffn","0") },
          @{ name = "ssffn1"; args = @("--fuxi_use_pos","1","--fuxi_use_time","1","--fuxi_use_latent","1","--fuxi_single_stage_ffn","1") }
        )

        foreach ($ab in $ablations) {
          $fuxiArgs = @()
          $fuxiArgs += $ab.args
          if ($FuXiLr -ne "") {
            $fuxiArgs += @("--lr", $FuXiLr)
          }
          if ($FuXiAttnDropout -ne "") {
            $fuxiArgs += @("--attn_dropout", $FuXiAttnDropout)
          }
          Run-One -ModelName "FuXiAlpha" -Dataset $dataset -Seed $seed -ExtraArgs $fuxiArgs
        }
      }
    }
  }

  Write-Host "\n[POST] Summarize logs -> results/*.csv"
  & $Python "tools/summarize_results.py" --rechorus_root "."
  if ($LASTEXITCODE -ne 0) { throw "summarize_results.py failed" }

  Write-Host "\n[POST] Plot (test split, HR@10 & NDCG@10) -> figures/*.png"
  $plotModels = @()
  $plotModels += $Baselines
  $plotModels += @("FuXiAlpha")
  $plotDatasets = @()
  foreach ($ds in $Datasets) {
    if ($DataAppendix -ne "") {
      $plotDatasets += ($ds + $DataAppendix)
    } else {
      $plotDatasets += $ds
    }
  }
  $plotArgs = @(
    "tools/plot_results.py",
    "--rechorus_root", ".",
    "--split", "test",
    "--datasets"
  )
  $plotArgs += $plotDatasets
  $plotArgs += @("--models")
  $plotArgs += $plotModels
  $plotArgs += @("--metric", "HR@10", "NDCG@10")
  & $Python @plotArgs
  if ($LASTEXITCODE -ne 0) { throw "plot_results.py failed" }

  Write-Host "\n[DONE] Experiments finished."
  Write-Host "- results/summary_runs.csv"
  Write-Host "- results/summary_agg.csv"
  Write-Host "- figures/*__test__HR@10_NDCG@10.png"
}
finally {
  Pop-Location
}
