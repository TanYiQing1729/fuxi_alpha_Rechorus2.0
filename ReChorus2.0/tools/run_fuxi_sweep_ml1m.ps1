Param(
  [string]$Python = "python",
  [string]$ReChorusRoot = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path,
  [string]$Gpu = "",

  # Fixed target for this sweep
  [string]$Dataset = "MovieLens_1M",
  [int[]]$Seeds = @(0),

  # Sweep knobs (override in CLI if you want larger grid)
  # Default grid is intentionally small and centered around the best-known ML1M settings we already used.
  # (Keeps runtime reasonable on CPU; you can still override from CLI.)
  [double[]]$Lrs = @(0.001, 0.0005),
  [double[]]$AttnDropouts = @(0.0, 0.1),
  # Include deeper stacks (e.g., 4 layers) but keep other dims small to avoid multi-day CPU sweeps.
  [int[]]$NumLayers = @(1, 2, 4),
  [int[]]$NumHeads = @(4),
  # Keep FFN size fixed at the commonly used value to control runtime while sweeping depth.
  [int[]]$FfHiddenSizes = @(256),
  [int[]]$SingleStageFfns = @(0),

  # Keep history_max fixed in this sweep to avoid mixing caches/architectures.
  # If you want to sweep history_max too, run this script twice with different HistoryMax.
  [int]$HistoryMax = 20,

  # Training/eval controls
  [string]$Appendix = "_ml1m_fuxi_sweep_s0",
  # Tag for output CSV filenames (defaults to Appendix; can include date/time)
  [string]$OutTag = "",
  [int]$Epoch = 30,
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
    [int]$NumLayer,
    [int]$NumHead,
    [int]$FfHiddenSize,
    [int]$SingleStageFfn,
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
    "--save_final_results", "0",

    "--lr", ([string]$Lr),
    "--attn_dropout", ([string]$AttnDropout),
    "--num_layers", ([string]$NumLayer),
    "--num_heads", ([string]$NumHead),
    "--ff_hidden_size", ([string]$FfHiddenSize),
    "--fuxi_single_stage_ffn", ([string]$SingleStageFfn),

    "--fuxi_use_pos", "1",
    "--fuxi_use_time", "1",
    "--fuxi_use_latent", "1",

    "--time_buckets", "32",
    "--time_bucket_max", "1000000"
  )

  if ($Gpu -ne "") {
    $cmdArgs += @("--gpu", $Gpu)
  }

  Write-Host (
    "\n[RUN] FuXiAlpha dataset={0} seed={1} lr={2} attn_dropout={3} layers={4} heads={5} ff_hidden={6} ssffn={7} hmax={8} ep={9} es={10} appendix={11}" -f 
      $Dataset, $Seed, $Lr, $AttnDropout, $NumLayer, $NumHead, $FfHiddenSize, $SingleStageFfn, $HistoryMax, $Epoch, $EarlyStop, $Appendix
  )

  & $Python @cmdArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Run failed: seed=$Seed lr=$Lr attn_dropout=$AttnDropout layers=$NumLayer heads=$NumHead ff_hidden=$FfHiddenSize ssffn=$SingleStageFfn"
  }
}

function Safe-Copy-If-Exists {
  Param(
    [string]$Src,
    [string]$Dst
  )

  if (Test-Path $Src) {
    $dstDir = Split-Path -Parent $Dst
    if ($dstDir -ne "" -and -not (Test-Path $dstDir)) {
      New-Item -ItemType Directory -Path $dstDir | Out-Null
    }
    Copy-Item -Force -Path $Src -Destination $Dst
  }
}

Push-Location $ReChorusRoot
try {
  Ensure-Dataset -Dataset $Dataset

  if ($OutTag -eq "") {
    $OutTag = $Appendix
  }
  if ($OutTag -eq "") {
    $OutTag = "_untagged"
  }
  # Ensure tag starts with '_' to make filenames readable.
  if (-not $OutTag.StartsWith("_")) {
    $OutTag = "_" + $OutTag
  }

  $total = ($Seeds.Count * $Lrs.Count * $AttnDropouts.Count * $NumLayers.Count * $NumHeads.Count * $FfHiddenSizes.Count * $SingleStageFfns.Count)
  Write-Host "\n[GRID] Planned sweep size (seed0 only, FuXiAlpha only)"
  Write-Host ("- Seeds: {0}" -f ($Seeds -join ","))
  Write-Host ("- Lrs: {0}" -f ($Lrs -join ","))
  Write-Host ("- AttnDropouts: {0}" -f ($AttnDropouts -join ","))
  Write-Host ("- NumLayers: {0}" -f ($NumLayers -join ","))
  Write-Host ("- NumHeads: {0}" -f ($NumHeads -join ","))
  Write-Host ("- FfHiddenSizes: {0}" -f ($FfHiddenSizes -join ","))
  Write-Host ("- SingleStageFfns: {0}" -f ($SingleStageFfns -join ","))
  Write-Host ("- HistoryMax: {0}" -f $HistoryMax)
  Write-Host ("=> Total combinations: {0}" -f $total)

  # Backup existing result files to avoid confusion.
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $defaultSweepCsv = Join-Path $ReChorusRoot ("results/fuxi_sweep_{0}.csv" -f $Dataset)
  $defaultAggCsv = Join-Path $ReChorusRoot "results/summary_agg.csv"
  $defaultRunsCsv = Join-Path $ReChorusRoot "results/summary_runs.csv"
  Safe-Copy-If-Exists -Src $defaultSweepCsv -Dst (Join-Path $ReChorusRoot ("results/backup/fuxi_sweep_{0}__backup_{1}.csv" -f $Dataset, $stamp))
  Safe-Copy-If-Exists -Src $defaultAggCsv -Dst (Join-Path $ReChorusRoot ("results/backup/summary_agg__backup_{0}.csv" -f $stamp))
  Safe-Copy-If-Exists -Src $defaultRunsCsv -Dst (Join-Path $ReChorusRoot ("results/backup/summary_runs__backup_{0}.csv" -f $stamp))

  $i = 0
  foreach ($seed in $Seeds) {
    foreach ($lr in $Lrs) {
      foreach ($ad in $AttnDropouts) {
        foreach ($nl in $NumLayers) {
          foreach ($nh in $NumHeads) {
            foreach ($ff in $FfHiddenSizes) {
              foreach ($ss in $SingleStageFfns) {
                $i += 1
                Write-Host ("\n[PROGRESS] {0}/{1}" -f $i, $total)
                Run-One -Lr $lr -AttnDropout $ad -NumLayer $nl -NumHead $nh -FfHiddenSize $ff -SingleStageFfn $ss -Seed $seed
              }
            }
          }
        }
      }
    }
  }

  Write-Host "\n[POST] Summarize sweep (FuXiAlpha only) -> results/fuxi_sweep_*.csv"
  & $Python "tools/summarize_fuxi_sweep.py" --rechorus_root "." --dataset $Dataset --appendix $Appendix
  if ($LASTEXITCODE -ne 0) { throw "summarize_fuxi_sweep.py failed" }

  Write-Host "\n[POST] Summarize all logs -> results/summary_*.csv"
  $summarizeOk = $true
  & $Python "tools/summarize_results.py" --rechorus_root "."
  if ($LASTEXITCODE -ne 0) {
    $summarizeOk = $false
    Write-Warning "summarize_results.py failed (often because results/summary_runs.csv is open in Excel/preview). You can close it and re-run: $Python tools/summarize_results.py --rechorus_root ."
  }

  # Copy the default CSV outputs to tagged names to avoid mixing with existing runs.
  $taggedSweepCsv = Join-Path $ReChorusRoot ("results/fuxi_sweep_{0}{1}.csv" -f $Dataset, $OutTag)
  $taggedAggCsv = Join-Path $ReChorusRoot ("results/summary_agg{0}.csv" -f $OutTag)
  $taggedRunsCsv = Join-Path $ReChorusRoot ("results/summary_runs{0}.csv" -f $OutTag)
  Safe-Copy-If-Exists -Src $defaultSweepCsv -Dst $taggedSweepCsv
  if ($summarizeOk) {
    Safe-Copy-If-Exists -Src $defaultAggCsv -Dst $taggedAggCsv
    Safe-Copy-If-Exists -Src $defaultRunsCsv -Dst $taggedRunsCsv
  }

  Write-Host "\n[DONE] MovieLens_1M FuXiAlpha sweep finished."
  Write-Host ("- {0}" -f $taggedSweepCsv)
  Write-Host ("- {0}" -f $taggedAggCsv)
  Write-Host ("- {0}" -f $taggedRunsCsv)
}
finally {
  Pop-Location
}
