# ReChorus2.0/tools

这里放“实验汇总 + 可视化 + 写报告用的数据整理”脚本。

## 1) 汇总日志为表格

从 `ReChorus2.0/log/**.txt` 里抽取最终 `Dev/Test After Training` 指标。

```bash
# 在 ReChorus2.0 目录下运行
C:/Users/陈永鸿/.conda/envs/pcrec/python.exe tools/summarize_results.py --rechorus_root .
```

输出：
- `results/summary_runs.csv`：每个 run（每个 seed / 超参文件）一行（含 dev/test 两种 split）
- `results/summary_agg.csv`：按 (model,dataset,variant,split) 聚合后的均值/标准差（用于画误差棒）

## 2) 画图（柱状图 + 误差棒）

```bash
C:/Users/陈永鸿/.conda/envs/pcrec/python.exe tools/plot_results.py --rechorus_root . --split test --metric HR@10 NDCG@10
```

默认输出到 `ReChorus2.0/figures/`。

## 2.2) 画学习曲线（每个 epoch 的 dev 指标）

当你担心 “3 epoch 不够/不收敛” 时，建议画学习曲线来观察 dev 指标是否在上升以及何时趋于稳定。

该脚本会从日志中解析每个 epoch 的 `dev=(HR@K:...,NDCG@K:...)` 行，并按 (dataset, model) 生成曲线图。

- 输出目录：`ReChorus2.0/figures/learning_curves/`
- 默认扫描日志目录：`./log` 和 `../log`（兼容 ReChorus 默认把日志写到上级目录）

```bash
C:/Users/陈永鸿/.conda/envs/pcrec/python.exe tools/plot_learning_curves.py --rechorus_root . --datasets Grocery_and_Gourmet_Food MovieLens_1M MovieLens_100K --models SASRec GRU4Rec TiSASRec FuXiAlpha --metric HR@10 NDCG@10
```

## 2.5) 一键批量跑实验（推荐）

我们提供了一个 PowerShell 脚本把“跑模型 → 汇总 → 画图”串起来：

```bash
# 在 ReChorus2.0 目录下运行
./tools/run_experiments.ps1 -Python C:/Users/陈永鸿/.conda/envs/pcrec/python.exe
```

默认会跑：
- 3 个数据集：`Grocery_and_Gourmet_Food`、`MovieLens_1M`、`MovieLens_100K`
- 3 个 baselines（同为 sequential）：`SASRec`、`GRU4Rec`、`TiSASRec`
- 目标模型：`FuXiAlpha`
- 3 个 seed：0/1/2

如果你只想快速验证流程（冒烟测试）：

```bash
./tools/run_experiments.ps1 -Python C:/Users/陈永鸿/.conda/envs/pcrec/python.exe -Smoke
```

说明：在序列推荐里，`3 epoch` 往往远远不够（尤其是 Grocery 这类稀疏数据）。
建议把 `Epoch` 设大一些（例如 10~30），再用 `EarlyStop` 控制训练自动停止，这样既更容易“收敛”，也不会无意义地跑满全部 epoch。

如果你要跑消融：

```bash
./tools/run_experiments.ps1 -Python C:/Users/陈永鸿/.conda/envs/pcrec/python.exe -RunAblations
```

## 2.6) FuXiAlpha 小范围调参（sweep）

当 FuXiAlpha 明显落后 SASRec/TiSASRec 时，建议先做一个很小的调参 sweep（优先调 `lr` 与 `attn_dropout`）。

默认 sweep 配置：
- `lr`: 0.001 / 0.0005
- `attn_dropout`: 0.0 / 0.1 / 0.2
- 每组跑 3 个 seed：0/1/2

运行（以 MovieLens_100K 为例）：

```bash
./tools/run_fuxi_sweep.ps1 -Python C:/Users/陈永鸿/.conda/envs/pcrec/python.exe -Dataset MovieLens_100K -Epoch 25 -EarlyStop 5
```

输出：
- `results/fuxi_sweep_MovieLens_100K.csv`：按 (lr,attn_dropout,split) 聚合的 mean±std（跨 seed）
- 同时会更新 `results/summary_runs.csv` / `results/summary_agg.csv`，并重画该数据集的对比图

如果你想先快速筛选（只跑 seed=0）：

```bash
./tools/run_fuxi_sweep.ps1 -Python C:/Users/陈永鸿/.conda/envs/pcrec/python.exe -Dataset MovieLens_100K -Seeds 0 -Epoch 25 -EarlyStop 5
```

## 3) FuXiAlpha 的消融参数

在 `src/models/sequential/FuXiAlpha.py` 里增加了以下参数（都能出现在 log 文件名里，方便汇总）：

- `--fuxi_use_pos {0,1}`：是否启用 **位置通道**
- `--fuxi_use_time {0,1}`：是否启用 **时间通道**
- `--fuxi_use_latent {0,1}`：是否启用 **latent(qk) 通道**
- `--fuxi_single_stage_ffn {0,1}`：是否把 MSFFN 退化成单阶段（消融 stage-2）

推荐你做 4 组（每组跑 3 个 seed，两个数据集都跑）：

1) Full：pos=1,time=1,latent=1,ssffn=0
2) w/o time：pos=1,time=0,latent=1,ssffn=0
3) w/o pos：pos=0,time=1,latent=1,ssffn=0
4) w/o MSFFN stage-2：pos=1,time=1,latent=1,ssffn=1

（如果算力允许，再加：w/o latent 或只保留 latent）

## 4) 怎么“分析结果”（写作业建议结构）

你可以按下面思路写分析：

- **整体对比（FuXiAlpha vs SASRec/GRU4Rec）**：
  - 对每个数据集，报告 Test 的 `HR@10`、`NDCG@10`（最好带均值±标准差）。
  - 解释：`HR@K` 更像“命中率”，`NDCG@K` 会额外惩罚命中位置靠后。

- **跨数据集一致性**：
  - 如果 FuXiAlpha 在两个数据集都提升，说明更稳健。
  - 如果只在一个数据集提升，说明可能对数据稀疏度/序列长度敏感。

- **消融结论**：
  - 去掉 time 通道后下降幅度大：说明模型主要收益来自时间建模。
  - 去掉 pos 通道后下降幅度大：说明相对位置/顺序信息更关键。
  - MSFFN stage-2 关闭后下降：说明隐式特征交互（FFN 里的门控/二阶段）有贡献。

- **统计角度**：
  - 用 3 个 seed 的标准差解释“是否显著”：
    如果提升 < 1 个 std，建议谨慎表述（写成“轻微提升/趋势”）。

