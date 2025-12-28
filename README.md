# 机器学习课程大作业（ReChorus2.0 + FuXiAlpha 复现）

本仓库用于辅助老师/助教在 **Windows** 环境下快速把代码跑通，并复现我们组在报告中使用的结果汇总与图表。

- 主要框架：`ReChorus2.0/`（推荐从这里运行与复现实验）
- 复现模型：FuXiAlpha（已集成到 ReChorus2.0 的 sequential 模型体系中）
- 对比模型：SASRec / GRU4Rec / TiSASRec
- 数据集：MovieLens_1M、Grocery_and_Gourmet_Food（数据文件**不随仓库上传**；请按 [ReChorus2.0/data/README.md](ReChorus2.0/data/README.md) 放置）

> 如果只需要“验证能跑通 + 能生成报告用图”，推荐走本文的「快速复现（不重新训练）」路线。

---

## 目录结构（关键部分）

- `ReChorus2.0/`：实验主框架、数据、训练入口、汇总与绘图脚本
- `log/`：训练日志（本项目的日志默认写在仓库根目录的 `log/`）
- `机器学习课程大作业-模版.md`：实验报告正文（引用了 figures 中的图片）
- `实验复现流程与问题解决记录.md`：复现过程记录

---

## 环境准备

### 方式 A（推荐）：Conda + pip

1) 创建环境（Python >= 3.10；我本机使用 3.12）

```powershell
conda create -n finerec python=3.12 -y
conda activate finerec
```

2) 安装依赖

```powershell
pip install -r requirements.txt
```

> 如果老师机器没有 NVIDIA GPU 或不想用 GPU：该项目可在 CPU 上运行。

### 方式 B：venv（不使用 conda）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 从零训练

说明：本项目训练日志在我本机累计约数 GB，无法直接上传到 GitHub（且容易触发 100MB 单文件/仓库体积限制）。如果需要从零训练验证流程，可按以下方式跑一个最小例子。

### 1) 进入 ReChorus2.0 根目录

```powershell
Set-Location ReChorus2.0
```

### 2) 运行一个模型（示例：FuXiAlpha on MovieLens_1M，CPU）

```powershell
python src\main.py --model_name FuXiAlpha --dataset MovieLens_1M --path data/ --gpu '' --random_seed 0 --history_max 20 --epoch 5 --early_stop 2 --batch_size 256 --eval_batch_size 256
```

训练完成后会在日志中看到：
- `Dev  After Training: (...)`
- `Test After Training: (...)`

再回到仓库根目录运行 `summarize_results.py` 即可汇总到 CSV。

---

## 上传到 GitHub（Windows / PowerShell）

本仓库包含较大的本地产物目录（如 `venv/`、`log/`、`model/`、`ReChorus2.0/data/`），已经通过 `.gitignore` 排除。

在仓库根目录执行：

```powershell
Set-Location .
git init
git add -A
git commit -m "init: course project"
git branch -M main
```

然后在 GitHub 网页端新建一个空仓库（不要勾选添加 README/license/gitignore），拿到地址后执行：

```powershell
git remote add origin https://github.com/<your_name>/<repo>.git
git push -u origin main
```

如果仍然推送失败，最常见原因是误把大文件加入了提交历史：
- 先用 `git status` 确认没有把 `log/`、`model/`、`ReChorus2.0/data/` 加进去
- 若已经提交过大文件，请告诉我报错信息（例如 100MB 限制/HTTP 413），我再给你对应的清理步骤

---

## 常见问题（Windows）

1) **PowerShell 下不要用 `cd /d`**：那是 cmd 的写法；请使用 `Set-Location` 或 `cd`。

2) **CSV 被占用无法写入**：请关闭 Excel/预览器等占用 `ReChorus2.0/results/*.csv` 的程序，再重跑。

3) **日志文件名被截断导致超参缺失**：Windows 路径长度限制可能截断文件名。
   本项目已在 `summarize_results.py` 中做了兼容：会从日志头部的 `Arguments | Values` 表补全关键超参（例如 `attn_dropout`）。

---

## 复现产物位置

- 汇总 CSV：`ReChorus2.0/results/summary_runs.csv`、`ReChorus2.0/results/summary_agg.csv`
- 图表输出：`ReChorus2.0/figures/`

