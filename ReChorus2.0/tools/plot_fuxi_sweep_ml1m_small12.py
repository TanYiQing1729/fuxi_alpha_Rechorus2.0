import csv
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


APPENDIX = "MovieLens_1M_ml1m_fuxi_sweep_s0_e30_small12_ss0"
SUMMARY_RUNS = os.path.join(os.path.dirname(__file__), "..", "results", "summary_runs.csv")
OUT_FIG = os.path.join(os.path.dirname(__file__), "..", "figures", "ml1m_fuxi_sweep_small12_ss0.png")


def parse_hparams(hparams: str) -> dict:
    out: dict[str, str] = {}
    for part in (hparams or "").split(";"):
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def fnum(x: str | None) -> float | None:
    if x is None:
        return None
    x = x.strip()
    if x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def fint(x: str | None) -> int | None:
    if x is None:
        return None
    x = x.strip()
    if x == "":
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def main() -> None:
    # group by (run_id, split)
    by_run: dict[str, dict[str, dict]] = defaultdict(dict)

    with open(SUMMARY_RUNS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            f,
            fieldnames=[
                "model",
                "dataset",
                "seed",
                "group",
                "split",
                "run_id",
                "log_path",
                "hparams",
                "HR@5",
                "HR@10",
                "HR@20",
                "AUC",
                "NDCG@5",
                "NDCG@10",
                "NDCG@20",
                "MRR",
            ],
        )
        for row in reader:
            if row.get("model") != "FuXiAlpha":
                continue
            if row.get("dataset") != APPENDIX:
                continue
            split = row.get("split")
            if split not in ("dev", "test"):
                continue
            run_id = row.get("run_id")
            if not run_id:
                continue
            by_run[run_id][split] = row

    rows = []
    for run_id, splits in by_run.items():
        dev = splits.get("dev")
        test = splits.get("test")
        if not dev or not test:
            continue

        hp = parse_hparams(dev.get("hparams", ""))
        lr = fnum(hp.get("lr"))
        num_layers = fint(hp.get("num_layers"))
        attn_dropout = fnum(hp.get("attn_dropout"))

        rows.append(
            {
                "run_id": run_id,
                "lr": lr,
                "num_layers": num_layers,
                "attn_dropout": attn_dropout,
                "dev_hr10": fnum(dev.get("HR@10")),
                "dev_ndcg10": fnum(dev.get("NDCG@10")),
                "test_hr10": fnum(test.get("HR@10")),
                "test_ndcg10": fnum(test.get("NDCG@10")),
            }
        )

    # keep only well-formed rows
    rows = [
        r
        for r in rows
        if r["lr"] is not None and r["num_layers"] is not None and r["attn_dropout"] is not None
    ]

    # dedupe to configs by (lr, num_layers, attn_dropout)
    # each config should have exactly one run_id under seed0; if duplicates exist, keep best dev_ndcg10
    by_cfg = {}
    for r in rows:
        key = (r["lr"], r["num_layers"], r["attn_dropout"])
        cur = by_cfg.get(key)
        if cur is None:
            by_cfg[key] = r
            continue
        if (r["dev_ndcg10"] or -1) > (cur["dev_ndcg10"] or -1):
            by_cfg[key] = r
    rows = list(by_cfg.values())
    rows.sort(key=lambda r: (r["lr"], r["num_layers"], r["attn_dropout"]))

    if len(rows) != 12:
        print(f"[WARN] Expected 12 configs, got {len(rows)}")

    # best by dev NDCG@10
    best_dev = max(rows, key=lambda r: (r["dev_ndcg10"] or -1, r["dev_hr10"] or -1))
    best_test_hr = max(rows, key=lambda r: (r["test_hr10"] or -1, r["test_ndcg10"] or -1))

    print("Best by dev NDCG@10:")
    print(best_dev)
    print("Best by test HR@10:")
    print(best_test_hr)

    # compute layer means
    layer_vals = defaultdict(list)
    for r in rows:
        layer_vals[r["num_layers"]].append(r["dev_ndcg10"])
    print("Layer mean dev NDCG@10:")
    for k in sorted(layer_vals):
        vals = [v for v in layer_vals[k] if v is not None]
        print(k, sum(vals) / len(vals) if vals else None)

    # plot
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

    lrs = sorted({r["lr"] for r in rows})
    layers = sorted({r["num_layers"] for r in rows})
    drops = sorted({r["attn_dropout"] for r in rows})

    # offsets for different lr only
    lr_offsets = {lr: (i - (len(lrs) - 1) / 2) * 0.14 for i, lr in enumerate(lrs)}

    # visuals
    lr_colors = {
        lrs[0]: "C0",
        lrs[1]: "C1",
    } if len(lrs) == 2 else {lr: f"C{i}" for i, lr in enumerate(lrs)}
    drop_markers = {
        0.0: "o",
        0.1: "^",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    for metric, ax, ylabel in [
        ("dev_ndcg10", axes[0], "Dev NDCG@10"),
        ("test_hr10", axes[1], "Test HR@10"),
    ]:
        for r in rows:
            x_base = layers.index(r["num_layers"]) + 1
            x = x_base + lr_offsets[r["lr"]]
            y = r[metric]
            if y is None:
                continue
            color = lr_colors.get(r["lr"], "C0")
            marker = drop_markers.get(r["attn_dropout"], "o")
            ax.scatter(x, y, s=32, c=color, marker=marker, edgecolors="none")

        ax.set_xticks([i + 1 for i in range(len(layers))])
        ax.set_xticklabels([str(n) for n in layers])
        ax.set_xlabel("num_layers")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # legends
    lr_handles, lr_labels = [], []
    for lr in lrs:
        h = plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, color=lr_colors.get(lr, "C0"))
        lr_handles.append(h)
        lr_labels.append(f"lr={lr}")

    drop_handles, drop_labels = [], []
    for d in drops:
        h = plt.Line2D([0], [0], marker=drop_markers.get(d, "o"), linestyle="", markersize=6, color="black")
        drop_handles.append(h)
        drop_labels.append(f"attn_dropout={d}")

    fig.legend(lr_handles + drop_handles, lr_labels + drop_labels, loc="upper center", ncol=len(lrs) + len(drops), frameon=False)

    fig.suptitle("FuXiAlpha hyperparameter sweep (ML-1M, small12, ssffn=0; show attn_dropout)")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT_FIG)
    print("Saved figure to:", OUT_FIG)


if __name__ == "__main__":
    main()
