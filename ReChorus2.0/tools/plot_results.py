# -*- coding: utf-8 -*-

"""Plot aggregated results produced by summarize_results.py.

It reads `ReChorus2.0/results/summary_agg.csv` and generates bar charts
(with optional error bars) for selected datasets/models/variants.

Example:
  python tools/plot_results.py --rechorus_root . --split test --metric HR@10 NDCG@10
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib

# Use a headless backend (no Tcl/Tk dependency).
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _set_tight_ylim(
	ax: plt.Axes,
	values: List[float],
	errors: List[float],
	*,
	margin_ratio: float = 0.06,
	clip_01: bool = True,
) -> None:
	if not values:
		return

	low = min(v - (errors[i] if i < len(errors) else 0.0) for i, v in enumerate(values))
	high = max(v + (errors[i] if i < len(errors) else 0.0) for i, v in enumerate(values))

	# Guard against degenerate ranges.
	span = high - low
	if span <= 1e-12:
		span = max(abs(high), 1.0) * 0.01
		low -= span
		high += span
	else:
		pad = span * max(margin_ratio, 0.0)
		low -= pad
		high += pad

	if clip_01:
		low = max(low, 0.0)
		high = min(high, 1.0)

	# Still ensure valid ordering.
	if high <= low:
		high = low + 1e-3

	ax.set_ylim(low, high)


def _read_csv(path: str) -> List[Dict[str, str]]:
	rows: List[Dict[str, str]] = []
	with open(path, "r", encoding="utf-8") as fp:
		reader = csv.DictReader(fp)
		for r in reader:
			rows.append(r)
	return rows


def _as_float(v: str) -> float:
	try:
		return float(v)
	except Exception:
		return 0.0


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--rechorus_root", type=str, default=".")
	parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
	parser.add_argument(
		"--datasets",
		nargs="*",
		default=["Grocery_and_Gourmet_Food", "MovieLens_1M", "MovieLens_100K"],
	)
	parser.add_argument("--models", nargs="*", default=["SASRec", "GRU4Rec", "TiSASRec", "FuXiAlpha"])
	parser.add_argument("--metric", nargs="*", default=["HR@10", "NDCG@10"])
	parser.add_argument(
		"--ylim_margin",
		type=float,
		default=0.06,
		help="Tight y-axis margin ratio (default: 0.06).",
	)
	parser.add_argument(
		"--y_from_zero",
		action="store_true",
		help="Force y-axis to start from 0 (not recommended for highlighting small differences).",
	)
	parser.add_argument("--out_dir", type=str, default="figures")
	args = parser.parse_args()

	rechorus_root = os.path.abspath(args.rechorus_root)
	summary_path = os.path.join(rechorus_root, "results", "summary_agg.csv")
	rows = _read_csv(summary_path)

	out_dir = os.path.join(rechorus_root, args.out_dir)
	os.makedirs(out_dir, exist_ok=True)

	metrics = [m.upper() for m in args.metric]

	for dataset in args.datasets:
		filtered = [
			r
			for r in rows
			if r.get("dataset") == dataset
			and r.get("split") == args.split
			and r.get("model") in args.models
		]
		if not filtered:
			print(f"No rows for dataset={dataset}, split={args.split}")
			continue

		# consistent ordering: baseline models first, then FuXi variants
		labels: List[str] = []
		series: Dict[str, List[float]] = {m: [] for m in metrics}
		series_std: Dict[str, List[float]] = {m: [] for m in metrics}

		# group rows by model+variant (FuXiAlpha may have multiple ablations)
		grouped: Dict[str, List[Dict[str, str]]] = {}
		for r in filtered:
			model = r["model"]
			variant = r.get("variant", "default")
			key = model if (model != "FuXiAlpha" or variant == "default") else f"{model}({variant})"
			grouped.setdefault(key, []).append(r)

		# flatten in a stable order
		ordered_keys: List[str] = []
		for m in args.models:
			if m == "FuXiAlpha":
				for k in sorted([k for k in grouped.keys() if k.startswith("FuXiAlpha")]):
					ordered_keys.append(k)
			else:
				if m in grouped:
					ordered_keys.append(m)

		for key in ordered_keys:
			rs = grouped[key]
			# expect single row per group in summary_agg
			r = rs[0]
			labels.append(key)
			for mk in metrics:
				series[mk].append(_as_float(r.get(f"{mk}_mean", "0")))
				series_std[mk].append(_as_float(r.get(f"{mk}_std", "0")))

		# plot: one figure with subplots for each metric
		fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
		for j, mk in enumerate(metrics):
			ax = axes[0][j]
			x = list(range(len(labels)))
			ax.bar(x, series[mk], yerr=series_std[mk], capsize=4)
			if args.y_from_zero:
				ax.set_ylim(bottom=0.0)
			else:
				_set_tight_ylim(
					ax,
					series[mk],
					series_std[mk],
					margin_ratio=args.ylim_margin,
				)
			ax.set_xticks(x)
			ax.set_xticklabels(labels, rotation=30, ha="right")
			ax.set_title(f"{dataset} - {args.split} - {mk}")
			ax.set_ylabel(mk)
			ax.grid(axis="y", linestyle="--", alpha=0.3)
		fig.tight_layout()

		out_path = os.path.join(out_dir, f"{dataset}__{args.split}__{'_'.join(metrics)}.png")
		fig.savefig(out_path, dpi=200)
		plt.close(fig)
		print(f"Wrote: {out_path}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
