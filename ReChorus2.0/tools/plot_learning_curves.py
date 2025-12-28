# -*- coding: utf-8 -*-

"""Plot learning curves (dev metrics vs epoch) from ReChorus logs.

This script parses per-epoch log lines like:
  Epoch 1     loss=...    dev=(HR@5:...,NDCG@5:...) ...

and generates line plots for selected metrics (default HR@10 and NDCG@10).

It scans both:
- <rechorus_root>/log
- <rechorus_root>/../log  (ReChorus default log output)

Example:
  python tools/plot_learning_curves.py --rechorus_root . \
    --datasets Grocery_and_Gourmet_Food MovieLens_1M MovieLens_100K \
    --models SASRec GRU4Rec TiSASRec FuXiAlpha \
    --metric HR@10 NDCG@10
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

# Use headless backend for Windows environments without Tcl/Tk.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


_EPOCH_DEV_RE = re.compile(r"^.*?Epoch\s+(?P<epoch>\d+)\s+.*?dev=\((?P<body>[^)]*)\).*$")
_METRIC_RE = re.compile(r"(?P<name>[A-Za-z]+)@(?P<k>\d+):(?P<val>[0-9.]+)")


@dataclass
class Curve:
	model: str
	dataset: str
	seed: Optional[int]
	variant: str
	run_id: str
	log_path: str
	mtime: float
	epochs: List[int]
	metrics_by_epoch: Dict[str, List[float]]


def _safe_int(s: str) -> Optional[int]:
	try:
		return int(s)
	except Exception:
		return None


def _parse_filename(stem: str) -> Tuple[str, str, Optional[int], Dict[str, str]]:
	"""Parse `Model__Dataset__Seed__k=v__...` from file stem."""
	parts = stem.split("__")
	if len(parts) < 2:
		return stem, "", None, {}
	model = parts[0]
	dataset = parts[1] if len(parts) >= 2 else ""
	seed = _safe_int(parts[2]) if len(parts) >= 3 else None
	hparams: Dict[str, str] = {}
	for p in parts[3:]:
		if "=" in p:
			k, v = p.split("=", 1)
			hparams[k] = v
	return model, dataset, seed, hparams


def _variant_from(model: str, hparams: Dict[str, str]) -> str:
	if not model.startswith("FuXiAlpha"):
		return "default"
	use_pos = hparams.get("fuxi_use_pos", "1")
	use_time = hparams.get("fuxi_use_time", "1")
	use_latent = hparams.get("fuxi_use_latent", "1")
	ss_ffn = hparams.get("fuxi_single_stage_ffn", "0")
	return f"pos{use_pos}_time{use_time}_latent{use_latent}_ssffn{ss_ffn}"


def _walk_log_files(log_dir: str) -> List[str]:
	paths: List[str] = []
	for root, _dirs, files in os.walk(log_dir):
		for f in files:
			if f.lower().endswith(".txt"):
				paths.append(os.path.join(root, f))
	return sorted(paths)


def _scan_log_paths(rechorus_root: str, explicit: Optional[List[str]]) -> List[str]:
	if explicit is not None:
		log_dirs = explicit
	else:
		log_dirs = [
			os.path.join(rechorus_root, "log"),
			os.path.abspath(os.path.join(rechorus_root, os.pardir, "log")),
		]

	seen: set = set()
	out: List[str] = []
	for ld in log_dirs:
		if not os.path.isdir(ld):
			continue
		for p in _walk_log_files(ld):
			ap = os.path.abspath(p)
			if ap in seen:
				continue
			seen.add(ap)
			out.append(p)
	return sorted(out)


def _parse_metric_body(body: str) -> Dict[str, float]:
	out: Dict[str, float] = {}
	for m in _METRIC_RE.finditer(body):
		key = f"{m.group('name').upper()}@{m.group('k')}"
		out[key] = float(m.group("val"))
	return out


def _extract_epoch_dev(lines: Iterable[str]) -> Tuple[List[int], Dict[str, List[float]]]:
	epochs: List[int] = []
	metrics_by_epoch: Dict[str, List[float]] = {}
	for line in lines:
		m = _EPOCH_DEV_RE.match(line.strip())
		if not m:
			continue
		epoch = int(m.group("epoch"))
		metrics = _parse_metric_body(m.group("body"))
		epochs.append(epoch)
		for k, v in metrics.items():
			metrics_by_epoch.setdefault(k, []).append(v)
	return epochs, metrics_by_epoch


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--rechorus_root", type=str, default=".")
	parser.add_argument("--log_dirs", nargs="*", default=None)
	parser.add_argument(
		"--datasets",
		nargs="*",
		default=["Grocery_and_Gourmet_Food", "MovieLens_1M", "MovieLens_100K"],
	)
	parser.add_argument("--models", nargs="*", default=["SASRec", "GRU4Rec", "TiSASRec", "FuXiAlpha"])
	parser.add_argument(
		"--metric",
		nargs="*",
		default=None,
		help=(
			"Which dev metrics to plot (e.g., HR@5 NDCG@5). "
			"If omitted, defaults to HR@5 and NDCG@5 (ReChorus per-epoch logs usually only include @5)."
		),
	)
	parser.add_argument("--out_dir", type=str, default="figures/learning_curves")
	parser.add_argument("--max_runs_per_group", type=int, default=3)
	parser.add_argument(
		"--min_points",
		type=int,
		default=2,
		help="Skip curves with fewer than this many epochs/points (default: 2). Use 1 to include 1-epoch smoke runs.",
	)
	args = parser.parse_args()

	rechorus_root = os.path.abspath(args.rechorus_root)
	log_paths = _scan_log_paths(rechorus_root, args.log_dirs)

	want_datasets = set(args.datasets)
	want_models = set(args.models)
	if args.metric is None or len(args.metric) == 0:
		want_metrics = ["HR@5", "NDCG@5"]
	else:
		want_metrics = [m.upper() for m in args.metric]

	curves: List[Curve] = []
	for p in log_paths:
		stem = os.path.splitext(os.path.basename(p))[0]
		model, dataset, seed, hparams = _parse_filename(stem)
		if model not in want_models or dataset not in want_datasets:
			continue
		variant = _variant_from(model, hparams)
		mtime = 0.0
		try:
			mtime = os.path.getmtime(p)
		except Exception:
			mtime = 0.0
		with open(p, "r", encoding="utf-8", errors="ignore") as fp:
			epochs, metrics_by_epoch = _extract_epoch_dev(fp)
		if not epochs:
			continue
		# Keep only requested metrics if present; if none match, fall back to whatever exists.
		filtered: Dict[str, List[float]] = {}
		for mk in want_metrics:
			if mk in metrics_by_epoch:
				filtered[mk] = metrics_by_epoch[mk]
		if not filtered:
			# Fallback: plot whatever is available in per-epoch dev logs
			available = sorted(metrics_by_epoch.keys())
			print(
				f"[WARN] Requested metrics {want_metrics} not found in per-epoch dev logs for {stem}. "
				f"Falling back to available: {available}"
			)
			filtered = {k: metrics_by_epoch[k] for k in available}
		curves.append(
			Curve(
				model=model,
				dataset=dataset,
				seed=seed,
				variant=variant,
				run_id=stem,
				log_path=os.path.relpath(p, rechorus_root),
				mtime=mtime,
				epochs=epochs,
				metrics_by_epoch=filtered,
			)
		)

	out_dir = os.path.join(rechorus_root, args.out_dir)
	os.makedirs(out_dir, exist_ok=True)

	# Prefer the most complete/latest run per (dataset, model, variant, seed).
	best_by_seed: Dict[Tuple[str, str, str, Optional[int]], Curve] = {}
	for c in curves:
		key_variant = c.variant if c.model == "FuXiAlpha" else "default"
		key = (c.dataset, c.model, key_variant, c.seed)
		prev = best_by_seed.get(key)
		if prev is None:
			best_by_seed[key] = c
			continue
		# Pick the one with more epochs; tie-breaker by mtime.
		if len(c.epochs) > len(prev.epochs) or (len(c.epochs) == len(prev.epochs) and c.mtime > prev.mtime):
			best_by_seed[key] = c

	# group by dataset + model (+ variant for FuXiAlpha)
	groups: Dict[Tuple[str, str, str], List[Curve]] = {}
	for c in best_by_seed.values():
		key_variant = c.variant if c.model == "FuXiAlpha" else "default"
		key = (c.dataset, c.model, key_variant)
		groups.setdefault(key, []).append(c)

	for (dataset, model, variant), cs in sorted(groups.items()):
		cs = sorted(cs, key=lambda x: (x.seed is None, x.seed if x.seed is not None else 0, x.run_id))
		cs = cs[: max(1, int(args.max_runs_per_group))]
		cs = [c for c in cs if len(c.epochs) >= int(args.min_points)]
		if not cs:
			continue

		fig, ax = plt.subplots(1, 1, figsize=(7, 4))
		min_epoch = min(min(c.epochs) for c in cs)
		max_epoch = max(max(c.epochs) for c in cs)
		for c in cs:
			seed_str = "" if c.seed is None else f"seed={c.seed}"
			label_suffix = seed_str
			for mk, ys in c.metrics_by_epoch.items():
				label = f"{mk} {label_suffix}".strip()
				ax.plot(c.epochs[: len(ys)], ys, marker="o", linewidth=1.5, markersize=3, label=label)

		ax.set_title(f"{dataset} - {model}" + (f" ({variant})" if model == "FuXiAlpha" else ""))
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Dev metric")
		# Make x-axis look sane even when only a few points exist.
		if min_epoch == max_epoch:
			ax.set_xlim(min_epoch - 0.5, max_epoch + 0.5)
			ax.set_xticks([min_epoch])
		else:
			ax.set_xlim(min_epoch - 0.5, max_epoch + 0.5)
			step = 1 if (max_epoch - min_epoch) <= 30 else 5
			ax.set_xticks(list(range(min_epoch, max_epoch + 1, step)))
		ax.grid(True, linestyle="--", alpha=0.3)
		ax.legend(fontsize=8)
		fig.tight_layout()

		safe_variant = variant.replace("/", "-")
		out_path = os.path.join(out_dir, f"{dataset}__{model}__{safe_variant}__dev_curve.png")
		fig.savefig(out_path, dpi=200)
		plt.close(fig)
		print(f"Wrote: {out_path}")

	if not groups:
		print("No matching curves found. Make sure you have logs with per-epoch dev=... lines.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
