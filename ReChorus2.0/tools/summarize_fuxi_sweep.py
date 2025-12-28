# -*- coding: utf-8 -*-

"""Summarize FuXiAlpha hyperparameter sweep results.

Why this exists:
- The generic summarize_results.py aggregates FuXiAlpha by "variant" (ablation flags only).
- For tuning, we want to compare lr / attn_dropout (and potentially others) separately.

This script scans ReChorus log files, keeps only FuXiAlpha runs for a dataset,
extracts Dev/Test After Training metrics, then aggregates by (lr, attn_dropout)
across seeds.

Outputs:
- results/fuxi_sweep_<dataset>.csv

Example:
  python tools/summarize_fuxi_sweep.py --rechorus_root . --dataset MovieLens_100K
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


_METRIC_RE = re.compile(r"(?P<name>[A-Za-z]+)@(?P<k>\d+):(?P<val>[0-9.]+)")
_DEV_RE = re.compile(r"^Dev\s+After\s+Training:\s*\((?P<body>.*)\)\s*$")
_TEST_RE = re.compile(r"^Test\s+After\s+Training:\s*\((?P<body>.*)\)\s*$")
_ARG_ROW_RE = re.compile(r"^\s*(?P<key>[A-Za-z0-9_]+)\s*\|\s*(?P<val>.*?)\s*$")


@dataclass
class Record:
	seed: Optional[int]
	lr: str
	attn_dropout: str
	split: str
	metrics: Dict[str, float]
	log_path: str
	run_id: str


def _safe_int(s: str) -> Optional[int]:
	try:
		return int(s)
	except Exception:
		return None


def _parse_filename(stem: str) -> Tuple[str, str, Optional[int], Dict[str, str]]:
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


def _parse_metric_body(body: str) -> Dict[str, float]:
	out: Dict[str, float] = {}
	for m in _METRIC_RE.finditer(body):
		key = f"{m.group('name').upper()}@{m.group('k')}"
		out[key] = float(m.group("val"))
	return out


def _extract_dev_test_metrics(lines: Iterable[str]) -> Dict[str, Dict[str, float]]:
	out: Dict[str, Dict[str, float]] = {}
	for line in lines:
		line = line.strip()
		m = _DEV_RE.match(line)
		if m:
			out["dev"] = _parse_metric_body(m.group("body"))
			continue
		m = _TEST_RE.match(line)
		if m:
			out["test"] = _parse_metric_body(m.group("body"))
			continue
	return out


def _extract_args_table(lines: Iterable[str]) -> Dict[str, str]:
	"""Parse the pretty-printed Arguments table in ReChorus logs.

	We rely on this because Windows-safe filename truncation may drop hparams
	(e.g., attn_dropout) from the log filename.
	"""
	out: Dict[str, str] = {}
	for raw in lines:
		line = raw.rstrip("\n")
		m = _ARG_ROW_RE.match(line)
		if not m:
			continue
		key = (m.group("key") or "").strip()
		val = (m.group("val") or "").strip()
		if not key:
			continue
		# Skip table separators/headings
		if key.lower() in {"arguments", "values"}:
			continue
		if set(key) <= {"=", "-"}:
			continue
		out[key] = val
	return out


def _walk_log_files(log_dir: str) -> List[str]:
	paths: List[str] = []
	for root, _dirs, files in os.walk(log_dir):
		for f in files:
			if f.lower().endswith(".txt"):
				paths.append(os.path.join(root, f))
	return sorted(paths)


def _mean_std(values: List[float]) -> Tuple[float, float]:
	if not values:
		return 0.0, 0.0
	mean = sum(values) / len(values)
	var = sum((x - mean) ** 2 for x in values) / (len(values) - 1) if len(values) > 1 else 0.0
	return mean, var ** 0.5


def _as_float(v: object) -> float:
	try:
		return float(v)
	except Exception:
		return 0.0


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--rechorus_root", type=str, default=".")
	parser.add_argument("--dataset", type=str, default="MovieLens_100K")
	parser.add_argument(
		"--appendix",
		type=str,
		default="fuxi_sweep",
		help=(
			"Only include logs whose filename hparam data_appendix matches this value. "
			"This avoids mixing sweep runs with previous experiments. Use empty string to disable filtering."
		),
	)
	parser.add_argument(
		"--log_dirs",
		nargs="*",
		default=None,
		help="Optional explicit log dirs. If omitted, scans ./log and ../log.",
	)
	parser.add_argument("--metrics", nargs="*", default=["HR@10", "NDCG@10"])
	args = parser.parse_args()

	rechorus_root = os.path.abspath(args.rechorus_root)
	want_dataset = args.dataset
	want_appendix = (args.appendix or "").strip()
	want_metrics = [m.upper() for m in args.metrics]

	default_log_dirs = [
		os.path.join(rechorus_root, "log"),
		os.path.abspath(os.path.join(rechorus_root, os.pardir, "log")),
	]
	log_dirs = args.log_dirs if args.log_dirs is not None else default_log_dirs

	seen: set = set()
	log_files: List[str] = []
	for ld in log_dirs:
		if not os.path.isdir(ld):
			continue
		for p in _walk_log_files(ld):
			ap = os.path.abspath(p)
			if ap in seen:
				continue
			seen.add(ap)
			log_files.append(p)
	log_files = sorted(log_files)

	records: List[Record] = []
	for p in log_files:
		stem = os.path.splitext(os.path.basename(p))[0]
		model, dataset, seed, hparams = _parse_filename(stem)
		if model != "FuXiAlpha":
			continue
		# New sweep isolation strategy appends data_appendix to the dataset name in filenames.
		# Support both:
		# - dataset == MovieLens_100K (old)
		# - dataset == MovieLens_100K + appendix (new)
		if dataset != want_dataset and not (want_appendix and dataset == f"{want_dataset}{want_appendix}"):
			continue
		with open(p, "r", encoding="utf-8", errors="ignore") as fp:
			lines = list(fp)
		args_table = _extract_args_table(lines)
		# If we later decide to filter by hparam appendix again, this keeps us compatible.
		if want_appendix and hparams.get("data_appendix", "") and hparams.get("data_appendix", "") != want_appendix:
			continue
		lr = hparams.get("lr") or args_table.get("lr") or ""
		attn_dropout = hparams.get("attn_dropout") or args_table.get("attn_dropout") or ""
		metrics_by_split = _extract_dev_test_metrics(lines)
		for split, metrics in metrics_by_split.items():
			records.append(
				Record(
					seed=seed,
					lr=lr,
					attn_dropout=attn_dropout,
					split=split,
					metrics=metrics,
					log_path=os.path.relpath(p, rechorus_root),
					run_id=stem,
				)
			)

	# group by (lr, attn_dropout, split)
	group: Dict[Tuple[str, str, str], List[Record]] = {}
	for r in records:
		key = (r.lr, r.attn_dropout, r.split)
		group.setdefault(key, []).append(r)

	out_rows: List[Dict[str, object]] = []
	for (lr, attn_dropout, split), rs in sorted(group.items()):
		row: Dict[str, object] = {
			"dataset": want_dataset,
			"model": "FuXiAlpha",
			"appendix": want_appendix,
			"lr": lr,
			"attn_dropout": attn_dropout,
			"split": split,
			"n": len({r.seed for r in rs if r.seed is not None}) if any(r.seed is not None for r in rs) else len(rs),
		}
		for mk in want_metrics:
			vals = [r.metrics.get(mk) for r in rs if mk in r.metrics]
			vals_f = [float(v) for v in vals if v is not None]
			mean, std = _mean_std(vals_f)
			row[f"{mk}_mean"] = mean
			row[f"{mk}_std"] = std
		out_rows.append(row)

	out_dir = os.path.join(rechorus_root, "results")
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, f"fuxi_sweep_{want_dataset}.csv")
	fieldnames = ["dataset", "model", "appendix", "split", "lr", "attn_dropout", "n"]
	for mk in want_metrics:
		fieldnames += [f"{mk}_mean", f"{mk}_std"]

	with open(out_path, "w", newline="", encoding="utf-8") as fp:
		w = csv.DictWriter(fp, fieldnames=fieldnames)
		w.writeheader()
		for r in out_rows:
			w.writerow(r)

	print(f"Wrote: {out_path}")

	# Print best config on test by the first metric
	primary = want_metrics[0] if want_metrics else "HR@10"
	test_rows = [r for r in out_rows if r.get("split") == "test"]
	if test_rows:
		best = max(test_rows, key=lambda r: _as_float(r.get(f"{primary}_mean", 0.0)))
		print(
			f"Best(test) by {primary}: lr={best['lr']} attn_dropout={best['attn_dropout']} "
			f"{primary}={best.get(primary + '_mean'):.6f}"
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
