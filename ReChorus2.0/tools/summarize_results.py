# -*- coding: utf-8 -*-

"""Summarize ReChorus log files into CSV tables.

It scans `ReChorus2.0/log/**.txt` and extracts:
- model / dataset / seed / hyperparams (from filename)
- Dev After Training metrics
- Test After Training metrics

Outputs:
- results/summary_runs.csv: one row per (run, split)
- results/summary_agg.csv: mean/std across seeds per (model, dataset, split, variant)

Run examples:
  python tools/summarize_results.py --rechorus_root .
  python tools/summarize_results.py --rechorus_root . --metrics HR@10 NDCG@10
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


@dataclass
class RunRecord:
	model: str
	dataset: str
	seed: Optional[int]
	variant: str
	split: str  # dev/test
	metrics: Dict[str, float]
	log_path: str
	run_id: str
	hparams: Dict[str, str]


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
	base = f"pos{use_pos}_time{use_time}_latent{use_latent}_ssffn{ss_ffn}"
	use_ams = hparams.get("fuxi_use_ams", "1")
	# Backward compatible: only append when AMS is explicitly removed.
	if str(use_ams).strip() == "0":
		return f"{base}_noams"
	return base


_HPARAM_KEYS_FROM_LOG = {
	"attn_dropout",
	"fuxi_use_ams",
	"fuxi_use_pos",
	"fuxi_use_time",
	"fuxi_use_latent",
	"fuxi_single_stage_ffn",
}


def _merge_hparams_from_log(hparams: Dict[str, str], lines: List[str]) -> Dict[str, str]:
	"""Fill missing hparams from the log header table.

	Some Windows log filenames get truncated, which can drop keys like
	`fuxi_single_stage_ffn=1` from the filename. The log header contains a
	stable "Arguments | Values" table we can parse as a fallback.
	"""
	missing = {
		k
		for k in _HPARAM_KEYS_FROM_LOG
		if (k not in hparams) or (str(hparams.get(k, "")).strip() == "")
	}
	if not missing:
		return hparams
	merged = dict(hparams)
	for raw in lines[:250]:
		line = raw.strip()
		if "|" not in line:
			continue
		left, right = line.split("|", 1)
		key = left.strip()
		val = right.strip()
		if key in missing and val:
			merged[key] = val
	return merged


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


def _walk_log_files(log_dir: str) -> List[str]:
	paths: List[str] = []
	for root, _dirs, files in os.walk(log_dir):
		for f in files:
			if not f.lower().endswith(".txt"):
				continue
			paths.append(os.path.join(root, f))
	return sorted(paths)


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	tmp_path = f"{path}.tmp"
	try:
		with open(tmp_path, "w", newline="", encoding="utf-8") as fp:
			writer = csv.DictWriter(fp, fieldnames=fieldnames)
			writer.writeheader()
			for r in rows:
				writer.writerow(r)
		# Atomic replace on most platforms; may fail on Windows if target is open.
		os.replace(tmp_path, path)
	except PermissionError:
		# Fallback: keep the newly written file with a different name.
		fallback = f"{path}.new"
		try:
			if os.path.exists(fallback):
				os.remove(fallback)
			os.replace(tmp_path, fallback)
		except Exception:
			# Best effort cleanup.
			try:
				if os.path.exists(tmp_path):
					os.remove(tmp_path)
			except Exception:
				pass
			raise
		print(f"[WARN] Cannot overwrite locked file: {path}")
		print(f"[WARN] Wrote to: {fallback} (close the original file and rerun to overwrite)")


def _mean_std(values: List[float]) -> Tuple[float, float]:
	if not values:
		return 0.0, 0.0
	mean = sum(values) / len(values)
	var = sum((x - mean) ** 2 for x in values) / (len(values) - 1) if len(values) > 1 else 0.0
	return mean, var ** 0.5


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--rechorus_root", type=str, default=".", help="Path to ReChorus2.0")
	parser.add_argument(
		"--log_dirs",
		nargs="*",
		default=None,
		help=(
			"Optional explicit log directories to scan. "
			"If omitted, will scan both <rechorus_root>/log and <rechorus_root>/../log (ReChorus default)."
		),
	)
	parser.add_argument(
		"--metrics",
		nargs="*",
		default=["HR@10", "NDCG@10", "HR@20", "NDCG@20"],
		help="Which metrics to keep in aggregated table",
	)
	args = parser.parse_args()

	rechorus_root = os.path.abspath(args.rechorus_root)
	default_log_dirs = [
		os.path.join(rechorus_root, "log"),
		os.path.abspath(os.path.join(rechorus_root, os.pardir, "log")),
	]
	log_dirs = args.log_dirs if args.log_dirs is not None else default_log_dirs
	out_dir = os.path.join(rechorus_root, "results")

	log_files: List[str] = []
	seen: set = set()
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
	records: List[RunRecord] = []

	for p in log_files:
		stem = os.path.splitext(os.path.basename(p))[0]
		model, dataset, seed, hparams = _parse_filename(stem)
		with open(p, "r", encoding="utf-8", errors="ignore") as fp:
			lines = fp.readlines()
		metrics_by_split = _extract_dev_test_metrics(lines)
		if model.startswith("FuXiAlpha"):
			hparams = _merge_hparams_from_log(hparams, lines)
		variant = _variant_from(model, hparams)
		for split, metrics in metrics_by_split.items():
			records.append(
				RunRecord(
					model=model,
					dataset=dataset,
					seed=seed,
					variant=variant,
					split=split,
					metrics=metrics,
					log_path=os.path.relpath(p, rechorus_root),
					run_id=stem,
					hparams=hparams,
				)
			)

	# --- summary_runs.csv (wide)
	all_metric_keys = sorted({k for r in records for k in r.metrics.keys()})
	run_rows: List[Dict[str, object]] = []
	for r in records:
		row: Dict[str, object] = {
			"model": r.model,
			"dataset": r.dataset,
			"seed": r.seed if r.seed is not None else "",
			"variant": r.variant,
			"split": r.split,
			"run_id": r.run_id,
			"log_path": r.log_path,
		}
		# include hparams as a compact string (keep raw too for debugging)
		row["hparams"] = ";".join([f"{k}={v}" for k, v in sorted(r.hparams.items())])
		for k in all_metric_keys:
			row[k] = r.metrics.get(k, "")
		run_rows.append(row)

	_write_csv(
		os.path.join(out_dir, "summary_runs.csv"),
		run_rows,
		fieldnames=["model", "dataset", "seed", "variant", "split", "run_id", "log_path", "hparams"] + all_metric_keys,
	)

	# --- summary_agg.csv
	# group by (model,dataset,variant,split)
	group: Dict[Tuple[str, str, str, str], List[RunRecord]] = {}
	for r in records:
		key = (r.model, r.dataset, r.variant, r.split)
		group.setdefault(key, []).append(r)

	agg_rows: List[Dict[str, object]] = []
	keep_metrics = [m.upper() for m in args.metrics]
	for (model, dataset, variant, split), rs in sorted(group.items()):
		row: Dict[str, object] = {
			"model": model,
			"dataset": dataset,
			"variant": variant,
			"split": split,
			"n": len({r.seed for r in rs}),
		}
		for mk in keep_metrics:
			vals = [r.metrics.get(mk) for r in rs if mk in r.metrics]
			vals_f = [float(v) for v in vals if isinstance(v, (float, int))]
			mean, std = _mean_std(vals_f)
			row[f"{mk}_mean"] = mean
			row[f"{mk}_std"] = std
		agg_rows.append(row)

	_write_csv(
		os.path.join(out_dir, "summary_agg.csv"),
		agg_rows,
		fieldnames=["model", "dataset", "variant", "split", "n"]
		+ [f"{m.upper()}_mean" for m in args.metrics]
		+ [f"{m.upper()}_std" for m in args.metrics],
	)

	print(f"Wrote: {os.path.join(out_dir, 'summary_runs.csv')}")
	print(f"Wrote: {os.path.join(out_dir, 'summary_agg.csv')}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
