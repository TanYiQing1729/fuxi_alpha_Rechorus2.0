# -*- coding: utf-8 -*-

"""Analyze convergence from ReChorus log files.

This script scans ../log/<Model>/*.txt (relative to ReChorus2.0) and extracts:
- configured epoch cap from the argument table
- Best Iter(dev)
- last epoch's dev HR@10 / NDCG@10 (if present)

It prints a compact table to help decide whether epoch=10 is near convergence.

Usage (PowerShell):
  .\.venv\Scripts\python.exe tools\analyze_convergence.py --epoch_cap 10
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


_RE_EPOCH_ARG = re.compile(r"^\s*epoch\s*\|\s*(\d+)\s*$")
_RE_BEST = re.compile(r"^Best\s+Iter\(dev\)=\s*(\d+)\s+dev=\((.*)\)")
_RE_EPOCH_LINE = re.compile(r"^.*\bEpoch\s+(\d+)\s+.*dev=\((.*)\)")
_RE_DEV_AFTER = re.compile(r"^Dev\s+After\s+Training:\s*\((.*)\)\s*$")
_RE_METRIC = re.compile(r"([A-Za-z]+)@(\d+):([0-9.]+)")


def _parse_metrics(body: str) -> Dict[str, float]:
	out: Dict[str, float] = {}
	for name, k, val in _RE_METRIC.findall(body):
		out[f"{name.upper()}@{k}"] = float(val)
	return out


def _find_config_epoch(lines: Iterable[str]) -> Optional[int]:
	for line in lines:
		s = line.strip()
		m = _RE_EPOCH_ARG.match(s)
		if m:
			return int(m.group(1))
	return None


@dataclass
class RunInfo:
	path: str
	model: str
	dataset: str
	seed: int
	cfg_epoch: Optional[int]
	best_ep: Optional[int]
	dev_hr10: Optional[float]
	dev_ndcg10: Optional[float]
	last_ep: Optional[int]

	@property
	def best_at_cap(self) -> Optional[bool]:
		if self.cfg_epoch is None or self.best_ep is None:
			return None
		return self.best_ep == self.cfg_epoch

	@property
	def reached_cap(self) -> Optional[bool]:
		if self.cfg_epoch is None or self.last_ep is None:
			return None
		return self.last_ep == self.cfg_epoch



def _parse_log(path: str) -> Tuple[Optional[int], Optional[int], Dict[str, float], Optional[int]]:
	"""Return (cfg_epoch, best_ep, dev_after_metrics, last_ep)."""
	cfg_epoch: Optional[int] = None
	best_ep: Optional[int] = None
	dev_m: Dict[str, float] = {}
	last_ep: Optional[int] = None
	# Note: per-epoch dev lines typically log HR@5/NDCG@5 only.

	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		buf: List[str] = []
		for _ in range(400):
			line = f.readline()
			if not line:
				break
			buf.append(line.rstrip("\n"))
		cfg_epoch = _find_config_epoch(buf)

		# Parse buffered head + rest so we don't miss early lines.
		for raw in buf:
			s = raw.strip()
			m = _RE_EPOCH_LINE.match(s)
			if m:
				last_ep = int(m.group(1))
				continue
			m = _RE_BEST.match(s)
			if m:
				best_ep = int(m.group(1))
				continue
			m = _RE_DEV_AFTER.match(s)
			if m:
				dev_m = _parse_metrics(m.group(1))
				continue

		for raw in f:
			s = raw.strip()
			m = _RE_EPOCH_LINE.match(s)
			if m:
				last_ep = int(m.group(1))
				continue
			m = _RE_BEST.match(s)
			if m:
				best_ep = int(m.group(1))
				continue
			m = _RE_DEV_AFTER.match(s)
			if m:
				dev_m = _parse_metrics(m.group(1))
				continue

	return cfg_epoch, best_ep, dev_m, last_ep


def _fmt(x: Optional[float]) -> str:
	return "-" if x is None else f"{x:.4f}"


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--epoch_cap", type=int, default=10, help="Only include runs configured with this epoch cap")
	parser.add_argument("--datasets", nargs="*", default=["Grocery_and_Gourmet_Food", "MovieLens_1M", "MovieLens_100K"])
	parser.add_argument("--models", nargs="*", default=["FuXiAlpha", "SASRec", "GRU4Rec", "TiSASRec"])
	parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
	args = parser.parse_args()

	rechorus_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	repo_root = os.path.abspath(os.path.join(rechorus_root, os.pardir))
	log_root = os.path.join(repo_root, "log")

	runs: List[RunInfo] = []
	for model in args.models:
		for p in glob.glob(os.path.join(log_root, model, "*.txt")):
			stem = os.path.splitext(os.path.basename(p))[0]
			parts = stem.split("__")
			if len(parts) < 3:
				continue
			m, ds, seed_s = parts[0], parts[1], parts[2]
			if m != model or ds not in set(args.datasets):
				continue
			try:
				seed = int(seed_s)
			except Exception:
				continue
			if seed not in set(args.seeds):
				continue

			cfg_epoch, best_ep, dev_m, last_ep = _parse_log(p)
			if cfg_epoch != args.epoch_cap:
				continue

			runs.append(
				RunInfo(
					path=os.path.relpath(p, repo_root),
					model=model,
					dataset=ds,
					seed=seed,
					cfg_epoch=cfg_epoch,
					best_ep=best_ep,
					dev_hr10=dev_m.get("HR@10"),
					dev_ndcg10=dev_m.get("NDCG@10"),
					last_ep=last_ep,
				)
			)

	runs.sort(key=lambda r: (r.dataset, r.model, r.seed))

	print(f"Convergence snapshot (cfg epoch={args.epoch_cap}).")
	print("Columns: dataset model seed cfg_epoch best_ep last_ep reached_cap best_at_cap dev(HR10 NDCG10) log")
	for r in runs:
		print(
			f"{r.dataset:24} {r.model:8} s{r.seed} cfg={r.cfg_epoch} best={r.best_ep} last={r.last_ep} "
			f"reached_cap={r.reached_cap} best_at_cap={r.best_at_cap} dev({_fmt(r.dev_hr10)} {_fmt(r.dev_ndcg10)})  {r.path}"
		)

	# Aggregate: how often best hits the cap
	from collections import defaultdict

	agg: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"n": 0, "best_at_cap": 0})
	for r in runs:
		k = (r.dataset, r.model)
		agg[k]["n"] += 1
		if r.best_ep == args.epoch_cap:
			agg[k]["best_at_cap"] += 1

	print("\nBest epoch hits cap rate:")
	for (ds, model), v in sorted(agg.items()):
		print(f"{ds:24} {model:8}: {v['best_at_cap']}/{v['n']}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
