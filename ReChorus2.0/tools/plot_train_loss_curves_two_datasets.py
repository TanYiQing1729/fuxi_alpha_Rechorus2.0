"""Plot epoch-train-loss curves for multiple models on two datasets.

This project stores epoch loss in text logs under the workspace-level `log/` folder.
Despite the user phrasing "based on .pt", the `.pt` checkpoints typically do not
contain the full per-epoch loss history, so we parse the training logs.

Outputs:
- ReChorus2.0/figures/train_loss_MovieLens_1M_ep50_seed0.png
- ReChorus2.0/figures/train_loss_Grocery_and_Gourmet_Food_grocery_ep50_h50_cmp_seed0.png

Usage (from repo root):
  python ReChorus2.0/tools/plot_train_loss_curves_two_datasets.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Use non-interactive backend to avoid Windows Tcl/Tk issues.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DatasetSpec:
    tag: str
    title: str
    output_stem: str


EPOCH_LOSS_RE = re.compile(r"Epoch\s+(\d+)\s+loss=([0-9]*\.?[0-9]+)")
BEGIN_RE = re.compile(r"BEGIN:")


def _split_into_runs(text: str) -> list[str]:
    """Split a log into run chunks.

    Some logs may contain multiple runs appended over time.
    We split on 'BEGIN:' and keep non-empty chunks.
    """

    parts = BEGIN_RE.split(text)
    # If there is no BEGIN marker, keep the entire file as one run.
    if len(parts) <= 1:
        return [text]

    runs: list[str] = []
    # The first part is preamble before the first BEGIN; usually empty.
    for part in parts[1:]:
        part = part.strip()
        if part:
            runs.append(part)
    return runs or [text]


def parse_epoch_loss_from_log(log_path: Path) -> tuple[list[int], list[float]]:
    """Parse (epoch, loss) sequence from a training log.

    Strategy:
    - If multiple runs exist in a single file, prefer the run with the most epochs;
      break ties by picking the latest run.
    - If duplicate epoch numbers appear, keep the last occurrence.
    """

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    runs = _split_into_runs(text)

    best_epochs: list[int] = []
    best_losses: list[float] = []

    for run_text in runs:
        epoch_to_loss: dict[int, float] = {}
        for match in EPOCH_LOSS_RE.finditer(run_text):
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epoch_to_loss[epoch] = loss

        if not epoch_to_loss:
            continue

        epochs = sorted(epoch_to_loss)
        losses = [epoch_to_loss[e] for e in epochs]

        # Prefer longer curves; if equal length, prefer the later run.
        if len(epochs) > len(best_epochs) or (len(epochs) == len(best_epochs) and epochs and epochs[-1] >= (best_epochs[-1] if best_epochs else -1)):
            best_epochs, best_losses = epochs, losses

    return best_epochs, best_losses


def find_single_log(log_dir: Path, model_name: str, dataset_tag: str, seed: int = 0) -> Path:
    pattern = f"{model_name}__{dataset_tag}__{seed}__*.txt"
    matches = sorted((log_dir / model_name).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No log matched: log/{model_name}/{pattern}")
    return matches[0]


def plot_dataset_curves(
    *,
    log_dir: Path,
    fig_dir: Path,
    dataset: DatasetSpec,
    models: Iterable[str],
    seed: int = 0,
) -> Path:
    plt.figure(figsize=(8, 5))

    found_any = False
    for model in models:
        log_path = find_single_log(log_dir, model, dataset.tag, seed=seed)
        epochs, losses = parse_epoch_loss_from_log(log_path)
        if not epochs:
            print(f"[WARN] No epoch-loss parsed for {model} from {log_path}")
            continue

        found_any = True
        plt.plot(epochs, losses, label=model, linewidth=2)

    if not found_any:
        raise RuntimeError(f"No curves parsed for dataset {dataset.tag}")

    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(dataset.title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_path = fig_dir / f"{dataset.output_stem}_seed{seed}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    log_dir = repo_root / "log"
    fig_dir = repo_root / "ReChorus2.0" / "figures"

    models = ["FuXiAlpha", "SASRec", "TiSASRec", "GRU4Rec"]

    datasets = [
        DatasetSpec(
            tag="MovieLens_1M_ep50",
            title="MovieLens_1M (appendix: _ep50) — Train Loss",
            output_stem="train_loss_MovieLens_1M_ep50",
        ),
        DatasetSpec(
            tag="Grocery_and_Gourmet_Food_grocery_ep50_h50_cmp",
            title="Grocery_and_Gourmet_Food (appendix: _grocery_ep50_h50_cmp) — Train Loss",
            output_stem="train_loss_Grocery_and_Gourmet_Food_grocery_ep50_h50_cmp",
        ),
    ]

    seed = 0
    out_paths: list[Path] = []
    for ds in datasets:
        out_paths.append(
            plot_dataset_curves(
                log_dir=log_dir,
                fig_dir=fig_dir,
                dataset=ds,
                models=models,
                seed=seed,
            )
        )

    print("Generated:")
    for p in out_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
