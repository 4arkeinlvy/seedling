"""Plot the M3 curves written by seedling.tasks.es_demo into bench/results/es_*.png.

    uv run python -m bench.plot_es
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
PLOTS = {
    "cartpole": ("step", "policy_return", "generation", "return (max 500)"),
    "mnist": ("forward_passes", "test_accuracy", "forward passes (log)", "test accuracy"),
    "gpt": ("forward_passes", "train_loss", "forward passes (log)", "train loss"),
}


def main() -> None:
    for task, (x_key, y_key, x_label, y_label) in PLOTS.items():
        path = RESULTS / f"es_{task}.csv"
        if not path.exists():
            continue
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        with open(path) as f:
            for row in csv.DictReader(f):
                series[row["method"]].append((float(row[x_key]), float(row[y_key])))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for method, points in series.items():
            ax.plot(*zip(*points, strict=True), label=method)
        if x_key == "forward_passes":
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{task}: ES vs backprop" if len(series) > 1 else f"{task}: ES")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / f"es_{task}.png", dpi=120)
        print(f"wrote {RESULTS / f'es_{task}.png'}")


if __name__ == "__main__":
    main()
