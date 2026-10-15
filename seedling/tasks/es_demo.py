"""Run the M3 experiments and save their curves to bench/results/es_<task>.csv.

    uv run python -m seedling.tasks.es_demo --task cartpole
    uv run python -m seedling.tasks.es_demo --task mnist --steps 1000
    uv run python -m seedling.tasks.es_demo --task gpt --steps 600 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parents[2] / "bench" / "results"


def run_cartpole(steps: int, device: str) -> list[dict]:
    from seedling.tasks.cartpole import train_es

    _, history = train_es(generations=steps)
    return [dict(row, method="es") for row in history]


def run_mnist(steps: int, device: str) -> list[dict]:
    from seedling.tasks import mnist

    data = mnist.load()
    _, es_rows = mnist.train_es(data, steps=steps, device=device)
    _, bp_rows = mnist.train_backprop(data, steps=steps, device=device)
    return [dict(r, method="es") for r in es_rows] + [dict(r, method="backprop") for r in bp_rows]


def run_gpt(steps: int, device: str) -> list[dict]:
    from seedling.tasks.es_gpt import train_es_gpt
    from seedling.tasks.tiny_gpt import TrainConfig, train

    _, es_rows = train_es_gpt(steps=steps, device=device)
    _, losses = train(TrainConfig(steps=steps, device=device))
    bp_rows = [{"step": i + 1, "forward_passes": i + 1, "train_loss": v} for i, v in enumerate(losses)]
    return [dict(r, method="es") for r in es_rows] + [dict(r, method="backprop") for r in bp_rows]


TASKS = {"cartpole": (run_cartpole, 60), "mnist": (run_mnist, 300), "gpt": (run_gpt, 300)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ns = parser.parse_args()

    run, default_steps = TASKS[ns.task]
    t0 = time.time()
    rows = run(ns.steps or default_steps, ns.device)
    seconds = time.time() - t0

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"es_{ns.task}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shown = rows if len(rows) <= 20 else rows[:: max(1, len(rows) // 20)]
    for row in shown:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    print(f"{len(rows)} rows in {seconds:.1f}s -> {path}")


if __name__ == "__main__":
    main()
