"""Fine-tune Qwen2.5-0.5B-Instruct on Countdown with exact low-rank ES, in one process.

    uv run python -m seedling.llm.train --steps 40 --pop 32 --sigma 0.002 --lr 2e-5

Prints one JSON line per step and writes bench/results/llm_countdown.csv. For the
distributed version use the same workload through the runtime:

    seedling coordinator --task countdown --run-dir runs/qwen --frac-bits 24 ...
    seedling worker --connect tcp://...
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from seedling.llm.workload import CountdownWorkload
from seedling.runtime.config import RunConfig
from seedling.runtime.local import train_local

RESULTS = Path(__file__).resolve().parents[2] / "bench" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--pop", type=int, default=32, help="seed pairs per step")
    parser.add_argument("--sigma", type=float, default=0.002)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--frac-bits", type=int, default=24)
    parser.add_argument("--prompts", type=int, default=16, help="problems per step")
    parser.add_argument("--last-layers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-problems", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(RESULTS / "llm_countdown.csv"))
    ns = parser.parse_args()

    workload = CountdownWorkload(
        last_layers=ns.last_layers, prompts_per_step=ns.prompts, eval_problems=ns.eval_problems
    )
    config = RunConfig(
        task="countdown", steps=ns.steps, pop=ns.pop, sigma=ns.sigma, lr=ns.lr, frac_bits=ns.frac_bits, seed=ns.seed
    )
    t0 = time.time()
    before = workload.evaluate(workload.initial_state(ns.seed, ns.frac_bits), 0)
    print(json.dumps({"step": 0, **before}), flush=True)
    run = train_local(workload, config, eval_every=ns.eval_every, log=lambda row: print(json.dumps(row), flush=True))

    trainable = sum(s.size for s in workload.specs)
    summary = {
        "before": before,
        "after": {k: run.history[-1][k] for k in before},
        "trainable_params": trainable,
        "minutes": round((time.time() - t0) / 60, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0,
        "bytes_per_step": run.records[-1].nbytes(),
        "digest": run.state.digest(),
    }
    print(json.dumps(summary), flush=True)

    rows = [{"step": 0, **before}] + run.history
    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    with open(ns.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
