"""Fine-tune Qwen2.5-0.5B-Instruct on Countdown with exact low-rank ES, in one process.

    uv run python -m seedling.llm.train --steps 40 --pop 32 --sigma 0.002 --lr 2e-5

    # the TinyZero data, prompt and score of Qiu et al. (2025), population 30, sigma 0.001;
    # --run-dir journals every step (resume by running again; `seedling export` afterwards)
    uv run python -m seedling.llm.train --protocol tinyzero --pop 30 --sigma 0.001 --lr 2e-5 \
        --steps 200 --max-new-tokens 512 --run-dir runs/countdown-tz

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

from seedling.llm.workload import DEFAULT_MODEL, CountdownWorkload
from seedling.runtime.config import RunConfig
from seedling.runtime.journal import Journal
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--protocol", choices=("seedling", "tinyzero"), default="seedling")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-rows", type=int, default=128, help="generation batch size (lower it on small GPUs)")
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16", help="float16 on GPUs without bfloat16 (Kaggle T4)")
    parser.add_argument("--run-dir", default=None, help="journal every step here; running again resumes")
    parser.add_argument("--out", default=str(RESULTS / "llm_countdown.csv"))
    ns = parser.parse_args()

    task_args = dict(
        model_name=ns.model, last_layers=ns.last_layers, prompts_per_step=ns.prompts, eval_problems=ns.eval_problems,
        max_new_tokens=ns.max_new_tokens, max_rows=ns.max_rows, eval_batch=ns.eval_batch, dtype=ns.dtype,
        protocol=ns.protocol,
    )  # fmt: skip
    workload = CountdownWorkload(**task_args)
    config = RunConfig(
        task="countdown", steps=ns.steps, pop=ns.pop, sigma=ns.sigma, lr=ns.lr, frac_bits=ns.frac_bits, seed=ns.seed,
        task_args=task_args,
    )  # fmt: skip
    journal = Journal(ns.run_dir) if ns.run_dir else None
    t0 = time.time()
    before = workload.evaluate(workload.initial_state(ns.seed, ns.frac_bits), 0)
    print(json.dumps({"step": 0, **before}), flush=True)
    run = train_local(
        workload, config, eval_every=ns.eval_every, log=lambda row: print(json.dumps(row), flush=True), journal=journal
    )

    trainable = sum(s.size for s in workload.specs)
    summary = {
        "before": before,
        "after": {k: run.history[-1][k] for k in before} if run.history else before,
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
