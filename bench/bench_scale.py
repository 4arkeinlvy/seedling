"""M9 experiment: dishonest workers on bigger workloads, against the defense baselines.

Every benchmark trains with 10 simulated workers. Three of them (30%) attack, or none do.
Each run uses one defense:

    none     centered ranks only (1x work)
    median3  every pair scored by 3 workers, coordinate-wise median (CyBeR-0-style
             robust aggregation, Egger et al. 2025): 3x work, no bans
    spot     spot checks at 10%, probation 10, vote and ban (M8): about 1.1x work

Attacks: flip (swap F+ and F-), lazy (zeros, no work), sleeper (honest through probation,
then flip). Each finished run is appended to bench/results/scale_<benchmark>.jsonl (runs
already there are skipped, so an interrupted sweep resumes); --report only prints tables.

    uv run python -m bench.bench_scale --bench mnist halfcheetah tinylm      # CPU, hours
    uv run python -m bench.bench_scale --bench sst2 --seeds 1 --procs 1      # needs the HF hub
    uv run python -m bench.bench_scale --bench halfcheetah --steps 20 --seeds 1   # quick look
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"
WORKERS = 10
DISHONEST = 3
ATTACKS = ("flip", "lazy", "sleeper")
DEFENSES = {
    "none": {},
    "median3": {"aggregate": "median", "replicas": 3},
    "spot": {"rate": 0.1, "probation": 10},
}
PROBATION = 10


def benchmarks() -> dict[str, dict]:
    from seedling.runtime import RunConfig

    return {
        "mnist": dict(
            config=RunConfig(task="mnist", steps=300, pop=50, sigma=0.05, lr=0.01, pairs_per_task=5),
            metric="test_accuracy", higher=True, label="MNIST MLP (0.1M), test accuracy",
        ),
        "halfcheetah": dict(
            config=RunConfig(task="halfcheetah", steps=150, pop=32, sigma=0.05, lr=0.05, pairs_per_task=4),
            metric="return", higher=True, label="HalfCheetah-v5 MLP, return (1000 steps)",
        ),
        "tinylm": dict(
            config=RunConfig(task="tinylm", steps=200, pop=32, sigma=0.01, lr=0.002, frac_bits=24, pairs_per_task=4),
            metric="val_loss", higher=False, label="Char GPT (0.8M, trained by ES), val loss",
        ),
        "sst2": dict(
            config=RunConfig(task="sst2", steps=100, pop=16, sigma=0.002, lr=3e-4, frac_bits=24, pairs_per_task=2),
            metric="accuracy", higher=True, label="SmolLM2-135M on SST-2, accuracy",
        ),
        "countdown": dict(  # GPU only; see bench/gpu/README.md
            config=RunConfig(task="countdown", steps=200, pop=30, sigma=0.001, lr=2e-5, frac_bits=24, pairs_per_task=2,
                             task_args={"protocol": "tinyzero", "max_new_tokens": 512}),
            metric="accuracy", higher=True, label="Qwen2.5-0.5B on Countdown (TinyZero protocol), accuracy",
        ),
    }  # fmt: skip


def _init(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def run_one(job: dict) -> dict:
    from seedling.runtime.audit import AuditPolicy
    from seedling.runtime.simulate import simulate

    bench = benchmarks()[job["bench"]]
    config = replace(bench["config"], seed=job["seed"], steps=job["steps"] or bench["config"].steps)
    bad = 0 if job["attack"] == "honest" else DISHONEST
    behaviours = ["honest"] * (WORKERS - bad) + [job["attack"]] * bad
    audit = AuditPolicy(**DEFENSES[job["defense"]])
    r = simulate(config, behaviours, audit, seed=job["seed"], sleeper_after=2 * PROBATION)
    return {
        **job,
        "finished": r.finished,
        "metric": r.metrics[bench["metric"]],
        "lies": r.lies_committed,
        "cost": r.cost,
        "liars_banned": r.banned(honest=False),
        "honest_banned": r.banned(honest=True),
        "digest": r.digest,
    }


def jobs(names: list[str], steps: int, seeds: int) -> list[dict]:
    return [
        dict(bench=b, defense=d, attack=a, seed=s, steps=steps)
        for b, s, d, a in itertools.product(names, range(seeds), DEFENSES, ("honest", *ATTACKS))
    ]


def summarize(rows: list[dict]) -> list[dict]:
    honest = {(r["bench"], r["seed"]): r["digest"] for r in rows if r["attack"] == "honest" and r["defense"] == "none"}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["bench"], r["attack"], r["defense"])].append(r)
    attacks, defenses = ("honest", *ATTACKS), list(DEFENSES)
    table = []
    for (bench, attack, defense), rs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], attacks.index(kv[0][1]), defenses.index(kv[0][2]))
    ):
        metric = np.array([r["metric"] for r in rs])
        table.append({
            "bench": bench, "attack": attack, "defense": defense, "runs": len(rs),
            "metric": float(metric.mean()), "min": float(metric.min()), "max": float(metric.max()),
            "lies": float(np.mean([r["lies"] for r in rs])),
            "cost": float(np.mean([r["cost"] for r in rs])),
            "liars_banned": f"{sum(r['liars_banned'] for r in rs)}/{(0 if attack == 'honest' else DISHONEST) * len(rs)}",
            "honest_banned": sum(r["honest_banned"] for r in rs),
            "bit_identical": sum(r["digest"] == honest.get((bench, r["seed"])) for r in rs),
        })  # fmt: skip
    return table


def markdown(table: list[dict], bench: str) -> str:
    info = benchmarks()[bench]
    lines = [
        f"**{info['label']}** ({'higher' if info['higher'] else 'lower'} is better)",
        "",
        "| attack (30%) | defense | metric: mean (min–max over seeds) | lies committed | liars banned | honest banned | work | = honest run |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in table:
        if t["bench"] != bench:
            continue
        spread = f" ({t['min']:.4g}–{t['max']:.4g})" if t["runs"] > 1 and t["min"] != t["max"] else ""
        lines.append(
            f"| {t['attack']} | {t['defense']} | {t['metric']:.4g}{spread} | {t['lies']:.0f} | {t['liars_banned']} "
            f"| {t['honest_banned']} | {t['cost']:.2f}x | {t['bit_identical']}/{t['runs']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--bench", nargs="+", default=["mnist", "halfcheetah", "tinylm"], choices=sorted(benchmarks()))
    parser.add_argument("--steps", type=int, default=0, help="override each benchmark's step count")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1, help="torch threads per process")
    parser.add_argument("--report", action="store_true", help="print tables from saved runs, run nothing")
    ns = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = {bench: load(bench) for bench in ns.bench}
    key = lambda r: (r["bench"], r["defense"], r["attack"], r["seed"], r["steps"])  # noqa: E731
    seen = {key(r) for rows in done.values() for r in rows}
    todo = [] if ns.report else [j for j in jobs(ns.bench, ns.steps, ns.seeds) if key(j) not in seen]
    os.environ.setdefault("OMP_NUM_THREADS", str(ns.threads))
    with multiprocessing.get_context("spawn").Pool(ns.procs, initializer=_init, initargs=(ns.threads,)) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, todo), 1):
            done[row["bench"]].append(row)
            with open(RESULTS / f"scale_{row['bench']}.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{k}/{len(todo)}] {row['bench']} {row['attack']}/{row['defense']} seed {row['seed']}: "
                  f"{row['metric']:.4g} (lies {row['lies']}, work {row['cost']:.2f}x)", flush=True)  # fmt: skip
    for bench, rows in done.items():
        print(markdown(summarize(rows), bench) + "\n", flush=True)


def load(bench: str) -> list[dict]:
    path = RESULTS / f"scale_{bench}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


if __name__ == "__main__":
    main()
