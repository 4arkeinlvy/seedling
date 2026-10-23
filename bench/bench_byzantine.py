"""M8 experiment: MNIST trained by 10 simulated workers, some of them dishonest.

For each attack and each fraction of dishonest workers, train with and without spot checks
and report test accuracy, lies committed, bans, extra work, and whether the final weights
are bit-identical to an all-honest run with the same seed.

    uv run python -m bench.bench_byzantine                 # ~6 minutes on 4 cores
    uv run python -m bench.bench_byzantine --steps 60 --seeds 1   # quick look
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"
ATTACKS = ("flip", "inflate", "lazy", "random", "sleeper")
WORKERS = 10
AUDIT = {"rate": 0.1, "probation": 10}


def _init() -> None:
    import torch

    torch.set_num_threads(1)


def run_one(job: dict) -> dict:
    from seedling.runtime import RunConfig
    from seedling.runtime.audit import AuditPolicy
    from seedling.runtime.simulate import simulate

    config = RunConfig(
        task="mnist", steps=job["steps"], pop=50, sigma=0.05, lr=0.01,
        pairs_per_task=5, deadline=30.0, seed=job["seed"],
    )  # fmt: skip
    bad = job["dishonest"]
    behaviours = ["honest"] * (WORKERS - bad) + [job["attack"]] * bad
    audit = AuditPolicy(**AUDIT) if job["audit"] else AuditPolicy()
    r = simulate(config, behaviours, audit, seed=job["seed"], sleeper_after=2 * AUDIT["probation"])
    return {
        **job,
        "finished": r.finished,
        "accuracy": r.metrics["test_accuracy"],
        "lies": r.lies_committed,
        "dropped": r.pairs_dropped,
        "cost": r.cost,
        "liars_banned": r.banned(honest=False),
        "honest_banned": r.banned(honest=True),
        "digest": r.digest,
    }


def jobs(steps: int, seeds: int) -> list[dict]:
    out = []
    for seed, audit in itertools.product(range(seeds), (False, True)):
        out.append(dict(attack="none", dishonest=0, audit=audit, seed=seed, steps=steps))
        for attack, bad in itertools.product(ATTACKS, (1, 2, 3, 4)):
            out.append(dict(attack=attack, dishonest=bad, audit=audit, seed=seed, steps=steps))
    return out


def summarize(rows: list[dict]) -> list[dict]:
    honest = {r["seed"]: r["digest"] for r in rows if r["attack"] == "none" and not r["audit"]}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["attack"], r["dishonest"], r["audit"])].append(r)
    table = []
    order = ("none", *ATTACKS)
    for (attack, bad, audit), rs in sorted(groups.items(), key=lambda kv: (order.index(kv[0][0]), *kv[0][1:])):
        acc = np.array([r["accuracy"] for r in rs])
        table.append({
            "attack": attack, "dishonest": bad, "audit": audit, "runs": len(rs),
            "accuracy": float(acc.mean()), "acc_min": float(acc.min()), "acc_max": float(acc.max()),
            "lies": float(np.mean([r["lies"] for r in rs])),
            "cost": float(np.mean([r["cost"] for r in rs])),
            "liars_banned": f"{sum(r['liars_banned'] for r in rs)}/{bad * len(rs)}",
            "honest_banned": sum(r["honest_banned"] for r in rs),
            "bit_identical": f"{sum(r['digest'] == honest[r['seed']] for r in rs)}/{len(rs)}",
            "finished": all(r["finished"] for r in rs),
        })  # fmt: skip
    return table


def markdown(table: list[dict]) -> str:
    lines = [
        "| attack | dishonest | spot checks | accuracy | lies committed | liars banned | honest banned | work | = honest run |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in table:
        lines.append(
            f"| {t['attack']} | {10 * t['dishonest']}% | {'on' if t['audit'] else 'off'} "
            f"| {100 * t['accuracy']:.1f}% | {t['lies']:.0f} | {t['liars_banned']} | {t['honest_banned']} "
            f"| {t['cost']:.2f}x | {t['bit_identical']} |"
        )
    return "\n".join(lines)


def plot(table: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = {False: ("no spot checks", "#eb6834"), True: ("spot checks (10%, probation 10)", "#2a78d6")}
    base = {t["audit"]: t for t in table if t["attack"] == "none"}
    plt.rcParams.update({"font.size": 9, "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted})  # fmt: skip
    fig, axes = plt.subplots(1, len(ATTACKS), figsize=(12, 3.0), sharey=True, facecolor=surface)
    for ax, attack in zip(axes, ATTACKS, strict=True):
        ax.set_facecolor(surface)
        ax.axhline(100 * base[False]["accuracy"], color=muted, lw=1, ls=(0, (3, 3)), zorder=1)
        for audit, (label, color) in series.items():
            pts = [base[audit]] + sorted(
                (t for t in table if t["attack"] == attack and t["audit"] == audit), key=lambda t: t["dishonest"]
            )
            x = [10 * t["dishonest"] for t in pts]
            y = [100 * t["accuracy"] for t in pts]
            lo = [100 * (t["accuracy"] - t["acc_min"]) for t in pts]
            hi = [100 * (t["acc_max"] - t["accuracy"]) for t in pts]
            ax.errorbar(x, y, yerr=[lo, hi], color=color, lw=2, marker="o", ms=5, capsize=0, elinewidth=1,
                        mec=surface, mew=1.5, label=label, zorder=3)  # fmt: skip
        ax.set_title(attack, color=ink, fontsize=10, loc="left")
        ax.set_xticks([0, 10, 20, 30, 40], ["0", "10", "20", "30", "40%"])
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(length=0)
        ax.set_xlabel("dishonest workers")
    axes[0].set_ylabel("MNIST test accuracy (%)")
    axes[0].annotate("all-honest run", (40, 100 * base[False]["accuracy"]), xytext=(0, 4),
                     textcoords="offset points", ha="right", color=muted, fontsize=8)  # fmt: skip
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle("Dishonest workers vs. spot checks: 10 workers, 300 steps", x=0.01, ha="left", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=surface)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--plot", default=str(REPO / "assets" / "byzantine.png"))
    ns = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    todo = jobs(ns.steps, ns.seeds)
    with multiprocessing.get_context("spawn").Pool(ns.procs, initializer=_init) as pool:
        rows = []
        for k, row in enumerate(pool.imap_unordered(run_one, todo), 1):
            rows.append(row)
            print(f"[{k}/{len(todo)}] {row['attack']} x{row['dishonest']} audit={row['audit']} "
                  f"seed={row['seed']}: {100 * row['accuracy']:.1f}%", flush=True)  # fmt: skip
    (RESULTS / "byzantine.json").write_text(json.dumps(rows, indent=1))
    table = summarize(rows)
    (RESULTS / "byzantine.md").write_text(markdown(table) + "\n")
    print(markdown(table))
    plot(table, Path(ns.plot))
    print(f"wrote {ns.plot}")


if __name__ == "__main__":
    main()
