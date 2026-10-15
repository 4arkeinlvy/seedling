"""Ring all-reduce bandwidth vs message size (M1).

    uv run python -m bench.bench_allreduce --world 2 4 --backend python c

algbw = bytes / time
busbw = algbw * 2(W-1)/W  -- the bytes each link really carries in a ring all-reduce
                            (same definition as NVIDIA's nccl-tests)
Small messages are latency-bound (the alpha term), big ones bandwidth-bound (beta).
Results go to bench/results/allreduce.csv (+ .png if matplotlib is installed).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

from seedling.foundation.collectives import ProcessGroup
from seedling.foundation.launch import run_ranks

RESULTS = Path(__file__).resolve().parent / "results"


def bench_rank(rank, world, backend, sizes, iters):
    pg = ProcessGroup.from_env(backend=backend)
    try:
        out = []
        for n in sizes:
            x = np.ones(n, dtype=np.float32)
            pg.ring_all_reduce(x)  # warm-up
            times = []
            for _ in range(iters):
                x.fill(1.0)
                pg.barrier()
                t0 = time.perf_counter()
                pg.ring_all_reduce(x)
                times.append(time.perf_counter() - t0)
            out.append(statistics.median(times))
        return out
    finally:
        pg.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--world", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--backend", nargs="+", default=["python", "c"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-log2", type=int, default=24, help="largest size = 2**this floats")
    ns = parser.parse_args()

    sizes = [2**k for k in range(8, ns.max_log2 + 1, 2)]
    rows = []
    for backend in ns.backend:
        for world in ns.world:
            per_rank = run_ranks(bench_rank, world, args=(backend, sizes, ns.iters), timeout=1800)
            print(f"\nbackend={backend} world={world}")
            print(f"{'floats':>10} {'bytes':>12} {'time (us)':>12} {'algbw GB/s':>11} {'busbw GB/s':>11}")
            for i, n in enumerate(sizes):
                t = max(r[i] for r in per_rank)  # the slowest rank decides
                nbytes = 4 * n
                algbw = nbytes / t / 1e9
                busbw = algbw * 2 * (world - 1) / world
                rows.append([backend, world, n, nbytes, t, algbw, busbw])
                print(f"{n:>10} {nbytes:>12} {t * 1e6:>12.1f} {algbw:>11.3f} {busbw:>11.3f}")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "allreduce.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "world", "floats", "bytes", "seconds", "algbw_GBps", "busbw_GBps"])
        w.writerows(rows)
    print(f"\nwrote {RESULTS / 'allreduce.csv'}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for backend in ns.backend:
        for world in ns.world:
            pts = [(r[3], r[6]) for r in rows if r[0] == backend and r[1] == world]
            ax.plot(*zip(*pts, strict=True), marker="o", label=f"{backend}, W={world}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("bus bandwidth (GB/s)")
    ax.set_title("ring all-reduce on this machine")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "allreduce.png", dpi=120)
    print(f"wrote {RESULTS / 'allreduce.png'}")


if __name__ == "__main__":
    main()
