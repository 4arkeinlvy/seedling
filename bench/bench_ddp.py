"""DDP step time on this laptop: compute vs communication, overlap on/off (M2).

    uv run python -m bench.bench_ddp                 # W = 1, 2
    uv run python -m bench.bench_ddp --world 1 2 4   # W = 4 needs ~1.6 GB free RAM

Same global batch (16 x 64 tokens) for every W: the work is split, not repeated.
Each configuration gets 4 CPU threads in total (4 // W per rank).
"""

from __future__ import annotations

import argparse
import statistics

from seedling.foundation.launch import run_ranks

WARMUP = 5


def bench_rank(rank, world, backend, overlap, steps, bucket_mb):
    import torch

    from seedling.foundation.collectives import ProcessGroup
    from seedling.tasks.tiny_gpt import TrainConfig
    from seedling.tasks.train_ddp import train_ddp

    torch.set_num_threads(max(1, 4 // world))
    pg = ProcessGroup.from_env(backend=backend)
    ddp = None
    try:
        cfg = TrainConfig(steps=steps + WARMUP)
        ddp, _, seconds = train_ddp(pg, cfg, overlap=overlap, bucket_cap_mb=bucket_mb)
        s = ddp.stats
        return {
            "step_ms": 1000 * statistics.median(seconds[WARMUP:]),
            "comm_ms": 1000 * s.comm_seconds / s.steps,
            "exposed_ms": 1000 * s.wait_seconds / s.steps,
            "payload_kb": s.payload_bytes / s.steps / 1024,
            "wire_kb": s.wire_bytes / s.steps / 1024,
            "buckets": len(ddp.buckets),
        }
    finally:
        if ddp is not None:
            ddp.close()
        pg.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--world", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--backend", nargs="+", default=["python", "c"])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--bucket-mb", type=float, default=0.1)
    ns = parser.parse_args()

    header = f"{'W':>2} {'backend':>7} {'overlap':>7} {'buckets':>7} {'step ms':>8} {'comm ms':>8} {'exposed ms':>10} {'payload KB':>10} {'wire KB':>8}"
    print(header)
    baseline = None
    for world in ns.world:
        configs = [("python", False)] if world == 1 else [(b, o) for b in ns.backend for o in (False, True)]
        for backend, overlap in configs:
            reports = run_ranks(
                bench_rank, world, args=(backend, overlap, ns.steps, ns.bucket_mb), timeout=1800
            )
            r = max(reports, key=lambda rep: rep["step_ms"])
            baseline = baseline or r["step_ms"]
            print(
                f"{world:>2} {backend:>7} {str(overlap):>7} {r['buckets']:>7} {r['step_ms']:>8.1f} "
                f"{r['comm_ms']:>8.1f} {r['exposed_ms']:>10.1f} {r['payload_kb']:>10.0f} {r['wire_kb']:>8.0f}"
                f"   speedup x{baseline / r['step_ms']:.2f}"
            )


if __name__ == "__main__":
    main()
