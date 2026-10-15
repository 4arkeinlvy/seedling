"""Code that runs inside each rank during the multi-process tests.

It lives here, not in the test files, because ranks start with "spawn": every child
imports this module by name. Keep the imports light (no torch at the top), so each rank
stays small in RAM.
"""

from __future__ import annotations

import hashlib
import os
import signal
import time

import numpy as np

from seedling.foundation.collectives import ProcessGroup
from seedling.foundation.store import TCPStore

# ------------------------------------------------------------------ launcher (given code)


def echo(rank, world):
    return rank, world, os.environ["RANK"], os.environ["WORLD_SIZE"]


def fail_on_rank_1(rank, world):
    if rank == 1:
        raise ValueError("boom from rank 1")
    time.sleep(30)  # the launcher must not wait for these


def sleep_forever(rank, world):
    time.sleep(3600)


def die_on_rank_0(rank, world):
    if rank == 0:
        os.kill(os.getpid(), signal.SIGKILL)  # like a segfault: no Python traceback
    time.sleep(30)


# ------------------------------------------------------------------ store


def _store(rank, world, timeout=10.0) -> TCPStore:
    return TCPStore(
        os.environ["MASTER_ADDR"],
        int(os.environ["MASTER_PORT"]),
        is_server=(rank == 0),
        world_size=world,
        timeout=timeout,
    )


def store_atomic_add(rank, world, n_adds):
    store = _store(rank, world)
    try:
        return [store.add("counter", 1) for _ in range(n_adds)]
    finally:
        store.close()


def store_rendezvous(rank, world):
    store = _store(rank, world)
    try:
        time.sleep(0.05 * (world - rank))  # publish in reverse rank order
        store.set(f"ready/{rank}", str(rank).encode())
        store.wait([f"ready/{r}" for r in range(world)])
        return [store.get(f"ready/{r}") for r in range(world)]
    finally:
        store.close()


# ------------------------------------------------------------------ collectives

SIZES = [0, 1, 7, 1000, 1_000_003]
# 16 MB: much more than a Unix socketpair buffers (~200 KB). In tests/thread_ranks.py a
# design where every rank sends before it receives deadlocks at this size, every time.
BIG = 4_000_037


def make_input(rank: int, n: int, seed: int, exact: bool = True) -> np.ndarray:
    """Rank's input. exact=True: small integers, so float32 sums are exact in any order.
    exact=False: random floats, so the order of additions changes the last bits."""
    rng = np.random.default_rng([seed, rank])
    if exact:
        return rng.integers(-100, 101, size=n, dtype=np.int16).astype(np.float32)
    return rng.standard_normal(n, dtype=np.float32)


def expected_sum(world: int, n: int, seed: int, exact: bool = True) -> np.ndarray:
    total = np.zeros(n, dtype=np.float32 if exact else np.float64)
    for r in range(world):
        total += make_input(r, n, seed, exact)
    return total.astype(np.float32)


def mismatch(got: np.ndarray, want: np.ndarray) -> str | None:
    """None if equal, else a short description of the first difference."""
    if got.shape != want.shape:
        return f"shape {got.shape}, want {want.shape}"
    bad = np.flatnonzero(got != want)
    if bad.size == 0:
        return None
    i = int(bad[0])
    return f"{bad.size} wrong values, first at index {i}: got {got[i]}, want {want[i]}"


def digest(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def all_reduce_report(rank, world, backend, method, sizes):
    """Run pg.<method> (ring_all_reduce or all_reduce) for every size.
    Returns [(n, error or None, digest of the random-float result or None, close or None)]."""
    pg = ProcessGroup.from_env(backend=backend)
    try:
        report = []
        for n in sizes:
            x = make_input(rank, n, seed=n)
            getattr(pg, method)(x)
            err = mismatch(x, expected_sum(world, n, seed=n))
            dig = close = None
            if n <= 1_000_003:
                y = make_input(rank, n, seed=n + 1, exact=False)
                getattr(pg, method)(y)
                dig = digest(y)
                want = expected_sum(world, n, seed=n + 1, exact=False)
                close = bool(np.allclose(y, want, rtol=1e-5, atol=1e-5))
            report.append((n, err, dig, close))
        return report
    finally:
        pg.close()


def broadcast_report(rank, world, sizes, src):
    pg = ProcessGroup.from_env()
    try:
        report = []
        for n in sizes:
            x = make_input(rank, n, seed=n)
            pg.broadcast(x, src=src)
            report.append((n, mismatch(x, make_input(src, n, seed=n))))
        return report
    finally:
        pg.close()


def all_gather_report(rank, world, sizes):
    pg = ProcessGroup.from_env()
    try:
        report = []
        for n in sizes:
            got = pg.all_gather(make_input(rank, n, seed=n))
            if not isinstance(got, list) or len(got) != world:
                report.append((n, f"expected a list of {world} arrays, got {type(got).__name__}"))
                continue
            errs = [mismatch(np.asarray(got[r]), make_input(r, n, seed=n)) for r in range(world)]
            bad = [f"entry {r}: {e}" for r, e in enumerate(errs) if e]
            report.append((n, "; ".join(bad) or None))
        return report
    finally:
        pg.close()


def send_recv_report(rank, world):
    """Ping-pong between ranks 0 and 1; other ranks only join and leave."""
    pg = ProcessGroup.from_env()
    try:
        x = np.arange(1000, dtype=np.float32)
        if rank == 0:
            pg.send(x, dst=1)
            pg.send(x[:3] + 7, dst=1)  # two messages in a row keep their boundaries
            back = np.empty_like(x)
            pg.recv(back, src=1)
            pg.send(np.zeros(10, dtype=np.float32), dst=1)  # rank 1 receives it into the wrong size
            return mismatch(back, x * 2)
        if rank == 1:
            first = np.empty_like(x)
            second = np.empty(3, dtype=np.float32)
            pg.recv(first, src=0)
            pg.recv(second, src=0)
            pg.send(first * 2, dst=0)
            errs = [mismatch(first, x), mismatch(second, x[:3] + 7)]
            try:
                pg.recv(np.empty(5, dtype=np.float32), src=0)
                errs.append("recv into a too-small array did not raise ValueError")
            except ValueError:
                pass
            return "; ".join(e for e in errs if e) or None
        return None
    finally:
        pg.close()


def barrier_times(rank, world):
    pg = ProcessGroup.from_env()
    try:
        time.sleep(0.3 * rank)
        arrive = time.monotonic()
        pg.barrier()
        return arrive, time.monotonic()
    finally:
        pg.close()


def compare_with_gloo(rank, world, gloo_port):
    """Same inputs through our collectives and torch.distributed (gloo)."""
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{gloo_port}", rank=rank, world_size=world
    )
    pg = ProcessGroup.from_env()
    try:
        x = make_input(rank, 10_007, seed=5, exact=False)
        out = {}

        ours = x.copy()
        pg.ring_all_reduce(ours)
        theirs = torch.from_numpy(x.copy())
        dist.all_reduce(theirs)
        out["all_reduce"] = bool(np.allclose(ours, theirs.numpy(), rtol=1e-6, atol=1e-6))

        ours = x.copy()
        pg.broadcast(ours, src=world - 1)
        theirs = torch.from_numpy(x.copy())
        dist.broadcast(theirs, src=world - 1)
        out["broadcast"] = bool(np.array_equal(ours, theirs.numpy()))

        ours_list = pg.all_gather(x)
        theirs_list = [torch.empty(x.size) for _ in range(world)]
        dist.all_gather(theirs_list, torch.from_numpy(x))
        out["all_gather"] = all(
            np.array_equal(a, b.numpy()) for a, b in zip(ours_list, theirs_list, strict=True)
        )
        return out
    finally:
        pg.close()
        dist.destroy_process_group()


# ------------------------------------------------------------------ M2: DDP in real processes


def ddp_train_report(rank, world, steps):
    import torch

    from seedling.tasks.tiny_gpt import TrainConfig
    from seedling.tasks.train_ddp import params_digest, train_ddp

    torch.set_num_threads(2)
    pg = ProcessGroup.from_env(backend="c")
    ddp = None
    try:
        ddp, losses, _ = train_ddp(pg, TrainConfig(steps=steps))
        return {"digest": params_digest(ddp.module), "loss_first": losses[0], "loss_last": losses[-1]}
    finally:
        if ddp is not None:
            ddp.close()
        pg.close()
