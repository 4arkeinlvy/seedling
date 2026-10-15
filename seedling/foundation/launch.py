"""Run one function in N processes ("ranks"), like a tiny `torchrun`.

    results = run_ranks(fn, world_size=4, args=(...))

Each child process gets the env vars RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT,
then calls fn(rank, world_size, *args). Return values come back to the parent in rank
order (they must be picklable). If any rank raises, all ranks are stopped and RankError
shows that child's traceback. If they run longer than `timeout` (a deadlock, usually),
all ranks are killed and TimeoutError is raised.

fn must be importable by name (defined at the top level of a module), because children
are started fresh with the "spawn" method: they don't inherit the parent's memory, so
they start fast and small even when the parent has imported torch.

CLI:  python -m seedling.foundation.launch -n 4 some.module:function
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import queue
import signal
import socket
import time
import traceback
from collections.abc import Callable
from typing import Any


class RankError(RuntimeError):
    """A rank raised an exception or died."""


def free_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for a free TCP port. (Another program could grab it before we use it,
    but that's rare on a laptop.)"""
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _child(fn, rank, world_size, args, addr, port, results) -> None:
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), MASTER_ADDR=addr, MASTER_PORT=str(port)
    )
    try:
        results.put((rank, True, fn(rank, world_size, *args)))
    except BaseException:
        results.put((rank, False, traceback.format_exc()))


def _describe_exit(code: int) -> str:
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exit code {code}"


def run_ranks(
    fn: Callable[..., Any],
    world_size: int,
    args: tuple = (),
    timeout: float = 60.0,
    addr: str = "127.0.0.1",
) -> list[Any]:
    ctx = mp.get_context("spawn")
    port = free_port(addr)
    results = ctx.Queue()
    procs = [
        ctx.Process(
            target=_child, args=(fn, r, world_size, args, addr, port, results), daemon=True
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()

    out: list[Any] = [None] * world_size
    pending = set(range(world_size))
    deadline = time.monotonic() + timeout
    try:
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"ranks {sorted(pending)} did not finish within {timeout:.0f}s "
                    "(deadlock? a rank waiting for a message nobody sends?)"
                )
            try:
                rank, ok, payload = results.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                for r in pending:
                    code = procs[r].exitcode
                    if code is not None and code != 0:
                        raise RankError(
                            f"rank {r} died without a Python error ({_describe_exit(code)}). "
                            "A crash inside C code? Try the C tests under ASan: make test-c"
                        ) from None
                continue
            if not ok:
                last_line = payload.strip().splitlines()[-1]  # e.g. "ValueError: boom"
                raise RankError(f"rank {rank} failed: {last_line}\n\n{payload}")
            out[rank] = payload
            pending.discard(rank)
        for p in procs:
            p.join(timeout=10)
        return out
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
        for p in procs:
            p.join(timeout=5)
        results.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run module:function in N ranks.")
    parser.add_argument("-n", "--nproc", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("target", help="module:function, called as function(rank, world_size)")
    ns = parser.parse_args()
    module_name, _, fn_name = ns.target.partition(":")
    fn = getattr(importlib.import_module(module_name), fn_name)
    for rank, result in enumerate(run_ranks(fn, ns.nproc, timeout=ns.timeout)):
        print(f"rank {rank}: {result!r}")


if __name__ == "__main__":
    main()
