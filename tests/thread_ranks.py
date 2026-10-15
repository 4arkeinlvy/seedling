"""Ranks as threads in one process, joined by Unix socketpairs.

Why: a Unix socketpair buffers only ~200 KB, with no TCP timers in the way. So if your
collective sends a 16 MB chunk before it receives, every rank blocks in sendall() and
this harness reports the deadlock within seconds, every time. (Over TCP on loopback the
kernel can buffer ~32 MB, so the same bug can pass by luck.)

Python threads release the GIL while they wait in send/recv, so threads deadlock exactly
like processes would. One process also means little RAM.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from typing import Any

from seedling.foundation.collectives import ProcessGroup


def run_threads(
    world: int, fn: Callable[..., Any], *args: Any, backend: str = "python", timeout: float = 30.0
) -> list[Any]:
    """Call fn(pg, *args) for each rank in its own thread; return results in rank order."""
    peers: list[dict[int, socket.socket]] = [{} for _ in range(world)]
    socks: list[socket.socket] = []
    for i in range(world):
        for j in range(i + 1, world):
            a, b = socket.socketpair()
            peers[i][j], peers[j][i] = a, b
            socks += [a, b]
    groups = [ProcessGroup(r, world, peers[r], backend=backend) for r in range(world)]

    results: list[Any] = [None] * world
    errors: list[BaseException | None] = [None] * world

    def body(rank: int) -> None:
        try:
            results[rank] = fn(groups[rank], *args)
        except BaseException as e:  # re-raised below, in the test's thread
            errors[rank] = e

    threads = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(world)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    stuck = [r for r, t in enumerate(threads) if t.is_alive()]

    for s in socks:  # wake up anything still blocked, then clean up
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    for t in threads:
        t.join(2.0)
    for s in socks:
        s.close()

    if stuck:
        raise TimeoutError(
            f"ranks {stuck} still blocked after {timeout:.0f}s: deadlock. Does every rank "
            "send its whole chunk before receiving? Send and receive at the same time."
        )
    for e in errors:
        if e is not None:
            raise e
    return results
