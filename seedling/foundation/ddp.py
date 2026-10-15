"""M2: data-parallel training from scratch (what torch's DistributedDataParallel does).

    ddp = DDP(model, pg, bucket_cap_mb=0.1, overlap=True)
    _, loss = ddp(x_local, y_local)
    opt.zero_grad()
    ddp.backward(loss)
    opt.step()

CPU float32 only: the transport is our CPU sockets. Explained in docs/M2-explained.md.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from seedling.foundation.collectives import ProcessGroup


@dataclass
class SyncStats:
    """Totals since the last reset."""

    steps: int = 0
    payload_bytes: int = 0
    wire_bytes: float = 0.0
    comm_seconds: float = 0.0
    wait_seconds: float = 0.0

    def reset(self) -> None:
        self.steps = self.payload_bytes = 0
        self.wire_bytes = self.comm_seconds = self.wait_seconds = 0.0


class Bucket:
    """Parameters whose gradients travel together as one flat float32 buffer."""

    def __init__(self, index: int, params: list[nn.Parameter]):
        self.index = index
        self.params = params
        self.offsets = [0]
        for p in params:
            self.offsets.append(self.offsets[-1] + p.numel())
        self.buffer = torch.zeros(self.offsets[-1], dtype=torch.float32)
        self.ready = [False] * len(params)

    @property
    def pending(self) -> int:
        return self.ready.count(False)

    def _slice(self, i: int) -> torch.Tensor:
        return self.buffer[self.offsets[i] : self.offsets[i + 1]]

    def load_grad(self, i: int) -> None:
        """Copy parameter i's gradient (zeros if it has none) into the buffer."""
        grad = self.params[i].grad
        if grad is None:
            self._slice(i).zero_()
        else:
            self._slice(i).copy_(grad.reshape(-1))
        self.ready[i] = True

    def load_missing(self) -> None:
        for i, done in enumerate(self.ready):
            if not done:
                self.load_grad(i)

    def store_grads(self) -> None:
        """Write the buffer back into every p.grad and get ready for the next step."""
        for i, p in enumerate(self.params):
            value = self._slice(i).view_as(p)
            if p.grad is None:
                p.grad = value.clone()
            else:
                p.grad.copy_(value)
        self.ready = [False] * len(self.params)


def make_buckets(params: list[nn.Parameter], cap_bytes: int) -> list[Bucket]:
    """Pack params, in order, into buckets of at most cap_bytes (a bigger param gets its own)."""
    groups: list[list[nn.Parameter]] = []
    size = cap_bytes + 1
    for p in params:
        nbytes = p.numel() * p.element_size()
        if size + nbytes > cap_bytes:
            groups.append([])
            size = 0
        groups[-1].append(p)
        size += nbytes
    return [Bucket(i, g) for i, g in enumerate(groups)]


class GradReducer(ABC):
    """Strategy for averaging bucket gradients across ranks."""

    def __init__(self, pg: ProcessGroup, buckets: list[Bucket], stats: SyncStats):
        self.pg = pg
        self.buckets = buckets
        self.stats = stats

    def all_reduce(self, bucket: Bucket) -> None:
        t0 = time.perf_counter()
        flat = bucket.buffer.numpy()
        self.pg.ring_all_reduce(flat)
        flat /= self.pg.world_size
        self.stats.comm_seconds += time.perf_counter() - t0

    def grad_ready(self, p: nn.Parameter) -> None:
        """Called from backward when p.grad is complete; strategies may start early."""

    @abstractmethod
    def wait(self) -> None:
        """Return once every bucket holds the average over all ranks."""

    def close(self) -> None:
        pass


class SyncAfterBackward(GradReducer):
    """Simplest strategy: all-reduce every bucket after backward has finished."""

    def wait(self) -> None:
        for bucket in self.buckets:
            bucket.load_missing()
            self.all_reduce(bucket)


class OverlapWithBackward(GradReducer):
    """All-reduce each bucket on a background thread as soon as backward fills it."""

    def __init__(self, pg: ProcessGroup, buckets: list[Bucket], stats: SyncStats, timeout: float):
        super().__init__(pg, buckets, stats)
        self.timeout = timeout
        self._where = {id(p): (b, i) for b in buckets for i, p in enumerate(b.params)}
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._step_done = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def grad_ready(self, p: nn.Parameter) -> None:
        bucket, i = self._where[id(p)]
        bucket.load_grad(i)
        if bucket.pending == 0:
            self._queue.put(bucket.index)

    def _run(self) -> None:
        ready: set[int] = set()
        next_index = 0
        while (index := self._queue.get()) is not None:
            ready.add(index)
            while next_index in ready:
                ready.remove(next_index)
                try:
                    self.all_reduce(self.buckets[next_index])
                except BaseException as e:
                    self._error = e
                next_index += 1
            if next_index == len(self.buckets):
                next_index = 0
                self._step_done.set()

    def wait(self) -> None:
        for bucket in self.buckets:
            if bucket.pending:
                bucket.load_missing()
                self._queue.put(bucket.index)
        if not self._step_done.wait(self.timeout):
            raise TimeoutError("gradient all-reduce did not finish (a rank died or hung?)")
        self._step_done.clear()
        if self._error is not None:
            raise RuntimeError("gradient all-reduce failed") from self._error

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=5)


class DDP(nn.Module):
    """Wraps a model so that every rank ends each step with the same averaged gradients."""

    def __init__(
        self,
        module: nn.Module,
        pg: ProcessGroup,
        bucket_cap_mb: float = 0.1,
        overlap: bool = True,
        timeout: float = 300.0,
    ):
        super().__init__()
        _check_cpu_float32(module)
        self.module = module
        self.pg = pg
        self.stats = SyncStats()
        self._sync = True
        self._broadcast_from_rank0()

        params = [p for p in module.parameters() if p.requires_grad]
        self.buckets = make_buckets(params[::-1], int(bucket_cap_mb * 2**20))
        self._hooks: list = []
        if overlap:
            self.reducer: GradReducer = OverlapWithBackward(pg, self.buckets, self.stats, timeout)
            self._hooks = [p.register_post_accumulate_grad_hook(self._on_grad_ready) for p in params]
        else:
            self.reducer = SyncAfterBackward(pg, self.buckets, self.stats)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def _broadcast_from_rank0(self) -> None:
        for t in [*self.module.parameters(), *self.module.buffers()]:
            self.pg.broadcast(t.detach().numpy(), src=0)

    def _on_grad_ready(self, p: nn.Parameter) -> None:
        if self._sync:
            self.reducer.grad_ready(p)

    def finish_grad_sync(self) -> None:
        """Average p.grad over all ranks. Call after backward, before optimizer.step()."""
        if not self._sync:
            return
        t0 = time.perf_counter()
        self.reducer.wait()
        self.stats.wait_seconds += time.perf_counter() - t0
        for bucket in self.buckets:
            bucket.store_grads()
        self._count_step()

    def _count_step(self) -> None:
        payload = sum(b.buffer.numel() * b.buffer.element_size() for b in self.buckets)
        world = self.pg.world_size
        self.stats.steps += 1
        self.stats.payload_bytes += payload
        self.stats.wire_bytes += payload * 2 * (world - 1) / world

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()
        self.finish_grad_sync()

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Backward passes inside this block only accumulate gradients locally."""
        before, self._sync = self._sync, False
        try:
            yield
        finally:
            self._sync = before

    def reset_stats(self) -> None:
        self.stats.reset()

    def close(self) -> None:
        """Stop the background thread and remove the hooks (the ProcessGroup stays open)."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self.reducer.close()


def _check_cpu_float32(module: nn.Module) -> None:
    for t in [*module.parameters(), *module.buffers()]:
        if t.device.type != "cpu" or t.dtype != torch.float32 or not t.is_contiguous():
            raise ValueError("this DDP needs contiguous CPU float32 tensors")
