"""M1: collectives from scratch: send/recv, broadcast, all_gather, all_reduce.

    pg = ProcessGroup.from_env()              # reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    x = np.full(1000, pg.rank, dtype=np.float32)
    pg.ring_all_reduce(x)                     # in place: x = sum over all ranks
    pg.close()

Buffers are C-contiguous numpy arrays. For a CPU torch tensor t, pass t.numpy(): it
shares memory with t, so in-place results show up in t.

A ProcessGroup is just: my rank, the world size, and one connected socket per other rank
(self.peers[p]). from_env() builds those sockets over TCP (connect_mesh below). The
tests also build them from Unix socketpairs, with ranks as threads.

The C backend (backend="c") runs ring_all_reduce in csrc/ring.c instead.
Explained in docs/M1-explained.md.
"""

from __future__ import annotations

import os
import socket
import struct
import threading

import numpy as np

from seedling import _c
from seedling.foundation.store import TCPStore
from seedling.foundation.wire import recv_msg, send_msg


def connect_mesh(rank: int, world_size: int, store: TCPStore, host: str) -> dict[int, socket.socket]:
    """Given. Every rank listens on a free port and publishes "host:port" in the store.
    Then rank r connects to every lower rank and accepts one connection from every higher
    rank. Returns {peer rank: connected socket}."""
    listener = socket.create_server((host, 0))
    listener.settimeout(store.timeout)
    port = listener.getsockname()[1]
    store.set(f"pg/addr/{rank}", f"{host}:{port}".encode())

    peers: dict[int, socket.socket] = {}
    for peer in range(rank):  # connect to every lower rank...
        peer_host, peer_port = store.get(f"pg/addr/{peer}").decode().rsplit(":", 1)
        s = socket.create_connection((peer_host, int(peer_port)), timeout=store.timeout)
        s.sendall(struct.pack("!I", rank))  # say who we are
        peers[peer] = s
    for _ in range(rank + 1, world_size):  # ...accept every higher rank
        s, _ = listener.accept()
        (peer,) = struct.unpack("!I", s.recv(4, socket.MSG_WAITALL))
        peers[peer] = s
    listener.close()

    for s in peers.values():
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Plain blocking sockets. Python implements socket timeouts by switching the fd
        # to non-blocking mode, and the C code gets the same fds.
        s.settimeout(None)
    return peers


class ProcessGroup:
    def __init__(
        self,
        rank: int,
        world_size: int,
        peers: dict[int, socket.socket],
        store: TCPStore | None = None,
        backend: str = "python",
    ):
        if backend not in ("python", "c"):
            raise ValueError(f"unknown backend {backend!r}")
        self.rank = rank
        self.world_size = world_size
        self.peers = peers
        self.store = store  # needed only by barrier() and close()
        self.backend = backend
        self._barrier_gen = 0

    @classmethod
    def from_env(cls, backend: str = "python", timeout: float = 30.0) -> ProcessGroup:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])
        store = TCPStore(host, port, is_server=(rank == 0), world_size=world_size, timeout=timeout)
        peers = connect_mesh(rank, world_size, store, host)
        return cls(rank, world_size, peers, store, backend)

    # ---------------------------------------------------------------- point to point

    def send(self, arr: np.ndarray, dst: int) -> None:
        """Send arr's raw bytes to rank dst as one framed message."""
        send_msg(self.peers[dst], memoryview(np.ascontiguousarray(arr)))

    def recv(self, arr: np.ndarray, src: int) -> None:
        """Receive one message from rank src into arr, in place (sizes must match)."""
        payload = recv_msg(self.peers[src])
        if len(payload) != arr.nbytes:
            raise ValueError(f"got {len(payload)} bytes from rank {src}, expected {arr.nbytes}")
        if arr.nbytes:
            memoryview(arr).cast("B")[:] = payload

    # ---------------------------------------------------------------- collectives

    def broadcast(self, arr: np.ndarray, src: int = 0) -> None:
        """In place: afterwards every rank's arr equals rank src's arr."""
        if self.rank == src:
            for peer in range(self.world_size):
                if peer != src:
                    self.send(arr, peer)
        else:
            self.recv(arr, src)

    def all_gather(self, arr: np.ndarray) -> list[np.ndarray]:
        """[arr of rank 0, arr of rank 1, ...]. Sends run in threads, so big arrays can't
        deadlock while this thread receives."""
        arr = np.ascontiguousarray(arr)
        out: list[np.ndarray] = [np.empty(0)] * self.world_size
        out[self.rank] = arr.copy()
        senders = [
            threading.Thread(target=self.send, args=(arr, peer))
            for peer in range(self.world_size)
            if peer != self.rank
        ]
        for t in senders:
            t.start()
        for peer in range(self.world_size):
            if peer != self.rank:
                buf = np.empty_like(arr)
                self.recv(buf, peer)
                out[peer] = buf
        for t in senders:
            t.join()
        return out

    def all_reduce(self, arr: np.ndarray) -> None:
        """Naive SUM all-reduce, in place: gather to rank 0, add in rank order, broadcast."""
        if self.world_size == 1:
            return
        if self.rank == 0:
            incoming = np.empty_like(arr)
            for peer in range(1, self.world_size):
                self.recv(incoming, peer)
                arr += incoming
        else:
            self.send(arr, 0)
        self.broadcast(arr, src=0)

    def _ring_all_reduce_py(self, arr: np.ndarray) -> None:
        """Ring all-reduce in Python: reduce-scatter (W-1 steps), then all-gather (W-1 steps)."""
        W, r = self.world_size, self.rank
        flat = arr.reshape(-1)
        n = flat.size
        bounds = [c * n // W for c in range(W + 1)]  # chunk c = flat[bounds[c]:bounds[c+1]]
        left, right = self._ring_neighbors()
        left_sock, right_sock = self.peers[left], self.peers[right]

        def step(send_chunk: int, recv_chunk: int, add: bool) -> None:
            outgoing = memoryview(flat[bounds[send_chunk] : bounds[send_chunk + 1]])
            sender = threading.Thread(target=send_msg, args=(right_sock, outgoing))
            sender.start()  # send to the right while we receive from the left
            incoming = np.frombuffer(recv_msg(left_sock), dtype=flat.dtype)
            sender.join()
            dst = flat[bounds[recv_chunk] : bounds[recv_chunk + 1]]
            if add:
                dst += incoming
            else:
                dst[...] = incoming

        for s in range(W - 1):  # reduce-scatter: afterwards we own chunk (r + 1) % W
            step((r - s) % W, (r - s - 1) % W, add=True)
        for s in range(W - 1):  # all-gather: pass the finished chunks around the ring
            step((r + 1 - s) % W, (r - s) % W, add=False)

    # ---------------------------------------------------------------- given

    def ring_all_reduce(self, arr: np.ndarray) -> None:
        """In-place SUM all-reduce with the ring algorithm. Afterwards all ranks hold the
        same bits."""
        if self.world_size == 1:
            return
        if self.backend == "c":
            left, right = self._ring_neighbors()
            _c.ring_allreduce_f32(
                self.peers[left].fileno(), self.peers[right].fileno(), arr, self.rank, self.world_size
            )
        else:
            self._ring_all_reduce_py(arr)

    def _ring_neighbors(self) -> tuple[int, int]:
        """(left, right) = (rank - 1, rank + 1), wrapping around."""
        return (self.rank - 1) % self.world_size, (self.rank + 1) % self.world_size

    def barrier(self) -> None:
        """Block until every rank has called barrier() (uses your store)."""
        if self.store is None:
            raise RuntimeError("barrier() needs a store: build the group with from_env()")
        self._barrier_gen += 1
        key = f"pg/barrier/{self._barrier_gen}"
        if self.store.add(key, 1) == self.world_size:
            self.store.set(f"{key}/done", b"1")
        self.store.wait([f"{key}/done"])

    def close(self) -> None:
        for s in self.peers.values():
            s.close()
        self.peers.clear()
        if self.store is not None:
            self.store.close()

    def __enter__(self) -> ProcessGroup:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
