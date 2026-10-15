"""M1 spec: message framing. Run: uv run pytest tests/test_m1_wire.py -x"""

import os
import socket
import threading

import numpy as np
import pytest

from seedling.foundation.wire import HEADER, recv_exact, recv_msg, send_msg


class TrickleSocket:
    """A fake socket that hands out at most `chunk` bytes per recv call, like a slow network.
    After its data runs out it behaves like a closed connection (returns b"" / 0)."""

    def __init__(self, data: bytes, chunk: int = 3):
        self.data = memoryview(data)
        self.pos = 0
        self.chunk = chunk
        self.calls = 0

    def _take(self, n: int) -> memoryview:
        self.calls += 1
        k = min(n, self.chunk, len(self.data) - self.pos)
        out = self.data[self.pos : self.pos + k]
        self.pos += k
        return out

    def recv(self, n: int, flags: int = 0) -> bytes:
        return bytes(self._take(n))

    def recv_into(self, buf, nbytes: int = 0, flags: int = 0) -> int:
        view = memoryview(buf).cast("B")
        part = self._take(nbytes or len(view))
        view[: len(part)] = part
        return len(part)


@pytest.fixture
def pair():
    a, b = socket.socketpair()
    a.settimeout(10)  # a broken implementation fails instead of hanging forever
    b.settimeout(10)
    yield a, b
    a.close()
    b.close()


def test_recv_exact_handles_partial_reads():
    data = bytes(range(256)) * 4
    sock = TrickleSocket(data, chunk=3)
    assert recv_exact(sock, len(data)) == data
    assert sock.calls >= len(data) // 3


def test_recv_exact_zero_bytes_does_not_read():
    sock = TrickleSocket(b"abc")
    assert recv_exact(sock, 0) == b""
    assert sock.calls == 0


def test_recv_exact_raises_when_peer_closes_early():
    with pytest.raises(ConnectionError):
        recv_exact(TrickleSocket(b"abc"), 10)


def test_recv_msg_over_trickling_socket():
    sock = TrickleSocket(HEADER.pack(5) + b"hello" + HEADER.pack(0), chunk=2)
    assert recv_msg(sock) == b"hello"
    assert recv_msg(sock) == b""


def test_wire_format_is_exact(pair):
    a, b = pair
    send_msg(a, b"abc")
    raw = b""
    while len(raw) < 11:
        raw += b.recv(11 - len(raw))
    assert raw == b"\x00\x00\x00\x00\x00\x00\x00\x03abc", "header must be 8-byte big-endian length"


def test_roundtrip(pair):
    a, b = pair
    send_msg(a, b"hello")
    assert recv_msg(b) == b"hello"


def test_empty_message(pair):
    a, b = pair
    send_msg(a, b"")
    assert recv_msg(b) == b""


def test_messages_keep_their_boundaries(pair):
    a, b = pair
    msgs = [b"one", b"", b"three" * 1000, bytes(1)]
    for m in msgs:
        send_msg(a, m)
    assert [recv_msg(b) for _ in msgs] == msgs


def test_large_message(pair):
    a, b = pair
    payload = os.urandom(8 * 1024 * 1024)  # far bigger than the socket buffer
    sender = threading.Thread(target=send_msg, args=(a, payload))
    sender.start()
    got = recv_msg(b)
    sender.join()
    assert got == payload


def test_numpy_payload_counts_bytes_not_elements(pair):
    a, b = pair
    arr = np.arange(10, dtype=np.float32)
    send_msg(a, memoryview(arr))
    assert recv_msg(b) == arr.tobytes(), "a float32 memoryview has 10 elements but 40 bytes"
