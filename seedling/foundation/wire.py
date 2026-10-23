"""M1: message framing over TCP.

TCP is a byte *stream*, not a sequence of messages, so every message is framed as:

    +---------------------------------------+------------------------+
    | length: 8 bytes, unsigned, big-endian | payload (length bytes) |
    +---------------------------------------+------------------------+

Later milestones speak the same format from C. Explained in docs/M1-explained.md.
"""

from __future__ import annotations

import socket
import struct

HEADER = struct.Struct("!Q")  # "!" = network byte order (big-endian), "Q" = uint64


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes; ConnectionError if the peer closes first."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError(f"peer closed the connection after {got} of {n} bytes")
        got += k
    return bytes(buf)


def send_msg(sock: socket.socket, payload) -> None:
    """Send one framed message: 8-byte length header, then the payload bytes."""
    view = memoryview(payload)
    sock.sendall(HEADER.pack(view.nbytes))  # nbytes, not len(): a float32 view counts floats
    if view.nbytes:
        sock.sendall(view)


def recv_msg(sock: socket.socket, max_bytes: int | None = None) -> bytes:
    """Receive one framed message and return its payload.

    A server must pass max_bytes: the length comes from the peer, and an 8-byte length of
    2**63 would otherwise make it try to allocate that much memory."""
    (n,) = HEADER.unpack(recv_exact(sock, HEADER.size))
    if max_bytes is not None and n > max_bytes:
        raise ConnectionError(f"peer sent a {n}-byte message; the limit is {max_bytes}")
    return recv_exact(sock, n)
