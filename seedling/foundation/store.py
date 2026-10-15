"""M1: a tiny TCP key-value store for rendezvous (like torch.distributed.TCPStore).

Rank 0 runs the server in a background thread. Every rank, rank 0 included, connects as
a client. Ranks use it to find each other: rank r calls store.set(f"addr/{r}", b"host:port"),
and store.get(f"addr/{peer}") blocks until that peer has published its address.

    store = TCPStore("127.0.0.1", 29500, is_server=(rank == 0), world_size=4)
    store.set("hello", b"world")
    store.get("hello")          # -> b"world"; blocks until the key exists, else TimeoutError
    store.add("counter", 1)     # -> the new value (int); atomic across all clients
    store.wait(["a", "b"])      # blocks until every key exists, else TimeoutError
    store.close()

Protocol: one request message -> one reply message, framed by wire.send_msg / recv_msg.
    request = op (1 byte) | key length (4 bytes, "!I") | key (utf-8) | value bytes
    reply   = status (1 byte: 0 = ok, 1 = timeout) | value bytes
No pickle on network data: unpickling bytes from a socket can run any code.
Explained in docs/M1-explained.md.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

from seedling.foundation.wire import recv_msg, send_msg

OP_SET, OP_GET, OP_ADD = b"S", b"G", b"A"
OK, TIMEOUT = b"\x00", b"\x01"
_KEY_LEN = struct.Struct("!I")


def encode_request(op: bytes, key: str, value: bytes = b"") -> bytes:
    k = key.encode("utf-8")
    return op + _KEY_LEN.pack(len(k)) + k + value


def decode_request(request: bytes) -> tuple[bytes, str, bytes]:
    (key_len,) = _KEY_LEN.unpack_from(request, 1)
    start = 1 + _KEY_LEN.size
    return request[:1], request[start : start + key_len].decode("utf-8"), request[start + key_len :]


class _Server:
    """Lives inside rank 0: a listening socket plus one thread per connected client."""

    def __init__(self, host: str, port: int, timeout: float):
        self.timeout = timeout  # how long a blocking get may wait for its key
        self.data: dict[str, bytes] = {}
        self.cond = threading.Condition()  # guards self.data; notify_all() after changes

        self._clients = 0  # connected right now
        self._seen = 0  # ever connected
        self._clients_cond = threading.Condition()

        self.sock = socket.create_server((host, port))
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return  # listening socket closed
            with self._clients_cond:
                self._clients += 1
                self._seen += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    try:
                        request = recv_msg(conn)
                    except (ConnectionError, OSError):
                        return  # client went away
                    send_msg(conn, self._handle(request))
        finally:
            with self._clients_cond:
                self._clients -= 1
                self._clients_cond.notify_all()

    def _handle(self, request: bytes) -> bytes:
        """Apply one request to self.data under the lock and return the encoded reply."""
        op, key, value = decode_request(request)
        with self.cond:
            if op == OP_SET:
                self.data[key] = value
                self.cond.notify_all()
                return OK
            if op == OP_GET:
                if not self.cond.wait_for(lambda: key in self.data, timeout=self.timeout):
                    return TIMEOUT
                return OK + self.data[key]
            if op == OP_ADD:
                new = int(self.data.get(key, b"0")) + int(value)
                self.data[key] = str(new).encode()
                self.cond.notify_all()
                return OK + self.data[key]
        raise ValueError(f"unknown store op {op!r}")

    def close(self, expect_clients: int | None, wait: float) -> None:
        """Wait (at most `wait` seconds) until every client has come and gone, then stop
        listening. Rank 0 must not vanish while other ranks still use the store."""
        with self._clients_cond:
            self._clients_cond.wait_for(
                lambda: self._clients == 0 and self._seen >= (expect_clients or 0),
                timeout=wait,
            )
        try:
            self.sock.shutdown(socket.SHUT_RDWR)  # wakes up the blocked accept()
        except OSError:
            pass
        self.sock.close()


class TCPStore:
    def __init__(
        self,
        host: str,
        port: int,
        is_server: bool,
        world_size: int | None = None,
        timeout: float = 30.0,
    ):
        """is_server: True in exactly one process (rank 0).
        world_size: number of clients that will connect (server only; used by close()).
        timeout: how long get/wait may block, in seconds."""
        self.timeout = timeout
        self._world_size = world_size
        self._server = _Server(host, port, timeout) if is_server else None
        self._sock = self._connect(host, port, timeout)
        self._lock = threading.Lock()  # one request at a time on this connection

    @staticmethod
    def _connect(host: str, port: int, timeout: float) -> socket.socket:
        """Retry until the server is up: rank 0 may start later than us."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                sock = socket.create_connection((host, port), timeout=timeout)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no store server at {host}:{port}") from None
                time.sleep(0.05)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # The server enforces `timeout` for blocking gets; this is only a safety net in
        # case the server itself dies.
        sock.settimeout(timeout + 5.0)
        return sock

    def _request(self, payload: bytes) -> bytes:
        """Send one request, return the raw reply."""
        with self._lock:
            send_msg(self._sock, payload)
            return recv_msg(self._sock)

    def _value(self, reply: bytes, key: str) -> bytes:
        if reply[:1] == TIMEOUT:
            raise TimeoutError(f"store key {key!r} not set within {self.timeout}s")
        return reply[1:]

    def set(self, key: str, value: bytes) -> None:
        """Store value under key (overwrite if it exists)."""
        self._value(self._request(encode_request(OP_SET, key, bytes(value))), key)

    def get(self, key: str) -> bytes:
        """Value of key; blocks until the key exists, TimeoutError after `timeout`."""
        return self._value(self._request(encode_request(OP_GET, key)), key)

    def add(self, key: str, amount: int) -> int:
        """Atomically add `amount` to the counter at key (missing = 0); return the new value."""
        reply = self._request(encode_request(OP_ADD, key, str(int(amount)).encode()))
        return int(self._value(reply, key))

    def wait(self, keys: list[str]) -> None:
        """Block until every key exists, else TimeoutError."""
        for key in keys:
            self.get(key)

    def close(self) -> None:
        self._sock.close()
        if self._server is not None:
            self._server.close(self._world_size, wait=self.timeout)

    def __enter__(self) -> TCPStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
