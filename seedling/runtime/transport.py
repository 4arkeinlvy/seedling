"""How messages travel. Workers use a Transport (client); the coordinator runs a Server.

Both directions only ever see `Message` objects; the bytes on the wire are protocol.encode().
Workers always open the connection, so a worker behind NAT (Kaggle, Colab, home Wi-Fi) can
reach a coordinator through a tunnel. Two implementations:

    tcp://host:port    length-prefixed frames from seedling.foundation.wire (fast, LAN)
    http://host:port   one POST /rpc per message (works through HTTP tunnels and proxies)
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import threading
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from urllib.parse import urlparse

from seedling.foundation.wire import recv_msg, send_msg
from seedling.runtime import protocol as p

Handler = Callable[[p.Message], p.Message]


def serialized(handler: Handler) -> Handler:
    """Run one request at a time: the coordinator is a plain object, not thread-safe."""
    lock = threading.Lock()

    def locked(message: p.Message) -> p.Message:
        with lock:
            return handler(message)

    return locked


def _respond(handler: Handler, data: bytes) -> bytes:
    try:
        return p.encode(handler(p.decode(data)))
    except p.ProtocolError as e:
        return p.encode(p.Error(str(e)))


# ---------------------------------------------------------------- client side


class Transport(ABC):
    @abstractmethod
    def request(self, message: p.Message) -> p.Message:
        """Send one message, wait for the reply. Raises ConnectionError if unreachable."""

    def close(self) -> None:
        pass


class TcpTransport(Transport):
    def __init__(self, host: str, port: int, timeout: float = 60.0):
        self.address = (host, port)
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def request(self, message: p.Message) -> p.Message:
        try:
            if self._sock is None:
                self._sock = socket.create_connection(self.address, timeout=self.timeout)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            send_msg(self._sock, p.encode(message))
            return p.decode(recv_msg(self._sock))
        except OSError as e:
            self.close()
            raise ConnectionError(f"coordinator at {self.address}: {e}") from e

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class HttpTransport(Transport):
    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url.rstrip("/") + "/rpc"
        self.timeout = timeout

    def request(self, message: p.Message) -> p.Message:
        req = urllib.request.Request(
            self.url, data=p.encode(message), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return p.decode(response.read())
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f"coordinator at {self.url}: {e}") from e


def connect(url: str) -> Transport:
    """tcp://host:port or http(s)://host[:port]."""
    parsed = urlparse(url)
    if parsed.scheme == "tcp":
        return TcpTransport(parsed.hostname or "127.0.0.1", parsed.port or 7070)
    if parsed.scheme in ("http", "https"):
        return HttpTransport(url)
    raise ValueError(f"unsupported URL {url!r}: use tcp://host:port or http://host:port")


# ---------------------------------------------------------------- server side


class Server(ABC):
    @property
    @abstractmethod
    def port(self) -> int: ...

    @abstractmethod
    def serve_forever(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    def start(self) -> Server:
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self


class _ThreadingTCP(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class TcpServer(Server):
    def __init__(self, handler: Handler, host: str = "0.0.0.0", port: int = 7070):
        class Connection(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    try:
                        data = recv_msg(self.request)
                    except (ConnectionError, OSError):
                        return
                    send_msg(self.request, _respond(handler, data))

        self._server = _ThreadingTCP((host, port), Connection)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def serve_forever(self) -> None:
        self._server.serve_forever(poll_interval=0.1)

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class _ThreadingHTTP(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class HttpServer(Server):
    MAX_BODY = 64 * 2**20

    def __init__(self, handler: Handler, host: str = "0.0.0.0", port: int = 7070):
        max_body = self.MAX_BODY

        class Rpc(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/rpc" or not 0 < length <= max_body:
                    self.send_error(400, "POST /rpc with a JSON body")
                    return
                body = _respond(handler, self.rfile.read(length))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = _ThreadingHTTP((host, port), Rpc)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def serve_forever(self) -> None:
        self._server.serve_forever(poll_interval=0.1)

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def make_server(kind: str, handler: Handler, host: str = "0.0.0.0", port: int = 7070) -> Server:
    servers: dict[str, type[Server]] = {"tcp": TcpServer, "http": HttpServer}
    if kind not in servers:
        raise ValueError(f"unknown server kind {kind!r}: use tcp or http")
    return servers[kind](handler, host, port)  # type: ignore[call-arg]
