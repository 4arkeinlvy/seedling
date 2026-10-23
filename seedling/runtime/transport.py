"""How messages travel. Workers use a Transport (client); the coordinator runs a Server.

Both directions only ever see `Message` objects; the bytes on the wire are protocol.encode().
Workers always open the connection, so a worker behind NAT (Kaggle, Colab, home Wi-Fi) can
reach a coordinator through a tunnel. Two implementations:

    tcp://host:port    length-prefixed frames from seedling.foundation.wire (fast, LAN)
    http://host:port   one POST /rpc per message (works through HTTP tunnels and proxies)

and their TLS versions tcps:// and https:// (M10). The HTTP server can also serve
GET /metrics in the Prometheus text format.
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import ssl
import threading
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from urllib.parse import urlparse

from seedling.foundation.wire import recv_msg, send_msg
from seedling.runtime import protocol as p

Handler = Callable[[p.Message], p.Message]
Metrics = Callable[[], str]
MAX_REQUEST = 64 * 2**20  # bytes a worker may send in one message
HANDSHAKE_SECONDS = 30.0


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
    def __init__(self, host: str, port: int, timeout: float = 60.0, tls: ssl.SSLContext | None = None):
        self.address = (host, port)
        self.timeout = timeout
        self.tls = tls
        self._sock: socket.socket | None = None

    def request(self, message: p.Message) -> p.Message:
        try:
            if self._sock is None:
                sock = socket.create_connection(self.address, timeout=self.timeout)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = self.tls.wrap_socket(sock, server_hostname=self.address[0]) if self.tls else sock
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
    def __init__(self, url: str, timeout: float = 60.0, tls: ssl.SSLContext | None = None):
        self.url = url.rstrip("/") + "/rpc"
        self.timeout = timeout
        self.tls = tls

    def request(self, message: p.Message) -> p.Message:
        req = urllib.request.Request(
            self.url, data=p.encode(message), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.tls) as response:
                return p.decode(response.read())
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f"coordinator at {self.url}: {e}") from e


def connect(url: str, tls: ssl.SSLContext | None = None) -> Transport:
    """tcp(s)://host:port or http(s)://host[:port]. TLS verifies with `tls`, else the system CAs."""
    parsed = urlparse(url)
    secure = parsed.scheme in ("tcps", "https")
    context = (tls or ssl.create_default_context()) if secure else None
    if parsed.scheme in ("tcp", "tcps"):
        return TcpTransport(parsed.hostname or "127.0.0.1", parsed.port or 7070, tls=context)
    if parsed.scheme in ("http", "https"):
        return HttpTransport(url, tls=context)
    raise ValueError(f"unsupported URL {url!r}: use tcp(s)://host:port or http(s)://host:port")


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


class _TlsAccept:
    """Wrap accepted sockets in TLS without handshaking in the accept loop: a slow or silent
    client must not stall everyone else. The handshake runs in the connection's thread."""

    tls: ssl.SSLContext | None = None

    def get_request(self):  # type: ignore[no-untyped-def]
        sock, address = super().get_request()  # type: ignore[misc]
        if self.tls is not None:
            sock = self.tls.wrap_socket(sock, server_side=True, do_handshake_on_connect=False)
        return sock, address


def _handshake(sock: socket.socket) -> bool:
    if not isinstance(sock, ssl.SSLSocket):
        return True
    try:
        sock.settimeout(HANDSHAKE_SECONDS)
        sock.do_handshake()
        sock.settimeout(None)
        return True
    except (ssl.SSLError, OSError):
        return False


class _ThreadingTCP(_TlsAccept, socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class TcpServer(Server):
    def __init__(self, handler: Handler, host: str = "0.0.0.0", port: int = 7070, tls: ssl.SSLContext | None = None):
        open_connections: set[socket.socket] = set()
        lock = threading.Lock()

        class Connection(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                with lock:
                    open_connections.add(self.request)
                try:
                    if not _handshake(self.request):
                        return
                    self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    while True:
                        try:
                            data = recv_msg(self.request, MAX_REQUEST)
                        except (ConnectionError, OSError):
                            return
                        send_msg(self.request, _respond(handler, data))
                finally:
                    with lock:
                        open_connections.discard(self.request)

        self._server = _ThreadingTCP((host, port), Connection)
        self._server.tls = tls
        self._connections, self._lock = open_connections, lock

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def serve_forever(self) -> None:
        self._server.serve_forever(poll_interval=0.1)

    def shutdown(self) -> None:
        """Stop accepting, and hang up on every connected worker (they would otherwise keep
        being served by their connection threads)."""
        self._server.shutdown()
        self._server.server_close()
        with self._lock:
            connections = list(self._connections)
        for sock in connections:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class _ThreadingHTTP(_TlsAccept, http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class HttpServer(Server):
    MAX_BODY = MAX_REQUEST

    def __init__(
        self,
        handler: Handler,
        host: str = "0.0.0.0",
        port: int = 7070,
        tls: ssl.SSLContext | None = None,
        metrics: Metrics | None = None,
    ):
        max_body = self.MAX_BODY

        class Rpc(http.server.BaseHTTPRequestHandler):
            def setup(self) -> None:
                if not _handshake(self.request):
                    raise ConnectionAbortedError("TLS handshake failed")
                super().setup()

            def handle(self) -> None:
                try:
                    super().handle()
                except ConnectionAbortedError:
                    pass

            def do_GET(self) -> None:
                if self.path != "/metrics" or metrics is None:
                    self.send_error(404)
                    return
                body = metrics().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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
        self._server.tls = tls

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def serve_forever(self) -> None:
        self._server.serve_forever(poll_interval=0.1)

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def make_server(
    kind: str,
    handler: Handler,
    host: str = "0.0.0.0",
    port: int = 7070,
    tls: ssl.SSLContext | None = None,
    metrics: Metrics | None = None,
) -> Server:
    """kind: tcp, tcps, http or https (the last two also serve GET /metrics if given)."""
    if kind not in ("tcp", "tcps", "http", "https"):
        raise ValueError(f"unknown server kind {kind!r}: use tcp(s) or http(s)")
    secure = kind.endswith("s")
    if secure and tls is None:
        raise ValueError(f"{kind}:// needs a TLS certificate and key")
    context = tls if secure else None
    if kind.startswith("tcp"):
        return TcpServer(handler, host, port, tls=context)
    return HttpServer(handler, host, port, tls=context, metrics=metrics)
