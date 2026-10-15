"""M5: the distributed runtime. A coordinator keeps the step log; workers anywhere keep exact
replicas and score seeds. Only seeds, scores and integer coefficients travel.

Explained in docs/M5-explained.md.
"""

from seedling.runtime.config import RunConfig
from seedling.runtime.coordinator import Coordinator, initial_state
from seedling.runtime.journal import Journal
from seedling.runtime.tasks import create_task, register
from seedling.runtime.transport import (
    HttpServer,
    HttpTransport,
    Server,
    TcpServer,
    TcpTransport,
    Transport,
    connect,
    make_server,
    serialized,
)
from seedling.runtime.worker import ReplicaMismatch, Worker

__all__ = [
    "Coordinator",
    "HttpServer",
    "HttpTransport",
    "Journal",
    "ReplicaMismatch",
    "RunConfig",
    "Server",
    "TcpServer",
    "TcpTransport",
    "Transport",
    "Worker",
    "connect",
    "create_task",
    "initial_state",
    "make_server",
    "register",
    "serialized",
]
