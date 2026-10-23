"""M5: the distributed runtime. A coordinator keeps the step log; workers anywhere keep exact
replicas and score seeds. Only seeds, scores and integer coefficients travel.

With an AuditPolicy (M8), pairs are spot-checked by other workers and liars are banned.

Explained in docs/M5-explained.md.
"""

from seedling.runtime.audit import AuditPolicy, influence_bound
from seedling.runtime.config import RunConfig
from seedling.runtime.coordinator import Coordinator, build_workload, initial_state
from seedling.runtime.journal import Journal
from seedling.runtime.tasks import create_workload, register
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
from seedling.runtime.workload import VectorWorkload, Workload, replay_workload

__all__ = [
    "AuditPolicy",
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
    "VectorWorkload",
    "Workload",
    "build_workload",
    "connect",
    "create_workload",
    "influence_bound",
    "initial_state",
    "make_server",
    "register",
    "replay_workload",
    "serialized",
]
