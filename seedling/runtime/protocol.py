"""Messages between workers and the coordinator, encoded as versioned JSON.

Never pickle: a coordinator may be reachable from the internet, and unpickling untrusted
bytes runs arbitrary code. Each message is a frozen dataclass; `encode`/`decode` add the
type name and protocol version.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any

PROTOCOL_VERSION = 1
_MESSAGES: dict[str, type[Message]] = {}


class ProtocolError(ValueError):
    pass


class Message:
    """Base class: subclasses register themselves by name."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _MESSAGES[cls.__name__] = cls


def encode(message: Message) -> bytes:
    body = asdict(message)  # type: ignore[call-overload]
    return json.dumps({"v": PROTOCOL_VERSION, "type": type(message).__name__, "body": body}).encode()


def decode(data: bytes) -> Message:
    try:
        envelope = json.loads(data)
        version, name, body = envelope["v"], envelope["type"], envelope["body"]
    except (ValueError, KeyError, TypeError) as e:
        raise ProtocolError(f"malformed message: {e}") from None
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {version}, expected {PROTOCOL_VERSION}")
    cls = _MESSAGES.get(name)
    if cls is None:
        raise ProtocolError(f"unknown message type {name!r}")
    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    if set(body) != allowed:
        raise ProtocolError(f"{name}: fields {sorted(body)} != {sorted(allowed)}")
    return cls(**body)


# ---------------------------------------------------------------- worker -> coordinator


@dataclass(frozen=True)
class Hello(Message):
    worker_id: str


@dataclass(frozen=True)
class Sync(Message):
    from_step: int


@dataclass(frozen=True)
class GetWork(Message):
    worker_id: str


@dataclass(frozen=True)
class Submit(Message):
    worker_id: str
    step: int
    pairs: list[int]
    f_plus: list[float]
    f_minus: list[float]
    digest: str


# ---------------------------------------------------------------- coordinator -> worker


@dataclass(frozen=True)
class Welcome(Message):
    config: dict[str, Any]
    initial_digest: str


@dataclass(frozen=True)
class Records(Message):
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class Snapshot(Message):
    step: int
    frac_bits: int
    values_b64: str
    digest: str


@dataclass(frozen=True)
class Work(Message):
    step: int
    pairs: list[int]


@dataclass(frozen=True)
class Wait(Message):
    seconds: float


@dataclass(frozen=True)
class Ack(Message):
    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class Finished(Message):
    steps: int
    digest: str


@dataclass(frozen=True)
class Error(Message):
    reason: str
