"""Messages between workers and the coordinator, encoded as versioned JSON.

Never pickle: a coordinator may be reachable from the internet, and unpickling untrusted
bytes runs arbitrary code. Each message is a frozen dataclass; `encode`/`decode` add the
type name and protocol version. `decode` also checks every field against its annotation
(ints are not bools, floats are finite), so a handler never sees a malformed value. Fields
with a default may be left out; unknown fields are refused.
"""

from __future__ import annotations

import json
import math
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any

PROTOCOL_VERSION = 3
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
    if not isinstance(body, dict):
        raise ProtocolError(f"{name}: body is not an object")
    kinds = {f.name: f.type for f in fields(cls)}  # type: ignore[arg-type]
    required = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}  # type: ignore[arg-type]
    if not required <= set(body) <= set(kinds):
        raise ProtocolError(f"{name}: fields {sorted(body)}; required {sorted(required)}, allowed {sorted(kinds)}")
    return cls(**{key: _checked(value, kinds[key], f"{name}.{key}") for key, value in body.items()})


def _checked(value: Any, kind: str, where: str) -> Any:
    """`value` as the annotated type `kind` (a string, see the dataclasses below), or raise."""
    if kind.startswith("list[") and kind.endswith("]"):
        if not isinstance(value, list):
            raise ProtocolError(f"{where}: expected a list")
        return [_checked(item, kind[5:-1], where) for item in value]
    if kind == "dict[str, Any]":
        ok = isinstance(value, dict)
    elif kind == "str":
        ok = isinstance(value, str)
    elif kind == "bool":
        ok = isinstance(value, bool)
    elif kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        value = float(value) if ok else value
    else:
        raise TypeError(f"{where}: no check for annotation {kind!r}")
    if not ok:
        raise ProtocolError(f"{where}: {value!r} is not a valid {kind}")
    return value


# ---------------------------------------------------------------- worker -> coordinator


@dataclass(frozen=True)
class Hello(Message):
    worker_id: str
    token: str = ""  # the join token's secret, when the coordinator requires one (M10)


@dataclass(frozen=True)
class Sync(Message):
    from_step: int


@dataclass(frozen=True)
class GetWork(Message):
    worker_id: str
    fingerprint: str = ""  # the worker's base model, for workloads that have one (M10)


@dataclass(frozen=True)
class Submit(Message):
    worker_id: str
    step: int
    pairs: list[int]
    f_plus: list[float]
    f_minus: list[float]
    digest: str
    evidence: list[str] = field(default_factory=list)  # per pair, from workloads that send it (M10)


# ---------------------------------------------------------------- coordinator -> worker


@dataclass(frozen=True)
class Welcome(Message):
    config: dict[str, Any]
    initial_digest: str
    worker_id: str  # the id to use from now on; with audits on, issued by the coordinator


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
    claims: list[str] = field(default_factory=list)  # per pair: another worker's evidence to check, or ""


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
