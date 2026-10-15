"""M1 spec: the key-value store. Run: uv run pytest tests/test_m1_store.py -x"""

import threading
import time

import pytest

from seedling.foundation.launch import free_port, run_ranks
from seedling.foundation.store import TCPStore
from tests import rank_programs as rp

TIMEOUT = 2.0


@pytest.fixture
def stores():
    """A server-side store plus a second client, both in this process."""
    port = free_port()
    server = TCPStore("127.0.0.1", port, is_server=True, world_size=2, timeout=TIMEOUT)
    client = TCPStore("127.0.0.1", port, is_server=False, timeout=TIMEOUT)
    yield server, client
    client.close()
    server.close()


def test_set_then_get(stores):
    server, client = stores
    server.set("a", b"1")
    assert client.get("a") == b"1"
    client.set("b", b"\x00\xff" * 1000)
    assert server.get("b") == b"\x00\xff" * 1000


def test_empty_value_and_unicode_key(stores):
    server, client = stores
    client.set("ключ/ü", b"")
    assert server.get("ключ/ü") == b""


def test_overwrite(stores):
    server, client = stores
    client.set("k", b"old")
    client.set("k", b"new")
    assert server.get("k") == b"new"


def test_large_value(stores):
    server, client = stores
    value = bytes(range(256)) * 20_000  # ~5 MB
    client.set("big", value)
    assert server.get("big") == value


def test_get_blocks_until_the_key_is_set(stores):
    server, client = stores
    threading.Timer(0.3, server.set, args=("late", b"x")).start()
    t0 = time.monotonic()
    assert client.get("late") == b"x"
    assert time.monotonic() - t0 >= 0.25, "get() returned before the key was set"


def test_get_times_out(stores):
    _, client = stores
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        client.get("never")
    elapsed = time.monotonic() - t0
    assert TIMEOUT * 0.75 <= elapsed < TIMEOUT + 3, f"timed out after {elapsed:.2f}s"


def test_add_counter(stores):
    server, client = stores
    assert client.add("c", 1) == 1
    assert server.add("c", 5) == 6
    assert client.add("c", -2) == 4
    assert client.get("c") == b"4", "counters are stored as ASCII decimal bytes"


def test_wait_for_several_keys(stores):
    server, client = stores
    threading.Timer(0.1, server.set, args=("k1", b"1")).start()
    threading.Timer(0.2, server.set, args=("k2", b"2")).start()
    client.wait(["k1", "k2"])
    with pytest.raises(TimeoutError):
        client.wait(["k1", "missing"])


def test_add_is_atomic_across_processes():
    results = run_ranks(rp.store_atomic_add, 4, args=(200,), timeout=60)
    values = sorted(v for per_rank in results for v in per_rank)
    assert values == list(range(1, 801)), "every add() must return a different new value"


def test_rendezvous_across_processes():
    results = run_ranks(rp.store_rendezvous, 3, timeout=60)
    assert results == [[b"0", b"1", b"2"]] * 3
