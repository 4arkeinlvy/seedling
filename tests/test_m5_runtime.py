"""M5 spec: the distributed runtime. Run: uv run pytest tests/test_m5_runtime.py -x"""

import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from seedling.exact.integer_es import replay
from seedling.foundation.launch import free_port
from seedling.runtime import (
    Coordinator,
    Journal,
    RunConfig,
    Worker,
    initial_state,
    make_server,
    serialized,
)
from seedling.runtime import protocol as p
from seedling.runtime.transport import HttpTransport, TcpTransport

REPO = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def quadratic(steps: int = 3, pop: int = 6, **overrides) -> RunConfig:
    args = dict(task="quadratic", steps=steps, pop=pop, sigma=0.1, lr=0.1, pairs_per_task=2, deadline=5.0)
    args.update(overrides)
    return RunConfig(**args)


def finish_step(coordinator: Coordinator) -> None:
    """Score every pair of the current step with a fake but valid submission."""
    digest = coordinator.state.digest()
    pairs = list(range(coordinator.config.pop))
    scores = [float(i) for i in pairs]
    reply = coordinator.handle(p.Submit("w", coordinator.step, pairs, scores, scores[::-1], digest))
    assert reply == p.Ack(True)


# ------------------------------------------------------------------ protocol


@pytest.mark.parametrize(
    "message",
    [
        p.Hello("w1"),
        p.Sync(3),
        p.GetWork("w1"),
        p.Submit("w1", 2, [0, 1], [1.5, -2.0], [0.0, 3.25], "00ff"),
        p.Welcome({"task": "cartpole", "steps": 3}, "abcd", "w1~00ff"),
        p.Records([{"step": 0, "seeds": [1], "coefs": [2]}]),
        p.Snapshot(5, 20, "AAAA", "beef"),
        p.Work(4, [0, 3]),
        p.Wait(0.05),
        p.Ack(False, "stale"),
        p.Finished(10, "cafe"),
        p.Error("nope"),
    ],
)
def test_every_message_survives_encode_decode(message):
    assert p.decode(p.encode(message)) == message


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        json.dumps({"v": 99, "type": "Hello", "body": {"worker_id": "w"}}).encode(),
        json.dumps({"v": p.PROTOCOL_VERSION, "type": "Pickle", "body": {}}).encode(),
        json.dumps({"v": p.PROTOCOL_VERSION, "type": "Hello", "body": {"worker_id": "w", "token": "", "extra": 1}}).encode(),
    ],
)
def test_bad_messages_are_rejected(raw):
    with pytest.raises(p.ProtocolError):
        p.decode(raw)


# ------------------------------------------------------------------ journal


def test_journal_keeps_records_and_cuts_a_torn_last_line(tmp_path):
    coordinator = Coordinator(quadratic(steps=5), Journal(tmp_path))
    finish_step(coordinator)
    finish_step(coordinator)
    with open(tmp_path / "log.jsonl", "ab") as f:
        f.write(b'{"step": 2, "seeds": [1')
    records = Journal(tmp_path).records()
    assert [r.step for r in records] == [0, 1]
    assert (tmp_path / "log.jsonl").read_bytes().endswith(b"\n"), "the torn line must be removed"


def test_journal_refuses_a_different_run(tmp_path):
    Coordinator(quadratic(), Journal(tmp_path))
    with pytest.raises(ValueError):
        Coordinator(quadratic(seed=1), Journal(tmp_path))


def test_snapshot_round_trip(tmp_path):
    journal = Journal(tmp_path)
    state = initial_state(quadratic())
    journal.write_snapshot(state, 7)
    step, loaded = journal.snapshot()
    assert step == 7 and loaded.digest() == state.digest()


# ------------------------------------------------------------------ coordinator (no sockets)


def test_welcome_carries_the_config_and_initial_digest(tmp_path):
    config = quadratic()
    reply = Coordinator(config, Journal(tmp_path)).handle(p.Hello("w"))
    assert reply == p.Welcome(config.to_json(), initial_state(config).digest(), "w")


def test_work_is_handed_out_once_then_again_after_the_deadline(tmp_path):
    clock = FakeClock()
    coordinator = Coordinator(quadratic(pop=4), Journal(tmp_path), clock=clock)
    first = coordinator.handle(p.GetWork("a"))
    second = coordinator.handle(p.GetWork("b"))
    assert first == p.Work(0, [0, 1]) and second == p.Work(0, [2, 3])
    assert isinstance(coordinator.handle(p.GetWork("c")), p.Wait)
    clock.now = 10.0
    assert coordinator.handle(p.GetWork("c")) == p.Work(0, [0, 1]), "expired pairs go out again"


def test_submissions_are_checked(tmp_path):
    coordinator = Coordinator(quadratic(pop=2), Journal(tmp_path))
    digest = coordinator.state.digest()
    assert coordinator.handle(p.Submit("w", 1, [0], [1.0], [0.0], digest)) == p.Ack(False, "stale")
    assert coordinator.handle(p.Submit("w", 0, [0], [1.0], [0.0], "bad")) == p.Ack(False, "resync")
    assert coordinator.handle(p.Submit("w", 0, [0], [1.0], [0.0], digest)) == p.Ack(True)
    assert coordinator.handle(p.Submit("w", 0, [0], [9.0], [9.0], digest)) == p.Ack(True)
    assert coordinator.results[0] == (1.0, 0.0), "the first answer for a pair wins"
    assert coordinator.step == 0
    coordinator.handle(p.Submit("w", 0, [1], [0.0], [1.0], digest))
    assert coordinator.step == 1, "all pairs in: the step commits"


def test_finished_run_says_so(tmp_path):
    coordinator = Coordinator(quadratic(steps=1), Journal(tmp_path))
    finish_step(coordinator)
    assert coordinator.handle(p.GetWork("w")) == p.Finished(1, coordinator.state.digest())


def test_restart_recovers_from_the_journal(tmp_path):
    first = Coordinator(quadratic(steps=5, snapshot_every=2), Journal(tmp_path))
    for _ in range(3):
        finish_step(first)
    again = Coordinator(quadratic(steps=5, snapshot_every=2), Journal(tmp_path))
    assert again.step == 3 and again.state.digest() == first.state.digest()


def test_far_behind_workers_get_a_snapshot(tmp_path):
    coordinator = Coordinator(quadratic(steps=5, snapshot_every=1), Journal(tmp_path))
    for _ in range(3):
        finish_step(coordinator)
    assert isinstance(coordinator.handle(p.Sync(0)), p.Snapshot)
    assert isinstance(coordinator.handle(p.Sync(2)), p.Records)


# ------------------------------------------------------------------ end to end (threads, real sockets)


def run_cluster(tmp_path, kind: str, workers: int, config: RunConfig) -> tuple[Coordinator, list[str]]:
    coordinator = Coordinator(config, Journal(tmp_path))
    server = make_server(kind, serialized(coordinator.handle), "127.0.0.1", 0).start()
    transport = TcpTransport if kind == "tcp" else lambda h, port: HttpTransport(f"http://{h}:{port}")
    digests: list[str] = []
    threads = [
        threading.Thread(target=lambda i=i: digests.append(Worker(transport("127.0.0.1", server.port), f"w{i}").run()))
        for i in range(workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    server.shutdown()
    return coordinator, digests


@pytest.mark.parametrize("kind", ["tcp", "http"])
def test_workers_end_with_the_coordinators_exact_state(tmp_path, kind):
    coordinator, digests = run_cluster(tmp_path, kind, workers=3, config=quadratic(steps=8, pop=8))
    assert coordinator.finished
    assert digests == [coordinator.state.digest()] * 3
    rebuilt = replay(initial_state(coordinator.config), Journal(tmp_path).records())
    assert rebuilt.digest() == coordinator.state.digest(), "the journal alone rebuilds the run"


def test_the_run_learns(tmp_path):
    coordinator, _ = run_cluster(tmp_path, "tcp", workers=2, config=quadratic(steps=20, pop=8))
    target = torch.linspace(-1, 1, 16)
    before = (coordinator.initial.to_float() - target).norm()
    after = (coordinator.state.to_float() - target).norm()
    assert after < 0.25 * before, f"distance to the optimum {before:.2f} -> {after:.2f}"


# ------------------------------------------------------------------ chaos: real processes, kill -9


def _seedling(*args: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "seedling", *args],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _cartpole_cli(run_dir: Path, port: int) -> list[str]:
    return [
        "coordinator", "--task", "cartpole", "--steps", "6", "--run-dir", str(run_dir),
        "--listen", f"tcp://127.0.0.1:{port}", "--pairs-per-task", "1", "--deadline", "2", "--linger", "3",
    ]  # fmt: skip


def _final_digest(output: str) -> str:
    return json.loads(output.strip().splitlines()[-1])["digest"]


def _run(run_dir: Path, chaos: bool) -> str:
    port = free_port()
    coordinator = _seedling(*_cartpole_cli(run_dir, port))
    url = f"tcp://127.0.0.1:{port}"
    worker_args = ["--connect", url, "--threads", "1", "--max-retry", "20"]
    workers = [_seedling("worker", *worker_args, "--id", f"w{i}") for i in range(2)]
    rng = random.Random(0)
    coordinator_restarted = False
    deadline = time.monotonic() + 240
    while coordinator.poll() is None and time.monotonic() < deadline:
        time.sleep(1.0)
        if not chaos:
            continue
        victim = rng.randrange(len(workers))
        workers[victim].send_signal(signal.SIGKILL)
        workers[victim].wait()
        workers[victim] = _seedling("worker", *worker_args, "--id", f"w{victim}r")
        if not coordinator_restarted and Journal(run_dir).records():
            coordinator.send_signal(signal.SIGKILL)
            coordinator.wait()
            coordinator = _seedling(*_cartpole_cli(run_dir, port))
            coordinator_restarted = True
    output, _ = coordinator.communicate(timeout=60)
    for w in workers:  # a worker restarted after the end can never hear "finished"
        try:
            w.wait(timeout=15)
        except subprocess.TimeoutExpired:
            w.kill()
            w.wait()
    if chaos:
        assert coordinator_restarted, "the chaos run never killed the coordinator"
    return _final_digest(output)


@pytest.mark.heavy
def test_chaos_run_ends_bit_identical_to_a_clean_run(tmp_path):
    clean = _run(tmp_path / "clean", chaos=False)
    chaotic = _run(tmp_path / "chaos", chaos=True)
    assert chaotic == clean, "kill -9 of workers and coordinator must not change a single bit"
