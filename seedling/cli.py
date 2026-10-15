"""The `seedling` command.

    seedling coordinator --task cartpole --steps 30 --run-dir runs/cartpole \
        --listen tcp://0.0.0.0:7070 --listen http://0.0.0.0:7080
    seedling worker --connect tcp://127.0.0.1:7070      (or http://... through a tunnel)
    seedling replay --run-dir runs/cartpole
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from typing import Any
from urllib.parse import urlparse

from seedling.runtime.config import RunConfig
from seedling.runtime.coordinator import Coordinator, initial_state
from seedling.runtime.journal import Journal
from seedling.runtime.tasks import create_task
from seedling.runtime.transport import connect, make_server, serialized
from seedling.runtime.worker import Worker


def _parse_value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _task_args(pairs: list[str]) -> dict[str, Any]:
    args = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--task-arg expects key=value, got {pair!r}")
        args[key] = _parse_value(value)
    return args


def _config_from(ns: argparse.Namespace) -> RunConfig:
    journal = Journal(ns.run_dir)
    if journal.header_path.exists():
        config, _ = journal.header()
        print(f"resuming the run in {ns.run_dir} (its saved config wins over flags)", flush=True)
        return config
    return RunConfig(
        task=ns.task,
        steps=ns.steps,
        task_args=_task_args(ns.task_arg),
        seed=ns.seed,
        pop=ns.pop,
        sigma=ns.sigma,
        lr=ns.lr,
        frac_bits=ns.frac_bits,
        pairs_per_task=ns.pairs_per_task,
        deadline=ns.deadline,
        snapshot_every=ns.snapshot_every,
    )


def run_coordinator(ns: argparse.Namespace) -> int:
    config = _config_from(ns)
    coordinator = Coordinator(config, Journal(ns.run_dir))
    handler = serialized(coordinator.handle)
    servers = []
    for url in ns.listen or ["tcp://0.0.0.0:7070"]:
        parsed = urlparse(url)
        servers.append(make_server(parsed.scheme, handler, parsed.hostname or "0.0.0.0", parsed.port or 0).start())
        print(f"listening on {parsed.scheme}://{parsed.hostname}:{servers[-1].port}", flush=True)
    print(f"coordinator: {config.task}, step {coordinator.step}/{config.steps}", flush=True)
    last = coordinator.step
    while not coordinator.finished:
        time.sleep(0.2)
        if coordinator.step != last:
            last = coordinator.step
            print(f"step {last}/{config.steps}  digest {coordinator.state.digest()}", flush=True)
    time.sleep(ns.linger)
    for server in servers:
        server.shutdown()
    task = create_task(config.task, **config.task_args)
    metrics = task.evaluate(coordinator.state.to_float(), config.steps)
    print(json.dumps({"digest": coordinator.state.digest(), **metrics}), flush=True)
    return 0


def run_worker(ns: argparse.Namespace) -> int:
    if ns.threads:
        import torch

        torch.set_num_threads(ns.threads)
    worker = Worker(connect(ns.connect), ns.id, max_retry_seconds=ns.max_retry)
    digest = worker.run()
    print(json.dumps({"worker": ns.id, "digest": digest, **vars(worker.stats)}), flush=True)
    return 0


def run_replay(ns: argparse.Namespace) -> int:
    from seedling.exact.integer_es import replay

    journal = Journal(ns.run_dir)
    config, initial_digest = journal.header()
    initial = initial_state(config)
    if initial.digest() != initial_digest:
        raise SystemExit("initial state differs from the run's: different code version?")
    records = journal.records()
    state = replay(initial, records)
    metrics = create_task(config.task, **config.task_args).evaluate(state.to_float(), len(records))
    print(json.dumps({"steps": len(records), "digest": state.digest(), **metrics}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seedling", description="Train with seeds and scores.")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("coordinator", help="run (or resume) a training run and serve workers")
    c.add_argument("--task", default="cartpole")
    c.add_argument("--task-arg", action="append", default=[], metavar="KEY=VALUE")
    c.add_argument("--steps", type=int, default=30)
    c.add_argument("--run-dir", required=True)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--pop", type=int, default=32)
    c.add_argument("--sigma", type=float, default=0.2)
    c.add_argument("--lr", type=float, default=0.3)
    c.add_argument("--frac-bits", type=int, default=20)
    c.add_argument("--pairs-per-task", type=int, default=4)
    c.add_argument("--deadline", type=float, default=30.0)
    c.add_argument("--snapshot-every", type=int, default=50)
    c.add_argument("--listen", action="append", metavar="URL",
                   help="tcp://host:port or http://host:port; repeat to serve both (default tcp://0.0.0.0:7070)")
    c.add_argument("--linger", type=float, default=2.0, help="seconds to keep serving after the end")
    c.set_defaults(run=run_coordinator)

    w = sub.add_parser("worker", help="join a run and score seeds")
    w.add_argument("--connect", required=True, help="tcp://host:port or http://host:port")
    w.add_argument("--id", default=f"{socket.gethostname()}-{os.getpid()}")
    w.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    w.add_argument("--max-retry", type=float, default=60.0)
    w.set_defaults(run=run_worker)

    r = sub.add_parser("replay", help="rebuild the final state from a run directory")
    r.add_argument("--run-dir", required=True)
    r.set_defaults(run=run_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    return ns.run(ns)


if __name__ == "__main__":
    sys.exit(main())
