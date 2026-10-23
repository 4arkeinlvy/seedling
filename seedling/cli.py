"""The `seedling` command.

    seedling coordinator --task cartpole --steps 30 --run-dir runs/cartpole \
        --listen tcp://0.0.0.0:7070 --listen http://0.0.0.0:7080 \
        --audit-rate 0.1 --probation 10          (spot checks against dishonest workers)
    seedling worker --connect tcp://127.0.0.1:7070      (or http://... through a tunnel)
    seedling replay --run-dir runs/cartpole
    seedling train --task mnist --steps 300 --pop 50 --sigma 0.05 --lr 0.01   (one process)
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

from seedling.runtime.audit import AuditPolicy
from seedling.runtime.config import RunConfig
from seedling.runtime.coordinator import Coordinator, build_workload
from seedling.runtime.journal import Journal
from seedling.runtime.local import train_local
from seedling.runtime.workload import replay_workload
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
    return _new_config(ns)


def _new_config(ns: argparse.Namespace) -> RunConfig:
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
    audit = AuditPolicy(
        rate=ns.audit_rate, probation=ns.probation, rtol=ns.audit_rtol, aggregate=ns.aggregate, replicas=ns.replicas
    )
    coordinator = Coordinator(config, Journal(ns.run_dir), audit=audit)
    handler = serialized(coordinator.handle)
    servers = []
    for url in ns.listen or ["tcp://0.0.0.0:7070"]:
        parsed = urlparse(url)
        servers.append(make_server(parsed.scheme, handler, parsed.hostname or "0.0.0.0", parsed.port or 0).start())
        print(f"listening on {parsed.scheme}://{parsed.hostname}:{servers[-1].port}", flush=True)
    print(f"coordinator: {config.task}, step {coordinator.step}/{config.steps}", flush=True)
    last, banned = coordinator.step, set(coordinator.ledger.banned)
    while not coordinator.finished:
        time.sleep(0.2)
        for worker in set(coordinator.ledger.banned) - banned:
            print(f"banned {worker}: {coordinator.ledger.banned[worker]}", flush=True)
            banned.add(worker)
        if coordinator.step != last:
            last = coordinator.step
            print(f"step {last}/{config.steps}  digest {coordinator.state.digest()}", flush=True)
    time.sleep(ns.linger)
    for server in servers:
        server.shutdown()
    metrics = {} if ns.no_eval else coordinator.workload.evaluate(coordinator.state, config.steps)
    if audit.enabled:
        metrics["banned"] = len(coordinator.ledger.banned)
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


def run_train(ns: argparse.Namespace) -> int:
    config = _new_config(ns)
    workload = build_workload(config)
    run = train_local(workload, config, ns.eval_every, log=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"steps": len(run.records), "digest": run.state.digest()}), flush=True)
    return 0


def run_replay(ns: argparse.Namespace) -> int:
    journal = Journal(ns.run_dir)
    config, initial_digest = journal.header()
    workload = build_workload(config)
    initial = workload.initial_state(config.seed, config.frac_bits)
    if initial.digest() != initial_digest:
        raise SystemExit("initial state differs from the run's: different code version?")
    records = journal.records()
    state = replay_workload(workload, initial, records)
    metrics = {} if ns.no_eval else workload.evaluate(state, len(records))
    print(json.dumps({"steps": len(records), "digest": state.digest(), **metrics}))
    return 0


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", default="cartpole", help="quadratic | cartpole | mnist | countdown")
    parser.add_argument("--task-arg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop", type=int, default=32, help="seed pairs per step")
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--frac-bits", type=int, default=20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seedling", description="Train with seeds and scores.")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("coordinator", help="run (or resume) a training run and serve workers")
    _add_run_args(c)
    c.add_argument("--run-dir", required=True)
    c.add_argument("--pairs-per-task", type=int, default=4)
    c.add_argument("--deadline", type=float, default=30.0)
    c.add_argument("--snapshot-every", type=int, default=50)
    c.add_argument("--listen", action="append", metavar="URL",
                   help="tcp://host:port or http://host:port; repeat to serve both (default tcp://0.0.0.0:7070)")
    c.add_argument("--linger", type=float, default=2.0, help="seconds to keep serving after the end")
    c.add_argument("--no-eval", action="store_true", help="skip the final evaluation")
    c.add_argument("--audit-rate", type=float, default=0.0,
                   help="chance that a pair is re-scored by a second worker (0 = trust everyone)")
    c.add_argument("--probation", type=int, default=0,
                   help="audit every pair of a new worker until it has passed this many audits")
    c.add_argument("--audit-rtol", type=float, default=1e-3,
                   help="relative tolerance when two workers' scores are compared")
    c.add_argument("--aggregate", choices=("vote", "median"), default="vote",
                   help="median: score every pair --replicas times and take the median (no bans)")
    c.add_argument("--replicas", type=int, default=3, help="scorers per pair with --aggregate median")
    c.set_defaults(run=run_coordinator)

    t = sub.add_parser("train", help="train in this process (no network)")
    _add_run_args(t)
    t.add_argument("--eval-every", type=int, default=10)
    t.set_defaults(run=run_train, pairs_per_task=4, deadline=30.0, snapshot_every=50)

    w = sub.add_parser("worker", help="join a run and score seeds")
    w.add_argument("--connect", required=True, help="tcp://host:port or http://host:port")
    w.add_argument("--id", default=f"{socket.gethostname()}-{os.getpid()}")
    w.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    w.add_argument("--max-retry", type=float, default=60.0)
    w.set_defaults(run=run_worker)

    r = sub.add_parser("replay", help="rebuild the final state from a run directory")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--no-eval", action="store_true")
    r.set_defaults(run=run_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    return ns.run(ns)


if __name__ == "__main__":
    sys.exit(main())
