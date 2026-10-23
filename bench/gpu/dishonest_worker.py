"""A worker that lies, to test a real coordinator's defenses. Never point it at a real run.

    python -m bench.gpu.dishonest_worker --connect tcp://127.0.0.1:7070 --behaviour flip

    flip     does the work, then swaps F+ and F- (pushes the model the wrong way)
    lazy     skips the work and sends zeros (a free rider)
    inflate  skips the work and claims F+ = 1e9, F- = -1e9
"""

from __future__ import annotations

import argparse
import json
import os
import socket

from seedling.runtime import Worker
from seedling.runtime import protocol as p
from seedling.runtime.transport import connect

BEHAVIOURS = ("flip", "lazy", "inflate")


class LyingWorker(Worker):
    def __init__(self, *args, behaviour: str = "flip", **kwargs):
        if behaviour not in BEHAVIOURS:
            raise ValueError(f"behaviour is one of {BEHAVIOURS}")
        super().__init__(*args, **kwargs)
        self.behaviour = behaviour

    def _score(self, work: p.Work) -> p.Submit:
        n = len(work.pairs)
        if self.behaviour == "flip":
            honest = super()._score(work)
            f_plus, f_minus = honest.f_minus, honest.f_plus
        elif self.behaviour == "lazy":
            f_plus, f_minus = [0.0] * n, [0.0] * n
        else:
            f_plus, f_minus = [1e9] * n, [-1e9] * n
        return p.Submit(self.worker_id, work.step, list(work.pairs), f_plus, f_minus, self.state.digest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--connect", required=True)
    parser.add_argument("--behaviour", choices=BEHAVIOURS, default="flip")
    parser.add_argument("--id", default=f"liar-{socket.gethostname()}-{os.getpid()}")
    ns = parser.parse_args()
    worker = LyingWorker(connect(ns.connect), ns.id, behaviour=ns.behaviour)
    try:
        digest = worker.run()
        print(json.dumps({"worker": ns.id, "caught": False, "digest": digest, **vars(worker.stats)}))
    except RuntimeError as e:
        print(json.dumps({"worker": ns.id, "caught": "banned" in str(e), "reason": str(e), **vars(worker.stats)}))


if __name__ == "__main__":
    main()
