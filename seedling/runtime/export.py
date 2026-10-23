"""M10: turn a run directory into something usable without seedling.

    seedling export --run-dir runs/qwen --out qwen-countdown/

The journal is replayed into the final state (bit for bit, as `seedling replay` does). A
workload with an `export(state, out)` method writes its own format: the LLM workloads a
standard Hugging Face folder (load it with transformers or vLLM), TinyLM its weights. Any
other workload gets params.npy, its parameters as float32.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from seedling.runtime.journal import Journal
from seedling.runtime.tasks import create_workload
from seedling.runtime.workload import Workload, replay_workload


def export_run(run_dir: str, out: str, workload_factory: Callable[..., Workload] = create_workload) -> dict:
    journal = Journal(run_dir)
    config, initial_digest = journal.header()
    workload = workload_factory(config.task, **config.task_args)
    initial = workload.initial_state(config.seed, config.frac_bits)
    if initial.digest() != initial_digest:
        raise ValueError("initial state differs from the run's: different code or model version?")
    records = journal.records()
    state = replay_workload(workload, initial, records)
    exporter = getattr(workload, "export", None)
    if exporter is not None:
        exporter(state, out)
    else:
        Path(out).mkdir(parents=True, exist_ok=True)
        np.save(Path(out) / "params.npy", state.to_float().numpy().astype(np.float32))
    summary = {"task": config.task, "steps": len(records), "digest": state.digest(), "out": str(out)}
    (Path(out) / "seedling.json").write_text(json.dumps({**summary, "config": config.to_json()}, indent=1))
    return summary
