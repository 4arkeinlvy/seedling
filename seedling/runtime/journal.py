"""The coordinator's durable memory: the run header, an append-only step log, snapshots.

    run.json        RunConfig + initial digest (written once)
    log.jsonl       one StepRecord per line, flushed to disk before the step counts
    snapshot.npz    latest state + its step (replaced atomically)

Write-ahead rule: a step is committed only after its line is on disk, so a coordinator
restart can never forget a step that workers have already seen.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np

from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.runtime.config import RunConfig


def atomic_write(path: Path, data: bytes) -> None:
    """Write to a temp file, flush it to disk, then rename over the target in one step."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Journal:
    def __init__(self, directory: Path | str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.header_path = self.dir / "run.json"
        self.log_path = self.dir / "log.jsonl"
        self.snapshot_path = self.dir / "snapshot.npz"

    def open(self, config: RunConfig, initial_digest: str) -> None:
        """Create the run, or check that an existing run directory belongs to this run."""
        header = {"config": config.to_json(), "initial_digest": initial_digest}
        if self.header_path.exists():
            existing = json.loads(self.header_path.read_text())
            if existing != header:
                raise ValueError(f"{self.dir} holds a different run")
        else:
            atomic_write(self.header_path, json.dumps(header, indent=2).encode())

    def header(self) -> tuple[RunConfig, str]:
        data = json.loads(self.header_path.read_text())
        return RunConfig.from_json(data["config"]), data["initial_digest"]

    def records(self) -> list[StepRecord]:
        """All complete records. A torn last line (crash mid-write) is cut off."""
        if not self.log_path.exists():
            return []
        records: list[StepRecord] = []
        good_bytes = 0
        with open(self.log_path, "rb") as f:
            for line in f:
                try:
                    record = StepRecord.from_json(json.loads(line))
                except (ValueError, KeyError):
                    break
                if record.step != len(records) or not line.endswith(b"\n"):
                    break
                records.append(record)
                good_bytes += len(line)
        if good_bytes != self.log_path.stat().st_size:
            with open(self.log_path, "r+b") as f:
                f.truncate(good_bytes)
        return records

    def append(self, record: StepRecord) -> None:
        with open(self.log_path, "ab") as f:
            f.write(json.dumps(record.to_json()).encode() + b"\n")
            f.flush()
            os.fsync(f.fileno())

    def write_snapshot(self, state: FixedPointState, step: int) -> None:
        buffer = io.BytesIO()
        np.savez(buffer, values=state.values, frac_bits=state.frac_bits, step=step)
        atomic_write(self.snapshot_path, buffer.getvalue())

    def snapshot(self) -> tuple[int, FixedPointState] | None:
        if not self.snapshot_path.exists():
            return None
        with np.load(self.snapshot_path) as data:
            state = FixedPointState(np.ascontiguousarray(data["values"]), int(data["frac_bits"]))
            return int(data["step"]), state
