"""M9: do honest workers agree? Score the same pairs under different conditions.

Spot checks compare two workers' scores of one pair within a tolerance (rtol 1e-3 by
default). That only works if honest scores barely move with things a worker may change
without lying: which other pairs share its batch, its thread count, its float precision, its
device. This scores step 0's pairs under each condition and reports how far they move and
what share of pairs a 1e-3 audit would wrongly call a lie.

    uv run python -m bench.bench_agreement --bench mnist halfcheetah tinylm

Across devices: run it on each machine (it saves the raw scores), then compare the files:

    python -m bench.bench_agreement --bench sst2 countdown --device cuda --out gpu.json
    python -m bench.bench_agreement --bench sst2 countdown --device cpu --out cpu.json
    python -m bench.bench_agreement --compare cpu.json gpu.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"
RTOL = 1e-3
PAIRS = 16


LLM_BENCHES = ("sst2", "countdown")


def _workload(bench: str, **overrides):
    from seedling.runtime.tasks import create_workload

    args = {"sst2": {"dtype": "float32"}, "countdown": {"prompts_per_step": 4, "max_new_tokens": 96}}.get(bench, {})
    args.update(overrides)
    return create_workload(bench, **args)


def _sigma(bench: str) -> float:
    from bench.bench_scale import benchmarks

    return benchmarks()[bench]["config"].sigma


def _score(workload, seeds: list[int], sigma: float, one_by_one: bool = False) -> np.ndarray:
    state = workload.initial_state(seed=0, frac_bits=24)
    if one_by_one:
        parts = [workload.score(state, 0, [s], sigma) for s in seeds]
        return np.concatenate([np.r_[p[0], p[1]] for p in parts])
    f_plus, f_minus = workload.score(state, 0, seeds, sigma)
    return np.stack([f_plus, f_minus], axis=1).ravel()


def compare(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    rel = np.abs(reference - other) / np.maximum(np.maximum(np.abs(reference), np.abs(other)), 1e-12)
    pairs_flagged = (rel.reshape(-1, 2) > RTOL).any(axis=1)
    return {"max_rel_diff": float(rel.max()), "median_rel_diff": float(np.median(rel)), "flagged": float(pairs_flagged.mean())}


def study(bench: str, device: str | None = None) -> dict:
    """{"scores": reference scores, "conditions": {name: comparison with the reference}}."""
    from seedling.es.es import step_seeds

    seeds, sigma = step_seeds(0, 0, PAIRS), _sigma(bench)
    extra = {"device": device} if device and bench in LLM_BENCHES else {}
    threads = torch.get_num_threads()
    reference = _score(_workload(bench, **extra), seeds, sigma)
    out = {"repeat": compare(reference, _score(_workload(bench, **extra), seeds, sigma))}
    out["one_pair_per_batch"] = compare(reference, _score(_workload(bench, **extra), seeds, sigma, one_by_one=True))
    if not device or device == "cpu":
        torch.set_num_threads(1)
        out["threads_1_vs_default"] = compare(reference, _score(_workload(bench, **extra), seeds, sigma))
        torch.set_num_threads(threads)
    if bench in ("tinylm", *LLM_BENCHES):
        other = "float32" if bench == "countdown" else "bfloat16"
        out[f"{other}_vs_default"] = compare(reference, _score(_workload(bench, dtype=other, **extra), seeds, sigma))
    return {"device": device or "cpu", "torch": torch.__version__, "scores": reference.tolist(), "conditions": out}


def compare_files(a: str, b: str) -> None:
    left, right = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    print(f"| workload | {Path(a).name} vs {Path(b).name} | max rel. diff | median rel. diff | pairs a {RTOL:g} audit flags |")
    print("|---|---|---|---|---|")
    for bench in sorted(set(left) & set(right)):
        r = compare(np.array(left[bench]["scores"]), np.array(right[bench]["scores"]))
        devices = f"{left[bench]['device']} vs {right[bench]['device']}"
        print(f"| {bench} | {devices} | {r['max_rel_diff']:.1e} | {r['median_rel_diff']:.1e} | {100 * r['flagged']:.0f}% |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--bench", nargs="+", default=["mnist", "halfcheetah", "tinylm"])
    parser.add_argument("--device", default=None, help="for the LLM workloads: cpu or cuda")
    parser.add_argument("--out", default=str(RESULTS / "agreement.json"))
    parser.add_argument("--compare", nargs=2, metavar="FILE", help="compare the scores saved in two files")
    ns = parser.parse_args()
    if ns.compare:
        compare_files(*ns.compare)
        return
    results = {bench: study(bench, ns.device) for bench in ns.bench}
    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.out).write_text(json.dumps(results, indent=1))
    print(f"| workload | condition | max rel. diff | median rel. diff | pairs a {RTOL:g} audit flags |")
    print("|---|---|---|---|---|")
    for bench, result in results.items():
        for name, r in result["conditions"].items():
            print(f"| {bench} | {name} | {r['max_rel_diff']:.1e} | {r['median_rel_diff']:.1e} | {100 * r['flagged']:.0f}% |")


if __name__ == "__main__":
    main()
