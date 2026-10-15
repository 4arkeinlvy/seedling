"""M1 spec: collectives. Run: uv run pytest tests/test_m1_collectives.py -x

Two kinds of tests:
  - real processes over TCP (run_ranks): the real thing, like torch.distributed;
  - threads over Unix socketpairs (run_threads) with 16 MB chunks: catches deadlocks
    reliably (see tests/thread_ranks.py).
A deadlock shows up as TimeoutError; a crash in C as RankError "died without a Python error".
"""

import numpy as np
import pytest

from seedling import _c
from seedling.foundation.launch import free_port, run_ranks
from tests import rank_programs as rp
from tests.thread_ranks import run_threads

WORLDS = [1, 2, 3, 4]
# Distinct ids, so `-k backend_c` selects only the C variants.
BACKENDS = pytest.mark.parametrize("backend", ["python", "c"], ids=["backend_python", "backend_c"])


def check_all_reduce(reports):
    for rank, report in enumerate(reports):
        for n, err, _, close in report:
            assert err is None, f"rank {rank}, n={n}: {err}"
            assert close in (None, True), f"rank {rank}, n={n}: random floats are not the sum"
    for i, (n, *_) in enumerate(reports[0]):
        assert len({report[i][2] for report in reports}) == 1, (
            f"n={n}: ranks ended with different bits. Each chunk must be summed by ONE rank "
            "and then copied to the others, never re-added."
        )


# ------------------------------------------------------------------ processes over TCP


@pytest.mark.parametrize("world", WORLDS)
def test_send_recv(world):
    if world == 1:
        pytest.skip("needs two ranks")
    for rank, err in enumerate(run_ranks(rp.send_recv_report, world, timeout=60)):
        assert err is None, f"rank {rank}: {err}"


@pytest.mark.parametrize("src", ["first", "last"])
@pytest.mark.parametrize("world", WORLDS)
def test_broadcast(world, src):
    src_rank = 0 if src == "first" else world - 1
    reports = run_ranks(rp.broadcast_report, world, args=([0, 1, 1000, 1_000_003], src_rank), timeout=60)
    for rank, report in enumerate(reports):
        for n, err in report:
            assert err is None, f"rank {rank}, n={n}: {err}"


@pytest.mark.parametrize("world", WORLDS)
def test_all_gather(world):
    reports = run_ranks(rp.all_gather_report, world, args=([0, 1, 1000, 1_000_003],), timeout=60)
    for rank, report in enumerate(reports):
        for n, err in report:
            assert err is None, f"rank {rank}, n={n}: {err}"


@pytest.mark.parametrize("world", WORLDS)
def test_naive_all_reduce(world):
    reports = run_ranks(
        rp.all_reduce_report, world, args=("python", "all_reduce", [0, 1, 7, 1000, 100_003]), timeout=60
    )
    check_all_reduce(reports)


@BACKENDS
@pytest.mark.parametrize("world", WORLDS)
def test_ring_all_reduce(world, backend):
    reports = run_ranks(rp.all_reduce_report, world, args=(backend, "ring_all_reduce", rp.SIZES), timeout=60)
    check_all_reduce(reports)


def test_barrier_waits_for_everyone():
    times = run_ranks(rp.barrier_times, 3, timeout=60)
    last_arrival = max(arrive for arrive, _ in times)
    first_leave = min(leave for _, leave in times)
    assert first_leave >= last_arrival, "a rank left the barrier before the last one arrived"


# ------------------------------------------------------------------ deadlock checks (threads)


def _ring_big(pg, want):
    x = rp.make_input(pg.rank, rp.BIG, seed=1)
    pg.ring_all_reduce(x)
    return rp.mismatch(x, want)


def _all_gather_big(pg):
    got = pg.all_gather(rp.make_input(pg.rank, rp.BIG, seed=2))
    return [rp.mismatch(np.asarray(g), rp.make_input(r, rp.BIG, seed=2)) for r, g in enumerate(got)]


@BACKENDS
@pytest.mark.parametrize("world", [2, 3, 4])
def test_ring_all_reduce_big_chunks_do_not_deadlock(world, backend):
    want = rp.expected_sum(world, rp.BIG, seed=1)
    for rank, err in enumerate(run_threads(world, _ring_big, want, backend=backend)):
        assert err is None, f"rank {rank}: {err}"


@pytest.mark.parametrize("world", [2, 3])
def test_all_gather_big_arrays_do_not_deadlock(world):
    for rank, errs in enumerate(run_threads(world, _all_gather_big)):
        assert not any(errs), f"rank {rank}: {errs}"


# ------------------------------------------------------------------ C kernel


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 1000, (1 << 20) + 3])
def test_c_sum_matches_numpy_bit_for_bit(n):
    rng = np.random.default_rng(n)
    dst = rng.standard_normal(n, dtype=np.float32)
    src = rng.standard_normal(n, dtype=np.float32)
    want = dst + src  # IEEE-754 addition is exactly rounded: C must give the same bits
    _c.sum_f32(dst, src)
    assert np.array_equal(dst, want)


# ------------------------------------------------------------------ vs torch.distributed


@pytest.mark.heavy
def test_matches_torch_distributed_gloo():
    results = run_ranks(rp.compare_with_gloo, 2, args=(free_port(),), timeout=180)
    for rank, checks in enumerate(results):
        assert all(checks.values()), f"rank {rank}: differs from torch.distributed: {checks}"
