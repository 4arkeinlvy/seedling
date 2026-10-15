"""M4 spec: the bit-exact core. Run: uv run pytest tests/test_m4_exact.py -x"""

import os

import numpy as np
import pytest
import torch
import xxhash

from seedling import _c
from seedling.exact import (
    FixedPointState,
    IntegerES,
    StepRecord,
    integer_coefficients,
    rademacher_noise,
    replay,
    train_integer_es,
)
from seedling.exact import reference as ref
from seedling.tasks.cartpole import CartPoleTask

# ------------------------------------------------------------------ C == oracle, bit for bit


@pytest.mark.parametrize(
    "counter, key, want",
    [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
         (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
    ],
)
def test_philox_known_answers(counter, key, want):
    assert tuple(ref.philox4x32(counter, key)) == want
    assert _c.philox4x32(counter, key) == want


def test_philox_c_matches_python_on_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(50):
        counter = tuple(int(v) for v in rng.integers(0, 2**32, 4))
        key = tuple(int(v) for v in rng.integers(0, 2**32, 2))
        assert _c.philox4x32(counter, key) == tuple(ref.philox4x32(counter, key))


@pytest.mark.parametrize("seed, stream, offset", [(0, 0, 0), (2**63 - 1, 7, 0), (123, 1, 1000), (5, 0, 2**40)])
def test_sign_stream_c_matches_python(seed, stream, offset):
    assert np.array_equal(_c.rademacher_i8(seed, 300, stream, offset), ref.rademacher(seed, 300, stream, offset))


def test_signs_are_balanced_and_random_access():
    signs = _c.rademacher_i8(9, 100_000)
    assert set(np.unique(signs)) == {-1, 1}
    assert abs(int(signs.sum())) < 1500
    assert np.array_equal(_c.rademacher_i8(9, 500, offset=777), signs[777:1277])


@pytest.mark.parametrize("n", [0, 1, 3, 4, 7, 8, 31, 32, 33, 100, 4096])
def test_xxh64_matches_the_reference_implementation(n):
    data = os.urandom(n)
    want = xxhash.xxh64_intdigest(data, 0)
    assert ref.xxh64(data) == want
    assert _c.xxh64(data) == want


def test_apply_c_matches_python_oracle():
    state = np.arange(-500, 500, dtype=np.int32)
    seeds, coefs = [1, 2, 2**62, 99], [17, -4, 100_000, -1]
    want = ref.apply_rademacher(state, seeds, coefs)
    _c.apply_rademacher_i32(state, np.array(seeds, np.uint64), np.array(coefs, np.int32))
    assert np.array_equal(state, want)


def test_apply_gives_the_same_bits_for_any_thread_count():
    seeds = np.arange(100, 132, dtype=np.uint64)
    coefs = (np.arange(32, dtype=np.int32) - 16) * 1000
    results = []
    for threads in (1, 3, 8):
        _c.set_num_threads(threads)
        state = np.zeros(2_000_003, dtype=np.int32)
        _c.apply_rademacher_i32(state, seeds, coefs)
        results.append(_c.xxh64(state))
    _c.set_num_threads(os.cpu_count() or 1)
    assert len(set(results)) == 1


# ------------------------------------------------------------------ fixed-point state and records


def test_fixed_point_round_trip_is_within_one_unit():
    theta = torch.randn(1000, generator=torch.Generator().manual_seed(0))
    state = FixedPointState.from_float(theta, frac_bits=20)
    assert (state.to_float() - theta).abs().max().item() <= state.unit


def test_fixed_point_refuses_values_that_do_not_fit():
    with pytest.raises(OverflowError):
        FixedPointState.from_float(np.array([5000.0]), frac_bits=20)


def test_step_record_json_round_trip():
    record = StepRecord(3, (1, 2**63 - 1), (-5, 7))
    assert StepRecord.from_json(record.to_json()) == record
    with pytest.raises(ValueError):
        StepRecord(0, (1, 2), (3,))


def test_coefficients_are_integers_with_zero_sum_for_ranks():
    coefs = integer_coefficients(np.array([3.0, 1.0, 2.0]), np.array([0.0, 5.0, 2.5]), 0.1, 0.05, 20)
    assert all(isinstance(c, int) for c in coefs)
    assert coefs[0] > 0 > coefs[1]


def test_rademacher_noise_is_plus_minus_one_float():
    eps = rademacher_noise(4, 1000)
    assert eps.dtype == torch.float32 and set(eps.unique().tolist()) == {-1.0, 1.0}


# ------------------------------------------------------------------ the point of M4: exact replay


def test_replaying_the_records_rebuilds_the_exact_state():
    run = train_integer_es(CartPoleTask(), steps=8, sigma=0.2, lr=0.3)
    rebuilt = replay(run.initial, run.records)
    assert rebuilt.digest() == run.es.state.digest()
    assert np.array_equal(rebuilt.values, run.es.state.values)


def test_two_runs_with_the_same_seed_agree_bit_for_bit():
    a = train_integer_es(CartPoleTask(), steps=5, sigma=0.2, lr=0.3, seed=11)
    b = train_integer_es(CartPoleTask(), steps=5, sigma=0.2, lr=0.3, seed=11)
    assert a.es.state.digest() == b.es.state.digest()
    assert [r.coefs for r in a.records] == [r.coefs for r in b.records]


def test_a_record_is_tiny():
    es = IntegerES(FixedPointState(np.zeros(10**6, np.int32)), pop=32)
    record = es.tell(es.ask(), np.arange(32.0), np.zeros(32))
    assert record.nbytes() == 8 + 32 * 12, "seeds + coefficients only, whatever the model size"


# ------------------------------------------------------------------ it still learns


def test_integer_es_solves_cartpole():
    run = train_integer_es(CartPoleTask(), steps=30, sigma=0.2, lr=0.3, eval_every=5)
    assert run.history[-1]["policy_return"] >= 475


@pytest.fixture(scope="module")
def mnist_task():
    from seedling.tasks.mnist import MnistTask, load

    try:
        return MnistTask(load())
    except ConnectionError as e:
        pytest.skip(f"MNIST not cached and download failed: {e}")


def test_integer_es_learns_mnist(mnist_task):
    run = train_integer_es(mnist_task, steps=300, pop=50, sigma=0.05, lr=0.01, eval_every=300)
    assert run.history[-1]["test_accuracy"] >= 0.82
