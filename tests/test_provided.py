"""Tests for the code that comes with the scaffold. These pass before you write anything;
if one fails, your environment is broken (not your code)."""

import time

import pytest
import torch

from seedling import _c
from seedling.data import CharTokenizer, get_batch, load_text
from seedling.foundation.launch import RankError, run_ranks
from tests import rank_programs as rp


def test_tokenizer_roundtrip():
    text = load_text()
    tok = CharTokenizer(text)
    assert 30 < tok.vocab_size < 100
    assert tok.decode(tok.encode(text[:500])) == text[:500]


def test_get_batch_targets_are_shifted_inputs():
    data = torch.arange(1000)
    x, y = get_batch(data, batch_size=4, block_size=8, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_get_batch_is_reproducible():
    data = torch.arange(1000)
    a = get_batch(data, 4, 8, torch.Generator().manual_seed(3))
    b = get_batch(data, 4, 8, torch.Generator().manual_seed(3))
    assert torch.equal(a[0], b[0])


def test_c_library_loads():
    assert _c.version() == "seedling 0.0.1"


def test_run_ranks_returns_results_in_rank_order():
    assert run_ranks(rp.echo, 3) == [(0, 3, "0", "3"), (1, 3, "1", "3"), (2, 3, "2", "3")]


def test_run_ranks_reports_a_failing_rank():
    t0 = time.monotonic()
    with pytest.raises(RankError, match="boom from rank 1"):
        run_ranks(rp.fail_on_rank_1, 3)
    assert time.monotonic() - t0 < 15


def test_run_ranks_kills_hung_ranks():
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        run_ranks(rp.sleep_forever, 2, timeout=2)
    assert time.monotonic() - t0 < 15


def test_run_ranks_detects_a_crashed_rank():
    with pytest.raises(RankError, match="SIGKILL"):
        run_ranks(rp.die_on_rank_0, 2)


def test_cuda_works(cuda):
    x = torch.ones(1000, device=cuda)
    assert (x * 2).sum().item() == 2000
