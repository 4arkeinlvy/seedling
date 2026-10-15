"""M2 spec: DDP from scratch. Run: uv run pytest tests/test_m2_ddp.py -x

Most tests run ranks as threads over Unix socketpairs (tests/thread_ranks.py): torch is
imported once, so 4 ranks fit in the laptop's RAM. One heavy test uses real processes.
"""

import copy

import numpy as np
import pytest
import torch

from seedling.data import CharTokenizer, get_batch, load_text
from seedling.foundation.ddp import DDP, make_buckets
from seedling.foundation.launch import run_ranks
from seedling.tasks.tiny_gpt import TrainConfig, train
from seedling.tasks.train_ddp import build_model, train_ddp
from tests import rank_programs as rp
from tests.thread_ranks import run_threads

TEXT = load_text()
TOK = CharTokenizer(TEXT)
V = TOK.vocab_size
DATA = torch.tensor(TOK.encode(TEXT), dtype=torch.long)
CFG = TrainConfig(steps=10, batch_size=16, seed=0)
SMALL_BUCKETS_MB = 0.05  # 12 buckets for this 438 KB model, so overlap really happens
BACKENDS = pytest.mark.parametrize("backend", ["python", "c"], ids=["backend_python", "backend_c"])


@pytest.fixture(autouse=True)
def one_torch_thread_per_rank():
    """Ranks are threads here: give torch one thread, restore it afterwards."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def flat_grads(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]).clone()


# ------------------------------------------------------------------ rank bodies (threads)


def _construct(pg, models):
    ddp = DDP(models[pg.rank], pg, overlap=False)
    ddp.close()
    return flat_params(ddp.module)


def _one_backward(pg, models, batches, overlap):
    ddp = DDP(models[pg.rank], pg, overlap=overlap, bucket_cap_mb=SMALL_BUCKETS_MB)
    x, y = batches[pg.rank]
    _, loss = ddp(x, y)
    ddp.backward(loss)
    ddp.close()
    return flat_grads(ddp.module)


def _train(pg, models, cfg, overlap):
    ddp, losses, _ = train_ddp(
        pg, cfg, overlap=overlap, bucket_cap_mb=SMALL_BUCKETS_MB, model=models[pg.rank]
    )
    ddp.close()
    return losses, flat_params(ddp.module)


def _accumulate(pg, models, micro, full):
    """Path A: two micro-batches, gradients synced only after the second.
    Path B: one backward on both micro-batches together."""
    a = DDP(models[pg.rank][0], pg, bucket_cap_mb=SMALL_BUCKETS_MB)
    (x1, y1), (x2, y2) = micro[pg.rank]
    with a.no_sync():
        a.backward(a(x1, y1)[1] / 2)
    a.backward(a(x2, y2)[1] / 2)
    a.close()
    b = DDP(models[pg.rank][1], pg, bucket_cap_mb=SMALL_BUCKETS_MB)
    x, y = full[pg.rank]
    b.backward(b(x, y)[1])
    b.close()
    return flat_grads(a.module), flat_grads(b.module)


# ------------------------------------------------------------------ tests


@pytest.mark.parametrize("world", [2, 4])
def test_all_ranks_start_from_rank0_weights(world):
    models = [build_model(TrainConfig(seed=r), V) for r in range(world)]  # all different
    rank0 = flat_params(models[0])
    for rank, params in enumerate(run_threads(world, _construct, models, timeout=120)):
        assert torch.equal(params, rank0), f"rank {rank} does not hold rank 0's weights"


@pytest.mark.parametrize("overlap", [False, True], ids=["sync_after_backward", "overlap"])
def test_gradients_are_the_average_over_ranks(overlap):
    world = 3
    base = build_model(CFG, V)
    gen = torch.Generator().manual_seed(1)
    batches = [get_batch(DATA, 4, CFG.block_size, gen) for _ in range(world)]  # one per rank
    expected = torch.zeros(sum(p.numel() for p in base.parameters()))
    for x, y in batches:  # each rank's own gradient, computed without DDP
        m = copy.deepcopy(base)
        m(x, y)[1].backward()
        expected += flat_grads(m)
    expected /= world

    models = [copy.deepcopy(base) for _ in range(world)]
    grads = run_threads(world, _one_backward, models, batches, overlap, timeout=120)
    for rank, g in enumerate(grads):
        assert torch.equal(g, grads[0]), f"rank {rank} ended with different gradient bits"
        assert torch.allclose(g, expected, rtol=1e-5, atol=1e-7), "not the mean of the ranks"


@pytest.fixture(scope="module")
def single_process_run():
    model, losses = train(CFG)
    return losses, flat_params(model)


@BACKENDS
@pytest.mark.parametrize("world", [2, 4])
def test_ddp_matches_single_process_training(world, backend, single_process_run):
    ref_losses, ref_params = single_process_run
    models = [build_model(CFG, V) for _ in range(world)]  # same init as train()
    results = run_threads(world, _train, models, CFG, True, backend=backend, timeout=180)
    for rank, (losses, params) in enumerate(results):
        assert torch.equal(params, results[0][1]), f"rank {rank}'s weights drifted from rank 0"
        np.testing.assert_allclose(losses, ref_losses, rtol=1e-5)
        assert torch.allclose(params, ref_params, atol=1e-4), (
            f"max weight difference {float((params - ref_params).abs().max()):.2e} vs 1 process"
        )


def test_overlap_gives_the_same_bits_as_sync_after_backward():
    cfg = TrainConfig(steps=5, batch_size=16, seed=0)
    runs = []
    for overlap in (False, True):
        models = [build_model(cfg, V) for _ in range(2)]
        runs.append(run_threads(2, _train, models, cfg, overlap, timeout=120))
    assert torch.equal(runs[0][0][1], runs[1][0][1]), "same buckets + same order = same bits"


def test_buckets_cover_every_parameter_once_and_respect_the_cap():
    params = list(reversed(list(build_model(CFG, V).parameters())))
    cap = 50_000
    buckets = make_buckets(params, cap)
    assert [id(p) for b in buckets for p in b.params] == [id(p) for p in params], "order or coverage"
    for b in buckets:
        assert b.buffer.numel() * 4 <= cap or len(b.params) == 1
    assert len(make_buckets(params, 10**9)) == 1
    assert len(make_buckets(params, 1)) == len(params)


def test_no_sync_accumulation_equals_one_big_batch():
    world = 2
    base = build_model(CFG, V)
    gen = torch.Generator().manual_seed(2)
    micro, full = [], []
    for _ in range(world):
        (x1, y1), (x2, y2) = get_batch(DATA, 4, 64, gen), get_batch(DATA, 4, 64, gen)
        micro.append(((x1, y1), (x2, y2)))
        full.append((torch.cat([x1, x2]), torch.cat([y1, y2])))
    models = [(copy.deepcopy(base), copy.deepcopy(base)) for _ in range(world)]
    for rank, (accumulated, big) in enumerate(run_threads(world, _accumulate, models, micro, full)):
        assert torch.allclose(accumulated, big, rtol=1e-5, atol=1e-7), f"rank {rank}"


@pytest.mark.heavy
def test_real_processes_end_bit_identical():
    reports = run_ranks(rp.ddp_train_report, 2, args=(30,), timeout=300)
    assert reports[0]["digest"] == reports[1]["digest"], "the two processes' weights differ"
    assert reports[0]["loss_last"] < reports[0]["loss_first"]
