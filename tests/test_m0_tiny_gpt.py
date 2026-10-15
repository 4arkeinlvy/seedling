"""M0 spec: the tiny GPT. Run: uv run pytest tests/test_m0_tiny_gpt.py -x"""

import math

import pytest
import torch

from seedling.data import CharTokenizer, get_batch, load_text
from seedling.tasks.tiny_gpt import GPTConfig, TinyGPT, TrainConfig, generate, train

TEXT = load_text()
TOK = CharTokenizer(TEXT)
V = TOK.vocab_size
DATA = torch.tensor(TOK.encode(TEXT), dtype=torch.long)
BLOCK = 32


def make_model(seed: int = 0) -> TinyGPT:
    torch.manual_seed(seed)
    return TinyGPT(GPTConfig(vocab_size=V, block_size=BLOCK, n_layer=2, n_head=4, n_embd=64))


def test_logits_and_loss_shapes():
    model = make_model()
    x, y = get_batch(DATA, 3, BLOCK, torch.Generator().manual_seed(0))
    logits, loss = model(x)
    assert logits.shape == (3, BLOCK, V)
    assert loss is None, "loss must be None when no targets are given"
    logits, loss = model(x, y)
    assert loss.ndim == 0 and torch.isfinite(loss), "loss must be one finite number"


def test_shorter_sequences_work():
    logits, _ = make_model()(torch.zeros(2, 5, dtype=torch.long))
    assert logits.shape == (2, 5, V)


def test_too_long_sequence_raises():
    with pytest.raises(ValueError):
        make_model()(torch.zeros(1, BLOCK + 1, dtype=torch.long))


def test_attention_is_causal():
    model = make_model().eval()
    a = torch.randint(0, V, (1, 20), generator=torch.Generator().manual_seed(0))
    b = a.clone()
    b[0, 10] = (a[0, 10] + 1) % V  # change only token 10
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-6), (
        "changing token 10 changed the logits at positions 0-9: the model can see the future"
    )
    assert not torch.allclose(la[:, 10], lb[:, 10]), "position 10 ignored its own token"


def test_initial_loss_is_near_uniform():
    model = make_model()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        losses = [model(*get_batch(DATA, 8, BLOCK, gen))[1].item() for _ in range(4)]
    mean = sum(losses) / len(losses)
    assert abs(mean - math.log(V)) < 0.07, (
        f"initial loss {mean:.3f}, expected about ln({V}) = {math.log(V):.3f}: an untrained "
        "model should guess uniformly. Did you call self.apply(self._init_weights)?"
    )


def test_can_overfit_one_batch():
    model = make_model()
    x, y = get_batch(DATA, 4, BLOCK, torch.Generator().manual_seed(1))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3, f"loss {loss.item():.3f} after 200 steps on ONE batch; should memorize it"


def test_generate_crops_to_block_size():
    out = generate(make_model().eval(), torch.zeros(1, 1, dtype=torch.long), max_new_tokens=40)
    assert out.shape == (1, 41)


def test_train_returns_one_float_loss_per_step():
    model, losses = train(TrainConfig(steps=5))
    assert isinstance(model, TinyGPT)
    assert len(losses) == 5 and all(isinstance(v, float) for v in losses)


def test_train_is_deterministic():
    _, a = train(TrainConfig(steps=30, seed=7))
    _, b = train(TrainConfig(steps=30, seed=7))
    _, c = train(TrainConfig(steps=30, seed=8))
    assert a == b, "same seed, different losses: seed the model init AND the batch generator"
    assert a != c, "different seeds gave identical losses"


def test_train_reduces_loss():
    _, losses = train(TrainConfig(steps=300, seed=0))
    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    assert last < first - 0.5, f"loss only went from {first:.2f} to {last:.2f} in 300 steps"


def test_train_on_gpu(cuda):
    model, losses = train(TrainConfig(steps=20, device="cuda"))
    assert next(model.parameters()).is_cuda
    assert all(math.isfinite(v) for v in losses)
