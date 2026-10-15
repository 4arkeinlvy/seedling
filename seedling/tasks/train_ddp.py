"""M2: train the tiny GPT with our DDP on W ranks.

Every rank draws the SAME global batch (same seed) and keeps only its slice of rows. So W
ranks with batch B/W each compute exactly the gradient of the single-process run with
batch B: the test suite checks that the two runs end with the same weights.

    run (2 processes):  uv run python -m seedling.foundation.launch -n 2 seedling.tasks.train_ddp:main
    backend:            SEEDLING_BACKEND=python|c   (default c)
    explained:          docs/M2-explained.md
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import torch

from seedling.data import CharTokenizer, get_batch, load_text, split
from seedling.foundation.collectives import ProcessGroup
from seedling.foundation.ddp import DDP
from seedling.tasks.tiny_gpt import GPTConfig, TinyGPT, TrainConfig


def build_model(cfg: TrainConfig, vocab_size: int) -> TinyGPT:
    """Same initialization as tiny_gpt.train(): seed the global RNG, then build."""
    torch.manual_seed(cfg.seed)
    return TinyGPT(
        GPTConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
        )
    )


def train_ddp(
    pg: ProcessGroup,
    cfg: TrainConfig,
    *,
    overlap: bool = True,
    bucket_cap_mb: float = 0.1,
    model: TinyGPT | None = None,
) -> tuple[DDP, list[float], list[float]]:
    """Train on pg.world_size ranks. Returns (ddp, global loss per step, seconds per step).
    `model` lets the caller pass a ready model (ranks-as-threads share one global RNG)."""
    world, rank = pg.world_size, pg.rank
    if cfg.batch_size % world:
        raise ValueError(f"batch_size {cfg.batch_size} must be divisible by world size {world}")
    if cfg.device != "cpu":
        raise ValueError("this DDP runs on CPU (our transport is CPU sockets)")

    text = load_text(cfg.text)
    tok = CharTokenizer(text)
    train_data, _ = split(torch.tensor(tok.encode(text), dtype=torch.long))
    if model is None:
        model = build_model(cfg, tok.vocab_size)

    ddp = DDP(model, pg, bucket_cap_mb=bucket_cap_mb, overlap=overlap)
    opt = torch.optim.AdamW(ddp.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    local = cfg.batch_size // world
    rows = slice(rank * local, (rank + 1) * local)

    losses: list[float] = []
    step_seconds: list[float] = []
    loss_sum = np.zeros(1, dtype=np.float32)
    for _ in range(cfg.steps):
        t0 = time.perf_counter()
        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, gen)
        _, loss = ddp(x[rows], y[rows])
        opt.zero_grad(set_to_none=True)
        ddp.backward(loss)
        opt.step()
        step_seconds.append(time.perf_counter() - t0)

        loss_sum[0] = loss.item()
        pg.ring_all_reduce(loss_sum)
        losses.append(float(loss_sum[0]) / world)
    return ddp, losses, step_seconds


def params_digest(model: torch.nn.Module) -> str:
    """Hash of all weights: equal digests = bit-identical models."""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main(rank: int, world: int) -> dict:
    torch.set_num_threads(max(1, 4 // world))
    pg = ProcessGroup.from_env(backend=os.environ.get("SEEDLING_BACKEND", "c"))
    ddp = None
    try:
        ddp, losses, seconds = train_ddp(pg, TrainConfig(steps=300))
        return {
            "loss_first_20": round(sum(losses[:20]) / 20, 3),
            "loss_last_20": round(sum(losses[-20:]) / 20, 3),
            "ms_per_step": round(1000 * sorted(seconds)[len(seconds) // 2], 1),
            "weights_digest": params_digest(ddp.module),
        }
    finally:
        if ddp is not None:
            ddp.close()
        pg.close()
