"""M3: train the M0 tiny GPT with ES instead of backprop (forward passes only).

Uses the in-place (MeZO) evaluation: one model in memory, perturbed and restored per seed.

    run:  uv run python -m seedling.tasks.es_demo --task gpt
"""

from __future__ import annotations

import numpy as np
import torch

from seedling.data import CharTokenizer, get_batch, load_text, split
from seedling.es import ES, Adam, assign, evaluate_inplace, flatten
from seedling.tasks.tiny_gpt import GPTConfig, TinyGPT


def train_es_gpt(
    steps: int = 300,
    pop: int = 32,
    sigma: float = 0.01,
    lr: float = 0.01,
    batch_size: int = 16,
    block_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[TinyGPT, list[dict]]:
    text = load_text()
    tok = CharTokenizer(text)
    train_data, _ = split(torch.tensor(tok.encode(text), dtype=torch.long))

    torch.manual_seed(seed)
    model = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=block_size)).to(device).eval()
    es = ES(flatten(model), Adam(lr=lr), pop=pop, sigma=sigma, seed=seed)
    generator = torch.Generator().manual_seed(seed)

    history = []
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, batch_size, block_size, generator, device)

        def fitness(m: TinyGPT) -> float:
            return -m(x, y)[1].item()

        seeds = es.ask()
        f_plus, f_minus = evaluate_inplace(model, fitness, es.theta, seeds, sigma)
        es.tell(seeds, f_plus, f_minus)
        history.append(
            {
                "step": step,
                "forward_passes": step * 2 * pop,
                "train_loss": float(-np.mean(np.concatenate([f_plus, f_minus]))),
            }
        )
    assign(model, es.theta)
    return model, history
