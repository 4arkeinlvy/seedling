"""M9: a character-level GPT trained by low-rank ES alone, no downloads needed.

ES pretraining of a language model is what EGGROLL (Sarkar et al. 2025) does at scale with
low-rank perturbations; this is the same recipe at laptop scale. Every Linear layer of a
TinyGPT trains through rank-1 +-1 noise and exact int32 updates (seedling.llm.lowrank); the
embeddings and LayerNorms keep their seeded random init, identical on every worker.

    fitness   minus the next-character cross-entropy on this step's text windows
    evaluate  cross-entropy on the held-out end of the text (nats per character)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from seedling.data import CharTokenizer, get_batch, load_text, split
from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.llm.lowrank import PerturbableModel, apply_lowrank, plan_layers
from seedling.tasks.tiny_gpt import GPTConfig, TinyGPT


class TinyLMWorkload:
    name = "tinylm"

    def __init__(
        self,
        n_layer: int = 4,
        n_embd: int = 128,
        n_head: int = 4,
        block_size: int = 64,
        batch: int = 8,
        eval_batches: int = 8,
        max_rows: int = 256,
        model_seed: int = 0,
        text: str = "the-verdict.txt",
        dtype: str = "float32",
    ):
        self.tokenizer = CharTokenizer(load_text(text))
        data = torch.tensor(self.tokenizer.encode(load_text(text)), dtype=torch.long)
        self.train_data, self.val_data = split(data)
        self.block_size = block_size
        self.batch = batch
        self.eval_batches = eval_batches
        self.max_rows = max_rows
        config = GPTConfig(self.tokenizer.vocab_size, block_size, n_layer, n_head, n_embd)
        with torch.random.fork_rng(devices=[]):  # the base weights depend on model_seed only
            torch.manual_seed(model_seed)
            self.model = TinyGPT(config).eval().to(getattr(torch, dtype))
        shapes = [
            (name, m.weight.shape[0], m.weight.shape[1])
            for name, m in self.model.named_modules()
            if isinstance(m, torch.nn.Linear)
        ]
        self.specs = plan_layers(shapes)
        self.net = PerturbableModel(self.model, self.specs)

    @property
    def trainable_parameters(self) -> int:
        return sum(s.size for s in self.specs)

    def initial_state(self, seed: int, frac_bits: int) -> FixedPointState:
        return FixedPointState(np.zeros(self.trainable_parameters, dtype=np.int32), frac_bits)

    def apply(self, state: FixedPointState, record: StepRecord) -> None:
        apply_lowrank(state, record, self.specs)

    @torch.no_grad()
    def _losses(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy of each row of x (rows, T)."""
        logits, _ = self.model(x)
        losses = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")
        return losses.mean(dim=1)

    def score(
        self, state: FixedPointState, step: int, seeds: Sequence[int], sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = get_batch(self.train_data, self.batch, self.block_size, torch.Generator().manual_seed(step))
        self.net.load(state)
        per_chunk = max(1, self.max_rows // (2 * self.batch))
        losses = []
        for start in range(0, len(seeds), per_chunk):
            chunk = seeds[start : start + per_chunk]
            with self.net.population(chunk, sigma, self.batch):
                rows = 2 * len(chunk)
                losses.append(self._losses(x.repeat(rows, 1), y.repeat(rows, 1)))
        per_member = -torch.cat(losses).double().numpy().reshape(2 * len(seeds), self.batch).mean(axis=1)
        return per_member[0::2], per_member[1::2]

    def evaluate(self, state: FixedPointState, step: int) -> dict[str, float]:
        self.net.load(state)
        generator = torch.Generator().manual_seed(10**9)
        losses = [
            float(self._losses(*get_batch(self.val_data, 32, self.block_size, generator)).mean())
            for _ in range(self.eval_batches)
        ]
        return {"val_loss": float(np.mean(losses))}
