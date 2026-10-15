"""Text data for the tiny GPT (M0): a character tokenizer and a batch sampler."""

from __future__ import annotations

import os
from pathlib import Path

import torch

ASSETS = Path(__file__).resolve().parent / "assets"


def data_dir() -> Path:
    """Where downloaded datasets live: $SEEDLING_DATA, else ~/.cache/seedling."""
    return Path(os.environ.get("SEEDLING_DATA", Path.home() / ".cache" / "seedling"))


def load_text(name: str = "the-verdict.txt") -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


class CharTokenizer:
    """One id per distinct character. Sorted, so the same text always gives the same ids."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


def split(data: torch.Tensor, train_frac: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    """First 90% for training, last 10% for validation."""
    k = int(len(data) * train_frac)
    return data[:k], data[k:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows of the text. y is x shifted by one: the "next character" targets.

    x[b] = data[i : i + block_size]
    y[b] = data[i + 1 : i + block_size + 1]
    """
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
