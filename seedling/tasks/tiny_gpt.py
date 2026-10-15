"""M0: a tiny character-level GPT, the first workload for seedling.

Same architecture as Raschka's *Build an LLM From Scratch* ch. 4 (and Karpathy's nanoGPT),
just tiny (~110k parameters), so it trains on a laptop CPU in seconds. In M3, ES will
train this same model.

    run:      uv run python -m seedling.tasks.tiny_gpt
    test:     uv run pytest tests/test_m0_tiny_gpt.py
    explained: docs/M0-explained.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from seedling.data import CharTokenizer, get_batch, load_text, split


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64  # max context length (number of positions)
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64  # must be divisible by n_head
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t only looks at positions <= t."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)  # each (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))

        out = weights @ v  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads
        return self.resid_dropout(self.projection(out))


class MLP(nn.Module):
    """Per-position feed-forward network: expand 4x, GELU, project back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer block: pre-LayerNorm, attention and MLP, each with a residual."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    """Token + position embeddings, n_layer Blocks, final LayerNorm, linear head to logits."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.config.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 init: normal(0, 0.02) weights, zero biases. PyTorch's default for
        nn.Embedding is normal(0, 1), which is far too big here."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def generate(
    model: TinyGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append max_new_tokens sampled tokens to idx (B, T)."""
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.config.block_size :])
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 16
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.01
    seed: int = 0
    device: str = "cpu"
    text: str = "the-verdict.txt"


def train(cfg: TrainConfig) -> tuple[TinyGPT, list[float]]:
    """Train a TinyGPT on cfg.text with AdamW; return (model, loss of every step)."""
    torch.manual_seed(cfg.seed)  # model init uses the global RNG
    gen = torch.Generator().manual_seed(cfg.seed)  # batches use their own RNG

    text = load_text(cfg.text)
    tok = CharTokenizer(text)
    train_data, _ = split(torch.tensor(tok.encode(text), dtype=torch.long))

    config = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
    )
    model = TinyGPT(config).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    losses: list[float] = []
    for _ in range(cfg.steps):
        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, gen, cfg.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig(steps=1500, device=device)
    model, losses = train(cfg)
    for i in range(0, len(losses), 250):
        print(f"step {i:5d}  loss {sum(losses[i : i + 50]) / len(losses[i : i + 50]):.3f}")
    tok = CharTokenizer(load_text(cfg.text))
    start = torch.tensor([tok.encode("I ")], device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    print("\nsample:\n" + tok.decode(generate(model.eval(), start, 300, generator=gen)[0].tolist()))


if __name__ == "__main__":
    main()
