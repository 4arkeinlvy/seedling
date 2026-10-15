"""Turning prompts into token ids and back. The workload depends on this small interface,
so tests can use a stub instead of a real tokenizer."""

from __future__ import annotations

from typing import Protocol

import torch


class ChatCodec(Protocol):
    pad_token_id: int

    def encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Chat-formatted, left-padded (input_ids, attention_mask)."""

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        """One string per row of generated tokens."""


class HFChatCodec:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
            for prompt in prompts
        ]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left")
        return batch["input_ids"], batch["attention_mask"]

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
