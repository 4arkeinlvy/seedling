"""What every low-rank LLM workload shares: the model, the trained layers, the exact state.

A subclass decides only how a population is scored (generate and grade, or read label
log-probabilities). The coordinator needs layer shapes alone (read from the model's config,
no weights), so it can run without a GPU; workers load the model the first time they score.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.llm.codec import ChatCodec, HFChatCodec
from seedling.llm.lowrank import LayerSpec, PerturbableModel, apply_lowrank, plan_layers

DEFAULT_MODULES = ("self_attn.o_proj", "mlp.down_proj")


def linear_shapes(model: nn.Module, last_layers: int, modules: Sequence[str]) -> list[tuple[str, int, int]]:
    """(name, out, in) of the chosen Linear modules in the last `last_layers` decoder layers."""
    blocks = model.get_submodule("model.layers")
    first = len(blocks) - last_layers
    shapes = []
    for i in range(first, len(blocks)):
        for module in modules:
            name = f"model.layers.{i}.{module}"
            weight = model.get_submodule(name).weight
            shapes.append((name, weight.shape[0], weight.shape[1]))
    return shapes


def positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids that ignore left padding, so a row's result never depends on its batch."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


class LowRankLLMWorkload:
    """Base of the LLM workloads: rank-1 noise on chosen Linear layers, exact int32 updates."""

    name = "llm"
    chat = True  # chat template for prompts (instruction tasks) or plain text (log-likelihood tasks)

    def __init__(
        self,
        model_name: str,
        last_layers: int = 8,
        modules: Sequence[str] = DEFAULT_MODULES,
        device: str | None = None,
        dtype: str = "bfloat16",
        model: nn.Module | None = None,
        codec: ChatCodec | None = None,
    ):
        self.model_name = model_name
        self.last_layers = last_layers
        self.modules = tuple(modules)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)
        self._model = model
        self._codec = codec
        self._tokenizer = None
        self._perturbable: PerturbableModel | None = None
        self.specs: list[LayerSpec] = plan_layers(linear_shapes(self._skeleton(), last_layers, self.modules))

    # ---------------------------------------------------------------- model loading

    def _skeleton(self) -> nn.Module:
        """The model's structure without weights (meta device): enough for shapes."""
        if self._model is not None:
            return self._model
        from transformers import AutoConfig, AutoModelForCausalLM

        with torch.device("meta"):
            return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

    def perturbable(self) -> PerturbableModel:
        if self._perturbable is None:
            model = self._model
            if model is None:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=self.dtype)
            model = model.to(self.device).eval()
            self._perturbable = PerturbableModel(model, self.specs)
        return self._perturbable

    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def codec(self) -> ChatCodec:
        if self._codec is None:
            self._codec = HFChatCodec(self.tokenizer(), chat=self.chat)
        return self._codec

    @property
    def trainable_parameters(self) -> int:
        return sum(s.size for s in self.specs)

    def export(self, state: FixedPointState, out: str) -> None:
        """A standard Hugging Face folder (safetensors) holding the trained weights."""
        net = self.perturbable()
        net.load(state)
        model = net.unwrap()
        self._perturbable = None
        model.save_pretrained(out, safe_serialization=True)
        if self._model is None:  # a hub model: ship its tokenizer too
            self.tokenizer().save_pretrained(out)

    # ---------------------------------------------------------------- Workload interface (shared part)

    def initial_state(self, seed: int, frac_bits: int) -> FixedPointState:
        return FixedPointState(np.zeros(self.trainable_parameters, dtype=np.int32), frac_bits)

    def apply(self, state: FixedPointState, record: StepRecord) -> None:
        apply_lowrank(state, record, self.specs)

    # ---------------------------------------------------------------- shared measurements

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, keep: int = 0) -> torch.Tensor:
        """Logits of every position, or of the last `keep` only (the vocabulary projection is
        the costly part of a long forward pass)."""
        net = self.perturbable()
        input_ids, attention_mask = input_ids.to(net.device), attention_mask.to(net.device)
        out = net.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=positions(attention_mask), logits_to_keep=keep
        )
        return out.logits.float()

    @torch.no_grad()
    def language_loss(self, state: FixedPointState, texts: Sequence[str], batch: int = 8) -> float:
        """Mean next-token cross-entropy on `texts` (plain text): what the model forgets."""
        self.perturbable().load(state)
        codec = HFChatCodec(self.tokenizer(), chat=False) if self._codec is None else self.codec()
        total, count = 0.0, 0
        for start in range(0, len(texts), batch):
            input_ids, mask = codec.encode(list(texts[start : start + batch]))
            logits = self.forward_logits(input_ids, mask)[:, :-1]
            targets = input_ids[:, 1:].to(logits.device)
            keep = (mask[:, 1:] * mask[:, :-1]).bool().to(logits.device)
            losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
            total += float(losses[keep].sum())
            count += int(keep.sum())
        return total / max(count, 1)
