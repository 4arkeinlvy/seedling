"""Countdown with a real LLM, as a runtime Workload.

The coordinator only needs layer shapes (read from the model's config, no weights), so it
can run without a GPU. Workers load the model lazily the first time they score.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.llm import countdown
from seedling.llm.codec import ChatCodec, HFChatCodec
from seedling.llm.lowrank import LayerSpec, PerturbableModel, apply_lowrank, plan_layers

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODULES = ("self_attn.o_proj", "mlp.down_proj")
EVAL_SEED = 10**9


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


class CountdownWorkload:
    name = "countdown"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        last_layers: int = 8,
        modules: Sequence[str] = DEFAULT_MODULES,
        prompts_per_step: int = 8,
        max_new_tokens: int = 160,
        eval_problems: int = 64,
        eval_batch: int = 32,
        max_rows: int = 128,
        device: str | None = None,
        model: nn.Module | None = None,
        codec: ChatCodec | None = None,
    ):
        self.model_name = model_name
        self.last_layers = last_layers
        self.modules = tuple(modules)
        self.prompts_per_step = prompts_per_step
        self.max_new_tokens = max_new_tokens
        self.eval_problems = eval_problems
        self.eval_batch = eval_batch
        self.max_rows = max_rows
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model
        self._codec = codec
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

                model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.bfloat16)
            model = model.to(self.device).eval()
            self._perturbable = PerturbableModel(model, self.specs)
        return self._perturbable

    def codec(self) -> ChatCodec:
        if self._codec is None:
            from transformers import AutoTokenizer

            self._codec = HFChatCodec(AutoTokenizer.from_pretrained(self.model_name))
        return self._codec

    # ---------------------------------------------------------------- Workload interface

    def initial_state(self, seed: int, frac_bits: int) -> FixedPointState:
        return FixedPointState(np.zeros(sum(s.size for s in self.specs), dtype=np.int32), frac_bits)

    def apply(self, state: FixedPointState, record: StepRecord) -> None:
        apply_lowrank(state, record, self.specs)

    def score(
        self, state: FixedPointState, step: int, seeds: Sequence[int], sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        problems = countdown.generate(self.prompts_per_step, seed=step)
        prompts = [p.prompt() for p in problems]
        net = self.perturbable()
        net.load(state)
        repeat = len(problems)
        per_chunk = max(1, self.max_rows // (2 * repeat))
        rewards = []
        for start in range(0, len(seeds), per_chunk):
            chunk = seeds[start : start + per_chunk]
            with net.population(chunk, sigma, repeat):
                texts = self._generate(prompts * (2 * len(chunk)))
            rewards += [countdown.reward(t, problems[i % repeat]) for i, t in enumerate(texts)]
        per_member = np.array(rewards).reshape(2 * len(seeds), repeat).mean(axis=1)
        return per_member[0::2], per_member[1::2]

    def evaluate(self, state: FixedPointState, step: int) -> dict[str, float]:
        problems = countdown.generate(self.eval_problems, seed=EVAL_SEED)
        self.perturbable().load(state)
        rewards = []
        for start in range(0, len(problems), self.eval_batch):
            chunk = problems[start : start + self.eval_batch]
            texts = self._generate([p.prompt() for p in chunk])
            rewards += [countdown.reward(t, p) for t, p in zip(texts, chunk, strict=True)]
        rewards_arr = np.array(rewards)
        return {
            "accuracy": float(np.mean(rewards_arr == countdown.CORRECT)),
            "well_formed": float(np.mean(rewards_arr > 0)),
        }

    @torch.no_grad()
    def _generate(self, prompts: list[str]) -> list[str]:
        codec = self.codec()
        net = self.perturbable()
        input_ids, attention_mask = codec.encode(prompts)
        output = net.model.generate(
            input_ids=input_ids.to(net.device),
            attention_mask=attention_mask.to(net.device),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=codec.pad_token_id,
        )
        return codec.decode(output[:, input_ids.shape[1] :])
