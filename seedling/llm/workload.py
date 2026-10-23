"""Countdown with a real LLM, as a runtime Workload: generate an answer, grade it.

The coordinator only needs layer shapes (read from the model's config, no weights), so it
can run without a GPU. Workers load the model lazily the first time they score.

Two protocols:
    seedling   generated 3-number problems, partial credit for close answers (M6)
    tinyzero   the TinyZero dataset, prompt and score used by Qiu et al. (2025), so that
               accuracies compare with theirs; the last `eval_problems` are held out
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from seedling.exact.state import FixedPointState
from seedling.llm import countdown
from seedling.llm.base import DEFAULT_MODULES, LowRankLLMWorkload, linear_shapes
from seedling.llm.codec import ChatCodec, HFChatCodec

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EVAL_SEED = 10**9

__all__ = ["DEFAULT_MODEL", "CountdownWorkload", "linear_shapes"]


class CountdownWorkload(LowRankLLMWorkload):
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
        dtype: str = "bfloat16",
        model: nn.Module | None = None,
        codec: ChatCodec | None = None,
        protocol: str = "seedling",
        problems: Sequence[countdown.Problem] | None = None,
    ):
        if protocol not in ("seedling", "tinyzero"):
            raise ValueError("protocol is 'seedling' or 'tinyzero'")
        super().__init__(model_name, last_layers, modules, device, dtype, model, codec)
        self.prompts_per_step = prompts_per_step
        self.max_new_tokens = max_new_tokens
        self.eval_problems = eval_problems
        self.eval_batch = eval_batch
        self.max_rows = max_rows
        self.protocol = protocol
        self._problems = list(problems) if problems is not None else None

    # ---------------------------------------------------------------- protocol

    def codec(self) -> ChatCodec:
        if self._codec is None and self.protocol == "tinyzero":
            self._codec = HFChatCodec(self.tokenizer(), system=countdown.TINYZERO_SYSTEM, prefix=countdown.TINYZERO_PREFIX)
        return super().codec()

    def dataset(self) -> list[countdown.Problem]:
        if self._problems is None:
            self._problems = countdown.load_tinyzero()
        return self._problems

    def step_problems(self, step: int) -> list[countdown.Problem]:
        if self.protocol == "seedling":
            return countdown.generate(self.prompts_per_step, seed=step)
        pool = len(self.dataset()) - self.eval_problems
        return [self.dataset()[i] for i in random.Random(step).sample(range(pool), self.prompts_per_step)]

    def held_out_problems(self) -> list[countdown.Problem]:
        if self.protocol == "seedling":
            return countdown.generate(self.eval_problems, seed=EVAL_SEED)
        return self.dataset()[-self.eval_problems :]

    def prompt(self, problem: countdown.Problem) -> str:
        return problem.prompt() if self.protocol == "seedling" else countdown.tinyzero_prompt(problem)

    def reward(self, text: str, problem: countdown.Problem) -> float:
        return countdown.reward(text, problem) if self.protocol == "seedling" else countdown.tinyzero_reward(text, problem)

    # ---------------------------------------------------------------- Workload interface

    def score(
        self, state: FixedPointState, step: int, seeds: Sequence[int], sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        problems = self.step_problems(step)
        prompts = [self.prompt(p) for p in problems]
        net = self.perturbable()
        net.load(state)
        repeat = len(problems)
        per_chunk = max(1, self.max_rows // (2 * repeat))
        rewards = []
        for start in range(0, len(seeds), per_chunk):
            chunk = seeds[start : start + per_chunk]
            with net.population(chunk, sigma, repeat):
                texts = self._generate(prompts * (2 * len(chunk)))
            rewards += [self.reward(t, problems[i % repeat]) for i, t in enumerate(texts)]
        per_member = np.array(rewards).reshape(2 * len(seeds), repeat).mean(axis=1)
        return per_member[0::2], per_member[1::2]

    def evaluate(self, state: FixedPointState, step: int) -> dict[str, float]:
        problems = self.held_out_problems()
        self.perturbable().load(state)
        rewards = []
        for start in range(0, len(problems), self.eval_batch):
            chunk = problems[start : start + self.eval_batch]
            texts = self._generate([self.prompt(p) for p in chunk])
            rewards += [self.reward(t, p) for t, p in zip(texts, chunk, strict=True)]
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
