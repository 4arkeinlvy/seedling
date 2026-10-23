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
from seedling.llm import countdown, evidence
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
        verify_tolerance: float = 0.5,
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
        self.verify_tolerance = verify_tolerance
        self.verified = self.regenerated = 0  # audits settled by a check vs by generating again

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
        f_plus, f_minus, _ = self.score_with_evidence(state, step, seeds, sigma)
        return f_plus, f_minus

    def score_with_evidence(
        self, state: FixedPointState, step: int, seeds: Sequence[int], sigma: float
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Scores, and per seed the generated tokens of its 2 x prompts rows (see evidence.py)."""
        problems = self.step_problems(step)
        prompts = [self.prompt(p) for p in problems]
        net = self.perturbable()
        net.load(state)
        repeat = len(problems)
        per_chunk = max(1, self.max_rows // (2 * repeat))
        rewards, packs = [], []
        for start in range(0, len(seeds), per_chunk):
            chunk = seeds[start : start + per_chunk]
            with net.population(chunk, sigma, repeat):
                completions = self._generate_ids(prompts * (2 * len(chunk)))
            rewards += self._grade(completions, problems)
            packs += [evidence.pack(rows) for rows in np.split(completions, len(chunk))]
        per_member = np.array(rewards).reshape(2 * len(seeds), repeat).mean(axis=1)
        return per_member[0::2], per_member[1::2], packs

    def check(
        self, state: FixedPointState, step: int, seeds: Sequence[int], sigma: float, claims: Sequence[str]
    ) -> list[tuple[float, float] | None]:
        """Per seed: the scores of the claimed tokens if one teacher-forced pass confirms them,
        else None. Grading the same tokens gives the same scores on any machine."""
        problems = self.step_problems(step)
        prompts = [self.prompt(p) for p in problems] * 2
        self.perturbable().load(state)
        out: list[tuple[float, float] | None] = []
        for seed, claim in zip(seeds, claims, strict=True):
            rewards = self._check(prompts, problems, seed, sigma, claim)
            if rewards is None:
                out.append(None)
            else:
                plus, minus = np.array(rewards).reshape(2, len(problems)).mean(axis=1)
                out.append((float(plus), float(minus)))
        return out

    def verify(
        self, state: FixedPointState, step: int, seeds: Sequence[int], sigma: float, claims: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """An auditor's scores: those of the claimed tokens if they check out (the claim is then
        its evidence too), else its own, from generating again (with its own evidence)."""
        f_plus, f_minus, packs = np.zeros(len(seeds)), np.zeros(len(seeds)), list(claims)
        for k, checked in enumerate(self.check(state, step, seeds, sigma, claims)):
            if checked is None:
                self.regenerated += 1
                fp, fm, (packs[k],) = self.score_with_evidence(state, step, [seeds[k]], sigma)
                f_plus[k], f_minus[k] = fp[0], fm[0]
            else:
                self.verified += 1
                f_plus[k], f_minus[k] = checked
        return f_plus, f_minus, packs

    def _check(
        self, prompts: list[str], problems: list[countdown.Problem], seed: int, sigma: float, claim: str
    ) -> list[float] | None:
        """Rewards of the claimed completions of one seed, or None if they do not hold up."""
        try:
            claimed = evidence.unpack(claim)
        except ValueError:
            return None
        rows, length = claimed.shape
        if rows != len(prompts) or not 0 < length <= self.max_new_tokens:
            return None
        tokens, valid = evidence.canonical(claimed, self._eos(), self.codec().pad_token_id)
        unfinished = ~np.isin(tokens, self._eos()).any(axis=1)
        if unfinished.any() and length != self.max_new_tokens:
            return None  # a row that never stopped must have used every token: no cutting text short
        net = self.perturbable()
        prompt_ids, prompt_mask = self.codec().encode(prompts)
        ids = torch.cat([prompt_ids, torch.from_numpy(tokens).long()], dim=1)
        mask = torch.cat([prompt_mask, torch.from_numpy(valid).long()], dim=1)
        width = prompt_ids.shape[1]
        with net.population([seed], sigma, len(problems)):
            logits = self.forward_logits(ids, mask, keep=length + 1)[:, :length]  # predictions of the claimed tokens
        ok = evidence.greedy_rows(
            logits, torch.from_numpy(tokens).to(logits.device), torch.from_numpy(valid).to(logits.device),
            self.verify_tolerance,
        )  # fmt: skip
        if not bool(ok.all()):
            return None
        return self._grade(tokens, problems)

    def _eos(self) -> list[int]:
        eos = self.perturbable().model.generation_config.eos_token_id
        return [] if eos is None else [eos] if isinstance(eos, int) else list(eos)

    def _grade(self, completions: np.ndarray, problems: list[countdown.Problem]) -> list[float]:
        """Rewards of completion rows (row r answers problem r mod len(problems))."""
        tokens, _ = evidence.canonical(completions, self._eos(), self.codec().pad_token_id)
        texts = self.codec().decode(torch.from_numpy(tokens))
        return [self.reward(t, problems[i % len(problems)]) for i, t in enumerate(texts)]

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

    def _generate(self, prompts: list[str]) -> list[str]:
        return self.codec().decode(torch.from_numpy(self._generate_ids(prompts)))

    @torch.no_grad()
    def _generate_ids(self, prompts: list[str]) -> np.ndarray:
        """Greedy completions (rows, new tokens), padded after each row's end."""
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
        return output[:, input_ids.shape[1] :].cpu().numpy().astype(np.int32)
