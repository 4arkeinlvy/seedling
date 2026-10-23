"""SST-2 sentiment by label-word likelihood, the standard zeroth-order LLM benchmark.

The prompt and verbalizer follow MeZO (Malladi et al. 2023) and ZO-Bench (Zhang et al. 2024):

    "<sentence> It was"  ->  " great" (positive) or " terrible" (negative)

The fitness of a population member is the mean log-probability of the right label word,
normalized over the two label words, on this step's examples. Only forward passes: no
generation, so a whole population is one batched call and the task runs on a CPU.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from seedling.data import data_dir
from seedling.exact.state import FixedPointState
from seedling.llm.base import DEFAULT_MODULES, LowRankLLMWorkload
from seedling.llm.codec import ChatCodec

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TEMPLATE = "{sentence} It was"
LABEL_WORDS = (" terrible", " great")  # label 0 = negative, 1 = positive
SST2 = "stanfordnlp/sst2"
SST2_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
}


@dataclass(frozen=True)
class Example:
    sentence: str
    label: int

    def prompt(self) -> str:
        return TEMPLATE.format(sentence=self.sentence.strip())


def load_sst2(split: str) -> list[Example]:
    """SST-2 from the Hugging Face hub, cached under seedling's data directory."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SST2, SST2_FILES[split], repo_type="dataset", cache_dir=data_dir() / "hf")
    table = pq.read_table(path, columns=["sentence", "label"]).to_pydict()
    return [Example(s, int(y)) for s, y in zip(table["sentence"], table["label"], strict=True)]


class Sst2Workload(LowRankLLMWorkload):
    name = "sst2"
    chat = False

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        last_layers: int = 8,
        modules: Sequence[str] = DEFAULT_MODULES,
        examples_per_step: int = 16,
        eval_examples: int = 872,
        eval_batch: int = 64,
        max_rows: int = 256,
        device: str | None = None,
        dtype: str = "float32",
        model: nn.Module | None = None,
        codec: ChatCodec | None = None,
        train: Sequence[Example] | None = None,
        validation: Sequence[Example] | None = None,
        label_token_ids: tuple[int, int] | None = None,
    ):
        super().__init__(model_name, last_layers, modules, device, dtype, model, codec)
        self.examples_per_step = examples_per_step
        self.eval_examples = eval_examples
        self.eval_batch = eval_batch
        self.max_rows = max_rows
        self._train = list(train) if train is not None else None
        self._validation = list(validation) if validation is not None else None
        self._label_ids = label_token_ids

    # ---------------------------------------------------------------- data

    def train_set(self) -> list[Example]:
        if self._train is None:
            self._train = load_sst2("train")
        return self._train

    def validation_set(self) -> list[Example]:
        if self._validation is None:
            self._validation = load_sst2("validation")
        return self._validation[: self.eval_examples]

    def step_examples(self, step: int) -> list[Example]:
        """This step's minibatch: a pure function of the step, like every other task."""
        return random.Random(step).sample(self.train_set(), self.examples_per_step)

    def label_ids(self) -> tuple[int, int]:
        if self._label_ids is None:
            ids = [self.tokenizer().encode(word, add_special_tokens=False) for word in LABEL_WORDS]
            if any(len(i) != 1 for i in ids):
                raise ValueError(f"label words must be single tokens for {self.model_name}: {ids}")
            self._label_ids = (ids[0][0], ids[1][0])
        return self._label_ids

    # ---------------------------------------------------------------- scoring

    def _label_logprobs(self, prompts: list[str], labels: torch.Tensor) -> torch.Tensor:
        """log P(right label word | prompt), normalized over the two label words, per row."""
        input_ids, mask = self.codec().encode(prompts)
        logits = self.forward_logits(input_ids, mask)[:, -1, list(self.label_ids())]
        return torch.log_softmax(logits, dim=-1).gather(1, labels.to(logits.device)[:, None])[:, 0].cpu()

    def score(
        self, state: FixedPointState, step: int, seeds: Sequence[int], sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        examples = self.step_examples(step)
        prompts = [e.prompt() for e in examples]
        labels = torch.tensor([e.label for e in examples])
        net = self.perturbable()
        net.load(state)
        repeat = len(examples)
        per_chunk = max(1, self.max_rows // (2 * repeat))
        scores = []
        for start in range(0, len(seeds), per_chunk):
            chunk = seeds[start : start + per_chunk]
            with net.population(chunk, sigma, repeat):
                rows = 2 * len(chunk)
                scores.append(self._label_logprobs(prompts * rows, labels.repeat(rows)))
        per_member = torch.cat(scores).double().numpy().reshape(2 * len(seeds), repeat).mean(axis=1)
        return per_member[0::2], per_member[1::2]

    def evaluate(self, state: FixedPointState, step: int) -> dict[str, float]:
        examples = self.validation_set()
        self.perturbable().load(state)
        right, logprob = 0, 0.0
        for start in range(0, len(examples), self.eval_batch):
            chunk = examples[start : start + self.eval_batch]
            labels = torch.tensor([e.label for e in chunk])
            lp = self._label_logprobs([e.prompt() for e in chunk], labels)
            right += int((lp > np.log(0.5)).sum())
            logprob += float(lp.sum())
        return {"accuracy": right / len(examples), "label_logprob": logprob / len(examples)}
