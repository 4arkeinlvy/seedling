"""M6: evolution strategies on a real LLM (Qwen2.5-0.5B-Instruct) solving Countdown.

Explained in docs/M6-explained.md.
"""

from seedling.llm.countdown import Problem, check, extract_answer, generate, reward
from seedling.llm.lowrank import LayerSpec, PerturbableModel, PopulationLinear, apply_lowrank, plan_layers
from seedling.llm.workload import CountdownWorkload

__all__ = [
    "CountdownWorkload",
    "LayerSpec",
    "PerturbableModel",
    "PopulationLinear",
    "Problem",
    "apply_lowrank",
    "check",
    "extract_answer",
    "generate",
    "plan_layers",
    "reward",
]
