"""Countdown: use each given number once, with + - * /, to hit the target.

    numbers 3, 7, 25 and target 46  ->  7 * 3 + 25 = 46

Problems are generated so that a solution always exists. The answer is the last <answer> block,
or else the model's last line (small models often skip the tags). It is checked by walking the
expression's syntax tree: only numbers, + - * / and parentheses are allowed, and nothing is
ever executed with eval().
"""

from __future__ import annotations

import ast
import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
OPERATORS = {ast.Add: Fraction.__add__, ast.Sub: Fraction.__sub__, ast.Mult: Fraction.__mul__, ast.Div: Fraction.__truediv__}
CORRECT, WELL_FORMED, CLOSENESS, WRONG = 1.0, 0.1, 0.4, 0.0


@dataclass(frozen=True)
class Problem:
    numbers: tuple[int, ...]
    target: int

    def prompt(self) -> str:
        numbers = ", ".join(str(n) for n in self.numbers)
        return (
            f"Using the numbers [{numbers}], create an equation that equals {self.target}. "
            "Use +, -, * and /, and use each number exactly once. "
            "Think briefly, then give only the equation inside <answer> </answer> tags, "
            "for example <answer> (1 + 2) * 3 </answer>."
        )


def generate(count: int, seed: int, size: int = 3, low: int = 1, high: int = 20) -> list[Problem]:
    """Random solvable problems: pick numbers, build a random expression, use its value."""
    rng = random.Random(seed)
    problems: list[Problem] = []
    while len(problems) < count:
        numbers = [rng.randint(low, high) for _ in range(size)]
        value = Fraction(numbers[0])
        for n in numbers[1:]:
            op = rng.choice([ast.Add, ast.Sub, ast.Mult])
            value = OPERATORS[op](value, Fraction(n))
        if value.denominator == 1 and 0 < value <= 400:
            shuffled = numbers[:]
            rng.shuffle(shuffled)
            problems.append(Problem(tuple(shuffled), int(value)))
    return problems


def extract_answer(text: str) -> str | None:
    """The last <answer> block; without tags, the last line (left of any "=")."""
    found = ANSWER.findall(text)
    if found:
        return found[-1].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1].split("=")[0].strip() if lines else None


def _evaluate(node: ast.AST, used: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, used)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.left, used), _evaluate(node.right, used))
    raise ValueError(f"not allowed: {type(node).__name__}")


def parse(expression: str) -> tuple[Fraction, list[int]] | None:
    """(exact value, numbers used), or None if it is not a plain arithmetic expression."""
    used: list[int] = []
    try:
        return _evaluate(ast.parse(expression, mode="eval"), used), used
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return None


def check(expression: str, problem: Problem) -> tuple[bool, bool]:
    """(well_formed, correct). Well-formed: parses and uses exactly the given numbers."""
    parsed = parse(expression)
    if parsed is None:
        return False, False
    value, used = parsed
    well_formed = Counter(used) == Counter(problem.numbers)
    return well_formed, well_formed and value == problem.target


def reward(text: str, problem: Problem) -> float:
    """1 if correct. A well-formed miss earns 0.1 plus up to 0.4 for being close to the target,
    so the search gets a signal before it finds exact answers. Anything else earns 0."""
    expression = extract_answer(text)
    parsed = parse(expression) if expression else None
    if parsed is None or Counter(parsed[1]) != Counter(problem.numbers):
        return WRONG
    if parsed[0] == problem.target:
        return CORRECT
    closeness = max(0.0, 1.0 - abs(float(parsed[0]) - problem.target) / problem.target)
    return WELL_FORMED + CLOSENESS * closeness


# ---------------------------------------------------------------- TinyZero protocol (Qiu et al. 2025)
#
# Qiu et al., "Evolution Strategies at Scale" (2025), fine-tune on the TinyZero Countdown set
# (3-4 numbers) with TinyZero's prompt and score. Using the same data, prompt and score makes
# our accuracies comparable with theirs.

TINYZERO = "Jiayi-Pan/Countdown-Tasks-3to4"
TINYZERO_SYSTEM = (
    "You are a helpful assistant. You first thinks about the reasoning process in the mind "
    "and then provides the user with the answer."
)
TINYZERO_PREFIX = "Let me solve this step by step.\n<think>"


def tinyzero_prompt(problem: Problem) -> str:
    return (
        f"Using the numbers {list(problem.numbers)}, create an equation that equals {problem.target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, "
        "for example <answer> (1 + 2) / 3 </answer>."
    )


def tinyzero_reward(text: str, problem: Problem) -> float:
    """TinyZero's score: 0 without an <answer> block, 0.1 for any other answer, 1 if correct."""
    found = ANSWER.findall(text)
    if not found:
        return WRONG
    parsed = parse(found[-1].strip())
    if parsed is not None and Counter(parsed[1]) == Counter(problem.numbers) and parsed[0] == problem.target:
        return CORRECT
    return WELL_FORMED


def tinyzero_rows(rows: dict[str, list]) -> list[Problem]:
    """Problems from the dataset's columns ("nums", "target")."""
    return [Problem(tuple(int(n) for n in nums), int(t)) for nums, t in zip(rows["nums"], rows["target"], strict=True)]


def load_tinyzero() -> list[Problem]:
    """The whole TinyZero Countdown set (~490k problems), cached under seedling's data directory."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    from seedling.data import data_dir

    files = sorted(f for f in HfApi().list_repo_files(TINYZERO, repo_type="dataset") if f.endswith(".parquet"))
    problems: list[Problem] = []
    for name in files:
        path = hf_hub_download(TINYZERO, name, repo_type="dataset", cache_dir=data_dir() / "hf")
        problems += tinyzero_rows(pq.read_table(path, columns=["nums", "target"]).to_pydict())
    return problems
