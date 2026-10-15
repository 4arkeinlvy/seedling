"""M4: the bit-exact core: Philox noise, int32 fixed-point state, exact replay, xxHash64 digests.

Explained in docs/M4-explained.md.
"""

from seedling.exact.integer_es import IntegerES, integer_coefficients, rademacher_noise, replay
from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.exact.train import Run, train_integer_es

__all__ = [
    "FixedPointState",
    "IntegerES",
    "Run",
    "StepRecord",
    "integer_coefficients",
    "rademacher_noise",
    "replay",
    "train_integer_es",
]
