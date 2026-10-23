"""M10: audit generated text cheaply, and tolerate honest differences between devices.

After TOPLOC (INTELLECT-2, Prime Intellect 2025): a worker that generates text sends the
token ids with its scores ("evidence"). An auditor does not generate again; it runs ONE
teacher-forced forward pass over prompt + claimed tokens under the same perturbation and
checks that every claimed token is the greedy choice up to a tolerance:

    logit[claimed token] >= max logit - delta        at every position up to the first EOS

Then it grades the claimed text itself. So:
    an honest claim         agrees exactly, even when another GPU breaks a near-tie the other way
    a false score           the regraded text gives a different score: the vote catches it
    forged text             fails the check; only then does the auditor generate for itself

One forward pass over a completion costs about as much as one or two decoding steps, so an
audit costs a small fraction of the generation it checks.
"""

from __future__ import annotations

import base64
import zlib
from collections.abc import Sequence

import numpy as np
import torch


def pack(tokens: np.ndarray) -> str:
    """(rows, length) token ids -> compact text (zlib-compressed int32, base64)."""
    tokens = np.ascontiguousarray(tokens, dtype=np.int32)
    header = np.array(tokens.shape, dtype=np.int32).tobytes()
    return base64.b64encode(zlib.compress(header + tokens.tobytes(), 6)).decode()


def unpack(text: str, max_tokens: int = 1 << 24) -> np.ndarray:
    """Inverse of pack. Raises ValueError on anything malformed (the text comes from a peer)."""
    try:
        raw = zlib.decompress(base64.b64decode(text, validate=True), bufsize=1 << 16)
    except (ValueError, zlib.error) as e:
        raise ValueError(f"bad evidence: {e}") from None
    if len(raw) < 8:
        raise ValueError("bad evidence: too short")
    rows, length = (int(v) for v in np.frombuffer(raw[:8], dtype=np.int32))
    if rows < 0 or length < 0 or rows * length > max_tokens or len(raw) != 8 + 4 * rows * length:
        raise ValueError("bad evidence: shape does not match its data")
    return np.frombuffer(raw[8:], dtype=np.int32).reshape(rows, length).copy()


def canonical(tokens: np.ndarray, eos: Sequence[int], pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Tokens after each row's first EOS become `pad`. Returns (tokens, valid): valid marks the
    positions up to and including the first EOS, the only ones a check or a grade looks at."""
    tokens = tokens.copy()
    is_eos = np.isin(tokens, np.asarray(list(eos), dtype=tokens.dtype))
    first = np.where(is_eos.any(axis=1), is_eos.argmax(axis=1), tokens.shape[1])
    valid = np.arange(tokens.shape[1])[None, :] <= first[:, None]
    tokens[~valid] = pad
    return tokens, valid


def greedy_rows(
    logits: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor, delta: float
) -> torch.Tensor:
    """Per row: is every valid claimed token within delta of that position's top logit?

    logits (rows, L, vocab) predict tokens (rows, L); valid (rows, L) marks positions to check."""
    claimed = logits.gather(-1, tokens[..., None].long())[..., 0]
    gap = logits.max(dim=-1).values - claimed
    return ((gap <= delta) | ~valid).all(dim=1)
