"""Low-rank evolution strategies for a big model (EGGROLL-style), exact like M4.

Only some Linear layers train. For seed i and layer l the noise is rank 1:

    E_il = b a^T,   a = sign(seed_i, stream 2l, in),  b = sign(seed_i, stream 2l+1, out)

so a whole population fits in ONE batched forward pass: batch row r belongs to member m and
computes   x (W + sigma s_m b_m a_m^T)^T = x W^T + sigma s_m (x . a_m) b_m.

The trainable state is one flat int32 array holding every selected layer's Delta (out x in);
the effective weight is W_base + Delta * 2**-frac_bits.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from seedling import _c
from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState


@dataclass(frozen=True)
class LayerSpec:
    name: str
    index: int
    rows: int
    cols: int
    offset: int

    @property
    def size(self) -> int:
        return self.rows * self.cols

    @property
    def streams(self) -> tuple[int, int]:
        return 2 * self.index, 2 * self.index + 1


def plan_layers(shapes: Sequence[tuple[str, int, int]]) -> list[LayerSpec]:
    """Give each (name, out, in) layer its place in the flat state and its noise streams."""
    specs, offset = [], 0
    for index, (name, rows, cols) in enumerate(shapes):
        specs.append(LayerSpec(name, index, rows, cols, offset))
        offset += rows * cols
    return specs


def delta_view(state: FixedPointState, spec: LayerSpec) -> np.ndarray:
    return state.values[spec.offset : spec.offset + spec.size].reshape(spec.rows, spec.cols)


def apply_lowrank(state: FixedPointState, record: StepRecord, specs: Sequence[LayerSpec]) -> None:
    seeds = np.asarray(record.seeds, dtype=np.uint64)
    coefs = np.asarray(record.coefs, dtype=np.int32)
    for spec in specs:
        _c.apply_lowrank_i32(delta_view(state, spec), seeds, coefs, *spec.streams)


def signs(seed: int, stream: int, n: int) -> torch.Tensor:
    return torch.from_numpy(_c.rademacher_i8(seed, n, stream).astype(np.float32))


class PopulationLinear(nn.Module):
    """nn.Linear whose batch rows can each carry their own rank-1 perturbation."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.a: torch.Tensor | None = None
        self.b: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def set_population(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> None:
        """a (rows, in), b (rows, out), scale (rows,): row r adds scale_r * (x . a_r) * b_r."""
        self.a, self.b, self.scale = a, b, scale

    def clear_population(self) -> None:
        self.a = self.b = self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.a is None:
            return y
        projection = torch.einsum("rti,ri->rt", x, self.a) * self.scale[:, None]
        return y + projection[..., None] * self.b[:, None, :]


def _swap_module(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent_name) if parent_name else model, child, new)


class PerturbableModel:
    """A model whose selected Linear layers follow an exact int32 state and take populations."""

    def __init__(self, model: nn.Module, specs: Sequence[LayerSpec]):
        self.model = model
        self.specs = list(specs)
        self.layers: list[PopulationLinear] = []
        self._base: list[torch.Tensor] = []
        for spec in self.specs:
            linear = model.get_submodule(spec.name)
            if not isinstance(linear, nn.Linear) or tuple(linear.weight.shape) != (spec.rows, spec.cols):
                raise ValueError(f"{spec.name} is not a {spec.rows}x{spec.cols} Linear")
            wrapped = PopulationLinear(linear)
            _swap_module(model, spec.name, wrapped)
            self.layers.append(wrapped)
            self._base.append(linear.weight.detach().clone())
        self._loaded: str | None = None

    @property
    def device(self) -> torch.device:
        return self._base[0].device

    @torch.no_grad()
    def load(self, state: FixedPointState) -> None:
        """Set every selected weight to W_base + Delta * unit (skipped if already loaded)."""
        digest = state.digest()
        if digest == self._loaded:
            return
        for spec, layer, base in zip(self.specs, self.layers, self._base, strict=True):
            delta = torch.from_numpy(delta_view(state, spec)).to(self.device, torch.float32) * state.unit
            layer.base.weight.copy_((base.float() + delta).to(base.dtype))
        self._loaded = digest

    @contextmanager
    def population(self, seeds: Sequence[int], sigma: float, repeat: int) -> Iterator[None]:
        """Rows: for each seed, `repeat` rows with +sigma then `repeat` rows with -sigma."""
        signs_per_member = torch.tensor([1.0, -1.0] * len(seeds))
        scale = (sigma * signs_per_member).repeat_interleave(repeat)
        try:
            for spec, layer in zip(self.specs, self.layers, strict=True):
                dtype = layer.base.weight.dtype
                stream_a, stream_b = spec.streams
                a = torch.stack([signs(s, stream_a, spec.cols) for s in seeds])
                b = torch.stack([signs(s, stream_b, spec.rows) for s in seeds])
                rows = torch.arange(len(seeds)).repeat_interleave(2 * repeat)
                layer.set_population(
                    a[rows].to(self.device, dtype), b[rows].to(self.device, dtype), scale.to(self.device, dtype)
                )
            yield
        finally:
            for layer in self.layers:
                layer.clear_population()
