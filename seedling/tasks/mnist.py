"""M3: MNIST digits with a small MLP, trained by ES and by backprop (same model code).

The four idx files (~11 MB) are downloaded once into ~/.cache/seedling/mnist and checked by MD5.

    run:  uv run python -m seedling.tasks.es_demo --task mnist
"""

from __future__ import annotations

import gzip
import hashlib
import os
import ssl
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from seedling.data import data_dir
from seedling.es import ES, Adam, BatchedMLP, evaluate_batched

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
MODEL = BatchedMLP([784, 32, 10], activation=torch.relu)
CA_BUNDLES = ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/cert.pem")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _tls_context() -> ssl.SSLContext:
    """Verified TLS. Falls back to common CA bundle paths (NixOS has no /etc/ssl/cert.pem)."""
    if os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context()
    bundle = next((path for path in CA_BUNDLES if Path(path).exists()), None)
    return ssl.create_default_context(cafile=bundle)


def download(root: Path | None = None) -> Path:
    root = root or data_dir() / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    context = _tls_context()
    for name, md5 in FILES.items():
        target = root / name
        if target.exists() and _md5(target) == md5:
            continue
        errors = []
        for mirror in MIRRORS:
            try:
                with urllib.request.urlopen(mirror + name, timeout=60, context=context) as response:
                    partial = target.with_suffix(".part")
                    partial.write_bytes(response.read())
                if _md5(partial) != md5:
                    raise ValueError("checksum mismatch")
                partial.replace(target)
                break
            except (OSError, ValueError) as e:
                errors.append(f"{mirror}: {e}")
        else:
            raise ConnectionError(f"could not download {name}: " + "; ".join(errors))
    return root


def read_idx(path: Path) -> np.ndarray:
    """The idx format: 2 zero bytes, a type code (8 = uint8), the number of dims, then sizes."""
    with gzip.open(path, "rb") as f:
        _, type_code, ndim = struct.unpack(">HBB", f.read(4))
        if type_code != 0x08:
            raise ValueError(f"{path}: unsupported idx type {type_code:#x}")
        shape = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


@dataclass
class MNIST:
    train_x: torch.Tensor  # (60000, 784) uint8
    train_y: torch.Tensor
    test_x: torch.Tensor  # (10000, 784) uint8
    test_y: torch.Tensor

    def batch(self, size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(len(self.train_y), (size,), generator=generator)
        return self.train_x[idx].float() / 255, self.train_y[idx]


def load(root: Path | None = None) -> MNIST:
    root = download(root)

    def images(name: str) -> torch.Tensor:
        return torch.from_numpy(read_idx(root / name).reshape(-1, 784).copy())

    def labels(name: str) -> torch.Tensor:
        return torch.from_numpy(read_idx(root / name).astype(np.int64))

    return MNIST(
        images("train-images-idx3-ubyte.gz"),
        labels("train-labels-idx1-ubyte.gz"),
        images("t10k-images-idx3-ubyte.gz"),
        labels("t10k-labels-idx1-ubyte.gz"),
    )


def fitness(thetas: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minus the cross-entropy of each of the P models on the same batch: shape (P,)."""
    logits = MODEL(thetas, x)
    targets = y.expand(thetas.shape[0], -1)
    return -F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").mean(dim=1)


@torch.no_grad()
def accuracy(theta: torch.Tensor, x_uint8: torch.Tensor, y: torch.Tensor) -> float:
    logits = MODEL(theta[None].to(x_uint8.device), x_uint8.float() / 255)[0]
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_es(
    data: MNIST,
    steps: int = 300,
    pop: int = 50,
    sigma: float = 0.05,
    lr: float = 0.02,
    batch: int = 256,
    seed: int = 0,
    eval_every: int = 50,
    device: str = "cpu",
) -> tuple[ES, list[dict]]:
    generator = torch.Generator().manual_seed(seed)
    es = ES(MODEL.init(generator), Adam(lr=lr), pop=pop, sigma=sigma, seed=seed)
    test_x, test_y = data.test_x.to(device), data.test_y.to(device)
    history = []
    for step in range(1, steps + 1):
        x, y = data.batch(batch, generator)
        x, y = x.to(device), y.to(device)
        seeds = es.ask()
        f_plus, f_minus = evaluate_batched(lambda t: fitness(t.to(device), x, y), es.theta, seeds, sigma)
        es.tell(seeds, f_plus, f_minus)
        if step % eval_every == 0 or step == steps:
            history.append(
                {
                    "step": step,
                    "forward_passes": step * 2 * pop,
                    "train_loss": float(-np.mean(np.concatenate([f_plus, f_minus]))),
                    "test_accuracy": accuracy(es.theta, test_x, test_y),
                }
            )
    return es, history


def train_backprop(
    data: MNIST,
    steps: int = 300,
    lr: float = 1e-3,
    batch: int = 256,
    seed: int = 0,
    eval_every: int = 50,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[dict]]:
    """The same BatchedMLP (P = 1), trained with gradients from autograd and torch's Adam."""
    generator = torch.Generator().manual_seed(seed)
    theta = MODEL.init(generator).to(device).requires_grad_()
    opt = torch.optim.Adam([theta], lr=lr)
    test_x, test_y = data.test_x.to(device), data.test_y.to(device)
    history = []
    for step in range(1, steps + 1):
        x, y = data.batch(batch, generator)
        loss = -fitness(theta[None], x.to(device), y.to(device))[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            history.append(
                {
                    "step": step,
                    "forward_passes": step,
                    "train_loss": loss.item(),
                    "test_accuracy": accuracy(theta.detach(), test_x, test_y),
                }
            )
    return theta.detach(), history


class MnistTask:
    """MNIST as a Task: fitness = minus the cross-entropy on a minibatch chosen by the step."""

    name = "mnist"
    dim = MODEL.size

    def __init__(self, data: MNIST | None = None, batch: int = 256):
        self._data = data
        self.batch = batch

    @property
    def data(self) -> MNIST:
        if self._data is None:
            self._data = load()
        return self._data

    def init(self, seed: int) -> torch.Tensor:
        return MODEL.init(torch.Generator().manual_seed(seed))

    def fitness(self, thetas: torch.Tensor, step: int) -> torch.Tensor:
        x, y = self.data.batch(self.batch, torch.Generator().manual_seed(step))
        return fitness(thetas, x, y)

    def evaluate(self, theta: torch.Tensor, step: int) -> dict[str, float]:
        return {"test_accuracy": accuracy(theta, self.data.test_x, self.data.test_y)}
