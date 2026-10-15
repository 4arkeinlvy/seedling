from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def cuda():
    """The CUDA device. Skips on machines without an NVIDIA GPU, but FAILS when the GPU is
    there and PyTorch can't see it (the usual NixOS problem), so you don't miss it."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if Path("/dev/nvidia0").exists():
        pytest.fail(
            "This machine has an NVIDIA GPU, but PyTorch can't see it. On NixOS, run inside "
            "`nix-shell` (it sets LD_LIBRARY_PATH=/run/opengl-driver/lib)."
        )
    pytest.skip("no NVIDIA GPU on this machine")
