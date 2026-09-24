"""
Reproducibility and device selection.

Call set_seed() as the FIRST thing in every entry point, before any dataset,
model or DataLoader is created.

Scope, honestly: seeding makes runs repeatable on the same device with the same
library versions. It does not make a CUDA run match an MPS run. Every number
must therefore carry its device and version stamp (see runmeta.py). Do not
compare your numbers against the 18 committed logs, which were all produced on
Apple MPS.
"""
from __future__ import annotations

import os
import platform
import random
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int = 42, strict: bool = False) -> None:
    """Seed every RNG this pipeline touches.

    strict=True additionally requests deterministic kernels. Slower, and some
    ops have no deterministic implementation (warn_only=True keeps the run
    alive and tells you which op broke the guarantee). Use strict for the runs
    whose exact numbers you intend to report.

    CUBLAS_WORKSPACE_CONFIG only takes effect before CUDA initialises, which is
    why this has to be the first call in the program.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN autotuning picks different algorithms run to run.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn.

    AugmentedDataset calls np.random inside __getitem__. With num_workers > 0
    each worker inherits a copy of the parent RNG state, so without this the
    workers duplicate or diverge unreproducibly. Only needed when
    num_workers > 0.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """Generator for random_split and DataLoader(shuffle=True).

    Using an explicit generator means batch order and splits do not shift when
    augmentation consumes global RNG state.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    try:
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


def pick_device(prefer: Optional[str] = None) -> torch.device:
    """Resolve a device.

    'auto' (default) reproduces the original cuda -> mps -> cpu order.
    Naming a device explicitly errors instead of falling back silently,
    because a silent CPU fallback is how a bad wheel goes unnoticed for weeks.
    """
    if prefer in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but torch.cuda.is_available() is False. "
                f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}. "
                "A bare 'pip install torch' on Windows yields a CPU-only wheel."
            )
        return torch.device("cuda")

    if prefer == "mps":
        if not _mps_available():
            raise RuntimeError("device='mps' requested but MPS is unavailable.")
        return torch.device("mps")

    if prefer == "cpu":
        return torch.device("cpu")

    raise ValueError(f"unknown device {prefer!r}; use auto|cuda|mps|cpu")


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, cwd=_REPO_ROOT
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _pkg_version(name: str) -> Optional[str]:
    """Version via importlib.metadata.

    ramanspy 0.2.10 exposes no __version__ attribute, so reading rp.__version__
    raises AttributeError. Package metadata works for every installed dist.
    """
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        try:
            return getattr(__import__(name), "__version__", None)
        except Exception:
            return None


def describe_env(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Everything needed to decide whether two numbers are comparable."""
    dirty = _git("status", "--porcelain")
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "numpy": np.__version__,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
    }
    for pkg in ("ramanspy", "scikit-learn", "scipy", "pybaselines"):
        env[pkg] = _pkg_version(pkg)
    if device is not None:
        env["device"] = str(device)
        if device.type == "cuda":
            try:
                env["device_name"] = torch.cuda.get_device_name(device)
            except Exception:
                env["device_name"] = None
    return env
