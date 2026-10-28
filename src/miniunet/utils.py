"""Small helpers shared by the command-line tools: config loading, seeding, device."""

from __future__ import annotations

import random

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    """Read a YAML config file into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    """Seed ``random``, NumPy and PyTorch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    """Return ``name`` if given, else CUDA when available, else CPU."""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
