"""Small shared helpers: devices, seeding, tensor conversion, atomic file writes."""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def resolve_device(device: str = "auto") -> str:
    """'auto' picks CUDA when available, otherwise CPU."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(a, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)


def to_numpy(a: torch.Tensor) -> np.ndarray:
    return a.detach().cpu().numpy()


@contextmanager
def timer(store: dict, key: str):
    start = time.time()
    yield
    store[key] = time.time() - start


def atomic_write_json(path: str | os.PathLike, obj) -> None:
    """Write JSON via a temporary file and rename, so parallel jobs never see partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        tmp = f.name
    os.replace(tmp, path)


def atomic_torch_save(obj, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as f:
        tmp = f.name
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")
