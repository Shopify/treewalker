"""Content-addressed writes: each file is written unless it already holds the bytes, and
its hash comes from the bytes in memory, the same bytes the formats' writers produce."""

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from .. import formats as fm


def put_bytes(path: Path, data: bytes) -> str:
    """Write ``data`` unless the file already holds it; return its hash."""
    digest = hashlib.sha256(data).hexdigest()
    if not (path.exists() and path.stat().st_size == len(data) and fm.sha256_file(path) == digest):
        fm.write_bytes(path, data)
    return digest


def put_matrix(path: Path, X: np.ndarray) -> str:
    return put_bytes(path, fm.matrix_bytes(X))


def put_offsets(path: Path, offsets: np.ndarray) -> str:
    return put_bytes(path, fm.offsets_bytes(offsets))


def put_walker_config(path: Path, config: dict[str, Any]) -> str:
    return put_bytes(path, fm.walker_config_bytes(config))


def put_npy(path: Path, array: np.ndarray) -> str:
    return put_bytes(path, fm.npy_bytes(array))


def files_match(base: Path, files: dict[str, dict[str, str]]) -> bool:
    """Whether every recorded file under ``base`` exists with its recorded hash."""
    for rec in files.values():
        p = base / rec["path"]
        if not p.exists() or fm.sha256_file(p) != rec["sha256"]:
            return False
    return True


def rel(path: Path, start: Path) -> str:
    return os.path.relpath(path, start)
