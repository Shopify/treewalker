"""Content-addressed writes: each file is written unless it already holds the bytes, and
its hash comes from the bytes in memory, the same bytes the formats' writers produce."""

import hashlib
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .. import formats as fm


def matrix_bytes(X: np.ndarray) -> bytes:
    return struct.pack("<QQ", *X.shape) + np.ascontiguousarray(X, dtype="<f8").tobytes()


def offsets_bytes(offsets: np.ndarray) -> bytes:
    o = np.asarray(offsets, dtype="<u8")
    return struct.pack("<Q", len(o) - 1) + o.tobytes()


def put_bytes(path: Path, data: bytes) -> str:
    """Write ``data`` unless the file already holds it; return its hash."""
    digest = hashlib.sha256(data).hexdigest()
    if not (path.exists() and path.stat().st_size == len(data) and fm.sha256_file(path) == digest):
        fm.write_bytes(path, data)
    return digest


def put_matrix(path: Path, X: np.ndarray) -> str:
    return put_bytes(path, matrix_bytes(X))


def put_offsets(path: Path, offsets: np.ndarray) -> str:
    fm.check_offsets(np.asarray(offsets, dtype=np.uint64))
    return put_bytes(path, offsets_bytes(offsets))


def put_walker_config(path: Path, config: dict[str, Any]) -> str:
    import json

    return put_bytes(path, json.dumps(config, indent=2).encode())


def put_npy(path: Path, array: np.ndarray) -> str:
    import io

    buf = io.BytesIO()
    np.save(buf, array)
    return put_bytes(path, buf.getvalue())


def files_match(base: Path, files: dict[str, dict[str, str]]) -> bool:
    for rec in files.values():
        p = base / rec["path"]
        if not p.exists() or fm.sha256_file(p) != rec["sha256"]:
            return False
    return True


def rel(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(path, start)
