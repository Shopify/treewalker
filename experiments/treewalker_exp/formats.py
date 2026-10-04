"""The artifact file formats: one reader and one writer each.

- ``test_data.bin`` (also ``train_data.bin`` and ``reference.bin``): u64 rows,
  u64 columns, then row-major little-endian f64. ``benchmarks/src/data.rs``
  reads the same layout.
- ``group_offsets.bin``: u64 group count, then count + 1 little-endian u64
  cumulative row offsets, starting at 0.
- ``walker_config.json``: TreeWalker's feature classification.
- ``cell.json``, ``model.json`` and the execution manifests: JSON documents.
- ``predictions.npy`` and ``entities.npy``: NumPy arrays.

Every writer replaces its file atomically, so an interrupted run never leaves
a partial file behind under the final name.
"""

import hashlib
import json
import os
import struct
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import numpy as np


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


# mkstemp creates files readable by the owner only; files get the mode a
# plain open() would give them.
_FILE_MODE = 0o666 & ~_umask()


@contextmanager
def _atomic(path: Path) -> Iterator[IO[bytes]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
        os.chmod(tmp, _FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_bytes(path: Path, data: bytes) -> None:
    with _atomic(path) as f:
        f.write(data)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash of a JSON value in canonical form (sorted keys, no whitespace)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


# --- test_data.bin -----------------------------------------------------------


def write_matrix(path: Path, data: np.ndarray) -> None:
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {data.shape}")
    n_rows, n_cols = data.shape
    with _atomic(path) as f:
        f.write(struct.pack("<QQ", n_rows, n_cols))
        f.write(np.ascontiguousarray(data, dtype="<f8").tobytes())


def read_matrix(path: Path) -> np.ndarray:
    """Read a matrix as a read-only memory map; copy it to modify it."""
    with open(path, "rb") as f:
        n_rows, n_cols = struct.unpack("<QQ", f.read(16))
    expected = 16 + 8 * n_rows * n_cols
    if (size := path.stat().st_size) != expected:
        raise ValueError(f"{path}: {size} bytes, header says {expected}")
    if n_rows * n_cols == 0:
        return np.zeros((n_rows, n_cols), dtype="<f8")
    return np.memmap(path, dtype="<f8", mode="r", offset=16, shape=(n_rows, n_cols))


# --- group_offsets.bin ---------------------------------------------------------


def check_offsets(offsets: np.ndarray, n_rows: int | None = None) -> None:
    if len(offsets) < 1 or offsets[0] != 0:
        raise ValueError("group offsets must start at 0")
    if np.any(np.diff(offsets.astype(np.int64)) <= 0):
        raise ValueError("group offsets must strictly increase")
    if n_rows is not None and int(offsets[-1]) != n_rows:
        raise ValueError(f"group offsets end at {int(offsets[-1])}, data has {n_rows} rows")


def write_group_offsets(path: Path, offsets: np.ndarray) -> None:
    offsets = np.asarray(offsets, dtype=np.uint64)
    check_offsets(offsets)
    with _atomic(path) as f:
        f.write(struct.pack("<Q", len(offsets) - 1))
        f.write(offsets.astype("<u8").tobytes())


def read_group_offsets(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    (n_groups,) = struct.unpack("<Q", raw[:8])
    offsets = np.frombuffer(raw[8:], dtype="<u8")
    if len(offsets) != n_groups + 1:
        raise ValueError(f"{path}: header says {n_groups} groups, found {len(offsets) - 1}")
    return offsets


# --- JSON documents ------------------------------------------------------------


def write_walker_config(path: Path, config: dict[str, Any]) -> None:
    # No trailing newline: the released artifacts were written this way.
    write_bytes(path, json.dumps(config, indent=2).encode())


def read_walker_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


# cell.json and model.json share the generic JSON writer; the names say which
# document a call site handles.
write_cell = write_json
read_cell = read_json


# --- NumPy arrays --------------------------------------------------------------


def write_npy(path: Path, array: np.ndarray) -> None:
    with _atomic(path) as f:
        np.save(f, array)


def read_npy(path: Path) -> np.ndarray:
    return np.load(path)


write_predictions: Callable[[Path, np.ndarray], None] = write_npy
read_predictions: Callable[[Path], np.ndarray] = read_npy
