"""Pack a finished run: one Parquet file per table instead of one per cell.

The runner writes each cell's tables under ``cells/<cell>/``: a cell is renamed into
place when complete, and a resumed run skips the finished ones. Nothing needs that
layout once the run is over. ``pack_run`` writes, beside ``run.json``:

    samples.parquet  groups.parquet  counters.parquet  hw.parquet
    cells.json                 every cell's manifest, by cell ID
    sentinel/<table>.parquet   the sentinel's cells, with a leading `sentinel` column
    sentinel/cells.json        their manifests, by attempt and cell ID

Cells keep their directory order and rows their original order. Every packed table
is read back, cast to the runner's types and compared with its cells before anything
moves into place; ``remove`` then deletes the per-cell directories. The readers
(summarize, validation-readout, budget) read packed runs only.

Types follow what a column can hold, not the largest value seen: strings are
dictionary-typed (polars reads them as Categorical); small structural indices are
uint16; indices and counts bounded by a cell's groups or a dataset's rows are
uint32; ticks, hardware events and TreeWalker's counters stay uint64, since they can
pass 2**32 (the final run's ticks reached 3.0e9); ``timed`` is a bool and ``entity``
keeps the source's int64. Columns not named keep the runner's type. Narrowing casts
are checked, so a value that does not fit fails the pack.

Encodings, from a 10% sample of the final factorial: dictionary for strings and small
categorical integers, DELTA_BINARY_PACKED for indices that increase within a cell,
BYTE_STREAM_SPLIT for every other integer, zstd level 9 (level 15 saves about 5% more
on samples). Against the runner's per-cell files (zstd 3, dictionary everywhere) the
final run packed to 0.38x on both machines.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from . import formats as fm

TABLES = ("samples", "groups", "counters", "hw")
CELLS = "cells.json"
SENTINEL = "sentinel"
LEVEL = 9
ROW_GROUP = 1 << 20
_TMP = ".pack-tmp"

STRINGS = frozenset({"suite", "cell", "variant", "method", "mode", "interface", "status"})
_SMALL = dict.fromkeys(("block", "batch", "position", "repetition", "process"), "uint16")
TYPES: dict[str, dict[str, str]] = {
    "samples": {
        **_SMALL,
        "sentinel": "uint32",
        "sample_id": "uint32",
        "group": "uint32",
        "n_groups": "uint32",
        "rows": "uint32",
        "ticks": "uint64",
    },
    "groups": {
        "sentinel": "uint32",
        "group": "uint32",
        "entity": "int64",
        "first_row": "uint32",
        "rows": "uint32",
        "timed": "bool_",
    },
    "counters": {"sentinel": "uint32", "group": "uint32", "counters_version": "uint32"},
    "hw": {**_SMALL, "sentinel": "uint32"},
}
NOT_NULL = frozenset({"sentinel"})
DELTA = {
    "samples": {"sample_id"},
    "groups": {"group", "first_row"},
    "counters": {"group"},
    "hw": set(),
}
DICTIONARY_INTS = frozenset({*_SMALL, "n_groups", "counters_version", "sentinel"})


def is_packed(run_dir: Path) -> bool:
    """cells.json moves into place last, so it marks a complete pack."""
    return (run_dir / CELLS).exists()


def require_packed(run_dir: Path) -> None:
    if not is_packed(run_dir):
        raise ValueError(f"{run_dir} is not packed: treewalker-exp pack {run_dir}")


def target_schema(source: Any, table: str) -> Any:
    """The packed schema for a table whose cells have the arrow schema ``source``."""
    import pyarrow as pa

    fields = []
    for f in source:
        if f.name in STRINGS:
            typ = pa.dictionary(pa.int32(), pa.string())
        elif f.name in TYPES[table]:
            typ = getattr(pa, TYPES[table][f.name])()
        else:
            typ = f.type
        fields.append(pa.field(f.name, typ, f.nullable and f.name not in NOT_NULL))
    return pa.schema(fields)


def writer_options(schema: Any, table: str) -> dict[str, Any]:
    import pyarrow as pa

    dictionary, encoding = [], {}
    for f in schema:
        if pa.types.is_boolean(f.type):
            continue  # Parquet bit-packs booleans
        if not pa.types.is_integer(f.type) or f.name in DICTIONARY_INTS:
            dictionary.append(f.name)
        elif f.name in DELTA[table]:
            encoding[f.name] = "DELTA_BINARY_PACKED"
        else:
            encoding[f.name] = "BYTE_STREAM_SPLIT"
    return {"use_dictionary": dictionary, "column_encoding": encoding}


def _cells(root: Path) -> list[Path]:
    """Finished cells, in name order; cells being written have names starting with '.'."""
    d = root / "cells"
    if not d.is_dir():
        return []
    cells = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
    for c in cells:
        names = ("manifest.json", *(f"{t}.parquet" for t in TABLES))
        missing = [n for n in names if not (c / n).exists()]
        if missing:
            raise ValueError(f"{c} is not a finished cell: no {', '.join(missing)}")
    return cells


def _attempts(run_dir: Path) -> list[Path]:
    d = run_dir / SENTINEL
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir() and p.name.isdigit())


def _read(cell: Path, table: str, attempt: int | None) -> Any:
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pq.read_table(cell / f"{table}.parquet")
    if attempt is None:
        return t
    return t.add_column(0, "sentinel", pa.array([attempt] * t.num_rows, pa.uint32()))


def _to_target(t: Any, target: Any, where: Path) -> Any:
    """Cast one row group to the packed types; strings get one dictionary per row group."""
    import pyarrow as pa
    import pyarrow.compute as pc

    cols = []
    for f in target:
        c = t[f.name]
        if not f.nullable and c.null_count:
            raise ValueError(f"{where}: {f.name} has {c.null_count} nulls")
        if pa.types.is_boolean(f.type) and not pa.types.is_boolean(c.type):
            over = pc.sum(pc.greater(c, 1)).as_py() or 0
            if over:
                raise ValueError(f"{where}: {f.name} is a flag but {over} values exceed 1")
        if pa.types.is_dictionary(f.type):
            cols.append(c.combine_chunks().dictionary_encode())
        else:
            cols.append(c.cast(f.type))  # a safe cast: a value that does not fit fails
    return pa.Table.from_arrays(cols, schema=target)


def _write(sources: list[tuple[int | None, Path]], table: str, dst: Path, level: int) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    first = _read(sources[0][1], table, sources[0][0])
    target = target_schema(first.schema, table)
    rows = 0
    with pq.ParquetWriter(
        dst, target, compression="zstd", compression_level=level, **writer_options(target, table)
    ) as w:
        buf, n = [], 0
        for i, (attempt, cell) in enumerate(sources):
            t = first if i == 0 else _read(cell, table, attempt)
            if t.schema != first.schema:
                raise ValueError(f"{cell}/{table}.parquet: schema differs from {sources[0][1]}")
            buf.append(t)
            n += t.num_rows
            rows += t.num_rows
            if n >= ROW_GROUP:
                big = pa.concat_tables(buf)
                k = n // ROW_GROUP * ROW_GROUP
                w.write_table(_to_target(big.slice(0, k), target, dst), row_group_size=ROW_GROUP)
                buf, n = [big.slice(k)], n - k
        if n:
            w.write_table(_to_target(pa.concat_tables(buf), target, dst), row_group_size=ROW_GROUP)
    return rows


def _verify(sources: list[tuple[int | None, Path]], table: str, dst: Path) -> None:
    """The packed table, cast back to the runner's types, equals its cells in order."""
    import pyarrow.parquet as pq

    packed = pq.read_table(dst)
    first = _read(sources[0][1], table, sources[0][0])
    if not packed.schema.remove_metadata().equals(target_schema(first.schema, table)):
        raise ValueError(f"{dst}: unexpected schema\n{packed.schema}")
    off = 0
    for attempt, cell in sources:
        src = _read(cell, table, attempt)
        if not packed.slice(off, src.num_rows).cast(src.schema).equals(src):
            raise ValueError(f"{dst}: rows {off}..{off + src.num_rows} differ from {cell}")
        off += src.num_rows
    if off != packed.num_rows:
        raise ValueError(f"{dst}: {packed.num_rows} rows, its cells {off}")


def pack_run(run_dir: Path, remove: bool = False, level: int = LEVEL) -> list[str]:
    """Pack a finished run in place; one report line per table written."""
    run_dir = Path(run_dir)
    if is_packed(run_dir) or any((run_dir / f"{t}.parquet").exists() for t in TABLES):
        raise ValueError(f"{run_dir} is already packed")
    cells, attempts = _cells(run_dir), _attempts(run_dir)
    if not cells:
        raise ValueError(f"{run_dir} has no finished cells")
    groups: dict[str, list[tuple[int | None, Path]]] = {"": [(None, c) for c in cells]}
    groups[SENTINEL] = [(int(a.name), c) for a in attempts for c in _cells(a)]

    tmp = run_dir / _TMP
    shutil.rmtree(tmp, ignore_errors=True)  # an interrupted pack's
    lines, moves = [], []
    try:
        for sub, sources in groups.items():
            if not sources:
                continue
            (tmp / sub).mkdir(parents=True, exist_ok=True)
            for table in TABLES:
                t0 = time.monotonic()
                dst = tmp / sub / f"{table}.parquet"
                rows = _write(sources, table, dst, level)
                _verify(sources, table, dst)
                before = sum((c / f"{table}.parquet").stat().st_size for _, c in sources)
                after = dst.stat().st_size
                moves.append(Path(sub) / dst.name)
                lines.append(
                    f"{run_dir.name} {sub + '/' if sub else ''}{table}: {len(sources)} cells, "
                    f"{rows:,} rows, {before / 1e6:.1f} -> {after / 1e6:.1f} MB "
                    f"({after / max(before, 1):.2f}x), verified, {time.monotonic() - t0:.1f} s"
                )
        if groups[SENTINEL]:
            by_attempt: dict[str, dict[str, Any]] = {}
            for attempt, cell in groups[SENTINEL]:
                m = fm.read_json(cell / "manifest.json")
                by_attempt.setdefault(f"{attempt:03d}", {})[m["id"]] = m
            fm.write_json(tmp / SENTINEL / CELLS, by_attempt)
            moves.append(Path(SENTINEL) / CELLS)
        manifests: dict[str, Any] = {}
        for c in cells:
            m = fm.read_json(c / "manifest.json")
            if m["id"] in manifests:
                raise ValueError(f"{c}: cell {m['id']} appears twice")
            manifests[m["id"]] = m
        fm.write_json(tmp / CELLS, manifests)
        if fm.read_json(tmp / CELLS) != manifests:
            raise ValueError(f"{tmp / CELLS} does not read back")
        moves.append(Path(CELLS))  # last: it marks the pack complete
        for rel in moves:
            os.replace(tmp / rel, run_dir / rel)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if remove:
        shutil.rmtree(run_dir / "cells")
        for a in attempts:
            shutil.rmtree(a)
        lines.append(
            f"{run_dir.name}: removed {len(cells)} cell and {len(attempts)} sentinel directories"
        )
    return lines
