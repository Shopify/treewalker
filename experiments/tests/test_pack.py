"""pack: one verified file per table, the packed types, and what it refuses."""

import polars as pl
import pytest

from treewalker_exp import analysis, pack
from treewalker_exp import formats as fm

U32, U64 = pl.UInt32, pl.UInt64


def write_cell(root, cell, block=3, timed=(True, False)):
    """One finished cell in the runner's layout and types (2 groups, 1 row each)."""
    d = root / "cells" / cell.replace("/", "_2f")
    d.mkdir(parents=True)
    ids = {"suite": "s", "cell": cell}
    pl.DataFrame(
        {"sample_id": [0, 1], **ids, "method": ["treewalker", "lleaves"], "block": [block] * 2}
        | {"group": [0, 1], "rows": [1, 1], "ticks": [5_000_000_000, 7]},
        schema_overrides={"sample_id": U64, "block": U32, "group": U32, "rows": U32, "ticks": U64},
    ).write_parquet(d / "samples.parquet")
    timed_dtype = pl.Boolean if isinstance(timed[0], bool) else U32
    pl.DataFrame(
        {**ids, "group": [0, 1], "entity": [-1, 9], "first_row": [0, 1], "timed": list(timed)},
        schema_overrides={"group": U32, "first_row": U64, "timed": timed_dtype},
    ).write_parquet(d / "groups.parquet")
    pl.DataFrame(
        {**ids, "group": [0, 1], "leaf_hits": [4, 5]},
        schema_overrides={"group": U32, "leaf_hits": U64},
    ).write_parquet(d / "counters.parquet")
    pl.DataFrame(
        {**ids, "block": [block], "cycles": [None], "status": ["unsupported"]},
        schema_overrides={"block": U32, "cycles": U64},
    ).write_parquet(d / "hw.parquet")
    fm.write_json(d / "manifest.json", {"id": cell, "seconds": 1.0})


def write_run(tmp_path, **kw):
    fm.write_json(tmp_path / "run.json", {})
    write_cell(tmp_path, "b/cell", **kw)
    write_cell(tmp_path, "a/cell", **kw)
    write_cell(tmp_path / "sentinel" / "000", "a/cell", **kw)
    return tmp_path


def test_pack_writes_one_verified_file_per_table_in_the_packed_types(tmp_path):
    run = write_run(tmp_path)
    lines = pack.pack_run(run)
    assert len(lines) == 8 and all("verified" in line for line in lines)
    s = pl.read_parquet(run / "samples.parquet")
    # Cells in directory order, rows in their order; nothing narrowed that cannot be.
    assert s["cell"].cast(pl.String).to_list() == ["a/cell"] * 2 + ["b/cell"] * 2
    assert s["ticks"].to_list() == [5_000_000_000, 7] * 2
    assert dict(s.schema) == {
        "sample_id": U32,
        "suite": pl.Categorical,
        "cell": pl.Categorical,
        "method": pl.Categorical,
        "block": pl.UInt16,
        "group": U32,
        "rows": U32,
        "ticks": U64,
    }
    g = pl.read_parquet(run / "groups.parquet")
    assert g["timed"].dtype == pl.Boolean and g["entity"].to_list() == [-1, 9, -1, 9]
    assert pl.read_parquet(run / "hw.parquet")["cycles"].null_count() == 2
    assert list(fm.read_json(run / "cells.json")) == ["a/cell", "b/cell"]
    sentinel = pl.read_parquet(run / "sentinel" / "samples.parquet")
    assert sentinel.columns[0] == "sentinel" and sentinel["sentinel"].to_list() == [0, 0]
    assert fm.read_json(run / "sentinel" / "cells.json") == {
        "000": {"a/cell": {"id": "a/cell", "seconds": 1.0}}
    }
    assert (run / "cells").is_dir() and not (run / ".pack-tmp").exists()  # kept by default


def test_readers_need_a_packed_run_and_pack_refuses_one(tmp_path):
    run = write_run(tmp_path)
    with pytest.raises(ValueError, match="is not packed: treewalker-exp pack"):
        analysis.load(run)
    pack.pack_run(run)
    assert analysis.load(run)["manifests"]["b/cell"]["seconds"] == 1.0
    with pytest.raises(ValueError, match="already packed"):
        pack.pack_run(run)


def test_remove_deletes_the_cell_directories_after_the_pack(tmp_path):
    run = write_run(tmp_path)
    lines = pack.pack_run(run, remove=True)
    assert lines[-1].endswith("removed 2 cell and 1 sentinel directories")
    assert not (run / "cells").exists() and not (run / "sentinel" / "000").exists()
    assert sorted(p.name for p in (run / "sentinel").iterdir()) == [
        "cells.json",
        "counters.parquet",
        "groups.parquet",
        "hw.parquet",
        "samples.parquet",
    ]


@pytest.mark.parametrize(
    ("kw", "error"),
    [
        ({"block": 70_000}, "70000 not in range: 0 to 65535"),  # uint16
        ({"timed": (1, 2)}, "timed is a flag but 2 values exceed 1"),
    ],
)
def test_a_value_that_does_not_fit_fails_the_pack_and_keeps_the_cells(tmp_path, kw, error):
    run = write_run(tmp_path, **kw)
    with pytest.raises(Exception, match=error):
        pack.pack_run(run, remove=True)
    assert not pack.is_packed(run) and not (run / "samples.parquet").exists()
    assert not (run / ".pack-tmp").exists() and len(list((run / "cells").iterdir())) == 2


def test_a_uint32_flag_from_older_runs_packs_as_a_bool(tmp_path):
    run = write_run(tmp_path, timed=(1, 0))
    pack.pack_run(run)
    g = pl.read_parquet(run / "groups.parquet")
    assert g.filter(pl.col("cell") == "b/cell")["timed"].to_list() == [True, False]
