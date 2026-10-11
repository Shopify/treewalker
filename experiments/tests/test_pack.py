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
    assert lines[-1].endswith("removed 3 packed cell directories")
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


def test_readers_name_a_git_lfs_pointer(tmp_path):
    run = write_run(tmp_path)
    pack.pack_run(run)
    (run / "samples.parquet").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n"
    )
    with pytest.raises(ValueError, match="is a git-lfs pointer: run `git lfs pull`"):
        analysis.load(run)


def write_stages(cell_dir, cell, rows=(3, 4)):
    """The runner's stages table: f64 stages on the oracle's rows."""
    n = len(rows)
    pl.DataFrame(
        {"suite": ["s"] * n, "cell": [cell] * n, "group": [1] * n, "row": list(rows)}
        | {"tree_sum": [0.1 * r for r in rows], "raw_margin": [0.2] * n, "output": [0.5] * n},
        schema_overrides={"group": U32, "row": U64},
    ).write_parquet(cell_dir / "stages.parquet")


def test_stages_pack_when_every_cell_has_them(tmp_path):
    import pyarrow.parquet as pq

    run = write_run(tmp_path)
    for d in [*(run / "cells").iterdir(), *(run / "sentinel" / "000" / "cells").iterdir()]:
        write_stages(d, fm.read_json(d / "manifest.json")["id"])
    lines = pack.pack_run(run)
    assert (
        len(lines) == 10 and sum(" stages: " in line or "/stages: " in line for line in lines) == 2
    )
    st = pl.read_parquet(run / "stages.parquet")
    assert st["cell"].cast(pl.String).to_list() == ["a/cell"] * 2 + ["b/cell"] * 2
    assert st["row"].dtype == U32 and st["tree_sum"].dtype == pl.Float64
    # The f64 stages survive the pack bit for bit.
    assert st["tree_sum"].to_list() == [0.1 * 3, 0.1 * 4] * 2
    meta = pq.ParquetFile(run / "stages.parquet").metadata.row_group(0)
    encodings = {meta.column(i).path_in_schema: meta.column(i).encodings for i in range(7)}
    assert "BYTE_STREAM_SPLIT" in encodings["raw_margin"]
    assert "DELTA_BINARY_PACKED" in encodings["row"]
    assert pl.read_parquet(run / "sentinel" / "stages.parquet")["sentinel"].to_list() == [0, 0]


def test_a_run_where_only_some_cells_have_stages_fails(tmp_path):
    run = write_run(tmp_path)
    write_stages(run / "cells" / "a_2fcell", "a/cell")
    with pytest.raises(ValueError, match=r"b_2fcell has no stages\.parquet, other cells do"):
        pack.pack_run(run)
    assert not pack.is_packed(run)


def test_remove_keeps_unfinished_cells_and_failed_attempts(tmp_path):
    run = write_run(tmp_path)
    (run / "cells" / ".tmp-c_2fcell").mkdir()  # a cell the runner was writing
    (run / "cells" / ".tmp-c_2fcell" / "samples.parquet").write_bytes(b"partial")
    failed = run / "sentinel" / "001" / "failed" / "a_2fcell.json"
    failed.parent.mkdir(parents=True)
    fm.write_json(failed, {"reason": "the stage oracle did not pass", "diagnostics": {}})
    (run / "failed").mkdir()
    fm.write_json(run / "failed" / "d_2fcell.json", {"reason": "refused"})
    lines = pack.pack_run(run, remove=True)
    assert lines[-1] == (
        f"{run.name}: removed 3 packed cell directories; kept 2 other files: "
        "cells/.tmp-c_2fcell/samples.parquet, sentinel/001/failed/a_2fcell.json"
    )
    # Only the packed cells went; the failed attempt, its diagnosis and the run's
    # failed/ stay, and the empty attempt 000 is gone.
    assert sorted(p.name for p in (run / "cells").iterdir()) == [".tmp-c_2fcell"]
    assert fm.read_json(failed)["reason"] == "the stage oracle did not pass"
    assert (run / "failed" / "d_2fcell.json").exists()
    assert not (run / "sentinel" / "000").exists()


def test_schema_6_requires_stages_in_the_cells_and_in_the_sentinels(tmp_path):
    for where in ("cells", "sentinel"):
        run = write_run(tmp_path / where)
        fm.write_json(run / "run.json", {"output_schema": 6})
        # Every main cell has stages and no sentinel cell does, or the reverse.
        dirs = run / "cells" if where == "sentinel" else run / "sentinel" / "000" / "cells"
        for d in dirs.iterdir():
            write_stages(d, fm.read_json(d / "manifest.json")["id"])
        required = r"has no stages\.parquet, output schema 6 requires it"
        with pytest.raises(ValueError, match=required):
            pack.pack_run(run, remove=True)
        assert not pack.is_packed(run) and len(list((run / "cells").iterdir())) == 2


def test_older_schemas_pack_without_stages(tmp_path):
    for schema in (5, None):
        run = write_run(tmp_path / str(schema))
        fm.write_json(run / "run.json", {} if schema is None else {"output_schema": schema})
        pack.pack_run(run)
        assert pack.is_packed(run) and not (run / "stages.parquet").exists()


def test_a_run_directory_state_for_a_restart(tmp_path):
    runs = tmp_path / "runs"
    assert pack.run_state(runs / "nothing") == "absent"
    (runs / "camera-ready-x86_64").mkdir(parents=True)
    run = write_run(runs / "camera-ready-x86_64")
    # No invocation of the runner finished (run.json records none): resumed, never packed.
    assert pack.run_state(run) == "partial"
    fm.write_json(run / "run.json", {"invocations": [{"cells_done": [], "finished": 1}]})
    assert pack.run_state(run) == "raw"
    (run / "sentinel" / "001" / "failed").mkdir(parents=True)  # a failed attempt pack keeps
    (run / "cells" / ".tmp-unfinished").mkdir()  # an unfinished cell pack keeps
    pack.pack_run(run, remove=True)
    assert pack.run_state(run) == "packed"
    write_cell(run, "c/later")  # raw cells of a later run beside the packed tables
    assert pack.run_state(run) == "mixed"
    half = runs / "half"
    half.mkdir()
    (half / "samples.parquet").write_bytes(b"x")  # tables without the cells.json
    assert pack.run_state(half) == "mixed"


def test_run_and_retime_run_refuse_a_packed_run_directory(tmp_path, monkeypatch):
    import shutil

    import typer

    from treewalker_exp import cli
    from treewalker_exp.paths import Paths, find_repo

    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    shutil.copy(find_repo() / "experiments" / "grids.toml", repo / "experiments" / "grids.toml")
    monkeypatch.setattr(cli, "_paths", lambda: Paths(repo))
    runs = repo / "experiments" / "data" / "runs"
    for rid in ("camera-ready-x86_64", "camera-ready-retime-r8-x86_64"):
        (runs / rid).mkdir(parents=True)
        (runs / rid / "cells.json").write_text("{}")
    with pytest.raises(typer.BadParameter, match="is packed"):
        cli.run(["camera-ready"], run_id="camera-ready-x86_64")
    with pytest.raises(typer.BadParameter, match="is packed"):
        cli.retime_run(tmp_path / "m.json", run_id="camera-ready-retime-r8-x86_64", parent="x")
