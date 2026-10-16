"""treewalker_exp.counters, and results' work table: a run's counters, of any
version, in the released stats CSVs' columns."""

from pathlib import Path

import polars as pl
import pytest

from treewalker_exp import counters, results

V2 = {
    "constant_steps": 100,
    "varying_splits": 20,
    "unsplit_skips": 5,
    "recursive_calls": 30,
    "leaf_hits": 40,
    "leaf_adds": 41,
    "partition_row_evals": 7,
    "scan_row_evals": 0,
    "scan_compares": 0,
    "scan_missing_checks": 0,
    "precompute_row_evals": 9,
    "precompute_sort_compares": 0,
    "precompute_sweep_compares": 0,
    "precompute_mask_writes": 0,
}
V3 = {
    "groups": 1,
    "single_rows": 1,
    "pieces": 0,
    "bins_computed": 1,
    "precompute_row_evals": 9,
    "precompute_sort_compares": 0,
    "precompute_sweep_compares": 0,
    "precompute_mask_writes": 0,
    "scan_rows": 8,
    "scan_evals": 7,
    "prefix_steps": 3,
    "walk_steps": 200,
    "lane_slots": 256,
    "passes": 1,
    "varying_events": 20,
    "splits": 15,
    "empty_sides": 0,
    "leaf_events": 40,
    "runs_written": 44,
    "arena_slots": 0,
}
CELL = "support/nt10_md8_h16/lightgbm/panel"


def write_run(tmp: Path, version: int, variants: list[str]) -> Path:
    """A packed run's counters and groups: one cell of a 10-tree model, two groups
    of 16 rows, each variant's counters the same in both groups."""
    if version == 2:
        values = V2
    else:
        values = {**V3, "walk_ns": 5}
        if version >= 6:
            values = {k: v for k, v in values.items() if k != "pieces"}
    rows = [
        {"suite": "s", "cell": CELL, "variant": v, "group": g, "counters_version": version} | values
        for v in variants
        for g in range(2)
    ]
    pl.DataFrame(rows).with_columns(pl.col("counters_version").cast(pl.UInt32)).write_parquet(
        tmp / "counters.parquet"
    )
    pl.DataFrame({"cell": [CELL] * 2, "group": [0, 1], "rows": [16, 16]}).write_parquet(
        tmp / "groups.parquet"
    )
    return tmp


def work(tmp: Path, version: int, variants: list[str]) -> dict[str, dict]:
    results._work.cache_clear()
    w = results._work(write_run(tmp, version, variants))
    return {r["variant"]: r for r in w.iter_rows(named=True)}


def test_version_2_keeps_its_names(tmp_path):
    rows = work(tmp_path, 2, ["all-on", "disable_varying_precompute|disable_tree_ordering"])
    r = rows["all-on"]
    assert (r["groups"], r["rows"], r["T"], r["L"], r["horizon"]) == (2, 32, 10, 8, 16)
    assert {k: r[k] for k in counters.COUNTERS} == {k: 2 * V2[k] for k in counters.COUNTERS}
    assert counters.terms(2, V2, 10) == {k: V2[k] for k in counters.COUNTERS}


def test_version_3_maps_onto_the_csv_terms(tmp_path):
    rows = work(tmp_path, 3, ["all-on", "lanes=4", "disable_unsplit"])
    assert set(rows) == {"all-on", "lanes=4", "disable_unsplit"}
    r = rows["all-on"]
    # Two groups; leaves are 40 leaf events plus a single row through 10 trees.
    leaves = 2 * (40 + 10)
    assert r["leaf_hits"] == leaves
    assert r["constant_steps"] == 2 * (200 + 3 - 20) - leaves
    assert (r["varying_splits"], r["unsplit_skips"]) == (40, 10)
    assert (r["partition_row_evals"], r["precompute_row_evals"]) == (14, 18)
    assert r["recursive_calls"] is None
    # The group count is the groups', and the 3.x counters stay beside the terms.
    assert (r["groups"], r["walk_steps"]) == (2, 400)


def test_version_4_counts_single_rows_in_leaf_events(tmp_path):
    r = work(tmp_path, 4, ["all-on"])["all-on"]
    assert r["leaf_hits"] == 2 * 40
    assert r["constant_steps"] == 2 * (200 + 3 - 20) - 2 * 40


def test_versions_4_5_and_6_read_alike(tmp_path):
    # Version 5 changes only `pieces`, which no column reads, and version 6 drops it.
    terms = {}
    for version in (4, 5, 6):
        (tmp_path / str(version)).mkdir()
        r = work(tmp_path / str(version), version, ["all-on"])["all-on"]
        terms[version] = {k: r[k] for k in counters.COUNTERS}
    assert terms[4] == terms[5] == terms[6]


def test_unknown_versions_are_refused(tmp_path):
    with pytest.raises(ValueError, match="version 7"):
        work(tmp_path, 7, ["all-on"])
    with pytest.raises(ValueError, match="version 1"):
        counters.terms(1, V2, 10)
