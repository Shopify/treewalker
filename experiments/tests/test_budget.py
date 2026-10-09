"""The budget on empty, partial and full pools."""

from pathlib import Path

import pytest

from treewalker_exp import budget as bg
from treewalker_exp import formats as fm
from treewalker_exp import grids
from treewalker_exp.paths import Paths

GRIDS = Path(__file__).resolve().parents[1] / "grids.toml"


@pytest.fixture
def setup(tmp_path):
    doc = grids.load(GRIDS)
    return Paths(tmp_path, tmp_path / "artifacts"), doc


def prepare(paths, cell, rows=160, groups=10):
    fm.write_cell(
        cell.dir(paths.artifacts) / "cell.json",
        {"grouping": {"rows": rows, "n_groups": groups}, "feature_order": ["a", "b"]},
    )


def test_a_missing_size_is_an_error_naming_the_cell_and_no_total(setup):
    paths, doc = setup
    s = grids.suite(doc, "acceptance")
    support = [c for c in s.cells if c.model.dataset == "support"]
    prepare(paths, support[0])
    b = bg.estimate(paths, doc, "acceptance", bg.Pilot(10.0, 1e-9))
    assert b.models == 4  # two SUPPORT and two credit models; fixtures have none
    assert len(b.missing) == b.cells - 1 and support[0].id not in b.missing
    assert b.hours is None and b.sample_bytes is None
    text = "\n".join(bg.describe(b, 3))
    assert f"ERROR: {b.cells - 1} without a prepared size, so no total" in text
    assert f"{b.cells} cells: 1 of {b.cells} sized from their cell.json" in text
    assert "wall time" not in text and b.missing[0] in text


def test_wall_time_scales_with_rows_trees_and_depth_without_halving(setup):
    paths, doc = setup
    s = grids.suite(doc, "acceptance")
    for c in s.cells:
        prepare(paths, c)
    pilot = bg.Pilot(bytes_per_sample=4.0, secs_per_row_tree_level=1e-6)
    b = bg.estimate(paths, doc, "acceptance", pilot)
    assert b.missing == []
    real = [c for c in s.cells if c.framework != "treelite"]
    plan = s.plan_for(real[0])
    m = len(plan["variants"]) + bg.OTHER_METHODS[plan["methods"]]
    # The suite's one mode's timed rows, scaled by 500 trees x depth 4: no halving.
    assert plan["modes"] == ["serving"]
    assert b.work >= len(real) * m * 3 * 160 * 500 * 4
    assert b.hours == pytest.approx(b.work * 1e-6 / 3600)
    assert b.sample_bytes == pytest.approx(b.samples * 4.0)


def test_a_sampled_cell_times_its_sample_rows(setup):
    run = {"max_rows_per_round": 65_536, "min_groups_per_round": 200}
    assert bg.timed_fraction(10_000, 100, run) == 1.0
    # FLCHAIN h1024: 1,304 groups of 1,008 rows; the minimum of 200 groups binds.
    f = bg.timed_fraction(1_314_432, 1304, run)
    assert f == 200 / 1304
    # A large pool of small groups: the cap binds.
    assert bg.timed_fraction(1_000_000, 100_000, run) == 65_536 / 1_000_000
    assert bg.timed_fraction(1_000_000, 100, {"max_rows_per_round": 0}) == 1.0


def test_replicates_and_expedia_filled_take_their_twins_sizes(setup):
    paths, doc = setup
    cells = {c.id: c for c in grids.suite(doc, "factorial").cells}
    twins = {
        "support/nt500_md4_h16_r2/xgboost/panel": "support/nt500_md4_h16/xgboost/panel",
        "credit/nt500_md4_r1/xgboost/whatif-v2-k4-G16": "credit/nt500_md4/xgboost/whatif-v2-k4-G16",
        "expedia/nt500_md8_r4/lightgbm/sessions": "expedia/nt500_md8/lightgbm/sessions",
        "expedia-filled/nt50_md2/xgboost/sessions": "expedia/nt50_md2/xgboost/sessions",
    }
    for cid, twin in twins.items():
        t = bg.data_twin(cells[cid])
        assert t is not None and t.id == twin and twin in cells
    assert bg.data_twin(cells["expedia/nt50_md2/xgboost/sessions"]) is None
    # Sized from the twin's cell.json; without it, an error naming the cell.
    filled = cells["expedia-filled/nt50_md2/xgboost/sessions"]
    assert bg.cell_size(paths, filled) is None
    prepare(paths, cells["expedia/nt50_md2/xgboost/sessions"], rows=248_076, groups=10_000)
    assert bg.cell_size(paths, filled) == ((248_076, 10_000, 2), True)
    b = bg.estimate(paths, doc, "factorial", None)
    assert filled.id not in b.missing and b.from_twins == 1
    assert "expedia/nt50_md2/xgboost/sessions" not in b.missing
