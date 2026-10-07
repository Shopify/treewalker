"""The command line's cell selection."""

from treewalker_exp import cli, grids
from treewalker_exp.paths import Paths, find_repo


def test_a_workload_filter_finds_the_cells_it_shares(monkeypatch):
    """whatif-credit-full overlaps whatif-credit-core at T=500, L=8; the shared
    cells keep the first workload's name but belong to both."""
    repo = find_repo()
    monkeypatch.setattr(cli, "_paths", lambda: Paths(repo))
    doc = grids.load(repo / "experiments/grids.toml")
    full = grids.workload_cells(doc, "whatif-credit-full")
    _, _, cells = cli._cells(["factorial"], None, ["whatif-credit-full"], None, None)
    assert {c.id for c in cells} == {c.id for c in full}
    assert any(c.workload == "whatif-credit-core" for c in cells)  # shared, named for the first
    # The filter selects; it changes no cell's identity.
    factorial = {c.id: c for c in grids.suite(doc, "factorial").cells}
    assert all(factorial[c.id] == c for c in cells)
