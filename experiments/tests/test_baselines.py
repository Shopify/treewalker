from pathlib import Path

import pytest

from treewalker_exp import baselines as bl
from treewalker_exp import formats as fm


@pytest.fixture
def probes(monkeypatch):
    """Fixed toolchain probes; tests change one field at a time."""
    state = {
        "compiler": {"path": "/usr/bin/gcc", "version": "gcc 13.3.0", "target": "x86_64-linux-gnu"},
        "host_cpu": "Intel(R) Xeon(R) Platinum 8581C",
        "versions": {"tl2cgen": "1.0.0", "treelite": "4.7.2"},
    }
    monkeypatch.setattr(bl, "_which", lambda name, configured="": f"/usr/bin/{name}")
    monkeypatch.setattr(bl, "probe_compiler", lambda path: dict(state["compiler"]))
    monkeypatch.setattr(bl, "host_cpu", lambda: state["host_cpu"])
    monkeypatch.setattr(bl, "_dist_version", lambda name: state["versions"][name])
    return state


def tl2cgen_key(settings=None):
    return bl.identity_key(bl.tl2cgen_identity("m" * 64, settings or bl.tl2cgen_settings(500)))


def lleaves_key(script, **request):
    req = {**bl.lleaves_request(script.parent, "/usr/bin/cc"), **request}
    return bl.identity_key(bl.lleaves_identity(script, "m" * 64, req))


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["compiler"].update(version="gcc 14.2.0"),
        lambda s: s["compiler"].update(target="aarch64-linux-gnu"),
        lambda s: s["compiler"].update(path="/opt/gcc/bin/gcc"),
        lambda s: s.update(host_cpu="Google Axion"),
        lambda s: s["versions"].update(tl2cgen="1.0.1"),
        lambda s: s["versions"].update(treelite="4.7.3"),
    ],
)
def test_tl2cgen_identity_changes(probes, change):
    before = tl2cgen_key()
    change(probes)
    assert tl2cgen_key() != before


def test_tl2cgen_identity_settings_and_model(probes):
    base = tl2cgen_key()
    # Compile threads do not shape the library, so they are not in the settings.
    assert "nthread" not in bl.tl2cgen_settings(500)
    assert tl2cgen_key(bl.tl2cgen_settings(500, 32)) != base
    other = bl.identity_key(bl.tl2cgen_identity("n" * 64, bl.tl2cgen_settings(500)))
    assert other != base
    assert tl2cgen_key() == base  # deterministic


@pytest.fixture
def script(tmp_path):
    p = tmp_path / "compile.py"
    p.write_text("# script\n")
    (tmp_path / "compile.py.lock").write_text("lock v1\n")
    return p


@pytest.mark.parametrize(
    "change",
    [
        lambda s, p: s["compiler"].update(version="clang 21"),
        lambda s, p: s["compiler"].update(target="aarch64-linux-gnu"),
        lambda s, p: s.update(host_cpu="Apple M4 Pro"),
        lambda s, p: p.with_name("compile.py.lock").write_text("lock v2, new llvmlite\n"),
        lambda s, p: p.write_text("# script, edited\n"),
    ],
)
def test_lleaves_identity_changes(probes, script, change):
    before = lleaves_key(script)
    change(probes, script)
    assert lleaves_key(script) != before


def test_lleaves_identity_settings(probes, script):
    base = lleaves_key(script)
    assert lleaves_key(script, opt_level="2") != base
    assert lleaves_key(script, fp_contract="fast") != base
    assert lleaves_key(script, fblocksize=2) != base
    # The chunk count is the recipe's, 4, and changing it changes the identity.
    assert bl.lleaves_request(script.parent, "/usr/bin/cc")["n_jobs"] == 4
    assert lleaves_key(script, n_jobs=8) != base
    # The model and output paths are not part of the identity; the model's hash is.
    assert lleaves_key(script, output="/elsewhere/lleaves.so") == base


def test_reusable_requires_key_and_library_hash(tmp_path):
    lib, record = tmp_path / "tl2cgen.so", tmp_path / "tl2cgen.json"
    lib.write_bytes(b"library")
    fm.write_json(record, {"key": "k1", "sha256": fm.sha256_file(lib)})
    assert bl.reusable(record, lib, "k1")
    assert not bl.reusable(record, lib, "k2")  # another compiler, target or setting
    lib.write_bytes(b"rebuilt elsewhere")
    assert not bl.reusable(record, lib, "k1")  # the library changed
    # A record from before compile identities (request and model hash only).
    fm.write_json(record, {"model_sha256": "m", "request": {}, "sha256": fm.sha256_file(lib)})
    assert not bl.reusable(record, lib, "k1")
    record.unlink()
    assert not bl.reusable(record, lib, "k1")


def test_tl2cgen_targets_the_host_and_sets_its_file_count():
    s = bl.tl2cgen_settings(500, 32)
    assert s["options"] == ["-march=native"]  # finding 0.19: CFLAGS never reached gcc
    assert "env" not in s
    assert s["params"]["parallel_comp"] == 32
    assert bl.tl2cgen_settings(500)["params"]["parallel_comp"] == 500
    assert bl.tl2cgen_settings(20, 32)["params"]["parallel_comp"] == 20


def test_tl2cgen_files_stay_under_the_node_cap():
    def files(trees, nodes):
        return bl.tl2cgen_settings(trees, 32, nodes)["params"]["parallel_comp"]

    assert files(500, 170_000) == 32  # T=500, L=8: the pilot's 32 files
    assert files(500, 256_000) == 32
    assert files(500, 256_001) == 33
    # Expedia T=1000, L=16, XGBoost: 32 files took 60 GB in one cc1.
    assert files(1000, 4_021_510) == 503
    assert files(1000, 7_749_154) == 969
    assert files(1000, 9_000_000) == 1000  # never more files than trees
    assert bl.tl2cgen_settings(1000, "trees", 4_021_510)["params"]["parallel_comp"] == 1000


def test_compiles_run_largest_first_within_the_process_budget(tmp_path):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from treewalker_exp.cli import compile_order

    dirs = {}
    for name, nodes in [("small", 100), ("big", 90_000), ("mid", 5_000)]:
        d = tmp_path / name
        fm.write_json(d / "model.json", {"structure": {"total_nodes": nodes}})
        dirs[name] = d
    tasks = [(t, d) for d in dirs.values() for t in ("tl2cgen", "lleaves")]
    order = compile_order(tasks)
    assert [d.name for _, d in order] == ["big", "big", "mid", "mid", "small", "small"]
    assert bl.model_nodes(tmp_path / "none") == 0

    # 2-thread tl2cgen and 4-chunk lleaves compiles never exceed 6 processes.
    lock, state = threading.Lock(), {"now": 0, "peak": 0, "started": []}

    def work(task, c):
        with lock:
            state["now"] += c
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.01)
        with lock:
            state["now"] -= c

    def cost(task):
        return bl.compile_cost(task[0], 2)

    def submit(pool, task):
        state["started"].append(task)
        return pool.submit(work, task, cost(task))

    with ThreadPoolExecutor(max_workers=6) as pool:
        done = list(bl.run_budgeted(pool, order, cost, 6, submit))
    assert len(done) == 6 and all(f.exception() is None for _, f in done)
    assert state["peak"] <= 6
    assert state["started"] == order  # strictly in order
    assert (bl.compile_cost("lleaves", 2), bl.compile_cost("quickscorer", 2)) == (4, 1)


def xgb_model(tmp_path, categorical=False, depth=2, rounds=3):
    import numpy as np

    from treewalker_exp import train as tr

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)).astype(np.float32).astype(np.float64)
    if categorical:
        X[:, 2] = rng.integers(0, 4, size=400)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    if categorical:
        y = (X[:, 2] >= 2).astype(np.float64)
    native = tmp_path / "model.json"
    cat = [2] if categorical else []
    tr.train("xgboost", X, y, ["a", "b", "c"], cat, rounds, depth, native)
    return tr.load_treelite("xgboost", native), X


def test_quickscorer_xml_is_exact_for_xgboost(tmp_path):
    import numpy as np
    import treelite

    tl, X = xgb_model(tmp_path)
    xml, status = bl.quickscorer_xml(tl)
    assert status == "ready" and xml is not None
    # QuickScorer sends x <= t' left; t' is the next float32 below XGBoost's t.
    for t in range(tl.num_tree):
        acc = tl.get_tree_accessor(t)
        split = acc.get_field("cleft") >= 0
        for thr in np.asarray(acc.get_field("threshold"))[split]:
            below = bl._f32_below(float(np.float32(thr)))
            assert f"<threshold>{bl._xml_number(below)}</threshold>" in xml
            x = np.float32(thr)
            assert not (x < np.float32(thr)) and not (x <= np.float32(below))
            x = np.nextafter(x, np.float32(-np.inf))
            assert (x < np.float32(thr)) and (x <= np.float32(below))
    # A tiny scorer over the XML gives XGBoost's tree sums.
    import xml.etree.ElementTree as ET

    def walk(node, row):
        out = node.find("output")
        if out is not None:
            return float(np.float32(out.text))
        f, t = int(node.find("feature").text) - 1, np.float32(node.find("threshold").text)
        left = np.float32(row[f]) <= t
        return walk(node.find(f"split[@pos='{'left' if left else 'right'}']"), row)

    trees = ET.fromstring(xml.split("\n", 1)[1]).find("ensemble").findall("tree")
    sums = [sum(walk(tree.find("split"), row) for tree in trees) for row in X[:50]]
    ref = treelite.gtil.predict_per_tree(tl, X[:50].astype(np.float32)).sum(axis=1).ravel()
    assert np.allclose(sums, ref, rtol=0, atol=1e-6)


def test_quickscorer_rejects_categories_and_wide_trees(tmp_path):
    tl, _ = xgb_model(tmp_path, categorical=True)
    xml, status = bl.quickscorer_xml(tl)
    assert xml is None and "categorical" in status
    tl, _ = xgb_model(tmp_path, depth=8, rounds=1)
    n_leaves = int((tl.get_tree_accessor(0).get_field("cleft") < 0).sum())
    xml, status = bl.quickscorer_xml(tl)
    assert (xml is None) == (n_leaves > 128)


def test_a_compile_budget_below_lleaves_chunks_is_rejected(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from treewalker_exp import cli

    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    r = CliRunner().invoke(
        cli.app,
        ["--artifacts-dir", str(tmp_path), "compile-baselines", "--suite", "smoke", "--jobs", "2"],
    )
    assert r.exit_code != 0 and "at least" in r.output
    # QuickScorer alone fits any budget.
    r = CliRunner().invoke(
        cli.app,
        [
            *("--artifacts-dir", str(tmp_path), "compile-baselines", "--suite", "smoke"),
            *("--jobs", "1", "--only", "quickscorer"),
        ],
    )
    assert r.exit_code == 0, r.output
