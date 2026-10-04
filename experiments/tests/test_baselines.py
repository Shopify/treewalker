import pytest

from treewalker_exp import baselines as bl
from treewalker_exp import formats as fm


@pytest.fixture
def probes(monkeypatch):
    """Fixed toolchain probes; tests change one field at a time."""
    state = {
        "compiler": {"path": "/usr/bin/gcc", "version": "gcc 13.3.0", "target": "x86_64-linux-gnu"},
        "llc": {
            "path": "/usr/bin/llc",
            "version": "LLVM version 20.1.8",
            "default_target": "x86_64-unknown-linux-gnu",
            "host_cpu": "emeraldrapids",
        },
        "host_cpu": "Intel(R) Xeon(R) Platinum 8581C",
        "versions": {"tl2cgen": "1.0.0", "treelite": "4.7.2"},
    }
    monkeypatch.setattr(bl, "_which", lambda name, configured="": f"/usr/bin/{name}")
    monkeypatch.setattr(bl, "probe_compiler", lambda path: dict(state["compiler"]))
    monkeypatch.setattr(bl, "probe_llc", lambda path: dict(state["llc"]))
    monkeypatch.setattr(bl, "host_cpu", lambda: state["host_cpu"])
    monkeypatch.setattr(bl, "_dist_version", lambda name: state["versions"][name])
    return state


def tl2cgen_key(settings=None):
    return bl.identity_key(bl.tl2cgen_identity("m" * 64, settings or bl.tl2cgen_settings(500, 4)))


def lleaves_key(script, **request):
    req = {**bl.lleaves_request(script.parent, 4, "/usr/bin/llc", "/usr/bin/clang"), **request}
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
    assert tl2cgen_key(bl.tl2cgen_settings(500, 8)) != base
    other = bl.identity_key(bl.tl2cgen_identity("n" * 64, bl.tl2cgen_settings(500, 4)))
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
        lambda s, p: s["llc"].update(host_cpu="neoverse-v2"),
        lambda s, p: s["llc"].update(version="LLVM version 21.1.0"),
        lambda s, p: s["llc"].update(default_target="aarch64-unknown-linux-gnu"),
        lambda s, p: s["compiler"].update(version="clang 21"),
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
