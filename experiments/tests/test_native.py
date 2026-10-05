"""Native library sources: versions from uv.lock, commits pinned."""

import pytest

from treewalker_exp import native
from treewalker_exp.paths import find_repo, resolve


@pytest.fixture
def paths():
    return resolve(find_repo())


@pytest.mark.parametrize("lib", native.LIBRARIES)
def test_locked_versions_have_pinned_commits(paths, lib):
    src = native.source(paths, lib)
    assert src["version"] == native.locked_version(paths, lib)
    assert len(src["commit"]) == 40 and src["tag"].endswith(src["version"])


def test_lightgbm_comes_from_its_new_home(paths):
    assert "lightgbm-org/LightGBM" in native.source(paths, "lightgbm")["url"]


def test_an_unpinned_version_is_an_error(paths, monkeypatch):
    monkeypatch.setattr(native, "locked_version", lambda p, lib: "9.9.9")
    with pytest.raises(KeyError, match="no pinned source commit"):
        native.source(paths, "xgboost")


def test_the_cache_key_follows_the_compiler_and_flags(paths, monkeypatch):
    src = native.source(paths, "lightgbm")
    base = native.identity(src)
    monkeypatch.setattr(native, "FLAGS", "-march=x86-64")
    assert native.identity(src)["cmake_args"] != base["cmake_args"]
    monkeypatch.setenv("CXX", "/nonexistent/old-g++")
    assert native.identity(src)["cxx"]["path"] == "/nonexistent/old-g++"
