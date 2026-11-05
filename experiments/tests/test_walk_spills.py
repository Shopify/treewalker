"""The spill check reads only assembly emitted by its build."""

import subprocess

import pytest
import walk_spills


def test_failed_build_does_not_reuse_assembly(tmp_path, monkeypatch):
    monkeypatch.setattr(walk_spills, "ROOT", tmp_path)
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "walk_asm.rs").touch()
    examples = tmp_path / "target" / "walk-spills" / "release" / "examples"
    examples.mkdir(parents=True)
    stale = examples / "walk_asm-old.s"
    stale.write_text("stale")

    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "invalid lanes")

    monkeypatch.setattr(walk_spills.subprocess, "run", failed)
    with pytest.raises(SystemExit, match=r"no assembly was emitted:.*\ninvalid lanes"):
        walk_spills.build(None, "native", 13)

    def emitted(*args, **kwargs):
        stale.write_text("fresh assembly")
        return subprocess.CompletedProcess(args, 1, "", "expected link failure")

    monkeypatch.setattr(walk_spills.subprocess, "run", emitted)
    assert walk_spills.build(None, "native", None).read_text() == "fresh assembly"
