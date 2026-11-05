import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import walk_spills as ws


def test_a_failed_build_does_not_return_an_earlier_runs_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "walk_asm.rs").touch()
    examples = tmp_path / "target" / "walk-spills" / "release" / "examples"
    examples.mkdir(parents=True)
    (examples / "walk_asm-0123.s").write_text("an earlier run's assembly\n")
    monkeypatch.setattr(ws, "ROOT", tmp_path)

    def failed(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 101, "", "error: could not compile")

    monkeypatch.setattr(ws.subprocess, "run", failed)
    with pytest.raises(SystemExit, match="could not compile"):
        ws.build(None, "native", 13)
