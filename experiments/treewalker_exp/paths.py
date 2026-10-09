"""Every path the package uses, relative to one repository checkout.

treewalker-exp is a repository tool, not a distributed wheel. It finds the
checkout by walking up to ``experiments/grids.toml``, from the working
directory first and then from the package itself, or takes ``--repo``.
"""

from dataclasses import dataclass
from pathlib import Path

MARKER = Path("experiments/grids.toml")


class RepoNotFound(RuntimeError):
    pass


def _walk_up(start: Path) -> Path | None:
    for d in (start, *start.parents):
        if (d / MARKER).is_file():
            return d
    return None


def find_repo(explicit: Path | None = None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not (root / MARKER).is_file():
            raise RepoNotFound(f"{root} has no {MARKER}")
        return root
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        if (found := _walk_up(start)) is not None:
            return found
    raise RepoNotFound(f"no {MARKER} above {Path.cwd()}; pass --repo")


@dataclass(frozen=True, slots=True)
class Paths:
    repo: Path
    artifacts_override: Path | None = None

    @property
    def experiments(self) -> Path:
        return self.repo / "experiments"

    @property
    def grids(self) -> Path:
        return self.experiments / "grids.toml"

    @property
    def data(self) -> Path:
        return self.experiments / "data"

    @property
    def raw(self) -> Path:
        """Download cache for the public datasets (git-ignored)."""
        return self.data / "raw"

    @property
    def expedia(self) -> Path:
        return self.data / "expedia.parquet"

    @property
    def artifacts(self) -> Path:
        return self.artifacts_override or self.experiments / "artifacts"

    @property
    def runs(self) -> Path:
        """Run directories: run.json and per-cell Parquet tables."""
        return self.data / "runs"

    @property
    def figures(self) -> Path:
        """The paper's figures, tables and numbers, regenerated from the runs."""
        return self.experiments / "figures"

    @property
    def manifests(self) -> Path:
        return self.artifacts / "manifests"

    @property
    def compile_script(self) -> Path:
        return self.experiments / "compile.py"

    @property
    def target_dir(self) -> Path:
        return self.repo / "target"

    @property
    def sweep_bench(self) -> Path:
        return self.target_dir / "release" / "sweep_bench"


def resolve(repo: Path | None = None, artifacts: Path | None = None) -> Paths:
    return Paths(find_repo(repo), artifacts.resolve() if artifacts else None)
