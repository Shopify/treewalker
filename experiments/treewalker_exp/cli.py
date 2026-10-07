"""The treewalker-exp command line.

Heavy libraries are imported inside the commands, so ``--help`` stays fast.
"""

import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="Prepare data, models and baselines for the TreeWalker benchmarks.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

Suites = Annotated[list[str], typer.Option("--suite", "-s", help="Suite from grids.toml.")]


class State:
    repo: Path | None = None
    artifacts: Path | None = None


state = State()


@app.callback()
def main(
    repo: Annotated[
        Path | None, typer.Option(help="Repository checkout; found from the working directory.")
    ] = None,
    artifacts_dir: Annotated[
        Path | None, typer.Option(help="Artifacts directory (default: experiments/artifacts).")
    ] = None,
) -> None:
    state.repo, state.artifacts = repo, artifacts_dir


def _paths():
    from .paths import RepoNotFound, resolve

    try:
        return resolve(state.repo, state.artifacts)
    except RepoNotFound as e:
        raise typer.BadParameter(str(e)) from e


def _cells(suites: list[str], datasets, workloads, models, frameworks, ids=None):
    from fnmatch import fnmatchcase

    from . import grids

    paths = _paths()
    doc = grids.load(paths.grids)
    cells: dict[str, grids.Cell] = {}
    # Every workload naming each cell: overlapping workloads share cells, and a
    # workload filter must find a shared cell under each of them.
    member: dict[str, set[str]] = {}
    for name in suites:
        s = grids.suite(doc, name)
        for c in s.cells:
            cells.setdefault(c.id, c)
            member.setdefault(c.id, set()).update(s.workloads_of(c))
    out = [
        c
        for c in cells.values()
        if (not datasets or c.model.dataset in datasets)
        and (not workloads or not member[c.id].isdisjoint(workloads))
        and (not models or c.model.name in models)
        and (not frameworks or c.framework in frameworks)
        and (not ids or any(fnmatchcase(c.id, pattern) for pattern in ids))
    ]
    return paths, doc, out


Datasets = Annotated[list[str] | None, typer.Option("--dataset", help="Only these datasets.")]
Workloads = Annotated[list[str] | None, typer.Option("--workload", help="Only these workloads.")]
Models = Annotated[
    list[str] | None, typer.Option("--model", help="Only these models, e.g. nt500_md8_h16.")
]
Frameworks = Annotated[list[str] | None, typer.Option("--framework", help="lightgbm or xgboost.")]
CellIds = Annotated[
    list[str] | None, typer.Option("--cell", help="Only cells whose ID matches this glob.")
]


@app.command()
def fetch(
    datasets: Annotated[list[str] | None, typer.Argument(help="support, flchain, credit.")] = None,
) -> None:
    """Download the public datasets, checked against their pinned SHA-256s."""
    from . import datasets as ds

    paths = _paths()
    for name in datasets or list(ds.SOURCES):
        print(f"{name}: {ds.fetch(paths, name)}")


@app.command("fetch-expedia")
def fetch_expedia(
    train_csv: Annotated[
        Path | None, typer.Option(help="Kaggle train.csv, or its data.zip.")
    ] = None,
    check_only: Annotated[bool, typer.Option(help="Verify the existing parquet.")] = False,
) -> None:
    """Build experiments/data/expedia.parquet from the Kaggle download.

    The Expedia data (ICDM 2013, kaggle.com/competitions/expedia-personalized-sort)
    may not be redistributed. Accept the competition rules, download data.zip
    (kaggle competitions download -c expedia-personalized-sort -f data.zip),
    and pass it here. The result's content fingerprint is checked against the
    file behind the paper (9,917,530 rows, 399,344 sessions).
    """
    import polars as pl

    from . import datasets as ds

    paths = _paths()
    if check_only == (train_csv is not None):
        raise typer.BadParameter("pass exactly one of --train-csv and --check-only")
    if check_only:
        ok, _ = ds.check_expedia(pl.read_parquet(paths.expedia))
    else:
        assert train_csv is not None
        ok = ds.convert_expedia(train_csv, paths.expedia)
        print(f"wrote {paths.expedia}")
    print(
        "MATCH: identical to the data behind the paper"
        if ok
        else "MISMATCH: the Expedia results will not reproduce exactly"
    )
    raise typer.Exit(0 if ok else 1)


@app.command()
def prepare(
    suite: Suites,
    dataset: Datasets = None,
    workload: Workloads = None,
    model: Models = None,
    framework: Frameworks = None,
    cell: CellIds = None,
    force: Annotated[bool, typer.Option(help="Rebuild even when manifests match.")] = False,
    dry_run: Annotated[bool, typer.Option(help="List the models and cells only.")] = False,
    validate: Annotated[
        bool, typer.Option(help="Check whatif-v1 cells with sweep_bench --validate.")
    ] = False,
) -> None:
    """Train models and write each cell's data, references and cell.json."""
    from . import grids
    from . import manifest as mf

    paths, doc, cells = _cells(suite, dataset, workload, model, framework, cell)
    models = sorted({(c.model, c.framework) for c in cells})
    print(f"{len(models)} models, {len(cells)} cells under {paths.artifacts}", file=sys.stderr)
    if dry_run:
        for c in sorted(cells, key=lambda c: c.id):
            print(c.id)
        return

    from .prepare import Context
    from .prepare import prepare as run

    ctx = Context(paths, grids.defaults(doc), grids.treelite_json_models(doc))
    report = run(ctx, cells, force)
    for name in suite:
        out, m = mf.write(paths, grids.suite(doc, name))
        print(f"manifest {out}: {m['counts']}", file=sys.stderr)
    print(f"prepared {report.models} models; cells {report.cells}", file=sys.stderr)
    failed = list(report.failed)
    if validate:
        failed += _validate_scenario(paths, [c for c in cells if c.generator == "whatif-v1"])
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        raise typer.Exit(1)


def _validate_scenario(paths, cells) -> list[str]:
    from . import bench

    if not cells:
        return []
    binary = bench.build(paths)
    failed = []
    for c in cells:
        ok, line = bench.validate_scenario_cell(
            binary, c.model.dir(paths.artifacts) / c.framework, c.dir(paths.artifacts)
        )
        print(f"  {c.id}: {line}", file=sys.stderr)
        if not ok:
            failed.append(c.id)
    return failed


@app.command()
def manifest(suite: Suites) -> None:
    """Resolve suites into execution manifests from the prepared cells."""
    from . import grids
    from . import manifest as mf

    paths = _paths()
    doc = grids.load(paths.grids)
    for name in suite:
        out, m = mf.write(paths, grids.suite(doc, name))
        print(f"{out}: {m['counts']}")


@app.command("compile-baselines")
def compile_baselines(
    suite: Suites,
    dataset: Datasets = None,
    workload: Workloads = None,
    model: Models = None,
    framework: Frameworks = None,
    cell: CellIds = None,
    only: Annotated[list[str] | None, typer.Option(help="tl2cgen or lleaves.")] = None,
    workers: Annotated[int | None, typer.Option(help="Parallel compiles.")] = None,
    llc: Annotated[str, typer.Option(help="llc for lleaves [default: from PATH].")] = "",
    clang: Annotated[str, typer.Option(help="clang for lleaves [default: from PATH].")] = "",
    force: Annotated[bool, typer.Option(help="Recompile even when records match.")] = False,
) -> None:
    """Compile tl2cgen (both frameworks) and lleaves (LightGBM) once per model.

    tl2cgen comes from the baselines dependency group:
    uv run --group baselines treewalker-exp compile-baselines ...
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from . import baselines as bl
    from . import formats as fm

    paths, _, cells = _cells(suite, dataset, workload, model, framework, cell)
    tools = only or ["tl2cgen", "lleaves"]
    ncpu = os.cpu_count() or 8
    n_workers = workers or min(max(ncpu // 2, 1), 4)
    threads = max(1, ncpu // n_workers)
    tasks = []
    for fw_dir, fw in sorted(
        {(c.model.dir(paths.artifacts) / c.framework, c.framework) for c in cells}
    ):
        if not (fw_dir / "model_treelite.bin").exists():
            continue
        if "tl2cgen" in tools:
            tasks.append(("tl2cgen", fw_dir))
        if "lleaves" in tools and fw == "lightgbm":
            tasks.append(("lleaves", fw_dir))
    print(f"{len(tasks)} compiles, {n_workers} workers x {threads} threads", file=sys.stderr)
    failed = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _compile_one, tool, fw_dir, threads, paths.compile_script, llc, clang, force
            ): (
                tool,
                fw_dir,
            )
            for tool, fw_dir in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            tool, fw_dir = futures[fut]
            label = f"{fw_dir.relative_to(paths.artifacts)}/{tool}"
            try:
                fut.result()
                print(f"  [{i}/{len(tasks)}] ok: {label}", file=sys.stderr)
            except Exception as e:
                failed.append(label)
                print(f"  [{i}/{len(tasks)}] FAIL: {label}: {e}", file=sys.stderr)
    # Record each model's compiled baselines in its cells.
    for c in cells:
        cell_json = c.dir(paths.artifacts) / "cell.json"
        fw_dir = c.model.dir(paths.artifacts) / c.framework
        if not cell_json.exists():
            continue
        doc = fm.read_cell(cell_json)
        doc["baselines"] = {
            tool: bl.summary(fm.read_json(fw_dir / f"{tool}.json"))
            for tool in ("tl2cgen", "lleaves")
            if (fw_dir / f"{tool}.json").exists()
        }
        fm.write_cell(cell_json, doc)
    if failed:
        raise typer.Exit(1)


def _compile_one(tool, fw_dir, threads, script, llc, clang, force) -> None:
    from . import baselines as bl

    if tool == "tl2cgen":
        bl.compile_tl2cgen(fw_dir, threads, force)
    else:
        bl.compile_lleaves(script, fw_dir, threads, llc, clang, force)


@app.command("build-bench")
def build_bench(
    features: Annotated[str, typer.Option(help="Cargo features.")] = "external-bench",
) -> None:
    """Build sweep_bench (release, -C target-cpu=native)."""
    from . import bench

    print(bench.build(_paths(), features))
