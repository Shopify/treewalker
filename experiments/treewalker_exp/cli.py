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
        bool, typer.Option(help="Check the prepared cells with sweep_bench validate.")
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
        failed += _validate(paths, cells)
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        raise typer.Exit(1)


def _validate(paths, cells) -> list[str]:
    from . import bench

    if not cells:
        return []
    binary = bench.build(paths)
    failed = []
    for c in cells:
        ok, line = bench.validate_cell(binary, c.dir(paths.artifacts))
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
    only: Annotated[list[str] | None, typer.Option(help="tl2cgen, lleaves or quickscorer.")] = None,
    jobs: Annotated[
        int | None,
        typer.Option(help="Compiler processes at once, over all compiles [default: CPUs]."),
    ] = None,
    threads: Annotated[int, typer.Option(help="Compiler threads per tl2cgen compile.")] = 2,
    cc: Annotated[str, typer.Option(help="Linker for lleaves [default: cc from PATH].")] = "",
    force: Annotated[bool, typer.Option(help="Recompile even when records match.")] = False,
) -> None:
    """Compile tl2cgen (both frameworks) and lleaves (LightGBM) once per model, and
    write QuickScorer XML for XGBoost models.

    tl2cgen comes from the baselines dependency group:
    uv run --group baselines treewalker-exp compile-baselines ...
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    from . import baselines as bl
    from . import formats as fm
    from . import grids

    paths, doc, cells = _cells(suite, dataset, workload, model, framework, cell)
    # Only the factorial method set times baselines (MethodSet::baselines in
    # sweep_bench); the ablation suite's cells need none.
    timed = {
        cid
        for name in suite
        for cid, p in grids.suite(doc, name).plan.items()
        if p["methods"] == "factorial"
    }
    cells = [c for c in cells if c.id in timed]
    # The sentinel times its cell's baselines in every suite's run, the ablation
    # suite's included, so an unfiltered compile covers its model too.
    if not (dataset or workload or model or framework or cell):
        extra = sentinel_cell(doc)
        if extra is not None and extra.id not in {c.id for c in cells}:
            cells.append(extra)
    files = grids.baseline_settings(doc).get("tl2cgen_parallel_comp", "trees")
    tools = only or ["tl2cgen", "lleaves", "quickscorer"]
    # As many compiler processes at once as CPUs, as before, but in more, narrower
    # compiles: tl2cgen with a few threads each, lleaves its 4 chunks, so one
    # model's last, largest file no longer holds the machine. Largest models first.
    budget = max(1, jobs or os.cpu_count() or 8)
    if "lleaves" in tools and budget < bl.LLEAVES_CHUNKS:
        # Its chunk count is its recipe, so it cannot shrink to fit.
        raise typer.BadParameter(
            f"lleaves runs {bl.LLEAVES_CHUNKS} compiler processes; --jobs must be at least that",
            param_hint="--jobs",
        )
    threads = max(1, min(threads, budget))
    tasks = []
    for fw_dir, fw in sorted(
        {(c.model.dir(paths.artifacts) / c.framework, c.framework) for c in cells}
    ):
        if fw not in ("lightgbm", "xgboost") or not (fw_dir / "model_treelite.bin").exists():
            continue
        if "tl2cgen" in tools:
            tasks.append(("tl2cgen", fw_dir))
        if "lleaves" in tools and fw == "lightgbm":
            tasks.append(("lleaves", fw_dir))
        if "quickscorer" in tools and fw == "xgboost":
            tasks.append(("quickscorer", fw_dir))
    tasks = compile_order(tasks)
    print(
        f"{len(tasks)} compiles, {budget} compiler processes at once, "
        f"tl2cgen {threads} threads each, lleaves {bl.LLEAVES_CHUNKS} chunks",
        file=sys.stderr,
    )
    failed = []

    def submit(pool, task):
        tool, fw_dir = task
        return pool.submit(
            _compile_one, tool, fw_dir, threads, paths.compile_script, cc, force, files
        )

    with ProcessPoolExecutor(max_workers=budget) as pool:
        done = bl.run_budgeted(
            pool, tasks, lambda t: bl.compile_cost(t[0], threads), budget, submit
        )
        for i, ((tool, fw_dir), fut) in enumerate(done, 1):
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
        cdoc = fm.read_cell(cell_json)
        cdoc["baselines"] = {
            tool: bl.summary(fm.read_json(fw_dir / f"{tool}.json"))
            for tool in ("tl2cgen", "lleaves", "quickscorer")
            if (fw_dir / f"{tool}.json").exists()
        }
        fm.write_cell(cell_json, cdoc)
    if failed:
        raise typer.Exit(1)


def compile_order(tasks):
    """Largest models first, by node count; a model's compiles together."""
    from . import baselines as bl

    nodes = {fw_dir: bl.model_nodes(fw_dir) for _, fw_dir in tasks}
    return sorted(tasks, key=lambda t: (-nodes[t[1]], str(t[1]), t[0]))


def sentinel_cell(doc):
    """The sentinel's cell, when [run] names one and turns it on."""
    from . import grids

    run = grids.run_config(doc)
    if not run.get("sentinel_cell") or not run.get("sentinel_every"):
        return None
    return grids.find_cell(doc, run["sentinel_cell"])


def _compile_one(tool, fw_dir, threads, script, cc, force, files) -> None:
    from . import baselines as bl

    if tool == "tl2cgen":
        bl.compile_tl2cgen(fw_dir, threads, force, files)
    elif tool == "quickscorer":
        bl.write_quickscorer(fw_dir, force)
    else:
        bl.compile_lleaves(script, fw_dir, cc, force)


@app.command("build-native")
def build_native(
    library: Annotated[
        list[str] | None, typer.Argument(help="lightgbm, xgboost [default: both].")
    ] = None,
    jobs: Annotated[int | None, typer.Option(help="Parallel compile jobs.")] = None,
    force: Annotated[bool, typer.Option(help="Rebuild even when the cache matches.")] = False,
) -> None:
    """Build the native LightGBM and XGBoost C libraries from pinned sources.

    Versions come from uv.lock and sources are pinned by commit SHA; the build is
    Release with -march=native and OpenMP, into experiments/.cache/native.
    """
    from . import native

    paths = _paths()
    for lib in library or list(native.LIBRARIES):
        rec = native.build(paths, lib, jobs, force)
        print(f"{lib} {rec['version']} ({rec['commit'][:12]}): {rec['path']}")


@app.command("build-bench")
def build_bench(
    features: Annotated[str | None, typer.Option(help="Cargo features.")] = None,
    rustflags: Annotated[str, typer.Option(help="RUSTFLAGS.")] = "-C target-cpu=native",
) -> None:
    """Build sweep_bench (release, -C target-cpu=native; hardware counters on Linux)."""
    from . import bench

    print(bench.build(_paths(), features or bench.FEATURES, rustflags))


@app.command("validation-readout")
def validation_readout(
    run_dir: Annotated[Path, typer.Argument(help="The validation deployment's run directory.")],
) -> None:
    """What the final run's settings rest on: round 0 against later rounds, XGBoost's
    mixed share, counters and children, the sentinel's first load, the probes'
    margins, the drawn samples and the measured per-row rate."""
    from . import readout

    print("\n".join(readout.readout(run_dir)))


@app.command()
def pack(
    run_dirs: Annotated[
        list[Path], typer.Argument(help="Finished runs: experiments/data/runs/<run_id>")
    ],
    remove: Annotated[
        bool, typer.Option(help="Delete the per-cell directories once the packed tables verify.")
    ] = False,
    level: Annotated[int, typer.Option(help="zstd level.")] = 9,
) -> None:
    """One Parquet file per table and one cells.json per run, each table verified against
    its cells; summarize, validation-readout and budget read packed runs only."""
    from . import pack as pk

    for run_dir in run_dirs:
        for line in pk.pack_run(run_dir, remove=remove, level=level):
            print(line, flush=True)


@app.command()
def figures(
    out: Annotated[
        Path | None, typer.Option(help="Output directory (experiments/figures).")
    ] = None,
) -> None:
    """The paper's figures from the final runs: PNG and PDF, and the heatmap as TikZ."""
    from . import figures as fg

    paths = _paths()
    fg.generate(paths.runs, out or paths.figures)


@app.command()
def summarize(
    run_dir: Annotated[Path, typer.Argument(help="experiments/data/runs/<run_id>")],
    reference: Annotated[str, typer.Option(help="The method speedups are against.")] = "treewalker",
    boot: Annotated[int, typer.Option(help="Bootstrap replicates.")] = 1000,
) -> None:
    """Per-group p50/p99, row-weighted speedups and one bootstrap interval each."""
    from . import analysis

    print("\n".join(analysis.summarize(run_dir, reference, boot)))


def _run_id(suite: str) -> str:
    import platform
    from datetime import UTC, datetime

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{suite}-{platform.machine()}-{stamp}"


@app.command()
def preflight(
    suite: Suites,
    require_pmu: Annotated[bool, typer.Option(help="Fail without hardware counters.")] = False,
) -> None:
    """Resolve the suites' manifests and check them with sweep_bench preflight: which
    methods are required and which optional, which variants are unsupported."""
    import subprocess

    from . import bench, grids
    from . import manifest as mf

    paths = _paths()
    doc = grids.load(paths.grids)
    binary = bench.build(paths)
    ok = True
    for name in suite:
        out, _ = mf.write(paths, grids.suite(doc, name))
        cmd = [str(binary), "preflight", str(out)] + (["--require-pmu"] if require_pmu else [])
        ok &= subprocess.run(cmd, check=False).returncode == 0
    raise typer.Exit(0 if ok else 1)


@app.command()
def run(
    suite: Suites,
    run_id: Annotated[str | None, typer.Option(help="Run directory name.")] = None,
    cells: Annotated[str | None, typer.Option(help="Only cells matching this glob.")] = None,
    core: Annotated[int | None, typer.Option(help="Pin to this core (taskset).")] = 0,
    system_info: Annotated[
        Path | None, typer.Option(help="JSON of host facts for run.json (image, turbo mode).")
    ] = None,
    require_pmu: Annotated[bool, typer.Option(help="Fail without hardware counters.")] = False,
    only: Annotated[str | None, typer.Option(help="Only these methods, comma-separated.")] = None,
    rustflags: Annotated[
        str, typer.Option(help="RUSTFLAGS for sweep_bench, e.g. for the layout check.")
    ] = "-C target-cpu=native",
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override a grids.toml [run] setting for this run: KEY=VALUE."),
    ] = None,
) -> None:
    """Run suites with sweep_bench into experiments/data/runs/<run_id>/, one run per
    suite. A rerun with the same run ID resumes: finished cells whose manifests
    match are reused. The run settings, overrides included, are in run.json."""
    import subprocess

    from . import bench, grids
    from . import manifest as mf

    paths = _paths()
    doc = grids.load(paths.grids)
    overrides = tuple(set_ or ())
    try:
        grids.run_config(doc, overrides)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--set") from None
    binary = bench.build(paths, rustflags=rustflags)
    failed = False
    for name in suite:
        out, _ = mf.write(paths, grids.suite(doc, name), overrides)
        rid = run_id if run_id and len(suite) == 1 else _run_id(name)
        run_dir = paths.runs / rid
        cmd = [str(binary), "run", str(out), "--output-dir", str(run_dir)]
        cmd += ["--cells", cells] if cells else []
        cmd += ["--system-info", str(system_info)] if system_info else []
        cmd += ["--require-pmu"] if require_pmu else []
        cmd += ["--only", only] if only else []
        print(f"{name}: {run_dir}", file=sys.stderr)
        failed |= subprocess.run(bench.pinned(cmd, core), check=False).returncode != 0
    raise typer.Exit(1 if failed else 0)


@app.command("pilot-tl2cgen")
def pilot_tl2cgen(
    suite: Suites,
    cell: CellIds = None,
    files: Annotated[
        list[str] | None, typer.Option(help="parallel_comp settings: trees, or a file count.")
    ] = None,
    workers: Annotated[int, typer.Option(help="Compile threads.")] = 8,
) -> None:
    """Time tl2cgen's parallel_comp settings on the selected cells' models.

    The released recipe compiles one file per tree, so every tree is a call the
    compiler cannot inline. Each setting is compiled next to the model as
    tl2cgen_pc<N>.so and timed with sweep_bench cell (tl2cgen only, both modes).
    Run it on the largest and a mid-size model, then set grids.toml [baselines]
    tl2cgen_parallel_comp to the faster setting.
    """
    import json
    import subprocess
    import tempfile

    import polars as pl

    from . import baselines as bl
    from . import bench
    from . import formats as fm
    from . import manifest as mf
    from . import pack as pk

    paths, _, cells = _cells(suite, None, None, None, None, cell)
    runtime = mf.tl2cgen_runtime()
    if runtime is None:
        raise typer.BadParameter("no libtl2cgen: uv run --group baselines ...")
    binary = bench.build(paths)
    results = []
    for c in cells:
        fw_dir = c.model.dir(paths.artifacts) / c.framework
        if c.framework not in ("lightgbm", "xgboost"):
            continue
        for setting in files or ["trees", "32"]:
            n = setting if setting == "trees" else int(setting)
            name = f"tl2cgen_pc{setting}.so"
            rec = bl.compile_tl2cgen(fw_dir, workers, False, n, name)
            out = Path(tempfile.mkdtemp(prefix="tl2cgen-pilot-"))
            cmd = [str(binary), "cell", str(c.dir(paths.artifacts)), "--output-dir", str(out)]
            cmd += ["--only", "tl2cgen", "--tl2cgen-lib", str(fw_dir / name)]
            cmd += ["--tl2cgen-runtime", runtime["path"], "--no-hardware-counters"]
            subprocess.run(bench.pinned(cmd, 0), check=True, capture_output=True)
            hz = fm.read_json(out / "run.json")["timer"]["calibration"]["hz"]
            pk.pack_run(out)
            s = pl.read_parquet(out / "samples.parquet")
            per_row = {
                r["mode"]: r["ticks"] / r["rows"] / hz * 1e6
                for r in s.group_by("mode")
                .agg(pl.col("ticks").sum(), pl.col("rows").sum())
                .iter_rows(named=True)
            }
            row = {
                "cell": c.id,
                "parallel_comp": rec["effective"]["source_files"],
                "setting": setting,
                "compile_seconds": rec["seconds"],
                "us_per_row": per_row,
            }
            results.append(row)
            print(json.dumps(row))
    fm.write_json(paths.artifacts / "pilots" / "tl2cgen.json", results)


@app.command()
def budget(
    suite: Suites,
    pilot_run: Annotated[
        Path | None, typer.Option(help="A finished run to scale from: the validation deployment's.")
    ] = None,
    arch: Annotated[
        str | None,
        typer.Option(help="The machine (x86_64, aarch64), for XGBoost's extra processes."),
    ] = None,
) -> None:
    """Print a suite's expected models, compilations, cells, calls, sample bytes,
    disk, RAM and wall time, scaled from a pilot run where one is given."""
    from . import budget as bg
    from . import grids

    paths = _paths()
    doc = grids.load(paths.grids)
    pilot = bg.Pilot.from_run(pilot_run) if pilot_run else None
    # The inputs decide the estimate: name them.
    print(
        f"budget from pilot {pilot_run or 'none'}, arch {arch or 'any'}, sizes from the "
        f"prepared cells in {paths.artifacts}"
    )
    rounds = int(grids.run_config(doc).get("min_rounds", 3))
    missing = False
    for name in suite:
        b = bg.estimate(paths, doc, name, pilot, arch)
        print("\n".join(bg.describe(b, rounds)))
        missing |= bool(b.missing)
    if missing:
        raise typer.Exit(1)
