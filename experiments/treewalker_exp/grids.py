"""``experiments/grids.toml``, resolved into models and cells."""

import tomllib
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

SURVIVAL = {"support", "flchain"}
GENERATORS = {
    "panel-v2",
    "ranking-sessions-v2",
    "ranking-cohort-v2",
    "whatif-v1",
    "whatif-v2",
    "fixture-v1",
}
# Generators whose models can be seed replicates: the released fixtures and
# scenario layout cannot.
REPLICABLE = {"panel-v2", "ranking-sessions-v2", "ranking-cohort-v2", "whatif-v2"}
# The load options a variant can turn off; prefix grouping off is prefix_depth 0.
LOAD_FLAGS = (
    "disable_tree_ordering",
    "disable_prefix_grouping",
    "disable_bitset_intern",
    "disable_predicate_dedup",
)
MODES = ("serving", "batch")
RUNTIME_FLAGS = (
    "disable_varying_precompute",
    "disable_predicate_sweep",
    "disable_unsplit",
    "disable_monotonic",
    "disable_exact_sums",
)


@dataclass(frozen=True, slots=True, order=True)
class Model:
    dataset: str
    n_trees: int
    max_depth: int
    horizon: int | None = None  # requested; survival only
    layout: str = "standard"  # "scenario-v1": the released artifacts/scenario_credit
    fixture: str | None = None  # an import fixture (tests/fixtures/import), by name
    # A seed replicate: k >= 1 trains on a seeded sample of this fraction of the
    # training split's entities; 0 is the released model, on all of them.
    replicate: int = 0
    replicate_fraction: float = 0.0

    @property
    def name(self) -> str:
        if self.fixture is not None:
            return self.fixture
        base = f"nt{self.n_trees}_md{self.max_depth}"
        base = base if self.horizon is None else f"{base}_h{self.horizon}"
        return f"{base}_r{self.replicate}" if self.replicate else base

    @property
    def released(self) -> Model:
        """The released model a replicate resamples; a released model itself."""
        return replace(self, replicate=0, replicate_fraction=0.0)

    @property
    def id(self) -> str:
        if self.layout == "scenario-v1":
            return "scenario_credit"
        return f"{self.dataset}/{self.name}"

    def dir(self, artifacts: Path) -> Path:
        """The model's directory: shared split data, one subdirectory per framework."""
        return artifacts / self.id

    def cost(self) -> float:
        """A rough relative training cost, to prepare the slowest models first."""
        return self.n_trees * 2.0 ** min(self.max_depth, 12) * (self.horizon or 16)


@dataclass(frozen=True, slots=True)
class Cell:
    workload: str  # the grids.toml workload table
    generator: str
    model: Model
    framework: str
    params: tuple[tuple[str, Any], ...] = ()  # k, G, size, min_candidates

    @property
    def param(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def workload_id(self) -> str:
        p = self.param
        match self.generator:
            case "panel-v2":
                return "panel"
            case "ranking-sessions-v2":
                return "sessions"
            case "ranking-cohort-v2":
                return f"cohort{p['min_candidates']}-n{p['size']}"
            case "fixture-v1":
                return "import"
            case _:
                return f"{self.generator}-k{p['k']}-G{p['G']}"

    @property
    def id(self) -> str:
        return f"{self.model.id}/{self.framework}/{self.workload_id}"

    def dir(self, artifacts: Path) -> Path:
        if self.model.layout == "scenario-v1":
            p = self.param
            return artifacts / "scenario_credit" / "cells" / f"k{p['k']}_G{p['G']}"
        return self.model.dir(artifacts) / self.framework / "cells" / self.workload_id


@dataclass(slots=True)
class Suite:
    name: str
    description: str
    cells: list[Cell]
    extra: dict[str, Any] = field(default_factory=dict)
    # Per cell ID: what the runner times, {"methods", "variants", "modes"}.
    plan: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per cell ID, every workload that names the cell, in grids.toml's order. A
    # cell shared by overlapping workloads keeps the first as its own (part of
    # its identity); a workload filter matches any of them.
    memberships: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def plan_for(self, cell: Cell) -> dict[str, Any]:
        default = {
            "methods": "factorial",
            "variants": [],
            "modes": ["serving"],
        }
        return {**default, **self.plan.get(cell.id, {})}

    def workloads_of(self, cell: Cell) -> tuple[str, ...]:
        return self.memberships.get(cell.id, (cell.workload,))


def variant(runtime: list[str] | tuple[str, ...] = (), load: list[str] | tuple[str, ...] = ()):
    """A research variant: runtime ablation flags plus load options."""
    for f in runtime:
        if f not in RUNTIME_FLAGS:
            raise ValueError(f"unknown runtime flag {f}")
    for f in load:
        if f not in LOAD_FLAGS:
            raise ValueError(f"unknown load flag {f}")
    return {"runtime": sorted(runtime), "load": sorted(load)}


def _variant_id(v: dict[str, Any]) -> str:
    rt, ld = "+".join(v["runtime"]), "+".join(v["load"])
    return f"{rt}|{ld}" if ld else (rt or "all-on")


def suite_modes(name: str, s: dict[str, Any]) -> list[str]:
    """The timing modes a suite declares: serving, batch or both, in order."""
    modes = s.get("modes")
    if not modes or not isinstance(modes, list) or any(m not in MODES for m in modes):
        raise ValueError(f"suite {name}: modes must list one or both of {', '.join(MODES)}")
    if len(set(modes)) != len(modes):
        raise ValueError(f"suite {name}: modes {modes} repeat")
    return list(modes)


def load(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    if doc.get("schema") != 1:
        raise ValueError(f"{path}: unsupported schema {doc.get('schema')}")
    return doc


def _workload_cells(name: str, w: dict[str, Any], defaults: dict[str, Any]) -> list[Cell]:
    gen = w["generator"]
    if gen not in GENERATORS:
        raise ValueError(f"workload {name}: unknown generator {gen}")
    if gen == "fixture-v1":
        return [
            Cell(name, gen, Model("fixtures", 0, 0, fixture=f), "treelite") for f in w["models"]
        ]
    frameworks = w.get("frameworks", defaults["frameworks"])
    replicates, fraction = _replicates(name, w)
    out = []
    for dataset, nt, md, r in product(w["datasets"], w["n_trees"], w["max_depth"], replicates):
        horizons: list[int | None] = [None]
        if dataset in SURVIVAL:
            horizons = list(w.get("horizon", [16]))
        for h in horizons:
            layout = "scenario-v1" if gen == "whatif-v1" else "standard"
            model = Model(dataset, nt, md, h, layout, replicate=r, replicate_fraction=fraction)
            params: list[tuple[tuple[str, Any], ...]]
            if gen.startswith("whatif"):
                params = [(("k", k), ("G", g)) for k, g in product(w["k"], w["G"])]
            elif gen == "ranking-cohort-v2":
                params = [
                    (("min_candidates", w["min_candidates"]), ("size", n)) for n in w["sizes"]
                ]
            else:
                params = [()]
            for fw, p in product(frameworks, params):
                out.append(Cell(name, gen, model, fw, p))
    return out


def _replicates(name: str, w: dict[str, Any]) -> tuple[list[int], float]:
    """A workload's replicates (``replicates``, ``replicate_fraction``), or the
    released models alone, [0] at fraction 0."""
    if "replicates" not in w:
        if "replicate_fraction" in w:
            raise ValueError(f"workload {name}: replicate_fraction without replicates")
        return [0], 0.0
    ks, fraction = w["replicates"], w.get("replicate_fraction")
    if w["generator"] not in REPLICABLE:
        raise ValueError(f"workload {name}: {w['generator']} has no replicates")
    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError(f"workload {name}: replicates must be distinct integers >= 1")
    if type(fraction) is not float or not 0.0 < fraction < 1.0:
        raise ValueError(f"workload {name}: replicate_fraction must be a float in (0, 1)")
    return list(ks), fraction


def workload_cells(doc: dict[str, Any], name: str) -> list[Cell]:
    return _workload_cells(name, doc["workloads"][name], doc["defaults"])


def suite(doc: dict[str, Any], name: str) -> Suite:
    s = doc["suites"].get(name)
    if s is None:
        raise KeyError(f"unknown suite {name}; grids.toml has {', '.join(doc['suites'])}")
    if name == "ablation":
        return _ablation(doc, s)
    cells: dict[str, Cell] = {}
    memberships: dict[str, tuple[str, ...]] = {}
    for w in s["workloads"]:
        for c in workload_cells(doc, w):
            first = cells.setdefault(c.id, c)  # overlapping workloads share a cell
            if first.model != c.model:
                raise ValueError(f"suite {name}: {c.id} names two models: {first.model}, {c.model}")
            if w not in memberships.get(c.id, ()):
                memberships[c.id] = (*memberships.get(c.id, ()), w)
    if "cells" in s:
        wanted = set(s["cells"])
        missing = wanted - set(cells)
        if missing:
            raise ValueError(f"suite {name}: no such cells {sorted(missing)}")
        cells = {k: c for k, c in cells.items() if k in wanted or c.generator == "fixture-v1"}
    variants = [variant(v) for v in s.get("runtime_variants", [])]
    modes = suite_modes(name, s)
    plan = {}
    for c in cells.values():
        methods = "treewalker" if c.generator == "fixture-v1" else s.get("methods", "factorial")
        plan[c.id] = {
            "methods": methods,
            "variants": variants,
            "modes": modes,
        }
    return Suite(name, s["description"], list(cells.values()), plan=plan, memberships=memberships)


def find_cell(doc: dict[str, Any], cell_id: str) -> Cell | None:
    """The cell with this ID in any workload of grids.toml."""
    for name in doc["workloads"]:
        for c in workload_cells(doc, name):
            if c.id == cell_id:
                return c
    return None


def ablation_variants(s: dict[str, Any], interaction: bool) -> list[dict[str, Any]]:
    """Every runtime variant with the default load options; each load flag alone;
    and at the interaction anchors every runtime variant crossed with all eight
    combinations of the first three load flags. Disabling predicate dedup runs
    alone, with bitset interning on, because the two interact."""
    runtime = [list(v) for v in s["runtime_variants"]]
    out = [variant(rt) for rt in runtime]
    out += [variant((), [f]) for f in s["parse_flags"]]
    if interaction:
        crossed = [f for f in s["parse_flags"] if f != "disable_predicate_dedup"][:3]
        for bits in range(8):
            load = [f for i, f in enumerate(crossed) if bits >> i & 1]
            out += [variant(rt, load) for rt in runtime]
    unique: dict[str, dict[str, Any]] = {}
    for v in out:
        unique.setdefault(_variant_id(v), v)
    return list(unique.values())


def _ablation(doc: dict[str, Any], s: dict[str, Any]) -> Suite:
    frameworks = doc["defaults"]["frameworks"]
    inter = {tuple(a) for a in s["interaction_anchors"]}
    inter_tl = {(t, d) for t, d, _ in inter}
    cells = []
    interaction: dict[str, bool] = {}
    for nt, md, g in s["anchors"]:
        for dataset in s["panel_datasets"]:
            for fw in frameworks:
                c = Cell("panel", "panel-v2", Model(dataset, nt, md, g), fw)
                cells.append(c)
                interaction[c.id] = (nt, md, g) in inter
        for dataset in s["ranking_datasets"]:
            for fw in frameworks:
                c = Cell("ranking", "ranking-sessions-v2", Model(dataset, nt, md), fw)
                cells.append(c)
                interaction[c.id] = (nt, md) in inter_tl
    unique = list({c.id: c for c in cells}.values())
    extra = {k: s[k] for k in ("anchors", "runtime_variants", "parse_flags", "interaction_anchors")}
    modes = suite_modes("ablation", s)
    plan = {
        c.id: {
            "methods": "ablation",
            "variants": ablation_variants(s, interaction[c.id]),
            "modes": modes,
            "interaction": interaction[c.id],
        }
        for c in unique
    }
    return Suite("ablation", s["description"], unique, extra, plan)


def defaults(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["defaults"]


def treelite_json_models(doc: dict[str, Any]) -> frozenset[str]:
    return frozenset(doc.get("treelite_json", {}).get("models", []))


def run_config(
    doc: dict[str, Any], overrides: list[str] | tuple[str, ...] = (), suite: str | None = None
) -> dict[str, Any]:
    """grids.toml [run], then the suite's own [suites.NAME.run] settings, then
    ``KEY=VALUE`` overrides. A key must be in [run] and keep its type; an
    override's value is read as TOML, and a bare word as a string."""
    run = dict(doc.get("run", {}))
    own = doc.get("suites", {}).get(suite, {}).get("run", {}) if suite else {}
    for key, value in own.items():
        if key not in run:
            raise ValueError(f"suite {suite}: run.{key} is not a [run] setting")
        if type(run[key]) is float and type(value) is int:
            value = float(value)
        if type(value) is not type(run[key]):
            raise ValueError(f"suite {suite}: run.{key} is a {type(run[key]).__name__} in [run]")
        run[key] = value
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or key not in run:
            raise ValueError(f"{item}: expected KEY=VALUE with KEY one of {', '.join(run)}")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw
        if type(run[key]) is float and type(value) is int:
            value = float(value)
        if type(value) is not type(run[key]):
            raise ValueError(f"{item}: {key} is a {type(run[key]).__name__} in [run]")
        run[key] = value
    return run


def baseline_settings(doc: dict[str, Any]) -> dict[str, Any]:
    return dict(doc.get("baselines", {}))
