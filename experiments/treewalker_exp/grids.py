"""``experiments/grids.toml``, resolved into models and cells."""

import tomllib
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

SURVIVAL = {"support", "flchain"}
GENERATORS = {"panel-v2", "ranking-sessions-v2", "ranking-cohort-v2", "whatif-v1", "whatif-v2"}
# Generators whose models can be seed replicates: the scenario layout cannot.
REPLICABLE = {"panel-v2", "ranking-sessions-v2", "ranking-cohort-v2", "whatif-v2"}


@dataclass(frozen=True, slots=True, order=True)
class Model:
    dataset: str
    n_trees: int
    max_depth: int
    horizon: int | None = None  # requested; survival only
    layout: str = "standard"  # "scenario-v1": the released artifacts/scenario_credit
    # A seed replicate: k >= 1 trains on a seeded sample of this fraction of the
    # training split's entities; 0 is the released model, on all of them.
    replicate: int = 0
    replicate_fraction: float = 0.0

    @property
    def name(self) -> str:
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
    # Per cell ID, every workload that names the cell, in grids.toml's order. A
    # cell shared by overlapping workloads keeps the first as its own (part of
    # its identity); a workload filter matches any of them.
    memberships: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def workloads_of(self, cell: Cell) -> tuple[str, ...]:
        return self.memberships.get(cell.id, (cell.workload,))


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
    return Suite(name, s["description"], list(cells.values()), memberships=memberships)


def _ablation(doc: dict[str, Any], s: dict[str, Any]) -> Suite:
    frameworks = doc["defaults"]["frameworks"]
    cells = []
    for nt, md, g in s["anchors"]:
        for dataset in s["panel_datasets"]:
            for fw in frameworks:
                cells.append(Cell("panel", "panel-v2", Model(dataset, nt, md, g), fw))
        for dataset in s["ranking_datasets"]:
            for fw in frameworks:
                cells.append(Cell("ranking", "ranking-sessions-v2", Model(dataset, nt, md), fw))
    unique = list({c.id: c for c in cells}.values())
    extra = {k: s[k] for k in ("anchors", "runtime_variants", "parse_flags", "interaction_anchors")}
    return Suite("ablation", s["description"], unique, extra)


def defaults(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["defaults"]


def treelite_json_models(doc: dict[str, Any]) -> frozenset[str]:
    return frozenset(doc.get("treelite_json", {}).get("models", []))
