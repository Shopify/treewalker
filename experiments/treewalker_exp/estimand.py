"""The camera-ready estimand: v1's nested medians over blocks.

- A block is one (round, batch) pass: the samples' ``block`` id, which the runner
  numbers ``round * blocks_per_round + batch`` (``repetition`` is the round).
- A block's value is the median of its per-group times, by v1's nearest-rank
  convention (`rank_index`, ``src/bench/timing.rs:24-29`` at the neurips2026 tag:
  index ``round(p / 100 * (n - 1))``, half away from zero, so the median of an even
  count is the upper one). In batch mode, where no per-group time is observed, a
  block's value is its ticks over its groups.
- A method's latency is the median over its blocks, every block weighted equally,
  from process 0; a speedup is the ratio of two latencies. p5-p95 over blocks, by
  the same convention, is descriptive dispersion, not an interval.
- The headline interval (serving mode) resamples entities and rounds as crossed
  clusters (`analysis.resample_weights`), the same draws for every method of a
  cell, so the two methods of a ratio are paired, and recomputes both nested
  medians in every draw: a drawn entity enters each block that holds it as many
  times as it was drawn, a drawn round brings all its blocks as many times.
- A sampled cell is used unweighted only if every group had the same inclusion
  probability (`check_equal_probability`); otherwise the estimate fails.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from . import analysis


def rank_index(n: Any, p: float) -> np.ndarray:
    """v1's nearest-rank index into ``n`` sorted values for percentile ``p``:
    ``round(p / 100 * (n - 1))``, half away from zero, clamped to ``n - 1``."""
    n = np.asarray(n, dtype=np.int64)
    x = p / 100.0 * (n - 1).astype(float)
    f = np.floor(x)
    idx = (f + (x - f >= 0.5)).astype(np.int64)
    return np.clip(idx, 0, np.maximum(n - 1, 0))


def nearest_rank(values: np.ndarray, p: float) -> float:
    """v1's ``percentile``: the value at `rank_index` of the sorted ``values``."""
    if len(values) == 0:
        return float("nan")
    return float(np.sort(values)[int(rank_index(len(values), p))])


def block_values(ticks: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Each block's nearest-rank median of its samples' ticks, in block order."""
    order = np.lexsort((ticks, block))
    t, b = ticks[order], block[order]
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    n = np.diff(np.r_[starts, len(b)])
    return t[starts + rank_index(n, 50)].astype(float)


def nested_median(values: np.ndarray) -> tuple[float, float, float]:
    """The median over block values and their p5 and p95, nearest rank."""
    return tuple(nearest_rank(values, p) for p in (50, 5, 95))  # type: ignore[return-value]


class Method:
    """One method's serving samples in a cell, sorted by block and ticks, for
    `boot`: ``ticks``, ``entity`` (dense, per sample), the blocks' bounds and
    each block's round (dense)."""

    def __init__(
        self, ticks: np.ndarray, block: np.ndarray, entity: np.ndarray, rounds: np.ndarray
    ) -> None:
        order = np.lexsort((ticks, block))
        self.ticks = ticks[order].astype(float)
        self.entity = entity[order]
        b = block[order]
        self.starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
        self.ends = np.r_[self.starts[1:], len(b)]
        self.block_round = rounds[order][self.starts]

    def point(self) -> np.ndarray:
        """The block values (every weight 1)."""
        return self.ticks[self.starts + rank_index(self.ends - self.starts, 50)]

    def boot(self, we: np.ndarray, wr: np.ndarray, chunk: int = 1 << 21) -> np.ndarray:
        """The nested median under each draw: ``we`` (n_boot, entities) and ``wr``
        (n_boot, rounds) are `analysis.resample_weights`' counts. NaN for a draw
        that leaves the method no sample."""
        s, nb, n_boot = len(self.ticks), len(self.starts), len(we)
        out = np.full(n_boot, np.nan)
        we_i, wr_i = we.astype(np.int32), wr.astype(np.int32)
        step = max(1, chunk // max(s, 1))
        for a in range(0, n_boot, step):
            b = min(n_boot, a + step)
            n = b - a
            # One cumulative sum over every draw's weights in turn: monotone, so a
            # single search finds each draw's weighted median position in each block.
            c = np.cumsum(we_i[a:b][:, self.entity].ravel(), dtype=np.int64)
            cp = np.concatenate([[0], c])
            off = (np.arange(n, dtype=np.int64) * s)[:, None]
            before, after = cp[off + self.starts], cp[off + self.ends]
            count = after - before
            pos = np.searchsorted(c, (before + count // 2).ravel(), side="right")
            med = self.ticks[np.minimum(pos.reshape(n, nb) - off, s - 1)]
            w = wr_i[a:b][:, self.block_round] * (count > 0)
            order = np.argsort(med, axis=1)
            cw = np.cumsum(np.take_along_axis(w, order, axis=1), axis=1)
            total = cw[:, -1]
            k = np.minimum((cw <= (total // 2)[:, None]).sum(axis=1), nb - 1)
            lat = np.take_along_axis(np.take_along_axis(med, order, axis=1), k[:, None], axis=1)
            out[a:b] = np.where(total > 0, lat[:, 0], np.nan)
        return out


def check_equal_probability(cell: str, manifest: dict[str, Any], groups: Any) -> None:
    """Fail unless every group of the cell had the same inclusion probability.

    ``groups`` is the cell's rows of the groups table (``rows``, ``timed``). A cell
    without a ``sample`` record must time every group. A sampled cell must hold
    the runner's design (``schedule.rs``, `sample_groups`): strata by power-of-two
    size, each taking ``floor`` or ``ceil`` of its ``pool * fraction`` groups by
    systematic rounding with one uniform, so every prefix of strata takes within
    one group of its expected count, which gives every group probability
    ``fraction``. The strata are checked against the groups table."""
    rows = np.asarray(groups["rows"].to_list(), dtype=np.int64)
    timed = np.asarray(groups["timed"].to_list(), dtype=bool)
    sample = manifest.get("sample")
    if not sample:
        if not timed.all():
            raise ValueError(
                f"{cell}: {int((~timed).sum())} groups untimed without a sample record"
            )
        return
    f = float(sample["fraction"])
    strata = sample.get("strata") or []
    if not 0 < f <= 1 or not strata:
        raise ValueError(f"{cell}: the sample record has no fraction or strata")
    bins = np.floor(np.log2(np.maximum(rows, 1))).astype(np.int64)
    expected = taken = 0.0
    for st in strata:
        b = int(np.log2(st["min_size"]))
        in_bin = bins == b
        if int(in_bin.sum()) != st["pool"] or int((in_bin & timed).sum()) != st["sampled"]:
            raise ValueError(
                f"{cell}: stratum {st['min_size']}-{st['max_size']} disagrees with the groups"
                f" table ({int(in_bin.sum())} groups, {int((in_bin & timed).sum())} timed;"
                f" record {st['pool']}, {st['sampled']})"
            )
        expected += st["pool"] * f
        taken += st["sampled"]
        if abs(taken - expected) >= 1 + 1e-9:
            raise ValueError(
                f"{cell}: unequal inclusion probabilities: strata up to "
                f"{st['max_size']} rows take {taken:.0f} groups, expected {expected:.2f}"
            )
    if sum(st["pool"] for st in strata) != len(rows) or int(timed.sum()) != taken:
        raise ValueError(f"{cell}: the strata do not cover the groups table")


INTERVAL = (2.5, 97.5)


def paper_against(method: str, variant: str) -> tuple[str, str] | None:
    """The denominator of an ablation effect the paper reports: a research variant
    against the research build's all-on, and a monotonic variant (whose scans act
    only without the precompute) against the same variant with monotonic on,
    ``disable_monotonic+disable_varying_precompute[|layout]`` against
    ``disable_varying_precompute[|layout]``. None for anything else."""
    if method != "treewalker_research" or variant == "all-on":
        return None
    head, sep, tail = variant.partition("|")
    runtime = [f for f in head.split("+") if f]
    if "disable_monotonic" in runtime:
        rest = "+".join(f for f in runtime if f != "disable_monotonic")
        return (method, rest + sep + tail if rest or tail else "all-on")
    return (method, "all-on")


def _interval(draws: np.ndarray) -> tuple[float | None, float | None]:
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return None, None
    lo, hi = np.percentile(draws, INTERVAL)
    return float(lo), float(hi)


def _serving_cell(
    methods: dict[tuple[str, str], Method],
    ref_key: tuple[str, str],
    we: np.ndarray,
    wr: np.ndarray,
) -> dict[tuple[str, str], dict[str, Any]]:
    """One serving cell's rows of `block_estimates`, per (method, variant). Every
    ratio's interval comes from the same draws of both its sides."""
    boots: dict[tuple[str, str], np.ndarray] = {}
    lats = {key: nested_median(m.point()) for key, m in methods.items()}

    def boot(key: tuple[str, str]) -> np.ndarray:
        if key not in boots:
            boots[key] = methods[key].boot(we, wr)
        return boots[key]

    out = {}
    for key in methods:
        lat, p5, p95 = lats[key]
        row: dict[str, Any] = {
            "latency": lat,
            "p5": p5,
            "p95": p95,
            "blocks": len(methods[key].starts),
            "ratio": float("nan"),
            "lo": None,
            "hi": None,
        }
        if ref_key in methods:
            row["ratio"] = lat / lats[ref_key][0]
            row["lo"], row["hi"] = _interval(boot(key) / boot(ref_key))
        against = paper_against(*key)
        if against is not None and against in methods:
            row["effect_against"] = against[1]
            row["effect"] = lat / lats[against][0]
            row["effect_lo"], row["effect_hi"] = _interval(boot(key) / boot(against))
        out[key] = row
    return out


def block_estimates(
    samples: Any,
    groups: Any,
    manifests: dict[str, Any],
    reference: str = "treewalker",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Every (cell, method, variant, mode)'s latency in ticks per group (the
    nested median), its p5 and p95 over blocks, its block count, and its ratio to
    ``reference`` (variant all-on): ``ratio``, with ``lo`` and ``hi`` (the crossed,
    paired bootstrap, `INTERVAL`) in serving mode. A research variant also has
    its ablation effect against its `paper_against` denominator: ``effect`` with
    ``effect_lo`` and ``effect_hi`` from the same draws, and ``effect_against``.
    ``samples`` is process 0's, with `analysis.SAMPLE_COLUMNS`; ``groups`` the groups
    table."""
    import polars as pl

    groups = groups.with_columns(pl.col("cell").cast(pl.String))
    by_cell = {cell: g for (cell,), g in groups.partition_by("cell", as_dict=True).items()}
    weights: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    s = (
        samples.with_columns(pl.col("cell", "method", "variant", "mode").cast(pl.String))
        .join(groups.select("cell", "group", "entity"), on=["cell", "group"], how="left")
        .with_columns(
            e=pl.col("entity").rank("dense").over("cell", "mode") - 1,
            r=pl.col("repetition").rank("dense").over("cell", "mode") - 1,
        )
    )
    # The bootstrap's numpy work releases the GIL: cells run in parallel.
    pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))
    jobs: list[tuple[str, str, Any]] = []
    for (cell, mode), d in s.partition_by("cell", "mode", as_dict=True).items():
        g = by_cell.get(cell)
        if g is None or d["entity"].null_count():
            raise ValueError(f"{cell}: a timed group is missing from the groups table")
        check_equal_probability(cell, manifests.get(cell, {}), g)
        by_method = d.partition_by("method", "variant", as_dict=True)
        ref_key = (reference, "all-on")
        if mode != "serving":
            vals = {}
            for (method, variant), m in by_method.items():
                per = m.group_by("block").agg(pl.col("ticks").sum(), pl.col("n_groups").sum())
                vals[(method, variant)] = (per["ticks"] / per["n_groups"]).to_numpy()
            ref = nested_median(vals[ref_key])[0] if ref_key in vals else None
            for (method, variant), v in vals.items():
                lat, p5, p95 = nested_median(v)
                row: dict[str, Any] = {
                    "latency": lat,
                    "p5": p5,
                    "p95": p95,
                    "blocks": len(v),
                    "ratio": float("nan") if ref is None else lat / ref,
                    "lo": None,
                    "hi": None,
                }
                against = paper_against(method, variant)
                if against is not None and against in vals:
                    row["effect_against"] = against[1]
                    row["effect"] = lat / nested_median(vals[against])[0]
                    row["effect_lo"] = row["effect_hi"] = None
                out[(cell, method, variant, mode)] = row
            continue
        if int(d["n_groups"].max()) != 1:
            raise ValueError(f"{cell}: a serving sample covers more than one group")
        methods = {
            key: Method(
                m["ticks"].to_numpy(),
                m["block"].to_numpy(),
                m["e"].to_numpy().astype(np.int64),
                m["r"].to_numpy().astype(np.int64),
            )
            for key, m in by_method.items()
        }
        shape = (int(d["e"].max()) + 1, int(d["r"].max()) + 1)
        if shape not in weights:
            weights[shape] = analysis.resample_weights(*shape, n_boot, seed)
        we, wr = weights[shape]
        jobs.append((cell, mode, pool.submit(_serving_cell, methods, ref_key, we, wr)))
    for cell, mode, job in jobs:
        for (method, variant), row in job.result().items():
            out[(cell, method, variant, mode)] = row
    pool.shutdown()
    return out
