"""TreeWalker's work counters in the released stats CSVs' columns.

The released stats (grid3_stats_*.csv, scenario_credit_stats_*.csv) hold the
version 1 counters of treewalker-gbdt 1.x, and the paper's cost terms are defined
on their columns. A run's counters.parquet records counters_version: 2 for the
2.x recursive kernel, 3 to 6 for the 3.x kernel. `terms` maps one cell and
variant's counters, summed over its groups, onto those columns:

    CSV column            version 2               versions 3 to 6
    constant_steps        constant_steps          walk_steps + prefix_steps
                                                  - varying_events - leaf events
    varying_splits        varying_splits          varying_events
    unsplit_skips         unsplit_skips           varying_events - splits
    recursive_calls       recursive_calls         (none: no recursion)
    leaf_hits             leaf_hits               leaf events
    partition_row_evals   partition_row_evals     scan_evals
    precompute_row_evals  precompute_row_evals    precompute_row_evals

Version 3's leaf events are the masked path's leaf_events plus one per tree
for each single row, whose walk steps through its leaf too; version 4's
leaf_events count single rows' leaves already, version 5 changes only
``pieces``, which no column reads, and version 6 drops it. The counted constant
steps are the walk's and the prefix's node steps that end in neither event; at
G = 1 the varying splits are constant ones.
"""

from collections.abc import Mapping

# The CSVs' counter columns.
COUNTERS = (
    "constant_steps",
    "varying_splits",
    "unsplit_skips",
    "recursive_calls",
    "leaf_hits",
    "partition_row_evals",
    "precompute_row_evals",
)


def terms(version: int, c: Mapping[str, int], trees: int) -> dict[str, int | None]:
    """One cell and variant's summed counters ``c`` of ``version``, as the CSVs'
    counter columns; ``trees`` is the model's tree count."""
    if version == 2:
        return {k: c[k] for k in COUNTERS}
    if version in (3, 4, 5, 6):
        leaves = c["leaf_events"] + (c["single_rows"] * trees if version == 3 else 0)
        return {
            "constant_steps": c["walk_steps"] + c["prefix_steps"] - c["varying_events"] - leaves,
            "varying_splits": c["varying_events"],
            "unsplit_skips": c["varying_events"] - c["splits"],
            "recursive_calls": None,
            "leaf_hits": leaves,
            "partition_row_evals": c["scan_evals"],
            "precompute_row_evals": c["precompute_row_evals"],
        }
    raise ValueError(f"counters version {version} is not supported")
