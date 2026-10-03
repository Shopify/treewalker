# Exploratory constant-predicate hoisting benchmarks

The first survival-model sweep found real rewrite opportunities but no general
latency improvement. The pass remains disabled by default. These are local
macOS measurements, not replacements for the paper's pinned Linux runs.

## Method

- Engine: `01e6af685e5e09f95128a084c05afa07d9513285`;
  release `hoist_bench`, Rust 1.94.1, macOS 26.6.2 (Darwin 25.6.0), arm64,
  2026-10-03. The repository build configuration sets `target-cpu=native`.
- Existing trained artifacts; no retraining. Prefer Treelite binary, fall back
  to JSON. Translate legacy configuration field names without changing values.
- Each invocation loads baseline and hoisted forests. Check all input rows
  against the baseline full walk at absolute tolerance 1e-14, then collect work
  counters outside timing.
- Serial execution, three warmup passes over the first batch, 12 group batches,
  rotating method order, 11–21 timing blocks, adaptive stop at CV below 3%.
  Latencies are microseconds
  per **group**. Median/p5/p95 summarize the per-block medians; they are not
  confidence intervals.
- `default`: current tree ordering and prefix depth 2. `isolated`: original
  tree order and prefix depth 0; heavy-path layout and the other prediction
  optimizations still run.
- Speedup = baseline median / hoisted median. Equal weight per cell for
  across-cell summaries. A ratio greater than one favors hoisting.
- No core pinning or frequency control on this local machine. No-op controls
  and separate process repetitions help assess small timing differences.
- Validate declared constant features and group boundaries separately for all
  44 unique inputs (891,555 rows). Input fingerprints and raw reports stay in
  the local output directory; model/data artifacts are not copied into git.

## Survival sweep

There are 13 parameter configurations for each of SUPPORT, FLCHAIN and
METABRIC, with both LightGBM and XGBoost: 78 models, each measured in both modes.
The available configurations vary tree count from 50 to 1,000, maximum depth
from 4 to 10, and group width from 1 to 16. This is the available artifact
grid, not a full factorial grid.

| Dataset | Models | Varying roots before → after | Median constant-visit reduction (default) | Median speedup (default) | Median speedup (isolated) |
|---|---:|---:|---:|---:|---:|
| FLCHAIN | 26 | 2,438 → 2,430 | 0.044% | 1.005× | 1.007× |
| METABRIC | 26 | 4,886 → 4,875 | 0.015% | 0.997× | 0.999× |
| SUPPORT | 26 | 4,988 → 4,848 | 0.133% | 1.003× | 0.999× |
| All | 78 | 12,312 → 12,153 | 0.048% | 1.001× | 1.000× |

The pass changes 1,747 of 36,900 trees across 72 models. It performs 1,840
swaps and 16 collapses, removing 32 of 5,605,234 nodes. Only 29 models lose a
varying root. No tree reaches the 16-pass limit.

All 156 runs pass correctness. Maximum absolute prediction difference is
6.0e-15 with ordering enabled and exactly zero with original tree order.
Varying-split, recursive-call, leaf-hit and precompute-row counters are
unchanged in every cell. The largest reduction in constant visits is 0.645%
with default optimizations and 0.492% in the ablation.

Default speedups range from 0.954× to 1.040× (geometric mean 0.999×). Ablation
speedups range from 0.943× to 1.050× (geometric mean 1.001×). Six models have
no rewrites at all; their default ratios range from 0.970× to 1.029×.
These controls and the tiny work reductions do not support interpreting the
small positive ratios as an established hoisting benefit.

The 500-tree, depth-8, width-16 anchors illustrate the absolute timings. Each
entry is median [p5, p95] in **µs/group** from the main sweep's default mode:

| Dataset | Framework | Baseline | Hoisted |
|---|---|---:|---:|
| FLCHAIN | LightGBM | 36.792 [33.792, 44.167] | 36.834 [33.667, 40.541] |
| FLCHAIN | XGBoost | 32.333 [29.750, 37.791] | 32.041 [29.667, 35.542] |
| METABRIC | LightGBM | 58.500 [53.583, 63.542] | 59.167 [52.250, 72.459] |
| METABRIC | XGBoost | 48.459 [43.041, 52.916] | 49.333 [44.125, 54.666] |
| SUPPORT | LightGBM | 40.334 [39.209, 41.584] | 40.666 [40.125, 41.542] |
| SUPPORT | XGBoost | 41.917 [40.750, 43.500] | 41.875 [40.833, 43.334] |

## Expedia

The two depth-16, 2,000-tree models have many varying roots, but few can be
changed by the paired-child rule:

| Framework | Nodes before → after | Varying roots before → after | Swaps | Collapses |
|---|---:|---:|---:|---:|
| LightGBM | 15,864,790 → 15,863,978 | 1,007 → 1,000 | 3,234 | 406 |
| XGBoost | 5,989,228 → 5,989,228 | 1,335 → 1,328 | 1,673 | 0 |

On the empirical input (10,000 groups, 248,293 rows, maximum width 37), default
timings are again median [p5, p95] in **µs/group**:

| Framework | Baseline | Hoisted | Speedup |
|---|---:|---:|---:|
| LightGBM | 2,410.792 [2,351.250, 3,553.208] | 2,405.541 [2,335.833, 2,641.542] | 1.002× |
| XGBoost | 1,750.833 [1,680.584, 2,003.792] | 1,756.750 [1,666.666, 2,158.625] | 0.997× |

LightGBM saves 0.149% of constant visits, 924 of 845,255,830 varying-split
visits, and 38 of 174,068,972 recursive calls. XGBoost saves 0.155% of constant
visits and leaves varying-split and recursive-call counts unchanged. Both
precompute-row counters are unchanged. The ordering/prefix ablation yields
0.996× for both frameworks on this input, with exact prediction agreement.

All five available group distributions were measured in both modes, for 20
successful runs. Maximum prediction difference is 6.2e-15 with ordering and
zero without it; no tree reaches the pass limit. Default constant-visit
reductions range from 0.115% to 0.163%.

| Input | Framework | Groups | Default speedup | Isolated speedup |
|---|---|---:|---:|---:|
| empirical | LightGBM | 10,000 | 1.002× | 0.996× |
| empirical | XGBoost | 10,000 | 0.997× | 0.996× |
| fixed16 | LightGBM | 171 | 0.994× | 0.992× |
| fixed16 | XGBoost | 171 | 0.986× | 1.034× |
| fixed32 | LightGBM | 1,459 | 0.995× | 0.979× |
| fixed32 | XGBoost | 1,459 | 0.990× | 1.011× |
| fixed8 | LightGBM | 194 | 1.002× | 1.029× |
| fixed8 | XGBoost | 194 | 1.017× | 0.988× |
| geom8 | LightGBM | 2,907 | 0.978× | 1.028× |
| geom8 | XGBoost | 2,907 | 1.049× | 1.013× |

The median across these ten model/input pairs is 0.996× in default mode and
1.003× in the ablation. The geometric means are 1.001× and 1.007×.

## Repetitions and interpretation

Repeat the six survival anchors, six width-one cases, and both METABRIC
width-two models three times in default mode: 42 additional successful runs.
These use exactly 24 blocks, covering all 12 batches twice with balanced
method order. Selection of the width-two cases followed the initial slowdown;
this is an exploratory follow-up, not a pre-registered significance test.

The 18 anchor ratios have median 0.996× and range 0.982–1.033×. The initial
0.954× METABRIC LightGBM width-two result becomes 0.990–0.993×. Five width-one
models have no rewrites at all; their repeated ratios span 0.952–1.021×.
The largest positive Expedia result, XGBoost/geom8 at 1.049×, also gets three
separate 24-block repetitions: 1.004×, 1.026×, and 1.006×. The initial gain
does not recur at the same magnitude.

All 221 main and repeat invocations pass correctness. These results support
keeping the prototype opt-in. The matched-predicate rule produces too little
work reduction on this artifact set to establish a general execution benefit.
Varying-root count is a diagnostic; a useful next transform should remove
measurable work while controlling representation growth and cache effects.
Downstream execution performance remains a separate measurement.

## Reproduction

Build and run from the repository root, using new output directories:

```bash
cargo build --release --manifest-path benchmarks/Cargo.toml --bin hoist_bench
python3 paper/experiments/scripts/hoist_sweep.py ARTIFACTS target/hoist-survival \
  --dataset support --dataset flchain --dataset metabric
python3 paper/experiments/scripts/hoist_sweep.py ARTIFACTS target/hoist-expedia \
  --dataset expedia
python3 paper/experiments/scripts/hoist_sweep.py ARTIFACTS target/hoist-repeats \
  --cell '*/nt500_md8_h16' --cell '*/nt500_md8_h1' --cell 'metabric/nt500_md8_h2' \
  --mode default --repeats 3 --min-blocks 24 --max-blocks 24
python3 paper/experiments/scripts/hoist_sweep.py ARTIFACTS target/hoist-outlier \
  --cell 'expedia/nt2000_md16/geom8' --framework xgboost \
  --mode default --repeats 3 --min-blocks 24 --max-blocks 24
```

The runner retains raw JSON, stderr, commands, a binary hash and an incremental
CSV. The original exploratory outputs are under the ignored local directory
`target/hoisting-private-2026-10-03/`. Single-sample load times are available
there but are not used for performance conclusions because load/cache order
is not controlled.
