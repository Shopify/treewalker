//! The work counters per group, and each ablation's counter signature.

use std::collections::BTreeMap;

use anyhow::Result;
use serde_json::{Value, json};
use treewalker_gbdt::Forest;
use treewalker_gbdt::research::{Ablation, COUNTERS_VERSION, Phases, WorkCounters};

use crate::artifacts::{Cell, VariantSpec};
use crate::output::{Table, V};

use super::load::{Loaded, load_key};
use super::validate::drop_failed;
use super::{Context, Excluded};

/// One counted group: its counters and the time its phases took.
type Counted = (WorkCounters, Phases);

pub(super) fn count(
    forest: &Forest,
    ablation: Ablation,
    cell: &Cell,
    per_group: bool,
) -> (Counted, Vec<Counted>) {
    let mut r = forest.research_predictor(ablation);
    let mut out = vec![0.0; cell.max_group_rows()];
    let mut total = Counted::default();
    let mut groups = Vec::new();
    for g in 0..cell.n_groups() {
        let rows = cell.group(g);
        let c = r.predict_group_counted(rows, &mut out[..rows.len() / cell.n_features]);
        let p = r.take_phases();
        total.0 += c;
        total.1 += p;
        if per_group {
            groups.push((c, p));
        }
    }
    (total, groups)
}

/// The layout the model was built with, the rules' choices on this machine and
/// their inputs (`Forest::layout_stats`), beside the host's L1D size.
pub fn layout_stats(forest: &Forest) -> Value {
    let s = forest.layout_stats();
    json!({
        "varying_share": s.varying_share,
        "lanes": s.lanes,
        "lanes_constant": s.lanes_constant,
        "registers": s.registers,
        "vector_bits": s.vector_bits,
        "pool_bytes": s.pool_bytes,
        "leaf_bytes": s.leaf_bytes,
        "rank_table_bytes": s.rank_table_bytes,
        "flip_table_bytes": s.flip_table_bytes,
        "chain_slots": s.chain_slots,
        "chain_sets": s.chain_sets,
        "chains": s.chains,
        "packed": s.packed,
        "line_bytes": s.line_bytes,
        "l1_bytes": s.l1_bytes,
        "l2_share_bytes": s.l2_share_bytes,
        "cache_source": s.cache_source,
        "chunk_bytes": s.chunk_bytes,
        "steps_leaf": s.steps_leaf,
        "steps_stop": s.steps_stop,
        "pool_level": s.pool_level,
        "distance_constant": s.distance_constant,
        "distance": s.distance,
        "rate_constant": s.rate_constant,
        "pool_latency_l1s": s.pool_latency_l1s,
        "pool_latency_ns": s.pool_latency_ns,
        "pool_latency_ratio": s.pool_latency_ratio,
        "prefix_groups": s.prefix_groups,
        "prefix_trees": s.prefix_trees,
        "prefix_max_depth": s.prefix_max_depth,
        "prefix_mean_depth": s.prefix_mean_depth,
        "prefix_saved_steps": s.prefix_saved_steps,
        "prefix_cost_steps": s.prefix_cost_steps,
        "l1d_bytes": crate::system::l1d_bytes(),
    })
}

/// The phases' names and times in nanoseconds, in the counters table's order.
pub fn phase_fields(p: &Phases) -> [(&'static str, u64); 7] {
    let ns = |d: std::time::Duration| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
    [
        ("bins_ns", ns(p.bins)),
        ("precompute_ns", ns(p.precompute)),
        ("prefix_ns", ns(p.prefix)),
        ("walk_ns", ns(p.walk)),
        ("resolve_ns", ns(p.resolve)),
        ("leaf_pass_ns", ns(p.leaf_pass)),
        ("finish_ns", ns(p.finish)),
    ]
}

/// What a cell's no-op rules and signatures depend on.
#[derive(Debug, Clone)]
pub struct Facts {
    /// The cell declares monotonic features.
    pub monotonic: bool,
    /// The model has varying predicates: splits on a varying feature. A horizon-1
    /// survival model has none: its time features are constant in training.
    pub varying_predicates: bool,
    /// The walk lanes the rules chose for the model's walks with and without
    /// varying stops, their restart prefetch distances, and the first's rate.
    pub lanes: usize,
    pub lanes_constant: usize,
    pub distance: usize,
    pub distance_constant: usize,
    pub rate_constant: usize,
    /// Some group is a single row.
    pub single_rows: bool,
    /// The mask widths the groups of two or more rows take ([`mask_width`]), in
    /// increasing order, and the configured widest group's.
    pub mask_widths: Vec<usize>,
    pub config_width: usize,
}

/// The mask width in bits a group of `rows` rows takes: 32 for up to 32 rows, then
/// 64 to 1,024 by powers of two, then runtime widths of 128-bit units.
pub const fn mask_width(rows: usize) -> usize {
    if rows <= 32 {
        32
    } else if rows <= 1024 {
        rows.next_power_of_two()
    } else {
        rows.div_ceil(128) * 128
    }
}

impl Facts {
    pub fn of(forest: &Forest, cell: &Cell) -> Self {
        let config = forest.config();
        let rows = cell.group(0);
        let first = &rows[..rows.len().min(32 * cell.n_features)];
        let stats = forest.layout_stats();
        let sizes = cell.offsets.windows(2).map(|w| w[1] - w[0]);
        Self {
            monotonic: (0..config.n_features())
                .any(|f| config.is_increasing(f) || config.is_decreasing(f)),
            varying_predicates: !forest.predicate_masks(first, true).is_empty(),
            lanes: stats.lanes,
            lanes_constant: stats.lanes_constant,
            distance: stats.distance,
            distance_constant: stats.distance_constant,
            rate_constant: stats.rate_constant,
            single_rows: sizes.clone().any(|s| s == 1),
            mask_widths: {
                let mut widths: Vec<usize> = sizes.filter(|&s| s > 1).map(mask_width).collect();
                widths.sort_unstable();
                widths.dedup();
                widths
            },
            config_width: mask_width(config.max_group_width()),
        }
    }

    /// The mask widths the variant's masked groups take with width dispatch, in
    /// increasing order: single rows join them when they stop at varying splits.
    fn dispatched(&self, spec: &VariantSpec) -> Vec<usize> {
        let g1_off = spec.runtime.iter().any(|x| x == "disable_g1_constant");
        let mut widths = self.mask_widths.clone();
        if self.single_rows && g1_off && widths.first() != Some(&32) {
            widths.insert(0, 32);
        }
        widths
    }

    /// The mask widths the variant's masked groups take: every group takes the
    /// configured width without width dispatch.
    fn widths(&self, spec: &VariantSpec) -> Vec<usize> {
        let c = self.dispatched(spec);
        if !c.is_empty() && spec.runtime.iter().any(|x| x == "disable_width_dispatch") {
            vec![self.config_width]
        } else {
            c
        }
    }
}

/// Whether `flag` is a no-op for this variant and cell, and why.
pub fn no_op(flag: &str, spec: &VariantSpec, facts: &Facts) -> Option<&'static str> {
    let has = |f: &str| spec.runtime.iter().any(|x| x == f);
    let widths = facts.widths(spec);
    if let Some((name, value)) = flag.split_once('=') {
        return match name {
            // Forced lanes prefetch one item ahead every round.
            "lanes"
                if value.parse() == Ok(facts.lanes)
                    && facts.lanes == facts.lanes_constant
                    && (facts.distance, facts.distance_constant) == (1, 1)
                    && facts.rate_constant == 1 =>
            {
                Some("the rules choose these lanes for every walk, one item ahead")
            }
            _ => None,
        };
    }
    match flag {
        "disable_g1_constant" if !facts.single_rows => Some("no group is a single row"),
        "disable_width_dispatch" if widths.is_empty() => Some("no group takes the masked path"),
        "disable_width_dispatch" if facts.dispatched(spec) == [facts.config_width] => {
            Some("every group takes the configured width")
        }
        "disable_monotonic" if !has("disable_varying_precompute") => {
            Some("precompute is on, and it ignores monotonicity")
        }
        "disable_monotonic" if !facts.monotonic => Some("the cell has no monotonic features"),
        "disable_predicate_sweep" if has("disable_varying_precompute") => Some("precompute is off"),
        "disable_unsplit"
        | "disable_varying_precompute"
        | "disable_predicate_sweep"
        | "disable_monotonic"
            if !facts.varying_predicates =>
        {
            Some("the model has no varying predicates")
        }
        _ => None,
    }
}

/// Whether `v`, the variant's counters, show that `flag`'s path ran.
///
/// Judged by the path's own counters, never by totals differing from all-on: with
/// two identical rows, exact and per-row accumulation both make 4 leaf adds.
pub fn signature(flag: &str, v: &WorkCounters, _facts: &Facts) -> bool {
    match flag {
        // The event scans ran, and the precompute did not.
        "disable_varying_precompute" => {
            v.precompute_row_evals == 0
                && v.precompute_mask_writes == 0
                && (v.scan_rows > 0 || v.varying_events == 0)
        }
        // Empty sides were passed on, wherever an event left one.
        "disable_unsplit" => v.empty_sides > 0 || v.splits == v.varying_events,
        // The brute force wrote the masks, with no sort or sweep, unless no group
        // reads masks: single rows step through varying splits as constants.
        "disable_predicate_sweep" => {
            v.precompute_sweep_compares == 0
                && v.precompute_sort_compares == 0
                && (v.precompute_mask_writes > 0 || v.groups == v.single_rows)
        }
        // Every scan evaluated all its rows: no monotonic early stop.
        "disable_monotonic" => v.scan_evals == v.scan_rows,
        // No single row stepped through varying splits as constants.
        "disable_g1_constant" => v.single_rows == 0,
        // No counter tells these apart from all-on.
        "disable_width_dispatch" | "disable_restart_prefetch" => true,
        t => t.starts_with("lanes="),
    }
}

fn push_counters(
    t: &mut Table,
    suite: &str,
    cell: &str,
    variant: &str,
    g: usize,
    (c, p): &Counted,
) {
    let mut row = vec![
        V::S(suite),
        V::S(cell),
        V::S(variant),
        V::U32(g as u32),
        V::U32(COUNTERS_VERSION),
    ];
    row.extend(c.fields().iter().map(|&(_, v)| V::U64(v)));
    row.extend(phase_fields(p).iter().map(|&(_, v)| V::U64(v)));
    t.push(&row);
}

/// Record per-group counters for every research variant (and all-on), and check
/// each ablation's signature. Variants whose counters do not show an active flag
/// are removed from timing.
pub(super) fn counters(
    ctx: &Context,
    cell: &Cell,
    l: &mut Loaded,
    table: &mut Table,
) -> Result<BTreeMap<String, Value>> {
    let facts = Facts::of(&l.forests[""], cell);
    let mut totals: BTreeMap<(String, Vec<String>), Counted> = BTreeMap::new();
    let mut total_of =
        |forests: &BTreeMap<String, Forest>, key: &str, rt: &[String], per_group: bool| {
            let mut rt = rt.to_vec();
            rt.sort();
            let spec = VariantSpec {
                runtime: rt.clone(),
                load: Vec::new(),
            };
            if !per_group && let Some(t) = totals.get(&(key.to_string(), rt.clone())) {
                return Ok::<_, anyhow::Error>((*t, Vec::new()));
            }
            let (t, g) = count(&forests[key], spec.ablation()?, cell, per_group);
            totals.insert((key.to_string(), rt), t);
            Ok((t, g))
        };
    let (_, all_on) = total_of(&l.forests, "", &[], true)?;
    for (g, c) in all_on.iter().enumerate() {
        push_counters(table, &ctx.suite, &cell.id, "all-on", g, c);
    }
    let mut report = BTreeMap::new();
    let mut failed = vec![false; l.methods.len()];
    let variants: Vec<(usize, VariantSpec)> = (l.methods.iter().enumerate())
        .filter_map(|(i, m)| Some((i, m.spec.clone()?)))
        .collect();
    for (i, spec) in variants {
        let id = spec.id();
        let key = load_key(&spec);
        let ((v, phases), groups) = if id == "all-on" {
            (
                all_on.iter().fold(Counted::default(), |mut a, c| {
                    a.0 += c.0;
                    a.1 += c.1;
                    a
                }),
                Vec::new(),
            )
        } else {
            total_of(&l.forests, &key, &spec.runtime, true)?
        };
        for (g, c) in groups.iter().enumerate() {
            push_counters(table, &ctx.suite, &cell.id, &id, g, c);
        }
        let mut flags = BTreeMap::new();
        for flag in &spec.runtime {
            let status = match no_op(flag, &spec, &facts) {
                Some(why) => format!("no-op: {why}"),
                None if signature(flag, &v, &facts) => "active".to_string(),
                None => {
                    failed[i] = true;
                    "failed: the counters do not show this flag's path".to_string()
                }
            };
            flags.insert(flag.clone(), Value::String(status));
        }
        if failed[i] {
            l.excluded.push(Excluded {
                method: "treewalker_research".into(),
                variant: id.clone(),
                reason: "ablation self-check failed".into(),
            });
        }
        report.insert(
            id,
            json!({"flags": flags, "totals": format!("{v}"), "load": spec.load,
                   "phases_ns": phase_fields(&phases).iter().copied().collect::<BTreeMap<_, _>>()}),
        );
    }
    drop_failed(l, &failed);
    Ok(report)
}
