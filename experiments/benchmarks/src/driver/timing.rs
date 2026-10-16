//! Timing one mode: the probe, the rounds of timed passes, the conversion pass.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::Result;

use crate::artifacts::Cell;
use crate::methods::TimedMethod;
use crate::output::{Table, V};
use crate::pmu::Pmu;
use crate::schedule::Schedule;

use super::load::{Call, Gathered, calls_for};
use super::{Context, Excluded};

/// The mode of a timing phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Serving,
    Batch,
}

impl Mode {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Serving => "serving",
            Self::Batch => "batch",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "serving" => Ok(Self::Serving),
            "batch" => Ok(Self::Batch),
            other => anyhow::bail!("unknown mode {other}; expected serving or batch"),
        }
    }

    /// A suite's modes, in order, each once; at least one.
    pub fn parse_list(names: &[String]) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        for n in names {
            let m = Self::parse(n)?;
            if !out.contains(&m) {
                out.push(m);
            }
        }
        anyhow::ensure!(!out.is_empty(), "no timing modes");
        Ok(out)
    }
}

/// Why a mode stopped, and how much it timed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Stopped {
    pub reason: &'static str,
    pub rounds: usize,
    pub blocks: usize,
    pub blocks_per_round: usize,
    pub seconds: f64,
    /// Relative standard error of each method's per-round mean ticks per row, in
    /// percent, at the stop.
    pub rse_pct: BTreeMap<String, f64>,
    /// Each method's timed ticks and rows over every round, by `method/variant`.
    pub totals: BTreeMap<String, [u64; 2]>,
    /// Methods that failed during timing; their samples up to the failure are kept.
    pub failures: Vec<Excluded>,
    /// Blocks whose counter group was multiplexed or could not be read.
    pub incomplete_reads: usize,
    /// Rounds per order cycle: forward, then reversed. A mode can stop after any
    /// round from `min_rounds` on, so an odd round count leaves the positions
    /// balanced only up to the last round.
    pub cycle: usize,
    /// Each method's interface probe, where it had a choice.
    pub probes: Vec<Probe>,
    /// The input-conversion passes of the methods that convert outside their
    /// library (QuickScorer's f64-to-f32 copy), one per round.
    pub conversion: Vec<Conversion>,
}

/// One input-conversion pass over the batch the method timed last.
///
/// Right after its timed passes: the calls' conversions alone, back to back in one interval
/// (amortized: one timer read pair for the batch), against a headline-style pass
/// over the same calls. Neither enters the samples.
#[derive(Debug, Clone, serde::Serialize)]
#[expect(
    clippy::struct_field_names,
    reason = "the fields are the record's keys"
)]
pub struct Conversion {
    pub method: String,
    pub round: usize,
    pub batch: usize,
    pub calls: usize,
    pub rows: usize,
    pub ticks: u64,
    pub conversion_ticks: u64,
}

/// The conversion pass of a method over one batch: calls, rows, the ticks of a
/// headline-style pass, and the ticks of the same calls' conversions alone, all in
/// one interval, so the timer's overhead is one read pair over the batch. `None`
/// when the method has no conversion of its own, or a call failed.
fn conversion_pass(
    m: &mut TimedMethod,
    g: &Gathered,
    calls: Option<&[Call]>,
    nf: usize,
) -> Option<(usize, usize, u64, u64)> {
    let list: Vec<(&[f64], usize, bool)> = calls.map_or_else(
        || {
            (0..g.offsets.len() - 1)
                .map(|i| (g.group(i, nf), g.offsets[i + 1] - g.offsets[i], false))
                .collect()
        },
        |calls| {
            let call = |c: &Call| {
                (
                    &g.rows[c.rows.clone()],
                    *c.offsets.last().unwrap_or(&0),
                    true,
                )
            };
            calls.iter().map(call).collect()
        },
    );
    let (first, n, multi) = *list.first()?;
    if !m.convert(first, n, multi) {
        return None;
    }
    let t0 = crate::timer::start();
    let mut ok = true;
    for &(rows, n, multi) in &list {
        ok &= m.convert(rows, n, multi);
    }
    let conversion = crate::timer::stop().wrapping_sub(t0);
    let mut ticks = 0u64;
    run_pass(m, g, calls, nf, |_, _, _, t| ticks += t).ok()?;
    let rows = list.iter().map(|l| l.1).sum();
    ok.then_some((list.len(), rows, ticks, conversion))
}

fn rse_pct(per_round: &[f64]) -> f64 {
    let n = per_round.len() as f64;
    if per_round.len() < 2 {
        return f64::INFINITY;
    }
    let mean = per_round.iter().sum::<f64>() / n;
    let var = per_round.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (var / n).sqrt() / mean * 100.0
}

/// One mode's timing over the given groups: every round gives each method its own
/// phase, a warm-up on the batch it times last and then a timed pass over every
/// batch, in the schedule's method order.
#[expect(clippy::too_many_lines, reason = "the rounds and phases of one mode")]
pub(super) fn time_mode(
    ctx: &mut Context,
    cell: &Cell,
    groups: &[usize],
    methods: &mut [TimedMethod],
    mode: Mode,
    samples: &mut Table,
    hw: &mut Table,
) -> Stopped {
    let cfg = ctx.config.clone();
    let nf = cell.n_features;
    // Seeded per cell, so the method order's position advantage differs by cell.
    let seed = crate::schedule::cell_seed(cfg.seed, &cell.id);
    let schedule = Schedule::ordered(
        groups.len(),
        methods.len(),
        cfg.target_batches,
        seed,
        cell.group_order(),
    );
    let bpr = schedule.blocks_per_round();
    let batch_groups =
        |b: usize| -> Vec<usize> { schedule.batches[b].iter().map(|&i| groups[i]).collect() };
    let gather = |b: usize| {
        let g = Gathered::new(cell, &batch_groups(b));
        let calls = (mode == Mode::Batch).then(|| calls_for(&g, nf, cfg.batch_rows));
        (g, calls)
    };
    let mut alive = vec![true; methods.len()];
    let mut failures = Vec::new();
    let mut per_round: Vec<Vec<f64>> = vec![Vec::new(); methods.len()];
    let started = Instant::now();
    let mut sample_id = samples.len() as u64;
    let (mut rounds, mut blocks, mut reason) = (0, 0, "block_cap");
    let cycle = schedule.cycle();
    // The minimum binds first: neither the precision target nor the budget stops a
    // mode before it.
    let min_rounds = cfg.min_rounds.max(1);
    let max_rounds = cfg.max_rounds.max(min_rounds);
    let mut incomplete_reads = 0;
    let mut probes = Vec::new();
    let mut conversion = Vec::new();
    let mut totals = vec![[0u64; 2]; methods.len()];
    // Serving calls of one row always use the single-row interface, so only cells
    // with wider groups, and batch mode, have a choice to probe.
    let multi_row = mode == Mode::Batch
        || groups
            .iter()
            .any(|&g| cell.offsets[g + 1] - cell.offsets[g] > 1);
    for round in 0..max_rounds {
        let mut round_ticks = vec![0u64; methods.len()];
        let mut round_rows = vec![0u64; methods.len()];
        for (pos, mi) in schedule.method_order(round).into_iter().enumerate() {
            if !alive[mi] {
                continue;
            }
            let m = &mut methods[mi];
            let (method, variant) = (m.method.clone(), m.variant.clone());
            let (wg, wcalls) = gather(schedule.warm_batch());
            // The mode's first phase of each method chooses its interface: every
            // candidate runs on the warm-up batch, and the faster is timed in every
            // round.
            if round == 0 {
                let names = m.candidates(mode == Mode::Batch);
                let fixed = ctx
                    .interface_override
                    .get(&m.method)
                    .and_then(|by_mode| by_mode.get(mode.name()))
                    .and_then(|want| names.iter().position(|n| n == want))
                    .filter(|i| m.allowed.contains(i));
                let choice = if let Some(i) = fixed {
                    Ok(i)
                } else if multi_row && m.allowed.len() > 1 {
                    probe(m, &wg, wcalls.as_deref(), nf, schedule.warm_batch()).map(|p| {
                        let i = p.index;
                        probes.push(p);
                        i
                    })
                } else {
                    m.allowed
                        .first()
                        .copied()
                        .ok_or_else(|| "no interface".into())
                };
                match choice {
                    Ok(i) => {
                        m.select(i);
                        // All-1-row serving cells only ever use the single-row call.
                        let iface = if multi_row {
                            m.interface(2, mode == Mode::Batch)
                        } else {
                            m.interface(1, false)
                        };
                        m.chosen.insert(mode.name().into(), iface);
                    }
                    Err(e) => {
                        alive[mi] = false;
                        failures.push(Excluded {
                            method,
                            variant,
                            reason: format!("probe: {e}"),
                        });
                        continue;
                    }
                }
            }
            let (iface_one, iface_multi) =
                (m.interface(1, false), m.interface(2, mode == Mode::Batch));
            // Warm up, untimed, on the batch this phase times last.
            let warm = run_pass(m, &wg, wcalls.as_deref(), nf, |_, _, _, _| {});
            drop(wg);
            if let Err(e) = warm {
                alive[mi] = false;
                failures.push(Excluded {
                    method,
                    variant,
                    reason: e,
                });
                continue;
            }
            for b in 0..bpr {
                let block = round * bpr + b;
                let batch = batch_groups(b);
                let (g, calls) = gather(b);
                let marked = ctx.pmu.as_mut().map(Pmu::mark);
                let result = run_pass(m, &g, calls.as_deref(), nf, |first, count, rows, ticks| {
                    let iface = if mode == Mode::Serving && rows == 1 {
                        iface_one
                    } else {
                        iface_multi
                    };
                    samples.push(&[
                        V::U64(sample_id),
                        V::S(&ctx.suite),
                        V::S(&cell.id),
                        V::S(&variant),
                        V::S(&method),
                        V::S(mode.name()),
                        V::S(iface),
                        V::U32(block as u32),
                        V::U32(b as u32),
                        V::U32(pos as u32),
                        V::U32(round as u32),
                        V::U32(ctx.process),
                        V::U32(batch[first] as u32),
                        V::U32(count as u32),
                        V::U32(rows as u32),
                        V::U64(ticks),
                    ]);
                    sample_id += 1;
                    round_ticks[mi] += ticks;
                    totals[mi][0] += ticks;
                    round_rows[mi] += rows as u64;
                    totals[mi][1] += rows as u64;
                });
                let reading = match (marked, ctx.pmu.as_mut()) {
                    (Some(Ok(())), Some(p)) => Some(p.since_mark()),
                    (Some(Err(e)), _) => Some(Err(e)),
                    _ => None,
                };
                if matches!(&reading, Some(Err(_)))
                    || matches!(&reading, Some(Ok(r)) if !r.complete())
                {
                    incomplete_reads += 1;
                }
                push_hw(hw, ctx, &cell.id, &variant, &method, mode, block, reading);
                if let Err(e) = result {
                    alive[mi] = false;
                    failures.push(Excluded {
                        method: method.clone(),
                        variant: variant.clone(),
                        reason: e,
                    });
                    break;
                }
            }
            // QuickScorer's f32 copy, timed on its own over the batch just timed
            // last, outside the samples.
            if alive[mi] {
                let wb = schedule.warm_batch();
                let (g, calls) = gather(wb);
                if let Some((n_calls, rows, ticks, conversion_ticks)) =
                    conversion_pass(m, &g, calls.as_deref(), nf)
                {
                    conversion.push(Conversion {
                        method: method.clone(),
                        round,
                        batch: wb,
                        calls: n_calls,
                        rows,
                        ticks,
                        conversion_ticks,
                    });
                }
            }
        }
        blocks += bpr;
        rounds = round + 1;
        for mi in 0..methods.len() {
            if alive[mi] && round_rows[mi] > 0 {
                per_round[mi].push(round_ticks[mi] as f64 / round_rows[mi] as f64);
            }
        }
        if rounds < min_rounds {
            continue;
        }
        // The relative standard error needs two rounds.
        let precise = (0..methods.len())
            .filter(|&mi| alive[mi])
            .all(|mi| rse_pct(&per_round[mi]) < cfg.precision_pct);
        if precise {
            reason = "precision";
            break;
        }
        if started.elapsed().as_secs_f64() >= cfg.mode_budget_secs && rounds < max_rounds {
            reason = "time_budget";
            break;
        }
    }
    Stopped {
        reason,
        rounds,
        blocks,
        blocks_per_round: bpr,
        seconds: started.elapsed().as_secs_f64(),
        rse_pct: methods
            .iter()
            .enumerate()
            .map(|(mi, m)| {
                (
                    format!("{}/{}", m.method, m.variant),
                    rse_pct(&per_round[mi]),
                )
            })
            .collect(),
        totals: methods
            .iter()
            .zip(&totals)
            .map(|(m, t)| (format!("{}/{}", m.method, m.variant), *t))
            .collect(),
        failures,
        incomplete_reads,
        cycle,
        probes,
        conversion,
    }
}

/// The interface probe of one method on one batch.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Probe {
    pub method: String,
    pub variant: String,
    /// The warm-up batch, by index.
    pub batch: usize,
    pub rows: usize,
    pub candidates: Vec<ProbeCandidate>,
    /// How much slower the runner-up was than the chosen, in percent: below the
    /// probe's noise (about 1-2%) the choice is a coin toss between near-equals.
    pub margin_pct: Option<f64>,
    /// The candidate timed in the mode.
    pub chosen: String,
    #[serde(skip)]
    pub index: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProbeCandidate {
    pub interface: String,
    /// Ticks of its two timed passes; `None` when a call failed.
    pub ticks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Probe a method's validated candidates on one batch, untimed for the mode: each
/// runs one warm pass, then timed passes in order and in reverse order (A B B A).
/// The candidate with the fewest ticks is chosen. An error when every one fails.
fn probe(
    m: &mut TimedMethod,
    g: &Gathered,
    calls: Option<&[Call]>,
    nf: usize,
    batch: usize,
) -> Result<Probe, String> {
    let names = m.candidates(calls.is_some());
    let allowed = m.allowed.clone();
    let mut ticks: Vec<Option<u64>> = vec![Some(0); allowed.len()];
    let mut errors: Vec<Option<String>> = vec![None; allowed.len()];
    let mut pass = |m: &mut TimedMethod, k: usize, timed: bool, ticks: &mut [Option<u64>]| {
        if ticks[k].is_none() {
            return;
        }
        m.select(allowed[k]);
        let mut sum = 0u64;
        match run_pass(m, g, calls, nf, |_, _, _, t| sum += t) {
            Ok(()) if timed => ticks[k] = ticks[k].map(|t| t + sum),
            Ok(()) => {}
            Err(e) => {
                ticks[k] = None;
                errors[k] = Some(e);
            }
        }
    };
    for k in 0..allowed.len() {
        pass(m, k, false, &mut ticks);
    }
    for k in (0..allowed.len()).chain((0..allowed.len()).rev()) {
        pass(m, k, true, &mut ticks);
    }
    let best = (0..allowed.len())
        .filter_map(|k| ticks[k].map(|t| (t, k)))
        .min()
        .map(|(_, k)| k);
    let candidates = (0..allowed.len())
        .map(|k| ProbeCandidate {
            interface: names[allowed[k]].into(),
            ticks: ticks[k],
            error: errors[k].clone(),
        })
        .collect();
    let Some(best) = best else {
        return Err(format!("every candidate failed: {errors:?}"));
    };
    Ok(Probe {
        method: m.method.clone(),
        variant: m.variant.clone(),
        batch,
        rows: *g.offsets.last().unwrap_or(&0),
        margin_pct: (0..allowed.len())
            .filter(|&k| k != best)
            .filter_map(|k| ticks[k])
            .min()
            .zip(ticks[best])
            .map(|(second, first)| (second as f64 / first.max(1) as f64 - 1.0) * 100.0),
        candidates,
        chosen: names[allowed[best]].into(),
        index: allowed[best],
    })
}

/// One pass of a method over a block: one call per group, or the batch calls.
/// `record(first, count, rows, ticks)` receives each sample.
fn run_pass(
    m: &mut TimedMethod,
    g: &Gathered,
    calls: Option<&[Call]>,
    nf: usize,
    mut record: impl FnMut(usize, usize, usize, u64),
) -> Result<(), String> {
    match calls {
        None => {
            for i in 0..g.offsets.len() - 1 {
                let rows = g.offsets[i + 1] - g.offsets[i];
                let ticks = m.serve(g.group(i, nf), rows)?;
                record(i, 1, rows, ticks);
            }
        }
        Some(calls) => {
            for c in calls {
                let ticks = m.batch(&g.rows[c.rows.clone()], &c.offsets)?;
                record(c.first, c.count, *c.offsets.last().unwrap_or(&0), ticks);
            }
        }
    }
    Ok(())
}

#[expect(clippy::too_many_arguments, reason = "one row of the hw table")]
fn push_hw(
    hw: &mut Table,
    ctx: &Context,
    cell: &str,
    variant: &str,
    method: &str,
    mode: Mode,
    block: usize,
    reading: Option<Result<crate::pmu::Reading, String>>,
) {
    let mut row = vec![
        V::S(&ctx.suite),
        V::S(cell),
        V::S(variant),
        V::S(method),
        V::S(mode.name()),
        V::U32(block as u32),
        V::U32(ctx.process),
    ];
    let status = match reading {
        Some(Ok(r)) => {
            row.extend(r.values.iter().map(|v| V::Opt(*v)));
            row.extend([
                V::Opt(Some(r.time_enabled)),
                V::Opt(Some(r.time_running)),
                V::Opt(Some(r.rusage_nvcsw)),
                V::Opt(Some(r.rusage_nivcsw)),
            ]);
            if r.complete() { "ok" } else { "multiplexed" }.to_string()
        }
        Some(Err(e)) => {
            row.extend([V::Opt(None); 12]);
            format!("read failed: {e}")
        }
        None => {
            row.extend([V::Opt(None); 12]);
            format!("unsupported: {}", ctx.pmu_status)
        }
    };
    row.push(V::S(&status));
    hw.push(&row);
}

/// One untimed pass of a method over every batch of the mode's schedule, in the
/// timed rounds' batch order (the schedule is seeded per cell).
#[cfg_attr(
    all(not(feature = "external-bench"), not(test)),
    expect(dead_code, reason = "only XGBoost's extra processes warm a round up")
)]
pub(super) fn warm_round(
    ctx: &Context,
    cell: &Cell,
    groups: &[usize],
    m: &mut TimedMethod,
    mode: Mode,
) {
    let cfg = &ctx.config;
    let seed = crate::schedule::cell_seed(cfg.seed, &cell.id);
    let schedule = Schedule::ordered(
        groups.len(),
        1,
        cfg.target_batches,
        seed,
        cell.group_order(),
    );
    for batch in &schedule.batches {
        let ids: Vec<usize> = batch.iter().map(|&i| groups[i]).collect();
        let g = Gathered::new(cell, &ids);
        let calls = (mode == Mode::Batch).then(|| calls_for(&g, cell.n_features, cfg.batch_rows));
        let _ = run_pass(m, &g, calls.as_deref(), cell.n_features, |_, _, _, _| {});
    }
}
