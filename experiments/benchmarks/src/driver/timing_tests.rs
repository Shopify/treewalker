use std::cell::RefCell;
use std::rc::Rc;

use super::load::*;
use super::processes::*;
use super::timing::*;
use super::validate::*;
use super::*;
use crate::artifacts::VariantSpec;
use crate::external::ExternalMethod;
use crate::external::Precision;
use crate::methods::{Engine, TimedMethod};
use crate::schedule::Schedule;
use treewalker_gbdt::research::{Ablation, LoadAblation, Stages};

/// Every call a fake method receives: its name, the first row's group and the
/// row count.
type Log = Rc<RefCell<Vec<(String, usize, usize)>>>;

struct Fake {
    name: String,
    log: Log,
    per_row: u64,
    /// Candidate interfaces and each one's ticks per row, when there is a choice.
    candidates: Vec<(&'static str, u64)>,
    selected: usize,
    /// Report a conversion of one tick per row.
    split: bool,
}

impl ExternalMethod for Fake {
    fn interface(&self, n: usize, multi: bool) -> &'static str {
        match self.candidates.get(self.selected) {
            Some(&(name, _)) if n > 1 || multi => name,
            _ => "fake",
        }
    }
    fn candidates(&self) -> Vec<&'static str> {
        if self.candidates.is_empty() {
            vec!["fake"]
        } else {
            self.candidates.iter().map(|c| c.0).collect()
        }
    }
    fn select(&mut self, i: usize) {
        self.selected = i;
    }
    fn precision(&self) -> Precision {
        Precision::F64
    }
    fn predict_timed(&mut self, rows: &[f64], n: usize, _: bool) -> Result<u64, String> {
        let name = format!("{}{}", self.name, self.selected);
        self.log.borrow_mut().push((name, rows[0] as usize, n));
        let per_row = self
            .candidates
            .get(self.selected)
            .map_or(self.per_row, |c| c.1);
        Ok(per_row * n as u64)
    }
    fn output(&self, n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
    fn convert_only(&mut self, _: &[f64], _: usize, _: bool) -> bool {
        self.split
    }
}

/// A cell of `sizes.len()` groups whose rows hold their group index.
pub(super) fn cell(sizes: &[usize]) -> Cell {
    let nf = 2;
    let mut offsets = vec![0];
    let mut data = Vec::new();
    for (g, &s) in sizes.iter().enumerate() {
        data.extend(std::iter::repeat_n(g as f64, s * nf));
        offsets.push(offsets.last().unwrap() + s);
    }
    Cell {
        id: "test/cell".into(),
        dir: PathBuf::new(),
        doc: Value::Null,
        framework: "test".into(),
        status: "ready".into(),
        n_rows: *offsets.last().unwrap(),
        n_features: nf,
        entities: (0..sizes.len() as i64).collect(),
        offsets,
        walker_config: PathBuf::new(),
        model_treelite: PathBuf::new(),
        model_native: None,
        reference: None,
        oracle: None,
        tl2cgen: None,
        lleaves: None,
        quickscorer: Err(String::new()),
        quickscorer_sha256: None,
        data,
    }
}

pub(super) fn context(config: RunConfig) -> Context {
    Context {
        suite: "test".into(),
        config,
        natives: Natives::default(),
        tl2cgen_runtime: None,
        only: None,
        tl2cgen_override: None,
        pmu: None,
        pmu_status: "test".into(),
        require_pmu: false,
        process: 0,
        timing_began: 0,
        interface_override: BTreeMap::new(),
        invocation: 0,
    }
}

fn fake(name: &str, log: &Log, per_row: u64) -> TimedMethod {
    choosing(name, log, per_row, vec![])
}

fn choosing(
    name: &str,
    log: &Log,
    per_row: u64,
    candidates: Vec<(&'static str, u64)>,
) -> TimedMethod {
    let m = Fake {
        name: name.into(),
        log: log.clone(),
        per_row,
        candidates,
        selected: 0,
        split: name == "q",
    };
    TimedMethod::new(name, "-", Engine::External(Box::new(m)), 64)
}

#[test]
fn extra_xgboost_processes_run_only_where_configured() {
    let log = Log::default();
    let modes = [Mode::Serving];
    let (cell, forest) = (cell(&[1]), super::tests::fixture_forest(&[]));
    let totals = BTreeMap::new();
    let shared = Shared {
        cell_dir: Path::new("."),
        cell: &cell,
        forest: &forest,
        groups: &[0],
        modes: &modes,
        max_rows: 1,
        totals: &totals,
    };
    let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
    let mut run = |config: RunConfig, name: &str| {
        let mut methods = [fake(name, &log, 1)];
        extra_processes(
            &context(config),
            &mut methods,
            &shared,
            &mut samples,
            &mut hw,
        )
    };
    // Not an XGBoost cell: nothing to record.
    assert_eq!(run(RunConfig::default(), "lightgbm_native"), Value::Null);
    let elsewhere = RunConfig {
        xgboost_process_arches: vec!["no-such-arch".into()],
        ..RunConfig::default()
    };
    let r = run(elsewhere, "xgboost_native");
    assert_eq!(r["count"], 0);
    assert!(
        r["skipped"]
            .as_str()
            .unwrap()
            .contains("not in xgboost_process_arches")
    );
    let none = RunConfig {
        xgboost_processes: 0,
        ..RunConfig::default()
    };
    assert_eq!(
        run(none, "xgboost_native")["skipped"],
        "xgboost_processes is 0"
    );
    let here = RunConfig {
        xgboost_process_arches: vec![std::env::consts::ARCH.into()],
        ..RunConfig::default()
    };
    assert_eq!(
        run(here, "xgboost_native")["skipped"],
        "no native XGBoost library"
    );
    let s = process_summary(2, "serving", Some([300, 3]), "ok");
    assert_eq!(s["ticks_per_row"].as_f64(), Some(100.0));
    assert_eq!(
        process_summary(2, "serving", None, "failed")["ticks_per_row"],
        Value::Null
    );
}

#[test]
fn the_childs_warm_up_round_follows_the_timed_batches() {
    let cell = cell(&[1; 30]);
    let config = RunConfig {
        target_batches: 4,
        ..RunConfig::default()
    };
    let ctx = context(config.clone());
    let log = Log::default();
    let mut m = fake("w", &log, 1);
    let groups: Vec<usize> = (0..30).rev().collect();
    warm_round(&ctx, &cell, &groups, &mut m, Mode::Serving);
    let seed = crate::schedule::cell_seed(config.seed, &cell.id);
    let want: Vec<usize> = Schedule::new(30, 1, 4, seed)
        .batches
        .iter()
        .flatten()
        .map(|&i| groups[i])
        .collect();
    let got: Vec<usize> = log.borrow().iter().map(|c| c.1).collect();
    assert_eq!(got, want);
}

#[test]
fn a_child_with_unread_counters_or_other_outputs_is_invalid() {
    let modes = [Mode::Serving];
    let ok = json!({"check": {"passed": true}, "pmu_status": "ok",
                    "stops": {"serving": {"failures": [], "incomplete_reads": 0}}});
    assert!(child_invalid(&ok, &modes, true).is_empty());
    let mut unread = ok.clone();
    unread["stops"]["serving"]["incomplete_reads"] = json!(2);
    assert_eq!(child_invalid(&unread, &modes, true).len(), 1);
    // Without counters in the parent, unread blocks do not count.
    assert!(child_invalid(&unread, &modes, false).is_empty());
    let mut no_pmu = ok.clone();
    no_pmu["pmu_status"] = json!("perf_event_open: EACCES");
    assert!(child_invalid(&no_pmu, &modes, true)[0].contains("no hardware counters"));
    let mut differ = ok;
    differ["check"] = json!({"passed": false, "mismatches": 3});
    assert!(child_invalid(&differ, &modes, false)[0].contains("outputs differ"));
}

#[test]
fn the_conversion_pass_follows_the_timed_passes_outside_the_samples() {
    let cell = cell(&[2; 6]);
    let config = RunConfig {
        target_batches: 3,
        min_rounds: 2,
        max_rounds: 2,
        hardware_counters: false,
        ..RunConfig::default()
    };
    let log = Log::default();
    let mut methods = vec![fake("q", &log, 10), fake("b", &log, 20)];
    let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
    let s = time_mode(
        &mut context(config),
        &cell,
        &(0..6).collect::<Vec<_>>(),
        &mut methods,
        Mode::Serving,
        &mut samples,
        &mut hw,
    );
    // One pass per round, of the method with a conversion, on the last batch:
    // 2 groups of 2 rows at 10 ticks a row, and the conversions in one interval.
    assert_eq!(s.conversion.len(), 2);
    let c = &s.conversion[0];
    assert_eq!(
        (c.method.as_str(), c.batch, c.calls, c.rows),
        ("q", 2, 2, 4)
    );
    assert_eq!(c.ticks, 40);
    assert_eq!(samples.len(), 2 * 2 * 6);
}

#[test]
fn all_one_row_serving_cells_record_the_single_row_interface() {
    let cell = cell(&[1; 6]);
    let config = RunConfig {
        min_rounds: 1,
        max_rounds: 1,
        hardware_counters: false,
        ..RunConfig::default()
    };
    let log = Log::default();
    let mut methods = vec![choosing("a", &log, 0, vec![("mat", 30), ("loop", 10)])];
    let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
    let s = time_mode(
        &mut context(config),
        &cell,
        &(0..6).collect::<Vec<_>>(),
        &mut methods,
        Mode::Serving,
        &mut samples,
        &mut hw,
    );
    assert!(s.probes.is_empty());
    assert_eq!(methods[0].chosen["serving"], "fake");
}

#[test]
fn the_probe_times_the_faster_candidate_and_records_it() {
    let cell = cell(&[1, 3, 3, 3, 3, 3]);
    let config = RunConfig {
        target_batches: 2,
        min_rounds: 2,
        max_rounds: 2,
        hardware_counters: false,
        ..RunConfig::default()
    };
    let log = Log::default();
    let mut methods = vec![
        choosing("a", &log, 0, vec![("mat", 30), ("loop", 10)]),
        fake("b", &log, 20),
    ];
    let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
    let s = time_mode(
        &mut context(config),
        &cell,
        &(0..6).collect::<Vec<_>>(),
        &mut methods,
        Mode::Serving,
        &mut samples,
        &mut hw,
    );
    // One probe, of the method with a choice, on its warm-up batch.
    assert_eq!(s.probes.len(), 1);
    let p = &s.probes[0];
    assert_eq!(
        (p.method.as_str(), p.chosen.as_str(), p.batch),
        ("a", "loop", 1)
    );
    let ticks: Vec<Option<u64>> = p.candidates.iter().map(|c| c.ticks).collect();
    // The runner-up, the multi-row call at 30 ticks a row, is 200% slower.
    assert!((p.margin_pct.unwrap() - 200.0).abs() < 1e-9);
    // Two timed passes each over the batch's rows: 30 and 10 ticks a row.
    assert_eq!(ticks, [Some(60 * p.rows as u64), Some(20 * p.rows as u64)]);
    assert_eq!(methods[0].chosen["serving"], "loop");
    assert_eq!(methods[1].chosen["serving"], "fake");
    // The probe ran each candidate three times on the warm-up batch: one warm
    // pass and two timed ones; every later call of "a" used the loop.
    let calls = log.borrow();
    let a: Vec<&String> = calls
        .iter()
        .filter(|c| c.0.starts_with('a'))
        .map(|c| &c.0)
        .collect();
    let probe_calls = 3
        * 2
        * Schedule::new(6, 2, 2, crate::schedule::cell_seed(42, "test/cell")).batches[1].len();
    assert!(a[..probe_calls].contains(&&"a0".to_string()));
    assert!(a[probe_calls..].iter().all(|n| *n == "a1"));
}

#[test]
fn a_dropped_method_takes_its_variant_spec_with_it() {
    let log = Log::default();
    let mut l = Loaded {
        methods: Vec::new(),
        excluded: Vec::new(),
        forests: BTreeMap::new(),
        load_seconds: BTreeMap::new(),
        memory: Vec::new(),
    };
    for name in ["a", "b", "c"] {
        let mut m = fake(name, &log, 1);
        m.spec = Some(VariantSpec {
            runtime: vec![name.into()],
            load: Vec::new(),
        });
        l.methods.push(m);
    }
    l.methods.push(fake("baseline", &log, 1));
    drop_failed(&mut l, &[false, true, false, false]);
    let left: Vec<(&str, Option<&str>)> = l
        .methods
        .iter()
        .map(|m| {
            let spec = m.spec.as_ref().map(|s| s.runtime[0].as_str());
            (m.method.as_str(), spec)
        })
        .collect();
    assert_eq!(
        left,
        [("a", Some("a")), ("c", Some("c")), ("baseline", None)]
    );
}

/// The import fixture `identity.bin` as a cell of `sizes` groups, with a walker
/// config wide enough for its widest group, under a temporary directory.
fn fixture_cell(sizes: &[usize], tag: &str) -> (Cell, PathBuf) {
    let dir = std::env::temp_dir().join(format!("tw-fixture-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let width = sizes.iter().copied().max().unwrap_or(1);
    let config = dir.join("walker_config.json");
    std::fs::write(
        &config,
        json!({"n_features": 2, "max_group_width": width, "varying_features": [0],
               "mono_inc_features": [], "mono_dec_features": []})
        .to_string(),
    )
    .unwrap();
    let mut c = cell(sizes);
    c.model_treelite = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/import/identity.bin"
    ));
    c.walker_config = config;
    (c, dir)
}

/// A cell's recorded `max_rank_table_bytes` reaches every load of its model: a cap
/// below the categorical fixture's rank tables rejects it through `forest`, an
/// ablated `forest` and `forest_at_width`, and without the override all three load.
#[test]
fn recorded_load_options_reach_every_load() {
    let (mut cell, _dir) = fixture_cell(&[2, 3], "load-options");
    cell.model_treelite = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/import/categories_left.bin"
    ));
    let ablated = LoadAblation {
        disable_tree_ordering: true,
        ..LoadAblation::default()
    };
    for (doc, loads) in [
        (json!({}), true),
        (json!({"load_options": {"max_rank_table_bytes": 1}}), false),
    ] {
        cell.doc = doc;
        let results = [
            cell.forest(&LoadAblation::default()).err(),
            cell.forest(&ablated).err(),
            cell.forest_at_width(4).err(),
        ];
        for (what, err) in ["forest", "ablated forest", "forest at width 4"]
            .into_iter()
            .zip(results)
        {
            match err {
                None => assert!(loads, "{what} loaded under a 1-byte cap"),
                Some(e) => assert!(
                    !loads && format!("{e:#}").contains("max_rank_table_bytes"),
                    "{what}: {e:#}"
                ),
            }
        }
    }
}

#[test]
fn the_oracle_rows_stages_are_written_in_the_oracles_order() {
    let (mut cell, dir) = fixture_cell(&[3, 5, 2], "stages");
    // Rows whose varying feature 0 differs within each group, so every row has its
    // own stages; feature 1 is constant, so the same in every row.
    for (i, v) in cell.data.iter_mut().enumerate() {
        *v = if i % 2 == 0 {
            (i % 7) as f64 - 3.0
        } else {
            0.0
        };
    }
    let forest = cell.forest(&LoadAblation::default()).unwrap();
    let mut r = forest.research_predictor(Ablation::default());
    let mut want = Vec::new();
    let mut stages = Stages::default();
    for g in [0, 2] {
        r.predict_group_stages(cell.group(g), &mut stages);
        for i in 0..stages.output.len() {
            let row = cell.offsets[g] + i;
            want.push((
                g,
                row,
                stages.tree_sum[i],
                stages.raw_margin[i],
                stages.output[i],
            ));
        }
    }
    cell.oracle = Some(crate::artifacts::Oracle {
        groups: vec![0, 2],
        rows: want
            .iter()
            .map(|&(_, row, ts, rm, _)| (row, ts, rm))
            .collect(),
        divisor: 1.0,
        base_score: 0.0,
        postprocessor: "identity".into(),
        sigmoid_alpha: 1.0,
    });
    let mut t = output::stages_table();
    let got = check_oracle(&cell, &forest, "s", &mut t).unwrap();
    assert_eq!(got["status"], "pass");
    let rows = t.rows_json();
    assert_eq!(rows.len(), 5);
    for (r, &(g, row, ts, rm, out)) in rows.iter().zip(&want) {
        assert_eq!(
            r[..4],
            [json!("s"), json!("test/cell"), json!(g), json!(row)]
        );
        for (v, w) in r[4..].iter().zip([ts, rm, out]) {
            assert_eq!(v.as_f64().unwrap().to_bits(), w.to_bits());
        }
    }
    t.write(&dir.join("stages.parquet")).unwrap();
    // No oracle, no rows: the table is still written, empty.
    cell.oracle = None;
    let mut t = output::stages_table();
    assert_eq!(
        check_oracle(&cell, &forest, "s", &mut t).unwrap()["status"],
        "no oracle"
    );
    assert_eq!(t.len(), 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_cell_that_fails_before_timing_does_not_count_as_timed() {
    let mut ctx = context(RunConfig::default());
    let mc = ManifestCell {
        id: "x/y".into(),
        dir: PathBuf::from("no/such/cell"),
        status: "ready".into(),
        key: "k".into(),
        methods: "factorial".into(),
        variants: Vec::new(),
        modes: vec!["serving".into()],
    };
    let tmp = std::env::temp_dir().join(format!("tw-notimed-{}", std::process::id()));
    assert!(run_cell(&mut ctx, &mc, Path::new("/no/such/cell"), &tmp, "", true).is_err());
    assert_eq!(ctx.timing_began, 0);
}

#[test]
fn the_minimum_rounds_bind_before_the_budget() {
    let cell = cell(&[1; 6]);
    for (precision_pct, reason) in [(0.0, "time_budget"), (1e9, "precision")] {
        let config = RunConfig {
            min_rounds: 3,
            max_rounds: 10,
            mode_budget_secs: 0.0,
            precision_pct,
            hardware_counters: false,
            ..RunConfig::default()
        };
        let log = Log::default();
        let mut methods = vec![fake("a", &log, 10), fake("b", &log, 20)];
        let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
        let s = time_mode(
            &mut context(config),
            &cell,
            &(0..6).collect::<Vec<_>>(),
            &mut methods,
            Mode::Serving,
            &mut samples,
            &mut hw,
        );
        // A spent budget and a met target both wait for the third round, an
        // odd one: stopping is not held to the A B B A cycle.
        assert_eq!((s.rounds, s.reason), (3, reason));
    }
}

#[test]
fn every_method_times_the_same_group_sample() {
    let sizes: Vec<usize> = (0..40).map(|g| 1 + g % 3).collect();
    let cell = cell(&sizes);
    let sample = crate::schedule::sample_groups(&sizes, 20, 0, 5).unwrap();
    let config = RunConfig {
        min_rounds: 2,
        max_rounds: 2,
        hardware_counters: false,
        ..RunConfig::default()
    };
    let log = Log::default();
    let mut methods = vec![fake("a", &log, 10), fake("b", &log, 20)];
    let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
    time_mode(
        &mut context(config),
        &cell,
        &sample.groups,
        &mut methods,
        Mode::Serving,
        &mut samples,
        &mut hw,
    );
    for name in ["a", "b"] {
        let mut seen: Vec<usize> = log
            .borrow()
            .iter()
            .filter(|(m, _, _)| *m == format!("{name}0"))
            .map(|&(_, g, _)| g)
            .collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen, sample.groups, "{name}");
    }
}

#[test]
fn each_method_times_every_batch_in_its_own_phase() {
    let cell = cell(&[2; 10]);
    let config = RunConfig {
        target_batches: 3,
        min_rounds: 4,
        max_rounds: 4,
        precision_pct: 0.0,
        hardware_counters: false,
        ..RunConfig::default()
    };
    let mut ctx = context(config.clone());
    let log = Log::default();
    let mut methods = vec![fake("a", &log, 10), fake("b", &log, 20)];
    let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
    let groups: Vec<usize> = (0..10).collect();
    let s = time_mode(
        &mut ctx,
        &cell,
        &groups,
        &mut methods,
        Mode::Serving,
        &mut samples,
        &mut hw,
    );
    assert_eq!((s.rounds, s.blocks_per_round), (4, 3));
    let schedule = Schedule::new(
        10,
        2,
        3,
        crate::schedule::cell_seed(config.seed, "test/cell"),
    );
    let batch = |b: usize| schedule.batches[b].clone();
    // Each phase: the warm-up on the last batch, then batches 0, 1, 2.
    let mut want = Vec::new();
    for round in 0..4 {
        for mi in schedule.method_order(round) {
            let name = ["a", "b"][mi];
            let warm = batch(schedule.warm_batch());
            for b in std::iter::once(warm).chain((0..3).map(batch)) {
                want.extend(b.iter().map(|&g| (format!("{name}0"), g, 2)));
            }
        }
    }
    assert_eq!(*log.borrow(), want);
    // A B B A across rounds, and every method times every group every round.
    let order: Vec<Vec<usize>> = (0..4).map(|r| schedule.method_order(r)).collect();
    assert_eq!(order[0], order[1].iter().rev().copied().collect::<Vec<_>>());
    assert_eq!(order[0], order[2]);
    assert_eq!(samples.len(), 4 * 2 * 10);
}

#[test]
fn a_cell_with_a_row_order_is_timed_in_file_order() {
    let mut cell = cell(&[1; 10]);
    cell.doc = json!({"row_order": "independent"});
    let config = RunConfig {
        target_batches: 3,
        min_rounds: 2,
        max_rounds: 2,
        precision_pct: 0.0,
        hardware_counters: false,
        ..RunConfig::default()
    };
    let mut ctx = context(config);
    let log = Log::default();
    let mut methods = vec![fake("a", &log, 10), fake("b", &log, 20)];
    let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
    let groups: Vec<usize> = (0..10).collect();
    time_mode(
        &mut ctx,
        &cell,
        &groups,
        &mut methods,
        Mode::Serving,
        &mut samples,
        &mut hw,
    );
    // Each phase warms up on the last batch, groups 7 to 9, then times every
    // group in order; both methods, both rounds.
    assert_eq!(cell.group_order(), crate::schedule::GroupOrder::File);
    let calls = log.borrow();
    let phases: Vec<Vec<usize>> = calls
        .chunks(13)
        .map(|p| p.iter().map(|c| c.1).collect())
        .collect();
    assert_eq!(phases.len(), 4);
    for p in &phases {
        assert_eq!(*p, [7, 8, 9].into_iter().chain(0..10).collect::<Vec<_>>());
    }
}
