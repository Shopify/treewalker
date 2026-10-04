//! Minimal driver for instruction-level profiling (valgrind/cachegrind) of one kernel.
//! probe <model_dir> <data_dir> <n_groups> <reps> <variant>
use std::path::PathBuf;
use treewalker_gbdt::ParseConfig;
use treewalker_gbdt::opt::{LeafMode, OptFlags, OptWorkspace, RunWorkspace};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (md, dd) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]));
    let n_groups: usize = a[3].parse().unwrap();
    let reps: usize = a[4].parse().unwrap();
    let variant = a[5].as_str();
    let sched: u128 = a.get(6).map_or(0, |s| {
        s.split(',')
            .filter(|t| !t.is_empty())
            .fold(0u128, |m, t| m | 1u128 << t.parse::<u32>().unwrap())
    });
    let model = if md.join("model_treelite.bin").exists() {
        md.join("model_treelite.bin")
    } else {
        md.join("model_treelite.json")
    };
    let (data, n_rows, _nc) = treewalker_bench::load_raw_f64(dd.join("test_data.bin"));
    let (mut f, g) = treewalker_bench::wide::load_forest_any_width(
        &model,
        &dd.join("walker_config.json"),
        &ParseConfig::default(),
    )
    .expect("load forest");
    let groups: Vec<(usize, usize)> = (0..n_rows / g)
        .skip(1)
        .step_by(2)
        .take(n_groups)
        .map(|k| (k * g, (k + 1) * g))
        .collect();
    if variant.starts_with("hint") || variant == "flip" {
        f.add_child_hints();
    }
    if variant == "flip" || variant == "exact" || variant.starts_with("runs") {
        if variant != "flip" {
            f.add_child_hints();
        }
        f.flip_transform();
    }
    let ee = f.exact_scale().unwrap_or(0);
    let mut rws = RunWorkspace::new(&f, g);
    if variant == "runs2" || variant == "runs3" {
        rws.fast = true;
        rws.schedule = sched;
        rws.uninit = variant == "runs3";
    }
    let mut res = vec![0.0f64; n_rows];
    let mut ws = OptWorkspace::new(&f);
    let leaf = if cfg!(target_feature = "avx512f") {
        LeafMode::Avx512
    } else {
        LeafMode::Halves
    };
    let flags = match variant {
        "best" => OptFlags {
            leaf,
            schedule_features: sched,
            ..OptFlags::default()
        },
        "opt0" | "base" => OptFlags::default(),
        "catfree" => OptFlags {
            leaf,
            schedule_features: sched,
            lockstep_k: 5,
            ..OptFlags::default()
        },
        "flip" => OptFlags {
            leaf,
            schedule_features: sched,
            lockstep_k: 7,
            ..OptFlags::default()
        },
        "exact" | "runs" | "runs2" | "runs3" => OptFlags {
            leaf: LeafMode::Exact,
            schedule_features: if g > 128 { 0 } else { sched },
            lockstep_k: 7,
            ..OptFlags::default()
        },
        "hint2" => OptFlags {
            leaf,
            schedule_features: sched,
            lockstep_k: 6,
            ..OptFlags::default()
        },
        "hint" => OptFlags {
            leaf,
            schedule_features: sched,
            lockstep_k: 3,
            ..OptFlags::default()
        },
        v => panic!("unknown variant {v}"),
    };
    let mut acc = 0.0;
    for _ in 0..reps {
        for &(s, e) in &groups {
            if variant == "base" {
                for c in (s..e).step_by(128) {
                    f.predict(&data, &mut res, c, (c + 128).min(e))
                }
            } else if variant.starts_with("runs") {
                assert!(f.predict_runs(&mut rws, &data, &mut res, s, e, ee));
            } else {
                let fl = OptFlags {
                    exact_e: ee,
                    ..flags
                };
                for c in (s..e).step_by(128) {
                    f.predict_opt(&mut ws, &data, &mut res, c, (c + 128).min(e), &fl)
                }
            }
            acc += res[s];
        }
    }
    println!("{acc}");
}
