//! Paired benchmark for the run-list kernel against chunked bitmask kernels.
//! run_bench <model_dir> <data_dir> [--blocks N] [--batch N] [--sched i,j] [--variants a,b]
use std::path::PathBuf;
use std::time::Instant;
use treewalker_gbdt::ParseConfig;
use treewalker_gbdt::forest::Forest;
use treewalker_gbdt::opt::{LeafMode, OptFlags, OptWorkspace, RunWorkspace};

enum R {
    Base,
    Opt(OptFlags, Box<OptWorkspace>),
    Runs(Box<RunWorkspace>, OptFlags, Box<OptWorkspace>),
}

fn rws(fb: &Forest, g: usize, knock: u8, fast: bool, sched: u128) -> Box<RunWorkspace> {
    let mut w = RunWorkspace::new(fb, g);
    w.knock = knock;
    w.fast = fast;
    w.schedule = sched;
    Box::new(w)
}

fn chunks(s: usize, e: usize) -> impl Iterator<Item = (usize, usize)> {
    (s..e).step_by(128).map(move |c| (c, (c + 128).min(e)))
}

fn run(
    r: &mut R,
    fa: &mut Forest,
    fb: &Forest,
    data: &[f64],
    res: &mut [f64],
    s: usize,
    e: usize,
    ee: i32,
) {
    match r {
        R::Base => {
            for (a, b) in chunks(s, e) {
                fa.predict(data, res, a, b)
            }
        }
        R::Opt(f, ws) => {
            for (a, b) in chunks(s, e) {
                fb.predict_opt(ws, data, res, a, b, f)
            }
        }
        R::Runs(rws, f, ws) => {
            if !fb.predict_runs(rws, data, res, s, e, ee) {
                for (a, b) in chunks(s, e) {
                    fb.predict_opt(ws, data, res, a, b, f)
                }
            }
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}
fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (md, dd) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]));
    let (mut blocks, mut batch, mut sched, mut names) = (
        21usize,
        16usize,
        0u128,
        "base,flip,flipexact,runs".to_string(),
    );
    let mut i = 3;
    while i < a.len() {
        match a[i].as_str() {
            "--blocks" => blocks = a[i + 1].parse().unwrap(),
            "--batch" => batch = a[i + 1].parse().unwrap(),
            "--sched" => {
                sched = a[i + 1]
                    .split(',')
                    .filter(|t| !t.is_empty())
                    .fold(0, |m, t| m | 1u128 << t.parse::<u32>().unwrap())
            }
            "--variants" => names = a[i + 1].clone(),
            o => panic!("unknown arg {o}"),
        }
        i += 2;
    }
    let model = if md.join("model_treelite.bin").exists() {
        md.join("model_treelite.bin")
    } else {
        md.join("model_treelite.json")
    };
    let cfg = dd.join("walker_config.json");
    let (data, n_rows, _nc) = treewalker_bench::load_raw_f64(dd.join("test_data.bin"));
    // Cells wider than 128 rows load with max_group_width clamped to 128 (predict() is
    // chunked); `g` is the cell's actual group width.
    let load = || {
        treewalker_bench::wide::load_forest_any_width(&model, &cfg, &ParseConfig::default())
            .expect("load forest")
    };
    let (mut fa, g) = load();
    let (mut fb, _) = load();
    fb.add_child_hints();
    fb.flip_transform();
    let ee = fb.exact_scale().expect("exact scale");
    // Schedule-mask caching is keyed on chunk width, which is wrong once a group is chunked.
    // The run kernel never chunks, so it can always cache schedule features.
    let sched_all = sched;
    let sched = if g > 128 { 0 } else { sched };
    let groups: Vec<(usize, usize)> = (0..n_rows / g)
        .skip(1)
        .step_by(2)
        .map(|k| (k * g, (k + 1) * g))
        .collect();
    let d = OptFlags::default();
    let flip = OptFlags {
        leaf: LeafMode::Avx512,
        lockstep_k: 7,
        schedule_features: sched,
        ..d
    };
    let fexact = OptFlags {
        leaf: LeafMode::Exact,
        lockstep_k: 7,
        schedule_features: sched,
        exact_e: ee,
        ..d
    };
    let mut vars: Vec<(String, R)> = names
        .split(',')
        .map(|n| {
            (
                n.to_string(),
                match n {
                    "base" => R::Base,
                    "flip" => R::Opt(flip, Box::new(OptWorkspace::new(&fb))),
                    "flipexact" => R::Opt(fexact, Box::new(OptWorkspace::new(&fb))),
                    "runs" => R::Runs(
                        rws(&fb, g, 0, false, 0),
                        fexact,
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "runs2" => R::Runs(
                        rws(&fb, g, 0, true, sched_all),
                        fexact,
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "runs3" => R::Runs(
                        {
                            let mut w = rws(&fb, g, 0, true, sched_all);
                            w.uninit = true;
                            w
                        },
                        fexact,
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "runs3_noleaf" => R::Runs(
                        {
                            let mut w = rws(&fb, g, 2, true, sched_all);
                            w.uninit = true;
                            w
                        },
                        fexact,
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "stackexact" => R::Opt(
                        OptFlags {
                            lockstep_k: 9,
                            ..fexact
                        },
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "stackexact_u" => R::Opt(
                        OptFlags {
                            lockstep_k: 10,
                            ..fexact
                        },
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "stackflip" => R::Opt(
                        OptFlags {
                            lockstep_k: 9,
                            ..flip
                        },
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "stackflip_u" => R::Opt(
                        OptFlags {
                            lockstep_k: 10,
                            ..flip
                        },
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "runs2_pre" => R::Runs(
                        rws(&fb, g, 1, true, sched_all),
                        fexact,
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "runs2_noleaf" => R::Runs(
                        rws(&fb, g, 2, true, sched_all),
                        fexact,
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "runs2_noepi" => R::Runs(
                        rws(&fb, g, 3, true, sched_all),
                        fexact,
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "runs_pre" => R::Runs(
                        rws(&fb, g, 1, false, 0),
                        fexact,
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "runs_noleaf" => R::Runs(
                        rws(&fb, g, 2, false, 0),
                        fexact,
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "runs_noepi" => R::Runs(
                        rws(&fb, g, 3, false, 0),
                        fexact,
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "exact_noleaf" => R::Opt(
                        OptFlags {
                            leaf: LeafMode::Noop,
                            ..fexact
                        },
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    "exact_pre" => R::Opt(
                        OptFlags {
                            skip_trees: true,
                            ..fexact
                        },
                        Box::new(OptWorkspace::new(&fb)),
                    ),
                    o => panic!("unknown variant {o}"),
                },
            )
        })
        .collect();
    assert_eq!(vars[0].0, "base");

    // Exactness: every variant on every measured group.
    let mut outs: Vec<Vec<f64>> = Vec::new();
    for (_, r) in vars.iter_mut() {
        let mut res = vec![0.0f64; n_rows];
        for &(s, e) in &groups {
            run(r, &mut fa, &fb, &data, &mut res, s, e, ee);
        }
        outs.push(res);
    }
    let cmp = |x: &Vec<f64>, y: &Vec<f64>| -> (usize, f64) {
        let mut nd = 0;
        let mut md = 0.0f64;
        for &(s, e) in &groups {
            for r in s..e {
                if x[r].to_bits() != y[r].to_bits() {
                    nd += 1;
                }
                md = md.max((x[r] - y[r]).abs());
            }
        }
        (nd, md)
    };
    let names_v: Vec<String> = vars.iter().map(|v| v.0.clone()).collect();
    let fe = names_v.iter().position(|n| n == "flipexact");
    let ru = names_v.iter().position(|n| n == "runs");
    let runs_vs_exact = match (fe, ru) {
        (Some(x), Some(y)) => cmp(&outs[x], &outs[y]).0 as i64,
        _ => -1,
    };
    let pos = |n: &str| vars.iter().position(|v| v.0 == n);
    if let (Some(x), Some(y)) = (pos("runs2"), pos("runs3")) {
        eprintln!("runs2_vs_runs3_bits_differ={}", cmp(&outs[x], &outs[y]).0);
    }
    let (rej, ovf) = match &vars.iter().find(|v| v.0 == "runs").map(|v| &v.1) {
        Some(R::Runs(w, _, _)) => (w.rejected, w.overflows),
        _ => (0, 0),
    };

    // Timing: blocked, paired, rotated order.
    let mut rng = Lcg(42);
    let mut order: Vec<usize> = (0..groups.len()).collect();
    for k in (1..order.len()).rev() {
        let j = (rng.next() as usize) % (k + 1);
        order.swap(k, j);
    }
    let mbs: Vec<Vec<(usize, usize)>> = order
        .chunks(batch)
        .map(|c| c.iter().map(|&k| groups[k]).collect())
        .collect();
    let nv = vars.len();
    let mut res = vec![0.0f64; n_rows];
    for (_, r) in vars.iter_mut() {
        for &(s, e) in &mbs[0] {
            run(r, &mut fa, &fb, &data, &mut res, s, e, ee);
        }
    }
    let mut bm = vec![vec![0.0f64; blocks]; nv];
    let mut t = Vec::new();
    for b in 0..blocks {
        let mb = &mbs[b % mbs.len()];
        for o in 0..nv {
            let vi = (o + b) % nv;
            t.clear();
            for &(s, e) in mb {
                let t0 = Instant::now();
                run(&mut vars[vi].1, &mut fa, &fb, &data, &mut res, s, e, ee);
                t.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            bm[vi][b] = median(&mut t);
        }
    }
    let mut br = Lcg(7);
    for vi in 0..nv {
        let med = median(&mut bm[vi].clone());
        let ratios: Vec<f64> = (0..blocks).map(|b| bm[vi][b] / bm[0][b]).collect();
        let rm = median(&mut ratios.clone());
        let mut boots: Vec<f64> = (0..2000)
            .map(|_| {
                let mut s: Vec<f64> = (0..blocks)
                    .map(|_| ratios[(br.next() as usize) % blocks])
                    .collect();
                median(&mut s)
            })
            .collect();
        let (lo, hi) = (
            quantile(&mut boots.clone(), 0.025),
            quantile(&mut boots, 0.975),
        );
        let (nd, mdiff) = cmp(&outs[vi], &outs[0]);
        println!(
            "{{\"cell\":\"{}\",\"G\":{g},\"variant\":\"{}\",\"median_us\":{med:.3},\"ratio\":{rm:.4},\"ci_lo\":{lo:.4},\"ci_hi\":{hi:.4},\"n_bits_differ\":{nd},\"max_abs_diff\":{mdiff:e},\"us_per_row\":{:.4}}}",
            md.display(),
            names_v[vi],
            med / g as f64
        );
    }
    println!(
        "{{\"cell\":\"{}\",\"G\":{g},\"meta\":true,\"exact_e\":{ee},\"runs_vs_flipexact_bits_differ\":{runs_vs_exact},\"rejected_groups\":{rej},\"overflow_groups\":{ovf},\"n_groups\":{}}}",
        md.display(),
        groups.len()
    );
    std::process::exit(0);
}
