//! Paired benchmark for the opt-experiments branch.
//!
//! opt_bench <model_dir> <data_dir> [--blocks N] [--batch N] [--sched i,j,k] [--alpha A]
//!           [--variants a,b,c] [--xgb-lib PATH] [--lgb-lib PATH]
//!
//! Groups are fixed-width (max_group_width). Even-indexed groups are the calibration
//! set for profile-guided layout; odd-indexed groups are used for every measurement.
//! Each block times every variant on the same mini-batch in rotated order; the
//! block statistic is the median per-group latency. Prints one JSON object per variant.

use std::path::PathBuf;
use std::time::Instant;

use treewalker_gbdt::ParseConfig;
use treewalker_gbdt::forest::Forest;
use treewalker_gbdt::opt::{LeafMode, OptFlags, OptWorkspace};
use treewalker_gbdt::parser::layout;

enum Runner {
    Base(usize),
    Full(usize),
    Opt(usize, OptFlags, Box<OptWorkspace>),
    Compact(
        usize,
        Box<treewalker_gbdt::opt::CompactForest>,
        OptFlags,
        Box<OptWorkspace>,
    ),
    #[cfg(feature = "external-bench")]
    Xgb(treewalker_bench::external::XGBoostBench),
    #[cfg(feature = "external-bench")]
    Lgb(treewalker_bench::external::LightGBMBench),
}

struct Variant {
    name: String,
    runner: Runner,
}

fn run(
    v: &mut Variant,
    forests: &mut [Forest],
    data: &[f64],
    res: &mut [f64],
    s: usize,
    e: usize,
    n_cols: usize,
) {
    match &mut v.runner {
        Runner::Base(i) => forests[*i].predict(data, res, s, e),
        Runner::Full(i) => forests[*i].predict_full(data, res, s, e),
        Runner::Opt(i, f, ws) => forests[*i].predict_opt(ws, data, res, s, e, f),
        Runner::Compact(i, c, f, ws) => forests[*i].predict_compact(c, ws, data, res, s, e, f),
        #[cfg(feature = "external-bench")]
        Runner::Xgb(b) => {
            use treewalker_bench::external::ExternalMethod;
            b.predict_group(data, n_cols, s, e);
            let out = b.last_output();
            for (j, o) in out.iter().enumerate() {
                res[s + j] = f64::from(*o);
            }
        }
        #[cfg(feature = "external-bench")]
        Runner::Lgb(b) => {
            use treewalker_bench::external::ExternalMethod;
            b.predict_group(data, n_cols, s, e);
            let out = b.last_output(e - s);
            res[s..e].copy_from_slice(out);
        }
    }
    let _ = n_cols;
}

fn anon_huge_kb() -> u64 {
    std::fs::read_to_string("/proc/self/smaps_rollup")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("AnonHugePages:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
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
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
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
    let args: Vec<String> = std::env::args().collect();
    let model_dir = PathBuf::from(&args[1]);
    let data_dir = PathBuf::from(&args[2]);
    let mut blocks = 21usize;
    let mut batch = 64usize;
    let mut sched: u128 = 0;
    let mut alpha = 2.0f64;
    let mut variant_names: Vec<String> = Vec::new();
    let mut xgb_lib: Option<PathBuf> = None;
    let mut lgb_lib: Option<PathBuf> = None;
    let mut subset: Option<(usize, f64, bool)> = None; // (feature, quantile, keep_high)
    let mut analyze = false;
    let mut hot_cover = 1.0f64; // fraction of calibration visits the hot region must cover
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--blocks" => {
                blocks = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--batch" => {
                batch = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--alpha" => {
                alpha = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--sched" => {
                for t in args[i + 1].split(',').filter(|t| !t.is_empty()) {
                    sched |= 1u128 << t.parse::<u32>().unwrap();
                }
                i += 2;
            }
            "--variants" => {
                variant_names = args[i + 1].split(',').map(str::to_string).collect();
                i += 2;
            }
            "--xgb-lib" => {
                xgb_lib = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--lgb-lib" => {
                lgb_lib = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--subset" => {
                let p: Vec<&str> = args[i + 1].split(':').collect();
                subset = Some((p[0].parse().unwrap(), p[1].parse().unwrap(), p[2] == "hi"));
                i += 2;
            }
            "--analyze" => {
                analyze = true;
                i += 1;
            }
            "--hot-cover" => {
                hot_cover = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let _ = (&xgb_lib, &lgb_lib);

    let cfg_path = data_dir.join("walker_config.json");
    // Same preference as the paper's harness (src/bench/mod.rs): binary first.
    let model_path = if model_dir.join("model_treelite.bin").exists() {
        model_dir.join("model_treelite.bin")
    } else {
        model_dir.join("model_treelite.json")
    };
    let (data, n_rows, n_cols) = treewalker_bench::load_raw_f64(data_dir.join("test_data.bin"));

    let mut forests: Vec<Forest> = Vec::new();
    forests.push(Forest::load_with_config(
        &model_path,
        &cfg_path,
        &ParseConfig::default(),
    )); // 0 base
    let g = forests[0].config.max_group_width;
    assert_eq!(
        n_rows % g,
        0,
        "rows {n_rows} not a multiple of group width {g}"
    );
    let mut groups: Vec<(usize, usize)> = (0..n_rows / g).map(|k| (k * g, (k + 1) * g)).collect();
    if let Some((f, q, hi)) = subset {
        // Covariate-shift experiment: keep groups whose constant feature f is above
        // (hi) or below (lo) its q-quantile across all test groups.
        let mut vals: Vec<f64> = groups
            .iter()
            .map(|&(s, _)| data[s * n_cols + f])
            .filter(|v| !v.is_nan())
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let thr = vals[((vals.len() - 1) as f64 * q) as usize];
        groups.retain(|&(s, _)| {
            let v = data[s * n_cols + f];
            if hi { v >= thr } else { v <= thr }
        });
        eprintln!(
            "subset: feature {f} {} q{q} (thr {thr}) -> {} groups",
            if hi { ">=" } else { "<=" },
            groups.len()
        );
    }
    let calib: Vec<(usize, usize)> = groups.iter().step_by(2).copied().collect();
    let meas: Vec<(usize, usize)> = groups.iter().skip(1).step_by(2).copied().collect();

    // --- Profile-guided layout ---
    let t0 = Instant::now();
    layout::install(None, true);
    let prof = Forest::load_with_config(
        &model_path,
        &cfg_path,
        &ParseConfig {
            disable_tree_ordering: true,
            prefix_depth: 0,
            ..ParseConfig::default()
        },
    );
    let visit_orders = layout::take_visit_orders();
    let mut counts = vec![(0u64, 0u64); prof.n_nodes()];
    let mut pws = OptWorkspace::new(&prof);
    for &(s, e) in &calib {
        prof.profile_group(&mut pws, &data, s, e, &mut counts);
    }
    let ranges = prof.tree_ranges();
    assert_eq!(ranges.len(), visit_orders.len());
    let mut hint: Vec<Vec<(u64, u64)>> = Vec::with_capacity(ranges.len());
    for (t, &(st, cnt)) in ranges.iter().enumerate() {
        let vo = &visit_orders[t];
        assert_eq!(vo.len(), cnt);
        let mut h = vec![(0u64, 0u64); cnt];
        for (j, &old) in vo.iter().enumerate() {
            h[old as usize] = counts[st + j];
        }
        hint.push(h);
    }
    let profile_ms = t0.elapsed().as_secs_f64() * 1e3;

    if analyze {
        // Layout-headroom analysis on the profiling forest (annotation layout, file tree order).
        let mut cm = vec![(0u64, 0u64); prof.n_nodes()];
        for &(s, e) in &meas {
            prof.profile_group(&mut pws, &data, s, e, &mut cm);
        }
        let nodes = prof.nodes();
        let (mut v, mut ft_ann, mut ft_pgo, mut ft_orc) = (0u64, 0u64, 0u64, 0u64);
        let (mut dis_nodes, mut dis_visits, mut visited) = (0u64, 0u64, 0u64);
        let (mut vv, mut vft_ann, mut vft_pgo, mut vft_orc) = (0u64, 0u64, 0u64, 0u64);
        let mut minority: Vec<(f64, u64)> = Vec::new();
        for (k, nd) in nodes.iter().enumerate() {
            if nd.is_leaf() {
                continue;
            }
            let (l, r) = cm[k];
            let (cl, cr) = counts[k];
            let tot = l + r;
            if tot == 0 {
                continue;
            }
            let hil = nd.heavy_is_left();
            let pgo_left = if cl == cr { hil } else { cl > cr };
            if nd.is_constant() {
                visited += 1;
                v += tot;
                ft_ann += if hil { l } else { r };
                ft_pgo += if pgo_left { l } else { r };
                ft_orc += l.max(r);
                if pgo_left != hil {
                    dis_nodes += 1;
                    dis_visits += tot;
                }
                minority.push((l.min(r) as f64 / tot as f64, tot));
            } else {
                vv += tot;
                vft_ann += if hil { l } else { r };
                vft_pgo += if pgo_left { l } else { r };
                vft_orc += l.max(r);
            }
        }
        minority.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let wq = |q: f64| {
            let target = q * v as f64;
            let mut acc = 0.0;
            for &(m, w) in &minority {
                acc += w as f64;
                if acc >= target {
                    return m;
                }
            }
            1.0
        };
        let near_tie: u64 = minority.iter().filter(|x| x.0 > 0.4).map(|x| x.1).sum();
        let f = |a: u64, b: u64| {
            if b == 0 {
                f64::NAN
            } else {
                a as f64 / b as f64
            }
        };
        println!(
            "{{\"cell\":\"{}\",\"analysis\":true,\"const_visits_per_group\":{:.1},\"ft_annotation\":{:.4},\"ft_pgo_rule\":{:.4},\"ft_oracle\":{:.4},\"disagree_node_frac\":{:.4},\"disagree_visit_frac\":{:.4},\"minority_share_p25\":{:.3},\"minority_share_p50\":{:.3},\"minority_share_p75\":{:.3},\"near_tie_visit_frac\":{:.4},\"vary_onesided_per_group\":{:.1},\"vary_ft_annotation\":{:.4},\"vary_ft_pgo_rule\":{:.4},\"vary_ft_oracle\":{:.4}}}",
            model_dir.display(),
            v as f64 / meas.len() as f64,
            f(ft_ann, v),
            f(ft_pgo, v),
            f(ft_orc, v),
            f(dis_nodes, visited),
            f(dis_visits, v),
            wq(0.25),
            wq(0.5),
            wq(0.75),
            f(near_tie, v),
            vv as f64 / meas.len() as f64,
            f(vft_ann, vv),
            f(vft_pgo, vv),
            f(vft_orc, vv),
        );
    }
    layout::install(
        Some(layout::Mode::Profile {
            counts: hint.clone(),
            alpha,
        }),
        false,
    );
    forests.push(Forest::load_with_config(
        &model_path,
        &cfg_path,
        &ParseConfig::default(),
    )); // 1 pgo
    layout::install(Some(layout::Mode::AlwaysLeft), false);
    forests.push(Forest::load_with_config(
        &model_path,
        &cfg_path,
        &ParseConfig::default(),
    )); // 2 left
    layout::install(None, false);

    // Exactness forests: tree ordering off so summation order is identical.
    let noord = ParseConfig {
        disable_tree_ordering: true,
        ..ParseConfig::default()
    };
    forests.push(Forest::load_with_config(&model_path, &cfg_path, &noord)); // 3 base_noord
    layout::install(
        Some(layout::Mode::Profile {
            counts: hint,
            alpha,
        }),
        false,
    );
    forests.push(Forest::load_with_config(&model_path, &cfg_path, &noord)); // 4 pgo_noord
    layout::install(None, false);

    // BOLT-style experiments: visit counts on calibration groups -> hot-first layout.
    let mut cvis = vec![0u64; prof.n_nodes()];
    let mut tws = OptWorkspace::new(&prof);
    let _ = prof.touch_profile(&mut tws, &data, &calib, &mut cvis);
    let mut hv: Vec<Vec<u64>> = Vec::with_capacity(ranges.len());
    for (t, &(st, cnt)) in ranges.iter().enumerate() {
        let mut h = vec![0u64; cnt];
        for (j, &old) in visit_orders[t].iter().enumerate() {
            h[old as usize] = cvis[st + j];
        }
        hv.push(h);
    }
    // Hot threshold: smallest visit count such that nodes at or above it cover `hot_cover`
    // of all calibration visits (1.0 = every visited node is hot).
    let min_visits = {
        let mut sv: Vec<u64> = cvis.iter().copied().filter(|&x| x > 0).collect();
        sv.sort_unstable_by(|a, b| b.cmp(a));
        let tot: u64 = sv.iter().sum();
        let mut acc = 0u64;
        let mut thr = 1u64;
        for &x in &sv {
            acc += x;
            thr = x;
            if acc as f64 >= hot_cover * tot as f64 {
                break;
            }
        }
        if hot_cover >= 1.0 { 1 } else { thr }
    };
    let anon_hp_before = anon_huge_kb();
    let hv_all = hv.clone();
    layout::install(
        Some(layout::Mode::HotFirst {
            visits: hv.clone(),
            min_visits,
        }),
        false,
    );
    forests.push(Forest::load_with_config(
        &model_path,
        &cfg_path,
        &ParseConfig::default(),
    )); // 5 hot
    layout::install(
        Some(layout::Mode::HotFirst {
            visits: hv.clone(),
            min_visits,
        }),
        false,
    );
    forests.push(Forest::load_with_config(
        &model_path,
        &cfg_path,
        &ParseConfig::default(),
    )); // 6 hot -> hothuge
    layout::install(
        Some(layout::Mode::HotFirst {
            visits: hv,
            min_visits,
        }),
        false,
    );
    forests.push(Forest::load_with_config(&model_path, &cfg_path, &noord)); // 7 hot_noord (exactness)
    layout::install(None, false);
    forests.push(Forest::load_with_config(
        &model_path,
        &cfg_path,
        &ParseConfig::default(),
    )); // 8 base -> huge
    forests.push(Forest::load_with_config(
        &model_path,
        &cfg_path,
        &ParseConfig::default(),
    )); // 9 base + child hints
    forests[9].add_child_hints();
    forests[6].add_child_hints(); // invisible to the other kernels; used by "all"
    forests.push(Forest::load_with_config(
        &model_path,
        &cfg_path,
        &ParseConfig::default(),
    )); // 10 base + hints + flip
    forests[10].add_child_hints();
    let flip_slow = forests[10].flip_transform();
    // 11: hot-first layout + child hints + flip transform + huge pages
    layout::install(
        Some(layout::Mode::HotFirst {
            visits: hv_all.clone(),
            min_visits,
        }),
        false,
    );
    forests.push(Forest::load_with_config(
        &model_path,
        &cfg_path,
        &ParseConfig::default(),
    ));
    layout::install(None, false);
    forests[11].add_child_hints();
    forests[11].flip_transform();
    forests[11].hugepage_nodes();
    eprintln!("flip: {flip_slow} nodes on the slow path");
    let hp6 = forests[6].hugepage_nodes();
    let hp8 = forests[8].hugepage_nodes();
    let anon_hp_after = anon_huge_kb();
    // Footprint on measurement groups: base vs hot-first.
    let mut v0 = vec![0u64; forests[0].n_nodes()];
    let mut w0 = OptWorkspace::new(&forests[0]);
    let (lines_pg_base, union_base) = forests[0].touch_profile(&mut w0, &data, &meas, &mut v0);
    let mut v5 = vec![0u64; forests[5].n_nodes()];
    let mut w5 = OptWorkspace::new(&forests[5]);
    let (lines_pg_hot, union_hot) = forests[5].touch_profile(&mut w5, &data, &meas, &mut v5);
    let hot_nodes = v0.iter().filter(|&&x| x > 0).count();
    // Lines needed to cover 90% / 99% of node visits, base vs hot-first layout.
    let line_cover = |f: &Forest, v: &[u64], q: f64| -> usize {
        let a = f.nodes().as_ptr() as usize;
        let mut per: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();
        for (k, &x) in v.iter().enumerate() {
            if x > 0 {
                *per.entry((a % 64 + k * 16) / 64).or_insert(0) += x;
            }
        }
        let mut w: Vec<u64> = per.into_values().collect();
        w.sort_unstable_by(|a, b| b.cmp(a));
        let tot: u64 = w.iter().sum();
        let mut acc = 0u64;
        for (k, &x) in w.iter().enumerate() {
            acc += x;
            if acc as f64 >= q * tot as f64 {
                return k + 1;
            }
        }
        w.len()
    };
    let lc = [
        line_cover(&forests[0], &v0, 0.90),
        line_cover(&forests[0], &v0, 0.99),
        line_cover(&forests[5], &v5, 0.90),
        line_cover(&forests[5], &v5, 0.99),
    ];
    let total_nodes = forests[0].n_nodes();
    // Visit mass coverage: smallest node set covering 90% / 99% of visits.
    let mut sv: Vec<u64> = v0.clone();
    sv.sort_unstable_by(|a, b| b.cmp(a));
    let tot: u64 = sv.iter().sum();
    let cover = |q: f64| {
        let mut acc = 0u64;
        for (k, &x) in sv.iter().enumerate() {
            acc += x;
            if acc as f64 >= q * tot as f64 {
                return k + 1;
            }
        }
        sv.len()
    };
    let (c90, c99) = (cover(0.90), cover(0.99));
    // Hot-first vs base with identical tree order: must be bit-exact.
    let (mut ha, mut hb) = (vec![0.0f64; n_rows], vec![0.0f64; n_rows]);
    for &(s, e) in &meas {
        forests[3].predict(&data, &mut ha, s, e);
        forests[7].predict(&data, &mut hb, s, e);
    }
    let hot_noord_ndiff = meas
        .iter()
        .flat_map(|&(s, e)| s..e)
        .filter(|&r| ha[r].to_bits() != hb[r].to_bits())
        .count();
    println!(
        "{{\"cell\":\"{}\",\"footprint\":true,\"pool_kb\":{:.0},\"nodes\":{total_nodes},\"visited_nodes\":{hot_nodes},\"nodes_90pct_visits\":{c90},\"nodes_99pct_visits\":{c99},\"lines_per_group_base\":{lines_pg_base:.0},\"lines_per_group_hot\":{lines_pg_hot:.0},\"union_lines_base\":{union_base},\"union_lines_hot\":{union_hot},\"union_kb_base\":{:.0},\"union_kb_hot\":{:.0},\"hugepage_bytes\":[{hp6},{hp8}],\"min_visits\":{min_visits},\"lines90_base\":{},\"lines99_base\":{},\"lines90_hot\":{},\"lines99_hot\":{},\"anon_huge_kb_delta\":{},\"hot_vs_base_noord_bits_differ\":{hot_noord_ndiff}}}",
        model_dir.display(),
        total_nodes as f64 * 16.0 / 1024.0,
        union_base as f64 * 64.0 / 1024.0,
        union_hot as f64 * 64.0 / 1024.0,
        lc[0],
        lc[1],
        lc[2],
        lc[3],
        anon_hp_after as i64 - anon_hp_before as i64,
    );

    // Fall-through rates on the measurement groups.
    let mut fws = OptWorkspace::new(&forests[0]);
    let ft_base = forests[0].fallthrough_rate(&mut fws, &data, &meas);
    let mut fws1 = OptWorkspace::new(&forests[1]);
    let ft_pgo = forests[1].fallthrough_rate(&mut fws1, &data, &meas);
    let mut fws2 = OptWorkspace::new(&forests[2]);
    let ft_left = forests[2].fallthrough_rate(&mut fws2, &data, &meas);

    // --- Variants ---
    let mk = |fi: usize, f: OptFlags, forests: &Vec<Forest>| {
        Runner::Opt(fi, f, Box::new(OptWorkspace::new(&forests[fi])))
    };
    let d = OptFlags::default();
    if variant_names.is_empty() {
        variant_names = "base,opt0,halves,avx512,diff,k4,k8,sched,wide,pgo,left,combo,precomp"
            .split(',')
            .map(str::to_string)
            .collect();
    }
    let mut variants: Vec<Variant> = Vec::new();
    for name in &variant_names {
        let runner = match name.as_str() {
            "base" => Runner::Base(0),
            "opt0" => mk(0, d, &forests),
            "halves" => mk(
                0,
                OptFlags {
                    leaf: LeafMode::Halves,
                    ..d
                },
                &forests,
            ),
            "avx512" => mk(
                0,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    ..d
                },
                &forests,
            ),
            "diff" => mk(
                0,
                OptFlags {
                    leaf: LeafMode::DiffArray,
                    ..d
                },
                &forests,
            ),
            "k4" => mk(0, OptFlags { lockstep_k: 4, ..d }, &forests),
            "k8" => mk(0, OptFlags { lockstep_k: 8, ..d }, &forests),
            "sched" => mk(
                0,
                OptFlags {
                    schedule_features: sched,
                    ..d
                },
                &forests,
            ),
            "wide" => mk(
                0,
                OptFlags {
                    force_wide: true,
                    ..d
                },
                &forests,
            ),
            "pgo" => Runner::Base(1),
            "hot" => Runner::Base(5),
            "catfree" => mk(
                0,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    lockstep_k: 5,
                    ..d
                },
                &forests,
            ),
            "flipstack" => mk(
                10,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    lockstep_k: 9,
                    ..d
                },
                &forests,
            ),
            "flipdiff" => mk(
                10,
                OptFlags {
                    leaf: LeafMode::DiffArray,
                    schedule_features: sched,
                    lockstep_k: 7,
                    ..d
                },
                &forests,
            ),
            "flipexact" => mk(
                10,
                OptFlags {
                    leaf: LeafMode::Exact,
                    schedule_features: sched,
                    lockstep_k: 7,
                    exact_e: forests[10].exact_scale().expect("exact scale"),
                    ..d
                },
                &forests,
            ),
            "stackexact" => mk(
                10,
                OptFlags {
                    leaf: LeafMode::Exact,
                    schedule_features: sched,
                    lockstep_k: 9,
                    exact_e: forests[10].exact_scale().expect("exact scale"),
                    ..d
                },
                &forests,
            ),
            "allexact_u" => mk(
                11,
                OptFlags {
                    leaf: LeafMode::Exact,
                    schedule_features: sched,
                    lockstep_k: 10,
                    exact_e: forests[11].exact_scale().expect("exact scale"),
                    ..d
                },
                &forests,
            ),
            "stackexact_u" => mk(
                10,
                OptFlags {
                    leaf: LeafMode::Exact,
                    schedule_features: sched,
                    lockstep_k: 10,
                    exact_e: forests[10].exact_scale().expect("exact scale"),
                    ..d
                },
                &forests,
            ),
            "flipstack_u" => mk(
                10,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    lockstep_k: 10,
                    ..d
                },
                &forests,
            ),
            "allexact" => mk(
                11,
                OptFlags {
                    leaf: LeafMode::Exact,
                    schedule_features: sched,
                    lockstep_k: 9,
                    exact_e: forests[11].exact_scale().expect("exact scale"),
                    ..d
                },
                &forests,
            ),
            "allflip" => mk(
                11,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    lockstep_k: 7,
                    ..d
                },
                &forests,
            ),
            "flip" => mk(
                10,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    lockstep_k: 7,
                    ..d
                },
                &forests,
            ),
            "all" => mk(
                6,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    lockstep_k: 6,
                    ..d
                },
                &forests,
            ),
            "hint2" => mk(
                9,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    lockstep_k: 6,
                    ..d
                },
                &forests,
            ),
            "hint" => mk(
                9,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    lockstep_k: 3,
                    ..d
                },
                &forests,
            ),
            "compact" => Runner::Compact(
                0,
                Box::new(forests[0].compact()),
                d,
                Box::new(OptWorkspace::new(&forests[0])),
            ),
            "compact_best" => Runner::Compact(
                0,
                Box::new(forests[0].compact()),
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    ..d
                },
                Box::new(OptWorkspace::new(&forests[0])),
            ),
            "compact_huge_best" => {
                let mut c = forests[0].compact();
                c.hugepage_nodes();
                Runner::Compact(
                    0,
                    Box::new(c),
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        schedule_features: sched,
                        ..d
                    },
                    Box::new(OptWorkspace::new(&forests[0])),
                )
            }
            "compact_hot_huge_best" => {
                let mut c = forests[6].compact();
                c.hugepage_nodes();
                Runner::Compact(
                    6,
                    Box::new(c),
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        schedule_features: sched,
                        ..d
                    },
                    Box::new(OptWorkspace::new(&forests[6])),
                )
            }
            "pf" => mk(0, OptFlags { lockstep_k: 2, ..d }, &forests),
            "hothuge_best_pf" => mk(
                6,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    lockstep_k: 2,
                    ..d
                },
                &forests,
            ),
            "hothuge" => Runner::Base(6),
            "huge" => Runner::Base(8),
            "hothuge_best" => mk(
                6,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    ..d
                },
                &forests,
            ),
            "full" => Runner::Full(0),
            "left" => Runner::Base(2),
            "combo" => mk(
                1,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    lockstep_k: 4,
                    schedule_features: sched,
                    ..d
                },
                &forests,
            ),
            "combo8" => mk(
                1,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    lockstep_k: 8,
                    schedule_features: sched,
                    ..d
                },
                &forests,
            ),
            "best" => mk(
                0,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    schedule_features: sched,
                    ..d
                },
                &forests,
            ),
            "bestdiff" => mk(
                0,
                OptFlags {
                    leaf: LeafMode::DiffArray,
                    schedule_features: sched,
                    ..d
                },
                &forests,
            ),
            "precomp" => mk(
                0,
                OptFlags {
                    skip_trees: true,
                    ..d
                },
                &forests,
            ),
            #[cfg(feature = "external-bench")]
            "xgb_native" => {
                let lib = xgb_lib.clone().expect("--xgb-lib");
                let mp = model_dir.join("model_native.json");
                Runner::Xgb(
                    treewalker_bench::external::XGBoostBench::load(&lib, &mp, n_cols)
                        .expect("xgboost load"),
                )
            }
            #[cfg(feature = "external-bench")]
            "lgb_native" => {
                let lib = lgb_lib.clone().expect("--lgb-lib");
                let mp = model_dir.join("model_native.txt");
                Runner::Lgb(
                    treewalker_bench::external::LightGBMBench::load(&lib, &mp, n_cols, g)
                        .expect("lightgbm load"),
                )
            }
            other => panic!("unknown variant {other}"),
        };
        variants.push(Variant {
            name: name.clone(),
            runner,
        });
    }
    assert_eq!(variants[0].name, "base", "first variant must be base");

    // --- Correctness: every variant vs base on all measurement groups ---
    let mut ref_res = vec![0.0f64; n_rows];
    for &(s, e) in &meas {
        run(
            &mut variants[0],
            &mut forests,
            &data,
            &mut ref_res,
            s,
            e,
            n_cols,
        );
    }
    let mut exact: Vec<(f64, usize)> = Vec::new();
    let mut tmp = vec![0.0f64; n_rows];
    let mut exact_outputs: Vec<(String, Vec<f64>)> = Vec::new();
    for v in variants.iter_mut() {
        if v.name == "precomp" {
            exact.push((f64::NAN, 0));
            continue;
        }
        for &(s, e) in &meas {
            run(v, &mut forests, &data, &mut tmp, s, e, n_cols);
        }
        let mut maxd = 0.0f64;
        let mut ndiff = 0usize;
        for &(s, e) in &meas {
            for r in s..e {
                let dlt = (tmp[r] - ref_res[r]).abs();
                if tmp[r].to_bits() != ref_res[r].to_bits() {
                    ndiff += 1;
                }
                if dlt > maxd || dlt.is_nan() {
                    maxd = dlt;
                }
            }
        }
        exact.push((maxd, ndiff));
        if v.name.contains("exact") {
            exact_outputs.push((v.name.clone(), tmp.clone()));
        }
    }
    // Exact accumulation is order-independent: every exact variant must agree bit for bit,
    // whatever the tree order or kernel.
    let mut exact_cross_bits = 0usize;
    for w in exact_outputs.windows(2) {
        exact_cross_bits += meas
            .iter()
            .flat_map(|&(s, e)| s..e)
            .filter(|&r| w[0].1[r].to_bits() != w[1].1[r].to_bits())
            .count();
    }
    let exact_e = forests[10].exact_scale();
    eprintln!("exact: scale e={exact_e:?}, cross-variant bits differ={exact_cross_bits}");
    // PGO vs base with identical tree order: must be bit-exact.
    let (mut a, mut b) = (vec![0.0f64; n_rows], vec![0.0f64; n_rows]);
    for &(s, e) in &meas {
        forests[3].predict(&data, &mut a, s, e);
        forests[4].predict(&data, &mut b, s, e);
    }
    let pgo_noord_ndiff = meas
        .iter()
        .flat_map(|&(s, e)| s..e)
        .filter(|&r| a[r].to_bits() != b[r].to_bits())
        .count();

    // --- Timing ---
    let mut rng = Lcg(42);
    let mut order: Vec<usize> = (0..meas.len()).collect();
    for k in (1..order.len()).rev() {
        let j = (rng.next() as usize) % (k + 1);
        order.swap(k, j);
    }
    let minibatches: Vec<Vec<(usize, usize)>> = order
        .chunks(batch)
        .map(|c| c.iter().map(|&k| meas[k]).collect())
        .collect();
    let nv = variants.len();
    let mut res = vec![0.0f64; n_rows];
    for v in variants.iter_mut() {
        for &(s, e) in &minibatches[0] {
            run(v, &mut forests, &data, &mut res, s, e, n_cols);
        }
    }
    let mut block_med = vec![vec![0.0f64; blocks]; nv];
    let mut times: Vec<f64> = Vec::with_capacity(batch);
    for bi in 0..blocks {
        let mb = &minibatches[bi % minibatches.len()];
        for o in 0..nv {
            let vi = (o + bi) % nv;
            times.clear();
            let v = &mut variants[vi];
            for &(s, e) in mb {
                let t = Instant::now();
                run(v, &mut forests, &data, &mut res, s, e, n_cols);
                times.push(t.elapsed().as_secs_f64() * 1e6);
            }
            block_med[vi][bi] = median(&mut times);
        }
    }

    let cell = model_dir.display().to_string();
    let mut brng = Lcg(7);
    for (vi, v) in variants.iter().enumerate() {
        let mut bm = block_med[vi].clone();
        let med = median(&mut bm.clone());
        let p5 = quantile(&mut bm.clone(), 0.05);
        let p95 = quantile(&mut bm, 0.95);
        let ratios: Vec<f64> = (0..blocks)
            .map(|b| block_med[vi][b] / block_med[0][b])
            .collect();
        let rmed = median(&mut ratios.clone());
        let mut boots = Vec::with_capacity(2000);
        for _ in 0..2000 {
            let mut sample: Vec<f64> = (0..blocks)
                .map(|_| ratios[(brng.next() as usize) % blocks])
                .collect();
            boots.push(median(&mut sample));
        }
        let lo = quantile(&mut boots.clone(), 0.025);
        let hi = quantile(&mut boots, 0.975);
        let (maxd, ndiff) = exact[vi];
        println!(
            "{{\"cell\":\"{cell}\",\"G\":{g},\"variant\":\"{}\",\"median_us\":{med:.3},\"p5_us\":{p5:.3},\"p95_us\":{p95:.3},\"ratio\":{rmed:.4},\"ci_lo\":{lo:.4},\"ci_hi\":{hi:.4},\"max_abs_diff\":{},\"n_bits_differ\":{ndiff},\"blocks\":{blocks},\"batch\":{batch}}}",
            v.name,
            if maxd.is_nan() {
                "null".to_string()
            } else {
                format!("{maxd:e}")
            },
        );
    }
    println!(
        "{{\"cell\":\"{cell}\",\"G\":{g},\"meta\":true,\"profile_ms\":{profile_ms:.1},\"fallthrough_base\":{:.4},\"fallthrough_pgo\":{:.4},\"fallthrough_left\":{:.4},\"const_steps_per_group\":{:.1},\"pgo_vs_base_noord_bits_differ\":{pgo_noord_ndiff},\"exact_cross_bits_differ\":{exact_cross_bits},\"exact_e\":{},\"n_meas_groups\":{},\"n_calib_groups\":{}}}",
        ft_base.0 as f64 / ft_base.1 as f64,
        ft_pgo.0 as f64 / ft_pgo.1 as f64,
        ft_left.0 as f64 / ft_left.1 as f64,
        ft_base.1 as f64 / meas.len() as f64,
        exact_e.map_or(-1, |x| x),
        meas.len(),
        calib.len(),
    );
    std::process::exit(0);
}
