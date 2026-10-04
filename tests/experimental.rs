//! Experimental kernels (`--features experimental`) against `predict()` on the
//! independent import fixtures: identity/sigmoid/alpha/average/base-score outputs,
//! categorical membership in both directions, float32 models and boundary inputs.
#![cfg(feature = "experimental")]
#![allow(clippy::float_cmp)]
use serde::Deserialize;
use std::path::PathBuf;
use treewalker_gbdt::opt::{LeafMode, OptFlags, OptWorkspace, RunWorkspace};
use treewalker_gbdt::{Forest, ModelFormat, ParseConfig, WalkerConfig};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/import")
        .join(name)
}
fn bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap()
}
fn raw(name: &str) -> Vec<f64> {
    bytes(name)[16..]
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn load(name: &str, width: usize) -> Forest {
    let config = WalkerConfig::try_new(2, width, &[0], &[], &[]).unwrap();
    Forest::from_bytes(
        &bytes(name),
        ModelFormat::TreeliteBinaryV4,
        config,
        &ParseConfig::default(),
    )
    .unwrap()
}
#[derive(Deserialize)]
struct Manifest {
    models: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    atol: f64,
    rtol: f64,
}

/// Groups as in the import tests: blocks of 128 rows with a constant second feature,
/// split into chunks of the configured width.
fn assert_same_bits(actual: &[f64], expected: &[f64], label: &str) {
    if let Some(r) = (0..actual.len()).find(|&r| actual[r].to_bits() != expected[r].to_bits()) {
        panic!("{label}: row {r}: {} != {}", actual[r], expected[r]);
    }
}

fn groups(rows: usize, width: usize) -> Vec<(usize, usize)> {
    (0..rows)
        .step_by(128)
        .flat_map(|g| {
            (g..g + 128)
                .step_by(width)
                .map(move |s| (s, (s + width).min(g + 128)))
        })
        .collect()
}

#[test]
fn experimental_kernels_match_predict_on_import_fixtures() {
    let manifest: Manifest = simd_json::serde::from_slice(&mut bytes("manifest.json")).unwrap();
    let data = raw("data.bin");
    let rows = data.len() / 2;
    for case in manifest.models {
        let name = format!("{}.bin", case.name);
        let reference = raw(&format!("{}_reference.bin", case.name));
        for width in [32, 64, 128] {
            let gs = groups(rows, width);
            let mut base = load(&name, width);
            // predict() sums leaves exactly; the f64-accumulating kernels add them in
            // tree order, as the full walk does.
            let mut exact = vec![0.0; rows];
            let mut expected = vec![0.0; rows];
            for &(s, e) in &gs {
                base.predict(&data, &mut exact, s, e);
                base.predict_full(&data, &mut expected, s, e);
            }

            let mut hinted = load(&name, width);
            hinted.add_child_hints();
            let mut flipped = load(&name, width);
            flipped.add_child_hints();
            flipped.flip_transform();

            let d = OptFlags::default();
            // (label, forest, flags): all bit-identical to predict().
            let exact_variants: Vec<(&str, &Forest, OptFlags)> = vec![
                ("scalar", &base, d),
                (
                    "halves",
                    &base,
                    OptFlags {
                        leaf: LeafMode::Halves,
                        ..d
                    },
                ),
                (
                    "avx512",
                    &base,
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        ..d
                    },
                ),
                ("k4", &base, OptFlags { lockstep_k: 4, ..d }),
                ("k8", &base, OptFlags { lockstep_k: 8, ..d }),
                (
                    "per_group_width",
                    &base,
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        per_group_width: true,
                        ..d
                    },
                ),
                (
                    "force_wide",
                    &base,
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        force_wide: true,
                        ..d
                    },
                ),
                (
                    "catfree",
                    &base,
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        lockstep_k: 5,
                        ..d
                    },
                ),
                (
                    "hint",
                    &hinted,
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        lockstep_k: 3,
                        ..d
                    },
                ),
                (
                    "hint2",
                    &hinted,
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        lockstep_k: 6,
                        ..d
                    },
                ),
                (
                    "flip",
                    &flipped,
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        lockstep_k: 7,
                        ..d
                    },
                ),
                (
                    "flipstack",
                    &flipped,
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        lockstep_k: 9,
                        ..d
                    },
                ),
                (
                    "flipstack_u",
                    &flipped,
                    OptFlags {
                        leaf: LeafMode::Avx512,
                        lockstep_k: 10,
                        ..d
                    },
                ),
            ];
            for (label, forest, flags) in exact_variants {
                let mut ws = OptWorkspace::new(forest);
                let mut out = vec![f64::NAN; rows];
                for &(s, e) in &gs {
                    forest.predict_opt(&mut ws, &data, &mut out, s, e, &flags);
                }
                for r in 0..rows {
                    assert_eq!(
                        out[r].to_bits(),
                        expected[r].to_bits(),
                        "{name} width {width} {label}: row {r}: {} != {}",
                        out[r],
                        expected[r]
                    );
                }
            }

            let mut cf = base.compact();
            let mut ws = OptWorkspace::new(&base);
            let mut out = vec![f64::NAN; rows];
            for &(s, e) in &gs {
                base.predict_compact(
                    &mut cf,
                    &mut ws,
                    &data,
                    &mut out,
                    s,
                    e,
                    &OptFlags {
                        leaf: LeafMode::Avx512,
                        ..d
                    },
                );
            }
            for r in 0..rows {
                assert_eq!(
                    out[r].to_bits(),
                    expected[r].to_bits(),
                    "{name} width {width} compact: row {r}"
                );
            }

            // Order-independent accumulation: correctly rounded tree sum, so compare with
            // the independent oracle at the fixture's tolerance and with predict().
            let close = |x: f64, y: f64| (x - y).abs() <= case.rtol.mul_add(y.abs(), case.atol);
            let mut checked: Vec<(&str, Vec<f64>)> = Vec::new();
            if let Some(e) = flipped.exact_scale() {
                for (label, k) in [("flipexact", 7), ("stackexact", 9), ("stackexact_u", 10)] {
                    let flags = OptFlags {
                        leaf: LeafMode::Exact,
                        lockstep_k: k,
                        exact_e: e,
                        ..d
                    };
                    let mut ws = OptWorkspace::new(&flipped);
                    let mut out = vec![f64::NAN; rows];
                    for &(s, e) in &gs {
                        flipped.predict_opt(&mut ws, &data, &mut out, s, e, &flags);
                    }
                    assert_same_bits(&out, &exact, &format!("{name} width {width} {label}"));
                    checked.push((label, out));
                }
                // Run lists: groups whose varying feature is not monotone/unimodal/valley
                // shaped are rejected and left untouched; accepted ones must agree.
                let mut rws = RunWorkspace::new(&flipped, width);
                let mut out = exact.clone();
                for &(s, e2) in &gs {
                    let _ = flipped.predict_runs(&mut rws, &data, &mut out, s, e2, e);
                }
                assert_same_bits(&out, &exact, &format!("{name} width {width} runs"));
                checked.push(("runs", out));
            }
            let mut ws = OptWorkspace::new(&flipped);
            let mut out = vec![f64::NAN; rows];
            for &(s, e) in &gs {
                flipped.predict_opt(
                    &mut ws,
                    &data,
                    &mut out,
                    s,
                    e,
                    &OptFlags {
                        leaf: LeafMode::DiffArray,
                        lockstep_k: 7,
                        ..d
                    },
                );
            }
            checked.push(("flipdiff", out));
            for (label, out) in checked {
                for r in 0..rows {
                    assert!(
                        close(out[r], reference[r]) && close(out[r], expected[r]),
                        "{name} width {width} {label}: row {r}: {} vs oracle {} / predict {}",
                        out[r],
                        reference[r],
                        expected[r]
                    );
                }
            }
        }
    }
}

/// Constant categorical splits evaluated inline by the hint/catfree kernels follow
/// `eval_split`: NaN takes the missing branch; negative, fractional-negative and
/// out-of-range categories take the non-membership branch, whatever `default_left` says.
#[test]
fn inline_categorical_rules_match_predict() {
    let data = raw("data.bin");
    let rows = data.len() / 2;
    for name in ["categories_left.json", "categories_f32.json"] {
        let mut model = simd_json::to_owned_value(&mut bytes(name)).unwrap();
        // Tree 1 splits on the constant feature: include category 0 and route missing
        // values left, so -0.5, 40 and 64 distinguish membership from missing routing.
        let node = &mut model["trees"][1]["nodes"][0];
        node["category_list"] = simd_json::json!([0, 1, 2]);
        node["default_left"] = true.into();
        let json = simd_json::to_vec(&model).unwrap();
        let config = WalkerConfig::try_new(2, 128, &[0], &[], &[]).unwrap();
        let load = || {
            Forest::from_bytes(
                &json,
                ModelFormat::TreeliteJson,
                config.clone(),
                &ParseConfig::default(),
            )
            .unwrap()
        };
        let base = load();
        let mut expected = vec![0.0; rows];
        for s in (0..rows).step_by(128) {
            base.predict_full(&data, &mut expected, s, s + 128);
        }
        let mut hinted = load();
        hinted.add_child_hints();
        let d = OptFlags::default();
        for (label, forest, flags) in [
            (
                "catfree",
                &base,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    lockstep_k: 5,
                    ..d
                },
            ),
            (
                "hint",
                &hinted,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    lockstep_k: 3,
                    ..d
                },
            ),
            (
                "hint2",
                &hinted,
                OptFlags {
                    leaf: LeafMode::Avx512,
                    lockstep_k: 6,
                    ..d
                },
            ),
        ] {
            let mut ws = OptWorkspace::new(forest);
            let mut out = vec![f64::NAN; rows];
            for s in (0..rows).step_by(128) {
                forest.predict_opt(&mut ws, &data, &mut out, s, s + 128, &flags);
            }
            for r in 0..rows {
                assert_eq!(
                    out[r].to_bits(),
                    expected[r].to_bits(),
                    "{name} {label}: row {r}"
                );
            }
        }
    }
}

/// Run-list kernel on monotone panels: feature 0 sorted within each 128-row group.
/// Accepted groups must equal the exact bitmask kernel bit for bit.
#[test]
fn run_lists_match_exact_bitmask_kernel_on_monotone_groups() {
    let manifest: Manifest = simd_json::serde::from_slice(&mut bytes("manifest.json")).unwrap();
    let mut data = raw("data.bin");
    let rows = data.len() / 2;
    for g in (0..rows).step_by(128) {
        // The run kernel rejects NaN in varying features; replace them before sorting.
        let mut col: Vec<f64> = (g..g + 128)
            .map(|r| {
                if data[2 * r].is_nan() {
                    1.5
                } else {
                    data[2 * r]
                }
            })
            .collect();
        col.sort_by(f64::total_cmp);
        for (i, v) in col.into_iter().enumerate() {
            data[2 * (g + i)] = v;
        }
    }
    let mut accepted_total = 0;
    for case in manifest.models {
        let name = format!("{}.bin", case.name);
        let mut f = load(&name, 128);
        f.add_child_hints();
        f.flip_transform();
        let Some(e) = f.exact_scale() else { continue };
        let flags = OptFlags {
            leaf: LeafMode::Exact,
            lockstep_k: 7,
            exact_e: e,
            ..OptFlags::default()
        };
        let mut ws = OptWorkspace::new(&f);
        let mut rws = RunWorkspace::new(&f, 128);
        let (mut a, mut b) = (vec![0.0; rows], vec![0.0; rows]);
        for s in (0..rows).step_by(128) {
            f.predict_opt(&mut ws, &data, &mut a, s, s + 128, &flags);
            if f.predict_runs(&mut rws, &data, &mut b, s, s + 128, e) {
                accepted_total += 1;
                for r in s..s + 128 {
                    assert_eq!(a[r].to_bits(), b[r].to_bits(), "{name}: row {r}");
                }
            }
        }
    }
    assert!(accepted_total > 0, "no group exercised the run-list kernel");
}
