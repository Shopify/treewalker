//! Recorded vectors: production's outputs, recorded once and checked bit for bit, so
//! that they outlive the kernels they were checked against.
//!
//! `tests/fixtures/vectors.bin` holds one vector per model: the output of every row,
//! which does not depend on grouping, configuration or load options because every
//! model's leaves are summed exactly. The models are every `tests/fixtures/import`
//! model on `data.bin`, and 36 models from the property generator (`tests/common`),
//! ordinary and with leaves past the 126-bit rule, each with a group at one of the
//! mask-width boundaries, runtime widths past 1,024 rows included. Each vector carries a fingerprint of its model
//! and rows, so a changed fixture or generator fails loudly instead of comparing
//! other inputs.
//!
//! The checks predict the rows in many groupings, configurations and load options,
//! through all three `Predictor` calls. To record again, after a change that is meant
//! to change outputs, run `cargo test --test vectors record -- --ignored`; it checks
//! production against the exact full walk before it writes.
#![expect(clippy::float_cmp, reason = "the generator's exact category test")]

mod common;

use common::model::{self, Feature, Precision, Settings};
use common::rng::{Rng, mix};
use serde::Deserialize;
use std::collections::BTreeMap;
use treewalker_gbdt::research::LoadAblation;
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat, WalkerConfig};

const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/vectors.bin");
const MAGIC: &[u8; 8] = b"TWVEC001";
const SEED: u64 = 0x5EC0_4DED_2026_1008;

/// Group sizes at the mask-width and piece boundaries.
const BOUNDARIES: [usize; 18] = [
    1, 2, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513, 1024, 1025, 2048, 2049,
];

/// A model's recorded outputs.
struct Vector {
    fingerprint: u64,
    outputs: Vec<f64>,
}

/// FNV-1a, for fingerprints.
struct Fnv(u64);

impl Fnv {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        self
    }
    fn values(&mut self, values: &[f64]) -> &mut Self {
        for v in values {
            self.bytes(&v.to_le_bytes());
        }
        self
    }
    fn count(&mut self, n: usize) -> &mut Self {
        self.bytes(&(n as u64).to_le_bytes())
    }
}

fn read_vectors() -> BTreeMap<String, Vector> {
    let bytes = std::fs::read(PATH).unwrap_or_else(|e| panic!("{PATH}: {e}"));
    assert!(bytes.starts_with(MAGIC), "{PATH}: not a vector file");
    let mut at = MAGIC.len();
    let mut take = |n: usize| {
        let s = &bytes[at..at + n];
        at += n;
        s
    };
    let word = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
    let records = word(take(8));
    let mut out = BTreeMap::new();
    for _ in 0..records {
        let len = word(take(8)) as usize;
        let name = String::from_utf8(take(len).to_vec()).unwrap();
        let fingerprint = word(take(8));
        let n = word(take(8)) as usize;
        let outputs = take(8 * n)
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        out.insert(
            name,
            Vector {
                fingerprint,
                outputs,
            },
        );
    }
    out
}

fn write_vectors(vectors: &BTreeMap<String, Vector>) {
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&(vectors.len() as u64).to_le_bytes());
    for (name, v) in vectors {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&v.fingerprint.to_le_bytes());
        bytes.extend_from_slice(&(v.outputs.len() as u64).to_le_bytes());
        for o in &v.outputs {
            bytes.extend_from_slice(&o.to_le_bytes());
        }
    }
    std::fs::write(PATH, bytes).unwrap_or_else(|e| panic!("{PATH}: {e}"));
}

/// The recorded vector `name`, after checking its fingerprint.
fn vector<'a>(vectors: &'a BTreeMap<String, Vector>, name: &str, fingerprint: u64) -> &'a [f64] {
    let v = vectors
        .get(name)
        .unwrap_or_else(|| panic!("{name}: not recorded"));
    assert!(
        v.fingerprint == fingerprint,
        "{name}: the model or its rows changed since recording ({:#x} != {fingerprint:#x})",
        v.fingerprint
    );
    &v.outputs
}

/// Panic at the first row whose bits differ.
fn same(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths");
    for (r, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.to_bits() == e.to_bits(),
            "{what}: row {r}: {a:e} ({:#018x}) != recorded {e:e} ({:#018x})",
            a.to_bits(),
            e.to_bits()
        );
    }
}

/// Predict `data` in groups of `sizes` with `predict_group` and `predict_groups`,
/// and with `predict_fixed` when `fixed` is the width, and compare with `expected`.
fn check_groups(
    forest: &Forest,
    data: &[f64],
    sizes: &[usize],
    fixed: Option<usize>,
    expected: &[f64],
    what: &str,
) {
    let nf = forest.config().n_features();
    let n = data.len() / nf;
    assert_eq!(sizes.iter().sum::<usize>(), n, "{what}: sizes");
    let mut predictor = forest.predictor();
    let mut out = vec![f64::NAN; n];
    let mut start = 0;
    for &size in sizes {
        predictor.predict_group(
            &data[start * nf..(start + size) * nf],
            &mut out[start..start + size],
        );
        start += size;
    }
    same(&out, expected, &format!("{what}, predict_group"));
    let offsets: Vec<usize> = std::iter::once(0)
        .chain(sizes.iter().scan(0, |end, &n| {
            *end += n;
            Some(*end)
        }))
        .collect();
    out.fill(f64::NAN);
    predictor.predict_groups(data, &offsets, &mut out);
    same(&out, expected, &format!("{what}, predict_groups"));
    if let Some(width) = fixed {
        out.fill(f64::NAN);
        predictor.predict_fixed(data, width, &mut out);
        same(&out, expected, &format!("{what}, predict_fixed({width})"));
    }
}

/// Production's outputs for `data`, one row at a time and in one batch of groups,
/// checked against each other and against the exact full walk.
fn record_outputs(forest: &Forest, data: &[f64], offsets: &[usize], what: &str) -> Vec<f64> {
    let n = data.len() / forest.config().n_features();
    let mut full = vec![f64::NAN; n];
    forest.predict_full_walk(data, &mut full);
    let mut batch = vec![f64::NAN; n];
    forest.predictor().predict_groups(data, offsets, &mut batch);
    same(
        &batch,
        &full,
        &format!("{what}: production against the full walk"),
    );
    let mut single = vec![f64::NAN; n];
    forest.predictor().predict_fixed(data, 1, &mut single);
    same(
        &single,
        &full,
        &format!("{what}: single rows against the full walk"),
    );
    batch
}

// ---------------------------------------------------------------------------
// The import fixtures
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Manifest {
    models: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    json: bool,
}

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/import/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The two-feature rows every fixture model is recorded on.
fn fixture_rows() -> Vec<f64> {
    fixture("data.bin")[16..]
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Each fixture model's encodings: the binary export, and the JSON dump if any.
/// A model's encodings, each with its format.
type Encodings = Vec<(ModelFormat, Vec<u8>)>;

fn fixture_models() -> Vec<(String, Encodings)> {
    let manifest: Manifest = simd_json::serde::from_slice(&mut fixture("manifest.json")).unwrap();
    manifest
        .models
        .into_iter()
        .map(|case| {
            let mut encodings = vec![(
                ModelFormat::TreeliteBinaryV4,
                fixture(&format!("{}.bin", case.name)),
            )];
            if case.json {
                encodings.push((
                    ModelFormat::TreeliteJson,
                    fixture(&format!("{}.json", case.name)),
                ));
            }
            (case.name, encodings)
        })
        .collect()
}

fn fixture_fingerprint(encodings: &[(ModelFormat, Vec<u8>)], rows: &[f64]) -> u64 {
    let mut h = Fnv::new();
    for (_, bytes) in encodings {
        h.count(bytes.len()).bytes(bytes);
    }
    h.values(rows).0
}

fn two_features(max_width: usize, varying: &[usize], inc: &[usize], dec: &[usize]) -> WalkerConfig {
    WalkerConfig::builder(2)
        .max_group_width(max_width)
        .varying(varying.iter().copied())
        .increasing(inc.iter().copied())
        .decreasing(dec.iter().copied())
        .build()
        .unwrap()
}

/// Rows `indices` of the fixture data, with their recorded outputs.
fn pick(rows: &[f64], expected: &[f64], indices: &[usize]) -> (Vec<f64>, Vec<f64>) {
    (
        indices
            .iter()
            .flat_map(|&i| [rows[2 * i], rows[2 * i + 1]])
            .collect(),
        indices.iter().map(|&i| expected[i]).collect(),
    )
}

/// A configured maximum for groups of up to `width` rows: the width itself, or more.
const fn max_width(width: usize, k: usize) -> usize {
    if k.is_multiple_of(2) { width } else { 2500 }
}

/// Check one fixture model's encoding against its vector in every grouping the
/// fixture data allows.
fn check_fixture(
    forest: &dyn Fn(WalkerConfig) -> Forest,
    all: bool,
    rows: &[f64],
    expected: &[f64],
    what: &str,
) {
    let n = rows.len() / 2;
    // Every feature varying: any rows form a group. Groups of each boundary width, the
    // data cycled in a stride, then a single row.
    for (k, &w) in BOUNDARIES.iter().enumerate() {
        let f = forest(two_features(max_width(w, k), &[0, 1], &[], &[]));
        let indices: Vec<usize> = (0..=2 * w).map(|r| (r * 7 + w) % n).collect();
        let (data, want) = pick(rows, expected, &indices);
        check_groups(
            &f,
            &data,
            &[w, w, 1],
            Some(w),
            &want,
            &format!("{what}: all varying, groups of {w}"),
        );
    }
    if !all {
        return;
    }
    // Feature 1 constant: rows sharing its value, cycled to the group's width.
    let mut pools: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        pools.entry(rows[2 * i + 1].to_bits()).or_default().push(i);
    }
    let pools: Vec<Vec<usize>> = pools.into_values().collect();
    for (k, &w) in BOUNDARIES.iter().enumerate() {
        let f = forest(two_features(max_width(w, k), &[0], &[], &[]));
        let (a, b) = (&pools[k % pools.len()], &pools[(k + 3) % pools.len()]);
        let indices: Vec<usize> = (0..w)
            .map(|r| a[r % a.len()])
            .chain((0..w).map(|r| b[(r * 5) % b.len()]))
            .chain([b[0]])
            .collect();
        let (data, want) = pick(rows, expected, &indices);
        check_groups(
            &f,
            &data,
            &[w, w, 1],
            Some(w),
            &want,
            &format!("{what}: feature 1 constant, groups of {w}"),
        );
    }
    // Feature 0 monotonic within each pool, without NaN.
    for (k, &w) in [3, 64, 65, 128, 129, 300].iter().enumerate() {
        for increasing in [true, false] {
            let config = if increasing {
                two_features(max_width(w, k), &[], &[0], &[])
            } else {
                two_features(max_width(w, k), &[], &[], &[0])
            };
            let f = forest(config);
            let mut indices = Vec::new();
            let mut sizes = Vec::new();
            for pool in &pools {
                let mut group: Vec<usize> = (0..w)
                    .map(|r| pool[(r * 3 + k) % pool.len()])
                    .filter(|&i| !rows[2 * i].is_nan())
                    .collect();
                group.sort_by(|&i, &j| rows[2 * i].total_cmp(&rows[2 * j]));
                if !increasing {
                    group.reverse();
                }
                if !group.is_empty() {
                    sizes.push(group.len());
                    indices.extend(group);
                }
            }
            let (data, want) = pick(rows, expected, &indices);
            check_groups(
                &f,
                &data,
                &sizes,
                None,
                &want,
                &format!("{what}: monotonic, groups of {w}"),
            );
        }
    }
    // Nothing varying: each group repeats one row.
    for w in [1, 17, 1025] {
        let f = forest(two_features(w, &[], &[], &[]));
        let indices: Vec<usize> = (0..5)
            .flat_map(|g| std::iter::repeat_n(g * 131 % n, w))
            .collect();
        let (data, want) = pick(rows, expected, &indices);
        check_groups(
            &f,
            &data,
            &[w; 5],
            Some(w),
            &want,
            &format!("{what}: constant, groups of {w}"),
        );
    }
}

fn fixture_options() -> [LoadAblation; 3] {
    [
        LoadAblation::default(),
        LoadAblation {
            disable_tree_ordering: true,
            disable_bitset_intern: true,
            prefix_depth: Some(0),
            ..Default::default()
        },
        LoadAblation {
            prefix_depth: Some(5),
            disable_predicate_dedup: true,
            ..Default::default()
        },
    ]
}

#[test]
fn fixtures_match_their_recorded_vectors() {
    let vectors = read_vectors();
    let rows = fixture_rows();
    for (name, encodings) in fixture_models() {
        let key = format!("fixture/{name}");
        let expected = vector(&vectors, &key, fixture_fingerprint(&encodings, &rows));
        for (format, bytes) in &encodings {
            for (o, options) in fixture_options().iter().enumerate() {
                // The binary export at default options gets every grouping; the other
                // encodings and options get the boundary widths.
                let all = *format == ModelFormat::TreeliteBinaryV4 && o == 0;
                if *format == ModelFormat::TreeliteJson && o > 0 {
                    continue;
                }
                let forest = |config: WalkerConfig| {
                    Forest::from_bytes_ablated(
                        bytes,
                        *format,
                        config,
                        &LoadOptions::default(),
                        options,
                    )
                    .unwrap()
                };
                check_fixture(
                    &forest,
                    all,
                    &rows,
                    expected,
                    &format!("{key} {format:?} {options:?}"),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Generated models
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Constant,
    Varying,
    Increasing,
    Decreasing,
}

/// A generated case: a model, its configuration and load options, and its groups.
struct Generated {
    name: String,
    fingerprint: u64,
    bytes: Vec<u8>,
    format: ModelFormat,
    config: WalkerConfig,
    options: LoadAblation,
    sizes: Vec<usize>,
    fixed: Option<usize>,
    data: Vec<f64>,
}

/// Case `k` of 36: an ordinary model for k < 18, leaves past the 126-bit rule
/// otherwise, with a group of `BOUNDARIES[k % 18]` rows among a few small ones.
fn generated(k: usize) -> Generated {
    let wide = k >= BOUNDARIES.len();
    let boundary = BOUNDARIES[k % BOUNDARIES.len()];
    let mut rng = Rng::new(mix(SEED ^ mix(k as u64)));
    let settings = Settings {
        max_trees: 20,
        max_nodes: 1200,
        wide_leaves: if wide { 1.0 } else { 0.0 },
        categorical: 0.3,
        wide: false,
        max_depth: 12,
        nested: 0.0,
    };
    let (model, features) = model::generate(&mut rng, &settings);
    let binary = model.has_infinite_threshold() || rng.chance(0.25);
    let (bytes, format) = if binary {
        (model.to_binary(&mut rng), ModelFormat::TreeliteBinaryV4)
    } else {
        (model.to_json(&mut rng), ModelFormat::TreeliteJson)
    };
    let nf = model.n_features;
    let roles: Vec<Role> = (0..nf)
        .map(|_| {
            *rng.pick(&[
                Role::Constant,
                Role::Constant,
                Role::Varying,
                Role::Increasing,
                Role::Decreasing,
            ])
        })
        .collect();
    // Small boundaries also run through predict_fixed: groups of the boundary and a
    // shorter last one.
    let (sizes, fixed) = if boundary <= 257 && k.is_multiple_of(3) {
        let mut sizes = vec![boundary; 3];
        sizes.push(rng.range(1, boundary));
        (sizes, Some(boundary))
    } else {
        let mut sizes: Vec<usize> = (0..rng.range(0, 3)).map(|_| rng.range(1, 40)).collect();
        sizes.insert(rng.below(sizes.len() + 1), boundary);
        (sizes, None)
    };
    let widest = *sizes.iter().max().unwrap();
    let max = if k.is_multiple_of(2) {
        widest
    } else {
        widest + rng.range(1, 3000)
    };
    let roles_ref = &roles;
    let with = |role| (0..nf).filter(move |&f| roles_ref[f] == role);
    let config = WalkerConfig::builder(nf)
        .max_group_width(max)
        .varying(with(Role::Varying))
        .increasing(with(Role::Increasing))
        .decreasing(with(Role::Decreasing))
        .build()
        .unwrap();
    // Prefix depth None is the rule's groups; it hashes as 2, the default it
    // replaced, which keeps the recorded fingerprints.
    let options = LoadAblation {
        disable_tree_ordering: rng.chance(0.3),
        disable_bitset_intern: rng.chance(0.3),
        prefix_depth: *rng.pick(&[Some(0), Some(1), None, Some(2), Some(3), Some(5)]),
        disable_predicate_dedup: rng.chance(0.1),
    };
    let data: Vec<f64> = sizes
        .iter()
        .flat_map(|&n| group(&mut rng, &features, &roles, model.precision, n))
        .collect();
    let mut h = Fnv::new();
    h.count(bytes.len()).bytes(&bytes).values(&data).count(max);
    for &size in &sizes {
        h.count(size);
    }
    for role in &roles {
        h.count(*role as usize);
    }
    h.count(options.prefix_depth.unwrap_or(2)).bytes(&[
        u8::from(options.disable_tree_ordering),
        u8::from(options.disable_bitset_intern),
        u8::from(options.disable_predicate_dedup),
    ]);
    let name = format!(
        "generated/{k:02}-{}-{boundary}",
        if wide { "wide" } else { "ordinary" }
    );
    Generated {
        name,
        fingerprint: h.0,
        bytes,
        format,
        config,
        options,
        sizes,
        fixed,
        data,
    }
}

/// One group's rows, row-major: each constant feature equal in every row, each
/// monotonic feature ordered in its direction and free of NaN.
fn group(
    rng: &mut Rng,
    features: &[Feature],
    roles: &[Role],
    precision: Precision,
    n: usize,
) -> Vec<f64> {
    let columns: Vec<Vec<f64>> = features
        .iter()
        .zip(roles)
        .map(|(f, &role)| match role {
            Role::Constant => vec![model::input(rng, f, precision, true); n],
            Role::Varying => (0..n)
                .map(|_| model::input(rng, f, precision, true))
                .collect(),
            Role::Increasing | Role::Decreasing => {
                let mut column: Vec<f64> = (0..n)
                    .map(|_| model::input(rng, f, precision, false))
                    .collect();
                column.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
                if role == Role::Decreasing {
                    column.reverse();
                }
                column
            }
        })
        .collect();
    (0..n)
        .flat_map(|r| columns.iter().map(move |c| c[r]))
        .collect()
}

const GENERATED: usize = 2 * BOUNDARIES.len();

#[test]
fn generated_models_match_their_recorded_vectors() {
    let vectors = read_vectors();
    for k in 0..GENERATED {
        let case = generated(k);
        let expected = vector(&vectors, &case.name, case.fingerprint);
        for options in [case.options, LoadAblation::default()] {
            let forest = Forest::from_bytes_ablated(
                &case.bytes,
                case.format,
                case.config.clone(),
                &LoadOptions::default(),
                &options,
            )
            .unwrap();
            let what = format!("{} {options:?}", case.name);
            check_groups(
                &forest,
                &case.data,
                &case.sizes,
                case.fixed,
                expected,
                &what,
            );
            // Single rows, which are always a valid grouping.
            let mut single = vec![f64::NAN; expected.len()];
            forest.predictor().predict_fixed(&case.data, 1, &mut single);
            same(&single, expected, &format!("{what}, single rows"));
        }
    }
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

#[test]
#[ignore = "writes tests/fixtures/vectors.bin; run after a change meant to change outputs"]
fn record() {
    let mut vectors = BTreeMap::new();
    let rows = fixture_rows();
    let n = rows.len() / 2;
    for (name, encodings) in fixture_models() {
        let what = format!("fixture/{name}");
        let (format, bytes) = &encodings[0];
        let forest = Forest::from_bytes(
            bytes,
            *format,
            two_features(n, &[0, 1], &[], &[]),
            &LoadOptions::default(),
        )
        .unwrap();
        let outputs = record_outputs(&forest, &rows, &[0, n], &what);
        let fingerprint = fixture_fingerprint(&encodings, &rows);
        vectors.insert(
            what,
            Vector {
                fingerprint,
                outputs,
            },
        );
    }
    for k in 0..GENERATED {
        let case = generated(k);
        let forest = Forest::from_bytes_ablated(
            &case.bytes,
            case.format,
            case.config.clone(),
            &LoadOptions::default(),
            &case.options,
        )
        .unwrap();
        let offsets: Vec<usize> = std::iter::once(0)
            .chain(case.sizes.iter().scan(0, |end, &n| {
                *end += n;
                Some(*end)
            }))
            .collect();
        let outputs = record_outputs(&forest, &case.data, &offsets, &case.name);
        vectors.insert(
            case.name,
            Vector {
                fingerprint: case.fingerprint,
                outputs,
            },
        );
    }
    write_vectors(&vectors);
}
