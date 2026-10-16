//! Every node of the layout decides like the parsed node it replaces, on fixtures and
//! generated models, for random and adversarial inputs.

use std::fmt::Write as _;
use std::path::PathBuf;

use super::*;
use crate::config::{Choices, WalkerConfig};
use crate::forest::Forest;
use crate::model::Model;
use crate::rules::Prefix;
use crate::{LoadOptions, ModelFormat};

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

fn choices(rng: &mut Rng) -> Choices {
    Choices {
        tree_ordering: !rng.chance(0.2),
        bitset_intern: !rng.chance(0.2),
        prefix: rng.pick(&[
            Prefix::Depth(0),
            Prefix::Depth(1),
            Prefix::Rule,
            Prefix::Depth(3),
        ]),
        predicate_dedup: !rng.chance(0.2),
    }
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

/// Every split of `model` takes the same child through the layout as through the
/// parsed nodes, for
/// bins of every feature and, on constant splits, bins of the constant features.
fn check_decisions(model: &Model, row: &[f64]) {
    let layout = &model.layout;
    let mut all = layout.bins.new_bins();
    layout.bins.fill(row, &mut all, true);
    let mut constant = layout.bins.new_bins();
    layout.bins.fill(row, &mut constant, false);
    for tree in &model.trees {
        let start = tree.node_start as usize;
        for local in 0..tree.node_count as usize {
            let node = &model.nodes[start + local];
            if node.is_leaf() {
                continue;
            }
            let parsed = start
                + if layout.bins.f32 {
                    model.step::<true>(&model.nodes, start + local, local, row)
                } else {
                    model.step::<false>(&model.nodes, start + local, local, row)
                };
            let index = (start + local) as u32;
            assert_eq!(
                layout.next(index, &all) as usize,
                parsed,
                "node {index} {node:?} {:?}, row {row:?}",
                layout.nodes[index as usize]
            );
            if node.is_constant() {
                assert_eq!(
                    layout.next(index, &constant) as usize,
                    parsed,
                    "node {index}"
                );
            }
        }
    }
    let dummy = layout.dummy();
    assert_eq!(layout.next(dummy, &all), dummy);
    assert_eq!(layout.next(dummy, &constant), dummy);
}

/// The bins of `row` against each feature's thresholds: `x <= t_k` (`x < t_k` for
/// float32, after rounding) exactly when `bin <= k`, and missing values take bin 0
/// when sent left and the threshold count when sent right. `reused` holds the bins
/// of earlier rows, of every feature and of the constant ones, which a fill of this
/// row's must turn into a fresh buffer's.
fn check_bins(bins: &BinMap, row: &[f64], reused: &mut [Vec<u32>; 2]) {
    let mut b = bins.new_bins();
    bins.fill(row, &mut b, true);
    let mut constant = bins.new_bins();
    bins.fill(row, &mut constant, false);
    bins.fill(row, &mut reused[0], true);
    bins.fill(row, &mut reused[1], false);
    assert_eq!(reused[0], b);
    assert_eq!(reused[1], constant);
    for (f, &x) in row.iter().enumerate() {
        let t = bins.feature_thresholds(f);
        let Some(s) = bins.numeric_slot(f) else {
            continue;
        };
        assert_eq!(b[s + 2], u32::MAX - b[s]);
        assert_eq!(b[s + 3], u32::MAX - b[s + 1]);
        if x.is_nan() {
            assert_eq!((b[s], b[s + 1]), (0, t.len() as u32));
            continue;
        }
        assert_eq!(b[s], b[s + 1]);
        for (k, &tk) in t.iter().enumerate() {
            let holds = if bins.f32 {
                (x as f32) < (tk as f32)
            } else {
                x <= tk
            };
            assert_eq!(holds, b[s] as usize <= k, "feature {f}: x {x}, t_{k} {tk}");
        }
    }
    // A categorical feature's slots hold its code's ranks, the defaults but for its
    // exceptions: the first set of each chain that holds the code, by the layout's
    // own nodes in check_decisions.
    for (f, &x) in row.iter().enumerate() {
        if let Some(c) = bins.feature_categories(f) {
            let code = if bins.f32 {
                c.codes.code::<true>(&bins.books, x)
            } else {
                c.codes.code::<false>(&bins.books, x)
            };
            assert_eq!(b[c.slot..c.slot + c.chains], bins.ranks_of(c, code));
        }
    }
    assert_eq!(b[DUMMY_SLOT], u32::MAX);
}

/// The heavy child is next, light children lie later in the same tree, slots and
/// leaf-value indices are in range, the dummy loops on itself, and the fields match
/// the parsed nodes.
fn check_structure(model: &Model) {
    let layout = &model.layout;
    assert_eq!(layout.nodes.len(), model.nodes.len() + 1);
    let dummy = layout.dummy();
    let d = layout.nodes[dummy as usize];
    assert_eq!((d.w0, d.child(), d.slot()), (0, dummy, DUMMY_SLOT));
    assert!(!d.is_leaf() && !d.is_varying());

    let n_leaves = layout.leaves.len();
    let mut seen = vec![false; n_leaves];
    let (mut internal, mut varying) = (0, 0);
    for tree in &model.trees {
        let (start, end) = (
            tree.node_start as usize,
            (tree.node_start + tree.node_count) as usize,
        );
        for i in start..end {
            let (old, new) = (&model.nodes[i], layout.nodes[i]);
            assert!(new.slot() < layout.bins.n_slots(), "node {i}");
            assert_eq!(new.is_leaf(), old.is_leaf(), "node {i}");
            if new.is_leaf() {
                let leaf = new.child() as usize;
                assert_eq!(new.w0, leaf as u64);
                assert!(leaf < n_leaves && !seen[leaf], "leaf {leaf}");
                seen[leaf] = true;
                assert!(!new.is_varying());
                assert_eq!(new.mask_field(), 0, "leaves read the all-rows mask");
                match (&layout.leaves, model.plan) {
                    (LeafValues::Fixed(values), crate::exact::Plan::Fixed(e)) => {
                        assert_eq!(values[leaf], crate::exact::to_fixed(old.value, e));
                    }
                    (LeafValues::Wide(values), crate::exact::Plan::Wide(wide)) => {
                        assert_eq!(values[leaf], wide.leaf(old.value));
                    }
                    _ => panic!("leaf values differ from the plan"),
                }
                continue;
            }
            internal += 1;
            let child = new.child() as usize;
            assert!(i + 1 < end && child > i + 1 && child < end, "node {i}");
            assert_eq!(child, start + old.skip as usize);
            assert_eq!(new.is_varying(), !old.is_constant());
            // A categorical split reads one of its feature's chains; a numeric one, one
            // of its four numeric slots.
            let f = old.feature as usize;
            if old.is_categorical() {
                let c = layout
                    .bins
                    .feature_categories(f)
                    .expect("a categorical feature");
                assert!(
                    (c.slot..c.slot + c.chains).contains(&new.slot()),
                    "node {i}"
                );
            } else {
                let s = layout.bins.numeric_slot(f).expect("a numeric feature");
                assert!((s..s + 4).contains(&new.slot()), "node {i}");
            }
            if new.is_varying() {
                varying += 1;
                let side = usize::from(!old.heavy_is_left());
                assert_eq!(
                    new.mask_field(),
                    2 * old.varying_pred_id as usize + 1 + side
                );
                assert!(new.mask_field() <= 2 * model.varying_predicates.len());
            } else {
                assert_eq!(new.mask_field(), 0);
            }
        }
    }
    assert!(seen.iter().all(|&s| s));
    let share = if internal == 0 {
        0.0
    } else {
        f64::from(varying) / f64::from(internal)
    };
    assert_eq!(layout.varying_share.to_bits(), share.to_bits());
    assert_eq!(
        matches!(layout.leaves, LeafValues::Fixed(_)),
        matches!(model.plan, crate::exact::Plan::Fixed(_))
    );
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// Values that sit on, beside and between `thresholds` under both precisions, and
/// around the categorical boundaries.
fn interesting_values(thresholds: &[f64]) -> Vec<f64> {
    let mut v = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        -f64::from_bits(1),
        f64::from_bits(1),
        -0.5,
        -1.0,
        -1e10,
        0.25,
        0.9999,
        1.0,
        1.5,
        2.0,
        7.75,
        31.0,
        31.5,
        31.999_999_999,
        32.0,
        33.0,
        33.9,
        40.0,
        99_999.0,
        100_000.0,
        100_000.5,
        2_147_483_648.5,
        4_294_967_294.0,
        4_294_967_295.0,
        4_294_967_295.5,
        63.0,
        64.0,
        100.5,
        8158.0,
        8159.0,
        8159.5,
        8160.0,
        8161.0,
        16_777_217.0,
        2_147_483_647.5,
        2_147_483_648.0,
        4_294_967_296.0,
        1e10,
        1e39,
        1e300,
    ];
    for &t in thresholds {
        let r = f64::from(t as f32);
        v.extend([t, t.next_up(), t.next_down(), r, r.next_up(), r.next_down()]);
        let up = f64::from((t as f32).next_up());
        let down = f64::from((t as f32).next_down());
        v.extend([r.midpoint(up), r.midpoint(down), up, down]);
    }
    v
}

fn random_rows(rng: &mut Rng, model: &Model, n_rows: usize) -> Vec<Vec<f64>> {
    let nf = model.config.n_features();
    let values: Vec<Vec<f64>> = (0..nf)
        .map(|f| {
            let bins = &model.layout.bins;
            let mut v = interesting_values(bins.feature_thresholds(f));
            for &c in bins.feature_book(f) {
                let c = f64::from(c);
                v.extend([c, c + 0.5, c - 1.0, c + 1.0]);
            }
            v
        })
        .collect();
    (0..n_rows)
        .map(|_| {
            (0..nf)
                .map(|f| match rng.below(10) {
                    0 => (rng.unit() - 0.5) * 200.0,
                    1 => rng.below(9000) as f64,
                    _ => rng.pick(&values[f]),
                })
                .collect()
        })
        .collect()
}

fn check_model(rng: &mut Rng, forest: &Forest, n_rows: usize) {
    let model = &*forest.model;
    check_structure(model);
    let bins = &model.layout.bins;
    let mut reused = [bins.new_bins(), bins.new_bins()];
    for row in random_rows(rng, model, n_rows) {
        check_decisions(model, &row);
        check_bins(bins, &row, &mut reused);
    }
}

// ---------------------------------------------------------------------------
// Generated models
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Numeric,
    Inline,
    Pooled(u32),
    Mixed(u32),
}

struct Spec {
    f32: bool,
    kinds: Vec<Kind>,
    palettes: Vec<Vec<f64>>,
    max_depth: usize,
    wide_leaves: bool,
}

fn random_spec(rng: &mut Rng) -> Spec {
    let nf = if rng.chance(0.3) {
        rng.below(64) + 1
    } else {
        rng.below(6) + 1
    };
    let f32 = rng.chance(0.4);
    let kinds = (0..nf)
        .map(|_| match rng.below(6) {
            0 => Kind::Inline,
            1 => Kind::Pooled(rng.pick(&[40, 100, 1000, 8159, 100_000, 1 << 31, u32::MAX])),
            2 => Kind::Mixed(rng.pick(&[20, 40, 8159, u32::MAX])),
            _ => Kind::Numeric,
        })
        .collect();
    let base = [
        0.0,
        -0.0,
        0.5,
        1.0,
        -2.5,
        31.5,
        1e-300,
        -1e-300,
        1.0 + 2f64.powi(-30),
        1.0 + 2f64.powi(-31),
        f64::from_bits(1),
    ];
    let palettes = (0..nf)
        .map(|_| {
            let mut p: Vec<f64> = (0..=rng.below(5)).map(|_| rng.pick(&base)).collect();
            p.extend((0..rng.below(20)).map(|_| ((rng.unit() - 0.5) * 100.0).round() / 8.0));
            if !f32 && rng.chance(0.3) {
                p.extend([f64::MAX, f64::MIN, 1e300]);
            }
            if f32 && rng.chance(0.3) {
                p.push(1e30);
            }
            p
        })
        .collect();
    Spec {
        f32,
        kinds,
        palettes,
        max_depth: rng.below(12) + 1,
        wide_leaves: rng.chance(0.15),
    }
}

fn num(x: f64) -> String {
    format!("{x:?}")
}

/// One node of a generated tree, then its subtrees; returns its ID.
fn write_node(rng: &mut Rng, spec: &Spec, depth: usize, nodes: &mut Vec<String>) -> usize {
    let id = nodes.len();
    nodes.push(String::new());
    let count = rng.below(1000) + 1;
    if depth >= spec.max_depth || (depth > 0 && rng.chance(0.25)) {
        let value = if spec.wide_leaves && rng.chance(0.5) {
            if spec.f32 {
                rng.pick(&[1e30, -1e30, 1e-30])
            } else {
                rng.pick(&[1e300, -1e300, 1e-300])
            }
        } else {
            (rng.unit() - 0.5) * 2f64.powi(rng.below(20) as i32 - 10)
        };
        nodes[id] = format!(
            r#"{{"node_id": {id}, "leaf_value": {}, "data_count": {count}}}"#,
            num(value)
        );
        return id;
    }
    let f = rng.below(spec.kinds.len());
    let default_left = rng.chance(0.5);
    let categorical = match spec.kinds[f] {
        Kind::Numeric => None,
        Kind::Inline => Some(31),
        Kind::Pooled(max) => Some(max),
        Kind::Mixed(max) => rng.chance(0.5).then_some(max),
    };
    let test = if let Some(max) = categorical {
        // Pooled features need more than 30 categories in their splits.
        let n = if max > 31 && rng.chance(0.5) {
            rng.below(48)
        } else {
            rng.below(8)
        };
        let mut list: Vec<u32> = (0..n)
            .map(|_| {
                if rng.chance(0.5) {
                    rng.below(32) as u32
                } else {
                    rng.below(max as usize + 1) as u32
                }
            })
            .collect();
        if rng.chance(0.2) {
            list.push(max);
        }
        list.sort_unstable();
        list.dedup();
        let list: Vec<String> = list.iter().map(u32::to_string).collect();
        format!(
            r#""node_type": "categorical_test_node", "category_list": [{}], "category_list_right_child": {}"#,
            list.join(", "),
            rng.chance(0.5)
        )
    } else {
        let op = if spec.f32 || rng.chance(0.5) {
            "<"
        } else {
            "<="
        };
        format!(
            r#""node_type": "numerical_test_node", "comparison_op": "{op}", "threshold": {}"#,
            num(rng.pick(&spec.palettes[f]))
        )
    };
    let left = write_node(rng, spec, depth + 1, nodes);
    let right = write_node(rng, spec, depth + 1, nodes);
    nodes[id] = format!(
        r#"{{"node_id": {id}, "split_feature_id": {f}, "default_left": {default_left}, {test}, "left_child": {left}, "right_child": {right}, "data_count": {count}}}"#
    );
    id
}

fn random_model_json(rng: &mut Rng, spec: &Spec) -> String {
    let n_trees = rng.below(8) + 1;
    let mut trees = Vec::new();
    for _ in 0..n_trees {
        let mut nodes = Vec::new();
        write_node(rng, spec, 0, &mut nodes);
        let has_cat = nodes.iter().any(|n| n.contains("categorical_test_node"));
        trees.push(format!(
            r#"{{"num_nodes": {}, "has_categorical_split": {has_cat}, "nodes": [{}]}}"#,
            nodes.len(),
            nodes.join(", ")
        ));
    }
    let ty = if spec.f32 { "float32" } else { "float64" };
    let zeros = vec!["0"; n_trees].join(", ");
    let mut json = String::new();
    write!(
        json,
        r#"{{"threshold_type": "{ty}", "leaf_output_type": "{ty}", "num_feature": {}, "task_type": "kRegressor", "average_tree_output": false, "num_target": 1, "num_class": [1], "leaf_vector_shape": [1, 1], "target_id": [{zeros}], "class_id": [{zeros}], "postprocessor": "identity", "sigmoid_alpha": 1.0, "ratio_c": 1.0, "base_scores": [0.0], "attributes": "{{}}", "trees": [{}]}}"#,
        spec.kinds.len(),
        trees.join(", ")
    )
    .unwrap();
    json
}

fn random_config(rng: &mut Rng, nf: usize) -> WalkerConfig {
    let mut roles: Vec<usize> = (0..nf).map(|_| rng.below(5)).collect();
    if rng.chance(0.2) {
        roles.fill(0);
    }
    let of = |r: usize| (0..nf).filter(|&f| roles[f] == r).collect::<Vec<_>>();
    WalkerConfig::builder(nf)
        .max_group_width(64)
        .varying(of(2))
        .increasing(of(3))
        .decreasing(of(4))
        .build()
        .unwrap()
}

#[test]
fn generated_models_decide_like_the_parsed_nodes() {
    let mut rng = Rng(0x5EED);
    let (mut categorical, mut chained, mut fixed, mut float) = (0, 0, 0, 0);
    for _ in 0..300 {
        let spec = random_spec(&mut rng);
        let json = random_model_json(&mut rng, &spec);
        let config = random_config(&mut rng, spec.kinds.len());
        let forest = Forest::from_reader_with(
            json.as_bytes(),
            ModelFormat::TreeliteJson,
            config,
            &LoadOptions::default(),
            &choices(&mut rng),
        )
        .unwrap_or_else(|e| panic!("{e}: {json}"));
        let layout = &forest.model.layout;
        let slots = layout.bins.chain_slots();
        categorical += usize::from(slots > 0);
        chained += usize::from(layout.chain_sets > slots);
        match layout.leaves {
            LeafValues::Fixed(_) => fixed += 1,
            LeafValues::Wide(_) => float += 1,
        }
        check_model(&mut rng, &forest, 200);
    }
    assert!(categorical > 100 && chained > 50 && fixed > 50 && float > 10);
}

#[test]
fn fixtures_decide_like_the_parsed_nodes() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/import");
    let mut rng = Rng(7);
    let mut loaded = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        let format = match path.extension().and_then(|e| e.to_str()) {
            Some("bin") => ModelFormat::TreeliteBinaryV4,
            Some("json") => ModelFormat::TreeliteJson,
            _ => continue,
        };
        if name.starts_with("reject_")
            || name.contains("reference")
            || name.contains("data")
            || name == "walker_config.json"
            || name == "manifest.json"
        {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        for varying in [vec![], vec![0], vec![0, 1]] {
            let config = WalkerConfig::builder(2)
                .max_group_width(128)
                .varying(varying)
                .build()
                .unwrap();
            let forest = Forest::from_reader_with(
                std::io::Cursor::new(&bytes),
                format,
                config,
                &LoadOptions::default(),
                &choices(&mut rng),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            check_model(&mut rng, &forest, 2000);
            loaded += 1;
        }
    }
    assert!(loaded >= 3 * 20, "{loaded}");
}

// ---------------------------------------------------------------------------
// Hand-checked encodings
// ---------------------------------------------------------------------------

fn one_split(test: &str, default_left: bool, f32: bool) -> Forest {
    let ty = if f32 { "float32" } else { "float64" };
    let json = format!(
        r#"{{"threshold_type": "{ty}", "leaf_output_type": "{ty}", "num_feature": 1, "task_type": "kRegressor", "average_tree_output": false, "num_target": 1, "num_class": [1], "leaf_vector_shape": [1, 1], "target_id": [0], "class_id": [0], "postprocessor": "identity", "sigmoid_alpha": 1.0, "ratio_c": 1.0, "base_scores": [0.0], "attributes": "", "trees": [{{"num_nodes": 3, "has_categorical_split": {}, "nodes": [{{"node_id": 0, "split_feature_id": 0, "default_left": {default_left}, {test}, "left_child": 1, "right_child": 2}}, {{"node_id": 1, "leaf_value": 1.0, "data_count": 1}}, {{"node_id": 2, "leaf_value": 2.0, "data_count": 5}}]}}]}}"#,
        test.contains("categorical")
    );
    let config = WalkerConfig::builder(1)
        .max_group_width(8)
        .varying([])
        .build()
        .unwrap();
    Forest::from_bytes(
        json.as_bytes(),
        ModelFormat::TreeliteJson,
        config,
        &LoadOptions::default(),
    )
    .unwrap()
}

#[test]
fn heavy_right_splits_read_inverted_bins_and_complemented_sets() {
    // The right leaf is heavier, so every split here is heavy-right.
    let numeric = one_split(
        r#""node_type": "numerical_test_node", "comparison_op": "<=", "threshold": 1.5"#,
        true,
        false,
    );
    let n = numeric.model.layout.nodes[0];
    // Slot 0 is the dummy's; the feature's numeric slots follow.
    assert_eq!(
        (n.slot(), n.payload(), (n.w0 >> 32) & 3),
        (1 + 2, u32::MAX - 1, 0)
    );
    for (list, default_left, f32) in [
        (vec![0, 3], true, false),
        // 31 categories and 40, past the 30 that 2.x held in a word.
        ((0..=30).chain([40]).collect::<Vec<u32>>(), false, true),
    ] {
        let names: Vec<String> = list.iter().map(u32::to_string).collect();
        let forest = one_split(
            &format!(
                r#""node_type": "categorical_test_node", "category_list": [{}], "category_list_right_child": false"#,
                names.join(", ")
            ),
            default_left,
            f32,
        );
        let layout = &forest.model.layout;
        let n = layout.nodes[0];
        // One heavy set, the complement of the left one: one chain, after the dummy.
        assert_eq!(
            (n.slot(), n.payload(), layout.bins.chain_slots()),
            (1, 0, 1)
        );
        // Its ranks: no category and, sent right, missing are in the set, at the
        // default rank 0; the listed codes are not, their exceptions.
        let c = layout.bins.feature_categories(0).unwrap();
        let mut want = vec![1; list.len() + 2];
        want[NO_CATEGORY as usize] = 0;
        if !default_left {
            want[MISSING as usize] = 0;
        }
        let domain = c.codes.domain() as u32;
        let ranks: Vec<u32> = (0..domain).map(|e| layout.bins.ranks_of(c, e)[0]).collect();
        assert_eq!(ranks, want);
        assert_eq!(c.chains, 1);
        let listed = list.len() + usize::from(default_left);
        assert_eq!(layout.bins.exceptions.len(), listed);
        assert_eq!(
            layout.bins.rank_table_bytes(),
            4 * (1 + domain as usize + 1) + 8 * listed
        );
        let listed = |x: f64| x >= 0.0 && list.contains(&(x as u32));
        for x in [
            0.0,
            0.9,
            3.0,
            3.5,
            30.5,
            40.5,
            31.0,
            39.0,
            41.0,
            -0.0,
            -0.5,
            1e10,
            f64::NAN,
        ] {
            let mut bins = layout.bins.new_bins();
            layout.bins.fill(&[x], &mut bins, false);
            let left = if x.is_nan() { default_left } else { listed(x) };
            assert_eq!(n.goes_heavy(&bins), !left, "{x}");
        }
    }
}

/// A model of one categorical feature with a tree per split: `(categories,
/// default_left, heavy_left)`.
fn categorical_forest(splits: &[(Vec<u32>, bool, bool)], varying: bool) -> Forest {
    categorical_load(splits, varying, &LoadOptions::default()).unwrap()
}

fn categorical_load(
    splits: &[(Vec<u32>, bool, bool)],
    varying: bool,
    options: &LoadOptions,
) -> Result<Forest, LoadError> {
    let trees: Vec<String> = splits
        .iter()
        .map(|(list, default_left, heavy_left)| {
            let names: Vec<String> = list.iter().map(u32::to_string).collect();
            let (l, r) = if *heavy_left { (5, 1) } else { (1, 5) };
            format!(
                r#"{{"num_nodes": 3, "has_categorical_split": true, "nodes": [{{"node_id": 0, "split_feature_id": 0, "default_left": {default_left}, "node_type": "categorical_test_node", "category_list": [{}], "category_list_right_child": false, "left_child": 1, "right_child": 2}}, {{"node_id": 1, "leaf_value": 1.0, "data_count": {l}}}, {{"node_id": 2, "leaf_value": 2.0, "data_count": {r}}}]}}"#,
                names.join(", ")
            )
        })
        .collect();
    let json = format!(
        r#"{{"threshold_type": "float64", "leaf_output_type": "float64", "num_feature": 1, "task_type": "kRegressor", "average_tree_output": false, "num_target": 1, "num_class": [1], "leaf_vector_shape": [1, 1], "target_id": {ids}, "class_id": {ids}, "postprocessor": "identity", "sigmoid_alpha": 1.0, "ratio_c": 1.0, "base_scores": [0.0], "attributes": "", "trees": [{}]}}"#,
        trees.join(", "),
        ids = format!("{:?}", vec![0; splits.len()]),
    );
    let config = WalkerConfig::builder(1)
        .max_group_width(8)
        .varying(if varying { vec![0] } else { vec![] })
        .build()
        .unwrap();
    Forest::from_bytes(json.as_bytes(), ModelFormat::TreeliteJson, config, options)
}

#[test]
fn nested_heavy_sets_share_a_chain() {
    // Heavy-left {1} ⊂ {1, 2} ⊂ {1, 2, 3} nest; {5} and, heavy-right, the
    // complement of {1, 2} (no category, missing, 3 and 5) do not nest with them,
    // but {5} ⊂ that complement. The largest antichain is {1, 2, 3} and {5}: two
    // chains.
    let splits = [
        (vec![1, 2], false, true),
        (vec![1], false, true),
        (vec![5], false, true),
        (vec![1, 2, 3], false, true),
        (vec![1, 2], false, false),
    ];
    let mut rng = Rng(29);
    for varying in [false, true] {
        let forest = categorical_forest(&splits, varying);
        let layout = &forest.model.layout;
        assert_eq!((layout.chain_sets, layout.bins.chain_slots()), (5, 2));
        // Codes: 0 no category, 1 missing, then 1, 2, 3, 5 at 2 to 5.
        assert_eq!(layout.bins.feature_book(0), [1, 2, 3, 5]);
        check_model(&mut rng, &forest, 500);
    }
}

#[test]
fn chains_stop_at_the_rank_table_cap() {
    // Ten singletons, some with missing: an antichain, ten chains over 12 codes.
    // The rank tables take 212 bytes: 10 defaults, 13 exception bounds and 15
    // exceptions, each singleton's code and five times missing, at rank 0.
    let splits: Vec<_> = (0..10).map(|c| (vec![c], c % 2 == 0, true)).collect();
    let cap = |bytes| LoadOptions {
        max_rank_table_bytes: bytes,
        ..LoadOptions::default()
    };
    let mut rng = Rng(31);
    for (varying, bytes) in [(false, 212), (true, 212)] {
        let forest = categorical_load(&splits, varying, &cap(bytes)).unwrap();
        let bins = &forest.model.layout.bins;
        assert_eq!((bins.rank_table_bytes(), bins.chain_slots()), (212, 10));
        check_model(&mut rng, &forest, 500);
        assert!(matches!(
            categorical_load(&splits, varying, &cap(bytes - 1)),
            Err(LoadError::Limit(_))
        ));
    }
}

#[test]
fn any_u32_category_takes_a_code() {
    let mut rng = Rng(11);
    for list in [
        vec![u32::MAX],
        vec![0, u32::MAX - 1, u32::MAX],
        (0..40).map(|c| c * 100_000_000).collect(),
        (0..40).rev().map(|c| u32::MAX - c).collect(),
    ] {
        let names: Vec<String> = list.iter().map(u32::to_string).collect();
        for (default_left, f32) in [(false, false), (true, false), (false, true), (true, true)] {
            let forest = one_split(
                &format!(
                    r#""node_type": "categorical_test_node", "category_list": [{}], "category_list_right_child": false"#,
                    names.join(", ")
                ),
                default_left,
                f32,
            );
            let layout = &forest.model.layout;
            // The codebook holds the listed categories, whatever their values.
            assert_eq!(layout.bins.feature_book(0), list.as_slice());
            check_model(&mut rng, &forest, 500);
        }
    }
}

#[test]
fn signed_zero_thresholds_share_a_bin() {
    let mut rng = Rng(3);
    for f32 in [false, true] {
        let json_of = |t: f64| {
            format!(
                r#""node_type": "numerical_test_node", "comparison_op": "<", "threshold": {}"#,
                num(t)
            )
        };
        for t in [0.0, -0.0] {
            let forest = one_split(&json_of(t), rng.chance(0.5), f32);
            let thresholds = forest.model.layout.bins.feature_thresholds(0);
            assert_eq!(thresholds.len(), 1);
            check_model(&mut rng, &forest, 500);
        }
    }
}

#[test]
fn the_empty_layout_holds_the_dummy() {
    assert_eq!(Layout::empty().nodes.len(), 1);
}

#[test]
fn small_codebooks_rank_like_the_search() {
    let mut rng = Rng(19);
    for _ in 0..500 {
        let mut book: Vec<u32> = (0..rng.below(64)).map(|_| rng.below(64) as u32).collect();
        book.sort_unstable();
        book.dedup();
        let mut books = vec![7; rng.below(3)];
        let small = CategoryCodes::new(&mut books, &book).unwrap();
        assert!(small.small.is_some());
        let search = CategoryCodes {
            small: None,
            ..small
        };
        for c in (0..70).chain([u32::MAX, 1 << 31]) {
            assert_eq!(small.of(&books, c), search.of(&books, c), "{book:?} {c}");
        }
    }
}
