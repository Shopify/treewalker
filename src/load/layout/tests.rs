//! Every node of the layout decides like the parsed node it replaces, on fixtures and
//! generated models, for random and adversarial inputs.

use std::path::PathBuf;

use super::*;
use crate::config::{Choices, WalkerConfig};
use crate::forest::Forest;
use crate::load::json;
use crate::model::Model;
use crate::rng::Rng;
use crate::rules::Prefix;
use crate::{LoadOptions, ModelFormat};

fn choices(rng: &mut Rng) -> Choices {
    Choices {
        tree_ordering: !rng.chance(0.2),
        bitset_intern: !rng.chance(0.2),
        prefix: *rng.pick_mod(&[
            Prefix::Depth(0),
            Prefix::Depth(1),
            Prefix::Rule,
            Prefix::Depth(3),
        ]),
        predicate_dedup: !rng.chance(0.2),
        category_bits_above: *rng.pick_mod(&[0, 3, CATEGORY_BITS_ABOVE, CATEGORY_BITS_ABOVE]),
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
            let at = node_link(start + local);
            assert_eq!(
                node_index(layout.next(at, &all)),
                parsed,
                "node {} {node:?} {:?}, row {row:?}",
                start + local,
                layout.nodes[start + local]
            );
            if node.is_constant() {
                assert_eq!(
                    node_index(layout.next(at, &constant)),
                    parsed,
                    "node {}",
                    start + local
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
    // own nodes in check_decisions; or its row of bits.
    for (f, &x) in row.iter().enumerate() {
        if let Some(c) = bins.feature_categories(f) {
            let code = if bins.f32 {
                c.codes.code::<true>(&bins.books, x)
            } else {
                c.codes.code::<false>(&bins.books, x)
            };
            assert_eq!(b[c.slot..c.slot + c.slots()], bins.ranks_of(c, code));
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
    assert_eq!(dummy, node_link(model.nodes.len()));
    let d = layout.nodes[node_index(dummy)];
    assert_eq!((d.w0, d.child(), d.slot()), (0, dummy, DUMMY_SLOT));
    assert!(!d.is_leaf() && !d.is_varying());
    // Its bound is 0. With byte offsets, as k + 1, no bin is below it, so it loops
    // whatever its slot holds; with indices, as k, it loops on slot 0's `u32::MAX`
    // (`check_bins`).
    let bins: &[u32] = if BYTE_LINKS {
        &[0, 1, u32::MAX - 1, u32::MAX]
    } else {
        &[u32::MAX]
    };
    for &bin in bins {
        assert!(!d.goes_heavy::<true>(&[bin]), "{bin}");
    }

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
            let child = node_index(new.child());
            assert_eq!(new.child(), child as u32 * if BYTE_LINKS { 16 } else { 1 });
            assert!(i + 1 < end && child > i + 1 && child < end, "node {i}");
            assert_eq!(child, start + old.skip as usize);
            assert_eq!(new.is_varying(), !old.is_constant());
            // A categorical split reads one of its feature's chains, or one of its
            // bit slots, shifted with the bit split's k; a numeric one, one of its four
            // numeric slots, unshifted.
            let f = old.feature as usize;
            if old.is_categorical() {
                let c = layout
                    .bins
                    .feature_categories(f)
                    .expect("a categorical feature");
                assert!(
                    (c.slot..c.slot + c.slots()).contains(&new.slot()),
                    "node {i}"
                );
                assert!(!c.bits || !new.is_varying(), "node {i}");
                if c.bits {
                    assert_eq!(new.payload(), BIT_K + BOUND_BIAS, "node {i}");
                } else {
                    assert_eq!(new.shift(), 0, "node {i}");
                }
            } else {
                assert_eq!(new.shift(), 0, "node {i}");
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
                .map(|f| match rng.below_mod(10) {
                    0 => (rng.unit() - 0.5) * 200.0,
                    1 => rng.below_mod(9000) as f64,
                    _ => *rng.pick_mod(&values[f]),
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
        rng.below_mod(64) + 1
    } else {
        rng.below_mod(6) + 1
    };
    let f32 = rng.chance(0.4);
    let kinds = (0..nf)
        .map(|_| match rng.below_mod(6) {
            0 => Kind::Inline,
            1 => Kind::Pooled(*rng.pick_mod(&[40, 100, 1000, 8159, 100_000, 1 << 31, u32::MAX])),
            2 => Kind::Mixed(*rng.pick_mod(&[20, 40, 8159, u32::MAX])),
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
            let mut p: Vec<f64> = (0..=rng.below_mod(5))
                .map(|_| *rng.pick_mod(&base))
                .collect();
            p.extend((0..rng.below_mod(20)).map(|_| ((rng.unit() - 0.5) * 100.0).round() / 8.0));
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
        max_depth: rng.below_mod(12) + 1,
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
    let count = rng.below_mod(1000) + 1;
    if depth >= spec.max_depth || (depth > 0 && rng.chance(0.25)) {
        let value = if spec.wide_leaves && rng.chance(0.5) {
            if spec.f32 {
                *rng.pick_mod(&[1e30, -1e30, 1e-30])
            } else {
                *rng.pick_mod(&[1e300, -1e300, 1e-300])
            }
        } else {
            (rng.unit() - 0.5) * 2f64.powi(rng.below_mod(20) as i32 - 10)
        };
        nodes[id] = format!(
            r#"{{"node_id": {id}, "leaf_value": {}, "data_count": {count}}}"#,
            num(value)
        );
        return id;
    }
    let f = rng.below_mod(spec.kinds.len());
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
            rng.below_mod(48)
        } else {
            rng.below_mod(8)
        };
        let mut list: Vec<u32> = (0..n)
            .map(|_| {
                if rng.chance(0.5) {
                    rng.below_mod(32) as u32
                } else {
                    // `below_mod(max as usize + 1)`, without its overflow on 32-bit
                    // targets when `max` is `u32::MAX`.
                    (rng.next_u64() % (u64::from(max) + 1)) as u32
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
            num(*rng.pick_mod(&spec.palettes[f]))
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
    let n_trees = rng.below_mod(8) + 1;
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
    json::regressor(spec.f32, spec.kinds.len(), &trees)
}

fn random_config(rng: &mut Rng, nf: usize) -> WalkerConfig {
    let mut roles: Vec<usize> = (0..nf).map(|_| rng.below_mod(5)).collect();
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
    let mut rng = Rng::new(0x5EED);
    let (mut categorical, mut chained, mut bits, mut fixed, mut float) = (0, 0, 0, 0, 0);
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
        categorical += usize::from(slots + layout.bins.bit_slots() > 0);
        chained += usize::from(layout.chain_sets > slots);
        bits += usize::from(layout.bins.has_bits());
        match layout.leaves {
            LeafValues::Fixed(_) => fixed += 1,
            LeafValues::Wide(_) => float += 1,
        }
        check_model(&mut rng, &forest, 200);
    }
    assert!(categorical > 100 && chained > 50 && bits > 30 && fixed > 50 && float > 10);
}

#[test]
fn fixtures_decide_like_the_parsed_nodes() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/import");
    let mut rng = Rng::new(7);
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
    let json = json::regressor(f32, 1, &[json::stump(test, default_left, (1, 5))]);
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
    // Slot 0 is the dummy's; the feature's numeric slots follow. k = 0, inverted: its
    // bound `u32::MAX - 1`, or `u32::MAX` with byte offsets.
    let inverted = if BYTE_LINKS { u32::MAX } else { u32::MAX - 1 };
    assert_eq!((n.slot(), n.payload(), n.shift()), (1 + 2, inverted, 0));
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
        // One heavy set, the complement of the left one: one chain, after the dummy,
        // at k = 0, whose bound is 1 with byte offsets.
        assert_eq!(
            (n.slot(), n.payload(), layout.bins.chain_slots()),
            (1, u32::from(BYTE_LINKS), 1)
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
        assert_eq!(c.slots(), 1);
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
            assert_eq!(n.goes_heavy::<true>(&bins), !left, "{x}");
        }
    }
}

/// A numeric split at k = 0 stores its bound, k + 1 with byte offsets (x86-64), for
/// the test `bin < k + 1`, and k with node indices, for `bin <= k`: 1 or 0 on its slot
/// when its heavy child is on the left, and on the inverted slot when it is on the
/// right `u32::MAX`, the most a split stores, or `u32::MAX - 1`. Bins k and k + 1 fall
/// on either side, in filled bins and through `Layout::next`, whose heavy child is 16
/// bytes or 1 index on.
#[test]
fn split_bounds_part_bins_k_and_k_plus_1() {
    let test = r#""node_type": "numerical_test_node", "comparison_op": "<=", "threshold": 1.5"#;
    for (counts, heavy_left) in [((5, 1), true), ((1, 5), false)] {
        for default_left in [true, false] {
            let json = json::regressor(false, 1, &[json::stump(test, default_left, counts)]);
            let config = WalkerConfig::builder(1)
                .max_group_width(8)
                .varying([])
                .build()
                .unwrap();
            let forest = Forest::from_bytes(
                json.as_bytes(),
                ModelFormat::TreeliteJson,
                config,
                &LoadOptions::default(),
            )
            .unwrap();
            let layout = &forest.model.layout;
            let n = layout.nodes[0];
            // After the dummy's slot: missing sent left, sent right, both inverted.
            let slot = 1 + usize::from(!default_left) + if heavy_left { 0 } else { 2 };
            let bound = match (BYTE_LINKS, heavy_left) {
                (true, true) => 1,
                (true, false) => u32::MAX,
                (false, true) => 0,
                (false, false) => u32::MAX - 1,
            };
            assert_eq!((n.slot(), n.payload()), (slot, bound));
            let goes = |bins: &[u32], left: bool, what: &str| {
                assert_eq!(n.goes_heavy::<true>(bins), left == heavy_left, "{what}");
                let next = if left != heavy_left {
                    n.child()
                } else if BYTE_LINKS {
                    16
                } else {
                    1
                };
                assert_eq!(layout.next(0, bins), next, "{what}");
            };
            // Bin 0 (x <= 1.5) goes left, bin 1 right.
            for (bin, left) in [(0, true), (1, false)] {
                let mut bins = layout.bins.new_bins();
                bins[slot] = if heavy_left { bin } else { u32::MAX - bin };
                goes(&bins, left, &format!("bin {bin}"));
            }
            for x in [1.0, 1.5, 1.5f64.next_up(), 2.0, f64::NAN] {
                let mut bins = layout.bins.new_bins();
                layout.bins.fill(&[x], &mut bins, false);
                let left = if x.is_nan() { default_left } else { x <= 1.5 };
                goes(&bins, left, &format!("{x}"));
            }
        }
    }
}

/// A model of one categorical feature with a tree per split: `(categories,
/// default_left, heavy_left)`.
fn categorical_forest(splits: &[(Vec<u32>, bool, bool)], varying: bool) -> Forest {
    categorical_load(
        splits,
        varying,
        &LoadOptions::default(),
        CATEGORY_BITS_ABOVE,
    )
    .unwrap()
}

/// [`categorical_forest`] with `options`, and bits past `bits_above` heavy sets; the
/// trees in the model's order, so that tree t holds split t.
fn categorical_load(
    splits: &[(Vec<u32>, bool, bool)],
    varying: bool,
    options: &LoadOptions,
    bits_above: usize,
) -> Result<Forest, LoadError> {
    let varying: &[usize] = if varying { &[0] } else { &[] };
    categorical_features_load(&[splits], varying, options, bits_above)
}

/// [`categorical_load`] of a feature per entry of `features`, a tree per split, the
/// features in `varying` varying.
fn categorical_features_load(
    features: &[&[(Vec<u32>, bool, bool)]],
    varying: &[usize],
    options: &LoadOptions,
    bits_above: usize,
) -> Result<Forest, LoadError> {
    let trees: Vec<String> = features
        .iter()
        .enumerate()
        .flat_map(|(f, splits)| {
            splits.iter().map(move |(list, default_left, heavy_left)| {
                let test = format!(
                    r#""node_type": "categorical_test_node", "category_list": {list:?}, "category_list_right_child": false"#
                );
                let counts = if *heavy_left { (5, 1) } else { (1, 5) };
                json::stump(&test, *default_left, counts)
                    .replace(r#""split_feature_id": 0"#, &format!(r#""split_feature_id": {f}"#))
            })
        })
        .collect();
    let json = json::regressor(false, features.len(), &trees);
    let config = WalkerConfig::builder(features.len())
        .max_group_width(8)
        .varying(varying.to_vec())
        .build()
        .unwrap();
    let choices = Choices {
        tree_ordering: false,
        category_bits_above: bits_above,
        ..Choices::default()
    };
    Forest::from_reader_with(
        json.as_bytes(),
        ModelFormat::TreeliteJson,
        config,
        options,
        &choices,
    )
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
    let mut rng = Rng::new(29);
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
    // exceptions, each singleton's code and five times missing, at rank 0. Varying,
    // the masks' flips take 112 more: 13 bounds and a flip per exception, each into
    // its chain's one predicate (no row starts in it, the default rank being past
    // it).
    let splits: Vec<_> = (0..10).map(|c| (vec![c], c % 2 == 0, true)).collect();
    let cap = |bytes| LoadOptions {
        max_rank_table_bytes: bytes,
        ..LoadOptions::default()
    };
    let mut rng = Rng::new(31);
    for (varying, bytes) in [(false, 212), (true, 324)] {
        let forest = categorical_load(&splits, varying, &cap(bytes), CATEGORY_BITS_ABOVE).unwrap();
        let (bins, masks) = (&forest.model.layout.bins, &forest.model.masks);
        assert_eq!((bins.rank_table_bytes(), bins.chain_slots()), (212, 10));
        let flips = 4 * (masks.flip_bounds.len() + masks.flips.len());
        assert_eq!(flips, if varying { 112 } else { 0 });
        check_model(&mut rng, &forest, 500);
        assert!(matches!(
            categorical_load(&splits, varying, &cap(bytes - 1), CATEGORY_BITS_ABOVE),
            Err(LoadError::Limit(_))
        ));
    }
}

#[test]
fn bit_splits_shift_their_sets_bit_to_the_top() {
    // The five distinct heavy sets of `nested_heavy_sets_share_a_chain`, past a
    // threshold of 4: one slot of bits after the dummy's, over the six codes.
    let splits = [
        (vec![1, 2], false, true),
        (vec![1], false, true),
        (vec![5], false, true),
        (vec![1, 2, 3], false, true),
        (vec![1, 2], false, false),
    ];
    let mut rng = Rng::new(37);
    let options = LoadOptions::default();
    let forest = categorical_load(&splits, false, &options, 4).unwrap();
    let layout = &forest.model.layout;
    let bins = &layout.bins;
    assert_eq!((layout.chain_sets, layout.bit_sets), (0, 5));
    assert_eq!(
        (bins.chain_slots(), bins.bit_slots(), bins.has_bits()),
        (0, 1, true)
    );
    assert_eq!(bins.rank_table_bytes(), 4 * 6);
    let c = bins.feature_categories(0).unwrap();
    assert_eq!((c.slot, c.slots(), c.bits), (1, 1, true));
    // No category is in no list, so out of the listed sets and in the complement; its
    // row is the slot's default.
    assert_eq!(bins.defaults[1], bins.ranks_of(c, NO_CATEGORY)[0]);
    // Each split's bound is the bit split's, and its shift its set's: five distinct
    // shifts, 31 - j for set j.
    let mut shifts: Vec<u32> = forest
        .model
        .trees
        .iter()
        .map(|t| layout.nodes[t.node_start as usize])
        .map(|n| {
            assert_eq!((n.slot(), n.payload()), (1, BIT_K + BOUND_BIAS));
            n.shift()
        })
        .collect();
    shifts.sort_unstable();
    assert_eq!(shifts, [27, 28, 29, 30, 31]);
    // A code's bit is clear exactly in the sets that hold it: by value, missing, no
    // category, and a category of each code, against each split's heavy side.
    assert_eq!(bins.feature_book(0), [1, 2, 3, 5]);
    for x in [f64::NAN, 0.0, 4.0, 1.0, 2.0, 3.0, 5.0] {
        let mut b = bins.new_bins();
        bins.fill(&[x], &mut b, false);
        for (t, (list, _, heavy_left)) in splits.iter().enumerate() {
            let n = layout.nodes[forest.model.trees[t].node_start as usize];
            let left = !x.is_nan() && list.contains(&(x as u32));
            assert_eq!(
                n.goes_heavy::<true>(&b),
                left == *heavy_left,
                "{x}, split {t}"
            );
        }
    }
    check_model(&mut rng, &forest, 500);
    // Five sets are not past five, and varying features keep chains.
    for (varying, above) in [(false, 5), (true, 0)] {
        let forest = categorical_load(&splits, varying, &options, above).unwrap();
        let layout = &forest.model.layout;
        assert_eq!((layout.chain_sets, layout.bit_sets), (5, 0));
        assert_eq!(
            (layout.bins.chain_slots(), layout.bins.has_bits()),
            (2, false)
        );
        check_model(&mut rng, &forest, 200);
    }
}

#[test]
fn bits_that_pass_the_cap_fall_back_to_chains() {
    // 160 singletons over 162 codes: 160 chains, whose rank tables take 2,572 bytes
    // (160 defaults, 163 exception bounds and 160 exceptions), or five slots of bits,
    // whose rows take 3,240 (162 rows of five words).
    let splits: Vec<_> = (0..160).map(|c| (vec![c], false, true)).collect();
    let cap = |bytes| LoadOptions {
        max_rank_table_bytes: bytes,
        ..LoadOptions::default()
    };
    let mut rng = Rng::new(41);
    for (bytes, bits) in [(3240, true), (3239, false), (2572, false)] {
        let forest = categorical_load(&splits, false, &cap(bytes), 0).unwrap();
        let bins = &forest.model.layout.bins;
        let want = if bits { (3240, 0, 5) } else { (2572, 160, 0) };
        assert_eq!(
            (
                bins.rank_table_bytes(),
                bins.chain_slots(),
                bins.bit_slots()
            ),
            want,
            "{bytes}"
        );
        check_model(&mut rng, &forest, 200);
    }
    assert!(matches!(
        categorical_load(&splits, false, &cap(2571), 0),
        Err(LoadError::Limit(_))
    ));
}

type Splits = [(Vec<u32>, bool, bool)];

/// A layout's categorical rank-table bytes, chain slots and bit slots.
fn tables(forest: &Forest) -> (usize, usize, usize) {
    let bins = &forest.model.layout.bins;
    (
        bins.rank_table_bytes(),
        bins.chain_slots(),
        bins.bit_slots(),
    )
}

fn cap(bytes: u64) -> LoadOptions {
    LoadOptions {
        max_rank_table_bytes: bytes,
        ..LoadOptions::default()
    }
}

#[test]
fn the_cap_counts_bits_with_the_other_features_tables() {
    // Past ten sets, 160 singletons take bits, 3,240 bytes, or chains, 2,572 (as in
    // `bits_that_pass_the_cap_fall_back_to_chains`). Beside them, ten singletons take
    // bits, 48 bytes, fewer than their chains, 212 (as in
    // `chains_stop_at_the_rank_table_cap`).
    let ten: Vec<_> = (0..10).map(|c| (vec![c], c % 2 == 0, true)).collect();
    let many: Vec<_> = (0..160).map(|c| (vec![c], false, true)).collect();
    let mut rng = Rng::new(43);
    // Both take bits at 3,288 bytes. The features are laid out in order: below that,
    // the 160 first take bits that leave the ten too little, so the load takes
    // chains everywhere, 2,784, and below their bits they take chains and the ten
    // bits, 2,620; the ten first take bits and the 160 chains.
    let both = (3288, 0, 6);
    let all_chains = (2784, 170, 0);
    let ten_bits = (2620, 160, 1);
    for (bits_first, layouts) in [
        (
            true,
            &[
                (3288, both),
                (3287, all_chains),
                (3240, all_chains),
                (3239, ten_bits),
                (2620, ten_bits),
            ][..],
        ),
        (
            false,
            &[(3288, both), (3287, ten_bits), (2620, ten_bits)][..],
        ),
    ] {
        let features: [&Splits; 2] = if bits_first {
            [&many, &ten]
        } else {
            [&ten, &many]
        };
        for &(bytes, want) in layouts {
            let forest = categorical_features_load(&features, &[], &cap(bytes), 10).unwrap();
            assert_eq!(
                tables(&forest),
                want,
                "bits first {bits_first}, {bytes} bytes"
            );
            check_model(&mut rng, &forest, 100);
        }
        assert!(matches!(
            categorical_features_load(&features, &[], &cap(2619), 10),
            Err(LoadError::Limit(_))
        ));
    }
    // The varying features' flips share the cap with the bits: 40 singletons in bits,
    // 336 bytes (chains would take 652), then ten varying singletons' chains, 212, and
    // their flips, 112.
    let forty: Vec<_> = (0..40).map(|c| (vec![c], false, true)).collect();
    let features: [&Splits; 2] = [&forty, &ten];
    let forest = categorical_features_load(&features, &[1], &cap(660), 10).unwrap();
    let masks = &forest.model.masks;
    assert_eq!(tables(&forest), (548, 10, 2));
    assert_eq!(4 * (masks.flip_bounds.len() + masks.flips.len()), 112);
    check_model(&mut rng, &forest, 100);
    assert!(matches!(
        categorical_features_load(&features, &[1], &cap(659), 10),
        Err(LoadError::Limit(_))
    ));
}

#[test]
fn bits_that_leave_the_flips_too_little_fall_back_to_chains() {
    // 160 singletons past ten sets in bits, 3,240 bytes, or chains, 2,572; ten varying
    // singletons' chains, 212, and their flips, 112, which the layout bounds by 172
    // (13 bounds and two flips per exception) before the masks count them.
    let ten: Vec<_> = (0..10).map(|c| (vec![c], c % 2 == 0, true)).collect();
    let many: Vec<_> = (0..160).map(|c| (vec![c], false, true)).collect();
    let features: [&Splits; 2] = [&many, &ten];
    let mut rng = Rng::new(53);
    // Bits and the bound fit at 3,624 bytes. At 3,563 the bits' 3,452 and the flips
    // fit no more, the bound neither, and chains everywhere take fewer bytes: 2,784,
    // with the flips 2,896.
    for (bytes, want) in [
        (3624, (3452, 10, 5)),
        (3563, (2784, 170, 0)),
        (2896, (2784, 170, 0)),
    ] {
        let forest = categorical_features_load(&features, &[1], &cap(bytes), 10).unwrap();
        assert_eq!(tables(&forest), want, "{bytes} bytes");
        let masks = &forest.model.masks;
        assert_eq!(4 * (masks.flip_bounds.len() + masks.flips.len()), 112);
        check_model(&mut rng, &forest, 100);
    }
    assert!(matches!(
        categorical_features_load(&features, &[1], &cap(2895), 10),
        Err(LoadError::Limit(_))
    ));
    // A tie keeps the bits: 129 singletons, missing sent left by 68, take 2,620 bytes
    // in bits (131 rows of five words) and in chains (129 defaults, 132 bounds and 197
    // exceptions). Beside the ten varying singletons at 2,944 bytes the bound fits
    // neither layout, both build, and the bits stay; with the flips they take 2,944.
    let tie: Vec<_> = (0..129).map(|c| (vec![c], c < 68, true)).collect();
    let features: [&Splits; 2] = [&tie, &ten];
    let forest = categorical_features_load(&features, &[1], &cap(2944), 10).unwrap();
    assert_eq!(tables(&forest), (2832, 10, 5));
    check_model(&mut rng, &forest, 100);
    // Where neither fits, the error tells both: 40 singletons take bits, 336 bytes,
    // which leave the 160's chains too little at 2,900 bytes, and chains everywhere
    // take 3,224.
    let forty: Vec<_> = (0..40).map(|c| (vec![c], false, true)).collect();
    let features: [&Splits; 2] = [&forty, &many];
    let Err(LoadError::Limit(e)) = categorical_features_load(&features, &[], &cap(2900), 10) else {
        panic!("a limit error");
    };
    assert!(
        e.starts_with("with category bits, ") && e.contains("; with chains everywhere, "),
        "{e}"
    );
}

#[test]
fn bits_beside_that_crowd_out_the_bits_past_the_threshold_give_way() {
    // Past 100 sets, 200 nested prefixes over 202 codes take bits, 5,656 bytes; their
    // chains' inclusion graph, every pair of them, passes the cap. Beside them, 99
    // nested prefixes, of lengths 2 to 98, 103 and 10,000 over 10,002 codes, take bits
    // that take no more than their chains can, 160,032 bytes, though their one chain
    // takes 120,016. At 161,000 bytes the bits beside leave the 200 too little, and
    // chains everywhere fail on the graph: bits only past the threshold load, 125,672
    // bytes, in either order.
    let prefix = |n: u32| ((0..n).collect::<Vec<_>>(), false, true);
    let past: Vec<_> = (1..=200).map(prefix).collect();
    let beside: Vec<_> = (2..=98).chain([103, 10_000]).map(prefix).collect();
    let mut rng = Rng::new(61);
    for features in [[&beside, &past], [&past, &beside]] {
        let features: [&Splits; 2] = [features[0], features[1]];
        let forest = categorical_features_load(&features, &[], &cap(161_000), 100).unwrap();
        assert_eq!(tables(&forest), (125_672, 1, 7));
        check_model(&mut rng, &forest, 50);
        let forest =
            categorical_features_load(&features, &[], &LoadOptions::default(), 100).unwrap();
        assert_eq!(tables(&forest), (165_688, 0, 11));
    }
}

#[test]
fn features_beside_a_bit_feature_take_bits_no_larger_than_their_chains() {
    // Past 100 sets, 160 singletons take bits. Beside them, ten singletons take bits,
    // 48 bytes, no more than their chains can take, 212; 99 singletons over 101
    // codes keep chains, 1,596 bytes, as their bits would take 1,616 (4 slots).
    let ten: Vec<_> = (0..10).map(|c| (vec![c], c % 2 == 0, true)).collect();
    let many: Vec<_> = (0..160).map(|c| (vec![c], false, true)).collect();
    let sparse: Vec<_> = (0..99).map(|c| (vec![c], false, true)).collect();
    let mut rng = Rng::new(59);
    // Whichever feature comes first.
    for features in [[&ten, &sparse, &many], [&many, &sparse, &ten]] {
        let features: [&Splits; 3] = [features[0], features[1], features[2]];
        let forest =
            categorical_features_load(&features, &[], &LoadOptions::default(), 100).unwrap();
        let layout = &forest.model.layout;
        assert_eq!(tables(&forest), (3240 + 48 + 1596, 99, 6));
        assert_eq!((layout.chain_sets, layout.bit_sets), (99, 170));
        check_model(&mut rng, &forest, 100);
    }
    // Without a feature past the threshold, every feature keeps chains; a varying one
    // always does.
    let features: [&Splits; 2] = [&ten, &sparse];
    let forest = categorical_features_load(&features, &[], &LoadOptions::default(), 100).unwrap();
    assert_eq!(tables(&forest), (212 + 1596, 109, 0));
    let features: [&Splits; 2] = [&many, &ten];
    let forest = categorical_features_load(&features, &[1], &LoadOptions::default(), 100).unwrap();
    assert_eq!(tables(&forest), (3240 + 212, 10, 5));
    check_model(&mut rng, &forest, 100);
}

#[test]
fn bit_rows_span_words_with_both_sides_and_missing() {
    // 70 distinct heavy sets over 12 categories, in three slots, the last of 6 bits:
    // split i lists the categories of i + 1's bits, sends missing left on even i and
    // is heavy-right on every third, a complement.
    let splits: Vec<_> = (0..70u32)
        .map(|i| {
            let list = (0..12).filter(|&c| (i + 1) >> c & 1 != 0).collect();
            (list, i % 2 == 0, i % 3 != 0)
        })
        .collect();
    let forest = categorical_load(&splits, false, &LoadOptions::default(), 0).unwrap();
    let layout = &forest.model.layout;
    let bins = &layout.bins;
    assert_eq!(
        (layout.bit_sets, bins.bit_slots(), bins.chain_slots()),
        (70, 3, 0)
    );
    let nodes: Vec<LayoutNode> = forest
        .model
        .trees
        .iter()
        .map(|t| layout.nodes[t.node_start as usize])
        .collect();
    let mut bits: Vec<(usize, u32)> = nodes.iter().map(|n| (n.slot(), n.shift())).collect();
    bits.sort_unstable();
    bits.dedup();
    assert_eq!(bits.len(), 70);
    assert!(bits.iter().all(|&(slot, _)| (1..=3).contains(&slot)));
    // One buffer through no category, a category, the same again, missing, no
    // category and others, each fill equal to a fresh one and each split deciding by
    // its list and its heavy side.
    let mut reused = bins.new_bins();
    for x in [100.0, 3.0, 3.0, f64::NAN, 100.0, 0.0, 11.0, -1.0, 11.0, 5.5] {
        bins.fill(&[x], &mut reused, false);
        let mut fresh = bins.new_bins();
        bins.fill(&[x], &mut fresh, false);
        assert_eq!(reused, fresh, "{x}");
        for (n, (list, default_left, heavy_left)) in nodes.iter().zip(&splits) {
            let left = if x.is_nan() {
                *default_left
            } else {
                x >= 0.0 && list.contains(&(x as u32))
            };
            assert_eq!(
                n.goes_heavy::<true>(&reused),
                left == *heavy_left,
                "{x}, {list:?}"
            );
        }
    }
    check_model(&mut Rng::new(47), &forest, 300);
}

#[test]
fn any_u32_category_takes_a_code() {
    let mut rng = Rng::new(11);
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
    let mut rng = Rng::new(3);
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
    let mut rng = Rng::new(19);
    for _ in 0..500 {
        let mut book: Vec<u32> = (0..rng.below_mod(64))
            .map(|_| rng.below_mod(64) as u32)
            .collect();
        book.sort_unstable();
        book.dedup();
        let mut books = vec![7; rng.below_mod(3)];
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

/// A categorical feature's bins keep their size with the `bits` flag, as their slot
/// count is a `u32` beside it, and so does a varying feature's entry in the masks'
/// plan, which the group paths index: a model without bit splits runs the same code
/// as before bits.
#[test]
#[cfg(target_pointer_width = "64")]
fn category_bins_keep_their_size() {
    assert_eq!(size_of::<CategoryBins>(), 48);
    assert_eq!(size_of::<masks::MaskFeature>(), 96);
}

#[test]
fn slots_stop_at_the_layout_limit() {
    assert_eq!(MAX_SLOTS, 1 << 27);
    assert!(super::build::slots_fit(MAX_SLOTS).is_ok());
    assert!(matches!(
        super::build::slots_fit(MAX_SLOTS + 1),
        Err(LoadError::Limit(_))
    ));
}
