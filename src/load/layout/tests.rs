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
        pack_above: *rng.pick_mod(&[0, 3, PACK_ABOVE, usize::MAX]),
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
                if !layout.bins.has_shifts() {
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
        categorical += usize::from(slots > 0);
        chained += usize::from(layout.chain_sets > layout.chains);
        bits += usize::from(layout.bins.has_shifts());
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
    categorical_load(splits, varying, &LoadOptions::default(), PACK_ABOVE).unwrap()
}

/// [`categorical_forest`] with `options`, packed past `pack_above` chains; the trees
/// in the model's order, so that tree t holds split t.
fn categorical_load(
    splits: &[(Vec<u32>, bool, bool)],
    varying: bool,
    options: &LoadOptions,
    pack_above: usize,
) -> Result<Forest, LoadError> {
    let varying: &[usize] = if varying { &[0] } else { &[] };
    categorical_features_load(&[splits], varying, options, pack_above)
}

/// [`categorical_load`] of a feature per entry of `features`, a tree per split, the
/// features in `varying` varying.
fn categorical_features_load(
    features: &[&[(Vec<u32>, bool, bool)]],
    varying: &[usize],
    options: &LoadOptions,
    pack_above: usize,
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
        pack_above,
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
    // complement of {1, 2} (no category, missing, 3 and 5) do not nest with them.
    // {5} ⊂ that complement, but each kind is covered apart: three chains.
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
        assert_eq!((layout.chain_sets, layout.bins.chain_slots()), (5, 3));
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
        let forest = categorical_load(&splits, varying, &cap(bytes), PACK_ABOVE).unwrap();
        let (bins, masks) = (&forest.model.layout.bins, &forest.model.masks);
        assert_eq!((bins.rank_table_bytes(), bins.chain_slots()), (212, 10));
        let flips = 4 * (masks.flip_bounds.len() + masks.flips.len());
        assert_eq!(flips, if varying { 112 } else { 0 });
        check_model(&mut rng, &forest, 500);
        assert!(matches!(
            categorical_load(&splits, varying, &cap(bytes - 1), PACK_ABOVE),
            Err(LoadError::Limit(_))
        ));
    }
}

#[test]
fn packed_fields_share_a_word() {
    // The five distinct heavy sets of `nested_heavy_sets_share_a_chain`, packed: the
    // chain {1} ⊂ {1, 2} ⊂ {1, 2, 3} takes 2 bits, {5} and the complement 1 each,
    // one word after the dummy's slot, over the six codes.
    let splits = [
        (vec![1, 2], false, true),
        (vec![1], false, true),
        (vec![5], false, true),
        (vec![1, 2, 3], false, true),
        (vec![1, 2], false, false),
    ];
    let mut rng = Rng::new(37);
    let options = LoadOptions::default();
    for varying in [false, true] {
        let forest = categorical_load(&splits, varying, &options, 0).unwrap();
        let layout = &forest.model.layout;
        let bins = &layout.bins;
        assert_eq!((layout.chain_sets, layout.chains), (5, 3));
        assert_eq!((bins.chain_slots(), bins.has_shifts()), (1, true));
        let c = bins.feature_categories(0).unwrap();
        assert_eq!((c.slot, c.slots()), (1, 1));
        assert_eq!(bins.defaults[1], bins.ranks_of(c, NO_CATEGORY)[0]);
        // Fields of 2, 1 and 1 bits at bits 0, 2 and 3: shifts 30, 29 and 28.
        let mut shifts: Vec<u32> = forest
            .model
            .trees
            .iter()
            .map(|t| layout.nodes[t.node_start as usize].shift())
            .collect();
        shifts.sort_unstable();
        shifts.dedup();
        assert_eq!(shifts, [28, 29, 30]);
        // Each split goes heavy exactly when the value is in its heavy set.
        assert_eq!(bins.feature_book(0), [1, 2, 3, 5]);
        for x in [f64::NAN, 0.0, 4.0, 1.0, 2.0, 3.0, 5.0] {
            let mut b = bins.new_bins();
            bins.fill(&[x], &mut b, true);
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
        // Below the bound, each chain takes its word and nothing shifts.
        let forest = categorical_load(&splits, varying, &options, 3).unwrap();
        let bins = &forest.model.layout.bins;
        assert_eq!((bins.chain_slots(), bins.has_shifts()), (3, false));
        check_model(&mut rng, &forest, 200);
    }
}

#[test]
fn a_lone_field_takes_its_word() {
    // One chain, packed: its word holds no other field, so it takes the word whole.
    let splits = [(vec![1], false, true), (vec![1, 2], false, true)];
    let forest = categorical_load(&splits, false, &LoadOptions::default(), 0).unwrap();
    let bins = &forest.model.layout.bins;
    assert_eq!(
        (
            forest.model.layout.chains,
            bins.chain_slots(),
            bins.has_shifts()
        ),
        (1, 1, false)
    );
    check_model(&mut Rng::new(43), &forest, 200);
}

#[test]
fn packed_fields_span_words() {
    // 160 singletons over 162 codes: 160 1-bit fields in five words. The tables take
    // five defaults, 163 exception bounds and an exception per singleton's code,
    // 1,952 bytes, against 2,572 for 160 chains of a word each.
    let splits: Vec<_> = (0..160).map(|c| (vec![c], false, true)).collect();
    let mut rng = Rng::new(41);
    for varying in [false, true] {
        let forest = categorical_load(&splits, varying, &LoadOptions::default(), 0).unwrap();
        let bins = &forest.model.layout.bins;
        assert_eq!(
            (
                bins.rank_table_bytes(),
                bins.chain_slots(),
                bins.has_shifts()
            ),
            (1952, 5, true)
        );
        check_model(&mut rng, &forest, 200);
        let forest =
            categorical_load(&splits, varying, &LoadOptions::default(), usize::MAX).unwrap();
        let bins = &forest.model.layout.bins;
        assert_eq!((bins.rank_table_bytes(), bins.chain_slots()), (2572, 160));
    }
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
