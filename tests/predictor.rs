//! Predictor call contracts: group shapes, empty input, and the panics that reject
//! invalid calls before any output is written.
use std::panic::{AssertUnwindSafe, catch_unwind};
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat, Predictor, WalkerConfig};

/// The two-feature sigmoid fixture: feature 0 varies, feature 1 is constant.
fn forest(max_group_width: usize) -> Forest {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/import/sigmoid_f64.bin"
    );
    let config = WalkerConfig::builder(2)
        .max_group_width(max_group_width)
        .varying([0])
        .build()
        .unwrap();
    Forest::from_bytes(
        &std::fs::read(path).unwrap(),
        ModelFormat::TreeliteBinaryV4,
        config,
        &LoadOptions::default(),
    )
    .unwrap()
}

/// `n` rows whose varying feature sweeps the fixture's split points; the constant
/// feature is the same in every row, so any grouping is valid.
fn rows(n: usize) -> Vec<f64> {
    (0..n)
        .flat_map(|r| [(r % 17) as f64 / 4.0 - 1.0, 0.25])
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Run `call` on a NaN-filled output of `n` rows, require a panic whose message
/// contains `expected`, and require that the output was not touched.
fn rejects(expected: &str, n: usize, call: impl FnOnce(&mut Predictor, &mut [f64])) {
    rejects_with(32, expected, n, call);
}

/// As [`rejects`], with a predictor for groups of up to `max_group_width` rows.
fn rejects_with(
    max_group_width: usize,
    expected: &str,
    n: usize,
    call: impl FnOnce(&mut Predictor, &mut [f64]),
) {
    let mut p = forest(max_group_width).predictor();
    let mut out = vec![f64::NAN; n];
    let err = catch_unwind(AssertUnwindSafe(|| call(&mut p, &mut out))).unwrap_err();
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&str>().copied())
        .unwrap_or_default();
    assert!(msg.contains(expected), "{msg:?} lacks {expected:?}");
    assert!(out.iter().all(|v| v.is_nan()), "{expected}: output written");
}

#[test]
fn the_three_calls_agree_on_every_grouping() {
    let data = rows(100);
    let mut p = forest(32).predictor();
    let mut alone = vec![0.0; 100];
    for (row, out) in data.chunks(2).zip(alone.chunks_mut(1)) {
        p.predict_group(row, out);
    }
    for width in [1, 3, 16, 17, 32] {
        // 100 rows: the last group is shorter unless width divides 100.
        let mut fixed = vec![f64::NAN; 100];
        p.predict_fixed(&data, width, &mut fixed);
        assert_eq!(bits(&fixed), bits(&alone), "fixed {width}");
        let offsets: Vec<usize> = (0..100).step_by(width).chain([100]).collect();
        let mut grouped = vec![f64::NAN; 100];
        p.predict_groups(&data, &offsets, &mut grouped);
        assert_eq!(bits(&grouped), bits(&alone), "offsets {width}");
    }
    let offsets = [0, 1, 33, 34, 66, 98, 100];
    let mut grouped = vec![f64::NAN; 100];
    p.predict_groups(&data, &offsets, &mut grouped);
    assert_eq!(bits(&grouped), bits(&alone), "uneven offsets");
}

#[test]
fn empty_input_is_a_no_op() {
    let mut p = forest(32).predictor();
    p.predict_group(&[], &mut []);
    p.predict_groups(&[], &[0], &mut []);
    p.predict_fixed(&[], 5, &mut []);
}

#[test]
fn width_zero_is_rejected() {
    rejects("group width must be positive", 4, |p, out| {
        p.predict_fixed(&rows(4), 0, out);
    });
    rejects("group width must be positive", 0, |p, out| {
        p.predict_fixed(&[], 0, out);
    });
}

#[test]
fn groups_wider_than_max_group_width_are_rejected() {
    rejects(
        "group of 33 rows exceeds max_group_width 32",
        33,
        |p, out| {
            p.predict_group(&rows(33), out);
        },
    );
    rejects("group width 33 exceeds max_group_width 32", 40, |p, out| {
        p.predict_fixed(&rows(40), 33, out);
    });
    rejects(
        "group of 33 rows exceeds max_group_width 32",
        40,
        |p, out| {
            p.predict_groups(&rows(40), &[0, 7, 40], out);
        },
    );
}

#[test]
fn invalid_offsets_are_rejected() {
    rejects("offsets must start at 0", 0, |p, out| {
        p.predict_groups(&[], &[], out);
    });
    rejects("offsets must start at 0", 4, |p, out| {
        p.predict_groups(&rows(4), &[], out);
    });
    rejects("offsets must start at 0", 4, |p, out| {
        p.predict_groups(&rows(4), &[1, 4], out);
    });
    rejects("offsets must end at the row count 4", 4, |p, out| {
        p.predict_groups(&rows(4), &[0, 3], out);
    });
    rejects("offsets must strictly increase", 4, |p, out| {
        p.predict_groups(&rows(4), &[0, 2, 2, 4], out);
    });
    rejects("offsets must strictly increase", 4, |p, out| {
        p.predict_groups(&rows(4), &[0, 3, 2, 4], out);
    });
}

#[test]
fn mismatched_lengths_are_rejected() {
    rejects(
        "input length 7 is not a multiple of n_features 2",
        3,
        |p, out| {
            p.predict_group(&rows(4)[..7], out);
        },
    );
    rejects(
        "output length 3 differs from the row count 4",
        3,
        |p, out| {
            p.predict_group(&rows(4), out);
        },
    );
    rejects(
        "output length 5 differs from the row count 4",
        5,
        |p, out| {
            p.predict_fixed(&rows(4), 2, out);
        },
    );
    rejects(
        "output length 3 differs from the row count 4",
        3,
        |p, out| {
            p.predict_groups(&rows(4), &[0, 4], out);
        },
    );
}

#[test]
fn bounds_near_usize_max_do_not_overflow() {
    // Fixed groups of usize::MAX rows: the last group's end is clamped to the row
    // count instead of computed as start + width.
    let unbounded = forest(usize::MAX);
    let data = rows(1500);
    let mut p = unbounded.predictor();
    let (mut whole, mut fixed) = (vec![0.0; 1500], vec![f64::NAN; 1500]);
    p.predict_group(&data, &mut whole);
    p.predict_fixed(&data, usize::MAX, &mut fixed);
    assert_eq!(bits(&fixed), bits(&whole));
    p.predict_fixed(&data, usize::MAX - 1, &mut fixed);
    assert_eq!(bits(&fixed), bits(&whole));
    // Offsets past the row count fail their check rather than overflow a bound.
    for max in [32, usize::MAX] {
        rejects_with(max, "offsets must end at the row count 4", 4, |p, out| {
            p.predict_groups(&rows(4), &[0, usize::MAX], out);
        });
    }
    rejects_with(usize::MAX, "offsets must strictly increase", 4, |p, out| {
        p.predict_groups(&rows(4), &[0, usize::MAX, 4], out);
    });
    rejects("group of 18446744073709551615 rows exceeds", 4, |p, out| {
        p.predict_groups(&rows(4), &[0, usize::MAX, 4], out);
    });
}

#[test]
fn forests_and_predictors_move_across_threads() {
    let forest = forest(32);
    let data = rows(64);
    let mut expected = vec![0.0; 64];
    forest.predictor().predict_fixed(&data, 32, &mut expected);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let (forest, data) = (forest.clone(), data.clone());
            std::thread::spawn(move || {
                let mut out = vec![0.0; 64];
                forest.predictor().predict_fixed(&data, 32, &mut out);
                out
            })
        })
        .collect();
    for h in handles {
        assert_eq!(bits(&h.join().unwrap()), bits(&expected));
    }
    // A predictor keeps its model alive after the forest is dropped.
    let mut p = forest.predictor();
    drop(forest);
    let mut out = vec![0.0; 64];
    std::thread::spawn(move || p.predict_fixed(&data, 32, &mut out))
        .join()
        .unwrap();
}

/// Six trees whose first two heavy-path nodes are the same constant splits on
/// feature 1, so they form one prefix group at the default depth; below them, splits
/// on the varying feature 0. Constants above 0, in (-2, 0] and at most -2 leave the
/// prefix at levels 0, 1 and 2.
fn prefix_forest() -> Forest {
    let tree = |t: usize| {
        let leaf = |id: usize, k: usize, count: u32| {
            format!(
                r#"{{"node_id":{id},"leaf_value":{},"data_count":{count}}}"#,
                (t * 8 + k) as f64 / 64.0 - 0.3
            )
        };
        let split = |id: usize, f: usize, thr: f64, l: usize, r: usize, count: u32| {
            format!(
                r#"{{"node_id":{id},"split_feature_id":{f},"default_left":true,"node_type":"numerical_test_node","comparison_op":"<=","threshold":{thr:?},"left_child":{l},"right_child":{r},"data_count":{count}}}"#
            )
        };
        let nodes = [
            split(0, 1, 0.0, 1, 2, 100),
            split(1, 1, -2.0, 3, 4, 70),
            split(2, 0, 1.0, 5, 6, 30),
            split(3, 0, 0.5 + t as f64 / 8.0, 7, 8, 50),
            leaf(4, 0, 20),
            leaf(5, 1, 20),
            leaf(6, 2, 10),
            leaf(7, 3, 30),
            leaf(8, 4, 20),
        ];
        format!(
            r#"{{"num_nodes":9,"has_categorical_split":false,"nodes":[{}]}}"#,
            nodes.join(",")
        )
    };
    let trees: Vec<String> = (0..6).map(tree).collect();
    let json = format!(
        r#"{{"threshold_type":"float64","leaf_output_type":"float64","num_feature":2,"task_type":"kBinaryClf","average_tree_output":false,"num_target":1,"num_class":[1],"leaf_vector_shape":[1,1],"target_id":[0,0,0,0,0,0],"class_id":[0,0,0,0,0,0],"postprocessor":"sigmoid","sigmoid_alpha":1.0,"ratio_c":1.0,"base_scores":[0.125],"attributes":"{{}}","trees":[{}]}}"#,
        trees.join(",")
    );
    let config = WalkerConfig::builder(2)
        .max_group_width(32)
        .varying([0])
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

/// Rows of one group whose constant feature is `constant`.
fn rows_with_constant(n: usize, constant: f64) -> Vec<f64> {
    (0..n)
        .flat_map(|r| [(r % 17) as f64 / 4.0 - 1.0, constant])
        .collect()
}

/// A reused predictor's workspace (prefix starts, difference array) carries nothing
/// from one call to the next: after rejected calls and groups of alternating widths
/// and constants, each result equals a fresh predictor's.
#[test]
fn reused_predictors_match_fresh_ones() {
    let constants = [0.25, -3.0, -1.0, f64::NAN, 0.0, 1.0e6, -2.0];
    let widths = [1, 32, 3, 17, 2, 31, 16, 1, 32, 5];
    for forest in [forest(32), prefix_forest()] {
        let mut reused = forest.predictor();
        for (i, &width) in widths.iter().enumerate() {
            let constant = constants[i % constants.len()];
            // Rejected before any work: a group wider than max_group_width.
            let mut wide = vec![f64::NAN; 33];
            let rejected = catch_unwind(AssertUnwindSafe(|| {
                reused.predict_group(&rows_with_constant(33, constant), &mut wide);
            }));
            assert!(rejected.is_err(), "width 33 must be rejected");
            let data = rows_with_constant(width, constant);
            let mut got = vec![f64::NAN; width];
            reused.predict_group(&data, &mut got);
            let mut want = vec![f64::NAN; width];
            forest.predictor().predict_group(&data, &mut want);
            assert_eq!(
                bits(&got),
                bits(&want),
                "width {width}, constant {constant}"
            );
        }
        // One call over groups of alternating widths and constants.
        let mut data = Vec::new();
        let mut offsets = vec![0];
        for (i, &width) in widths.iter().enumerate() {
            data.extend(rows_with_constant(width, constants[i % constants.len()]));
            offsets.push(offsets.last().unwrap() + width);
        }
        let n = *offsets.last().unwrap();
        let mut got = vec![f64::NAN; n];
        reused.predict_groups(&data, &offsets, &mut got);
        let mut want = vec![f64::NAN; n];
        for g in 0..widths.len() {
            let (s, e) = (offsets[g], offsets[g + 1]);
            forest
                .predictor()
                .predict_group(&data[2 * s..2 * e], &mut want[s..e]);
        }
        assert_eq!(bits(&got), bits(&want), "alternating groups in one call");
    }
}
