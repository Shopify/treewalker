//! Predictor call contracts: group shapes, empty input, and the panics that reject
//! invalid calls before any output is written.
use std::panic::{AssertUnwindSafe, catch_unwind};
use treewalker_gbdt::{Forest, ModelFormat, ParseConfig, Predictor, WalkerConfig};

/// The two-feature sigmoid fixture: feature 0 varies, feature 1 is constant.
fn forest(max_group_width: usize) -> Forest {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/import/sigmoid_f64.bin"
    );
    let config = WalkerConfig::try_new(2, max_group_width, &[0], &[], &[]).unwrap();
    Forest::from_bytes(
        &std::fs::read(path).unwrap(),
        ModelFormat::TreeliteBinaryV4,
        config,
        &ParseConfig::default(),
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
