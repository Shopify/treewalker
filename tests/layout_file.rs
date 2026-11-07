//! Layout files are tested only with public repository fixtures.
#![cfg(feature = "layout-file")]

use std::path::PathBuf;
use treewalker_gbdt::{Forest, LoadError, LoadOptions, ModelFormat, WalkerConfig};
#[path = "common/json.rs"]
mod json;

struct File(PathBuf);
impl File {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "treewalker-layout-{}-{}.twl",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )))
    }
}
impl Drop for File {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

const fn fixtures() -> [&'static [u8]; 13] {
    [
        include_bytes!("fixtures/import/sigmoid_f64.bin"),
        include_bytes!("fixtures/import/sigmoid_f32.bin"),
        include_bytes!("fixtures/import/categories_left.bin"),
        include_bytes!("fixtures/import/categories_f32.bin"),
        include_bytes!("fixtures/import/sigmoid_alpha.bin"),
        include_bytes!("fixtures/import/identity.bin"),
        include_bytes!("fixtures/import/ranking.bin"),
        include_bytes!("fixtures/import/average_base.bin"),
        include_bytes!("fixtures/import/categories_right.bin"),
        include_bytes!("fixtures/import/lt_zero.bin"),
        include_bytes!("fixtures/import/positive_infinity.bin"),
        include_bytes!("fixtures/import/negative_infinity.bin"),
        include_bytes!("fixtures/import/random_forest.bin"),
    ]
}

fn load(bytes: &[u8]) -> Forest {
    Forest::from_bytes(
        bytes,
        ModelFormat::TreeliteBinaryV4,
        WalkerConfig::builder(2)
            .max_group_width(128)
            .varying([0])
            .build()
            .unwrap(),
        &LoadOptions::default(),
    )
    .unwrap()
}

#[test]
fn native_lightgbm_text_round_trips_layout_file() {
    let forest = Forest::from_bytes(
        include_bytes!("fixtures/import/lightgbm/category_words.txt"),
        ModelFormat::LightGbmText,
        WalkerConfig::builder(2)
            .max_group_width(4)
            .all_varying()
            .build()
            .unwrap(),
        &LoadOptions::default(),
    )
    .unwrap();
    let file = File::new();
    forest.save_layout(&file.0).unwrap();
    let loaded = Forest::load_layout(&file.0).unwrap();
    let rows = [1.0, 0.0, 32.0, 65.0, 64.0, f64::NAN, -1.0, 128.0];
    let (mut expected, mut actual) = ([0.0f64; 4], [0.0f64; 4]);
    forest.predictor().predict_group(&rows, &mut expected);
    loaded.predictor().predict_group(&rows, &mut actual);
    assert_eq!(expected.map(f64::to_bits), actual.map(f64::to_bits));
    #[cfg(all(unix, target_pointer_width = "64", not(miri)))]
    {
        // SAFETY: no writes or truncation occur while mapped owners live.
        let mapped = unsafe { Forest::map_layout(&file.0).unwrap() };
        mapped.predictor().predict_group(&rows, &mut actual);
        assert_eq!(expected.map(f64::to_bits), actual.map(f64::to_bits));
    }
}

#[test]
fn many_numeric_features_round_trip() {
    let features = if cfg!(miri) { 8 } else { 512 };
    let trees: Vec<_> =
        (0..features)
            .map(|f| {
                json::stump(
        r#""node_type": "numerical_test_node", "comparison_op": "<=", "threshold": 0.5"#,
        true, (3, 1)).replace("\"split_feature_id\": 0", &format!("\"split_feature_id\": {f}"))
            })
            .collect();
    let model = json::regressor(false, features, &trees);
    for varying in [false, true] {
        let forest = Forest::from_bytes(
            model.as_bytes(),
            ModelFormat::TreeliteJson,
            WalkerConfig::builder(features)
                .max_group_width(4)
                .varying(0..if varying { features } else { 0 })
                .build()
                .unwrap(),
            &LoadOptions::default(),
        )
        .unwrap();
        let file = File::new();
        forest.save_layout(&file.0).unwrap();
        let loaded = Forest::load_layout(&file.0).unwrap();
        let rows = vec![0.25; features * 4];
        let (mut a, mut b) = ([0.0f64; 4], [0.0f64; 4]);
        forest.predictor().predict_group(&rows, &mut a);
        loaded.predictor().predict_group(&rows, &mut b);
        assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
    }
}

#[test]
fn inline_fixed_value_extremes_round_trip() {
    let tiny = f64::from_bits(1);
    let high_shift = 2.0f64.powi(123);
    for [left, right] in [
        [0.0, -0.0],
        [tiny, -tiny],
        [f64::MIN_POSITIVE, -f64::MIN_POSITIVE],
        [f64::MAX, -f64::MAX],
        [1.0, high_shift],
        [-1.0, -high_shift],
        [tiny, f64::MAX],
    ] {
        let tree = json::stump(
            r#""node_type": "numerical_test_node", "comparison_op": "<=", "threshold": 0.5"#,
            true,
            (3, 1),
        )
        .replace("\"leaf_value\": 1.0", &format!("\"leaf_value\": {left:e}"))
        .replace("\"leaf_value\": 2.0", &format!("\"leaf_value\": {right:e}"));
        let model = json::regressor(false, 1, &[tree]);
        let forest = Forest::from_bytes(
            model.as_bytes(),
            ModelFormat::TreeliteJson,
            WalkerConfig::builder(1)
                .max_group_width(4)
                .all_varying()
                .build()
                .unwrap(),
            &LoadOptions::default(),
        )
        .unwrap();
        let file = File::new();
        forest.save_layout(&file.0).unwrap();
        let loaded = Forest::load_layout(&file.0).unwrap();
        let rows = [0.25, 0.75, f64::NAN, 0.5];
        let (mut expected, mut actual) = ([0.0f64; 4], [0.0f64; 4]);
        forest.predictor().predict_group(&rows, &mut expected);
        loaded.predictor().predict_group(&rows, &mut actual);
        assert_eq!(expected.map(f64::to_bits), actual.map(f64::to_bits));
        #[cfg(all(unix, target_pointer_width = "64", not(miri)))]
        {
            // SAFETY: this file is not written until mapped owners drop.
            let mapped = unsafe { Forest::map_layout(&file.0).unwrap() };
            mapped.predictor().predict_group(&rows, &mut actual);
            assert_eq!(expected.map(f64::to_bits), actual.map(f64::to_bits));
        }
    }
}

#[test]
fn many_two_sided_categorical_predicates_round_trip() {
    let count = if cfg!(miri) { 16 } else { 1024 };
    for nested in [false, true] {
        let mut trees = Vec::new();
        for category in 0..count {
            let categories: Vec<_> = if nested {
                (0..=category).collect()
            } else {
                vec![category]
            };
            let test = format!(
                r#""node_type": "categorical_test_node", "category_list": {categories:?}, "category_list_right_child": false"#
            );
            for weights in [(3, 1), (1, 3)] {
                trees.push(json::stump(&test, true, weights));
            }
        }
        let model = json::regressor(false, 1, &trees);
        let forest = Forest::from_bytes(
            model.as_bytes(),
            ModelFormat::TreeliteJson,
            WalkerConfig::builder(1)
                .max_group_width(4)
                .all_varying()
                .build()
                .unwrap(),
            &LoadOptions::default(),
        )
        .unwrap();
        let file = File::new();
        forest.save_layout(&file.0).unwrap();
        let loaded = Forest::load_layout(&file.0).unwrap();
        let rows = [0.0, 7.0, f64::from(count - 1), f64::NAN];
        let (mut expected, mut actual) = ([0.0f64; 4], [0.0f64; 4]);
        forest.predictor().predict_group(&rows, &mut expected);
        loaded.predictor().predict_group(&rows, &mut actual);
        assert_eq!(expected.map(f64::to_bits), actual.map(f64::to_bits));
        #[cfg(all(unix, target_pointer_width = "64", not(miri)))]
        {
            // SAFETY: the file remains immutable until mapped owners drop.
            let mapped = unsafe { Forest::map_layout(&file.0).unwrap() };
            mapped.predictor().predict_group(&rows, &mut actual);
            assert_eq!(expected.map(f64::to_bits), actual.map(f64::to_bits));
        }
    }
}

#[test]
fn copy_round_trips_numeric_and_categorical_precisions() {
    let path = File::new();
    for fixture in fixtures() {
        let forest = load(fixture);
        forest.save_layout(&path.0).unwrap();
        let loaded = Forest::load_layout(&path.0).unwrap();
        let rows = [0.0, 1.0, 0.1, 1.0, f64::NAN, 1.0, 10.0, 1.0];
        let (mut a, mut b) = ([0.0f64; 4], [0.0f64; 4]);
        forest.predictor().predict_group(&rows, &mut a);
        loaded.predictor().predict_group(&rows, &mut b);
        assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
        assert_eq!(forest.config(), loaded.config());
        #[cfg(all(unix, target_pointer_width = "64", not(miri)))]
        {
            // SAFETY: this test owns the file exclusively and does not write it
            // again until every forest/predictor retaining the mapping is dropped.
            let mapped = unsafe { Forest::map_layout(&path.0).unwrap() };
            #[cfg(feature = "research")]
            {
                mapped.predict_full_walk(&rows, &mut b);
                assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
            }
            let clone = mapped.clone();
            let mut predictor = mapped.predictor();
            drop(mapped);
            drop(clone);
            predictor.predict_group(&rows, &mut b);
            assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
        }
    }
}

#[test]
fn corrupt_truncated_and_extended_files_are_errors_not_panics() {
    let path = File::new();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for (fixture_index, fixture) in fixtures().into_iter().enumerate() {
        load(fixture).save_layout(&path.0).unwrap();
        let bytes = std::fs::read(&path.0).unwrap();
        let link = bytes.windows(6).position(|b| b == b"links=").unwrap();
        let header_end = link + bytes[link..].iter().position(|&b| b == 0).unwrap() + 1;
        for i in 0..if fixture_index == 0 { header_end } else { 0 } {
            let mut corrupt = bytes.clone();
            corrupt[i] ^= 1;
            std::fs::write(&path.0, corrupt).unwrap();
            assert!(matches!(
                Forest::load_layout(&path.0),
                Err(LoadError::Layout(_))
            ));
        }
        for n in [0, 1, 7, 8, bytes.len() / 2, bytes.len() - 1] {
            std::fs::write(&path.0, &bytes[..n]).unwrap();
            assert!(matches!(
                Forest::load_layout(&path.0),
                Err(LoadError::Layout(_))
            ));
        }
        for _ in 0..if cfg!(miri) { 8 } else { 128 } {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut corrupt = bytes.clone();
            corrupt[state as usize % bytes.len()] ^= 1 << (state % 8);
            std::fs::write(&path.0, &corrupt).unwrap();
            assert!(Forest::load_layout(&path.0).is_err());
            #[cfg(all(unix, target_pointer_width = "64", not(miri)))]
            {
                // SAFETY: no mapping from a previous call remains alive; no
                // writes occur during this call or while its result is alive.
                assert!(unsafe { Forest::map_layout(&path.0) }.is_err());
            }
        }
        let mut extended = bytes;
        extended.push(0);
        std::fs::write(&path.0, extended).unwrap();
        assert!(Forest::load_layout(&path.0).is_err());
    }
}
