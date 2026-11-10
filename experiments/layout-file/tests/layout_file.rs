//! Layout files are tested only with public repository fixtures.

use std::path::PathBuf;
use treewalker_gbdt::{Forest, LoadError, LoadOptions, ModelFormat, WalkerConfig};
#[path = "../../../tests/common/json.rs"]
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
        include_bytes!("../../../tests/fixtures/import/sigmoid_f64.bin"),
        include_bytes!("../../../tests/fixtures/import/sigmoid_f32.bin"),
        include_bytes!("../../../tests/fixtures/import/categories_left.bin"),
        include_bytes!("../../../tests/fixtures/import/categories_f32.bin"),
        include_bytes!("../../../tests/fixtures/import/sigmoid_alpha.bin"),
        include_bytes!("../../../tests/fixtures/import/identity.bin"),
        include_bytes!("../../../tests/fixtures/import/ranking.bin"),
        include_bytes!("../../../tests/fixtures/import/average_base.bin"),
        include_bytes!("../../../tests/fixtures/import/categories_right.bin"),
        include_bytes!("../../../tests/fixtures/import/lt_zero.bin"),
        include_bytes!("../../../tests/fixtures/import/positive_infinity.bin"),
        include_bytes!("../../../tests/fixtures/import/negative_infinity.bin"),
        include_bytes!("../../../tests/fixtures/import/random_forest.bin"),
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
        include_bytes!("../../../tests/fixtures/import/lightgbm/category_words.txt"),
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
        // A header that starts as a layout file's is another build's, else not one.
        // Its message escapes the file's bytes: an escape character, a byte that is
        // not UTF-8.
        for i in 0..if fixture_index == 0 { header_end } else { 0 } {
            for byte in [bytes[i] ^ 1, 0x1b, 0xff] {
                let mut corrupt = bytes.clone();
                corrupt[i] = byte;
                std::fs::write(&path.0, corrupt).unwrap();
                let error = Forest::load_layout(&path.0).unwrap_err();
                if i < b"TWLAYOUT\0".len() {
                    assert!(matches!(error, LoadError::Layout(_)), "{error}");
                } else {
                    assert!(matches!(error, LoadError::LayoutMismatch(_)), "{error}");
                }
                let message = error.to_string();
                assert!(
                    message.bytes().all(|b| (0x20..0x7f).contains(&b)),
                    "{message:?}"
                );
            }
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

// Independent implementation of the public format's checksum, not library code.
fn checksum(bytes: &[u8]) -> u64 {
    let mut state = 0u64;
    let mut word = |value: u64| {
        state = (state.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95);
    };
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        word(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    let tail = chunks.remainder();
    let mut padded = [0; 8];
    padded[..tail.len()].copy_from_slice(tail);
    word(u64::from_le_bytes(padded));
    word(tail.len() as u64);
    state
}

fn reject_rechecksummed(path: &File, mut bytes: Vec<u8>) {
    let end = bytes.len() - 8;
    let sum = checksum(&bytes[..end]);
    bytes[end..].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(&path.0, bytes).unwrap();
    assert!(matches!(
        Forest::load_layout(&path.0),
        Err(LoadError::Layout(_))
    ));
    #[cfg(all(unix, target_pointer_width = "64"))]
    {
        // SAFETY: the test owns the file, and no write occurs while the call's
        // result lives. A previous result has dropped before the next mutation.
        assert!(matches!(
            unsafe { Forest::map_layout(&path.0) },
            Err(LoadError::Layout(_))
        ));
    }
}

fn source_build_id() -> String {
    use std::path::Path;
    fn collect(root: &Path, relative: &Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(root.join(relative)).unwrap() {
            let entry = entry.unwrap();
            let path = relative.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                collect(root, &path, files);
            } else {
                files.push(path);
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = vec![PathBuf::from("build.rs")];
    collect(&root, Path::new("src"), &mut files);
    files.sort_unstable();
    let mut input = Vec::new();
    for path in files {
        input.extend_from_slice(path.to_str().unwrap().replace('\\', "/").as_bytes());
        input.extend_from_slice(&std::fs::read(root.join(path)).unwrap());
    }
    format!("{:016x}", checksum(&input))
}

/// Locate fields in the small numeric fixture's production schema. These are
/// wire sizes, not Rust object sizes; assertions make schema changes explicit.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl Cursor<'_> {
    fn byte(&mut self) -> u8 {
        let value = self.bytes[self.at];
        self.at += 1;
        value
    }
    fn count(&mut self) -> usize {
        let n = u64::from_ne_bytes(self.bytes[self.at..self.at + 8].try_into().unwrap());
        self.at += 8;
        n as usize
    }
    fn array(&mut self, width: usize) -> std::ops::Range<usize> {
        let len = self.count();
        self.at = (self.at + 15) & !15;
        let data = self.at..self.at + len * width;
        self.at = data.end;
        data
    }
}

#[test]
fn rechecksummed_body_corruption_is_rejected_by_production_readers() {
    let model = json::regressor(
        false,
        1,
        &[json::stump(
            r#""node_type": "numerical_test_node", "comparison_op": "<=", "threshold": 0.5"#,
            true,
            (3, 1),
        )],
    );
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
    let path = File::new();
    forest.save_layout(&path.0).unwrap();
    let bytes = std::fs::read(&path.0).unwrap();
    let footer = bytes.len() - 8;
    assert_eq!(
        checksum(&bytes[..footer]),
        u64::from_le_bytes(bytes[footer..].try_into().unwrap())
    );
    let link = bytes.windows(6).position(|b| b == b"links=").unwrap();
    let body = link + bytes[link..].iter().position(|&b| b == 0).unwrap() + 1;
    let mut c = Cursor {
        bytes: &bytes,
        at: body + 8 + 1,
    }; // fingerprint, precision
    let divisor = c.at + 8;
    c.at += 16; // Output's two f64s
    assert_eq!(c.byte(), 0); // Identity
    let trees_count = c.at;
    let trees = c.array(12);
    assert_eq!(trees.len(), 12);
    let config = c.at;
    c.at += 16; // feature count, group width
    let varying_tag = c.at;
    for _ in 0..3 {
        c.byte();
        c.array(4);
    } // the three FeatureSets
    let predicates = c.count();
    c.at = (c.at + 15) & !15;
    assert_eq!(predicates, 1);
    assert_eq!(c.byte(), 0); // numeric: feature u32, threshold f64, missing bool
    c.at += 13;
    c.byte(); // threshold precision; no mask plan, which the loader rebuilds
    assert!(c.array(16).is_empty()); // no constant prefixes
    c.at += 48; // PrefixStats
    assert_eq!(c.array(4).len(), 4); // one root
    c.at += 16; // Steps
    assert_eq!(c.byte(), 0); // Fixed plan, exponent i32
    c.at += 4;
    let nodes_count = c.at;
    let nodes = c.array(16);
    assert_eq!(nodes.len(), 4 * 16); // split, two leaves, dummy
    let root_w1 = u64::from_ne_bytes(bytes[nodes.start + 8..nodes.start + 16].try_into().unwrap());
    let changes: Vec<(usize, Vec<u8>)> = vec![
        (divisor, 0f64.to_ne_bytes().to_vec()),
        (trees_count, u64::MAX.to_ne_bytes().to_vec()),
        (trees.start + 4, u32::MAX.to_ne_bytes().to_vec()),
        (config + 8, 0u64.to_ne_bytes().to_vec()),
        (varying_tag, vec![2]),
        (nodes_count, u64::MAX.to_ne_bytes().to_vec()),
        (nodes.start, u64::MAX.to_ne_bytes().to_vec()), // split slot out of range
        (
            nodes.start + 8,
            (root_w1 | u64::from(u32::MAX)).to_ne_bytes().to_vec(),
        ), // child
        (nodes.start + 8, (root_w1 | 3 << 33).to_ne_bytes().to_vec()), // predicate ID
        (nodes.start + 16, 1u64.to_ne_bytes().to_vec()), // Fixed leaf's w0
        (nodes.end - 8, 0u64.to_ne_bytes().to_vec()),   // dummy self-loop
    ];
    for (at, value) in changes {
        let mut corrupt = bytes.clone();
        corrupt[at..at + value.len()].copy_from_slice(&value);
        reject_rechecksummed(&path, corrupt);
    }
    // Deterministic body flips target a Fixed leaf's zero w0: every bit must be
    // structurally rejected after the corruption checksum has been repaired.
    for bit in 0..64 {
        let mut corrupt = bytes.clone();
        corrupt[nodes.start + 16 + bit / 8] ^= 1 << (bit % 8);
        reject_rechecksummed(&path, corrupt);
    }
    let build = bytes.windows(6).position(|b| b == b"build=").unwrap() + 6;
    assert_eq!(
        std::str::from_utf8(&bytes[build..build + 16]).unwrap(),
        source_build_id()
    );
    // Another build's file is a mismatch, to rebuild from the source model, that
    // names both builds, even with its checksum repaired.
    let mut other_build = bytes;
    other_build[build] = if other_build[build] == b'0' {
        b'1'
    } else {
        b'0'
    };
    let theirs = String::from_utf8(other_build[build..build + 16].to_vec()).unwrap();
    let end = other_build.len() - 8;
    let sum = checksum(&other_build[..end]);
    other_build[end..].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(&path.0, other_build).unwrap();
    let mismatched = |error: LoadError| {
        let LoadError::LayoutMismatch(message) = &error else {
            panic!("{error}");
        };
        assert!(message.contains(&format!("build={theirs}")), "{message}");
        assert!(
            message.contains(&format!("build={}", source_build_id())),
            "{message}"
        );
    };
    mismatched(Forest::load_layout(&path.0).unwrap_err());
    #[cfg(all(unix, target_pointer_width = "64"))]
    // SAFETY: the test owns the file, and no write occurs while the call's result
    // lives.
    mismatched(unsafe { Forest::map_layout(&path.0) }.unwrap_err());
}
