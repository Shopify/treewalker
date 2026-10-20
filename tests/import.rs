//! Public import contract, independent Treelite/GTIL fixtures, and malformed input guards.
#![expect(clippy::float_cmp, reason = "exact hand-computable expectations")]

mod common;

use common::bits::same;
use common::fixtures::{self, bytes, values};
use serde::Deserialize;
use simd_json::{OwnedValue as Value, prelude::*};
use std::io::Cursor;
use treewalker_gbdt::research::Ablation;
use treewalker_gbdt::{Forest, LoadError, LoadOptions, ModelFormat, WalkerConfig};

fn config(width: usize) -> WalkerConfig {
    WalkerConfig::builder(2)
        .max_group_width(width)
        .varying([0])
        .build()
        .unwrap()
}
fn load(name: &str, format: ModelFormat, width: usize) -> Result<Forest, LoadError> {
    Forest::from_bytes(&bytes(name), format, config(width), &LoadOptions::default())
}
fn model_json() -> Value {
    simd_json::to_owned_value(&mut bytes("sigmoid_f64.json")).unwrap()
}
fn parse_json(v: &Value) -> Result<Forest, LoadError> {
    Forest::from_bytes(
        &simd_json::to_vec(v).unwrap(),
        ModelFormat::TreeliteJson,
        config(128),
        &LoadOptions::default(),
    )
}
/// Predict each 128-row block of the two-feature fixture data in groups of `width` rows.
fn predict_blocks(forest: &Forest, data: &[f64], width: usize, out: &mut [f64]) {
    let mut p = forest.predictor();
    for (block, out) in data.chunks(128 * 2).zip(out.chunks_mut(128)) {
        p.predict_fixed(block, width, out);
    }
}
/// As [`predict_blocks`], through the research timed build with `ablation`.
fn predict_blocks_ablated(
    forest: &Forest,
    ablation: Ablation,
    data: &[f64],
    width: usize,
    out: &mut [f64],
) {
    let mut r = forest.research_predictor(ablation);
    for (block, out) in data.chunks(128 * 2).zip(out.chunks_mut(128)) {
        for (rows, out) in block.chunks(width * 2).zip(out.chunks_mut(width)) {
            r.predict_group(rows, out);
        }
    }
}
fn check(actual: &[f64], expected: &[f64], atol: f64, rtol: f64, context: &str) {
    for (i, (&x, &y)) in actual.iter().zip(expected).enumerate() {
        assert!(
            x.is_finite() && (x - y).abs() <= rtol.mul_add(y.abs(), atol),
            "{context}: row {i}: {x} != {y}"
        );
    }
}
#[derive(Deserialize)]
struct Manifest {
    models: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    json: bool,
    atol: f64,
    rtol: f64,
}

#[test]
fn independent_oracles_all_formats_paths_and_workspace_widths() {
    let manifest: Manifest = simd_json::serde::from_slice(&mut bytes("manifest.json")).unwrap();
    let data = values("data.bin");
    let rows = data.len() / 2;
    for case in manifest.models {
        let expected = values(&format!("{}_reference.bin", case.name));
        for (extension, format) in [
            ("bin", ModelFormat::TreeliteBinaryV4),
            ("json", ModelFormat::TreeliteJson),
        ] {
            if extension == "json" && !case.json {
                continue;
            }
            let name = format!("{}.{extension}", case.name);
            for width in [1, 16, 17, 32, 33, 64, 65, 128] {
                let forest = load(&name, format, width).unwrap();
                let mut actual = vec![0.0; rows];
                forest.predict_full_walk(&data, &mut actual);
                check(&actual, &expected, case.atol, case.rtol, &name);
                predict_blocks(&forest, &data, width, &mut actual);
                check(&actual, &expected, case.atol, case.rtol, &name);
                for ablation in [
                    Ablation::default(),
                    Ablation {
                        disable_varying_precompute: true,
                        ..Default::default()
                    },
                    Ablation {
                        disable_predicate_sweep: true,
                        ..Default::default()
                    },
                    Ablation {
                        disable_g1_constant: true,
                        disable_width_dispatch: true,
                        ..Default::default()
                    },
                ] {
                    predict_blocks_ablated(&forest, ablation, &data, width, &mut actual);
                    check(&actual, &expected, case.atol, case.rtol, &name);
                }
            }
        }
    }
}

#[test]
fn leaves_past_the_126_bit_rule_sum_exactly() {
    // Leaf exponents 2^-997 apart leave no 126-bit fixed point: prediction and the full
    // walk sum in the wide accumulator, so the two agree bit for bit.
    let mut model = simd_json::to_owned_value(&mut bytes("identity.json")).unwrap();
    model["trees"][0]["nodes"][1]["leaf_value"] = 1e300.into();
    model["trees"][1]["nodes"][1]["leaf_value"] = 1e-300.into();
    let json = simd_json::to_vec(&model).unwrap();
    let forest = Forest::from_bytes(
        &json,
        ModelFormat::TreeliteJson,
        config(128),
        &LoadOptions::default(),
    )
    .unwrap();
    // The wide accumulator's leaf table holds each leaf's first slot and three
    // digits, 16 bytes, as an i128 leaf takes.
    let leaves = forest.nodes().iter().filter(|n| n.is_leaf()).count();
    assert_eq!(forest.layout_stats().leaf_bytes, 16 * leaves);
    let data = values("data.bin");
    let rows = data.len() / 2;
    let (mut partial, mut full) = (vec![0.0; rows], vec![0.0; rows]);
    forest.predictor().predict_fixed(&data, 128, &mut partial);
    forest.predict_full_walk(&data, &mut full);
    assert!(partial.iter().any(|&v| v == 1e300));
    same(&partial, &full, "prediction against the full walk");
}

#[test]
fn native_random_forest_and_hand_computable_aggregation() {
    let forest = load("random_forest.bin", ModelFormat::TreeliteBinaryV4, 128).unwrap();
    let data = values("rf_native_data.bin");
    let expected = values("rf_native_reference.bin");
    let mut actual = vec![0.0; expected.len()];
    forest.predict_full_walk(&data, &mut actual);
    check(&actual, &expected, 1e-14, 1e-14, "sklearn");
    let forest = load("average_base.bin", ModelFormat::TreeliteBinaryV4, 128).unwrap();
    let mut result = [0.0];
    forest.predictor().predict_group(&[0.0, 0.0], &mut result);
    assert_eq!(result, [3.375]);
}

#[test]
fn file_reader_and_memory_loaders() {
    let cfg = fixtures::path("walker_config.json");
    let a = Forest::load(fixtures::path("sigmoid_f64.bin"), &cfg).unwrap();
    let b = Forest::from_reader(
        Cursor::new(bytes("sigmoid_f64.bin")),
        ModelFormat::TreeliteBinaryV4,
        config(128),
        &LoadOptions::default(),
    )
    .unwrap();
    let c = Forest::load(fixtures::path("sigmoid_f64.json"), &cfg).unwrap();
    let d = Forest::load_with(
        fixtures::path("sigmoid_f64.bin"),
        &cfg,
        &LoadOptions::default(),
    )
    .unwrap();
    let input = values("data.bin");
    let expected = values("sigmoid_f64_reference.bin");
    for f in [a, b, c, d] {
        let mut output = vec![0.0; expected.len()];
        f.predict_full_walk(&input, &mut output);
        check(&output, &expected, 1e-14, 0.0, "loader");
    }
    assert!(matches!(
        Forest::load("missing.bin", &cfg),
        Err(LoadError::Io(_))
    ));
    assert!(matches!(
        Forest::load("unknown.format", &cfg),
        Err(LoadError::Unsupported(_))
    ));
    let mut config_file = bytes("walker_config.json");
    let _: WalkerConfig = WalkerConfig::from_json(&config_file).unwrap();
    config_file.truncate(10);
    assert!(WalkerConfig::from_json(&config_file).is_err());
    assert_eq!(WalkerConfig::from_file(&cfg).unwrap().n_features(), 2);
    assert!(matches!(
        WalkerConfig::from_file("missing.json"),
        Err(LoadError::Io(_))
    ));
}

#[test]
fn grouping_schema_and_configuration_validation() {
    let schema = br#"{"n_features":4,"max_group_width":14,"varying_features":[1,2,3],"mono_inc_features":[3],"mono_dec_features":[1]}"#;
    let parsed = WalkerConfig::from_json(schema).unwrap();
    assert_eq!((parsed.n_features(), parsed.max_group_width()), (4, 14));
    let roles = |c: &WalkerConfig| {
        (0..=c.n_features())
            .map(|f| (c.is_varying(f), c.is_increasing(f), c.is_decreasing(f)))
            .collect::<Vec<_>>()
    };
    let varying = (true, false, false);
    let (inc, dec, constant) = (
        (true, true, false),
        (true, false, true),
        (false, false, false),
    );
    assert_eq!(roles(&parsed), [constant, dec, varying, inc, constant]);
    // The builder gives the same configuration, and monotonic features imply varying.
    let built = WalkerConfig::builder(4)
        .max_group_width(14)
        .varying([2])
        .increasing([3])
        .decreasing([1])
        .build()
        .unwrap();
    assert_eq!(built, parsed);
    let all = WalkerConfig::builder(3)
        .max_group_width(1)
        .all_varying()
        .increasing([0])
        .build()
        .unwrap();
    assert_eq!(roles(&all), [inc, varying, varying, constant]);
    let builder = |nf: usize| WalkerConfig::builder(nf).max_group_width(1).varying([]);
    // Treelite's num_feature is a 32-bit signed integer.
    for nf in [0, i32::MAX as usize + 1, usize::MAX] {
        assert!(builder(nf).build().is_err());
    }
    for nf in [65, 128, 300, i32::MAX as usize] {
        let c = builder(nf)
            .varying([nf - 1])
            .increasing([0])
            .build()
            .unwrap();
        assert!(c.is_varying(nf - 1) && c.is_varying(0) && c.is_increasing(0));
        assert!(!c.is_varying(nf / 2) && !c.is_varying(nf) && !c.is_decreasing(0));
    }
    // Equal role sets compare equal however they were declared.
    let listed = builder(130).varying(0..130).build().unwrap();
    let all = WalkerConfig::builder(130)
        .max_group_width(1)
        .all_varying()
        .build()
        .unwrap();
    assert_eq!(listed, all);
    assert_eq!(
        builder(300).varying([5, 299]).build().unwrap(),
        builder(300).varying([299, 5]).build().unwrap()
    );
    assert_ne!(
        builder(300).varying([299]).increasing([5]).build().unwrap(),
        builder(300).varying([5, 299]).build().unwrap()
    );
    assert!(builder(1).max_group_width(0).build().is_err());
    for width in [129, 1_000_000, usize::MAX] {
        assert!(builder(1).max_group_width(width).build().is_ok());
    }
    let reject = |b: treewalker_gbdt::WalkerConfigBuilder, expected: &str| match b.build() {
        Err(LoadError::MalformedConfig(msg)) => assert!(msg.contains(expected), "{msg}"),
        other => panic!("{expected}: {other:?}"),
    };
    // Feature roles must be declared: an undeclared feature would be constant.
    reject(WalkerConfig::builder(2).max_group_width(1), "varying(..)");
    reject(
        WalkerConfig::builder(2).max_group_width(1).increasing([0]),
        "varying(..)",
    );
    reject(WalkerConfig::builder(2).varying([0]), "max_group_width");
    reject(builder(2).varying([0, 0]), "duplicate index 0");
    reject(builder(2).varying([2]), "index 2 >= n_features 2");
    reject(builder(2).increasing([0, 0]), "duplicate index 0");
    reject(builder(2).decreasing([5]), "index 5 >= n_features 2");
    reject(
        builder(2).increasing([0]).decreasing([1, 0]),
        "feature 0 is both increasing and decreasing",
    );
    assert!(
        WalkerConfig::from_json(
            br#"{"n_features":2,"max_group_width":4,"varying_features":[0],"mono_inc_features":[1],"mono_dec_features":[1]}"#
        )
        .is_err()
    );
    // Later calls to a setter replace earlier ones.
    let replaced = builder(2).varying([0]).varying([1]).build().unwrap();
    assert!(!replaced.is_varying(0) && replaced.is_varying(1));
    let mut value = model_json();
    value["num_feature"] = 64.into();
    value["trees"][0]["nodes"][0]["split_feature_id"] = 63.into();
    let cfg = WalkerConfig::builder(64)
        .max_group_width(128)
        .varying([63])
        .build()
        .unwrap();
    let forest = Forest::from_bytes(
        &simd_json::to_vec(&value).unwrap(),
        ModelFormat::TreeliteJson,
        cfg,
        &LoadOptions::default(),
    )
    .unwrap();
    let mut result = vec![0.0; 128];
    forest
        .predictor()
        .predict_group(&vec![0.0; 128 * 64], &mut result);
}

#[test]
fn rejects_unsupported_or_malformed_json_metadata() {
    for (key, value) in [
        ("num_target", Value::from(2)),
        ("num_feature", 65.into()),
        ("task_type", "kMultiClf".into()),
        ("postprocessor", "not_sigmoid".into()),
        ("sigmoid_alpha", 0.into()),
        ("sigmoid_alpha", (-1).into()),
        ("leaf_output_type", "float32".into()),
        ("threshold_type", "uint32".into()),
        ("average_tree_output", 1.into()),
        ("num_opt_field_per_model", 1.into()),
    ] {
        let mut v = model_json();
        v.as_object_mut().unwrap().insert(key.into(), value);
        assert!(parse_json(&v).is_err(), "accepted {key}");
    }
    for key in [
        "num_feature",
        "num_target",
        "postprocessor",
        "base_scores",
        "sigmoid_alpha",
        "ratio_c",
        "attributes",
        "average_tree_output",
        "leaf_output_type",
        "threshold_type",
        "target_id",
        "class_id",
    ] {
        let mut v = model_json();
        v.as_object_mut().unwrap().remove(key);
        assert!(parse_json(&v).is_err(), "accepted missing {key}");
    }
    for key in [
        "base_scores",
        "num_class",
        "leaf_vector_shape",
        "target_id",
        "class_id",
        "trees",
    ] {
        let mut v = model_json();
        v[key] = Value::Array(Box::default());
        assert!(parse_json(&v).is_err(), "accepted empty {key}");
    }
    for key in ["target_id", "class_id"] {
        let mut v = model_json();
        v[key][1] = 99.into(); // validate beyond first entry
        assert!(parse_json(&v).is_err());
        v[key][1] = (-2).into();
        assert!(parse_json(&v).is_err());
        v[key][1] = (-1).into();
        assert!(parse_json(&v).is_err()); // Wildcards require vector leaves in GTIL.
    }
}

#[test]
fn rejects_bad_json_nodes_and_topology() {
    for (key, value) in [
        ("threshold", Value::from(1)),
        ("comparison_op", "<=".into()),
    ] {
        let mut v = simd_json::to_owned_value(&mut bytes("categories_left.json")).unwrap();
        v["trees"][0]["nodes"][0]
            .as_object_mut()
            .unwrap()
            .insert(key.into(), value);
        assert!(parse_json(&v).is_err(), "accepted categorical {key}");
    }
    for (key, val) in [
        ("node_type", Value::from("unknown")),
        ("split_feature_id", 65536.into()),
        ("comparison_op", ">".into()),
        ("threshold", Value::from(())),
        ("left_child", 0.into()),
        ("right_child", 1.into()),
        ("left_child", 999.into()),
        ("default_left", Value::from(())),
        ("category_list", Value::Array(Box::new(vec![1.into()]))),
        ("category_list_right_child", false.into()),
    ] {
        let mut v = model_json();
        v["trees"][0]["nodes"][0]
            .as_object_mut()
            .unwrap()
            .insert(key.into(), val);
        assert!(parse_json(&v).is_err(), "accepted {key}");
    }
    let mut v = model_json();
    v["trees"][0]["nodes"][1]["node_id"] = 0.into();
    assert!(parse_json(&v).is_err());
    let mut v = model_json();
    v["trees"][0]
        .as_object_mut()
        .unwrap()
        .insert("root_id".into(), 999.into());
    assert!(parse_json(&v).is_err());
    let mut v = model_json();
    v["trees"][0]
        .as_object_mut()
        .unwrap()
        .insert("root_id".into(), 1.into());
    assert!(parse_json(&v).is_err());
    let mut v = model_json();
    v["trees"][0]["nodes"][1]["leaf_value"] = Value::Array(Box::new(vec![1.into()]));
    assert!(matches!(parse_json(&v), Err(LoadError::Unsupported(_))));
    let mut v = model_json();
    v["trees"][0]["nodes"][1]
        .as_object_mut()
        .unwrap()
        .insert("left_child".into(), 2.into());
    assert!(parse_json(&v).is_err());
    let mut v = model_json();
    v["trees"][0]["num_nodes"] = 99.into();
    assert!(parse_json(&v).is_err());
    let mut v = model_json();
    v["threshold_type"] = "float32".into();
    v["leaf_output_type"] = "float32".into();
    assert!(parse_json(&v).is_err()); // <= unsupported
    let bad = String::from_utf8(bytes("sigmoid_f64.json"))
        .unwrap()
        .replace("\"threshold\": 1.0", "\"threshold\": ");
    assert!(
        Forest::from_bytes(
            bad.as_bytes(),
            ModelFormat::TreeliteJson,
            config(128),
            &LoadOptions::default()
        )
        .is_err()
    );
    // Any u32 category ID loads and routes like the full walk.
    let mut v = simd_json::to_owned_value(&mut bytes("categories_left.json")).unwrap();
    v["trees"][0]["nodes"][0]["category_list"] =
        Value::Array(Box::new(vec![(u32::MAX - 1).into(), u32::MAX.into()]));
    let forest = parse_json(&v).unwrap();
    let rows: Vec<f64> = [
        4_294_967_295.0,
        4_294_967_295.9,
        4_294_967_294.0,
        4_294_967_296.0,
    ]
    .iter()
    // Feature 1 is constant, so the same in every row of the group.
    .flat_map(|&x| [x, 0.0])
    .collect();
    let (mut out, mut reference) = (vec![0.0; 4], vec![0.0; 4]);
    forest.predictor().predict_group(&rows, &mut out);
    forest.predict_full_walk(&rows, &mut reference);
    assert_eq!(out, reference);
}

// Offset map for mutating *real Treelite-generated* checkpoints. Not a serializer.
fn binary_fields(data: &[u8]) -> std::collections::BTreeMap<String, usize> {
    fn arr(
        data: &[u8],
        fields: &mut std::collections::BTreeMap<String, usize>,
        pos: &mut usize,
        key: &str,
        size: usize,
    ) {
        fields.insert(format!("{key}.len"), *pos);
        let n = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap()) as usize;
        *pos += 8;
        fields.insert(key.to_owned(), *pos);
        *pos += n * size;
    }
    let mut fields = std::collections::BTreeMap::new();
    let mut pos = 0;
    let mut scalar = |key: &str, size: usize| {
        fields.insert(key.to_owned(), pos);
        pos += size;
    };
    for (key, size) in [
        ("major_ver", 4),
        ("minor_ver", 4),
        ("patch_ver", 4),
        ("threshold_type", 1),
        ("leaf_output_type", 1),
        ("num_tree", 8),
        ("num_feature", 4),
        ("task_type", 1),
        ("average_tree_output", 1),
        ("num_target", 4),
    ] {
        scalar(key, size);
    }
    for (key, size) in [
        ("num_class", 4),
        ("leaf_vector_shape", 4),
        ("target_id", 4),
        ("class_id", 4),
        ("postprocessor", 1),
    ] {
        arr(data, &mut fields, &mut pos, key, size);
    }
    fields.insert("sigmoid_alpha".into(), pos);
    pos += 8; // also ratio_c
    arr(data, &mut fields, &mut pos, "base_scores", 8);
    arr(data, &mut fields, &mut pos, "attributes", 1);
    fields.insert("num_opt_field_per_model".into(), pos);
    pos += 4;
    fields.insert("num_nodes".into(), pos);
    pos += 4;
    fields.insert("has_categorical_split".into(), pos);
    pos += 1;
    for (key, size) in [
        ("node_type", 1),
        ("cleft", 4),
        ("cright", 4),
        ("split_index", 4),
        ("default_left", 1),
        ("leaf_value", 8),
        ("threshold", 8),
        ("cmp", 1),
        ("category_list_right_child", 1),
        ("leaf_vector", 8),
        ("leaf_vector_begin", 8),
        ("leaf_vector_end", 8),
        ("category_list", 4),
        ("category_list_begin", 8),
        ("category_list_end", 8),
        ("data_count", 8),
        ("data_count_present", 1),
        ("sum_hess", 8),
        ("sum_hess_present", 1),
        ("gain", 8),
        ("gain_present", 1),
    ] {
        arr(data, &mut fields, &mut pos, key, size);
    }
    fields.insert("num_opt_field_per_tree".into(), pos);
    pos += 4;
    fields.insert("num_opt_field_per_node".into(), pos);
    fields
}
fn binary_result(b: &[u8]) -> Result<Forest, LoadError> {
    Forest::from_bytes(
        b,
        ModelFormat::TreeliteBinaryV4,
        config(128),
        &LoadOptions::default(),
    )
}
#[test]
fn binary_truncation_lengths_extensions_metadata_and_nodes() {
    let original = bytes("sigmoid_f64.bin");
    for end in 0..original.len() {
        assert!(
            binary_result(&original[..end]).is_err(),
            "accepted truncation {end}"
        );
    }
    let fields = binary_fields(&original);
    let changes = [
        ("major_ver", 5i32.to_le_bytes().to_vec()),
        ("minor_ver", 8i32.to_le_bytes().to_vec()),
        ("leaf_output_type", vec![2]),
        ("num_tree", u64::MAX.to_le_bytes().to_vec()),
        ("task_type", vec![2]),
        ("average_tree_output", vec![2]),
        ("num_target", 2i32.to_le_bytes().to_vec()),
        ("target_id", 2i32.to_le_bytes().to_vec()),
        ("target_id", (-1i32).to_le_bytes().to_vec()),
        ("class_id", (-2i32).to_le_bytes().to_vec()),
        ("class_id", (-1i32).to_le_bytes().to_vec()),
        ("num_feature", 65i32.to_le_bytes().to_vec()),
        ("sigmoid_alpha", f32::NAN.to_le_bytes().to_vec()),
        ("base_scores", f64::INFINITY.to_le_bytes().to_vec()),
        ("num_opt_field_per_model", 1i32.to_le_bytes().to_vec()),
        ("num_opt_field_per_tree", 1i32.to_le_bytes().to_vec()),
        ("num_opt_field_per_node", 1i32.to_le_bytes().to_vec()),
        ("num_nodes", i32::MAX.to_le_bytes().to_vec()),
        ("node_type", vec![3]),
        ("cleft", 0i32.to_le_bytes().to_vec()),
        ("cright", 1i32.to_le_bytes().to_vec()),
        ("split_index", 65536i32.to_le_bytes().to_vec()),
        ("default_left", vec![2]),
        ("cmp", vec![1]),
        ("threshold", f64::NAN.to_le_bytes().to_vec()),
        ("category_list_begin", 1u64.to_le_bytes().to_vec()),
        ("leaf_vector.len", 1u64.to_le_bytes().to_vec()),
    ];
    for (name, change) in changes {
        let mut b = original.clone();
        let start = fields[name];
        b[start..start + change.len()].copy_from_slice(&change);
        assert!(binary_result(&b).is_err(), "accepted {name}");
    }
    for (key, &start) in &fields {
        if key
            .rsplit_once('.')
            .is_some_and(|(_, suffix)| suffix == "len")
        {
            let mut b = original.clone();
            b[start..start + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(binary_result(&b).is_err(), "accepted length {key}");
        }
    }
    let mut b = original.clone();
    let start = fields["leaf_value"] + 8;
    b[start..start + 8].copy_from_slice(&f64::INFINITY.to_le_bytes());
    assert!(binary_result(&b).is_err());
    let mut b = original.clone();
    let start = fields["threshold"];
    b[start..start + 8].copy_from_slice(&f64::NEG_INFINITY.to_le_bytes());
    b[fields["cmp"]] = 2;
    assert!(matches!(binary_result(&b), Err(LoadError::Unsupported(_))));
    let mut b = original.clone();
    b.push(0);
    assert!(binary_result(&b).is_err());
    // Deterministic mutation smoke test: accepted mutations must still predict safely.
    let mut seed = 123u64;
    for _ in 0..2000 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut b = original.clone();
        b[(seed as usize) % original.len()] ^= (seed >> 32) as u8;
        if let Ok(f) = binary_result(&b) {
            f.predictor().predict_group(&[0.0; 4], &mut [0.0; 2]);
        }
    }
}

#[test]
fn real_unsupported_exports_return_capability_errors() {
    for name in [
        "reject_multiclass",
        "reject_multitarget",
        "reject_vector",
        "reject_postprocessor",
    ] {
        for (ext, format) in [
            ("bin", ModelFormat::TreeliteBinaryV4),
            ("json", ModelFormat::TreeliteJson),
        ] {
            assert!(
                matches!(
                    load(&format!("{name}.{ext}"), format, 128),
                    Err(LoadError::Unsupported(_))
                ),
                "{name}.{ext}"
            );
        }
    }
}

#[test]
fn deep_trees_load_and_json_nesting_is_bounded() {
    // A chain of 300 splits, past the old 256-edge depth limit.
    let mut model = model_json();
    let split = model["trees"][0]["nodes"][0].clone();
    let leaf = model["trees"][0]["nodes"][1].clone();
    let depth = 300usize;
    let mut nodes = Vec::new();
    for i in 0..depth {
        let mut n = split.clone();
        n["node_id"] = (i as i64).into();
        n["threshold"] = (i as f64).into();
        n["left_child"] = (if i + 1 == depth { 2 * depth } else { i + 1 } as i64).into();
        n["right_child"] = ((depth + i) as i64).into();
        nodes.push(n);
    }
    for i in depth..=2 * depth {
        let mut n = leaf.clone();
        n["node_id"] = (i as i64).into();
        n["leaf_value"] = (i as f64).into();
        nodes.push(n);
    }
    model["trees"][0]["num_nodes"] = (nodes.len() as i64).into();
    model["trees"][0]["nodes"] = Value::Array(Box::new(nodes));
    let forest = parse_json(&model).unwrap();
    let feature = split["split_feature_id"].as_u64().unwrap() as usize;
    let rows: Vec<f64> = (0..128)
        .flat_map(|r: i32| {
            let mut row = [0.5; 2];
            row[feature] = f64::from(r).mul_add(2.5, -10.0);
            row
        })
        .collect();
    let (mut out, mut reference) = (vec![0.0; 128], vec![0.0; 128]);
    forest.predictor().predict_group(&rows, &mut out);
    forest.predict_full_walk(&rows, &mut reference);
    assert_eq!(out, reference);

    let nested = format!("{}0{}", "[".repeat(1000), "]".repeat(1000));
    assert!(matches!(
        Forest::from_bytes(
            nested.as_bytes(),
            ModelFormat::TreeliteJson,
            config(128),
            &LoadOptions::default()
        ),
        Err(LoadError::Limit(_))
    ));
    assert!(WalkerConfig::from_json(nested.as_bytes()).is_err());
}

#[test]
fn input_caps_are_options() {
    let json = bytes("categories_left.json");
    let bin = bytes("categories_left.bin");
    let with = |options: LoadOptions, model: &[u8], format| {
        Forest::from_bytes(model, format, config(128), &options)
    };
    let limit = |options: LoadOptions, model: &[u8], format| {
        matches!(with(options, model, format), Err(LoadError::Limit(_)))
    };
    let defaults = LoadOptions::default();
    let (json_format, binary_format) = (ModelFormat::TreeliteJson, ModelFormat::TreeliteBinaryV4);
    assert!(
        with(defaults, &json, json_format).is_ok() && with(defaults, &bin, binary_format).is_ok()
    );
    let n = json.len() as u64;
    assert!(limit(
        LoadOptions {
            max_json_bytes: n - 1,
            ..defaults
        },
        &json,
        json_format
    ));
    assert!(
        with(
            LoadOptions {
                max_json_bytes: n,
                ..defaults
            },
            &json,
            json_format
        )
        .is_ok()
    );
    assert!(limit(
        LoadOptions {
            max_json_nesting: 2,
            ..defaults
        },
        &json,
        json_format
    ));
    let n = bin.len() as u64;
    assert!(limit(
        LoadOptions {
            max_binary_bytes: n - 1,
            ..defaults
        },
        &bin,
        binary_format
    ));
    assert!(
        with(
            LoadOptions {
                max_binary_bytes: n,
                ..defaults
            },
            &bin,
            binary_format
        )
        .is_ok()
    );
    assert!(limit(
        LoadOptions {
            max_binary_array_bytes: 8,
            ..defaults
        },
        &bin,
        binary_format
    ));
    // A split on 41 categories with missing sent left takes, in its one chain, a
    // default rank, the bounds of its 43 codes' exceptions (the categories, no
    // category and missing), 44 of 4 bytes, and 42 exceptions of 8 bytes: 516 bytes.
    let mut v = simd_json::to_owned_value(&mut json.clone()).unwrap();
    v["trees"][0]["nodes"][0]["category_list"] =
        Value::Array(Box::new((0..=40).map(Value::from).collect()));
    let pooled = simd_json::to_vec(&v).unwrap();
    assert!(with(defaults, &pooled, json_format).is_ok());
    let capped = LoadOptions {
        max_rank_table_bytes: 4 + 4 * 44 + 8 * 42 - 1,
        ..defaults
    };
    assert!(limit(capped, &pooled, json_format));
    assert!(with(capped, &json, json_format).is_ok());
    // A configuration past 64 KiB loads once its cap is raised.
    let list: Vec<String> = (0..20_000).map(|f| f.to_string()).collect();
    let config = format!(
        r#"{{"n_features":20001,"max_group_width":4,"varying_features":[{}],"mono_inc_features":[],"mono_dec_features":[]}}"#,
        list.join(",")
    );
    assert!(config.len() > 64 << 10);
    assert!(matches!(
        WalkerConfig::from_json(config.as_bytes()),
        Err(LoadError::Limit(_))
    ));
    let raised = LoadOptions {
        max_config_bytes: 1 << 20,
        ..defaults
    };
    let c = WalkerConfig::from_json_with(config.as_bytes(), &raised).unwrap();
    assert!(c.is_varying(19_999) && !c.is_varying(20_000));
}
