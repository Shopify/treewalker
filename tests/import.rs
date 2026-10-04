//! Public import contract, independent Treelite/GTIL fixtures, and malformed input guards.
#![expect(clippy::float_cmp, reason = "exact hand-computable expectations")]
use serde::Deserialize;
use simd_json::{OwnedValue as Value, prelude::*};
use std::{io::Cursor, path::PathBuf};
use treewalker_gbdt::research::Ablation;
use treewalker_gbdt::{Forest, LoadError, ModelFormat, ParseConfig, WalkerConfig};

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
fn config(width: usize) -> WalkerConfig {
    WalkerConfig::try_new(2, width, &[0], &[], &[]).unwrap()
}
fn load(name: &str, format: ModelFormat, width: usize) -> Result<Forest, LoadError> {
    Forest::from_bytes(&bytes(name), format, config(width), &ParseConfig::default())
}
fn model_json() -> Value {
    simd_json::to_owned_value(&mut bytes("sigmoid_f64.json")).unwrap()
}
fn parse_json(v: &Value) -> Result<Forest, LoadError> {
    Forest::from_bytes(
        &simd_json::to_vec(v).unwrap(),
        ModelFormat::TreeliteJson,
        config(128),
        &ParseConfig::default(),
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
    let data = raw("data.bin");
    let rows = data.len() / 2;
    for case in manifest.models {
        let expected = raw(&format!("{}_reference.bin", case.name));
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
                        disable_exact_sums: true,
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
fn groups_of_any_width_match_single_rows() {
    // A row's prediction depends only on its own features, so a row predicted in a
    // group of any width, whole or in pieces, equals the row predicted alone.
    let manifest: Manifest = simd_json::serde::from_slice(&mut bytes("manifest.json")).unwrap();
    let data = raw("data.bin");
    let rows = data.len() / 2;
    for case in manifest.models {
        let name = format!("{}.bin", case.name);
        let mut single = load(&name, ModelFormat::TreeliteBinaryV4, 1)
            .unwrap()
            .predictor();
        for width in [129, 300, 1024, 1025, 2500] {
            let forest = load(&name, ModelFormat::TreeliteBinaryV4, width).unwrap();
            for block in [0, rows / 128 - 1] {
                // Varying feature cycles through every test value; the constant
                // feature is the block's.
                let constant = data[block * 128 * 2 + 1];
                let group: Vec<f64> = (0..width)
                    .flat_map(|r| [data[(r % rows) * 2], constant])
                    .collect();
                let mut alone = vec![0.0; width];
                single.predict_fixed(&group, 1, &mut alone);
                for ablation in [
                    None,
                    Some(Ablation::default()),
                    Some(Ablation {
                        disable_varying_precompute: true,
                        ..Default::default()
                    }),
                    Some(Ablation {
                        disable_predicate_sweep: true,
                        ..Default::default()
                    }),
                ] {
                    let mut wide = vec![f64::NAN; width];
                    match ablation {
                        None => forest.predictor().predict_group(&group, &mut wide),
                        Some(a) => forest
                            .research_predictor(a)
                            .predict_group(&group, &mut wide),
                    }
                    for r in 0..width {
                        assert_eq!(
                            wide[r].to_bits(),
                            alone[r].to_bits(),
                            "{name} width {width} block {block} row {r}: {} != {}",
                            wide[r],
                            alone[r]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn exact_sums_make_layout_invisible() {
    // Tree order, prefix sharing and bitset interning change the order in which
    // leaves are added; exact sums make the predictions bit-identical anyway.
    let manifest: Manifest = simd_json::serde::from_slice(&mut bytes("manifest.json")).unwrap();
    let data = raw("data.bin");
    let rows = data.len() / 2;
    for case in manifest.models {
        let name = format!("{}.bin", case.name);
        let predict_with = |parse: &ParseConfig| {
            let forest = Forest::from_bytes(
                &bytes(&name),
                ModelFormat::TreeliteBinaryV4,
                config(128),
                parse,
            )
            .unwrap();
            assert!(forest.exact_sums(), "{name}");
            let mut out = vec![0.0; rows];
            forest.predictor().predict_fixed(&data, 128, &mut out);
            out
        };
        let reference = predict_with(&ParseConfig::default());
        for parse in [
            ParseConfig {
                disable_tree_ordering: true,
                ..Default::default()
            },
            ParseConfig {
                prefix_depth: 0,
                ..Default::default()
            },
            ParseConfig {
                disable_tree_ordering: true,
                disable_bitset_intern: true,
                prefix_depth: 0,
                ..Default::default()
            },
        ] {
            let out = predict_with(&parse);
            for r in 0..rows {
                assert_eq!(out[r].to_bits(), reference[r].to_bits(), "{name} row {r}");
            }
        }
    }
}

#[test]
fn leaves_without_a_common_scale_add_in_tree_order() {
    // Leaf exponents 2^-997 apart leave no 126-bit fixed point: prediction adds in
    // f64 in tree order, exactly as the full walk does.
    let mut model = simd_json::to_owned_value(&mut bytes("identity.json")).unwrap();
    model["trees"][0]["nodes"][1]["leaf_value"] = 1e300.into();
    model["trees"][1]["nodes"][1]["leaf_value"] = 1e-300.into();
    let json = simd_json::to_vec(&model).unwrap();
    let forest = Forest::from_bytes(
        &json,
        ModelFormat::TreeliteJson,
        config(128),
        &ParseConfig::default(),
    )
    .unwrap();
    assert!(!forest.exact_sums());
    let data = raw("data.bin");
    let rows = data.len() / 2;
    let (mut partial, mut full) = (vec![0.0; rows], vec![0.0; rows]);
    forest.predictor().predict_fixed(&data, 128, &mut partial);
    forest.predict_full_walk(&data, &mut full);
    assert!(partial.iter().any(|&v| v == 1e300));
    for r in 0..rows {
        assert_eq!(partial[r].to_bits(), full[r].to_bits(), "row {r}");
    }
}

#[test]
fn native_random_forest_and_hand_computable_aggregation() {
    let forest = load("random_forest.bin", ModelFormat::TreeliteBinaryV4, 128).unwrap();
    let data = raw("rf_native_data.bin");
    let expected = raw("rf_native_reference.bin");
    let mut actual = vec![0.0; expected.len()];
    forest.predict_full_walk(&data, &mut actual);
    check(&actual, &expected, 1e-14, 1e-14, "sklearn");
    let forest = load("average_base.bin", ModelFormat::TreeliteBinaryV4, 128).unwrap();
    let mut result = [0.0];
    forest.predictor().predict_group(&[0.0, 0.0], &mut result);
    assert_eq!(result, [3.375]);
}

#[test]
fn file_reader_memory_and_legacy_paths() {
    let cfg = fixture("walker_config.json");
    let a = Forest::try_load(fixture("sigmoid_f64.bin"), &cfg).unwrap();
    let b = Forest::from_reader(
        Cursor::new(bytes("sigmoid_f64.bin")),
        ModelFormat::TreeliteBinaryV4,
        config(128),
        &ParseConfig::default(),
    )
    .unwrap();
    let c = Forest::load(fixture("sigmoid_f64.json"), &cfg);
    let d = Forest::load_with_config(fixture("sigmoid_f64.bin"), &cfg, &ParseConfig::default());
    let input = raw("data.bin");
    let expected = raw("sigmoid_f64_reference.bin");
    for f in [a, b, c, d] {
        let mut output = vec![0.0; expected.len()];
        f.predict_full_walk(&input, &mut output);
        check(&output, &expected, 1e-14, 0.0, "loader");
    }
    assert!(matches!(
        Forest::try_load("missing.bin", &cfg),
        Err(LoadError::Io(_))
    ));
    assert!(matches!(
        Forest::try_load("unknown.format", &cfg),
        Err(LoadError::Unsupported(_))
    ));
    let mut config_file = bytes("walker_config.json");
    let _: WalkerConfig = WalkerConfig::try_from_json(&config_file).unwrap();
    config_file.truncate(10);
    assert!(WalkerConfig::try_from_json(&config_file).is_err());
    assert_eq!(WalkerConfig::from_file(cfg).n_features, 2);
}

#[test]
fn grouping_schema_and_configuration_validation() {
    let schema = br#"{"n_features":4,"max_group_width":14,"varying_features":[1,2,3],"mono_inc_features":[3],"mono_dec_features":[1]}"#;
    let c = WalkerConfig::try_from_json(schema).unwrap();
    assert_eq!(
        (c.varying_mask, c.mono_inc_mask, c.mono_dec_mask),
        (14, 8, 2)
    );
    for nf in [0, 65, 128, usize::MAX] {
        assert!(WalkerConfig::try_new(nf, 1, &[], &[], &[]).is_err());
    }
    assert!(WalkerConfig::try_new(1, 0, &[], &[], &[]).is_err());
    for width in [129, 1_000_000, usize::MAX] {
        assert!(WalkerConfig::try_new(1, width, &[], &[], &[]).is_ok());
    }
    for (v, i, d) in [
        (&[0, 0][..], &[][..], &[][..]),
        (&[2][..], &[][..], &[][..]),
        (&[0][..], &[0][..], &[0][..]),
        (&[][..], &[0][..], &[][..]),
        (&[0][..], &[0, 0][..], &[][..]),
    ] {
        assert!(WalkerConfig::try_new(2, 1, v, i, d).is_err());
    }
    for mutate in 0..4 {
        let mut cfg = config(128);
        match mutate {
            0 => cfg.n_features = 65,
            1 => cfg.max_group_width = 0,
            2 => cfg.varying_mask = 1 << 64,
            _ => cfg.mono_inc_mask = 2,
        }
        assert!(matches!(
            Forest::from_bytes(
                &bytes("sigmoid_f64.bin"),
                ModelFormat::TreeliteBinaryV4,
                cfg,
                &ParseConfig::default()
            ),
            Err(LoadError::MalformedConfig(_))
        ));
    }
    let mut value = model_json();
    value["num_feature"] = 64.into();
    value["trees"][0]["nodes"][0]["split_feature_id"] = 63.into();
    let cfg = WalkerConfig::try_new(64, 128, &[63], &[], &[]).unwrap();
    let forest = Forest::from_bytes(
        &simd_json::to_vec(&value).unwrap(),
        ModelFormat::TreeliteJson,
        cfg,
        &ParseConfig::default(),
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
            &ParseConfig::default()
        )
        .is_err()
    );
    let mut v = simd_json::to_owned_value(&mut bytes("categories_left.json")).unwrap();
    v["trees"][0]["nodes"][0]["category_list"] = Value::Array(Box::new(vec![u32::MAX.into()]));
    assert!(matches!(parse_json(&v), Err(LoadError::Limit(_))));
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
        &ParseConfig::default(),
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
fn deep_trees_and_deep_json_are_bounded() {
    let mut model = model_json();
    let split = model["trees"][0]["nodes"][0].clone();
    let leaf = model["trees"][0]["nodes"][1].clone();
    let depth = 258usize;
    let mut nodes = Vec::new();
    for i in 0..depth {
        let mut n = split.clone();
        n["node_id"] = (i as i64).into();
        n["left_child"] = (if i + 1 == depth { 2 * depth } else { i + 1 } as i64).into();
        n["right_child"] = ((depth + i) as i64).into();
        nodes.push(n);
    }
    for i in depth..=2 * depth {
        let mut n = leaf.clone();
        n["node_id"] = (i as i64).into();
        nodes.push(n);
    }
    model["trees"][0]["num_nodes"] = (nodes.len() as i64).into();
    model["trees"][0]["nodes"] = Value::Array(Box::new(nodes));
    assert!(matches!(parse_json(&model), Err(LoadError::Limit(_))));
    let nested = format!("{}0{}", "[".repeat(1000), "]".repeat(1000));
    assert!(matches!(
        Forest::from_bytes(
            nested.as_bytes(),
            ModelFormat::TreeliteJson,
            config(128),
            &ParseConfig::default()
        ),
        Err(LoadError::Limit(_))
    ));
    assert!(WalkerConfig::try_from_json(nested.as_bytes()).is_err());
}
