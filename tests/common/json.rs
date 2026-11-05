//! Hand-written Treelite JSON models.

/// The Treelite JSON of a regressor over `n_features` features, without a
/// postprocessor or base score, from its trees' JSON.
pub fn regressor(f32: bool, n_features: usize, trees: &[String]) -> String {
    let ty = if f32 { "float32" } else { "float64" };
    let ids = format!("{:?}", vec![0; trees.len()]);
    format!(
        r#"{{"threshold_type": "{ty}", "leaf_output_type": "{ty}", "num_feature": {n_features}, "task_type": "kRegressor", "average_tree_output": false, "num_target": 1, "num_class": [1], "leaf_vector_shape": [1, 1], "target_id": {ids}, "class_id": {ids}, "postprocessor": "identity", "sigmoid_alpha": 1.0, "ratio_c": 1.0, "base_scores": [0.0], "attributes": "", "trees": [{}]}}"#,
        trees.join(", ")
    )
}

/// A tree of one split on feature 0, whose fields besides the node's own are `test`,
/// and two leaves, 1.0 on the left and 2.0 on the right, with data counts `counts`.
pub fn stump(test: &str, default_left: bool, counts: (u32, u32)) -> String {
    let (left, right) = counts;
    format!(
        r#"{{"num_nodes": 3, "has_categorical_split": {}, "nodes": [{{"node_id": 0, "split_feature_id": 0, "default_left": {default_left}, {test}, "left_child": 1, "right_child": 2}}, {{"node_id": 1, "leaf_value": 1.0, "data_count": {left}}}, {{"node_id": 2, "leaf_value": 2.0, "data_count": {right}}}]}}"#,
        test.contains("categorical")
    )
}
