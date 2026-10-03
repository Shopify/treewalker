//! Shared semantic validation and bounded import policy.
use crate::forest::{Node, ThresholdType, Tree};
use crate::{LoadError, WalkerConfig};

pub(super) const MAX_TREES: usize = 1_000_000;
pub(super) const MAX_NODES: usize = 32_000_000;
pub(super) const MAX_POOL_BYTES: usize = 256 * 1024 * 1024;
pub(super) const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;
pub(super) const MAX_DEPTH: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Postprocessor {
    Identity,
    Sigmoid(f64),
}

#[derive(Clone, Copy, Debug)]
pub struct Output {
    pub base_score: f64,
    pub divisor: f64,
    pub postprocessor: Postprocessor,
}

pub struct ParsedModel {
    pub trees: Vec<Tree>,
    pub nodes: Vec<Node>,
    pub bitsets: Vec<u8>,
    pub threshold_type: ThresholdType,
    pub output: Output,
}

pub(super) struct Metadata {
    pub num_tree: usize,
    pub num_feature: i64,
    pub task_type: String,
    pub average: bool,
    pub num_target: i64,
    pub num_class: Vec<i32>,
    pub leaf_shape: Vec<i32>,
    pub target_id: Vec<i32>,
    pub class_id: Vec<i32>,
    pub postprocessor: String,
    pub sigmoid_alpha: f64,
    pub base_scores: Vec<f64>,
}

impl Metadata {
    pub fn validate(&self, config: &WalkerConfig) -> Result<Output, LoadError> {
        if self.num_tree == 0 {
            return Err(LoadError::MalformedModel("empty forest".into()));
        }
        if self.num_tree > MAX_TREES {
            return Err(LoadError::Limit(format!(
                "num_tree {} > {MAX_TREES}",
                self.num_tree
            )));
        }
        if self.num_feature != config.n_features as i64 {
            return Err(LoadError::MalformedModel(format!(
                "num_feature {} differs from configured n_features {}",
                self.num_feature, config.n_features
            )));
        }
        if !matches!(
            self.task_type.as_str(),
            "kBinaryClf" | "kRegressor" | "kLearningToRank"
        ) {
            return Err(LoadError::Unsupported(format!(
                "task_type {:?}; require scalar binary classification, regression or ranking",
                self.task_type
            )));
        }
        if self.num_target != 1 || self.num_class != [1] || self.leaf_shape != [1, 1] {
            return Err(LoadError::Unsupported(format!(
                "scalar output required: num_target={}, num_class={:?}, leaf_vector_shape={:?}",
                self.num_target, self.num_class, self.leaf_shape
            )));
        }
        for (name, ids) in [("target_id", &self.target_id), ("class_id", &self.class_id)] {
            if ids.len() != self.num_tree {
                return Err(LoadError::MalformedModel(format!(
                    "{name}: length {}, expected {}",
                    ids.len(),
                    self.num_tree
                )));
            }
            for (tree, &id) in ids.iter().enumerate() {
                // Treelite's scalar-leaf evaluator requires explicit IDs.
                // Wildcard -1 assignments belong to vector leaves, even when
                // their output shape happens to be [1, 1].
                if id != 0 {
                    return Err(LoadError::Unsupported(format!(
                        "tree {tree}: {name}={id}; scalar leaves require 0 (wildcards require vector leaves)"
                    )));
                }
            }
        }
        if self.base_scores.len() != 1 || !self.base_scores[0].is_finite() {
            return Err(LoadError::Unsupported(format!(
                "base_scores must contain one finite value; got {:?}",
                self.base_scores
            )));
        }
        let postprocessor = match self.postprocessor.as_str() {
            "identity" => Postprocessor::Identity,
            "sigmoid" if self.sigmoid_alpha.is_finite() && self.sigmoid_alpha > 0.0 => {
                Postprocessor::Sigmoid(self.sigmoid_alpha)
            }
            "sigmoid" => {
                return Err(LoadError::Unsupported(format!(
                    "sigmoid_alpha must be finite and positive; got {}",
                    self.sigmoid_alpha
                )));
            }
            _ => {
                return Err(LoadError::Unsupported(format!(
                    "postprocessor {:?}; supported names are identity and sigmoid",
                    self.postprocessor
                )));
            }
        };
        Ok(Output {
            base_score: self.base_scores[0],
            divisor: if self.average {
                self.num_tree as f64
            } else {
                1.0
            },
            postprocessor,
        })
    }
}

pub(super) fn threshold_type(threshold: &str, leaf: &str) -> Result<ThresholdType, LoadError> {
    match (threshold, leaf) {
        ("float32", "float32") => Ok(ThresholdType::F32),
        ("float64", "float64") => Ok(ThresholdType::F64),
        _ => Err(LoadError::Unsupported(format!(
            "threshold_type={threshold:?}, leaf_output_type={leaf:?}; require matching float32 or float64"
        ))),
    }
}

pub(super) fn threshold(value: f64, op: &str, kind: ThresholdType) -> Result<f64, LoadError> {
    if value.is_nan() || (kind == ThresholdType::F64 && op == "<" && value == f64::NEG_INFINITY) {
        return Err(LoadError::Unsupported(format!(
            "threshold {value} with {op} cannot be represented exactly"
        )));
    }
    match (op, kind) {
        ("<", ThresholdType::F64) => Ok(super::common::next_down(value)),
        ("<=", ThresholdType::F64) => Ok(value),
        ("<", ThresholdType::F32) => {
            let rounded = f64::from(value as f32);
            if value.is_finite() && !rounded.is_finite() {
                return Err(LoadError::Unsupported(format!(
                    "threshold {value} overflows float32"
                )));
            }
            Ok(rounded)
        }
        _ => Err(LoadError::Unsupported(format!(
            "comparison_op={op:?} for {kind:?}"
        ))),
    }
}

pub(super) fn leaf(value: f64, kind: ThresholdType) -> Result<f64, LoadError> {
    let value = if kind == ThresholdType::F32 {
        f64::from(value as f32)
    } else {
        value
    };
    if !value.is_finite() {
        return Err(LoadError::Unsupported(format!(
            "nonfinite leaf_value {value}"
        )));
    }
    Ok(value)
}

/// Bound parser nesting before constructing the JSON DOM (also used by config).
/// Syntax itself is validated by simd-json, without repairing malformed dumps.
pub fn validate_json_depth(bytes: &[u8]) -> Result<(), LoadError> {
    let (mut depth, mut string, mut escape) = (0usize, false, false);
    for &b in bytes {
        if string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                string = false;
            }
        } else if b == b'"' {
            string = true;
        } else if b == b'[' || b == b'{' {
            depth += 1;
            if depth > 64 {
                return Err(LoadError::Limit("JSON nesting exceeds 64".into()));
            }
        } else if b == b']' || b == b'}' {
            depth = depth.saturating_sub(1);
        }
    }
    Ok(())
}

/// Attributes cannot affect prediction, but v4 requires a JSON object or empty string.
pub(super) fn attributes(value: &str) -> Result<(), LoadError> {
    use simd_json::prelude::*;
    if value.is_empty() {
        return Ok(());
    }
    if value.len() > 16 * 1024 * 1024 {
        return Err(LoadError::Limit("attributes exceeds 16 MiB".into()));
    }
    validate_json_depth(value.as_bytes())?;
    let parsed = simd_json::to_owned_value(&mut value.as_bytes().to_vec())
        .map_err(|e| LoadError::MalformedModel(format!("attributes: {e}")))?;
    if parsed.as_object().is_none() {
        return Err(LoadError::MalformedModel(
            "attributes must be a JSON object".into(),
        ));
    }
    Ok(())
}
