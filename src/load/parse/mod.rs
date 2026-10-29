//! Decoding a Treelite model: the binary checkpoint ([`binary`]) or the JSON dump
//! ([`json`]) into the decoded forest ([`decoded`]), each tree validated and put in
//! heavy-first preorder as it is read ([`tree`]), with the model-level checks and
//! the input caps ([`validation`]).
//!
//! # Threshold semantics
//!
//! Treelite normalizes each framework's model (`treelite.frontend.load_lightgbm_model`
//! or `load_xgboost_model`, then `serialize_bytes()` or `dump_as_json()`) but keeps
//! its comparison: LightGBM splits on `val <= threshold` in f64, XGBoost on
//! `val < threshold` in f32. The decoder reads the threshold type from the header:
//!
//! - F64: a `<=` threshold is kept, and a `<` threshold becomes `next_down(t)`, so
//!   that every F64 split compares `<=`.
//! - F32: thresholds are kept, and splits compare `(val as f32) < (t as f32)`.

pub(crate) mod decoded;
pub(crate) mod tree;
