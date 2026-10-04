//! Loading artifact cells whose groups are wider than the library's 128-row contract.

use std::path::Path;

use treewalker_gbdt::{Forest, LoadError, ModelFormat, ParseConfig, WalkerConfig};

#[derive(serde::Deserialize)]
struct CellConfig {
    n_features: usize,
    max_group_width: usize,
    varying_features: Vec<usize>,
    mono_inc_features: Vec<usize>,
    mono_dec_features: Vec<usize>,
}

/// Load an artifact cell's forest with `max_group_width` clamped to 128.
///
/// 128 is the library's limit for `predict`. Returns the forest and the cell's actual
/// group width; callers chunk wider groups or use a kernel without the 128-row limit
/// (the experimental run-list kernel). `max_group_width` does not affect parsing or
/// layout.
pub fn load_forest_any_width(
    model: &Path,
    config: &Path,
    parse: &ParseConfig,
) -> Result<(Forest, usize), LoadError> {
    let mut bytes = std::fs::read(config)?;
    let c: CellConfig =
        simd_json::from_slice(&mut bytes).map_err(|e| LoadError::MalformedConfig(e.to_string()))?;
    let cfg = WalkerConfig::try_new(
        c.n_features,
        c.max_group_width.min(128),
        &c.varying_features,
        &c.mono_inc_features,
        &c.mono_dec_features,
    )?;
    let format = match model.extension().and_then(|e| e.to_str()) {
        Some("bin") => ModelFormat::TreeliteBinaryV4,
        Some("json") => ModelFormat::TreeliteJson,
        _ => {
            return Err(LoadError::Unsupported(format!(
                "model path {} must end in .bin or .json",
                model.display()
            )));
        }
    };
    let forest = Forest::from_reader(std::fs::File::open(model)?, format, cfg, parse)?;
    Ok((forest, c.max_group_width))
}
