//! Grid definitions and artifact directory discovery.
//!
//! The released grids' constants and the artifact directory scan. The grids
//! now live in `experiments/grids.toml`; the runner redesign reads them from
//! the execution manifest instead.

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Grid combo constants (the released grids)
// ---------------------------------------------------------------------------

/// Grid 1: full factorial `T × L × H`.
pub const GRID1_N_TREES: &[usize] = &[50, 500, 1000, 2000];
pub const GRID1_MAX_DEPTH: &[usize] = &[2, 4, 8, 16];
pub const GRID1_HORIZON: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];

/// Grid 3 ablation anchors (B set).
pub const ABLATION_ANCHORS_B: &[(usize, usize, usize)] = &[
    (50, 4, 1),
    (500, 8, 4),
    (500, 8, 16),
    (1000, 16, 32),
    (500, 4, 16),
    (500, 8, 64),
    (500, 8, 128),
];

/// Grid 3 ablation anchors (B' set — full 64-combo cross).
pub const ABLATION_ANCHORS_B_PRIME: &[(usize, usize, usize)] = &[(500, 8, 16), (1000, 16, 32)];

/// External baselines run on the B set anchors.
pub const EXTERNAL_BASELINE_ANCHORS: &[(usize, usize, usize)] = ABLATION_ANCHORS_B;

// ---------------------------------------------------------------------------
// Grid enumeration
// ---------------------------------------------------------------------------

/// Which benchmark grid to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grid {
    /// Grid 1: full factorial T × L × H comparison.
    G1,
    /// Grid 3: ablation interaction study.
    G3,
    /// Grid 4: group distribution experiments (CTR datasets).
    G4,
    /// E1 scenario-analysis benchmark (UCI Default of Credit).
    /// Run via `run_grid_scen`; does not use `discover_cells`.
    Scen,
    /// All grids.
    All,
}

/// A single benchmark cell: one (dataset, param_combo, framework) triple.
#[derive(Debug, Clone)]
pub struct GridCell {
    pub dataset: String,
    pub param_dir: PathBuf,
    pub framework: String,
    pub fw_dir: PathBuf,
    pub nt: usize,
    pub md: usize,
    pub horizon: usize,
    pub is_ctr: bool,
}

impl GridCell {
    /// Human-readable label for logging.
    pub fn label(&self) -> String {
        let dir_name = self
            .param_dir
            .file_name()
            .map_or("?", |n| n.to_str().unwrap_or("?"));
        format!("{}/{}/{}", self.dataset, dir_name, self.framework)
    }
}

/// Check if a `(nt, md, h)` cell should run external baselines.
pub fn is_external_anchor(nt: usize, md: usize, h: usize, is_ctr: bool) -> bool {
    if is_ctr {
        return EXTERNAL_BASELINE_ANCHORS
            .iter()
            .any(|&(at, am, _)| at == nt && am == md);
    }
    EXTERNAL_BASELINE_ANCHORS
        .iter()
        .any(|&(at, am, ah)| at == nt && am == md && ah == h)
}

/// Generate the full factorial combo list for Grid 1.
pub fn grid1_combos() -> Vec<(usize, usize, usize)> {
    let mut combos = Vec::new();
    for &nt in GRID1_N_TREES {
        for &md in GRID1_MAX_DEPTH {
            for &h in GRID1_HORIZON {
                combos.push((nt, md, h));
            }
        }
    }
    combos
}

/// Parse a param directory name like `nt500_md8_h16` into `(nt, md, h)`.
/// For CTR datasets, parses `nt500_md8` into `(nt, md, 0)`.
/// Parse a param directory name like `nt500_md8_h16` into `(nt, md, h)`.
///
/// For CTR datasets, parses `nt500_md8` into `(nt, md, 0)`.
pub fn parse_param_dir(name: &str) -> Option<(usize, usize, usize)> {
    // Try nt{T}_md{L}_h{H} first
    if let Some(rest) = name.strip_prefix("nt") {
        let parts: Vec<&str> = rest.split('_').collect();
        if parts.len() == 3
            && let (Some(nt_s), Some(md_rest), Some(h_rest)) =
                (parts.first(), parts.get(1), parts.get(2))
        {
            let nt = nt_s.parse::<usize>().ok()?;
            let md = md_rest.strip_prefix("md")?.parse::<usize>().ok()?;
            let h = h_rest.strip_prefix('h')?.parse::<usize>().ok()?;
            return Some((nt, md, h));
        }
        // Try nt{T}_md{L} (CTR, no horizon)
        if parts.len() == 2
            && let (Some(nt_s), Some(md_rest)) = (parts.first(), parts.get(1))
        {
            let nt = nt_s.parse::<usize>().ok()?;
            let md = md_rest.strip_prefix("md")?.parse::<usize>().ok()?;
            return Some((nt, md, 0));
        }
    }
    None
}

/// Parse an E1 scenario cell directory name `k{K}_G{G}` into `(k, G)`.
///
/// Returns `None` for names that do not match the scenario layout.
pub fn parse_scen_cell(name: &str) -> Option<(usize, usize)> {
    let (k_part, g_part) = name.split_once('_')?;
    let k = k_part.strip_prefix('k')?.parse::<usize>().ok()?;
    let g = g_part.strip_prefix('G')?.parse::<usize>().ok()?;
    Some((k, g))
}

/// Discover benchmark cells by scanning the artifacts directory.
///
/// Looks for `<artifacts_dir>/<dataset>/<param_dir>/<framework>/walker_config.json`.
pub fn discover_cells(
    artifacts_dir: &Path,
    grid: Grid,
    datasets_filter: Option<&[String]>,
) -> Vec<GridCell> {
    let combos = match grid {
        Grid::G1 | Grid::All => grid1_combos(),
        Grid::G3 => ABLATION_ANCHORS_B.to_vec(),
        Grid::G4 | Grid::Scen => {
            // G4: CTR datasets only (filtered by is_ctr).
            // Scen: does not use discover_cells (run_grid_scen builds cells).
            Vec::new()
        }
    };
    let combo_set: std::collections::HashSet<(usize, usize, usize)> =
        combos.iter().copied().collect();

    let mut cells = Vec::new();

    // Scan dataset directories.
    let Ok(ds_entries) = std::fs::read_dir(artifacts_dir) else {
        eprintln!("Cannot read artifacts dir: {}", artifacts_dir.display());
        return cells;
    };

    let mut dataset_dirs: Vec<_> = ds_entries
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_ok_and(|ft| ft.is_dir()))
        .collect();
    dataset_dirs.sort_by_key(std::fs::DirEntry::file_name);

    for ds_entry in dataset_dirs {
        let ds_name = ds_entry.file_name().to_string_lossy().to_string();

        // Apply dataset filter.
        if let Some(filter) = datasets_filter
            && !filter.iter().any(|f| f == &ds_name)
        {
            continue;
        }

        let is_ctr = ds_name == "expedia";
        let ds_dir = ds_entry.path();

        // Scan param directories.
        let Ok(param_entries) = std::fs::read_dir(&ds_dir) else {
            continue;
        };
        let mut param_dirs: Vec<_> = param_entries
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_ok_and(|ft| ft.is_dir()))
            .collect();
        param_dirs.sort_by_key(std::fs::DirEntry::file_name);

        for param_entry in param_dirs {
            let param_name = param_entry.file_name().to_string_lossy().to_string();
            let Some((nt, md, h)) = parse_param_dir(&param_name) else {
                continue;
            };

            // Check if this combo is in the requested grid.
            match grid {
                Grid::G4 => {
                    if !is_ctr {
                        continue;
                    }
                }
                _ => {
                    if !combo_set.is_empty() {
                        let key = if is_ctr {
                            // CTR: match on (nt, md) only
                            combo_set.iter().any(|&(ct, cm, _)| ct == nt && cm == md)
                        } else {
                            combo_set.contains(&(nt, md, h))
                        };
                        if !key {
                            continue;
                        }
                    }
                }
            }

            let param_dir = param_entry.path();

            for framework in &["lightgbm", "xgboost"] {
                let fw_dir = param_dir.join(framework);
                // Require walker_config.json to exist.
                if !param_dir.join("walker_config.json").exists() {
                    continue;
                }
                // Require at least one model file.
                if !fw_dir.join("model_treelite.bin").exists()
                    && !fw_dir.join("model_treelite.json").exists()
                {
                    continue;
                }

                cells.push(GridCell {
                    dataset: ds_name.clone(),
                    param_dir: param_dir.clone(),
                    framework: (*framework).to_string(),
                    fw_dir,
                    nt,
                    md,
                    horizon: h,
                    is_ctr,
                });
            }
        }
    }

    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_param_dir() {
        assert_eq!(parse_param_dir("nt500_md8_h16"), Some((500, 8, 16)));
        assert_eq!(parse_param_dir("nt50_md4_h1"), Some((50, 4, 1)));
        assert_eq!(parse_param_dir("nt1000_md8"), Some((1000, 8, 0)));
        assert_eq!(parse_param_dir("invalid"), None);
        assert_eq!(parse_param_dir("nt_md_h"), None);
    }

    #[test]
    fn test_grid1_combos_count() {
        let combos = grid1_combos();
        assert_eq!(combos.len(), 4 * 4 * 8); // 128 combos
    }

    #[test]
    fn test_is_external_anchor() {
        assert!(is_external_anchor(500, 8, 16, false));
        assert!(is_external_anchor(50, 4, 1, false));
        assert!(!is_external_anchor(100, 4, 1, false)); // not in anchors
        // CTR: match on (nt, md) only
        assert!(is_external_anchor(500, 8, 999, true));
    }
}
