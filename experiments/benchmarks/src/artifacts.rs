//! Prepared cells and the execution manifest that lists them.
//!
//! `treewalker-exp manifest` resolves a suite into an execution manifest: every cell
//! with its directory, its key and the variants to time. Each cell directory holds a
//! `cell.json` that names its files with their hashes. The runner and the tests load
//! cells only through this module, so there is one artifact layout and no directory
//! scanning.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat, WalkerConfig};

use crate::data;

/// The execution manifest's schema version this runner reads.
pub const MANIFEST_SCHEMA: u64 = 3;

/// An execution manifest: one suite resolved into cells.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema_version: u64,
    pub suite: String,
    #[serde(default)]
    pub description: String,
    /// Where cell directories are, absolute; `run --artifacts-dir` overrides it.
    pub artifacts_dir: PathBuf,
    #[serde(default)]
    pub grids_sha256: String,
    #[serde(default)]
    pub source: Value,
    /// Measurement settings for every cell of the run.
    #[serde(default)]
    pub run: RunConfig,
    /// The native LightGBM and XGBoost libraries from `treewalker-exp build-native`.
    #[serde(default)]
    pub native: Natives,
    /// libtl2cgen from the installed tl2cgen package, by path and hash.
    #[serde(default)]
    pub tl2cgen_runtime: Option<NativeLib>,
    /// The sentinel cell ([`crate::sentinel`]): a cell entry with `status` ready,
    /// or its ID, status and the reason it cannot run.
    #[serde(default)]
    pub sentinel: Option<Value>,
    pub cells: Vec<ManifestCell>,
}

/// A native C library, by path and hash.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct NativeLib {
    pub path: PathBuf,
    pub sha256: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub commit: String,
    /// The build record from `build-native`: its identity, flags and compiler.
    #[serde(default)]
    pub build: Value,
}

#[derive(Debug, Clone, Default, Deserialize, serde::Serialize)]
pub struct Natives {
    pub lightgbm: Option<NativeLib>,
    pub xgboost: Option<NativeLib>,
}

/// One cell of the manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct ManifestCell {
    pub id: String,
    /// Relative to the artifacts directory.
    pub dir: PathBuf,
    pub status: String,
    #[serde(default)]
    pub key: String,
    /// Which methods to time: `factorial` (every method), `ablation` (the research
    /// variants and production `predict`) or `treewalker` (TreeWalker only).
    #[serde(default = "default_methods")]
    pub methods: String,
    /// Research variants to time, beyond production.
    #[serde(default)]
    pub variants: Vec<VariantSpec>,
    /// The modes the suite times, in order: `serving`, `batch` or both.
    pub modes: Vec<String>,
}

fn default_methods() -> String {
    "factorial".into()
}

/// A research variant: runtime ablation flags plus load options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, serde::Serialize)]
pub struct VariantSpec {
    /// Runtime flags, by `Ablation` field name.
    #[serde(default)]
    pub runtime: Vec<String>,
    /// Load-time flags: `disable_tree_ordering`, `disable_prefix_grouping`,
    /// `disable_bitset_intern`, `disable_predicate_dedup`.
    #[serde(default)]
    pub load: Vec<String>,
}

impl VariantSpec {
    /// A stable ID: `all-on`, or the flags joined by `+`, load flags after `|`.
    pub fn id(&self) -> String {
        let mut rt = self.runtime.clone();
        rt.sort();
        let mut ld = self.load.clone();
        ld.sort();
        match (rt.is_empty(), ld.is_empty()) {
            (true, true) => "all-on".into(),
            (false, true) => rt.join("+"),
            (true, false) => format!("|{}", ld.join("+")),
            (false, false) => format!("{}|{}", rt.join("+"), ld.join("+")),
        }
    }

    pub fn ablation(&self) -> Result<treewalker_gbdt::research::Ablation> {
        let mut a = treewalker_gbdt::research::Ablation::default();
        for flag in &self.runtime {
            match flag.as_str() {
                "disable_varying_precompute" => a.disable_varying_precompute = true,
                "disable_predicate_sweep" => a.disable_predicate_sweep = true,
                "disable_unsplit" => a.disable_unsplit = true,
                "disable_monotonic" => a.disable_monotonic = true,
                "disable_exact_sums" => a.disable_exact_sums = true,
                other => bail!("unknown runtime flag {other}"),
            }
        }
        Ok(a)
    }

    pub fn load_options(&self) -> Result<LoadOptions> {
        let mut o = LoadOptions::default();
        for flag in &self.load {
            match flag.as_str() {
                "disable_tree_ordering" => o.disable_tree_ordering = true,
                "disable_prefix_grouping" => o.prefix_depth = 0,
                "disable_bitset_intern" => o.disable_bitset_intern = true,
                "disable_predicate_dedup" => o.disable_predicate_dedup = true,
                other => bail!("unknown load flag {other}"),
            }
        }
        Ok(o)
    }
}

/// Measurement settings, from the manifest's `run` table (`grids.toml [run]`).
#[derive(Debug, Clone, Deserialize, serde::Serialize, PartialEq)]
#[serde(default)]
pub struct RunConfig {
    /// Seed of the group order and the method order.
    pub seed: u64,
    /// Batches per round, capped by the group count. A round covers every group
    /// once with every method.
    pub target_batches: usize,
    /// Rows per call in batch mode; a group wider than this is a call of its own.
    pub batch_rows: usize,
    /// Rounds before the precision target or the time budget may stop a mode.
    pub min_rounds: usize,
    /// The block cap, in rounds.
    pub max_rounds: usize,
    /// Stop once every method's relative standard error of its per-round mean
    /// time per row is below this, in percent.
    pub precision_pct: f64,
    /// Wall-time budget per cell and mode, checked after each round from
    /// `min_rounds` on.
    pub mode_budget_secs: f64,
    /// Read the hardware counters (Linux, `pmu` feature).
    pub hardware_counters: bool,
    /// Rows a round times per method, at most: a larger pool times a seeded sample
    /// of its groups, stratified by group size, the same for every method and
    /// mode, and validation checks every method on exactly those groups. 0 is no
    /// cap.
    pub max_rows_per_round: usize,
    /// Groups a sample keeps at least, beside the row cap; it wins when they
    /// conflict, and the manifest says so.
    pub min_groups_per_round: usize,
    /// The sentinel cell's ID; `treewalker-exp manifest` resolves it.
    pub sentinel_cell: String,
    /// Timed cells between sentinels; 0 turns the sentinel off.
    pub sentinel_every: usize,
    /// A sentinel method's time per row drifting more than this from the run's
    /// first measured sentinel, in percent, is flagged by the analysis.
    pub sentinel_drift_pct: f64,
    /// Extra one-round XGBoost-only processes per XGBoost cell, on the
    /// architectures in `xgboost_process_arches`: XGBoost's AVX2 block walk runs
    /// only when a heap buffer happens to be 32-byte aligned, so its speed has
    /// two modes, random across processes and sometimes switching within one.
    pub xgboost_processes: usize,
    /// `std::env::consts::ARCH` values the extra processes run on.
    pub xgboost_process_arches: Vec<String>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            target_batches: 12,
            batch_rows: 1024,
            min_rounds: 3,
            max_rounds: 10,
            precision_pct: 1.0,
            mode_budget_secs: 120.0,
            hardware_counters: true,
            max_rows_per_round: 65_536,
            min_groups_per_round: 200,
            sentinel_cell: "support/nt500_md4_h16/lightgbm/panel".into(),
            sentinel_every: 25,
            sentinel_drift_pct: 3.0,
            xgboost_processes: 6,
            xgboost_process_arches: vec!["x86_64".into()],
        }
    }
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let m: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        ensure!(
            m.schema_version == MANIFEST_SCHEMA,
            "{}: manifest schema {} (this runner reads {MANIFEST_SCHEMA}); rerun treewalker-exp manifest",
            path.display(),
            m.schema_version
        );
        Ok(m)
    }
}

/// A compiled baseline library recorded in `cell.json`.
#[derive(Debug, Clone)]
pub struct BaselineLib {
    pub path: PathBuf,
    pub sha256: String,
}

/// The stage oracle: `math.fsum` of Treelite GTIL's per-tree outputs on a sample of
/// groups, and the staged finalization recomputed from it.
#[derive(Debug, Clone)]
pub struct Oracle {
    /// Sampled groups, by index.
    pub groups: Vec<usize>,
    /// Per row of the sampled groups, in order: `(row, tree_sum, raw_margin)`.
    pub rows: Vec<(usize, f64, f64)>,
    pub divisor: f64,
    pub base_score: f64,
    /// `identity` or `sigmoid`.
    pub postprocessor: String,
    pub sigmoid_alpha: f64,
}

impl Oracle {
    /// The model's link applied to a raw margin, as production applies it.
    pub fn link(&self, raw_margin: f64) -> f64 {
        if self.postprocessor == "sigmoid" {
            1.0 / (1.0 + (-(raw_margin * self.sigmoid_alpha)).exp())
        } else {
            raw_margin
        }
    }
}

/// One prepared cell, loaded.
#[derive(Debug)]
pub struct Cell {
    pub id: String,
    pub dir: PathBuf,
    /// The whole `cell.json`.
    pub doc: Value,
    pub framework: String,
    pub status: String,
    /// Row-major test data.
    pub data: Vec<f64>,
    pub n_rows: usize,
    pub n_features: usize,
    /// Group boundaries: `offsets[g]..offsets[g + 1]`.
    pub offsets: Vec<usize>,
    /// One entity ID per group: patient, session or what-if base row.
    pub entities: Vec<i64>,
    pub walker_config: PathBuf,
    pub model_treelite: PathBuf,
    pub model_native: Option<PathBuf>,
    /// The reference predictions `prepare` wrote (GTIL for LightGBM, native XGBoost).
    pub reference: Option<Vec<f64>>,
    pub oracle: Option<Oracle>,
    pub tl2cgen: Option<BaselineLib>,
    pub lleaves: Option<BaselineLib>,
    /// QuickScorer's model: LightGBM text, or the XML `compile-baselines` writes for
    /// XGBoost. `Err` holds why there is none.
    pub quickscorer: std::result::Result<PathBuf, String>,
    /// The QuickScorer XML's recorded hash, for XGBoost.
    pub quickscorer_sha256: Option<String>,
}

fn str_field<'a>(doc: &'a Value, key: &str) -> &'a str {
    doc.get(key).and_then(Value::as_str).unwrap_or_default()
}

impl Cell {
    /// The path of a file role, relative paths resolved against the cell directory.
    fn file(dir: &Path, doc: &Value, role: &str) -> Option<PathBuf> {
        let rel = doc.get("files")?.get(role)?.get("path")?.as_str()?;
        Some(dir.join(rel))
    }

    /// Load `<dir>/cell.json` and the cell's data.
    pub fn load(dir: &Path) -> Result<Self> {
        let doc_path = dir.join("cell.json");
        let doc: Value = serde_json::from_str(
            &std::fs::read_to_string(&doc_path)
                .with_context(|| format!("reading {}", doc_path.display()))?,
        )
        .with_context(|| format!("parsing {}", doc_path.display()))?;
        let need = |role: &str| {
            Self::file(dir, &doc, role)
                .with_context(|| format!("{}: no {role} file", doc_path.display()))
        };
        let (data, n_rows, n_features) = data::try_load_raw_f64(&need("test_data")?)?;
        let walker_config = need("walker_config")?;
        let config = WalkerConfig::from_file(&walker_config)?;
        ensure!(
            config.n_features() == n_features,
            "{}: walker_config has {} features, test data {n_features}",
            dir.display(),
            config.n_features()
        );
        let offsets = if let Some(p) = Self::file(dir, &doc, "group_offsets") {
            data::load_group_offsets(&p)?
        } else {
            let w = config.max_group_width();
            ensure!(
                n_rows.is_multiple_of(w),
                "{}: {n_rows} rows are not groups of {w}",
                dir.display()
            );
            (0..=n_rows / w).map(|g| g * w).collect()
        };
        ensure!(
            offsets.last() == Some(&n_rows),
            "{}: group offsets end at {:?}, data has {n_rows} rows",
            dir.display(),
            offsets.last()
        );
        let n_groups = offsets.len() - 1;
        let entities = match Self::file(dir, &doc, "entities") {
            Some(p) => match data::read_npy(&p)? {
                data::Npy::I64(v) => v,
                data::Npy::F64(v) => v.into_iter().map(|x| x as i64).collect(),
            },
            None => (0..n_groups as i64).collect(),
        };
        ensure!(
            entities.len() == n_groups,
            "{}: {} entities for {n_groups} groups",
            dir.display(),
            entities.len()
        );
        let reference = match Self::file(dir, &doc, "predictions") {
            Some(p) if p.exists() => Some(data::read_npy(&p)?.into_f64()),
            _ => None,
        };
        let oracle = match doc.get("oracle") {
            Some(o) if !o.is_null() => Some(Self::load_oracle(dir, &doc, o)?),
            _ => None,
        };
        let baseline = |name: &str| -> Option<BaselineLib> {
            let rec = doc.get("baselines")?.get(name)?;
            let lib = rec.get("library")?.as_str()?;
            let path = Self::file(dir, &doc, "model_treelite_bin")?.with_file_name(lib);
            Some(BaselineLib {
                path,
                sha256: rec.get("sha256")?.as_str()?.to_string(),
            })
        };
        let framework = str_field(&doc, "framework").to_string();
        let model_native = Self::file(dir, &doc, "model_native");
        // Features with a missing value in the cell's inputs.
        let mut has_nan = vec![false; n_features];
        for row in data.chunks_exact(n_features.max(1)) {
            for (seen, v) in has_nan.iter_mut().zip(row) {
                *seen |= v.is_nan();
            }
        }
        let nan_features: std::collections::BTreeSet<usize> =
            (0..n_features).filter(|&f| has_nan[f]).collect();
        let missing = |split_on: &std::collections::BTreeSet<usize>| {
            (!nan_features.is_disjoint(split_on))
                .then(|| crate::external::QUICKSCORER_MISSING.to_string())
        };
        let quickscorer = match framework.as_str() {
            "lightgbm" => model_native.as_ref().map_or_else(
                || Err("no LightGBM text model".to_string()),
                |p| match std::fs::read_to_string(p) {
                    Ok(text) if crate::external::lightgbm_has_categorical(&text) => {
                        Err(crate::external::QUICKSCORER_CATEGORICAL.to_string())
                    }
                    Ok(text) => missing(&crate::external::lightgbm_missing_right_features(&text))
                        .map_or_else(|| Ok(p.clone()), Err),
                    Err(e) => Err(format!("reading {}: {e}", p.display())),
                },
            ),
            "xgboost" => {
                let rec = doc.get("baselines").and_then(|b| b.get("quickscorer"));
                match rec {
                    Some(r) if str_field(r, "status") == "ready" => {
                        let xml = Self::file(dir, &doc, "model_treelite_bin")
                            .unwrap_or_default()
                            .with_file_name(str_field(r, "library"));
                        // The XML keeps no default directions, so any missing value
                        // in a feature the model splits on excludes it: nearly all of
                        // XGBoost's splits send missing values right.
                        match std::fs::read_to_string(&xml) {
                            Ok(text) => missing(&crate::external::quickscorer_xml_features(&text))
                                .map_or(Ok(xml), Err),
                            Err(e) => Err(format!("reading {}: {e}", xml.display())),
                        }
                    }
                    Some(r) => Err(str_field(r, "status").to_string()),
                    None => Err("no QuickScorer XML; run compile-baselines".into()),
                }
            }
            _ => Err("QuickScorer reads LightGBM and XGBoost models only".into()),
        };
        let id = str_field(&doc, "id").to_string();
        Self::slug(&id).with_context(|| format!("{}", doc_path.display()))?;
        Ok(Self {
            id,
            dir: dir.to_path_buf(),
            framework,
            status: str_field(&doc, "status").to_string(),
            data,
            n_rows,
            n_features,
            offsets,
            entities,
            walker_config,
            model_treelite: need("model_treelite_bin")?,
            model_native,
            reference,
            oracle,
            tl2cgen: baseline("tl2cgen"),
            lleaves: baseline("lleaves"),
            quickscorer_sha256: doc
                .get("baselines")
                .and_then(|b| b.get("quickscorer"))
                .and_then(|r| r.get("sha256"))
                .and_then(Value::as_str)
                .map(ToString::to_string),
            quickscorer,
            doc,
        })
    }

    fn load_oracle(dir: &Path, doc: &Value, o: &Value) -> Result<Oracle> {
        let path = Self::file(dir, doc, "oracle").context("cell.json has no oracle file")?;
        let (m, n, cols) = data::try_load_raw_f64(&path)?;
        ensure!(cols == 3, "{}: expected 3 columns", path.display());
        let rows = (0..n)
            .map(|r| (m[r * 3] as usize, m[r * 3 + 1], m[r * 3 + 2]))
            .collect();
        let num = |k: &str| o.get(k).and_then(Value::as_f64).context(k.to_string());
        Ok(Oracle {
            groups: o
                .get("groups")
                .and_then(Value::as_array)
                .context("oracle groups")?
                .iter()
                .map(|v| v.as_u64().map(|g| g as usize).context("oracle group"))
                .collect::<Result<_>>()?,
            rows,
            divisor: num("divisor")?,
            base_score: num("base_score")?,
            postprocessor: str_field(o, "postprocessor").to_string(),
            sigmoid_alpha: num("sigmoid_alpha")?,
        })
    }

    /// Recompute every file hash `cell.json` records; the first mismatch is an error.
    pub fn verify_files(&self) -> Result<()> {
        let files = self
            .doc
            .get("files")
            .and_then(Value::as_object)
            .context("cell.json has no files")?;
        for (role, rec) in files {
            let path = self.dir.join(str_field(rec, "path"));
            let want = str_field(rec, "sha256");
            let got = data::sha256_file(&path)?;
            ensure!(
                got == want,
                "{}: {role} {} has sha256 {got}, cell.json records {want}",
                self.dir.display(),
                path.display()
            );
        }
        Ok(())
    }

    pub const fn n_groups(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn max_group_rows(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0)
    }

    /// Rows of group `g`.
    pub fn group(&self, g: usize) -> &[f64] {
        &self.data[self.offsets[g] * self.n_features..self.offsets[g + 1] * self.n_features]
    }

    /// Load the model with `options` at the configured maximum group width.
    pub fn forest(&self, options: &LoadOptions) -> Result<Forest> {
        Ok(Forest::load_with(
            &self.model_treelite,
            &self.walker_config,
            options,
        )?)
    }

    /// Load the model with the cell's feature roles at another maximum group width.
    pub fn forest_at_width(&self, width: usize) -> Result<Forest> {
        let file = WalkerConfig::from_file(&self.walker_config)?;
        let nf = file.n_features();
        let pick = |role: fn(&WalkerConfig, usize) -> bool| {
            (0..nf).filter(|&f| role(&file, f)).collect::<Vec<_>>()
        };
        let config = WalkerConfig::builder(nf)
            .max_group_width(width)
            .varying(pick(WalkerConfig::is_varying))
            .increasing(pick(WalkerConfig::is_increasing))
            .decreasing(pick(WalkerConfig::is_decreasing))
            .build()?;
        Ok(Forest::from_reader(
            std::fs::File::open(&self.model_treelite)?,
            ModelFormat::TreeliteBinaryV4,
            config,
            &LoadOptions::default(),
        )?)
    }

    /// The model's Treelite JSON dump, when `prepare` wrote one.
    pub fn model_json(&self) -> Option<PathBuf> {
        let p = self.model_treelite.with_file_name("model_treelite.json");
        p.exists().then_some(p)
    }

    /// The cell's output directory name. An ID must be nonempty ASCII with no empty,
    /// `.` or `..` `/`-component and no component starting with `.`. The encoding is
    /// injective: letters, digits, `.` and `-` stay, and every other byte, `/` and `_`
    /// included, becomes `_` and two hex digits, so distinct IDs never share a name.
    pub fn slug(id: &str) -> Result<String> {
        ensure!(
            !id.is_empty()
                && id.is_ascii()
                && !id.chars().any(char::is_control)
                && id
                    .split('/')
                    .all(|p| !p.is_empty() && p != "." && p != ".." && !p.starts_with('.')),
            "unsafe cell ID {id:?}: it would not name one directory under cells/"
        );
        Ok(id
            .bytes()
            .map(|b| {
                if b.is_ascii_alphanumeric() || b == b'.' || b == b'-' {
                    char::from(b).to_string()
                } else {
                    format!("_{b:02x}")
                }
            })
            .collect())
    }
}

/// Every prepared cell under `artifacts`: each directory holding a `cell.json`. For
/// tests, which run on whatever was prepared; the runner reads a manifest.
pub fn discover(artifacts: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if dir.join("cell.json").is_file() {
            out.push(dir.to_path_buf());
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .map(|e| e.path())
            .collect();
        dirs.sort();
        for d in dirs {
            walk(&d, out);
        }
    }
    let mut out = Vec::new();
    walk(artifacts, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_ids_are_stable() {
        let v = |rt: &[&str], ld: &[&str]| VariantSpec {
            runtime: rt.iter().map(ToString::to_string).collect(),
            load: ld.iter().map(ToString::to_string).collect(),
        };
        assert_eq!(v(&[], &[]).id(), "all-on");
        assert_eq!(
            v(&["disable_unsplit", "disable_monotonic"], &[]).id(),
            "disable_monotonic+disable_unsplit"
        );
        assert_eq!(
            v(&["disable_unsplit"], &["disable_tree_ordering"]).id(),
            "disable_unsplit|disable_tree_ordering"
        );
        assert_eq!(
            v(&[], &["disable_prefix_grouping"]).id(),
            "|disable_prefix_grouping"
        );
        assert!(v(&["nope"], &[]).ablation().is_err());
        assert_eq!(
            v(&[], &["disable_prefix_grouping"])
                .load_options()
                .unwrap()
                .prefix_depth,
            0
        );
    }
}
