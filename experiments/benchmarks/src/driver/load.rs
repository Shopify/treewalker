//! Loading a cell's methods, and the rows of the groups they are called on.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::Result;
use treewalker_gbdt::Forest;
use treewalker_gbdt::research::LoadAblation;

use crate::artifacts::{Cell, ManifestCell, VariantSpec};
use crate::methods::{CHUNK_ROWS, Engine, TimedMethod};
use crate::schedule::batch_calls;
use crate::suites::MethodSet;

use super::{Context, Excluded};

/// The rows of a set of groups, gathered contiguously.
pub(super) struct Gathered {
    pub(super) rows: Vec<f64>,
    /// Local offsets, from 0.
    pub(super) offsets: Vec<usize>,
}

impl Gathered {
    pub(super) fn new(cell: &Cell, groups: &[usize]) -> Self {
        let nf = cell.n_features;
        let mut rows = Vec::new();
        let mut offsets = vec![0];
        for &g in groups {
            rows.extend_from_slice(cell.group(g));
            offsets.push(rows.len() / nf);
        }
        Self { rows, offsets }
    }

    pub(super) fn group(&self, i: usize, nf: usize) -> &[f64] {
        &self.rows[self.offsets[i] * nf..self.offsets[i + 1] * nf]
    }

    pub(super) fn sizes(&self) -> Vec<usize> {
        self.offsets.windows(2).map(|w| w[1] - w[0]).collect()
    }
}

/// One batch-mode call: positions `first..first + count` of the block.
pub(super) struct Call {
    pub(super) first: usize,
    pub(super) count: usize,
    pub(super) rows: std::ops::Range<usize>,
    pub(super) offsets: Vec<usize>,
}

pub(super) fn calls_for(g: &Gathered, nf: usize, budget: usize) -> Vec<Call> {
    batch_calls(&g.sizes(), budget)
        .into_iter()
        .map(|(first, count)| {
            let base = g.offsets[first];
            Call {
                first,
                count,
                rows: base * nf..g.offsets[first + count] * nf,
                offsets: g.offsets[first..=first + count]
                    .iter()
                    .map(|o| o - base)
                    .collect(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Loading methods
// ---------------------------------------------------------------------------

pub(super) struct Loaded {
    pub(super) methods: Vec<TimedMethod>,
    pub(super) excluded: Vec<Excluded>,
    /// Forests by load options, for counters and full walks.
    pub(super) forests: BTreeMap<String, Forest>,
    pub(super) load_seconds: BTreeMap<String, f64>,
}

pub(super) fn load_key(spec: &VariantSpec) -> String {
    let mut l = spec.load.clone();
    l.sort();
    l.join("+")
}

pub(super) fn wanted(ctx: &Context, method: &str) -> bool {
    ctx.only
        .as_ref()
        .is_none_or(|only| only.iter().any(|m| m == method))
}

pub(super) fn load_methods(
    cell: &Cell,
    mc: &ManifestCell,
    ctx: &Context,
    max_rows: usize,
) -> Result<Loaded> {
    let set = MethodSet::parse(&mc.methods)?;
    let mut l = Loaded {
        methods: Vec::new(),
        excluded: Vec::new(),
        forests: BTreeMap::new(),
        load_seconds: BTreeMap::new(),
    };
    let t0 = Instant::now();
    let base = cell.forest(&LoadAblation::default())?;
    l.load_seconds
        .insert(String::new(), t0.elapsed().as_secs_f64());
    l.forests.insert(String::new(), base.clone());

    if wanted(ctx, "treewalker") {
        l.methods.push(TimedMethod::new(
            "treewalker",
            "all-on",
            Engine::Production(base.predictor()),
            max_rows,
        ));
    }
    if set.full_walk() && wanted(ctx, "treewalker_fullwalk") {
        l.methods.push(TimedMethod::new(
            "treewalker_fullwalk",
            "-",
            Engine::FullWalk(base),
            max_rows,
        ));
    }
    if set.full_walk() && cell.max_group_rows() > CHUNK_ROWS && wanted(ctx, "treewalker_chunked128")
    {
        match cell.forest_at_width(CHUNK_ROWS) {
            Ok(f) => l.methods.push(TimedMethod::new(
                "treewalker_chunked128",
                "-",
                Engine::Chunked(f.predictor()),
                max_rows,
            )),
            Err(e) => l.excluded.push(Excluded {
                method: "treewalker_chunked128".into(),
                variant: "-".into(),
                reason: format!("load: {e}"),
            }),
        }
    }
    for spec in &mc.variants {
        if !wanted(ctx, "treewalker_research") {
            break;
        }
        let id = spec.id();
        let key = load_key(spec);
        let forest = if let Some(f) = l.forests.get(&key) {
            f.clone()
        } else {
            let t0 = Instant::now();
            match spec.load_ablation().and_then(|o| cell.forest(&o)) {
                Ok(f) => {
                    l.load_seconds
                        .insert(key.clone(), t0.elapsed().as_secs_f64());
                    l.forests.insert(key, f.clone());
                    f
                }
                Err(e) => {
                    l.excluded.push(Excluded {
                        method: "treewalker_research".into(),
                        variant: id,
                        reason: format!("load: {e:#}"),
                    });
                    continue;
                }
            }
        };
        let ablation = spec.ablation()?;
        let mut m = TimedMethod::new(
            "treewalker_research",
            &id,
            Engine::Research(forest.research_predictor(ablation)),
            max_rows,
        );
        m.spec = Some(spec.clone());
        l.methods.push(m);
    }
    if set.baselines() {
        load_baselines(cell, ctx, max_rows, &mut l);
    }
    Ok(l)
}

#[cfg(not(feature = "quickscorer-bench"))]
fn load_baselines(_cell: &Cell, _ctx: &Context, _max_rows: usize, l: &mut Loaded) {
    l.excluded.push(Excluded {
        method: "baselines".into(),
        variant: "-".into(),
        reason: "sweep_bench was built without external-bench".into(),
    });
}

#[cfg(feature = "quickscorer-bench")]
fn load_baselines(cell: &Cell, ctx: &Context, max_rows: usize, l: &mut Loaded) {
    use crate::external::{Link, QuickScorerBench};

    let mut add = |name: &str, r: Result<Box<dyn crate::external::ExternalMethod>, String>| {
        if !wanted(ctx, name) {
            return;
        }
        match r {
            Ok(m) => l
                .methods
                .push(TimedMethod::new(name, "-", Engine::External(m), max_rows)),
            Err(reason) => l.excluded.push(Excluded {
                method: name.into(),
                variant: "-".into(),
                reason,
            }),
        }
    };
    #[cfg(feature = "external-bench")]
    {
        use crate::external::{LightGbmBench, LleavesBench, Tl2cgenBench, XgBoostBench};
        let nf = cell.n_features;
        // A compiled library is loaded only when it still has its recorded hash.
        let checked = |b: &crate::artifacts::BaselineLib| -> Result<std::path::PathBuf, String> {
            let got = crate::data::sha256_file(&b.path).map_err(|e| format!("{e:#}"))?;
            if got == b.sha256 {
                Ok(b.path.clone())
            } else {
                Err(format!(
                    "{} has sha256 {got}, cell.json records {}; rerun compile-baselines",
                    b.path.display(),
                    b.sha256
                ))
            }
        };
        let tl2cgen = match (&ctx.tl2cgen_override, &cell.tl2cgen) {
            (Some(p), _) => Some(Ok(p.clone())),
            (None, Some(b)) => Some(checked(b)),
            (None, None) => None,
        };
        add(
            "tl2cgen",
            match (tl2cgen, &ctx.tl2cgen_runtime) {
                (Some(Err(e)), _) => Err(e),
                (Some(Ok(lib)), Some((rt, want))) => {
                    // The runtime must still have the hash the run recorded.
                    match crate::data::sha256_file(rt) {
                        Ok(got) if got == *want => {
                            Tl2cgenBench::load(&lib, rt, nf, max_rows).map(|m| Box::new(m) as _)
                        }
                        Ok(got) => Err(format!(
                            "{} has sha256 {got}, the run recorded {want}",
                            rt.display()
                        )),
                        Err(e) => Err(format!("{e:#}")),
                    }
                }
                (None, _) => Err("not compiled; run compile-baselines".into()),
                (_, None) => Err("no libtl2cgen runtime (uv sync --group baselines)".into()),
            },
        );
        if cell.framework == "lightgbm" {
            add(
                "lleaves",
                cell.lleaves.as_ref().map_or_else(
                    || Err("not compiled; run compile-baselines".into()),
                    |lib| {
                        let path = checked(lib)?;
                        LleavesBench::load(&path, nf, max_rows).map(|m| Box::new(m) as _)
                    },
                ),
            );
            add(
                "lightgbm_native",
                match (&ctx.natives.lightgbm, &cell.model_native) {
                    (Some(lib), Some(model)) => LightGbmBench::load(&lib.path, model, nf, max_rows)
                        .map(|m| Box::new(m) as _),
                    (None, _) => Err("no native LightGBM library; run build-native".into()),
                    (_, None) => Err("no LightGBM text model".into()),
                },
            );
        }
        if cell.framework == "xgboost" {
            add(
                "xgboost_native",
                match (&ctx.natives.xgboost, &cell.model_native) {
                    (Some(lib), Some(model)) => {
                        XgBoostBench::load(&lib.path, model, nf, max_rows).map(|m| Box::new(m) as _)
                    }
                    (None, _) => Err("no native XGBoost library; run build-native".into()),
                    (_, None) => Err("no XGBoost JSON model".into()),
                },
            );
        }
    }
    let link = cell.oracle.as_ref().map(|o| Link {
        divisor: o.divisor,
        base_score: o.base_score,
        sigmoid_alpha: (o.postprocessor == "sigmoid").then_some(o.sigmoid_alpha),
    });
    // The XML must still have its recorded hash; LightGBM's text model is a cell
    // file, already checked.
    let quickscorer = match (&cell.quickscorer, cell.quickscorer_sha256.as_deref()) {
        (Ok(p), Some(want)) => match crate::data::sha256_file(p) {
            Ok(got) if got == want => Ok(p.clone()),
            Ok(got) => Err(format!(
                "{} has sha256 {got}, cell.json records {want}",
                p.display()
            )),
            Err(e) => Err(format!("{e:#}")),
        },
        (other, _) => other.clone(),
    };
    add(
        "quickscorer",
        match (&quickscorer, link) {
            (Ok(model), Some(link)) => {
                QuickScorerBench::load(model, link, cell.n_features, max_rows)
                    .map(|m| Box::new(m) as _)
            }
            (Err(reason), _) => Err(reason.clone()),
            (_, None) => Err("no oracle header with the model's link".into()),
        },
    );
}
