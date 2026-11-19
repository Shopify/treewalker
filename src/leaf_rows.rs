//! Rows that reach every leaf ([`leaf_rows`]), to compare a model's predictions with
//! another implementation's at each of its leaves: a layout file's with its source
//! model's, or the loaded model's with the framework that trained it.
//!
//! Random rows rarely reach a leaf whose path tests several features, since every
//! split on the path must send the row its way. These rows are built along the paths
//! instead. A row keeps, for each feature, the values it may still take; a walk down a
//! tree narrows them to the side each split takes, and backs out of a side that leaves
//! a feature no value, until it reaches a leaf no earlier row reached. One row walks
//! every tree, so it reaches a leaf in each tree it still fits, and takes, for each
//! feature, a value of those left.
//!
//! A numeric feature keeps an interval and whether NaN remains. A categorical feature
//! keeps the classes of values its splits tell apart: each category a split lists,
//! the values in none of them, and NaN. Both are exact: a row fits a side exactly when
//! some input reaches it, so the rows reach every leaf any input reaches.

use std::cmp::Reverse;
use std::path::Path;

use crate::config::{Choices, LoadOptions, WalkerConfig};
use crate::load::parse::{self, ParsedModel};
use crate::model::{ThresholdType, category, category_list};
use crate::{LoadError, ModelFormat};

/// Rows that together reach every leaf of the model at `model_path` that any input
/// reaches, for the grouping configuration at `config_path`.
///
/// The rows are row-major, the configuration's `n_features` values each, and meant to
/// be predicted alone (in groups of one): a constant feature may differ between rows.
/// Each row reaches a leaf no earlier row reaches, in one or more trees. A numeric
/// value lies inside the interval its splits leave rather than on a threshold, 0 if
/// the interval holds it, and a categorical one is a category the model's splits
/// list, or -1 for a category they do not.
///
/// The model's format follows its extension, as [`Forest::load`](crate::Forest::load)'s.
/// A feature tested both by numeric and by categorical splits is
/// [`LoadError::Unsupported`].
pub fn leaf_rows(
    model_path: impl AsRef<Path>,
    config_path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<Vec<f64>, LoadError> {
    let model_path = model_path.as_ref();
    let config = WalkerConfig::from_file_with(config_path.as_ref(), options)?;
    let format = ModelFormat::from_path(model_path)?;
    let file = std::fs::File::open(model_path)?;
    let parsed = parse::parse_file(file, format, &config, options, &Choices::default())?;
    rows(&parsed, config.n_features())
}

/// [`leaf_rows`] of a model in memory, of `format`, for `config`.
pub fn leaf_rows_from_bytes(
    bytes: &[u8],
    format: ModelFormat,
    config: &WalkerConfig,
    options: &LoadOptions,
) -> Result<Vec<f64>, LoadError> {
    let parsed = parse::parse_bytes(bytes, format, config, options, &Choices::default())?;
    rows(&parsed, config.n_features())
}

/// The value a row takes for a category no split lists: negative values are in no
/// category.
const OTHER: f64 = -1.0;

/// What a split tests, in terms of a row's state.
#[derive(Clone, Copy)]
enum Test {
    /// The largest value that goes left, and the smallest that goes right, in the
    /// model's precision; `None` when none does.
    Numeric {
        left_max: Option<f64>,
        right_min: Option<f64>,
    },
    /// The categories that go left, as a set of the feature's classes at `at` in
    /// [`Forest::sets`].
    Categorical { at: usize },
}

/// How a feature is split on.
#[derive(Clone)]
enum Feature {
    Unused,
    Numeric,
    /// Its splits' categories, sorted, whose indices are its classes, then one class
    /// for every other value but NaN; their set is `words` words at `at` in a row's
    /// classes.
    Categorical {
        categories: Vec<u32>,
        at: usize,
        words: usize,
    },
}

/// The decoded trees, with what the walks need: each node's light child, and each
/// split's test.
struct Forest<'a> {
    parsed: &'a ParsedModel,
    features: Vec<Feature>,
    tests: Vec<Test>,
    sets: Vec<u64>,
    /// Each split's light child, a tree-local index, by the split's index in the
    /// decoded store.
    light: Vec<u32>,
    class_words: usize,
}

/// A row's state: the values each feature may still take.
struct Row {
    /// A numeric feature's interval, empty when `lo > hi`.
    lo: Vec<f64>,
    hi: Vec<f64>,
    nan: Vec<bool>,
    /// The categorical features' classes still allowed.
    classes: Vec<u64>,
    /// What to restore, latest last, to undo the narrowing since a mark; a
    /// categorical feature's words are in `saved`.
    trail: Vec<Undo>,
    saved: Vec<u64>,
}

#[derive(Clone, Copy)]
enum Undo {
    Numeric {
        feature: usize,
        lo: f64,
        hi: f64,
        nan: bool,
    },
    /// A categorical feature's `words` class words at `at`, the last saved.
    Categorical {
        feature: usize,
        nan: bool,
        at: usize,
        words: usize,
    },
}

fn rows(parsed: &ParsedModel, n_features: usize) -> Result<Vec<f64>, LoadError> {
    let forest = Forest::new(parsed, n_features)?;
    let decoded = &parsed.decoded;
    // Each node's leaves that no row reaches yet, by its index in the decoded store.
    let mut uncovered = vec![0u32; forest.light.len()];
    for t in 0..decoded.trees.len() {
        let (start, nodes) = (decoded.trees[t].start as usize, decoded.nodes(t));
        for i in (0..nodes.len()).rev() {
            uncovered[start + i] = if nodes.at(i).is_leaf() {
                1
            } else {
                uncovered[start + i + 1] + uncovered[start + forest.light[start + i] as usize]
            };
        }
    }
    let root = |t: usize| decoded.trees[t].start as usize;
    let mut row = Row::new(&forest);
    let mut walk = Vec::new();
    let mut rows = Vec::new();
    let mut active: Vec<usize> = (0..decoded.trees.len()).collect();
    while !active.is_empty() {
        // The trees with the most leaves left first, while the row fits them most.
        active.sort_by_key(|&t| Reverse(uncovered[root(t)]));
        row.reset(&forest);
        let mut fresh = true;
        for &t in &active {
            if forest.reach(t, &mut row, &mut uncovered, &mut walk) {
                fresh = false;
            } else if fresh {
                // An unnarrowed row fits none of its leaves left: no input reaches them.
                uncovered[root(t)] = 0;
            }
        }
        if !fresh {
            row.values(&forest, &mut rows);
        }
        active.retain(|&t| uncovered[root(t)] > 0);
    }
    Ok(rows)
}

impl<'a> Forest<'a> {
    fn new(parsed: &'a ParsedModel, n_features: usize) -> Result<Self, LoadError> {
        let decoded = &parsed.decoded;
        let f32 = parsed.threshold_type == ThresholdType::F32;
        let category_of = |split: &crate::load::parse::decoded::Split| -> Vec<u32> {
            if split.inline_cat() {
                let word = split.value as u32;
                (0..32).filter(|c| word >> c & 1 != 0).collect()
            } else {
                category_list(&decoded.categories, split.value as u32).to_vec()
            }
        };
        // Each feature's kind, and a categorical one's categories: those an input has,
        // the category itself as a value (a float32 model rounds some to another).
        let mut numeric = vec![false; n_features];
        let mut categories: Vec<Option<Vec<u32>>> = vec![None; n_features];
        for split in &decoded.splits.all {
            let f = split.feature as usize;
            if split.is_categorical() {
                let listed = categories[f].get_or_insert_with(Vec::new);
                listed.extend(
                    category_of(split)
                        .into_iter()
                        .filter(|&c| category(f64::from(c), f32) == Some(c)),
                );
            } else {
                numeric[f] = true;
            }
        }
        let mut class_words = 0;
        let mut features = Vec::with_capacity(n_features);
        for (f, (numeric, categories)) in numeric.into_iter().zip(categories).enumerate() {
            features.push(match (numeric, categories) {
                (true, Some(_)) => {
                    return Err(LoadError::Unsupported(format!(
                        "leaf rows: feature {f} has numeric and categorical splits"
                    )));
                }
                (true, None) => Feature::Numeric,
                (false, None) => Feature::Unused,
                (false, Some(mut categories)) => {
                    categories.sort_unstable();
                    categories.dedup();
                    let words = (categories.len() + 1).div_ceil(64);
                    class_words += words;
                    Feature::Categorical {
                        categories,
                        at: class_words - words,
                        words,
                    }
                }
            });
        }
        let mut sets = Vec::new();
        let tests = decoded
            .splits
            .all
            .iter()
            .map(|split| match &features[split.feature as usize] {
                Feature::Categorical {
                    categories, words, ..
                } => {
                    let at = sets.len();
                    sets.resize(at + words, 0);
                    for c in category_of(split) {
                        if let Ok(class) = categories.binary_search(&c) {
                            sets[at + class / 64] |= 1 << (class % 64);
                        }
                    }
                    Test::Categorical { at }
                }
                _ => numeric_test(split.value, f32),
            })
            .collect();
        let store = decoded
            .trees
            .iter()
            .map(|t| (t.start + t.count) as usize)
            .max();
        let mut light = vec![0u32; store.unwrap_or(0)];
        for t in 0..decoded.trees.len() {
            let (start, nodes) = (decoded.trees[t].start as usize, decoded.nodes(t));
            // Subtree sizes from the last node back: in heavy-first preorder the heavy
            // child follows its parent, and the light child the heavy child's subtree.
            let mut size = vec![1u32; nodes.len()];
            for i in (0..nodes.len()).rev() {
                if !nodes.at(i).is_leaf() {
                    let l = i + 1 + size[i + 1] as usize;
                    light[start + i] = l as u32;
                    size[i] = 1 + size[i + 1] + size[l];
                }
            }
        }
        Ok(Self {
            parsed,
            features,
            tests,
            sets,
            light,
            class_words,
        })
    }

    /// Walk tree `t` to a leaf no row reaches yet that `row` fits, narrowing `row` to
    /// it and counting it reached in `uncovered`; false, with `row` as it was, if
    /// there is none. `walk` is scratch.
    fn reach(&self, t: usize, row: &mut Row, uncovered: &mut [u32], walk: &mut Vec<Step>) -> bool {
        let decoded = &self.parsed.decoded;
        let (start, nodes) = (decoded.trees[t].start as usize, decoded.nodes(t));
        if uncovered[start] == 0 {
            return false;
        }
        let base = row.trail.len();
        walk.clear();
        walk.push(Step {
            node: 0,
            mark: base,
            tried: 0,
        });
        while let Some(&Step { node, mark, tried }) = walk.last() {
            let packed = nodes.at(node as usize);
            if packed.is_leaf() {
                for step in walk.iter() {
                    uncovered[start + step.node as usize] -= 1;
                }
                row.commit();
                return true;
            }
            row.undo(mark);
            // The child with more leaves left first, the heavy one on ties.
            let (heavy, light) = (node + 1, self.light[start + node as usize]);
            let children = if uncovered[start + light as usize] > uncovered[start + heavy as usize]
            {
                [light, heavy]
            } else {
                [heavy, light]
            };
            let split = decoded.split(packed);
            let mut next = None;
            for (i, &child) in children.iter().enumerate().skip(tried as usize) {
                if uncovered[start + child as usize] == 0 {
                    continue;
                }
                let left = (child == heavy) == packed.heavy_left();
                let test = self.tests[packed.id() as usize];
                if row.narrow(
                    self,
                    split.feature as usize,
                    test,
                    split.default_left(),
                    left,
                ) {
                    next = Some((i, child));
                    break;
                }
                row.undo(mark);
            }
            let last = walk.len() - 1;
            match next {
                Some((i, child)) => {
                    walk[last].tried = i as u8 + 1;
                    walk.push(Step {
                        node: child,
                        mark: row.trail.len(),
                        tried: 0,
                    });
                }
                None => {
                    walk.pop();
                }
            }
        }
        row.undo(base);
        false
    }
}

/// A node on a walk's path: where the row's trail stood when the walk reached it, and
/// how many of its children the walk has tried, in [`Forest::reach`]'s order.
#[derive(Clone, Copy)]
struct Step {
    node: u32,
    mark: usize,
    tried: u8,
}

/// A numeric split's test on `threshold`, as the decoder keeps it: F64 models go left
/// at `value <= threshold`, F32 models at `(value as f32) < (threshold as f32)`, whose
/// rows take float32 values.
fn numeric_test(threshold: f64, f32: bool) -> Test {
    let (left_max, right_min) = if f32 {
        let t = threshold as f32;
        let left_max = (t != f32::NEG_INFINITY).then(|| f64::from(t.next_down()));
        (left_max, Some(f64::from(t)))
    } else {
        let right_min = (threshold != f64::INFINITY).then(|| threshold.next_up());
        (Some(threshold), right_min)
    };
    Test::Numeric {
        left_max,
        right_min,
    }
}

impl Row {
    fn new(forest: &Forest<'_>) -> Self {
        let n = forest.features.len();
        let mut row = Self {
            lo: vec![0.0; n],
            hi: vec![0.0; n],
            nan: vec![true; n],
            classes: vec![0; forest.class_words],
            trail: Vec::new(),
            saved: Vec::new(),
        };
        row.reset(forest);
        row
    }

    /// Every value allowed again.
    fn reset(&mut self, forest: &Forest<'_>) {
        self.lo.fill(f64::NEG_INFINITY);
        self.hi.fill(f64::INFINITY);
        self.nan.fill(true);
        for feature in &forest.features {
            if let Feature::Categorical {
                categories,
                at,
                words,
            } = feature
            {
                let classes = categories.len() + 1;
                let set = &mut self.classes[*at..at + words];
                set.fill(u64::MAX);
                if classes % 64 != 0 {
                    set[words - 1] = (1 << (classes % 64)) - 1;
                }
            }
        }
        self.commit();
    }

    /// Keep the values that `test` on `feature` sends `left`; false if none is left.
    fn narrow(
        &mut self,
        forest: &Forest<'_>,
        feature: usize,
        test: Test,
        default_left: bool,
        left: bool,
    ) -> bool {
        let nan = self.nan[feature];
        self.nan[feature] &= default_left == left;
        match test {
            Test::Numeric {
                left_max,
                right_min,
            } => {
                self.trail.push(Undo::Numeric {
                    feature,
                    lo: self.lo[feature],
                    hi: self.hi[feature],
                    nan,
                });
                let (lo, hi) = (&mut self.lo[feature], &mut self.hi[feature]);
                match (left, left_max, right_min) {
                    (true, Some(max), _) => *hi = hi.min(max),
                    (false, _, Some(min)) => *lo = lo.max(min),
                    _ => (*lo, *hi) = (f64::INFINITY, f64::NEG_INFINITY),
                }
                *lo <= *hi || self.nan[feature]
            }
            Test::Categorical { at } => {
                let Feature::Categorical {
                    at: here, words, ..
                } = forest.features[feature]
                else {
                    unreachable!("a categorical test on a categorical feature")
                };
                self.trail.push(Undo::Categorical {
                    feature,
                    nan,
                    at: here,
                    words,
                });
                let classes = &mut self.classes[here..here + words];
                self.saved.extend_from_slice(classes);
                let mut any = false;
                for (class, &set) in classes.iter_mut().zip(&forest.sets[at..at + words]) {
                    *class &= if left { set } else { !set };
                    any |= *class != 0;
                }
                any || self.nan[feature]
            }
        }
    }

    /// Undo the narrowing since the trail stood at `mark`.
    fn undo(&mut self, mark: usize) {
        while self.trail.len() > mark {
            match self.trail.pop().expect("an entry past the mark") {
                Undo::Numeric {
                    feature,
                    lo,
                    hi,
                    nan,
                } => {
                    (self.lo[feature], self.hi[feature], self.nan[feature]) = (lo, hi, nan);
                }
                Undo::Categorical {
                    feature,
                    nan,
                    at,
                    words,
                } => {
                    self.nan[feature] = nan;
                    let from = self.saved.len() - words;
                    self.classes[at..at + words].copy_from_slice(&self.saved[from..]);
                    self.saved.truncate(from);
                }
            }
        }
    }

    /// Keep the narrowing: forget how to undo it.
    fn commit(&mut self) {
        self.trail.clear();
        self.saved.clear();
    }

    /// Append the row's values to `rows`: for each feature, one it may still take.
    fn values(&self, forest: &Forest<'_>, rows: &mut Vec<f64>) {
        let f32 = forest.parsed.threshold_type == ThresholdType::F32;
        for (f, feature) in forest.features.iter().enumerate() {
            rows.push(match feature {
                Feature::Unused => 0.0,
                Feature::Numeric if self.lo[f] <= self.hi[f] => inside(self.lo[f], self.hi[f], f32),
                Feature::Categorical {
                    categories,
                    at,
                    words,
                } => {
                    let classes = &self.classes[*at..at + words];
                    // The first class left, a category before the other values.
                    classes.iter().position(|&w| w != 0).map_or(f64::NAN, |i| {
                        let class = 64 * i + classes[i].trailing_zeros() as usize;
                        categories.get(class).map_or(OTHER, |&c| f64::from(c))
                    })
                }
                Feature::Numeric => f64::NAN,
            });
        }
    }
}

/// A value in the interval `lo ..= hi`, not empty, inside it rather than on its
/// bounds where it can be: 0 if it holds it, else its midpoint, or a step past its
/// finite bound. Float32 models' values are float32.
fn inside(lo: f64, hi: f64, f32: bool) -> f64 {
    if lo <= 0.0 && 0.0 <= hi {
        return 0.0;
    }
    let v = match (lo.is_finite(), hi.is_finite()) {
        (true, true) => f64::midpoint(lo, hi),
        (false, true) => hi - hi.abs().max(1.0),
        (true, false) => lo + lo.abs().max(1.0),
        // Both infinite, one value: 0 is in any wider interval.
        (false, false) => lo,
    };
    let v = if f32 { f64::from(v as f32) } else { v };
    if lo <= v && v <= hi { v } else { hi }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::Forest;
    use crate::rng::Rng;

    fn config(n_features: usize) -> WalkerConfig {
        WalkerConfig::builder(n_features)
            .max_group_width(1)
            .varying([])
            .build()
            .expect("a configuration")
    }

    fn fixture(name: &str) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/import")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// The leaves that `rows` reach, as indices in the loaded model's parsed nodes:
    /// each row stepped down each tree by the full walk's split test, not the code
    /// under test.
    fn reached(forest: &Forest, rows: &[f64]) -> HashSet<usize> {
        let model = &forest.model;
        let mut leaves = HashSet::new();
        for row in rows.chunks_exact(model.config.n_features()) {
            for tree in &model.trees {
                let start = tree.node_start as usize;
                let mut local = 0;
                while !model.nodes[start + local].is_leaf() {
                    local = match model.threshold_type {
                        ThresholdType::F64 => {
                            model.step::<false>(&model.nodes, start + local, local, row)
                        }
                        ThresholdType::F32 => {
                            model.step::<true>(&model.nodes, start + local, local, row)
                        }
                    };
                }
                leaves.insert(start + local);
            }
        }
        leaves
    }

    fn leaves(forest: &Forest) -> usize {
        let model = &forest.model;
        model
            .trees
            .iter()
            .map(|t| {
                let nodes =
                    &model.nodes[t.node_start as usize..(t.node_start + t.node_count) as usize];
                nodes.iter().filter(|n| n.is_leaf()).count()
            })
            .sum()
    }

    /// `count` random rows of values worth trying: on and beside each feature's
    /// thresholds, its categories and their neighbors, -1, ±0, ±infinity and NaN.
    fn sampled(forest: &Forest, count: usize) -> Vec<f64> {
        let model = &forest.model;
        let n = model.config.n_features();
        let mut values = vec![vec![f64::NAN, -1.0, 0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY]; n];
        for node in model.nodes.iter().filter(|n| !n.is_leaf()) {
            let feature = &mut values[node.feature as usize];
            let categories: Vec<u32> = if !node.is_categorical() {
                let t = node.value;
                feature.extend([
                    t,
                    t.next_up(),
                    t.next_down(),
                    f64::from(t as f32),
                    f64::from((t as f32).next_up()),
                ]);
                continue;
            } else if node.inline_cat() {
                (0..32)
                    .filter(|c| node.value as u32 >> c & 1 != 0)
                    .collect()
            } else {
                category_list(&model.categories, node.value as u32).to_vec()
            };
            feature.extend(
                categories
                    .iter()
                    .flat_map(|&c| [f64::from(c), f64::from(c) + 1.0]),
            );
        }
        let mut rng = Rng::new(7);
        (0..count * n).map(|i| *rng.pick(&values[i % n])).collect()
    }

    /// Load `bytes` and their leaf rows; check that the rows reach every leaf that
    /// sampled rows do. The rows' leaves and the model's.
    fn check(bytes: &[u8], format: ModelFormat, n_features: usize) -> (usize, usize) {
        let options = LoadOptions::default();
        let forest =
            Forest::from_bytes(bytes, format, config(n_features), &options).expect("a model");
        let rows =
            leaf_rows_from_bytes(bytes, format, &config(n_features), &options).expect("leaf rows");
        assert_eq!(rows.len() % n_features, 0);
        let by_rows = reached(&forest, &rows);
        let missed: Vec<_> = reached(&forest, &sampled(&forest, 20_000))
            .difference(&by_rows)
            .copied()
            .collect();
        assert!(
            missed.is_empty(),
            "sampled rows reach leaves {missed:?} that the leaf rows miss"
        );
        (by_rows.len(), leaves(&forest))
    }

    /// LightGBM text of one complete tree of depth `n` on `n` features, splitting
    /// feature `d` at 0.5 at depth `d`, the ones of whose leaves the parity of `n` bits.
    fn parity(n: usize) -> String {
        let splits = (1 << n) - 1;
        let child = |i: usize| {
            if i < splits {
                i.to_string()
            } else {
                format!("-{}", i - splits + 1)
            }
        };
        let join = |values: Vec<String>| values.join(" ");
        let depth = |i: usize| (i + 1).ilog2() as usize;
        format!(
            "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nmax_feature_idx={}\n\
             objective=regression\n\nTree=0\nnum_leaves={}\nnum_cat=0\nsplit_feature={}\n\
             threshold={}\ndecision_type={}\nleft_child={}\nright_child={}\nleaf_value={}\n\
             leaf_count={}\ninternal_count={}\n\nend of trees\n",
            n - 1,
            splits + 1,
            join((0..splits).map(|i| depth(i).to_string()).collect()),
            join(vec!["0.5".into(); splits]),
            join(vec!["8".into(); splits]),
            join((0..splits).map(|i| child(2 * i + 1)).collect()),
            join((0..splits).map(|i| child(2 * i + 2)).collect()),
            join(
                (0..=splits)
                    .map(|l| f64::from(l.count_ones() % 2).to_string())
                    .collect()
            ),
            join(vec!["1".into(); splits + 1]),
            join(vec!["2".into(); splits]),
        )
    }

    /// Four interacting features, whose 16 leaves each need all four on their side,
    /// which values sampled per feature reach few of. One row per leaf is the fewest.
    #[test]
    #[expect(clippy::float_cmp, reason = "exact expectations")]
    fn rows_reach_every_leaf_of_interacting_features() {
        let text = parity(4);
        assert_eq!(
            check(text.as_bytes(), ModelFormat::LightGbmText, 4),
            (16, 16)
        );
        let rows = leaf_rows_from_bytes(
            text.as_bytes(),
            ModelFormat::LightGbmText,
            &config(4),
            &LoadOptions::default(),
        )
        .expect("leaf rows");
        assert_eq!(rows.len(), 16 * 4);
        // Inside the intervals the splits leave, not on the thresholds: 0 on the left.
        assert!(rows.iter().all(|&v| v == 0.0 || v == 1.5), "{rows:?}");
    }

    /// Every leaf of the trained and hand-written fixtures, numeric, categorical,
    /// float32 and missing-type, that sampled inputs reach; all of a trained model's.
    #[test]
    fn rows_reach_every_leaf_of_the_fixtures() {
        let lightgbm = [
            "binary",
            "categorical_trained",
            "category_words",
            "custom",
            "lambdarank",
            "missing_types",
            "rf",
            "single_leaf",
            "xentropy",
        ];
        for name in lightgbm {
            let (reached, leaves) = check(
                &fixture(&format!("lightgbm/{name}.txt")),
                ModelFormat::LightGbmText,
                2,
            );
            if !["category_words", "missing_types"].contains(&name) {
                assert_eq!(reached, leaves, "{name}");
            }
        }
        for name in [
            "average_base",
            "categories_f32",
            "categories_left",
            "categories_right",
            "identity",
            "lt_zero",
            "random_forest",
            "ranking",
            "sigmoid_alpha",
            "sigmoid_f32",
            "sigmoid_f64",
        ] {
            let (reached, leaves) = check(
                &fixture(&format!("{name}.json")),
                ModelFormat::TreeliteJson,
                2,
            );
            assert!(reached > 0 && reached <= leaves, "{name}");
        }
    }

    /// A leaf whose path no input satisfies gets no row, and the walks stop.
    #[test]
    fn a_leaf_no_input_reaches_gets_no_row() {
        // Feature 0 <= 0.5 then, on the left, feature 0 <= 1: its right leaf needs
        // both x <= 0.5 and x > 1, and NaN goes right at the root.
        let text = "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nmax_feature_idx=0\n\
            objective=regression\n\nTree=0\nnum_leaves=3\nnum_cat=0\nsplit_feature=0 0\n\
            threshold=0.5 1\ndecision_type=8 8\nleft_child=1 -1\nright_child=-3 -2\n\
            leaf_value=1 2 3\nleaf_count=1 1 1\ninternal_count=3 2\n\nend of trees\n";
        assert_eq!(check(text.as_bytes(), ModelFormat::LightGbmText, 1), (2, 3));
    }

    #[test]
    fn a_feature_split_both_ways_is_unsupported() {
        // Categories 1 and 2 of feature 0 go left, to a split of feature 0 at 0.5.
        let text = "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nmax_feature_idx=0\n\
            objective=regression\n\nTree=0\nnum_leaves=3\nnum_cat=1\nsplit_feature=0 0\n\
            threshold=0 0.5\ndecision_type=1 8\nleft_child=1 -1\nright_child=-3 -2\n\
            leaf_value=1 2 3\nleaf_count=1 1 1\ninternal_count=3 2\ncat_boundaries=0 1\n\
            cat_threshold=6\n\nend of trees\n";
        let result = leaf_rows_from_bytes(
            text.as_bytes(),
            ModelFormat::LightGbmText,
            &config(1),
            &LoadOptions::default(),
        );
        assert!(matches!(result, Err(LoadError::Unsupported(m)) if m.contains("feature 0")));
    }
}
