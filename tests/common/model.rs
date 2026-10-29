//! Random Treelite models within the loader's supported subset, their Treelite JSON
//! and v4 binary encodings, and an evaluator that follows the documented semantics
//! (docs/treelite-loading.md) without the library's normalizations.

use super::oracle;
use super::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F64,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Lt,
    Le,
}

#[derive(Clone, Debug)]
pub enum Node {
    Leaf {
        value: f64,
        count: Option<u64>,
    },
    Numeric {
        feature: usize,
        op: Op,
        threshold: f64,
        default_left: bool,
        left: usize,
        right: usize,
        count: Option<u64>,
    },
    Categorical {
        feature: usize,
        categories: Vec<u32>,
        /// Members take the right child instead of the left.
        right_child: bool,
        default_left: bool,
        left: usize,
        right: usize,
        count: Option<u64>,
    },
}

/// A tree in preorder: the root is node 0.
#[derive(Clone, Debug)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub precision: Precision,
    pub n_features: usize,
    pub trees: Vec<Tree>,
    pub average: bool,
    pub base_score: f64,
    /// The sigmoid's alpha, a float32 value; `None` for identity.
    pub sigmoid: Option<f64>,
    pub task: Task,
}

#[derive(Clone, Copy, Debug)]
pub enum Task {
    BinaryClf,
    Regressor,
    Ranking,
}

/// The inputs worth trying for one feature: its thresholds and categories.
#[derive(Clone, Debug, Default)]
pub struct Feature {
    pub thresholds: Vec<f64>,
    /// The largest category its splits may use; `None` if it has no categorical splits.
    pub max_category: Option<u32>,
    /// With [`Settings::wide`], the categories its splits draw from, sorted; empty
    /// otherwise, for categories up to `max_category`.
    pub palette: Vec<u32>,
    /// With [`Settings::nested`], categories in an order whose runs from its first
    /// few entries its splits take, so that their sets nest and overlap; empty
    /// otherwise.
    pub nest: Vec<u32>,
}

/// What the generator draws from.
#[derive(Clone, Debug)]
pub struct Settings {
    pub max_trees: usize,
    /// Nodes per model, at most.
    pub max_nodes: usize,
    /// Share of models whose leaves span past the 126-bit rule.
    pub wide_leaves: f64,
    /// Share of features with categorical splits.
    pub categorical: f64,
    /// Past the old representation limits: more than 64 features, and category IDs
    /// anywhere in `u32`, enough per feature for pooled splits. Off, the generator
    /// draws exactly what it drew before the option existed.
    pub wide: bool,
    /// Deepest tree: 12 draws exactly what the generator drew before the option
    /// existed; more makes some trees deep chains and complete trees.
    pub max_depth: usize,
    /// Share of categorical features whose splits take nested and overlapping
    /// category sets ([`Feature::nest`]): long chains, several per feature. 0 draws
    /// exactly what the generator drew before the option existed.
    pub nested: f64,
}

impl Model {
    /// A leaf value as the loader stores it: float32 models round it.
    fn leaf(&self, value: f64) -> f64 {
        match self.precision {
            Precision::F64 => value,
            Precision::F32 => f64::from(value as f32),
        }
    }

    /// Whether `x` takes the left child.
    fn goes_left(&self, node: &Node, row: &[f64]) -> bool {
        match *node {
            Node::Leaf { .. } => unreachable!("leaves do not split"),
            Node::Numeric {
                feature,
                op,
                threshold,
                default_left,
                ..
            } => {
                let x = row[feature];
                if x.is_nan() {
                    return default_left;
                }
                match (self.precision, op) {
                    (Precision::F64, Op::Lt) => x < threshold,
                    (Precision::F64, Op::Le) => x <= threshold,
                    (Precision::F32, _) => (x as f32) < (threshold as f32),
                }
            }
            Node::Categorical {
                feature,
                ref categories,
                right_child,
                default_left,
                ..
            } => {
                let x = row[feature];
                if x.is_nan() {
                    return default_left;
                }
                let x = match self.precision {
                    Precision::F64 => x,
                    Precision::F32 => f64::from(x as f32),
                };
                // Negative and other out-of-range values are not members; in-range
                // fractions truncate.
                let member = x >= 0.0 && categories.iter().any(|&c| f64::from(c) == x.trunc());
                member != right_child
            }
        }
    }

    /// The leaf value `row` reaches in `tree`.
    pub fn leaf_value(&self, tree: &Tree, row: &[f64]) -> f64 {
        let mut i = 0;
        loop {
            match tree.nodes[i] {
                Node::Leaf { value, .. } => return self.leaf(value),
                Node::Numeric { left, right, .. } | Node::Categorical { left, right, .. } => {
                    i = if self.goes_left(&tree.nodes[i], row) {
                        left
                    } else {
                        right
                    };
                }
            }
        }
    }

    /// The row's leaf values summed exactly, rounded once; +0.0 for zero.
    pub fn tree_sum(&self, row: &[f64]) -> f64 {
        let mut sum = oracle::Sum::new();
        for tree in &self.trees {
            sum.add(self.leaf_value(tree, row));
        }
        sum.round()
    }

    /// The prediction for `row`: the exact tree sum, finalized with production's
    /// steps and roundings (divisor, base score, link).
    #[expect(clippy::float_cmp, reason = "production's exact fast-path identities")]
    pub fn predict(&self, row: &[f64]) -> f64 {
        let divisor = if self.average {
            self.trees.len() as f64
        } else {
            1.0
        };
        let mut v = self.tree_sum(row);
        if divisor != 1.0 || self.base_score != 0.0 {
            v = v / divisor + self.base_score;
        }
        if let Some(alpha) = self.sigmoid {
            if alpha != 1.0 {
                v *= alpha;
            }
            v = 1.0 / (1.0 + (-v).exp());
        }
        v
    }

    pub fn leaves(&self) -> impl Iterator<Item = f64> + '_ {
        self.trees
            .iter()
            .flat_map(|t| &t.nodes)
            .filter_map(|n| match *n {
                Node::Leaf { value, .. } => Some(self.leaf(value)),
                _ => None,
            })
    }

    /// Whether a threshold is infinite, which JSON cannot hold.
    pub fn has_infinite_threshold(&self) -> bool {
        self.trees
            .iter()
            .flat_map(|t| &t.nodes)
            .any(|n| matches!(*n, Node::Numeric { threshold, .. } if threshold.is_infinite()))
    }

    pub fn has_pooled_categories(&self) -> bool {
        self.trees.iter().flat_map(|t| &t.nodes).any(|n| {
            matches!(n, Node::Categorical { categories, .. } if categories.iter().any(|&c| c >= 32))
        })
    }

    pub fn has_inline_categories(&self) -> bool {
        self.trees.iter().flat_map(|t| &t.nodes).any(|n| {
            matches!(n, Node::Categorical { categories, .. } if categories.iter().all(|&c| c < 32))
        })
    }

    /// The largest category any split names.
    pub fn max_category(&self) -> Option<u32> {
        self.trees
            .iter()
            .flat_map(|t| &t.nodes)
            .filter_map(|n| match n {
                Node::Categorical { categories, .. } => categories.last().copied(),
                _ => None,
            })
            .max()
    }

    pub fn node_count(&self) -> usize {
        self.trees.iter().map(|t| t.nodes.len()).sum()
    }

    pub fn depth(&self) -> usize {
        fn depth(tree: &Tree, i: usize) -> usize {
            match tree.nodes[i] {
                Node::Leaf { .. } => 0,
                Node::Numeric { left, right, .. } | Node::Categorical { left, right, .. } => {
                    1 + depth(tree, left).max(depth(tree, right))
                }
            }
        }
        self.trees.iter().map(|t| depth(t, 0)).max().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// A random model and the features its inputs should exercise.
pub fn generate(rng: &mut Rng, settings: &Settings) -> (Model, Vec<Feature>) {
    let precision = if rng.chance(0.3) {
        Precision::F32
    } else {
        Precision::F64
    };
    let n_features = if settings.wide {
        *rng.pick(&[65, 100, 127, 128, 129, 200, 300])
    } else {
        *rng.pick(&[1, 2, 3, 5, 8, 13, 31, 32, 33, 64])
    };
    let infinities = rng.chance(0.3);
    let features: Vec<Feature> = (0..n_features)
        .map(|_| feature(rng, precision, settings, infinities))
        .collect();
    let wide = rng.chance(settings.wide_leaves);
    let leaves = leaf_pool(rng, precision, wide);
    let deep = settings.max_depth > 12;
    let max_depth = if deep {
        rng.range(1, settings.max_depth)
    } else if rng.chance(0.3) {
        12
    } else {
        rng.range(1, 12)
    };
    let n_trees = rng.range(1, settings.max_trees);
    let per_tree = (settings.max_nodes / n_trees).max(3);
    // The chance that a node below the root splits. Roots nearly always split.
    // Some models have complete trees, as far as the node budget allows.
    let split = if rng.chance(0.15) || (deep && rng.chance(0.4)) {
        1.0
    } else {
        rng.unit().mul_add(0.4, 0.55)
    };
    let mut trees: Vec<Tree> = (0..n_trees)
        .map(|_| {
            let mut tree = Tree { nodes: Vec::new() };
            let mut budget = per_tree;
            grow(
                rng,
                &mut tree,
                &features,
                precision,
                &leaves,
                max_depth,
                split,
                &mut budget,
            );
            tree
        })
        .collect();
    // Copies of earlier trees with other leaves and a changed split, so that trees
    // share prefixes, predicates and bitsets.
    if rng.chance(0.4) {
        for i in 1..trees.len() {
            if rng.chance(0.5) {
                let source = rng.below(i);
                trees[i] = variant(rng, &trees[source], &features, &leaves);
            }
        }
    }
    let model = Model {
        precision,
        n_features,
        trees,
        average: rng.chance(0.2),
        base_score: match rng.below(4) {
            0 | 1 => 0.0,
            2 => (rng.unit() - 0.5) * 4.0,
            _ => *rng.pick(&[-0.0, 1e-300, -3.0, 1e10, 0.1]),
        },
        sigmoid: rng
            .chance(0.4)
            .then(|| f64::from(*rng.pick(&[1.0f32, 1.0, 0.5, 2.0, 0.3, 7.25]))),
        task: *rng.pick(&[Task::BinaryClf, Task::Regressor, Task::Ranking]),
    };
    (model, features)
}

/// A feature's thresholds and categories.
fn feature(rng: &mut Rng, precision: Precision, settings: &Settings, infinities: bool) -> Feature {
    let mut f = Feature::default();
    let kind = if rng.chance(settings.categorical) {
        if rng.chance(0.1) { 2 } else { 1 } // categorical, rarely also numeric
    } else {
        0
    };
    if kind != 1 {
        let n = rng.range(1, 12);
        while f.thresholds.len() < n {
            let t = threshold(rng, precision, &f.thresholds);
            if infinities || t.is_finite() {
                f.thresholds.push(t);
            }
        }
    }
    if kind != 0 && settings.wide {
        f.palette = palette(rng);
        f.max_category = f.palette.last().copied();
    } else if kind != 0 {
        f.max_category = Some(if rng.chance(0.5) {
            *rng.pick(&[1, 5, 30, 31])
        } else {
            *rng.pick(&[32, 33, 63, 64, 100, 1000, 8158, 8159])
        });
    }
    if kind != 0 && settings.nested > 0.0 && rng.chance(settings.nested) {
        // Up to 40 of the feature's categories, shuffled.
        let mut nest: Vec<u32> = if f.palette.is_empty() {
            let max = f.max_category.expect("a categorical feature") as usize;
            (0..rng.range(1, 40))
                .map(|_| rng.range(0, max) as u32)
                .collect()
        } else {
            (0..rng.range(1, 40))
                .map(|_| *rng.pick(&f.palette))
                .collect()
        };
        nest.sort_unstable();
        nest.dedup();
        for i in (1..nest.len()).rev() {
            nest.swap(i, rng.below(i + 1));
        }
        f.nest = nest;
    }
    f
}

/// Categories anywhere in `u32`: near 0, near 2^31 and 2^32, and at random, up to 80
/// of them, so that most features have more than the 30 an inline split holds.
fn palette(rng: &mut Rng) -> Vec<u32> {
    let n = rng.range(1, 80);
    let mut p: Vec<u32> = (0..n)
        .map(|_| match rng.below(5) {
            0 => rng.range(0, 64) as u32,
            1 => (1u32 << 31)
                .wrapping_add(rng.range(0, 64) as u32)
                .wrapping_sub(32),
            2 => u32::MAX - rng.range(0, 64) as u32,
            3 => (rng.unit() * 16_777_216.0) as u32,
            _ => (rng.unit() * 4_294_967_296.0) as u32,
        })
        .collect();
    if rng.chance(0.3) {
        p.push(u32::MAX);
    }
    p.sort_unstable();
    p.dedup();
    p
}

/// A threshold: ordinary values, signed zeros, infinities, extremes and neighbors of
/// earlier thresholds, so `<` and `<=` differ on some input.
fn threshold(rng: &mut Rng, precision: Precision, earlier: &[f64]) -> f64 {
    let t = match rng.below(12) {
        0 => rng.range(0, 10) as f64 - 5.0,
        1 => *rng.pick(&[0.0, -0.0]),
        2 => *rng.pick(&[f64::INFINITY, f64::NEG_INFINITY]),
        3 => *rng.pick(&[f64::MAX, -f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)]),
        4 if !earlier.is_empty() => {
            let t = *rng.pick(earlier);
            if rng.chance(0.5) {
                t.next_up()
            } else {
                t.next_down()
            }
        }
        5 => (rng.unit() - 0.5) * 2f64.powi(rng.range(0, 120) as i32 - 60),
        _ => (rng.unit() - 0.5) * 20.0,
    };
    match precision {
        Precision::F64 => t,
        // Thresholds that overflow float32 are rejected; others round at load, and
        // half of them are written unrounded.
        Precision::F32 => {
            let t = if t.is_finite() {
                t.clamp(-f64::from(f32::MAX), f64::from(f32::MAX))
            } else {
                t
            };
            if rng.chance(0.5) {
                t
            } else {
                f64::from(t as f32)
            }
        }
    }
}

/// The leaf values a model draws from. `wide` mixes in exponents far apart, so that
/// most such models have no 126-bit scale.
fn leaf_pool(rng: &mut Rng, precision: Precision, wide: bool) -> Vec<f64> {
    let n = rng.range(1, 40);
    let mut pool: Vec<f64> = (0..n)
        .map(|_| match rng.below(6) {
            0 => (rng.range(0, 64) as f64 - 32.0) / 16.0,
            1 => *rng.pick(&[0.0, -0.0]),
            _ => (rng.unit() - 0.5) * 2f64.powi(rng.range(0, 12) as i32 - 8),
        })
        .collect();
    if wide {
        let extremes: &[f64] = match precision {
            Precision::F64 => &[
                1e300,
                -1e300,
                1e-300,
                -1e-300,
                f64::MAX,
                -f64::MAX,
                f64::MIN_POSITIVE,
                f64::from_bits(1),
                -f64::from_bits(1),
                f64::from_bits(0x000f_ffff_ffff_ffff),
                2f64.powi(200),
                -(2f64.powi(-200)),
            ],
            Precision::F32 => &[
                1e38,
                -1e38,
                1e-38,
                -1e-38,
                f64::from(f32::MAX),
                -f64::from(f32::MAX),
                1.401_298_464_324_817e-45, // the smallest float32 subnormal
                -1.401_298_464_324_817e-45,
                2e30,
                -3e-30,
            ],
        };
        for _ in 0..rng.range(1, 6) {
            let e = *rng.pick(extremes);
            pool.push(e);
            // Extremes beside ordinary values, for cancellation.
            if rng.chance(0.3) {
                pool.push(-e);
            }
            if rng.chance(0.3) {
                pool.push((rng.unit() - 0.5) * e);
            }
        }
    }
    if precision == Precision::F32 {
        for v in &mut pool {
            *v = f64::from(*v as f32);
        }
    }
    pool
}

/// Append a subtree in preorder and return its index.
#[expect(clippy::too_many_arguments, reason = "the generator's state")]
fn grow(
    rng: &mut Rng,
    tree: &mut Tree,
    features: &[Feature],
    precision: Precision,
    leaves: &[f64],
    depth: usize,
    split: f64,
    budget: &mut usize,
) -> usize {
    let index = tree.nodes.len();
    let count = rng.chance(0.8).then(|| rng.below(1000) as u64);
    // A split needs room for its two leaves.
    let split = if index == 0 { 0.97 } else { split };
    if depth == 0 || *budget < 3 || !rng.chance(split) {
        *budget = budget.saturating_sub(1);
        tree.nodes.push(Node::Leaf {
            value: *rng.pick(leaves),
            count,
        });
        return index;
    }
    *budget -= 1;
    let feature = rng.below(features.len());
    let f = &features[feature];
    let default_left = rng.chance(0.5);
    let categorical = match (f.thresholds.is_empty(), f.max_category) {
        (true, _) => true,
        (false, None) => false,
        (false, Some(_)) => rng.chance(0.5),
    };
    let node = if categorical {
        Node::Categorical {
            feature,
            categories: categories_of(rng, f),
            right_child: rng.chance(0.5),
            default_left,
            left: 0,
            right: 0,
            count,
        }
    } else {
        let threshold = *rng.pick(&f.thresholds);
        // Float64 `< -infinity` is rejected; float32 models only have `<`.
        let op = match precision {
            Precision::F32 => Op::Lt,
            Precision::F64 if threshold == f64::NEG_INFINITY => Op::Le,
            Precision::F64 => *rng.pick(&[Op::Lt, Op::Le]),
        };
        Node::Numeric {
            feature,
            op,
            threshold,
            default_left,
            left: 0,
            right: 0,
            count,
        }
    };
    tree.nodes.push(node);
    let l = grow(
        rng,
        tree,
        features,
        precision,
        leaves,
        depth - 1,
        split,
        budget,
    );
    let r = grow(
        rng,
        tree,
        features,
        precision,
        leaves,
        depth - 1,
        split,
        budget,
    );
    match &mut tree.nodes[index] {
        Node::Numeric { left, right, .. } | Node::Categorical { left, right, .. } => {
            (*left, *right) = (l, r);
        }
        Node::Leaf { .. } => unreachable!(),
    }
    index
}

/// A copy of `tree` with leaves drawn again and, half the time, one split below the
/// root changed.
fn variant(rng: &mut Rng, tree: &Tree, features: &[Feature], leaves: &[f64]) -> Tree {
    let mut tree = tree.clone();
    for node in &mut tree.nodes {
        if let Node::Leaf { value, .. } = node {
            *value = *rng.pick(leaves);
        }
    }
    let i = rng.below(tree.nodes.len());
    if i > 0 && rng.chance(0.5) {
        match &mut tree.nodes[i] {
            Node::Leaf { .. } => {}
            Node::Numeric {
                feature,
                op,
                threshold,
                default_left,
                ..
            } => {
                let t = *rng.pick(&features[*feature].thresholds);
                if !(t == f64::NEG_INFINITY && *op == Op::Lt) {
                    *threshold = t;
                }
                *default_left ^= rng.chance(0.5);
            }
            Node::Categorical {
                feature,
                categories: list,
                default_left,
                ..
            } => {
                *list = categories_of(rng, &features[*feature]);
                *default_left ^= rng.chance(0.5);
            }
        }
    }
    tree
}

/// A split's categories on `f`: from its palette if it has one, else up to its
/// largest category.
fn categories_of(rng: &mut Rng, f: &Feature) -> Vec<u32> {
    if !f.nest.is_empty() {
        // A run from one of the first three entries: runs from one start nest, and
        // runs from different starts overlap.
        let n = f.nest.len();
        let a = rng.below(3.min(n));
        let b = rng.range(a, n);
        let mut list = f.nest[a..b].to_vec();
        list.sort_unstable();
        return list;
    }
    if f.palette.is_empty() {
        return categories(rng, f.max_category.expect("a categorical feature"));
    }
    let p = &f.palette;
    let mut list: Vec<u32> = match rng.below(4) {
        0 => Vec::new(),
        1 => {
            let a = rng.below(p.len());
            let b = rng.range(a, (a + 40).min(p.len() - 1));
            p[a..=b].to_vec()
        }
        _ => (0..rng.range(1, 12)).map(|_| *rng.pick(p)).collect(),
    };
    list.sort_unstable();
    list.dedup();
    list
}

/// A sorted category list up to `max`: sparse, dense runs, sometimes empty, and
/// sometimes holding `max` itself.
fn categories(rng: &mut Rng, max: u32) -> Vec<u32> {
    let max = max as usize;
    let mut list: Vec<u32> = match rng.below(5) {
        0 => Vec::new(),
        1 => {
            let a = rng.range(0, max);
            let b = rng.range(a, (a + 40).min(max));
            (a as u32..=b as u32).collect()
        }
        _ => (0..rng.range(1, 12))
            .map(|_| rng.range(0, max) as u32)
            .collect(),
    };
    if rng.chance(0.3) {
        list.push(max as u32);
    }
    list.sort_unstable();
    list.dedup();
    list
}

/// A value for `feature` worth trying: on and beside its thresholds (including the
/// float32 rounding boundaries), its categories and their neighbors, or a special
/// value. `nan` allows NaN.
pub fn input(rng: &mut Rng, feature: &Feature, precision: Precision, nan: bool) -> f64 {
    let special = rng.below(10) == 0;
    if special || (feature.thresholds.is_empty() && feature.max_category.is_none()) {
        let mut v = *rng.pick(&[
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            1e300,
            -1e300,
            f64::from_bits(1),
        ]);
        if v.is_nan() && !nan {
            v = 0.0;
        }
        return v;
    }
    let categorical = match (feature.thresholds.is_empty(), feature.max_category) {
        (true, _) => true,
        (false, None) => false,
        (false, Some(_)) => rng.chance(0.5),
    };
    if categorical && !feature.palette.is_empty() {
        let c = f64::from(*rng.pick(&feature.palette));
        return match rng.below(8) {
            0..=3 => c,
            4 => c + rng.unit(), // truncates
            5 => c + *rng.pick(&[-1.0, 1.0, 0.999_999_9, -0.5]),
            6 => *rng.pick(&[
                4_294_967_295.0,
                4_294_967_295.5,
                4_294_967_296.0,
                2f64.powi(31),
            ]),
            _ => *rng.pick(&[-1.0, -0.0, 0.0, 1e10, 16_777_217.0]),
        };
    }
    if categorical {
        let max = feature.max_category.expect("a categorical feature");
        let c = f64::from(rng.range(0, max as usize) as u32);
        return match rng.below(10) {
            0..=3 => c,
            4 => c + rng.unit(),  // truncates
            5 => c + 0.999_999_9, // rounds up in float32 near large categories
            6 => *rng.pick(&[-1.0, -0.5, -0.0, -1e-300]),
            7 => *rng.pick(&[31.0, 32.0, 33.0, 8159.0, 8160.0, f64::from(max) + 1.0]),
            8 => *rng.pick(&[2f64.powi(31), 2f64.powi(32), 1e10, 1e300]),
            _ => *rng.pick(&[31.999_999, 31.999_999_9, 0.5]),
        };
    }
    let t = *rng.pick(&feature.thresholds);
    if !t.is_finite() {
        return if rng.chance(0.5) { t } else { -t };
    }
    match rng.below(8) {
        0 | 1 => t,
        2 => t.next_up(),
        3 => t.next_down(),
        4 if precision == Precision::F32 => {
            // Beside the midpoint between t's float32 and the next: rounding to
            // float32 decides the side.
            let a = f64::from(t as f32);
            let b = f64::from((t as f32).next_up());
            let mid = f64::midpoint(a, b);
            *rng.pick(&[mid, mid.next_up(), mid.next_down(), b])
        }
        5 => t + (rng.unit() - 0.5),
        _ => t * rng.unit().mul_add(0.2, 0.9),
    }
}

// ---------------------------------------------------------------------------
// Encodings
// ---------------------------------------------------------------------------

const fn task_name(task: Task) -> &'static str {
    match task {
        Task::BinaryClf => "kBinaryClf",
        Task::Regressor => "kRegressor",
        Task::Ranking => "kLearningToRank",
    }
}

/// A JSON number that reads back as `v` exactly.
fn number(v: f64) -> String {
    assert!(v.is_finite(), "JSON has no {v}");
    format!("{v:?}")
}

impl Model {
    /// The Treelite JSON dump, with shuffled node IDs and node order.
    ///
    /// # Panics
    ///
    /// If a threshold is infinite: JSON cannot hold it.
    pub fn to_json(&self, rng: &mut Rng) -> Vec<u8> {
        use std::fmt::Write;
        let ty = match self.precision {
            Precision::F64 => "float64",
            Precision::F32 => "float32",
        };
        let n = self.trees.len();
        let zeros = vec!["0"; n].join(", ");
        let mut s = String::new();
        write!(
            s,
            r#"{{"threshold_type": "{ty}", "leaf_output_type": "{ty}", "num_feature": {}, "task_type": "{}", "average_tree_output": {}, "num_target": 1, "num_class": [1], "leaf_vector_shape": [1, 1], "target_id": [{zeros}], "class_id": [{zeros}], "postprocessor": "{}", "sigmoid_alpha": {}, "ratio_c": 1.0, "base_scores": [{}], "attributes": "{}","#,
            self.n_features,
            task_name(self.task),
            self.average,
            if self.sigmoid.is_some() { "sigmoid" } else { "identity" },
            number(self.sigmoid.unwrap_or(1.0)),
            number(self.base_score),
            if rng.chance(0.5) { "{}" } else { "" },
        )
        .unwrap();
        if rng.chance(0.2) {
            s.push_str(r#" "num_opt_field_per_model": 0,"#);
        }
        s.push_str(r#" "trees": ["#);
        for (t, tree) in self.trees.iter().enumerate() {
            if t > 0 {
                s.push_str(", ");
            }
            // Node IDs: a permutation of 0..n, or of spaced-out IDs.
            let mut ids: Vec<usize> = (0..tree.nodes.len()).collect();
            if rng.chance(0.7) {
                rng.shuffle(&mut ids);
            }
            if rng.chance(0.2) {
                for id in &mut ids {
                    *id = *id * 3 + 7;
                }
            }
            let has_cat = tree
                .nodes
                .iter()
                .any(|n| matches!(n, Node::Categorical { .. }));
            write!(
                s,
                r#"{{"num_nodes": {}, "has_categorical_split": {has_cat}, "#,
                tree.nodes.len()
            )
            .unwrap();
            if ids[0] != 0 || rng.chance(0.5) {
                write!(s, r#""root_id": {}, "#, ids[0]).unwrap();
            }
            s.push_str(r#""nodes": ["#);
            let mut order: Vec<usize> = (0..tree.nodes.len()).collect();
            if rng.chance(0.5) {
                rng.shuffle(&mut order);
            }
            for (k, &i) in order.iter().enumerate() {
                if k > 0 {
                    s.push_str(", ");
                }
                write!(s, r#"{{"node_id": {}"#, ids[i]).unwrap();
                let count = match tree.nodes[i] {
                    Node::Leaf { value, count } => {
                        write!(s, r#", "leaf_value": {}"#, number(value)).unwrap();
                        count
                    }
                    Node::Numeric {
                        feature,
                        op,
                        threshold,
                        default_left,
                        left,
                        right,
                        count,
                    } => {
                        let op = if op == Op::Lt { "<" } else { "<=" };
                        write!(
                            s,
                            r#", "split_feature_id": {feature}, "default_left": {default_left}, "node_type": "numerical_test_node", "comparison_op": "{op}", "threshold": {}, "left_child": {}, "right_child": {}"#,
                            number(threshold),
                            ids[left],
                            ids[right]
                        )
                        .unwrap();
                        count
                    }
                    Node::Categorical {
                        feature,
                        ref categories,
                        right_child,
                        default_left,
                        left,
                        right,
                        count,
                    } => {
                        let list: Vec<String> = categories.iter().map(u32::to_string).collect();
                        write!(
                            s,
                            r#", "split_feature_id": {feature}, "default_left": {default_left}, "node_type": "categorical_test_node", "category_list_right_child": {right_child}, "category_list": [{}], "left_child": {}, "right_child": {}"#,
                            list.join(", "),
                            ids[left],
                            ids[right]
                        )
                        .unwrap();
                        count
                    }
                };
                if let Some(c) = count {
                    write!(s, r#", "data_count": {c}"#).unwrap();
                    if rng.chance(0.3) {
                        write!(s, r#", "sum_hess": {c}.5, "gain": -1.25"#).unwrap();
                    }
                }
                s.push('}');
            }
            s.push_str("]}");
        }
        s.push_str("]}");
        s.into_bytes()
    }

    /// The Treelite v4 binary checkpoint.
    pub fn to_binary(&self, rng: &mut Rng) -> Vec<u8> {
        self.to_binary_with(rng, false)
    }

    /// The Treelite v4 binary checkpoint; with `overlap`, the second categorical
    /// split of each tree names the first one's category segment, which Treelite never
    /// writes.
    pub fn to_binary_with(&self, rng: &mut Rng, overlap: bool) -> Vec<u8> {
        let mut b = Binary(Vec::new());
        let n = self.trees.len();
        for v in [4, rng.range(0, 7) as i32, 0] {
            b.i32(v);
        }
        let ty = match self.precision {
            Precision::F64 => 3u8,
            Precision::F32 => 2,
        };
        b.u8(ty);
        b.u8(ty);
        b.u64(n as u64);
        b.i32(self.n_features as i32);
        b.u8(match self.task {
            Task::BinaryClf => 0,
            Task::Regressor => 1,
            Task::Ranking => 3,
        });
        b.u8(u8::from(self.average));
        b.i32(1); // num_target
        b.array(&[1i32], Binary::i32);
        b.array(&[1i32, 1], Binary::i32);
        b.array(&vec![0i32; n], Binary::i32);
        b.array(&vec![0i32; n], Binary::i32);
        let post = if self.sigmoid.is_some() {
            "sigmoid"
        } else {
            "identity"
        };
        b.array(post.as_bytes(), Binary::u8);
        b.f32(self.sigmoid.unwrap_or(1.0) as f32);
        b.f32(1.0); // ratio_c
        b.array(&[self.base_score], Binary::f64);
        b.array(b"{}", Binary::u8);
        b.i32(0); // num_opt_field_per_model
        for tree in &self.trees {
            self.binary_tree(&mut b, tree, rng, overlap);
        }
        b.0
    }

    fn binary_tree(&self, b: &mut Binary, tree: &Tree, rng: &mut Rng, overlap: bool) {
        let n = tree.nodes.len();
        b.i32(n as i32);
        let has_cat = tree
            .nodes
            .iter()
            .any(|n| matches!(n, Node::Categorical { .. }));
        b.u8(u8::from(has_cat));
        let kind: Vec<i8> = tree
            .nodes
            .iter()
            .map(|n| match n {
                Node::Leaf { .. } => 0,
                Node::Numeric { .. } => 1,
                Node::Categorical { .. } => 2,
            })
            .collect();
        b.array(&kind, |b, v| b.u8(v as u8));
        let children = |n: &Node| match *n {
            Node::Leaf { .. } => (-1, -1, -1),
            Node::Numeric {
                left,
                right,
                feature,
                ..
            }
            | Node::Categorical {
                left,
                right,
                feature,
                ..
            } => (left as i32, right as i32, feature as i32),
        };
        let triples: Vec<(i32, i32, i32)> = tree.nodes.iter().map(children).collect();
        b.array(&triples, |b, t| b.i32(t.0));
        b.array(&triples, |b, t| b.i32(t.1));
        b.array(&triples, |b, t| b.i32(t.2));
        let default_left: Vec<u8> = tree
            .nodes
            .iter()
            .map(|n| match *n {
                Node::Leaf { .. } => 0,
                Node::Numeric { default_left, .. } | Node::Categorical { default_left, .. } => {
                    u8::from(default_left)
                }
            })
            .collect();
        b.array(&default_left, Binary::u8);
        let leaf_values: Vec<f64> = tree
            .nodes
            .iter()
            .map(|n| match *n {
                Node::Leaf { value, .. } => value,
                _ => 0.0,
            })
            .collect();
        let thresholds: Vec<f64> = tree
            .nodes
            .iter()
            .map(|n| match *n {
                Node::Numeric { threshold, .. } => threshold,
                _ => 0.0,
            })
            .collect();
        for values in [&leaf_values, &thresholds] {
            match self.precision {
                Precision::F64 => b.array(values, Binary::f64),
                Precision::F32 => b.array(values, |b, v| b.f32(v as f32)),
            }
        }
        let cmp: Vec<u8> = tree
            .nodes
            .iter()
            .map(|n| match *n {
                Node::Numeric { op: Op::Lt, .. } => 2,
                Node::Numeric { op: Op::Le, .. } => 3,
                _ => 0,
            })
            .collect();
        b.array(&cmp, Binary::u8);
        let right_child: Vec<u8> = tree
            .nodes
            .iter()
            .map(|n| match *n {
                Node::Categorical { right_child, .. } => u8::from(right_child),
                _ => 0,
            })
            .collect();
        b.array(&right_child, Binary::u8);
        b.u64(0); // leaf_vector
        let zeros = if rng.chance(0.5) {
            vec![0u64; n]
        } else {
            Vec::new()
        };
        b.array(&zeros, Binary::u64);
        b.array(&zeros, Binary::u64);
        let (mut list, mut begin, mut end) = (Vec::new(), Vec::new(), Vec::new());
        let mut first = None;
        for node in &tree.nodes {
            begin.push(list.len() as u64);
            if let Node::Categorical { categories, .. } = node {
                list.extend_from_slice(categories);
            }
            end.push(list.len() as u64);
            if overlap && matches!(node, Node::Categorical { .. }) {
                let i = begin.len() - 1;
                match first {
                    None => first = Some((begin[i], end[i])),
                    Some((b0, e0)) => (begin[i], end[i]) = (b0, e0),
                }
            }
        }
        b.array(&list, Binary::u32);
        b.array(&begin, Binary::u64);
        b.array(&end, Binary::u64);
        let counts: Vec<Option<u64>> = tree
            .nodes
            .iter()
            .map(|n| match *n {
                Node::Leaf { count, .. }
                | Node::Numeric { count, .. }
                | Node::Categorical { count, .. } => count,
            })
            .collect();
        b.array(&counts, |b, c| b.u64(c.unwrap_or(0)));
        b.array(&counts, |b, c| b.u8(u8::from(c.is_some())));
        for _ in 0..2 {
            // sum_hess and gain, absent
            b.u64(0);
            b.u64(0);
        }
        b.i32(0);
        b.i32(0);
    }
}

/// A little-endian byte writer.
struct Binary(Vec<u8>);

impl Binary {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    /// A length-prefixed array.
    fn array<T: Copy>(&mut self, values: &[T], mut write: impl FnMut(&mut Self, T)) {
        self.u64(values.len() as u64);
        for &v in values {
            write(self, v);
        }
    }
}
