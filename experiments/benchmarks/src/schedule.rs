//! The sampling schedule: which groups each batch holds, and the method order.
//!
//! The timed groups are put in one seeded order and split into `B` batches of equal
//! size (±1). A round gives each method its own phase: it warms up on the batch it
//! times last, then times every batch in order, so every round times every group
//! exactly once with every method, the same groups in the same order. The warm-up
//! never immediately precedes the pass it warms, except with one batch. Methods run
//! one at a time, so no method's state (caches, branch predictor) is trained by
//! another between its warm-up and its timed passes.
//!
//! Method order follows a seeded base order, forward in even rounds and reversed in
//! odd ones: A B, then B A. Over each pair of rounds every method's mean position is
//! the same.

/// SplitMix64: a small, well-mixed generator for seeded orders.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`, by rejection.
    pub const fn below(&mut self, n: usize) -> usize {
        let n = n as u64;
        let zone = u64::MAX - u64::MAX % n;
        loop {
            let v = self.next_u64();
            if v < zone {
                return (v % n) as usize;
            }
        }
    }

    /// Fisher-Yates.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

/// The schedule of one cell and mode.
#[derive(Debug, Clone)]
pub struct Schedule {
    /// Positions in the timed group list, per batch, in timing order.
    pub batches: Vec<Vec<usize>>,
    /// The seeded base order of the methods.
    pub base_order: Vec<usize>,
}

impl Schedule {
    /// `n_groups` groups and `n_methods` methods, in `target_batches` batches, or one
    /// per group when there are fewer groups.
    pub fn new(n_groups: usize, n_methods: usize, target_batches: usize, seed: u64) -> Self {
        assert!(n_groups > 0 && n_methods > 0);
        let mut rng = Rng::new(seed);
        let mut order: Vec<usize> = (0..n_groups).collect();
        rng.shuffle(&mut order);
        let b = target_batches.clamp(1, n_groups);
        let (base, extra) = (n_groups / b, n_groups % b);
        let mut batches = Vec::with_capacity(b);
        let mut at = 0;
        for i in 0..b {
            let len = base + usize::from(i < extra);
            batches.push(order[at..at + len].to_vec());
            at += len;
        }
        let mut base_order: Vec<usize> = (0..n_methods).collect();
        rng.shuffle(&mut base_order);
        Self {
            batches,
            base_order,
        }
    }

    pub const fn blocks_per_round(&self) -> usize {
        self.batches.len()
    }

    /// Rounds per balanced cycle: a forward round and a reversed one.
    pub const fn cycle(&self) -> usize {
        if self.base_order.len() > 1 { 2 } else { 1 }
    }

    /// The method order of round `round`: the base order, reversed in odd rounds.
    pub fn method_order(&self, round: usize) -> Vec<usize> {
        let mut order = self.base_order.clone();
        if round % 2 == 1 {
            order.reverse();
        }
        order
    }

    /// The batch each phase warms up on: the one it times last.
    pub const fn warm_batch(&self) -> usize {
        self.batches.len() - 1
    }
}

/// One stratum of a [`GroupSample`]: the groups whose size falls in
/// `min_size..=max_size`, a power-of-two range.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Stratum {
    pub min_size: usize,
    pub max_size: usize,
    pub pool: usize,
    pub pool_rows: usize,
    pub sampled: usize,
    pub sampled_rows: usize,
    /// `pool / sampled`: what each sampled group stands for in a weighted
    /// estimate; `None` when the draw took none of this stratum.
    pub weight: Option<f64>,
}

/// A seeded sample of a cell's groups, stratified by group size, for a pool above
/// the row cap. Every method times the same sample.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GroupSample {
    /// `max_rows_per_round`: the expected rows, a target rather than a bound.
    pub cap: usize,
    /// `min_groups_per_round`.
    pub min_groups: usize,
    pub seed: u64,
    pub pool_groups: usize,
    pub pool_rows: usize,
    /// Every group's inclusion probability, exactly.
    pub fraction: f64,
    /// `fraction * pool_rows`.
    pub expected_rows: f64,
    /// The rows drawn, which vary around `expected_rows` with the groups' sizes.
    pub rows: usize,
    /// `(rows - expected_rows) / expected_rows`.
    pub overshoot: f64,
    /// Why `fraction` is above the cap's share, when it is: the minimum binds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exception: Option<String>,
    /// The sampled group indices, ascending.
    #[serde(skip)]
    pub groups: Vec<usize>,
    /// SHA-256 of the sampled indices as little-endian `u32`s, ascending.
    pub groups_sha256: String,
    /// Per power-of-two size range, ascending.
    pub strata: Vec<Stratum>,
}

/// The groups to time when the pool's `sizes` sum to more than `cap` rows.
///
/// Every group has the same inclusion probability `f = cap / pool rows`, so the
/// cap is the expected rows; `f` is raised so the sample keeps at least
/// `min_groups` groups (at least one). Groups are stratified by power-of-two size
/// range, and the strata's quotas `n f` are rounded systematically with one seeded
/// uniform: each stratum takes `floor(n f)` or `ceil(n f)` groups with expectation
/// `n f`, and the total is `floor(Q)` or `ceil(Q)` of `Q = N f`, so the minimum
/// always holds and the draw is never empty. The drawn rows vary around the
/// expected rows, since sizes vary within a stratum; their relative overshoot is
/// recorded. `None` when the pool fits or `cap` is 0 (no cap); when the minimum
/// raises `f`, the exception says so, and when it takes every group, the sample
/// is the whole pool.
pub fn sample_groups(
    sizes: &[usize],
    cap: usize,
    min_groups: usize,
    seed: u64,
) -> Option<GroupSample> {
    let pool_rows: usize = sizes.iter().sum();
    if cap == 0 || pool_rows <= cap || sizes.is_empty() {
        return None;
    }
    let mut by_bin: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for (g, &s) in sizes.iter().enumerate() {
        by_bin.entry(s.max(1).ilog2()).or_default().push(g);
    }
    let n = sizes.len();
    let by_cap = cap as f64 / pool_rows as f64;
    let fraction = by_cap.max(min_groups.max(1) as f64 / n as f64).min(1.0);
    let mut rng = Rng::new(seed);
    let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut picked: Vec<usize> = Vec::new();
    let mut cumulative = 0.0f64;
    for members in by_bin.values() {
        let before = (cumulative + u).floor();
        cumulative = (members.len() as f64).mul_add(fraction, cumulative);
        let k = ((cumulative + u).floor() - before) as usize;
        let mut m = members.clone();
        rng.shuffle(&mut m);
        picked.extend_from_slice(&m[..k.min(m.len())]);
    }
    let rows: usize = picked.iter().map(|&g| sizes[g]).sum();
    let exception = (fraction > by_cap).then(|| {
        format!(
            "the minimum of {} groups raises the fraction to {fraction:.4} from the cap's {by_cap:.4}",
            min_groups.max(1)
        )
    });
    let expected_rows = fraction * pool_rows as f64;
    let mut groups = picked;
    groups.sort_unstable();
    let mut strata = Vec::new();
    for (&bin, members) in &by_bin {
        let in_bin = |g: &usize| sizes[*g].max(1).ilog2() == bin;
        let taken: Vec<usize> = groups.iter().copied().filter(in_bin).collect();
        strata.push(Stratum {
            min_size: 1 << bin,
            max_size: (1usize << bin) * 2 - 1,
            pool: members.len(),
            pool_rows: members.iter().map(|&g| sizes[g]).sum(),
            sampled: taken.len(),
            sampled_rows: taken.iter().map(|&g| sizes[g]).sum(),
            weight: (!taken.is_empty()).then(|| members.len() as f64 / taken.len() as f64),
        });
    }
    let bytes: Vec<u8> = groups
        .iter()
        .flat_map(|&g| (g as u32).to_le_bytes())
        .collect();
    Some(GroupSample {
        cap,
        min_groups,
        seed,
        pool_groups: n,
        pool_rows,
        fraction,
        expected_rows,
        rows,
        overshoot: (rows as f64 - expected_rows) / expected_rows,
        exception,
        groups_sha256: crate::data::sha256_bytes(&bytes),
        groups,
        strata,
    })
}

/// A seed for a cell's sample: the run's seed mixed with the cell ID (FNV-1a).
pub fn cell_seed(seed: u64, cell_id: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in cell_id.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    seed ^ h
}

/// Split a block's groups, in order, into calls of at most `budget` rows; a group
/// wider than the budget is a call of its own. Returns `(first, count)` pairs of
/// positions within the block.
pub fn batch_calls(sizes: &[usize], budget: usize) -> Vec<(usize, usize)> {
    let mut calls = Vec::new();
    let (mut first, mut rows) = (0, 0);
    for (i, &s) in sizes.iter().enumerate() {
        if i > first && rows + s > budget {
            calls.push((first, i - first));
            first = i;
            rows = 0;
        }
        rows += s;
    }
    if first < sizes.len() {
        calls.push((first, sizes.len() - first));
    }
    calls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_cover_every_group_once() {
        let s = Schedule::new(103, 7, 12, 1);
        assert_eq!(s.blocks_per_round(), 12);
        assert_eq!(Schedule::new(5, 7, 12, 1).blocks_per_round(), 5);
        let mut seen: Vec<usize> = s.batches.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..103).collect::<Vec<_>>());
        let sizes: Vec<usize> = s.batches.iter().map(Vec::len).collect();
        assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
    }

    #[test]
    fn methods_alternate_forward_and_reversed() {
        let s = Schedule::new(100, 4, 12, 7);
        let (fwd, rev) = (s.method_order(0), s.method_order(1));
        assert_eq!(fwd, s.base_order);
        assert_eq!(rev, fwd.iter().rev().copied().collect::<Vec<_>>());
        // A B B A: round 2 is forward again, round 3 reversed.
        assert_eq!(s.method_order(2), fwd);
        assert_eq!(s.method_order(3), rev);
        // Each round has every method once, and over a cycle every method's
        // positions sum to the same total.
        let mut sums = [0usize; 4];
        for r in 0..s.cycle() {
            let mut o = s.method_order(r);
            for (pos, &m) in o.iter().enumerate() {
                sums[m] += pos;
            }
            o.sort_unstable();
            assert_eq!(o, [0, 1, 2, 3]);
        }
        assert!(sums.iter().all(|&x| x == sums[0]), "{sums:?}");
        assert_eq!(Schedule::new(100, 1, 12, 7).cycle(), 1);
    }

    #[test]
    fn the_warm_up_batch_is_timed_last() {
        let s = Schedule::new(100, 3, 12, 7);
        // Each phase warms on the last batch, then times batches 0..B: the warm-up
        // is followed by B - 1 other batches before its own pass.
        assert_eq!(s.warm_batch(), s.blocks_per_round() - 1);
        assert_ne!(s.warm_batch(), 0);
        assert_eq!(Schedule::new(1, 3, 12, 7).warm_batch(), 0);
    }

    #[test]
    fn calls_respect_the_row_budget() {
        assert_eq!(batch_calls(&[4, 4, 4], 8), vec![(0, 2), (2, 1)]);
        assert_eq!(batch_calls(&[20, 1, 1], 8), vec![(0, 1), (1, 2)]);
        assert_eq!(batch_calls(&[1; 5], 1024), vec![(0, 5)]);
        assert_eq!(batch_calls(&[], 8), vec![]);
    }

    #[test]
    fn samples_are_seeded_stratified_and_on_target() {
        // 600 groups of 1 row, 300 of 4 and 100 of 16: 3,400 rows.
        let sizes: Vec<usize> = (0..1000)
            .map(|g| match g % 10 {
                0 => 16,
                1..=3 => 4,
                _ => 1,
            })
            .collect();
        assert!(sample_groups(&sizes, 3400, 0, 1).is_none());
        assert!(sample_groups(&sizes, 0, 0, 1).is_none());
        let s = sample_groups(&sizes, 850, 0, 1).unwrap();
        // A quarter of the rows: every stratum's quota is whole, so exactly so.
        assert!((s.fraction - 0.25).abs() < 1e-12);
        let got: Vec<(usize, usize)> = s.strata.iter().map(|t| (t.pool, t.sampled)).collect();
        assert_eq!(got, [(600, 150), (300, 75), (100, 25)]);
        assert_eq!((s.rows, s.exception.as_deref()), (850, None));
        assert!(s.groups.windows(2).all(|w| w[0] < w[1]));
        let again = sample_groups(&sizes, 850, 0, 1).unwrap();
        assert_eq!(
            (&again.groups, &again.groups_sha256),
            (&s.groups, &s.groups_sha256)
        );
        assert_ne!(sample_groups(&sizes, 850, 0, 2).unwrap().groups, s.groups);
        assert_ne!(cell_seed(42, "a/b"), cell_seed(42, "a/c"));
    }

    /// Each group's share of `seeds` draws.
    fn inclusion(sizes: &[usize], cap: usize, min: usize, seeds: u64) -> Vec<f64> {
        let mut hits = vec![0u32; sizes.len()];
        for seed in 0..seeds {
            for g in sample_groups(sizes, cap, min, seed).unwrap().groups {
                hits[g] += 1;
            }
        }
        hits.iter().map(|&h| f64::from(h) / seeds as f64).collect()
    }

    #[test]
    fn every_group_has_the_same_inclusion_probability() {
        // One 60,000-row group among 100,000 single rows: with a trim the large
        // group's chance fell to 0.064 and the small ones' to 0.324, against 0.41.
        let mut sizes = vec![1usize; 100_000];
        sizes.push(60_000);
        let s = sample_groups(&sizes, 65_536, 0, 0).unwrap();
        let f = s.fraction;
        assert!((f - 65_536.0 / 160_000.0).abs() < 1e-12);
        let pi = inclusion(&sizes, 65_536, 0, 200);
        let small = pi[..100_000].iter().sum::<f64>() / 100_000.0;
        assert!((small - f).abs() < 0.005, "{small} {f}");
        assert!((pi[100_000] - f).abs() < 0.08, "{} {f}", pi[100_000]);
        // The drawn rows vary around the expected rows; no exception without the
        // minimum binding.
        let mut mean = 0.0;
        for seed in 0..50 {
            let s = sample_groups(&sizes, 65_536, 0, seed).unwrap();
            assert!(s.exception.is_none());
            let want = (s.rows as f64 - s.expected_rows) / s.expected_rows;
            assert!((s.overshoot - want).abs() < 1e-12);
            mean += s.overshoot / 50.0;
        }
        assert!(mean.abs() < 0.2, "{mean}");
    }

    #[test]
    fn the_minimum_always_holds_and_the_draw_is_never_empty() {
        // A pool where an independent rounding drew 199 of 200 groups.
        let mut sizes = Vec::new();
        for (count, size) in [(101, 256), (101, 512), (101, 1024), (697, 2048)] {
            sizes.extend(std::iter::repeat_n(size, count));
        }
        for seed in 0..200 {
            let s = sample_groups(&sizes, 65_536, 200, seed).unwrap();
            assert!((200..=201).contains(&s.groups.len()), "{}", s.groups.len());
        }
        // No minimum and a tiny cap, where every stratum rounded to zero: still
        // one group.
        for seed in 0..20 {
            let s = sample_groups(&[2, 4, 8], 1, 0, seed).unwrap();
            assert_eq!(s.groups.len(), 1);
            assert!(s.exception.unwrap().contains("minimum of 1 groups"));
        }
        // A minimum that takes every group: the whole pool, and why.
        let s = sample_groups(&vec![1008usize; 150], 65_536, 200, 3).unwrap();
        assert_eq!((s.groups.len(), s.fraction), (150, 1.0));
        assert!(s.exception.unwrap().contains("minimum of 200"));
        // 1,000 groups of 1,008 rows: 200 by the minimum.
        let s = sample_groups(&vec![1008usize; 1000], 65_536, 200, 3).unwrap();
        assert_eq!((s.groups.len(), s.rows), (200, 201_600));
    }

    #[test]
    fn seeds_reproduce() {
        let a = Schedule::new(50, 3, 12, 9);
        let b = Schedule::new(50, 3, 12, 9);
        assert_eq!(a.batches, b.batches);
        assert_eq!(a.base_order, b.base_order);
    }
}
