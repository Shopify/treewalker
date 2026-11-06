//! A categorical feature's heavy sets as chains: sequences of strictly nested sets
//! that hold every set once.
//!
//! A field holds one rank per row, and a split on it tests `rank <= k`, so the splits
//! one field serves have nested sublevel sets: a chain. The fewest chains are the
//! sets less a maximum matching of sets to strict supersets, each set's match its
//! successor in its chain (Dilworth's theorem, by König's). The cover finds that
//! matching by augmenting paths without building the inclusion graph: a set's strict
//! supersets are looked up as a path needs them ([`Supersets`]), and each search
//! stops after a bounded number of steps, so the work is linear in the lists. When no
//! search stops early, the cover is a minimum one.

/// The heavy sets of one feature, distinct, each a sorted list of codes over the
/// domain `0..domain`: the codes a listed set holds, or the codes a complemented set
/// lacks. Code 0, no category, is in no split's list, so a listed set never holds it
/// and a complemented one always does.
pub(crate) struct Sets<'a> {
    pub(crate) domain: usize,
    /// Set i's list, at `ranges[i]` in `codes`; sets can share codes.
    pub(crate) ranges: &'a [(u32, u32)],
    pub(crate) codes: &'a [u32],
    /// Whether each set is its list's complement.
    pub(crate) complement: &'a [bool],
}

impl Sets<'_> {
    pub(crate) const fn len(&self) -> usize {
        self.complement.len()
    }

    pub(crate) fn list(&self, i: usize) -> &[u32] {
        let (start, end) = self.ranges[i];
        &self.codes[start as usize..end as usize]
    }
}

/// The codes of sorted `a` that sorted `b` lacks.
pub(crate) fn difference<'a>(a: &'a [u32], b: &'a [u32]) -> impl Iterator<Item = u32> + 'a {
    let mut j = 0;
    a.iter().copied().filter(move |&x| {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        j == b.len() || b[j] != x
    })
}

/// A chain cover: the sets of each chain in increasing order, chain after chain, and
/// where each set is.
#[derive(Debug)]
pub(crate) struct Cover {
    /// Every set, chain by chain, each chain from its smallest set.
    pub(crate) order: Vec<u32>,
    /// Each chain's first entry in `order`, then `order`'s length.
    pub(crate) starts: Vec<u32>,
    /// Each set's chain and position in it.
    pub(crate) at: Vec<(u32, u32)>,
    /// Whether no search stopped at its bound, so that the cover is a minimum one.
    pub(crate) minimum: bool,
}

impl Cover {
    pub(crate) const fn chains(&self) -> usize {
        self.starts.len() - 1
    }

    /// Chain `c`'s sets, smallest first.
    pub(crate) fn chain(&self, c: usize) -> &[u32] {
        &self.order[self.starts[c] as usize..self.starts[c + 1] as usize]
    }
}

const NONE: u32 = u32::MAX;

/// Steps a set's lookup of a free superset may take, per code of its list and one
/// more, past those it draws from the cover's shared steps: one keeps the cover linear
/// in the lists, and a lookup it stops leaves the cover short, so its model packs.
const WORK_PER_CODE: u64 = 1;

/// Steps a model's covers' lookups and searches share, past the lookups' own. They
/// bound the load's time, not its layout: a model whose covers run out packs, where
/// an extra chain takes a field. 2^24 steps take 15 to 25 ms on Genoa over dense
/// synthetic families; the benchmark grid's models take at most 1.3 million.
pub(crate) const SHARED_WORK: u64 = 1 << 24;

/// A chain cover of `sets`: a matching of each set to a strict superset, its
/// successor, taken greedily and then grown by augmenting paths.
///
/// Largest sets first, each set takes the first of its strict supersets without a
/// predecessor that its lookup finds, tightest first. Then the sets still without a
/// successor search for augmenting paths, in phases (as Pothen and Fan's matching
/// does). At each set it reaches, a search looks first for a superset without a
/// predecessor, which completes the path; then it tries the others in turn, each
/// passing the search on to its predecessor, which may take another superset and free
/// this one. A superset once taken stays taken, so a set's look-ahead resumes where
/// it last stopped and scans its supersets at most once in all. A phase's searches
/// share their marks, so a superset is tried at most once a phase; a path found may
/// hide another behind a mark, which a later phase finds. Before a phase's first path
/// the marks are exact, so a set whose search fails then has no path later either (a
/// path from another set gives none to a set that had none) and searches no more. A
/// phase that finds no path ends the search: every search in it failed with exact
/// marks, so with every search run to its end the matching is a maximum one.
///
/// Steps, superset lookups and code checks alike, are bounded: the lookups and
/// searches draw on `pool`, which a model's covers share ([`SHARED_WORK`]), and once
/// it is spent a lookup takes at most [`WORK_PER_CODE`] per code of its set's list and
/// one more, and a search stops at once, its set a chain's top. So the work is linear
/// in the lists, and a cover whose pool doesn't run out ([`Cover::minimum`]) takes the
/// fewest chains.
pub(crate) fn cover(sets: &Sets<'_>, pool: &mut u64) -> Cover {
    cover_with(sets, WORK_PER_CODE, pool).0
}

/// [`cover`] with `per_code` steps per lookup and code and one more; with the steps
/// taken.
pub(crate) fn cover_with(sets: &Sets<'_>, per_code: u64, pool: &mut u64) -> (Cover, u64) {
    let n = sets.len();
    assert!(u32::try_from(n).is_ok_and(|n| n < NONE));
    let lookup = Supersets::new(sets);
    let size = |i: u32| {
        let len = sets.list(i as usize).len();
        if sets.complement[i as usize] {
            sets.domain - len
        } else {
            len
        }
    };
    let mut roots: Vec<u32> = (0..n as u32).collect();
    roots.sort_by_key(|&i| std::cmp::Reverse(size(i)));
    let (mut successor, mut predecessor) = (vec![NONE; n], vec![NONE; n]);
    let mut total = 0u64;
    // Each set's look-ahead for a superset without a predecessor.
    let mut ahead: Vec<Scan> = (0..n as u32).map(|u| lookup.start(u)).collect();
    for &root in &roots {
        let own = per_code.saturating_mul(sets.list(root as usize).len() as u64 + 1);
        let limit = own.saturating_add(*pool);
        let mut work = 0;
        let free = |v: u32| predecessor[v as usize] == NONE;
        if let Some(v) = lookup.next(root, &mut ahead[root as usize], &mut work, limit, free) {
            successor[root as usize] = v;
            predecessor[v as usize] = root;
        }
        *pool -= work.saturating_sub(own).min(*pool);
        total += work;
    }
    // The phase in which each set was last tried as a superset, and whether a set's
    // search failed with exact marks.
    let (mut tried, mut phase) = (vec![NONE; n], 0u32);
    let mut done = vec![false; n];
    let mut stack: Vec<Frame> = Vec::new();
    let mut minimum = true;
    'phases: loop {
        let mut found = false;
        for &root in &roots {
            if successor[root as usize] != NONE || done[root as usize] {
                continue;
            }
            if *pool == 0 {
                minimum = false;
                break 'phases;
            }
            let mut work = 0;
            stack.clear();
            stack.push(Frame {
                set: root,
                via: NONE,
                scan: lookup.start(root),
            });
            while let Some(top) = stack.last_mut() {
                let free = |v: u32| predecessor[v as usize] == NONE;
                let ahead_of = &mut ahead[top.set as usize];
                let next = match lookup.next(top.set, ahead_of, &mut work, *pool, free) {
                    None if work <= *pool => {
                        let untried = |v: u32| tried[v as usize] != phase;
                        lookup.next(top.set, &mut top.scan, &mut work, *pool, untried)
                    }
                    next => next,
                };
                let Some(v) = next else {
                    if work > *pool {
                        minimum = false;
                        break;
                    }
                    stack.pop();
                    continue;
                };
                tried[v as usize] = phase;
                let w = predecessor[v as usize];
                if w == NONE {
                    // Each frame takes the superset the frame above freed, the top one v.
                    let mut v = v;
                    for f in stack.iter().rev() {
                        successor[f.set as usize] = v;
                        predecessor[v as usize] = f.set;
                        v = f.via;
                    }
                    found = true;
                    break;
                }
                stack.push(Frame {
                    set: w,
                    via: v,
                    scan: lookup.start(w),
                });
            }
            *pool -= work.min(*pool);
            total += work;
            if !minimum {
                break 'phases;
            }
            if successor[root as usize] == NONE && !found {
                done[root as usize] = true;
            }
        }
        if !found {
            break;
        }
        phase += 1;
    }

    let mut cover = Cover {
        order: Vec::with_capacity(n),
        starts: Vec::new(),
        at: vec![(0, 0); n],
        minimum,
    };
    for first in (0..n).filter(|&i| predecessor[i] == NONE) {
        let chain = cover.starts.len() as u32;
        cover.starts.push(cover.order.len() as u32);
        let (mut i, mut position) = (first as u32, 0);
        while i != NONE {
            cover.order.push(i);
            cover.at[i as usize] = (chain, position);
            position += 1;
            i = successor[i as usize];
        }
    }
    cover.starts.push(cover.order.len() as u32);
    debug_assert_eq!(cover.order.len(), n, "every set is in one chain");
    (cover, total)
}

/// A search's frame: its set, the superset whose predecessor the set was (which the
/// frame below takes), and where the set's lookup of its supersets stands.
struct Frame {
    set: u32,
    via: u32,
    scan: Scan,
}

/// Where a lookup of a set's strict supersets stands: its phase, a code's position or
/// the code itself, and an entry's position.
#[derive(Clone, Copy)]
struct Scan {
    phase: u8,
    i: u32,
    j: u32,
}

/// Each set's strict supersets, by the indexes that find them.
///
/// A listed set `L_u` is inside a listed `L_v` exactly when `L_v` holds every code of
/// `L_u`, so among the listed sets that hold `L_u`'s rarest code; and inside a
/// complement `D \ L_v` exactly when `L_u` and `L_v` are disjoint, which only a scan
/// of the complements finds, past those whose lists are too long: no list holds code
/// 0, so a list of `|D| - |L_u|` codes or more meets `L_u`. A complement `D \ L_u` is
/// inside `D \ L_v` exactly when `L_v ⊊ L_u`, so when `L_v`'s rarest code among the
/// complements' lists is one of `L_u`'s. A complement holds code 0, no category, which
/// no listed set does, so it is in no listed set. A 64-bit signature of each list, a
/// code's bit at the code mod 64, rejects most candidates before their codes are
/// checked.
struct Supersets<'a> {
    sets: &'a Sets<'a>,
    signature: Vec<u64>,
    /// The listed sets that hold each code, shortest first: code c's at
    /// `holders[holder_starts[c]..holder_starts[c + 1]]`.
    holder_starts: Vec<u32>,
    holders: Vec<u32>,
    /// The complements by their list's rarest code, longest list first, as above.
    rare_starts: Vec<u32>,
    rare: Vec<u32>,
    /// The listed sets, shortest first, and the complements, longest list first.
    listed: Vec<u32>,
    complements: Vec<u32>,
    /// The complement of no list, the whole domain, if any.
    whole: u32,
    /// Whether the domain has at most 64 codes, so that the signatures are exact.
    exact: bool,
}

impl<'a> Supersets<'a> {
    fn new(sets: &'a Sets<'a>) -> Self {
        let n = sets.len() as u32;
        let len = |i: u32| sets.list(i as usize).len();
        let signature = (0..n as usize)
            .map(|i| sets.list(i).iter().fold(0u64, |s, &c| s | 1 << (c & 63)))
            .collect();
        let mut listed: Vec<u32> = (0..n).filter(|&i| !sets.complement[i as usize]).collect();
        listed.sort_by_key(|&i| len(i));
        let mut complements: Vec<u32> = (0..n).filter(|&i| sets.complement[i as usize]).collect();
        complements.sort_by_key(|&i| std::cmp::Reverse(len(i)));
        // Code c's entries, filled in the order of `ids`.
        let index = |ids: &[u32], code: &dyn Fn(u32) -> &'a [u32]| {
            let mut starts = vec![0u32; sets.domain + 1];
            for &i in ids {
                for &c in code(i) {
                    starts[c as usize + 1] += 1;
                }
            }
            for c in 0..sets.domain {
                starts[c + 1] += starts[c];
            }
            let mut next = starts[..sets.domain].to_vec();
            let mut entries = vec![0u32; starts[sets.domain] as usize];
            for &i in ids {
                for &c in code(i) {
                    entries[next[c as usize] as usize] = i;
                    next[c as usize] += 1;
                }
            }
            (starts, entries)
        };
        let (holder_starts, holders) = index(&listed, &|i| sets.list(i as usize));
        let mut count = vec![0u32; sets.domain];
        for &i in &complements {
            for &c in sets.list(i as usize) {
                count[c as usize] += 1;
            }
        }
        let rarest: Vec<Option<u32>> = (0..n as usize)
            .map(|i| {
                let list = sets.list(i);
                let at = (0..list.len()).min_by_key(|&k| (count[list[k] as usize], k))?;
                Some(at as u32)
            })
            .collect();
        let whole = complements
            .iter()
            .copied()
            .find(|&i| len(i) == 0)
            .unwrap_or(NONE);
        let (rare_starts, rare) = index(&complements, &|i| {
            rarest[i as usize].map_or(&[], |at| {
                std::slice::from_ref(&sets.list(i as usize)[at as usize])
            })
        });
        Self {
            sets,
            signature,
            holder_starts,
            holders,
            rare_starts,
            rare,
            listed,
            complements,
            whole,
            exact: sets.domain <= u64::BITS as usize,
        }
    }

    fn holders_of(&self, c: u32) -> &[u32] {
        &self.holders
            [self.holder_starts[c as usize] as usize..self.holder_starts[c as usize + 1] as usize]
    }

    fn rare_of(&self, c: u32) -> &[u32] {
        &self.rare[self.rare_starts[c as usize] as usize..self.rare_starts[c as usize + 1] as usize]
    }

    /// The start of a lookup of `u`'s supersets.
    fn start(&self, u: u32) -> Scan {
        let lu = self.sets.list(u as usize);
        let len = |v: &u32| self.sets.list(*v as usize).len();
        if self.sets.complement[u as usize] {
            return Scan {
                phase: 0,
                i: 0,
                j: NONE,
            };
        }
        // The listed sets that hold its rarest code, past those no longer than it.
        let (i, bucket) = lu
            .iter()
            .copied()
            .min_by_key(|&c| self.holder_starts[c as usize + 1] - self.holder_starts[c as usize])
            .map_or((NONE, &self.listed[..]), |c| (c, self.holders_of(c)));
        Scan {
            phase: 0,
            i,
            j: bucket.partition_point(|v| len(v) <= lu.len()) as u32,
        }
    }

    /// The next strict superset of `u` that `scan` finds and `allow` takes, while
    /// `work` is at most `limit`: a step per candidate, and one per code checked when
    /// the signatures, which are exact over at most 64 codes, leave it in doubt.
    fn next(
        &self,
        u: u32,
        scan: &mut Scan,
        work: &mut u64,
        limit: u64,
        allow: impl Fn(u32) -> bool,
    ) -> Option<u32> {
        let sets = self.sets;
        let lu = sets.list(u as usize);
        let su = self.signature[u as usize];
        // Charges `steps` and says whether they fit.
        let mut charge = |steps: usize| {
            *work += steps as u64;
            *work <= limit
        };
        if !sets.complement[u as usize] {
            if scan.phase == 0 {
                let bucket = if scan.i == NONE {
                    &self.listed[..]
                } else {
                    self.holders_of(scan.i)
                };
                while let Some(&v) = bucket.get(scan.j as usize) {
                    scan.j += 1;
                    if !charge(1) {
                        return None;
                    }
                    if !allow(v) || su & !self.signature[v as usize] != 0 {
                        continue;
                    }
                    if !self.exact {
                        if !charge(lu.len()) {
                            return None;
                        }
                        if !inside(lu, sets.list(v as usize)) {
                            continue;
                        }
                    }
                    return Some(v);
                }
                let room = sets.domain.saturating_sub(lu.len());
                let first = self
                    .complements
                    .partition_point(|&v| sets.list(v as usize).len() >= room);
                (scan.phase, scan.j) = (1, first as u32);
            }
            while let Some(&v) = self.complements.get(scan.j as usize) {
                scan.j += 1;
                if !charge(1) {
                    return None;
                }
                if !allow(v) {
                    continue;
                }
                if su & self.signature[v as usize] != 0 {
                    if self.exact {
                        continue;
                    }
                    let lv = sets.list(v as usize);
                    if !charge(lu.len().min(lv.len())) {
                        return None;
                    }
                    if !disjoint(lu, lv) {
                        continue;
                    }
                }
                return Some(v);
            }
            return None;
        }
        while let Some(&c) = lu.get(scan.i as usize) {
            let bucket = self.rare_of(c);
            if scan.j == NONE {
                scan.j =
                    bucket.partition_point(|&v| sets.list(v as usize).len() >= lu.len()) as u32;
            }
            while let Some(&v) = bucket.get(scan.j as usize) {
                scan.j += 1;
                if !charge(1) {
                    return None;
                }
                if !allow(v) || self.signature[v as usize] & !su != 0 {
                    continue;
                }
                if !self.exact {
                    let lv = sets.list(v as usize);
                    if !charge(lv.len()) {
                        return None;
                    }
                    if !inside(lv, lu) {
                        continue;
                    }
                }
                return Some(v);
            }
            (scan.i, scan.j) = (scan.i + 1, NONE);
        }
        if scan.phase == 0 {
            scan.phase = 1;
            if self.whole != NONE && !lu.is_empty() && allow(self.whole) {
                return Some(self.whole);
            }
        }
        None
    }
}

/// Whether sorted `a` is inside sorted `b`.
fn inside(a: &[u32], b: &[u32]) -> bool {
    let mut rest = b;
    a.iter().all(|x| {
        rest.binary_search(x).is_ok_and(|k| {
            rest = &rest[k + 1..];
            true
        })
    })
}

/// Whether sorted `a` and `b` share no code.
fn disjoint(a: &[u32], b: &[u32]) -> bool {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    short.iter().all(|x| long.binary_search(x).is_err())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Sets as bits over a domain of `domain` codes, sorted and distinct as the
    /// build makes them, and as lists: a set with code 0 is the complement of the
    /// codes it lacks.
    struct Family {
        bits: Vec<u64>,
        ranges: Vec<(u32, u32)>,
        codes: Vec<u32>,
        complement: Vec<bool>,
    }

    impl Family {
        /// The family of `bits`, each code `c` written as `c × scale`, over a domain
        /// of `domain × scale` codes: past 64 codes, lists collide in their bits.
        fn new(domain: u32, scale: u32, mut bits: Vec<u64>) -> Self {
            bits.sort_unstable_by_key(|&s| (s.count_ones(), s));
            bits.dedup();
            let mut family = Self {
                bits,
                ranges: Vec::new(),
                codes: Vec::new(),
                complement: Vec::new(),
            };
            for &set in &family.bits {
                let complement = set & 1 != 0;
                let list = if complement { !set } else { set };
                let at = family.codes.len() as u32;
                family.codes.extend(
                    (0..domain)
                        .filter(|&c| list >> c & 1 != 0)
                        .map(|c| c * scale),
                );
                family.ranges.push((at, family.codes.len() as u32));
                family.complement.push(complement);
            }
            family
        }

        fn sets(&self, domain: u32) -> Sets<'_> {
            Sets {
                domain: domain as usize,
                ranges: &self.ranges,
                codes: &self.codes,
                complement: &self.complement,
            }
        }
    }

    /// The largest antichain, by brute force over subsets of the sets.
    fn width(sets: &[u64]) -> usize {
        let n = sets.len();
        (0u32..1 << n)
            .filter(|&pick| {
                (0..n).all(|i| {
                    (0..n).all(|j| {
                        i == j
                            || pick >> i & 1 == 0
                            || pick >> j & 1 == 0
                            || (sets[i] & !sets[j] != 0 && sets[j] & !sets[i] != 0)
                    })
                })
            })
            .map(|pick| pick.count_ones() as usize)
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn covers_are_chains_of_the_fewest() {
        let mut rng = Rng::new(0x00C4_A125);
        let mut minimum = 0;
        for round in 0..2000 {
            let domain = 1 + rng.below_mod(8) as u32;
            let n = rng.below_mod(13);
            // Mostly small sets, so that some nest.
            let sparse = rng.below_mod(2) == 0;
            let scale = if rng.below_mod(2) == 0 { 1 } else { 41 };
            let family = Family::new(
                domain,
                scale,
                (0..n)
                    .map(|_| {
                        let s = rng.next_u64() & ((1 << domain) - 1);
                        if sparse {
                            s & rng.next_u64() & rng.next_u64()
                        } else {
                            s
                        }
                    })
                    .collect(),
            );
            let sets = &family.bits;
            let cover = cover(&family.sets(domain * scale), &mut { SHARED_WORK });
            // The largest antichain when no search stopped early, else at least it.
            if cover.minimum {
                assert_eq!(cover.chains(), width(sets), "round {round}: {sets:?}");
                minimum += 1;
            } else {
                assert!(cover.chains() >= width(sets), "round {round}: {sets:?}");
            }
            let mut seen = vec![false; sets.len()];
            for c in 0..cover.chains() {
                let chain = cover.chain(c);
                for (k, &i) in chain.iter().enumerate() {
                    assert!(!seen[i as usize]);
                    seen[i as usize] = true;
                    assert_eq!(cover.at[i as usize], (c as u32, k as u32));
                }
                for pair in chain.windows(2) {
                    let (a, b) = (sets[pair[0] as usize], sets[pair[1] as usize]);
                    assert!(a != b && a & !b == 0, "strictly nested");
                }
            }
            assert!(seen.iter().all(|&s| s));
        }
        assert!(minimum > 1900, "{minimum} of 2,000 covers minimum");
    }

    /// Augments the matching `right` from left vertex `u`, if a path is free.
    fn augment(u: usize, adj: &[Vec<usize>], seen: &mut [bool], right: &mut [usize]) -> bool {
        for &v in &adj[u] {
            if !seen[v] {
                seen[v] = true;
                if right[v] == usize::MAX || augment(right[v], adj, seen, right) {
                    right[v] = u;
                    return true;
                }
            }
        }
        false
    }

    /// The size of a maximum matching of the strict-inclusion graph of `sets`, as
    /// sorted lists of codes, by plain augmenting paths over every pair.
    fn matching(sets: &[Vec<u32>]) -> usize {
        let n = sets.len();
        let inside = |a: &[u32], b: &[u32]| a.len() < b.len() && a.iter().all(|c| b.contains(c));
        let adj: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| inside(&sets[i], &sets[j])).collect())
            .collect();
        let mut right = vec![usize::MAX; n];
        (0..n)
            .filter(|&u| augment(u, &adj, &mut vec![false; n], &mut right))
            .count()
    }

    #[test]
    fn covers_of_many_sets_match_a_plain_matching() {
        let mut rng = Rng::new(0x5EED);
        let mut pools = Rng::new(0x9_0015);
        let (mut minimum, mut short) = (0, 0);
        for round in 0..60 {
            // Domains past a word, with codes of every frequency.
            let domain = 2 + rng.below_mod(150) as u32;
            let n = 1 + rng.below_mod(300);
            let mut sets: Vec<(bool, Vec<u32>)> = (0..n)
                .map(|_| {
                    let complement = rng.below_mod(2) == 0;
                    let len = rng.below_mod(8);
                    let mut list: Vec<u32> = (0..len)
                        .map(|_| {
                            let code = rng.below_mod(domain as usize - 1);
                            1 + code.min(rng.below_mod(12)) as u32
                        })
                        .collect();
                    list.sort_unstable();
                    list.dedup();
                    (complement, list)
                })
                .collect();
            sets.sort_unstable();
            sets.dedup();
            let mut family = Family {
                bits: Vec::new(),
                ranges: Vec::new(),
                codes: Vec::new(),
                complement: Vec::new(),
            };
            for (complement, list) in &sets {
                let at = family.codes.len() as u32;
                family.codes.extend_from_slice(list);
                family.ranges.push((at, family.codes.len() as u32));
                family.complement.push(*complement);
            }
            let (cover, steps) =
                cover_with(&family.sets(domain), WORK_PER_CODE, &mut { SHARED_WORK });
            // Again with a pool too short for some, which past 64 codes may run out in
            // the code checks the signatures leave.
            let mut pool = pools.below_mod(steps as usize + 1) as u64;
            let (cut, _) = cover_with(&family.sets(domain), WORK_PER_CODE, &mut pool);
            // The sets themselves, for the reference.
            let full: Vec<Vec<u32>> = sets
                .iter()
                .map(|(complement, list)| {
                    if *complement {
                        (0..domain).filter(|c| !list.contains(c)).collect()
                    } else {
                        list.clone()
                    }
                })
                .collect();
            // A minimum cover's chains when no search stopped early, else at least
            // those and at most the sets, each set once.
            let fewest = full.len() - matching(&full);
            for (cover, full_pool) in [(&cover, true), (&cut, false)] {
                if cover.minimum {
                    assert_eq!(cover.chains(), fewest, "round {round}");
                    minimum += usize::from(full_pool);
                } else {
                    assert!(cover.chains() >= fewest, "round {round}");
                    short += usize::from(!full_pool);
                }
                assert!(cover.chains() <= full.len());
                let mut seen = vec![false; full.len()];
                for c in 0..cover.chains() {
                    for pair in cover.chain(c).windows(2) {
                        let (a, b) = (&full[pair[0] as usize], &full[pair[1] as usize]);
                        assert!(a.len() < b.len() && a.iter().all(|x| b.contains(x)));
                    }
                    for &i in cover.chain(c) {
                        assert!(!seen[i as usize]);
                        seen[i as usize] = true;
                    }
                }
                assert!(seen.iter().all(|&s| s));
            }
        }
        assert!(minimum > 50, "{minimum} of 60 covers minimum");
        assert!(short > 10, "{short} of 60 short pools ran out");
    }

    #[test]
    fn covers_of_dense_small_domains_match_a_plain_matching() {
        // Hundreds of sets over 11 to 16 codes, as the grid's slowest covers have, where
        // the greedy matching leaves many sets to the searches; then each family again
        // with a pool too short for some, whose covers keep chains all the same.
        let mut rng = Rng::new(0xD3_75E5);
        let mut short = 0;
        for round in 0..40 {
            let domain = 11 + rng.below_mod(6) as u32;
            let n = 1 + rng.below_mod(1200);
            let skew = 3 + 3 * rng.below_mod(6);
            let mut sets: Vec<(bool, Vec<u32>)> = (0..n)
                .map(|_| {
                    let complement = rng.below_mod(2) == 0;
                    let len = rng.below_mod(domain as usize);
                    let mut list: Vec<u32> = (0..len)
                        .map(|_| {
                            let code = rng.below_mod(domain as usize - 1);
                            1 + code.min(rng.below_mod(skew)) as u32
                        })
                        .collect();
                    list.sort_unstable();
                    list.dedup();
                    (complement, list)
                })
                .collect();
            sets.sort_unstable();
            sets.dedup();
            let mut family = Family {
                bits: Vec::new(),
                ranges: Vec::new(),
                codes: Vec::new(),
                complement: Vec::new(),
            };
            for (complement, list) in &sets {
                let at = family.codes.len() as u32;
                family.codes.extend_from_slice(list);
                family.ranges.push((at, family.codes.len() as u32));
                family.complement.push(*complement);
            }
            let full: Vec<Vec<u32>> = sets
                .iter()
                .map(|(complement, list)| {
                    if *complement {
                        (0..domain).filter(|c| !list.contains(c)).collect()
                    } else {
                        list.clone()
                    }
                })
                .collect();
            let fewest = full.len() - matching(&full);
            let chains_of = |cover: &Cover| {
                let mut seen = vec![false; full.len()];
                for c in 0..cover.chains() {
                    for pair in cover.chain(c).windows(2) {
                        let (a, b) = (&full[pair[0] as usize], &full[pair[1] as usize]);
                        assert!(a.len() < b.len() && a.iter().all(|x| b.contains(x)));
                    }
                    for &i in cover.chain(c) {
                        assert!(!seen[i as usize]);
                        seen[i as usize] = true;
                    }
                }
                assert!(seen.iter().all(|&s| s));
                cover.chains()
            };
            let family = family.sets(domain);
            let (cover, steps) = cover_with(&family, WORK_PER_CODE, &mut { SHARED_WORK });
            assert!(cover.minimum, "round {round}");
            assert_eq!(chains_of(&cover), fewest, "round {round}");
            let mut pool = rng.below_mod(steps as usize + 1) as u64;
            let (cover, _) = cover_with(&family, WORK_PER_CODE, &mut pool);
            if cover.minimum {
                assert_eq!(chains_of(&cover), fewest, "round {round}, short pool");
            } else {
                assert!(chains_of(&cover) >= fewest, "round {round}, short pool");
                short += 1;
            }
        }
        assert!(short > 10, "{short} of 40 pools ran out");
    }

    #[test]
    fn nested_lists_and_complements_make_chains() {
        // ∅ ⊊ {1} ⊊ {1, 2}, and the complements of {1, 2} and {2}, {0} ⊊ {0, 1}:
        // {1, 2} and {0} are an antichain, so two chains.
        let family = Family::new(3, 1, vec![0, 2, 6, 1 | !6 & 7, 1 | !4 & 7]);
        let cover = cover(&family.sets(3), &mut { SHARED_WORK });
        assert_eq!((cover.chains(), cover.minimum), (2, true));
    }

    #[test]
    fn a_listed_set_continues_into_a_complement() {
        // {1} ⊊ {1, 2}, inside the complement of {3}, {0, 1, 2}, inside that of the
        // empty list: one chain across both kinds.
        let family = Family::new(4, 1, vec![2, 6, 1 | !8 & 15, 15]);
        let cover = cover(&family.sets(4), &mut { SHARED_WORK });
        assert_eq!((cover.chains(), cover.minimum), (1, true));
        assert_eq!(cover.chain(0), [0, 1, 2, 3]);
    }

    #[test]
    fn differences_of_sorted_lists() {
        let d = |a: &[u32], b: &[u32]| difference(a, b).collect::<Vec<_>>();
        assert_eq!(d(&[1, 3, 5, 9], &[0, 3, 4, 9, 10]), [1, 5]);
        assert_eq!(d(&[2], &[]), [2]);
        assert!(d(&[], &[1]).is_empty());
    }
}
