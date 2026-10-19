//! A categorical feature's heavy sets as chains: the fewest sequences of strictly
//! nested sets that hold every set once.
//!
//! A slot holds one integer per row, and a split on it tests `rank <= k`, so the
//! splits one slot serves have nested sublevel sets: a chain. The fewest slots a
//! feature's heavy sets need is therefore the size of a minimum chain cover of the
//! sets under strict inclusion, which Dilworth's theorem makes the largest antichain.
//! Inclusion is transitive, so a cover by chains is a cover of the inclusion graph's
//! vertices by paths, and the fewest paths are the sets less a maximum matching of
//! the bipartite graph with an edge `i → j` for every `S_i ⊊ S_j` (each matched edge
//! links a set to its successor in its chain). Hopcroft–Karp finds the matching in
//! `O(E √V)`.

use crate::LoadError;

/// The heavy sets of one feature, distinct, each a sorted list of codes over the
/// domain `0..domain`: the codes a listed set holds, or the codes a complemented set
/// lacks. Code 0, no category, is in no split's list, so a listed set never holds it
/// and a complemented one always does.
pub(crate) struct Sets<'a> {
    pub(crate) domain: usize,
    /// Set i's list, at `starts[i]..starts[i + 1]` in `codes`.
    pub(crate) starts: &'a [u32],
    pub(crate) codes: &'a [u32],
    /// Whether each set is its list's complement.
    pub(crate) complement: &'a [bool],
}

impl Sets<'_> {
    pub(crate) const fn len(&self) -> usize {
        self.complement.len()
    }

    pub(crate) fn list(&self, i: usize) -> &[u32] {
        &self.codes[self.starts[i] as usize..self.starts[i + 1] as usize]
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

/// A minimum chain cover: the sets of each chain in increasing order, chain after
/// chain, and where each set is.
#[derive(Debug)]
pub(crate) struct Cover {
    /// Every set, chain by chain, each chain from its smallest set.
    pub(crate) order: Vec<u32>,
    /// Each chain's first entry in `order`, then `order`'s length.
    pub(crate) starts: Vec<u32>,
    /// Each set's chain and position in it.
    pub(crate) at: Vec<(u32, u32)>,
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

/// A minimum chain cover of `sets` under strict inclusion.
///
/// The inclusion graph has two kinds of edges. Between sets of one kind, a listed
/// set `L_a` is inside `L_b` when `L_b` lists all of `L_a`'s codes, and a
/// complement `D \ L_b` inside `D \ L_a` likewise: so a set's list is inside only
/// lists that hold its rarest code, which an index of each kind's sets by the codes
/// they list finds. Those edges are kept, 8 bytes each while they are gathered. A
/// listed set is inside a complement `D \ L_b` when no code of it is in `L_b`:
/// those edges join unrelated splits, so most pairs of a listed and a complemented
/// set may have one, and each listed set keeps them as a row of bits over the
/// complements. Both take `budget` bytes at most, else the load fails, as a hostile
/// file could make them quadratic in the splits. A complement, which holds no
/// category, is in no listed set.
///
/// An index entry of a code that a 32nd or more of its kind's sets list is also a
/// row of bits over them, so that a set whose codes are all that common intersects
/// their rows a word at a time, and a listed set's row of complements is all of
/// them less the rows of its codes. Those rows take at most 2 bits per entry.
pub(crate) fn cover(sets: &Sets<'_>, budget: u64) -> Result<Cover, LoadError> {
    let n = sets.len();
    assert!(u32::try_from(n).is_ok_and(|n| n < NONE));
    let graph = Graph::new(sets, budget)?;
    let (successor, predecessor) = graph.hopcroft_karp();

    let mut cover = Cover {
        order: Vec::with_capacity(n),
        starts: Vec::new(),
        at: vec![(0, 0); n],
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
    Ok(cover)
}

/// The sets of one kind, by their index among the kind's, and per code the ones
/// that list it: its entries in `members`, and for common codes a row of bits.
struct Kind {
    sets: Vec<u32>,
    start: Vec<u32>,
    members: Vec<u32>,
    /// Words per row, and each code's row in `rows`, or [`NONE`].
    words: usize,
    row: Vec<u32>,
    rows: Vec<u64>,
}

impl Kind {
    fn new(all: &Sets<'_>, complement: bool) -> Self {
        let sets: Vec<u32> = (0..all.len() as u32)
            .filter(|&i| all.complement[i as usize] == complement)
            .collect();
        let d = all.domain;
        let mut start = vec![0u32; d + 1];
        for &i in &sets {
            for &code in all.list(i as usize) {
                start[code as usize + 1] += 1;
            }
        }
        for c in 0..d {
            start[c + 1] += start[c];
        }
        let mut members = vec![0u32; start[d] as usize];
        let mut fill = start.clone();
        for (x, &i) in sets.iter().enumerate() {
            for &code in all.list(i as usize) {
                members[fill[code as usize] as usize] = x as u32;
                fill[code as usize] += 1;
            }
        }
        let words = sets.len().div_ceil(64);
        let (mut row, mut rows) = (vec![NONE; d], Vec::new());
        for c in 0..d {
            let listed = &members[start[c] as usize..start[c + 1] as usize];
            if 32 * listed.len() >= sets.len() && !listed.is_empty() {
                row[c] = (rows.len() / words) as u32;
                let at = rows.len();
                rows.resize(at + words, 0);
                for &x in listed {
                    rows[at + x as usize / 64] |= 1 << (x % 64);
                }
            }
        }
        Self {
            sets,
            start,
            members,
            words,
            row,
            rows,
        }
    }

    fn listing(&self, code: u32) -> &[u32] {
        &self.members[self.start[code as usize] as usize..self.start[code as usize + 1] as usize]
    }

    fn bits(&self, code: u32) -> Option<&[u64]> {
        let r = self.row[code as usize];
        (r != NONE).then(|| &self.rows[r as usize * self.words..(r as usize + 1) * self.words])
    }
}

/// Sets bits `0..len` of `words`, which hold `len.div_ceil(64)`, and clears the rest.
fn every(words: &mut [u64], len: usize) {
    words.fill(u64::MAX);
    if let Some(last) = words.last_mut()
        && !len.is_multiple_of(64)
    {
        *last = (1 << (len % 64)) - 1;
    }
}

/// The set bits of `words`, in increasing order.
fn ones(words: &[u64]) -> impl Iterator<Item = usize> + '_ {
    words.iter().enumerate().flat_map(|(w, &bits)| {
        let mut bits = bits;
        std::iter::from_fn(move || {
            (bits != 0).then(|| {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                64 * w + b
            })
        })
    })
}

/// The inclusion graph: the kept edges `adj[adj_start[i]..adj_start[i + 1]]` from
/// set `i`, between sets of one kind, and each listed set's row of the complements
/// its list is disjoint from, by their index among the complements.
struct Graph<'a> {
    sets: &'a Sets<'a>,
    adj_start: Vec<u32>,
    adj: Vec<u32>,
    /// Each set's index among its kind's.
    local: Vec<u32>,
    complements: Vec<u32>,
    /// Words per row of complements, and the rows by listed set.
    words: usize,
    rows: Vec<u64>,
}

impl<'a> Graph<'a> {
    fn new(sets: &'a Sets<'a>, budget: u64) -> Result<Self, LoadError> {
        let n = sets.len();
        let kinds = [Kind::new(sets, false), Kind::new(sets, true)];
        let mut local = vec![0u32; n];
        for kind in &kinds {
            for (x, &i) in kind.sets.iter().enumerate() {
                local[i as usize] = x as u32;
            }
        }
        let (listed, complements) = (&kinds[0], &kinds[1]);
        let words = complements.words;
        let row_bytes = listed.sets.len() as u64 * words as u64 * 8;
        let too_big = || {
            LoadError::Limit(format!(
                "the inclusion graph of a feature's categorical splits exceeds {budget} bytes \
                 (LoadOptions::max_rank_table_bytes)"
            ))
        };
        if row_bytes > budget {
            return Err(too_big());
        }
        // Each listed set's row: the complements less those listing one of its codes.
        let mut rows = vec![0u64; listed.sets.len() * words];
        for (x, &i) in listed.sets.iter().enumerate() {
            let row = &mut rows[x * words..(x + 1) * words];
            every(row, complements.sets.len());
            for &code in sets.list(i as usize) {
                match complements.bits(code) {
                    Some(bits) => {
                        for (r, &b) in row.iter_mut().zip(bits) {
                            *r &= !b;
                        }
                    }
                    None => {
                        for &y in complements.listing(code) {
                            row[y as usize / 64] &= !(1 << (y % 64));
                        }
                    }
                }
            }
        }

        // The kept edges: per kind and set x, by index among the kind's, each set y of
        // the kind whose list strictly holds x's, in increasing order. A listed set's
        // edges go to them; a complement's come from them, so they are turned round.
        let mut found: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        let mut found_start: [Vec<u32>; 2] = [vec![0], vec![0]];
        let mut acc = Vec::new();
        for (k, kind) in kinds.iter().enumerate() {
            for (x, &a) in kind.sets.iter().enumerate() {
                let out = &mut found[k];
                let list = sets.list(a as usize);
                match list.iter().min_by_key(|&&code| kind.listing(code).len()) {
                    // The empty list is inside every other list of its kind.
                    None => out.extend((0..kind.sets.len() as u32).filter(|&y| y as usize != x)),
                    Some(&rarest) if kind.bits(rarest).is_some() => {
                        // Every code of the list is common: the rows' intersection.
                        acc.clear();
                        acc.resize(kind.words, u64::MAX);
                        for &code in list {
                            let bits = kind.bits(code).expect("as common as the rarest");
                            for (r, &b) in acc.iter_mut().zip(bits) {
                                *r &= b;
                            }
                        }
                        acc[x / 64] &= !(1 << (x % 64));
                        out.extend(ones(&acc).map(|y| y as u32));
                    }
                    Some(&rarest) => {
                        out.extend(kind.listing(rarest).iter().copied().filter(|&y| {
                            y as usize != x
                                && difference(list, sets.list(kind.sets[y as usize] as usize))
                                    .next()
                                    .is_none()
                        }));
                    }
                }
                if row_bytes + 8 * (found[0].len() + found[1].len()) as u64 > budget {
                    return Err(too_big());
                }
                found_start[k].push(found[k].len() as u32);
            }
        }
        let edges = u32::try_from(found[0].len() + found[1].len())
            .map_err(|_| LoadError::Limit("an inclusion graph exceeds u32 edges".into()))?;
        let (listed, complements) = (&kinds[0], &kinds[1]);
        let mut adj_start = vec![0u32; n + 1];
        for (x, &a) in listed.sets.iter().enumerate() {
            adj_start[a as usize + 1] = found_start[0][x + 1] - found_start[0][x];
        }
        for &y in &found[1] {
            adj_start[complements.sets[y as usize] as usize + 1] += 1;
        }
        for i in 0..n {
            adj_start[i + 1] += adj_start[i];
        }
        debug_assert_eq!(adj_start[n], edges);
        let mut adj = vec![0u32; edges as usize];
        for (x, &a) in listed.sets.iter().enumerate() {
            let targets = &found[0][found_start[0][x] as usize..found_start[0][x + 1] as usize];
            let at = adj_start[a as usize] as usize;
            for (to, &y) in adj[at..at + targets.len()].iter_mut().zip(targets) {
                *to = listed.sets[y as usize];
            }
        }
        let mut fill = adj_start.clone();
        for (x, &a) in complements.sets.iter().enumerate() {
            for &y in &found[1][found_start[1][x] as usize..found_start[1][x + 1] as usize] {
                let from = complements.sets[y as usize] as usize;
                adj[fill[from] as usize] = a;
                fill[from] += 1;
            }
        }
        drop(found);
        debug_assert!(
            (0..n).all(|i| adj[adj_start[i] as usize..adj_start[i + 1] as usize].is_sorted())
        );
        let [_, complements] = kinds;
        Ok(Self {
            sets,
            adj_start,
            adj,
            local,
            complements: complements.sets,
            words,
            rows,
        })
    }

    fn kept(&self, u: usize) -> &[u32] {
        &self.adj[self.adj_start[u] as usize..self.adj_start[u + 1] as usize]
    }

    /// Listed set `u`'s row of complements.
    fn row(&self, u: usize) -> &[u64] {
        let x = self.local[u] as usize;
        &self.rows[x * self.words..(x + 1) * self.words]
    }

    /// A maximum matching of the bipartite graph with left and right vertices the
    /// sets and an edge `i → j` for each `S_i ⊊ S_j`: each left vertex's partner
    /// and each right vertex's, [`NONE`] when unmatched.
    ///
    /// A greedy matching starts it, which leaves the phases little to augment.
    /// Each phase finds the shortest augmenting paths' layers by breadth-first
    /// search from the free left vertices, over unmatched edges to the right and
    /// matched edges back, then vertex-disjoint shortest augmenting paths along the
    /// layers by depth-first search. A right vertex matters once per phase to
    /// each: the breadth-first search reaches its partner the first time, and the
    /// paths use it at most once. So a listed set's row is read a word at a time
    /// against the complements not yet reached, or not yet used and on the next
    /// layer.
    fn hopcroft_karp(&self) -> (Vec<u32>, Vec<u32>) {
        let n = self.sets.len();
        let complement = self.sets.complement;
        let words = self.words;
        let (mut left, mut right) = (vec![NONE; n], vec![NONE; n]);
        let mut dist = vec![NONE; n];
        let (mut next, mut cursor) = (vec![0u32; n], vec![0u32; n]);
        let (mut via, mut spent) = (vec![NONE; n], vec![false; n]);
        let (mut queue, mut stack) = (Vec::with_capacity(n), Vec::new());
        // The complements, by their index among them, not yet reached by a
        // phase's breadth-first search, then not yet used by its paths.
        let (mut open, mut by_layer) = (vec![0u64; words], Vec::new());
        let fresh = |open: &mut [u64]| every(open, self.complements.len());
        // A greedy matching first: each set, smallest first, takes its first free
        // successor, so the phases only augment what it missed.
        fresh(&mut open);
        for u in 0..n {
            let mut found = self
                .kept(u)
                .iter()
                .copied()
                .find(|&v| right[v as usize] == NONE);
            if found.is_none() && !complement[u] {
                found = self
                    .row(u)
                    .iter()
                    .zip(&open)
                    .enumerate()
                    .find_map(|(k, (&r, &o))| {
                        let bits = r & o;
                        (bits != 0).then(|| 64 * k + bits.trailing_zeros() as usize)
                    })
                    .map(|at| self.complements[at]);
            }
            if let Some(v) = found {
                left[u] = v;
                right[v as usize] = u as u32;
                if complement[v as usize] {
                    let at = self.local[v as usize];
                    open[at as usize / 64] &= !(1 << (at % 64));
                }
            }
        }
        loop {
            queue.clear();
            for u in 0..n {
                dist[u] = if left[u] == NONE {
                    queue.push(u as u32);
                    0
                } else {
                    NONE
                };
            }
            fresh(&mut open);
            // The last layer, where a free right vertex is reached.
            let mut limit = NONE;
            let mut head = 0;
            while head < queue.len() {
                let u = queue[head] as usize;
                head += 1;
                if dist[u] >= limit {
                    break;
                }
                let mut reach = |v: u32, dist: &mut [u32], queue: &mut Vec<u32>| {
                    let w = right[v as usize];
                    if w == NONE {
                        limit = dist[u];
                    } else if dist[w as usize] == NONE {
                        dist[w as usize] = dist[u] + 1;
                        queue.push(w);
                    }
                };
                for &v in self.kept(u) {
                    if complement[v as usize] {
                        let at = self.local[v as usize];
                        open[at as usize / 64] &= !(1 << (at % 64));
                    }
                    reach(v, &mut dist, &mut queue);
                }
                if !complement[u] {
                    for (k, (o, &r)) in open.iter_mut().zip(self.row(u)).enumerate() {
                        let mut new = *o & r;
                        *o &= !new;
                        while new != 0 {
                            let at = 64 * k + new.trailing_zeros() as usize;
                            new &= new - 1;
                            reach(self.complements[at], &mut dist, &mut queue);
                        }
                    }
                }
            }
            if limit == NONE {
                return (left, right);
            }

            next.copy_from_slice(&self.adj_start[..n]);
            cursor.fill(0);
            spent.fill(false);
            fresh(&mut open);
            // The complements by the layer they lead on to: matched ones by their
            // partners' layer, free ones past the last. A listed set at layer l
            // reads layer l + 1 with its row, unless the layers would take more
            // words than the rows, when its row is read whole.
            let layers = limit as usize + 2;
            let layered = layers * words <= self.rows.len().max(words);
            if layered {
                by_layer.clear();
                by_layer.resize(layers * words, 0);
                for (at, &v) in self.complements.iter().enumerate() {
                    let l = match right[v as usize] {
                        NONE => limit as usize + 1,
                        w => dist[w as usize] as usize,
                    };
                    if l < layers {
                        by_layer[l * words + at / 64] |= 1 << (at % 64);
                    }
                }
            }
            for root in 0..n {
                if left[root] != NONE || dist[root] != 0 {
                    continue;
                }
                stack.clear();
                stack.push(root as u32);
                while let Some(&u) = stack.last() {
                    let u = u as usize;
                    let du = dist[u];
                    // Whether right vertex v leads on from u along the layers.
                    let leads = |v: u32| match right[v as usize] {
                        NONE => du == limit,
                        w => du < limit && dist[w as usize] == du + 1,
                    };
                    let mut found = NONE;
                    let end = self.adj_start[u + 1];
                    while next[u] < end {
                        let v = self.adj[next[u] as usize];
                        next[u] += 1;
                        if !spent[v as usize] && leads(v) {
                            found = v;
                            break;
                        }
                    }
                    if found == NONE && !complement[u] {
                        let row = self.row(u);
                        let layer = if layered {
                            let l = du as usize + 1;
                            &by_layer[l * words..(l + 1) * words]
                        } else {
                            row
                        };
                        let mut at = cursor[u] as usize;
                        while at < self.complements.len() {
                            let k = at / 64;
                            let bits = row[k] & layer[k] & open[k] & (u64::MAX << (at % 64));
                            if bits == 0 {
                                at = 64 * (k + 1);
                                continue;
                            }
                            at = 64 * k + bits.trailing_zeros() as usize;
                            let v = self.complements[at];
                            at += 1;
                            if leads(v) {
                                found = v;
                                break;
                            }
                        }
                        cursor[u] = at as u32;
                    }
                    if found == NONE {
                        dist[u] = NONE;
                        stack.pop();
                        continue;
                    }
                    spent[found as usize] = true;
                    if complement[found as usize] {
                        let at = self.local[found as usize];
                        open[at as usize / 64] &= !(1 << (at % 64));
                    }
                    via[u] = found;
                    let w = right[found as usize];
                    if w == NONE {
                        for &u in &stack {
                            let v = via[u as usize];
                            left[u as usize] = v;
                            right[v as usize] = u;
                        }
                        break;
                    }
                    stack.push(w);
                }
            }
        }
    }
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
        starts: Vec<u32>,
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
                starts: vec![0],
                codes: Vec::new(),
                complement: Vec::new(),
            };
            for &set in &family.bits {
                let complement = set & 1 != 0;
                let list = if complement { !set } else { set };
                family.codes.extend(
                    (0..domain)
                        .filter(|&c| list >> c & 1 != 0)
                        .map(|c| c * scale),
                );
                family.starts.push(family.codes.len() as u32);
                family.complement.push(complement);
            }
            family
        }

        fn sets(&self, domain: u32) -> Sets<'_> {
            Sets {
                domain: domain as usize,
                starts: &self.starts,
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
            let cover = cover(&family.sets(domain * scale), u64::MAX).unwrap();
            assert_eq!(cover.chains(), width(sets), "round {round}: {sets:?}");
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
                starts: vec![0],
                codes: Vec::new(),
                complement: Vec::new(),
            };
            for (complement, list) in &sets {
                family.codes.extend_from_slice(list);
                family.starts.push(family.codes.len() as u32);
                family.complement.push(*complement);
            }
            let cover = cover(&family.sets(domain), u64::MAX).unwrap();
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
            assert_eq!(
                cover.chains(),
                full.len() - matching(&full),
                "round {round}"
            );
            for c in 0..cover.chains() {
                for pair in cover.chain(c).windows(2) {
                    let (a, b) = (&full[pair[0] as usize], &full[pair[1] as usize]);
                    assert!(a.len() < b.len() && a.iter().all(|x| b.contains(x)));
                }
            }
        }
    }

    #[test]
    fn the_inclusion_graph_stops_at_its_budget() {
        // ∅ ⊊ {1} ⊊ {1, 2}: three edges.
        let family = Family::new(3, 1, vec![0, 2, 6]);
        assert_eq!(cover(&family.sets(3), 24).unwrap().chains(), 1);
        assert!(matches!(
            cover(&family.sets(3), 11),
            Err(LoadError::Limit(_))
        ));
    }

    #[test]
    fn differences_of_sorted_lists() {
        let d = |a: &[u32], b: &[u32]| difference(a, b).collect::<Vec<_>>();
        assert_eq!(d(&[1, 3, 5, 9], &[0, 3, 4, 9, 10]), [1, 5]);
        assert_eq!(d(&[2], &[]), [2]);
        assert!(d(&[], &[1]).is_empty());
    }
}
