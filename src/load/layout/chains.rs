//! A categorical feature's heavy sets as chains: sequences of strictly nested sets
//! that hold every set once.
//!
//! A field holds one rank per row, and a split on it tests `rank <= k`, so the splits
//! one field serves have nested sublevel sets: a chain. A chain of m sets takes a
//! field of ⌈log2(m + 1)⌉ bits ([`super`], "Packed fields"), so an extra chain costs a
//! field, not a slot, and the cover need not be a minimum one. Each kind is covered
//! apart, greedily: a listed set is inside another exactly when its list is, and a
//! complement is inside another exactly when its list holds the other's, so both
//! kinds are chains of lists, a complement's read backwards. Code 0 is in every
//! complement and no listed set, so every chain is one kind's sets.

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

/// Steps a list may take, per code and one more, to find the chain it extends.
const WORK_PER_CODE: u64 = 64;

/// A chain cover of `sets`, each kind apart, greedily.
///
/// A kind's lists are taken in increasing length, and each extends the chain whose
/// last list is the longest one strictly inside it, else starts a chain. Each chain's
/// last list is indexed under its rarest code, the one the fewest lists of the kind
/// name, and its entry goes stale when the chain grows: a list inside A has that code
/// in A, so scanning the entries of A's codes finds every candidate once, checked
/// against a stamp of A's codes. Entries arrive in increasing length, so each code's
/// are scanned newest first, and the scan stops at the first match or at entries no
/// longer than the best so far. A chain whose last list is empty is a candidate for
/// every list. A list's steps, entries visited and codes checked, stop at
/// [`WORK_PER_CODE`] per code and one more; past them it extends the best chain found
/// so far or starts one, so the work is linear in the lists.
pub(crate) fn cover(sets: &Sets<'_>) -> Cover {
    let n = sets.len();
    assert!(u32::try_from(n).is_ok_and(|n| n < u32::MAX));
    let mut cover = Cover {
        order: Vec::with_capacity(n),
        starts: Vec::new(),
        at: vec![(0, 0); n],
    };
    let mut freq = vec![0u32; sets.domain];
    let mut stamp = vec![u32::MAX; sets.domain];
    let mut index: Vec<Vec<(u32, u32)>> = vec![Vec::new(); sets.domain];
    for complement in [false, true] {
        let mut ids: Vec<u32> = (0..n as u32)
            .filter(|&i| sets.complement[i as usize] == complement)
            .collect();
        ids.sort_by_key(|&i| sets.list(i as usize).len());
        freq.fill(0);
        for &i in &ids {
            for &c in sets.list(i as usize) {
                freq[c as usize] += 1;
            }
        }
        index.iter_mut().for_each(Vec::clear);
        // Each chain's lists in increasing length, and its last list.
        let mut chains: Vec<Vec<u32>> = Vec::new();
        let mut last: Vec<u32> = Vec::new();
        let mut empty: Option<u32> = None;
        for (stamp_of, &a) in ids.iter().enumerate() {
            let stamp_of = stamp_of as u32;
            let la = sets.list(a as usize);
            for &c in la {
                stamp[c as usize] = stamp_of;
            }
            let limit = WORK_PER_CODE * (la.len() as u64 + 1);
            let mut work = 0;
            let mut best: Option<(u32, usize)> = empty.map(|c| (c, 0));
            'codes: for &c in la {
                for &(chain, t) in index[c as usize].iter().rev() {
                    work += 1;
                    if work > limit {
                        break 'codes;
                    }
                    if last[chain as usize] != t {
                        continue;
                    }
                    let lt = sets.list(t as usize);
                    if lt.len() >= la.len() {
                        continue;
                    }
                    if best.is_some_and(|(_, len)| lt.len() <= len) {
                        break;
                    }
                    work += lt.len() as u64;
                    if lt.iter().all(|&x| stamp[x as usize] == stamp_of) {
                        best = Some((chain, lt.len()));
                        break;
                    }
                }
            }
            let chain = if let Some((chain, _)) = best {
                if empty == Some(chain) {
                    empty = None;
                }
                chains[chain as usize].push(a);
                last[chain as usize] = a;
                chain
            } else {
                chains.push(vec![a]);
                last.push(a);
                (chains.len() - 1) as u32
            };
            match la.iter().min_by_key(|&&c| (freq[c as usize], c)) {
                Some(&rarest) => index[rarest as usize].push((chain, a)),
                None => empty = Some(chain),
            }
        }
        for mut chain in chains {
            // A complement's list shrinks as the set grows.
            if complement {
                chain.reverse();
            }
            let c = cover.starts.len() as u32;
            cover.starts.push(cover.order.len() as u32);
            for (k, &i) in chain.iter().enumerate() {
                cover.at[i as usize] = (c, k as u32);
            }
            cover.order.extend_from_slice(&chain);
        }
    }
    cover.starts.push(cover.order.len() as u32);
    debug_assert_eq!(cover.order.len(), n, "every set is in one chain");
    cover
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
    fn covers_are_chains_of_one_kind() {
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
            let cover = cover(&family.sets(domain * scale));
            // At least the largest antichain; within a kind, at most the sets.
            assert!(cover.chains() >= width(sets), "round {round}: {sets:?}");
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
                    assert_eq!(a & 1, b & 1, "one kind");
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
    fn covers_of_many_sets_are_chains() {
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
            let cover = cover(&family.sets(domain));
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
            // At least a minimum cover's chains, and at most the sets.
            assert!(
                cover.chains() >= full.len() - matching(&full),
                "round {round}"
            );
            assert!(cover.chains() <= full.len());
            for c in 0..cover.chains() {
                for pair in cover.chain(c).windows(2) {
                    let (a, b) = (&full[pair[0] as usize], &full[pair[1] as usize]);
                    assert!(a.len() < b.len() && a.iter().all(|x| b.contains(x)));
                }
            }
        }
    }

    #[test]
    fn nested_lists_make_one_chain_per_kind() {
        // ∅ ⊊ {1} ⊊ {1, 2}, and the complements of {1, 2} ⊋ {2}: one chain each.
        let family = Family::new(3, 1, vec![0, 2, 6, 1 | !6 & 7, 1 | !4 & 7]);
        let cover = cover(&family.sets(3));
        assert_eq!(cover.chains(), 2);
    }

    #[test]
    fn differences_of_sorted_lists() {
        let d = |a: &[u32], b: &[u32]| difference(a, b).collect::<Vec<_>>();
        assert_eq!(d(&[1, 3, 5, 9], &[0, 3, 4, 9, 10]), [1, 5]);
        assert_eq!(d(&[2], &[]), [2]);
        assert!(d(&[], &[1]).is_empty());
    }
}
