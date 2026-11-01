//! Whether a model packs its categorical chains' rank fields ([`crate::load::layout`],
//! "Packed fields"): when its unpacked bins would leave L1, or when the fill work
//! packing saves a group outweighs the shift its walk then takes at every lane step.
//! A model whose chain covers ran out of steps packs regardless, so that an extra
//! chain costs a field rather than a slot.
//!
//! The lane rule ([`super::lanes()`]) prices a lane step at the larger of its issue
//! bound and its chain, `σ = max(U / W, C / k)` cycles, with `U` the step's uops, `W`
//! the issue width, `C` the dependent chain and `k` the lanes, and chooses lanes that
//! reach the issue bound. With the bins in L1 both walks stay there, so the shift
//! costs `ΔU / W` a lane step ([`super::lanes::shift_uops`]). A fill restores the held
//! code's exceptions and writes the new code's, independent loads and stores in two
//! loops of `U_r` and `U_w` uops an exception ([`FILL_UOPS`]), issue-bound too. So
//! for a group of `S` lane slots whose code change saves `ΔE` exceptions a code, the
//! width cancels:
//!
//! ```text
//! pack  ⇔  S · ΔU  <  (U_r + U_w) · ΔE
//! ```
//!
//! - `S`: the expected lane slots of a group at the configuration's
//!   `max_group_width` ([`super::GroupSums`]), its trees' steps, where rows share the
//!   constant splits' paths and part only at varying ones, and the round in which the
//!   lanes run out of items.
//! - `ΔE`: per categorical feature that a group fills (the constant ones; every one
//!   for single rows), its codes' exceptions less the packed words they fall in,
//!   weighted by the code's exceptions: a code that more chains rank apart is one more
//!   splits name, so more rows carry. On the categorical cell this came to 85% of the
//!   exceptions its rows' codes save; uniform over the codes, whose rare codes are in
//!   one chain each and pack to nothing, to under half.
//!
//! Once the unpacked bins leave L1 the bin's load on every step's chain comes from
//! L2: on x86-64 `C` goes from 25 to 36 cycles and chains take 36 / 8 = 4.5 a step,
//! against 4.17 for the packed walk's issue bound, so packing wins without the fill.
//!
//! The thresholds on `2 ΔE / S` this gives, `2 ΔU / (U_r + U_w)`: x86-64 2 × 2.25 / 13
//! = 0.35, AArch64 2 × 2 / 17 = 0.24. Over the categorical cell's cuts of 1 to 20
//! trees, and the grid's 42 cells of its 30 categorical models of the highest
//! `2 ΔE / S` and 10 spread below, packed against unpacked: on Genoa it leaves the
//! 1-tree cut unpacked (packing 1.000 latency, 1.029 batch) and packs from 2 trees
//! (1.000 and 0.990; 3 trees 0.958), and of the grid's 554 categorical models packs the
//! four of 0.35 to 0.39, one faster (0.988) and three 0.6–1.7% slower, and leaves one
//! 1.4% faster (0.34) unpacked; on the M4 it packs every cut (0.82–0.93) and the 14
//! grid cells above 0.24, all faster (0.895–0.995), and leaves unpacked the cells
//! packing slows (up to 1.14) and eight it speeds by at most 2.6%. Counting the M4's
//! fused compare-and-branch pairs once would put its threshold at 0.33, which leaves
//! the 1-tree cut, 6.8% faster packed, unpacked.
//!
//! Bins within 16 KiB, the assumed 32 KiB of L1 over two threads of a core
//! ([`LEAST_L1`]), stay in L1 on any core with that L1 or more and at most two
//! threads, so the rule probes the caches only for larger ones; the first predictor
//! probes them anyway.
//!
//! The rule reads the bins' size first, then counts `ΔE`, and takes `S` in a pass over
//! the trees that stops past the most at which packing pays: over the grid's 554
//! categorical models on Genoa it took a median 0.7% of their load, at most 8.6%, and
//! at most 1.0 ms.

use super::lanes::shift_uops;
use super::machine::{self, Caches};

/// The fill's uops per exception it restores and writes, in the units of the issue
/// width (x86-64 uops, a macro-fused compare-and-branch once; AArch64 instructions).
/// Counted from the post-LTO assembly of `examples/walk_asm.rs` (rustc 1.97.1), where
/// each loop is one block: the restore loads the slot, checks it against the bins and
/// the defaults, loads the default, stores it and advances, 7 on znver4 (10
/// instructions) and 9 on apple-m4; the write loads the slot, checks it, loads the word,
/// stores it and advances, 6 (8 instructions) and 8.
const FILL_UOPS: [usize; 2] = if cfg!(target_arch = "x86_64") {
    [7, 6]
} else {
    [9, 8]
};

/// The least share of L1 a CPU has on a core with at least the assumed L1
/// ([`machine::ASSUMED`]) and at most two threads.
pub(crate) const LEAST_L1: usize = machine::ASSUMED[0].bytes / 2;

/// [`resident`] on this machine, probing its caches only for bins past [`LEAST_L1`].
pub(crate) fn resident_here(bytes: usize) -> bool {
    bytes <= LEAST_L1 || resident(bytes, machine::caches())
}

/// Whether bins of `bytes` stay in L1: within one CPU's share of it, as the chunk
/// budget takes L2's ([`super::chunk_budget`]).
pub(crate) fn resident(bytes: usize, caches: &Caches) -> bool {
    bytes
        <= caches
            .level(1)
            .map_or(machine::ASSUMED[0].bytes, machine::Cache::per_cpu)
}

/// The most lane slots of a group for which packing pays with the bins in L1, on the
/// walk with varying stops with `varying`, against fills that save `saved` exceptions a
/// code: packing pays below it.
pub(crate) fn max_lane_slots(saved: f64, varying: bool) -> f64 {
    (FILL_UOPS[0] + FILL_UOPS[1]) as f64 * saved / shift_uops(varying)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::machine::{Cache, Source};

    #[test]
    fn bins_stay_in_l1_up_to_one_cpus_share() {
        let caches = |levels: Vec<Cache>| Caches {
            line: 64,
            levels,
            source: Source::Assumed,
        };
        // 48 KiB shared by two logical CPUs: 24 KiB each.
        let smt = caches(vec![Cache {
            level: 1,
            bytes: 48 << 10,
            shared_by: 2,
        }]);
        assert!(resident(24 << 10, &smt));
        assert!(!resident((24 << 10) + 1, &smt));
        // Without an L1, the assumed one.
        let none = caches(Vec::new());
        assert!(resident(machine::ASSUMED[0].bytes, &none));
        assert!(!resident(machine::ASSUMED[0].bytes + 1, &none));
        // Within the least share, whatever this machine's caches.
        assert_eq!(LEAST_L1, 16 << 10);
        assert!(resident_here(LEAST_L1));
    }

    #[test]
    fn packing_pays_below_the_fills_saving_over_the_shift() {
        // A lane step's shift against an exception's restore and write: on x86-64 2.25
        // uops (the shift's 2, and a spilled value's store and reload over 8 lanes)
        // against 7 and 6, on AArch64 2 instructions against 9 and 8.
        let (shift, fill) = if cfg!(target_arch = "x86_64") {
            (2.25, 13.0)
        } else {
            (2.0, 17.0)
        };
        for varying in [false, true] {
            assert!((shift_uops(varying) - shift).abs() < 1e-12);
            assert!((max_lane_slots(31.0, varying) - fill * 31.0 / shift).abs() < 1e-9);
            assert!(max_lane_slots(0.0, varying).abs() < 1e-12);
        }
    }
}
