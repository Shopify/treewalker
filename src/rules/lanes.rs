//! The walk's lanes: how many trees a walk steps at a time, from the step's dependent
//! chain, the core's issue width, and the registers where a spilled lane lengthens
//! that chain.

use super::machine::GENERAL_REGISTERS;

/// The most lanes the walk is written for: what 30 registers (AArch64 without a
/// reserved platform register) leave after the fewest invariants and temporaries
/// below, (30 − 5 − 1) / 2.
pub(crate) const MAX_LANES: usize = 12;

/// Registers the walk's loop holds besides its lanes and a step's temporaries: the
/// node pool, the bins, the items, the words written per step and the dummy's link;
/// with varying stops the stops written per step. On
/// x86-64 the 64-bit stop or leaf mask too, since its instructions take at most 32-bit
/// immediates (AArch64 encodes it as a logical immediate). On AArch64 a walk with
/// varying stops also holds its lane count, which the cursor advance selects
/// (`csel w, wzr, wK`, see [`one_instruction_advance`]).
///
/// They count against the registers even though a spilled invariant would only be
/// reloaded, off the lanes' dependent chains: LLVM's allocator weighs uses, and an
/// invariant is used by every lane's step while a lane's node or cursor is used by
/// one, so when the registers run short it spills lanes first, which then cost a store
/// and a load per step (x86-64 at 8 lanes stores 11 to 46 values per iteration; an
/// attempt to force the invariants to memory with volatile reads still stored 3 to 7).
const fn invariants(varying: bool) -> usize {
    let x86 = cfg!(target_arch = "x86_64");
    5 + varying as usize + x86 as usize + (varying && !x86) as usize
}

/// Registers live within one lane's step, which the lanes reuse in turn. Read from
/// the post-LTO assembly (rustc 1.97.1):
///
/// - AArch64 (apple-m4): word 0 and word 1, the bin, the heavy candidate, the next
///   item, and in a walk without varying stops the leaf bit, which `ands` keeps for
///   the shifted add of the advance: 5, or 4 with varying stops, whose `tst` sets
///   only the flags.
/// - x86-64 (emeraldrapids) folds the bin and item loads into the compare and the
///   conditional move: word 0, word 1 and the candidate (or the stop flag), 3.
///
/// Checked by building every walk at one lane more than the AArch64 rule's
/// (`walk_spills.py --lanes`): with these counts, the rule's lane count is the largest
/// without stack stores for every walk.
const fn temporaries(varying: bool) -> usize {
    if cfg!(target_arch = "x86_64") {
        3
    } else {
        5 - varying as usize
    }
}

/// The lane values (nodes and cursors) that do not fit the registers at `k` lanes:
/// each lane holds two, beside the invariants and a step's temporaries.
/// `experiments/scripts/walk_spills.py` expects a production walk loop to store to the
/// stack exactly when this is above zero.
pub(crate) const fn spilled(k: usize, varying: bool) -> usize {
    (2 * k + invariants(varying) + temporaries(varying)).saturating_sub(GENERAL_REGISTERS)
}

/// Whether advancing a lane's cursor by `k` items when it stops is one instruction
/// beyond the stop test. The advance is on the lane's dependent chain (the next
/// item's address), and every step pays it, so a lane count whose advance takes more
/// slows every lane.
///
/// - Without varying stops, the stop is the leaf bit at position 32, and for a power of
///   two it shifts straight to `k × stop`: AArch64 folds the shift into the add (`add
///   c, c, bit, lsr #(32 − log2 k)`), x86-64 takes a shift and an add as for one lane.
///   Any other `k` needs the bit shifted to 1 and a multiply: three instructions on
///   AArch64 for 9 (`lsr`, `add w, w, w, lsl #3`, `add`), which made 9 lanes 2.5–4%
///   slower than 8 on single rows of 500-tree models on the M4, and on x86-64 a `lea`
///   more for 3, 5 or 9.
/// - With varying stops, the stop is any of word 1's bits 32–62, so no shift gives
///   `k × stop`. AArch64 selects `k` from a register or zero (`tst`, `csel w, wzr, wK`,
///   `add`), the same for every `k`; x86-64 sets the flag in a register and adds it
///   scaled (`setne`, `lea (c, s, k)`), which `lea` scales by 1, 2, 4 or 8 only.
const fn one_instruction_advance(k: usize, varying: bool) -> bool {
    (varying && !cfg!(target_arch = "x86_64")) || k.is_power_of_two()
}

/// x86-64: load-to-use latencies of L1 and L2, in cycles: 5 and 16 on Golden Cove's
/// server and Raptor Lake cores (Sapphire and Emerald Rapids; Zen 4: 4 and 14). The
/// probe measured L2 at 3.19 L1 latencies on Emerald Rapids.
const X86_L1_CYCLES: usize = 5;
const X86_L2_CYCLES: usize = 16;

/// x86-64: one lane step's dependent chain, in cycles: the node's load, at the pool
/// plus the lane's byte offset ([`crate::load::layout`]), its slot (`shr`), the
/// bin's load folded into the compare (`cmp`), the heavy select (`cmovae`) and the
/// stop select (`cmovne`), four one-cycle operations and two loads; the path through
/// the stop to the next item is as long. Read from the post-LTO assembly of
/// `examples/walk_asm.rs` (emeraldrapids and x86-64-v4, rustc 1.97.1), alike for every
/// walk. The bins are small and stay in L1; the node comes from L2, the level a pool
/// comes from when it does not fit L1 (a pool in L1 is a model of a few dozen small
/// trees, whose walk is a small part of its time): 16 + 5 + 4 = 25. With node indices
/// the address took a `shl` more, and the heavy select was `cmova`, whose two flags
/// take 2 cycles on Intel.
const X86_CHAIN_CYCLES: usize = X86_L2_CYCLES + X86_L1_CYCLES + 4;

/// x86-64: the uops the core renames per cycle, which bound the walk's throughput
/// whatever its lanes: 6 on Golden Cove and its successors (Intel's optimization
/// manual, 248966-048) and on Zen 3 and 4.
const X86_ISSUE_WIDTH: usize = 6;

/// x86-64: the uops of one lane step with every lane in registers (2 lanes, a
/// prefetch every round): instructions, less a macro-fused compare-and-branch per
/// lane, plus one for each load-op with an index register (unlaminated) and for each
/// `cmova`/`cmovbe` (2 uops on Intel; the walks have none). Counted from the same
/// assembly: 21 for the walk without varying stops, 20 and 23 for those with them over
/// 8- and 16-byte items; the fewest, 20, which reach the issue bound last. With node
/// indices they were 24, 24 and 27.5: a `shl` for the node's address and one for the
/// prefetch's, and `cmova`'s or `cmovbe`'s second uop.
const X86_STEP_UOPS: usize = 20;

/// x86-64: uops a lane value that does not fit the registers adds per round, its store
/// and its reload. The assembly adds more per lane step (at 8 lanes 3 to 6 against
/// 2 lanes, invariant reloads and register moves included): more uops only reach the
/// issue bound sooner.
const X86_SPILL_UOPS: usize = 2;

/// x86-64: whether `k` lanes reach the issue bound, `chain / k <= uops per step /
/// width`, with the uops of the values [`spilled`] at `k`.
const fn x86_issue_bound(k: usize, varying: bool) -> bool {
    X86_CHAIN_CYCLES * X86_ISSUE_WIDTH <= k * X86_STEP_UOPS + X86_SPILL_UOPS * spilled(k, varying)
}

/// A lane count fixed at build time by `TREEWALKER_LANES`, for the spill check of
/// `experiments/scripts/walk_spills.py`; production builds leave it unset.
const FORCED_LANES: Option<usize> = match option_env!("TREEWALKER_LANES") {
    Some(s) => Some(parse(s)),
    None => None,
};

const fn parse(s: &str) -> usize {
    let b = s.as_bytes();
    let mut n = 0;
    let mut i = 0;
    while i < b.len() {
        assert!(b[i].is_ascii_digit(), "TREEWALKER_LANES is a number");
        n = 10 * n + (b[i] - b'0') as usize;
        i += 1;
    }
    assert!(
        n >= 1 && n <= MAX_LANES,
        "TREEWALKER_LANES is 1 to MAX_LANES"
    );
    n
}

/// Walk lanes for a walk stopping at varying splits with `varying`.
///
/// Each lane's step is a dependent chain (the node's load, its bin's load, the compare
/// and two selects), so `k` lanes finish at most `k` steps per chain, and the core's
/// issue width bounds the steps per cycle whatever `k`. The walk wants the fewest lanes
/// that reach that bound, among those whose cursor advance is one instruction, at
/// most [`MAX_LANES`]; what a lane costs beyond the registers decides how it gets there.
///
/// - AArch64 (and the targets with as many registers): a spilled lane stores and
///   reloads its node or cursor every step, and the store-to-load round trip lengthens
///   its chain (5 cycles on Neoverse V2, about 7 on the M4). So the lanes are the most
///   whose state, two registers each (the node and the cursor), fits the registers
///   beside the loop's invariants and a step's temporaries. They reach the issue bound
///   already: a chain of 13 cycles over 15 to 18 uops per step at the M4's 10 per cycle
///   wants 7 to 9 lanes, at Neoverse V2's 6, 4 to 5. On the M4, research lanes 2 to 12
///   over R5's 90 cells put 9 and 8 within 0.8% of the best everywhere (10 lanes 0.992
///   overall, 11 and 12 about 1.02, 8 for both 1.026).
/// - x86-64: 15 registers fit 2 lanes, which are bound by the chain (12.5 cycles per
///   step with nodes in L2, against an issue bound near 3.3). A spilled lane costs uops,
///   a store and a reload, but on Golden Cove and Zen 3 and later no chain latency, since
///   those cores forward a store to a load of the same address with zero latency. So
///   the lanes are the fewest powers of two that reach the issue bound
///   ([`x86_issue_bound`]) with the spilled values' uops counted: 8, `25 × 6 = 150 <=
///   8 × 20 + 2 × 10` (4 give 84). Without zero-latency forwarding a spilled lane's
///   chain is about 30 cycles, and 8 lanes still step every 3.75 cycles against 4
///   lanes' 6.25 to 7.5, so the choice does not depend on it. On Emerald Rapids, with
///   node indices, over R5's 90 cells at both alignments, research lanes=4 took 0.793
///   of 2 lanes' time (lanes=1 1.455); over 4 lanes, 5 to 8 lanes took 0.964, 0.936,
///   0.944 and 0.937. At 8: sessions 0.921, panels 0.93 to 0.94, what-ifs 0.972, grids
///   0.990, and the cells whose pools fit L1 0.98 to 1.00. Single rows are faster at 6
///   (0.841) than at 8 (0.892), but 6 and 8 tie overall, and a power of two keeps the
///   cursor advance one instruction on both walks.
///
/// With 29 registers (AArch64 on Apple) and 30 (Linux):
///
/// | Walk | invariants | temporaries | lanes that fit | lanes |
/// |---|---|---|---|---|
/// | varying | 7 | 4 | 9 | 9 |
/// | constant | 5 | 5 | 9; 10 on Linux | 8 (a power of two) |
///
/// On x86-64 (15 registers, 6 uops renamed per cycle, a chain of 25 cycles):
///
/// | Walk | uops per step | lanes that fit | values spilled at 8 | lanes |
/// |---|---|---|---|---|
/// | varying | 20 | 2 | 11 | 8 (`150 <= 160 + 22`; 4 give 86) |
/// | constant | 20 (21 counted) | 3 | 10 | 8 (`150 <= 160 + 20`; 4 give 84) |
///
/// The counts that changed with byte offsets and the one-flag select, 26 cycles to 25
/// and 24 uops to 20, leave both walks at 8.
///
/// Not capped by the items of a pass. A pass's last round leaves lanes on the dummy:
/// with `n` items of similar length, `⌈n / k⌉ k − n` slots, on average `k / 2n` of
/// the pass, so a lane more or fewer moves the waste by about `1 / 2n`, and only a
/// model of a few dozen trees notices it (50 trees on 8 lanes take 7 rounds, the last
/// 2/8 busy, against 6 on 9: 5% more lane slots and +3% on the support nt50_md4_h1
/// LightGBM single rows, while the XGBoost model's paths filled both alike). Choosing
/// the lanes per model would compile every walk once per lane count.
///
/// On Emerald Rapids the one-lane step is the documented chain: with node indices and a
/// pool in L1, 16.1 cycles per step without the restart prefetch and 17.4 with forced
/// lanes' prefetch, against 16 from uops.info's latencies (`shl` 1, the load 5, `shr`
/// 1, the compare from memory 6, `cmova` 2, `cmovne` 1). The about 20 cycles per step
/// the counted walk took at one lane, which no documented latency explained, came from
/// its counting and that prefetch. Beyond one lane the lanes' steps contend for ports 0
/// and 6, Golden Cove's only ports for shifts, `cmov`, `setcc` and branches, to which a
/// uop is bound when it is allocated. A 4-lane round over a pool in L1 took 21.8 cycles
/// against the 17.4-cycle chain, with 7.25 such uops per step, 5 of them on the chain,
/// and the two ports busy 66% of cycles; at 2 MiB it took 23.1. With byte offsets and
/// the one-flag select a step has 4.3 of them, and the round takes 16.7 cycles against
/// a 14.0-cycle chain. Measured under the performance counters with a benchmark of
/// these loops (instruction-identical with node indices; with byte offsets, the same
/// steps with registers allocated differently), on pools whose splits take the heavy
/// side as real models' do. So 4 lanes do not reach the issue bound there, and 8 stay
/// right: over a pool in L1 a 4-lane slot and an 8-lane one cost about the same (4.17
/// and 4.13 cycles), at 2 MiB 8 lanes cost 11% less (4.19 against 4.73).
///
/// The links are byte offsets on x86-64 and node indices elsewhere
/// ([`crate::load::layout`], "Links"), as each instruction set addresses a 16-byte
/// node. x86-64 scales an index register by at most 8, so a node's index took a `shl`
/// on every step's chain, one of the port 0 and 6 uops above on Golden Cove, and the
/// heavy select for `bin <= k`, `cmova`, reads two flags and takes 2 uops on Intel,
/// where `cmovae` for `bin < k + 1` takes 1. AArch64 scales the index by 16 in the add
/// of the node's address (`add x, nodes, w, uxtw #4`), and `csinc` selects the heavy
/// child, the index plus 1, in one instruction; byte offsets cost it an `add` per lane
/// step for the heavy child's offset, and on Apple a register: at 9 lanes the walk
/// with varying stops kept a lane's cursor on the stack. Over 90 cells at the default
/// alignment and with functions aligned to 64 bytes, the walks with byte offsets and
/// the test `bin < k + 1` took, against node indices and `bin <= k`, 0.961 and 0.959
/// of the time on Emerald Rapids, 0.962 and 0.958 on Ice Lake, 0.959 and 0.961 on
/// Milan, 0.958 and 0.958 on Genoa and 0.960 and 0.965 on Turin, but 1.008 and 1.007
/// on Axion (Neoverse V2) and 1.023 and 1.021 on Altra (Neoverse N1).
///
/// The x86-64 counts are Intel's. AMD's Zen 3, 4 and 5 count the same steps differently,
/// which the rule does not model (AMD's software optimization guides for Zen 3, 56665
/// rev. 3.00, Zen 4, 57647 rev. 1.00, and Zen 5, 58455 rev. 1.00, with their instruction
/// tables):
///
/// - A memory operand with a base and an index register takes 2 macro ops, a load or a
///   store as much as a load-op (56665 and 57647 section 2.3, 58455 appendix A). An
///   8-lane round of the walk with varying stops over 8-byte items, 192 instructions at
///   `znver4`, 64 of them addressed so, is about 248 macro ops.
/// - On Zen 3 and 4 a shift by an immediate runs only on ALU1 and ALU2, and `cmov` and
///   `setcc` only on ALU0 and ALU3: two ALUs each, where Golden Cove issues all three on
///   ports 0 and 6.
/// - On Zen 5 `prefetcht0` is microcoded, and `cmova` takes 2 macro ops where `cmovae`
///   takes 1, as on Intel.
pub(crate) const fn lanes(varying: bool) -> usize {
    if let Some(k) = FORCED_LANES {
        return k;
    }
    if cfg!(target_arch = "x86_64") {
        let mut k = 1;
        while k < MAX_LANES && !(one_instruction_advance(k, varying) && x86_issue_bound(k, varying))
        {
            k += 1;
        }
        return k;
    }
    let held = invariants(varying) + temporaries(varying);
    let fit = GENERAL_REGISTERS.saturating_sub(held) / 2;
    let mut k = if fit > MAX_LANES { MAX_LANES } else { fit };
    while k > 1 && !one_instruction_advance(k, varying) {
        k -= 1;
    }
    if k < 1 { 1 } else { k }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes_follow_the_rule() {
        for varying in [false, true] {
            let k = lanes(varying);
            assert!((1..=MAX_LANES).contains(&k));
            if FORCED_LANES.is_some() {
                continue;
            }
            assert!(one_instruction_advance(k, varying));
            if cfg!(target_arch = "x86_64") {
                // The fewest lanes that reach the issue bound: half as many do not.
                assert!(x86_issue_bound(k, varying) && !x86_issue_bound(k / 2, varying));
            } else {
                assert!(spilled(k, varying) == 0 || k == 1);
            }
        }
        if FORCED_LANES.is_none() && cfg!(all(target_arch = "aarch64", target_vendor = "apple")) {
            assert_eq!((lanes(true), lanes(false)), (9, 8));
        }
        if FORCED_LANES.is_none() && cfg!(target_arch = "x86_64") {
            assert_eq!((lanes(true), lanes(false)), (8, 8));
            assert_eq!(
                (spilled(8, true), spilled(8, false)),
                if cfg!(target_vendor = "apple") {
                    (12, 11)
                } else {
                    (11, 10)
                }
            );
        }
    }
}
