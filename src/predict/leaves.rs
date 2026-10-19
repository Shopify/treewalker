//! The leaf pass: each leaf entry's value added to its rows' runs in the difference
//! arrays.

use super::isa::Isa;
use super::mask::Mask;
use super::resolve::Carrier;
use super::walk::Item;
use crate::exact::RunSink;

/// [`leaf_pass`] at level `X` ([`Isa`]): where the build dispatches, out of line,
/// with the body compiled inside the level's trampoline; elsewhere inlined.
#[cfg_attr(not(dispatch), inline(always))]
pub(super) fn leaf_pass_at<
    X: Isa,
    M: Mask,
    C: Carrier<M>,
    S: RunSink + ?Sized,
    const STATS: bool,
>(
    entries: &[Item<C>],
    full: &[Item<C>],
    arena: &[M],
    u: usize,
    values: &[S::Value],
    even: &mut S,
    odd: &mut S,
) -> u64 {
    if X::NATIVE {
        return leaf_pass::<M, C, S, STATS>(entries, full, arena, u, values, even, odd);
    }
    X::with7(
        #[inline(always)]
        |entries, full, arena, u, values, even, odd| {
            leaf_pass::<M, C, S, STATS>(entries, full, arena, u, values, even, odd)
        },
        entries,
        full,
        arena,
        u,
        values,
        even,
        odd,
    )
}

/// Add each leaf entry's value to its rows: each of `full`, the first pass's entries,
/// which carry all rows, to a running sum in registers, and each of `entries` to its
/// runs, alternating between the sinks `even` and `odd`; with `STATS`, returns the
/// number of runs, a full entry being one. Arena masks have `u` units of `M`.
#[cfg_attr(dispatch, inline(always))]
pub(super) fn leaf_pass<M: Mask, C: Carrier<M>, S: RunSink + ?Sized, const STATS: bool>(
    entries: &[Item<C>],
    full: &[Item<C>],
    arena: &[M],
    u: usize,
    values: &[S::Value],
    even: &mut S,
    odd: &mut S,
) -> u64 {
    if STATS {
        let mut even = CountRuns {
            sink: even,
            runs: 0,
        };
        let mut odd = CountRuns { sink: odd, runs: 0 };
        add_entries(entries, full, arena, u, values, &mut even, &mut odd);
        return even.runs + odd.runs;
    }
    add_entries(entries, full, arena, u, values, even, odd);
    0
}

/// The loop of [`leaf_pass`]. Full entries, the leaves of trees whose path has no
/// varying split before the leaf, add to one sum rather than
/// to the first entry, where each add would wait on the previous one's store. The
/// others go two at a time.
#[inline(always)]
fn add_entries<M: Mask, C: Carrier<M>, S: RunSink + ?Sized>(
    entries: &[Item<C>],
    full: &[Item<C>],
    arena: &[M],
    u: usize,
    values: &[S::Value],
    even: &mut S,
    odd: &mut S,
) {
    let mut sum = S::Sum::default();
    even.add_all_of(&mut sum, full.iter().map(|e| values[e.node as usize]));
    even.flush(sum);
    let mut pairs = entries.chunks_exact(2);
    for pair in &mut pairs {
        let (a, b) = (&pair[0], &pair[1]);
        let (va, vb) = (values[a.node as usize], values[b.node as usize]);
        // SAFETY: a leaf entry's handle is a slot its passes wrote.
        unsafe {
            a.carry.add_runs(arena.as_ptr(), va, even, u);
            b.carry.add_runs(arena.as_ptr(), vb, odd, u);
        }
    }
    for e in pairs.remainder() {
        let v = values[e.node as usize];
        // SAFETY: as above.
        unsafe { e.carry.add_runs(arena.as_ptr(), v, even, u) };
    }
}

/// A sink that counts the runs added to the sink it wraps.
struct CountRuns<'a, S: ?Sized> {
    sink: &'a mut S,
    runs: u64,
}

impl<S: RunSink + ?Sized> RunSink for CountRuns<'_, S> {
    type Value = S::Value;
    type Sum = S::Sum;

    fn entries(&self) -> usize {
        self.sink.entries()
    }

    unsafe fn add(&mut self, at: usize, v: S::Value) {
        self.runs += 1;
        // SAFETY: the caller's bound on at holds for the wrapped sink.
        unsafe { self.sink.add(at, v) };
    }

    unsafe fn sub(&mut self, at: usize, v: S::Value) {
        // SAFETY: as above.
        unsafe { self.sink.sub(at, v) };
    }

    fn add_all_of(&mut self, sum: &mut S::Sum, values: impl Iterator<Item = S::Value>) {
        let runs = &mut self.runs;
        self.sink.add_all_of(sum, values.inspect(|_| *runs += 1));
    }

    fn flush(&mut self, sum: S::Sum) {
        self.sink.flush(sum);
    }
}
