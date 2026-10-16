//! How a group's passes run their walks ([`Walks`]): the lanes and prefetch distance
//! each walk is compiled for.

use super::kernel::Options;
use super::walk::{lanes_of, pad};
use crate::rules::{Distance, pass_distance};

/// How a group's passes run their walks: with the rule's lanes for each walk, or a
/// research ablation's, a prefetch distance from `rules::DISTANCES` and, for the walk
/// without varying stops, a rate from `rules::RATES`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Walks {
    pub(super) lanes: Option<usize>,
    pub(super) distance: usize,
    pub(super) rate: usize,
}

impl Walks {
    /// The walks of `options` with the rule's `distance` and `rate`. Forced lanes
    /// prefetch one item ahead every round, as before the rules, unless the distance
    /// or the rate is forced too.
    pub(super) fn of<const ABLATE: bool>(options: Options, rule: Distance) -> Self {
        if ABLATE {
            let forced = options.lanes.is_some();
            let rate = usize::from(rule.rate);
            Self {
                lanes: options.lanes,
                distance: options
                    .distance
                    .unwrap_or(if forced { 1 } else { rule.items }),
                rate: options.rate.unwrap_or(if forced { 1 } else { rate }),
            }
        } else {
            Self {
                lanes: None,
                distance: rule.items,
                rate: usize::from(rule.rate),
            }
        }
    }

    /// The walks of a pass of `n` items, with the distance of
    /// [`crate::rules::pass_distance`], and the padding such a walk reads past the
    /// items.
    pub(super) fn for_pass<const ABLATE: bool>(self, n: usize, varying: bool) -> (Self, usize) {
        let forced = if ABLATE { self.lanes } else { None };
        let lanes = lanes_of(forced.unwrap_or(0), varying);
        let wanted = match forced {
            Some(_) => self.distance.min(1),
            None => self.distance,
        };
        let distance = pass_distance(wanted, n, lanes);
        (
            Self {
                lanes: self.lanes,
                distance,
                rate: self.rate,
            },
            pad(lanes, distance),
        )
    }
}

/// Run `$call` with `K`, `D` and `R` as consts: the walk's lanes (0 for the rule's),
/// its prefetch distance, rounded up to a compiled one, and its rate (1 without
/// prefetch). Production, without `$ablate`, never instantiates forced lanes.
macro_rules! walk_with {
    (@forced $lanes:ident, $d:ident, $($l:literal)+) => {
        match $lanes {
            $($l => if $d == 0 { go!($l, 0, 1) } else { go!($l, 1, 1) },)+
            _ => unreachable!("lanes are checked at creation"),
        }
    };
    (@rate $r:expr, $dd:literal) => {
        match $r {
            1 => go!(0, $dd, 1),
            2 => go!(0, $dd, 2),
            _ => go!(0, $dd, 4),
        }
    };
    ($ablate:expr, $walks:expr, |$k:ident, $d:ident, $rr:ident| $call:expr) => {{
        macro_rules! go {
            ($kk:literal, $dd:literal, $r:literal) => {{
                const $k: usize = $kk;
                const $d: usize = $dd;
                const $rr: usize = $r;
                $call
            }};
        }
        let w: Walks = $walks;
        match (if $ablate { w.lanes } else { None }, w.distance) {
            (None, 0) => go!(0, 0, 1),
            (None, 1) => walk_with!(@rate w.rate, 1),
            (None, 2) => walk_with!(@rate w.rate, 2),
            (None, 3..=4) => walk_with!(@rate w.rate, 4),
            (None, 5..=8) => walk_with!(@rate w.rate, 8),
            (None, _) => walk_with!(@rate w.rate, 16),
            // Forced lanes, research only: 1 to MAX_LANES, without prefetch or one
            // item ahead every round.
            (Some(lanes), d) => walk_with!(@forced lanes, d, 1 2 3 4 5 6 7 8 9 10 11 12),
        }
    }};
}

pub(super) use walk_with;
