//! The research ablations: the runtime ones a [`ResearchPredictor`](super::ResearchPredictor) runs with, and the load-time ones
//! that change the layout, with their stable IDs.

use std::io::Read;
use std::path::Path;

use crate::config::{Choices, LoadOptions, WalkerConfig};
use crate::forest::Forest;
use crate::predict::Options;
use crate::rules::{MAX_LANES, Prefix};
use crate::{LoadError, ModelFormat};

/// Runtime ablations of the predict path: each field disables one optimization or
/// forces one choice. The default is production.
///
/// Load-time ablations are a [`LoadAblation`] instead, because they change the
/// model's layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ablation {
    /// Precompute the varying masks by evaluating every predicate on every row,
    /// instead of the sorted-threshold sweep.
    pub disable_predicate_sweep: bool,
    /// Compute no varying masks: each varying event scans its rows for its
    /// predicate's mask instead. Overrides `disable_predicate_sweep`.
    pub disable_varying_precompute: bool,
    /// Scan monotonic features like the others, without stopping at the first row
    /// past the threshold. Effective only with `disable_varying_precompute`.
    pub disable_monotonic: bool,
    /// Pass on the sides of varying events that have no rows, as items the walk then
    /// steps through.
    pub disable_unsplit: bool,
    /// Walk lanes, 1 to 12, for every walk; `None` for the rule's lanes for each
    /// walk. Forced lanes prefetch one item ahead every round, or not at all with
    /// `disable_restart_prefetch`.
    pub lanes: Option<usize>,
    /// Walk single rows like groups: stop at varying splits and read their
    /// precomputed one-row masks, instead of stepping through the splits as
    /// constants.
    pub disable_g1_constant: bool,
    /// Run every group at the widest mask width the configuration allows, as the 2.x
    /// kernel did, instead of the narrowest that holds it.
    pub disable_width_dispatch: bool,
    /// Do not prefetch the roots of a lane's coming items.
    pub disable_restart_prefetch: bool,
}

/// What [`Ablation::with`] accepts as `lanes=N`.
pub(super) fn valid_lanes(n: usize) -> bool {
    (1..=MAX_LANES).contains(&n)
}

/// A boolean field of [`Ablation`]: its name and the field.
type Flag = (&'static str, fn(&mut Ablation) -> &mut bool);

/// The boolean fields of [`Ablation`], in declaration order.
const FLAGS: [Flag; 7] = [
    ("disable_predicate_sweep", |a| {
        &mut a.disable_predicate_sweep
    }),
    ("disable_varying_precompute", |a| {
        &mut a.disable_varying_precompute
    }),
    ("disable_monotonic", |a| &mut a.disable_monotonic),
    ("disable_unsplit", |a| &mut a.disable_unsplit),
    ("disable_g1_constant", |a| &mut a.disable_g1_constant),
    ("disable_width_dispatch", |a| &mut a.disable_width_dispatch),
    ("disable_restart_prefetch", |a| {
        &mut a.disable_restart_prefetch
    }),
];

impl Ablation {
    /// Production with the one field named by `token` set: a flag's name or
    /// `lanes=N`.
    ///
    /// # Errors
    ///
    /// If `token` names no field or an invalid value.
    pub fn with(mut self, token: &str) -> Result<Self, String> {
        match token.split_once('=') {
            Some(("lanes", n)) => {
                let n: usize = n.parse().map_err(|_| format!("invalid lanes in {token}"))?;
                if !valid_lanes(n) {
                    return Err(format!("lanes must be 1 to {MAX_LANES}, got {n}"));
                }
                self.lanes = Some(n);
            }
            _ => {
                *self
                    .flag(token)
                    .ok_or_else(|| format!("unknown runtime ablation {token}"))? = true;
            }
        }
        Ok(self)
    }

    /// The boolean field named `name`.
    fn flag(&mut self, name: &str) -> Option<&mut bool> {
        let &(_, field) = FLAGS.iter().find(|(n, _)| *n == name)?;
        Some(field(self))
    }

    /// The tokens [`Self::with`] takes for each field that differs from production,
    /// sorted.
    #[must_use]
    pub fn tokens(&self) -> Vec<String> {
        let mut this = *self;
        let mut tokens: Vec<String> = FLAGS
            .iter()
            .filter(|(_, field)| *field(&mut this))
            .map(|&(name, _)| name.to_owned())
            .collect();
        if let Some(n) = self.lanes {
            tokens.push(format!("lanes={n}"));
        }
        tokens.sort();
        tokens
    }

    /// A stable ID: `all-on`, or the [`Self::tokens`] joined by `+`.
    #[must_use]
    pub fn id(&self) -> String {
        let tokens = self.tokens();
        if tokens.is_empty() {
            "all-on".into()
        } else {
            tokens.join("+")
        }
    }

    /// The ablation of an [`Self::id`].
    ///
    /// # Errors
    ///
    /// If a token is invalid.
    pub fn from_id(id: &str) -> Result<Self, String> {
        if id == "all-on" {
            return Ok(Self::default());
        }
        id.split('+').try_fold(Self::default(), Self::with)
    }

    /// The kernel's options for this ablation.
    pub(super) const fn options(&self) -> Options {
        Options {
            g1_constant: !self.disable_g1_constant,
            width_dispatch: !self.disable_width_dispatch,
            precompute: !self.disable_varying_precompute,
            sweep: !self.disable_predicate_sweep,
            monotonic: !self.disable_monotonic,
            unsplit: !self.disable_unsplit,
            lanes: self.lanes,
            distance: if self.disable_restart_prefetch {
                Some(0)
            } else {
                None
            },
            rate: None,
        }
    }
}

/// Load-time ablations: each field turns off one of the layout choices the library
/// makes, or forces one. The default is production.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LoadAblation {
    /// Keep the trees in the model's order instead of sorting trees with similar
    /// heavy paths next to each other. Prefix groups are runs of consecutive trees,
    /// so they shrink too.
    pub disable_tree_ordering: bool,
    /// Store every categorical list, even when an identical one is already pooled.
    pub disable_bitset_intern: bool,
    /// Group every run of trees sharing exactly this many heavy-path constant splits,
    /// instead of the rule's groups and depths; 0 disables prefix starts.
    pub prefix_depth: Option<usize>,
    /// Give every varying node its own predicate instead of sharing one between
    /// identical splits. This multiplies the precompute work.
    pub disable_predicate_dedup: bool,
}

impl LoadAblation {
    /// The load's choices for this ablation.
    const fn choices(&self) -> Choices {
        Choices {
            tree_ordering: !self.disable_tree_ordering,
            bitset_intern: !self.disable_bitset_intern,
            prefix: match self.prefix_depth {
                Some(k) => Prefix::Depth(k),
                None => Prefix::Rule,
            },
            predicate_dedup: !self.disable_predicate_dedup,
        }
    }
}

impl Forest {
    /// [`Forest::from_reader`] with load-time ablations.
    pub fn from_reader_ablated(
        reader: impl Read,
        format: ModelFormat,
        config: WalkerConfig,
        options: &LoadOptions,
        ablation: &LoadAblation,
    ) -> Result<Self, LoadError> {
        Self::from_reader_with(reader, format, config, options, &ablation.choices())
    }

    /// [`Forest::from_bytes`] with load-time ablations.
    pub fn from_bytes_ablated(
        bytes: &[u8],
        format: ModelFormat,
        config: WalkerConfig,
        options: &LoadOptions,
        ablation: &LoadAblation,
    ) -> Result<Self, LoadError> {
        Self::from_bytes_with(bytes, format, config, options, &ablation.choices())
    }

    /// [`Forest::load_with`] with load-time ablations.
    pub fn load_ablated(
        model_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        options: &LoadOptions,
        ablation: &LoadAblation,
    ) -> Result<Self, LoadError> {
        let config = WalkerConfig::from_file_with(config_path, options)?;
        let format = ModelFormat::from_path(model_path.as_ref())?;
        Self::from_file_with(
            std::fs::File::open(model_path)?,
            format,
            config,
            options,
            &ablation.choices(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip() {
        assert_eq!(Ablation::default().id(), "all-on");
        let a = Ablation {
            disable_unsplit: true,
            disable_monotonic: true,
            lanes: Some(4),
            ..Default::default()
        };
        assert_eq!(a.id(), "disable_monotonic+disable_unsplit+lanes=4");
        assert_eq!(Ablation::from_id(&a.id()), Ok(a));
        assert_eq!(Ablation::from_id("all-on"), Ok(Ablation::default()));
        assert!(Ablation::from_id("disable_exact_sums").is_err());
        assert!(Ablation::from_id("lanes=13").is_err());
        assert!(Ablation::from_id("distance=3").is_err());
        assert!(Ablation::from_id("rate=2").is_err());
        assert_eq!(Ablation::from_id("lanes=3").unwrap().lanes, Some(3));
        assert!(Ablation::from_id("layout=2e").is_err());
        for (flag, _) in super::FLAGS {
            let a = Ablation::default().with(flag).unwrap();
            assert_eq!(a.id(), flag);
        }
    }
}
