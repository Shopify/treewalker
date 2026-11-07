//! Layout schema: array fields live here, so rank-table and leaf-width changes
//! have one serialization site each. Bump the format version when changing them.
use super::exceptions::ExceptionSlots;
use super::{
    BinMap, CategoryBins, CategoryCodes, ChainPoint, FeatureBins, Layout, LayoutNode, Need,
};
use crate::layout_file::codec::tagged;
use crate::layout_file::codec::{fields, invalid};
use crate::layout_file::validate::{check, partition, span};
use crate::model::VaryingPredicate;
use std::io;

mod sides;

fields!(CategoryCodes { start, len, small });
fields!(CategoryBins {
    codes,
    slot,
    slots,
    offsets,
    held
});
fields!(
    FeatureBins {
        feature,
        thresholds,
        slot,
        categories
    },
    33
);
fields!(BinMap {
    f32,
    n_features,
    n_slots,
    thresholds,
    books,
    defaults,
    offsets,
    exception_slots,
    exception_words,
    shifts,
    n_categorical,
    constant,
    varying
});
fields!(Layout {
    nodes,
    bins,
    leaves,
    varying_share,
    chain_sets,
    chains
});
fields!(Need { leaves, slots }, 16);
// Width is explicit on disk, not inferred from Rust's enum representation.
tagged!(ExceptionSlots { 2 => Narrow(value), 4 => Wide(value) });

impl CategoryCodes {
    fn validate_extent(self, bins: &BinMap) -> io::Result<()> {
        span(
            self.start as usize,
            self.len as usize,
            bins.books.len(),
            "codebook range",
        )?;
        check(
            self.len <= u32::MAX - super::FIRST_CATEGORY,
            "code domain overflow",
        )?;
        if let Some(at) = self.small {
            span(at as usize, 65, bins.books.len(), "small code table")?;
        }
        Ok(())
    }

    pub(crate) fn validate_layout(self, bins: &BinMap) -> io::Result<()> {
        check(
            self.book(&bins.books).windows(2).all(|w| w[0] < w[1]),
            "sorted codebook",
        )?;
        if let Some(at) = self.small {
            check(
                bins.books[at as usize..at as usize + 65]
                    .iter()
                    .all(|&code| (code as usize) < self.domain()),
                "code table domain",
            )?;
        }
        Ok(())
    }
}

impl CategoryBins {
    fn validate_extent(self, bins: &BinMap) -> io::Result<()> {
        self.codes.validate_extent(bins)?;
        span(self.slot, self.slots, bins.n_slots, "categorical slots")?;
        check(
            self.slot > 0 && self.held < bins.n_categorical,
            "categorical slot/held index",
        )?;
        span(
            self.offsets,
            self.codes.domain() + 1,
            bins.offsets.len(),
            "category exception bounds",
        )
    }
}

impl BinMap {
    pub(crate) fn validate_layout(
        &self,
        config: &crate::WalkerConfig,
        f32: bool,
    ) -> io::Result<ValidatedBins<'_>> {
        check(
            self.n_features == config.n_features() && self.f32 == f32,
            "bin precision/features",
        )?;
        check((1..=super::MAX_SLOTS).contains(&self.n_slots), "slot count")?;
        check(
            self.defaults.len() == self.n_slots && self.defaults[super::DUMMY_SLOT] == u32::MAX,
            "dummy/default bins",
        )?;
        check(
            self.n_categorical <= self.constant.len() + self.varying.len(),
            "held code count",
        )?;
        check(
            self.offsets.windows(2).all(|w| w[0] <= w[1]),
            "monotonic exception offsets",
        )?;
        check(
            self.offsets
                .iter()
                .all(|&o| (o as usize) <= self.exception_words.len()),
            "exception offsets",
        )?;
        check(
            match &self.exception_slots {
                ExceptionSlots::Narrow(slots) => slots.len() == self.exception_words.len(),
                ExceptionSlots::Wide(slots) => slots.len() == self.exception_words.len(),
            },
            "exception slots/words lengths",
        )?;
        // Claim ownership before scanning any feature's tables. Otherwise many
        // small feature records could alias one large book or exception table.
        let mut writable: Vec<_> = std::iter::once(0..1).collect(); // dummy
        let (mut thresholds, mut books, mut offsets, mut exceptions) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut owners = Vec::new();
        for (features, varying) in [(&self.constant, false), (&self.varying, true)] {
            check(
                features.windows(2).all(|w| w[0].feature < w[1].feature),
                "sorted bin features",
            )?;
            for b in features {
                check(
                    b.feature < self.n_features && config.is_varying(b.feature) == varying,
                    "bin feature role",
                )?;
                check(
                    b.thresholds.start <= b.thresholds.end
                        && b.thresholds.end <= self.thresholds.len(),
                    "bin thresholds",
                )?;
                thresholds.push(b.thresholds.clone());
                if !b.thresholds.is_empty() {
                    span(b.slot, 4, self.n_slots, "numeric slots")?;
                    check(b.slot > 0, "numeric dummy collision")?;
                    writable.push(b.slot..b.slot + 4);
                    owners.push(SlotOwner {
                        slots: b.slot..b.slot + 4,
                        feature: b,
                        varying,
                        categorical: false,
                    });
                }
                if let Some(c) = b.categories {
                    c.validate_extent(self)?;
                    writable.push(c.slot..c.slot + c.slots());
                    writable.push(self.n_slots + c.held..self.n_slots + c.held + 1);
                    owners.push(SlotOwner {
                        slots: c.slot..c.slot + c.slots(),
                        feature: b,
                        varying,
                        categorical: true,
                    });
                    books.push(
                        c.codes.start as usize..c.codes.start as usize + c.codes.len as usize,
                    );
                    if let Some(at) = c.codes.small {
                        books.push(at as usize..at as usize + 65);
                    }
                    offsets.push(c.offsets..c.offsets + c.codes.domain() + 1);
                    exceptions.push(
                        self.offsets[c.offsets] as usize
                            ..self.offsets[c.offsets + c.codes.domain()] as usize,
                    );
                    check(
                        self.exception_range(c, super::NO_CATEGORY).is_empty(),
                        "code zero exceptions",
                    )?;
                }
            }
        }
        partition(
            &mut writable,
            self.n_slots + self.n_categorical,
            "writable slot ownership",
        )?;
        partition(
            &mut thresholds,
            self.thresholds.len(),
            "threshold ownership",
        )?;
        partition(&mut books, self.books.len(), "codebook ownership")?;
        partition(&mut offsets, self.offsets.len(), "offset table ownership")?;
        partition(
            &mut exceptions,
            self.exception_words.len(),
            "exception table ownership",
        )?;
        let wide = self.constant.iter().chain(&self.varying).any(|b| {
            b.categories
                .is_some_and(|c| c.slots() > super::exceptions::NARROW_SLOTS)
        });
        check(
            matches!(self.exception_slots, ExceptionSlots::Wide(_)) == wide,
            "canonical exception slot width",
        )?;
        for b in self.constant.iter().chain(&self.varying) {
            let thresholds = &self.thresholds[b.thresholds.clone()];
            check(
                thresholds.iter().all(|t| !t.is_nan())
                    && thresholds.windows(2).all(|w| w[0] < w[1]),
                "sorted thresholds",
            )?;
            if let Some(c) = b.categories {
                c.codes.validate_layout(self)?;
                for code in 0..c.codes.domain() {
                    let mut previous = None;
                    for i in self.exception_range(c, code as u32) {
                        let slot = self.exception_slots.at(i);
                        check(
                            slot < c.slots() && previous.is_none_or(|p| p < slot),
                            "relative exception slots",
                        )?;
                        previous = Some(slot);
                    }
                }
            }
        }
        owners.retain(|o| !o.slots.is_empty());
        owners.sort_unstable_by_key(|o| o.slots.start);
        Ok(ValidatedBins { bins: self, owners })
    }
}

struct SlotOwner<'a> {
    slots: std::ops::Range<usize>,
    feature: &'a FeatureBins,
    varying: bool,
    categorical: bool,
}

/// Disjoint slot owners, indexed once rather than searched per node by feature.
pub(crate) struct ValidatedBins<'a> {
    bins: &'a BinMap,
    owners: Vec<SlotOwner<'a>>,
}

impl ValidatedBins<'_> {
    #[cfg(any(test, feature = "research"))]
    pub(crate) fn feature(&self, slot: usize) -> io::Result<usize> {
        Ok(self.owner(slot)?.feature.feature)
    }

    fn owner(&self, slot: usize) -> io::Result<&SlotOwner<'_>> {
        let at = self.owners.partition_point(|o| o.slots.start <= slot);
        let owner = at
            .checked_sub(1)
            .and_then(|i| self.owners.get(i))
            .ok_or_else(|| invalid("unowned split slot"))?;
        check(owner.slots.contains(&slot), "unowned split slot")?;
        Ok(owner)
    }

    /// A split's owner determines whether a mask is required. Numeric masks must
    /// name the exact threshold/default/side encoding; categorical points are
    /// reconstructed from the constant encoding for the mask-plan rebuild.
    pub(crate) fn split(
        &self,
        node: LayoutNode,
        predicates: &[VaryingPredicate],
    ) -> io::Result<Option<(usize, ChainPoint)>> {
        let owner = self.owner(node.slot())?;
        let mask = ((node.w1 >> super::MASK_SHIFT) & super::MASK_FIELD) as usize;
        check(owner.varying == (mask != 0), "split mask ownership")?;
        if mask == 0 {
            return Ok(None);
        }
        let id = (mask - 1) / 2;
        let predicate = predicates
            .get(id)
            .ok_or_else(|| invalid("split predicate"))?;
        check(
            predicate.feature() as usize == owner.feature.feature,
            "split predicate feature",
        )?;
        let invert = mask & 1 == 0;
        if owner.categorical {
            check(
                !matches!(predicate, VaryingPredicate::Num { .. }),
                "categorical split predicate",
            )?;
            Ok(Some((
                id,
                ChainPoint {
                    slot: node.slot() as u32,
                    shift: ((node.w0 >> 32) & 31) as u32,
                    bound: node.payload(),
                    invert,
                },
            )))
        } else {
            let VaryingPredicate::Num {
                threshold,
                default_left,
                ..
            } = predicate
            else {
                return Err(invalid("numeric split predicate"));
            };
            let thresholds = &self.bins.thresholds[owner.feature.thresholds.clone()];
            let k = thresholds.partition_point(|&t| t < threshold.0);
            check(
                k < thresholds.len() && thresholds[k] <= threshold.0,
                "numeric predicate threshold",
            )?;
            let slot = owner.feature.slot + usize::from(!default_left) + 2 * usize::from(invert);
            let bound = if invert {
                u32::MAX - 1 - k as u32 + super::BOUND_BIAS
            } else {
                k as u32 + super::BOUND_BIAS
            };
            check(
                node.w0 == (slot as u64) << super::SLOT_SHIFT | u64::from(bound),
                "numeric predicate encoding",
            )?;
            Ok(Some((id, ChainPoint::NONE)))
        }
    }

    pub(crate) const fn records(&self) -> usize {
        self.bins.books.len()
            + self.bins.thresholds.len()
            + self.bins.offsets.len()
            + self.bins.exception_words.len()
            + self.bins.n_slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Forest;
    use crate::layout_file::codec::{Codec, Decoder};
    use crate::layout_file::testsupport as support;

    #[test]
    fn compact_model_with_many_numeric_features_round_trips() {
        let features = 256;
        let leaf = r#"{"num_nodes": 1, "has_categorical_split": false, "nodes": [{"node_id": 0, "leaf_value": 1.0}]}"#.to_owned();
        let mut forest = support::forest(features, false, &[leaf]);
        let bins = &mut std::sync::Arc::get_mut(&mut forest.model)
            .unwrap()
            .layout
            .bins;
        // Unused numeric tables are legal and do not change this leaf-only
        // forest. Their fieldwise encoding is smaller than their Rust records.
        for feature in 0..features {
            bins.constant.push(FeatureBins {
                feature,
                thresholds: feature..feature + 1,
                slot: 1 + 4 * feature,
                categories: None,
            });
            bins.thresholds.push(0.5);
        }
        bins.defaults.resize(1 + 4 * features, 0);
        bins.n_slots = bins.defaults.len();
        let file = support::File::new();
        forest.save_layout(&file.0).unwrap();
        // Fieldwise encoding can be smaller than in-memory FeatureBins records.
        assert!(
            (std::fs::metadata(&file.0).unwrap().len() as usize)
                < features * size_of::<FeatureBins>()
        );
        let loaded = Forest::load_layout(&file.0).unwrap();
        let rows = vec![0.25; features * 4];
        let (mut a, mut b) = ([0.0f64; 4], [0.0f64; 4]);
        forest.predictor().predict_group(&rows, &mut a);
        loaded.predictor().predict_group(&rows, &mut b);
        assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
    }

    #[test]
    fn checksummed_slot_overlap_is_rejected() {
        let mut forest = support::numeric(2, true);
        let b = &mut std::sync::Arc::get_mut(&mut forest.model)
            .unwrap()
            .layout
            .bins;
        b.varying[1].slot = b.varying[0].slot;
        support::rejects(&forest, "slot ownership");

        let mut forest = support::categorical(2);
        let b = &mut std::sync::Arc::get_mut(&mut forest.model)
            .unwrap()
            .layout
            .bins;
        b.varying[1].categories.as_mut().unwrap().slot = b.varying[0].categories.unwrap().slot;
        support::rejects(&forest, "slot ownership");

        let mut forest = support::categorical(2);
        let b = &mut std::sync::Arc::get_mut(&mut forest.model)
            .unwrap()
            .layout
            .bins;
        b.varying[1].categories.as_mut().unwrap().held = b.varying[0].categories.unwrap().held;
        support::rejects(&forest, "slot ownership");

        let tree = crate::load::json::stump(
            r#""node_type": "numerical_test_node", "comparison_op": "<=", "threshold": 0.5"#,
            true,
            (3, 1),
        );
        let mut forest = support::forest(1, true, &[tree, support::category_tree(&[1, 2], (3, 1))]);
        let b = &mut std::sync::Arc::get_mut(&mut forest.model)
            .unwrap()
            .layout
            .bins;
        b.varying[0].categories.as_mut().unwrap().slot = b.varying[0].slot;
        support::rejects(&forest, "slot ownership");
    }

    #[test]
    fn checksummed_code_zero_exceptions_are_rejected() {
        let mut forest = support::categorical(1);
        let b = &mut std::sync::Arc::get_mut(&mut forest.model)
            .unwrap()
            .layout
            .bins;
        let c = b.varying[0].categories.unwrap();
        assert!(b.offsets[c.offsets + 2] > b.offsets[c.offsets]);
        b.offsets[c.offsets + 1] = b.offsets[c.offsets + 2];
        support::rejects(&forest, "code zero exceptions");
    }

    #[test]
    fn checksummed_table_aliasing_is_rejected_before_scanning() {
        let mutations: &[fn(&mut BinMap)] = &[
            |b| {
                b.varying[1].categories.as_mut().unwrap().codes.start =
                    b.varying[0].categories.unwrap().codes.start;
            },
            |b| {
                b.varying[1].categories.as_mut().unwrap().codes.small =
                    b.varying[0].categories.unwrap().codes.small;
            },
            |b| {
                b.varying[1].categories.as_mut().unwrap().offsets =
                    b.varying[0].categories.unwrap().offsets;
            },
        ];
        for mutate in mutations {
            let mut forest = support::categorical(2);
            mutate(
                &mut std::sync::Arc::get_mut(&mut forest.model)
                    .unwrap()
                    .layout
                    .bins,
            );
            support::rejects(&forest, "ownership");
        }
        let mut forest = support::numeric(2, true);
        let b = &mut std::sync::Arc::get_mut(&mut forest.model)
            .unwrap()
            .layout
            .bins;
        b.varying[1].thresholds = b.varying[0].thresholds.clone();
        support::rejects(&forest, "threshold ownership");
    }

    #[test]
    fn rejects_bin_domains_and_feature_metadata() {
        let corruptions: &[fn(&mut BinMap)] = &[
            |b| b.n_slots = 0,
            |b| b.n_features = usize::MAX,
            |b| b.varying[0].feature = usize::MAX,
            |b| b.varying[0].thresholds.end = usize::MAX,
            |b| b.varying[0].categories.as_mut().unwrap().held = usize::MAX,
            |b| b.varying[0].categories.as_mut().unwrap().codes.start = u32::MAX,
            |b| b.varying[0].categories.as_mut().unwrap().codes.small = Some(u32::MAX),
            |b| b.varying[0].categories.as_mut().unwrap().offsets = usize::MAX,
            |b| b.exception_words = super::super::array(Vec::new()),
            |b| b.exception_slots = ExceptionSlots::Narrow(super::super::array(vec![])),
            |b| b.exception_slots.set(0, u32::from(u16::MAX)),
        ];
        for corrupt in corruptions {
            let mut forest = support::fixture();
            let config = forest.config().clone();
            let bins = &mut std::sync::Arc::get_mut(&mut forest.model)
                .unwrap()
                .layout
                .bins;
            bins.validate_layout(&config, false).unwrap();
            corrupt(bins);
            assert!(bins.validate_layout(&config, false).is_err());
        }
    }

    #[test]
    fn slot_widths_round_trip_and_reject_relative_overflow() {
        for wide in [false, true] {
            let mut forest = support::categorical(1);
            let config = forest.config().clone();
            let mut bins = forest.model.layout.bins.clone();
            let count = bins.exception_words.len();
            if wide {
                // Keep the fixture's live slot; add unused categorical slots to
                // exercise the canonical wide representation without huge trees.
                let model = std::sync::Arc::get_mut(&mut forest.model).unwrap();
                let c = bins.varying[0].categories.as_mut().unwrap();
                let extra = super::super::exceptions::NARROW_SLOTS + 1 - c.slots;
                c.slots += extra;
                model.masks.features[0].categories = Some(*c);
                bins.n_slots += extra;
                bins.defaults.resize(bins.n_slots, 0);
                bins.exception_slots = ExceptionSlots::Wide(super::super::array(
                    (0..count)
                        .map(|i| bins.exception_slots.at(i) as u32)
                        .collect(),
                ));
            }
            let mut w = crate::layout_file::codec::Encoder::new(Vec::new());
            bins.put(&mut w).unwrap();
            let mut r = crate::layout_file::codec::Decoder {
                inner: w.inner.as_slice(),
                at: 0,
                total: w.inner.len(),
                checksum: crate::layout_file::codec::Checksum::default(),
                #[cfg(all(unix, target_pointer_width = "64"))]
                mapping: None,
            };
            let mut loaded = BinMap::get(&mut r).unwrap();
            assert_eq!(r.at, r.total);
            assert_eq!(
                matches!(loaded.exception_slots, ExceptionSlots::Wide(_)),
                wide
            );
            loaded.validate_layout(&config, false).unwrap();
            let mut expected = bins.new_bins();
            let mut actual = loaded.new_bins();
            for x in [0.0, 1.0, 99.0, f64::NAN] {
                bins.fill_with::<false>(&[x], &mut expected, true);
                loaded.fill_with::<false>(&[x], &mut actual, true);
                assert_eq!(actual, expected);
            }
            let mut wrong = loaded.clone();
            wrong.exception_slots = if wide {
                ExceptionSlots::Narrow(super::super::array(
                    (0..count)
                        .map(|i| loaded.exception_slots.at(i) as u16)
                        .collect(),
                ))
            } else {
                ExceptionSlots::Wide(super::super::array(
                    (0..count)
                        .map(|i| loaded.exception_slots.at(i) as u32)
                        .collect(),
                ))
            };
            assert!(
                wrong
                    .validate_layout(&config, false)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("canonical exception slot width")
            );
            #[cfg(all(unix, target_pointer_width = "64", not(miri)))]
            {
                let mut forest = forest;
                let path = support::File::new();
                std::sync::Arc::get_mut(&mut forest.model)
                    .unwrap()
                    .layout
                    .bins = loaded.clone();
                forest.save_layout(&path.0).unwrap();
                let copied = Forest::load_layout(&path.0).unwrap();
                // SAFETY: the test exclusively owns the file and does not write
                // or truncate it until every mapped forest/predictor has dropped.
                let mapped = unsafe { Forest::map_layout(&path.0).unwrap() };
                let layout = &mapped.model.layout;
                assert!(layout.nodes.is_mapped());
                assert!(layout.bins.exception_words.is_mapped());
                match &layout.bins.exception_slots {
                    ExceptionSlots::Narrow(s) => {
                        assert!(!wide);
                        assert!(s.is_mapped());
                    }
                    ExceptionSlots::Wide(s) => {
                        assert!(wide);
                        assert!(s.is_mapped());
                    }
                }
                assert!(matches!(mapped.model.plan, crate::exact::Plan::Fixed(_)));
                assert!(layout.leaves.is_empty());
                let rows = [0.0, 1.0, 99.0, f64::NAN];
                let (mut a, mut b) = ([0.0f64; 4], [0.0f64; 4]);
                forest.predictor().predict_group(&rows, &mut a);
                copied.predictor().predict_group(&rows, &mut b);
                assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
                mapped.predictor().predict_group(&rows, &mut b);
                assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
            }
            loaded
                .exception_slots
                .set(0, if wide { u32::MAX } else { u32::from(u16::MAX) });
            assert!(loaded.validate_layout(&config, false).is_err());
        }
    }

    #[test]
    fn rejects_unknown_exception_width() {
        for tag in [0, 1, 8, 255] {
            let bytes = [tag];
            let mut r = Decoder {
                inner: bytes.as_slice(),
                at: 0,
                total: 1,
                checksum: crate::layout_file::codec::Checksum::default(),
                #[cfg(all(unix, target_pointer_width = "64"))]
                mapping: None,
            };
            assert!(ExceptionSlots::get(&mut r).is_err());
        }
    }
}
