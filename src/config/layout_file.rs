//! Configuration schema, separate from the JSON import schema.
use super::{FeatureSet, WalkerConfig};
use crate::layout_file::codec::fields;

impl WalkerConfig {
    pub(crate) fn validate_layout(&self) -> std::io::Result<()> {
        use crate::layout_file::validate::check;
        let nf = self.n_features;
        check(
            (1..=i32::MAX as usize).contains(&nf),
            "config feature count",
        )?;
        check(
            (1..=1 << 20).contains(&self.max_group_width),
            "layout group-width limit",
        )?;
        for set in [&self.varying, &self.increasing, &self.decreasing] {
            check(!set.all || set.indices.is_empty(), "canonical feature set")?;
            check(
                set.indices.iter().all(|&f| (f as usize) < nf),
                "config feature index",
            )?;
            check(
                set.indices.windows(2).all(|w| w[0] < w[1]),
                "sorted feature set",
            )?;
        }
        check(
            self.increasing.first_common(&self.decreasing, nf).is_none(),
            "overlapping monotonic features",
        )?;
        for set in [&self.increasing, &self.decreasing] {
            check(!set.all || self.varying.all, "monotonic implies varying")?;
            check(
                set.indices.iter().all(|&f| self.is_varying(f as usize)),
                "monotonic implies varying",
            )?;
        }
        Ok(())
    }
}

fields!(FeatureSet { all, indices });
fields!(WalkerConfig {
    n_features,
    max_group_width,
    varying,
    increasing,
    decreasing
});

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_feature_sets_and_width() {
        let corruptions: &[fn(&mut WalkerConfig)] = &[
            |c| c.n_features = 0,
            |c| c.max_group_width = 0,
            |c| c.varying.indices = vec![1, 1],
            |c| c.varying.indices = vec![2],
            |c| c.increasing = c.decreasing.clone(),
            |c| c.varying.indices.clear(),
        ];
        for corrupt in corruptions {
            let mut c = WalkerConfig::builder(2)
                .max_group_width(2)
                .decreasing([1])
                .varying([])
                .build()
                .unwrap();
            c.validate_layout().unwrap();
            corrupt(&mut c);
            assert!(c.validate_layout().is_err());
        }
    }
}
