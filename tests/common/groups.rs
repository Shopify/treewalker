//! Feature roles, the configuration they make, and groups of generated rows that keep
//! them. `WalkerConfig` comes from the including module: the library's own in its unit
//! tests, the public one in the integration tests.

use std::cmp::Ordering;

use super::WalkerConfig;
use super::model::{self, Feature, Precision};
use super::rng::Rng;

/// A feature's role in every group. `tests/vectors.rs` hashes the discriminants into
/// its fingerprints, so the order of the variants stays.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Constant,
    Varying,
    Increasing,
    Decreasing,
}

/// The configuration of `n_features` features with `roles`, for groups of up to
/// `max_width` rows.
pub fn config(n_features: usize, max_width: usize, roles: &[Role]) -> WalkerConfig {
    let with = |role| (0..n_features).filter(move |&f| roles[f] == role);
    WalkerConfig::builder(n_features)
        .max_group_width(max_width)
        .varying(with(Role::Varying))
        .increasing(with(Role::Increasing))
        .decreasing(with(Role::Decreasing))
        .build()
        .expect("a valid configuration")
}

/// The offsets of consecutive groups of `sizes` rows.
pub fn offsets(sizes: &[usize]) -> Vec<usize> {
    std::iter::once(0)
        .chain(sizes.iter().scan(0, |end, &n| {
            *end += n;
            Some(*end)
        }))
        .collect()
}

/// One group's rows, row-major: each constant feature equal in every row, each
/// monotonic feature free of NaN and sorted by `order` in its direction.
pub fn group(
    rng: &mut Rng,
    features: &[Feature],
    roles: &[Role],
    precision: Precision,
    n: usize,
    mut order: impl FnMut(&f64, &f64) -> Ordering,
) -> Vec<f64> {
    let columns: Vec<Vec<f64>> = features
        .iter()
        .zip(roles)
        .map(|(f, &role)| match role {
            Role::Constant => vec![model::input(rng, f, precision, true); n],
            Role::Varying => (0..n)
                .map(|_| model::input(rng, f, precision, true))
                .collect(),
            Role::Increasing | Role::Decreasing => {
                let mut column: Vec<f64> = (0..n)
                    .map(|_| model::input(rng, f, precision, false))
                    .collect();
                column.sort_by(&mut order);
                if role == Role::Decreasing {
                    column.reverse();
                }
                column
            }
        })
        .collect();
    (0..n)
        .flat_map(|r| columns.iter().map(move |c| c[r]))
        .collect()
}
