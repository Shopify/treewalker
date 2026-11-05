//! Outputs compared bit for bit.

/// The bits of each value, for `assert_eq!` on whole outputs.
pub fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Panic at the first row whose bits differ.
pub fn same(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths");
    for (r, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.to_bits() == e.to_bits(),
            "{what}: row {r}: {a:e} ({:#018x}) != {e:e} ({:#018x})",
            a.to_bits(),
            e.to_bits()
        );
    }
}
