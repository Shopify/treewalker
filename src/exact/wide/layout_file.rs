use super::{Wide, WideLeaf};
use crate::layout_file::codec::fields;

impl Wide {
    pub(crate) fn validate_layout(self, leaves: &[WideLeaf]) -> std::io::Result<()> {
        use crate::layout_file::validate::check;
        check(
            (3..=super::MAX_LIMBS).contains(&self.limbs) && (-1074..=1023).contains(&self.low),
            "wide plan bounds",
        )?;
        check(
            leaves.iter().all(|v| {
                v.first as usize <= self.limbs - super::LEAF_DIGITS
                    && v.digits
                        .iter()
                        .all(|d| d.unsigned_abs() < 1 << super::DIGIT_BITS)
            }),
            "wide leaf digit bounds",
        )
    }
}

fields!(Wide { limbs, low });
fields!(WideLeaf { first, digits }, 16);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_wide_layouts_and_digits() {
        let valid = Wide { limbs: 3, low: 0 };
        valid
            .validate_layout(&[WideLeaf {
                first: 0,
                digits: [0; 3],
            }])
            .unwrap();
        for wide in [
            Wide { limbs: 2, low: 0 },
            Wide {
                limbs: usize::MAX,
                low: 0,
            },
            Wide {
                limbs: 3,
                low: i32::MIN,
            },
        ] {
            assert!(wide.validate_layout(&[]).is_err());
        }
        assert!(
            valid
                .validate_layout(&[WideLeaf {
                    first: 1,
                    digits: [0; 3]
                }])
                .is_err()
        );
        assert!(
            valid
                .validate_layout(&[WideLeaf {
                    first: 0,
                    digits: [i32::MIN; 3]
                }])
                .is_err()
        );
    }
}
