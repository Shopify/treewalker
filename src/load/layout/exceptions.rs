//! Feature-relative slots of the rank tables' exceptions.

use crate::LoadError;

const NARROW_SLOTS: usize = 1 << u16::BITS;

/// Narrow until a feature needs more slots than u16 can name, then wide throughout.
#[derive(Clone, Debug)]
pub(super) enum ExceptionSlots {
    Narrow(Vec<u16>),
    Wide(Vec<u32>),
}

impl Default for ExceptionSlots {
    fn default() -> Self {
        Self::Narrow(Vec::new())
    }
}

impl ExceptionSlots {
    pub(super) const fn bytes(&self) -> usize {
        match self {
            Self::Narrow(slots) => 2 * slots.len(),
            Self::Wide(slots) => 4 * slots.len(),
        }
    }

    /// Bytes after growing to `len` for a feature of `feature_slots` slots, including any
    /// widening of the earlier features' exceptions.
    pub(super) const fn bytes_after(&self, len: usize, feature_slots: usize) -> u64 {
        let wide = matches!(self, Self::Wide(_)) || feature_slots > NARROW_SLOTS;
        len as u64 * if wide { 4 } else { 2 }
    }

    /// Grow after the byte cap is checked, widening the preceding slots at most once.
    pub(super) fn resize(&mut self, len: usize, feature_slots: usize) -> Result<(), LoadError> {
        match self {
            Self::Narrow(slots) if feature_slots > NARROW_SLOTS => {
                let mut wide = Vec::new();
                LoadError::reserve(wide.try_reserve_exact(len))?;
                wide.extend(slots.iter().map(|&s| u32::from(s)));
                wide.resize(len, 0);
                *self = Self::Wide(wide);
            }
            Self::Narrow(slots) => {
                LoadError::reserve(slots.try_reserve_exact(len - slots.len()))?;
                slots.resize(len, 0);
            }
            Self::Wide(slots) => {
                LoadError::reserve(slots.try_reserve_exact(len - slots.len()))?;
                slots.resize(len, 0);
            }
        }
        Ok(())
    }

    /// Write a feature-relative slot, whose feature's extent selected the width.
    pub(super) fn set(&mut self, at: usize, slot: u32) {
        match self {
            Self::Narrow(slots) => {
                debug_assert!(u16::try_from(slot).is_ok());
                slots[at] = slot as u16;
            }
            Self::Wide(slots) => slots[at] = slot,
        }
    }

    /// A feature-relative slot, for readers outside the fill's monomorphic loops.
    #[inline]
    pub(super) fn at(&self, at: usize) -> usize {
        match self {
            Self::Narrow(slots) => usize::from(slots[at]),
            Self::Wide(slots) => slots[at] as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widening_preserves_earlier_features_and_stays_wide() {
        let mut slots = ExceptionSlots::default();
        slots.resize(2, 65_536).unwrap();
        slots.set(0, 7);
        slots.set(1, 65_535);
        assert!(matches!(slots, ExceptionSlots::Narrow(_)));
        assert_eq!(slots.bytes(), 4);
        assert_eq!(slots.bytes_after(4, 65_537), 16);
        slots.resize(4, 65_537).unwrap();
        slots.set(2, 65_536);
        slots.set(3, 0);
        assert_eq!(
            (0..4).map(|i| slots.at(i)).collect::<Vec<_>>(),
            [7, 65_535, 65_536, 0]
        );
        assert_eq!(slots.bytes(), 16);
        slots.resize(5, 1).unwrap();
        assert!(matches!(slots, ExceptionSlots::Wide(_)));
        assert_eq!(slots.bytes(), 20);
    }
}
