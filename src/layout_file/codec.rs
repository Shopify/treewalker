//! Fieldwise metadata and aligned, length-prefixed arrays. Never persist pointers,
//! padding, bool representations or Rust enum discriminants.

use crate::exact::Plan;
use crate::load::layout::{MaskChain, MaskFeature, MaskPlan};
use crate::model::{
    Model, Output, Postprocessor, PrefixGroup, Threshold, ThresholdType, Tree, VaryingPredicate,
};
use crate::rules::{ItemLengths, PrefixStats, RuleInputs, Steps};
use std::io::{self, Read, Write};

pub(crate) fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// A streaming checksum, independent of the boundaries of calls to `write`.
/// Detects accidental corruption; not an authentication mechanism.
#[derive(Default)]
pub(crate) struct Checksum {
    state: u64,
    tail: [u8; 8],
    used: usize,
}

impl Checksum {
    const fn word(&mut self, word: u64) {
        self.state = (self.state.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    pub(crate) fn write(&mut self, mut bytes: &[u8]) {
        if self.used != 0 {
            let n = (8 - self.used).min(bytes.len());
            self.tail[self.used..self.used + n].copy_from_slice(&bytes[..n]);
            self.used += n;
            bytes = &bytes[n..];
            if self.used < 8 {
                return;
            }
            self.word(u64::from_le_bytes(self.tail));
            self.used = 0;
        }
        let (words, tail) = bytes.as_chunks::<8>();
        for word in words {
            self.word(u64::from_le_bytes(*word));
        }
        self.tail = [0; 8];
        self.tail[..tail.len()].copy_from_slice(tail);
        self.used = tail.len();
    }

    pub(crate) const fn finish(mut self) -> u64 {
        self.word(u64::from_le_bytes(self.tail));
        self.word(self.used as u64);
        self.state
    }
}

pub(crate) struct Encoder<W> {
    pub(crate) inner: W,
    pub(crate) at: usize,
    pub(crate) checksum: Checksum,
}

impl<W: Write> Encoder<W> {
    pub(crate) fn new(inner: W) -> Self {
        Self {
            inner,
            at: 0,
            checksum: Checksum::default(),
        }
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.inner.write_all(bytes)?;
        self.checksum.write(bytes);
        self.at += bytes.len();
        Ok(())
    }

    pub(crate) fn align(&mut self) -> io::Result<()> {
        let padding = (16 - self.at % 16) % 16;
        self.bytes(&[0; 16][..padding])
    }
}

pub(crate) struct Decoder<R> {
    pub(crate) inner: R,
    pub(crate) at: usize,
    pub(crate) total: usize,
    pub(crate) checksum: Checksum,
}

impl<R: Read> Decoder<R> {
    pub(crate) fn bytes(&mut self, bytes: &mut [u8]) -> io::Result<()> {
        if bytes.len() > self.total.saturating_sub(self.at) {
            return Err(invalid("truncated layout section"));
        }
        self.inner.read_exact(bytes)?;
        self.checksum.write(bytes);
        self.at += bytes.len();
        Ok(())
    }

    pub(crate) fn align(&mut self) -> io::Result<()> {
        let padding = (16 - self.at % 16) % 16;
        let mut bytes = [0; 16];
        self.bytes(&mut bytes[..padding])?;
        if bytes != [0; 16] {
            return Err(invalid("nonzero layout section padding"));
        }
        Ok(())
    }

    pub(crate) fn length<T: Codec>(&mut self) -> io::Result<usize> {
        let len = usize::get(self)?;
        self.align()?;
        // Bound reservations by minimum wire sizes, not in-memory Rust sizes.
        // All fieldwise Vec elements declare their true minimum; POD is exact.
        if len > self.total.saturating_sub(self.at) / T::MIN_BYTES.max(1) || len > 1 << 28 {
            return Err(invalid("layout array exceeds allocation limit"));
        }
        Ok(len)
    }
}

pub(crate) trait Codec: Sized {
    const MIN_BYTES: usize = 1;
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()>;
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self>;

    fn put_slice<W: Write>(values: &[Self], w: &mut Encoder<W>) -> io::Result<()> {
        for value in values {
            value.put(w)?;
        }
        Ok(())
    }

    fn get_vec<R: Read>(len: usize, r: &mut Decoder<R>) -> io::Result<Vec<Self>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| invalid("layout array allocation"))?;
        for _ in 0..len {
            values.push(Self::get(r)?);
        }
        Ok(values)
    }
}

/// A padding-free, pointer-free type accepting every bit pattern.
///
/// # Safety
/// Implementors must be `Copy`, have no padding, and accept every input byte
/// pattern, including all-zero bytes. Only explicit numeric/layout types qualify.
pub(crate) unsafe trait Pod: Copy {}

pub(crate) const fn pod_bytes<T: Pod>(values: &[T]) -> &[u8] {
    // SAFETY: Pod excludes padding and all values are initialized.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

fn pod_vec<T: Pod, R: Read>(len: usize, r: &mut Decoder<R>) -> io::Result<Vec<T>> {
    if len
        .checked_mul(size_of::<T>())
        .is_none_or(|n| n > r.total.saturating_sub(r.at))
    {
        return Err(invalid("layout array extent"));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| invalid("layout array allocation"))?;
    // SAFETY: Pod explicitly admits zero and has no destructor.
    values.resize(len, unsafe { std::mem::zeroed::<T>() });
    // SAFETY: initialized Pod values have no padding or invalid input patterns.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), size_of_val(values.as_slice()))
    };
    r.bytes(bytes)?;
    Ok(values)
}

macro_rules! pod {
    ($($ty:ty),* $(,)?) => { $(
        // SAFETY: these numeric types and layout nodes have no padding, pointers
        // or invalid Rust bit patterns. Node encodings are checked by the loader.
        unsafe impl Pod for $ty {}
        impl Codec for $ty {
            const MIN_BYTES: usize = size_of::<Self>();
            fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> { w.bytes(pod_bytes(std::slice::from_ref(self))) }
            fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
                // SAFETY: this Pod type admits all-zero bytes.
                let mut value = unsafe { std::mem::zeroed::<Self>() };
                // SAFETY: initialized, padding-free Pod accepts every input pattern.
                let bytes = unsafe { std::slice::from_raw_parts_mut((&raw mut value).cast(), size_of::<Self>()) };
                r.bytes(bytes)?;
                Ok(value)
            }
            fn put_slice<W: Write>(values: &[Self], w: &mut Encoder<W>) -> io::Result<()> { w.bytes(pod_bytes(values)) }
            fn get_vec<R: Read>(len: usize, r: &mut Decoder<R>) -> io::Result<Vec<Self>> { pod_vec(len, r) }
        }
    )* };
}
pod!(
    u8,
    u16,
    u32,
    u64,
    i32,
    f64,
    [i32; 3],
    [f64; 3],
    crate::load::layout::LayoutNode
);

macro_rules! fields {
    ($ty:ty { $($field:ident),* $(,)? }) => {
        $crate::layout_file::codec::fields!($ty { $($field),* }, 1);
    };
    ($ty:ty { $($field:ident),* $(,)? }, $min:expr) => {
        impl $crate::layout_file::codec::Codec for $ty {
            const MIN_BYTES: usize = $min;
            fn put<W: std::io::Write>(&self, w: &mut $crate::layout_file::codec::Encoder<W>) -> std::io::Result<()> {
                $( $crate::layout_file::codec::Codec::put(&self.$field, w)?; )*
                Ok(())
            }
            fn get<R: std::io::Read>(r: &mut $crate::layout_file::codec::Decoder<R>) -> std::io::Result<Self> {
                Ok(Self { $( $field: $crate::layout_file::codec::Codec::get(r)?, )* })
            }
        }
    };
}
pub(crate) use fields;

macro_rules! tagged {
    ($ty:ty { $( $tag:literal => $variant:ident $( ( $value:ident ) )? ),* $(,)? }) => {
        impl $crate::layout_file::codec::Codec for $ty {
            fn put<W: std::io::Write>(&self, w: &mut $crate::layout_file::codec::Encoder<W>) -> std::io::Result<()> {
                match self { $( Self::$variant $(($value))? => { $crate::layout_file::codec::Codec::put(&($tag as u8), w)?; $( $crate::layout_file::codec::Codec::put($value, w)?; )? } ),* }
                Ok(())
            }
            fn get<R: std::io::Read>(r: &mut $crate::layout_file::codec::Decoder<R>) -> std::io::Result<Self> {
                match <u8 as $crate::layout_file::codec::Codec>::get(r)? { $( $tag => Ok(Self::$variant $( ({ let $value = $crate::layout_file::codec::Codec::get(r)?; $value }) )? ), )* _ => Err($crate::layout_file::codec::invalid("invalid layout enum tag")) }
            }
        }
    };
}
pub(crate) use tagged;

fields!(
    Tree {
        node_start,
        node_count,
        category_start
    },
    12
);
fields!(
    PrefixGroup {
        node_base,
        depth,
        first,
        end
    },
    16
);
fields!(Output {
    base_score,
    divisor,
    postprocessor
});
fields!(Steps { leaf, stop });
fields!(PrefixStats {
    groups,
    trees,
    max_depth,
    mean_depth,
    saved,
    cost
});
fields!(
    MaskFeature {
        feature,
        num,
        chains,
        flips,
        categories
    },
    49
);
fields!(MaskChain { start, points, len }, 12);
fields!(MaskPlan {
    features,
    chains,
    points,
    flip_bounds,
    flips
});
tagged!(ThresholdType { 0 => F64, 1 => F32 });
tagged!(Postprocessor { 0 => Identity, 1 => Sigmoid(value) });
tagged!(Plan { 0 => Fixed(value), 1 => Wide(value) });

impl Codec for VaryingPredicate {
    const MIN_BYTES: usize = 10;
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> {
        match self {
            Self::Num {
                feature,
                threshold,
                default_left,
            } => {
                0u8.put(w)?;
                feature.put(w)?;
                threshold.0.put(w)?;
                default_left.put(w)
            }
            Self::CatInline {
                feature,
                default_left,
                word,
            } => {
                1u8.put(w)?;
                feature.put(w)?;
                default_left.put(w)?;
                word.put(w)
            }
            Self::CatPool {
                feature,
                default_left,
                offset,
            } => {
                2u8.put(w)?;
                feature.put(w)?;
                default_left.put(w)?;
                offset.put(w)
            }
        }
    }
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        match u8::get(r)? {
            0 => Ok(Self::Num {
                feature: Codec::get(r)?,
                threshold: Threshold(Codec::get(r)?),
                default_left: Codec::get(r)?,
            }),
            1 => Ok(Self::CatInline {
                feature: Codec::get(r)?,
                default_left: Codec::get(r)?,
                word: Codec::get(r)?,
            }),
            2 => Ok(Self::CatPool {
                feature: Codec::get(r)?,
                default_left: Codec::get(r)?,
                offset: Codec::get(r)?,
            }),
            _ => Err(invalid("invalid layout predicate tag")),
        }
    }
}

impl Codec for Model {
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> {
        self.output.put(w)?;
        self.trees.put(w)?;
        self.config.put(w)?;
        self.varying_predicates.put(w)?;
        self.masks.put(w)?;
        self.threshold_type.put(w)?;
        self.prefix_groups.put(w)?;
        self.prefix.put(w)?;
        self.tree_roots.put(w)?;
        self.steps.put(w)?;
        self.plan.put(w)?;
        self.layout.put(w)?;
        self.needs.put(w)?;
        self.rules.items.leaf_short.put(w)?;
        self.bounds.put(w)?;
        Ok(())
    }
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        Ok(Self {
            output: Codec::get(r)?,
            trees: Codec::get(r)?,
            config: Codec::get(r)?,
            categories: Vec::new(),
            varying_predicates: Codec::get(r)?,
            masks: Codec::get(r)?,
            threshold_type: Codec::get(r)?,
            prefix_groups: Codec::get(r)?,
            prefix: Codec::get(r)?,
            tree_roots: Codec::get(r)?,
            steps: Codec::get(r)?,
            plan: Codec::get(r)?,
            layout: Codec::get(r)?,
            needs: Codec::get(r)?,
            rules: Box::new(RuleInputs {
                items: ItemLengths {
                    leaf_short: Codec::get(r)?,
                },
                pool_latency: std::sync::OnceLock::new(),
            }),
            bounds: Codec::get(r)?,
            #[cfg(any(test, feature = "research"))]
            nodes: Vec::new(),
        })
    }
}

impl Codec for usize {
    const MIN_BYTES: Self = 8;
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> {
        (*self as u64).put(w)
    }
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        Self::try_from(u64::get(r)?).map_err(|_| invalid("layout usize overflow"))
    }
}
impl Codec for bool {
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> {
        u8::from(*self).put(w)
    }
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        match u8::get(r)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("invalid layout bool")),
        }
    }
}
impl<T: Codec> Codec for Vec<T> {
    const MIN_BYTES: usize = 8;
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> {
        self.len().put(w)?;
        w.align()?;
        T::put_slice(self, w)
    }
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        let len = r.length::<T>()?;
        T::get_vec(len, r)
    }
}
impl<T: Codec> Codec for Option<T> {
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> {
        self.is_some().put(w)?;
        if let Some(v) = self {
            v.put(w)?;
        }
        Ok(())
    }
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        if bool::get(r)? {
            Ok(Some(T::get(r)?))
        } else {
            Ok(None)
        }
    }
}
impl<T: Codec> Codec for std::ops::Range<T> {
    const MIN_BYTES: usize = 2 * T::MIN_BYTES;
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> {
        self.start.put(w)?;
        self.end.put(w)
    }
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        Ok(T::get(r)?..T::get(r)?)
    }
}
impl<A: Codec, B: Codec> Codec for (A, B) {
    const MIN_BYTES: usize = A::MIN_BYTES + B::MIN_BYTES;
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> {
        self.0.put(w)?;
        self.1.put(w)
    }
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        Ok((A::get(r)?, B::get(r)?))
    }
}
impl<T: Codec> Codec for Box<T> {
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> {
        (**self).put(w)
    }
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        Ok(Self::new(T::get(r)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checksum_is_independent_of_write_boundaries() {
        let input: Vec<u8> = (0..251).collect();
        let mut all = Checksum::default();
        all.write(&input);
        let expected = all.finish();
        for n in 1..=32 {
            let mut parts = Checksum::default();
            for part in input.chunks(n) {
                parts.write(part);
            }
            assert_eq!(parts.finish(), expected);
        }
    }
}
