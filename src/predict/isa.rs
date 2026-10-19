//! The instruction-set level the kernel is compiled for, as a type: [`Isa`]'s constants
//! say which optional instructions its code may use, and the kernel's choices between
//! code paths read them, so that each level's paths are chosen at compile time.
//!
//! [`Native`] is the build's own level: its constants are the target features the
//! build enables, the choices the kernel made by `cfg!` before there were levels.

/// An instruction-set level: which optional x86-64 instructions its code may use. Off
/// x86-64, every constant is false.
pub(crate) trait Isa: Copy + Eq + Send + Sync + 'static {
    /// SSE4.1: 64-bit compares in the masks' word maps.
    #[cfg_attr(
        not(target_arch = "x86_64"),
        expect(dead_code, reason = "x86-64's choices")
    )]
    const SSE41: bool;
    /// POPCNT: a mask of up to 8 words counts one word at a time.
    #[cfg_attr(
        not(target_arch = "x86_64"),
        expect(dead_code, reason = "x86-64's choices")
    )]
    const POPCNT: bool;
    /// AVX2: four words per vector for masks of 256 bits and up.
    #[cfg_attr(
        not(target_arch = "x86_64"),
        expect(dead_code, reason = "x86-64's choices")
    )]
    const AVX2: bool;
    /// AVX-512F and VL: resolve's compress listing and the masks' word tests.
    const AVX512: bool;
    /// AVX-512DQ: the listing's `kandnb`.
    const AVX512DQ: bool;
}

/// The build's own level: the target features it enables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Native;

impl Isa for Native {
    const SSE41: bool = cfg!(target_feature = "sse4.1");
    const POPCNT: bool = cfg!(target_feature = "popcnt");
    const AVX2: bool = cfg!(target_feature = "avx2");
    const AVX512: bool = cfg!(all(target_feature = "avx512f", target_feature = "avx512vl"));
    const AVX512DQ: bool = cfg!(target_feature = "avx512dq");
}
