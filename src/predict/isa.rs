//! The instruction-set level the kernel is compiled for, as a type: [`Isa`]'s constants
//! say which optional instructions its code may use, and the kernel's choices between
//! code paths read them, so that each level's paths are chosen at compile time.
//!
//! [`Native`] is the build's own level: its constants are the target features the
//! build enables. A build for x86-64 without AVX2 and FMA, such as the default
//! `x86-64`, also compiles the kernel at [`V3`] and [`V4`] and runs the highest level
//! the CPU has ([`level`]); a build that enables them, or for another architecture,
//! compiles only its own. The build script sets `cfg(dispatch)` for the first kind
//! ([`DISPATCH`]).
//!
//! A level reaches the kernel's out-of-line functions through its trampolines
//! (`Isa::with2` to `with8`), each of which calls a closure with its arguments in a
//! function that enables the level's target features. Each such function `f` has a
//! forwarder `f_at::<X>`, which callers call. Where the build dispatches, `f_at` is out of
//! line and calls `f`, `#[inline(always)]` there, inside `X`'s trampoline, so that `f`
//! compiles with `X`'s features; compiled out of line, `f`'s vector code would lack them
//! and call each intrinsic. Elsewhere `f_at` inlines into its caller and `f` keeps its
//! own attribute, so that those builds compile the kernel as before levels.

/// Every trampoline's arity: invokes `$m!` once per trampoline, with the tokens given
/// and then the trampoline's name and its arguments with their types. The arguments
/// stay separate, rather than one tuple, so that the hot functions' `&mut` arguments
/// keep their `noalias`.
macro_rules! arities {
    ($m:ident!($($head:tt)*)) => {
        $m!($($head)* with2(a: A, b: B));
        $m!($($head)* with3(a: A, b: B, c: C));
        $m!($($head)* with5(a: A, b: B, c: C, d: D, e: E));
        $m!($($head)* with7(a: A, b: B, c: C, d: D, e: E, f2: F, g: G));
        $m!($($head)* with8(a: A, b: B, c: C, d: D, e: E, f2: F, g: G, h: H));
    };
}

/// A trampoline's declaration in [`Isa`].
macro_rules! declare {
    ($n:ident($($a:ident: $t:ident),+)) => {
        /// `f` called with the arguments, with the level's target features.
        #[allow(clippy::too_many_arguments, reason = "a trampoline takes its function's arguments")]
        fn $n<$($t,)+ R>(f: impl FnOnce($($t),+) -> R, $($a: $t),+) -> R;
    };
}

/// A trampoline's body, `f` called with the arguments, in a function with the
/// attributes and visibility given.
macro_rules! direct {
    ($(#[$attr:meta])* [$($vis:tt)*] $n:ident($($a:ident: $t:ident),+)) => {
        $(#[$attr])*
        #[allow(clippy::too_many_arguments, reason = "a trampoline takes its function's arguments")]
        $($vis)* fn $n<$($t,)+ R>(f: impl FnOnce($($t),+) -> R, $($a: $t),+) -> R {
            f($($a),+)
        }
    };
}

/// An [`Isa`] method of a dispatched level: a call of the trampoline of the same name
/// in `$module`, whose function enables the level's target features.
#[cfg(dispatch)]
macro_rules! forward {
    ($module:ident; $n:ident($($a:ident: $t:ident),+)) => {
        #[inline(always)]
        fn $n<$($t,)+ R>(f: impl FnOnce($($t),+) -> R, $($a: $t),+) -> R {
            // SAFETY: the kernel runs at a level only where the CPU has every feature
            // its trampolines enable ([`level`]).
            unsafe { $module::$n(f, $($a),+) }
        }
    };
}

/// An instruction-set level: which optional x86-64 instructions its code may use, and
/// the trampolines that run code with them. Off x86-64, every constant is false.
pub trait Isa: Copy + Eq + Send + Sync + 'static {
    /// Whether this is [`Native`], whose forwarders call their functions directly rather
    /// than through a trampoline: through `Native`'s trampolines, which are the same
    /// call, rustc inlines differently and changes native builds' code.
    const NATIVE: bool;
    /// SSE4.1: 64-bit compares in the masks' word maps.
    #[cfg_attr(
        not(target_arch = "x86_64"),
        expect(dead_code, reason = "x86-64's choices")
    )]
    const SSE41: bool;
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

    arities!(declare!());
}

/// The build's own level: the target features it enables. Its forwarders call their
/// functions directly, not through its trampolines, which only make it an `Isa`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Native;

impl Isa for Native {
    const NATIVE: bool = true;
    const SSE41: bool = cfg!(target_feature = "sse4.1");
    const AVX2: bool = cfg!(target_feature = "avx2");
    const AVX512: bool = cfg!(all(target_feature = "avx512f", target_feature = "avx512vl"));
    const AVX512DQ: bool = cfg!(target_feature = "avx512dq");

    arities!(direct!(
        #[inline(always)]
        []
    ));
}

/// Whether the build compiles the kernel at [`V3`] and [`V4`] too and chooses a level
/// at run time: on x86-64 without AVX2 and FMA, where the build script sets
/// `cfg(dispatch)`.
pub(crate) const DISPATCH: bool = cfg!(dispatch);

// The build script reads the target features cargo passes it, which miss any given
// only to rustc (`cargo rustc -- -C target-feature=...`): such a build stops here
// rather than compile the kernel for the wrong levels.
const _: () = assert!(
    DISPATCH
        == cfg!(all(
            target_arch = "x86_64",
            not(all(target_feature = "avx2", target_feature = "fma"))
        )),
    "cfg(dispatch) from build.rs disagrees with the target features rustc compiles for"
);

/// A level above the base: its features, `$module::FEATURES` in every build (for
/// [`above_base`]); and where the build dispatches, its type and its trampolines,
/// `direct!`'s bodies in functions that enable the features, called from the level's
/// `Isa` methods.
macro_rules! level {
    ($(#[$doc:meta])* $name:ident, $module:ident, $features:literal, $avx512:literal) => {
        $(#[$doc])*
        #[cfg(dispatch)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $name;

        mod $module {
            /// The features the level's trampolines enable, which a CPU needs to run it.
            pub(super) const FEATURES: &str = $features;

            #[cfg(dispatch)]
            arities!(direct!(#[target_feature(enable = $features)] [pub(super)]));
        }

        #[cfg(dispatch)]
        impl Isa for $name {
            const NATIVE: bool = false;
            const SSE41: bool = true;
            const AVX2: bool = true;
            const AVX512: bool = $avx512;
            const AVX512DQ: bool = $avx512;

            arities!(forward!($module;));
        }
    };
}

level!(
    /// x86-64-v3: AVX2, FMA, BMI1 and 2, LZCNT, MOVBE and POPCNT, and what they imply.
    V3,
    v3,
    "avx,avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,lzcnt,movbe,popcnt,sse3,sse4.1,sse4.2,ssse3,xsave",
    false
);

level!(
    /// x86-64-v4, AVX-512F, BW, CD, DQ and VL on V3, with VPOPCNTDQ, BITALG and BF16:
    /// Sapphire Rapids and Zen 4 on. Skylake-SP, Cascade Lake and Cooper Lake, which lack
    /// VPOPCNTDQ and BITALG, run V3, as do Ice Lake, Tiger Lake and Rocket Lake, which
    /// lack BF16.
    ///
    /// The kernel uses no BF16 instruction: the requirement is a feature proxy, chosen
    /// from measurements, for the CPUs where V4 is the faster level. Over the 3.1
    /// survey's 90 cells, Ice Lake SP, which has V4's other features but not BF16, took
    /// 5.4% longer at V4 than at V3, and longer on 89 of them, at both function
    /// alignments; Emerald Rapids, Genoa and Turin, which have BF16, took 1.4-1.6% less
    /// at V4 at the default alignment (at 64 bytes Emerald Rapids 1.4% and Genoa 0.9%
    /// less, Turin 0.7% more). Of the families the requirement sends to V3, only Ice Lake
    /// SP was measured.
    V4,
    v4,
    "avx,avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,lzcnt,movbe,popcnt,sse3,sse4.1,sse4.2,ssse3,xsave,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512bitalg,avx512bf16",
    true
);

/// How many of the levels above the base, [`V3`] and then [`V4`], run on a CPU whose
/// features are those `has` accepts: 0 for the base, 1 for V3, 2 for V4, the index in
/// [`Level::ALL`] where the build dispatches. V4's features include V3's. A function of
/// the features alone, so that its tests run on any host.
#[cfg_attr(
    not(any(dispatch, test)),
    expect(dead_code, reason = "only a build that dispatches detects levels")
)]
fn above_base(has: impl Fn(&str) -> bool) -> usize {
    let all = |features: &str| features.split(',').all(&has);
    if all(v4::FEATURES) {
        2
    } else {
        usize::from(all(v3::FEATURES))
    }
}

/// Whether this CPU has target feature `f`, one a level's trampolines enable.
#[cfg(dispatch)]
fn detect(f: &str) -> bool {
    match f {
        "avx" => std::arch::is_x86_feature_detected!("avx"),
        "avx2" => std::arch::is_x86_feature_detected!("avx2"),
        "bmi1" => std::arch::is_x86_feature_detected!("bmi1"),
        "bmi2" => std::arch::is_x86_feature_detected!("bmi2"),
        "cmpxchg16b" => std::arch::is_x86_feature_detected!("cmpxchg16b"),
        "f16c" => std::arch::is_x86_feature_detected!("f16c"),
        "fma" => std::arch::is_x86_feature_detected!("fma"),
        "lzcnt" => std::arch::is_x86_feature_detected!("lzcnt"),
        "movbe" => std::arch::is_x86_feature_detected!("movbe"),
        "popcnt" => std::arch::is_x86_feature_detected!("popcnt"),
        "sse3" => std::arch::is_x86_feature_detected!("sse3"),
        "sse4.1" => std::arch::is_x86_feature_detected!("sse4.1"),
        "sse4.2" => std::arch::is_x86_feature_detected!("sse4.2"),
        "ssse3" => std::arch::is_x86_feature_detected!("ssse3"),
        "xsave" => std::arch::is_x86_feature_detected!("xsave"),
        "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
        "avx512bw" => std::arch::is_x86_feature_detected!("avx512bw"),
        "avx512cd" => std::arch::is_x86_feature_detected!("avx512cd"),
        "avx512dq" => std::arch::is_x86_feature_detected!("avx512dq"),
        "avx512vl" => std::arch::is_x86_feature_detected!("avx512vl"),
        "avx512vpopcntdq" => std::arch::is_x86_feature_detected!("avx512vpopcntdq"),
        "avx512bitalg" => std::arch::is_x86_feature_detected!("avx512bitalg"),
        "avx512bf16" => std::arch::is_x86_feature_detected!("avx512bf16"),
        _ => unreachable!("a level's features are listed above"),
    }
}

/// The level a predictor runs at: [`Native`] without [`DISPATCH`], where `TREEWALKER_ISA`
/// is not read; with it, the highest of [`V4`], [`V3`] and [`Native`] whose features the
/// CPU has, or a lower one that `TREEWALKER_ISA` asks for (`base`, `v3` or `v4`; any
/// other value asks for nothing).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    Native,
    #[cfg(dispatch)]
    V3,
    #[cfg(dispatch)]
    V4,
}

impl Level {
    /// Every level the build compiles, lowest first.
    pub(crate) const ALL: &[Self] = &[
        Self::Native,
        #[cfg(dispatch)]
        Self::V3,
        #[cfg(dispatch)]
        Self::V4,
    ];

    /// The level's name, as `TREEWALKER_ISA` writes it: [`Native`] is `base` where it is
    /// the lowest of the levels the build dispatches to, and `native` where it is the
    /// build's only level.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Native if DISPATCH => "base",
            Self::Native => "native",
            #[cfg(dispatch)]
            Self::V3 => "v3",
            #[cfg(dispatch)]
            Self::V4 => "v4",
        }
    }

    /// `detected`, or the lower level `asked` names.
    fn capped(detected: Self, asked: Option<&str>) -> Self {
        Self::ALL
            .iter()
            .find(|l| asked == Some(l.name()))
            .map_or(detected, |&asked| asked.min(detected))
    }
}

/// The level this process's predictors run at, as [`Level`] says.
pub(crate) fn level() -> Level {
    if DISPATCH {
        Level::capped(detected(), std::env::var("TREEWALKER_ISA").ok().as_deref())
    } else {
        Level::Native
    }
}

/// The highest level whose features this CPU has: [`Native`] without [`DISPATCH`].
#[cfg_attr(
    not(dispatch),
    expect(
        clippy::missing_const_for_fn,
        reason = "where it dispatches, detection is not const"
    )
)]
pub(crate) fn detected() -> Level {
    #[cfg(dispatch)]
    {
        Level::ALL[above_base(detect)]
    }
    #[cfg(not(dispatch))]
    Level::Native
}

/// Every level this CPU has, lowest first.
#[cfg(test)]
pub(crate) fn levels() -> Vec<Level> {
    let top = detected();
    Level::ALL.iter().copied().filter(|&l| l <= top).collect()
}

#[cfg(test)]
mod tests {
    use super::{DISPATCH, Level, above_base};

    /// x86-64-v3's features, as Haswell and Zen 1 on have them.
    const V3: &str = "avx,avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,lzcnt,movbe,popcnt,sse3,sse4.1,\
                      sse4.2,ssse3,xsave";
    /// The AVX-512 subsets Ice Lake, Sapphire Rapids and Zen 4 have, BF16 and FP16 aside.
    const ICE_LAKE_512: &str = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512ifma,\
                                avx512vbmi,avx512vbmi2,avx512vnni,avx512vpopcntdq,avx512bitalg";

    /// The levels above the base a CPU with the features in `lists` runs.
    fn levels_of(lists: &[&str]) -> usize {
        above_base(|f| lists.iter().any(|l| l.split(',').any(|g| g == f)))
    }

    #[test]
    fn v4_needs_bf16_and_v3_needs_avx2() {
        let sandy_bridge = "avx,cmpxchg16b,popcnt,sse3,sse4.1,sse4.2,ssse3,xsave";
        assert_eq!(levels_of(&[sandy_bridge]), 0, "Sandy Bridge, without AVX2");
        assert_eq!(levels_of(&[V3, "adx,aes,sha"]), 1, "Zen 3");
        let skylake_sp = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl";
        assert_eq!(levels_of(&[V3, skylake_sp]), 1, "Skylake-SP");
        assert_eq!(levels_of(&[V3, skylake_sp, "avx512bf16"]), 1, "Cooper Lake");
        assert_eq!(levels_of(&[V3, ICE_LAKE_512]), 1, "Ice Lake");
        assert_eq!(levels_of(&[V3, ICE_LAKE_512, "avx512bf16"]), 2, "Zen 4");
        assert_eq!(
            levels_of(&[V3, ICE_LAKE_512, "avx512bf16,avx512fp16"]),
            2,
            "Sapphire Rapids"
        );
    }

    #[test]
    fn treewalker_isa_lowers_the_level_and_never_raises_it() {
        // `base` is a dispatching build's lowest level, `native` another build's only one.
        let (own, other) = if DISPATCH {
            ("base", "native")
        } else {
            ("native", "base")
        };
        assert_eq!(Level::Native.name(), own);
        for &detected in Level::ALL {
            assert_eq!(Level::capped(detected, None), detected);
            assert_eq!(Level::capped(detected, Some("other")), detected);
            assert_eq!(Level::capped(detected, Some(other)), detected);
            for &asked in Level::ALL {
                assert_eq!(
                    Level::capped(detected, Some(asked.name())),
                    asked.min(detected)
                );
            }
        }
    }
}
