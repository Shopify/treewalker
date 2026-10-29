//! The machine the kernel runs on: what the target fixes at compile time, what the
//! system reports at runtime, and what a one-time probe measures because no system
//! reports it. The kernel's rules ([`crate::rules`]) derive its constants from these.
//!
//! # Compile time
//!
//! [`GENERAL_REGISTERS`], the integer registers a leaf function such as the walk can
//! hold values in, and [`VECTOR_BITS`], the widest vector registers the build enables.
//!
//! # Runtime: the data caches
//!
//! [`caches`] reads each data or unified cache's size and the logical CPUs that share
//! it, so that a cache's share per CPU is its size over its sharers, and the line size:
//!
//! - macOS: `sysctl`, the performance cores' `hw.perflevel0.*` keys, or on machines
//!   without performance levels `hw.l1dcachesize`, `hw.l2cachesize`, `hw.l3cachesize`
//!   and `hw.cacheconfig`.
//! - Linux: sysfs, `/sys/devices/system/cpu/cpu0/cache/index*/`.
//! - x86-64 without either: CPUID leaf 4 (Intel) or 0x8000001D (AMD). Its sharing
//!   count is the most logical CPUs that may share the cache, a bound rather than a
//!   count, so per-CPU shares come out no larger than the truth.
//! - Elsewhere, and under Miri: [`ASSUMED`], a 32 KiB L1 and a 256 KiB L2 per CPU with
//!   64-byte lines, the smallest data caches of the 64-bit cores of the last decade
//!   (Intel's client cores through Skylake had 256 KiB of L2). Assuming too little
//!   only makes chunks smaller and prefetches earlier, which costs a little time;
//!   assuming too much would let a chunk's working set spill out of the cache.
//!
//! # Runtime: latency
//!
//! [`latency`] is the load-to-use latency of a dependent load in each cache level
//! and in memory. A pointer chase over a random cyclic permutation of lines measures
//! it: each load's address is the previous load's value, so loads cannot overlap,
//! and the random order defeats the hardware prefetchers. Each level's buffer misses
//! the level below and fits the level. The caches' levels are measured together when
//! the first predictor is built (about 1–3 ms) and cached for the process.
//!
//! A model whose nodes do not fit a CPU's share of the last reported cache has its
//! pool's latency measured instead, once per model, by a walk of random nodes of the
//! pool itself, each chosen by a hash of the last one: exactly the memory the kernel's
//! walk touches, whatever holds it (a cache the system does not report, such as the
//! M4's system-level cache, the rest of a shared cache, or memory), and nothing to
//! allocate.
//!
//! The ratio to L1's latency that the rules read is rounded to the nearest power of
//! two in log space ([`quantize`]); see there for why. `TREEWALKER_LATENCY_NS`
//! overrides the probe with a comma-separated list, the caches' latencies from L1
//! then memory's, for benchmarks that must not depend on it; under Miri the probe
//! does not run and [`ASSUMED_LATENCY`] applies.

use std::sync::OnceLock;

use crate::load::layout::LayoutNode;

// ---------------------------------------------------------------------------
// Compile time
// ---------------------------------------------------------------------------

/// General-purpose registers that a leaf function can hold values in: the
/// architecture's integer registers less those the ABI reserves.
///
/// - AArch64: x0–x30, less x18 (the platform register on Apple and Windows) and x29
///   (the frame pointer, which Apple's ABI keeps in every function). The walk calls
///   nothing, so the link register x30 is free once saved.
/// - x86-64: 16, less rsp; Apple's ABI also keeps rbp as the frame pointer.
/// - RISC-V and LoongArch 64: 31 besides the zero register, less sp, gp and tp.
/// - Others: 15, the fewest of the targets above, so that lanes chosen for an
///   unknown target do not count on registers it may not have.
pub(crate) const GENERAL_REGISTERS: usize = if cfg!(target_arch = "aarch64") {
    if cfg!(any(target_vendor = "apple", target_os = "windows")) {
        29
    } else {
        30
    }
} else if cfg!(target_arch = "x86_64") {
    if cfg!(target_vendor = "apple") {
        14
    } else {
        15
    }
} else if cfg!(any(target_arch = "riscv64", target_arch = "loongarch64")) {
    28
} else {
    15
};

/// Bits of the widest vector registers the build enables: AVX-512, AVX2, SSE2 or
/// NEON; 0 without any. Reported by the research module.
#[cfg_attr(
    not(any(test, feature = "research")),
    expect(dead_code, reason = "reported by the research module")
)]
pub(crate) const VECTOR_BITS: usize = if cfg!(target_feature = "avx512f") {
    512
} else if cfg!(target_feature = "avx2") {
    256
} else if cfg!(any(target_feature = "sse2", target_feature = "neon")) {
    128
} else {
    0
};

// ---------------------------------------------------------------------------
// Caches
// ---------------------------------------------------------------------------

/// A data or unified cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cache {
    /// 1 for L1, and so on.
    pub(crate) level: u8,
    pub(crate) bytes: usize,
    /// Logical CPUs that share it, at least 1.
    pub(crate) shared_by: usize,
}

impl Cache {
    /// The cache's share per logical CPU: what each gets when every CPU runs a
    /// predictor, as a serving process with a worker per CPU does.
    pub(crate) const fn per_cpu(self) -> usize {
        self.bytes / self.shared_by
    }
}

/// Where [`Caches`] came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code, reason = "each target constructs only the sources it reads")]
pub(crate) enum Source {
    Sysctl,
    Sysfs,
    Cpuid,
    Assumed,
}

impl Source {
    #[cfg_attr(
        not(any(test, feature = "research")),
        expect(dead_code, reason = "reported by the research module")
    )]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Sysctl => "sysctl",
            Self::Sysfs => "sysfs",
            Self::Cpuid => "cpuid",
            Self::Assumed => "assumed",
        }
    }
}

/// The data cache hierarchy, L1 first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Caches {
    /// Bytes per cache line.
    pub(crate) line: usize,
    /// The data and unified caches, by level, at least L1 and L2.
    pub(crate) levels: Vec<Cache>,
    pub(crate) source: Source,
}

/// What [`caches`] assumes when the system reports nothing; see the module docs.
pub(crate) const ASSUMED: [Cache; 2] = [
    Cache {
        level: 1,
        bytes: 32 << 10,
        shared_by: 1,
    },
    Cache {
        level: 2,
        bytes: 256 << 10,
        shared_by: 1,
    },
];

impl Caches {
    fn assumed() -> Self {
        Self {
            line: 64,
            levels: ASSUMED.to_vec(),
            source: Source::Assumed,
        }
    }

    /// The caches of `levels`, sorted and checked: L1 and L2 present, sizes positive
    /// and sharing at least 1. `None` if they are not. For the readers of the OS's and
    /// the CPU's reports, which Miri and other platforms do without.
    #[cfg(all(
        any(target_vendor = "apple", target_os = "linux", target_arch = "x86_64"),
        not(miri)
    ))]
    fn of(line: usize, mut levels: Vec<Cache>, source: Source) -> Option<Self> {
        levels.retain(|c| c.bytes > 0);
        levels.sort_by_key(|c| c.level);
        levels.dedup_by_key(|c| c.level);
        for c in &mut levels {
            c.shared_by = c.shared_by.max(1);
        }
        let has = |l: u8| levels.iter().any(|c| c.level == l);
        (has(1) && has(2) && line.is_power_of_two()).then_some(Self {
            line,
            levels,
            source,
        })
    }

    /// The cache at `level`, if reported.
    pub(crate) fn level(&self, level: u8) -> Option<Cache> {
        self.levels.iter().copied().find(|c| c.level == level)
    }
}

/// The data caches, read once per process.
pub(crate) fn caches() -> &'static Caches {
    static CACHES: OnceLock<Caches> = OnceLock::new();
    CACHES.get_or_init(detect)
}

fn detect() -> Caches {
    if cfg!(miri) {
        return Caches::assumed();
    }
    #[cfg(all(target_vendor = "apple", not(miri)))]
    if let Some(c) = sysctl::caches() {
        return c;
    }
    #[cfg(all(target_os = "linux", not(miri)))]
    if let Some(c) = sysfs::caches() {
        return c;
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if let Some(c) = cpuid::caches() {
        return c;
    }
    Caches::assumed()
}

#[cfg(all(target_vendor = "apple", not(miri)))]
mod sysctl {
    use super::{Cache, Caches, Source};
    use std::ffi::{CString, c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// The integers of the sysctl `name`: one, or an array such as `hw.cacheconfig`.
    fn values(name: &str) -> Option<Vec<u64>> {
        let name = CString::new(name).ok()?;
        let mut buf = [0u64; 16];
        let mut len = size_of_val(&buf);
        // SAFETY: name is a NUL-terminated string, buf has len writable bytes, and
        // sysctlbyname writes at most len bytes and stores the count in len; it
        // sets nothing with a null new value.
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr(),
                buf.as_mut_ptr().cast(),
                &raw mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            return None;
        }
        match len {
            4 => Some(vec![u64::from(buf[0] as u32)]),
            n if n % 8 == 0 && n > 0 => Some(buf[..n / 8].to_vec()),
            _ => None,
        }
    }

    fn value(name: &str) -> Option<usize> {
        values(name)?.first().map(|&v| v as usize)
    }

    pub(super) fn caches() -> Option<Caches> {
        let line = value("hw.cachelinesize")?;
        let mut levels = Vec::new();
        if value("hw.perflevel0.l1dcachesize").is_some() {
            // The performance cores, which a latency-bound kernel should run on.
            let p = |key: &str| value(&format!("hw.perflevel0.{key}"));
            levels.push(Cache {
                level: 1,
                bytes: p("l1dcachesize")?,
                shared_by: 1,
            });
            for (level, size, by) in [
                (2, "l2cachesize", "cpusperl2"),
                (3, "l3cachesize", "cpusperl3"),
            ] {
                if let Some(bytes) = p(size) {
                    levels.push(Cache {
                        level,
                        bytes,
                        shared_by: p(by).unwrap_or(1),
                    });
                }
            }
        } else {
            // hw.cacheconfig: logical CPUs sharing memory, then each level.
            let config = values("hw.cacheconfig").unwrap_or_default();
            let shared = |l: usize| config.get(l).map_or(1, |&v| v as usize);
            for (level, key) in [
                (1u8, "hw.l1dcachesize"),
                (2, "hw.l2cachesize"),
                (3, "hw.l3cachesize"),
            ] {
                if let Some(bytes) = value(key) {
                    levels.push(Cache {
                        level,
                        bytes,
                        shared_by: shared(level as usize),
                    });
                }
            }
        }
        Caches::of(line, levels, Source::Sysctl)
    }
}

#[cfg(all(target_os = "linux", not(miri)))]
mod sysfs {
    use super::{Cache, Caches, Source};

    /// A size such as `48K`, `2048K` or `105M`.
    fn size(s: &str) -> Option<usize> {
        let s = s.trim();
        let (digits, unit) = match s.as_bytes().last()? {
            b'K' => (&s[..s.len() - 1], 1 << 10),
            b'M' => (&s[..s.len() - 1], 1 << 20),
            b'G' => (&s[..s.len() - 1], 1 << 30),
            _ => (s, 1),
        };
        digits.parse::<usize>().ok()?.checked_mul(unit)
    }

    /// The CPUs of a list such as `0-1,56-57`.
    fn count(list: &str) -> Option<usize> {
        list.trim()
            .split(',')
            .filter(|r| !r.is_empty())
            .try_fold(0, |n, r| {
                let (lo, hi) = r.split_once('-').unwrap_or((r, r));
                let (lo, hi) = (lo.parse::<usize>().ok()?, hi.parse::<usize>().ok()?);
                Some(n + hi.checked_sub(lo)? + 1)
            })
    }

    pub(super) fn caches() -> Option<Caches> {
        let base = std::path::Path::new("/sys/devices/system/cpu/cpu0/cache");
        let mut levels = Vec::new();
        let mut line = 0;
        for i in 0..16 {
            let dir = base.join(format!("index{i}"));
            let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
            let Some(kind) = read("type") else { break };
            if kind.trim() == "Instruction" {
                continue;
            }
            let level: u8 = read("level")?.trim().parse().ok()?;
            let bytes = size(&read("size")?)?;
            let shared_by = read("shared_cpu_list").and_then(|l| count(&l)).unwrap_or(1);
            if level == 1 {
                line = read("coherency_line_size")?.trim().parse().ok()?;
            }
            levels.push(Cache {
                level,
                bytes,
                shared_by,
            });
        }
        Caches::of(line, levels, Source::Sysfs)
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn parses_sizes_and_lists() {
            assert_eq!(super::size("48K\n"), Some(48 << 10));
            assert_eq!(super::size("105M"), Some(105 << 20));
            assert_eq!(super::count("0-1,56-57\n"), Some(4));
            assert_eq!(super::count("3"), Some(1));
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
mod cpuid {
    use super::{Cache, Caches, Source};
    use std::arch::x86_64::{__cpuid, __cpuid_count};

    /// The data and unified caches of the deterministic cache parameters `leaf`.
    fn leaf(leaf: u32) -> Vec<(Cache, usize)> {
        let mut out = Vec::new();
        for sub in 0..16 {
            let r = __cpuid_count(leaf, sub);
            let kind = r.eax & 0x1f;
            if kind == 0 {
                break;
            }
            if kind == 2 {
                continue;
            }
            let field = |v: u32, at: u32, bits: u32| ((v >> at) & ((1 << bits) - 1)) as usize + 1;
            let line = field(r.ebx, 0, 12);
            let bytes = field(r.ebx, 22, 10) * field(r.ebx, 12, 10) * line * (r.ecx as usize + 1);
            out.push((
                Cache {
                    level: ((r.eax >> 5) & 7) as u8,
                    bytes,
                    shared_by: field(r.eax, 14, 12),
                },
                line,
            ));
        }
        out
    }

    pub(super) fn caches() -> Option<Caches> {
        let mut found = if __cpuid(0).eax >= 4 {
            leaf(4)
        } else {
            Vec::new()
        };
        if found.is_empty() && __cpuid(0x8000_0000).eax >= 0x8000_001d {
            found = leaf(0x8000_001d);
        }
        let line = found.iter().find(|(c, _)| c.level == 1)?.1;
        Caches::of(
            line,
            found.into_iter().map(|(c, _)| c).collect(),
            Source::Cpuid,
        )
    }
}

// ---------------------------------------------------------------------------
// Latency
// ---------------------------------------------------------------------------

/// Latencies under Miri, where the probe does not run, in L1 latencies: L1, L2 and
/// memory, near current cores' ratios. Miri only checks the code that reads them.
pub(crate) const ASSUMED_LATENCY: [f64; 3] = [1.0, 8.0, 128.0];

/// The environment variable that overrides the probe: nanoseconds per cache level
/// from L1, then memory's.
pub(crate) const LATENCY_ENV: &str = "TREEWALKER_LATENCY_NS";

/// What the probe measured for a cache level or a node pool: nanoseconds, the ratio
/// to L1's latency, and that ratio quantized.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Latency {
    pub(crate) ns: f64,
    /// In L1 latencies, as measured.
    pub(crate) ratio: f64,
    /// In L1 latencies, quantized ([`quantize`]); 1 for L1.
    pub(crate) l1s: f64,
}

impl Latency {
    fn of(ns: f64, l1: f64) -> Self {
        let ratio = ns / l1;
        Self {
            ns,
            ratio,
            l1s: quantize(ratio),
        }
    }
}

/// The load-to-use latency of a dependent load in the cache at `level` (1 for L1), or,
/// for a level past the last cache, of the node pool `nodes`, measured by a [`Walk`]
/// over it on the first call with its `cache` and kept there. With no nodes to walk,
/// a pool past the last cache takes [`ASSUMED_LATENCY`]'s memory ratio.
///
/// The rules use the ratio to L1's latency, which the probe measures in the same
/// interleaved runs. Cache latencies are fixed in cycles, so their ratios do not
/// depend on the clock, which an idle core raises over tens of milliseconds (from
/// 2.6 to 0.89 ns per L1 load on the M4 over 30 ms): measuring in nanoseconds would
/// need the clock to settle first. Memory's latency is fixed in nanoseconds, so its
/// ratio grows with the clock; a cold probe underestimates it.
pub(crate) fn latency(level: u8, nodes: &[LayoutNode], cache: &OnceLock<Latency>) -> Latency {
    static LEVELS: OnceLock<Vec<Latency>> = OnceLock::new();
    let caches = caches();
    let n = caches.levels.len();
    let i = caches
        .levels
        .iter()
        .position(|c| c.level >= level)
        .unwrap_or(n);
    if let Some(over) = overridden(n) {
        let ratio = over[i] / over[0];
        return Latency {
            ns: over[i],
            ratio,
            l1s: ratio,
        };
    }
    if cfg!(miri) || (i == n && nodes.is_empty()) {
        let l1s = if i == n {
            ASSUMED_LATENCY[2]
        } else {
            ASSUMED_LATENCY[usize::from(i > 0)]
        };
        return Latency {
            ns: l1s,
            ratio: l1s,
            l1s,
        };
    }
    if i == n {
        return *cache.get_or_init(|| {
            let mut l1 = Chase::new(caches.levels[0].bytes / 2, caches.line);
            let mut walk = Walk {
                nodes,
                at: 0,
                step: 0,
            };
            walk.warm(caches);
            settle(&mut l1);
            let [l1, pool] = interleaved(&mut [&mut l1, &mut walk])[..] else {
                unreachable!("two probes")
            };
            Latency::of(pool, l1)
        });
    }
    LEVELS.get_or_init(|| {
        let mut chases = Vec::new();
        let mut below = 0;
        for c in &caches.levels {
            // Eight times the level below hits it in at most an eighth of the loads
            // (a quarter made the M4's L2 look 15% faster); half of this level fits
            // in it.
            let bytes = if below == 0 {
                c.bytes / 2
            } else {
                (8 * below).min(c.bytes / 2).max(2 * below)
            };
            chases.push(Chase::new(bytes, caches.line));
            below = c.bytes;
        }
        let mut probes: Vec<&mut dyn Probe> =
            chases.iter_mut().map(|c| c as &mut dyn Probe).collect();
        let ns = interleaved(&mut probes);
        ns.iter().map(|&ns_i| Latency::of(ns_i, ns[0])).collect()
    })[i]
}

/// Latencies from [`LATENCY_ENV`]: one per cache level, then memory's.
fn overridden(levels: usize) -> Option<&'static [f64]> {
    static OVERRIDE: OnceLock<Option<Vec<f64>>> = OnceLock::new();
    OVERRIDE
        .get_or_init(|| {
            let s = std::env::var(LATENCY_ENV).ok()?;
            let v: Vec<f64> = s
                .split(',')
                .map(|x| x.trim().parse::<f64>().ok().filter(|&x| x > 0.0))
                .collect::<Option<_>>()?;
            (v.len() == levels + 1).then_some(v)
        })
        .as_deref()
}

/// A latency ratio `x` rounded to the nearest power of two in log space, and at
/// least 1.
///
/// The rules turn latencies into powers of two (prefetch distances), so a power of
/// two is all the precision a ratio needs, and rounding in log space keeps the error
/// within a factor of √2 either way, the least any power-of-two grid allows; rounding
/// up, as the first form did, errs by up to 2. The boundaries fall at 2^(k+½), so a
/// measurement flips between runs only within its noise of one of them, which no
/// grid avoids; what keeps the noise small is the measurement (a settled clock, and
/// for pools the walk's warm-up). Rounding up put the M4 Pro's memory, 116–130 L1
/// latencies, across the boundary at 128, so that processes read 128 or 256; its
/// nearest boundaries are now 90.5 and 181.
fn quantize(x: f64) -> f64 {
    if x.is_finite() && x > 0.0 {
        2f64.powi(x.log2().round() as i32).max(1.0)
    } else {
        x
    }
}

/// A dependent load the probe times, run after run.
trait Probe {
    /// Nanoseconds per load over one run, continuing from where the last ended.
    fn run(&mut self) -> f64;
}

/// A walk over a node pool the model already holds, so that measuring its latency
/// allocates nothing and reads what the kernel's walk reads: each load's node is a
/// hash of the last node's index, its words and the step, so every load depends on
/// the one before, and random nodes defeat the prefetchers as the kernel's jumps
/// between trees do. The step keeps the walk from closing a short cycle, as a
/// function of the node alone would after about the square root of the pool's
/// nodes.
struct Walk<'a> {
    nodes: &'a [LayoutNode],
    at: usize,
    step: u64,
}

impl Walk<'_> {
    /// Loads per run.
    const LOADS: usize = 4096;

    /// Walk until the caches hold what a long random walk over the pool leaves in
    /// them, so that the measured runs start from the same state in every process,
    /// whatever ran before (loading, other models): twice as many loads as the pool has
    /// lines, which reaches every line about twice over, but no more than the last
    /// cache has lines (on an M4 Pro, 131,072 loads, about 15 ms from memory), past
    /// which the caches hold too little of the pool for their state to matter. Without
    /// it, what loading left cached decided the walk's hit rate on pools near the
    /// caches' size, and its latency varied by up to a factor of 1.4 between
    /// processes.
    fn warm(&mut self, caches: &Caches) {
        let lines = |bytes: usize| bytes / caches.line;
        let last = caches.levels.last().map_or(0, |c| c.bytes);
        let loads = (2 * lines(size_of_val(self.nodes))).min(lines(last));
        for _ in 0..loads.div_ceil(Self::LOADS) {
            self.run();
        }
    }
}

impl Probe for Walk<'_> {
    fn run(&mut self) -> f64 {
        const LOADS: usize = Walk::LOADS;
        let n = self.nodes.len() as u64;
        let mut at = self.at;
        let start = std::time::Instant::now();
        for _ in 0..LOADS {
            let node = self.nodes[at];
            self.step += 1;
            let h = (at as u64 ^ node.w0 ^ node.w1 ^ self.step << 32)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            // The hash's high 32 bits scaled to the pool: fewer than 2^32 nodes.
            at = (((h >> 32) * n) >> 32) as usize;
        }
        let ns = start.elapsed().as_secs_f64() * 1e9 / LOADS as f64;
        self.at = std::hint::black_box(at);
        ns
    }
}

/// A random cycle over lines: each line holds the index of the next.
struct Chase {
    next: Vec<usize>,
    start: usize,
    loads: usize,
    /// Slots in the cycle, walked before each run.
    warm: usize,
}

impl Chase {
    /// A cycle over `bytes` of `line`-byte lines, which one pass before each run loads
    /// into the cache that holds it.
    fn new(bytes: usize, line: usize) -> Self {
        let stride = (line / size_of::<usize>()).max(1);
        let slots = (bytes / (stride * size_of::<usize>())).max(16);
        // A random permutation of the slots (Fisher-Yates, with a fixed xorshift
        // seed), linked into one cycle in its order.
        let mut order: Vec<usize> = (0..slots).collect();
        let mut x = 0x9E37_79B9_7F4A_7C15_u64;
        for i in (1..slots).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            order.swap(i, (x % (i as u64 + 1)) as usize);
        }
        let mut next = vec![0usize; slots * stride];
        for (w, &slot) in order.iter().enumerate() {
            next[slot * stride] = order[(w + 1) % slots] * stride;
        }
        Self {
            next,
            start: order[0] * stride,
            loads: slots.clamp(4096, 1 << 15),
            warm: slots,
        }
    }
}

impl Probe for Chase {
    /// Nanoseconds per load over one run, continuing from where the last ended. The
    /// chase first walks its cycle once, which reloads it after other runs.
    fn run(&mut self) -> f64 {
        let mut p = self.start;
        for _ in 0..self.warm {
            p = self.next[p];
        }
        let start = std::time::Instant::now();
        for _ in 0..self.loads {
            p = self.next[p];
        }
        let ns = start.elapsed().as_secs_f64() * 1e9 / self.loads as f64;
        self.start = std::hint::black_box(p);
        ns
    }
}

/// Run `l1` until it stops getting faster: until no run in the last 2 ms beats the
/// best by 1%, or for 50 ms at most.
///
/// An idle core takes tens of milliseconds to reach its full clock (on the M4, from
/// 2.6 to 0.89 ns per L1 load over about 30 ms). Ratios between cache levels do not
/// depend on it, since each level's latency is fixed in cycles, but a pool held in
/// part by memory or a cache outside the core's clock has a latency fixed partly in
/// nanoseconds, whose ratio to L1's grows with the clock: measured cold it reads low
/// (on an M4 Pro, 4 L1 latencies instead of 12.5 for a 6.8 MiB pool in about one
/// process in three). The kernel's passes run at the full clock, so the pool is
/// measured there.
fn settle(l1: &mut Chase) {
    let start = std::time::Instant::now();
    let (mut best, mut since) = (f64::INFINITY, start);
    while start.elapsed().as_millis() < 50 && since.elapsed().as_micros() < 2000 {
        let ns = l1.run();
        if ns < 0.99 * best {
            (best, since) = (ns, std::time::Instant::now());
        }
    }
}

/// The fastest of several runs of each probe, interleaved so that every probe sees
/// the same clocks.
fn interleaved(probes: &mut [&mut dyn Probe]) -> Vec<f64> {
    let mut best = vec![f64::INFINITY; probes.len()];
    for _ in 0..5 {
        for (b, p) in best.iter_mut().zip(probes.iter_mut()) {
            *b = b.min(p.run());
        }
    }
    best
}

#[cfg(test)]
#[expect(clippy::float_cmp, reason = "powers of two compare exactly")]
mod tests {
    use super::*;

    #[test]
    fn caches_have_two_levels_and_a_line() {
        let c = caches();
        assert!(c.level(1).is_some() && c.level(2).is_some());
        assert!(c.line.is_power_of_two() && c.line >= 16);
        assert!(c.levels.iter().all(|l| l.per_cpu() > 0));
    }

    #[test]
    fn quantizes_to_the_nearest_power_of_two_in_log_space() {
        assert_eq!(quantize(0.67), 1.0);
        assert_eq!(quantize(1.0), 1.0);
        assert_eq!(quantize(5.0), 4.0);
        assert_eq!(quantize(6.0), 8.0);
        assert_eq!(quantize(97.0), 128.0);
        assert_eq!(quantize(125.0), 128.0);
        assert_eq!(quantize(180.0), 128.0);
        assert_eq!(quantize(182.0), 256.0);
    }

    #[test]
    fn latencies_are_l1_multiples() {
        let none = OnceLock::new();
        let l1 = latency(1, &[], &none);
        let l2 = latency(2, &[], &none);
        assert_eq!(l1.l1s, 1.0);
        assert!(l2.l1s >= 1.0 && l2.ns > 0.0);
    }
}
