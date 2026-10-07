//! The constant-rate timestamp counter: `rdtsc` on x86, `cntvct_el0` on Arm.
//!
//! These count fixed-frequency ticks, not core cycles. Reads are serialized:
//! `lfence; rdtsc` to start and `rdtscp; lfence` to stop on x86, `isb` before
//! `cntvct_el0` on Arm. Samples are stored as raw ticks; [`Calibration`] records the
//! counter frequency, how it was found and checked, and the overhead of back-to-back
//! reads. Other targets fall back to nanoseconds from a monotonic clock.

#![expect(
    clippy::inline_always,
    reason = "a read must not add a call to the interval it bounds"
)]

use std::sync::atomic::{Ordering, compiler_fence};

use serde::Serialize;

/// Read the counter at the start of an interval.
///
/// The compiler fences keep the compiler from moving the timed work's loads and
/// stores across the read; the instruction barriers order the hardware.
#[inline(always)]
pub fn start() -> u64 {
    compiler_fence(Ordering::SeqCst);
    let t = imp::start();
    compiler_fence(Ordering::SeqCst);
    t
}

/// Read the counter at the end of an interval.
#[inline(always)]
pub fn stop() -> u64 {
    compiler_fence(Ordering::SeqCst);
    let t = imp::stop();
    compiler_fence(Ordering::SeqCst);
    t
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::{__rdtscp, _mm_lfence, _rdtsc};

    pub const NAME: &str = "rdtsc";

    #[inline(always)]
    pub fn start() -> u64 {
        // SAFETY: lfence and rdtsc are available on every x86_64 CPU.
        unsafe {
            _mm_lfence();
            _rdtsc()
        }
    }

    #[inline(always)]
    pub fn stop() -> u64 {
        let mut aux = 0u32;
        // SAFETY: rdtscp is available on every x86_64 CPU this harness targets; aux
        // is a valid destination.
        unsafe {
            let t = __rdtscp(&raw mut aux);
            _mm_lfence();
            t
        }
    }

    /// The architected counter frequency; x86 has none, see `super::calibrate`.
    pub const fn architected_hz() -> Option<u64> {
        None
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use std::arch::asm;

    pub const NAME: &str = "cntvct_el0";

    #[inline(always)]
    fn read() -> u64 {
        let v: u64;
        // SAFETY: cntvct_el0 is readable at EL0 on Linux and macOS; isb orders the
        // read after every earlier instruction. Without `nomem`, the compiler treats
        // the asm as touching memory and keeps loads and stores on their side of it.
        unsafe { asm!("isb", "mrs {}, cntvct_el0", out(reg) v, options(nostack)) };
        v
    }

    #[inline(always)]
    pub fn start() -> u64 {
        read()
    }

    #[inline(always)]
    pub fn stop() -> u64 {
        read()
    }

    #[expect(
        clippy::unnecessary_wraps,
        reason = "the same signature on every architecture"
    )]
    pub fn architected_hz() -> Option<u64> {
        let v: u64;
        // SAFETY: cntfrq_el0 is readable at EL0 on Linux and macOS.
        unsafe { asm!("mrs {}, cntfrq_el0", out(reg) v, options(nostack, nomem)) };
        Some(v)
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod imp {
    use std::sync::OnceLock;
    use std::time::Instant;

    pub const NAME: &str = "monotonic-ns";

    fn epoch() -> Instant {
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        *EPOCH.get_or_init(Instant::now)
    }

    #[inline(always)]
    pub fn start() -> u64 {
        epoch().elapsed().as_nanos() as u64
    }

    #[inline(always)]
    pub fn stop() -> u64 {
        start()
    }

    pub const fn architected_hz() -> Option<u64> {
        Some(1_000_000_000)
    }
}

/// Nanoseconds from `CLOCK_MONOTONIC_RAW`, which NTP does not slew.
fn monotonic_raw_ns() -> u64 {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: ts is a valid timespec.
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &raw mut ts) };
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        use std::sync::OnceLock;
        static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
        EPOCH
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_nanos() as u64
    }
}

/// The kernel's TSC frequency, from the `perf_event` mmap page of a dummy software
/// event: `time_mult` and `time_shift` convert TSC ticks to nanoseconds.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn kernel_tsc_hz() -> Option<f64> {
    // struct perf_event_attr: type at 0, size at 4, config at 8; 136 bytes covers
    // every field the kernel needs to read here.
    let mut attr = [0u8; 136];
    attr[0..4].copy_from_slice(&1u32.to_ne_bytes()); // PERF_TYPE_SOFTWARE
    attr[4..8].copy_from_slice(&136u32.to_ne_bytes());
    attr[8..16].copy_from_slice(&9u64.to_ne_bytes()); // PERF_COUNT_SW_DUMMY
    // Flags at 40: disabled (bit 0), exclude_kernel (bit 5), exclude_hv (bit 6).
    attr[40..48].copy_from_slice(&(1u64 | 1 << 5 | 1 << 6).to_ne_bytes());
    // SAFETY: attr is a zeroed perf_event_attr of the declared size.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_perf_event_open,
            attr.as_ptr(),
            0,
            -1,
            -1,
            8, // PERF_FLAG_FD_CLOEXEC
        )
    };
    if fd < 0 {
        return None;
    }
    let fd = fd as libc::c_int;
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    // SAFETY: maps one read-only page of the event's metadata.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            page,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    let hz = if ptr == libc::MAP_FAILED {
        None
    } else {
        // struct perf_event_mmap_page: capabilities at 40, time_shift (u16) at 50,
        // time_mult (u32) at 52. cap_user_time is capability bit 3.
        // SAFETY: the page is mapped, page-aligned and at least 64 bytes long; each
        // field is read at its natural alignment.
        let (caps, shift, mult) = unsafe {
            (
                std::ptr::read_volatile(ptr.cast::<u64>().add(40 / 8)),
                std::ptr::read_volatile(ptr.cast::<u16>().add(50 / 2)),
                std::ptr::read_volatile(ptr.cast::<u32>().add(52 / 4)),
            )
        };
        // SAFETY: unmaps the page mapped above.
        unsafe { libc::munmap(ptr, page) };
        (caps & (1 << 3) != 0 && mult != 0)
            .then(|| 1e9 * 2f64.powi(i32::from(shift)) / f64::from(mult))
    };
    // SAFETY: fd is the descriptor opened above.
    unsafe { libc::close(fd) };
    hz
}

/// How the counter's frequency was found and checked, and what a read costs.
#[derive(Debug, Clone, Serialize)]
pub struct Calibration {
    /// The counter: `rdtsc`, `cntvct_el0` or `monotonic-ns`.
    pub counter: &'static str,
    /// Ticks per second used to convert samples.
    pub hz: f64,
    /// Where `hz` comes from: `cntfrq_el0`, `kernel-tsc` (the perf mmap page) or
    /// `measured` (against `CLOCK_MONOTONIC_RAW`).
    pub source: &'static str,
    /// Ticks per second measured against `CLOCK_MONOTONIC_RAW` over `check_ms`.
    pub measured_hz: f64,
    pub check_ms: u64,
    /// `hz / measured_hz - 1`.
    pub relative_difference: f64,
    /// The smallest step the counter was seen to advance by, in ticks. A counter
    /// can be scaled to a nominal frequency above its update rate.
    pub resolution_ticks: u64,
    /// Back-to-back `start(); stop()` pairs, in ticks.
    pub overhead: Overhead,
}

/// The distribution of back-to-back read pairs, in ticks. The raw pairs are written
/// to the run's `timer.parquet`; no single value is subtracted from samples.
#[derive(Debug, Clone, Serialize)]
pub struct Overhead {
    pub n: usize,
    pub min: u64,
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub max: u64,
    pub mean: f64,
}

/// Measure `n` back-to-back read pairs.
pub fn overhead_samples(n: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = start();
        let t1 = stop();
        out.push(t1.wrapping_sub(t0));
    }
    out
}

/// Summarize overhead samples.
pub fn summarize(samples: &[u64]) -> Overhead {
    let mut s = samples.to_vec();
    s.sort_unstable();
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    Overhead {
        n: s.len(),
        min: s[0],
        p50: q(0.5),
        p90: q(0.9),
        p99: q(0.99),
        max: s[s.len() - 1],
        mean: s.iter().map(|&v| v as f64).sum::<f64>() / s.len() as f64,
    }
}

/// Find the counter's frequency, check it against `CLOCK_MONOTONIC_RAW` over
/// `check_ms`, and measure the overhead of `overhead_n` read pairs.
pub fn calibrate(check_ms: u64, overhead_n: usize) -> (Calibration, Vec<u64>) {
    let (n0, t0) = (monotonic_raw_ns(), start());
    std::thread::sleep(std::time::Duration::from_millis(check_ms));
    let (n1, t1) = (monotonic_raw_ns(), stop());
    let measured_hz = t1.wrapping_sub(t0) as f64 / ((n1 - n0) as f64 * 1e-9);
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let kernel = kernel_tsc_hz().map(|hz| (hz, "kernel-tsc"));
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    let kernel: Option<(f64, &'static str)> = None;
    let (hz, source) = imp::architected_hz()
        .map(|hz| (hz as f64, "cntfrq_el0"))
        .or(kernel)
        .unwrap_or((measured_hz, "measured"));
    let source = if imp::NAME == "monotonic-ns" {
        "nanoseconds"
    } else {
        source
    };
    let samples = overhead_samples(overhead_n);
    let resolution_ticks = (0..1000)
        .filter_map(|_| {
            let t0 = start();
            let mut t1 = start();
            while t1 == t0 {
                t1 = start();
            }
            t1.checked_sub(t0)
        })
        .min()
        .unwrap_or(0);
    (
        Calibration {
            counter: imp::NAME,
            hz,
            source,
            measured_hz,
            check_ms,
            relative_difference: hz / measured_hz - 1.0,
            resolution_ticks,
            overhead: summarize(&samples),
        },
        samples,
    )
}

/// A check of per-call samples against one long interval over the same calls.
#[derive(Debug, Clone, Serialize)]
pub struct AmortizedCheck {
    /// Calls of the fixed workload.
    pub calls: usize,
    /// Mean of the per-call samples, in ticks; each includes one read pair.
    pub per_call_mean: f64,
    /// One interval over all calls, divided by the call count.
    pub amortized_mean: f64,
    /// `per_call_mean - amortized_mean`: what timing each call on its own adds.
    pub difference: f64,
    /// Median read-pair overhead, for comparison with `difference`.
    pub overhead_p50: u64,
}

/// Time a fixed workload call by call and as one long interval. The difference of
/// the means should be close to the read-pair overhead.
pub fn amortized_check(calls: usize, overhead_p50: u64) -> AmortizedCheck {
    fn work(seed: u64) -> u64 {
        // About a microsecond of dependent integer work.
        let mut x = seed;
        for _ in 0..1000 {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
        }
        x
    }
    let mut acc = 0u64;
    for i in 0..calls / 10 {
        acc ^= std::hint::black_box(work(i as u64));
    }
    let mut total = 0u64;
    for i in 0..calls {
        let t0 = start();
        acc ^= std::hint::black_box(work(i as u64));
        total += stop().wrapping_sub(t0);
    }
    let t0 = start();
    for i in 0..calls {
        acc ^= std::hint::black_box(work(i as u64));
    }
    let long = stop().wrapping_sub(t0);
    std::hint::black_box(acc);
    let per_call_mean = total as f64 / calls as f64;
    let amortized_mean = long as f64 / calls as f64;
    AmortizedCheck {
        calls,
        per_call_mean,
        amortized_mean,
        difference: per_call_mean - amortized_mean,
        overhead_p50,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_advances_and_calibrates() {
        let (cal, samples) = calibrate(20, 1000);
        assert_eq!(samples.len(), 1000);
        assert!(cal.hz > 1e6, "{cal:?}");
        assert!(cal.relative_difference.abs() < 0.05, "{cal:?}");
        assert!(cal.overhead.min <= cal.overhead.p50);
        let check = amortized_check(200, cal.overhead.p50);
        assert!(check.amortized_mean > 0.0);
    }
}
