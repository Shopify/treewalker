//! Process memory around each method's load and first predictor, read outside every
//! timed region.
//!
//! - Linux: `VmRSS` and `VmHWM` from `/proc/self/status`. Writing `5` to
//!   `/proc/self/clear_refs` resets `VmHWM` to the current RSS, so the high-water mark
//!   read after a load is that load's own peak.
//! - macOS: `resident_size` and `phys_footprint` from `task_info(TASK_VM_INFO)`, and
//!   `ru_maxrss`. The peak cannot be reset there: a load's peak is its own only when
//!   `ru_maxrss` rose during it, which holds for the first method (unless the cell's
//!   own data peaked higher) and for a later one only when it sets a new high.
//!
//! No counting allocator: it would perturb the load timings measured in the same run.
//! The reads bracket the loads' clocks, never sit inside them.

use serde::Serialize;

/// What the process holds at one moment, in bytes; `None` where the platform has no
/// such reading.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Reading {
    /// Resident set: Linux `VmRSS`, macOS `resident_size`.
    pub rss_bytes: Option<u64>,
    /// Linux `VmHWM`, the peak since the last reset.
    pub hwm_bytes: Option<u64>,
    /// macOS `phys_footprint`, what the system charges the process.
    pub footprint_bytes: Option<u64>,
    /// `ru_maxrss`, the process's lifetime peak resident set.
    pub maxrss_bytes: Option<u64>,
}

/// How the peaks were read on this platform, for the manifest.
pub const SOURCE: &str = if cfg!(target_os = "linux") {
    "linux: VmRSS and VmHWM from /proc/self/status; VmHWM reset by writing 5 to \
     /proc/self/clear_refs before each load, so each load's peak is its own"
} else if cfg!(target_os = "macos") {
    "macos: resident_size and phys_footprint from task_info(TASK_VM_INFO), ru_maxrss; \
     the peak cannot be reset, so a load's peak is its own only when ru_maxrss rose \
     during it (the first method's, unless the cell's data peaked higher)"
} else {
    "none: no memory readings on this platform"
};

/// Reset the high-water mark to the current resident set: Linux only.
pub fn reset_peak() -> bool {
    cfg!(target_os = "linux") && std::fs::write("/proc/self/clear_refs", "5").is_ok()
}

/// The process's memory now.
pub fn read() -> Reading {
    let mut r = Reading::default();
    #[cfg(target_os = "linux")]
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        r.rss_bytes = status_kib(&status, "VmRSS:");
        r.hwm_bytes = status_kib(&status, "VmHWM:");
    }
    #[cfg(target_os = "macos")]
    if let Some(vm) = task_vm_info() {
        r.rss_bytes = Some(vm.resident_size);
        r.footprint_bytes = Some(vm.phys_footprint);
    }
    // The lifetime peak last: it only rises, so it is never below the resident set read
    // above, even when another thread allocates between the two calls.
    r.maxrss_bytes = maxrss();
    r
}

/// A `/proc/self/status` field in kB, as bytes.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn status_kib(status: &str, key: &str) -> Option<u64> {
    let line = status.lines().find(|l| l.starts_with(key))?;
    let kib: u64 = line[key.len()..]
        .trim()
        .strip_suffix("kB")?
        .trim()
        .parse()
        .ok()?;
    Some(kib * 1024)
}

fn maxrss() -> Option<u64> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        // SAFETY: an all-zero rusage is valid, and getrusage fills it.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: ru is a valid rusage.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut ru) } != 0 {
            return None;
        }
        let v = u64::try_from(ru.ru_maxrss).ok()?;
        // Linux reports kB, macOS bytes.
        Some(if cfg!(target_os = "macos") {
            v
        } else {
            v * 1024
        })
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// `task_vm_info` through revision 1 (`<mach/task_info.h>`), which ends at
/// `phys_footprint`; the kernel fills as much as the count asks for.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Default)]
struct TaskVmInfo {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
}

#[cfg(target_os = "macos")]
fn task_vm_info() -> Option<TaskVmInfo> {
    // libc's `mach_task_self` is deprecated in favour of the mach2 crate; the symbol
    // itself is the port the C macro reads.
    unsafe extern "C" {
        static mach_task_self_: libc::mach_port_t;
    }
    const TASK_VM_INFO: libc::task_flavor_t = 22;
    let mut info = TaskVmInfo::default();
    let mut count = (size_of::<TaskVmInfo>() / size_of::<libc::natural_t>()) as u32;
    // SAFETY: `info` is a writable `task_vm_info` prefix of `count` naturals, and
    // `mach_task_self_` is the task's own port, set before main.
    let kr = unsafe {
        libc::task_info(
            mach_task_self_,
            TASK_VM_INFO,
            (&raw mut info).cast(),
            &raw mut count,
        )
    };
    (kr == libc::KERN_SUCCESS).then_some(info)
}

/// One method's load and first predictor.
#[derive(Clone, Debug, Serialize)]
pub struct MethodMemory {
    pub method: String,
    pub variant: String,
    /// What the load built: `forest` (the cell's model), `forest:<load options>`
    /// (a research load), `forest_at_width:<rows>`, `baseline` (the baseline's own
    /// load, its buffers included), or `shared:<load options>` for a method on a
    /// forest already loaded (no load of its own).
    pub load: String,
    /// The high-water mark was reset before the load (Linux).
    pub peak_reset: bool,
    /// `peak_load_bytes` is this load's own peak.
    pub peak_attributable: bool,
    /// The peak resident set during the load, when attributable: Linux `VmHWM`, macOS
    /// `ru_maxrss`.
    pub peak_load_bytes: Option<u64>,
    pub before: Reading,
    pub after_load: Reading,
    /// After the method's first predictor was built: `None` where the load builds
    /// everything (the baselines) or the method has no predictor (the full walk).
    pub after_predictor: Option<Reading>,
}

/// A load in progress: started before the load's clock, finished after it.
pub struct Meter {
    peak_reset: bool,
    before: Reading,
}

impl Meter {
    /// Reset the peak where the platform can, then read: call before the load's clock
    /// starts.
    #[must_use]
    pub fn start() -> Self {
        let peak_reset = reset_peak();
        Self {
            peak_reset,
            before: read(),
        }
    }

    /// Read after the load's clock stops.
    #[must_use]
    pub fn loaded(self, method: &str, variant: &str, load: &str) -> MethodMemory {
        let after = read();
        let (peak_attributable, peak_load_bytes) = if self.peak_reset {
            (after.hwm_bytes.is_some(), after.hwm_bytes)
        } else {
            match (self.before.maxrss_bytes, after.maxrss_bytes) {
                (Some(b), Some(a)) if a > b => (true, Some(a)),
                _ => (false, None),
            }
        };
        MethodMemory {
            method: method.into(),
            variant: variant.into(),
            load: load.into(),
            peak_reset: self.peak_reset,
            peak_attributable,
            peak_load_bytes,
            before: self.before,
            after_load: after,
            after_predictor: None,
        }
    }
}

impl MethodMemory {
    /// Read after the method's first predictor is built.
    pub fn predictor_built(&mut self) {
        self.after_predictor = Some(read());
    }

    /// A method on a forest already loaded: the readings around its predictor only.
    #[must_use]
    pub fn shared(method: &str, variant: &str, load: &str) -> Self {
        let now = read();
        Self {
            method: method.into(),
            variant: variant.into(),
            load: load.into(),
            peak_reset: false,
            peak_attributable: false,
            peak_load_bytes: None,
            before: now,
            after_load: now,
            after_predictor: None,
        }
    }

    /// The same record for another method on the same load (production and the full
    /// walk share the cell's forest).
    #[must_use]
    pub fn renamed(&self, method: &str, variant: &str) -> Self {
        Self {
            method: method.into(),
            variant: variant.into(),
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_fields_parse_as_bytes() {
        let s = "Name:\tsweep_bench\nVmHWM:\t  123456 kB\nVmRSS:\t   65536 kB\n";
        assert_eq!(status_kib(s, "VmHWM:"), Some(123_456 * 1024));
        assert_eq!(status_kib(s, "VmRSS:"), Some(65_536 * 1024));
        assert_eq!(status_kib(s, "VmSwap:"), None);
        assert_eq!(status_kib("VmRSS:\t12 MB\n", "VmRSS:"), None);
    }

    #[test]
    fn readings_see_an_allocation() {
        let before = read();
        if before.rss_bytes.is_none() {
            return; // no readings on this platform
        }
        let m = Meter::start();
        let big: Vec<u8> = vec![1; 64 << 20];
        std::hint::black_box(&big);
        let mut rec = m.loaded("treewalker", "all-on", "forest");
        rec.predictor_built();
        // Linux resets the high-water mark (kernel 4.0 and later).
        assert_eq!(rec.peak_reset, cfg!(target_os = "linux"), "{rec:?}");
        let after = rec.after_load.rss_bytes.unwrap();
        assert!(
            after >= rec.before.rss_bytes.unwrap() + (32 << 20),
            "{rec:?}"
        );
        if rec.peak_reset || rec.peak_attributable {
            assert!(rec.peak_load_bytes.unwrap() >= after, "{rec:?}");
        }
        assert!(rec.after_predictor.is_some());
        drop(big);
    }
}
