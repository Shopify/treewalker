//! Hardware counters through `perf_event`, on Linux with the `pmu` feature.
//!
//! One group of five raw core events, programmed per CPU because the generic perf
//! cache events map to LLC events the GCE `STANDARD` PMU level does not expose, plus
//! reference cycles and two software events. The group is read at block boundaries,
//! never per group. Counting is per thread and user-space only, which
//! `perf_event_paranoid=2` allows. Thread context switches from `getrusage` are
//! recorded next to the group, because the software event cannot see the kernel.

use serde::Serialize;

/// One event of the group: its column name, the vendor's event name and how it is
/// programmed.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EventDef {
    /// Column in the `hw` table.
    pub column: &'static str,
    /// The vendor's event name.
    pub name: &'static str,
    /// `raw` (a PMU event code), `hardware` (a generic perf event) or `software`.
    pub kind: &'static str,
    /// The raw code, or the generic perf event number.
    pub code: u64,
}

const fn raw(column: &'static str, name: &'static str, code: u64) -> EventDef {
    EventDef {
        column,
        name,
        kind: "raw",
        code,
    }
}

const SOFTWARE: [EventDef; 2] = [
    EventDef {
        column: "context_switches",
        name: "PERF_COUNT_SW_CONTEXT_SWITCHES",
        kind: "software",
        code: 3,
    },
    EventDef {
        column: "task_clock_ns",
        name: "PERF_COUNT_SW_TASK_CLOCK",
        kind: "software",
        code: 1,
    },
];

/// Intel Emerald Rapids and Granite Rapids (C4).
///
/// GCE's PMU tables give both the same codes. Raw codes are `umask << 8 | event`; reference cycles
/// are the fixed counter behind `PERF_COUNT_HW_REF_CPU_CYCLES`.
pub const EMERALD_RAPIDS: [EventDef; 8] = [
    raw("cycles", "CPU_CLK_UNHALTED.THREAD_P", 0x003C),
    raw("instructions", "INST_RETIRED.ANY_P", 0x00C0),
    raw("branch_misses", "BR_MISP_RETIRED.ALL_BRANCHES", 0x00C5),
    raw("l1d_load_misses", "MEM_LOAD_RETIRED.L1_MISS", 0x08D1),
    raw(
        "l2_data_read_misses",
        "L2_RQSTS.DEMAND_DATA_RD_MISS",
        0x2124,
    ),
    EventDef {
        column: "ref_cycles",
        name: "CPU_CLK_UNHALTED.REF_TSC",
        kind: "hardware",
        code: 9,
    },
    SOFTWARE[0],
    SOFTWARE[1],
];

/// Google Axion, Arm Neoverse V2 (C4A).
///
/// A C4A guest exposes four programmable counters and the cycle counter
/// (`hw perfevents: ... 5 (0,8000000f) counters available`), and the kernel rejects
/// a group that needs more (`perf_event_open` gives `EINVAL`). `CPU_CYCLES` takes the
/// cycle counter and the four other core events take the rest, so there are no
/// reference cycles: `CNT_CYCLES` would count at the generic timer's frequency, which
/// `cntvct_el0` already gives every block, and C4A has no turbo setting.
pub const NEOVERSE_V2: [EventDef; 7] = [
    raw("cycles", "CPU_CYCLES", 0x11),
    raw("instructions", "INST_RETIRED", 0x08),
    raw("branch_misses", "BR_MIS_PRED", 0x10),
    raw("l1d_load_misses", "L1D_CACHE_LMISS_RD", 0x39),
    raw("l2_data_read_misses", "L2D_CACHE_LMISS_RD", 0x4009),
    SOFTWARE[0],
    SOFTWARE[1],
];

/// Every column of the `hw` table that holds an event count.
pub const COLUMNS: [&str; 8] = [
    "cycles",
    "instructions",
    "branch_misses",
    "l1d_load_misses",
    "l2_data_read_misses",
    "ref_cycles",
    "context_switches",
    "task_clock_ns",
];

/// Intel family-6 models with a mapped event table: Emerald Rapids (0xCF) and
/// Granite Rapids (0xAD), the CPUs a C4 VM lands on.
const INTEL_MODELS: [(u32, &str); 2] = [(0xCF, "emerald-rapids"), (0xAD, "granite-rapids")];

/// The value of a `/proc/cpuinfo` field on the first CPU.
fn cpuinfo_field<'a>(cpuinfo: &'a str, key: &str) -> Option<&'a str> {
    cpuinfo
        .lines()
        .find(|l| l.split(':').next().is_some_and(|k| k.trim() == key))
        .and_then(|l| l.split_once(':'))
        .map(|(_, v)| v.trim())
}

/// The event table for a CPU, from its `/proc/cpuinfo`: Intel by family and model,
/// Arm by part number. Anything unmapped is unsupported.
pub fn events_for(
    cpuinfo: &str,
    arch: &str,
) -> Result<(&'static str, &'static [EventDef]), String> {
    if arch == "x86_64" && cpuinfo_field(cpuinfo, "vendor_id") == Some("GenuineIntel") {
        let family = cpuinfo_field(cpuinfo, "cpu family").and_then(|v| v.parse::<u32>().ok());
        let model = cpuinfo_field(cpuinfo, "model").and_then(|v| v.parse::<u32>().ok());
        if family == Some(6)
            && let Some((_, name)) = INTEL_MODELS.iter().find(|(m, _)| Some(*m) == model)
        {
            return Ok((name, &EMERALD_RAPIDS));
        }
        return Err(format!(
            "no event table for Intel family {family:?} model {model:?} (mapped: family 6, models 0xCF, 0xAD)"
        ));
    }
    if arch == "aarch64" && cpuinfo_field(cpuinfo, "CPU part") == Some("0xd4f") {
        return Ok(("neoverse-v2", &NEOVERSE_V2));
    }
    Err("no event table for this CPU (mapped: Emerald Rapids, Granite Rapids, Neoverse V2)".into())
}

/// The event table for this CPU, if it has one.
pub fn events_for_host() -> Result<(&'static str, &'static [EventDef]), String> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    events_for(&cpuinfo, std::env::consts::ARCH)
}

/// Counter deltas over one block, with the group's enabled and running times.
#[derive(Debug, Clone, Default)]
pub struct Reading {
    /// One value per [`COLUMNS`] entry; `None` where the table has no such event.
    pub values: [Option<u64>; 8],
    pub time_enabled: u64,
    pub time_running: u64,
    /// Thread context switches from `getrusage`: voluntary, involuntary.
    pub rusage_nvcsw: u64,
    pub rusage_nivcsw: u64,
}

impl Reading {
    fn delta(&self, earlier: &Self) -> Self {
        let mut values = [None; 8];
        for (i, v) in values.iter_mut().enumerate() {
            *v = self.values[i]
                .zip(earlier.values[i])
                .map(|(a, b)| a.wrapping_sub(b));
        }
        Self {
            values,
            time_enabled: self.time_enabled - earlier.time_enabled,
            time_running: self.time_running - earlier.time_running,
            rusage_nvcsw: self.rusage_nvcsw - earlier.rusage_nvcsw,
            rusage_nivcsw: self.rusage_nivcsw - earlier.rusage_nivcsw,
        }
    }

    /// Whether the whole group was scheduled for the whole interval.
    pub const fn complete(&self) -> bool {
        self.time_running == self.time_enabled
    }
}

/// Context switches of the calling thread.
fn rusage_switches() -> (u64, u64) {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        #[cfg(target_os = "linux")]
        let who = libc::RUSAGE_THREAD;
        #[cfg(target_os = "macos")]
        let who = libc::RUSAGE_SELF;
        // SAFETY: an all-zero rusage is valid, and getrusage fills it.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: ru is a valid rusage.
        if unsafe { libc::getrusage(who, &raw mut ru) } == 0 {
            return (ru.ru_nvcsw as u64, ru.ru_nivcsw as u64);
        }
        (0, 0)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        (0, 0)
    }
}

/// An open event group, or the reason it is not available.
pub struct Pmu {
    #[cfg(all(target_os = "linux", feature = "pmu"))]
    inner: linux::Group,
    pub table: &'static str,
    pub events: &'static [EventDef],
    last: Reading,
}

impl std::fmt::Debug for Pmu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pmu")
            .field("table", &self.table)
            .finish_non_exhaustive()
    }
}

impl Pmu {
    /// Open the host's event group and enable it.
    pub fn open() -> Result<Self, String> {
        let (table, events) = events_for_host()?;
        #[cfg(all(target_os = "linux", feature = "pmu"))]
        {
            let inner = linux::Group::open(events).map_err(|e| format!("perf_event_open: {e}"))?;
            let mut pmu = Self {
                inner,
                table,
                events,
                last: Reading::default(),
            };
            pmu.last = pmu.read_raw()?;
            Ok(pmu)
        }
        #[cfg(not(all(target_os = "linux", feature = "pmu")))]
        {
            let _ = (table, events);
            Err("hardware counters need Linux and the pmu feature".into())
        }
    }

    #[cfg_attr(
        not(all(target_os = "linux", feature = "pmu")),
        expect(
            clippy::unused_self,
            clippy::unnecessary_wraps,
            clippy::needless_pass_by_ref_mut,
            reason = "the same signature with and without counters"
        )
    )]
    fn read_raw(&mut self) -> Result<Reading, String> {
        let (nvcsw, nivcsw) = rusage_switches();
        #[cfg(all(target_os = "linux", feature = "pmu"))]
        {
            let (values, enabled, running) = self.inner.read().map_err(|e| e.to_string())?;
            let mut out = [None; 8];
            for (def, v) in self.events.iter().zip(values) {
                if let Some(i) = COLUMNS.iter().position(|c| *c == def.column) {
                    out[i] = Some(v);
                }
            }
            Ok(Reading {
                values: out,
                time_enabled: enabled,
                time_running: running,
                rusage_nvcsw: nvcsw,
                rusage_nivcsw: nivcsw,
            })
        }
        #[cfg(not(all(target_os = "linux", feature = "pmu")))]
        {
            Ok(Reading {
                rusage_nvcsw: nvcsw,
                rusage_nivcsw: nivcsw,
                ..Reading::default()
            })
        }
    }

    /// Mark the start of an interval.
    pub fn mark(&mut self) -> Result<(), String> {
        self.last = self.read_raw()?;
        Ok(())
    }

    /// The deltas since the last [`Self::mark`].
    pub fn since_mark(&mut self) -> Result<Reading, String> {
        let now = self.read_raw()?;
        Ok(now.delta(&self.last))
    }
}

/// Open the group, run a short loop, and check that every event was scheduled for
/// the whole interval. `preflight` fails a run whose group would be multiplexed.
pub fn check() -> Result<(&'static str, Reading), String> {
    let mut pmu = Pmu::open()?;
    pmu.mark()?;
    let mut x = 1u64;
    for i in 0..10_000_000u64 {
        x = std::hint::black_box(x.wrapping_mul(3).wrapping_add(i));
    }
    let r = pmu.since_mark()?;
    if !r.complete() {
        return Err(format!(
            "the event group was multiplexed: time_running {} < time_enabled {}",
            r.time_running, r.time_enabled
        ));
    }
    // Cycles, instructions and, where the table has them, reference cycles must
    // count; misses may be zero.
    for column in ["cycles", "instructions", "ref_cycles"] {
        if !pmu.events.iter().any(|e| e.column == column) {
            continue;
        }
        let i = COLUMNS.iter().position(|c| *c == column).unwrap_or(0);
        if r.values[i].is_none_or(|v| v == 0) {
            return Err(format!("{column} counted nothing: {:?}", r.values));
        }
    }
    Ok((pmu.table, r))
}

#[cfg(all(target_os = "linux", feature = "pmu"))]
mod linux {
    use perf_event::events::{Hardware, Software};
    use perf_event::{Builder, Counter};

    use super::EventDef;

    const PERF_TYPE_RAW: u32 = 4;

    pub struct Group {
        group: perf_event::Group,
        counters: Vec<Counter>,
    }

    impl Group {
        pub fn open(events: &[EventDef]) -> std::io::Result<Self> {
            let mut group = perf_event::Group::new()?;
            let mut counters = Vec::with_capacity(events.len());
            for def in events {
                let builder = Builder::new().group(&mut group);
                let mut builder = match def.kind {
                    "hardware" => builder.kind(Hardware::REF_CPU_CYCLES),
                    "software" if def.code == 3 => builder.kind(Software::CONTEXT_SWITCHES),
                    "software" => builder.kind(Software::TASK_CLOCK),
                    _ => builder,
                };
                if def.kind == "raw" {
                    let attrs = builder.attrs_mut();
                    attrs.type_ = PERF_TYPE_RAW;
                    attrs.config = def.code;
                }
                // Per thread, user space only (the Builder default), never inherited.
                builder.inherit(false);
                counters.push(builder.build()?);
            }
            group.enable()?;
            Ok(Self { group, counters })
        }

        /// Every counter's value, then the group's enabled and running times.
        pub fn read(&mut self) -> std::io::Result<(Vec<u64>, u64, u64)> {
            let counts = self.group.read()?;
            let values = self
                .counters
                .iter()
                .map(|c| counts.get(c).copied().unwrap_or(0))
                .collect();
            Ok((values, counts.time_enabled(), counts.time_running()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intel(model: u32) -> String {
        format!("vendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel\t\t: {model}\nmodel name\t: x\n")
    }

    #[test]
    fn event_tables_map_by_cpu_model() {
        assert_eq!(
            events_for(&intel(0xCF), "x86_64").unwrap().0,
            "emerald-rapids"
        );
        assert_eq!(
            events_for(&intel(0xAD), "x86_64").unwrap().0,
            "granite-rapids"
        );
        assert!(events_for(&intel(0x8F), "x86_64").is_err()); // Sapphire Rapids: unmapped
        assert!(events_for("vendor_id : AuthenticAMD\n", "x86_64").is_err());
        assert_eq!(
            events_for("CPU part\t: 0xd4f\n", "aarch64").unwrap().0,
            "neoverse-v2"
        );
        assert!(events_for("CPU part\t: 0xd40\n", "aarch64").is_err());
    }

    #[test]
    fn arm_group_fits_a_c4a_guest() {
        // The cycle counter plus four programmable counters; software events need none.
        let core: Vec<_> = NEOVERSE_V2.iter().filter(|e| e.kind == "raw").collect();
        assert_eq!(core[0].code, 0x11, "CPU_CYCLES first, on the cycle counter");
        assert!(
            core.len() - 1 <= 4,
            "{} programmable events",
            core.len() - 1
        );
    }
}
