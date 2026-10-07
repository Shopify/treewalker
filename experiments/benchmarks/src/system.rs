//! The host description recorded in `run.json`.

use serde_json::{Value, json};

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

fn command(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The CPU's model name.
pub fn cpu_model() -> String {
    if cfg!(target_os = "macos") {
        return command("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default();
    }
    read("/proc/cpuinfo")
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name") || l.starts_with("CPU part"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_string())
        })
        .unwrap_or_default()
}

/// The CPU's identity fields from `/proc/cpuinfo`: vendor, family, model and
/// stepping on x86; implementer and part on Arm. The event table is chosen from them.
fn cpu_id() -> Value {
    let info = read("/proc/cpuinfo").unwrap_or_default();
    let field = |key: &str| {
        info.lines()
            .find(|l| l.split(':').next().is_some_and(|k| k.trim() == key))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
    };
    json!({
        "vendor_id": field("vendor_id"),
        "cpu_family": field("cpu family"),
        "model": field("model"),
        "stepping": field("stepping"),
        "cpu_implementer": field("CPU implementer"),
        "cpu_part": field("CPU part"),
    })
}

/// The CPUs this process may run on.
#[cfg_attr(
    not(target_os = "linux"),
    expect(clippy::missing_const_for_fn, reason = "Linux reads the affinity mask")
)]
fn pinned_cpu() -> Value {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: an all-zero cpu_set_t is valid, and sched_getaffinity fills it.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        // SAFETY: set is a valid cpu_set_t of the size passed.
        let rc = unsafe {
            libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &raw mut set)
        };
        if rc == 0 {
            let cpus: Vec<usize> = (0..libc::CPU_SETSIZE as usize)
                // SAFETY: i is below CPU_SETSIZE.
                .filter(|&i| unsafe { libc::CPU_ISSET(i, &set) })
                .collect();
            return json!(cpus);
        }
        Value::Null
    }
    #[cfg(not(target_os = "linux"))]
    {
        Value::Null
    }
}

/// Peak resident set size of this process, in bytes.
pub fn peak_rss_bytes() -> u64 {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        // SAFETY: an all-zero rusage is valid, and getrusage fills it.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: ru is a valid rusage.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut ru) } == 0 {
            let rss = ru.ru_maxrss as u64;
            return if cfg!(target_os = "macos") {
                rss
            } else {
                rss * 1024
            };
        }
        0
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

/// CPU, frequency governor, SMT, kernel and the build of this binary.
pub fn host() -> Value {
    let governor = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu": cpu_model(),
        "cpu_id": cpu_id(),
        "logical_cpus": std::thread::available_parallelism().map_or(0, usize::from),
        "pinned_cpus": pinned_cpu(),
        "governor": governor,
        "smt_active": read("/sys/devices/system/cpu/smt/active"),
        "perf_event_paranoid": read("/proc/sys/kernel/perf_event_paranoid"),
        "aslr": read("/proc/sys/kernel/randomize_va_space"),
        "kernel": command("uname", &["-r"]),
        "kernel_version": command("uname", &["-v"]),
        "rustc": env!("TW_RUSTC_VERSION"),
        "rustflags": env!("TW_RUSTFLAGS"),
        "profile": env!("TW_PROFILE"),
        "features": features(),
    })
}

/// The Cargo features this binary was built with.
pub fn features() -> Vec<&'static str> {
    let mut f = Vec::new();
    if cfg!(feature = "external-bench") {
        f.push("external-bench");
    }
    if cfg!(feature = "quickscorer-bench") {
        f.push("quickscorer-bench");
    }
    if cfg!(feature = "pmu") {
        f.push("pmu");
    }
    f
}
