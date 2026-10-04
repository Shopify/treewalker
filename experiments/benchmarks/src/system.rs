//! Process memory measurements for benchmark reporting.

/// Get process RSS in kilobytes.
///
/// Linux reports current RSS; macOS reports peak RSS.
pub fn get_rss_kb() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    return rest
                        .trim()
                        .trim_end_matches(" kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                }
            }
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        unsafe {
            let mut info: libc::rusage = std::mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &raw mut info) == 0 {
                return info.ru_maxrss as usize / 1024;
            }
        }
        0
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}
