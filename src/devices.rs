//! Device discovery — what compute exists on this machine.
//!
//! PH-101 Task 2. Three sources, no guessing:
//!
//! - CPU is always reported (source `cpu-always`).
//! - NVIDIA via `nvidia-smi --query-gpu=name,memory.total
//!   --format=csv,noheader` (source `nvidia-smi`); absent binary or failing
//!   command means no NVIDIA device, not an error.
//! - AMD via `/sys/class/drm` (source `sysfs`): cards whose PCI vendor is
//!   AMD (0x1002) with `mem_info_vram_total` when readable. `rocm-smi` output
//!   is deliberately not parsed — its format is unstable across versions and
//!   sysfs already answers "is AMD graphics hardware present".
//!
//! [`HardwareDiscovery::scan`] never fails: an empty extra-device list just
//! means this box is CPU-only.

use serde::{Deserialize, Serialize};

use crate::backend::BackendKind;

/// One compute device found on this machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub kind: BackendKind,
    /// Human name: `nvidia-smi` GPU name, DRM card id, or `"cpu"`.
    pub name: String,
    /// Total memory in bytes, when the source reports it.
    pub memory_bytes: Option<u64>,
    /// Extra identifier (CUDA compute capability, PCI device id…).
    pub detail: Option<String>,
    /// Where this record came from: `cpu-always`, `nvidia-smi`, `sysfs`.
    pub source: String,
}

/// Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`.
///
/// Expected line shape: `NVIDIA GeForce RTX 4090, 24564 MiB`. Malformed lines
/// are skipped, never fatal.
pub fn parse_nvidia_smi_csv(output: &str) -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, ',');
        let (Some(name), Some(mem)) = (parts.next(), parts.next()) else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        out.push(DeviceInfo {
            kind: BackendKind::Cuda,
            name: name.to_string(),
            memory_bytes: parse_mib(mem.trim()),
            detail: None,
            source: "nvidia-smi".to_string(),
        });
    }
    out
}

fn parse_mib(s: &str) -> Option<u64> {
    let num: f64 = s
        .strip_suffix("MiB")
        .unwrap_or(s)
        .trim()
        .parse()
        .ok()?;
    if num < 0.0 {
        return None;
    }
    Some((num * 1024.0 * 1024.0) as u64)
}

fn read_trim(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

/// AMD cards via sysfs: `/sys/class/drm/cardN/device/vendor == 0x1002`.
fn scan_sysfs_drm() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return out;
    };
    let mut cards: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("card") && !n.contains('-'))
        .collect();
    cards.sort();
    cards.dedup();
    for card in cards {
        let dev = std::path::PathBuf::from("/sys/class/drm").join(&card).join("device");
        let vendor = read_trim(&dev.join("vendor"));
        if vendor.as_deref() != Some("0x1002") {
            continue;
        }
        let device_id = read_trim(&dev.join("device"));
        let vram = read_trim(&dev.join("mem_info_vram_total"))
            .and_then(|s| s.parse::<u64>().ok());
        out.push(DeviceInfo {
            kind: BackendKind::Rocm,
            name: card,
            memory_bytes: vram,
            detail: device_id,
            source: "sysfs".to_string(),
        });
    }
    out
}

fn scan_nvidia_smi() -> Vec<DeviceInfo> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) if o.status.success() => parse_nvidia_smi_csv(&String::from_utf8_lossy(&o.stdout)),
        _ => Vec::new(),
    }
}

/// Find this machine's compute devices. Never fails; CPU is always present.
pub struct HardwareDiscovery;

impl HardwareDiscovery {
    pub fn scan() -> Vec<DeviceInfo> {
        let mut devices = vec![DeviceInfo {
            kind: BackendKind::Cpu,
            name: "cpu".to_string(),
            memory_bytes: None,
            detail: None,
            source: "cpu-always".to_string(),
        }];
        devices.extend(scan_nvidia_smi());
        devices.extend(scan_sysfs_drm());
        devices
    }

    /// The best backend kind the scan justifies: CUDA > ROCm > CPU.
    pub fn best_kind(devices: &[DeviceInfo]) -> BackendKind {
        if devices.iter().any(|d| d.kind == BackendKind::Cuda) {
            BackendKind::Cuda
        } else if devices.iter().any(|d| d.kind == BackendKind::Rocm) {
            BackendKind::Rocm
        } else {
            BackendKind::Cpu
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_csv_parses_name_and_memory() {
        let devs = parse_nvidia_smi_csv(
            "NVIDIA GeForce RTX 4090, 24564 MiB\nTesla T4, 15360 MiB\n",
        );
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(devs[0].memory_bytes, Some(24564 * 1024 * 1024));
        assert_eq!(devs[0].kind, BackendKind::Cuda);
        assert_eq!(devs[0].source, "nvidia-smi");
    }

    #[test]
    fn nvidia_csv_skips_garbage_lines() {
        let devs = parse_nvidia_smi_csv("garbage without comma\n, 1024 MiB\nGood Card, 8192 MiB\n");
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].name, "Good Card");
    }

    #[test]
    fn scan_always_reports_cpu() {
        let devs = HardwareDiscovery::scan();
        assert!(devs.iter().any(|d| d.kind == BackendKind::Cpu && d.source == "cpu-always"));
        // No device backend is wired yet, so selection-relevant kinds stay CPU
        // unless real hardware was found by the scan itself.
        let _ = HardwareDiscovery::best_kind(&devs);
    }

    #[test]
    fn best_kind_prefers_cuda_then_rocm() {
        let cpu = DeviceInfo {
            kind: BackendKind::Cpu,
            name: "cpu".to_string(),
            memory_bytes: None,
            detail: None,
            source: "cpu-always".to_string(),
        };
        assert_eq!(
            HardwareDiscovery::best_kind(std::slice::from_ref(&cpu)),
            BackendKind::Cpu
        );
        let rocm = DeviceInfo { kind: BackendKind::Rocm, ..cpu.clone() };
        assert_eq!(HardwareDiscovery::best_kind(&[cpu.clone(), rocm]), BackendKind::Rocm);
        let cuda = DeviceInfo { kind: BackendKind::Cuda, ..cpu.clone() };
        assert_eq!(
            HardwareDiscovery::best_kind(&[cpu, cuda]),
            BackendKind::Cuda
        );
    }
}
