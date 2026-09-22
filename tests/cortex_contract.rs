//! Cortex contract (PH-101, Tasks 1+2+8 partial).
//!
//! Covers the surface that exists: backend names resolve, CPU embeds
//! deterministically, discovery always reports CPU, and non-CPU requests fall
//! back loudly. Encoder/latent/retrieval/router tasks are not built yet, so
//! no contract here claims them.

use physis_core::backend::{select_backend, BackendKind};
use physis_core::devices::{parse_nvidia_smi_csv, HardwareDiscovery};
use physis_core::model_provider::Capability;

#[test]
fn backend_names_are_total_and_honest() {
    assert_eq!(BackendKind::parse("cpu").unwrap(), BackendKind::Cpu);
    assert_eq!(BackendKind::parse("rocm").unwrap(), BackendKind::Rocm);
    assert_eq!(BackendKind::parse("cuda").unwrap(), BackendKind::Cuda);
    assert!(BackendKind::parse("tpu").is_err());
}

#[test]
fn cpu_backend_embeds_and_declares_embeddings() {
    let resolved = select_backend(Some(BackendKind::Cpu));
    assert_eq!(resolved.backend.kind(), BackendKind::Cpu);
    assert!(resolved.fallback_note.is_none());
    assert!(resolved.backend.is_available());
    assert!(resolved.backend.capabilities().contains(&Capability::Embeddings));
    let a = resolved.backend.embed_text("the spindle overheated").expect("cpu embeds");
    let b = resolved.backend.embed_text("the spindle overheated").expect("cpu embeds");
    assert!(!a.is_empty());
    assert_eq!(a, b, "same input must embed identically");
}

#[test]
fn requested_gpu_falls_back_with_a_note_not_silently() {
    for kind in [BackendKind::Rocm, BackendKind::Cuda] {
        let resolved = select_backend(Some(kind));
        assert_eq!(resolved.backend.kind(), BackendKind::Cpu);
        assert!(
            resolved.fallback_note.is_some(),
            "{kind} must fall back loudly until a device backend lands"
        );
    }
}

#[test]
fn discovery_always_reports_cpu() {
    let devices = HardwareDiscovery::scan();
    assert!(
        devices.iter().any(|d| d.kind == BackendKind::Cpu),
        "CPU must always be present"
    );
    // best_kind is a pure function of the scan: CPU-only scan selects CPU.
    let cpu_only: Vec<_> = devices
        .iter()
        .filter(|d| d.kind == BackendKind::Cpu)
        .cloned()
        .collect();
    assert_eq!(HardwareDiscovery::best_kind(&cpu_only), BackendKind::Cpu);
}

#[test]
fn synthetic_nvidia_smi_parses() {
    let devices =
        parse_nvidia_smi_csv("NVIDIA GeForce RTX 4090, 24564 MiB\nnot a gpu line\n");
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].kind, BackendKind::Cuda);
    assert_eq!(devices[0].memory_bytes, Some(24564 * 1024 * 1024));
}
