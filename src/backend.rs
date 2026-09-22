//! Compute backends — one trait, no per-vendor code paths in callers.
//!
//! PH-101 Task 1. Hardware differences live behind [`ComputeBackend`]; semantic
//! code never branches on `#[cfg(rocm)]` / `#[cfg(cuda)]` (there are none in
//! this crate outside `embed_onnx`'s own EP plumbing). What ships today:
//!
//! - [`BackendKind::Cpu`] — always available, backed by
//!   [`RandomProjectionEmbedder`](crate::embed::RandomProjectionEmbedder).
//! - [`BackendKind::Rocm`] / [`BackendKind::Cuda`] — recognised names that
//!   resolve to the CPU backend with an explicit fallback note until a real
//!   device backend lands. The ONNX ROCm execution provider is a documented
//!   dead end (do not retry); CUDA arrives via `ort` when the weights and the
//!   driver exist on the machine.
//!
//! Selection: explicit argument beats `PHYSIS_BACKEND`, which beats CPU.

use serde::{Deserialize, Serialize};

use crate::embed::{RandomProjectionEmbedder, VectorEmbed};
use crate::model_provider::Capability;

/// Where compute runs. Names, not code paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Cpu,
    Rocm,
    Cuda,
}

impl BackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Rocm => "rocm",
            BackendKind::Cuda => "cuda",
        }
    }

    /// Parse a backend name. Case-insensitive; unknown names are an error,
    /// never a silent CPU.
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name.trim().to_lowercase().as_str() {
            "cpu" => Ok(BackendKind::Cpu),
            "rocm" | "hip" => Ok(BackendKind::Rocm),
            "cuda" | "nvidia" => Ok(BackendKind::Cuda),
            other => anyhow::bail!("unknown backend '{other}' (expected cpu|rocm|cuda)"),
        }
    }

    /// `PHYSIS_BACKEND`, when set and non-blank.
    pub fn from_env() -> Option<Self> {
        std::env::var("PHYSIS_BACKEND")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .and_then(|v| Self::parse(&v).ok())
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The single seam hardware plugs into. Object-safe so callers hold
/// `Box<dyn ComputeBackend>` without generics leaking into semantic code.
pub trait ComputeBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn label(&self) -> String;
    fn capabilities(&self) -> Vec<Capability>;
    /// False means "present by name only" — the caller must fall back.
    fn is_available(&self) -> bool;
    /// Embed text when [`Capability::Embeddings`] is declared, else `None`.
    /// `None` is a refusal, not an error: the caller picks another backend.
    fn embed_text(&self, text: &str) -> Option<Vec<f32>>;
}

/// The always-available backend: deterministic random-projection embeddings.
pub struct CpuBackend {
    dim: usize,
    embedder: RandomProjectionEmbedder,
}

impl CpuBackend {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            embedder: RandomProjectionEmbedder::new(dim),
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn label(&self) -> String {
        format!("cpu:random-projection:{}", self.dim)
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::Embeddings]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
        Some(self.embedder.embed(text))
    }
}

impl VectorEmbed for CpuBackend {
    fn embed(&self, text: &str) -> Vec<f32> {
        self.embedder.embed(text)
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

/// A resolved backend plus the honesty note when it is not what was asked.
pub struct ResolvedBackend {
    pub backend: Box<dyn ComputeBackend>,
    /// `Some` when the requested kind fell back to CPU and why.
    pub fallback_note: Option<String>,
}

/// Resolve `preferred`, else `PHYSIS_BACKEND`, else CPU. Non-CPU kinds fall
/// back to CPU with a note — never an error, never a silent swap.
pub fn select_backend(preferred: Option<BackendKind>) -> ResolvedBackend {
    let want = preferred.or_else(BackendKind::from_env).unwrap_or(BackendKind::Cpu);
    match want {
        BackendKind::Cpu => ResolvedBackend {
            backend: Box::new(CpuBackend::new(384)),
            fallback_note: None,
        },
        BackendKind::Rocm => ResolvedBackend {
            backend: Box::new(CpuBackend::new(384)),
            fallback_note: Some(
                "rocm requested but no ROCm device backend is wired yet \
                 (ONNX ROCm EP is a documented dead end); using CPU"
                    .to_string(),
            ),
        },
        BackendKind::Cuda => ResolvedBackend {
            backend: Box::new(CpuBackend::new(384)),
            fallback_note: Some(
                "cuda requested but no CUDA device backend is wired yet; using CPU".to_string(),
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_parse_and_round_trip() {
        assert_eq!(BackendKind::parse("cpu").unwrap(), BackendKind::Cpu);
        assert_eq!(BackendKind::parse("ROCM").unwrap(), BackendKind::Rocm);
        assert_eq!(BackendKind::parse("cuda").unwrap(), BackendKind::Cuda);
        assert_eq!(BackendKind::parse("hip").unwrap(), BackendKind::Rocm);
        assert!(BackendKind::parse("tpu").is_err());
        assert_eq!(BackendKind::Cpu.as_str(), "cpu");
    }

    #[test]
    fn cpu_backend_is_available_and_deterministic() {
        let b = CpuBackend::new(64);
        assert!(b.is_available());
        assert_eq!(b.kind(), BackendKind::Cpu);
        assert_eq!(b.dimension(), 64);
        assert_eq!(b.embed("hello world"), b.embed("hello world"));
        assert!(b.capabilities().contains(&Capability::Embeddings));
    }

    #[test]
    fn non_cpu_kinds_fall_back_loudly() {
        let r = select_backend(Some(BackendKind::Cuda));
        assert_eq!(r.backend.kind(), BackendKind::Cpu);
        assert!(r.fallback_note.is_some());
        let r = select_backend(Some(BackendKind::Cpu));
        assert!(r.fallback_note.is_none());
    }
}
