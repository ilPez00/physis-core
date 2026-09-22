//! Shared encoder + probe heads (PH-101 Task 3).
//!
//! One encoder invocation per text; [`NeuralRole`] heads read the same
//! [`CortexOutput`] as cosine probes over caller-supplied exemplars. Adapters
//! and probes first — no joint training, no second model. The encoder backend
//! is any [`VectorEmbed`](crate::embed::VectorEmbed); the honest default is
//! the CPU backend from [`crate::backend`].

use serde::{Deserialize, Serialize};

use crate::backend::{select_backend, ComputeBackend, CpuBackend};
use crate::embed::VectorEmbed;
use crate::models::{cosine_sim, Score};

/// Which consumer reads an encoding. One invocation feeds all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NeuralRole {
    Intent,
    Entity,
    Relation,
}

/// One shared encoder invocation, tagged for its consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortexOutput {
    pub embedding: Vec<f32>,
    pub role: NeuralRole,
    pub backend: String,
}

/// A probe head: a label with exemplar texts, embedded once at build time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeHead {
    pub role: NeuralRole,
    pub label: String,
    pub exemplars: Vec<Vec<f32>>,
}

/// The shared encoder. Embeds once; heads probe often.
pub struct SharedEncoder {
    backend: Box<dyn ComputeBackend>,
}

impl SharedEncoder {
    /// CPU default — deterministic, always available.
    pub fn cpu(dim: usize) -> Self {
        Self {
            backend: Box::new(CpuBackend::new(dim)),
        }
    }

    /// Resolve via `PHYSIS_BACKEND` / scan-best, falling back loudly.
    pub fn auto() -> Self {
        Self {
            backend: select_backend(None).backend,
        }
    }

    pub fn backend_label(&self) -> String {
        self.backend.label()
    }

    /// The single invocation everything downstream shares.
    pub fn encode(&self, text: &str, role: NeuralRole) -> CortexOutput {
        let embedding = self
            .backend
            .embed_text(text)
            .unwrap_or_else(|| crate::embed::RandomProjectionEmbedder::new(384).embed(text));
        CortexOutput {
            embedding,
            role,
            backend: self.backend.label(),
        }
    }

    /// Build a probe head by embedding its exemplars once.
    pub fn probe_head(&self, role: NeuralRole, label: &str, exemplars: &[&str]) -> ProbeHead {
        ProbeHead {
            role,
            label: label.to_string(),
            exemplars: exemplars
                .iter()
                .map(|e| {
                    self.backend
                        .embed_text(e)
                        .unwrap_or_else(|| crate::embed::RandomProjectionEmbedder::new(384).embed(e))
                })
                .collect(),
        }
    }

    /// Rank head labels by best-exemplar cosine similarity.
    pub fn probe(&self, output: &CortexOutput, heads: &[ProbeHead]) -> Vec<(String, Score)> {
        let mut scored: Vec<(String, Score)> = heads
            .iter()
            .filter(|h| h.role == output.role)
            .map(|h| {
                let best = h
                    .exemplars
                    .iter()
                    .map(|e| cosine_sim(&output.embedding, e))
                    .fold(f32::NEG_INFINITY, f32::max);
                (h.label.clone(), best)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_invocation_feeds_all_roles() {
        let enc = SharedEncoder::cpu(64);
        for role in [NeuralRole::Intent, NeuralRole::Entity, NeuralRole::Relation] {
            let out = enc.encode("the pump overheated", role);
            assert_eq!(out.role, role);
            assert!(!out.embedding.is_empty());
        }
        // Same text, same backend: identical bytes.
        let a = enc.encode("the pump overheated", NeuralRole::Intent);
        let b = enc.encode("the pump overheated", NeuralRole::Intent);
        assert_eq!(a.embedding, b.embedding);
    }

    #[test]
    fn probe_ranks_matching_head_first() {
        let enc = SharedEncoder::cpu(64);
        let hot = enc.probe_head(NeuralRole::Intent, "overheat", &["the pump overheated"]);
        let cold = enc.probe_head(NeuralRole::Intent, "routine", &["scheduled maintenance complete"]);
        let out = enc.encode("the pump overheated", NeuralRole::Intent);
        let ranked = enc.probe(&out, &[cold, hot]);
        assert_eq!(ranked[0].0, "overheat");
        assert!(ranked[0].1 > 0.99, "identical text must near-match");
    }

    #[test]
    fn probe_ignores_other_roles() {
        let enc = SharedEncoder::cpu(64);
        let entity = enc.probe_head(NeuralRole::Entity, "pump", &["the pump overheated"]);
        let out = enc.encode("the pump overheated", NeuralRole::Intent);
        assert!(enc.probe(&out, &[entity]).is_empty());
    }
}
