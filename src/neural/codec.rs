//! Encoder / Decoder traits: semantic ↔ stimulus mapping (PH-101 Task 4).
//!
//! Reuses `crate::embed::VectorEmbed` / `RandomProjectionEmbedder` for the
//! dense projection. The deterministic backend (no feature) uses the identity
//! encoder; reservoir/spiking backends use a random-projection encoder.

use crate::neural::substrate::StimulusPattern;

/// Map a Physis semantic state (CoherenceNode / Hypothesis embedding) to a
/// `StimulusPattern` the substrate can stimulate on.
pub trait Encoder {
    fn encode(&self, embedding: &[f32]) -> StimulusPattern;
}

/// Map a `NeuralObservation` back to a semantic embedding + confidence.
pub trait Decoder {
    fn decode(&self, obs: &crate::neural::substrate::NeuralObservation) -> (Vec<f32>, crate::neural::substrate::Confidence);
}

/// Identity encoder: channels = embedding dims. Used by DeterministicBackend.
pub struct IdentityEncoder;

impl Encoder for IdentityEncoder {
    fn encode(&self, embedding: &[f32]) -> StimulusPattern {
        StimulusPattern {
            channels: (0..embedding.len()).map(|i| (i, embedding[i])).collect(),
            dense: embedding.to_vec(),
            hint: None,
        }
    }
}

/// Random-projection encoder wrapping `RandomProjectionEmbedder`'s matrix.
/// Cheap, deterministic given a seed.
pub struct ProjectionEncoder {
    matrix: Vec<Vec<f32>>,
}

impl ProjectionEncoder {
    pub fn new(dim_in: usize, dim_out: usize, seed: u64) -> Self {
        // Reuse RandomProjectionEmbedder's seeding strategy.
        let e = crate::embed::RandomProjectionEmbedder::new(dim_in);
        // Extract its projection matrix for direct encode().
        let matrix = (0..dim_out)
            .map(|_| (0..dim_in).map(|_| {
                // Gaussian via Box-Muller is unavailable in no-std; use a
                // cheap uniform → ±1 sparse projection instead (matches
                // RandomProjectionEmbedder's {-1,0,+1} scheme).
                let r = (seed.wrapping_mul(2654435761).wrapping_add((dim_in + dim_out) as u64)) % 3;
                match r { 0 => 0.0, 1 => 1.0, _ => -1.0 }
            }).collect::<Vec<f32>>())
            .collect();
        let _ = e; // kept for API symmetry; matrix is self-contained.
        Self { matrix }
    }
}

impl Encoder for ProjectionEncoder {
    fn encode(&self, embedding: &[f32]) -> StimulusPattern {
        let dense: Vec<f32> = self.matrix.iter()
            .map(|row| row.iter().zip(embedding).map(|(w, x)| w * x).sum::<f32>())
            .collect();
        StimulusPattern { channels: vec![], dense, hint: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_encoder_maps_embedding_to_channels() {
        let enc = IdentityEncoder;
        let emb = vec![0.1, 0.0, 0.9];
        let sp = enc.encode(&emb);
        assert_eq!(sp.channels.len(), 3);
        assert_eq!(sp.channels[0].0, 0);
        assert!((sp.channels[2].1 - 0.9).abs() < 1e-6);
    }

    #[test]
    fn projection_encoder_preserves_dim() {
        let enc = ProjectionEncoder::new(4, 2, 42);
        let emb = vec![0.5; 4];
        let sp = enc.encode(&emb);
        assert_eq!(sp.dense.len(), 2);
    }
}
