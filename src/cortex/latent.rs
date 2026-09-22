//! The canonical Physis latent (PH-101 Task 4).
//!
//! One latent space with four named projections — `W_e` in, `W_d` out, `W_c`
//! centering, `W_g` gating — so encoder, decoder and gate agree on what a
//! vector *means* instead of each assuming compatibility. Weights are
//! seeded-deterministic (no training here); identity/linear/MLP forms are
//! chosen per projection and benchmarked, not assumed.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// A vector in the canonical latent space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysisLatent {
    pub dim: usize,
    pub vec: Vec<f32>,
}

impl PhysisLatent {
    pub fn new(vec: Vec<f32>) -> Self {
        let dim = vec.len();
        Self { dim, vec }
    }
}

/// One projection's form. Identity for shape-preserving passes, Linear for
/// learned-size maps, Mlp for the two shapes a linear map cannot separate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectionKind {
    Identity,
    Linear,
    Mlp { hidden: usize },
}

/// A seeded-deterministic projection matrix (pair). No training: weights come
/// from `seed`, reproducibly, so two runs agree bit-for-bit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projection {
    pub kind: ProjectionKind,
    pub rows: usize,
    pub cols: usize,
    pub seed: u64,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl Projection {
    pub fn new(kind: ProjectionKind, rows: usize, cols: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = match kind {
            ProjectionKind::Identity => 0.0,
            ProjectionKind::Linear => (2.0 / cols as f32).sqrt(),
            ProjectionKind::Mlp { .. } => (2.0 / cols as f32).sqrt(),
        };
        let weights = (0..rows * cols)
            .map(|_| rng.gen_range(-scale..=scale))
            .collect();
        let bias = (0..rows).map(|_| rng.gen_range(-scale..=scale)).collect();
        Self { kind, rows, cols, seed, weights, bias }
    }

    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.cols, "projection input dim mismatch");
        match &self.kind {
            ProjectionKind::Identity => {
                assert_eq!(self.rows, self.cols, "identity needs square dims");
                input.to_vec()
            }
            ProjectionKind::Linear => self.linear(input),
            ProjectionKind::Mlp { hidden } => {
                let mid = Projection::new(ProjectionKind::Linear, *hidden, self.cols, self.seed ^ 0x9e37);
                let out = Projection::new(ProjectionKind::Linear, self.rows, *hidden, self.seed ^ 0x51f3);
                let h: Vec<f32> = mid.linear(input).into_iter().map(|x| x.max(0.0)).collect();
                out.linear(&h)
            }
        }
    }

    fn linear(&self, input: &[f32]) -> Vec<f32> {
        (0..self.rows)
            .map(|r| {
                self.bias[r]
                    + (0..self.cols).map(|c| self.weights[r * self.cols + c] * input[c]).sum::<f32>()
            })
            .collect()
    }
}

/// The four canonical projections. `w_e` maps encoder output into the latent
/// space, `w_d` maps back out, `w_c` recenters, `w_g` gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentProjections {
    pub w_e: Projection,
    pub w_d: Projection,
    pub w_c: Projection,
    pub w_g: Projection,
}

impl LatentProjections {
    /// Seeded family: same seed, same matrices, every run.
    pub fn seeded(encoder_dim: usize, latent_dim: usize, seed: u64) -> Self {
        Self {
            w_e: Projection::new(ProjectionKind::Linear, latent_dim, encoder_dim, seed),
            w_d: Projection::new(ProjectionKind::Linear, encoder_dim, latent_dim, seed ^ 0x1),
            w_c: Projection::new(ProjectionKind::Identity, latent_dim, latent_dim, seed ^ 0x2),
            w_g: Projection::new(ProjectionKind::Linear, latent_dim, latent_dim, seed ^ 0x3),
        }
    }

    /// Encode into the latent space, then center: `W_c(W_e(x) − mean)`.
    /// Centering is parameter-free by design — a learned center would smuggle
    /// corpus bias into the canonical space.
    pub fn encode(&self, input: &[f32]) -> PhysisLatent {
        let mut v = self.w_e.apply(input);
        let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
        for x in &mut v {
            *x -= mean;
        }
        PhysisLatent::new(self.w_c.apply(&v))
    }

    /// Decode back out: `W_d(latent)`.
    pub fn decode(&self, latent: &PhysisLatent) -> Vec<f32> {
        self.w_d.apply(&latent.vec)
    }

    /// Gate: `sigmoid(W_g(latent)) ⊙ latent`. Values near 0 suppress
    /// dimensions the gate distrusts; near 1 passes them through.
    pub fn gate(&self, latent: &PhysisLatent) -> PhysisLatent {
        let g = self.w_g.apply(&latent.vec);
        PhysisLatent::new(
            latent
                .vec
                .iter()
                .zip(g)
                .map(|(x, g)| x * (1.0 / (1.0 + (-g).exp())))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_passes_through_exactly() {
        let p = Projection::new(ProjectionKind::Identity, 4, 4, 7);
        let x = vec![1.0, -2.0, 3.5, 0.0];
        assert_eq!(p.apply(&x), x);
    }

    #[test]
    fn seeded_projections_are_reproducible_and_seed_sensitive() {
        let a = LatentProjections::seeded(8, 4, 42);
        let b = LatentProjections::seeded(8, 4, 42);
        let c = LatentProjections::seeded(8, 4, 43);
        let x = vec![0.5; 8];
        assert_eq!(a.encode(&x).vec, b.encode(&x).vec);
        assert_ne!(a.encode(&x).vec, c.encode(&x).vec);
        assert_eq!(a.encode(&x).dim, 4);
        assert_eq!(a.decode(&a.encode(&x)).len(), 8);
    }

    #[test]
    fn centering_removes_the_mean() {
        let p = LatentProjections::seeded(6, 6, 1);
        let x = vec![10.0; 6];
        let latent = p.encode(&x);
        let mean = latent.vec.iter().sum::<f32>() / latent.vec.len() as f32;
        assert!(mean.abs() < 1e-5, "centered latent must have ~zero mean");
    }

    #[test]
    fn gate_suppresses_toward_zero_and_keeps_shape() {
        let p = LatentProjections::seeded(4, 4, 9);
        let latent = PhysisLatent::new(vec![1.0, -1.0, 0.5, 2.0]);
        let gated = p.gate(&latent);
        assert_eq!(gated.dim, 4);
        for (x, g) in latent.vec.iter().zip(gated.vec.iter()) {
            assert!(g.abs() <= x.abs(), "gate must only suppress");
        }
    }

    #[test]
    fn mlp_form_runs_at_declared_dims() {
        let p = Projection::new(ProjectionKind::Mlp { hidden: 8 }, 3, 5, 11);
        assert_eq!(p.apply(&[1.0; 5]).len(), 3);
    }
}
