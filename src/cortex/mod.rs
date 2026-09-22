//! Physis Cortex — the semantic perception layer (PH-101, Tasks 3–5).
//!
//! Deliberately NOT the world model: the cortex perceives (encode → project →
//! retrieve) while `transform`/`transition`/`query` reason and act. What ships:
//!
//! - [`encoder`]: one shared encoder invocation feeding [`encoder::NeuralRole`]
//!   probe heads (adapters first, no joint training).
//! - [`latent`]: the canonical [`latent::PhysisLatent`] with `W_e/W_d/W_c/W_g`
//!   projections. Weights are seeded-deterministic, not trained — benchmarked,
//!   not assumed.
//! - [`retrieval`]: the progressive refinement ladder over the existing
//!   [`crate::query`] primitives. ColBERT is a named rung that reports
//!   unavailable until weights exist on the machine.
//!
//! No `#[cfg(rocm)]` / `#[cfg(cuda)]` anywhere: compute placement is the
//! [`crate::backend`] decision, not the semantic layer's.

pub mod encoder;
pub mod latent;
pub mod retrieval;
