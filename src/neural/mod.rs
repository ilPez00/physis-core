//! PHYSIS × Neural Substrate — substrate-independent semantic interface.
//!
//! > Engineering hypothesis (research packet P0): a structured Physis semantic
//! > state can remain stable across radically different computational
//! > substrates, provided each substrate exposes a common
//! > stimulation/observation interface.
//!
//! This module is **additive**: it reuses existing Physis structures and
//! deletes nothing. The core builds without any biological dependency — the
//! `NeuralSubstrate` trait and `DeterministicBackend` are always available;
//! the reservoir, spiking, recorded-data, and FlyWire backends are behind the
//! `neural` feature (off by default).
//!
//! ## Architecture (PH-101 Task 4)
//!
//! ```text
//! PHYSIS Semantic State (CoherenceNode / Hypothesis)
//!         ↓  encoder  (semantic → StimulusPattern)
//! substrate adapter  (stimulus → channel activations)
//!         ↓
//! computation / dynamics  (backend: deterministic | reservoir | spiking | …)
//!         ↓
//! substrate observation  (NeuralObservation)
//!         ↓  decoder  (observation → OntologyMutation → evaluate_mutation)
//! PHYSIS Semantic State
//! ```
//!
//! ## Reuse map
//!
//! - [`crate::CoherenceNode`] `embedding: Vec<f32>` — the activation vector of
//!   a `PopulationState`. Physis does not care if the numbers came from a
//!   lookup table or 139k fly neurons.
//! - [`crate::embed::VectorEmbed`] / [`crate::embed::RandomProjectionEmbedder`]
//!   — the default encoder projection (deterministic mock).
//! - [`crate::classify::CellClassifier`] — cell/domain assignment of decoded
//!   states; the `asserted` score maps to `confidence`.
//! - [`crate::delta_engine::{OntologyMutation, MutationOp, evaluate_mutation}`]
//!   — **semantic delta** decoding: the primary signal is the
//!   `OntologyDeltaReport` between `State(t)` and `State(t+dt)`, not a full
//!   reconstruction. This is research note #4.
//! - [`crate::Hypothesis`] — `confidence`, `fitness_breakdown`, and `fitness`
//!   carry the distribution-over-labels model (ANIMAL/DOG/CAT/WOLF) rather
//!   than winner-take-all. The `coherence_profile` feeds uncertainty.
//! - [`crate::ontology::OntologyLoader`] — domain × mode cell pins.
//! - [`crate::cortex::encoder::NeuralRole`] — informs the `Encoder` trait's
//!   probe-head mapping (research note: cortex encodes, this decodes).
//!
//! ## Backends
//!
//! | Backend | Feature | Notes |
//! |---|---|---|
//! | `DeterministicBackend` | (none) | Pass-through; the null hypothesis. |
//! | `RandomReservoirBackend` | `neural` | Echo State Network, CPU-only. |
//! | `ArtificialSpikingBackend` | `neural` | Toy LIF; stub dynamics. |
//! | `RecordedNeuralBackend` | `neural` | CSV/JSON replay; no data bundled. |
//! | `FlyConnectomeBackend` | `neural` | Reads `data/fly/connectome.csv` (absent). |
//! | `CorticalSimulatorBackend` | `neural` | Mock only; no device API. |
//! | `FutureWetwareBackend` | `neural` | Stub; raises live-hardware error. |

pub mod substrate;

#[cfg(feature = "neural")]
pub mod backends;
#[cfg(feature = "neural")]
pub mod codec;
#[cfg(feature = "neural")]
pub mod delta;

pub use substrate::{
    ChannelId, Confidence, NeuralObservation, NeuralSubstrate, PopulationState, StimulusPattern,
    SubstrateCapabilities, SubstrateConfig, SubstrateHealth, SubstrateKind,
    SubstrateError, SubstrateResult,
};
