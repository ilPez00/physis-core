//! The `NeuralSubstrate` trait + normalized representation types.
//!
//! These types are the common language between Physis semantic state and any
//! computational substrate. They carry no biological assumptions — a channel
//! is just an index, an activation is just an `f32`. What makes them "neural"
//! is that they model **distributed, noisy, history-dependent** activations
//! rather than single-valued lookups.
//!
//! All types serialize to plain Rust primitives so a backend can be a pure
//! function (deterministic) or a wgpu kernel (spiking) without touching the
//! trait boundary.

use std::collections::HashMap;

use crate::models::Score;

/// A normalized substrate configuration. Backends interpret fields loosely;
/// the deterministic backend ignores most of these.
#[derive(Debug, Clone)]
pub struct SubstrateConfig {
    pub kind: SubstrateKind,
    pub population_size: usize,
    pub seed: u64,
    pub dt: f32,
    pub noise: f32,
    pub extra: HashMap<String, String>,
}

impl Default for SubstrateConfig {
    fn default() -> Self {
        Self {
            kind: SubstrateKind::Deterministic,
            population_size: 256,
            seed: 0,
            dt: 1.0,
            noise: 0.0,
            extra: HashMap::new(),
        }
    }
}

/// Distinguishes backends for the HUD selector and benchmark routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstrateKind {
    Deterministic,
    RandomReservoir,
    ArtificialSpiking,
    RecordedNeural,
    FlyConnectome,
    CorticalSimulator,
    /// Placeholder for live wetware. Construction raises an error.
    FutureWetware,
}

/// What the substrate can do — a capability advertisement, not a promise.
#[derive(Debug, Clone)]
pub struct SubstrateCapabilities {
    pub supports_stimulation: bool,
    pub supports_observation: bool,
    pub supports_reset: bool,
    pub population_size: usize,
    pub latency_ms: f32,
    pub notes: Vec<String>,
}

/// A single stimulation channel. Reuses the Physis `Score` range `[-1, 1]`
/// conceptually — negative = inhibition.
pub type ChannelId = usize;

/// Sparse or dense stimulation vector. `channel → activation`.
#[derive(Debug, Clone)]
pub struct StimulusPattern {
    pub channels: Vec<(ChannelId, f32)>,
    /// Dense fallback for read-out populations. If non-empty, overrides
    /// `channels` for substrates that prefer dense vectors.
    pub dense: Vec<f32>,
    /// Optional semantic hint (domain/mode) reused from the ontology. The
    /// substrate does not interpret it — the encoder did — but carrying it
    /// side-channel lets deterministic decode without a round-trip.
    pub hint: Option<(String, String)>,
}

/// Observation window for `observe`. A substrate may integrate over time.
#[derive(Debug, Clone, Copy)]
pub struct ObservationWindow {
    pub steps: usize,
    pub from_t: usize,
    pub to_t: usize,
}

impl Default for ObservationWindow {
    fn default() -> Self {
        Self { steps: 1, from_t: 0, to_t: 0 }
    }
}

/// The activation snapshot a substrate returns. `values` is the read-out
/// population; `trace` is optional history (spiking/recorded backends). The
/// layout matches `CoherenceNode.embedding` so a decoder can project directly
/// to a `CoherenceNode` without a copy.
#[derive(Debug, Clone)]
pub struct PopulationState {
    pub values: Vec<f32>,
    pub trace: Vec<Vec<f32>>,
}

/// A timestamped observation over a window.
#[derive(Debug, Clone)]
pub struct NeuralObservation {
    pub t: usize,
    pub populations: HashMap<String, PopulationState>,
    pub window: ObservationWindow,
}

/// Normalized confidence/uncertainty. `score` in `[-1, 1]` (maps to
/// `CoherenceNode::asserted` / `Hypothesis::confidence`). `entropy` is the
/// Shannon entropy of the activation distribution — high = uncertain.
#[derive(Debug, Clone, Copy)]
pub struct Confidence {
    pub score: Score,
    pub entropy: f32,
    pub uncertainty: f32,
}

/// Liveness + capacity health from the substrate.
#[derive(Debug, Clone)]
pub struct SubstrateHealth {
    pub online: bool,
    pub occupancy: f32,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum SubstrateError {
    /// Raised by `FutureWetwareBackend` / unimplemented backends.
    LiveHardwareUnavailable,
    /// Backend-specific failure (data file missing, runtime error).
    Backend(String),
}

pub type SubstrateResult<T> = std::result::Result<T, SubstrateError>;

/// The common stimulation/observation interface (PH-037 / research Task 4).
///
/// A substrate does not see "APPLE" or "EAT" — it sees channel indices and
/// floats. The encoder/decoder owns the semantic mapping.
pub trait NeuralSubstrate {
    /// Advertise what this backend supports.
    fn capabilities(&self) -> SubstrateCapabilities;

    /// Re-configure dynamics. Cheap for deterministic; may rebuild matrices
    /// for reservoir/spiking.
    fn configure(&mut self, config: SubstrateConfig) -> SubstrateResult<()>;

    /// Inject one stimulus timestep.
    fn stimulate(&mut self, input: StimulusPattern) -> SubstrateResult<()>;

    /// Read out over a window.
    fn observe(&mut self, window: ObservationWindow) -> SubstrateResult<NeuralObservation>;

    /// Clear transient state. Persistent topology (a connectome edge-list, a
    /// reservoir weight matrix) is preserved.
    fn reset(&mut self) -> SubstrateResult<()>;

    /// Liveness probe; never blocks.
    fn health(&self) -> SubstrateHealth;
}

/// DeterministicBackend (P2) — the null-hypothesis substrate.
///
/// Stimulus passes straight through: channels are written to an internal
/// buffer, `observe` returns the buffer as a `PopulationState`. No dynamics,
/// no noise. This is the upper-bound for invariance: if Physis cannot keep
/// state stable through this, no substrate will.
///
/// Always available (no feature gate) so the core builds without any
/// biological dependency.
#[derive(Debug, Clone)]
pub struct DeterministicBackend {
    cfg: SubstrateConfig,
    activation: Vec<f32>,
    t: usize,
}

impl DeterministicBackend {
    pub fn new(population_size: usize) -> Self {
        Self {
            cfg: SubstrateConfig {
                population_size,
                ..SubstrateConfig::default()
            },
            activation: vec![0.0; population_size],
            t: 0,
        }
    }
}

impl NeuralSubstrate for DeterministicBackend {
    fn capabilities(&self) -> SubstrateCapabilities {
        SubstrateCapabilities {
            supports_stimulation: true,
            supports_observation: true,
            supports_reset: true,
            population_size: self.cfg.population_size,
            latency_ms: 0.0,
            notes: vec!["deterministic pass-through".into()],
        }
    }

    fn configure(&mut self, config: SubstrateConfig) -> SubstrateResult<()> {
        self.activation.resize(config.population_size, 0.0);
        self.cfg = config;
        Ok(())
    }

    fn stimulate(&mut self, input: StimulusPattern) -> SubstrateResult<()> {
        // Dense wins; otherwise sparse-channel write.
        if !input.dense.is_empty() {
            let n = self.activation.len().min(input.dense.len());
            self.activation[..n].copy_from_slice(&input.dense[..n]);
        } else {
            for (ch, v) in input.channels {
                if ch < self.activation.len() {
                    self.activation[ch] = v;
                }
            }
        }
        self.t += 1;
        Ok(())
    }

    fn observe(&mut self, _window: ObservationWindow) -> SubstrateResult<NeuralObservation> {
        let mut populations = std::collections::HashMap::new();
        populations.insert(
            "readout".to_string(),
            PopulationState {
                values: self.activation.clone(),
                trace: vec![],
            },
        );
        Ok(NeuralObservation { t: self.t, populations, window: ObservationWindow::default() })
    }

    fn reset(&mut self) -> SubstrateResult<()> {
        for v in &mut self.activation {
            *v = 0.0;
        }
        self.t = 0;
        Ok(())
    }

    fn health(&self) -> SubstrateHealth {
        SubstrateHealth { online: true, occupancy: 0.0, error: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_backend_passes_stimulus_through() {
        let mut b = DeterministicBackend::new(4);
        let sp = StimulusPattern {
            channels: vec![(0, 0.9), (2, 0.1)],
            dense: vec![],
            hint: None,
        };
        b.stimulate(sp).unwrap();
        let obs = b.observe(ObservationWindow::default()).unwrap();
        let pop = obs.populations.get("readout").unwrap();
        assert!((pop.values[0] - 0.9).abs() < 1e-6);
        assert!((pop.values[2] - 0.1).abs() < 1e-6);
        assert!(b.health().online);
    }

    #[test]
    fn deterministic_reset_clears_activation() {
        let mut b = DeterministicBackend::new(2);
        b.stimulate(StimulusPattern { channels: vec![(0, 0.5)], dense: vec![], hint: None }).unwrap();
        b.reset().unwrap();
        let obs = b.observe(ObservationWindow::default()).unwrap();
        assert!(obs.populations["readout"].values.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn default_config_is_deterministic() {
        let cfg = SubstrateConfig::default();
        assert_eq!(cfg.kind, SubstrateKind::Deterministic);
        assert_eq!(cfg.population_size, 256);
    }

    #[test]
    fn stimulus_pattern_dense_overrides_channels() {
        let sp = StimulusPattern {
            channels: vec![(0, 0.5)],
            dense: vec![0.0, 0.0, 0.9, 0.0],
            hint: None,
        };
        assert!(sp.dense.len() > sp.channels.len());
    }

    #[test]
    fn observation_window_default_is_single_step() {
        let w = ObservationWindow::default();
        assert_eq!(w.steps, 1);
        assert_eq!(w.from_t, 0);
        assert_eq!(w.to_t, 0);
    }

    #[test]
    fn confidence_score_range_is_signed() {
        let c = Confidence { score: -0.3, entropy: 0.2, uncertainty: 0.8 };
        assert!(c.score < 0.0);
    }
}