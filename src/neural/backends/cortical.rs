//! CorticalSimulatorBackend / CorticalRemoteBackend (P10).
//!
//! Mock only. No Cortical Labs API is wired — that would require a proprietary
//! SDK behind a `cortical` feature gate. This stub preserves the interaction
//! model so simulation-written code can later target real hardware with
//! minimal changes. Construction succeeds; `stimulate`/`observe` return a
//! clear "not connected" error.

use crate::neural::substrate::{
    NeuralObservation, ObservationWindow, SubstrateCapabilities, SubstrateConfig, SubstrateHealth,
    SubstrateError,
};
use crate::neural::NeuralSubstrate;

#[derive(Debug, Clone)]
pub struct CorticalSimulatorBackend {
    cfg: SubstrateConfig,
}

impl Default for CorticalSimulatorBackend {
    fn default() -> Self { Self::new() }
}

impl CorticalSimulatorBackend {
    pub fn new() -> Self {
        Self { cfg: SubstrateConfig::default() }
    }
}

impl NeuralSubstrate for CorticalSimulatorBackend {
    fn capabilities(&self) -> SubstrateCapabilities {
        SubstrateCapabilities {
            supports_stimulation: false,
            supports_observation: false,
            supports_reset: true,
            population_size: 0,
            latency_ms: 0.0,
            notes: vec!["mock — no device API wired".into()],
        }
    }

    fn configure(&mut self, cfg: SubstrateConfig) -> Result<(), SubstrateError> {
        self.cfg = cfg;
        Ok(())
    }

    fn stimulate(&mut self, _input: crate::neural::substrate::StimulusPattern) -> Result<(), SubstrateError> {
        Err(SubstrateError::Backend("CorticalSimulatorBackend: no device connected (mock)".into()))
    }

    fn observe(&mut self, _window: ObservationWindow) -> Result<NeuralObservation, SubstrateError> {
        Err(SubstrateError::Backend("CorticalSimulatorBackend: no device connected (mock)".into()))
    }

    fn reset(&mut self) -> Result<(), SubstrateError> { Ok(()) }
    fn health(&self) -> SubstrateHealth {
        SubstrateHealth { online: false, occupancy: 0.0, error: Some("not connected (mock)".into()) }
    }
}
