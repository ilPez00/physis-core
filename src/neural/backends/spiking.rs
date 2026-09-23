//! ArtificialSpikingBackend (P7) — toy Leaky Integrate-and-Fire.
//!
//! Stub dynamics: `alpha = 0.1`, no plasticity, no synapse model. This is a
//! **spiking-shaped wire**, not a neuron model. Marked
//! `#[cfg(feature = "neural")]` because real dynamics would pull in a spiking
//! crate; keep it minimal so the core builds without one.

use crate::neural::substrate::{
    NeuralObservation, ObservationWindow, PopulationState, StimulusPattern, SubstrateCapabilities,
    SubstrateConfig, SubstrateHealth,
};
use crate::neural::{NeuralSubstrate, SubstrateResult};

#[derive(Debug, Clone)]
pub struct ArtificialSpikingBackend {
    cfg: SubstrateConfig,
    membrane: Vec<f32>,
    fired: Vec<bool>,
    t: usize,
}

impl ArtificialSpikingBackend {
    pub fn new(population_size: usize) -> Self {
        Self {
            cfg: SubstrateConfig { population_size, ..SubstrateConfig::default() },
            membrane: vec![0.0; population_size],
            fired: vec![false; population_size],
            t: 0,
        }
    }
}

// Biological LIF parameters are NOT modeled here — α = 0.1 is a toy decay.
// See docs/research/wetware/LIMITATIONS.md.
const DECAY: f32 = 0.1;
const THRESHOLD: f32 = 1.0;

impl NeuralSubstrate for ArtificialSpikingBackend {
    fn capabilities(&self) -> SubstrateCapabilities {
        SubstrateCapabilities {
            supports_stimulation: true,
            supports_observation: true,
            supports_reset: true,
            population_size: self.cfg.population_size,
            latency_ms: 2.0,
            notes: vec!["toy LIF (decay=0.1, no plasticity)".into()],
        }
    }

    fn configure(&mut self, config: SubstrateConfig) -> SubstrateResult<()> {
        let n = config.population_size;
        self.cfg = config;
        self.membrane.resize(n, 0.0);
        self.fired.resize(n, false);
        Ok(())
    }

    fn stimulate(&mut self, input: StimulusPattern) -> SubstrateResult<()> {
        let n = self.cfg.population_size;
        let drive = if !input.dense.is_empty() { &input.dense } else {
            // Channel write is sparse; accumulate onto membrane.
            for (ch, v) in input.channels {
                if ch < n { self.membrane[ch] += v; }
            }
            return Ok(());
        };
        for (i, m) in self.membrane.iter_mut().enumerate().take(n.min(drive.len())) {
            *m += drive[i];
        }
        // LIF step: leaky integration + spike threshold.
        for i in 0..self.membrane.len() {
            self.membrane[i] *= 1.0 - DECAY;
            self.fired[i] = self.membrane[i] >= THRESHOLD;
            if self.fired[i] { self.membrane[i] = 0.0; }
        }
        self.t += 1;
        Ok(())
    }

    fn observe(&mut self, _window: ObservationWindow) -> SubstrateResult<NeuralObservation> {
        let mut pops = std::collections::HashMap::new();
        // Read out membrane + spike raster (boolean → 0/1).
        let raster: Vec<f32> = self.fired.iter().map(|f| if *f { 1.0 } else { 0.0 }).collect();
        pops.insert("membrane".to_string(), PopulationState { values: self.membrane.clone(), trace: vec![] });
        pops.insert("raster".to_string(), PopulationState { values: raster, trace: vec![] });
        Ok(NeuralObservation { t: self.t, populations: pops, window: ObservationWindow::default() })
    }

    fn reset(&mut self) -> SubstrateResult<()> {
        for v in &mut self.membrane { *v = 0.0; }
        for f in &mut self.fired { *f = false; }
        self.t = 0;
        Ok(())
    }

    fn health(&self) -> SubstrateHealth {
        SubstrateHealth { online: !self.membrane.is_empty(), occupancy: 0.0, error: None }
    }
}
