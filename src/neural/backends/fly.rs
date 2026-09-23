//! FlyConnectomeBackend (P9) — adult Drosophila connectome as a weighted graph.
//!
//! Topology only. Dynamics are simplified (rate neurons, weighted propagation,
//! threshold). The module reads an edge-list CSV at `data/fly/connectome.csv`
//! — **absent from this repo** (public FAFB/FlyWire data is multi-GB).
//!
//! > connectome topology ≠ full biological dynamics. This is explicitly documented.

use crate::neural::substrate::{
    NeuralObservation, ObservationWindow, PopulationState, StimulusPattern, SubstrateCapabilities,
    SubstrateConfig, SubstrateHealth, SubstrateError,
};
use crate::neural::NeuralSubstrate;

#[derive(Debug, Clone)]
pub struct FlyConnectomeBackend {
    cfg: SubstrateConfig,
    /// (source, target, weight).
    edges: Vec<(usize, usize, f32)>,
    state: Vec<f32>,
    n_neurons: usize,
    step: usize,
}

impl Default for FlyConnectomeBackend {
    fn default() -> Self { Self::new() }
}

impl FlyConnectomeBackend {
    pub fn new() -> Self {
        Self {
            cfg: SubstrateConfig::default(),
            edges: Vec::new(),
            state: Vec::new(),
            n_neurons: 0,
            step: 0,
        }
    }
    /// Load an edge-list CSV: `source,target,weight` per row. Neurons are
    /// inferred from the max ID.
    pub fn load_csv(path: &str) -> Result<Self, SubstrateError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| SubstrateError::Backend(format!("fly connectome {path}: {e}")))?;
        let mut edges = Vec::new();
        let mut n = 0;
        for line in contents.lines() {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 3 { continue; }
            let s: usize = parts[0].trim().parse().unwrap_or(0);
            let t: usize = parts[1].trim().parse().unwrap_or(0);
            let w: f32 = parts[2].trim().parse().unwrap_or(0.0);
            n = n.max(s).max(t) + 1;
            edges.push((s, t, w));
        }
        Ok(Self {
            edges,
            state: vec![0.0; n],
            n_neurons: n,
            step: 0,
            cfg: SubstrateConfig { population_size: n, ..SubstrateConfig::default() },
        })
    }
}

impl NeuralSubstrate for FlyConnectomeBackend {
    fn capabilities(&self) -> SubstrateCapabilities {
        SubstrateCapabilities {
            supports_stimulation: true,
            supports_observation: true,
            supports_reset: true,
            population_size: self.n_neurons,
            latency_ms: 10.0,
            notes: vec!["fly connectome graph diffusion (NO biological fidelity)".into()],
        }
    }

    fn configure(&mut self, cfg: SubstrateConfig) -> Result<(), SubstrateError> {
        self.cfg = cfg;
        Ok(())
    }

    fn stimulate(&mut self, input: StimulusPattern) -> Result<(), SubstrateError> {
        if !input.dense.is_empty() {
            let n = self.state.len().min(input.dense.len());
            for i in 0..n { self.state[i] += input.dense[i] * 0.5; }
        }
        // One diffusion step: state ← state + W^T · state (rate model).
        let mut next = self.state.clone();
        for (s, t, w) in &self.edges {
            if *t < next.len() && *s < self.state.len() {
                next[*t] = (next[*t] + self.state[*s] * *w * 0.1).clamp(-1.0, 1.0);
            }
        }
        self.state = next;
        self.step += 1;
        Ok(())
    }

    fn observe(&mut self, window: ObservationWindow) -> Result<NeuralObservation, SubstrateError> {
        let mut pops = std::collections::HashMap::new();
        pops.insert(
            "readout".to_string(),
            PopulationState { values: self.state.clone(), trace: vec![] },
        );
        Ok(NeuralObservation { t: self.step, populations: pops, window })
    }

    fn reset(&mut self) -> Result<(), SubstrateError> {
        for v in &mut self.state { *v = 0.0; }
        self.step = 0;
        Ok(())
    }

    fn health(&self) -> SubstrateHealth {
        SubstrateHealth {
            online: !self.state.is_empty(),
            occupancy: self.edges.len() as f32 / (self.n_neurons as f32 + 1.0),
            error: None,
        }
    }
}