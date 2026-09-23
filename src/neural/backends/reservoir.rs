//! RandomReservoirBackend (P3) — Echo State Network / reservoir.
//!
//! The reference stochastic substrate. Not a model of anything biological:
//! a fixed random recurrent matrix with leaky-integrator neurons. The only
//! point is to give Physis something with temporal dynamics, recurrence,
//! state memory, and noisy distributed activations — cheaply and in pure
//! CPU Rust so the invariance benchmark can run anywhere.
//!
//! Stimulus is injected into `input_size` neurons; the full `population_size`
//! state is read out. The recurrent weight matrix is built from a seed and
//! spectral radius at `configure` time.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::neural::substrate::{
    NeuralObservation, ObservationWindow, PopulationState, StimulusPattern, SubstrateCapabilities,
    SubstrateConfig, SubstrateHealth, SubstrateResult,
};
use crate::neural::NeuralSubstrate;

#[derive(Debug, Clone)]
pub struct RandomReservoirBackend {
    cfg: SubstrateConfig,
    state: Vec<f32>,
    // Flattened recurrent weight matrix (population_size × population_size).
    rec: Vec<f32>,
    // Input projection (input_size → population_size).
    inp: Vec<f32>,
    input_size: usize,
    t: usize,
}

impl RandomReservoirBackend {
    pub fn new(population_size: usize, input_size: usize, seed: u64) -> Self {
        Self {
            cfg: SubstrateConfig { population_size, seed, ..SubstrateConfig::default() },
            state: vec![0.0; population_size],
            rec: Vec::new(),
            inp: Vec::new(),
            input_size,
            t: 0,
        }
    }

    fn build_matrices(&mut self) {
        let n = self.cfg.population_size;
        let mut rng = StdRng::seed_from_u64(self.cfg.seed);
        // Spectral radius scaling.
        let rho = self.cfg.extra.get("spectral_radius")
            .and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.9);

        // Recurrent matrix: sparse random, then scaled.
        self.rec = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                // ~5% connectivity.
                if rng.gen_bool(0.05) {
                    let w = rng.gen_range(-1.0..1.0);
                    self.rec[i * n + j] = w * rho;
                }
            }
        }

        // Input projection.
        self.inp = (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect();

        // Store leak for step.
        // (No runtime storage beyond cfg.extra; leak parsed in step.)
    }

    fn step(&mut self, stimulus: &[f32]) {
        let n = self.cfg.population_size;
        let leak: f32 = self.cfg.extra.get("leak_rate")
            .and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.3);
        let noise: f32 = self.cfg.noise;

        let mut next = vec![0.0f32; n];
        for (i, next_i) in next.iter_mut().enumerate() {
            let mut acc = 0.0;
            // Recurrent: state · W_rec column
            for (j, state_j) in self.state.iter().enumerate() {
                acc += self.rec[j * n + i] * state_j;
            }
            // Input injection: first `input_size` stimulus channels.
            let inp = stimulus.get(i).copied().unwrap_or(0.0);
            acc += self.inp[i] * inp;
            // Leaky integration + tanh nonlinearity.
            let driven = acc.tanh();
            let mut rng = StdRng::seed_from_u64(self.cfg.seed.wrapping_add(self.t as u64).wrapping_add(i as u64));
            let noise_term = if noise > 0.0 { rng.gen_range(-noise..noise) } else { 0.0 };
            *next_i = (self.state[i] + leak * (driven + noise_term - self.state[i]).tanh()).clamp(-1.0, 1.0);
        }
        self.state = next;
        self.t += 1;
    }
}

impl NeuralSubstrate for RandomReservoirBackend {
    fn capabilities(&self) -> SubstrateCapabilities {
        SubstrateCapabilities {
            supports_stimulation: true,
            supports_observation: true,
            supports_reset: true,
            population_size: self.cfg.population_size,
            latency_ms: 1.0,
            notes: vec!["echo-state reservoir".into()],
        }
    }

    fn configure(&mut self, config: SubstrateConfig) -> SubstrateResult<()> {
        self.cfg = config;
        if self.rec.is_empty() {
            self.build_matrices();
        }
        Ok(())
    }

    fn stimulate(&mut self, input: StimulusPattern) -> SubstrateResult<()> {
        let dense = if !input.dense.is_empty() {
            input.dense.clone()
        } else {
            let mut d = vec![0.0; self.input_size];
            for (ch, v) in input.channels {
                if ch < self.input_size { d[ch] = v; }
            }
            d
        };
        self.step(&dense);
        Ok(())
    }

    fn observe(&mut self, window: ObservationWindow) -> SubstrateResult<NeuralObservation> {
        let mut trace = Vec::new();
        if window.steps > 1 && !self.state.is_empty() {
            // Windowed replay is a placeholder; real backends carry history.
            let _ = window;
            trace = vec![self.state.clone(); window.steps];
        }
        let mut pops = std::collections::HashMap::new();
        pops.insert(
            "readout".to_string(),
            PopulationState { values: self.state.clone(), trace },
        );
        Ok(NeuralObservation { t: self.t, populations: pops, window })
    }

    fn reset(&mut self) -> SubstrateResult<()> {
        for v in &mut self.state { *v = 0.0; }
        self.t = 0;
        Ok(())
    }

    fn health(&self) -> SubstrateHealth {
        SubstrateHealth { online: !self.state.is_empty(), occupancy: 0.0, error: None }
    }
}
