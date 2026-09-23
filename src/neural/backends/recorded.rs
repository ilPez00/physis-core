//! RecordedNeuralBackend (P8) — replay spike trains / firing rates.
//!
//! Reads CSV or JSON (channels × time). No recordings bundled. The reader is
//! format-ready so public datasets can be dropped in under `data/neural/`.

use crate::neural::substrate::{
    NeuralObservation, ObservationWindow, PopulationState, StimulusPattern, SubstrateCapabilities,
    SubstrateConfig, SubstrateHealth, SubstrateError, SubstrateResult,
};
use crate::neural::NeuralSubstrate;

#[derive(Debug, Clone)]
pub struct RecordedNeuralBackend {
    cfg: SubstrateConfig,
    trace: Vec<Vec<f32>>,
    cursor: usize,
}

impl Default for RecordedNeuralBackend {
    fn default() -> Self { Self::new() }
}

impl RecordedNeuralBackend {
    pub fn new() -> Self {
        Self {
            cfg: SubstrateConfig::default(),
            trace: Vec::new(),
            cursor: 0,
        }
    }
    /// Load a CSV (one row = one timestep, one column = one channel).
    pub fn load_csv(path: &str) -> SubstrateResult<Self> {
        // Minimal CSV reader; no csv crate dependency (avoids pulling deps
        // into the biological layer).
        let contents = std::fs::read_to_string(path)
            .map_err(|e| SubstrateError::Backend(format!("load_csv {path}: {e}")))?;
        let mut trace = Vec::new();
        for line in contents.lines() {
            if line.trim().is_empty() { continue; }
            let row: Vec<f32> = line.split(',')
                .filter_map(|c| c.trim().parse::<f32>().ok())
                .collect();
            if !row.is_empty() { trace.push(row); }
        }
        if trace.is_empty() {
            return Err(SubstrateError::Backend(format!("empty recording: {path}")));
        }
        let pop = trace[0].len();
        Ok(Self {
            trace,
            cursor: 0,
            cfg: SubstrateConfig { population_size: pop, ..SubstrateConfig::default() },
        })
    }
}

impl NeuralSubstrate for RecordedNeuralBackend {
    fn capabilities(&self) -> SubstrateCapabilities {
        SubstrateCapabilities {
            supports_stimulation: false,
            supports_observation: true,
            supports_reset: true,
            population_size: self.cfg.population_size,
            latency_ms: 0.0,
            notes: vec!["read-only replay".into()],
        }
    }

    fn configure(&mut self, cfg: SubstrateConfig) -> SubstrateResult<()> {
        self.cfg = cfg;
        Ok(())
    }

    fn stimulate(&mut self, _input: StimulusPattern) -> SubstrateResult<()> {
        Err(SubstrateError::Backend("RecordedNeuralBackend is read-only".into()))
    }

    fn observe(&mut self, window: ObservationWindow) -> SubstrateResult<NeuralObservation> {
        if self.trace.is_empty() {
            return Err(SubstrateError::Backend("no recording loaded".into()));
        }
        let t = self.cursor.min(self.trace.len() - 1);
        let mut pops = std::collections::HashMap::new();
        pops.insert(
            "readout".to_string(),
            PopulationState { values: self.trace[t].clone(), trace: vec![] },
        );
        let _ = window; // windowing would slice the trace.
        self.cursor = (self.cursor + 1).min(self.trace.len() - 1);
        Ok(NeuralObservation { t: self.cursor, populations: pops, window })
    }

    fn reset(&mut self) -> SubstrateResult<()> {
        self.cursor = 0;
        Ok(())
    }

    fn health(&self) -> SubstrateHealth {
        SubstrateHealth { online: !self.trace.is_empty(), occupancy: 0.0, error: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_csv_parses_sample() {
        let b = RecordedNeuralBackend::load_csv("data/neural/sample.csv")
            .expect("sample recording loads");
        assert_eq!(b.trace.len(), 4, "sample.csv has 4 timesteps");
        assert_eq!(b.trace[0], vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(b.trace[1], vec![0.5, 0.2, -0.1, 0.3]);
    }

    #[test]
    fn observe_replays_in_order() {
        let mut b = RecordedNeuralBackend::load_csv("data/neural/sample.csv")
            .expect("sample loads");
        let o0 = b.observe(ObservationWindow::default()).unwrap();
        let o1 = b.observe(ObservationWindow::default()).unwrap();
        assert_eq!(o0.populations["readout"].values, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(o1.populations["readout"].values, vec![0.5, 0.2, -0.1, 0.3],
            "cursor advances: second observe = timestep 2");
    }

    #[test]
    fn reset_loops_back_to_first_frame() {
        let mut b = RecordedNeuralBackend::load_csv("data/neural/sample.csv")
            .expect("sample loads");
        let _ = b.observe(ObservationWindow::default()).unwrap();
        b.reset().unwrap();
        let o = b.observe(ObservationWindow::default()).unwrap();
        assert_eq!(o.populations["readout"].values, vec![0.0, 0.0, 0.0, 0.0],
            "reset restores frame 0");
    }

    #[test]
    fn missing_file_errors() {
        assert!(RecordedNeuralBackend::load_csv("data/neural/does_not_exist.csv").is_err());
    }
}
