//! PH-037 P4 — substrate-invariance benchmark harness.
//!
//! Runs the same semantic task through multiple substrates and prints the
//! reconstructed state, delta-recovery, and invariant-preservation columns.
//!
//! Run:
//!   cargo run --example neural_invariance --features neural -- --task object-state
//!   cargo run --example neural_invariance --features neural -- --task object-state --noise 0.1

use std::env;
use std::time::Instant;

use physis_core::neural::substrate::{
    DeterministicBackend, NeuralSubstrate, ObservationWindow, PopulationState,
    StimulusPattern, SubstrateConfig, SubstrateKind,
};
use physis_core::neural::backends::RandomReservoirBackend;
use physis_core::neural::delta::{build_delta_report, decode_semantic_delta};

struct BackendSlot {
    kind: SubstrateKind,
    name: &'static str,
}

fn unit(axis: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    if axis < v.len() { v[axis] = 1.0; }
    v
}

fn parse_args() -> (String, f32) {
    let args: Vec<String> = env::args().collect();
    let mut task = "object-state".into();
    let mut noise = 0.0;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--task" => { task = args[i + 1].clone(); i += 2; }
            "--noise" => { noise = args[i + 1].parse().unwrap_or(0.0); i += 2; }
            _ => i += 1,
        }
    }
    (task, noise)
}

fn main() {
    let (task, noise) = parse_args();
    let dim = 4;
    let threshold = 0.01_f32;

    // The semantic task: flip axis 0 from +1 → -1
    // (t0: cup ON table → t1: cup HELD_BY self, PH-101 §11 "object state").
    let prev = unit(0, dim);
    let stimulus = vec![-1.0_f32, 0.0, 0.0, 0.0];

    // Mix noise into the stimulus for the noisy-substrate runs.
    let stim_noisy = if noise > 0.0 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        stimulus.iter().map(|x| x + rng.gen_range(-noise..noise)).collect()
    } else {
        stimulus.clone()
    };

    let backends: Vec<BackendSlot> = vec![
        BackendSlot { kind: SubstrateKind::Deterministic, name: "DETERMINISTIC" },
        BackendSlot { kind: SubstrateKind::RandomReservoir, name: "RESERVOIR" },
        BackendSlot { kind: SubstrateKind::ArtificialSpiking, name: "SPIKING" },
    ];

    println!("PH-037 substrate-invariance benchmark");
    println!("task={task} dim={dim} threshold={threshold:.3} noise={noise}");
    println!("┌──────────────┬───────────┬─────────┬──────────┬────────────┬──────┐");
    println!("│ backend       │ recon_acc │ delta   │ invariant│ latency_ms │ err  │");
    println!("├──────────────┼───────────┼─────────┼──────────┼────────────┼──────┤");

    let t0 = prev.clone();
    let prev_state = PopulationState { values: prev.clone(), trace: vec![] };

    for slot in backends {
        let t_start = Instant::now();
        let mut b: Box<dyn NeuralSubstrate> = match slot.kind {
            SubstrateKind::Deterministic => Box::new({
                let mut b = DeterministicBackend::new(dim);
                b.configure(SubstrateConfig { kind: slot.kind, population_size: dim, ..Default::default() }).unwrap();
                b
            }),
            SubstrateKind::RandomReservoir => Box::new({
                let mut b = RandomReservoirBackend::new(dim, dim, 42);
                b.configure(SubstrateConfig {
                    kind: slot.kind,
                    population_size: dim,
                    noise: if slot.kind == SubstrateKind::RandomReservoir { noise } else { 0.0 },
                    ..Default::default()
                }).unwrap();
                b
            }),
            SubstrateKind::ArtificialSpiking => Box::new({
                let mut b = physis_core::neural::backends::ArtificialSpikingBackend::new(dim);
                b.configure(SubstrateConfig {
                    kind: slot.kind,
                    population_size: dim,
                    noise: if slot.kind == SubstrateKind::ArtificialSpiking { noise } else { 0.0 },
                    ..Default::default()
                }).unwrap();
                b
            }),
            _ => unreachable!("slot"),
        };

        let sp = StimulusPattern {
            channels: vec![],
            dense: stim_noisy.clone(),
            hint: None,
        };
        let err = match b.stimulate(sp) { Ok(_) => "", Err(_) => "ERR" };
        let obs = b.observe(ObservationWindow::default()).unwrap();
        let readout = obs.populations.get("readout").map(|p| p.values.clone()).unwrap_or_default();
        let latency_ms = t_start.elapsed().as_micros() as f64 / 1000.0;

        // Reconstruction accuracy: how close readout is to expected (sign on axis 0).
        let recon_acc = if readout.len() == dim {
            let correct = (readout[0].partial_cmp(&0.0).unwrap() == stimulus[0].partial_cmp(&0.0).unwrap()) as u8;
            correct as f32
        } else { 0.0 };

        // Delta recovery: did decode_semantic_delta fire?
        let delta_fired = decode_semantic_delta(&t0, &obs, "cup", threshold).is_some();
        let delta_report = build_delta_report(&prev_state, &PopulationState { values: readout.clone(), trace: vec![] }, threshold);

        println!(
            "│ {:<13} │ {:>9.2} │ {:>7} │ {:>8} │ {:>10.3} │ {:^6} │",
            slot.name,
            recon_acc,
            delta_fired,
            if delta_report.changed_activation { "yes" } else { "no" },
            latency_ms,
            err,
        );
    }
    println!("└──────────────┴───────────┴─────────┴──────────┴────────────┴──────┘");
    println!("(recon_acc is binary sign-match on axis 0; delta = drift signal fired; invariant = semantic delta detected)");
}
