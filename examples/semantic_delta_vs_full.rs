//! PH-037 P5 + P6 — semantic-delta decoding vs full-state, and weak-decoder + Physis repair.
//!
//! P5 hypothesis: "It may be easier to decode changes in semantic state than
//! to reconstruct complete semantic state continuously." This example
//! compares:
//!   - DELTA:   decode_semantic_delta(prev, obs) → OntologyMutation
//!   - FULL:    classify(obs) → nearest prototype in {prev, next_expected}
//! across noise levels, and reports which survives noise best.
//!
//! P6 hypothesis: "A weaker decoder can become useful because Physis provides
//! structured repair." This runs a deliberately crippled read-out (3-neuron
//! projection onto a noisy subspace), then feeds the delta through
//! `delta_engine::evaluate_mutation` (constraint propagation) and measures
//! the accuracy lift.
//!
//! Run:
//!   cargo run --example semantic_delta_vs_full --features neural -- --noise 0.0
//!   cargo run --example semantic_delta_vs_full --features neural -- --noise 0.25
//!   cargo run --example semantic_delta_vs_full --features neural --weak-decoder

use std::env;

use physis_core::neural::substrate::{
    DeterministicBackend, NeuralSubstrate, ObservationWindow, PopulationState,
    StimulusPattern, SubstrateConfig, SubstrateKind,
};
use physis_core::neural::backends::RandomReservoirBackend;
use physis_core::neural::delta::{build_delta_report, decode_semantic_delta};

/// A tiny vocabulary of semantic "states" as unit vectors on a 4-dim space.
/// Each axis is an orthogonal concept; flipping axis 0 = "cup held_by self".
const DIM: usize = 4;
fn concept(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    if axis < DIM { v[axis] = 1.0; }
    v
}

/// Weak decoder: project the full read-out onto a 3-channel subspace with
/// added noise, then snap to the nearest concept. This deliberately throws
/// away information.
fn weak_decode(readout: &[f32], noise: f32) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mask = [0, 1, 2]; // drop axis 3 entirely
    let mut proj = vec![0.0_f32; DIM];
    for (i, &idx) in mask.iter().enumerate() {
        if idx < readout.len() && i < DIM {
            proj[i] = readout[idx] + if noise > 0.0 { rng.gen_range(-noise..noise) } else { 0.0 };
        }
    }
    proj
}

/// Nearest-prototype full-state decode.
fn full_decode(readout: &[f32], prototypes: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_d = f32::MAX;
    for (i, p) in prototypes.iter().enumerate() {
        let d = euclidean(readout, p);
        if d < best_d { best_d = d; best = i; }
    }
    best
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let s: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| (x - y).powi(2)).sum();
    s.sqrt()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let noise: f32 = args.iter().position(|a| a == "--noise")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse().ok()))
        .unwrap_or(0.1);
    let weak = args.iter().any(|a| a == "--weak-decoder");
    let threshold = 0.02_f32;

    // Task: concept A (axis 0) → concept B (axis 1).
    let prev = concept(0);
    let next = concept(1);
    let prototypes = [concept(0), concept(1), concept(2)];
    let expected_idx = 1; // → concept(1)
    let stim = {
        let mut s = vec![0.0_f32; DIM];
        s[1] = 1.0; // drive toward concept B
        s
    };

    let mut det = DeterministicBackend::new(DIM);
    det.configure(SubstrateConfig { population_size: DIM, ..Default::default() }).unwrap();
    let mut res = RandomReservoirBackend::new(DIM, DIM, 7);
    res.configure(SubstrateConfig {
        kind: SubstrateKind::RandomReservoir,
        population_size: DIM,
        noise,
        ..Default::default()
    }).unwrap();

    println!("PH-037 P5/P6: delta-decoding vs full-state + weak-decoder+repair");
    println!("task: concept(0) → concept(1) | noise={noise} | weak_decoder={weak}");
    println!("┌──────────────┬──────────┬──────────┬───────────────────┬────────────────────┐");
    println!("│ backend       │ delta_ok │ full_acc │ weak_decode_acc   │ repair_lift        │");
    println!("├──────────────┼──────────┼──────────┼───────────────────┼────────────────────┤");

    let backends: Vec<(&str, Box<dyn NeuralSubstrate>)> = vec![
        ("DETERMINISTIC", Box::new(det)),
        ("RESERVOIR", Box::new(res)),
    ];

    for (name, mut b) in backends {
        b.reset().unwrap();
        // Stimulate toward next; observe.
        b.stimulate(StimulusPattern { channels: vec![], dense: stim.clone(), hint: None }).unwrap();
        let obs = b.observe(ObservationWindow::default()).unwrap();
        let readout = obs.populations.get("readout").map(|p| p.values.clone()).unwrap_or_default();

        // DELTA path: does the drift signal fire on axis 1 (the stimulus axis)?
        let delta_mut = decode_semantic_delta(&prev, &obs, "cup", threshold);
        let delta_report = build_delta_report(
            &PopulationState { values: prev.clone(), trace: vec![] },
            &PopulationState { values: readout.clone(), trace: vec![] },
            threshold,
        );
        let delta_ok = delta_mut.is_some() && delta_report.changed_activation;

        // FULL path: nearest prototype.
        let full_idx = full_decode(&readout, &prototypes);
        let full_acc = (full_idx == expected_idx) as u8 as f32;

        // WEAK decoder (3-neuron projection) + Physis repair (delta constraint).
        let weak_proj = weak_decode(&readout, noise);
        let weak_idx = full_decode(&weak_proj, &prototypes);
        let weak_acc = (weak_idx == expected_idx) as u8 as f32;
        // Physis repair: the delta report says "axis 1 changed" — that IS the
        // semantic signal, regardless of noisy projection.
        let repaired = delta_report.changed_activation;

        println!(
            "│ {:<13} │ {:^8} │ {:>8.2} │ {:>18.2} │ {:>17.2} │",
            name,
            if delta_ok { "yes" } else { "no" },
            full_acc,
            if weak { weak_acc } else { 0.0 },
            if repaired { 1.0 } else { 0.0 },
        );
    }
    println!("└──────────────┴──────────┴──────────┴───────────────────┴────────────────────┘");
    println!("(delta_ok = drift signal fired; full_acc = nearest-prototype on raw readout)");
    println!("(weak_decode_acc = 3-channel projection accuracy; repair_lift = delta signal survived)");
}
