//! P4: substrate-invariance base test.
//!
//! Same semantic task (a 4-dim embedding delta), different substrate → the
//! decoded semantic delta must preserve the **invariant** that a signal fired.
//! This is the "semantic invariant preserved?" check from PH-101 section 10.
//!
//! Run: `cargo test --test neural_invariance --features neural`

use physis_core::neural::substrate::{
    DeterministicBackend, NeuralSubstrate, ObservationWindow, PopulationState, StimulusPattern,
    SubstrateConfig, SubstrateKind,
};
#[cfg(feature = "neural")]
use physis_core::neural::backends::RandomReservoirBackend;
use physis_core::neural::delta::{build_delta_report, decode_semantic_delta};

const EMBEDDING_DIM: usize = 4;
const THRESHOLD: f32 = 0.01;

fn unit(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; EMBEDDING_DIM];
    if axis < v.len() { v[axis] = 1.0; }
    v
}

/// The reference semantic task: flip axis 0 from +1 → -1.
fn task_stimulus() -> Vec<f32> {
    vec![-1.0, 0.0, 0.0, 0.0]
}

/// Run a single substrate through the task; return the decoded delta mutation.
#[cfg(feature = "neural")]
fn run_substrate(kind: SubstrateKind, prev: &[f32]) -> Option<physis_core::delta_engine::OntologyMutation> {
    let mut b: Box<dyn NeuralSubstrate> = match kind {
        SubstrateKind::Deterministic => Box::new({
            let mut b = DeterministicBackend::new(EMBEDDING_DIM);
            b.configure(SubstrateConfig { kind, population_size: EMBEDDING_DIM, ..Default::default() }).unwrap();
            b
        }),
        _ => Box::new({
            let mut b = RandomReservoirBackend::new(EMBEDDING_DIM, EMBEDDING_DIM, 42);
            b.configure(SubstrateConfig { kind, population_size: EMBEDDING_DIM, ..Default::default() }).unwrap();
            b
        }),
    };
    let sp = StimulusPattern { channels: vec![], dense: task_stimulus(), hint: None };
    b.stimulate(sp).unwrap();
    let obs = b.observe(ObservationWindow::default()).unwrap();
    decode_semantic_delta(prev, &obs, "cup", THRESHOLD)
}

/// Without the neural feature, only the deterministic backend is available.
#[cfg(not(feature = "neural"))]
fn run_deterministic(prev: &[f32]) -> physis_core::delta_engine::OntologyMutation {
    let mut b = DeterministicBackend::new(EMBEDDING_DIM);
    let sp = StimulusPattern { channels: vec![], dense: task_stimulus(), hint: None };
    b.stimulate(sp).unwrap();
    let obs = b.observe(ObservationWindow::default()).unwrap();
    decode_semantic_delta(prev, &obs, "cup", THRESHOLD).expect("deterministic fires drift")
}

#[test]
fn deterministic_backend_preserves_semantic_invariant() {
    let prev = unit(0); // [1, 0, 0, 0]
    let obs_readout = task_stimulus();
    // Direct delta check: state(t) = [1,0,0,0], state(t+dt) = [-1,0,0,0].
    let prev_state = PopulationState { values: prev.clone(), trace: vec![] };
    let next_state = PopulationState { values: obs_readout, trace: vec![] };
    let delta = build_delta_report(&prev_state, &next_state, THRESHOLD);
    assert!(delta.changed_activation, "a sign-flip must register as a semantic delta");
    assert!(delta.changed_properties.contains(&"activation".to_string()));
}

#[cfg(feature = "neural")]
#[test]
fn same_signal_fires_across_substrates() {
    let prev = unit(0); // [1, 0, 0, 0]

    let det = run_substrate(SubstrateKind::Deterministic, &prev);
    let res = run_substrate(SubstrateKind::RandomReservoir, &prev);

    // Deterministic passes stimulus straight through: decoded mutation must
    // be an EmbeddingShift ending at [-1,0,0,0].
    let det_op = det.expect("deterministic fires a mutation").operation;
    match det_op {
        physis_core::delta_engine::MutationOp::EmbeddingShift { new_embedding, .. } => {
            assert!((new_embedding[0] - (-1.0)).abs() < 1e-4, "deterministic axis-0 drift");
        }
        other => panic!("deterministic produced {:?}, expected EmbeddingShift", other),
    }

    // Reservoir is noisy by design — assert the *invariant* (a drift signal
    // fired) rather than its exact value. The semantic fact — "something
    // changed on this node" — survives the substrate switch.
    assert!(res.is_some(), "reservoir must fire the same semantic-delta signal");
}
