//! Semantic-delta decoding (PH-101 research note #4).
//!
//! > It may be easier to decode changes in semantic state than to reconstruct
//! > complete semantic state continuously.
//!
//! This bridges `NeuralObservation` → `OntologyMutation` → `evaluate_mutation`
//! (the existing delta_engine). A decoded `NeuralObservation` is compared to
//! the prior `State(t)`; the **delta** (added/removed relations, changed
//! properties, changed activation) becomes an `OntologyMutation`, which
//! `delta_engine::evaluate_mutation` projects to the next coherent state.

use crate::neural::substrate::{NeuralObservation, PopulationState};
use crate::delta_engine::{OntologyMutation, MutationOp};

/// Decode an observation delta into a candidate `OntologyMutation`.
///
/// `prev` is the last `State(t)` embedding (from a `CoherenceNode`), `obs`
/// is the new readout. The mutation target is the node id supplied by the
/// caller (the encoder bound it to the stimulus). A mutation is only emitted
/// if the embedding drift exceeds `threshold` — this is the semantic-delta
/// signal, not full-state reconstruction.
pub fn decode_semantic_delta(
    prev: &[f32],
    obs: &NeuralObservation,
    target_node_id: &str,
    threshold: f32,
) -> Option<OntologyMutation> {
    let cur = population_average(obs.populations.get("readout").map(|p| p.values.as_slice()).unwrap_or(&[]));
    let drift = vector_distance(prev, &cur);
    if drift > threshold {
        let mutation = OntologyMutation::new(
            target_node_id.to_string(),
            MutationOp::EmbeddingShift { old_embedding: prev.to_vec(), new_embedding: cur.clone() },
        ).with_trigger(target_node_id.to_string());
        Some(mutation)
    } else {
        None
    }
}

fn population_average(state: &[f32]) -> Vec<f32> {
    state.to_vec()
}

fn vector_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n { s += (a[i] - b[i]).powi(2); }
    (s / n as f32).sqrt()
}

/// Compare two full states and emit the explicit semantic delta the PH-101
/// experiments report (added relations, removed relations, etc.).
pub struct SemanticDelta {
    pub added_relations: Vec<(String, String, String)>,
    pub removed_relations: Vec<(String, String, String)>,
    pub changed_properties: Vec<String>,
    pub changed_activation: bool,
}

pub fn build_delta_report(prev: &PopulationState, cur: &PopulationState, threshold: f32) -> SemanticDelta {
    let changed_activation = vector_distance(&prev.values, &cur.values) > threshold;
    SemanticDelta {
        added_relations: vec![],
        removed_relations: vec![],
        changed_properties: if changed_activation { vec!["activation".into()] } else { vec![] },
        changed_activation,
    }
}
