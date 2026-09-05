//! The epistemic loop, end to end: believe, predict, find out, let fitness move.
//!
//! This is the code from README §1 "Competing Hypotheses & Evidence Attestation",
//! kept here as a compiled example so the README cannot drift away from the API.
//! If you change the signatures it uses, this stops building and CI says so.
//!
//!     cargo run -p physis-core --example hypothesis_loop

use physis_core::{
    Evidence, EvidencePolarity, Hypothesis, HypothesisStatus, PhysisCore, Prediction,
    RandomProjectionEmbedder, VectorEmbed,
};

fn main() {
    let embedder = RandomProjectionEmbedder::new(64);
    let mut core = PhysisCore::new();

    // Two competing explanations for one production defect. Both are kept.
    let emb_a = embedder.embed("Extrusion temperature too low causing delamination");
    let mut hyp_a = Hypothesis::new("Low nozzle temperature", emb_a);
    hyp_a
        .assumptions
        .push("Thermistor calibration is accurate".to_string());
    let id_a = core.register_hypothesis(hyp_a);

    let emb_b = embedder.embed("Filament moisture absorption causing steam bubbles");
    let hyp_b = Hypothesis::new("Wet filament spool", emb_b);
    let _id_b = core.register_hypothesis(hyp_b);

    let thermocouple = Evidence {
        source: "thermal_camera_infrared".to_string(),
        polarity: EvidencePolarity::Supports,
        confidence: 0.95,
        claim: "Melt zone thermal gradient is 18C below target setpoint".to_string(),
        observed_at: Some(chrono::Utc::now()),
        embedding: vec![],
        context: vec!["nozzle_diameter: 0.4mm".to_string()],
    };
    core.hypotheses
        .get_mut(&id_a)
        .expect("just registered")
        .add_supporting_evidence(thermocouple);

    // Commit to something falsifiable BEFORE the next run.
    let before = core.hypotheses[&id_a].fitness;
    if let Some(h) = core.hypotheses.get_mut(&id_a) {
        h.add_prediction(Prediction::new(
            "Raising the setpoint 18C eliminates delamination on the next run",
        ));

        // ... the run happens, and it does not ...
        assert!(h.resolve_prediction(0, "Delamination unchanged at +18C", false));
        assert_eq!(h.predictions[0].correct, Some(false));

        // Write-once: a second call is refused, not applied.
        assert!(!h.resolve_prediction(0, "actually it worked", true));
        assert_eq!(h.predictions[0].correct, Some(false));

        h.transition_to(HypothesisStatus::Failed, "prediction falsified", None);
    }

    let after = core.hypotheses[&id_a].fitness;
    println!("fitness {before:.3} -> {after:.3} after one prediction resolved --wrong");
    assert!(after < before, "a falsified prediction must cost fitness");

    for (idx, pending) in core.hypotheses[&id_a].open_predictions() {
        println!("#{idx} still open: {}", pending.statement);
    }
    println!("open predictions remaining: {}", core.hypotheses[&id_a].pending_predictions());
}
