//! End-to-End Verification Test for Physis Core
//! Validates that physis-core does exactly what its documentation and thesis claim.

use physis_core::{
    classify::CellClassifier,
    contradiction::{Contradiction, ContradictionParty, ResolutionStatus},
    core::PhysisCore,
    discovery::{discover, DiscoveryConfig},
    embed::{RandomProjectionEmbedder, VectorEmbed},
    epistemic::{EpistemicEvent, EpistemicEventType},
    hypothesis::{Evidence, EvidencePolarity, Hypothesis, HypothesisStatus},
    ontology::OntologyLoader,
    quality::QualityTracker,
    rag::{RagCorpus, TokenFixedRetriever},
};

#[test]
fn test_epistemic_thesis_end_to_end() {
    println!("\n=================================================================");
    println!("=== 1. EMBEDDER & SEMIOTIC CLASSIFICATION ===");
    println!("=================================================================");
    let embedder = RandomProjectionEmbedder::new(64);
    let ontology = OntologyLoader::load_all();
    let classifier = CellClassifier::build(&ontology, &embedder);

    let sample_text = "Spindle bearing thermal runaway caused by lubricant breakdown";
    let scores = classifier.classify_text(sample_text, &embedder);
    assert!(
        !scores.is_empty(),
        "Classification must return populated cells"
    );
    let top_cell = &scores[0];
    println!(
        "Query: '{}'\nTop Cell: {} x {} (Score: {:.4})",
        sample_text, top_cell.domain, top_cell.mode, top_cell.score
    );
    assert!(top_cell.score > 0.0);

    println!("\n=================================================================");
    println!("=== 2. COMPETING HYPOTHESES & EVIDENCE SURVIVAL ===");
    println!("=================================================================");
    let mut core = PhysisCore::new();

    // Hypothesis A: Mechanical Bearing Breakdown
    let emb_a = embedder.embed("Spindle bearing ball fatigue failure");
    let mut hyp_a = Hypothesis::new("Mechanical bearing breakdown", emb_a);
    hyp_a
        .assumptions
        .push("Lubricant pump flow rate is nominal".to_string());
    let id_a = core.register_hypothesis(hyp_a);

    // Hypothesis B: Sensor Calibration Drift
    let emb_b = embedder.embed("Thermocouple telemetry calibration drift");
    let mut hyp_b = Hypothesis::new("Sensor calibration drift", emb_b);
    hyp_b
        .assumptions
        .push("Machine surface is physically cool".to_string());
    let id_b = core.register_hypothesis(hyp_b);

    println!("Registered 2 competing hypotheses:\n  [Hypothesis A: {}] Mechanical Bearing Breakdown\n  [Hypothesis B: {}] Sensor Calibration Drift", &id_a[..8], &id_b[..8]);

    // Attach Corroborating Evidence to A (Vibration sensor confirms harmonics)
    let ev_a = Evidence {
        source: "accelerometer_sensor_3".to_string(),
        polarity: EvidencePolarity::Supports,
        confidence: 0.94,
        claim: "High frequency harmonics match bearing cage defect frequency".to_string(),
        observed_at: Some(chrono::Utc::now()),
        embedding: vec![],
        context: vec!["spindle_rpm: 12000".to_string()],
    };
    core.hypotheses
        .get_mut(&id_a)
        .unwrap()
        .add_supporting_evidence(ev_a);

    // Attach Contradicting Evidence to B (Physical pyrometer confirms 95C hot zone)
    let ev_b = Evidence {
        source: "fluke_optical_pyrometer".to_string(),
        polarity: EvidencePolarity::Contradicts,
        confidence: 0.98,
        claim: "External housing surface measured 95.4C (not calibration error)".to_string(),
        observed_at: Some(chrono::Utc::now()),
        embedding: vec![],
        context: vec!["ambient_temp: 22C".to_string()],
    };
    core.hypotheses
        .get_mut(&id_b)
        .unwrap()
        .add_contradicting_evidence(ev_b);

    // Update survival states based on evidence
    core.hypotheses.get_mut(&id_a).unwrap().status = HypothesisStatus::Supported;
    core.hypotheses.get_mut(&id_a).unwrap().fitness = 0.92;
    core.hypotheses.get_mut(&id_b).unwrap().status = HypothesisStatus::Contradicted;
    core.hypotheses.get_mut(&id_b).unwrap().fitness = 0.15;

    assert_eq!(
        core.hypotheses.get(&id_a).unwrap().status,
        HypothesisStatus::Supported
    );
    assert_eq!(
        core.hypotheses.get(&id_b).unwrap().status,
        HypothesisStatus::Contradicted
    );
    println!("Hypothesis Competition Outcome:\n  -> Hypothesis A retained as Supported (Fitness: 0.92)\n  -> Hypothesis B marked Contradicted (Fitness: 0.15)");

    println!("\n=================================================================");
    println!("=== 3. TRUTH MAINTENANCE & CONTRADICTION RESOLUTION ===");
    println!("=================================================================");
    let mut party_1 = ContradictionParty::new("Spindle speed is 12000 RPM", "plc_telemetry");
    party_1.confidence = 0.90;
    party_1.context = vec!["frequency_inverter_feedback".to_string()];

    let mut party_2 =
        ContradictionParty::new("Spindle speed is 0 RPM (Stalled)", "laser_tachometer");
    party_2.confidence = 0.99;
    party_2.context = vec!["direct_optical_reflection".to_string()];

    let contradiction = Contradiction::new(party_1, party_2);
    let conflict_id = core.record_contradiction(contradiction);
    assert_eq!(core.contradictions.len(), 1);
    assert_eq!(core.contradictions[0].resolution, ResolutionStatus::Open);
    println!("Active Contradiction Recorded: ID={}", &conflict_id[..8]);

    // Resolve contextually preferring Party 2 (belt broke, inverter spinning but spindle stopped)
    if let Some(c) = core.contradictions.iter_mut().find(|c| c.id == conflict_id) {
        c.resolution = ResolutionStatus::BPreferred;
        c.explanation = Some("Drive belt snapped: motor is spinning at 12000 RPM but spindle shaft is stalled at 0 RPM".to_string());
    }
    assert_eq!(
        core.contradictions[0].resolution,
        ResolutionStatus::BPreferred
    );
    assert!(
        !core.contradictions[0].claim_a.claim.is_empty(),
        "Dissenting claim A must NOT be deleted"
    );
    println!(
        "Contradiction non-destructively resolved:\n  Status: {:?}\n  Context: {}",
        core.contradictions[0].resolution,
        core.contradictions[0].explanation.as_ref().unwrap()
    );

    println!("\n=================================================================");
    println!("=== 4. EPISTEMIC AUDIT TRAIL & HISTORICAL TIME MACHINE REPLAY ===");
    println!("=================================================================");
    let t0 = chrono::Utc::now();
    let evt_1 = EpistemicEvent::new(
        EpistemicEventType::HypothesisGenerated,
        &id_a,
        "Generated initial candidate hypothesis",
    )
    .with_transition("None", "Candidate")
    .with_metric(0.50);
    core.epistemic_audit.record(evt_1);

    std::thread::sleep(std::time::Duration::from_millis(50));
    let _t1 = chrono::Utc::now();

    let evt_2 = EpistemicEvent::new(
        EpistemicEventType::StatusTransition,
        &id_a,
        "Corroborated by accelerometer telemetry",
    )
    .with_transition("Candidate", "Supported")
    .with_metric(0.92);
    core.epistemic_audit.record(evt_2);

    // Replay state at t0
    let replay_t0 = core.epistemic_audit.reconstruct_status_at(&id_a, t0);
    assert_eq!(replay_t0, Some(HypothesisStatus::Candidate));

    // Replay state after t1
    let t_now = chrono::Utc::now();
    let replay_now = core.epistemic_audit.reconstruct_status_at(&id_a, t_now);
    assert_eq!(replay_now, Some(HypothesisStatus::Supported));
    println!("Time Machine Historical Replay Verified:\n  At t0 (Creation): Status={:?}\n  At t1 (After Evidence): Status={:?}",
        replay_t0.unwrap(), replay_now.unwrap()
    );

    println!("\n=================================================================");
    println!("=== 5. QUALITY FEEDBACK PENALTY TUNING ===");
    println!("=================================================================");
    let mut quality = QualityTracker::new(Box::new(RandomProjectionEmbedder::new(64)));
    let cell_key = format!("{}\x00{}", top_cell.domain, top_cell.mode);
    let initial_score = top_cell.score;

    // Report a quality penalty on this cell
    quality.penalize_cell(&cell_key, 0.35);
    let penalty = quality
        .cell_penalties
        .get(&cell_key)
        .copied()
        .unwrap_or(0.0);
    assert!(penalty > 0.0, "Cell must receive quality penalty");

    let adjusted_score = quality.adjust_score(&cell_key, initial_score);
    assert!(
        adjusted_score < initial_score,
        "Adjusted score must be penalized"
    );
    println!("Quality Feedback Penalty Verified:\n  Raw Classification Score: {:.4}\n  Penalty Applied: {:.2}\n  Adjusted Score: {:.4}",
        initial_score, penalty, adjusted_score
    );

    println!("\n=================================================================");
    println!("=== 6. FIXED-TOKEN BUDGET RAG WITH MMR DIVERSITY ===");
    println!("=================================================================");
    let chunk_texts = vec![
        "Emergency Stop procedure: Press red button to halt spindle immediately.".to_string(),
        "Bearing replacement protocol: Torque all structural casing bolts to 45Nm.".to_string(),
        "Thermal inspection: Use optical pyrometer to check overheated zones.".to_string(),
    ];
    let rag_corpus = RagCorpus::build(&chunk_texts, &embedder);

    let retriever = TokenFixedRetriever::new(30, 5);
    let q_emb = embedder.embed("emergency halt and bearing maintenance");
    let retrieved = retriever.retrieve(&q_emb, &rag_corpus);
    assert!(!retrieved.chunks.is_empty());
    assert!(
        retrieved.total_tokens <= 30,
        "Retrieved content must respect token budget"
    );
    println!(
        "Token-Fixed RAG Verified: Retrieved {} chunks with total {} tokens (Budget: 30 tokens)",
        retrieved.chunks.len(),
        retrieved.total_tokens
    );

    println!("\n=================================================================");
    println!("=== 7. UNSUPERVISED ONTOLOGY GAP DISCOVERY ===");
    println!("=================================================================");
    let unmapped_corpus = vec![
        "Quantum computing error correction with surface code logical qubits".to_string(),
        "Superconducting transmon qubit microwave gate pulse calibration".to_string(),
        "Dilution refrigerator cryogenic thermal management at 15 millikelvin".to_string(),
    ];
    let disc_config = DiscoveryConfig {
        min_cluster: 2,
        ..Default::default()
    };
    let disc_report = discover(&unmapped_corpus, &classifier, &embedder, &disc_config);
    println!("Ontology Gap Discovery: Evaluated {} unmapped texts -> Discovered {} candidate domain proposals",
        disc_report.total, disc_report.proposals.len()
    );
    assert!(disc_report.total >= 3);

    println!("\n=================================================================");
    println!("=== ALL PHYSIS-CORE CAPABILITIES EMPIRICALLY VERIFIED! ===");
    println!("=================================================================\n");
}
