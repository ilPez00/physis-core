//! # Physis Core (`physis-core`)
//!
//! **An engine that maintains competing interpretations of reality, evaluates their
//! coherence with observations, processes, evidence, and outcomes, and preferentially
//! retains interpretations that continue to work.**
//!
//! ---
//!
//! ## Core Epistemic Thesis
//!
//! Traditional knowledge systems enforce premature convergence onto single interpretations,
//! discard contradictory evidence, or treat truth as static. `physis-core` operates under
//! an alternative paradigm:
//!
//! 1. **Competing Hypotheses**: Multiple conflicting interpretations of an observation or process
//!    coexist simultaneously in the epistemic graph.
//! 2. **Multidimensional Coherence**: Interpretations are evaluated across five distinct dimensions:
//!    - **Semantic Fit**: Alignment with known linguistic and conceptual definitions.
//!    - **Ontological Grounding**: Proximity to validated domain/mode categories in the Semiotic Grid.
//!    - **Logical Consistency**: Lack of active contradictions or structural invalidities.
//!    - **Empirical Support**: Corroboration by observed evidence and outcomes.
//!    - **Process Coherence**: Temporal and causal alignment with operational cycles.
//! 3. **Truth Maintenance & Non-Destructive Contradictions**: When evidence conflicts, the system
//!    preserves both claims, records the tension as a typed `Contradiction`, and applies contextual
//!    preferencing without destroying dissenting historical data.
//! 4. **Epistemic Audit & Time Machine Replay**: Every hypothesis generation, evidence attachment,
//!    confidence drift, and contradiction resolution is recorded in an immutable, append-only
//!    `EpistemicAuditTrail`. Any belief state can be replayed and reconstructed at any historical
//!    timestamp $t$.
//! 5. **Quality Feedback Loop**: Cell-level and context-dependent penalties penalize domains that
//!    lead to failed predictions, dynamically modifying retrieval and classification scores.
//!
//! ---
//!
//! ## Architectural Overview
//!
//! ```text
//!                              ┌────────────────────────────────────────┐
//!                              │         Text / Sensor Stream           │
//!                              └───────────────────┬────────────────────┘
//!                                                  │
//!                                           [Vector Embedder]
//!                                                  │
//!                                                  ▼
//! ┌──────────────────────┐             ┌──────────────────────┐             ┌──────────────────────┐
//! │  Semiotic Grid (70)  │ ◄────────── │    Cell Classifier   │ ──────────► │ Unsupervised Discover│
//! │ 5 Domains × 14 Modes │             │ (Nearest-Centroid)   │             │ (Ontology Gap / HDBS)│
//! └──────────────────────┘             └──────────┬───────────┘             └──────────────────────┘
//!                                                 │
//!                                                 ▼
//!                                      ┌──────────────────────┐
//!                                      │      PhysisCore      │
//!                                      └──────────┬───────────┘
//!                                                 │
//!            ┌────────────────────────────────────┼────────────────────────────────────┐
//!            ▼                                    ▼                                    ▼
//! ┌──────────────────────┐             ┌──────────────────────┐             ┌──────────────────────┐
//! │ Competing Hypotheses │             │ Contradiction Engine │             │  Epistemic Audit     │
//! │ & Predictive Fitness │             │ & Truth Maintenance  │             │ & Historical Replay  │
//! └──────────────────────┘             └──────────────────────┘             └──────────────────────┘
//!            │                                    │                                    │
//!            └────────────────────────────────────┼────────────────────────────────────┘
//!                                                 │
//!                                                 ▼
//!                                      ┌──────────────────────┐
//!                                      │ Contextual Feedback  │
//!                                      │  & Quality Penalties │
//!                                      └──────────────────────┘
//! ```
//!
//! ---
//!
//! ## Quick Start Example
//!
//! ```rust
//! use physis_core::{
//!     PhysisCore, RandomProjectionEmbedder, Hypothesis, HypothesisStatus, Evidence, VectorEmbed,
//! };
//!
//! // 1. Initialize Embedder and Engine Core
//! let embedder = RandomProjectionEmbedder::new(64);
//! let mut core = PhysisCore::new();
//!
//! // 2. Create Competing Hypotheses for a Machine Temperature Anomaly
//! let emb_a = embedder.embed("Spindle bearing lubrication breakdown causing friction");
//! let mut hyp_a = Hypothesis::new("Spindle bearing lubrication breakdown", emb_a);
//! hyp_a.assumptions.push("Coolant pump flow is nominal".to_string());
//! // 3. Attach Empirical Observations to the hypothesis before registering
//! hyp_a.supporting_evidence.push(Evidence::supports(
//!     "vibration_sensor_accelerometer",
//!     "High frequency harmonics match bearing ball-pass frequency",
//! ));
//! let id_a = core.register_hypothesis(hyp_a);
//!
//! let emb_b = embedder.embed("Thermal sensor telemetry calibration drift");
//! let hyp_b = Hypothesis::new("Thermal sensor telemetry calibration drift", emb_b);
//! let id_b = core.register_hypothesis(hyp_b);
//!
//! // 4. Evaluate Fitness and Resolve Preferred Interpretation
//! core.transition_hypothesis(
//!     &id_a,
//!     HypothesisStatus::Supported,
//!     "Vibration telemetry corroborated bearing breakdown",
//!     Some("vibration_sensor_accelerometer".to_string()),
//! );
//!
//! // 5. Epistemic Audit Trail records registration + status transitions
//! assert_eq!(core.hypotheses.len(), 2);
//! assert_eq!(core.epistemic_audit.events.len(), 3);
//! ```
//!
//! ---
//!
//! ## Module Index
//!
//! - [`classify`]: Nearest-centroid semiotic classification against 70 canonical cells + custom axes.
//! - [`coherence_dimensions`]: Multidimensional coherence scoring (`Semantic`, `Ontological`, `Logical`, `Empirical`, `Process`).
//! - [`coherence_query`]: Query builder for filtering nodes and hypotheses by coherence bounds and verdict states.
//! - [`contradiction`]: Tension tracking, polarity detection, and contextual preferencing without information loss.
//! - [`core`]: The main [`PhysisCore`] knowledge graph containing coherence nodes, hypotheses, edges, and dreaming loops.
//! - [`discovery`]: Unsupervised ontology gap analysis and proposal clustering for novel domains.
//! - [`embed`]: Vector embedding trait [`VectorEmbed`] and lightweight deterministic [`RandomProjectionEmbedder`].
//! - [`embed_onnx`]: Optional high-fidelity ONNX embedding runtime (MiniLM / BERT) via `ort`.
//! - [`epistemic`]: Append-only audit stream and time-machine historical replay.
//! - [`explanation`]: Structured explanation report generation with provenance chains and causal grounding.
//! - [`history`]: Ingestion adapters for browser bookmarks, browser history, OPML feeds, and chat logs.
//! - [`hypothesis`]: Hypotheses, evidence polarity, predictions, revisions, and composite fitness breakdowns.
//! - [`ontology`]: Multi-domain ontology loaders (Praxis, Machine Process, Agent Workflow, Office Operations).
//! - [`praxis`]: Life-log and behavioral records with asserted verdicts and feedback integration.
//! - [`process`]: Industrial process cycles, tasks, state machines, and temporal deviations.
//! - [`provenance`]: Cryptographic hash chains and provenance tracking for epistemological traceability.
//! - [`quality`]: Reinforcement feedback loops with cell penalties and contextual fitness adjustments.
//! - [`rag`]: Token-budget bounded retrieval-augmented generation with MMR diversity filtering.
//! - [`vault`]: Markdown knowledge vault and Git commit history importers.
//! - [`studio`]: Embedded lightweight web studio GUI and RESTful API endpoints.

pub mod classify;
pub mod coherence_dimensions;
pub mod coherence_query;
pub mod contradiction;
pub mod core;
pub mod delta_engine;
pub mod discovery;
pub mod edition;
pub mod embed;
pub mod epistemic;
pub mod explanation;
pub mod history;
pub mod hypothesis;
pub mod models;
pub mod ontology;
pub mod praxis;
pub mod process;
pub mod provenance;
pub mod experiments;
pub mod quality;
pub mod rag;
pub mod relation;
pub mod store;
pub mod temporal;
pub mod vault;

#[cfg(feature = "studio")]
pub mod studio;
#[cfg(feature = "studio")]
pub mod studio_communities;

#[cfg(feature = "studio")]
pub mod studio_lab;

#[cfg(feature = "embed-onnx")]
pub mod embed_onnx;

pub use classify::{CellClassifier, CellScore, TopKStrategy};
pub use coherence_dimensions::{CoherenceDimension, CoherenceProfile};
pub use coherence_query::{
    EpistemicQuery, EpistemicQueryResult, FailedPredictionSummary, HypothesisSummary,
};
pub use contradiction::{Contradiction, ContradictionParty, ResolutionStatus};
pub use core::PhysisCore;
pub use delta_engine::{
    evaluate_mutation, EvaluationContext, HypothesisTransition, MutationOp, NodeDelta,
    OntologyDeltaReport, OntologyMutation, DEGRADATION_THRESHOLD, GAMMA, MAX_PROPAGATION_DEPTH,
    MIN_IMPACT,
};
pub use discovery::{discover, DiscoveryConfig, DiscoveryReport, ProposedDomain};
pub use embed::{RandomProjectionEmbedder, VectorEmbed};
pub use epistemic::{EpistemicAuditTrail, EpistemicEvent, EpistemicEventType};
pub use explanation::{ExplanationReport, HistoricalPrecedent};
pub use history::importer_for as history_importer_for;
pub use hypothesis::{
    Evidence, EvidencePolarity, FitnessBreakdown, Hypothesis, HypothesisStatus, Prediction,
    Revision,
};
pub use models::*;
pub use ontology::OntologyLoader;
pub use praxis::{BehaviourRecord, BehaviourStatus};
pub use process::{
    ProcessConstraint, ProcessCycle, ProcessDeviation, ProcessGoal, ProcessIntervention,
    ProcessMeasurement, ProcessOutcome, ProcessPlan, ProcessResource, ProcessState, ProcessTask,
    StateTransition, TaskState,
};
pub use provenance::{ProvenanceChain, ProvenanceLink};
pub use quality::{
    ContextualQualityTracker, FitnessContext, FitnessRecord, QualityFailure, QualityTracker,
};
pub use rag::{count_tokens, RagChunk, RagCorpus, RetrievalResult, TokenFixedRetriever};
pub use relation::{RelationType, TypedEdge};
pub use temporal::TemporalValidity;
pub use vault::{collect_labels as collect_vault_labels, scan_vault, VaultDoc};

#[cfg(feature = "studio")]
pub use studio::{run, run_with_model, StudioState};

#[cfg(feature = "embed-onnx")]
pub use embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
