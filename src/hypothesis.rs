//! First-class hypothesis model.
//!
//! A Hypothesis is a structured interpretation of an observation. It carries:
//! - statement and semantic representation (embedding)
//! - ontology references
//! - supporting and contradicting evidence
//! - assumptions
//! - predictions, expected outcomes, and actual outcomes
//! - confidence, multidimensional coherence, and decomposed fitness scores
//! - provenance
//! - revision history
//! - status in an extensible closed state machine
//!
//! Physis distinguishes:
//!   similarity ≠ consistency ≠ ontological validity ≠ empirical support ≠ predictive success ≠ causal support ≠ operational fitness ≠ truth.

use serde::{Deserialize, Serialize};

use crate::coherence_dimensions::CoherenceProfile;
use crate::models::Score;
use crate::temporal::TemporalValidity;

/// Closed status machine for a hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisStatus {
    /// Under evaluation; not yet supported or refuted.
    Candidate,
    /// Supported by evidence but not yet certified.
    Supported,
    /// Contradicted by at least one piece of evidence.
    Contradicted,
    /// Fully confirmed by multiple independent sources.
    Confirmed,
    /// Neither supported nor refuted; inactive.
    Inert,
    /// Refuted by observation or experiment.
    Failed,
    /// Replaced by a newer hypothesis.
    Superseded,
    /// No current relationship to observations.
    Isolated,
    /// Certified by an authority (human or process).
    Certified,
}

impl HypothesisStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            HypothesisStatus::Candidate => "candidate",
            HypothesisStatus::Supported => "supported",
            HypothesisStatus::Contradicted => "contradicted",
            HypothesisStatus::Confirmed => "confirmed",
            HypothesisStatus::Inert => "inert",
            HypothesisStatus::Failed => "failed",
            HypothesisStatus::Superseded => "superseded",
            HypothesisStatus::Isolated => "isolated",
            HypothesisStatus::Certified => "certified",
        }
    }
}

/// One piece of evidence for or against a hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Short label, e.g. "maintenance_report_2026_08_12.txt"
    pub source: String,
    /// Supports or contradicts.
    pub polarity: EvidencePolarity,
    /// How confident we are in this evidence (0.0–1.0).
    pub confidence: Score,
    /// Free-text summary or extracted claim.
    pub claim: String,
    /// When this evidence was observed or recorded.
    pub observed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Optional embedding of the claim for similarity search.
    #[serde(default)]
    pub embedding: Vec<f32>,
    /// Optional contextual tags (e.g. machine, operator, environment).
    #[serde(default)]
    pub context: Vec<String>,
}

impl Evidence {
    pub fn new(claim: impl Into<String>, source: impl Into<String>) -> Self {
        Self::supports(source, claim)
    }

    pub fn supports(source: impl Into<String>, claim: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            polarity: EvidencePolarity::Supports,
            confidence: 1.0,
            claim: claim.into(),
            observed_at: Some(chrono::Utc::now()),
            embedding: Vec::new(),
            context: Vec::new(),
        }
    }

    pub fn contradicts(source: impl Into<String>, claim: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            polarity: EvidencePolarity::Contradicts,
            confidence: 1.0,
            claim: claim.into(),
            observed_at: Some(chrono::Utc::now()),
            embedding: Vec::new(),
            context: Vec::new(),
        }
    }

    pub fn with_weight(mut self, weight: Score) -> Self {
        self.confidence = weight;
        self
    }
}

/// Whether evidence supports or contradicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidencePolarity {
    Supports,
    Contradicts,
}

/// One prediction made by a hypothesis, with its expected and actual outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// What was predicted.
    pub statement: String,
    /// Expected outcome if hypothesis holds.
    #[serde(default)]
    pub expected_outcome: Option<String>,
    /// When the prediction was made.
    pub made_at: chrono::DateTime<chrono::Utc>,
    /// When the outcome was observed (None = pending).
    pub observed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// What actually happened.
    pub actual_outcome: Option<String>,
    /// Did the prediction match reality? None = pending.
    pub correct: Option<bool>,
    /// Confidence in this prediction.
    #[serde(default = "default_one")]
    pub confidence: Score,
}

impl Prediction {
    pub fn new(statement: impl Into<String>) -> Self {
        Self {
            statement: statement.into(),
            expected_outcome: None,
            made_at: chrono::Utc::now(),
            observed_at: None,
            actual_outcome: None,
            correct: None,
            confidence: 1.0,
        }
    }
}

fn default_one() -> Score {
    1.0
}

/// Decomposed fitness score representing empirical and operational survival.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FitnessBreakdown {
    /// Semantic fit with domain vocabulary.
    pub semantic_fit: Score,
    /// Fit within the ontology schema and constraints.
    pub ontological_fit: Score,
    /// Internal non-contradiction with beliefs.
    pub logical_consistency: Score,
    /// Ratio and quality of supporting empirical observations.
    pub empirical_support: Score,
    /// Historical accuracy of predictions made by this hypothesis.
    pub predictive_success: Score,
    /// Operational outcomes in real workflows (PDCA / tasks).
    pub outcome_success: Score,
    /// Penalty deducted from active contradictions.
    pub contradiction_penalty: Score,
    /// Penalty deducted from failed predictions.
    pub failed_prediction_penalty: Score,
    /// Inspectable composite fitness in [0.0, 1.0].
    pub composite_fitness: Score,
}

impl Default for FitnessBreakdown {
    fn default() -> Self {
        Self {
            semantic_fit: 0.5,
            ontological_fit: 0.5,
            logical_consistency: 0.5,
            empirical_support: 0.5,
            predictive_success: 0.5,
            outcome_success: 0.5,
            contradiction_penalty: 0.0,
            failed_prediction_penalty: 0.0,
            composite_fitness: 0.5,
        }
    }
}

/// One revision step in a hypothesis's history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Revision {
    /// ISO timestamp of the revision.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// What changed (free text for audit).
    pub description: String,
    /// What the status was before.
    pub previous_status: HypothesisStatus,
    /// What the status is now.
    pub new_status: HypothesisStatus,
    /// Provenance reference (e.g. feedback id, observation id).
    pub trigger: Option<String>,
}

/// A first-class interpretation of reality.
///
/// Hypotheses are the unit of evaluation, revision, and explanation in Physis.
/// They replace implicit "current classification" with an explicit, revisable,
/// evidence-tracked interpretation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    /// Unique identifier.
    pub id: String,
    /// Human-readable statement.
    pub statement: String,
    /// Semantic embedding of the statement.
    pub embedding: Vec<f32>,
    /// Ontology cells this hypothesis maps to (DOMAIN×MODE keys).
    pub ontology_refs: Vec<String>,
    /// Explicit assumptions underlying this hypothesis.
    #[serde(default)]
    pub assumptions: Vec<String>,
    /// Supporting evidence.
    pub supporting_evidence: Vec<Evidence>,
    /// Contradicting evidence.
    pub contradicting_evidence: Vec<Evidence>,
    /// Predictions made and their outcomes.
    pub predictions: Vec<Prediction>,
    /// Expected outcomes listed explicitly.
    #[serde(default)]
    pub expected_outcomes: Vec<String>,
    /// Actual outcomes observed.
    #[serde(default)]
    pub actual_outcomes: Vec<String>,
    /// Revision history (append-only).
    pub revision_history: Vec<Revision>,
    /// Current status in the closed state machine.
    pub status: HypothesisStatus,
    /// Confidence from semantic similarity / source authority (0.0–1.0).
    pub confidence: Score,
    /// Coherence profile with the current belief graph.
    pub coherence_profile: CoherenceProfile,
    /// Coherence score (composite 0.0–1.0).
    pub coherence: Score,
    /// Decomposed fitness score.
    pub fitness_breakdown: FitnessBreakdown,
    /// Fitness: how well this hypothesis has survived interaction with outcomes (0.0–1.0).
    pub fitness: Score,
    /// When this hypothesis was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// When this hypothesis was last revised.
    pub revised_at: chrono::DateTime<chrono::Utc>,
    /// Temporal validity window.
    pub temporal: TemporalValidity,
    /// Optional provenance chain entry.
    pub provenance: Option<String>,
}

impl Hypothesis {
    pub fn new(statement: impl Into<String>, embedding: Vec<f32>) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            statement: statement.into(),
            embedding,
            ontology_refs: Vec::new(),
            assumptions: Vec::new(),
            supporting_evidence: Vec::new(),
            contradicting_evidence: Vec::new(),
            predictions: Vec::new(),
            expected_outcomes: Vec::new(),
            actual_outcomes: Vec::new(),
            revision_history: vec![Revision {
                timestamp: now,
                description: "Created".to_string(),
                previous_status: HypothesisStatus::Candidate,
                new_status: HypothesisStatus::Candidate,
                trigger: None,
            }],
            status: HypothesisStatus::Candidate,
            confidence: 0.5,
            coherence_profile: CoherenceProfile::new(),
            coherence: 0.5,
            fitness_breakdown: FitnessBreakdown::default(),
            fitness: 0.5,
            created_at: now,
            revised_at: now,
            temporal: TemporalValidity::permanent(),
            provenance: None,
        }
    }

    /// Add an assumption.
    pub fn add_assumption(&mut self, assumption: impl Into<String>) {
        self.assumptions.push(assumption.into());
    }

    /// Add supporting evidence and update status and fitness.
    pub fn add_supporting_evidence(&mut self, evidence: Evidence) {
        self.supporting_evidence.push(evidence);
        if self.status == HypothesisStatus::Candidate && !self.supporting_evidence.is_empty() {
            self.status = HypothesisStatus::Supported;
        }
        self.recompute_fitness();
        self.revise("Added supporting evidence", None);
    }

    pub fn add_supporting(&mut self, evidence: Evidence) {
        self.add_supporting_evidence(evidence);
    }

    /// Add contradicting evidence and update status and fitness.
    pub fn add_contradicting_evidence(&mut self, evidence: Evidence) {
        self.contradicting_evidence.push(evidence);
        self.status = HypothesisStatus::Contradicted;
        self.recompute_fitness();
        self.revise("Added contradicting evidence", None);
    }

    pub fn add_contradicting(&mut self, evidence: Evidence) {
        self.add_contradicting_evidence(evidence);
    }

    /// Record a prediction and its outcome.
    pub fn record_prediction(&mut self, prediction: Prediction) {
        if let Some(ref outcome) = prediction.actual_outcome {
            self.actual_outcomes.push(outcome.clone());
        }
        self.predictions.push(prediction);
        self.recompute_fitness();
        self.revise("Recorded prediction outcome", None);
    }

    pub fn add_prediction(&mut self, prediction: Prediction) {
        self.record_prediction(prediction);
    }

    /// Transition to a new status, recording the revision.
    pub fn transition_to(
        &mut self,
        new_status: HypothesisStatus,
        description: impl Into<String>,
        trigger: Option<String>,
    ) {
        let previous = self.status;
        self.status = new_status;
        self.revised_at = chrono::Utc::now();
        self.revision_history.push(Revision {
            timestamp: self.revised_at,
            description: description.into(),
            previous_status: previous,
            new_status,
            trigger,
        });
    }

    /// Recompute the decomposed fitness score.
    pub fn recompute_fitness(&mut self) {
        let support_len = self.supporting_evidence.len() as Score;
        let contradict_len = self.contradicting_evidence.len() as Score;
        let total_ev = support_len + contradict_len;

        let empirical = if total_ev == 0.0 {
            0.5
        } else {
            let support_conf: Score = self.supporting_evidence.iter().map(|e| e.confidence).sum();
            let contradict_conf: Score = self
                .contradicting_evidence
                .iter()
                .map(|e| e.confidence)
                .sum();
            (0.5 + 0.5 * ((support_conf - contradict_conf) / total_ev)).clamp(0.0, 1.0)
        };

        let resolved_preds: Vec<&Prediction> = self
            .predictions
            .iter()
            .filter(|p| p.correct.is_some())
            .collect();
        let (pred_success, failed_preds_penalty) = if resolved_preds.is_empty() {
            (0.5, 0.0)
        } else {
            let succ = resolved_preds
                .iter()
                .filter(|p| p.correct == Some(true))
                .count() as Score;
            let fail = resolved_preds
                .iter()
                .filter(|p| p.correct == Some(false))
                .count() as Score;
            let s_ratio = succ / resolved_preds.len() as Score;
            let f_penalty = (fail * 0.15).min(0.5);
            (s_ratio, f_penalty)
        };

        let contra_penalty = (contradict_len * 0.1).min(0.4);

        let composite = (0.20 * self.fitness_breakdown.semantic_fit
            + 0.15 * self.fitness_breakdown.ontological_fit
            + 0.15 * self.fitness_breakdown.logical_consistency
            + 0.25 * empirical
            + 0.25 * pred_success
            - contra_penalty
            - failed_preds_penalty)
            .clamp(0.0, 1.0);

        self.fitness_breakdown = FitnessBreakdown {
            semantic_fit: self.fitness_breakdown.semantic_fit,
            ontological_fit: self.fitness_breakdown.ontological_fit,
            logical_consistency: self.fitness_breakdown.logical_consistency,
            empirical_support: empirical,
            predictive_success: pred_success,
            outcome_success: self.fitness_breakdown.outcome_success,
            contradiction_penalty: contra_penalty,
            failed_prediction_penalty: failed_preds_penalty,
            composite_fitness: composite,
        };

        self.fitness = composite;
    }

    /// Resolve a pending prediction: record what actually happened and whether
    /// it matched, then let fitness absorb it.
    ///
    /// `recompute_fitness` has always read `Prediction.correct` — it drives both
    /// `predictive_success` (0.25 of the composite) and
    /// `failed_prediction_penalty`. Nothing could ever *write* it, so a quarter
    /// of the fitness score was pinned at its 0.5 "no resolved predictions"
    /// default for every hypothesis that has ever existed. Resolving a
    /// prediction is the one operation that distinguishes this from a notes
    /// file, and it was the only one missing.
    ///
    /// Returns false if `index` is out of range or that prediction is already
    /// resolved. Re-resolving is refused rather than silently overwritten: a
    /// prediction whose recorded outcome can change on a second call is not a
    /// record of anything.
    pub fn resolve_prediction(
        &mut self,
        index: usize,
        actual_outcome: impl Into<String>,
        correct: bool,
    ) -> bool {
        let Some(pred) = self.predictions.get_mut(index) else {
            return false;
        };
        if pred.correct.is_some() {
            return false;
        }
        pred.actual_outcome = Some(actual_outcome.into());
        pred.observed_at = Some(chrono::Utc::now());
        pred.correct = Some(correct);
        let statement = pred.statement.clone();
        self.recompute_fitness();
        self.revise(
            format!(
                "Resolved prediction {} ({}): {}",
                index,
                if correct { "correct" } else { "WRONG" },
                statement
            ),
            None,
        );
        true
    }

    /// Predictions still awaiting an outcome, with their index so they can be
    /// resolved by position.
    pub fn open_predictions(&self) -> Vec<(usize, &Prediction)> {
        self.predictions
            .iter()
            .enumerate()
            .filter(|(_, p)| p.correct.is_none())
            .collect()
    }

    /// Convenience: total evidence count.
    pub fn evidence_count(&self) -> usize {
        self.supporting_evidence.len() + self.contradicting_evidence.len()
    }

    /// Convenience: unresolved predictions.
    pub fn pending_predictions(&self) -> usize {
        self.predictions
            .iter()
            .filter(|p| p.correct.is_none())
            .count()
    }

    fn revise(&mut self, description: impl Into<String>, trigger: Option<String>) {
        let previous = self.status;
        self.revised_at = chrono::Utc::now();
        self.revision_history.push(Revision {
            timestamp: self.revised_at,
            description: description.into(),
            previous_status: previous,
            new_status: self.status,
            trigger,
        });
    }
}

#[cfg(test)]
mod resolution_tests {
    use super::*;

    fn hyp() -> Hypothesis {
        Hypothesis::new("clogs correlate with humidity", vec![0.1; 8])
    }

    /// `recompute_fitness` has always weighted `predictive_success` at 0.25 of
    /// the composite, and nothing could write `Prediction.correct`, so that
    /// quarter was pinned at its 0.5 default for every hypothesis that ever
    /// existed. This is the test that the write path exists at all.
    #[test]
    fn resolving_a_prediction_moves_fitness() {
        let mut h = hyp();
        h.add_prediction(Prediction::new("humidity above 60% precedes a clog"));
        let before = h.fitness;
        assert!(h.resolve_prediction(0, "three clogs, all above 62%", true));
        assert!(
            h.fitness > before,
            "a correct prediction must raise fitness ({before} -> {})",
            h.fitness
        );
        assert_eq!(h.predictions[0].correct, Some(true));
        assert!(h.predictions[0].observed_at.is_some());
    }

    #[test]
    fn a_wrong_prediction_lowers_fitness() {
        let mut h = hyp();
        h.add_prediction(Prediction::new("humidity above 60% precedes a clog"));
        let before = h.fitness;
        assert!(h.resolve_prediction(0, "clogged at 31% humidity", false));
        assert!(
            h.fitness < before,
            "a wrong prediction must lower fitness ({before} -> {})",
            h.fitness
        );
    }

    /// A record whose recorded outcome can change on a second call is not a
    /// record of anything. Re-resolving is refused, not silently overwritten.
    #[test]
    fn a_resolved_prediction_cannot_be_rewritten() {
        let mut h = hyp();
        h.add_prediction(Prediction::new("p"));
        assert!(h.resolve_prediction(0, "it held", true));
        assert!(
            !h.resolve_prediction(0, "actually it did not", false),
            "re-resolving must be refused"
        );
        assert_eq!(h.predictions[0].correct, Some(true));
        assert_eq!(h.predictions[0].actual_outcome.as_deref(), Some("it held"));
    }

    #[test]
    fn resolving_out_of_range_is_refused_not_panicking() {
        let mut h = hyp();
        assert!(!h.resolve_prediction(7, "x", true));
    }

    /// `open_predictions` carries the index, because that index is what the CLI
    /// asks the reader to type back into `hypothesis resolve`.
    #[test]
    fn open_predictions_carry_a_usable_index() {
        let mut h = hyp();
        h.add_prediction(Prediction::new("first"));
        h.add_prediction(Prediction::new("second"));
        h.add_prediction(Prediction::new("third"));
        assert!(h.resolve_prediction(1, "done", true));

        let open = h.open_predictions();
        assert_eq!(open.len(), 2);
        assert_eq!(h.pending_predictions(), 2);
        assert_eq!(open[0].0, 0);
        assert_eq!(open[1].0, 2, "indices must survive a gap, not be renumbered");
        assert_eq!(open[1].1.statement, "third");
    }

    /// Resolving is a revision: the trajectory has to show it, or the audit
    /// trail is missing the only event that carries an outcome.
    #[test]
    fn resolving_appends_to_the_revision_history() {
        let mut h = hyp();
        h.add_prediction(Prediction::new("p"));
        let before = h.revision_history.len();
        h.resolve_prediction(0, "it did not hold", false);
        assert_eq!(h.revision_history.len(), before + 1);
        let last = h.revision_history.last().unwrap();
        assert!(
            last.description.contains("WRONG"),
            "the revision must say which way it went, got {:?}",
            last.description
        );
    }

    /// `correct` is a plain serialised field, so a resolved prediction must
    /// survive the JSON round-trip that `PhysisCore::persist` performs.
    #[test]
    fn a_resolution_survives_serialisation() {
        let mut h = hyp();
        h.add_prediction(Prediction::new("p"));
        h.resolve_prediction(0, "observed", false);
        let json = serde_json::to_string(&h).unwrap();
        let back: Hypothesis = serde_json::from_str(&json).unwrap();
        assert_eq!(back.predictions[0].correct, Some(false));
        assert_eq!(
            back.predictions[0].actual_outcome.as_deref(),
            Some("observed")
        );
        assert!((back.fitness - h.fitness).abs() < 1e-6);
    }
}
