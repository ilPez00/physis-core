//! Epistemic Audit Trail — replayable timeline of beliefs, observations, and revisions.
//!
//! Answers:
//!   - "What did Physis believe at time T?"
//!   - "What evidence caused the belief to change?"
//!   - "Which hypotheses evolved, were contradicted, or were superseded?"

use serde::{Deserialize, Serialize};

use crate::hypothesis::HypothesisStatus;
use crate::models::Score;
use crate::temporal::TemporalValidity;

/// Types of epistemic events recorded on the timeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EpistemicEventType {
    /// Initial assertion or baseline belief.
    BeliefFormed,
    /// New empirical observation ingested.
    ObservationIngested,
    /// Conflict/contradiction detected between two claims.
    ContradictionDetected,
    /// New candidate hypothesis generated.
    HypothesisGenerated,
    /// Prediction formulated by a hypothesis.
    PredictionFormulated,
    /// Outcome observed for a prior prediction.
    OutcomeObserved,
    /// Fitness score adjusted based on evidence or outcomes.
    FitnessShifted,
    /// Status transition of a hypothesis (e.g. Supported -> Contradicted).
    StatusTransition,
    /// Discovery of an ontology gap requiring conceptual extension.
    OntologyGapDiscovered,
    /// Promotion of a candidate ontology domain to active status.
    OntologyDomainPromoted,
}

/// A single step in the epistemic audit trail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpistemicEvent {
    /// Unique event ID.
    pub id: String,
    /// When this event occurred.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Type of epistemic change.
    pub event_type: EpistemicEventType,
    /// ID of the subject entity (hypothesis, node, observation, contradiction).
    pub subject_id: String,
    /// Human-readable explanation of what occurred.
    pub description: String,
    /// Prior state or status (if applicable).
    #[serde(default)]
    pub prior_state: Option<String>,
    /// Posterior state or status (if applicable).
    #[serde(default)]
    pub posterior_state: Option<String>,
    /// Source or trigger of this change (file, operator, sensor, feedback ID).
    #[serde(default)]
    pub source: Option<String>,
    /// Confidence or fitness value at this instant.
    #[serde(default)]
    pub metric_value: Option<Score>,
    /// G3: when the fact was *asserted by its source* (episode reference
    /// time), distinct from `timestamp` (arrival / transaction time). None
    /// on pre-G3 events — replay then treats arrival as assertion.
    #[serde(default)]
    pub asserted_at: Option<chrono::DateTime<chrono::Utc>>,
    /// G4: id of the intake episode recorded by this event — the per-link
    /// intake id the trail carries so an evidence citation resolves to the
    /// concrete ingest that produced it.
    #[serde(default)]
    pub intake_id: Option<String>,
    /// Day 4: the *world* time this revision is about, as distinct from
    /// [`Self::assertion_time`] (when it was learned) and [`Self::timestamp`]
    /// (when it arrived).
    ///
    /// `None` means **unknown**, never "always true": a record with no world
    /// time cannot be placed at any `valid_at`, and [`EpistemicAuditTrail::state`]
    /// reports it as such rather than inventing a date. Records written before
    /// this field existed load as `None`, which preserves their meaning.
    #[serde(default)]
    pub validity: Option<TemporalValidity>,
}

impl EpistemicEvent {
    pub fn new(
        event_type: EpistemicEventType,
        subject_id: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            event_type,
            subject_id: subject_id.into(),
            description: description.into(),
            prior_state: None,
            posterior_state: None,
            source: None,
            metric_value: None,
            asserted_at: None,
            intake_id: None,
            validity: None,
        }
    }

    /// G3: stamp the episode reference time (when the source asserted the
    /// fact), as opposed to `timestamp` (when the engine received it).
    pub fn with_asserted_at(mut self, asserted_at: chrono::DateTime<chrono::Utc>) -> Self {
        self.asserted_at = Some(asserted_at);
        self
    }

    /// G3: the replay clock for this event — assertion time when known,
    /// arrival otherwise (pre-G3 events). Replay is ordered by this key, so
    /// an out-of-order intake replays to the same state as an in-order one.
    pub fn assertion_time(&self) -> chrono::DateTime<chrono::Utc> {
        self.asserted_at.unwrap_or(self.timestamp)
    }

    /// Day 4: state the world-time interval this revision is about.
    ///
    /// Separate from [`Self::with_asserted_at`] on purpose — that one answers
    /// "when did we learn it", this one "when was it true". A January reading
    /// learned in March sets `asserted_at` to March and `validity` to January;
    /// nothing else in the trail can express that.
    pub fn with_validity(mut self, validity: TemporalValidity) -> Self {
        self.validity = Some(validity);
        self
    }

    pub fn with_transition(
        mut self,
        prior: impl Into<String>,
        posterior: impl Into<String>,
    ) -> Self {
        self.prior_state = Some(prior.into());
        self.posterior_state = Some(posterior.into());
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_metric(mut self, metric: Score) -> Self {
        self.metric_value = Some(metric);
        self
    }

    /// G4: stamp which intake episode this event recorded.
    pub fn with_intake_id(mut self, intake_id: impl Into<String>) -> Self {
        self.intake_id = Some(intake_id.into());
        self
    }
}

/// The full epistemic audit trail of the engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpistemicAuditTrail {
    pub events: Vec<EpistemicEvent>,
    /// G3: per-stream intake watermarks (see [`HighWaterMark`]).
    #[serde(default)]
    pub watermarks: Vec<HighWaterMark>,
}

/// G3: per-stream intake watermark — the two clocks of the trail.
///
/// `last_arrival` is the transaction clock (wall time of the latest intake
/// for this stream); `last_episode_valid_at` is the episode reference clock
/// (the newest assertion time seen). A stream that delivers an episode
/// older than the mark is *late* — the receipt says so, and replay is
/// ordered by assertion time, so the late episode still lands where it
/// belongs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HighWaterMark {
    pub stream_id: String,
    pub last_arrival: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    pub last_episode_valid_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Day 4: the answer to [`EpistemicAuditTrail::state`] — one world instant
/// asked with one knowledge cut.
///
/// Every field is reported, including the ones that produced nothing, so a
/// caller can tell "no revision covers that instant" apart from "revisions
/// cover it but their world time is unknown". Collapsing those two would make
/// the unknown case look like evidence of absence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateAt {
    /// The status the applicable revisions support for `valid_at`.
    pub status: Option<HypothesisStatus>,
    /// Events known by `known_at` whose world time is unknown, so they could
    /// not be placed at `valid_at`. They contributed nothing to `status`.
    pub unknown_time: Vec<String>,
    /// Events that produced `status`, in replay order.
    pub supporting_event_ids: Vec<String>,
    /// Events known by `known_at` whose validity interval does not cover
    /// `valid_at`. They were true at some other time, not this one.
    pub excluded_by_validity: Vec<String>,
}

/// The single place a status is derived from an ordered event list.
///
/// Both [`EpistemicAuditTrail::reconstruct_status_at`] (one clock) and
/// [`EpistemicAuditTrail::state`] (two clocks) go through this, so the two
/// queries cannot drift into disagreeing about what a given sequence of
/// revisions means.
fn evaluate_status(events: &[&EpistemicEvent]) -> Option<HypothesisStatus> {
    let mut last_status: Option<HypothesisStatus> = None;
    for ev in events {
        if ev.event_type == EpistemicEventType::HypothesisGenerated {
            last_status = Some(HypothesisStatus::Candidate);
        }
        if let Some(ref post) = ev.posterior_state {
            match post.to_lowercase().as_str() {
                "candidate" => last_status = Some(HypothesisStatus::Candidate),
                "supported" => last_status = Some(HypothesisStatus::Supported),
                "contradicted" => last_status = Some(HypothesisStatus::Contradicted),
                "confirmed" => last_status = Some(HypothesisStatus::Confirmed),
                "inert" => last_status = Some(HypothesisStatus::Inert),
                "failed" => last_status = Some(HypothesisStatus::Failed),
                "superseded" => last_status = Some(HypothesisStatus::Superseded),
                "isolated" => last_status = Some(HypothesisStatus::Isolated),
                "certified" => last_status = Some(HypothesisStatus::Certified),
                _ => {}
            }
        }
    }
    last_status
}

/// G3: what [`EpistemicAuditTrail::note_intake`] reports for one intake.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntakeReceipt {
    pub stream_id: String,
    /// True when this episode's assertion time predates the mark's.
    pub late: bool,
    /// The mark after advancing.
    pub mark: HighWaterMark,
}

impl EpistemicAuditTrail {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, event: EpistemicEvent) {
        self.events.push(event);
    }

    /// G3: note an intake on `stream_id`, advancing the per-stream
    /// watermark. Returns a receipt that flags a *late* episode (assertion
    /// time older than the mark's). Late arrivals are expected, not errors —
    /// replay is ordered by assertion time, so they still land where they
    /// belong. Neither clock moves backwards.
    pub fn note_intake(
        &mut self,
        stream_id: impl Into<String>,
        arrival: chrono::DateTime<chrono::Utc>,
        episode_valid_at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> IntakeReceipt {
        let stream_id = stream_id.into();
        let mut late = false;
        let mark = match self
            .watermarks
            .iter_mut()
            .find(|w| w.stream_id == stream_id)
        {
            Some(w) => {
                if let (Some(prev), Some(v)) = (w.last_episode_valid_at, episode_valid_at) {
                    if v < prev {
                        late = true;
                    }
                }
                w.last_arrival = w.last_arrival.max(arrival);
                if w.last_episode_valid_at
                    .is_none_or(|prev| episode_valid_at.is_some_and(|v| v > prev))
                {
                    w.last_episode_valid_at = episode_valid_at;
                }
                w.clone()
            }
            None => {
                let w = HighWaterMark {
                    stream_id: stream_id.clone(),
                    last_arrival: arrival,
                    last_episode_valid_at: episode_valid_at,
                };
                self.watermarks.push(w.clone());
                w
            }
        };
        IntakeReceipt {
            stream_id,
            late,
            mark,
        }
    }

    /// Retrieve all events relevant to a specific entity ID in chronological order.
    pub fn history_for(&self, subject_id: &str) -> Vec<&EpistemicEvent> {
        self.events
            .iter()
            .filter(|e| e.subject_id == subject_id)
            .collect()
    }

    pub fn reconstruct_status_at(
        &self,
        subject_id: &str,
        when: chrono::DateTime<chrono::Utc>,
    ) -> Option<HypothesisStatus> {
        self.replay_to(subject_id, when)
    }

    /// G3 replay core: the status after applying every event known by `when`,
    /// in assertion order.
    fn replay_to(
        &self,
        subject_id: &str,
        when: chrono::DateTime<chrono::Utc>,
    ) -> Option<HypothesisStatus> {
        let mut relevant: Vec<&EpistemicEvent> = self
            .events
            .iter()
            .filter(|ev| ev.subject_id == subject_id && ev.assertion_time() <= when)
            .collect();
        relevant.sort_by_key(|ev| ev.assertion_time());
        evaluate_status(&relevant)
    }

    /// Day 4: **two clocks, asked separately.**
    ///
    /// [`Self::reconstruct_status_at`] answers "what did we believe at K" and
    /// [`Self::point_in_time_status_at`] answers "was it visible at T" — but
    /// both take one instant, so neither can be asked with `T != K`. This one
    /// can: it is the plan's `state(valid_at=T, known_at=K)`.
    ///
    /// The three steps, in order, because the order is the semantics:
    ///
    /// 1. **Knowledge cut.** Keep events with `assertion_time() <= known_at` —
    ///    what was known by K. A January reading learned in March is not in a
    ///    February cut.
    /// 2. **Replay.** Apply those in assertion order, so arrival order cannot
    ///    change the answer.
    /// 3. **Validity partition.** Keep only the revisions whose world-time
    ///    covers `valid_at`. A revision with **no** world time is not silently
    ///    dropped and not silently trusted: it is returned in
    ///    [`StateAt::unknown_time`] and contributes nothing to the status,
    ///    which is what "do not invent missing dates" means operationally.
    ///
    /// The status is computed from the applicable revisions alone, so a
    /// revision that was true in January does not decide a query about
    /// February just because it is the newest thing known.
    pub fn state(
        &self,
        subject_id: &str,
        valid_at: chrono::DateTime<chrono::Utc>,
        known_at: chrono::DateTime<chrono::Utc>,
    ) -> StateAt {
        let mut known: Vec<&EpistemicEvent> = self
            .events
            .iter()
            .filter(|ev| ev.subject_id == subject_id && ev.assertion_time() <= known_at)
            .collect();
        known.sort_by_key(|ev| ev.assertion_time());

        let mut applicable: Vec<&EpistemicEvent> = Vec::new();
        let mut unknown_time: Vec<String> = Vec::new();
        let mut excluded_by_validity: Vec<String> = Vec::new();
        for ev in known {
            match &ev.validity {
                // Unknown world time: reported, never assumed.
                None => unknown_time.push(ev.id.clone()),
                Some(v) if v.is_valid_at(valid_at) => applicable.push(ev),
                Some(_) => excluded_by_validity.push(ev.id.clone()),
            }
        }

        StateAt {
            status: evaluate_status(&applicable),
            supporting_event_ids: applicable.iter().map(|ev| ev.id.clone()).collect(),
            unknown_time,
            excluded_by_validity,
        }
    }

    /// G6: a point-in-time query — the audited status of `subject_id` at
    /// `when`, gated by `validity` (which carries the system-invalidation
    /// `expired_at` leg). The audit trail is the authority for *what was
    /// believed*; the validity window is the authority for *whether that
    /// belief was visible at T*. They are allowed to disagree exactly on
    /// purpose: history keeps superseded and invalidated states, the
    /// validity surface does not.
    ///
    /// `None` therefore means one of two things: the subject had no
    /// audited status at `when`, or the status existed but the validity
    /// window (or the system invalidation) closed it. To tell them apart,
    /// call [`EpistemicAuditTrail::reconstruct_status_at`] — the trail
    /// still shows the state that the surface no longer exposes.
    pub fn point_in_time_status_at(
        &self,
        subject_id: &str,
        when: chrono::DateTime<chrono::Utc>,
        validity: &TemporalValidity,
    ) -> Option<HypothesisStatus> {
        if !validity.is_valid_at(when) {
            return None;
        }
        self.reconstruct_status_at(subject_id, when)
    }

    /// Format chronological summary for audit reporting.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        for (i, ev) in self.events.iter().enumerate() {
            let t = ev.timestamp.format("%Y-%m-%d %H:%M:%S UTC");
            lines.push(format!(
                "t{i} [{t}] [{:?}]: {} (subject: {})",
                ev.event_type, ev.description, ev.subject_id
            ));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone, Utc};

    fn at(month: u32, day: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, month, day, 0, 0, 0).unwrap()
    }

    /// A revision with both clocks set independently, which is the whole point
    /// of the type: `arrived` is when the engine got it, `asserted` is when the
    /// source said it, `validity` is when it was true.
    fn revision(
        subject: &str,
        posterior: &str,
        arrived: DateTime<Utc>,
        asserted: DateTime<Utc>,
        validity: Option<TemporalValidity>,
    ) -> EpistemicEvent {
        let mut ev = EpistemicEvent::new(
            EpistemicEventType::StatusTransition,
            subject,
            format!("{posterior} @ {arrived}"),
        )
        .with_asserted_at(asserted);
        ev.timestamp = arrived;
        ev.posterior_state = Some(posterior.to_string());
        ev.validity = validity;
        ev
    }

    /// The plan's Day 4 gate, verbatim: a January reading learned in March is
    /// absent from a February knowledge query and available to an April query
    /// about January.
    #[test]
    fn a_january_reading_learned_in_march_is_absent_in_february_and_present_in_april() {
        let mut trail = EpistemicAuditTrail::new();
        trail.record(revision(
            "H1",
            "supported",
            at(3, 10),                                          // arrived in March
            at(3, 10),                                          // asserted in March
            Some(TemporalValidity::during(at(1, 1), at(2, 1))), // about January
        ));

        let february = trail.state("H1", at(1, 15), at(2, 15));
        assert_eq!(
            february.status, None,
            "a March intake is not in a February knowledge cut"
        );
        assert!(february.supporting_event_ids.is_empty());
        assert!(
            february.unknown_time.is_empty(),
            "the revision has a world time; it is the cut that excludes it"
        );

        let april = trail.state("H1", at(1, 15), at(4, 1));
        assert_eq!(
            april.status,
            Some(HypothesisStatus::Supported),
            "by April the January reading is known and covers 15 January"
        );
        assert_eq!(april.supporting_event_ids.len(), 1);
    }

    /// The same two revisions, delivered in either order, must answer the same
    /// question about the past. Intermediate *knowledge* answers may differ —
    /// that is the second clock doing its job, not a bug.
    #[test]
    fn arrival_order_does_not_change_the_retrospective_answer() {
        // Built once and cloned, so the two trails hold the *same* revisions
        // and only the delivery order differs. Constructing them twice would
        // give each event a fresh uuid and the comparison would be measuring
        // the ids rather than the replay.
        let january = revision(
            "H1",
            "supported",
            at(1, 5),
            at(1, 5),
            Some(TemporalValidity::during(at(1, 1), at(2, 1))),
        );
        let february = revision(
            "H1",
            "contradicted",
            at(2, 5),
            at(2, 5),
            Some(TemporalValidity::during(at(2, 1), at(3, 1))),
        );

        let in_order = {
            let mut t = EpistemicAuditTrail::new();
            t.record(january.clone());
            t.record(february.clone());
            t
        };
        let backfilled = {
            let mut t = EpistemicAuditTrail::new();
            // February's revision arrives first, January's lands late. Both
            // carry their own assertion times, so replay sorts them back.
            t.record(february);
            t.record(january);
            t
        };

        for valid_at in [at(1, 15), at(2, 15)] {
            assert_eq!(
                in_order.state("H1", valid_at, at(4, 1)),
                backfilled.state("H1", valid_at, at(4, 1)),
                "arrival order changed the answer for {valid_at}"
            );
        }
        // And the January instant resolves to January's revision, not to the
        // newest thing on the trail.
        assert_eq!(
            in_order.state("H1", at(1, 15), at(4, 1)).status,
            Some(HypothesisStatus::Supported)
        );
        assert_eq!(
            in_order.state("H1", at(2, 15), at(4, 1)).status,
            Some(HypothesisStatus::Contradicted)
        );
    }

    /// Interval endpoints: start inclusive, end exclusive — the existing
    /// `TemporalValidity` contract, which this must not quietly change.
    #[test]
    fn validity_endpoints_are_start_inclusive_and_end_exclusive() {
        let mut trail = EpistemicAuditTrail::new();
        trail.record(revision(
            "H1",
            "supported",
            at(1, 1),
            at(1, 1),
            Some(TemporalValidity::during(at(1, 1), at(2, 1))),
        ));

        assert_eq!(
            trail.state("H1", at(1, 1), at(4, 1)).status,
            Some(HypothesisStatus::Supported),
            "the start instant is inside the interval"
        );
        assert_eq!(
            trail.state("H1", at(2, 1), at(4, 1)).status,
            None,
            "the end instant is outside the interval"
        );
        assert_eq!(
            trail.state("H1", at(12, 31), at(4, 1)).status,
            None,
            "an instant after the interval is outside it"
        );
    }

    /// A revision with no world time is reported, never assumed. Dropping it
    /// silently would make "unknown" look like "not true", and trusting it
    /// would make it look like "true then".
    #[test]
    fn unknown_world_time_is_reported_and_contributes_nothing() {
        let mut trail = EpistemicAuditTrail::new();
        let ev = revision("H1", "supported", at(1, 5), at(1, 5), None);
        let id = ev.id.clone();
        trail.record(ev);

        let answer = trail.state("H1", at(1, 15), at(4, 1));
        assert_eq!(answer.status, None, "an unknown date proves nothing");
        assert_eq!(answer.unknown_time, vec![id]);
        assert!(answer.supporting_event_ids.is_empty());
    }

    /// A later correction inside its own interval supersedes; outside it, it
    /// does not touch the earlier instant.
    #[test]
    fn a_correction_only_applies_inside_its_own_interval() {
        let mut trail = EpistemicAuditTrail::new();
        trail.record(revision(
            "H1",
            "supported",
            at(1, 5),
            at(1, 5),
            Some(TemporalValidity::during(at(1, 1), at(6, 1))),
        ));
        trail.record(revision(
            "H1",
            "contradicted",
            at(3, 5),
            at(3, 5),
            // The correction is about March onward only.
            Some(TemporalValidity::during(at(3, 1), at(6, 1))),
        ));

        assert_eq!(
            trail.state("H1", at(2, 1), at(4, 1)).status,
            Some(HypothesisStatus::Supported),
            "before the correction's interval, the original reading stands"
        );
        assert_eq!(
            trail.state("H1", at(4, 1), at(4, 1)).status,
            Some(HypothesisStatus::Contradicted),
            "inside the correction's interval, it wins"
        );
        let excluded = trail.state("H1", at(2, 1), at(4, 1)).excluded_by_validity;
        assert_eq!(excluded.len(), 1, "the correction is excluded, not ignored");
    }

    /// Restart: the world time survives a round trip, and a record written
    /// before the field existed still loads — as unknown, not as always-true.
    #[test]
    fn world_time_survives_serialization_and_legacy_records_load_as_unknown() {
        let mut trail = EpistemicAuditTrail::new();
        trail.record(revision(
            "H1",
            "supported",
            at(1, 5),
            at(1, 5),
            Some(TemporalValidity::during(at(1, 1), at(2, 1))),
        ));
        let json = serde_json::to_string(&trail).unwrap();
        let reloaded: EpistemicAuditTrail = serde_json::from_str(&json).unwrap();
        assert_eq!(trail.events, reloaded.events);
        assert_eq!(
            reloaded.state("H1", at(1, 15), at(4, 1)).status,
            Some(HypothesisStatus::Supported)
        );

        // A pre-Day-4 record: no `validity`, no `asserted_at`, no `intake_id`.
        let legacy = r#"{"events":[{"id":"e1","timestamp":"2026-01-05T00:00:00Z",
            "event_type":"StatusTransition","subject_id":"H1","description":"legacy",
            "posterior_state":"supported"}],"watermarks":[]}"#;
        let old: EpistemicAuditTrail = serde_json::from_str(legacy).unwrap();
        let answer = old.state("H1", at(1, 15), at(4, 1));
        assert_eq!(
            answer.unknown_time,
            vec!["e1".to_string()],
            "a record with no world time must load as unknown, not as valid"
        );
        assert_eq!(answer.status, None);
        // The one-clock query keeps its documented meaning for that record.
        assert_eq!(
            old.reconstruct_status_at("H1", at(4, 1)),
            Some(HypothesisStatus::Supported)
        );
    }
}
