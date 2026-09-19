//! Temporal validity — when is a statement true?
//!
//! Physis must distinguish:
//!
//!   X is true
//!   X was true at time T
//!   X is true during interval [start, end]
//!
//! This module provides the primitives.

use serde::{Deserialize, Serialize};

/// A point in time or an interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalValidity {
    /// When this becomes valid. None = valid from the beginning of time.
    pub valid_from: Option<chrono::DateTime<chrono::Utc>>,
    /// When this ceases to be valid. None = still valid / permanent.
    pub valid_until: Option<chrono::DateTime<chrono::Utc>>,
    /// Optional event that triggered the validity (e.g. "startup", "intervention").
    #[serde(default)]
    pub trigger: Option<String>,
    /// System-recorded invalidation time (Graphiti `expired_at` leg): when the
    /// record itself was superseded, distinct from `valid_until` (when the
    /// claim stopped being true). None = never invalidated. **P2/G6 reads
    /// this leg**: a point-in-time query excludes records past
    /// `expired_at` while the audit trail keeps them replayable (invalidate,
    /// don't delete).
    #[serde(default)]
    pub expired_at: Option<chrono::DateTime<chrono::Utc>>,
    /// R1 (development-cycle-2026-09-15): this window is *unplaceable* in
    /// world time, not merely unbounded. `valid_from: None, valid_until:
    /// None` already means "permanent" (always valid) — the two are
    /// opposite claims and conflating them was the gap `UNKNOWN_TIME` exists
    /// to close: undated evidence must satisfy no `valid_at` query, not
    /// every one. `#[serde(default)]` so every pre-existing `TemporalValidity`
    /// (all Hypothesis-level windows to date) keeps meaning "known, unbounded"
    /// on deserialization, not "unknown".
    #[serde(default)]
    pub valid_unknown: bool,
}

impl TemporalValidity {
    /// Permanent validity (no time bounds).
    pub fn permanent() -> Self {
        Self {
            valid_from: None,
            valid_until: None,
            trigger: None,
            expired_at: None,
            valid_unknown: false,
        }
    }

    /// Valid from a specific point onward.
    pub fn from(instant: chrono::DateTime<chrono::Utc>) -> Self {
        Self {
            valid_from: Some(instant),
            valid_until: None,
            trigger: None,
            expired_at: None,
            valid_unknown: false,
        }
    }

    /// Valid during an interval.
    pub fn during(
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        Self {
            valid_from: Some(start),
            valid_until: Some(end),
            trigger: None,
            expired_at: None,
            valid_unknown: false,
        }
    }

    /// R1: no extractable world-time interval at all (E77's "dates in
    /// prose" case, or a genuinely undated report). Distinct from
    /// [`Self::permanent`] — that means "known to always hold"; this means
    /// "we cannot place it in time," and must satisfy no [`Self::is_valid_at`]
    /// query rather than every one.
    pub fn unknown() -> Self {
        Self {
            valid_from: None,
            valid_until: None,
            trigger: None,
            expired_at: None,
            valid_unknown: true,
        }
    }

    /// Is this statement valid at the given instant?
    ///
    /// P2/G6: read all three legs. The claim's own interval
    /// (`valid_from` / `valid_until`) says when the statement was true; the
    /// system leg (`expired_at`) says when the *record* was superseded. A
    /// point-in-time query honours both — history keeps the old record, the
    /// validity surface does not.
    pub fn is_valid_at(&self, when: chrono::DateTime<chrono::Utc>) -> bool {
        if self.valid_unknown {
            return false;
        }
        if let Some(from) = self.valid_from {
            if when < from {
                return false;
            }
        }
        if let Some(until) = self.valid_until {
            if when >= until {
                return false;
            }
        }
        if let Some(expired) = self.expired_at {
            if when >= expired {
                return false;
            }
        }
        true
    }

    /// Is this statement valid now?
    pub fn is_current(&self) -> bool {
        self.is_valid_at(chrono::Utc::now())
    }

    /// Correct the world-time end of this window when later evidence says the
    /// claim stopped being true earlier than recorded. Returns whether the
    /// window actually moved.
    ///
    /// ## Why this only ever narrows
    ///
    /// The schema carried both bi-temporal legs from the start, but nothing in
    /// this crate ever wrote `valid_until` except a constructor — so a claim's
    /// world-time end was whatever the caller asserted at creation, and
    /// evidence arriving later could not correct it.
    ///
    /// The failure that motivates the restriction was measured in graphiti,
    /// which writes its equivalent leg once from the first contradiction it
    /// sees and then never narrows it
    /// (`computer-remake-research/experiments/graphiti/RESULTS.md`): an
    /// employment fact was left recorded as ending 2026-02-20 while the same
    /// store held a document saying it ended 2025-03-10, so a query for
    /// mid-2025 returned two employers at once.
    ///
    /// Narrowing is a *correction* — the claim was always false after `when`,
    /// and the record merely failed to say so. Widening is not: evidence that
    /// a claim lasted longer than recorded is a new assertion about the
    /// interval, and letting a setter make it silently would hide the one
    /// thing replay exists to show. So a `when` at or after the current
    /// `valid_until` changes nothing and reports `false`.
    ///
    /// A `when` before `valid_from` is refused for the same reason: that is a
    /// contradiction about the interval itself, not a narrowing of it, and
    /// quietly producing an empty window would bury it.
    pub fn narrow_until(
        &mut self,
        when: chrono::DateTime<chrono::Utc>,
        trigger: Option<String>,
    ) -> bool {
        if let Some(from) = self.valid_from {
            if when < from {
                return false;
            }
        }
        if let Some(until) = self.valid_until {
            if when >= until {
                return false;
            }
        }
        self.valid_until = Some(when);
        if trigger.is_some() {
            self.trigger = trigger;
        }
        true
    }

    /// Does this overlap with another validity window?
    pub fn overlaps(&self, other: &TemporalValidity) -> bool {
        let start_a = self
            .valid_from
            .unwrap_or_else(|| chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap());
        let end_a = self.valid_until.unwrap_or_else(|| {
            chrono::DateTime::<chrono::Utc>::from_timestamp(i64::MAX, 0).unwrap()
        });
        let start_b = other
            .valid_from
            .unwrap_or_else(|| chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap());
        let end_b = other.valid_until.unwrap_or_else(|| {
            chrono::DateTime::<chrono::Utc>::from_timestamp(i64::MAX, 0).unwrap()
        });
        start_a < end_b && start_b < end_a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permanent_is_always_valid() {
        let t = TemporalValidity::permanent();
        assert!(t.is_current());
    }

    #[test]
    fn interval_validity() {
        let start = chrono::Utc::now();
        let end = start + chrono::Duration::hours(1);
        let t = TemporalValidity::during(start, end);
        assert!(t.is_valid_at(start));
        assert!(t.is_valid_at(start + chrono::Duration::minutes(30)));
        assert!(!t.is_valid_at(end));
    }

    #[test]
    fn narrow_until_closes_an_open_window() {
        let start = chrono::Utc::now();
        let mut t = TemporalValidity::from(start);
        assert!(t.valid_until.is_none());
        assert!(t.narrow_until(start + chrono::Duration::hours(1), None));
        assert_eq!(t.valid_until, Some(start + chrono::Duration::hours(1)));
    }

    #[test]
    fn narrow_until_refuses_to_widen() {
        let start = chrono::Utc::now();
        let end = start + chrono::Duration::hours(1);
        let mut t = TemporalValidity::during(start, end);

        // At the boundary and beyond it: both are widenings, both refused.
        assert!(!t.narrow_until(end, None));
        assert!(!t.narrow_until(end + chrono::Duration::hours(5), None));
        assert_eq!(t.valid_until, Some(end), "window must not move");
    }

    #[test]
    fn narrow_until_refuses_an_instant_before_the_window_opens() {
        let start = chrono::Utc::now();
        let mut t = TemporalValidity::during(start, start + chrono::Duration::hours(1));
        assert!(!t.narrow_until(start - chrono::Duration::hours(1), None));
        assert_eq!(t.valid_until, Some(start + chrono::Duration::hours(1)));
    }

    #[test]
    fn narrow_until_records_its_trigger() {
        let start = chrono::Utc::now();
        let mut t = TemporalValidity::from(start);
        assert!(t.narrow_until(
            start + chrono::Duration::hours(1),
            Some("contradicted by E2".to_string())
        ));
        assert_eq!(t.trigger.as_deref(), Some("contradicted by E2"));
    }

    /// The graphiti case, by name.
    ///
    /// A window is first closed at T2 by the only contradiction then
    /// available; evidence arriving afterwards shows the claim actually ended
    /// at T1, earlier. graphiti leaves the window at T2 forever, so a query
    /// between T1 and T2 reports the claim as still true — measured there as
    /// one person holding two jobs at once.
    ///
    /// Fails before `narrow_until` exists: there is no way to move the
    /// boundary at all.
    #[test]
    fn a_window_closed_by_weak_evidence_is_corrected_by_better_evidence() {
        let t0 = chrono::Utc::now();
        let t1 = t0 + chrono::Duration::days(400); // when it really ended
        let t2 = t0 + chrono::Duration::days(800); // the first, worse guess

        let mut t = TemporalValidity::from(t0);
        assert!(t.narrow_until(t2, Some("first contradiction".into())));
        let between = t0 + chrono::Duration::days(600);
        assert!(
            t.is_valid_at(between),
            "with only the weak boundary, the claim still reads as true here"
        );

        // Better evidence arrives, referring to an earlier world time.
        assert!(t.narrow_until(t1, Some("earlier contradiction".into())));
        assert!(
            !t.is_valid_at(between),
            "after the correction the claim must be false between T1 and T2"
        );
        assert!(t.is_valid_at(t0 + chrono::Duration::days(100)));
    }
}
