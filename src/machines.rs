//! Structural machines — a stance toward a world, and what it saw.
//!
//! ## Why this is in the library
//!
//! These types were written inside
//! `examples/impossible_machine_experiment.rs`, where they worked and where
//! nothing else could reach them. By this crate's own rule — *declared ≠
//! called* — a symbol reachable only from one example is a symbol the library
//! does not have, however good it is.
//!
//! Three things here exist nowhere else in the engine:
//!
//! 1. **A numeric claim that carries its own certification level.**
//!    [`ProofStatus`] grades a finding from `Established` down to `Heuristic`,
//!    which is documented as compatibility only and *never* certification. A
//!    quantity without a grade is the shape most numeric pipelines ship and
//!    the reason their results cannot be audited afterwards.
//! 2. **A scaling exponent.** [`scaling_exponent`] is KAIROS's operation:
//!    the invariant is not the value, it is how the value *grows* across
//!    scales.
//! 3. **Refusal as a finding.** [`Observation::refusal`] is EUCLID's stance —
//!    "this layer contains no witness either way" is recorded, not left as
//!    silence. An empty result and a check that could not run must not look
//!    alike.
//!
//! ## What is deliberately NOT here
//!
//! The seven machines' bodies stay with their domain. EUCLID's
//! `verify_von_mangoldt` and KAIROS's `psi_recon` are Riemann arithmetic;
//! moving them here to make the trait look better-used would reproduce exactly
//! the problem this module exists to fix.
//!
//! There is also no world abstraction beyond the type parameter on
//! [`StructuralMachine`]. There is one world today. A second one — a price
//! series is the live candidate — is what would tell us what the abstraction
//! should be, and guessing before it exists means guessing wrong.

use serde::{Deserialize, Serialize};

/// Proof status: exactly one per proposition.
///
/// The ordering of the variants is the ordering of epistemic strength, and the
/// distinction that matters most is the bottom of it: `Heuristic` and
/// `Conjectural` are *not* weak evidence, they are **not evidence**. A pipeline
/// that lets them be summed, averaged or ranked alongside `Established` has
/// lost the only thing this enum is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofStatus {
    /// Classical theorem, cited with witness.
    Established,
    /// Derived from established propositions inside this run.
    Derived,
    /// Verified in this run on a finite exact instance — proves the instance,
    /// and only the instance.
    EstablishedFinite,
    /// Numerically observed, error-bounded.
    Numerical,
    /// Generated for compatibility only — **never certification**.
    Heuristic,
    /// Awaiting adjudication; no test was available here.
    Conjectural,
    /// Killed by computation or by the literature. Preserved, never deleted.
    Contradicted,
    /// The layer was asked and had no witness either way. Distinct from
    /// `Contradicted` (which is an answer) and from absence (which is silence).
    InsufficientData,
}

impl ProofStatus {
    /// May a finding with this status be used to certify anything?
    ///
    /// This exists so the question is answered in one place rather than by
    /// each caller remembering the rule. `Heuristic` and `Conjectural` are
    /// compatibility and candidacy respectively; neither certifies.
    /// `Contradicted` and `InsufficientData` are not certifications either,
    /// for the opposite reason.
    pub fn certifies(&self) -> bool {
        matches!(
            self,
            Self::Established | Self::Derived | Self::EstablishedFinite | Self::Numerical
        )
    }
}

/// What one structural operator saw in a world.
///
/// `quantity` is the number; `status` is what the number is worth. They travel
/// together on purpose — separating them is how a heuristic ends up in a
/// results table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Which machine produced this — provenance, so a finding can be traced to
    /// the stance that generated it.
    pub machine: String,
    /// What the observation is about.
    pub target: String,
    /// The finding, in words, including the reasoning that produced it.
    pub claim: String,
    /// The measured value. Meaningless without `status`.
    pub quantity: f64,
    /// What the value is worth.
    pub status: ProofStatus,
}

impl Observation {
    /// A measurement.
    pub fn new(
        machine: impl Into<String>,
        target: impl Into<String>,
        claim: impl Into<String>,
        quantity: f64,
        status: ProofStatus,
    ) -> Self {
        Self {
            machine: machine.into(),
            target: target.into(),
            claim: claim.into(),
            quantity,
            status,
        }
    }

    /// A refusal: the layer was asked and has no witness either way.
    ///
    /// EUCLID's stance, and the reason it is a constructor rather than a
    /// convention: recording "I looked and could not see" has to be as cheap
    /// as recording a number, or it will not be recorded, and the absence will
    /// later be read as agreement. `quantity` is 0.0 because there is no
    /// measurement — `status` is what carries the meaning.
    pub fn refusal(
        machine: impl Into<String>,
        target: impl Into<String>,
        claim: impl Into<String>,
    ) -> Self {
        Self::new(machine, target, claim, 0.0, ProofStatus::InsufficientData)
    }

    /// Does this observation carry weight?
    pub fn certifies(&self) -> bool {
        self.status.certifies()
    }
}

/// One structural operator: a stance toward a world `W`.
///
/// Generic over the world because the stance is the portable part. A machine
/// that fits exponents does not care whether the series came from a prime
/// counting function or anywhere else; a machine that refuses to guess does
/// not care what it is refusing to guess about.
pub trait StructuralMachine<W> {
    /// Stable name, used as `Observation::machine`.
    fn name(&self) -> &'static str;
    /// Inspect the world and report what this stance can see.
    fn inspect(&self, world: &W) -> Vec<Observation>;
}

/// The empirical growth exponent of `y` against `x`, both already in log space.
///
/// KAIROS's operation: the invariant is the exponent, not the value — the
/// thing that survives a change of scale. `λ = d log y / d log x`, by least
/// squares.
///
/// # The top half, and why
///
/// Only the upper half of the range is fitted. In the original use a
/// truncation floor dominates at small `x`, so including the low end measures
/// the artefact rather than the growth. Any series with a resolution floor,
/// a burn-in, or a minimum sample size has the same problem, so the rule is
/// kept here rather than left for each caller to rediscover.
///
/// Points must be supplied sorted by `x`. Returns `None` when fewer than two
/// points survive the halving, because a slope through one point is not a
/// slope — reporting 0.0 there would be a measurement that never happened.
pub fn scaling_exponent(log_points: &[(f64, f64)]) -> Option<f64> {
    if log_points.len() < 4 {
        // Fewer than four points cannot be halved into a fittable upper range.
        // Fit the whole set instead of refusing outright, but still require two.
        return least_squares_slope(log_points);
    }
    least_squares_slope(&log_points[log_points.len() / 2..])
}

/// Least-squares slope of `y` on `x`. `None` when there are fewer than two
/// points, or when every `x` is identical (a vertical fit has no slope).
pub fn least_squares_slope(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let denom = n * sxx - sx * sx;
    // Guarding rather than clamping: the original used `.max(1e-12)`, which
    // turns a degenerate fit into an enormous finite slope instead of saying
    // the fit was degenerate.
    if denom.abs() < f64::EPSILON {
        return None;
    }
    Some((n * sxy - sx * sy) / denom)
}

/// One graded proposition from an impossible-machine run.
///
/// Mirrors the `Proposition` record in
/// `examples/impossible_machine_experiment.rs` (same field names, same
/// [`ProofStatus`] grades) so a run artifact deserializes here directly.
/// Claim and grade travel together — a proposition without a grade is the
/// shape most numeric pipelines ship and the reason their results cannot
/// be audited afterwards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropositionView {
    pub id: String,
    #[serde(default)]
    pub machines: Vec<String>,
    pub claim: String,
    pub status: ProofStatus,
    #[serde(default)]
    pub witness: String,
    #[serde(default)]
    pub quarantined: bool,
}

impl PropositionView {
    /// Does this proposition carry weight? Same rule as [`ProofStatus`]:
    /// `Heuristic` and `Conjectural` are compatibility and candidacy,
    /// never certification.
    pub fn certifies(&self) -> bool {
        self.status.certifies()
    }
}

/// One structural operator's digest from a run artifact: what stance it
/// took, what it saw (graded), and how it can fail.
///
/// Hypotheses and dreams are intentionally not carried: this run's
/// machine hypothesis lists are empty and dreams live under their own
/// top-level key. A view that silently re-grades them would be the
/// duplication trap, not a reader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineDigest {
    pub name: String,
    #[serde(default)]
    pub operation: String,
    #[serde(default)]
    pub failure_mode: String,
    #[serde(default)]
    pub physis_equivalent: String,
    #[serde(default)]
    pub observations: Vec<Observation>,
}

/// A recorded disagreement between two machines. Their disagreement is
/// data (mission first principle), so it is carried, never merged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisagreementView {
    pub a: String,
    pub b: String,
    pub subject: String,
    pub a_claim: String,
    pub b_claim: String,
}

/// The display subset of a `results.json` run artifact: graded machines,
/// propositions, disagreements, and the verdict. Field renames match the
/// artifact keys (`machine_reports`, not `machines`); everything has a
/// default so a partial artifact still loads what it has.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImpossibleReport {
    #[serde(default, rename = "machine_reports")]
    pub machines: Vec<MachineDigest>,
    #[serde(default)]
    pub propositions: Vec<PropositionView>,
    #[serde(default)]
    pub disagreements: Vec<DisagreementView>,
    #[serde(default)]
    pub verdict: String,
    #[serde(default)]
    pub verdict_witness: String,
}

impl ImpossibleReport {
    /// Load a run artifact. `Err` is a plain string: a missing or broken
    /// artifact is a display gap, never a reason to invent numbers.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Graded observations across all machines (quantity + status together).
    pub fn observations(&self) -> impl Iterator<Item = &Observation> {
        self.machines.iter().flat_map(|m| m.observations.iter())
    }

    /// Observations that carry weight (see [`ProofStatus::certifies`]).
    pub fn certified_observations(&self) -> impl Iterator<Item = &Observation> {
        self.observations().filter(|o| o.certifies())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points on `y = a + λx`, in the shape `scaling_exponent` expects.
    fn line(intercept: f64, slope: f64, n: usize) -> Vec<(f64, f64)> {
        (1..=n)
            .map(|i| {
                let x = i as f64;
                (x, intercept + slope * x)
            })
            .collect()
    }

    #[test]
    fn recovers_a_known_exponent() {
        let half = scaling_exponent(&line(3.0, 0.5, 40)).unwrap();
        assert!((half - 0.5).abs() < 1e-9, "got {half}");
    }

    /// A second, different slope — so the test cannot pass by returning a
    /// constant that happens to be 0.5.
    #[test]
    fn recovers_a_different_exponent() {
        let steep = scaling_exponent(&line(-1.0, 0.8, 40)).unwrap();
        assert!((steep - 0.8).abs() < 1e-9, "got {steep}");
    }

    /// The upper half is what is fitted, so a distorted low end must not move
    /// the answer. This is the truncation-floor rule doing its job.
    #[test]
    fn a_distorted_low_end_does_not_move_the_exponent() {
        let mut pts = line(0.0, 0.5, 40);
        for p in pts.iter_mut().take(20) {
            p.1 += 5.0; // a floor that only affects small x
        }
        let lambda = scaling_exponent(&pts).unwrap();
        assert!(
            (lambda - 0.5).abs() < 1e-9,
            "the low-end floor leaked into the fit: {lambda}"
        );
    }

    #[test]
    fn too_few_points_is_none_not_zero() {
        assert_eq!(scaling_exponent(&[]), None);
        assert_eq!(scaling_exponent(&[(1.0, 1.0)]), None);
        assert_eq!(least_squares_slope(&[(1.0, 1.0)]), None);
    }

    /// A vertical fit has no slope. The original clamped the denominator and
    /// returned an enormous finite number, which reads downstream as a real
    /// measurement.
    #[test]
    fn a_degenerate_fit_is_none_not_a_huge_number() {
        assert_eq!(least_squares_slope(&[(2.0, 1.0), (2.0, 5.0)]), None);
    }

    #[test]
    fn a_refusal_keeps_its_status_and_certifies_nothing() {
        let r = Observation::refusal("euclid", "offline-visibility", "no witness either way");
        assert_eq!(r.status, ProofStatus::InsufficientData);
        assert!(!r.certifies());
        let round: Observation = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(round.status, ProofStatus::InsufficientData);
        assert!(
            !round.certifies(),
            "a refusal must not survive a round-trip as a measurement"
        );
    }

    #[test]
    fn heuristic_never_certifies() {
        assert!(!ProofStatus::Heuristic.certifies());
        assert!(!ProofStatus::Conjectural.certifies());
        assert!(!ProofStatus::Contradicted.certifies());
        assert!(!ProofStatus::InsufficientData.certifies());
        assert!(ProofStatus::Established.certifies());
        assert!(ProofStatus::EstablishedFinite.certifies());
        assert!(ProofStatus::Numerical.certifies());
    }

    /// The trait is usable against a world it was not written for — which is
    /// the whole reason it is generic.
    #[test]
    fn a_machine_can_inspect_an_arbitrary_world() {
        struct Series(Vec<(f64, f64)>);
        struct Kairos;
        impl StructuralMachine<Series> for Kairos {
            fn name(&self) -> &'static str {
                "kairos"
            }
            fn inspect(&self, w: &Series) -> Vec<Observation> {
                match scaling_exponent(&w.0) {
                    Some(l) => vec![Observation::new(
                        self.name(),
                        "growth-exponent",
                        format!("lambda = {l:.3}"),
                        l,
                        ProofStatus::Numerical,
                    )],
                    None => vec![Observation::refusal(
                        self.name(),
                        "growth-exponent",
                        "fewer than two usable points",
                    )],
                }
            }
        }

        let seen = Kairos.inspect(&Series(line(0.0, 0.5, 40)));
        assert_eq!(seen.len(), 1);
        assert!((seen[0].quantity - 0.5).abs() < 1e-9);
        assert!(seen[0].certifies());

        let refused = Kairos.inspect(&Series(vec![]));
        assert_eq!(refused[0].status, ProofStatus::InsufficientData);
    }

    /// A run artifact deserializes into the report views with grades intact.
    #[test]
    fn a_report_loads_graded_machines_and_propositions() {
        let doc = serde_json::json!({
            "machine_reports": [{
                "name": "euclid",
                "operation": "sieve",
                "failure_mode": "RH invisible to construction",
                "physis_equivalent": "EXACT layer",
                "observations": [
                    {"machine": "euclid", "target": "t", "claim": "c",
                     "quantity": 0.0, "status": "EstablishedFinite"},
                    {"machine": "euclid", "target": "u", "claim": "d",
                     "quantity": 0.0, "status": "Heuristic"},
                ],
            }],
            "propositions": [
                {"id": "P1", "machines": ["euclid"], "claim": "c",
                 "status": "Contradicted", "witness": "w", "quarantined": false},
            ],
            "disagreements": [
                {"a": "euclid", "b": "nous", "subject": "s",
                 "a_claim": "invisible", "b_claim": "visible"},
            ],
            "verdict": "REDUCED",
            "verdict_witness": "seven stances",
        });
        let rep: ImpossibleReport = serde_json::from_value(doc).unwrap();
        assert_eq!(rep.machines.len(), 1);
        assert_eq!(rep.observations().count(), 2);
        assert_eq!(rep.certified_observations().count(), 1);
        assert!(!rep.propositions[0].certifies());
        assert_eq!(rep.disagreements.len(), 1);
        assert_eq!(rep.verdict, "REDUCED");
    }

    /// A partial artifact loads what it has; a missing file reports its path.
    #[test]
    fn a_partial_report_loads_and_a_missing_file_names_itself() {
        let rep: ImpossibleReport = serde_json::from_str(r#"{"verdict": "V"}"#).unwrap();
        assert!(rep.machines.is_empty());
        assert_eq!(rep.verdict, "V");
        let err = ImpossibleReport::load(std::path::Path::new("/no/such/results.json"))
            .unwrap_err();
        assert!(err.contains("/no/such/results.json"), "{err}");
    }

    /// An ungraded status is rejected, not guessed: a typo in a grade must
    /// fail the load rather than certify under a wrong name.
    #[test]
    fn an_unknown_grade_fails_the_load() {
        let bad = r#"{"propositions": [{"id": "P", "claim": "c", "status": "Proven"}]}"#;
        assert!(serde_json::from_str::<ImpossibleReport>(bad).is_err());
    }
}
