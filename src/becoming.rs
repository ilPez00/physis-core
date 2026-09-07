//! Becoming — did a term's meaning *change*, or does one name cover two things?
//!
//! Every other similarity method in this engine is order-blind. Clustering a
//! term's occurrences reports "two senses" whether or not one turned into the
//! other, because it never looks at *when* each occurrence happened. That is
//! precisely the information which separates the two cases:
//!
//! ```text
//!   AAAAAABBBBBB   two senses, time-SEPARATED   -> one thing changed (becoming)
//!   ABABBABAABAB   two senses, INTERLEAVED      -> two things sharing a name (split)
//! ```
//!
//! Both partitions are identical to a clusterer. Only the sequence tells them
//! apart, and the statistic for "are these two labels time-separated or
//! interleaved" is the Wald-Wolfowitz **runs test**, which has a null
//! distribution — so the verdict carries a z-score rather than a threshold
//! someone picked.
//!
//! ## Mistake vs variation
//!
//! A single occurrence that sits far from the term's usual context is either
//! noise or the start of something. The distinction is not similarity but
//! **uptake**: a deviation that never recurs is a mistake; one that recurs is a
//! variation the text went on to build upon. The judge is the document's own
//! later behaviour, not a reference ontology.
//!
//! ## What this is not
//!
//! It does not decide whether a sense is *correct*. It reports the shape of a
//! trajectory. A term can drift into a wrong meaning and this will faithfully
//! call it a becoming.
//!
//! ```no_run
//! use physis_core::becoming::{Occurrence, classify};
//! # fn demo(occs: Vec<Occurrence>) {
//! let verdict = classify(&occs);
//! println!("{:?} (runs z = {:.2})", verdict.trajectory, verdict.runs_z);
//! # }
//! ```

use crate::models::cosine_sim;

/// One occurrence of a term: when it happened, and the context it sat in.
///
/// `at` only has to be monotonically ordered — a timestamp, a byte offset, or a
/// sequence number all work. `context` is expected to be L2-normalized, which is
/// [`crate::embed::VectorEmbed`]'s documented invariant.
#[derive(Debug, Clone)]
pub struct Occurrence {
    pub at: i64,
    pub context: Vec<f32>,
}

/// The shape of a term's trajectory through a corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trajectory {
    /// One sense. No separation worth splitting.
    Stable,
    /// Two senses, time-separated: the term meant one thing, then another.
    Becoming,
    /// Two senses, interleaved: one name over two things, both live throughout.
    Split,
    /// Fewer occurrences than any of this can be said about.
    TooFew,
}

/// Why a deviating occurrence deviated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deviation {
    /// Deviates and never recurs — noise.
    Mistake,
    /// Deviates and recurs later — the text took it up.
    Variation,
}

/// A trajectory verdict with the evidence behind it.
#[derive(Debug, Clone)]
pub struct Verdict {
    pub trajectory: Trajectory,
    /// Separation between the two candidate senses: 1 - cosine between their
    /// centroids. Near 0 means there was never really a second sense.
    pub separation: f32,
    /// Runs-test z. Strongly negative = time-separated (becoming); near 0 or
    /// positive = interleaved (split).
    pub runs_z: f64,
    /// Index in the supplied order where the change sits, when `Becoming`.
    pub change_at: Option<usize>,
    /// Per-occurrence deviation classification, parallel to the input.
    pub deviations: Vec<Option<Deviation>>,
}

/// Below this centroid separation the two "senses" are one sense with noise.
const MIN_SEPARATION: f32 = 0.15;
/// Runs z at or below this counts as time-separated rather than interleaved.
/// -1.96 is the usual two-sided 5% point; the test is one-sided here because
/// only *too few* runs indicates separation.
const RUNS_Z_SEPARATED: f64 = -1.96;

fn centroid(v: &[&Vec<f32>]) -> Vec<f32> {
    let dim = v.first().map(|x| x.len()).unwrap_or(0);
    let mut acc = vec![0.0f32; dim];
    for e in v {
        for d in 0..dim {
            acc[d] += e[d];
        }
    }
    let n: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    acc.iter().map(|x| x / n).collect()
}

/// Deterministic 2-means: seeded from the farthest-apart pair (ties broken on
/// index), so the same input always yields the same partition. No RNG, because
/// a verdict that changes between runs is not a verdict.
fn two_means(occs: &[Occurrence]) -> Vec<bool> {
    let n = occs.len();
    let (mut bi, mut bj, mut worst) = (0usize, 1usize, f32::INFINITY);
    for i in 0..n {
        for j in (i + 1)..n {
            let s = cosine_sim(&occs[i].context, &occs[j].context);
            if s < worst {
                worst = s;
                bi = i;
                bj = j;
            }
        }
    }
    let (mut ca, mut cb) = (occs[bi].context.clone(), occs[bj].context.clone());
    let mut assign = vec![false; n];
    for _ in 0..20 {
        let mut changed = false;
        for (k, o) in occs.iter().enumerate() {
            let b = cosine_sim(&o.context, &cb) > cosine_sim(&o.context, &ca);
            if assign[k] != b {
                assign[k] = b;
                changed = true;
            }
        }
        let a: Vec<&Vec<f32>> = occs.iter().zip(&assign).filter(|(_, &b)| !b).map(|(o, _)| &o.context).collect();
        let b: Vec<&Vec<f32>> = occs.iter().zip(&assign).filter(|(_, &b)| b).map(|(o, _)| &o.context).collect();
        if a.is_empty() || b.is_empty() {
            break;
        }
        ca = centroid(&a);
        cb = centroid(&b);
        if !changed {
            break;
        }
    }
    assign
}

/// Wald-Wolfowitz runs test z for a binary sequence. Negative z means fewer
/// runs than chance — the labels clump in time — which is what a genuine
/// change looks like. Positive means they alternate more than chance.
pub fn runs_z(seq: &[bool]) -> f64 {
    let n = seq.len() as f64;
    let n1 = seq.iter().filter(|b| **b).count() as f64;
    let n2 = n - n1;
    if n1 == 0.0 || n2 == 0.0 {
        return 0.0;
    }
    let runs = 1 + seq.windows(2).filter(|w| w[0] != w[1]).count();
    let mu = 2.0 * n1 * n2 / n + 1.0;
    let var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n * n * (n - 1.0));
    if var <= 0.0 {
        return 0.0;
    }
    (runs as f64 - mu) / var.sqrt()
}

/// Classify a term's trajectory. `occs` must be sorted by `at`; it is not
/// re-sorted here, because the caller's ordering is the evidence.
pub fn classify(occs: &[Occurrence]) -> Verdict {
    if occs.len() < 6 {
        return Verdict {
            trajectory: Trajectory::TooFew,
            separation: 0.0,
            runs_z: 0.0,
            change_at: None,
            deviations: vec![None; occs.len()],
        };
    }
    let assign = two_means(occs);
    let a: Vec<&Vec<f32>> = occs.iter().zip(&assign).filter(|(_, &b)| !b).map(|(o, _)| &o.context).collect();
    let b: Vec<&Vec<f32>> = occs.iter().zip(&assign).filter(|(_, &b)| b).map(|(o, _)| &o.context).collect();
    if a.is_empty() || b.is_empty() {
        return Verdict {
            trajectory: Trajectory::Stable,
            separation: 0.0,
            runs_z: 0.0,
            change_at: None,
            deviations: deviations(occs),
        };
    }
    let separation = 1.0 - cosine_sim(&centroid(&a), &centroid(&b));
    let z = runs_z(&assign);

    let trajectory = if separation < MIN_SEPARATION {
        Trajectory::Stable
    } else if z <= RUNS_Z_SEPARATED {
        Trajectory::Becoming
    } else {
        Trajectory::Split
    };
    // The change point is the last index of the first run, which is only
    // meaningful when the labels are actually time-separated.
    let change_at = if trajectory == Trajectory::Becoming {
        assign.windows(2).position(|w| w[0] != w[1])
    } else {
        None
    };

    Verdict { trajectory, separation, runs_z: z, change_at, deviations: deviations(occs) }
}

/// Classify each occurrence as a mistake, a variation, or neither.
///
/// An occurrence deviates when it sits below the mean similarity-to-centroid by
/// more than one standard deviation. It is a *variation* if some LATER
/// occurrence resembles it more than it resembles the centroid — the text took
/// it up — and a *mistake* otherwise.
pub fn deviations(occs: &[Occurrence]) -> Vec<Option<Deviation>> {
    let n = occs.len();
    if n < 3 {
        return vec![None; n];
    }
    let all: Vec<&Vec<f32>> = occs.iter().map(|o| &o.context).collect();
    let ctr = centroid(&all);
    let sims: Vec<f32> = occs.iter().map(|o| cosine_sim(&o.context, &ctr)).collect();
    let mean = sims.iter().sum::<f32>() / n as f32;
    let sd = (sims.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n as f32).sqrt();
    let bar = mean - sd;

    (0..n)
        .map(|i| {
            if sims[i] >= bar {
                return None;
            }
            // Uptake: does anything LATER look more like this than like the norm?
            let taken_up = ((i + 1)..n).any(|j| cosine_sim(&occs[i].context, &occs[j].context) > sims[j]);
            Some(if taken_up { Deviation::Variation } else { Deviation::Mistake })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v.iter().map(|x| x / n).collect()
    }
    fn occ(at: i64, v: Vec<f32>) -> Occurrence {
        Occurrence { at, context: norm(v) }
    }
    /// Two well-separated senses, laid out in the given order.
    fn build(order: &[u8]) -> Vec<Occurrence> {
        order
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let v = if c == 0 { vec![1.0, 0.0, 0.05] } else { vec![0.0, 1.0, 0.05] };
                occ(i as i64, v)
            })
            .collect()
    }

    #[test]
    fn time_separated_senses_are_a_becoming() {
        let v = classify(&build(&[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]));
        assert_eq!(v.trajectory, Trajectory::Becoming);
        assert!(v.runs_z < RUNS_Z_SEPARATED, "runs z was {}", v.runs_z);
        assert_eq!(v.change_at, Some(5));
    }

    #[test]
    fn interleaved_senses_are_a_split() {
        // Same two senses, same counts — only the ORDER differs. A clusterer
        // cannot tell this from the case above; that is the whole point.
        let v = classify(&build(&[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]));
        assert_eq!(v.trajectory, Trajectory::Split);
        assert!(v.change_at.is_none());
    }

    #[test]
    fn one_sense_is_stable_however_it_is_ordered() {
        let occs: Vec<Occurrence> = (0..12)
            .map(|i| occ(i, vec![1.0, 0.02 * (i % 3) as f32, 0.01]))
            .collect();
        assert_eq!(classify(&occs).trajectory, Trajectory::Stable);
    }

    #[test]
    fn too_few_occurrences_says_so_rather_than_guessing() {
        assert_eq!(classify(&build(&[0, 0, 1])).trajectory, Trajectory::TooFew);
    }

    #[test]
    fn runs_z_is_negative_when_clumped_and_positive_when_alternating() {
        let clumped = [false, false, false, false, true, true, true, true];
        let alternating = [false, true, false, true, false, true, false, true];
        assert!(runs_z(&clumped) < -1.5, "clumped z = {}", runs_z(&clumped));
        assert!(runs_z(&alternating) > 1.5, "alternating z = {}", runs_z(&alternating));
    }

    #[test]
    fn a_lone_outlier_is_a_mistake_but_a_repeated_one_is_a_variation() {
        // Baseline sense, one odd occurrence that never recurs.
        let mut v: Vec<Occurrence> = (0..9).map(|i| occ(i, vec![1.0, 0.01, 0.0])).collect();
        v[4] = occ(4, vec![0.0, 0.0, 1.0]);
        let d = deviations(&v);
        assert_eq!(d[4], Some(Deviation::Mistake), "unrepeated outlier must be a mistake");

        // The same oddity, taken up again later.
        let mut w: Vec<Occurrence> = (0..9).map(|i| occ(i, vec![1.0, 0.01, 0.0])).collect();
        w[4] = occ(4, vec![0.0, 0.0, 1.0]);
        w[7] = occ(7, vec![0.0, 0.0, 1.0]);
        let e = deviations(&w);
        assert_eq!(e[4], Some(Deviation::Variation), "a recurring deviation is owned");
    }

    #[test]
    fn verdicts_are_reproducible() {
        // No RNG anywhere: the same input must give the same verdict every time.
        let occs = build(&[0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]);
        let first = classify(&occs);
        for _ in 0..8 {
            let again = classify(&occs);
            assert_eq!(first.trajectory, again.trajectory);
            assert_eq!(first.change_at, again.change_at);
            assert!((first.runs_z - again.runs_z).abs() < 1e-12);
        }
    }
}
