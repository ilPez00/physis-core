//! D4 — does a cell of the grid carve anything, or is it just full?
//!
//! ## Why this comes before the rest of D
//!
//! Conceptual problem 3 is that the ontology is the one belief that cannot be
//! contradicted. The design is to make cells **proposals** with a fitness, so
//! evidence can revise them. The immediate hazard is that a fitness which only
//! ever rises with population would licence exactly the wrong conclusion:
//! *this cell is well-supported* meaning nothing but *this cell is large*.
//!
//! So the null comes first and gates the rest. **Permute the labels.** Reassign
//! every entry to a cell at random, keeping every cell's size exactly, and
//! recompute the same fitness. Real structure survives that; volume does not.
//! A cell whose real fitness sits inside its permuted distribution is a cell
//! the evidence does not support, whatever its population.
//!
//! ## The fitness
//!
//! Leave-one-out self-recovery. For each entry, score it against every cell
//! *with that entry removed from its own cell*, and ask whether its own cell
//! still ranks first. A cell's fitness is the fraction of its entries that come
//! home.
//!
//! The leave-one-out is not a detail. Scoring an entry against a cell that
//! contains it returns cosine 1.0 and every cell scores a perfect 1.000 — the
//! first version of this module did exactly that and reported a flawless grid.
//! A fitness that cannot fail is the thing this file exists to prevent.
//!
//! ## Reading the result
//!
//! | | |
//! |---|---|
//! | real ≫ permuted | the cell carves something the labels alone do not |
//! | real ≈ permuted | population, not fit — a merge or retirement candidate |
//! | real < permuted | the label is actively misleading |
//!
//! `z` is how many permuted standard deviations the real fitness sits above the
//! permuted mean. It is reported per cell and not only in aggregate, because
//! the grid's problem is known to be *specific cells* — `FABRICATE` ↔
//! `CONSTRUCT` confusing at 0.22/0.21 (E22), `WALK` scoring top-1 0.18 on 27
//! entries — and an average over 70 cells would hide every one of them.

//! ## What it measured, 2026-09-13
//!
//! 70 cells, 731 entries, 200 permutations, bge-base-en-v1.5:
//!
//! ```text
//! overall self-recovery 0.438   permuted 0.038   (11.5x the null)
//!   49 of 70 cells clear their own null
//!    8 fail it with two or more entries
//!   13 have fewer than two entries: unmeasurable
//! ```
//!
//! **The grid is not an arbitrary carve.** Overall self-recovery is 11.5x the
//! label-permuted null, which is the first time this has been checked against
//! anything. Whatever else is wrong with the 5x14, the labels are carrying real
//! structure and a reader should stop assuming otherwise.
//!
//! The eight that fail with entries in them are the result:
//!
//! | cell | n | fitness | null | z |
//! |---|---|---|---|---|
//! | `BOND/MAINTAIN` | 15 | 0.067 | 0.015 | +1.40 |
//! | `STUDY/PLAN` | 5 | 0.000 | 0.008 | −0.14 |
//! | `STUDY/PLAY` | 4 | 0.000 | 0.006 | −0.13 |
//! | `BOND/PLAY`, `BOND/SENSE`, `FABRICATE/PLAY`, `HEAL/DESTROY`, `STUDY/WALK` | 3 | 0.000 | ~0.002 | ~−0.08 |
//!
//! **`BOND/MAINTAIN` is the finding.** Fifteen entries — a well-populated cell
//! by this grid's standards — recovering 6.7% of them. It is not starved and it
//! does not carve. Nothing in the existing research named it; E22 named
//! `FABRICATE`↔`CONSTRUCT` and `WALK`, and this is a third case that only a
//! null could surface, because 0.067 looks like a small number rather than a
//! verdict until you know the null is 0.015 and the bar is two standard
//! deviations away.
//!
//! The thirteen singletons are **unmeasurable, not failing** — leave-one-out
//! empties a one-entry cell, so 0.000 there is arithmetic. Counting them as
//! failures would be E7's starved-class error one level up, and the report
//! separates them for that reason.
//!
//! ### What this does not license
//!
//! Self-recovery over ontology *entries* is not classification accuracy over
//! documents. It says a cell's own entries are mutually nearest; it does not
//! say a document lands there. And the entries are the grid's own seed text, so
//! this measures the grid's internal coherence and not its fit to anything
//! outside it — which is exactly the leg D1's alignment to a top-down ontology
//! is for.
//!
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// How many label permutations the null is averaged over.
pub const PERMUTATIONS: usize = 200;

/// One cell, its fitness, and the null it has to beat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellFitness {
    /// `DOMAIN/MODE`.
    pub cell: String,
    pub entries: usize,
    /// Leave-one-out self-recovery, 0.0–1.0.
    pub fitness: f32,
    /// Mean of the same statistic under label permutation.
    pub null_mean: f32,
    pub null_sd: f32,
    /// `(fitness − null_mean) / null_sd`. `None` when the null has no spread.
    pub z: Option<f32>,
    /// `true` when the real fitness clears the permuted mean by two permuted
    /// standard deviations. Not a p-value and not called one.
    pub above_null: bool,
    /// A cell with one entry scores 0.000 **by construction** — leave-one-out
    /// empties it, so nothing can come home. It has not failed; it has not been
    /// measured. Counting it as a failure would be the same error E7 names for
    /// starved classes, one level up.
    pub unmeasurable: bool,
}

/// The whole grid, and how much of it survives its own null.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridFitness {
    pub cells: usize,
    pub entries: usize,
    pub permutations: usize,
    pub seed: u64,
    pub embedder: String,
    /// Self-recovery over every entry, ignoring cell boundaries.
    pub overall_fitness: f32,
    pub overall_null_mean: f32,
    pub cells_above_null: usize,
    /// Cells with fewer than two entries: unmeasurable, not failing.
    pub cells_unmeasurable: usize,
    /// Cells with two or more entries that still do not clear their null.
    /// These are the result.
    pub cells_failing: usize,
    pub per_cell: Vec<CellFitness>,
}

impl GridFitness {
    pub fn render(&self) -> String {
        let mut o = String::new();
        o.push_str(&format!(
            "── GRID FITNESS vs LABEL-PERMUTED NULL ──\n{} cells · {} entries · {} permutations · embedder {}\n\n",
            self.cells, self.entries, self.permutations, self.embedder
        ));
        o.push_str(&format!(
            "  overall self-recovery {:.3}   permuted {:.3}   ({:.1}x the null)\n\n  {} of {} cells clear their own null\n  {} fail it with two or more entries  <-- the result\n  {} have fewer than two entries: UNMEASURABLE, not failing — leave-one-out\n     empties a singleton cell, so 0.000 is arithmetic and not evidence\n\n",
            self.overall_fitness,
            self.overall_null_mean,
            if self.overall_null_mean > 0.0 { self.overall_fitness / self.overall_null_mean } else { f32::NAN },
            self.cells_above_null,
            self.cells,
            self.cells_failing,
            self.cells_unmeasurable
        ));
        if self.embedder == "random-projection" {
            o.push_str("  NOTE: random-projection is a lexical hash. This is a floor.\n\n");
        }
        let mut worst: Vec<&CellFitness> = self
            .per_cell
            .iter()
            .filter(|c| !c.above_null && !c.unmeasurable)
            .collect();
        worst.sort_by_key(|c| std::cmp::Reverse(c.entries));
        o.push_str("  populated cells that do NOT clear their null, largest first —\n  these have entries and carve nothing with them:\n");
        o.push_str("  cell                            n   fitness    null      z\n");
        for c in worst.iter().take(20) {
            o.push_str(&format!(
                "  {:<28} {:>4}    {:>5.3}   {:>5.3}  {}\n",
                c.cell,
                c.entries,
                c.fitness,
                c.null_mean,
                match c.z {
                    Some(z) => format!("{z:>+6.2}"),
                    None => "     ·".to_string(),
                }
            ));
        }
        if worst.is_empty() {
            o.push_str("  (none — every measurable cell clears its null)\n");
        }
        let mut singles: Vec<&CellFitness> =
            self.per_cell.iter().filter(|c| c.unmeasurable).collect();
        singles.sort_by(|a, b| a.cell.cmp(&b.cell));
        if !singles.is_empty() {
            o.push_str(&format!(
                "\n  unmeasurable ({}): {}\n",
                singles.len(),
                singles
                    .iter()
                    .map(|c| c.cell.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        o
    }
}

/// Score the grid against its label-permuted null.
///
/// `entries` is `(cell, embedding)` per ontology entry. Nothing here knows what
/// a cell means, which is the point: the same function scores the real labels
/// and the shuffled ones.
pub fn run(entries: &[(String, Vec<f32>)], seed: u64, embedder_kind: &str) -> GridFitness {
    let labels: Vec<&str> = entries.iter().map(|(c, _)| c.as_str()).collect();

    // The pairwise similarity matrix does not depend on the labels, so it is
    // computed once and every permutation reads it. The first version
    // recomputed it inside the null and turned a two-second job into an
    // overnight one: 200 permutations times a thousand-entry O(n^2) cosine over
    // 768 dimensions. The null is supposed to be cheap enough that nobody is
    // tempted to skip it.
    let sim = similarity_matrix(entries);

    let real = self_recovery(&labels, &sim);

    // The null. Sizes are preserved exactly — a permutation, not a resample —
    // so any difference is about which entries sit together and not about how
    // many do.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut per_cell_null: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut overall_null: Vec<f32> = Vec::new();
    let mut shuffled: Vec<&str> = labels.clone();
    for _ in 0..PERMUTATIONS {
        shuffled.shuffle(&mut rng);
        let r = self_recovery(&shuffled, &sim);
        overall_null.push(r.overall);
        for (cell, v) in r.per_cell {
            per_cell_null.entry(cell).or_default().push(v);
        }
    }

    let mut per_cell = Vec::new();
    let mut cells_above_null = 0usize;
    for (cell, fitness) in &real.per_cell {
        let nulls = per_cell_null.get(cell).cloned().unwrap_or_default();
        let (null_mean, null_sd) = mean_sd(&nulls);
        let z = if null_sd > 0.0 {
            Some((fitness - null_mean) / null_sd)
        } else {
            None
        };
        let above_null = match z {
            Some(z) => z >= 2.0,
            // No spread in the null: the real value clears it only by being
            // strictly greater. Calling a tie "above" would be the same
            // mistake as calling a Δ of zero a result.
            None => *fitness > null_mean,
        };
        if above_null {
            cells_above_null += 1;
        }
        let n = real.sizes.get(cell).copied().unwrap_or(0);
        per_cell.push(CellFitness {
            cell: cell.clone(),
            entries: n,
            fitness: *fitness,
            null_mean,
            null_sd,
            z,
            above_null,
            unmeasurable: n < 2,
        });
    }
    per_cell.sort_by(|a, b| a.cell.cmp(&b.cell));

    let cells_unmeasurable = per_cell.iter().filter(|c| c.unmeasurable).count();
    let cells_failing = per_cell
        .iter()
        .filter(|c| !c.unmeasurable && !c.above_null)
        .count();

    let (overall_null_mean, _) = mean_sd(&overall_null);
    GridFitness {
        cells_unmeasurable,
        cells_failing,
        cells: real.per_cell.len(),
        entries: entries.len(),
        permutations: PERMUTATIONS,
        seed,
        embedder: embedder_kind.to_string(),
        overall_fitness: real.overall,
        overall_null_mean,
        cells_above_null,
        per_cell,
    }
}

struct Recovery {
    overall: f32,
    per_cell: BTreeMap<String, f32>,
    sizes: BTreeMap<String, usize>,
}

/// Cosine between every pair, once. Symmetric, so only the upper triangle is
/// computed and it is mirrored.
fn similarity_matrix(entries: &[(String, Vec<f32>)]) -> Vec<Vec<f32>> {
    let n = entries.len();
    let mut m = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s = crate::models::cosine_sim(&entries[i].1, &entries[j].1);
            m[i][j] = s;
            m[j][i] = s;
        }
        m[i][i] = f32::MIN; // never its own neighbour
    }
    m
}

/// Leave-one-out: does each entry's own cell still rank first without it?
fn self_recovery(labels: &[&str], sim: &[Vec<f32>]) -> Recovery {
    let mut sizes: BTreeMap<String, usize> = BTreeMap::new();
    for l in labels {
        *sizes.entry((*l).to_string()).or_insert(0) += 1;
    }
    let mut hits: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_hits = 0usize;

    for i in 0..labels.len() {
        // Best cosine to any *other* entry, per cell. `Fixed(1)` — max, not
        // mean — because that is what `SemioticClassifier::classify` ships.
        let mut best: BTreeMap<&str, f32> = BTreeMap::new();
        for j in 0..labels.len() {
            if i == j {
                continue;
            }
            let s = sim[i][j];
            let e = best.entry(labels[j]).or_insert(f32::MIN);
            if s > *e {
                *e = s;
            }
        }
        // Ties go to the lexicographically first cell, so the result does not
        // depend on map iteration order.
        let winner = best
            .iter()
            .max_by(|a, b| {
                a.1.partial_cmp(b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(b.0.cmp(a.0))
            })
            .map(|(c, _)| *c);
        if winner == Some(labels[i]) {
            *hits.entry(labels[i].to_string()).or_insert(0) += 1;
            total_hits += 1;
        }
    }

    let per_cell = sizes
        .iter()
        .map(|(c, n)| {
            let h = hits.get(c).copied().unwrap_or(0);
            (c.clone(), h as f32 / (*n).max(1) as f32)
        })
        .collect();
    Recovery {
        overall: total_hits as f32 / labels.len().max(1) as f32,
        per_cell,
        sizes,
    }
}

fn mean_sd(xs: &[f32]) -> (f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Entries that genuinely cluster by cell must clear the null.
    #[test]
    fn real_structure_beats_the_permutation() {
        let mut entries = Vec::new();
        for (cell, base) in [("A/x", [1.0f32, 0.0]), ("B/y", [0.0, 1.0])] {
            for k in 0..6 {
                let jitter = k as f32 * 0.01;
                entries.push((cell.to_string(), vec![base[0] + jitter, base[1] - jitter]));
            }
        }
        let r = run(&entries, 7, "test");
        assert!(r.overall_fitness > 0.9, "got {}", r.overall_fitness);
        assert!(
            r.overall_fitness > r.overall_null_mean + 0.2,
            "real {} vs null {}",
            r.overall_fitness,
            r.overall_null_mean
        );
        assert_eq!(r.cells_above_null, 2);
    }

    /// And labels sprinkled over one undifferentiated blob must NOT. This is
    /// the case the null exists to catch: two full cells that carve nothing.
    #[test]
    fn labels_over_a_single_blob_do_not_beat_the_permutation() {
        let mut entries = Vec::new();
        for k in 0..12 {
            let cell = if k % 2 == 0 { "A/x" } else { "B/y" };
            // One cloud; the label is arbitrary.
            entries.push((cell.to_string(), vec![1.0 + k as f32 * 0.001, 1.0]));
        }
        let r = run(&entries, 7, "test");
        assert_eq!(
            r.cells_above_null, 0,
            "a blob with labels on it scored {} cells above null",
            r.cells_above_null
        );
    }

    /// Leave-one-out is load-bearing: without it every entry matches itself at
    /// cosine 1.0 and every cell scores a flawless 1.000.
    #[test]
    fn an_entry_does_not_score_against_itself() {
        let entries = vec![
            ("A/x".to_string(), vec![1.0f32, 0.0]),
            ("B/y".to_string(), vec![0.99, 0.01]),
            ("B/y".to_string(), vec![0.0, 1.0]),
        ];
        let r = run(&entries, 7, "test");
        // The lone A entry is nearest to a B entry, so it must NOT come home.
        let a = r.per_cell.iter().find(|c| c.cell == "A/x").unwrap();
        assert_eq!(a.fitness, 0.0, "self-matching leaked back in");
    }

    /// Same seed, same numbers.
    #[test]
    fn the_null_is_deterministic() {
        let entries: Vec<(String, Vec<f32>)> = (0..10)
            .map(|k| (format!("C{}/m", k % 3), vec![k as f32, 1.0]))
            .collect();
        let a = run(&entries, 11, "test");
        let b = run(&entries, 11, "test");
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }
}
