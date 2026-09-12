//! EXPERIMENT 53: SYMBOLIC ANALOGY — the first null that `transform` has faced.
//!
//! `transform.rs` is a symbolic homomorphism engine: deterministic, no
//! embeddings, patterns matched by exact token equality with variable binding.
//! Because it touches no embedding, it inherits none of the settled negatives
//! of the E1-E18 track. It has also never been measured against a control,
//! which `docs/WHAT_PHYSIS_IS.md` §5 records under "never asked". This is the
//! asking.
//!
//! ## Task: link prediction over a real dependency graph
//!
//! Facts are extracted from this crate's own source: `crate::X` appearing in
//! module `M` yields the triple `(M, Requires, X)`. Real data, verifiable by
//! grep, and not authored for the experiment.
//!
//! One edge `(M, Requires, T)` is held out at a time. The method sees every
//! other edge and must rank candidates for `T`.
//!
//! ## Method (ANALOGY)
//!
//! The A->B path made concrete. For held-out subject `M`:
//!   1. Find modules `M'` structurally similar to `M` — sharing the most
//!      *other* `Requires` targets. That similarity is the homomorphism:
//!      a binding `M' -> M` justified by the edges they already share.
//!   2. Each `M'` votes for its own targets that `M` does not yet have,
//!      weighted by the overlap that justified the binding.
//!   3. Rank by summed vote.
//!
//! No embeddings, no model, no training. Exact token equality only.
//!
//! ## Nulls — the experiment is built to be able to lose
//!
//!   POPULARITY  always rank by global in-degree. The "majority" baseline;
//!               the one that beats naive methods embarrassingly often.
//!   PERMUTED    identical ANALOGY code over an edge-shuffled graph that
//!               preserves both degree sequences. Construction-matched: it
//!               isolates *structure* from *degree*, so if ANALOGY only
//!               exploits popularity, PERMUTED matches it.
//!   RANDOM      uniform over candidates. The floor.
//!
//! Standing rule of this track, earned seven times: treat any positive as an
//! artifact until a construction-matched control says otherwise. PERMUTED is
//! that control.
//!
//! Run: cargo run --release --example experiment53_transform_analogy

use physis_core::transform::{PatElem, Predicate, Triple, TriplePattern, find_homomorphisms};
use std::collections::{BTreeMap, BTreeSet};

// ── Deterministic RNG so the run reproduces bit-for-bit ─────────────────────
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() % n as u64) as usize }
    }
}

/// Extract `(module, Requires, target)` from `crate::TARGET` occurrences.
fn extract(dir: &std::path::Path) -> Vec<Triple> {
    // (subject, object) pairs rather than Triple: Triple is not Ord, and the
    // set is only here to dedupe + fix iteration order for reproducibility.
    let mut out: BTreeSet<(String, String)> = BTreeSet::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            let Some(m) = p.file_stem().and_then(|s| s.to_str()) else { continue };
            if m == "lib" || m == "main" {
                continue;
            }
            let Ok(body) = std::fs::read_to_string(&p) else { continue };
            for (i, _) in body.match_indices("crate::") {
                let rest = &body[i + 7..];
                let tgt: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if tgt.is_empty() || tgt == m {
                    continue;
                }
                // Modules only: a lowercase head is a module path here.
                if tgt.chars().next().is_some_and(|c| c.is_lowercase()) {
                    out.insert((m.to_string(), tgt));
                }
            }
        }
    }
    out.into_iter()
        .map(|(s, o)| Triple::new(&s, Predicate::Requires, &o))
        .collect()
}

type Edges = BTreeMap<String, BTreeSet<String>>;

fn to_edges(ts: &[Triple]) -> Edges {
    let mut e: Edges = BTreeMap::new();
    for t in ts {
        e.entry(t.s.clone()).or_default().insert(t.o.clone());
    }
    e
}

/// ANALOGY: rank targets for `subj` by votes from structurally similar modules.
fn analogy(edges: &Edges, subj: &str, known: &BTreeSet<String>) -> Vec<(String, f32)> {
    let mut votes: BTreeMap<String, f32> = BTreeMap::new();
    for (other, targets) in edges {
        if other == subj {
            continue;
        }
        let overlap = known.intersection(targets).count();
        if overlap == 0 {
            continue;
        }
        // The binding M' -> M is justified by shared edges; weight by how much.
        let w = overlap as f32 / (known.len().max(1) as f32).sqrt();
        for t in targets {
            if !known.contains(t) {
                *votes.entry(t.clone()).or_insert(0.0) += w;
            }
        }
    }
    rank(votes)
}

fn popularity(edges: &Edges, subj: &str, known: &BTreeSet<String>) -> Vec<(String, f32)> {
    let mut votes: BTreeMap<String, f32> = BTreeMap::new();
    for (other, targets) in edges {
        if other == subj {
            continue;
        }
        for t in targets {
            if !known.contains(t) {
                *votes.entry(t.clone()).or_insert(0.0) += 1.0;
            }
        }
    }
    rank(votes)
}

fn random_rank(all: &BTreeSet<String>, known: &BTreeSet<String>, rng: &mut Rng) -> Vec<(String, f32)> {
    let mut c: Vec<String> = all.difference(known).cloned().collect();
    for i in (1..c.len()).rev() {
        c.swap(i, rng.below(i + 1));
    }
    c.into_iter().map(|s| (s, 0.0)).collect()
}

/// Deterministic ranking: score desc, then name asc so ties never flap.
fn rank(votes: BTreeMap<String, f32>) -> Vec<(String, f32)> {
    let mut v: Vec<(String, f32)> = votes.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    v
}

/// Degree-preserving edge shuffle: repeatedly swap the targets of two edges.
/// Both degree sequences survive, so POPULARITY is unchanged by construction
/// and only the *pairing* — the structure ANALOGY reads — is destroyed.
fn permute(ts: &[Triple], rng: &mut Rng) -> Vec<Triple> {
    let mut pairs: Vec<(String, String)> = ts.iter().map(|t| (t.s.clone(), t.o.clone())).collect();
    for _ in 0..pairs.len() * 8 {
        let (i, j) = (rng.below(pairs.len()), rng.below(pairs.len()));
        if i == j {
            continue;
        }
        let (a, b) = (pairs[i].clone(), pairs[j].clone());
        if a.0 == b.0 || a.1 == b.1 {
            continue;
        }
        pairs[i] = (a.0, b.1);
        pairs[j] = (b.0, a.1);
    }
    let set: BTreeSet<(String, String)> = pairs.into_iter().collect();
    set.into_iter().map(|(s, o)| Triple::new(&s, Predicate::Requires, &o)).collect()
}

#[derive(Default, Clone, Copy)]
struct Hits {
    n: usize,
    t1: usize,
    t3: usize,
    t5: usize,
}
impl Hits {
    fn add(&mut self, ranked: &[(String, f32)], truth: &str) {
        self.n += 1;
        let pos = ranked.iter().position(|(c, _)| c == truth);
        if let Some(p) = pos {
            if p < 1 { self.t1 += 1 }
            if p < 3 { self.t3 += 1 }
            if p < 5 { self.t5 += 1 }
        }
    }
    fn row(&self, name: &str) -> String {
        let f = |x: usize| x as f32 / self.n.max(1) as f32;
        format!("  {name:<12} top1 {:.3}  top3 {:.3}  top5 {:.3}   (n={})", f(self.t1), f(self.t3), f(self.t5), self.n)
    }
}

/// Leave-one-out over every edge whose subject keeps >= 2 other edges — a
/// subject with no remaining context is not a test of analogy, it is a test of
/// the prior, and mixing them would inflate every arm equally.
fn evaluate(edges: &Edges, all: &BTreeSet<String>, rng: &mut Rng) -> (Hits, Hits, Hits) {
    let (mut an, mut pop, mut rnd) = (Hits::default(), Hits::default(), Hits::default());
    for (subj, targets) in edges {
        if targets.len() < 3 {
            continue;
        }
        for held in targets {
            let known: BTreeSet<String> = targets.iter().filter(|t| *t != held).cloned().collect();
            let mut trimmed = edges.clone();
            trimmed.insert(subj.clone(), known.clone());
            an.add(&analogy(&trimmed, subj, &known), held);
            pop.add(&popularity(&trimmed, subj, &known), held);
            rnd.add(&random_rank(all, &known, rng), held);
        }
    }
    (an, pop, rnd)
}

fn main() {
    // Second corpus for replication: one positive is an artifact until it
    // repeats on data it was not tuned on. `../src` is physis_pro — a tree
    // roughly 4x larger, written against different conventions.
    let arg = std::env::args().nth(1).unwrap_or_else(|| "src".into());
    let dir = std::path::Path::new(&arg);
    let triples = extract(dir);
    let edges = to_edges(&triples);
    let all: BTreeSet<String> = triples.iter().map(|t| t.o.clone()).collect();

    println!("EXPERIMENT 53 — symbolic analogy vs a construction-matched null");
    println!("corpus root: {}", dir.display());
    println!("corpus: {} modules, {} Requires edges, {} distinct targets",
             edges.len(), triples.len(), all.len());

    // Sanity: the engine's own matcher must see these facts. If
    // find_homomorphisms cannot bind a pattern over the extracted triples, the
    // triples are not in the form transform.rs consumes and every number below
    // would be measuring the harness instead of the engine.
    let pat = TriplePattern {
        s: PatElem::Var("m".into()),
        p: Some(Predicate::Requires),
        o: PatElem::Var("t".into()),
    };
    let binds = find_homomorphisms(&[pat], &triples);
    println!("transform::find_homomorphisms binds {} facts (must be > 0)\n", binds.len());
    assert!(!binds.is_empty(), "the engine cannot read its own corpus");

    let mut rng = Rng(0x5EED_1234_9ABC_DEF0);
    let (an, pop, rnd) = evaluate(&edges, &all, &mut rng);

    println!("REAL GRAPH");
    println!("{}", an.row("ANALOGY"));
    println!("{}", pop.row("POPULARITY"));
    println!("{}", rnd.row("RANDOM"));

    let permuted = permute(&triples, &mut Rng(0xC0FFEE_1234_5678));
    let pedges = to_edges(&permuted);
    let pall: BTreeSet<String> = permuted.iter().map(|t| t.o.clone()).collect();
    let (pan, ppop, _) = evaluate(&pedges, &pall, &mut rng);

    println!("\nPERMUTED GRAPH (degree-preserving — the construction-matched null)");
    println!("{}", pan.row("ANALOGY"));
    println!("{}", ppop.row("POPULARITY"));

    let f = |x: usize, n: usize| x as f32 / n.max(1) as f32;
    let real = f(an.t3, an.n);
    let null = f(pan.t3, pan.n);
    let popt = f(pop.t3, pop.n);
    println!("\nVERDICT (top-3)");
    println!("  ANALOGY real          {real:.3}");
    println!("  ANALOGY permuted null {null:.3}   delta {:+.3}", real - null);
    println!("  POPULARITY baseline   {popt:.3}   delta {:+.3}", real - popt);
    println!();
    if real - null > 0.05 && real - popt > 0.05 {
        println!("  SUPPORTED: analogy reads structure, not degree, and beats the");
        println!("  majority baseline. First measured positive for transform.rs.");
    } else if real - null <= 0.05 {
        println!("  NOT SUPPORTED: the permuted null matches it. Whatever ANALOGY");
        println!("  scores is degree, not structure. This is the FAIL branch and it");
        println!("  is a result: transform.rs does not beat its control on this task.");
    } else {
        println!("  PARTIAL: beats the permuted null but not the majority baseline.");
        println!("  Structure is real but does not pay against simply ranking by");
        println!("  popularity. Not a shippable positive.");
    }
}
