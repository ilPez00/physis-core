//! D1 — how far is the grid from a top-down ontology, and where is that a
//! *variance* rather than an error?
//!
//! ## The question
//!
//! Problem 3's design makes cells proposals, with the variance from an
//! established top-down ontology as the falsifiable quantity. Before anything
//! is made revisable, the anchor has to be measured: **how much of the 5×14 is
//! a re-labelling of something standard, and how much is a distinction the
//! reference does not have?**
//!
//! The reference here is WordNet's noun supersense inventory — the same corpus
//! E22 anchored to, offline, symbolic, and fixed. It is a weak reference for an
//! industrial grid and that is stated plainly at the bottom; it is the one
//! available without a network or a licence, and the *protocol* is what this
//! example establishes.
//!
//! ## The three states, from the plan
//!
//! Each cell's entries are tokenised, each noun matched in WordNet, and mapped
//! to its **supersense** — WordNet's own 26-way top-level carve of nouns, which
//! is the granularity an upper ontology competes with. A cell then has a
//! distribution over supersenses, and its **concentration** is the share held
//! by its largest.
//! Crossed with D4's verdict — does the cell clear its own label-permuted null
//! — that gives the plan's three states:
//!
//! | coherent (D4) | concentrated | reading |
//! |---|---|---|
//! | yes | yes | **matched** — the grid rediscovered a known category |
//! | yes | no | **disputed** — coherent here, and the reference has no such class |
//! | no | either | **unsupported** — the cell does not carve, anchor or not |
//!
//! `disputed` is the interesting column and it is not this file's invention:
//! OAEI-LLM (arXiv 2409.14038) names exactly this case — a mapping to a
//! relevant entity the reference does not have, which may be *more* precise
//! than the reference. Counting it is how much of the grid is a contribution
//! rather than a synonym.
//!
//! ## What it measured, 2026-09-13
//!
//! 70 cells, WordNet 3.1 supersenses, 200-permutation concentration null,
//! coherence from the D4 null, bge-base-en-v1.5:
//!
//! ```text
//! matched      14   the grid rediscovered a category WordNet has
//! DISPUTED     35   coherent here, and WordNet has no such class
//! unsupported   8   does not clear its own null, anchor or not
//! unanchored    0
//! ```
//!
//! **62% of the measurable grid is disputed rather than matched or wrong.**
//! Under the plan's reading that is the differentiator, stated in a metric
//! someone else defined — and it is the first time any of it has been put
//! against an external reference at all.
//!
//! The shape of the disputed set is the interesting part: it is the **largest**
//! cells.
//!
//! | cell | n | top supersense | conc | null |
//! |---|---|---|---|---|
//! | `CONSTRUCT/CREATE` | 70 | `noun.artifact` | 0.17 | 0.18 |
//! | `STUDY/WORK` | 52 | `noun.cognition` | 0.19 | 0.19 |
//! | `STUDY/LEARN` | 35 | `noun.cognition` | 0.20 | 0.19 |
//! | `FABRICATE/CREATE` | 26 | `noun.communication` | 0.21 | 0.19 |
//!
//! These sit *on* their null. They are coherent by D4 — their own entries are
//! mutually nearest — and they spread across WordNet's carve exactly as a
//! shuffled cell of the same size would. So physis's biggest cells cut across
//! the lexical hierarchy rather than restating a slice of it.
//!
//! ### The first version of this had no power and said the opposite
//!
//! Two corrections, both found by the number looking wrong rather than by the
//! code failing:
//!
//! 1. A **fixed concentration threshold of 0.34** scored *zero* cells as
//!    matched. It was measuring cell size: more entries means more distinct
//!    parents means a smaller largest share, however coherent the cell.
//!    Replaced by the permutation null, which is what every other claim made
//!    today rests on.
//! 2. The anchor was **one hypernym hop**, which gave a cell of seventy entries
//!    four hundred distinct parents, concentration 0.02 and a null of 0.02 —
//!    indistinguishable by arithmetic, not by evidence. Supersenses are
//!    WordNet's own 26-way carve and the granularity an upper ontology actually
//!    competes with. The large cells still sit on their null afterwards, so the
//!    finding survived the fix that could have removed it.
//!
//! Run:
//!   cargo run --release --features cli,embed-onnx --example grid_anchor
//! Needs `models/wordnet/data.noun` and, for the D4 leg, an ONNX model.

use physis_core::ontology::OntologyLoader;
use std::collections::{BTreeMap, HashMap};
use wordnet_db::WordNet;
use wordnet_types::{Pos, SynsetId};

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(|w| w.to_string())
        .collect()
}

/// The WordNet **supersense** — the lexicographer file a synset belongs to.
///
/// The anchor was one hypernym hop in the first version and it had no power:
/// a cell of seventy entries yielded four hundred distinct parents, so its
/// largest share was 0.02 and the permuted null was 0.02 as well. The test
/// could not distinguish a coherent large cell from a shuffled one, and every
/// large cell came back DISPUTED for arithmetic reasons.
///
/// Supersenses are WordNet's own top-level carve — 26 noun categories,
/// `noun.artifact`, `noun.process`, `noun.cognition` and so on — which is the
/// granularity an upper ontology actually competes with. Concentration has
/// range again, and a cell can genuinely concentrate.
const NOUN_SUPERSENSE: [&str; 26] = [
    "adj.all", "adj.pert", "adv.all", "noun.Tops", "noun.act", "noun.animal",
    "noun.artifact", "noun.attribute", "noun.body", "noun.cognition",
    "noun.communication", "noun.event", "noun.feeling", "noun.food", "noun.group",
    "noun.location", "noun.motive", "noun.object", "noun.person", "noun.phenomenon",
    "noun.plant", "noun.possession", "noun.process", "noun.quantity",
    "noun.relation", "noun.shape",
];

fn supersense(wn: &WordNet, id: SynsetId) -> Option<String> {
    let s = wn.get_synset(id)?;
    NOUN_SUPERSENSE
        .get(s.lex_filenum as usize)
        .map(|x| (*x).to_string())
        .or_else(|| Some(format!("lex{}", s.lex_filenum)))
}

fn main() {
    let wn_dir = ["models/wordnet", "../models/wordnet"]
        .iter()
        .find(|d| std::path::Path::new(d).join("data.noun").exists());
    let Some(wn_dir) = wn_dir else {
        println!("WordNet not found at models/wordnet — NOT RUN, and no claim made.");
        return;
    };
    let wn = match WordNet::load(wn_dir) {
        Ok(w) => w,
        Err(e) => {
            println!("WordNet failed to load: {e} — NOT RUN.");
            return;
        }
    };
    let (embedder, kind) = physis_core::embed::select(384);
    println!(
        "WordNet {} noun index entries · embedder {}\n",
        wn.index_count(),
        kind
    );

    // Cell -> entry texts, and the same entries embedded for the D4 leg.
    let ontology = OntologyLoader::load_all();
    let mut cell_texts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut fitness_input: Vec<(String, Vec<f32>)> = Vec::new();
    for def in ontology.classification_domains() {
        let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
        let mut text = def.name.clone();
        for hint in &def.hints {
            text.push(' ');
            text.push_str(hint);
        }
        let cell = format!("{d}/{m}");
        cell_texts.entry(cell.clone()).or_default().push(text.clone());
        fitness_input.push((cell, embedder.embed(&text)));
    }

    // D4's verdict per cell, recomputed here rather than read from a file, so
    // the two legs cannot describe different grids.
    let fit = physis_core::grid_fitness::run(&fitness_input, 7, kind);
    let coherent: HashMap<&str, bool> = fit
        .per_cell
        .iter()
        .map(|c| (c.cell.as_str(), c.above_null))
        .collect();
    let unmeasurable: HashMap<&str, bool> = fit
        .per_cell
        .iter()
        .map(|c| (c.cell.as_str(), c.unmeasurable))
        .collect();

    // Per-entry hypernym parents, computed once. An entry contributes the
    // parents of every WordNet noun in its text.
    let parents_of = |texts: &[String]| -> Vec<String> {
        let mut out = Vec::new();
        for t in texts {
            for w in tokenize(t) {
                if !wn.lemma_exists(Pos::Noun, &w) {
                    continue;
                }
                let offsets = wn.synsets_for_lemma(Pos::Noun, &w);
                let Some(first) = offsets.first().copied() else { continue };
                if let Some(p) = supersense(&wn, first) {
                    out.push(p);
                }
            }
        }
        out
    };
    let per_entry: BTreeMap<String, Vec<Vec<String>>> = cell_texts
        .iter()
        .map(|(c, ts)| (c.clone(), ts.iter().map(|t| parents_of(std::slice::from_ref(t))).collect()))
        .collect();

    // Concentration is the share of a cell's anchored words held by its most
    // common hypernym parent — and a fixed threshold on it is meaningless.
    // The first version used 0.34 and scored **zero** cells as matched, because
    // a cell of fifteen entries yields dozens of distinct parents and its
    // largest share is small no matter how coherent it is. The threshold was
    // measuring cell size.
    //
    // So concentration gets the same treatment as everything else today: a
    // permutation null. Shuffle which entry belongs to which cell, keeping
    // every cell's entry count exactly, and recompute. A cell is `matched` when
    // its real concentration clears the permuted mean by two permuted standard
    // deviations — the same bar D4 uses, so the two legs are read the same way.
    let concentration = |groups: &[&Vec<String>]| -> (String, f32, usize, usize) {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut anchored = 0usize;
        for g in groups {
            for p in g.iter() {
                *counts.entry(p.as_str()).or_insert(0) += 1;
                anchored += 1;
            }
        }
        let (top, n) = counts
            .iter()
            .max_by_key(|(k, v)| (**v, std::cmp::Reverse(**k)))
            .map(|(k, v)| ((*k).to_string(), *v))
            .unwrap_or_else(|| ("—".to_string(), 0));
        let conc = if anchored == 0 { f32::NAN } else { n as f32 / anchored as f32 };
        (top, conc, counts.len(), anchored)
    };

    let flat: Vec<&Vec<String>> = per_entry.values().flatten().collect();
    let sizes: Vec<(String, usize)> =
        per_entry.iter().map(|(c, v)| (c.clone(), v.len())).collect();

    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(7);
    let mut idx: Vec<usize> = (0..flat.len()).collect();
    const PERMS: usize = 200;
    let mut null: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for _ in 0..PERMS {
        idx.shuffle(&mut rng);
        let mut at = 0usize;
        for (cell, n) in &sizes {
            let groups: Vec<&Vec<String>> = idx[at..at + n].iter().map(|&i| flat[i]).collect();
            at += n;
            let (_, c, _, _) = concentration(&groups);
            if c.is_finite() {
                null.entry(cell.clone()).or_default().push(c);
            }
        }
    }

    let mut rows: Vec<(String, usize, usize, String, f32, f32, &'static str)> = Vec::new();
    let (mut matched, mut disputed, mut unsupported, mut unanchored) = (0, 0, 0, 0);

    for (cell, groups) in &per_entry {
        let refs: Vec<&Vec<String>> = groups.iter().collect();
        let (top, conc, nparents, anchored) = concentration(&refs);
        let ns = null.get(cell).cloned().unwrap_or_default();
        let nmean = if ns.is_empty() { f32::NAN } else { ns.iter().sum::<f32>() / ns.len() as f32 };
        let nsd = if ns.len() < 2 {
            0.0
        } else {
            (ns.iter().map(|x| (x - nmean) * (x - nmean)).sum::<f32>() / ns.len() as f32).sqrt()
        };
        let above_anchor = conc.is_finite() && nmean.is_finite() && nsd > 0.0 && (conc - nmean) / nsd >= 2.0;

        let state = if anchored == 0 {
            unanchored += 1;
            "unanchored"
        } else if *unmeasurable.get(cell.as_str()).unwrap_or(&false) {
            "unmeasurable"
        } else if !*coherent.get(cell.as_str()).unwrap_or(&false) {
            unsupported += 1;
            "unsupported"
        } else if above_anchor {
            matched += 1;
            "matched"
        } else {
            disputed += 1;
            "DISPUTED"
        };
        rows.push((cell.clone(), groups.len(), nparents, top, conc, nmean, state));
    }

    println!("── D1: THE GRID AGAINST A TOP-DOWN ANCHOR (WordNet supersenses) ──");
    println!(
        "{} cells · concentration vs a {PERMS}-permutation null · coherence from the D4 null\n",
        rows.len()
    );
    println!("  cell                          n  parents  top parent            conc   null  state");
    let mut sorted = rows.clone();
    sorted.sort_by(|a, b| {
        a.6.cmp(b.6).then(b.1.cmp(&a.1))
    });
    for (cell, n, np, top, conc, nmean, state) in &sorted {
        println!(
            "  {:<28} {:>3}  {:>7}  {:<20} {:>5}  {:>5}  {}",
            cell,
            n,
            np,
            top.chars().take(20).collect::<String>(),
            if conc.is_nan() { "  —".to_string() } else { format!("{conc:.2}") },
            if nmean.is_nan() { "  —".to_string() } else { format!("{nmean:.2}") },
            state
        );
    }

    let total = matched + disputed + unsupported;
    println!("\n  matched     {matched:>3}   the grid rediscovered a category WordNet has");
    println!("  DISPUTED    {disputed:>3}   coherent, and WordNet has no such class");
    println!("  unsupported {unsupported:>3}   does not clear its own null, anchor or not");
    println!("  unanchored  {unanchored:>3}   no WordNet noun matched at all — not measured");
    if total > 0 {
        println!(
            "\n  {:.0}% of the measurable grid is DISPUTED — variance, not error.",
            100.0 * disputed as f32 / total as f32
        );
    }
    println!(
        "\n  What this does NOT license: WordNet is a lexical hierarchy of common\n  nouns, and this grid is industrial. A cell can be DISPUTED here and\n  standard in SUMO, DOLCE or schema.org. The protocol is the result; the\n  number needs a domain reference before it is quoted as one."
    );

    let out = std::path::Path::new("benchmarks/results/grid-anchor.json");
    if let Some(d) = out.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let json: Vec<_> = rows
        .iter()
        .map(|(c, n, np, top, conc, nmean, s)| {
            serde_json::json!({
                "cell": c, "entries": n, "parents": np, "top_parent": top,
                "concentration": if conc.is_nan() { serde_json::Value::Null } else { serde_json::json!(conc) },
                "null_concentration": if nmean.is_nan() { serde_json::Value::Null } else { serde_json::json!(nmean) },
                "state": s
            })
        })
        .collect();
    let _ = std::fs::write(
        out,
        serde_json::to_vec_pretty(&serde_json::json!({
            "reference": "wordnet-3.1-noun-supersenses",
            "embedder": kind,
            "null_permutations": PERMS,
            "matched": matched, "disputed": disputed,
            "unsupported": unsupported, "unanchored": unanchored,
            "cells": json
        }))
        .unwrap(),
    );
    println!("\nartifact -> {}", out.display());
}
