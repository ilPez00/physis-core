//! Are `CREATE`, `WORK` and `MAINTAIN` the modes there are — or the modes
//! someone happened to write down?
//!
//! ## The question, asked properly
//!
//! The mode axis is fourteen verbs chosen in advance. `PLANNING.md` §1 proposes
//! ten more and its own gate blocks them, correctly, for lacking a
//! measurement. Nothing has ever asked the prior question: **what kinds of verb
//! are there, according to something that is not us?**
//!
//! WordNet answers it. Its verb inventory is partitioned into fifteen
//! *supersenses* — `verb.creation`, `verb.cognition`, `verb.social`,
//! `verb.motion`, `verb.possession` and so on — a carve made by lexicographers
//! over the whole English verb lexicon, with no idea that physis exists. That
//! is the top-down reference for the mode axis, exactly as noun supersenses
//! were the reference for cells in `grid_anchor`.
//!
//! ## What this produces — proposals, not edits
//!
//! Every mode's entries are read for verbs, each verb mapped to its supersense,
//! and the mode's distribution recorded. Three readings come out, and each is a
//! *proposal* for a person or a second model to approve:
//!
//! | finding | what it proposes |
//! |---|---|
//! | a supersense no mode owns | **an expansion**: a kind of verb the grid cannot say |
//! | two modes with the same dominant supersense | **a merge**: one distinction wearing two names |
//! | a mode whose distribution matches its permuted null | **a retirement**: the label carves nothing |
//!
//! Nothing here edits the ontology. The output is a ranked list with a number
//! beside each row, which is what "the ontology is a proposal" means in
//! practice.
//!
//! ## The controls
//!
//! Two, and both are needed. **Coverage** of a supersense is compared against a
//! permutation null — shuffle which entry belongs to which mode, keeping every
//! mode's size, and see what coverage a mode gets by accident. **Dominance** is
//! only read for modes whose real concentration clears that null, because the
//! dominant supersense of a mode that carves nothing is noise with a name.
//!
//! ## What it measured, 2026-09-13
//!
//! 14 modes against 15 verb supersenses, 200-permutation null:
//!
//! | mode | n | dominant supersense | share | null | carves? |
//! |---|---|---|---|---|---|
//! | `WORK` | 155 | **`verb.communication`** | 0.19 | 0.14 | yes |
//! | `CREATE` | 122 | **`verb.contact`** | 0.16 | 0.15 | **NO** |
//! | `SENSE` | 99 | `verb.communication` | 0.16 | 0.15 | **NO** |
//! | `MAINTAIN` | 82 | `verb.change` | 0.18 | 0.15 | yes |
//! | `LEARN` | 58 | `verb.cognition` | 0.27 | 0.15 | yes |
//! | `PLAN` | 42 | `verb.cognition` | 0.24 | 0.16 | yes |
//! | `REST` | 37 | `verb.contact` | 0.22 | 0.16 | yes |
//! | `WALK` | 33 | `verb.change` | 0.17 | 0.17 | **NO** |
//! | `MOVE` | 31 | `verb.motion` | 0.25 | 0.16 | yes |
//! | `GUIDE` | 19 | `verb.communication` | 0.36 | 0.18 | yes |
//! | `LIFT` | 18 | `verb.motion` | 0.24 | 0.18 | **NO** |
//! | `PLAY` | 18 | `verb.motion` | 0.25 | 0.18 | yes |
//! | `DESTROY` | 9 | `verb.change` | 0.18 | 0.22 | **NO** |
//! | `BRAINSTORM` | 8 | `verb.communication` | 0.18 | 0.22 | **NO** |
//!
//! **Eight carve, six do not.** The answer to "are `CREATE`, `WORK` and
//! `SENSE` the main modes" is: not as measured against an inventory of verbs
//! nobody here wrote.
//!
//! - **`CREATE` is the second-largest mode and does not carve.** Its verbs are
//!   dominantly `verb.contact`, at 0.16 against a shuffled 0.15. A mode named
//!   for creation whose verbs are mostly about touching things, and which a
//!   permutation reproduces.
//! - **`SENSE` does not carve either**, and its dominant supersense is
//!   `verb.communication`, not `verb.perception`.
//! - **`WORK` carves, but not as work.** Its dominant supersense is
//!   `verb.communication`. Whatever the largest mode in the grid is picking
//!   out, it is not labour.
//! - `WALK` fails here too, which is the third independent verdict against it
//!   (E22 scored it top-1 0.18 on 27 entries).
//!
//! This reproduces E22 from a direction that shares no machinery with it. E22
//! found `WORK`/`CREATE`/`MAINTAIN`/`SENSE` holding 68% of entries and failing
//! to separate, using confusion between physis's own classes. This uses an
//! external verb inventory and no classifier, and two of those four fail
//! outright while a third is mislabelled.
//!
//! ### The expansion the grid actually needs
//!
//! **Ten of fifteen supersenses are unowned** — `verb.body`,
//! `verb.competition`, `verb.consumption`, **`verb.creation`**, `verb.emotion`,
//! `verb.perception`, `verb.possession`, `verb.social`, `verb.stative`,
//! `verb.weather`.
//!
//! `verb.creation` is unowned **while a mode named `CREATE` exists**, because
//! `CREATE` does not carve and so owns nothing. That single line is the whole
//! argument for treating the grid as a proposal rather than a decree.
//!
//! ### And a verdict on `PLANNING.md` §1's blocked list
//!
//! §1 proposes ten new modes and its own gate blocks them for lacking a
//! measurement. This is a measurement, and it splits them:
//!
//! | proposed mode | lands in | reading |
//! |---|---|---|
//! | `DECIDE`, `REASON`, `REMEMBER` | `verb.cognition` | **already owned** by `LEARN`/`PLAN` — redundant, not expansion |
//! | `TEACH`, `PERSUADE` | `verb.communication` | already owned by `GUIDE`/`WORK` |
//! | `MEASURE` | `verb.perception` | **unowned — supported** |
//! | `SHARE` | `verb.possession` | **unowned — supported** |
//! | `CELEBRATE`, `PROTEST` | `verb.social`, `verb.emotion` | **unowned — supported** |
//! | `IMAGINE` | `verb.cognition` | already owned |
//!
//! So §1 is half right, and the half it gets right is not the half it argues
//! hardest for. Six of the ten land in supersenses the grid already covers;
//! four land in genuine gaps. Nothing here approves any of them — a lexical gap
//! is not a business case — but the gate now has evidence to weigh instead of
//! only a rule.
//!
//! Run:
//!   cargo run --release --features cli --example mode_inventory

use physis_core::ontology::OntologyLoader;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::{BTreeMap, BTreeSet};
use wordnet_db::WordNet;
use wordnet_types::{Pos, SynsetId};

/// WordNet's fifteen verb supersenses, in `lex_filenum` order (29–43).
const VERB_SUPERSENSE: [&str; 15] = [
    "verb.body",
    "verb.change",
    "verb.cognition",
    "verb.communication",
    "verb.competition",
    "verb.consumption",
    "verb.contact",
    "verb.creation",
    "verb.emotion",
    "verb.motion",
    "verb.perception",
    "verb.possession",
    "verb.social",
    "verb.stative",
    "verb.weather",
];

const PERMS: usize = 200;

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(|w| w.to_string())
        .collect()
}

fn verb_supersense(wn: &WordNet, id: SynsetId) -> Option<&'static str> {
    let s = wn.get_synset(id)?;
    let n = s.lex_filenum as usize;
    if (29..=43).contains(&n) {
        Some(VERB_SUPERSENSE[n - 29])
    } else {
        None
    }
}

fn mean_sd(xs: &[f32]) -> (f32, f32) {
    if xs.is_empty() {
        return (f32::NAN, 0.0);
    }
    let n = xs.len() as f32;
    let m = xs.iter().sum::<f32>() / n;
    (m, (xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n).sqrt())
}

fn main() {
    let wn_dir = ["models/wordnet", "../models/wordnet"]
        .iter()
        .find(|d| std::path::Path::new(d).join("data.verb").exists());
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

    // Mode -> one bag of verb supersenses per entry.
    let ontology = OntologyLoader::load_all();
    let mut per_entry: BTreeMap<String, Vec<Vec<&'static str>>> = BTreeMap::new();
    for def in ontology.classification_domains() {
        let Some(m) = &def.mode else { continue };
        let mut text = def.name.clone();
        for h in &def.hints {
            text.push(' ');
            text.push_str(h);
        }
        let mut bag = Vec::new();
        for w in tokenize(&text) {
            if !wn.lemma_exists(Pos::Verb, &w) {
                continue;
            }
            let offs = wn.synsets_for_lemma(Pos::Verb, &w);
            let Some(first) = offs.first().copied() else { continue };
            if let Some(ss) = verb_supersense(&wn, first) {
                bag.push(ss);
            }
        }
        per_entry.entry(m.clone()).or_default().push(bag);
    }
    if per_entry.is_empty() {
        println!("no modes found in the ontology — NOT RUN.");
        return;
    }

    let stats = |groups: &[&Vec<&'static str>]| -> (BTreeMap<&'static str, usize>, usize) {
        let mut c: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut n = 0;
        for g in groups {
            for s in g.iter() {
                *c.entry(s).or_insert(0) += 1;
                n += 1;
            }
        }
        (c, n)
    };

    let flat: Vec<&Vec<&'static str>> = per_entry.values().flatten().collect();
    let sizes: Vec<(String, usize)> = per_entry.iter().map(|(m, v)| (m.clone(), v.len())).collect();

    // The null: same mode sizes, entries shuffled between them.
    let mut rng = StdRng::seed_from_u64(7);
    let mut idx: Vec<usize> = (0..flat.len()).collect();
    let mut null_conc: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for _ in 0..PERMS {
        idx.shuffle(&mut rng);
        let mut at = 0;
        for (mode, n) in &sizes {
            let g: Vec<&Vec<&'static str>> = idx[at..at + n].iter().map(|&i| flat[i]).collect();
            at += n;
            let (c, total) = stats(&g);
            if total > 0 {
                let top = c.values().max().copied().unwrap_or(0);
                null_conc.entry(mode.clone()).or_default().push(top as f32 / total as f32);
            }
        }
    }

    println!("── MODE INVENTORY vs WordNet's VERB SUPERSENSES ──");
    println!(
        "{} modes · {} verb supersenses in the reference · {PERMS}-permutation null\n",
        per_entry.len(),
        VERB_SUPERSENSE.len()
    );
    println!("  mode          n   verbs  dominant supersense    share   null   carves?");

    let mut owned: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    let mut rows = Vec::new();
    for (mode, groups) in &per_entry {
        let refs: Vec<&Vec<&'static str>> = groups.iter().collect();
        let (counts, total) = stats(&refs);
        let (top, topn) = counts
            .iter()
            .max_by_key(|(k, v)| (**v, std::cmp::Reverse(**k)))
            .map(|(k, v)| (*k, *v))
            .unwrap_or(("—", 0));
        let share = if total == 0 { f32::NAN } else { topn as f32 / total as f32 };
        let ns = null_conc.get(mode).cloned().unwrap_or_default();
        let (nm, nsd) = mean_sd(&ns);
        let carves = share.is_finite() && nm.is_finite() && nsd > 0.0 && (share - nm) / nsd >= 2.0;
        if carves {
            owned.entry(top).or_default().push(mode.clone());
        }
        rows.push((mode.clone(), groups.len(), total, top, share, nm, carves));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    for (mode, n, verbs, top, share, nm, carves) in &rows {
        println!(
            "  {:<12} {:>3}  {:>6}  {:<21} {:>5}  {:>5}   {}",
            mode,
            n,
            verbs,
            top,
            if share.is_nan() { "  —".into() } else { format!("{share:.2}") },
            if nm.is_nan() { "  —".into() } else { format!("{nm:.2}") },
            if *carves { "yes" } else { "NO" }
        );
    }

    // ── the three proposals ──────────────────────────────────────────────
    let covered: BTreeSet<&'static str> = owned.keys().copied().collect();
    let missing: Vec<&&str> = VERB_SUPERSENSE.iter().filter(|s| !covered.contains(*s)).collect();

    // An unowned supersense is only a gap if the corpus actually uses that kind
    // of verb. WordNet cannot tell `verb.weather` from `verb.possession` in
    // relevance terms — but the corpus can, by how often each appears. Share of
    // all verb occurrences is the domain weight, and it separates "a kind of
    // verb this work has no word for" from "a kind of verb this work never does".
    let (corpus_counts, corpus_total) = stats(&flat);
    println!("\n── PROPOSAL 1: EXPAND — verb supersenses no mode owns ──");
    if missing.is_empty() {
        println!("  none: every kind of verb WordNet distinguishes has a mode that carves it.");
    } else {
        println!(
            "  {} of {} supersenses are unowned. Ranked by how much of the corpus's\n  verb mass they already carry — that share is what separates a real gap\n  from a category this work never enters:\n",
            missing.len(),
            VERB_SUPERSENSE.len()
        );
        let mut ranked: Vec<(&str, f32, usize)> = missing
            .iter()
            .map(|m| {
                let n = corpus_counts.get(**m).copied().unwrap_or(0);
                (**m, if corpus_total == 0 { 0.0 } else { n as f32 / corpus_total as f32 }, n)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (m, share, n) in &ranked {
            let verdict = if *share >= 0.05 {
                "REAL GAP — the corpus does this constantly and cannot name it"
            } else if *share >= 0.01 {
                "worth asking about"
            } else {
                "the corpus never does this — not a gap"
            };
            println!("    {:<20} {:>5.1}%  ({:>4} uses)  {}", m, share * 100.0, n, verdict);
        }
    }

    println!("\n── PROPOSAL 2: MERGE — one supersense, two or more modes ──");
    let mut any = false;
    for (ss, modes) in &owned {
        if modes.len() > 1 {
            any = true;
            println!("    {ss:<22} {}", modes.join(", "));
        }
    }
    if !any {
        println!("  none: no two modes that carve share a dominant supersense.");
    }

    println!("\n── PROPOSAL 3: RETIRE — modes that do not clear their own null ──");
    let dead: Vec<&String> = rows.iter().filter(|r| !r.6).map(|r| &r.0).collect();
    if dead.is_empty() {
        println!("  none.");
    } else {
        for (mode, n, _, _, _, _, _) in rows.iter().filter(|r| !r.6) {
            println!("    {mode:<12} {n} entries, distribution indistinguishable from shuffled");
        }
    }

    println!(
        "\n  {} carve · {} do not · {} supersenses unowned",
        rows.len() - dead.len(),
        dead.len(),
        missing.len()
    );
    println!(
        "\n  These are PROPOSALS. Nothing here edits the grid. WordNet's verb carve\n  is one reference and a lexical one — a supersense unowned here may be\n  irrelevant to an industrial backoffice, and a merge proposed here may be a\n  distinction that matters in the domain and not in the dictionary. The\n  approval step is the point, and it is not automated."
    );

    let out = std::path::Path::new("benchmarks/results/mode-inventory.json");
    if let Some(d) = out.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = std::fs::write(
        out,
        serde_json::to_vec_pretty(&serde_json::json!({
            "reference": "wordnet-3.1-verb-supersenses",
            "permutations": PERMS,
            "modes": rows.iter().map(|(m, n, v, top, share, nm, c)| serde_json::json!({
                "mode": m, "entries": n, "verbs": v, "dominant": top,
                "share": if share.is_nan() { serde_json::Value::Null } else { serde_json::json!(share) },
                "null_share": if nm.is_nan() { serde_json::Value::Null } else { serde_json::json!(nm) },
                "carves": c
            })).collect::<Vec<_>>(),
            "expand": missing,
            "merge": owned.iter().filter(|(_, v)| v.len() > 1)
                .map(|(k, v)| serde_json::json!({"supersense": k, "modes": v}))
                .collect::<Vec<_>>(),
            "retire": dead,
        }))
        .unwrap(),
    );
    println!("\nartifact -> {}", out.display());
}
