// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 22 — user-proposed: "word discovery compared to an OWL
//! database, that only then binds to the embedder." Symbolic, versioned,
//! deterministic lexical structure (WordNet — same STABILITY property as
//! physis-core's hand-authored 70-cell grid, but here supplying a much
//! finer-grained, general-purpose vocabulary) supplies the PRIMARY
//! anchors; the embedder is used only afterward, for a bounded,
//! secondary job: relating text that has NO lexical match to the
//! nearest text that DOES.
//!
//! Princeton WordNet 3.1 (noun data), pulled from a GitHub mirror of the
//! standard `data.noun`/`index.noun` database format
//! (extjwnl/extjwnl-data-wn31), parsed via the `wordnet-db` crate — no
//! full OWL/DL reasoner needed for term matching against a controlled
//! vocabulary + hypernym hierarchy, which is all this needs.
//!
//! Pipeline:
//!   1. WORD DISCOVERY: tokenize each real ontology entry's text, check
//!      every candidate word against WordNet's noun index (deterministic
//!      lexical lookup — the same word always either is or isn't in
//!      WordNet 3.1, forever, independent of any embedder).
//!   2. For each match, walk ONE hypernym pointer ("@") to get a parent
//!      category — a genuinely fixed, symbolic anchor.
//!   3. BIND TO THE EMBEDDER: for entries with ZERO WordNet-recognized
//!      nouns (pure domain jargon — "OKR", "SCADA", "PDCA" are not
//!      general-English words), find their nearest neighbor AMONG
//!      entries that DO have a symbolic anchor, extending lexical
//!      coverage via embedding similarity rather than asking the
//!      embedder to do the primary discovery work.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment22_wordnet_discovery

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::HashMap;
use wordnet_db::WordNet;
use wordnet_types::{Pos, SynsetId};

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|w| w.len() > 2)
        .map(|w| w.to_lowercase())
        .collect()
}

/// One hop up the hypernym chain, if one exists — a fixed, symbolic
/// parent category, not derived from any embedding.
fn hypernym_parent(wn: &WordNet, id: SynsetId) -> Option<String> {
    let synset = wn.get_synset(id)?;
    let ptr = synset.pointers.iter().find(|p| p.symbol == "@")?;
    let parent = wn.get_synset(ptr.target)?;
    parent.words.first().map(|w| w.text.replace('_', " "))
}

/// For a tokenized entry, the set of (matched_word, hypernym_parent) pairs
/// discovered via WordNet alone — no embedding involved at this stage.
fn wordnet_anchors(wn: &WordNet, words: &[String]) -> Vec<(String, String)> {
    let mut seen_parents = std::collections::HashSet::new();
    let mut out = Vec::new();
    for w in words {
        if !wn.lemma_exists(Pos::Noun, w) { continue; }
        let offsets = wn.synsets_for_lemma(Pos::Noun, w);
        let Some(&first) = offsets.first() else { continue };
        if let Some(parent) = hypernym_parent(wn, first) {
            if seen_parents.insert(parent.clone()) {
                out.push((w.clone(), parent));
            }
        }
    }
    out
}

fn main() {
    println!("Experiment 22: word discovery vs. WordNet, binding to the embedder only afterward\n");

    let wn_dir = ["models/wordnet", "../models/wordnet"].iter().find(|d| std::path::Path::new(d).join("data.noun").exists());
    let wn_dir = match wn_dir {
        Some(d) => *d,
        None => { println!("WARNING: WordNet data not found — aborting."); return; }
    };
    let wn = match WordNet::load(wn_dir) {
        Ok(wn) => wn,
        Err(e) => { println!("WARNING: failed to load WordNet: {e} — aborting."); return; }
    };
    println!("Loaded WordNet: {} noun index entries, {} synsets total.\n", wn.index_count(), wn.synset_count());

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        let minilm_dir = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists());
        let minilm = match minilm_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() })) {
            Some(e) if e.is_available() => e,
            _ => { println!("WARNING: MiniLM not available — aborting."); return; }
        };

        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        for def in ontology.classification_domains() {
            if def.domain.is_none() || def.mode.is_none() { continue; }
            let mut text = def.name.clone();
            for hint in &def.hints { text.push(' '); text.push_str(hint); }
            texts.push(text);
        }
        let n = texts.len();
        println!("Loaded {n} real ontology entries.\n");

        // ── Step 1-2: word discovery against WordNet, per entry ──
        let anchors: Vec<Vec<(String, String)>> = texts.iter().map(|t| wordnet_anchors(&wn, &tokenize(t))).collect();
        let with_anchor: Vec<usize> = (0..n).filter(|&i| !anchors[i].is_empty()).collect();
        let without_anchor: Vec<usize> = (0..n).filter(|&i| anchors[i].is_empty()).collect();
        println!("=== Word discovery coverage ===");
        println!("{}/{} entries ({:.1}%) have at least one WordNet-recognized noun with a hypernym parent.", with_anchor.len(), n, 100.0 * with_anchor.len() as f32 / n as f32);
        println!("{}/{} entries ({:.1}%) are pure jargon by this lexicon — zero WordNet nouns matched.\n", without_anchor.len(), n, 100.0 * without_anchor.len() as f32 / n as f32);

        let mut parent_counts: HashMap<String, usize> = HashMap::new();
        for a in &anchors { for (_, p) in a { *parent_counts.entry(p.clone()).or_default() += 1; } }
        let mut top_parents: Vec<(&String, &usize)> = parent_counts.iter().collect();
        top_parents.sort_by(|a, b| b.1.cmp(a.1));
        println!("Top 15 hypernym parent categories discovered (fixed, symbolic — never change on rerun):");
        for (p, c) in top_parents.iter().take(15) { println!("  {p:<20} {c} entries touch this category"); }

        println!("\nExample entries WITH a WordNet anchor:");
        for &i in with_anchor.iter().take(6) {
            println!("  '{}'  anchors={:?}", texts[i].chars().take(60).collect::<String>(), anchors[i]);
        }
        println!("\nMulti-category entries (matched words map to 2+ DIFFERENT hypernym parents — a symbolically-grounded analog to multi-membership):");
        let multi: Vec<usize> = with_anchor.iter().filter(|&&i| anchors[i].len() >= 2).copied().collect();
        println!("{}/{} anchored entries touch 2+ distinct parent categories.", multi.len(), with_anchor.len());
        for &i in multi.iter().take(6) {
            println!("  '{}'  anchors={:?}", texts[i].chars().take(60).collect::<String>(), anchors[i]);
        }

        // ── Step 3: bind pure-jargon entries to the embedder, extending symbolic coverage ──
        println!("\n=== Binding to the embedder: pure-jargon entries linked to their nearest WordNet-anchored entry ===");
        let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| minilm.embed(t)).collect();
        for &i in without_anchor.iter().take(10) {
            let (best_j, best_sim) = with_anchor.iter()
                .map(|&j| (j, cosine_sim(&embeddings[i], &embeddings[j])))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            println!(
                "  '{}' (no lexical match)  -->  nearest anchored entry: '{}' (sim={:.3}, inherits anchors={:?})",
                texts[i].chars().take(45).collect::<String>(),
                texts[best_j].chars().take(45).collect::<String>(),
                best_sim, anchors[best_j]
            );
        }

        println!("\n(the word-discovery step (1-2) is fully deterministic given a fixed WordNet version — same word, same hypernym, forever, no embedder involved. Only step 3, extending coverage to jargon the lexicon doesn't reach, uses the embedder, in a bounded, clearly-scoped role rather than as the primary discovery mechanism.)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; word-discovery-only mode not implemented in this experiment.");
}
