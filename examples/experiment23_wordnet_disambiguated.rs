//! Experiment 23 — fixes the two real flaws Iteration 22 found and
//! reported honestly rather than glossed over: (1) several "jargon"
//! entries were actually Italian text colliding with unrelated English
//! WordNet words ("con"->"argument", "piano"->"keyboard instrument");
//! (2) naive first-sense lookup picked wrong senses for genuinely
//! ambiguous English words ("give"->"elasticity" instead of "donation").
//!
//! Fix 1 — language detection (`whatlang`, new dependency) BEFORE
//! WordNet lookup: entries not confidently detected as English are
//! excluded from anchoring entirely rather than forced through an
//! English-only lexicon.
//!
//! Fix 2 — gloss-based sense disambiguation, a second, DIFFERENT role
//! for "binding to the embedder" than Iteration 22's coverage-extension
//! use: for every word with more than one WordNet noun sense, embed each
//! candidate synset's gloss definition and pick whichever sense's gloss
//! is most similar (cosine) to the entry's OWN embedding — the entry's
//! context should favor the contextually-correct sense over an obscure
//! one that merely happens to be indexed first. Words with only one
//! sense need no disambiguation and are looked up directly, avoiding
//! unnecessary embedding calls.
//!
//! Verification method: rerun the exact spot-check examples Iteration 22
//! flagged as wrong and check whether they are actually fixed, not just
//! whether the aggregate coverage number looks good.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment23_wordnet_disambiguated

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

/// Language check tied to the SAME lexicon doing the anchoring, not an
/// external general-purpose detector. A first attempt using `whatlang`
/// (tuned for natural prose) massively over-rejected this dataset's
/// short, telegraphic, keyword-list style text — genuinely English
/// entries like "Rest & Recovery sleep rest nap recovery" were flagged
/// non-English, dropping coverage from ~98% to 28.6%. Fixed with a
/// self-referential heuristic instead: an entry is English-enough to
/// trust if at least `threshold` of its tokens exist in ANY WordNet POS
/// index (not just nouns) — genuinely non-English entries (Italian
/// "cura", "terapia", "medicazione") mostly won't match at all, while
/// genuine English keyword-lists will match at a high rate even without
/// full sentence grammar.
fn wordnet_coverage_fraction(wn: &WordNet, words: &[String]) -> f32 {
    if words.is_empty() { return 0.0; }
    let hits = words.iter().filter(|w| {
        wn.lemma_exists(Pos::Noun, w) || wn.lemma_exists(Pos::Verb, w) || wn.lemma_exists(Pos::Adj, w) || wn.lemma_exists(Pos::Adv, w)
    }).count();
    hits as f32 / words.len() as f32
}

struct Anchor { word: String, sense_count: usize, chosen_gloss: String, parent: String }

/// Disambiguated word discovery: for ambiguous words, the embedder picks
/// whichever WordNet sense's gloss best matches a LOCAL context window
/// around the target word — not the whole entry's embedding. A first
/// version compared the entry's full embedding against each gloss and
/// found a real, diagnosed confound: the entry's general theme ("metric",
/// "quantify", "track") spuriously favored "log" -> "exponent" (also
/// mathematical-flavored) over the intended "written record" sense, even
/// though "logarithm" has nothing to do with a tracking log. A small
/// window (target word +/- 2 tokens) keeps the LOCAL sense cue without
/// the whole-document theme drowning it out.
#[cfg(feature = "embed-onnx")]
fn wordnet_anchors_disambiguated(
    wn: &WordNet,
    words: &[String],
    embed: &mut impl FnMut(&str) -> Vec<f32>,
    gloss_cache: &mut HashMap<(String, u32), Vec<f32>>,
    window_cache: &mut HashMap<String, Vec<f32>>,
) -> Vec<Anchor> {
    let mut seen_parents = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (idx, w) in words.iter().enumerate() {
        if !wn.lemma_exists(Pos::Noun, w) { continue; }
        let offsets = wn.synsets_for_lemma(Pos::Noun, w);
        if offsets.is_empty() { continue; }

        let chosen = if offsets.len() == 1 {
            offsets[0]
        } else {
            let start = idx.saturating_sub(2);
            let end = (idx + 3).min(words.len());
            let window = words[start..end].join(" ");
            let window_embedding = window_cache.entry(window.clone()).or_insert_with(|| embed(&window)).clone();
            // Disambiguate: pick the sense whose gloss is most similar to the LOCAL window.
            offsets.iter().copied().max_by(|&a, &b| {
                let sim_a = gloss_similarity(wn, a, &window_embedding, embed, gloss_cache, w);
                let sim_b = gloss_similarity(wn, b, &window_embedding, embed, gloss_cache, w);
                sim_a.partial_cmp(&sim_b).unwrap()
            }).unwrap()
        };

        let Some(synset) = wn.get_synset(chosen) else { continue };
        let Some(ptr) = synset.pointers.iter().find(|p| p.symbol == "@") else { continue };
        let Some(parent_synset) = wn.get_synset(ptr.target) else { continue };
        let Some(parent) = parent_synset.words.first().map(|l| l.text.replace('_', " ")) else { continue };
        if seen_parents.insert(parent.clone()) {
            out.push(Anchor { word: w.clone(), sense_count: offsets.len(), chosen_gloss: synset.gloss.definition.to_string(), parent });
        }
    }
    out
}

#[cfg(feature = "embed-onnx")]
fn gloss_similarity(
    wn: &WordNet, id: SynsetId, entry_embedding: &[f32],
    embed: &mut impl FnMut(&str) -> Vec<f32>,
    gloss_cache: &mut HashMap<(String, u32), Vec<f32>>,
    word: &str,
) -> f32 {
    let key = (word.to_string(), id.offset);
    if let Some(cached) = gloss_cache.get(&key) {
        return cosine_sim(entry_embedding, cached);
    }
    let Some(synset) = wn.get_synset(id) else { return f32::NEG_INFINITY };
    let gloss_emb = embed(synset.gloss.definition);
    let sim = cosine_sim(entry_embedding, &gloss_emb);
    gloss_cache.insert(key, gloss_emb);
    sim
}

fn main() {
    println!("Experiment 23: WordNet discovery, fixed — language detection + gloss-based sense disambiguation\n");

    let wn_dir = ["models/wordnet", "../models/wordnet"].iter().find(|d| std::path::Path::new(d).join("data.noun").exists());
    let wn_dir = match wn_dir { Some(d) => *d, None => { println!("WARNING: WordNet data not found — aborting."); return; } };
    let wn = match WordNet::load(wn_dir) { Ok(wn) => wn, Err(e) => { println!("WARNING: failed to load WordNet: {e} — aborting."); return; } };
    println!("Loaded WordNet: {} noun index entries.\n", wn.index_count());

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        let minilm_dir = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists());
        let minilm = match minilm_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() })) {
            Some(e) if e.is_available() => e,
            _ => { println!("WARNING: MiniLM not available — aborting."); return; }
        };
        let mut embed_fn = |t: &str| minilm.embed(t);

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

        // ── Fix 1: language detection, WordNet-self-referential heuristic ──
        let tokens_per_entry: Vec<Vec<String>> = texts.iter().map(|t| tokenize(t)).collect();
        let coverage: Vec<f32> = tokens_per_entry.iter().map(|w| wordnet_coverage_fraction(&wn, w)).collect();
        println!("=== Diagnostic: WordNet-coverage-fraction distribution, before picking a threshold ===");
        for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7] {
            let above = coverage.iter().filter(|&&c| c >= t).count();
            println!("  threshold={t:.1}: {above}/{n} entries ({:.1}%) pass", 100.0 * above as f32 / n as f32);
        }
        // Spot-check known cases from Iteration 22/23's first attempt to pick a threshold
        // that separates them correctly, rather than guessing.
        for (i, t) in texts.iter().enumerate() {
            if t.starts_with("Cura e Terapia") || t.starts_with("Rest & Recovery") || t.starts_with("Workshop & Tools") {
                println!("  known case '{}...': coverage={:.3}", t.chars().take(30).collect::<String>(), coverage[i]);
            }
        }
        let threshold = 0.4;
        let english_flags: Vec<bool> = coverage.iter().map(|&c| c >= threshold).collect();
        let n_english = english_flags.iter().filter(|&&b| b).count();
        println!("\n=== Language detection (chosen threshold={threshold:.1}) ===");
        println!("{n_english}/{n} entries ({:.1}%) pass the WordNet-coverage bar.", 100.0 * n_english as f32 / n as f32);
        println!("{} entries excluded from WordNet anchoring as insufficiently English-lexical (not forced through an English-only lexicon).\n", n - n_english);
        let excluded_examples: Vec<&str> = texts.iter().zip(&english_flags).filter(|(_, &e)| !e).map(|(t, _)| t.as_str()).take(8).collect();
        println!("Examples of excluded entries: {:?}\n", excluded_examples.iter().map(|s| s.chars().take(40).collect::<String>()).collect::<Vec<_>>());

        // ── Fix 2: gloss-based disambiguation using a LOCAL context window, only for English entries ──
        let mut gloss_cache: HashMap<(String, u32), Vec<f32>> = HashMap::new();
        let mut window_cache: HashMap<String, Vec<f32>> = HashMap::new();
        let anchors: Vec<Vec<Anchor>> = (0..n).map(|i| {
            if !english_flags[i] { return Vec::new(); }
            wordnet_anchors_disambiguated(&wn, &tokens_per_entry[i], &mut embed_fn, &mut gloss_cache, &mut window_cache)
        }).collect();

        let with_anchor = (0..n).filter(|&i| !anchors[i].is_empty()).count();
        println!("=== Disambiguated word discovery coverage ===");
        println!("{with_anchor}/{n_english} English entries ({:.1}%) have at least one disambiguated WordNet anchor.\n", 100.0 * with_anchor as f32 / n_english.max(1) as f32);

        // ── Verification: rerun Iteration 22's exact known-wrong examples ──
        println!("=== Verification: the specific errors Iteration 22 found, rechecked ===");
        let checks = ["give", "log", "serve"];
        for (i, t) in texts.iter().enumerate() {
            if t.starts_with("Volunteering") || t.starts_with("Self-Tracking") {
                println!("\n'{}':", t.chars().take(50).collect::<String>());
                for a in &anchors[i] {
                    let flag = if checks.contains(&a.word.as_str()) { " <-- was WRONG in Iteration 22, check now" } else { "" };
                    println!("  {} ({} senses) -> parent='{}' gloss=\"{}\"{}", a.word, a.sense_count, a.parent, a.chosen_gloss.chars().take(60).collect::<String>(), flag);
                }
            }
        }
        println!("\n=== Debug: all candidate senses for 'log' in the Self-Tracking entry, with LOCAL-WINDOW scores ===");
        for (i, t) in texts.iter().enumerate() {
            if !t.starts_with("Self-Tracking") { continue; }
            let toks = &tokens_per_entry[i];
            if let Some(idx) = toks.iter().position(|w| w == "log") {
                let start = idx.saturating_sub(2);
                let end = (idx + 3).min(toks.len());
                let window = toks[start..end].join(" ");
                println!("  local window: \"{window}\"");
                let window_embedding = embed_fn(&window);
                let offsets = wn.synsets_for_lemma(Pos::Noun, "log");
                for &id in offsets {
                    if let Some(synset) = wn.get_synset(id) {
                        let sim = gloss_similarity(&wn, id, &window_embedding, &mut embed_fn, &mut gloss_cache, "log");
                        println!("  sim={sim:.3}  \"{}\"", synset.gloss.definition);
                    }
                }
            }
        }

        println!("\n=== Italian-text false positives from Iteration 22 — now correctly excluded? ===");
        for (i, t) in texts.iter().enumerate() {
            if t.starts_with("Cura e Terapia") || t.starts_with("Pianificazione Strategica") || t.starts_with("Montaggio e Installazione") {
                println!("  '{}' -> detected as English: {} (Iteration 22 wrongly matched 'con'/'piano'/'trave' as English words here)", t.chars().take(50).collect::<String>(), english_flags[i]);
            }
        }
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
