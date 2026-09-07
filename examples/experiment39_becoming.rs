//! Experiment 39 — does `becoming` tell a meaning-change from a name collision?
//!
//! `physis_core::becoming` claims that ORDER separates two cases every
//! order-blind method conflates: a term whose senses are time-separated (one
//! turned into the other) versus interleaved (one name, two things). The unit
//! tests prove the statistic behaves on synthetic sequences. This tests it on
//! real text with ground truth that can be checked independently.
//!
//! ## The ground truth, and why it is unusually good
//!
//! The operational corpus spans 2026-05-22 to 2026-09-07 and INCLUDES the
//! session that produced it. So there are terms whose meaning changed at a
//! datable moment, verifiable from git rather than asserted:
//!
//!   BECOMING expected — a generic word that became a module name today:
//!     "coverage"  generic English until physis-core 0.1.17 (07-08 Sep)
//!     "linkage"   generic until physis-core 0.1.16 (06 Sep)
//!     "disposer"  no prior use in this sense at all
//!
//!   SPLIT expected — one name over senses that coexist throughout:
//!     "key"       licence key, signing key, event_key, HashMap key
//!     "core"      physis-core the crate, core the engine, CPU core
//!
//!   STABLE expected — ordinary vocabulary with no shift:
//!     "commit", "test", "file"
//!
//! A term is not scored on whether the verdict is pleasant; it is scored
//! against what git says actually happened.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment39_becoming -- <corpus.json>

use physis_core::becoming::{classify, Deviation, Trajectory};
use serde::Deserialize;

#[derive(Deserialize)]
struct Event {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    evidence: Vec<String>,
    observed_at: String,
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn main() {
    println!("Experiment 39: becoming vs split, on real terms with datable histories\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::becoming::Occurrence;
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let path = std::env::args().nth(1).unwrap_or_else(|| "operational_corpus.json".into());
        let Ok(raw) = std::fs::read_to_string(&path) else {
            println!("WARNING: cannot read {path} — run populate_operational first.");
            return;
        };
        let Ok(mut events) = serde_json::from_str::<Vec<Event>>(&raw) else {
            println!("WARNING: {path} is not a Vec<Event>.");
            return;
        };
        // Chronological order IS the evidence, so sort explicitly rather than
        // trusting the file.
        events.sort_by(|a, b| a.observed_at.cmp(&b.observed_at));
        let texts: Vec<String> = events
            .iter()
            .map(|e| format!("{} {}", e.subject, e.evidence.join(" ")).to_lowercase())
            .collect();
        println!("corpus: {} events, {} .. {}", events.len(),
                 events.first().map(|e| e.observed_at.as_str()).unwrap_or("?"),
                 events.last().map(|e| e.observed_at.as_str()).unwrap_or("?"));

        let Some(dir) = ["models", "../models"]
            .iter()
            .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        else {
            println!("WARNING: MiniLM not available — aborting.");
            return;
        };
        let embedder = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 384,
            model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if !embedder.is_available() {
            println!("WARNING: embedder unavailable — aborting.");
            return;
        }

        // CONTEXT IS A WINDOW AROUND THE OCCURRENCE, not the whole event.
        //
        // The first version embedded the entire event text and called it "the
        // context of the word". That clusters DOCUMENT GENRE — a commit message
        // and a README paragraph embed differently whatever the word means — so
        // it reported a split for every term including stable ones, with change
        // points at index 0 or 1 (i.e. "the first document is different").
        // Local context is what carries word sense.
        const WINDOW: usize = 12;
        let occurrences = |term: &str, embedder: &OnnxEmbedder| -> Vec<Occurrence> {
            let mut out = Vec::new();
            for (i, t) in texts.iter().enumerate() {
                let toks: Vec<&str> = t.split_whitespace().collect();
                for (k, w) in toks.iter().enumerate() {
                    // whole-word match, not substring: "invoices" must not
                    // match "voice" (the same bug the other agent just fixed in
                    // the ingest path).
                    let clean: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
                    if clean != term {
                        continue;
                    }
                    let lo = k.saturating_sub(WINDOW);
                    let hi = (k + WINDOW + 1).min(toks.len());
                    // The term itself is dropped from its own window: every
                    // occurrence contains it, so it contributes nothing but a
                    // constant pull toward the same direction.
                    let win: Vec<&str> = toks[lo..hi]
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| lo + j != k)
                        .map(|(_, s)| *s)
                        .collect();
                    if win.len() < 6 {
                        continue;
                    }
                    out.push(Occurrence {
                        at: i as i64,
                        context: normalize(&embedder.embed(&win.join(" "))),
                    });
                }
            }
            out
        };

        let cases: [(&str, &[&str]); 3] = [
            ("BECOMING expected", &["coverage", "linkage", "disposer"]),
            ("SPLIT expected", &["key", "core"]),
            ("STABLE expected", &["commit", "test", "file"]),
        ];

        let mut correct = 0usize;
        let mut scored = 0usize;
        println!("  term         n    trajectory   separation   runs z   change@   verdict");
        for (label, terms) in cases {
            println!("\n  --- {label} ---");
            for t in terms {
                let occs = occurrences(t, &embedder);
                let v = classify(&occs);
                let expected = match label {
                    "BECOMING expected" => Trajectory::Becoming,
                    "SPLIT expected" => Trajectory::Split,
                    _ => Trajectory::Stable,
                };
                let ok = v.trajectory == expected;
                if v.trajectory != Trajectory::TooFew {
                    scored += 1;
                    if ok { correct += 1; }
                }
                println!(
                    "  {:<11} {:>4}   {:<11?} {:>9.3}   {:>+6.2}   {:>7}   {}",
                    t,
                    occs.len(),
                    v.trajectory,
                    v.separation,
                    v.runs_z,
                    v.change_at.map(|c| c.to_string()).unwrap_or_else(|| "-".into()),
                    if v.trajectory == Trajectory::TooFew { "(not enough occurrences)" }
                    else if ok { "as expected" } else { "MISMATCH" }
                );
                // For a becoming, name the moment: what was the corpus doing
                // at the change point?
                if let Some(c) = v.change_at {
                    if let Some(o) = occs.get(c) {
                        let e = &events[o.at as usize];
                        println!("        change at {} — {}", e.observed_at, e.subject.chars().take(70).collect::<String>());
                    }
                }
                let mis = v.deviations.iter().filter(|d| **d == Some(Deviation::Mistake)).count();
                let var = v.deviations.iter().filter(|d| **d == Some(Deviation::Variation)).count();
                if mis + var > 0 {
                    println!("        deviations: {mis} mistake(s), {var} variation(s)");
                }
            }
        }

        println!("\n=== Verdict ===\n");
        println!("  {correct}/{scored} terms classified as their git history says they should be.");
        if scored > 0 && correct * 2 > scored {
            println!("  Order carries the distinction on real text, not only on synthetic sequences.");
        } else {
            println!("  Order does NOT carry the distinction here. The statistic behaves on");
            println!("  synthetic sequences (unit tests) but does not survive real contexts.");
        }
        println!(
            "\n(What is honest about this ground truth: it comes from git, not from my reading\n of the text — \"coverage\" became a module name at a commit with a timestamp. What\n is weak about it: n = 8 terms, chosen by me, and a corpus that is one project's\n own documentation, so \"becoming\" here means \"a word this project started using\n differently\", not meaning-change in general. The unit tests, not this, are what\n prove the statistic separates the two orderings.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
