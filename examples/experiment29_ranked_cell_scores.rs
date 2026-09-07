// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 29 — roadmap item 25: "surface `CellClassifier`'s per-cell
//! similarity scores instead of collapsing to one winning (domain, mode)".
//!
//! Reading the code first changed the question. `CellClassifier::classify()`
//! does **not** collapse to a winner — it returns `Vec<CellScore>` with every
//! populated cell, scored and ranked, and the consumers in `studio.rs`,
//! `studio_lab.rs` and `main.rs` all receive the full vector. So the
//! recommendation as written is largely already implemented, and the report
//! needs correcting rather than the code.
//!
//! What is genuinely single-valued is *storage*: `OntologyEntry`
//! (`src/models.rs`) has `domain: String, mode: String`, one each. That is an
//! authoring limit, not a query limit — an entry that truly spans two fields
//! can only be written into one cell.
//!
//! So the useful measurement is: **how much does the ranked list actually
//! carry beyond its argmax?** If the answer is "almost nothing", surfacing it
//! is cosmetic. If real information sits at ranks 2..k, the storage limit is
//! costing something concrete, and Iteration 8's claim has real support at
//! scale rather than on two hand-built examples.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment29_ranked_cell_scores

use physis_core::classify::CellClassifier;
use physis_core::ontology::OntologyLoader;

fn main() {
    println!("Experiment 29: what does the ranked cell list carry beyond its argmax?\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use std::collections::HashMap;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let dir = match ["models", "../models"]
            .iter()
            .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        {
            Some(d) => *d,
            None => {
                println!("WARNING: MiniLM not available — aborting.");
                return;
            }
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

        let ontology = OntologyLoader::load_all();
        let classifier = CellClassifier::build(&ontology, &embedder);
        println!("Classifier built: {} populated cells.\n", classifier.cell_count());

        // Every real entry, with the cell it was authored into.
        let mut items: Vec<(String, String, String)> = Vec::new(); // (text, domain, mode)
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut t = def.name.clone();
            for h in &def.hints {
                t.push(' ');
                t.push_str(h);
            }
            items.push((t, d.clone(), m.clone()));
        }
        println!("Classifying {} real entries against their own ontology...\n", items.len());

        let mut rank_hist: HashMap<usize, usize> = HashMap::new();
        let mut unranked = 0usize;
        let mut second_diff_domain = 0usize;
        let mut margin_sum = 0.0f64;
        let mut margin_n = 0usize;
        let mut n_results_total = 0usize;

        for (text, true_d, true_m) in &items {
            let emb = embedder.embed(text);
            let results = classifier.classify(&emb);
            n_results_total += results.len();
            if results.len() >= 2 && results[0].domain != results[1].domain {
                second_diff_domain += 1;
            }
            if results.len() >= 2 {
                margin_sum += (results[0].score - results[1].score) as f64;
                margin_n += 1;
            }
            match results
                .iter()
                .position(|r| &r.domain == true_d && &r.mode == true_m)
            {
                Some(p) => *rank_hist.entry(p + 1).or_default() += 1,
                None => unranked += 1,
            }
        }

        let n = items.len() as f64;
        let at = |k: usize| -> f64 {
            (1..=k).map(|r| *rank_hist.get(&r).unwrap_or(&0)).sum::<usize>() as f64 / n
        };

        println!("=== Where does an entry's OWN authored cell land in the ranking? ===\n");
        println!("  cells returned per query (mean): {:.1}", n_results_total as f64 / n);
        println!("  top-1 (the argmax alone)  : {:.3}", at(1));
        println!("  top-2                     : {:.3}", at(2));
        println!("  top-3                     : {:.3}", at(3));
        println!("  top-5                     : {:.3}", at(5));
        println!("  top-10                    : {:.3}", at(10));
        println!("  never ranked at all       : {unranked}");

        let gain_2 = at(2) - at(1);
        let gain_5 = at(5) - at(1);
        println!("\n  information the argmax DISCARDS:");
        println!("    +{:.3} recovered by looking at rank 2 as well ({:.0}% relative)", gain_2, 100.0 * gain_2 / at(1).max(1e-9));
        println!("    +{:.3} recovered by looking at ranks 2-5      ({:.0}% relative)", gain_5, 100.0 * gain_5 / at(1).max(1e-9));

        println!("\n=== Cross-cutting evidence (Iteration 8's claim, at scale) ===\n");
        println!(
            "  entries whose 2nd-ranked cell is in a DIFFERENT domain: {}/{} ({:.1}%)",
            second_diff_domain,
            items.len(),
            100.0 * second_diff_domain as f64 / n
        );
        if margin_n > 0 {
            println!("  mean score margin between rank 1 and rank 2: {:.4}", margin_sum / margin_n as f64);
        }

        println!("\n=== Verdict for roadmap item 25 ===\n");
        println!("  classify() already returns the full ranked list, so 'surface per-cell");
        println!("  scores' needs no code change — the report was wrong about the current");
        println!("  behavior. The measured question is whether that list is worth reading:");
        if gain_5 > 0.05 {
            println!("    YES — ranks 2-5 recover {:.1} points of an entry's own cell that the", 100.0 * gain_5);
            println!("    argmax alone misses. The single-valued OntologyEntry storage is");
            println!("    discarding information the classifier already computes.");
        } else {
            println!("    MARGINAL — ranks 2-5 add only {:.1} points over the argmax, so", 100.0 * gain_5);
            println!("    reading past rank 1 buys little on this corpus.");
        }
        println!(
            "\n(Caveat: entries are classified against an ontology BUILT FROM THEM, so top-1\n here is an upper bound, not held-out accuracy — the entry's own text is a member\n of its own cell. That inflation applies equally to every rank, so the GAIN figures\n above, which are differences, are the trustworthy part; the absolute levels are not.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
