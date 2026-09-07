// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 18 — Next Experiment 9: does a different embedding model
//! show meaningfully better raw separability on physis-core's REAL
//! 5-domain ontology than all-MiniLM-L6-v2 (which Iterations 15 and 17
//! showed achieves only 40-46% 1-NN accuracy / ~59.5% centroid-argmax
//! accuracy — barely double the 20% chance rate)?
//!
//! BAAI/bge-base-en-v1.5 (768-d BertModel) weights were pulled from
//! HuggingFace for this test (were previously absent — only
//! config/tokenizer existed locally). Per the user's instruction to test
//! multiple embedding formats: BGE is conventionally used with
//! [CLS]-token pooling (unlike MiniLM's mean pooling), and physis-core's
//! `OnnxEmbedder` already supports both strategies as a config option —
//! this experiment tests BOTH pooling formats empirically rather than
//! assuming CLS is correct, since guessing and reporting an unverified
//! number would repeat this whole track's core mistake.
//!
//! Three embedder configurations compared on the identical task:
//!   - MiniLM (384-d, mean pooling) — the baseline used throughout
//!     Iterations 1-17
//!   - BGE-base (768-d, mean pooling)
//!   - BGE-base (768-d, CLS pooling) — BGE's conventional format
//!
//! Metrics, all reused verbatim from Iteration 15's methodology so
//! numbers are directly comparable: leave-one-out 1-NN accuracy,
//! centroid-argmax accuracy, mean margin, negative-margin fraction
//! (= 1 - centroid-argmax accuracy, reported for continuity with
//! Iteration 15's framing).
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment18_multi_embedder_comparison

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::HashMap;

fn centroid(embeddings: &[&Vec<f32>]) -> Vec<f32> {
    let dim = embeddings[0].len();
    let mut sum = vec![0.0f32; dim];
    for e in embeddings { for d in 0..dim { sum[d] += e[d]; } }
    let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    sum.iter().map(|x| x / norm).collect()
}

struct Report {
    name: &'static str,
    dim: usize,
    loo_1nn_acc: f32,
    centroid_argmax_acc: f32,
    mean_margin: f32,
    negative_margin_frac: f32,
}

fn evaluate(name: &'static str, embeddings: &[Vec<f32>], labels: &[usize], n_classes: usize) -> Report {
    let n = embeddings.len();
    let dim = embeddings[0].len();

    // Leave-one-out 1-NN: for each item, find its nearest OTHER item, check label match.
    let mut loo_hits = 0usize;
    for i in 0..n {
        let (mut best_j, mut best_sim) = (0usize, f32::NEG_INFINITY);
        for j in 0..n {
            if i == j { continue; }
            let s = cosine_sim(&embeddings[i], &embeddings[j]);
            if s > best_sim { best_sim = s; best_j = j; }
        }
        if labels[best_j] == labels[i] { loo_hits += 1; }
    }
    let loo_1nn_acc = loo_hits as f32 / n as f32;

    // Centroid-argmax + margin (Iteration 15's methodology, verbatim).
    let centroids: Vec<Vec<f32>> = (0..n_classes).map(|c| {
        let members: Vec<&Vec<f32>> = (0..n).filter(|&i| labels[i] == c).map(|i| &embeddings[i]).collect();
        centroid(&members)
    }).collect();
    let mut argmax_hits = 0usize;
    let mut total_margin = 0.0f32;
    for i in 0..n {
        let own = labels[i];
        let own_sim = cosine_sim(&embeddings[i], &centroids[own]);
        let (best_class, best_sim) = (0..n_classes).map(|c| (c, cosine_sim(&embeddings[i], &centroids[c])))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        if best_class == own { argmax_hits += 1; }
        let second_best = (0..n_classes).filter(|&c| c != own)
            .map(|c| cosine_sim(&embeddings[i], &centroids[c]))
            .fold(f32::NEG_INFINITY, f32::max);
        total_margin += own_sim - second_best;
        let _ = best_sim;
    }

    Report {
        name, dim,
        loo_1nn_acc,
        centroid_argmax_acc: argmax_hits as f32 / n as f32,
        mean_margin: total_margin / n as f32,
        negative_margin_frac: (0..n).filter(|&i| {
            let own = labels[i];
            let own_sim = cosine_sim(&embeddings[i], &centroids[own]);
            let second_best = (0..n_classes).filter(|&c| c != own).map(|c| cosine_sim(&embeddings[i], &centroids[c])).fold(f32::NEG_INFINITY, f32::max);
            own_sim - second_best < 0.0
        }).count() as f32 / n as f32,
    }
}

fn print_report(r: &Report) {
    println!(
        "{:<28} dim={:<5} LOO-1NN={:.3}  centroid-argmax={:.3}  mean-margin={:+.3}  neg-margin-frac={:.3}",
        r.name, r.dim, r.loo_1nn_acc, r.centroid_argmax_acc, r.mean_margin, r.negative_margin_frac
    );
}

fn main() {
    println!("Experiment 18: multi-embedder comparison on physis-core's real 5-domain ontology\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        let mut domains = Vec::new();
        for def in ontology.classification_domains() {
            let (Some(d), Some(_m)) = (&def.domain, &def.mode) else { continue };
            let mut text = def.name.clone();
            for hint in &def.hints { text.push(' '); text.push_str(hint); }
            texts.push(text);
            domains.push(d.clone());
        }
        let mut domain_ids: HashMap<String, usize> = HashMap::new();
        let labels: Vec<usize> = domains.iter().map(|d| { let next = domain_ids.len(); *domain_ids.entry(d.clone()).or_insert(next) }).collect();
        let n_classes = domain_ids.len();
        println!("Loaded {} real ontology entries, {} domain classes.\n", texts.len(), n_classes);

        // ── MiniLM (384-d, mean pooling) — the baseline used throughout Iterations 1-17 ──
        let minilm_dir = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists());
        let minilm = minilm_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() }));

        // ── BGE-base (768-d), tested with BOTH pooling formats ──
        let bge_dir = ["models/bge-base-en-v1.5", "../models/bge-base-en-v1.5"].iter().find(|d| std::path::Path::new(d).join("onnx/model.onnx").exists());
        let bge_mean = bge_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 768, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() }));
        let bge_cls = bge_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 768, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Cls, ..OnnxConfig::default() }));

        println!("MiniLM available: {}", minilm.as_ref().map(|e| e.is_available()).unwrap_or(false));
        println!("BGE available: {}\n", bge_mean.as_ref().map(|e| e.is_available()).unwrap_or(false));

        let mut reports = Vec::new();

        if let Some(embedder) = &minilm {
            if embedder.is_available() {
                let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| embedder.embed(t)).collect();
                reports.push(evaluate("MiniLM (mean)", &embeddings, &labels, n_classes));
            }
        }
        if let Some(embedder) = &bge_mean {
            if embedder.is_available() {
                let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| embedder.embed(t)).collect();
                reports.push(evaluate("BGE-base (mean)", &embeddings, &labels, n_classes));
            }
        }
        if let Some(embedder) = &bge_cls {
            if embedder.is_available() {
                let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| embedder.embed(t)).collect();
                reports.push(evaluate("BGE-base (CLS)", &embeddings, &labels, n_classes));
            }
        }

        println!("=== Comparison (Iteration 15's exact methodology, applied to 3 embedder configs) ===");
        for r in &reports { print_report(r); }

        if reports.len() >= 2 {
            let minilm_r = reports.iter().find(|r| r.name.starts_with("MiniLM"));
            let best_bge = reports.iter().filter(|r| r.name.starts_with("BGE")).max_by(|a, b| a.loo_1nn_acc.partial_cmp(&b.loo_1nn_acc).unwrap());
            if let (Some(m), Some(b)) = (minilm_r, best_bge) {
                println!(
                    "\nBest BGE config ({}) vs MiniLM: LOO-1NN {:.3} vs {:.3} (delta {:+.3}), centroid-argmax {:.3} vs {:.3} (delta {:+.3})",
                    b.name, b.loo_1nn_acc, m.loo_1nn_acc, b.loo_1nn_acc - m.loo_1nn_acc,
                    b.centroid_argmax_acc, m.centroid_argmax_acc, b.centroid_argmax_acc - m.centroid_argmax_acc
                );
                println!(
                    "\nVerdict: {}",
                    if b.loo_1nn_acc - m.loo_1nn_acc > 0.10 {
                        "BGE shows a SUBSTANTIAL improvement — supports the hypothesis that MiniLM's embedding quality is a real bottleneck, not just the taxonomy being inherently ambiguous."
                    } else if b.loo_1nn_acc - m.loo_1nn_acc > 0.02 {
                        "BGE shows a SMALL improvement — some representation-quality effect, but the domains remain hard to separate even with a stronger embedder, suggesting genuine taxonomy overlap is at least part of the story."
                    } else {
                        "BGE performs SIMILARLY to MiniLM — evidence AGAINST embedding quality being the primary bottleneck; physis-core's domain taxonomy itself likely has genuine, model-independent overlap (consistent with the qualitative spot-check in Iteration 15 reading several borderline entries as plausibly dual-domain)."
                    }
                );
            }
            if reports.iter().any(|r| r.name == "BGE-base (mean)") && reports.iter().any(|r| r.name == "BGE-base (CLS)") {
                let mean_r = reports.iter().find(|r| r.name == "BGE-base (mean)").unwrap();
                let cls_r = reports.iter().find(|r| r.name == "BGE-base (CLS)").unwrap();
                println!(
                    "\nPooling format check: mean={:.3} vs CLS={:.3} LOO-1NN — {} pooling is empirically better for BGE on this task (do not assume CLS is correct just because it's BGE's conventional default).",
                    mean_r.loo_1nn_acc, cls_r.loo_1nn_acc,
                    if mean_r.loo_1nn_acc >= cls_r.loo_1nn_acc { "MEAN" } else { "CLS" }
                );
            }
        }
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
