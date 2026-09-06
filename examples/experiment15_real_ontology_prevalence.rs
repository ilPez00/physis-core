//! Experiment 15 — Next Experiment 2 (FINAL_REPORT.md): measure how
//! common cross-cutting concepts actually are in physis-core's REAL
//! 730-entry ontology, not another hand-built toy corpus. Every dataset
//! used in Iterations 8-14 (Dataset D, the Vehicle ontology) was
//! deliberately constructed to contain cross-cutting concepts — none of
//! them answer whether real production content actually needs this
//! feature, or whether multi-membership is a toy-corpus phenomenon.
//!
//! This is a MEASUREMENT, not a calibration exercise: physis-core's real
//! ontology has no ground-truth "this entry genuinely belongs to two
//! domains" labels, so there is nothing to compute precision/recall
//! against. What CAN be measured, honestly: for every real entry, how
//! close is its similarity to its second-best domain/cell relative to its
//! best one, and does the resulting "looks arguably cross-cutting" rate
//! look like a real, qualitatively sane signal (spot-checked by reading
//! actual entry text, per this track's standing discipline) or noise.
//!
//! Also directly tests whether Iteration 12's synthetic finding
//! (calibration/overlap gets worse at finer granularity) replicates on
//! real data: the same margin analysis is run at both the 5-DOMAIN level
//! (coarse) and the CELL level (domain/mode, finer — Iteration 4's
//! well-populated subset, cells with >=5 entries).
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment15_real_ontology_prevalence

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

/// For each item, similarity to its OWN true class's centroid minus the
/// best similarity among all OTHER classes' centroids. Small/negative
/// margin = a plausible secondary-membership candidate (proxy only — no
/// ground truth exists to confirm this is genuine cross-cutting rather
/// than noise or a mislabeled/ambiguous entry).
fn compute_margins(embeddings: &[Vec<f32>], labels: &[usize], n_classes: usize) -> Vec<(f32, usize, usize)> {
    let centroids: Vec<Vec<f32>> = (0..n_classes).map(|c| {
        let members: Vec<&Vec<f32>> = (0..embeddings.len()).filter(|&i| labels[i] == c).map(|i| &embeddings[i]).collect();
        centroid(&members)
    }).collect();
    embeddings.iter().enumerate().map(|(i, e)| {
        let own = labels[i];
        let own_sim = cosine_sim(e, &centroids[own]);
        let (second_best_class, second_best_sim) = (0..n_classes)
            .filter(|&c| c != own)
            .map(|c| (c, cosine_sim(e, &centroids[c])))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        (own_sim - second_best_sim, i, second_best_class)
    }).collect()
}

fn main() {
    println!("Experiment 15: real cross-cutting prevalence in physis-core's actual ontology\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder};
        let mut embedder = None;
        for dir in ["models", "../models"] {
            let p = std::path::Path::new(dir);
            if !(p.join("model.onnx").exists() || p.join("onnx/model.onnx").exists()) { continue; }
            let cfg = OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), ..OnnxConfig::default() };
            let e = OnnxEmbedder::with_config(&cfg);
            if e.is_available() {
                println!("Using real semantic embedder: {dir}/model.onnx\n");
                embedder = Some(e);
                break;
            }
        }
        let embedder = match embedder { Some(e) => e, None => { println!("WARNING: no ONNX model — aborting."); return; } };

        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        let mut domains = Vec::new();
        let mut cells = Vec::new();
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut text = def.name.clone();
            for hint in &def.hints { text.push(' '); text.push_str(hint); }
            texts.push(text);
            domains.push(d.clone());
            cells.push(format!("{d}/{m}"));
        }
        println!("Loaded {} real ontology entries. Embedding...", texts.len());
        let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| embedder.embed(t)).collect();
        println!("Done.\n");

        // ── DOMAIN level (5 coarse classes) ──
        let mut domain_ids: HashMap<String, usize> = HashMap::new();
        let domain_labels: Vec<usize> = domains.iter().map(|d| { let next = domain_ids.len(); *domain_ids.entry(d.clone()).or_insert(next) }).collect();
        let mut domain_names = vec![String::new(); domain_ids.len()];
        for (name, &id) in &domain_ids { domain_names[id] = name.clone(); }

        println!("=== DOMAIN level (5 coarse classes, n={}) ===", embeddings.len());
        let domain_margins = compute_margins(&embeddings, &domain_labels, domain_ids.len());
        print_prevalence_curve(&domain_margins);
        println!("\nTop 15 most-arguably-cross-cutting entries at DOMAIN level (smallest margin):");
        let mut sorted = domain_margins.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for (margin, i, second_class) in sorted.iter().take(15) {
            println!(
                "  margin={margin:+.3}  '{}'  [true={}, arguable-second={}]",
                texts[*i].chars().take(80).collect::<String>(),
                domain_names[domain_labels[*i]],
                domain_names[*second_class]
            );
        }

        // ── CELL level (domain/mode, finer — Iteration 4's well-populated subset) ──
        let mut cell_counts: HashMap<String, usize> = HashMap::new();
        for c in &cells { *cell_counts.entry(c.clone()).or_default() += 1; }
        let keep: Vec<usize> = (0..cells.len()).filter(|&i| cell_counts[&cells[i]] >= 5).collect();
        let cell_embeddings: Vec<Vec<f32>> = keep.iter().map(|&i| embeddings[i].clone()).collect();
        let cell_texts: Vec<&String> = keep.iter().map(|&i| &texts[i]).collect();
        let mut cell_ids: HashMap<String, usize> = HashMap::new();
        let cell_labels: Vec<usize> = keep.iter().map(|&i| { let next = cell_ids.len(); *cell_ids.entry(cells[i].clone()).or_insert(next) }).collect();
        let mut cell_names = vec![String::new(); cell_ids.len()];
        for (name, &id) in &cell_ids { cell_names[id] = name.clone(); }

        println!("\n\n=== CELL level (domain/mode, finer granularity, n={}, {} cells with >=5 entries) ===", cell_embeddings.len(), cell_ids.len());
        let cell_margins = compute_margins(&cell_embeddings, &cell_labels, cell_ids.len());
        print_prevalence_curve(&cell_margins);
        println!("\nTop 15 most-arguably-cross-cutting entries at CELL level (smallest margin):");
        let mut sorted_cell = cell_margins.clone();
        sorted_cell.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for (margin, i, second_class) in sorted_cell.iter().take(15) {
            println!(
                "  margin={margin:+.3}  '{}'  [true={}, arguable-second={}]",
                cell_texts[*i].chars().take(80).collect::<String>(),
                cell_names[cell_labels[*i]],
                cell_names[*second_class]
            );
        }

        println!("\n=== Granularity comparison (does real data replicate Iteration 12's synthetic finding?) ===");
        let domain_frac_below = |d: f32| domain_margins.iter().filter(|(m, ..)| *m < d).count() as f32 / domain_margins.len() as f32;
        let cell_frac_below = |d: f32| cell_margins.iter().filter(|(m, ..)| *m < d).count() as f32 / cell_margins.len() as f32;
        println!("{:<10} {:>20} {:>20}", "delta", "DOMAIN frac<delta", "CELL frac<delta");
        for d in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10] {
            println!("{d:<10.2} {:>20.3} {:>20.3}", domain_frac_below(d), cell_frac_below(d));
        }
        println!("\n(if CELL fractions are consistently higher than DOMAIN fractions at the same delta, that replicates Iteration 12's synthetic finding on real production data: finer granularity produces more apparent overlap, independent of any specific toy corpus design choice.)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}

fn print_prevalence_curve(margins: &[(f32, usize, usize)]) {
    println!("delta -> fraction of entries with margin < delta (a plausible secondary-membership proxy):");
    for delta in [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20] {
        let frac = margins.iter().filter(|(m, ..)| *m < delta).count() as f32 / margins.len() as f32;
        println!("  delta={delta:.2}: {frac:.3}");
    }
    let mean_margin: f32 = margins.iter().map(|(m, ..)| m).sum::<f32>() / margins.len() as f32;
    let min_margin = margins.iter().map(|(m, ..)| *m).fold(f32::INFINITY, f32::min);
    println!("mean margin: {mean_margin:.3}, min margin: {min_margin:.3} (negative = closer to a DIFFERENT class's centroid than its own true one)");
}
