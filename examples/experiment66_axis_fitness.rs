//! E66 — is the grid's domain axis structural, or is it just written down?
//!
//! The design question, put by the author: *why 70 and not n? The 14 modes make
//! sense — tool use, method, workflow, the operative half. Domains dictate
//! content and should be adaptable.*
//!
//! That is testable with machinery already here. D4's fitness (leave-one-out
//! self-recovery against a size-preserving label permutation) does not care what
//! the labels mean, so run it three times on the same entries and the same
//! embeddings, relabelled:
//!
//!   full    DOMAIN/MODE, 70 classes  (D4's original run)
//!   domain  DOMAIN only, 5 classes
//!   mode    MODE only, 14 classes
//!
//! An axis that carves reality shows self-recovery well above its own null. An
//! axis that is a convention shows the null.
//!
//! Run: PHYSIS_MODEL_DIR=../models/bge-base-en-v1.5 PHYSIS_MODEL_DIM=768 \
//!      cargo run -p physis-core --release --features embed-onnx \
//!          --example experiment66_axis_fitness
//! Artifact: benchmarks/results/worldstate-e66-axes.json
fn main() {
    let (embedder, kind) = physis_core::embed::select(768);
    if kind == "random-projection" {
        eprintln!("WARNING: lexical hash — an axis result on this is not an axis result.");
    }
    let ontology = physis_core::ontology::OntologyLoader::load_all();
    let mut full: Vec<(String, Vec<f32>)> = Vec::new();
    let mut dom: Vec<(String, Vec<f32>)> = Vec::new();
    let mut mode: Vec<(String, Vec<f32>)> = Vec::new();
    for def in ontology.classification_domains() {
        let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
        let mut text = def.name.clone();
        for hint in &def.hints {
            text.push(' ');
            text.push_str(hint);
        }
        let v = embedder.embed(&text);
        full.push((format!("{d}/{m}"), v.clone()));
        dom.push((d.clone(), v.clone()));
        mode.push((m.clone(), v));
    }
    println!("{} ontology entries, embedder {kind}\n", full.len());

    let mut rows = Vec::new();
    println!("axis     classes  fitness  null    ratio  above-null  failing");
    for (label, entries) in [("full", &full), ("domain", &dom), ("mode", &mode)] {
        let run = physis_core::grid_fitness::run(entries, 7, kind);
        let ratio = if run.overall_null_mean > 0.0 {
            run.overall_fitness / run.overall_null_mean
        } else {
            0.0
        };
        println!(
            "{label:<8} {:<8} {:.3}    {:.3}   {ratio:.1}x   {:<11} {}",
            run.cells, run.overall_fitness, run.overall_null_mean, run.cells_above_null,
            run.cells_failing
        );
        rows.push(serde_json::json!({
            "axis": label,
            "classes": run.cells,
            "fitness": run.overall_fitness,
            "null": run.overall_null_mean,
            "ratio": ratio,
            "classes_above_null": run.cells_above_null,
            "classes_failing": run.cells_failing,
            "per_class": run.per_cell.iter().map(|c| serde_json::json!({
                "class": c.cell, "n": c.entries, "fitness": c.fitness, "null": c.null_mean
            })).collect::<Vec<_>>(),
        }));
    }

    let out = std::path::Path::new("benchmarks/results");
    let _ = std::fs::create_dir_all(out);
    let path = out.join("worldstate-e66-axes.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
    println!("\nartifact -> {}", path.display());
}
