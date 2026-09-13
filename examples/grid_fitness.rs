//! D4 — the label-permuted null that gates the rest of problem 3.
//!
//! Loads the real ontology, scores every cell's leave-one-out self-recovery,
//! and compares each cell against the same statistic under 200 label
//! permutations that preserve every cell's size. See `src/grid_fitness.rs`.
fn main() {
    let (embedder, kind) = physis_core::embed::select(384);
    if kind == "random-projection" {
        eprintln!("warning: random-projection is a lexical hash — this is a floor, not a claim.");
    }
    let ontology = physis_core::ontology::OntologyLoader::load_all();
    let mut entries: Vec<(String, Vec<f32>)> = Vec::new();
    for def in ontology.classification_domains() {
        let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
        let mut text = def.name.clone();
        for hint in &def.hints {
            text.push(' ');
            text.push_str(hint);
        }
        entries.push((format!("{d}/{m}"), embedder.embed(&text)));
    }
    println!("{} ontology entries loaded\n", entries.len());
    let run = physis_core::grid_fitness::run(&entries, 7, kind);
    print!("{}", run.render());
    let path = std::path::Path::new("benchmarks/results/grid-fitness.json");
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    std::fs::write(path, serde_json::to_vec_pretty(&run).unwrap()).unwrap();
    println!("\nartifact -> {}", path.display());
}
