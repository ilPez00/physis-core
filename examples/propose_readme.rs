//! The `Proposing a Filing` example from README.md, kept compilable so the
//! documented snippet cannot rot. `cargo run --example propose_readme`.
use physis_core::embed::RandomProjectionEmbedder;
use physis_core::propose::{embed_entry, Proposer, DEFAULT_HINT_WEIGHT};

fn main() {
    let embedder = RandomProjectionEmbedder::new(256);

    let confirmed = [
        ("HEAL", "REST", "Sleep Hygiene", vec!["circadian".to_string()]),
        ("STUDY", "LEARN", "Spaced Repetition", vec!["recall".to_string()]),
        ("BOND", "CREATE", "Team Charter", vec!["agreement".to_string()]),
    ];

    let proposer = Proposer::from_decisions(confirmed.iter().map(|(d, m, name, hints)| {
        (*d, *m, embed_entry(name, hints, &embedder, DEFAULT_HINT_WEIGHT))
    }));

    let entry = embed_entry("Nap Protocol", &["rest".to_string()], &embedder, DEFAULT_HINT_WEIGHT);
    for p in proposer.propose(&entry, 3) {
        println!("{}  {:.3}", p.cell(), p.score);
    }
}
