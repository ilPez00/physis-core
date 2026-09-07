//! Demonstrates `physis_core::coverage` on the real ontology — the production
//! form of Iteration 33's mechanical disposer.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example coverage_demo

fn main() {
    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::classify::{Cell, CellClassifier};
        use physis_core::coverage::{rank_candidates, uncovered};
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        use physis_core::models::Facets;
        use physis_core::ontology::OntologyLoader;

        let Some(dir) = ["models", "../models"]
            .iter()
            .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        else { println!("no model"); return };
        let embedder = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 384, model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean, ..OnnxConfig::default()
        });
        if !embedder.is_available() { println!("embedder unavailable"); return }

        let ontology = OntologyLoader::load_all();
        let defs: Vec<_> = ontology
            .classification_domains()
            .filter(|d| d.domain.is_some() && d.mode.is_some())
            .collect();
        let text_of = |d: &physis_core::models::DomainDef| {
            let mut t = d.name.clone();
            for h in &d.hints {
                t.push(' ');
                t.push_str(h);
            }
            t
        };

        // Hold out one entry in five. Without this the demo is degenerate: the
        // records ARE the ontology entries, every record scores 1.000 against
        // itself, nothing is uncovered, and every candidate scores zero — the
        // case `nothing_uncovered_means_no_candidate_can_gain` documents.
        let mut kept: std::collections::HashMap<(String, String), Vec<(String, Vec<f32>)>> =
            std::collections::HashMap::new();
        let mut records: Vec<(String, Vec<f32>)> = Vec::new();
        for (i, d) in defs.iter().enumerate() {
            let e = embedder.embed(&text_of(d));
            if i % 5 == 0 {
                records.push((d.name.clone(), e));
            } else {
                kept.entry((d.domain.clone().unwrap(), d.mode.clone().unwrap()))
                    .or_default()
                    .push((d.name.clone(), e));
            }
        }
        let mut keys: Vec<&(String, String)> = kept.keys().collect();
        keys.sort();
        let classifier = CellClassifier::from_cells(
            keys.iter()
                .map(|k| {
                    let ms = &kept[*k];
                    Cell {
                        domain: k.0.clone(),
                        mode: k.1.clone(),
                        entries: ms.iter().map(|(n, _)| n.clone()).collect(),
                        facets: ms.iter().map(|_| Facets::default()).collect(),
                        embeddings: ms.iter().map(|(_, e)| e.clone()).collect(),
                    }
                })
                .collect(),
        );

        // Threshold at the median live score, so about half start uncovered.
        let mut live: Vec<f32> = records
            .iter()
            .map(|(_, e)| classifier.best_entry_sim(e).map(|(s, _, _)| s).unwrap_or(0.0))
            .collect();
        live.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold = live[live.len() / 2];

        let gaps = uncovered(&classifier, &records, threshold);
        println!("{} records, {} uncovered at threshold {:.3}\n", records.len(), gaps.len(), threshold);

        // Two candidates: a duplicate of an existing entry, and a synthetic
        // cell sitting exactly on an uncovered record.
        let dup_text = text_of(defs[1]);
        let target = records.iter().find(|(id, _)| gaps.contains(id)).unwrap();
        let mk = |name: &str, v: Vec<f32>| Cell {
            domain: "CANDIDATE".into(), mode: name.into(),
            entries: vec![name.into()], facets: vec![Facets::default()], embeddings: vec![v],
        };
        let candidates = vec![
            mk("duplicate-of-existing", embedder.embed(&dup_text)),
            mk("on-an-uncovered-record", target.1.clone()),
        ];

        println!("candidate                     newly covered");
        for (i, g) in rank_candidates(&classifier, &candidates, &records, threshold) {
            println!("  {:<28} {:>3}   useful={}", candidates[i].mode, g.count(), g.is_useful());
        }
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("built without embed-onnx");
}
