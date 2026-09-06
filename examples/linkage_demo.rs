//! Demonstrates `physis_core::linkage` on the real ontology — the production
//! form of Iteration 21's fixed-cell linkage layer.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example linkage_demo

fn main() {
    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::classify::CellClassifier;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        use physis_core::linkage::LinkageGraph;
        use physis_core::ontology::OntologyLoader;

        let Some(dir) = ["models", "../models"]
            .iter()
            .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        else {
            println!("no model");
            return;
        };
        let embedder = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 384,
            model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if !embedder.is_available() {
            println!("embedder unavailable");
            return;
        }

        let ontology = OntologyLoader::load_all();
        let classifier = CellClassifier::build(&ontology, &embedder);
        let texts: Vec<String> = ontology
            .classification_domains()
            .filter(|d| d.domain.is_some() && d.mode.is_some())
            .map(|d| {
                let mut t = d.name.clone();
                for h in &d.hints {
                    t.push(' ');
                    t.push_str(h);
                }
                t
            })
            .collect();

        let graph = LinkageGraph::build(&classifier, &embedder, &texts);
        println!("{} texts -> {} bridged cell pairs\n", texts.len(), graph.len());
        println!("Strongest links:");
        for l in graph.strongest(10) {
            println!(
                "  {}/{} <-> {}/{}   {} bridges",
                l.a.0, l.a.1, l.b.0, l.b.1, l.bridge_count
            );
        }
        println!("\nStrongest CROSS-DOMAIN links:");
        for l in graph.cross_domain().take(10) {
            println!(
                "  {}/{} <-> {}/{}   {} bridges",
                l.a.0, l.a.1, l.b.0, l.b.1, l.bridge_count
            );
        }
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("built without embed-onnx");
}
