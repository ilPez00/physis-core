//! Anchor separation — how distinguishable are the 70 cell anchors?
//!
//! The anchors are the classifier's centroids: `classify` embeds `name + hints`
//! and nothing else. Rewriting them to span human, machine and organisational
//! registers removes a separation the old set got for free — HEAL spoke only
//! about bodies, CONSTRUCT only about building sites, so cells were partly told
//! apart by vocabulary rather than by meaning. That is a cheat when the corpus
//! is mostly neither, but dropping it can only make the anchors *more* alike.
//!
//! This measures how much. Run it on the old and new anchor files and compare:
//! mean off-diagonal cosine (how alike anchors are in general), the confusable
//! tail (pairs above 0.90), and per-anchor nearest neighbour — the pairs a
//! record could plausibly land in either of.
//!
//!   cargo run -p physis-core --features embed-onnx --release \
//!     --example anchor_separation -- config/mode_anchors_ontology.json

fn main() {
    #[cfg(not(feature = "embed-onnx"))]
    println!("built without embed-onnx");
    #[cfg(feature = "embed-onnx")]
    run();
}

#[cfg(feature = "embed-onnx")]
fn run() {
    use physis_core::embed::VectorEmbed;
    use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

    let path = std::env::args().nth(1).unwrap_or_else(|| "config/mode_anchors_ontology.json".into());
    let raw = std::fs::read_to_string(&path).expect("anchor file");
    let doc: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    let entries = doc["domains"].as_array().expect("domains array");

    let dirs = ["models", "../models", "physis-core/models"];
    let model_dir = dirs
        .iter()
        .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        .expect("models/model.onnx not found");
    let embedder = OnnxEmbedder::with_config(&OnnxConfig {
        model_dir: Some(model_dir.to_string()),
        pooling: PoolingStrategy::Mean,
        ..OnnxConfig::default()
    });
    if !embedder.is_available() {
        println!("WARNING: embedder unavailable — aborting.");
        return;
    }

    let mut cells = Vec::new();
    let mut emb = Vec::new();
    for e in entries {
        let mut t = e["name"].as_str().unwrap_or_default().to_string();
        for h in e["hints"].as_array().into_iter().flatten() {
            t.push(' ');
            t.push_str(h.as_str().unwrap_or_default());
        }
        cells.push(format!(
            "{}/{}",
            e["domain"].as_str().unwrap_or("?"),
            e["mode"].as_str().unwrap_or("?")
        ));
        emb.push(normalize(&embedder.embed(&t)));
    }
    let n = cells.len();
    println!("{path}: {n} anchors\n");

    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((cos(&emb[i], &emb[j]), i, j));
        }
    }
    let mean = pairs.iter().map(|p| p.0 as f64).sum::<f64>() / pairs.len() as f64;
    let mut sorted: Vec<f32> = pairs.iter().map(|p| p.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| sorted[(((sorted.len() - 1) as f64) * f).round() as usize];
    println!("off-diagonal cosine over {} pairs:", pairs.len());
    println!("  mean {mean:.4}   p50 {:.4}   p90 {:.4}   p99 {:.4}   max {:.4}", q(0.5), q(0.9), q(0.99), q(1.0));
    for t in [0.85f32, 0.90, 0.95] {
        println!("  pairs above {t:.2}: {}", pairs.iter().filter(|p| p.0 > t).count());
    }

    // Nearest neighbour per anchor: the cell it is most confusable with.
    let mut nn: Vec<(f32, usize, usize)> = (0..n)
        .map(|i| {
            let (s, j) = (0..n)
                .filter(|&j| j != i)
                .map(|j| (cos(&emb[i], &emb[j]), j))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap();
            (s, i, j)
        })
        .collect();
    let nn_mean = nn.iter().map(|p| p.0 as f64).sum::<f64>() / n as f64;
    println!("\nnearest-neighbour cosine: mean {nn_mean:.4}");
    nn.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\nten most confusable anchors (its nearest rival):");
    for (s, i, j) in nn.iter().take(10) {
        println!("  {:>10.4}  {:<22} -> {}", s, cells[*i], cells[*j]);
    }

    // Does the domain or the mode dominate the geometry? If same-mode pairs are
    // much closer than same-domain pairs the grid is really a mode axis with a
    // decorative second dimension (or vice versa).
    let (mut sd, mut sdn, mut sm, mut smn, mut ot, mut otn) = (0.0f64, 0u32, 0.0f64, 0u32, 0.0f64, 0u32);
    for (s, i, j) in &pairs {
        let (di, mi) = cells[*i].split_once('/').unwrap();
        let (dj, mj) = cells[*j].split_once('/').unwrap();
        if di == dj {
            sd += *s as f64;
            sdn += 1;
        } else if mi == mj {
            sm += *s as f64;
            smn += 1;
        } else {
            ot += *s as f64;
            otn += 1;
        }
    }
    println!("\nstructure of the similarity:");
    println!("  same domain, different mode: {:.4}  (n={sdn})", sd / sdn as f64);
    println!("  same mode, different domain: {:.4}  (n={smn})", sm / smn as f64);
    println!("  neither shared:              {:.4}  (n={otn})", ot / otn as f64);
}

#[cfg(feature = "embed-onnx")]
fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(feature = "embed-onnx")]
fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 { v.iter().map(|x| x / n).collect() } else { v.to_vec() }
}
