//! Refiling fit — does an assignment put entries in cells whose anchor
//! actually describes them?
//!
//! The Gate 0 question, made numeric and independent of the judgement that
//! produced the filing. `classify` embeds `name + hints`, so "does this entry
//! belong in this cell" has a measurable proxy: how close the entry's own text
//! sits to its cell's anchor text, and whether that anchor is the *nearest* of
//! the seventy.
//!
//! Neither the old nor the new filing was made with embeddings — both are
//! judgements over text — so the embedder is an independent instrument here.
//!
//! Run it as a 2x2 (old/new anchors x old/new filing) so the anchor rewrite
//! and the re-filing can be credited separately. Scoring both filings against
//! the *same* anchors is what makes the comparison fair; scoring each against
//! its own would be rigged.
//!
//!   cargo run -p physis-core --features embed-onnx --release \
//!     --example refiling_fit -- <anchors.json> <assignments.tsv>
//!
//! assignments.tsv: `name<TAB>hints-joined-by-space<TAB>DOMAIN/MODE`

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
    use std::collections::HashMap;

    let mut args = std::env::args().skip(1);
    let anchors_path = args.next().expect("usage: refiling_fit <anchors.json> <assignments.tsv>");
    let assign_path = args.next().expect("usage: refiling_fit <anchors.json> <assignments.tsv>");

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

    // ---- anchors ----
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&anchors_path).expect("anchors")).expect("parse");
    let mut cell_name = Vec::new();
    let mut cell_emb = Vec::new();
    for e in doc["domains"].as_array().expect("domains") {
        let mut t = e["name"].as_str().unwrap_or_default().to_string();
        for h in e["hints"].as_array().into_iter().flatten() {
            t.push(' ');
            t.push_str(h.as_str().unwrap_or_default());
        }
        cell_name.push(format!(
            "{}/{}",
            e["domain"].as_str().unwrap_or("?"),
            e["mode"].as_str().unwrap_or("?")
        ));
        cell_emb.push(normalize(&embedder.embed(&t)));
    }
    let index: HashMap<&str, usize> =
        cell_name.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();

    // ---- entries ----
    let raw = std::fs::read_to_string(&assign_path).expect("assignments");
    let mut names = Vec::new();
    let mut assigned = Vec::new();
    let mut emb = Vec::new();
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let mut f = line.split('\t');
        let name = f.next().unwrap_or_default();
        let hints = f.next().unwrap_or_default();
        let cell = f.next().unwrap_or_default();
        let Some(&ci) = index.get(cell) else {
            println!("WARNING: unknown cell {cell} — skipped");
            continue;
        };
        names.push(name.to_string());
        assigned.push(ci);
        emb.push(normalize(&embedder.embed(&format!("{name} {hints}"))));
    }
    let n = emb.len();
    println!("{anchors_path}\n{assign_path}\n{n} entries, {} anchors\n", cell_emb.len());

    // ---- fit of each entry to the cell it was put in ----
    let mut fit_sum = 0.0f64;
    let mut rank_sum = 0.0f64;
    let (mut top1, mut top3, mut top5, mut bottom_half) = (0usize, 0usize, 0usize, 0usize);
    let mut worst: Vec<(f32, usize, usize)> = Vec::new();
    for i in 0..n {
        let sims: Vec<f32> = cell_emb.iter().map(|c| cos(&emb[i], c)).collect();
        let mine = sims[assigned[i]];
        // Rank of the assigned cell among all 70, 1 = best. Ties count against.
        let rank = 1 + sims.iter().filter(|&&s| s > mine).count();
        fit_sum += mine as f64;
        rank_sum += rank as f64;
        if rank == 1 { top1 += 1; }
        if rank <= 3 { top3 += 1; }
        if rank <= 5 { top5 += 1; }
        if rank > cell_emb.len() / 2 { bottom_half += 1; }
        let best = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        worst.push((mine - best, i, sims.iter().position(|s| *s == best).unwrap()));
    }
    let p = |k: usize| 100.0 * k as f64 / n as f64;
    println!("fit of each entry to the cell it was filed in:");
    println!("  mean cosine to its own anchor      {:.4}", fit_sum / n as f64);
    println!("  its anchor is the NEAREST of 70    {top1:>4}  {:.1}%", p(top1));
    println!("  within the nearest 3               {top3:>4}  {:.1}%", p(top3));
    println!("  within the nearest 5               {top5:>4}  {:.1}%", p(top5));
    println!("  in the WORSE HALF of all 70 cells  {bottom_half:>4}  {:.1}%   <- the misfiling proxy", p(bottom_half));
    println!("  mean rank of its own anchor        {:.1} of 70", rank_sum / n as f64);

    worst.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("\nten worst-fitting entries (filed cell -> the anchor it is actually nearest):");
    for (gap, i, best) in worst.iter().take(10) {
        println!("  {:>7.3}  {:<34} {:<20} -> {}", gap, names[*i], cell_name[assigned[*i]], cell_name[*best]);
    }
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
