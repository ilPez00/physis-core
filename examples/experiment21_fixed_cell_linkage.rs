//! Experiment 21 — the concrete build resulting from Iteration 20's
//! resolution: stop trying to discover new stable fixed points from raw
//! embedding geometry (it failed decisively under four different
//! mechanisms). Use physis-core's EXISTING (domain, mode) cells as the
//! fixed points instead — genuinely fixed by design (hand-authored, not
//! discovered, so immune to every instability found in this whole
//! track) — and build the LINKAGE layer between them: which cells does
//! real data show meaningfully bridging together, and via which items?
//!
//! Method, reusing Iteration 13's validated local/split calibration
//! directly (same-domain "sibling" cells get a tight threshold, since no
//! recall is sacrificed calibrating that tightly; cross-domain cells get
//! a separately-calibrated, looser threshold): for every real entry,
//! compute its calibrated multi-membership set among the well-populated
//! cells (>=5 entries, Iteration 4's filter, 31 cells). The link between
//! two cells is the BRIDGE COUNT — how many real entries multi-belong to
//! both — grounded in actual data, not a synthetic shifting centroid.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment21_fixed_cell_linkage

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

fn main() {
    println!("Experiment 21: linkage layer over physis-core's existing, genuinely fixed cells\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        let minilm_dir = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists());
        let minilm = match minilm_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() })) {
            Some(e) if e.is_available() => e,
            _ => { println!("WARNING: MiniLM not available — aborting."); return; }
        };

        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        let mut cells = Vec::new();
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut text = def.name.clone();
            for hint in &def.hints { text.push(' '); text.push_str(hint); }
            texts.push(text);
            cells.push((d.clone(), m.clone()));
        }
        println!("Loaded {} real entries, embedding...", texts.len());
        let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| minilm.embed(t)).collect();
        println!("Done.\n");

        // Well-populated cells only (Iteration 4's filter: >=5 entries) — a fixed
        // point built from too few examples is a noisy anchor, not a stable one.
        let mut cell_counts: HashMap<(String, String), usize> = HashMap::new();
        for c in &cells { *cell_counts.entry(c.clone()).or_default() += 1; }
        let cell_names: Vec<(String, String)> = {
            let mut v: Vec<(String, String)> = cell_counts.iter().filter(|(_, &c)| c >= 5).map(|(k, _)| k.clone()).collect();
            v.sort();
            v
        };
        let n_cells = cell_names.len();
        let cell_id: HashMap<(String, String), usize> = cell_names.iter().enumerate().map(|(i, c)| (c.clone(), i)).collect();
        println!("{n_cells} well-populated fixed cells (>=5 entries each) out of {} total distinct (domain,mode) pairs seen.\n", cell_counts.len());

        let keep: Vec<usize> = (0..texts.len()).filter(|&i| cell_counts[&cells[i]] >= 5).collect();
        println!("{} of {} entries fall in a well-populated cell.\n", keep.len(), texts.len());

        let cell_centroids: Vec<Vec<f32>> = (0..n_cells).map(|c| {
            let members: Vec<&Vec<f32>> = keep.iter().filter(|&&i| cell_id[&cells[i]] == c).map(|&i| &embeddings[i]).collect();
            centroid(&members)
        }).collect();
        let sims_for = |e: &Vec<f32>| -> Vec<f32> { cell_centroids.iter().map(|c| cosine_sim(e, c)).collect() };
        let keep_sims: HashMap<usize, Vec<f32>> = keep.iter().map(|&i| (i, sims_for(&embeddings[i]))).collect();

        // NOT a threshold search — Iteration 13's delta-calibration needs known-good/
        // known-bad examples to calibrate against, which do not exist for this real
        // dataset (Iterations 15/18/19 established this repeatedly). Attempting it
        // here converged on a degenerate result the first time this file was run:
        // the search found no delta with a low same-domain flag rate and silently
        // fell back to an arbitrary default that flagged 98% of entries — a
        // near-universal flag rate carries no discriminative signal, it just
        // restates "domains are broad and internally overlapping" (already known).
        //
        // Deterministic, threshold-free alternative: every entry's membership set
        // is its own assigned cell PLUS its single next-best-scoring OTHER cell,
        // always exactly one extra candidate, regardless of similarity magnitude.
        // This makes no claim about which items are "genuinely" cross-cutting at
        // the item level — the aggregate BRIDGE COUNT across many items is what
        // carries the signal (a cell pair that keeps coming up as many different
        // items' second choice is meaningfully linked; a rare pairing is not),
        // and it requires no delta to guess.
        let domain_of = |c: usize| -> &str { &cell_names[c].0 };
        let membership_set = |i: usize| -> Vec<usize> {
            let sims = &keep_sims[&i];
            let own = cell_id[&cells[i]];
            let second_best = (0..n_cells).filter(|&c| c != own)
                .max_by(|&a, &b| sims[a].partial_cmp(&sims[b]).unwrap())
                .unwrap();
            vec![own, second_best]
        };

        let mut bridge_counts: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for &i in &keep {
            let ms = membership_set(i);
            for a in 0..ms.len() {
                for b in (a + 1)..ms.len() {
                    let key = (ms[a].min(ms[b]), ms[a].max(ms[b]));
                    bridge_counts.entry(key).or_default().push(i);
                }
            }
        }

        println!("=== Fixed-cell linkage graph: {} cell pairs bridged by at least one real entry ===\n", bridge_counts.len());
        let mut sorted_links: Vec<(&(usize, usize), &Vec<usize>)> = bridge_counts.iter().collect();
        sorted_links.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        for (&(a, b), items) in sorted_links.iter().take(20) {
            let examples: Vec<&str> = items.iter().take(3).map(|&i| texts[i].split_whitespace().next().unwrap_or("")).collect();
            println!(
                "  {}/{}  <->  {}/{}   bridged by {} entries (e.g. {:?})",
                cell_names[a].0, cell_names[a].1, cell_names[b].0, cell_names[b].1, items.len(), examples
            );
        }

        // Cross-domain-only view: these are the links that couldn't exist in a
        // single-domain hard classification at all — the genuinely new structure.
        println!("\n=== Cross-DOMAIN links only (structurally invisible to single-label classification) ===\n");
        let mut cross_links: Vec<(&(usize, usize), &Vec<usize>)> = bridge_counts.iter().filter(|((a, b), _)| domain_of(*a) != domain_of(*b)).collect();
        cross_links.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        for (&(a, b), items) in cross_links.iter().take(15) {
            let examples: Vec<&str> = items.iter().take(3).map(|&i| texts[i].split_whitespace().next().unwrap_or("")).collect();
            println!(
                "  {}/{}  <->  {}/{}   bridged by {} entries (e.g. {:?})",
                cell_names[a].0, cell_names[a].1, cell_names[b].0, cell_names[b].1, items.len(), examples
            );
        }

        println!("\n(this graph is fully reproducible given the same embeddings — no delta to guess, no K, no random seed. The FIXED POINTS (cells) never move because they are not discovered, and the SAME entry always produces the SAME top-2 membership on rerun; the only non-determinism this inherits is the embedder's own, already characterized in Iterations 15-19)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
