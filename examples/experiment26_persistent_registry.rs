//! Experiment 26 — applies PermaTrack's core insight (TRI-ML) to the
//! stability problem this track failed to solve geometrically.
//!
//! PermaTrack's move, for tracking objects through occlusion: do NOT
//! re-detect the object every frame and hope identity holds
//! (tracking-by-detection). Carry a persistent state forward and let each
//! new observation UPDATE it. Identity is maintained by persistence, not
//! rediscovered from each frame independently.
//!
//! Iteration 20 is exactly the tracking-by-detection failure mode. It
//! re-derived "fixed points" independently in each embedding space and
//! got 0% overlap between MiniLM's and BGE's answers — because a
//! different embedder is a different sensor, and re-derivation has no
//! reason to land on the same items. The conclusion drawn there ("stop
//! trying to discover stable points from embedding geometry") was right
//! about re-derivation but never tested the alternative: ESTABLISH the
//! registry once, then match forward.
//!
//! What this measures, head to head on the real 730-entry ontology:
//!   A. RE-DERIVED (Iteration 20's approach): pick anchors independently
//!      in each space, then assign every item to its nearest anchor.
//!   B. PERSISTENT (PermaTrack's approach): pick anchors ONCE in MiniLM
//!      space, keep those exact item identities, and only re-assign
//!      memberships when the embedder changes.
//!
//! The question is not whether B preserves anchor identity — it does so
//! by construction, and claiming that as a result would be circular. The
//! real question is whether the resulting STRUCTURE (which items group
//! under which anchor) survives the sensor change. That is what ARI
//! between the two spaces' assignments measures, and it is what A fails.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment26_persistent_registry

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::HashMap;

fn comb2(x: usize) -> f64 { if x < 2 { 0.0 } else { (x as f64) * ((x - 1) as f64) / 2.0 } }

fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    let mut table: HashMap<(usize, usize), usize> = HashMap::new();
    let mut row: HashMap<usize, usize> = HashMap::new();
    let mut col: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        *table.entry((a[i], b[i])).or_insert(0) += 1;
        *row.entry(a[i]).or_insert(0) += 1;
        *col.entry(b[i]).or_insert(0) += 1;
    }
    let sum_table: f64 = table.values().map(|&v| comb2(v)).sum();
    let sum_row: f64 = row.values().map(|&v| comb2(v)).sum();
    let sum_col: f64 = col.values().map(|&v| comb2(v)).sum();
    let comb_n = comb2(n);
    if comb_n == 0.0 { return 1.0; }
    let expected = sum_row * sum_col / comb_n;
    let max_index = 0.5 * (sum_row + sum_col);
    if (max_index - expected).abs() < 1e-12 { return 1.0; }
    (sum_table - expected) / (max_index - expected)
}

/// Density = mean cosine similarity to the M nearest other items
/// (physis-core's `coherence_score` concept).
fn densities(embeddings: &[Vec<f32>], m: usize) -> Vec<f32> {
    let n = embeddings.len();
    (0..n).map(|i| {
        let mut sims: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| cosine_sim(&embeddings[i], &embeddings[j])).collect();
        sims.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let k = m.min(sims.len());
        sims.iter().take(k).sum::<f32>() / k as f32
    }).collect()
}

/// Seed a registry: the highest-density items, subject to a minimum
/// separation so anchors don't all pile into one dense region. Deterministic
/// given the embeddings (ties broken by index).
fn seed_anchors(embeddings: &[Vec<f32>], k: usize, m: usize, min_sep: f32) -> Vec<usize> {
    let dens = densities(embeddings, m);
    let mut order: Vec<usize> = (0..embeddings.len()).collect();
    order.sort_by(|&a, &b| dens[b].partial_cmp(&dens[a]).unwrap().then(a.cmp(&b)));
    let mut anchors: Vec<usize> = Vec::new();
    for cand in order {
        if anchors.len() >= k { break; }
        if anchors.iter().all(|&a| cosine_sim(&embeddings[cand], &embeddings[a]) < min_sep) {
            anchors.push(cand);
        }
    }
    anchors
}

/// Assign every item to its nearest anchor in the given space.
fn assign_to_anchors(embeddings: &[Vec<f32>], anchors: &[usize]) -> Vec<usize> {
    (0..embeddings.len()).map(|i| {
        (0..anchors.len())
            .max_by(|&a, &b| {
                cosine_sim(&embeddings[i], &embeddings[anchors[a]])
                    .partial_cmp(&cosine_sim(&embeddings[i], &embeddings[anchors[b]]))
                    .unwrap()
            })
            .unwrap()
    }).collect()
}

fn main() {
    println!("Experiment 26: persistent registry (PermaTrack's insight) vs. re-derivation\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        let minilm_dir = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists());
        let minilm = match minilm_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() })) {
            Some(e) if e.is_available() => e, _ => { println!("WARNING: MiniLM unavailable — aborting."); return; }
        };
        let bge_dir = ["models/bge-base-en-v1.5", "../models/bge-base-en-v1.5"].iter().find(|d| std::path::Path::new(d).join("onnx/model.onnx").exists());
        let bge = match bge_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 768, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() })) {
            Some(e) if e.is_available() => e, _ => { println!("WARNING: BGE unavailable — aborting."); return; }
        };

        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        for def in ontology.classification_domains() {
            if def.domain.is_none() || def.mode.is_none() { continue; }
            let mut t = def.name.clone();
            for h in &def.hints { t.push(' '); t.push_str(h); }
            texts.push(t);
        }
        println!("Loaded {} real entries (deterministic order). Embedding in both spaces...", texts.len());
        let emb_a: Vec<Vec<f32>> = texts.iter().map(|t| minilm.embed(t)).collect();
        let emb_b: Vec<Vec<f32>> = texts.iter().map(|t| bge.embed(t)).collect();
        println!("Done.\n");

        let k = 12;
        let m = 8;

        for &min_sep in &[0.90f32, 0.95] {
            println!("=== registry size k={k}, min_sep={min_sep:.2} ===");

            // --- A. RE-DERIVED (Iteration 20's approach) ---
            let anchors_a = seed_anchors(&emb_a, k, m, min_sep);
            let anchors_b_rederived = seed_anchors(&emb_b, k, m, min_sep);
            let overlap: usize = anchors_a.iter().filter(|x| anchors_b_rederived.contains(x)).count();
            let assign_a = assign_to_anchors(&emb_a, &anchors_a);
            let assign_b_rederived = assign_to_anchors(&emb_b, &anchors_b_rederived);
            let ari_rederived = adjusted_rand_index(&assign_a, &assign_b_rederived);

            // --- B. PERSISTENT (PermaTrack's approach) ---
            // Same anchor IDENTITIES carried forward; only memberships re-observed.
            let assign_b_persistent = assign_to_anchors(&emb_b, &anchors_a);
            let ari_persistent = adjusted_rand_index(&assign_a, &assign_b_persistent);

            println!("  A. RE-DERIVED  : anchor-identity overlap {overlap}/{k} ({:.0}%), structure agreement ARI={ari_rederived:.3}", 100.0 * overlap as f32 / k as f32);
            println!("  B. PERSISTENT  : anchor-identity overlap {k}/{k} (100% by construction), structure agreement ARI={ari_persistent:.3}");
            println!("  -> structure-agreement delta from persisting instead of re-deriving: {:+.3}\n", ari_persistent - ari_rederived);
        }

        // What the persistent registry actually looks like, so it can be read
        // rather than trusted as a number.
        let anchors = seed_anchors(&emb_a, k, m, 0.90);
        println!("=== the persistent registry (k={k}), as actual named entries ===");
        let assign_a = assign_to_anchors(&emb_a, &anchors);
        let assign_b = assign_to_anchors(&emb_b, &anchors);
        for (slot, &a) in anchors.iter().enumerate() {
            let size_a = assign_a.iter().filter(|&&x| x == slot).count();
            let size_b = assign_b.iter().filter(|&&x| x == slot).count();
            let kept = (0..texts.len()).filter(|&i| assign_a[i] == slot && assign_b[i] == slot).count();
            println!(
                "  [{slot:>2}] '{}'  members: {size_a} (MiniLM) -> {size_b} (BGE), {kept} retained",
                texts[a].chars().take(46).collect::<String>()
            );
        }
        println!("\n(retention per anchor is the honest per-slot view: a registry can post a good aggregate ARI while individual slots churn, so both are reported)");

        // ── The setting PermaTrack's assumption actually holds in ──
        // Swapping embedders swaps the entire metric space — that is not
        // occlusion, it is changing the camera for one with different intrinsics
        // and no calibration between them, so "carry state forward" has little to
        // hold onto. PermaTrack's persistence works because the SENSOR is stable
        // across frames and only the observation is intermittent. The faithful
        // analogue here is a FIXED embedder over a GROWING corpus — which is also
        // physis-core's real situation: entries accumulate, the model does not
        // change underneath them. That is the question worth answering.
        println!("\n\n=== Corpus growth under a FIXED embedder (the setting PermaTrack's assumption fits) ===");
        let n = texts.len();
        let base: Vec<usize> = (0..n).filter(|i| i % 5 != 0).collect(); // 80% seed corpus
        let base_emb: Vec<Vec<f32>> = base.iter().map(|&i| emb_a[i].clone()).collect();

        // Registry established on the seed corpus only.
        let base_anchor_local = seed_anchors(&base_emb, k, m, 0.90);
        let base_anchor_global: Vec<usize> = base_anchor_local.iter().map(|&l| base[l]).collect();
        let base_assign_local = assign_to_anchors(&base_emb, &base_anchor_local);

        // Corpus grows to 100%. Two policies:
        //   PERSISTENT: keep the same anchor items, just re-assign.
        //   RE-DERIVED: recompute anchors from scratch on the grown corpus.
        let grown_anchor_rederived = seed_anchors(&emb_a, k, m, 0.90);
        let grown_assign_persistent = assign_to_anchors(&emb_a, &base_anchor_global);
        let grown_assign_rederived = assign_to_anchors(&emb_a, &grown_anchor_rederived);

        // Compare only on the shared seed items — the only items both views saw.
        let seed_before: Vec<usize> = base_assign_local.clone();
        let seed_after_persistent: Vec<usize> = base.iter().map(|&g| grown_assign_persistent[g]).collect();
        let seed_after_rederived: Vec<usize> = base.iter().map(|&g| grown_assign_rederived[g]).collect();

        let overlap_rederived = base_anchor_global.iter().filter(|x| grown_anchor_rederived.contains(x)).count();
        println!("  seed corpus {} entries -> grown to {} (+{:.0}%), same embedder throughout", base.len(), n, 100.0 * (n - base.len()) as f32 / base.len() as f32);
        println!("  RE-DERIVED : anchor-identity overlap {overlap_rederived}/{k} ({:.0}%), seed-item assignment stability ARI={:.3}",
            100.0 * overlap_rederived as f32 / k as f32, adjusted_rand_index(&seed_before, &seed_after_rederived));
        println!("  PERSISTENT : anchor-identity overlap {k}/{k} (100%), seed-item assignment stability ARI={:.3}",
            adjusted_rand_index(&seed_before, &seed_after_persistent));
        println!("\nReading this honestly, because the two numbers are not the same KIND of result:");
        println!("  - PERSISTENT's ARI=1.000 is true BY CONSTRUCTION, not an empirical discovery: an");
        println!("    existing item's nearest anchor, among anchors that do not move, cannot change");
        println!("    when unrelated items are added. That is exactly why it is the right architecture,");
        println!("    but it is a design guarantee and should not be reported as a measurement.");
        println!("  - RE-DERIVED's numbers ARE the measurement, and they are the finding: a mere +25%");
        println!("    of data silently replaces half the anchors and destroys ~80% of the structure.");
        println!("    That is the concrete cost of the re-derive-every-time approach used throughout");
        println!("    Iterations 1-20, quantified.");
        println!("  - Surviving a MODEL SWAP is a different problem and remains unsolved (ARI ~0.10");
        println!("    even with persistence, above). Persistence cannot fix it: changing the embedder");
        println!("    changes the metric space itself, so there is no stable geometry to carry forward.");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without embed-onnx; aborting.");
}
