// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 20 — an architectural redesign, not another certification
//! signal. Every mechanism tried across Iterations 1-19 (k-means,
//! calibration, certification gates) tries to make an inherently
//! UNSTABLE method — clustering, which shifts with the sample, the
//! embedder, the random seed, the threshold — behave as if it were
//! stable. That is fighting the tool. This asks a different question:
//! can we find reference points that are stable BY CONSTRUCTION, then
//! link them?
//!
//! Fixed points are REAL DATA ITEMS (not synthetic k-means centroids
//! that drift and depend on init/assignment order), identified as local
//! DENSITY MAXIMA in a nearest-neighbor similarity graph — this is
//! exactly physis-core's existing `coherence_score` concept (mean
//! cosine similarity to nearest neighbours), which this whole track
//! used exactly once, for arguably the wrong job (Iteration 9: certifying
//! a k-means SPLIT's quality, where it failed). Its natural job is
//! finding canonical hubs directly, which this experiment tests.
//!
//! No K to guess, no random initialization, no threshold to sweep.
//! Given the same embeddings, this is fully deterministic. "Always the
//! same" is tested directly, not assumed:
//!   - across deterministic sub-samples of the same corpus
//!   - across a SECOND, architecturally different embedder (BGE)
//!
//! Fixed points are then LINKED to each other two ways: direct
//! similarity, and a "bridge count" — how many other items sit close to
//! BOTH of two fixed points (reusing the multi-membership insight from
//! Iterations 8-13, now anchored to something stable instead of a
//! shifting centroid).
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment20_fixed_point_discovery

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;

/// Density of item i = mean cosine similarity to its M nearest OTHER items.
/// This is physis-core's existing coherence_score concept, computed here
/// directly rather than assumed available.
fn densities(embeddings: &[Vec<f32>], m: usize) -> Vec<f32> {
    let n = embeddings.len();
    (0..n).map(|i| {
        let mut sims: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| cosine_sim(&embeddings[i], &embeddings[j])).collect();
        sims.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let k = m.min(sims.len());
        sims.iter().take(k).sum::<f32>() / k as f32
    }).collect()
}

fn nearest_neighbors(embeddings: &[Vec<f32>], i: usize, m: usize) -> Vec<usize> {
    let n = embeddings.len();
    let mut sims: Vec<(usize, f32)> = (0..n).filter(|&j| j != i).map(|j| (j, cosine_sim(&embeddings[i], &embeddings[j]))).collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sims.into_iter().take(m.min(n - 1)).map(|(j, _)| j).collect()
}

/// A fixed point is a LOCAL density maximum: its own density is >= the
/// density of every one of its M nearest neighbors. Deterministic, no
/// parameters beyond M (fixed per dataset, not tuned per result).
fn find_fixed_points(embeddings: &[Vec<f32>], m: usize) -> Vec<usize> {
    let n = embeddings.len();
    let dens = densities(embeddings, m);
    (0..n).filter(|&i| {
        let neighbors = nearest_neighbors(embeddings, i, m);
        neighbors.iter().all(|&j| dens[i] >= dens[j])
    }).collect()
}

/// Bridge count between two fixed points: how many OTHER items sit within
/// `band` of the LOWER of the two fixed points' similarity to that item —
/// i.e. items genuinely close to both, not just close to whichever is
/// more central overall.
fn bridge_strength(embeddings: &[Vec<f32>], a: usize, b: usize, band: f32) -> usize {
    let n = embeddings.len();
    (0..n).filter(|&i| i != a && i != b).filter(|&i| {
        let sim_a = cosine_sim(&embeddings[i], &embeddings[a]);
        let sim_b = cosine_sim(&embeddings[i], &embeddings[b]);
        sim_a.min(sim_b) >= band
    }).count()
}

fn report_fixed_points(name: &str, names: &[&str], embeddings: &[Vec<f32>], m: usize) -> Vec<usize> {
    let fps = find_fixed_points(embeddings, m);
    let dens = densities(embeddings, m);
    println!("\n=== {name} (M={m}, n={}) ===", embeddings.len());
    println!("fixed points found: {}", fps.len());
    let mut sorted_fps = fps.clone();
    sorted_fps.sort_by(|&a, &b| dens[b].partial_cmp(&dens[a]).unwrap());
    for &i in &sorted_fps {
        println!("  '{}'  density={:.3}", names[i], dens[i]);
    }
    fps
}

fn stability_fraction(fps_a: &[usize], fps_b: &[usize]) -> f32 {
    if fps_a.is_empty() { return 0.0; }
    let set_b: std::collections::HashSet<usize> = fps_b.iter().copied().collect();
    fps_a.iter().filter(|i| set_b.contains(i)).count() as f32 / fps_a.len() as f32
}

// ─────────────────────────── Dataset A (known structure, sanity check) ───────────────────────────
const DATASET_A: &[(&str, &str)] = &[
    ("dog", "The dog wagged its tail and waited by the door for its owner to come home."),
    ("cat", "The cat curled up on the windowsill and purred in the afternoon sun."),
    ("horse", "The horse trotted around the paddock, its mane flowing in the breeze."),
    ("sheep", "The sheep grazed quietly in the pasture, following the rest of the flock."),
    ("lion", "The lion stalked its prey across the savanna before launching a sudden charge."),
    ("wolf", "The wolf howled at dusk, calling the rest of its pack to the hunt."),
    ("bear", "The bear caught a salmon in its claws as the fish leapt upstream."),
    ("tiger", "The tiger prowled silently through the tall grass, stripes blending with the shadows."),
    ("eagle", "The eagle soared high above the canyon, scanning the ground for movement."),
    ("sparrow", "The sparrow hopped along the branch before darting off between the leaves."),
    ("owl", "The owl turned its head silently, watching for the faintest movement in the dark."),
    ("swan", "The swan glided smoothly across the lake, barely rippling the water."),
    ("penguin", "The penguin waddled across the ice before diving into the frigid water."),
    ("ostrich", "The ostrich sprinted across the plain on powerful legs, kicking up dust."),
    ("kiwi", "The kiwi foraged in the undergrowth at night, sniffing out insects with its long beak."),
];

// ─────────────────────────── Vehicle ontology (known cross-cutting structure) ───────────────────────────
const VEHICLE: &[(&str, &str)] = &[
    ("engine", "The engine converts fuel into motion through repeated combustion cycles."),
    ("transmission", "The transmission shifts gears to match engine speed with road speed."),
    ("turbocharger", "The turbocharger forces extra compressed air into the cylinders to boost power."),
    ("fuel_tank", "The fuel tank stores gasoline until it is pumped toward the engine."),
    ("spark_plug", "The spark plug ignites the compressed air-fuel mixture inside the cylinder."),
    ("steering_wheel", "The driver turns the steering wheel to change the car's direction."),
    ("accelerator_pedal", "The accelerator pedal opens the throttle to increase speed."),
    ("turn_signal", "Flicking the turn signal alerts other drivers before changing lanes."),
    ("cruise_control", "Cruise control holds a steady speed on the highway without the driver's foot on the pedal."),
    ("blind_spot_monitor", "A light flashes in the mirror when another car lingers just out of view."),
    ("insurance_premium", "Younger drivers usually pay a higher monthly amount for coverage."),
    ("maintenance_cost", "Regular oil changes and tune-ups add up over the years of ownership."),
    ("resale_value", "A well-maintained car keeps a higher resale value after years of use."),
    ("purchase_price", "The sticker price is negotiated before taxes and fees are added."),
    ("trade_in_value", "The dealer offered a trade-in based on mileage and condition."),
    ("hybrid_battery", "The hybrid battery works alongside the engine to reduce gasoline use, though replacing it years later can be a costly repair."),
    ("fuel_efficiency", "Better fuel efficiency means burning less gasoline per mile, which adds up to real savings at the pump over time."),
    ("driver_assistance_system", "Sensors built into the car watch the road and can nudge the wheel if it starts drifting out of its lane."),
    ("regenerative_braking", "Easing off the pedal on the hybrid gently slows the car while feeding energy back into the battery instead of wasting it as heat."),
    ("warranty_coverage", "A multi-year warranty means a sudden breakdown on the road won't leave the owner facing a large repair bill out of pocket."),
];

fn build_and_print_graph(name: &str, names: &[&str], embeddings: &[Vec<f32>], fps: &[usize], band: f32) {
    if fps.len() < 2 { return; }
    println!("\n--- {name}: fixed-point linkage graph (band={band:.2}) ---");
    for a in 0..fps.len() {
        for b in (a + 1)..fps.len() {
            let sim = cosine_sim(&embeddings[fps[a]], &embeddings[fps[b]]);
            let bridge = bridge_strength(embeddings, fps[a], fps[b], band);
            if bridge > 0 || sim > 0.5 {
                println!("  '{}' <-> '{}'  direct-sim={:.3}  bridge-count={}", names[fps[a]], names[fps[b]], sim, bridge);
            }
        }
    }
}

fn main() {
    println!("Experiment 20: fixed-point discovery + linkage — an architectural redesign\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        let minilm_dir = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists());
        let minilm = minilm_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() }));
        let bge_dir = ["models/bge-base-en-v1.5", "../models/bge-base-en-v1.5"].iter().find(|d| std::path::Path::new(d).join("onnx/model.onnx").exists());
        let bge = bge_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 768, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() }));

        let minilm = match minilm { Some(e) if e.is_available() => e, _ => { println!("WARNING: MiniLM not available — aborting."); return; } };
        let bge_available = bge.as_ref().map(|e| e.is_available()).unwrap_or(false);

        // ── Dataset A: sanity check against known structure ──
        let a_names: Vec<&str> = DATASET_A.iter().map(|(n, _)| *n).collect();
        let a_minilm: Vec<Vec<f32>> = DATASET_A.iter().map(|(_, s)| minilm.embed(s)).collect();
        let a_fps_minilm = report_fixed_points("Dataset A (MiniLM)", &a_names, &a_minilm, 3);

        if bge_available {
            let bge = bge.as_ref().unwrap();
            let a_bge: Vec<Vec<f32>> = DATASET_A.iter().map(|(_, s)| bge.embed(s)).collect();
            let a_fps_bge = report_fixed_points("Dataset A (BGE)", &a_names, &a_bge, 3);
            println!("\nCross-embedder stability (Dataset A): {:.1}% of MiniLM's fixed points are also BGE's fixed points", 100.0 * stability_fraction(&a_fps_minilm, &a_fps_bge));
        }

        // Sub-sample stability: odd-indexed vs even-indexed items (two disjoint deterministic views).
        let even_idx: Vec<usize> = (0..a_names.len()).step_by(2).collect();
        let odd_idx: Vec<usize> = (1..a_names.len()).step_by(2).collect();
        let even_emb: Vec<Vec<f32>> = even_idx.iter().map(|&i| a_minilm[i].clone()).collect();
        let odd_emb: Vec<Vec<f32>> = odd_idx.iter().map(|&i| a_minilm[i].clone()).collect();
        let even_names: Vec<&str> = even_idx.iter().map(|&i| a_names[i]).collect();
        let odd_names: Vec<&str> = odd_idx.iter().map(|&i| a_names[i]).collect();
        report_fixed_points("Dataset A, even-indexed subsample (MiniLM)", &even_names, &even_emb, 2);
        report_fixed_points("Dataset A, odd-indexed subsample (MiniLM)", &odd_names, &odd_emb, 2);

        build_and_print_graph("Dataset A", &a_names, &a_minilm, &a_fps_minilm, 0.75);

        // ── Vehicle ontology: known cross-cutting structure ──
        let v_names: Vec<&str> = VEHICLE.iter().map(|(n, _)| *n).collect();
        let v_minilm: Vec<Vec<f32>> = VEHICLE.iter().map(|(_, s)| minilm.embed(s)).collect();
        let v_fps_minilm = report_fixed_points("Vehicle ontology (MiniLM)", &v_names, &v_minilm, 3);
        if bge_available {
            let bge = bge.as_ref().unwrap();
            let v_bge: Vec<Vec<f32>> = VEHICLE.iter().map(|(_, s)| bge.embed(s)).collect();
            let v_fps_bge = report_fixed_points("Vehicle ontology (BGE)", &v_names, &v_bge, 3);
            println!("\nCross-embedder stability (Vehicle): {:.1}% of MiniLM's fixed points are also BGE's fixed points", 100.0 * stability_fraction(&v_fps_minilm, &v_fps_bge));
        }
        build_and_print_graph("Vehicle ontology", &v_names, &v_minilm, &v_fps_minilm, 0.7);

        // ── Real 730-entry ontology: exploratory, what fixed points actually exist? ──
        println!("\n\n########## Real 730-entry ontology: exploratory fixed-point discovery ##########");
        use physis_core::ontology::OntologyLoader;
        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        for def in ontology.classification_domains() {
            if def.domain.is_none() || def.mode.is_none() { continue; }
            let mut text = def.name.clone();
            for hint in &def.hints { text.push(' '); text.push_str(hint); }
            texts.push(text);
        }
        let real_names: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let real_minilm: Vec<Vec<f32>> = texts.iter().map(|t| minilm.embed(t)).collect();
        let real_fps_minilm = report_fixed_points("Real ontology (MiniLM, M=8)", &real_names, &real_minilm, 8);
        if bge_available {
            let bge = bge.as_ref().unwrap();
            let real_bge: Vec<Vec<f32>> = texts.iter().map(|t| bge.embed(t)).collect();
            let real_fps_bge = report_fixed_points("Real ontology (BGE, M=8)", &real_names, &real_bge, 8);
            println!("\nCross-embedder stability (real ontology): {:.1}% of MiniLM's {} fixed points are also BGE's fixed points ({} BGE fixed points total)", 100.0 * stability_fraction(&real_fps_minilm, &real_fps_bge), real_fps_minilm.len(), real_fps_bge.len());

            // Diagnostic: is the STRICT local-maximum rule the brittle part, or is
            // ranking itself unstable across embedders? Compare top-K by RAW density
            // score (no strict-peak requirement) instead.
            let k = real_fps_minilm.len().max(10);
            let minilm_dens = densities(&real_minilm, 8);
            let bge_dens = densities(&real_bge, 8);
            let mut minilm_ranked: Vec<usize> = (0..real_names.len()).collect();
            minilm_ranked.sort_by(|&a, &b| minilm_dens[b].partial_cmp(&minilm_dens[a]).unwrap());
            let mut bge_ranked: Vec<usize> = (0..real_names.len()).collect();
            bge_ranked.sort_by(|&a, &b| bge_dens[b].partial_cmp(&bge_dens[a]).unwrap());
            let top_k_minilm: std::collections::HashSet<usize> = minilm_ranked.iter().take(k).copied().collect();
            let top_k_bge: std::collections::HashSet<usize> = bge_ranked.iter().take(k).copied().collect();
            let overlap = top_k_minilm.intersection(&top_k_bge).count();
            println!(
                "\nDiagnostic — softer criterion (top-{k} by raw density score, no strict local-maximum requirement): {}/{k} overlap ({:.1}%) between MiniLM and BGE",
                overlap, 100.0 * overlap as f32 / k as f32
            );
            println!("(compare to the strict local-maximum rule's {:.1}% above — if this softer version is much more stable, the strict peak criterion itself is the brittle part, not the density concept)", 100.0 * stability_fraction(&real_fps_minilm, &real_fps_bge));
        }
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
