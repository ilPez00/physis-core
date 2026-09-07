// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 24 — roadmap item 11: cross-embedder AGREEMENT as a
//! certification signal for CLUSTERING SPLITS — a different question
//! from item 15 (cross-embedder corroboration for MULTI-MEMBERSHIP
//! candidates, tested in Iteration 19 and found not supported). Item 11
//! was never actually tested. Iterations 9/11/13/14 all analyzed
//! properties of ONE embedding's ONE clustering; this asks whether an
//! INDEPENDENTLY-discovered partition in a second, architecturally
//! different embedding space (BGE) agrees with the partition being
//! evaluated (whether that partition is oracle-true or MiniLM-discovered)
//! more for good splits than for bad ones.
//!
//! Same 6 known real cases from Iterations 13-14, 17, 19. For each case:
//! the GIVEN partition (as already established — oracle labels for the
//! two GOOD cases, real MiniLM k-means for the four BAD cases) is
//! compared via Adjusted Rand Index against BGE's OWN, independently
//! discovered k-means partition of the same items in its own embedding
//! space — never told what MiniLM's given partition looked like.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment24_cross_embedder_split_agreement

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use std::collections::HashMap;

fn kmeans(embeddings: &[Vec<f32>], k: usize, iterations: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n < k { return (0..n).collect(); }
    let dim = embeddings[0].len();
    let mut centroid_idx = vec![0usize];
    while centroid_idx.len() < k {
        let next = (0..n)
            .max_by(|&a, &b| {
                let da = centroid_idx.iter().map(|&c| 1.0 - cosine_sim(&embeddings[a], &embeddings[c])).fold(f32::INFINITY, f32::min);
                let db = centroid_idx.iter().map(|&c| 1.0 - cosine_sim(&embeddings[b], &embeddings[c])).fold(f32::INFINITY, f32::min);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        centroid_idx.push(next);
    }
    let mut centroids: Vec<Vec<f32>> = centroid_idx.iter().map(|&i| embeddings[i].clone()).collect();
    let mut assignment = vec![0usize; n];
    for _ in 0..iterations {
        for i in 0..n {
            assignment[i] = (0..k).max_by(|&a, &b| cosine_sim(&embeddings[i], &centroids[a]).partial_cmp(&cosine_sim(&embeddings[i], &centroids[b])).unwrap()).unwrap();
        }
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignment[i];
            counts[c] += 1;
            for d in 0..dim { sums[c][d] += embeddings[i][d]; }
        }
        for c in 0..k {
            if counts[c] == 0 { continue; }
            let norm: f32 = sums[c].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            centroids[c] = sums[c].iter().map(|x| x / norm).collect();
        }
    }
    assignment
}

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

const DATASET_A: &[(&str, &str, &str)] = &[
    ("dog", "The dog wagged its tail and waited by the door for its owner to come home.", "mammal"),
    ("cat", "The cat curled up on the windowsill and purred in the afternoon sun.", "mammal"),
    ("horse", "The horse trotted around the paddock, its mane flowing in the breeze.", "mammal"),
    ("sheep", "The sheep grazed quietly in the pasture, following the rest of the flock.", "mammal"),
    ("lion", "The lion stalked its prey across the savanna before launching a sudden charge.", "mammal"),
    ("wolf", "The wolf howled at dusk, calling the rest of its pack to the hunt.", "mammal"),
    ("bear", "The bear caught a salmon in its claws as the fish leapt upstream.", "mammal"),
    ("tiger", "The tiger prowled silently through the tall grass, stripes blending with the shadows.", "mammal"),
    ("eagle", "The eagle soared high above the canyon, scanning the ground for movement.", "bird"),
    ("sparrow", "The sparrow hopped along the branch before darting off between the leaves.", "bird"),
    ("owl", "The owl turned its head silently, watching for the faintest movement in the dark.", "bird"),
    ("swan", "The swan glided smoothly across the lake, barely rippling the water.", "bird"),
    ("penguin", "The penguin waddled across the ice before diving into the frigid water.", "bird"),
    ("ostrich", "The ostrich sprinted across the plain on powerful legs, kicking up dust.", "bird"),
    ("kiwi", "The kiwi foraged in the undergrowth at night, sniffing out insects with its long beak.", "bird"),
];

const BIRD_BRANCH: &[(&str, &str)] = &[
    ("eagle", "The eagle soared high above the canyon, scanning the ground for movement."),
    ("sparrow", "The sparrow hopped along the branch before darting off between the leaves."),
    ("owl", "The owl turned its head silently, watching for the faintest movement in the dark."),
    ("swan", "The swan glided smoothly across the lake, barely rippling the water."),
    ("penguin", "The penguin waddled across the ice before diving into the frigid water."),
    ("ostrich", "The ostrich sprinted across the plain on powerful legs, kicking up dust."),
    ("kiwi", "The kiwi foraged in the undergrowth at night, sniffing out insects with its long beak."),
    ("farm_duck", "Raised alongside the rest of the poultry on the farm, the duck paddled across the pond each morning and burst into the air whenever the fox padded too close."),
    ("cassowary", "The cassowary paced through the rainforest undergrowth on powerful legs, stubby wings folded uselessly at its sides, ready to slash at anything that startled it with the dagger-like claw on each foot."),
    ("farm_chicken", "Kept in a pen behind the farmhouse, the chicken spent its days pecking at the ground, only managing a clumsy flap up to the low roost at dusk."),
];

const VEHICLES: &[(&str, &str, &str)] = &[
    ("motorcycle", "The motorcycle leaned into the curve as the rider downshifted before the hairpin turn.", "two"),
    ("bicycle", "The bicycle wobbled slightly as the child pedaled up the steep driveway.", "two"),
    ("car", "The car merged smoothly onto the highway, accelerating past the slower traffic in the right lane.", "many"),
    ("truck", "The truck rumbled down the gravel road, its heavy bed loaded with stacked lumber.", "many"),
    ("bus", "The bus pulled up to the curb and opened its doors for the line of waiting passengers.", "many"),
    ("van", "The van idled outside the warehouse while the driver checked the delivery manifest.", "many"),
    ("suv", "The vehicle climbed the rocky trail with ease, its high clearance keeping the underside clear of the boulders.", "many"),
    ("pickup", "It towed the boat trailer down the ramp and into the shallow water at the marina.", "many"),
    ("minivan", "Its sliding doors opened automatically as the family unloaded groceries in the driveway.", "many"),
    ("tractor", "The tractor crawled across the muddy field, dragging the plow through the freshly turned soil.", "many"),
    ("forklift", "The forklift lifted the pallet of crates and carried it carefully across the warehouse floor.", "many"),
    ("golf_cart", "The golf cart puttered along the cart path, carrying clubs and coolers toward the ninth hole.", "many"),
];

const ECONOMIC_BRANCH: &[(&str, &str)] = &[
    ("insurance_premium", "insurance premium"), ("maintenance_cost", "maintenance cost"),
    ("resale_value", "resale value"), ("purchase_price", "purchase price"),
    ("trade_in_value", "trade-in value"), ("warranty_coverage", "warranty coverage"),
];

const MECHANICAL_BRANCH: &[(&str, &str)] = &[
    ("engine", "engine"), ("transmission", "transmission"), ("turbocharger", "turbocharger"),
    ("fuel_tank", "fuel tank"), ("spark_plug", "spark plug"),
    ("hybrid_battery", "hybrid battery"), ("fuel_efficiency", "fuel efficiency"),
];

fn main() {
    println!("Experiment 24: cross-embedder agreement as a certification signal for CLUSTERING splits\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        let minilm_dir = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists());
        let minilm = match minilm_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() })) {
            Some(e) if e.is_available() => e, _ => { println!("WARNING: MiniLM not available — aborting."); return; }
        };
        let bge_dir = ["models/bge-base-en-v1.5", "../models/bge-base-en-v1.5"].iter().find(|d| std::path::Path::new(d).join("onnx/model.onnx").exists());
        let bge = match bge_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 768, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() })) {
            Some(e) if e.is_available() => e, _ => { println!("WARNING: BGE not available — aborting."); return; }
        };
        println!("Both embedders loaded.\n");

        struct Case { name: &'static str, correct: bool, _minilm_emb: Vec<Vec<f32>>, bge_emb: Vec<Vec<f32>>, given_partition: Vec<usize> }

        let a_minilm: Vec<Vec<f32>> = DATASET_A.iter().map(|(_, s, _)| minilm.embed(s)).collect();
        let a_bge: Vec<Vec<f32>> = DATASET_A.iter().map(|(_, s, _)| bge.embed(s)).collect();
        let a_true: Vec<usize> = DATASET_A.iter().map(|(_, _, l)| if *l == "mammal" { 0 } else { 1 }).collect();
        let a_real = kmeans(&a_minilm, 2, 30);

        let bird_minilm: Vec<Vec<f32>> = BIRD_BRANCH.iter().map(|(_, s)| minilm.embed(s)).collect();
        let bird_bge: Vec<Vec<f32>> = BIRD_BRANCH.iter().map(|(_, s)| bge.embed(s)).collect();
        let bird_real = kmeans(&bird_minilm, 2, 30);

        let veh_minilm: Vec<Vec<f32>> = VEHICLES.iter().map(|(_, s, _)| minilm.embed(s)).collect();
        let veh_bge: Vec<Vec<f32>> = VEHICLES.iter().map(|(_, s, _)| bge.embed(s)).collect();
        let veh_true: Vec<usize> = VEHICLES.iter().map(|v| if v.2 == "two" { 0 } else { 1 }).collect();

        let econ_minilm: Vec<Vec<f32>> = ECONOMIC_BRANCH.iter().map(|(_, s)| minilm.embed(s)).collect();
        let econ_bge: Vec<Vec<f32>> = ECONOMIC_BRANCH.iter().map(|(_, s)| bge.embed(s)).collect();
        let econ_real = kmeans(&econ_minilm, 2, 30);

        let mech_minilm: Vec<Vec<f32>> = MECHANICAL_BRANCH.iter().map(|(_, s)| minilm.embed(s)).collect();
        let mech_bge: Vec<Vec<f32>> = MECHANICAL_BRANCH.iter().map(|(_, s)| bge.embed(s)).collect();
        let mech_real = kmeans(&mech_minilm, 2, 30);

        let cases = [Case { name: "1: GOOD-balanced (Dataset A true 8v7)", correct: true, _minilm_emb: a_minilm, bge_emb: a_bge, given_partition: a_true },
            Case { name: "2: BAD-imbalanced (Dataset A real, 13v2)", correct: false, _minilm_emb: DATASET_A.iter().map(|(_, s, _)| minilm.embed(s)).collect(), bge_emb: DATASET_A.iter().map(|(_, s, _)| bge.embed(s)).collect(), given_partition: a_real },
            Case { name: "3: BAD-balanced (Iter.10 bird branch, 6v4)", correct: false, _minilm_emb: bird_minilm, bge_emb: bird_bge, given_partition: bird_real },
            Case { name: "4: GOOD-imbalanced (true wheel count, 2v10)", correct: true, _minilm_emb: veh_minilm, bge_emb: veh_bge, given_partition: veh_true },
            Case { name: "5: BAD-balanced (Iter.12 economic branch)", correct: false, _minilm_emb: econ_minilm, bge_emb: econ_bge, given_partition: econ_real },
            Case { name: "6: BAD-imbalanced (Iter.12 mechanical branch)", correct: false, _minilm_emb: mech_minilm, bge_emb: mech_bge, given_partition: mech_real }];

        let case_names: [&[(&str, &str)]; 6] = [
            &[], &[], &[], &[], &[], MECHANICAL_BRANCH,
        ];
        println!("{:<50} {:>18}", "case", "cross-embedder ARI");
        let mut scored: Vec<(bool, f64)> = Vec::new();
        for (ci, c) in cases.iter().enumerate() {
            let bge_independent = kmeans(&c.bge_emb, 2, 30);
            let ari = adjusted_rand_index(&c.given_partition, &bge_independent);
            println!("{:<50} {ari:>18.3}", c.name);
            if ci == 0 || ci == 5 {
                let names: Vec<&str> = if ci == 0 { DATASET_A.iter().map(|(n, ..)| *n).collect() } else { case_names[5].iter().map(|(n, _)| *n).collect() };
                println!("    given:          {:?}", c.given_partition);
                println!("    bge-independent: {:?}", bge_independent);
                println!("    items:          {:?}", names);
            }
            scored.push((c.correct, ari));
        }

        let min_good = scored.iter().filter(|(c, _)| *c).map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
        let max_bad = scored.iter().filter(|(c, _)| !*c).map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
        println!("\n=== Verdict: does min(correct-case ARI) > max(wrong-case ARI)? ===");
        println!("min(good)={min_good:.3}, max(bad)={max_bad:.3} -> separates: {}", min_good > max_bad);
        println!("\n(if TRUE, a partition that agrees with an independently-derived clustering in a DIFFERENT embedding architecture is real evidence of a robust, non-model-specific structure — a mechanism genuinely different from anything that failed in Iterations 9, 11, 13, 14)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
