//! Experiment 25 — roadmap item 13, finishing Iteration 17's original
//! proposal properly. Correction to the record first: Iteration 17's
//! code DID already compute and test final training loss as a candidate
//! certification signal on the 6 known toy cases (`loss_scores` /
//! `separates(&loss_scores)` in that file) — it was never written up as
//! its own finding, and the roadmap incorrectly listed this as "not
//! tested." It was tested, and it failed: all 6 cases converge to a
//! near-zero loss (0.001-0.003), indistinguishable from each other.
//!
//! Diagnosed reason, not just observed: Iteration 17's rank-8 predictor
//! is massively overparameterized relative to the 4-14 item TRAINING
//! sets these tiny toy cases produce — with that much capacity it can
//! trivially MEMORIZE any partition, good or bad, to near-zero loss.
//! Training loss under those conditions reflects "does the model have
//! enough capacity to memorize this training set" (always yes here), not
//! "does this partition reflect genuine structure."
//!
//! This tests the natural fix: deliberately UNDER-parameterize the
//! predictor (sweep rank down to 1-2, far below the 8 used before) so it
//! CANNOT trivially memorize a bad partition, forcing loss to reflect
//! actual linear separability instead.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment25_capacity_constrained_loss_gate

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;

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

fn centroid(embeddings: &[&Vec<f32>]) -> Vec<f32> {
    let dim = embeddings[0].len();
    let mut sum = vec![0.0f32; dim];
    for e in embeddings { for d in 0..dim { sum[d] += e[d]; } }
    let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    sum.iter().map(|x| x / norm).collect()
}

struct Predictor { u: Vec<Vec<f32>>, v: Vec<Vec<f32>>, dim: usize, k: usize }

impl Predictor {
    fn new(dim: usize, k: usize) -> Self {
        let mut u = vec![vec![0.0f32; k]; dim];
        let mut v = vec![vec![0.0f32; k]; dim];
        for i in 0..k.min(dim) { u[i][i] = 0.1; v[i][i] = 0.1; }
        Predictor { u, v, dim, k }
    }
    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0f32; self.k];
        for kk in 0..self.k { let mut s = 0.0; for d in 0..self.dim { s += self.u[d][kk] * x[d]; } h[kk] = s; }
        let mut pred_raw = vec![0.0f32; self.dim];
        for d in 0..self.dim { let mut s = 0.0; for kk in 0..self.k { s += self.v[d][kk] * h[kk]; } pred_raw[d] = s; }
        (pred_raw, h)
    }
    fn train_step(&mut self, xs: &[Vec<f32>], own_centroids: &[Vec<f32>], other_centroids: &[Vec<f32>], tau: f32, lr: f32) -> f32 {
        let n = xs.len();
        let mut grad_u = vec![vec![0.0f32; self.k]; self.dim];
        let mut grad_v = vec![vec![0.0f32; self.k]; self.dim];
        let mut total_loss = 0.0f32;
        for i in 0..n {
            let (pred_raw, h) = self.forward(&xs[i]);
            let z_own: f32 = pred_raw.iter().zip(&own_centroids[i]).map(|(a, b)| a * b).sum::<f32>() / tau;
            let z_other: f32 = pred_raw.iter().zip(&other_centroids[i]).map(|(a, b)| a * b).sum::<f32>() / tau;
            let m = z_own.max(z_other);
            let p_own = (z_own - m).exp() / ((z_own - m).exp() + (z_other - m).exp());
            total_loss += -(p_own.max(1e-8)).ln();
            let coef = (1.0 - p_own) / tau;
            let grad_pred_raw: Vec<f32> = (0..self.dim).map(|d| coef * (other_centroids[i][d] - own_centroids[i][d])).collect();
            let mut grad_h = vec![0.0f32; self.k];
            for d in 0..self.dim {
                for kk in 0..self.k { grad_v[d][kk] += grad_pred_raw[d] * h[kk]; grad_h[kk] += self.v[d][kk] * grad_pred_raw[d]; }
            }
            for d in 0..self.dim { for kk in 0..self.k { grad_u[d][kk] += xs[i][d] * grad_h[kk]; } }
        }
        let scale = lr / n as f32;
        for d in 0..self.dim { for kk in 0..self.k { self.u[d][kk] -= scale * grad_u[d][kk]; self.v[d][kk] -= scale * grad_v[d][kk]; } }
        total_loss / n as f32
    }
}

fn holdout_split(assignment: &[usize]) -> Vec<bool> {
    let n = assignment.len();
    let mut held_out = vec![false; n];
    for i in 0..n { if i % 3 == 0 { held_out[i] = true; } }
    for c in 0..2 {
        let members: Vec<usize> = (0..n).filter(|&i| assignment[i] == c).collect();
        let ho_count = members.iter().filter(|&&i| held_out[i]).count();
        if ho_count == 0 && members.len() > 1 { held_out[members[0]] = true; }
        if ho_count == members.len() { held_out[*members.last().unwrap()] = false; }
    }
    held_out
}

fn final_loss_for_rank(embeddings: &[Vec<f32>], assignment: &[usize], rank: usize) -> f32 {
    let dim = embeddings[0].len();
    let held_out = holdout_split(assignment);
    let train_idx: Vec<usize> = (0..embeddings.len()).filter(|&i| !held_out[i]).collect();
    let train_centroids: [Vec<f32>; 2] = [0, 1].map(|c| {
        let members: Vec<&Vec<f32>> = train_idx.iter().filter(|&&i| assignment[i] == c).map(|&i| &embeddings[i]).collect();
        centroid(&members)
    });
    let mut predictor = Predictor::new(dim, rank.min(dim));
    let train_xs: Vec<Vec<f32>> = train_idx.iter().map(|&i| embeddings[i].clone()).collect();
    let train_own: Vec<Vec<f32>> = train_idx.iter().map(|&i| train_centroids[assignment[i]].clone()).collect();
    let train_other: Vec<Vec<f32>> = train_idx.iter().map(|&i| train_centroids[1 - assignment[i]].clone()).collect();
    let mut loss = 0.0;
    for _ in 0..150 { loss = predictor.train_step(&train_xs, &train_own, &train_other, 0.1, 0.5); }
    loss
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
    println!("Experiment 25: does a capacity-CONSTRAINED predictor's training loss separate good from bad splits?\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        let minilm_dir = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists());
        let minilm = match minilm_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() })) {
            Some(e) if e.is_available() => e, _ => { println!("WARNING: MiniLM not available — aborting."); return; }
        };

        let a_emb: Vec<Vec<f32>> = DATASET_A.iter().map(|(_, s, _)| minilm.embed(s)).collect();
        let a_true: Vec<usize> = DATASET_A.iter().map(|(_, _, l)| if *l == "mammal" { 0 } else { 1 }).collect();
        let a_real = kmeans(&a_emb, 2, 30);
        let bird_emb: Vec<Vec<f32>> = BIRD_BRANCH.iter().map(|(_, s)| minilm.embed(s)).collect();
        let bird_real = kmeans(&bird_emb, 2, 30);
        let veh_emb: Vec<Vec<f32>> = VEHICLES.iter().map(|(_, s, _)| minilm.embed(s)).collect();
        let veh_true: Vec<usize> = VEHICLES.iter().map(|v| if v.2 == "two" { 0 } else { 1 }).collect();
        let econ_emb: Vec<Vec<f32>> = ECONOMIC_BRANCH.iter().map(|(_, s)| minilm.embed(s)).collect();
        let econ_real = kmeans(&econ_emb, 2, 30);
        let mech_emb: Vec<Vec<f32>> = MECHANICAL_BRANCH.iter().map(|(_, s)| minilm.embed(s)).collect();
        let mech_real = kmeans(&mech_emb, 2, 30);

        let cases: Vec<(&str, bool, &Vec<Vec<f32>>, &Vec<usize>)> = vec![
            ("1: GOOD-balanced", true, &a_emb, &a_true),
            ("2: BAD-imbalanced", false, &a_emb, &a_real),
            ("3: BAD-balanced", false, &bird_emb, &bird_real),
            ("4: GOOD-imbalanced", true, &veh_emb, &veh_true),
            ("5: BAD-balanced", false, &econ_emb, &econ_real),
            ("6: BAD-imbalanced", false, &mech_emb, &mech_real),
        ];

        for rank in [1usize, 2, 4, 8] {
            println!("=== rank={rank} ===");
            let mut scored: Vec<(bool, f32)> = Vec::new();
            for (name, correct, emb, assignment) in &cases {
                let loss = final_loss_for_rank(emb, assignment, rank);
                println!("  {name:<20} loss={loss:.4}");
                scored.push((*correct, -loss)); // lower loss = better, invert for "higher=better" convention
            }
            let min_good = scored.iter().filter(|(c, _)| *c).map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
            let max_bad = scored.iter().filter(|(c, _)| !*c).map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            println!("  separates (min-good > max-bad, i.e. good cases have LOWER loss than all bad ones): {}\n", min_good > max_bad);
        }
        println!("(if no rank achieves separation, the memorization-capacity hypothesis, while a correct diagnosis of Iteration 17's specific near-zero-loss result, is not the WHOLE story — even a properly capacity-constrained predictor may carry no more certification signal than the geometric statistics already ruled out)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
