//! Embedder benchmark harness.
//!
//! Evaluates every embedder available on this machine across four suites:
//!
//!   1. STS-lite      — graded sentence pairs, Spearman rank correlation
//!   2. Retrieval     — needle-in-haystack, MRR + recall@1/@3
//!   3. Clustering    — themed corpora, centroid-assignment purity
//!   4. Speed         — single latency + batch throughput
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --example embedder_bench -- [models-dir...]
//!
//! With no args it probes `models/` and `models/bge-base-en-v1.5/` relative to
//! the workspace root, plus any dirs passed on the command line.

use physis_core::embed::{RandomProjectionEmbedder, VectorEmbed};
use std::time::Instant;

// ── Suite 1: STS-lite ────────────────────────────────────────────────

/// (text_a, text_b, human similarity 0.0–5.0)
const STS_PAIRS: &[(&str, &str, f32)] = &[
    // near-duplicates (~5)
    (
        "The pump bearing overheated after three hours of continuous operation.",
        "After running continuously for three hours, the pump bearing overheated.",
        4.8,
    ),
    (
        "A crack was found in the weld seam of the pressure vessel.",
        "Inspectors discovered a crack along the pressure vessel's weld seam.",
        4.7,
    ),
    // strongly related (~4)
    (
        "Coolant temperature exceeded 90°C and triggered an alarm.",
        "The cooling loop ran too hot, which set off the high-temperature alarm.",
        4.2,
    ),
    (
        "The robot arm stalled mid-cycle due to servo overload.",
        "Servo overload caused the robotic arm to stop during its cycle.",
        4.3,
    ),
    (
        "Vibration readings spiked on the main spindle bearing.",
        "High-frequency vibration was measured at the spindle bearing assembly.",
        4.0,
    ),
    (
        "The operator replaced the worn conveyor belt before it snapped.",
        "A frayed conveyor belt was swapped out by the operator just in time.",
        3.9,
    ),
    // moderately related (~3)
    (
        "Quarterly revenue rose 12% driven by strong parts sales.",
        "Sales of spare parts grew, lifting the quarterly financial results.",
        3.2,
    ),
    (
        "The safety interlock prevented the press from cycling with the guard open.",
        "Opening the guard stops the press; the interlock worked as designed.",
        3.4,
    ),
    (
        "New hires must complete forklift certification within 30 days.",
        "Forklift training and certification is required for recent employees.",
        3.0,
    ),
    // loosely related (~2)
    (
        "The maintenance backlog reached 240 open work orders.",
        "Budget planning for next year starts in November.",
        1.5,
    ),
    (
        "Humidity in the paint booth affected coating thickness uniformity.",
        "The canteen menu rotates between three weekly options.",
        0.8,
    ),
    // unrelated (~0-1)
    (
        "Torque specs for the M8 bolts are documented in the assembly guide.",
        "The marketing team launched a social media campaign on Tuesday.",
        0.6,
    ),
    (
        "Ultrasonic testing found no flaws in the rotor shaft.",
        "Quarterly corporate tax accounting spreadsheet needs review.",
        0.7,
    ),
    (
        "The CNC machine completed 400 parts before tool change.",
        "A photograph of a dog wearing sunglasses went viral.",
        0.3,
    ),
    (
        "Thermal camera shows a hotspot on phase C of the motor junction box.",
        ("Quantum qubit decoherence limits computation time in dilution refrigerators."),
        1.0,
    ),
];

// ── Suite 2: Retrieval (needle-in-haystack) ─────────────────────────

const RETRIEVAL_QUERIES: &[&str] = &[
    "how do we calibrate the thermocouple",
    "cooling system pressure loss diagnosis",
    "employee onboarding paperwork checklist",
    "customer refund policy for defective parts",
    "spindle vibration root cause analysis",
];

/// For each query: the index of the single relevant doc in RETRIEVAL_CORPUS.
const RETRIEVAL_RELEVANT: &[usize] = &[6, 7, 8, 9, 10];

const RETRIEVAL_CORPUS: &[&str] = &[
    "The quarterly financial report shows a 4% margin improvement across all product lines.",
    "Safety glasses must be worn at all times inside the production hall.",
    "The company picnic is scheduled for the second Friday of July at the lakeside park.",
    "Laser alignment of the drive coupling reduced energy consumption by 3 percent.",
    "Marketing approved the new brochure design featuring the compact actuator line.",
    "Inventory counts for hydraulic fittings are reconciled at each month end.",
    // needle 0 → thermocouple calibration
    "Thermocouple calibration procedure: immerse the probe in ice bath, verify zero offset, then check boiling point against reference.",
    // needle 1 → cooling pressure loss
    "Diagnosing coolant pressure loss: inspect pump seals, check accumulator precharge, and pressure-test the circuit section by section.",
    // needle 2 → onboarding paperwork
    "New employee onboarding checklist: signed contract, tax forms, badge photo, IT account request, safety briefing attendance record.",
    // needle 3 → refunds
    "Defective part returns are eligible for a full customer refund when the failure report is filed within thirty days of delivery.",
    // needle 4 → spindle vibration RCA
    "Spindle vibration analysis points to bearing raceway spalling; perform root cause analysis using the spectrum waterfall captured at 8000 rpm.",
];

// ── Suite 3: Clustering ──────────────────────────────────────────────

const CLUSTER_THEMES: &[&[&str]] = &[
    &[
        "Replace the hydraulic filter and top up reservoir fluid.",
        "The gearbox oil change interval was shortened to 2000 hours.",
        "Grease the linear guides and inspect wipers for damage.",
        "Lubrication schedule updated for the chain drive assemblies.",
        "Drain water from the compressed air receiver tank.",
    ],
    &[
        "Q3 marketing campaign focuses on renewable-energy customers.",
        "The sales team closed two enterprise contracts this month.",
        "Customer satisfaction survey results improved to 4.4 of 5.",
        "Trade-show booth bookings confirmed for the Hannover fair.",
        "Brand refresh rolled out across the website and packaging.",
    ],
    &[
        "Migrate the database cluster to the new availability zone.",
        "CI pipeline now runs integration tests on every merge request.",
        "Kubernetes pod autoscaling tuned based on p95 memory usage.",
        "Rotate TLS certificates before expiry using the automation job.",
        "Backup retention policy raised from 14 to 30 days.",
    ],
    &[
        "Weld penetration depth verified by ultrasonic inspection.",
        "Surface roughness on machined bores held within Ra 0.8 spec.",
        "First-article inspection passed for the new bracket revision.",
        "Coordinate measuring machine reported all dimensions in tolerance.",
        "Paint thickness measured at 80 microns average across panels.",
    ],
];

// ── Metrics helpers ──────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

fn rank<T>(vals: &[T]) -> Vec<usize>
where
    T: PartialOrd,
{
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| {
        vals[b]
            .partial_cmp(&vals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0usize; vals.len()];
    for (pos, &i) in idx.iter().enumerate() {
        ranks[i] = pos;
    }
    ranks
}

fn spearman(xs: &[f32], ys: &[f32]) -> f32 {
    let rx = rank(xs);
    let ry = rank(ys);
    let n = xs.len() as f32;
    let mean_rx = (rx.iter().sum::<usize>() as f32) / n;
    let mean_ry = (ry.iter().sum::<usize>() as f32) / n;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..xs.len() {
        let dx = rx[i] as f32 - mean_rx;
        let dy = ry[i] as f32 - mean_ry;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    if dx2 <= f32::EPSILON || dy2 <= f32::EPSILON {
        return 0.0;
    }
    num / (dx2.sqrt() * dy2.sqrt())
}

struct BenchResult {
    sts_spearman: f32,
    mrr: f32,
    recall_at_1: f32,
    recall_at_3: f32,
    cluster_purity: f32,
    dup_separation: f32,
    single_us: f64,
    batch_per_embed_us: f64,
}

fn bench(e: &dyn VectorEmbed) -> BenchResult {
    // STS
    let sims: Vec<f32> = STS_PAIRS
        .iter()
        .map(|(a, b, _)| cosine(&e.embed(a), &e.embed(b)))
        .collect();
    let humans: Vec<f32> = STS_PAIRS.iter().map(|(_, _, s)| s / 5.0).collect();
    let sts = spearman(&sims, &humans);

    // Retrieval
    let corpus_emb: Vec<Vec<f32>> = RETRIEVAL_CORPUS.iter().map(|d| e.embed(d)).collect();
    let mut mrr_sum = 0.0;
    let mut r1 = 0.0;
    let mut r3 = 0.0;
    for (qi, q) in RETRIEVAL_QUERIES.iter().enumerate() {
        let qv = e.embed(q);
        let mut scored: Vec<(usize, f32)> = corpus_emb
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine(&qv, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let pos = scored
            .iter()
            .position(|(i, _)| *i == RETRIEVAL_RELEVANT[qi]);
        if let Some(rank1based) = pos.map(|p| p + 1) {
            mrr_sum += 1.0 / rank1based as f32;
            if rank1based == 1 {
                r1 += 1.0;
            }
            if rank1based <= 3 {
                r3 += 1.0;
            }
        }
    }
    let nq = RETRIEVAL_QUERIES.len() as f32;

    // Clustering purity: assign each doc to nearest theme-centroid.
    let theme_embs: Vec<Vec<Vec<f32>>> = CLUSTER_THEMES
        .iter()
        .map(|docs| docs.iter().map(|d| e.embed(d)).collect())
        .collect();
    let centroids: Vec<Vec<f32>> = theme_embs
        .iter()
        .map(|docs| {
            let n = docs.len() as f32;
            let dim = docs[0].len();
            let mut c = vec![0.0f32; dim];
            for d in docs {
                for (i, x) in d.iter().enumerate() {
                    c[i] += x / n;
                }
            }
            let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            c.iter_mut().for_each(|x| *x /= norm);
            c
        })
        .collect();
    let mut correct = 0usize;
    let mut total = 0usize;
    for (ti, docs) in theme_embs.iter().enumerate() {
        for d in docs {
            let best = centroids.iter().enumerate().max_by(|(_, a), (_, b)| {
                cosine(d, a)
                    .partial_cmp(&cosine(d, b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some((bi, _)) = best {
                if bi == ti {
                    correct += 1;
                }
            }
            total += 1;
        }
    }
    let purity = correct as f32 / total.max(1) as f32;

    // Near-dup separation: dup pair sim minus unrelated-pair mean sim.
    let dup_sim = cosine(
        &e.embed("Bearing temperature alarm threshold set to 75 degrees Celsius."),
        &e.embed("Bearing temperature alarm threshold set at 75°C."),
    );
    let unrelated_sims: Vec<f32> = [
        (
            "Bearing temperature alarm threshold set to 75 degrees Celsius.",
            "The canteen serves lunch between noon and two.",
        ),
        (
            "Bearing temperature alarm threshold set to 75 degrees Celsius.",
            "Quarterly board meeting minutes were circulated yesterday.",
        ),
        (
            "Hydraulic pump flow rate dropped below nominal range.",
            "The hotel reservation includes breakfast for two guests.",
        ),
    ]
    .iter()
    .map(|pair| {
        let (a, b) = *pair;
        cosine(&e.embed(a), &e.embed(b))
    })
    .collect();
    let unrelated_mean = unrelated_sims.iter().sum::<f32>() / unrelated_sims.len() as f32;

    // Speed
    let t0 = Instant::now();
    for i in 0..50 {
        let _ = e.embed(&format!("latency probe sentence number {i}"));
    }
    let single_us = t0.elapsed().as_secs_f64() * 1e6 / 50.0;

    let texts: Vec<String> = (0..100)
        .map(|i| format!("batch throughput sentence {i} about pumps and bearings"))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let t1 = Instant::now();
    let _ = e.embed_batch(&refs);
    let batch_us = t1.elapsed().as_secs_f64() * 1e6 / texts.len() as f64;

    BenchResult {
        sts_spearman: sts,
        mrr: mrr_sum / nq,
        recall_at_1: r1 / nq,
        recall_at_3: r3 / nq,
        cluster_purity: purity,
        dup_separation: dup_sim - unrelated_mean,
        single_us,
        batch_per_embed_us: batch_us,
    }
}

fn print_row(name: &str, dims: usize, semantic: bool, r: &BenchResult) {
    println!(
        "| {name:<34} | {dims:>4} | {:>8} | {:>6.3} | {:>5.3} | {:>5.3} | {:>5.3} | {:>6.3} | {:>7.3} | {:>9.1} | {:>9.1} |",
        if semantic { "semantic" } else { "lexical" },
        r.sts_spearman,
        r.recall_at_1,
        r.recall_at_3,
        r.mrr,
        r.cluster_purity,
        r.dup_separation,
        r.single_us,
        r.batch_per_embed_us
    );
}

fn main() {
    let mut model_dirs: Vec<String> = std::env::args().skip(1).collect();
    if model_dirs.is_empty() {
        for candidate in ["models", "models/bge-base-en-v1.5"] {
            let p = std::path::Path::new(candidate);
            if p.join("model.onnx").exists() || p.join("onnx/model.onnx").exists() {
                model_dirs.push(candidate.to_string());
            }
        }
    }

    println!("\nPhysis embedder benchmark");
    println!("═════════════════════════");
    println!(
        "| {:<34} | {:>4} | {:>8} | {:>7} | {:>5} | {:>6} | {:>7} | {:>7} | {:>8} | {:>8} |",
        "embedder", "dims", "kind", "STS", "R@1", "R@3", "MRR", "purity", "dup-delta", "us/emb"
    );
    println!(
        "|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|",
        "-".repeat(35),
        "-".repeat(4),
        "-".repeat(8),
        "-".repeat(7),
        "-".repeat(5),
        "-".repeat(6),
        "-".repeat(7),
        "-".repeat(7),
        "-".repeat(8),
        "-".repeat(8),
        "-".repeat(8)
    );
    println!(
        "|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|",
        "-".repeat(35),
        "-".repeat(4),
        "-".repeat(8),
        "-".repeat(7),
        "-".repeat(5),
        "-".repeat(6),
        "-".repeat(7),
        "-".repeat(7),
        "-".repeat(8),
        "-".repeat(8),
        "-".repeat(8)
    );

    for dim in [64usize, 128, 384, 768] {
        let e = RandomProjectionEmbedder::new(dim);
        print_row(
            &format!("random-projection ({dim}d)"),
            dim,
            false,
            &bench(&e),
        );
    }

    #[cfg(feature = "embed-onnx")]
    for dir in &model_dirs {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        // Native width lives in config.json (`hidden_size`); defaulting to 384
        // would silently truncate a 768-d model's output vector.
        let native_dim = std::fs::read_to_string(format!("{dir}/config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("hidden_size").and_then(|h| h.as_u64()))
            .map(|h| h as usize);
        let cfg = OnnxConfig {
            dim: native_dim.unwrap_or(384),
            model_dir: Some(dir.clone()),
            ..OnnxConfig::default()
        };
        let probe = OnnxEmbedder::with_config(&cfg);
        if !probe.is_available() {
            println!("| {dir:<34} | — model/tokenizer missing, skipped");
            continue;
        }
        let dim = probe.dimension();
        let name = format!(
            "onnx ({}) ",
            std::path::Path::new(dir)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(dir)
        );
        print_row(name.trim(), dim, true, &bench(&probe));

        // Also try CLS pooling for comparison on MiniLM-class models.
        let cfg = OnnxConfig {
            dim,
            model_dir: Some(dir.clone()),
            pooling: PoolingStrategy::Cls,
            ..OnnxConfig::default()
        };
        let cls = OnnxEmbedder::with_config(&cfg);
        if cls.is_available() {
            print_row(&format!("{name}·cls"), dim, true, &bench(&cls));
        }
    }

    #[cfg(not(feature = "embed-onnx"))]
    {
        let _ = model_dirs;
        println!("(built without --features embed-onnx; ONNX models skipped)");
    }

    println!("\nSuites: STS ρ = Spearman vs 16 graded pairs · R@k/MRR over 5 needles in 11 docs ·");
    println!(
        "purity = centroid clustering of 20 themed docs · dupΔ = duplicate sim − unrelated sim."
    );
    println!();
}
