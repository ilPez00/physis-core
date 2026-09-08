// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 46 — should the crowded cells be split? The expansion question.
//!
//! After the Gate 0 regeneration the corpus sits in 57 of 70 cells, but seven
//! cells hold 43% of it — `CONSTRUCT/CREATE` alone has 69 entries, bundling
//! "model the schema", "architect a service", "found an institution" and "lay
//! out a vehicle chassis". That is the junk-drawer shape Stage 7 diagnosed,
//! reappearing at a new address. Meanwhile BRAINSTORM holds 3 entries across
//! all five domains, DESTROY 4, PLAY 13.
//!
//! So: does a crowded cell contain real sub-structure worth giving its own
//! coordinate, or is it just large?
//!
//! ## The two nulls, because this track has killed five positives that lacked one
//!
//! Any set of points splits into two clusters, and any split has some
//! separation. Two controls say whether this one means anything.
//!
//! **PERMUTATION null.** The same entries, partitioned at random into the same
//! two sizes, 200 times. If the k-means split is not far out in that
//! distribution, the "structure" is what any partition of any point cloud
//! shows.
//!
//! **MIXTURE reference, and read its direction carefully.** A size-matched
//! group drawn at random from ACROSS the corpus contains entries from many
//! different cells, so it has genuine cluster structure and should split
//! cleanly. It is not a floor to beat — it is the scale:
//!
//!   cell z ~= mixture z   the cell is as internally divided as a mixture of
//!                         unrelated cells -> a real case for splitting it
//!   cell z << mixture z   the cell is homogeneous -> it is merely large, and
//!                         splitting would invent a distinction
//!
//! **STABILITY.** Structure that moves when you drop a fifth of the entries is
//! not a coordinate. 25 replicates at 80% subsampling; the score is how often
//! two entries that clustered together still do.
//!
//! The machine proposes and ranks. It does not name the split or decide it —
//! Stage 9 measured the machine's judgement of "does this belong here" at AUC
//! 0.410 on real misfilings, chance inside the interval. Members of each
//! sub-cluster are printed so a person can see whether the split names
//! anything.
//!
//!   cargo run -p physis-core --features embed-onnx --release \
//!     --example experiment46_cell_substructure

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
    use physis_core::ontology::OntologyLoader;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use std::collections::HashMap;

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

    // Which source file each entry came from. A split is far more legible as
    // "one side is entirely semiotic_ontology" than as a list of names, and on
    // this corpus that turns out to be the whole story.
    let mut source_of: HashMap<String, String> = HashMap::new();
    for (kind, json) in physis_core::ontology::ONTOLOGY_SOURCES {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(json) {
            for e in v["domains"].as_array().into_iter().flatten() {
                if let Some(nm) = e["name"].as_str() {
                    source_of.entry(nm.to_string()).or_insert_with(|| (*kind).to_string());
                }
            }
        }
    }

    let ontology = OntologyLoader::load_all();
    let mut names = Vec::new();
    let mut cells = Vec::new();
    let mut srcs = Vec::new();
    let mut proses: Vec<bool> = Vec::new();
    let mut emb: Vec<Vec<f32>> = Vec::new();
    for def in ontology.classification_domains() {
        let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
        let mut t = def.name.clone();
        for h in &def.hints {
            t.push(' ');
            t.push_str(h);
        }
        // Hint FORMAT, the confound this track would otherwise walk into: only
        // semiotic_ontology and agent_ontology write prose hints ("A sign that
        // denotes by virtue of resemblance"), everything else writes keyword
        // lists ("bolt | screw | nut"). Embeddings separate those two shapes on
        // sight, so any split that lines up with format is measuring house
        // style, not meaning. Reported per side so it cannot hide.
        let prose = !def.hints.is_empty()
            && def.hints.iter().map(|h| h.len()).sum::<usize>() / def.hints.len() > 25;
        proses.push(prose);
        // PHYSIS_TEXT=name drops the hints entirely, which removes the format
        // difference along with them. If a split survives that, it is about the
        // concepts; if it evaporates, it was house style.
        if std::env::var("PHYSIS_TEXT").as_deref() == Ok("name") {
            t = def.name.clone();
        }
        srcs.push(source_of.get(&def.name).cloned().unwrap_or_else(|| "?".into()));
        names.push(def.name.clone());
        cells.push(format!("{d}/{m}"));
        emb.push(normalize(&embedder.embed(&t)));
    }
    let n_prose = proses.iter().filter(|p| **p).count();
    println!(
        "corpus: {} entries, {} with prose hints ({:.0}%) — the format confound\n",
        names.len(),
        n_prose,
        100.0 * n_prose as f64 / names.len() as f64
    );

    let mut members: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, c) in cells.iter().enumerate() {
        members.entry(c.clone()).or_default().push(i);
    }
    let mut crowded: Vec<(&String, &Vec<usize>)> =
        members.iter().filter(|(_, v)| v.len() >= 25).collect();
    crowded.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));

    let mut rng = rand::rngs::StdRng::seed_from_u64(20260908);

    println!(
        "{:<20} {:>4} {:>8} {:>8} {:>9} {:>10} {:>8}   {}",
        "cell", "n", "sep", "z(perm)", "stability", "z(mixture)", "balance", "verdict"
    );
    println!("{}", "-".repeat(112));
    let mut printed: Vec<(String, Vec<usize>, Vec<usize>)> = Vec::new();
    for (cell, idx) in &crowded {
        let (a, b) = split2(idx, &emb);
        let sep = separation(&a, &b, &emb);
        let z = perm_z(idx, &emb, sep, &mut rng);
        let stab = stability(idx, &emb, &mut rng);

        // Size-matched mixture drawn from across the corpus: the scale, not a floor.
        let mut zs = Vec::new();
        for _ in 0..5 {
            let mut pool: Vec<usize> = (0..emb.len()).collect();
            pool.shuffle(&mut rng);
            let mix: Vec<usize> = pool.into_iter().take(idx.len()).collect();
            let (ma, mb) = split2(&mix, &emb);
            let msep = separation(&ma, &mb, &emb);
            zs.push(perm_z(&mix, &emb, msep, &mut rng));
        }
        let mz = zs.iter().sum::<f64>() / zs.len() as f64;

        // Balance matters: a 58/4 "split" is a handful of outliers peeling off,
        // not a cell dividing in two, and it argues for removing those entries
        // rather than for a new coordinate.
        let small = a.len().min(b.len());
        let balance = small as f64 / idx.len() as f64;
        // A z-score alone is not enough: with 200 permutations a separation of
        // 0.011 can sit many sigma out and still be nothing. Require the split
        // to be visible in absolute terms too.
        let verdict = if sep < 0.05 {
            "no — separation is negligible in absolute terms, whatever its z"
        } else if z < 2.0 {
            "no — separation is inside the permutation null"
        } else if stab < 0.75 {
            "no — structure moves when entries are dropped"
        } else if balance < 0.20 {
            "PEEL-OFF — a minority breaks away; look at what it is, do not split"
        } else if z >= mz * 0.8 {
            "SPLIT CANDIDATE — as divided as a mixture of unrelated cells"
        } else {
            "large, not divided — homogeneous next to the mixture scale"
        };
        println!(
            "{:<20} {:>4} {:>8.4} {:>8.2} {:>9.3} {:>10.2} {:>7.0}%   {}",
            cell, idx.len(), sep, z, stab, mz, 100.0 * balance, verdict
        );
        if sep >= 0.05 && z >= 2.0 && stab >= 0.75 {
            printed.push(((*cell).clone(), a, b));
        }
    }

    if printed.is_empty() {
        println!("\nNo cell cleared both bars. The crowding is size, not structure —\nsplitting these cells would invent a distinction the data does not carry.");
        return;
    }
    println!("\n=== what each side actually is, for a person to name (or reject) ===");
    for (cell, a, b) in &printed {
        println!("\n--- {cell} ---");
        for (label, side) in [("A", a), ("B", b)] {
            let mut by: HashMap<&str, usize> = HashMap::new();
            for &i in side {
                *by.entry(srcs[i].as_str()).or_insert(0) += 1;
            }
            let mut comp: Vec<(&str, usize)> = by.into_iter().collect();
            comp.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(y.0)));
            let top: Vec<String> = comp
                .iter()
                .take(4)
                .map(|(k, v)| format!("{k} {:.0}%", 100.0 * *v as f64 / side.len() as f64))
                .collect();
            let mut ns: Vec<&str> = side.iter().map(|&i| names[i].as_str()).collect();
            ns.sort();
            let pr = 100.0 * side.iter().filter(|&&i| proses[i]).count() as f64 / side.len() as f64;
            println!("  {label} ({:>2})  prose-hints {pr:.0}%   sources: {}", ns.len(), top.join(", "));
            println!("        {}", ns.join(", "));
        }
    }
}

/// Deterministic 2-means, best of 8 seeded initialisations by inertia.
#[cfg(feature = "embed-onnx")]
fn split2(idx: &[usize], emb: &[Vec<f32>]) -> (Vec<usize>, Vec<usize>) {
    let mut best: Option<(f64, Vec<usize>, Vec<usize>)> = None;
    for seed in 0..8usize {
        let (mut ca, mut cb) = (emb[idx[seed % idx.len()]].clone(), emb[idx[(seed * 7 + 3) % idx.len()]].clone());
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for _ in 0..25 {
            a.clear();
            b.clear();
            for &i in idx {
                if cos(&emb[i], &ca) >= cos(&emb[i], &cb) { a.push(i) } else { b.push(i) }
            }
            if a.is_empty() || b.is_empty() { break }
            ca = centroid(&a, emb);
            cb = centroid(&b, emb);
        }
        if a.is_empty() || b.is_empty() { continue }
        let inertia: f64 = a.iter().map(|&i| 1.0 - cos(&emb[i], &ca) as f64).sum::<f64>()
            + b.iter().map(|&i| 1.0 - cos(&emb[i], &cb) as f64).sum::<f64>();
        if best.as_ref().map(|(x, _, _)| inertia < *x).unwrap_or(true) {
            best = Some((inertia, a.clone(), b.clone()));
        }
    }
    let (_, a, b) = best.expect("a split");
    (a, b)
}

/// Mean within-side cosine minus mean across-side cosine.
#[cfg(feature = "embed-onnx")]
fn separation(a: &[usize], b: &[usize], emb: &[Vec<f32>]) -> f64 {
    let mut win = (0.0f64, 0usize);
    for side in [a, b] {
        for i in 0..side.len() {
            for j in (i + 1)..side.len() {
                win.0 += cos(&emb[side[i]], &emb[side[j]]) as f64;
                win.1 += 1;
            }
        }
    }
    let mut acr = (0.0f64, 0usize);
    for &i in a {
        for &j in b {
            acr.0 += cos(&emb[i], &emb[j]) as f64;
            acr.1 += 1;
        }
    }
    if win.1 == 0 || acr.1 == 0 { return 0.0 }
    win.0 / win.1 as f64 - acr.0 / acr.1 as f64
}

/// How far the real split sits from 200 random partitions of the same sizes.
#[cfg(feature = "embed-onnx")]
fn perm_z(idx: &[usize], emb: &[Vec<f32>], sep: f64, rng: &mut impl rand::Rng) -> f64 {
    use rand::seq::SliceRandom;
    let half = idx.len() / 2;
    let mut v: Vec<f64> = Vec::with_capacity(200);
    let mut scratch = idx.to_vec();
    for _ in 0..200 {
        scratch.shuffle(rng);
        v.push(separation(&scratch[..half], &scratch[half..], emb));
    }
    let mu = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    if sd < 1e-12 { 0.0 } else { (sep - mu) / sd }
}

/// Co-assignment consistency across 25 subsamples at 80%.
#[cfg(feature = "embed-onnx")]
fn stability(idx: &[usize], emb: &[Vec<f32>], rng: &mut impl rand::Rng) -> f64 {
    use rand::seq::SliceRandom;
    use std::collections::HashMap;
    let mut together: HashMap<(usize, usize), (u32, u32)> = HashMap::new();
    let keep = (idx.len() as f64 * 0.8).round() as usize;
    let mut scratch = idx.to_vec();
    for _ in 0..25 {
        scratch.shuffle(rng);
        let sub = &scratch[..keep];
        let (a, b) = split2(sub, emb);
        let side: HashMap<usize, u8> =
            a.iter().map(|&i| (i, 0u8)).chain(b.iter().map(|&i| (i, 1u8))).collect();
        for x in 0..sub.len() {
            for y in (x + 1)..sub.len() {
                let (p, q) = (sub[x].min(sub[y]), sub[x].max(sub[y]));
                let e = together.entry((p, q)).or_insert((0, 0));
                e.1 += 1;
                if side[&sub[x]] == side[&sub[y]] { e.0 += 1 }
            }
        }
    }
    // Per pair, how one-sided its co-assignment was; 1.0 = always the same answer.
    let mut s = 0.0;
    let mut n = 0usize;
    for (_, (same, total)) in together {
        if total < 5 { continue }
        let f = same as f64 / total as f64;
        s += f.max(1.0 - f);
        n += 1;
    }
    if n == 0 { 0.0 } else { s / n as f64 }
}

#[cfg(feature = "embed-onnx")]
fn centroid(idx: &[usize], emb: &[Vec<f32>]) -> Vec<f32> {
    let d = emb[idx[0]].len();
    let mut c = vec![0.0f32; d];
    for &i in idx {
        for k in 0..d { c[k] += emb[i][k] }
    }
    normalize(&c)
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
