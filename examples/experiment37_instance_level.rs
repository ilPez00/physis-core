// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 37 — number the instances instead of pooling them.
//!
//! Iteration 36 came back inconclusive with the two matching rules disagreeing
//! in sign (z = -2.78 loose, z = +3.27 strict). The diagnosis was that under
//! the loose rule an entry whose name carries common terms matches many
//! episodes and accumulates a large generic bag, and two large generic bags
//! overlap regardless of relatedness.
//!
//! The user's correction goes straight at that:
//!
//!   "can't we number instances of the same name per document, so we know if an
//!    instance of the word 'dog' corresponds to a later, maybe different,
//!    'dog'?"
//!
//! Iteration 36's bug was POOLING. It unioned every episode matching an entry
//! into one bag, which is precisely the move that says every `dog` is the same
//! `dog`. The corpus already supports the alternative: each event carries its
//! own `event_key`, so an occurrence is addressable. This treats
//! (entry, episode) as a numbered INSTANCE and never unions across them.
//!
//! Three statistics, weakest to strongest:
//!
//!   POOLED       Iteration 36's union, kept only as the baseline to beat.
//!   BEST-PAIR    max shared substructure over instance pairs — "SOME instance
//!                of dog and SOME instance of homework share white", rather
//!                than "the dog-bag and the homework-bag overlap somewhere".
//!   CO-EPISODE   the two entries occur in the SAME episode. This is the
//!                literal form of "the dog ate my homework": one context,
//!                both primitives present.
//!
//! Independence is enforced exactly as in 35/36: a shared token counts only
//! when NEITHER entry's embedding read it, and an entry's own name terms are
//! dropped since they are what matched the episode.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment37_instance_level -- <corpus.json>

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Deserialize)]
struct Event {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    evidence: Vec<String>,
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn z_prop(p1: f64, n1: f64, p2: f64, n2: f64) -> f64 {
    if n1 <= 0.0 || n2 <= 0.0 {
        return 0.0;
    }
    let pooled = (p1 * n1 + p2 * n2) / (n1 + n2);
    let se = (pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)).sqrt();
    if se <= 0.0 { 0.0 } else { (p1 - p2) / se }
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| w.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|w| w.len() > 2)
        .collect()
}

/// IDF-weighted intersection of two sorted token slices, skipping anything
/// either embedding already read.
fn shared(a: &[String], b: &[String], own_i: &HashSet<String>, own_j: &HashSet<String>, idf: &dyn Fn(&str) -> f64) -> f64 {
    let (mut x, mut y, mut acc) = (0usize, 0usize, 0.0f64);
    while x < a.len() && y < b.len() {
        match a[x].cmp(&b[y]) {
            std::cmp::Ordering::Less => x += 1,
            std::cmp::Ordering::Greater => y += 1,
            std::cmp::Ordering::Equal => {
                if !own_i.contains(&a[x]) && !own_j.contains(&a[x]) {
                    acc += idf(&a[x]);
                }
                x += 1;
                y += 1;
            }
        }
    }
    acc
}

fn main() {
    println!("Experiment 37: numbered instances instead of one pooled bag per entry\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let path = std::env::args().nth(1).unwrap_or_else(|| "operational_corpus.json".into());
        let Ok(raw) = std::fs::read_to_string(&path) else {
            println!("WARNING: cannot read {path}"); return;
        };
        let Ok(events) = serde_json::from_str::<Vec<Event>>(&raw) else {
            println!("WARNING: {path} is not a Vec<Event>"); return;
        };
        let ep_toks: Vec<Vec<String>> = events
            .iter()
            .map(|e| {
                let mut t = words(&format!("{} {}", e.subject, e.evidence.join(" ")));
                t.sort();
                t.dedup();
                t
            })
            .collect();
        println!("corpus: {} episodes", ep_toks.len());

        let Some(dir) = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists()) else {
            println!("WARNING: MiniLM not available"); return;
        };
        let embedder = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 384, model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean, ..OnnxConfig::default()
        });
        if !embedder.is_available() { println!("WARNING: embedder unavailable"); return; }

        let ontology = OntologyLoader::load_all();
        let (mut texts, mut cells, mut own, mut names) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut t = def.name.clone();
            for h in &def.hints { t.push(' '); t.push_str(h); }
            own.push(words(&t).into_iter().collect::<HashSet<String>>());
            names.push(words(&def.name));
            texts.push(t);
            cells.push((d.clone(), m.clone()));
        }
        let n = texts.len();
        println!("{n} ontology entries, embedding...");
        let emb: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

        // NUMBERED INSTANCES: (entry, episode) pairs, capped for cost, never
        // unioned. This is the whole difference from Iteration 36.
        const MAX_INST: usize = 24;
        let mut inst: Vec<Vec<usize>> = Vec::with_capacity(n);
        for name in &names {
            let mut v = Vec::new();
            if name.len() >= 2 {
                for (e, toks) in ep_toks.iter().enumerate() {
                    if name.iter().filter(|w| toks.binary_search(w).is_ok()).count() >= 2 {
                        v.push(e);
                        if v.len() >= MAX_INST { break; }
                    }
                }
            }
            inst.push(v);
        }
        let with = inst.iter().filter(|v| !v.is_empty()).count();
        let total_inst: usize = inst.iter().map(|v| v.len()).sum();
        println!(
            "entries with >=1 instance: {with}/{n}, total numbered instances: {total_inst} (mean {:.1})\n",
            total_inst as f64 / with.max(1) as f64
        );

        // IDF over episode tokens.
        let mut df: HashMap<&str, usize> = HashMap::new();
        for t in &ep_toks { for w in t { *df.entry(w.as_str()).or_default() += 1; } }
        let ne = ep_toks.len() as f64;
        let idf = move |w: &str| -> f64 { (ne / *df.get(w).unwrap_or(&1) as f64).ln() };

        struct P { cos: f32, pooled: f64, best: f64, coep: f64, sc: bool, sd: bool }
        let mut pairs: Vec<P> = Vec::new();
        for i in 0..n {
            if inst[i].is_empty() { continue }
            // Pooled bag, only to reproduce Iteration 36's baseline.
            let mut bag_i: Vec<String> = Vec::new();
            for &e in &inst[i] { bag_i.extend(ep_toks[e].iter().cloned()); }
            bag_i.sort(); bag_i.dedup();
            for j in (i + 1)..n {
                if inst[j].is_empty() { continue }
                let mut bag_j: Vec<String> = Vec::new();
                for &e in &inst[j] { bag_j.extend(ep_toks[e].iter().cloned()); }
                bag_j.sort(); bag_j.dedup();

                let pooled = shared(&bag_i, &bag_j, &own[i], &own[j], &idf);

                // BEST-PAIR over distinct instances, and CO-EPISODE where the
                // two entries occur in one and the same episode.
                let (mut best, mut coep) = (0.0f64, 0.0f64);
                let set_j: HashSet<usize> = inst[j].iter().copied().collect();
                for &a in &inst[i] {
                    if set_j.contains(&a) {
                        // Same episode: both primitives in one context. Score
                        // its content, minus what either embedding already had.
                        let s = shared(&ep_toks[a], &ep_toks[a], &own[i], &own[j], &idf);
                        if s > coep { coep = s; }
                    }
                    for &b in &inst[j] {
                        if a == b { continue }
                        let s = shared(&ep_toks[a], &ep_toks[b], &own[i], &own[j], &idf);
                        if s > best { best = s; }
                    }
                }
                pairs.push(P {
                    cos: cosine_sim(&emb[i], &emb[j]),
                    pooled, best, coep,
                    sc: cells[i] == cells[j], sd: cells[i].0 == cells[j].0,
                });
            }
        }
        let co_n = pairs.iter().filter(|p| p.coep > 0.0).count();
        println!("{} pairs; {} share at least one episode ({:.1}%)\n", pairs.len(), co_n, 100.0 * co_n as f64 / pairs.len() as f64);

        let mut order: Vec<usize> = (0..pairs.len()).collect();
        order.sort_by(|&a, &b| pairs[a].cos.partial_cmp(&pairs[b].cos).unwrap());
        const STRATA: usize = 40;
        const MAX_IMB: f64 = 0.01;
        let per = order.len() / STRATA;

        println!("  statistic     target        all-strata z    well-controlled z");
        for (sname, pick) in [
            ("POOLED    ", 0usize),
            ("BEST-PAIR ", 1),
            ("CO-EPISODE", 2),
        ] {
            for (tname, fine) in [("same-cell  ", true), ("same-domain", false)] {
                let (mut th, mut nh, mut tl, mut nl) = (0usize, 0usize, 0usize, 0usize);
                let (mut wth, mut wnh, mut wtl, mut wnl, mut wk) = (0usize, 0usize, 0usize, 0usize, 0usize);
                for s in 0..STRATA {
                    let lo = s * per;
                    let hi = if s == STRATA - 1 { order.len() } else { (s + 1) * per };
                    let mut idx: Vec<usize> = order[lo..hi].to_vec();
                    if idx.len() < 20 { continue }
                    let val = |k: usize| match pick { 0 => pairs[k].pooled, 1 => pairs[k].best, _ => pairs[k].coep };
                    idx.sort_by(|&a, &b| val(a).partial_cmp(&val(b)).unwrap());
                    let mid = idx.len() / 2;
                    let hit = |k: &usize| if fine { pairs[*k].sc } else { pairs[*k].sd };
                    let (lo_i, hi_i) = idx.split_at(mid);
                    let (hc, lc) = (hi_i.iter().filter(|k| hit(k)).count(), lo_i.iter().filter(|k| hit(k)).count());
                    let mc = |v: &[usize]| v.iter().map(|&k| pairs[k].cos as f64).sum::<f64>() / v.len() as f64;
                    th += hc; nh += hi_i.len(); tl += lc; nl += lo_i.len();
                    if (mc(hi_i) - mc(lo_i)).abs() < MAX_IMB {
                        wth += hc; wnh += hi_i.len(); wtl += lc; wnl += lo_i.len(); wk += 1;
                    }
                }
                let z = z_prop(th as f64 / nh as f64, nh as f64, tl as f64 / nl as f64, nl as f64);
                let wz = z_prop(wth as f64 / wnh.max(1) as f64, wnh as f64, wtl as f64 / wnl.max(1) as f64, wnl as f64);
                println!("  {sname}    {tname}      {z:+7.2}         {wz:+7.2}   ({wk}/{STRATA} strata)");
            }
        }

        println!(
            "\n(The comparison that matters is POOLED vs BEST-PAIR: identical corpus, identical\n matching, identical controls, differing only in whether occurrences are unioned\n into one bag or kept as numbered instances. Iteration 36 only ever computed the\n POOLED row, and its sign flipped with the matching rule. CO-EPISODE is the\n literal \"dog ate my homework\" case — both primitives in one context — and is\n scored separately because it is a different claim from \"two separate episodes\n happen to share a word\". Still uncontrolled: this corpus covers one domain,\n physis's own development, so matched entries are a biased subset.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
