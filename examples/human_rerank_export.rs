// NOTE: the no-embed-onnx build is a stub; the helpers below serve the
// embed-onnx path only.
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Human-as-reranker — the elicitation half.
//!
//! Every automated certification mechanism this track tried lost to plain
//! cosine, and cosine itself scores AUC 0.410 on the corpus's REAL misfilings
//! (Stage 9) while reaching 0.866 on injected ones (E1). The gap between those
//! two numbers is the whole problem: the machine can rank a worklist, and
//! cannot tell you whether the thing at the top is actually wrong. Stage 10
//! put it exactly — the disposer ranks, at the head of the list only.
//!
//! So the last filter is a person. This exports the task; `human_rerank_score`
//! reads the judgements back and scores them.
//!
//! ## The design, and the one thing it gets to measure that nothing else does
//!
//! The worklist is a BLIND mix of two populations:
//!
//!   INJECTED   deterministically misfiled by the Stage 12 rule. Gold
//!              POSITIVES — certain, no argument about the answer.
//!   NATIVE     the corpus's own filing, untouched. Ground truth UNKNOWN;
//!              the post-0.1.21 misfiling proxy puts it near 16%.
//!
//! The injected items score the rater (recall is exact). The native items are
//! the deliverable: a flag on one is a real misfiling candidate, which this
//! track has never had labelled. The rate of native flags against the ~16%
//! prior is a consistency check on both at once.
//!
//! ## Blinding
//!
//! The task file carries NO labels. `injected` lives only in the gold file,
//! which the UI never reads. Do not merge them.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example human_rerank_export

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::{HashMap, HashSet};

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|w| w.len() > 2)
        .collect()
}

fn load_wordnet(path: &std::path::Path) -> (HashMap<String, Vec<u32>>, HashMap<u32, Vec<u32>>) {
    let mut lemma: HashMap<String, Vec<u32>> = HashMap::new();
    let mut hyper: HashMap<u32, Vec<u32>> = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return (lemma, hyper);
    };
    for line in text.lines() {
        if line.starts_with("  ") || line.trim().is_empty() {
            continue;
        }
        let head = line.split(" | ").next().unwrap_or("");
        let f: Vec<&str> = head.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        let Ok(off) = f[0].parse::<u32>() else {
            continue;
        };
        let Ok(wcnt) = usize::from_str_radix(f[3], 16) else {
            continue;
        };
        for k in 0..wcnt {
            if let Some(w) = f.get(4 + k * 2) {
                lemma
                    .entry(w.replace('_', " ").to_lowercase())
                    .or_default()
                    .push(off);
            }
        }
        let pstart = 4 + wcnt * 2;
        if let Some(pc) = f.get(pstart).and_then(|s| s.parse::<usize>().ok()) {
            for p in 0..pc {
                let b = pstart + 1 + p * 4;
                if let (Some(sym), Some(o), Some(pos)) = (f.get(b), f.get(b + 1), f.get(b + 2)) {
                    if (*sym == "@" || *sym == "@i") && *pos == "n" {
                        if let Ok(t) = o.parse::<u32>() {
                            hyper.entry(off).or_default().push(t);
                        }
                    }
                }
            }
        }
    }
    (lemma, hyper)
}

fn ancestors(seed: &[u32], hyper: &HashMap<u32, Vec<u32>>, max_depth: usize) -> HashSet<u32> {
    let mut seen: HashSet<u32> = seed.iter().copied().collect();
    let mut frontier: Vec<u32> = seed.to_vec();
    for _ in 0..max_depth {
        let mut next = Vec::new();
        for n in frontier.drain(..) {
            if let Some(ps) = hyper.get(&n) {
                for &p in ps {
                    if seen.insert(p) {
                        next.push(p);
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    seen
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

fn jesc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn main() {
    println!("Human-as-reranker — exporting the adjudication task\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let Some(wn) = ["models/wordnet", "../models/wordnet"]
            .iter()
            .map(std::path::Path::new)
            .find(|d| d.join("data.noun").exists())
        else {
            println!("WARNING: WordNet not found — aborting.");
            return;
        };
        let (lemma, hyper) = load_wordnet(&wn.join("data.noun"));

        let Some(m1dir) = ["models", "../models"]
            .iter()
            .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        else {
            println!("WARNING: MiniLM not available — aborting.");
            return;
        };
        let Some(m2dir) = ["models/bge-base-en-v1.5", "../models/bge-base-en-v1.5"]
            .iter()
            .find(|d| std::path::Path::new(d).join("onnx/model.onnx").exists())
        else {
            println!("WARNING: BGE not available — aborting.");
            return;
        };

        let e1 = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 384,
            model_dir: Some(m1dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        // BGE is the ranker: E1 measured it at 0.866 mean AUC against MiniLM's
        // 0.733, on 22 of 22 configurations. The worklist a human sees should
        // be the best one available.
        let e2 = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 768,
            model_dir: Some(m2dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if !e1.is_available() || !e2.is_available() {
            println!("WARNING: an embedder is unavailable — aborting.");
            return;
        }

        // ── cell definitions, so the rater judges against a written standard ──
        // Stage 11's root cause was five words nobody had defined. A rater with
        // no definition would reproduce that failure.
        let anchor_path = ["physis-core/config", "config", "../config"]
            .iter()
            .map(|d| std::path::Path::new(d).join("mode_anchors_ontology.json"))
            .find(|p| p.exists());
        let mut anchor_hints: HashMap<(String, String), (String, Vec<String>)> = HashMap::new();
        let mut anchor_names: HashSet<String> = HashSet::new();
        if let Some(ap) = &anchor_path {
            if let Ok(txt) = std::fs::read_to_string(ap) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    for e in v["domains"].as_array().into_iter().flatten() {
                        let (Some(d), Some(m), Some(nm)) =
                            (e["domain"].as_str(), e["mode"].as_str(), e["name"].as_str())
                        else {
                            continue;
                        };
                        let hs: Vec<String> = e["hints"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(|h| h.as_str().map(|s| s.to_string()))
                            .collect();
                        anchor_hints.insert((d.to_string(), m.to_string()), (nm.to_string(), hs));
                        anchor_names.insert(nm.to_string());
                    }
                }
            }
        }

        // ── corpus, loaded exactly as experiments 38 and 47 load it ──
        let ontology = OntologyLoader::load_all();
        let (mut texts, mut true_cell, mut names) = (Vec::new(), Vec::new(), Vec::new());
        let (mut entry_names, mut entry_hints, mut entry_meta) =
            (Vec::new(), Vec::new(), Vec::new());
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else {
                continue;
            };
            if anchor_names.contains(&def.name) {
                continue;
            }
            let mut t = def.name.clone();
            for h in &def.hints {
                t.push(' ');
                t.push_str(h);
            }
            entry_names.push(def.name.clone());
            entry_hints.push(def.hints.clone());
            entry_meta.push((
                def.category.clone().unwrap_or_default(),
                def.axis_name.clone().unwrap_or_default(),
            ));
            names.push(words(&def.name));
            texts.push(t);
            true_cell.push((d.clone(), m.clone()));
        }
        let n = texts.len();
        println!("{n} entries, embedding...");
        let emb1: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&e1.embed(t))).collect();
        let emb2: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&e2.embed(t))).collect();

        let anc: Vec<HashSet<u32>> = (0..n)
            .map(|i| {
                let mut seed: Vec<u32> = Vec::new();
                for w in &names[i] {
                    if let Some(offs) = lemma.get(w) {
                        seed.extend(offs.iter().take(2));
                    }
                }
                if seed.is_empty() {
                    HashSet::new()
                } else {
                    ancestors(&seed, &hyper, 6)
                }
            })
            .collect();

        // ── the Stage 12 injection, published configuration, unchanged ──
        let cell_names: Vec<(String, String)> = {
            let mut v: Vec<(String, String)> = true_cell.clone();
            v.sort();
            v.dedup();
            v
        };
        let (stride, mult) = (7usize, 13usize);
        let mut cell = true_cell.clone();
        let mut injected: Vec<bool> = vec![false; n];
        let mut k = 0usize;
        for i in 0..n {
            if anc[i].is_empty() {
                continue;
            }
            k += 1;
            #[allow(clippy::manual_is_multiple_of)]
            if k % stride == 0 {
                let mut c = cell_names[(i * mult + 5) % cell_names.len()].clone();
                if c == true_cell[i] {
                    c = cell_names[(i * mult + 6) % cell_names.len()].clone();
                }
                if c != true_cell[i] {
                    cell[i] = c;
                    injected[i] = true;
                }
            }
        }
        println!(
            "injected: {} of {} anchored",
            injected.iter().filter(|x| **x).count(),
            anc.iter().filter(|a| !a.is_empty()).count()
        );

        let mut members: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for (i, c) in cell.iter().enumerate() {
            members.entry(c.clone()).or_default().push(i);
        }

        // ── score every scorable entry, and rank alternatives for each ──
        struct Cand {
            i: usize,
            s1: f64,
            s2: f64,
            alts: Vec<(String, String, f64)>,
        }
        let mut cands: Vec<Cand> = Vec::new();
        for i in 0..n {
            if anc[i].is_empty() {
                continue;
            }
            let peers: Vec<usize> = members[&cell[i]]
                .iter()
                .copied()
                .filter(|&j| j != i && !anc[j].is_empty())
                .collect();
            if peers.len() < 2 {
                continue;
            }
            let ctr = |e: &Vec<Vec<f32>>, dim: usize, ps: &[usize]| -> Vec<f32> {
                let mut acc = vec![0.0f32; dim];
                for &j in ps {
                    for d in 0..dim {
                        acc[d] += e[j][d];
                    }
                }
                normalize(&acc)
            };
            let s1 = cosine_sim(&emb1[i], &ctr(&emb1, 384, &peers)) as f64;
            let s2 = cosine_sim(&emb2[i], &ctr(&emb2, 768, &peers)) as f64;
            // Where else could it go? Same centroid rule, every other cell.
            let mut alts: Vec<(String, String, f64)> = Vec::new();
            for (c, ms) in members.iter() {
                if *c == cell[i] {
                    continue;
                }
                let ps: Vec<usize> = ms
                    .iter()
                    .copied()
                    .filter(|&j| j != i && !anc[j].is_empty())
                    .collect();
                if ps.len() < 2 {
                    continue;
                }
                alts.push((
                    c.0.clone(),
                    c.1.clone(),
                    cosine_sim(&emb2[i], &ctr(&emb2, 768, &ps)) as f64,
                ));
            }
            alts.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            alts.truncate(3);
            cands.push(Cand { i, s1, s2, alts });
        }
        println!("scorable: {}", cands.len());

        // ── the worklist: the machine's head, plus a control tail ──
        // TOP  is the operational scenario — machine proposes, human disposes.
        // TAIL is uniformly sampled from the rest, so the rater is not scored
        //      only on items the machine already believes are wrong, and so a
        //      rater who flags everything is visibly distinguishable.
        cands.sort_by(|a, b| b.s2.partial_cmp(&a.s2).unwrap()); // ascending suspicion later
        cands.reverse(); // now most suspect (lowest s2) first
        let ev = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let n_top = ev("PHYSIS_RERANK_TOP", 50).min(cands.len());
        let n_tail = ev("PHYSIS_RERANK_TAIL", 25);
        let mut pick: Vec<usize> = (0..n_top).collect();
        let mut rng = Lcg(0xA11CE);
        let rest: Vec<usize> = (n_top..cands.len()).collect();
        let mut taken: HashSet<usize> = HashSet::new();
        while taken.len() < n_tail.min(rest.len()) {
            taken.insert(rest[(rng.next() as usize) % rest.len()]);
        }
        let mut tail: Vec<usize> = taken.into_iter().collect();
        tail.sort();
        pick.extend(tail);

        // Present in a SHUFFLED order, so the rater cannot infer the machine's
        // rank from position and anchor on it.
        let mut order = pick.clone();
        for i in (1..order.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            order.swap(i, j);
        }

        let n_inj_in_task = order.iter().filter(|&&c| injected[cands[c].i]).count();
        println!(
            "task: {} items ({} from the worklist head, {} sampled tail)",
            order.len(),
            n_top,
            order.len() - n_top
        );
        println!("  of which injected (gold positives): {n_inj_in_task}");
        println!("  the rest are NATIVE filings — ground truth unknown, ~16% expected wrong");

        // ── write the task (no labels) ──
        let mut t = String::from("{\n  \"generated\": \"experiment: human_rerank_export\",\n");
        t.push_str(&format!("  \"machine\": \"BGE-base-en-v1.5 cosine to cell centroid (E1: 0.866 mean AUC)\",\n  \"n_items\": {},\n", order.len()));
        t.push_str("  \"grid\": {\n    \"domains\": {\"HEAL\":\"condition — the state and capacity of something already running\",\"CONSTRUCT\":\"structure — the durable arrangement other work happens inside or on top of\",\"FABRICATE\":\"output — discrete artifacts and throughput a process emits\",\"BOND\":\"relation — coupling between agents or components\",\"STUDY\":\"knowledge — acquiring and reasoning about information\"},\n");
        t.push_str("    \"modes\": {\"LIFT\":\"peak, intensity, maximum effort or load\",\"REST\":\"recover, idle, pause\",\"WALK\":\"steady flow, ongoing operation at nominal rhythm\",\"WORK\":\"execute, labour, do the routine doing\",\"CREATE\":\"make or design something that did not exist\",\"LEARN\":\"study or practise, self-directed acquisition\",\"DESTROY\":\"tear down, remove, end\",\"SENSE\":\"perceive, measure, observe\",\"GUIDE\":\"lead, mentor, direct others\",\"PLAY\":\"explore or improvise, low stakes\",\"BRAINSTORM\":\"ideate, diverge\",\"MAINTAIN\":\"upkeep, repair, preserve against decay\",\"MOVE\":\"relocate, transport, logistics\",\"PLAN\":\"sequence, organise, converge\"}\n  },\n");
        t.push_str("  \"items\": [\n");
        let mut g = String::from("{\n  \"note\": \"GOLD LABELS. The task file and the UI must never read this.\",\n  \"labels\": {\n");
        for (pos, &c) in order.iter().enumerate() {
            let cd = &cands[c];
            let i = cd.i;
            let (adom, amod) = (&cell[i].0, &cell[i].1);
            let (aname, ahints) = anchor_hints
                .get(&cell[i])
                .cloned()
                .unwrap_or_else(|| (String::new(), Vec::new()));
            let hints_json = entry_hints[i]
                .iter()
                .map(|h| format!("\"{}\"", jesc(h)))
                .collect::<Vec<_>>()
                .join(", ");
            let anchor_json = ahints
                .iter()
                .take(6)
                .map(|h| format!("\"{}\"", jesc(h)))
                .collect::<Vec<_>>()
                .join(", ");
            let alts_json = cd
                .alts
                .iter()
                .map(|(d, m, s)| format!("{{\"cell\": \"{d}/{m}\", \"fit\": {s:.4}}}"))
                .collect::<Vec<_>>()
                .join(", ");
            t.push_str(&format!(
                "    {{\"id\": \"e{i}\", \"pos\": {pos}, \"name\": \"{}\", \"hints\": [{hints_json}], \"category\": \"{}\", \"axis\": \"{}\", \"cell\": \"{adom}/{amod}\", \"domain\": \"{adom}\", \"mode\": \"{amod}\", \"anchor\": \"{}\", \"anchor_hints\": [{anchor_json}], \"fit\": {:.4}, \"fit_minilm\": {:.4}, \"machine_rank\": {}, \"in_head\": {}, \"alts\": [{alts_json}]}}{}\n",
                jesc(&entry_names[i]), jesc(&entry_meta[i].0), jesc(&entry_meta[i].1),
                jesc(&aname), cd.s2, cd.s1, c + 1, c < n_top,
                if pos + 1 == order.len() { "" } else { "," }
            ));
            g.push_str(&format!(
                "    \"e{i}\": {{\"injected\": {}, \"true_cell\": \"{}/{}\", \"assigned_cell\": \"{adom}/{amod}\", \"in_head\": {}, \"machine_rank\": {}}}{}\n",
                injected[i], true_cell[i].0, true_cell[i].1, c < n_top, c + 1,
                if pos + 1 == order.len() { "" } else { "," }
            ));
        }
        t.push_str("  ]\n}\n");
        g.push_str("  }\n}\n");

        let outdir = std::env::var("PHYSIS_RERANK_DIR")
            .unwrap_or_else(|_| "research/perspective-discovery".into());
        let tp = format!("{outdir}/rerank_task.json");
        let gp = format!("{outdir}/rerank_gold.json");
        std::fs::write(&tp, t).expect("write task");
        std::fs::write(&gp, g).expect("write gold");
        println!("\nwrote {tp}   (no labels — safe to publish)");
        println!("wrote {gp}   (gold — keep local, do not feed to the UI)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
