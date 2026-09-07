// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 38 — a SOUNDNESS check, which is the half `coverage` does not have.
//!
//! Iteration 33 shipped `physis_core::coverage` with a caveat: "coverage is not
//! correctness — a candidate that swallows records into a wrong cell scores
//! exactly like one that captures a real gap". Cheung's second paper names why:
//!
//!   "Soundness means all items below each facet are relevant to the facet.
//!    Completeness means any items or facets relevant to a specific facet are
//!    already contained in and accessible through the facet. Incomplete facets
//!    will reduce recall, while unsound facets will reduce precision."
//!
//! `coverage` is a COMPLETENESS check. This is the missing SOUNDNESS one: does a
//! cell contain items that do not belong in it?
//!
//! The paper points at the mechanism — "non-lattice auditing methods can
//! precisely identify and potentially fix such issues" — a purely structural,
//! label-free audit: where a pair of concepts has more than one MINIMAL common
//! ancestor, the hierarchy fails to be a lattice there, and such fragments are
//! empirically enriched for modelling errors.
//!
//! ## Ground truth, constructed rather than assumed
//!
//! physis has no labelled misfilings, so they are INJECTED: a deterministic
//! sample of entries is moved into a different cell. The perturbation is exact,
//! which is the one advantage this has over every embedding test in this track —
//! there is no argument about what the right answer is.
//!
//! ## The bar it has to clear
//!
//! A structural audit is only interesting if it beats the geometric detector we
//! already have. So the injected entries are also scored by plain distance to
//! their cell centroid, and the question is not "does the audit work" but "does
//! it add anything over cosine". Same discipline as Iterations 34-37.
//!
//! Scorers:
//!   NON-LATTICE   fraction of an entry's same-cell pairs whose minimal common
//!                 ancestors in the WordNet noun DAG number != 1
//!   ANCESTRY      1 - mean Jaccard of the entry's ancestor set against its
//!                 cellmates' (structural, no lattice condition)
//!   COSINE        distance to the cell centroid — the baseline to beat
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment38_soundness_audit

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::{HashMap, HashSet};

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| w.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|w| w.len() > 2)
        .collect()
}

/// WordNet noun DAG: lemma -> synset offsets, and offset -> hypernym offsets.
fn load_wordnet(path: &std::path::Path) -> (HashMap<String, Vec<u32>>, HashMap<u32, Vec<u32>>) {
    let mut lemma: HashMap<String, Vec<u32>> = HashMap::new();
    let mut hyper: HashMap<u32, Vec<u32>> = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else { return (lemma, hyper) };
    for line in text.lines() {
        if line.starts_with("  ") || line.trim().is_empty() {
            continue;
        }
        let head = line.split(" | ").next().unwrap_or("");
        let f: Vec<&str> = head.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        let Ok(off) = f[0].parse::<u32>() else { continue };
        let Ok(wcnt) = usize::from_str_radix(f[3], 16) else { continue };
        // words start at index 4, each is (word, lex_id)
        for k in 0..wcnt {
            if let Some(w) = f.get(4 + k * 2) {
                lemma.entry(w.replace('_', " ").to_lowercase()).or_default().push(off);
            }
        }
        // pointers follow the word list: p_cnt then p_cnt * (sym, off, pos, st)
        let pstart = 4 + wcnt * 2;
        if let Some(pc) = f.get(pstart).and_then(|s| s.parse::<usize>().ok()) {
            for p in 0..pc {
                let b = pstart + 1 + p * 4;
                if let (Some(sym), Some(o), Some(pos)) = (f.get(b), f.get(b + 1), f.get(b + 2)) {
                    // "@" is hypernym, "@i" instance hypernym; nouns only.
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

/// Transitive hypernym closure, depth-capped. WordNet's noun DAG is acyclic but
/// the cap also bounds cost and keeps very abstract roots (entity, abstraction)
/// from dominating every ancestor set.
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

/// Minimal elements of a common-ancestor set: those with no other member of the
/// set strictly below them. |MCA| != 1 is the non-lattice condition.
fn minimal(ca: &HashSet<u32>, hyper: &HashMap<u32, Vec<u32>>) -> usize {
    if ca.len() <= 1 {
        return ca.len();
    }
    // x is non-minimal if some other member of ca is a descendant of x, i.e.
    // x is reachable upward from that member.
    let mut count = 0;
    for &x in ca {
        let mut dominated = false;
        for &y in ca {
            if x == y {
                continue;
            }
            // y strictly below x?
            if ancestors(&[y], hyper, 12).contains(&x) {
                dominated = true;
                break;
            }
        }
        if !dominated {
            count += 1;
        }
    }
    count
}

/// Area under ROC for `score` separating flagged (true) from clean (false).
fn auc(scored: &[(f64, bool)]) -> f64 {
    let mut v: Vec<&(f64, bool)> = scored.iter().collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (pos, neg) = (v.iter().filter(|x| x.1).count(), v.iter().filter(|x| !x.1).count());
    if pos == 0 || neg == 0 {
        return 0.5;
    }
    // rank-sum with ties averaged
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1].0 == v[i].0 {
            j += 1;
        }
        let avg = ((i + 1 + j + 1) as f64) / 2.0;
        for x in &v[i..=j] {
            if x.1 {
                rank_sum += avg;
            }
        }
        i = j + 1;
    }
    (rank_sum - (pos * (pos + 1)) as f64 / 2.0) / (pos * neg) as f64
}

fn main() {
    println!("Experiment 38: a soundness audit — the half `coverage` does not have\n");

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
        println!("WordNet noun DAG: {} lemmas, {} synsets with hypernyms", lemma.len(), hyper.len());

        let Some(dir) = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists()) else {
            println!("WARNING: MiniLM not available — aborting.");
            return;
        };
        let embedder = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 384, model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean, ..OnnxConfig::default()
        });
        if !embedder.is_available() {
            println!("WARNING: embedder unavailable — aborting.");
            return;
        }

        // MODE ANCHORS: config/mode_anchors_ontology.json holds exactly one
        // entry per (domain, mode) cell, with hand-authored hints defining what
        // that cell MEANS. This is the controlled vocabulary item 34 found
        // missing — informative about domain x mode by construction, and
        // authored separately from the 957 corpus entries.
        //
        // The anchors are themselves loaded by OntologyLoader, so they are
        // excluded from the tested entries below; otherwise each anchor would
        // match its own cell perfectly and inflate the result.
        let anchor_path = ["physis-core/config", "config", "../config"]
            .iter()
            .map(|d| std::path::Path::new(d).join("mode_anchors_ontology.json"))
            .find(|p| p.exists());
        let mut anchors: HashMap<(String, String), Vec<String>> = HashMap::new();
        let mut anchor_names: HashSet<String> = HashSet::new();
        if let Some(ap) = &anchor_path {
            if let Ok(txt) = std::fs::read_to_string(ap) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    for e in v["domains"].as_array().into_iter().flatten() {
                        let (Some(d), Some(m), Some(nm)) =
                            (e["domain"].as_str(), e["mode"].as_str(), e["name"].as_str())
                        else { continue };
                        let mut toks: Vec<String> = words(nm);
                        for h in e["hints"].as_array().into_iter().flatten() {
                            if let Some(hs) = h.as_str() { toks.extend(words(hs)); }
                        }
                        toks.sort(); toks.dedup();
                        anchors.insert((d.to_string(), m.to_string()), toks);
                        anchor_names.insert(nm.to_string());
                    }
                }
            }
        }
        println!("mode anchors: {} cells with an authored vocabulary", anchors.len());

        // AUTHORED CELL VOCABULARY (scripts/gen_cell_vocab.py). Built from a
        // deterministic TRAIN half of each cell and scored only on the held-out
        // half, so a cell's vocabulary is never tested against the entries it
        // was derived from. `frequency` is the no-LLM arm; `generated` is
        // qwen3:4b's, present only if that arm ran.
        let mut vocab_freq: HashMap<(String, String), Vec<String>> = HashMap::new();
        let mut vocab_gen: HashMap<(String, String), Vec<String>> = HashMap::new();
        let mut test_names: HashSet<String> = HashSet::new();
        let vocab_paths: Vec<String> = match std::env::var("PHYSIS_VOCAB") {
            Ok(v) => vec![v],
            Err(_) => ["../research/perspective-discovery/cell_vocab_A.json", "cell_vocab_A.json"]
                .iter().map(|s| s.to_string()).collect(),
        };
        for vp in vocab_paths {
            let Ok(txt) = std::fs::read_to_string(&vp) else { continue };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else { continue };
            let mut take = |key: &str, into: &mut HashMap<(String, String), Vec<String>>| {
                for (k, arr) in v[key].as_object().into_iter().flatten() {
                    let Some((d, m)) = k.split_once('/') else { continue };
                    let mut t: Vec<String> = arr.as_array().into_iter().flatten()
                        .filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
                    t.sort(); t.dedup();
                    into.insert((d.to_string(), m.to_string()), t);
                }
            };
            take("frequency", &mut vocab_freq);
            take("generated", &mut vocab_gen);
            for (_, arr) in v["test_entries"].as_object().into_iter().flatten() {
                for x in arr.as_array().into_iter().flatten() {
                    if let Some(n) = x.as_str() { test_names.insert(n.to_string()); }
                }
            }
            println!(
                "cell vocabulary: {} freq cells, {} generated cells, {} held-out entries ({vp})",
                vocab_freq.len(), vocab_gen.len(), test_names.len()
            );
            break;
        }

        let ontology = OntologyLoader::load_all();
        let (mut texts, mut true_cell, mut names) = (Vec::new(), Vec::new(), Vec::new());
        // Structured fields, for the DEPENDENCY-CONSTRAINT scorer. Measured
        // first (cardinality-matched shuffle control): axis_name and category
        // carry real signal about DOMAIN (+0.140, +0.111 purity over a
        // shuffle) and essentially none about MODE (+0.041, +0.023). So a
        // constraint is statable for half the grid, and this tests whether
        // that half is worth anything against the geometric baseline.
        let mut fields: Vec<(String, String)> = Vec::new();
        let mut entry_toks: Vec<Vec<String>> = Vec::new();
        let mut entry_names: Vec<String> = Vec::new();
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut t = def.name.clone();
            for h in &def.hints { t.push(' '); t.push_str(h); }
            if anchor_names.contains(&def.name) { continue }
            entry_toks.push({
                let mut t = words(&def.name);
                for h in &def.hints { t.extend(words(h)); }
                t.sort(); t.dedup(); t
            });
            entry_names.push(def.name.clone());
            names.push(words(&def.name));
            fields.push((
                def.axis_name.clone().unwrap_or_default(),
                def.category.clone().unwrap_or_default(),
            ));
            texts.push(t);
            true_cell.push((d.clone(), m.clone()));
        }
        let n = texts.len();
        println!("{n} entries, embedding...");
        let emb: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

        // Ancestor set per entry, from its NAME terms' synsets.
        let anc: Vec<HashSet<u32>> = (0..n)
            .map(|i| {
                let mut seed: Vec<u32> = Vec::new();
                for w in &names[i] {
                    if let Some(offs) = lemma.get(w) {
                        seed.extend(offs.iter().take(2)); // first two senses
                    }
                }
                if seed.is_empty() { HashSet::new() } else { ancestors(&seed, &hyper, 6) }
            })
            .collect();
        let anchored = anc.iter().filter(|a| !a.is_empty()).count();
        println!(
            "entries anchored into the DAG: {anchored}/{n} ({:.0}%), mean {} ancestors\n",
            100.0 * anchored as f64 / n as f64,
            anc.iter().map(|a| a.len()).sum::<usize>() / anchored.max(1)
        );

        // INJECT: deterministically misfile every 7th anchored entry into a
        // different cell. Exact ground truth, no argument about the answer.
        let cell_names: Vec<(String, String)> = {
            let mut v: Vec<(String, String)> = true_cell.clone();
            v.sort();
            v.dedup();
            v
        };
        // Injection parameters. The published run is stride 7, mult 13 — these
        // env vars exist so the SAME corpus can be perturbed many different
        // ways, which is the only way to tell a real scorer margin from one
        // subsample's luck.
        let ev = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let stride = ev("PHYSIS_AUDIT_STRIDE", 7);
        let mult = ev("PHYSIS_AUDIT_MULT", 13);
        let half = std::env::var("PHYSIS_AUDIT_HALF").unwrap_or_else(|_| "test".into());
        let mut cell = true_cell.clone();
        let mut injected: Vec<bool> = vec![false; n];
        let mut k = 0usize;
        for i in 0..n {
            if anc[i].is_empty() { continue }
            k += 1;
            if stride != 0 && k % stride == 0 {
                // move to a deterministically-chosen DIFFERENT cell
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
        let n_inj = injected.iter().filter(|x| **x).count();
        println!("injected misfilings: {n_inj} of {anchored} anchored entries\n");

        // Cell membership under the PERTURBED assignment — what an auditor sees.
        let mut members: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for (i, c) in cell.iter().enumerate() {
            members.entry(c.clone()).or_default().push(i);
        }

        let mut rows: Vec<(f64, f64, f64, bool, f64, f64, f64, f64)> = Vec::new(); // nonlattice, ancestry, cosine, injected, constraint, anchor_vocab
        let (mut stat_ca, mut stat_mca): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        for i in 0..n {
            if anc[i].is_empty() { continue }
            // Only the held-out half. A cell's vocabulary was built from the
            // other half, so scoring a train entry would be self-confirming.
            let in_test = test_names.contains(&entry_names[i]);
            match half.as_str() {
                "test" if !test_names.is_empty() && !in_test => continue,
                "train" if !test_names.is_empty() && in_test => continue,
                _ => {}
            }
            let peers: Vec<usize> = members[&cell[i]].iter().copied().filter(|&j| j != i && !anc[j].is_empty()).collect();
            if peers.len() < 2 { continue }

            // NON-LATTICE, stated correctly: |MCA| > 1. The first version used
            // |MCA| != 1, which lumps "no common ancestor at all" (|CA| = 0)
            // together with "more than one minimal common ancestor". Those are
            // opposite situations — the first means unrelated, the second is the
            // actual lattice violation — and conflating them makes the score
            // mostly a detector of unrelatedness.
            let mut nl = 0usize;
            let mut orphan = 0usize;
            let mut considered = 0usize;
            for &j in peers.iter().take(12) {
                let ca: HashSet<u32> = anc[i].intersection(&anc[j]).copied().collect();
                considered += 1;
                let m = minimal(&ca, &hyper);
                if m > 1 { nl += 1; }
                if ca.is_empty() { orphan += 1; }
                stat_ca.push(ca.len());
                stat_mca.push(m);
            }
            let nonlattice = nl as f64 / considered.max(1) as f64;
            let orphan_rate = orphan as f64 / considered.max(1) as f64;

            // ANCESTRY: 1 - mean Jaccard against cellmates
            let mut jsum = 0.0;
            for &j in peers.iter().take(12) {
                let inter = anc[i].intersection(&anc[j]).count() as f64;
                let uni = anc[i].union(&anc[j]).count() as f64;
                jsum += if uni > 0.0 { inter / uni } else { 0.0 };
            }
            let ancestry = 1.0 - jsum / peers.len().min(12) as f64;

            // COSINE: distance to the cell centroid, computed WITHOUT the entry
            // itself so a misfiled item cannot drag its own target toward it.
            let mut acc = vec![0.0f32; 384];
            for &j in &peers { for d in 0..384 { acc[d] += emb[j][d]; } }
            let ctr = normalize(&acc);
            let cosine = 1.0 - cosine_sim(&emb[i], &ctr) as f64;

            // DEPENDENCY CONSTRAINT: is the entry's axis_name / category
            // attested among the other members of its cell's DOMAIN? Learned
            // from the PERTURBED assignment, excluding the entry itself, so it
            // gets exactly the information cosine gets.
            let dom = &cell[i].0;
            let (mut axis_seen, mut cat_seen) = (false, false);
            for j in 0..n {
                if j == i || &cell[j].0 != dom { continue }
                if !fields[i].0.is_empty() && fields[j].0 == fields[i].0 { axis_seen = true; }
                if !fields[i].1.is_empty() && fields[j].1 == fields[i].1 { cat_seen = true; }
                if axis_seen && cat_seen { break }
            }
            let constraint = (!axis_seen) as u8 as f64 + (!cat_seen) as u8 as f64;

            // ANCHOR-VOCAB: purely LEXICAL overlap between the entry and its
            // cell's authored anchor vocabulary. No embedder. This is the
            // controlled-vocabulary constraint done properly, where axis_name
            // (553 free-text values) could not be. Fewer shared markers = more
            // suspect filing, so the score is negated.
            let anchor_vocab = match anchors.get(&cell[i]) {
                Some(av) => {
                    let (mut x, mut y, mut hit) = (0usize, 0usize, 0usize);
                    while x < entry_toks[i].len() && y < av.len() {
                        match entry_toks[i][x].cmp(&av[y]) {
                            std::cmp::Ordering::Less => x += 1,
                            std::cmp::Ordering::Greater => y += 1,
                            std::cmp::Ordering::Equal => { hit += 1; x += 1; y += 1; }
                        }
                    }
                    -(hit as f64)
                }
                None => 0.0,
            };
            // VOCAB scorers: lexical overlap with the cell's AUTHORED vocabulary.
            // Fewer shared markers = more suspect filing, hence negated.
            let overlap = |v: &HashMap<(String, String), Vec<String>>| -> f64 {
                match v.get(&cell[i]) {
                    Some(av) if !av.is_empty() => {
                        let (mut x, mut y, mut hit) = (0usize, 0usize, 0usize);
                        while x < entry_toks[i].len() && y < av.len() {
                            match entry_toks[i][x].cmp(&av[y]) {
                                std::cmp::Ordering::Less => x += 1,
                                std::cmp::Ordering::Greater => y += 1,
                                std::cmp::Ordering::Equal => { hit += 1; x += 1; y += 1; }
                            }
                        }
                        -(hit as f64)
                    }
                    _ => 0.0,
                }
            };
            let vfreq = overlap(&vocab_freq);
            let vgen = overlap(&vocab_gen);
            let _ = orphan_rate;

            rows.push((nonlattice, ancestry, cosine, injected[i], constraint, anchor_vocab, vfreq, vgen));
        }
        println!("scored {} entries ({} injected)\n", rows.len(), rows.iter().filter(|r| r.3).count());
        // Is the lattice condition even firing? If almost every pair has no
        // common ancestor, |MCA|>1 can never discriminate and the whole method
        // is inapplicable rather than merely unhelpful.
        let np = stat_ca.len().max(1);
        println!("  same-cell pair diagnostics over {np} pairs:");
        println!("    |CA| = 0 (no common ancestor): {:.1}%", 100.0 * stat_ca.iter().filter(|&&c| c == 0).count() as f64 / np as f64);
        println!("    |MCA| = 1 (lattice, fine)    : {:.1}%", 100.0 * stat_mca.iter().filter(|&&m| m == 1).count() as f64 / np as f64);
        println!("    |MCA| > 1 (NON-LATTICE)      : {:.1}%", 100.0 * stat_mca.iter().filter(|&&m| m > 1).count() as f64 / np as f64);
        println!();

        let a_nl = auc(&rows.iter().map(|r| (r.0, r.3)).collect::<Vec<_>>());
        let a_an = auc(&rows.iter().map(|r| (r.1, r.3)).collect::<Vec<_>>());
        let a_co = auc(&rows.iter().map(|r| (r.2, r.3)).collect::<Vec<_>>());
        let a_cn = auc(&rows.iter().map(|r| (r.4, r.3)).collect::<Vec<_>>());
        let a_av = auc(&rows.iter().map(|r| (r.5, r.3)).collect::<Vec<_>>());
        let a_vf = auc(&rows.iter().map(|r| (r.6, r.3)).collect::<Vec<_>>());
        let a_vg = auc(&rows.iter().map(|r| (r.7, r.3)).collect::<Vec<_>>());

        println!("=== Can each scorer find the injected misfilings? (AUC, 0.5 = chance) ===\n");
        println!("  NON-LATTICE (structural, label-free)   AUC = {a_nl:.3}");
        println!("  ANCESTRY    (structural, label-free)   AUC = {a_an:.3}");
        println!("  CONSTRAINT  (axis_name/category unattested) AUC = {a_cn:.3}");
        println!("  ANCHOR-VOCAB (lexical, authored per cell)   AUC = {a_av:.3}");
        println!("  VOCAB-FREQ  (authored from TRAIN half, no LLM) AUC = {a_vf:.3}");
        if !vocab_gen.is_empty() {
            println!("  VOCAB-GEN   (qwen3:4b, same inputs)         AUC = {a_vg:.3}");
        }
        println!("  COSINE      (geometric baseline)       AUC = {a_co:.3}");

        // The question that matters: does structure add anything over cosine?
        // Stratify by cosine and ask whether the structural scores still
        // separate injected from clean WITHIN a stratum.
        println!(
            "SWEEP stride={stride} mult={mult} half={half} n={} inj={} constraint={a_cn:.3} cosine={a_co:.3} vfreq={a_vf:.3} delta={:+.3}",
            rows.len(), rows.iter().filter(|r| r.3).count(), a_cn - a_co
        );
        println!("\n=== Does structure add anything BEYOND cosine? ===\n");
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by(|&a, &b| rows[a].2.partial_cmp(&rows[b].2).unwrap());
        const STRATA: usize = 8;
        let per = order.len() / STRATA;
        for (label, pick) in [("NON-LATTICE", 0usize), ("ANCESTRY   ", 1)] {
            let mut within: Vec<(f64, bool)> = Vec::new();
            for s in 0..STRATA {
                let lo = s * per;
                let hi = if s == STRATA - 1 { order.len() } else { (s + 1) * per };
                // rank within stratum, so cosine level cannot carry the signal
                let mut idx: Vec<usize> = order[lo..hi].to_vec();
                let val = |k: usize| if pick == 0 { rows[k].0 } else { rows[k].1 };
                idx.sort_by(|&a, &b| val(a).partial_cmp(&val(b)).unwrap());
                for (r, &k) in idx.iter().enumerate() {
                    within.push((r as f64 / idx.len().max(1) as f64, rows[k].3));
                }
            }
            println!("  {label} within cosine strata:  AUC = {:.3}", auc(&within));
        }

        // DIAGNOSTIC: is physis's grid taxonomic at all? If real cells are no
        // more lattice-coherent than an arbitrary grouping of the same sizes,
        // then a kind-hierarchy has nothing to say about them and no amount of
        // tuning this audit will help — the grid groups by FUNCTION, not KIND.
        println!("\n=== Is the 5x14 grid taxonomically coherent at all? ===\n");
        let coherence = |assign: &Vec<(String, String)>| -> (f64, f64) {
            let mut mem: HashMap<(String, String), Vec<usize>> = HashMap::new();
            for i in 0..n {
                if !anc[i].is_empty() { mem.entry(assign[i].clone()).or_default().push(i); }
            }
            let (mut lat, mut tot, mut jac, mut jn) = (0usize, 0usize, 0.0f64, 0usize);
            for peers in mem.values() {
                for a in 0..peers.len().min(14) {
                    for b in (a + 1)..peers.len().min(14) {
                        let (i, j) = (peers[a], peers[b]);
                        let ca: HashSet<u32> = anc[i].intersection(&anc[j]).copied().collect();
                        if minimal(&ca, &hyper) == 1 { lat += 1; }
                        tot += 1;
                        let uni = anc[i].union(&anc[j]).count() as f64;
                        if uni > 0.0 { jac += ca.len() as f64 / uni; jn += 1; }
                    }
                }
            }
            (lat as f64 / tot.max(1) as f64, jac / jn.max(1) as f64)
        };
        let (real_lat, real_jac) = coherence(&true_cell);
        // Size-matched arbitrary grouping: same cells, membership permuted.
        let mut shuffled = true_cell.clone();
        for i in 0..n { shuffled[i] = true_cell[(i * 37 + 11) % n].clone(); }
        let (rand_lat, rand_jac) = coherence(&shuffled);
        println!("                        real cells   size-matched shuffle");
        println!("  pairs with |MCA| = 1     {real_lat:.3}          {rand_lat:.3}");
        println!("  mean ancestor Jaccard    {real_jac:.3}          {rand_jac:.3}");
        let taxonomic = (real_jac - rand_jac).abs() > 0.02 || (real_lat - rand_lat).abs() > 0.05;
        println!(
            "  -> the grid is {} taxonomically structured",
            if taxonomic { "measurably" } else { "NOT" }
        );

        println!("\n=== Verdict ===\n");
        let best_struct = a_nl.max(a_an).max(a_cn).max(a_av).max(a_vf).max(a_vg);
        if best_struct > a_co + 0.03 {
            println!("  The structural audit BEATS the geometric baseline outright");
            println!("  ({best_struct:.3} vs {a_co:.3}). A label-free soundness check is available.");
        } else if best_struct > 0.55 {
            println!("  The structural audit finds injected misfilings ({best_struct:.3}) but does");
            println!("  NOT beat plain cosine ({a_co:.3}). It is a real signal that is already");
            println!("  contained in the geometry — not the missing soundness half.");
        } else {
            println!("  The structural audit does NOT find injected misfilings ({best_struct:.3},");
            println!("  chance = 0.500) while cosine does ({a_co:.3}). Non-lattice auditing does");
            println!("  not transfer to physis's flat 5x14 grid: it detects missing or redundant");
            println!("  IS-A links in a deep taxonomy, and a misfiled entry in a two-level facet");
            println!("  system is simply not that kind of defect.");
        }
        println!(
            "\n(What is honest about this test and was not about the embedding ones: the ground\n truth is INJECTED, so there is no argument over what the right answer is. What it\n still cannot tell you: whether physis's REAL cell assignments contain misfilings\n of the same shape as the injected ones. A synthetic perturbation is a lower bound\n on difficulty — real misfilings are the ones a careful author already thought\n were right.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
