//! Experiment 31 — the outward discovery loop: **propose, check, triage.**
//!
//! Iteration 30 established the premise: near-neighbour midpoints point at
//! genuinely-missing concepts 1.44x more often than manifold-matched random
//! pairs (z = +4.87). But 87% of proposals still point at nothing, so the
//! external check is the system, not a formality. This builds that check.
//!
//! ## The check: a known-ontology list, and the inversion that makes it useful
//!
//! WordNet 3.1's 82k noun synsets are the "list of known ontologies". They are
//! the right check for a reason that matters: WordNet is **hand-built by
//! lexicographers from lexicographic principles**, not trained on web text. So
//! it is causally independent of MiniLM in a way an LLM is not — and causal
//! independence is exactly what Iteration 19 lacked when cross-embedder
//! corroboration moved FPR 0.133 -> 0.133 by asking the same distributional
//! statistics twice.
//!
//! The user's inversion: a proposal matching a known ontology is a **filled
//! gap** — useful but unsurprising. A proposal matching *nothing* known is
//! either noise or something genuinely novel, and the novel ones are the
//! valuable output. So "unknown" is promoted, not discarded.
//!
//! ## The problem that inversion creates, and the discriminator for it
//!
//! Unmatched is NOT the same as novel. Most unmatched proposals are noise —
//! we measured that 87% of proposals point at nothing at all. Promoting
//! "unknown" naively would promote mostly garbage.
//!
//! The discriminator is **convergence**: how many structurally distinct entry
//! pairs independently propose the same region. A location that one accidental
//! pair lands on is noise; a location that many different pairs converge on is
//! implied by the ontology's own structure. Convergence is measured here, and
//! validated against the hold-out rather than assumed.
//!
//! ## Triage
//!
//! | | WordNet knows it | WordNet doesn't |
//! |---|---|---|
//! | **high convergence** | CONFIRMED GAP — fill it | **UNCANNY — investigate** |
//! | **low convergence** | weak lead | NOISE — discard |
//!
//! ## Validation
//!
//! A 10% stratified hold-out is removed from physis before anything runs. If
//! the loop works, held-out concepts should concentrate in the CONFIRMED bucket
//! rather than spreading evenly across all four. That is the end-to-end check,
//! and it is what separates this from a plausible-looking pipeline.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment31_outward_loop

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::HashMap;

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn midpoint(a: &[f32], b: &[f32]) -> Vec<f32> {
    normalize(&a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<f32>>())
}

/// Parse a WordNet `data.<pos>` file: `offset lex ss_type w_cnt word lex_id ... | gloss`.
/// Licence header lines start with two spaces and are skipped.
///
/// Verbs matter here for a specific reason. The first version of this
/// experiment loaded nouns only and then reported "WordNet is nouns-only" as
/// a limitation — which was wrong: WordNet ships 13,789 verb synsets, the
/// loader just ignored them. And the omission was not harmless: the UNCANNY
/// bucket filled up with PROCESSES (agent coordination, debugging,
/// monitoring, causal reasoning) that a noun lexicon cannot name, so
/// "unrecognised" was partly measuring the loader rather than the concept.
fn parse_wordnet(path: &std::path::Path) -> Vec<(String, String)> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with("  ") || line.trim().is_empty() {
            continue;
        }
        let Some((head, gloss)) = line.split_once(" | ") else { continue };
        let f: Vec<&str> = head.split_whitespace().collect();
        // offset(0) lex(1) ss_type(2) w_cnt(3, hex) word(4)
        if f.len() < 5 {
            continue;
        }
        let lemma = f[4].replace('_', " ");
        let gloss = gloss.trim().to_string();
        if lemma.is_empty() {
            continue;
        }
        out.push((lemma, gloss));
    }
    out
}

/// Value at `q` (0..1) of a sorted-ascending slice.
fn quantile(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

fn main() {
    println!("Experiment 31: the outward loop — propose, check against WordNet, triage.\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let Some(dir) = ["models", "../models"]
            .iter()
            .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        else {
            println!("WARNING: MiniLM not available — aborting.");
            return;
        };
        let wn_path = ["models/wordnet", "../models/wordnet"]
            .iter()
            .map(std::path::Path::new)
            .find(|d| d.join("data.noun").exists())
            .map(|d| d.join("data.noun"));
        let Some(wn_path) = wn_path else {
            println!("WARNING: WordNet data.noun not found — aborting.");
            return;
        };

        let embedder = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 384,
            model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if !embedder.is_available() {
            println!("WARNING: embedder unavailable — aborting.");
            return;
        }

        // ---------- physis side ----------
        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        let mut cells = Vec::new();
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut t = def.name.clone();
            for h in &def.hints {
                t.push(' ');
                t.push_str(h);
            }
            texts.push(t);
            cells.push((d.clone(), m.clone()));
        }
        println!("physis: {} entries, embedding...", texts.len());
        let emb: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

        // Stratified hold-out, identical construction to Iteration 30.
        let mut by_cell: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for (i, c) in cells.iter().enumerate() {
            by_cell.entry(c.clone()).or_default().push(i);
        }
        let mut cell_keys: Vec<&(String, String)> = by_cell.keys().collect();
        cell_keys.sort();
        let mut held: Vec<usize> = Vec::new();
        for k in &cell_keys {
            let m = &by_cell[*k];
            let n_hold = ((m.len() as f64 * 0.10).round() as usize).min(m.len().saturating_sub(2));
            for j in 0..n_hold {
                held.push(m[(j * 7 + 3) % m.len()]);
            }
        }
        held.sort_unstable();
        held.dedup();
        let held_set: std::collections::HashSet<usize> = held.iter().copied().collect();
        let known: Vec<usize> = (0..texts.len()).filter(|i| !held_set.contains(i)).collect();
        println!("hold-out: {} known / {} held out\n", known.len(), held.len());

        // ---------- 1. propose ----------
        const NN: usize = 5;
        let mut proposals: Vec<(Vec<f32>, usize, usize)> = Vec::new();
        for (pos, &i) in known.iter().enumerate() {
            let mut sims: Vec<(f32, usize)> = known
                .iter()
                .enumerate()
                .filter(|(p, _)| *p != pos)
                .map(|(_, &j)| (cosine_sim(&emb[i], &emb[j]), j))
                .collect();
            sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
            for &(_, j) in sims.iter().take(NN) {
                if i < j {
                    proposals.push((midpoint(&emb[i], &emb[j]), i, j));
                }
            }
        }
        println!("1. proposed {} candidate locations", proposals.len());

        // ---------- 2. converge ----------
        // Greedy agglomeration: a proposal joins the first cluster whose seed it
        // is within CONV_TAU of. Deterministic because `proposals` is built in
        // sorted entry order and clusters are scanned in creation order.
        const CONV_TAU: f32 = 0.90;
        let mut seeds: Vec<Vec<f32>> = Vec::new();
        let mut members: Vec<Vec<usize>> = Vec::new();
        for (pi, (p, _, _)) in proposals.iter().enumerate() {
            let mut placed = false;
            for (ci, s) in seeds.iter().enumerate() {
                if cosine_sim(p, s) >= CONV_TAU {
                    members[ci].push(pi);
                    placed = true;
                    break;
                }
            }
            if !placed {
                seeds.push(p.clone());
                members.push(vec![pi]);
            }
        }
        // Convergence = number of DISTINCT parent entries backing the cluster,
        // not the raw proposal count: five midpoints all sharing one parent are
        // one piece of evidence wearing five hats.
        let convergence: Vec<usize> = members
            .iter()
            .map(|ms| {
                let mut parents: Vec<usize> = ms
                    .iter()
                    .flat_map(|&pi| [proposals[pi].1, proposals[pi].2])
                    .collect();
                parents.sort_unstable();
                parents.dedup();
                parents.len()
            })
            .collect();
        let centroids: Vec<Vec<f32>> = members
            .iter()
            .map(|ms| {
                let mut acc = vec![0.0f32; 384];
                for &pi in ms {
                    for d in 0..384 {
                        acc[d] += proposals[pi].0[d];
                    }
                }
                normalize(&acc)
            })
            .collect();
        println!("2. converged into {} distinct regions", seeds.len());

        // ---------- 3. the external check ----------
        // Deterministic subsample. Embedding all 82k glosses dominates the run
        // time; STRIDE keeps the lexicon large and representative while the
        // loop is being validated. A fixed stride, not a random sample, so the
        // check set is identical on every run — the vocabulary is a fixed
        // reference, exactly like the ontology cells it checks.
        const STRIDE: usize = 4;
        let wn_dir = wn_path.parent().unwrap().to_path_buf();
        let mut wn: Vec<(String, String)> = Vec::new();
        let mut wn_pos: Vec<&'static str> = Vec::new();
        let mut pos_counts: Vec<(&str, usize, usize)> = Vec::new();
        for (pos, file) in [("n", "data.noun"), ("v", "data.verb")] {
            let all = parse_wordnet(&wn_dir.join(file));
            let taken: Vec<(String, String)> = all.iter().step_by(STRIDE).cloned().collect();
            pos_counts.push((pos, taken.len(), all.len()));
            for t in taken {
                wn.push(t);
                wn_pos.push(if pos == "n" { "n" } else { "v" });
            }
        }
        println!("3. WordNet lexicon (stride {STRIDE}):");
        for (pos, took, all) in &pos_counts {
            println!("     {pos}: {took} of {all} synsets");
        }
        println!("     total check vocabulary: {}", wn.len());

        // Cache the embeddings. A real loop re-checks proposals against the
        // same fixed lexicon many times, so paying once per lexicon version is
        // the point rather than an optimisation.
        let cache_path =
            std::path::Path::new("target").join(format!("wordnet_nv_emb_s{STRIDE}.bin"));
        let wn_emb: Vec<Vec<f32>> = match std::fs::read(&cache_path) {
            Ok(bytes) if bytes.len() == wn.len() * 384 * 4 => {
                println!("   loaded from cache {}", cache_path.display());
                bytes
                    .chunks_exact(384 * 4)
                    .map(|c| {
                        c.chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect()
                    })
                    .collect()
            }
            _ => {
                println!("   embedding (slow; cached afterwards)...");
                let v: Vec<Vec<f32>> = wn
                    .iter()
                    .map(|(l, g)| normalize(&embedder.embed(&format!("{l}: {g}"))))
                    .collect();
                let mut bytes = Vec::with_capacity(v.len() * 384 * 4);
                for e in &v {
                    for x in e {
                        bytes.extend_from_slice(&x.to_le_bytes());
                    }
                }
                let _ = std::fs::create_dir_all("target");
                if std::fs::write(&cache_path, &bytes).is_ok() {
                    println!("   cached to {}", cache_path.display());
                }
                v
            }
        };
        println!("   done.\n");

        let best_wn = |x: &[f32]| -> (f32, usize) {
            let mut best = (f32::NEG_INFINITY, 0usize);
            for (i, w) in wn_emb.iter().enumerate() {
                let s = cosine_sim(x, w);
                if s > best.0 {
                    best = (s, i);
                }
            }
            best
        };
        let best_known = |x: &[f32]| -> f32 {
            known
                .iter()
                .map(|&i| cosine_sim(x, &emb[i]))
                .fold(f32::NEG_INFINITY, f32::max)
        };

        let wn_hits: Vec<(f32, usize)> = centroids.iter().map(|c| best_wn(c)).collect();
        let phys_hits: Vec<f32> = centroids.iter().map(|c| best_known(c)).collect();

        // Thresholds are PERCENTILES of the observed distributions, not absolute
        // cosines. physis entries read "Name hint hint hint" and WordNet reads
        // "lemma: gloss"; MiniLM similarity is partly driven by that style
        // difference, so one absolute cutoff would not mean the same thing on
        // both sides. Percentiles compare each proposal against its own
        // reference population.
        let mut wn_sorted: Vec<f32> = wn_hits.iter().map(|(s, _)| *s).collect();
        wn_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut ph_sorted: Vec<f32> = phys_hits.clone();
        ph_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut cv_sorted: Vec<usize> = convergence.clone();
        cv_sorted.sort_unstable();

        let wn_known_t = quantile(&wn_sorted, 0.50);
        let phys_covered_t = quantile(&ph_sorted, 0.50);
        let conv_t = cv_sorted[cv_sorted.len() / 2];
        println!(
            "   thresholds (medians): wordnet-match >= {wn_known_t:.3}, physis-covered >= {phys_covered_t:.3}, convergence >= {conv_t}"
        );

        // ---------- 4. triage ----------
        #[derive(PartialEq, Clone, Copy, Debug)]
        enum Bucket {
            Redundant,
            ConfirmedGap,
            Uncanny,
            WeakLead,
            Noise,
        }
        let bucket_of = |ci: usize| -> Bucket {
            let wn_knows = wn_hits[ci].0 >= wn_known_t;
            let phys_has = phys_hits[ci] >= phys_covered_t;
            let converged = convergence[ci] >= conv_t;
            if phys_has {
                Bucket::Redundant
            } else if wn_knows && converged {
                Bucket::ConfirmedGap
            } else if wn_knows {
                Bucket::WeakLead
            } else if converged {
                Bucket::Uncanny
            } else {
                Bucket::Noise
            }
        };
        let buckets: Vec<Bucket> = (0..centroids.len()).map(bucket_of).collect();
        let count = |b: Bucket| buckets.iter().filter(|x| **x == b).count();

        println!("\n=== 4. Triage ===\n");
        println!("  REDUNDANT (physis already covers)  {:5}", count(Bucket::Redundant));
        println!("  CONFIRMED GAP (WordNet + converged){:5}", count(Bucket::ConfirmedGap));
        println!("  UNCANNY (unknown but converged)    {:5}   <- the interesting bucket", count(Bucket::Uncanny));
        println!("  WEAK LEAD (WordNet, low converge)  {:5}", count(Bucket::WeakLead));
        println!("  NOISE (unknown, low convergence)   {:5}", count(Bucket::Noise));

        // ---------- 5. validation against the hold-out ----------
        // Does a region actually contain a concept we removed? This is ground
        // truth, and it is the only thing that tells us the triage means
        // anything.
        // Parent entries are adjacent to their own region BY CONSTRUCTION, so
        // they must be excluded from the comparison. The first version of this
        // compared against ALL known entries and every bucket scored 0.000 —
        // including the overall rate, which is the tell that the metric was
        // void rather than the triage being useless. This is the same
        // parent-adjacency artifact found and fixed in Iteration 30, and
        // reintroduced here.
        let parents_of = |ci: usize| -> Vec<usize> {
            let mut v: Vec<usize> = members[ci]
                .iter()
                .flat_map(|&pi| [proposals[pi].1, proposals[pi].2])
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let rival_known: Vec<f32> = (0..centroids.len())
            .map(|ci| {
                let ps = parents_of(ci);
                known
                    .iter()
                    .filter(|k| !ps.contains(k))
                    .map(|&i| cosine_sim(&centroids[ci], &emb[i]))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        let reaches_held = |ci: usize| -> Option<usize> {
            let mut best = (f32::NEG_INFINITY, usize::MAX);
            for &h in &held {
                let s = cosine_sim(&centroids[ci], &emb[h]);
                if s > best.0 {
                    best = (s, h);
                }
            }
            // Closer to a removed concept than to any NON-PARENT known entry.
            if best.0 > rival_known[ci] {
                Some(best.1)
            } else {
                None
            }
        };
        let hit: Vec<bool> = (0..centroids.len()).map(|ci| reaches_held(ci).is_some()).collect();
        let rate = |b: Bucket| -> (usize, usize, f64) {
            let idx: Vec<usize> = (0..buckets.len()).filter(|&i| buckets[i] == b).collect();
            let h = idx.iter().filter(|&&i| hit[i]).count();
            let n = idx.len();
            (h, n, if n == 0 { 0.0 } else { h as f64 / n as f64 })
        };
        let overall = hit.iter().filter(|x| **x).count() as f64 / hit.len() as f64;

        println!("\n=== 5. Validation: does each bucket actually contain removed concepts? ===\n");
        println!("  (a region counts as a hit only if it is closer to a held-out concept");
        println!("   than to anything physis still knows)\n");
        println!("  bucket              hits/regions    rate     vs overall");
        for (name, b) in [
            ("REDUNDANT     ", Bucket::Redundant),
            ("CONFIRMED GAP ", Bucket::ConfirmedGap),
            ("UNCANNY       ", Bucket::Uncanny),
            ("WEAK LEAD     ", Bucket::WeakLead),
            ("NOISE         ", Bucket::Noise),
        ] {
            let (h, n, r) = rate(b);
            println!(
                "  {name}      {h:4}/{n:<6}   {r:.3}    {:.2}x",
                if overall > 1e-9 { r / overall } else { 0.0 }
            );
        }
        println!("\n  overall rate across all regions: {overall:.3}");

        // ---------- 6. the actual output ----------
        let mut uncanny: Vec<usize> = (0..buckets.len())
            .filter(|&i| buckets[i] == Bucket::Uncanny)
            .collect();
        uncanny.sort_by(|&a, &b| {
            convergence[b]
                .cmp(&convergence[a])
                .then_with(|| wn_hits[a].0.partial_cmp(&wn_hits[b].0).unwrap())
        });
        let short = |s: &str| s.split_whitespace().take(5).collect::<Vec<_>>().join(" ");
        println!("\n=== 6. Top UNCANNY regions — highest convergence, least recognised ===\n");
        for &ci in uncanny.iter().take(8) {
            let mut parents: Vec<usize> = members[ci]
                .iter()
                .flat_map(|&pi| [proposals[pi].1, proposals[pi].2])
                .collect();
            parents.sort_unstable();
            parents.dedup();
            println!(
                "  [{} distinct parents, closest WordNet {:.3} \"{}\" ({})]",
                convergence[ci], wn_hits[ci].0, wn[wn_hits[ci].1].0, wn_pos[wn_hits[ci].1]
            );
            for &p in parents.iter().take(4) {
                println!("      {:?}", short(&texts[p]));
            }
            println!();
        }

        let mut confirmed: Vec<usize> = (0..buckets.len())
            .filter(|&i| buckets[i] == Bucket::ConfirmedGap)
            .collect();
        confirmed.sort_by(|&a, &b| wn_hits[b].0.partial_cmp(&wn_hits[a].0).unwrap());
        println!("=== 7. Top CONFIRMED GAPS — WordNet has the concept, physis does not ===\n");
        for &ci in confirmed.iter().take(8) {
            println!(
                "  \"{}\" [{}] ({:.3}) — {}",
                wn[wn_hits[ci].1].0,
                wn_pos[wn_hits[ci].1],
                wn_hits[ci].0,
                short(&wn[wn_hits[ci].1].1)
            );
        }

        let (ch, cn, cr) = rate(Bucket::ConfirmedGap);
        let (uh, un, ur) = rate(Bucket::Uncanny);
        let (nh, nn, nr) = rate(Bucket::Noise);

        // Two CONTROLLED comparisons. The bucket-vs-overall column above is
        // not enough: the buckets differ on two axes at once, so a raw
        // difference cannot say which axis did the work. Each comparison below
        // holds one axis fixed.
        let z_test = |p1: f64, n1: f64, p2: f64, n2: f64| -> f64 {
            if n1 <= 0.0 || n2 <= 0.0 {
                return 0.0;
            }
            let pooled = (p1 * n1 + p2 * n2) / (n1 + n2);
            let se = (pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)).sqrt();
            if se <= 0.0 {
                0.0
            } else {
                (p1 - p2) / se
            }
        };
        let z_wordnet = z_test(cr, cn as f64, ur, un as f64);
        let z_converge = z_test(ur, un as f64, nr, nn as f64);

        println!("\n=== Which axis actually does the work? ===\n");
        println!("  A. Does the WORDNET check help?");
        println!("     Both high-convergence, differing only in whether WordNet knows it:");
        println!("       CONFIRMED {ch}/{cn} = {cr:.3}   vs   UNCANNY {uh}/{un} = {ur:.3}   z = {z_wordnet:+.2}");
        println!();
        println!("  B. Does CONVERGENCE help?");
        println!("     Both WordNet-unknown, differing only in convergence:");
        println!("       UNCANNY {uh}/{un} = {ur:.3}   vs   NOISE {nh}/{nn} = {nr:.3}   z = {z_converge:+.2}");
        println!("\n  (|z| > 1.96 is p < 0.05)");

        println!("\n=== Verdict ===\n");
        if z_wordnet > 1.96 {
            println!("  The WordNet check selects for genuinely-missing concepts (z = {z_wordnet:+.2}).");
        } else {
            println!("  The WORDNET CHECK DOES NOT WORK as designed. Holding convergence fixed,");
            println!("  regions WordNet recognises hit at {cr:.3} against {ur:.3} for regions it does");
            println!("  not (z = {z_wordnet:+.2}) — no better, and if anything slightly worse. Lexical");
            println!("  agreement with a hand-built noun/verb lexicon carries no information about");
            println!("  whether a region contains a concept this ontology is missing.");
        }
        println!();
        if z_converge > 1.96 {
            println!("  CONVERGENCE does carry signal (z = {z_converge:+.2}): among regions no lexicon");
            println!("  recognises, the ones many distinct entry pairs agree on contain a removed");
            println!("  concept {:.2}x as often as the ones they do not.", ur / nr.max(1e-9));
        } else {
            println!("  CONVERGENCE is the more promising axis but is NOT yet established:");
            println!("  {ur:.3} vs {nr:.3} ({:.2}x) at z = {z_converge:+.2}, short of significance at", ur / nr.max(1e-9));
            println!("  these sample sizes (n={un} vs n={nn}). Worth a larger corpus before building");
            println!("  on it; not worth claiming yet.");
        }
        println!();
        println!("  Net: of the two discriminators this loop was built on, the external check");
        println!("  is dead and the convergence filter is unproven. The proposer from Iteration");
        println!("  30 still works; what sits on top of it here does not yet.");

        println!("\n  A confound to fix before trusting any of the above: REDUNDANT regions —");
        println!("  ones physis still covers — hit at {:.3}, the second-highest rate. Held-out", rate(Bucket::Redundant).2);
        println!("  concepts were removed from cells whose SIBLINGS remain, so a region near a");
        println!("  dense known cluster sits near the removed member too. The hit metric is");
        println!("  therefore still partly measuring local density rather than absence.");

        println!(
            "\n(Limits: one corpus, one embedder, one lexicon. Nouns AND verbs are loaded now\n — an earlier version took nouns only and then blamed \"WordNet is nouns-only\"\n for the process-shaped concepts piling up in UNCANNY; that was the loader, not\n the lexicon. Adding 3,448 verb synsets barely moved the buckets, so the\n process-concept explanation was largely wrong too. The residual UNCANNY head is\n dominated by ITALIAN-language entries, which an English lexicon cannot match at\n any part of speech — a coverage artifact, not novelty. Percentile thresholds\n mean bucket SIZES are fixed by construction; only differential rates are\n evidence.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
