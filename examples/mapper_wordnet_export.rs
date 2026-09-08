//! E11 — export WordNet noun synsets as ontology-entry CANDIDATES.
//!
//! The expansion spec recorded a negative result: mining free prose for
//! ontology entries does not work, because an entry is a short domain concept
//! and prose contains none in extractable form. The stated fix was "a source
//! that already contains concept-shaped items — a glossary, an index, a
//! taxonomy". WordNet is exactly that and is already in `models/`.
//!
//! Each synset gives a lemma plus a gloss, which is the shape of a physis entry
//! (name plus hints). This exports them embedded, so the miner can rank them
//! against a cell's own members and anchor.
//!
//! ## The filter, and why it is not arbitrary
//!
//! Physis entries are processes, conditions, fields and acts — not concrete
//! objects. WordNet's lexicographer file id (field 2 of each data line) already
//! partitions nouns that way, so the export keeps the noun files that match
//! what an entry IS and drops animal/plant/food/artifact/person/location.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example mapper_wordnet_export

fn jesc(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '"' => "\\\"".to_string(),
            '\\' => "\\\\".to_string(),
            '\n' | '\r' | '\t' => " ".to_string(),
            c => c.to_string(),
        })
        .collect()
}

fn main() {
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
        let Some(dir) = ["models/bge-base-en-v1.5", "../models/bge-base-en-v1.5"]
            .iter()
            .find(|d| std::path::Path::new(d).join("onnx/model.onnx").exists())
        else {
            println!("WARNING: BGE not available — aborting.");
            return;
        };
        let e = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 768,
            model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if !e.is_available() {
            println!("WARNING: embedder unavailable — aborting.");
            return;
        }

        // WordNet lexicographer files for nouns. Keep the ones whose members
        // are the KIND of thing a physis entry is; drop concrete taxa.
        //  03 act, 04 animal, 05 artifact, 06 attribute, 07 body, 09 cognition,
        //  10 communication, 11 event, 12 feeling, 13 food, 14 group,
        //  15 location, 16 motive, 17 object, 18 person, 19 phenomenon,
        //  20 plant, 21 possession, 22 process, 23 quantity, 24 relation,
        //  25 shape, 26 state, 27 substance, 28 time
        const KEEP: [u32; 12] = [3, 6, 9, 10, 11, 12, 16, 19, 22, 24, 26, 28];

        let text = std::fs::read_to_string(wn.join("data.noun")).expect("read");
        let mut items: Vec<(String, String, u32)> = Vec::new();
        for line in text.lines() {
            if line.starts_with("  ") || line.trim().is_empty() {
                continue;
            }
            let Some((head, gloss)) = line.split_once(" | ") else {
                continue;
            };
            let f: Vec<&str> = head.split_whitespace().collect();
            if f.len() < 5 {
                continue;
            }
            let Ok(lex) = f[1].parse::<u32>() else {
                continue;
            };
            if !KEEP.contains(&lex) {
                continue;
            }
            let Ok(wcnt) = usize::from_str_radix(f[3], 16) else {
                continue;
            };
            let Some(w) = f.get(4) else { continue };
            let lemma = w.replace('_', " ");
            // skip multi-sense clutter and very long compounds
            if wcnt == 0 || lemma.len() < 3 || lemma.split_whitespace().count() > 3 {
                continue;
            }
            let g = gloss.split(';').next().unwrap_or(gloss).trim().to_string();
            if g.len() < 12 || g.len() > 220 {
                continue;
            }
            items.push((lemma, g, lex));
        }
        println!("{} candidate synsets after the lexfile filter", items.len());

        let cap: usize = std::env::var("MAPPER_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20000);
        if items.len() > cap {
            let stride = items.len() / cap + 1;
            items = items.into_iter().step_by(stride).collect();
            println!("strided to {} (cap {cap})", items.len());
        }

        let mut meta =
            String::from("{\n  \"source\": \"mapper_wordnet_export\",\n  \"synsets\": [\n");
        let mut vecs: Vec<f32> = Vec::with_capacity(items.len() * 768);
        for (k, (lemma, gloss, lex)) in items.iter().enumerate() {
            // "name + hints" shape, matching how corpus entries are embedded
            let v = e.embed(&format!("{lemma} {gloss}"));
            vecs.extend_from_slice(&v);
            if k > 0 {
                meta.push_str(",\n");
            }
            meta.push_str(&format!(
                "    {{\"i\": {k}, \"lemma\": \"{}\", \"gloss\": \"{}\", \"lex\": {lex}}}",
                jesc(lemma),
                jesc(gloss)
            ));
            if k % 2000 == 0 && k > 0 {
                println!("  {k}/{}", items.len());
            }
        }
        meta.push_str("\n  ]\n}\n");
        std::fs::create_dir_all("research/mapper").ok();
        std::fs::write("research/mapper/wordnet_candidates.json", meta).expect("write");
        let mut raw: Vec<u8> = Vec::with_capacity(vecs.len() * 4);
        for v in &vecs {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write("research/mapper/wordnet_vecs.f32", raw).expect("write vecs");
        println!(
            "wrote research/mapper/wordnet_candidates.json + wordnet_vecs.f32 ({} x 768)",
            items.len()
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
