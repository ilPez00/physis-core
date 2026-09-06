//! Determinism probe — isolates WHERE run-to-run variation comes from,
//! rather than guessing. Iterations 21/24 observed a graph that was
//! structurally stable but not byte-identical across runs; the working
//! hypothesis was ONNX Runtime's multi-threaded reductions, but setting
//! `intra_threads: Some(1)` did NOT fix it, so the hypothesis needs
//! testing directly instead of more speculation.
//!
//! Checks, in order:
//!   1. Same text embedded twice IN ONE PROCESS — is the embedder even
//!      internally deterministic?
//!   2. Exact bytes of one embedding, printed, so two separate runs can
//!      be diffed to see whether the nondeterminism is cross-process.
//!
//! Run (twice, then diff):
//!   cargo run -p physis-core --features embed-onnx --release --example probe_embedding_determinism

#[cfg(feature = "embed-onnx")]
use physis_core::embed::VectorEmbed;

fn main() {
    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
        let dir = match ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists()) {
            Some(d) => *d,
            None => { println!("no model"); return; }
        };

        for threads in [None, Some(1usize)] {
            let e = OnnxEmbedder::with_config(&OnnxConfig {
                dim: 384,
                model_dir: Some(dir.to_string()),
                pooling: PoolingStrategy::Mean,
                intra_threads: threads,
                ..OnnxConfig::default()
            });
            if !e.is_available() { println!("embedder unavailable"); return; }

            let text = "The hybrid battery works alongside the engine to reduce gasoline use.";
            let a = e.embed(text);
            let b = e.embed(text);
            let same_in_process = a == b;

            // A second, different text first, then the target again — checks whether
            // preceding calls perturb later ones (shared buffers / accumulated state).
            let _ = e.embed("an unrelated sentence about weather and rainfall patterns");
            let c = e.embed(text);
            let stable_after_other_calls = a == c;

            let label = match threads { None => "all-cores", Some(n) => { let _ = n; "1-thread" } };
            println!(
                "[{label}] same-text-twice-in-process: {same_in_process}   stable-after-other-calls: {stable_after_other_calls}"
            );
            println!("[{label}] first 8 dims: {:?}", &a[..8]);
            let checksum: f64 = a.iter().map(|&x| x as f64).sum();
            println!("[{label}] sum-of-all-dims (cross-run comparison key): {checksum:.12}");
        }
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("built without embed-onnx");
}
