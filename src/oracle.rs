//! Big-model oracle **types and deterministic comparison** (Directive 1 §4).
//!
//! A substantially larger model is used as **reference/oracle**, never as part
//! of the deployed system. Core keeps what a benchmark needs to report that leg
//! honestly — the case type, the env-resolved configuration, and the two
//! deterministic checks (token agreement, evidence citation) — and **not the
//! transport**: the OpenAI-compatible call moved to the product repository in
//! the 2026-09-14 Core/Product split (see `MOVED_TO_PRODUCT.md`). Offline or
//! keyless the leg reports `not_configured`/`failed` and the benchmark carries
//! on. Nothing is ever faked.
//!
//! Provider configuration this module still resolves (parsing only, no network):
//!
//! ```text
//! PHYSIS_ORACLE_URL    (default https://openrouter.ai/api/v1)
//! PHYSIS_ORACLE_KEY    (fallback: OPENROUTER_API_KEY, then GROQ_API_KEY)
//! PHYSIS_ORACLE_MODEL  (default openai/gpt-4o)
//! ```

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleConfig {
    pub url: String,
    pub model: String,
    pub has_key: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleCase {
    pub query: String,
    pub context_chars: usize,
    pub decisive_docs: Vec<String>,
    pub must_mention: Vec<String>,
    pub physis_answer: String,
    pub oracle_answer: String,
    pub agreement_jaccard: f32,
    pub oracle_cites_evidence: bool,
    pub latency_ms: f64,
}

/// Resolve config from env. `None` ⇒ nothing to call; reported honestly.
pub fn from_env() -> Option<(OracleConfig, String)> {
    let url = std::env::var("PHYSIS_ORACLE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "https://openrouter.ai/api/v1".into());
    let key = std::env::var("PHYSIS_ORACLE_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok().filter(|s| !s.trim().is_empty()))
        .or_else(|| std::env::var("GROQ_API_KEY").ok().filter(|s| !s.trim().is_empty()));
    let model = std::env::var("PHYSIS_ORACLE_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "openai/gpt-4o".into());
    key.map(|k| (OracleConfig { url, model, has_key: true }, k))
}

/// Token Jaccard over lowercase alnum runs ≥3 chars — the linguistic-agreement
/// input. Deterministic, dependency-free, in-repo so tests own it.
pub fn token_jaccard(a: &str, b: &str) -> f32 {
    let toks = |t: &str| -> std::collections::BTreeSet<String> {
        t.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .map(|w| w.to_lowercase())
            .collect()
    };
    let (a, b) = (toks(a), toks(b));
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(&b).count();
    let uni = a.union(&b).count().max(1);
    inter as f32 / uni as f32
}

/// Do all `must_mention` tokens literally occur (case-insensitive)? The
/// decisive-evidence check: did the model recover the facts that decide?
/// Deterministic, offline — which is why it stays in Core while the call does not.
pub fn cites_evidence(answer: &str, must_mention: &[String]) -> bool {
    let low = answer.to_lowercase();
    must_mention.iter().all(|m| low.contains(&m.to_lowercase()))
}

/// The provider call as an injectable hook.
///
/// Core does not ship an HTTP client: the transport was moved to the product
/// repository by the 2026-09-14 Core/Product split. A benchmark still *can*
/// run the oracle leg, because the product (or a test) installs the transport
/// here. With no transport installed the leg reports `not_configured` — an
/// honest answer, never a fabricated one.
pub type OracleTransport =
    fn(&OracleConfig, &str, &str, &str, u32) -> anyhow::Result<(String, f64)>;

static ORACLE_TRANSPORT: std::sync::OnceLock<OracleTransport> = std::sync::OnceLock::new();

/// Install the transport once (product startup, or a test). Second install is
/// ignored, so a caller cannot silently swap the instrument mid-run.
pub fn install_transport(f: OracleTransport) {
    let _ = ORACLE_TRANSPORT.set(f);
}

/// Run one oracle call through the installed transport, or refuse in a way the
/// benchmark reports as `not_configured` rather than as a number.
pub fn complete_with(
    cfg: &OracleConfig,
    key: &str,
    system: &str,
    prompt: &str,
    max_tokens: u32,
) -> anyhow::Result<(String, f64)> {
    match ORACLE_TRANSPORT.get() {
        Some(f) => f(cfg, key, system, prompt, max_tokens),
        None => anyhow::bail!(
            "oracle transport not installed (provider call lives in the product;              Core reports not_configured)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_is_sane() {
        assert!((token_jaccard("the pump failed", "the pump failed") - 1.0).abs() < 1e-6);
        assert!(token_jaccard("invoice approved today", "quantum origami folds") < 0.2);
        assert!(token_jaccard("", "") > 0.99);
    }

    #[test]
    fn evidence_check_requires_every_fact() {
        let m = vec!["2.6".into(), "4.2".into()];
        assert!(cites_evidence("opens at 2.6 bar, counterpart 4.2 bar", &m));
        assert!(!cites_evidence("opens at 2.6 bar", &m));
    }

    #[test]
    fn env_resolution_prefers_explicit_key() {
        std::env::set_var("PHYSIS_ORACLE_KEY", "k-explicit");
        std::env::set_var("PHYSIS_ORACLE_MODEL", "test/big");
        let (cfg, key) = from_env().expect("key set");
        assert_eq!(key, "k-explicit");
        assert_eq!(cfg.model, "test/big");
        std::env::remove_var("PHYSIS_ORACLE_KEY");
        std::env::remove_var("PHYSIS_ORACLE_MODEL");
    }
}
