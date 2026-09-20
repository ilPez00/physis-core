//! Provenance tracking — traceable chains of evidence and reasoning.
//!
//! Every important assertion should answer:
//!   *Who said this?*
//!   *When?*
//!   *Based on what?*
//!   *Why do we believe it?*

use serde::{Deserialize, Serialize};

use crate::models::Score;

/// A single link in a provenance chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceLink {
    /// What this link asserts.
    pub claim: String,
    /// Source of the claim (file, observation, operator, sensor, etc.).
    pub source: String,
    /// When the claim was made or observed.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Confidence in this specific claim (0.0–1.0).
    pub confidence: Score,
    /// Optional reference to the raw data (path, id, URL).
    #[serde(default)]
    pub raw_reference: Option<String>,
    /// Free-text justification or method.
    #[serde(default)]
    pub method: Option<String>,
    /// G4: id of the intake episode that produced this link — citations
    /// resolve to the raw intake, not only to a free-text source label.
    #[serde(default)]
    pub intake_id: Option<String>,
    /// A4: SHA-256 over this link's own content (all fields above), hex.
    /// `None` on links written before A4 and on links never sealed.
    #[serde(default)]
    pub content_hash: Option<String>,
    /// A4: the previous link's `content_hash`, so the chain is a hash chain.
    /// `None` for the first link (genesis).
    #[serde(default)]
    pub prev_hash: Option<String>,
}

/// A4: the outcome of verifying a provenance chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainIntegrity {
    /// Every link carries a hash and each links to the previous one.
    Sealed,
    /// The first link with no `content_hash` (legacy or never sealed).
    Unsealed { first: usize },
    /// The first link whose hash or back-link does not match — tampering.
    Broken { first: usize },
}

impl ProvenanceLink {
    pub fn new(claim: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            claim: claim.into(),
            source: source.into(),
            timestamp: chrono::Utc::now(),
            confidence: 1.0,
            raw_reference: None,
            method: None,
            intake_id: None,
            content_hash: None,
            prev_hash: None,
        }
    }

    /// G4: stamp which intake episode produced this provenance link.
    pub fn with_intake_id(mut self, intake_id: impl Into<String>) -> Self {
        self.intake_id = Some(intake_id.into());
        self
    }

    /// A4: deterministic SHA-256 over this link's content. The hash fields are
    /// deliberately excluded, so a link's digest is stable and a tampered field
    /// is detectable. `timestamp` is hashed as RFC-3339, which is canonical.
    fn content_digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for part in [
            self.claim.as_str(),
            self.source.as_str(),
            self.raw_reference.as_deref().unwrap_or(""),
            self.method.as_deref().unwrap_or(""),
            self.intake_id.as_deref().unwrap_or(""),
        ] {
            h.update(part.as_bytes());
            h.update([0u8]);
        }
        h.update(self.timestamp.to_rfc3339().as_bytes());
        h.update([0u8]);
        h.update(self.confidence.to_bits().to_le_bytes());
        format!("{:x}", h.finalize())
    }
}

/// A complete provenance chain — why Physis believes something.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProvenanceChain {
    /// Ordered chain of evidence/reasoning steps.
    pub links: Vec<ProvenanceLink>,
}

impl ProvenanceChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_link(&mut self, mut link: ProvenanceLink) {
        // A4: seal on append, so a chain built through the API is a hash chain
        // by construction; a link mutated after this is detectable by verify().
        link.prev_hash = self.links.last().and_then(|l| l.content_hash.clone());
        link.content_hash = Some(link.content_digest());
        self.links.push(link);
    }

    /// A4: (re)compute every link's hash and back-link in order. Use after
    /// constructing a chain by pushing into `links` directly, or after a
    /// deliberate edit that is meant to be re-sealed.
    pub fn seal(&mut self) {
        let mut prev: Option<String> = None;
        for link in &mut self.links {
            link.prev_hash = prev.clone();
            link.content_hash = Some(link.content_digest());
            prev = link.content_hash.clone();
        }
    }

    /// A4: verify the hash chain. Reports the **first** link that breaks, with
    /// its sequence coordinate. A legacy chain (no hashes) is `Unsealed`, not
    /// `Broken` — absence of a seal is not evidence of tampering.
    ///
    /// Limitation: this detects an edit that does not re-seal the tail. An
    /// attacker who rewrites a link and every hash after it produces a chain
    /// that verifies; detecting that needs an external anchor (a signed head).
    pub fn verify(&self) -> ChainIntegrity {
        let mut prev: Option<&str> = None;
        for (i, link) in self.links.iter().enumerate() {
            let Some(stored) = link.content_hash.as_deref() else {
                return ChainIntegrity::Unsealed { first: i };
            };
            if link.prev_hash.as_deref() != prev || link.content_digest() != stored {
                return ChainIntegrity::Broken { first: i };
            }
            prev = Some(stored);
        }
        ChainIntegrity::Sealed
    }

    /// G4: the distinct intake ids cited across this chain, in encounter
    /// order — each resolves to the ingest episode that produced a link.
    pub fn cited_intake_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for l in &self.links {
            if let Some(id) = &l.intake_id {
                if !ids.contains(id) {
                    ids.push(id.clone());
                }
            }
        }
        ids
    }

    /// Short summary for display.
    pub fn summary(&self) -> String {
        if self.links.is_empty() {
            return "No provenance recorded".to_string();
        }
        let last = &self.links[self.links.len() - 1];
        format!("{} (via {})", last.claim, last.source)
    }
}
