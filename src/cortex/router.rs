//! Deterministic-first routing (PH-101 Task 6).
//!
//! [`Intent`] classifies the *verb*, never the meaning: a fixed keyword table
//! maps known verbs, everything else is [`Intent::Unknown`] and routes to the
//! cheapest capable backend with a note. No model is consulted — the ladder is
//! pure data: capability match first, explicit preference second, scan-best
//! third, CPU last. Every skipped candidate lands in the rationale.

use serde::{Deserialize, Serialize};

use crate::backend::{select_backend, BackendKind};
use crate::devices::DeviceInfo;
use crate::model_provider::Capability;

/// What the caller wants to do, from the verb alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Intent {
    Embed,
    Inspect,
    Execute,
    Plan,
    Unknown,
}

impl Intent {
    /// Fixed keyword table. Unknown verbs are [`Intent::Unknown`] — the
    /// router degrades the placement, never the classification.
    pub fn parse(verb: &str) -> Self {
        match verb.trim().to_lowercase().as_str() {
            "embed" | "classify" | "probe" | "encode" => Intent::Embed,
            "read" | "inspect" | "show" | "list" | "search" | "query" => Intent::Inspect,
            "execute" | "run" | "apply" | "write" | "mutate" => Intent::Execute,
            "route" | "plan" | "dispatch" => Intent::Plan,
            _ => Intent::Unknown,
        }
    }

    /// The capability an intent needs. Execute wants no model capability by
    /// itself — execution authority lives in the world's permission layer,
    /// not in the router.
    pub fn requires(&self) -> Option<Capability> {
        match self {
            Intent::Embed => Some(Capability::Embeddings),
            Intent::Inspect | Intent::Execute | Intent::Plan | Intent::Unknown => None,
        }
    }
}

/// Where a request goes and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub intent: Intent,
    pub backend: BackendKind,
    pub backend_label: String,
    /// Every candidate considered, in order, with its fate.
    pub rationale: Vec<String>,
    /// Set when the chosen backend is not what was preferred.
    pub fallback_note: Option<String>,
}

/// Ladder inputs. Devices come from [`crate::devices::HardwareDiscovery`];
/// preference from `PHYSIS_BACKEND` or the caller.
pub struct RoutingContext {
    pub verb: String,
    pub preferred: Option<BackendKind>,
    pub devices: Vec<DeviceInfo>,
}

/// Route deterministically: capable first, preferred second, scan-best
/// third, CPU last. Always resolves — the worst answer is CPU with notes.
pub fn route(ctx: &RoutingContext) -> RoutingDecision {
    let intent = Intent::parse(&ctx.verb);
    let mut rationale = Vec::new();
    rationale.push(format!("verb '{}' parses as {:?}", ctx.verb, intent));

    // Candidate order: explicit preference, scan-best, CPU.
    let mut candidates = Vec::new();
    if let Some(p) = ctx.preferred {
        candidates.push((p, "explicit preference"));
    }
    let scan_best = crate::devices::HardwareDiscovery::best_kind(&ctx.devices);
    if !candidates.iter().any(|(k, _)| *k == scan_best) {
        candidates.push((scan_best, "scan-best hardware"));
    }
    if !candidates.iter().any(|(k, _)| *k == BackendKind::Cpu) {
        candidates.push((BackendKind::Cpu, "always-available fallback"));
    }

    for (kind, why) in candidates {
        let resolved = select_backend(Some(kind));
        let capable = match intent.requires() {
            Some(need) => resolved.backend.capabilities().contains(&need),
            None => true,
        };
        if !capable {
            rationale.push(format!("{kind} ({why}) skipped: lacks required capability"));
            continue;
        }
        if resolved.backend.kind() != kind {
            rationale.push(format!(
                "{kind} ({why}) unwired — {}",
                resolved.fallback_note.as_deref().unwrap_or("fell back")
            ));
            // Fall through to the next candidate; CPU resolves below.
            continue;
        }
        rationale.push(format!("{kind} ({why}) selected: {}", resolved.backend.label()));
        return RoutingDecision {
            intent,
            backend: kind,
            backend_label: resolved.backend.label(),
            rationale,
            fallback_note: None,
        };
    }

    // Unreachable in practice (CPU always resolves), but the ladder keeps
    // one honest floor instead of an unwrap.
    let cpu = select_backend(Some(BackendKind::Cpu));
    rationale.push("floor: CPU".to_string());
    RoutingDecision {
        intent,
        backend: BackendKind::Cpu,
        backend_label: cpu.backend.label(),
        rationale,
        fallback_note: cpu.fallback_note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::HardwareDiscovery;

    fn ctx(verb: &str) -> RoutingContext {
        RoutingContext {
            verb: verb.to_string(),
            preferred: None,
            devices: HardwareDiscovery::scan(),
        }
    }

    #[test]
    fn verbs_parse_to_intents_without_guessing() {
        assert_eq!(Intent::parse("embed"), Intent::Embed);
        assert_eq!(Intent::parse("Classify"), Intent::Embed);
        assert_eq!(Intent::parse("read"), Intent::Inspect);
        assert_eq!(Intent::parse("execute"), Intent::Execute);
        assert_eq!(Intent::parse("plan"), Intent::Plan);
        assert_eq!(Intent::parse("frobnicate"), Intent::Unknown);
    }

    #[test]
    fn embed_routes_to_capable_cpu_with_rationale() {
        let d = route(&ctx("embed"));
        assert_eq!(d.intent, Intent::Embed);
        assert_eq!(d.backend, BackendKind::Cpu);
        assert!(d.rationale.len() >= 2, "rationale must show its work");
    }

    #[test]
    fn unknown_verb_still_resolves() {
        let d = route(&ctx("frobnicate"));
        assert_eq!(d.intent, Intent::Unknown);
        assert_eq!(d.backend, BackendKind::Cpu);
    }

    #[test]
    fn unwired_preference_falls_through_loudly() {
        let mut c = ctx("embed");
        c.preferred = Some(BackendKind::Cuda);
        let d = route(&c);
        assert_eq!(d.backend, BackendKind::Cpu);
        assert!(
            d.rationale.iter().any(|r| r.contains("unwired")),
            "skipped CUDA must be on the record: {:?}",
            d.rationale
        );
    }
}
