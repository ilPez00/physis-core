//! The observation log — what the machine saw, append-only, forever.
//!
//! ## Why this shape
//!
//! A survey of eleven systems in this space (`computer-remake-research/`) found
//! that each plays exactly one of three roles and collapses the other two into
//! it: **observers** record honestly but their events never mean anything;
//! **indexers** make content findable but hold no position on what is true and
//! overwrite themselves; **interpreters** let a model decide, and the model
//! becomes the memory. They want to be three layers.
//!
//! This is the bottom layer. It borrows ActivityWatch's decomposition —
//! *bucket / event / watcher* — because it is the only model in that survey with
//! nothing file-specific in it. A git commit, a terminal command, a downloaded
//! page, a sensor reading and a model's own output are all the same shape:
//! something, from some source, over some interval.
//!
//! ## Three rules, and they are the layer
//!
//! 1. **Append-only.** An [`Observation`] is never rewritten and never deleted.
//!    It records what was seen, not what is true. It cannot be wrong, because it
//!    does not claim anything — which is exactly what makes it safe to build
//!    belief on top of.
//! 2. **Interval, not instant.** `duration_ms` makes *"what was happening while
//!    X"* a query rather than a join. ActivityWatch's `duration` is the detail
//!    most re-implementations drop and then miss.
//! 3. **Provenance in the record.** `produced_by` names the watcher, so a claim
//!    derived from an observation can be walked back to the thing that saw it.
//!
//! ## What this is not
//!
//! Not an index — nothing here is ranked or embedded. Not a filesystem: thirty
//! four years of semantic-filesystem attempts (Gifford 1991, BeFS, WinFS,
//! Nepomuk, and LSFS in 2025) died of being in the write path, of needing
//! applications to integrate, and of metadata that does not survive a file
//! leaving the machine. This sits beside the filesystem and observes it.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One thing that was seen.
///
/// Deliberately flat and deliberately unconstrained in `body`: a schema-first
/// design (Nepomuk) cannot record what its ontology never enumerated, which is
/// why so few applications ever integrated with it. Structure is imposed later,
/// by the layer above, and may be imposed differently tomorrow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// Monotonic per-log sequence. Stable across reads; the address a claim
    /// cites when it says where it came from.
    pub seq: u64,
    pub at: chrono::DateTime<chrono::Utc>,
    /// How long it went on. `None` = an instant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// The bucket: `fs`, `git`, `terminal`, `model`, `human`, `sensor`…
    /// Free-form on purpose — a new source is a new string, not a migration.
    pub source: String,
    /// What was seen: a path, a command, a URL, a device id.
    pub subject: String,
    /// Content or summary. May be empty; an observation that something *changed*
    /// is still an observation.
    #[serde(default)]
    pub body: String,
    /// Which watcher produced this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub produced_by: Option<String>,
}

impl Observation {
    pub fn new(source: impl Into<String>, subject: impl Into<String>) -> Self {
        Self {
            seq: 0,
            at: chrono::Utc::now(),
            duration_ms: None,
            source: source.into(),
            subject: subject.into(),
            body: String::new(),
            produced_by: None,
        }
    }
    pub fn with_body(mut self, b: impl Into<String>) -> Self {
        self.body = b.into();
        self
    }
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }
    pub fn by(mut self, watcher: impl Into<String>) -> Self {
        self.produced_by = Some(watcher.into());
        self
    }
}

/// Path of the append-only log. JSONL: one observation per line, so a partial
/// write costs one record rather than the file, and `tail -f` works.
pub fn log_path() -> PathBuf {
    crate::store::data_dir().join("observations.jsonl")
}

/// Append observations, assigning sequence numbers. Returns the range written.
///
/// Opens in append mode and never rewrites: a corrupted line is a lost record,
/// not a lost log. That is the trade an append-only store exists to make.
pub fn append(path: &Path, obs: &mut [Observation]) -> anyhow::Result<(u64, u64)> {
    use std::io::Write;
    let mut next = last_seq(path)? + 1;
    let first = next;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    for o in obs.iter_mut() {
        o.seq = next;
        writeln!(f, "{}", serde_json::to_string(o)?)?;
        next += 1;
    }
    Ok((first, next.saturating_sub(1)))
}

/// Highest sequence in the log, or 0 when empty.
pub fn last_seq(path: &Path) -> anyhow::Result<u64> {
    Ok(read(path)?.last().map(|o| o.seq).unwrap_or(0))
}

/// Read the whole log, skipping unparseable lines.
///
/// Skipping rather than failing is deliberate: a truncated final line from a
/// killed process must not make the entire history unreadable.
pub fn read(path: &Path) -> anyhow::Result<Vec<Observation>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    Ok(std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<Observation>(l).ok())
        .collect())
}

/// Observations from one source, newest first.
pub fn by_source(all: &[Observation], source: &str) -> Vec<Observation> {
    let mut v: Vec<Observation> =
        all.iter().filter(|o| o.source == source).cloned().collect();
    v.sort_by_key(|o| std::cmp::Reverse(o.seq));
    v
}

/// What else was going on while `seq` was observed.
///
/// This is the query `duration_ms` exists for: overlap in time, across sources.
/// Point events are treated as zero-length intervals, so a command that ran
/// during a long file edit is found by the same call that finds two overlapping
/// edits.
pub fn concurrent(all: &[Observation], seq: u64) -> Vec<Observation> {
    let Some(target) = all.iter().find(|o| o.seq == seq) else {
        return Vec::new();
    };
    let span = |o: &Observation| {
        let start = o.at.timestamp_millis();
        (start, start + o.duration_ms.unwrap_or(0) as i64)
    };
    let (ts, te) = span(target);
    let mut v: Vec<Observation> = all
        .iter()
        .filter(|o| o.seq != seq)
        .filter(|o| {
            let (os, oe) = span(o);
            os <= te && oe >= ts
        })
        .cloned()
        .collect();
    v.sort_by_key(|o| o.seq);
    v
}

// ── Watchers ────────────────────────────────────────────────────────────────
//
// A watcher is a function that looks at one kind of thing and returns
// observations. It holds no state beyond what it reads, so watchers are
// independent, cheap to add, and impossible to get wrong in a way that damages
// the log. `recall` (Go) reaches the same conclusion from the other direction:
// read-only adapters into stores you do not own, writes only ever to your own.

/// Watch a directory tree: emit one observation per file whose content hash
/// differs from the last observation of that path.
///
/// Hash-gating before doing anything expensive is OpenRecall's instinct
/// (`is_similar` before embedding) applied at the right level: content, not
/// pixels. A file touched but unchanged produces nothing, so re-running this is
/// close to free and the log does not fill with noise.
pub fn watch_fs(root: &Path, known: &[Observation], max_files: usize) -> Vec<Observation> {
    use sha2::{Digest, Sha256};
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    let mut seen = 0usize;
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            if seen >= max_files {
                return out;
            }
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            // Skip the noise every tree has. Not a security boundary — a
            // signal-to-noise one.
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let Ok(meta) = p.metadata() else { continue };
            if meta.len() > 2_000_000 {
                continue;
            }
            let Ok(body) = std::fs::read_to_string(&p) else { continue };
            seen += 1;
            let hash = format!("{:x}", Sha256::digest(body.as_bytes()));
            let subject = p.to_string_lossy().to_string();
            let unchanged = known
                .iter()
                .rev()
                .find(|o| o.source == "fs" && o.subject == subject)
                .map(|o| o.body == hash)
                .unwrap_or(false);
            if unchanged {
                continue;
            }
            out.push(Observation::new("fs", subject).with_body(hash).by("watch_fs"));
        }
    }
    out
}

/// Watch git history: one observation per commit, with its duration spanning
/// back to the previous commit.
///
/// The duration is the point. A commit is an instant, but the *work* it records
/// is the interval since the last one — which makes [`concurrent`] able to
/// answer "what files were being edited while this commit was being written".
pub fn watch_git(repo: &Path, limit: usize) -> Vec<Observation> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["log", &format!("-{limit}"), "--format=%H%x1f%aI%x1f%s"])
        .output();
    let Ok(out) = out else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let mut rows: Vec<(String, chrono::DateTime<chrono::Utc>, String)> = Vec::new();
    for line in text.lines() {
        let mut f = line.split('\u{1f}');
        let (Some(h), Some(t), Some(s)) = (f.next(), f.next(), f.next()) else { continue };
        let Ok(at) = chrono::DateTime::parse_from_rfc3339(t) else { continue };
        rows.push((h.to_string(), at.with_timezone(&chrono::Utc), s.to_string()));
    }
    // `git log` is newest-first; the gap to the NEXT row is the previous commit.
    let mut obs = Vec::new();
    for (i, (hash, at, subject)) in rows.iter().enumerate() {
        let dur = rows
            .get(i + 1)
            .map(|(_, prev, _)| (*at - *prev).num_milliseconds().max(0) as u64);
        let mut o = Observation::new("git", hash.clone())
            .with_body(subject.clone())
            .by("watch_git");
        o.at = *at;
        o.duration_ms = dur;
        obs.push(o);
    }
    obs.reverse(); // oldest first, so sequence order matches time order
    obs
}

// ── The claim path ──────────────────────────────────────────────────────────
//
// This is the join the survey found missing everywhere. Observers record; they
// never assert. Indexers rank; they hold no position. The step from "this was
// seen" to "this is so, and here is why, and it might be wrong" exists in no
// surveyed system — and it is the only step that lets a machine notice it was
// mistaken rather than merely outdated.

/// Turn an observation into a claim, citing the observation as its evidence.
///
/// The claim enters as `Candidate` with the observation attached as supporting
/// evidence, sourced to `obs:<seq>` so the citation resolves back into the log.
/// Three properties follow, and each is deliberate:
///
/// - **The observation is untouched.** It stays in the append-only log exactly
///   as recorded. A claim derived from it is a separate object with its own
///   status, and refuting the claim never edits the record it came from.
/// - **The claim can be wrong.** That is the entire point of promoting it.
/// - **Provenance is an address, not a copy.** `obs:<seq>` is stable, so the
///   evidence can always be re-read rather than trusted.
pub fn promote(
    obs: &Observation,
    statement: impl Into<String>,
    embedding: Vec<f32>,
) -> crate::hypothesis::Hypothesis {
    let mut h = crate::hypothesis::Hypothesis::new(statement, embedding);
    let claim = if obs.body.is_empty() {
        format!("{} observed {} at {}", obs.source, obs.subject, obs.at.to_rfc3339())
    } else {
        format!(
            "{} observed {} at {}: {}",
            obs.source,
            obs.subject,
            obs.at.to_rfc3339(),
            obs.body.chars().take(200).collect::<String>()
        )
    };
    h.add_supporting_evidence(crate::hypothesis::Evidence::supports(
        format!("obs:{}", obs.seq),
        claim,
    ));
    // Provenance is not corroboration, and conflating them is an epistemic bug
    // rather than a cosmetic one.
    //
    // `add_supporting_evidence` promotes Candidate -> Supported as soon as any
    // evidence lands, which is right for a measurement and wrong here: the
    // observation is what the claim is ABOUT, not a reason to believe it. That a
    // commit happened is no evidence that an assertion about that commit is
    // true. Left alone, every promoted observation would enter the shared state
    // already looking established, and `ground` would show a wall of Supported
    // claims nobody had tested.
    //
    // The citation stays — `cited_by` walks it, and the evidence can be re-read.
    // Only the standing is reset.
    h.status = crate::hypothesis::HypothesisStatus::Candidate;
    h.recompute_fitness();
    h
}

/// The observation a claim's evidence cites, if any.
///
/// Walks `obs:<seq>` sources back into the log. Returns the observations that
/// actually exist: a citation to a sequence that is not in the log is reported
/// by its absence rather than by a silent empty result, because a dangling
/// citation means the log and the claims have diverged.
pub fn cited_by(h: &crate::hypothesis::Hypothesis, all: &[Observation]) -> Vec<Observation> {
    let mut out = Vec::new();
    for e in h.supporting_evidence.iter().chain(h.contradicting_evidence.iter()) {
        let Some(rest) = e.source.strip_prefix("obs:") else { continue };
        let Ok(seq) = rest.trim().parse::<u64>() else { continue };
        if let Some(o) = all.iter().find(|o| o.seq == seq) {
            out.push(o.clone());
        }
    }
    out.sort_by_key(|o| o.seq);
    out.dedup_by_key(|o| o.seq);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("physis-obs-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The one invariant the whole layer rests on: appending never disturbs what
    /// is already there, and sequence numbers keep rising across calls.
    #[test]
    fn appending_never_rewrites_and_sequences_keep_rising() {
        let p = tmp("append").join("o.jsonl");
        let (a1, b1) = append(&p, &mut [Observation::new("fs", "x")]).unwrap();
        assert_eq!((a1, b1), (1, 1));
        let (a2, b2) =
            append(&p, &mut [Observation::new("git", "y"), Observation::new("git", "z")]).unwrap();
        assert_eq!((a2, b2), (2, 3));
        let all = read(&p).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].subject, "x", "the first record must be untouched");
        assert_eq!(all.iter().map(|o| o.seq).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    /// A killed process leaves a half-written final line. That must cost one
    /// record, not the history.
    #[test]
    fn a_truncated_line_does_not_destroy_the_log() {
        use std::io::Write;
        let p = tmp("trunc").join("o.jsonl");
        append(&p, &mut [Observation::new("fs", "good")]).unwrap();
        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        write!(f, "{{\"seq\":2,\"at\":\"not-a-date").unwrap();
        drop(f);
        let all = read(&p).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].subject, "good");
    }

    /// `duration_ms` earns its keep here: overlap across sources, which is the
    /// query a point-event log cannot answer at all.
    #[test]
    fn concurrent_finds_overlap_across_sources() {
        let t0 = chrono::Utc::now();
        let mut long = Observation::new("fs", "editing.rs");
        long.seq = 1;
        long.at = t0;
        long.duration_ms = Some(60_000);

        let mut during = Observation::new("terminal", "cargo test");
        during.seq = 2;
        during.at = t0 + chrono::Duration::seconds(30);

        let mut after = Observation::new("terminal", "git push");
        after.seq = 3;
        after.at = t0 + chrono::Duration::seconds(120);

        let all = vec![long, during, after];
        let c = concurrent(&all, 1);
        assert_eq!(c.len(), 1, "only the overlapping event");
        assert_eq!(c[0].subject, "cargo test");
    }

    /// Re-running a watcher over an unchanged tree must produce nothing. If this
    /// regresses, continuous observation becomes unaffordable and the log fills
    /// with noise that means nothing.
    #[test]
    fn the_fs_watcher_is_silent_when_nothing_changed() {
        let d = tmp("fs");
        std::fs::write(d.join("a.txt"), "hello").unwrap();
        let first = watch_fs(&d, &[], 100);
        assert_eq!(first.len(), 1);
        let second = watch_fs(&d, &first, 100);
        assert!(second.is_empty(), "unchanged tree must produce no observations");

        std::fs::write(d.join("a.txt"), "changed").unwrap();
        let third = watch_fs(&d, &first, 100);
        assert_eq!(third.len(), 1, "a content change must be seen");
    }

    /// The join, stated as a test: promoting an observation must leave the
    /// observation alone and produce something that CAN be contradicted.
    #[test]
    fn promoting_leaves_the_observation_untouched_and_yields_a_refutable_claim() {
        let mut o = Observation::new("git", "abc123").with_body("fix the null");
        o.seq = 7;
        let before = o.clone();

        let h = promote(&o, "the null was invalid", vec![0.1, 0.2]);
        assert_eq!(o, before, "promotion must not mutate the observation");
        assert_eq!(h.supporting_evidence.len(), 1);
        assert_eq!(h.supporting_evidence[0].source, "obs:7");
        // Provenance is not corroboration: citing the observation must NOT make
        // the claim look established before anyone has tested it.
        assert_eq!(
            h.status,
            crate::hypothesis::HypothesisStatus::Candidate,
            "a promoted observation is asserted, not supported"
        );

        // And it can be refuted — which no surveyed system's record can be.
        let mut h = h;
        h.add_contradicting_evidence(crate::hypothesis::Evidence::contradicts(
            "measurement", "delta was exactly zero",
        ));
        assert_eq!(h.status, crate::hypothesis::HypothesisStatus::Contradicted);
    }

    /// Provenance is an address: the evidence must resolve back into the log.
    #[test]
    fn a_claim_walks_back_to_the_observation_that_caused_it() {
        let mut o = Observation::new("fs", "src/main.rs").with_body("deadbeef");
        o.seq = 42;
        let h = promote(&o, "main.rs changed", vec![0.0]);
        let found = cited_by(&h, &[o.clone()]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].seq, 42);
        // A citation into an empty log resolves to nothing, not to a guess.
        assert!(cited_by(&h, &[]).is_empty());
    }

    #[test]
    fn by_source_filters_and_orders_newest_first() {
        let mut a = Observation::new("fs", "1");
        a.seq = 1;
        let mut b = Observation::new("git", "2");
        b.seq = 2;
        let mut c = Observation::new("fs", "3");
        c.seq = 3;
        let v = by_source(&[a, b, c], "fs");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].subject, "3");
    }
}
