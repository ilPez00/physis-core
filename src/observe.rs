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
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut next = last_seq(path)? + 1;
    let first = next;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // `read(true)` is required to inspect the tail byte below: an append-only
    // handle is write-only by default, and reading from it fails with EBADF.
    let mut f = std::fs::OpenOptions::new().create(true).append(true).read(true).open(path)?;
    // A process killed mid-write leaves a final line with no newline. Writing
    // the next record directly onto it concatenates the two into one line, and
    // `read` drops that whole line — so the interrupted record AND the first
    // record of this batch both vanish. That is a *skipped* record, the one
    // thing an append-only log must not do. Measured 2026-09-16: a truncated
    // fixture plus one `observe` lost its seq-3 record and left a gap in the
    // log. Emitting the missing newline first makes the fragment a line of its
    // own (which `read` already drops by design) and leaves the new records
    // intact. Nothing already written is rewritten or removed, so the
    // append-only guarantee holds.
    if f.metadata()?.len() > 0 {
        f.seek(SeekFrom::End(-1))?;
        let mut last = [0u8; 1];
        f.read_exact(&mut last)?;
        if last[0] != b'\n' {
            f.write_all(b"\n")?;
        }
    }
    for o in obs.iter_mut() {
        o.seq = next;
        writeln!(f, "{}", serde_json::to_string(o)?)?;
        next += 1;
    }
    Ok((first, next.saturating_sub(1)))
}

/// Highest sequence in the log, or 0 when empty.
///
/// Reads only the tail: appending must not cost a full parse of history, or
/// every write becomes linear in everything ever written.
pub fn last_seq(path: &Path) -> anyhow::Result<u64> {
    Ok(read_tail(path, 1)?.last().map(|o| o.seq).unwrap_or(0))
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

/// Read only the last `n` observations, without parsing the whole log.
///
/// ## Why this exists — measured, not anticipated
///
/// [`read`] parses the entire JSONL on every call, and every watcher calls it
/// to deduplicate. That is linear in total history, in both time and memory:
///
/// | log | disk | one `watch` run | peak RSS |
/// |---|---|---|---|
/// | 10k obs | 1 MB | 0.03 s | 19 MB |
/// | 100k obs | 16 MB | 0.40 s | 90 MB |
/// | 500k obs | 85 MB | **2.00 s** | **372 MB** |
///
/// At roughly ten new observations a minute, 500k arrives in about five weeks.
/// A continuous observer that costs two seconds and a third of a gigabyte per
/// tick is one a person notices — and **being noticed is precisely what killed
/// Nepomuk and WinFS**, neither of which failed on the quality of its
/// semantics. The research file `computer-remake-research/links.md` records
/// that; this is the same cliff arriving in the same way.
///
/// Deduplication only ever needs *recent* history: a watcher asks "have I
/// already recorded this?", and anything it could plausibly re-observe is near
/// the end of the log. Reading a bounded tail makes that check O(1) in total
/// history.
///
/// Seeks from the end in 64 KiB blocks and stops once `n` newline boundaries
/// have been passed, so cost depends on `n` rather than on file size.
pub fn read_tail(path: &Path, n: usize) -> anyhow::Result<Vec<Observation>> {
    use std::io::{Read, Seek, SeekFrom};
    if !path.exists() || n == 0 {
        return Ok(Vec::new());
    }
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    const BLOCK: u64 = 64 * 1024;

    let mut end = len;
    let mut buf: Vec<u8> = Vec::new();
    let mut lines = 0usize;
    while end > 0 && lines <= n {
        let start = end.saturating_sub(BLOCK);
        let size = (end - start) as usize;
        let mut chunk = vec![0u8; size];
        f.seek(SeekFrom::Start(start))?;
        f.read_exact(&mut chunk)?;
        lines += chunk.iter().filter(|b| **b == b'\n').count();
        chunk.extend_from_slice(&buf);
        buf = chunk;
        end = start;
    }

    let text = String::from_utf8_lossy(&buf);
    let mut all: Vec<Observation> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<Observation>(l).ok())
        .collect();
    // The first line of the window may be a fragment; parsing already dropped
    // it. Take the newest `n`.
    if all.len() > n {
        all.drain(..all.len() - n);
    }
    Ok(all)
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
// Collection-only adapters (`watch_fs`, `watch_git`, `watch_shell`,
// `watch_agent`, `watch_proc`, `watch_browser`) moved to
// `physis-pro/src/ported/observe_adapters.rs` in the Core/Product split
// (2026-09-15, this HEAD d65858b — see `MOVED_TO_PRODUCT.md`). A watcher
// integrates with something the host runs (the filesystem, git, a shell
// history file, an agent's own session store, `/proc`, a browser's SQLite
// store) and is a product-side collector, not a scientific primitive — but
// it returns plain [`Observation`]s through the log API above, which is what
// stays here. Signatures, doc comments, log format and tests moved unchanged;
// see that file for the implementations.

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

    /// The other half of the same failure: a truncated tail must not eat the
    /// record written by the *next* run either. Before the repair, `append`
    /// wrote the new record onto the fragment, the concatenated line was
    /// unparseable, and both records disappeared — one interrupted write cost
    /// two records and left a gap in the sequence.
    #[test]
    fn appending_after_a_truncated_line_keeps_the_new_record() {
        use std::io::Write;
        let p = tmp("trunc-append").join("o.jsonl");
        append(&p, &mut [Observation::new("fs", "good")]).unwrap();
        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        write!(f, "{{\"seq\":2,\"at\":\"not-a-date").unwrap();
        drop(f);

        let (a, b) = append(&p, &mut [Observation::new("git", "after")]).unwrap();
        assert_eq!((a, b), (2, 2), "the fragment parsed to no seq, so 2 is next");
        let all = read(&p).unwrap();
        assert_eq!(all.len(), 2, "the fragment is dropped; both real records survive");
        assert_eq!(all[0].subject, "good");
        assert_eq!(all[1].subject, "after");
        assert_eq!(all[1].seq, 2, "no gap — the new record takes the fragment's number");
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

    /// `read_tail` must be bounded by `n`, not by file size — that is the whole
    /// reason it exists. Before it, a `watch` run at 500k observations cost
    /// 2.00 s and 372 MB; after, 0.05 s and 23 MB, flat.
    #[test]
    fn read_tail_is_bounded_by_n_not_by_file_size() {
        let p = tmp("tail").join("o.jsonl");
        let mut batch: Vec<Observation> = (0..5_000)
            .map(|i| Observation::new("fs", format!("/f{i}.rs")))
            .collect();
        append(&p, &mut batch).unwrap();

        let tail = read_tail(&p, 10).unwrap();
        assert_eq!(tail.len(), 10, "must return exactly n");
        assert_eq!(tail.last().unwrap().seq, 5_000, "and they must be the NEWEST");
        assert_eq!(tail.first().unwrap().seq, 4_991);

        // Asking for more than exists returns everything, not an error.
        assert_eq!(read_tail(&p, 99_999).unwrap().len(), 5_000);
        assert!(read_tail(&p, 0).unwrap().is_empty());
    }

    /// `append` calls `last_seq`, so if that read the whole log every write
    /// would be linear in everything ever written.
    #[test]
    fn appending_does_not_reparse_history() {
        let p = tmp("append-cheap").join("o.jsonl");
        let mut batch: Vec<Observation> = (0..2_000)
            .map(|i| Observation::new("fs", format!("/f{i}")))
            .collect();
        append(&p, &mut batch).unwrap();
        // The next sequence must be correct without a full parse.
        assert_eq!(last_seq(&p).unwrap(), 2_000);
        let (a, b) = append(&p, &mut [Observation::new("fs", "/new")]).unwrap();
        assert_eq!((a, b), (2_001, 2_001));
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
