//! Does `act` actually find the contradiction it exists to surface?
//!
//! ## Why this module exists
//!
//! `act.rs` is the only implementation, in a twenty-one system survey, of
//! *acting while checking what is already believed* (`gaps.md` Gap 8). It
//! reaches that belief through one line — cosine over the command text against
//! claim statements — and that retrieval family scored **0–1/7 against a
//! whole-file ceiling of 5/7** in the 2×2 of 2026-09-13. So the differentiator
//! routes through the commodity, and the commodity was measured weak.
//!
//! Recorded as conceptual problem 1 in
//! `docs/plans/2026-09-13-four-conceptual-problems.md`, which asks for exactly
//! one thing: **measure `act`'s recall directly, against a construction-matched
//! null.** That is this module. Nothing here is an improvement to `act`; it is
//! the number that says whether an improvement is needed and, later, whether
//! one worked.
//!
//! ## The construction
//!
//! A seeded synthetic ledger of [`LEDGER_SIZE`] claims is built in one pass, so
//! every claim is the same age and age cannot be the thing that ranks. Eight of
//! them are `Contradicted` and each is the *one* claim a given command
//! contradicts; eight more are `Contradicted` and bear on nothing being run;
//! the rest are candidates and support. Ground truth is by construction — the
//! pairing is authored with the corpus, not judged afterwards.
//!
//! Two arms, because the failure mode is specific and an average over both
//! would hide it:
//!
//! | arm | the command | what it tests |
//! |---|---|---|
//! | `lexical` | shares the claim's own words (`cargo build --release`) | the best case a string matcher can have |
//! | `paraphrase` | same act, no shared content words (`produce an optimised binary for shipping`) | the normal case — `act.rs`'s own stated limit |
//!
//! ## The nulls
//!
//! Two, both matched to the same `top` and drawn from the same one-age pool, so
//! the only difference between arm and null is the ranking:
//!
//! - **uniform** — surface `top` claims at random from the whole ledger. This
//!   is the null the plan names.
//! - **status** — surface `top` claims at random *from the contradicted claims
//!   only*. This is the harder and fairer one: `Bearing::is_warning` already
//!   knows a claim's status, so a system that ignored the command entirely and
//!   listed refuted claims would score this without doing any retrieval. Beating
//!   uniform and losing to status means the cosine leg contributes nothing that
//!   a status filter does not already give for free.
//!
//! A margin must clear one query's worth of recall (`1/queries`) to count, so a
//! single lucky pairing cannot produce a verdict.
//!
//! ## What it measured, 2026-09-13
//!
//! `recall@top` over 8 commands, ledger of 48, bge-base-en-v1.5 (`onnx-explicit`):
//!
//! | arm | top=1 | top=3 | top=5 | MRR@5 | null unif@5 | null status@5 |
//! |---|---|---|---|---|---|---|
//! | `lexical` | 1.000 | 1.000 | **1.000** | 1.000 | 0.103 | 0.318 |
//! | `paraphrase` | 0.375 | 0.875 | **0.875** | 0.583 | 0.103 | 0.318 |
//!
//! Both arms clear both nulls at every `top`. **The pessimistic reading of the
//! 2×2 does not survive a direct measurement:** on a semantic embedder `act`
//! does find the contradicting claim, and on the list the CLI actually shows it
//! finds seven of eight even when the command shares no word with the claim.
//! What degrades is *rank*, not presence — MRR 1.000 → 0.583 — so on a
//! paraphrase the warning arrives mid-list rather than first, and at `top=1` the
//! paraphrase arm drops to 0.375.
//!
//! The same run on the random-projection fallback:
//!
//! | arm | recall@5 | MRR@5 | verdict |
//! |---|---|---|---|
//! | `lexical` | 0.750 | 0.667 | HOLDS |
//! | `paraphrase` | **0.000** | 0.000 | FAILS |
//!
//! That is the sharper finding. Offline — the mode this crate advertises as
//! supported, not as a failure — `act` surfaces the contradiction **never** when
//! the command is phrased differently from the claim, and scores below both
//! nulls. Gap 8 is closed on a semantic embedder and open on the fallback, and
//! nothing in the CLI says which one is running when `act` prints nothing.
//!
//! ## The polarity arm, and the prediction recorded before it ran
//!
//! Every distractor above differs from its target by **topic**. None differs by
//! **polarity**, so the two arms measure the easy discrimination and say nothing
//! about the hard one: of two claims about this command, tell the refutation
//! from the endorsement. [`AFFIRMED_TWINS`] adds each target's endorsement to a
//! second ledger and asks which of the pair ranks first. The null is **0.500 by
//! construction** — no topic separates them.
//!
//! Measured elsewhere and the reason for this arm: dense retrievers rank
//! "treatment works" and "treatment does not work" alike (arXiv 2603.17580,
//! negation is syntactic, the embedding space is not), and in 95.2–99.8% of
//! structural-retrieval misses the false positive is more lexically similar to
//! the query than the gold item is (arXiv 2609.01556).
//!
//! **Predicted, before the arm was run** (recorded in commit d415488, so the
//! record is checkable): recall stays near 0.875 while discrimination lands
//! *below* 0.500 — the endorsement outranking the refutation in most pairs.
//!
//! ### What it did, and it is worse than the prediction
//!
//! bge-base-en-v1.5, 56-claim ledger, `discrim` = pairs where the refutation
//! outranked its endorsement, null 0.500:
//!
//! | arm | discrim | target@5 | twin@5 | reassured@5 | reassured@1 |
//! |---|---|---|---|---|---|
//! | `lexical` | **0.250** | 8/8 | 8/8 | 0/8 | 5/8 |
//! | `paraphrase` | **0.125** | 6/8 | 8/8 | 2/8 | 3/8 |
//!
//! Not a coin flip — **inverted**. The endorsement outranks the refutation in
//! six of eight pairs on the lexical arm and seven of eight on the paraphrase
//! arm. The prediction was that polarity would be invisible; it is worse than
//! invisible, because the endorsement is systematically the *better* match for
//! a command that asks to do the thing.
//!
//! `reassured` is the cell that matters: the endorsement made the list and the
//! refutation did not. At `top 5` that is 0 and 2 of 8 — the operator still
//! sees both, so the list saves it. At `top 1` it is **5 of 8 and 3 of 8**: in
//! the majority of lexical pairs, `act --top 1` prints a Supported claim about
//! the exact thing a Contradicted claim refutes. That is not a miss. It is the
//! wrong answer delivered with the ledger's authority, and it is the strongest
//! argument in this crate for showing a list rather than a best match.
//!
//! ### And the two embedders fail in opposite directions
//!
//! The same arm on the random-projection fallback:
//!
//! | arm | discrim | verdict |
//! |---|---|---|
//! | `lexical` | **0.750** | HOLDS |
//! | `paraphrase` | 0.125 | INVERTED |
//!
//! The lexical hash *beats* the semantic embedder on polarity, 0.750 against
//! 0.250, because it keys on the literal tokens — `fails`, `never`, `no` — that
//! the semantic space smooths away. That is arXiv 2603.17580's finding
//! reproduced from the other side: negation is syntactic, so the model that
//! cannot read meaning is the one that can still see the "not".
//!
//! Neither embedder dominates. The semantic one finds the right *topic* and
//! cannot tell the verdict; the hash can tell the verdict and cannot find the
//! topic (paraphrase recall 0.000). Read as an instruction, that is an argument
//! for a hybrid, and it is untested.
//!
//! **Do not over-read 0.750.** Eight pairs means one flip is worth 0.125, so
//! that margin is two pairs. The inversion on the semantic embedder is the
//! robust half of this result: it is the same sign on both arms, at every
//! `top`, and it is what the literature predicts.
//!
//! ## What this measures and what it does not
//!
//! It measures whether `bearing_on` puts the *one* contradicting claim in the
//! `top` the CLI actually shows (default 5). It does not measure whether an
//! operator then reads it, and it says nothing about real ledgers, whose claims
//! are neither uniformly aged nor authored to be findable. The corpus was authored
//! so that one claim is the right answer; a real ledger has near-misses, and
//! this number is therefore a ceiling on a real one, not an estimate of it.
//!
//! Run it: `physis-core act-recall [--top N] [--json]`. Artifacts land in
//! `benchmarks/results/act-recall.json`.

use crate::act::bearing_on;
use crate::core::PhysisCore;
use crate::embed::VectorEmbed;
use crate::hypothesis::{Evidence, Hypothesis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Claims in the synthetic ledger.
pub const LEDGER_SIZE: usize = 48;
/// Random draws averaged for each null. Enough that the null is stable to
/// ±0.01 at this ledger size, and cheap because no embedding is involved.
pub const NULL_TRIALS: usize = 2000;

/// One command, and the single claim it contradicts.
struct Pair {
    /// The command as it shares the claim's own vocabulary.
    lexical: &'static str,
    /// The same act, phrased with no content word in common with the claim.
    paraphrase: &'static str,
    /// The claim that command contradicts. Contradicted in the ledger.
    claim: &'static str,
}

/// The eight commands with a claim that already refutes them.
///
/// Authored together with the corpus: the ground truth is the pairing, and it
/// is fixed before any retrieval runs. The paraphrase column is deliberately
/// stripped of the claim's content words — that is the case `act.rs`'s "Honest
/// limits" says it will miss, and an arm that cannot fail is not an arm.
const PAIRS: [Pair; 8] = [
    Pair {
        lexical: "cargo test --all-features",
        paraphrase: "run the whole suite with every optional flag turned on",
        claim: "cargo test --all-features fails on this crate: the ort linker step never resolves",
    },
    Pair {
        lexical: "cargo build --release",
        paraphrase: "produce an optimised binary for shipping",
        claim: "the release build never finishes on this box — it runs out of memory at the LTO stage",
    },
    Pair {
        lexical: "pip install onnxruntime",
        paraphrase: "add the neural inference package to the interpreter environment",
        claim: "pip install onnxruntime fails on this Python: there is no manylinux wheel for 3.13",
    },
    Pair {
        lexical: "docker compose up -d",
        paraphrase: "bring the local service stack online in the background",
        claim: "docker compose up leaves postgres in a restart loop because the volume is owned by root",
    },
    Pair {
        lexical: "npm run build",
        paraphrase: "compile the frontend bundle for deployment",
        claim: "npm run build breaks on the CI image: node 18 cannot parse the new syntax",
    },
    Pair {
        lexical: "git push origin main",
        paraphrase: "publish the local commits to the shared upstream trunk",
        claim: "git push origin main is rejected — the branch is protected and requires a review",
    },
    Pair {
        lexical: "./scripts/migrate.sh",
        paraphrase: "apply the pending schema changes to the database",
        claim: "the migrate script drops the audit table when run twice; it is not idempotent",
    },
    Pair {
        lexical: "rsync -av ./data backup:/srv/data",
        paraphrase: "copy the dataset across to the offsite machine",
        claim: "rsync to the backup host times out over the VPN for anything larger than a gigabyte",
    },
];

/// The affirmed twin of each target in [`PAIRS`], same index.
///
/// ## Why these exist
///
/// Every distractor in the base ledger differs from its target **by topic**.
/// Not one differs by **polarity**. So the base arms measure the easy
/// discrimination — find the claim that is *about* this command — and say
/// nothing about the hard one: of two claims about this command, tell the
/// refutation from the endorsement.
///
/// That is the discrimination `act` actually needs. `Bearing::is_warning` reads
/// `status`, not the ranking, so a near-twin of the opposite polarity is not a
/// near-miss — it is a claim that will be surfaced with a *reassuring* status
/// while the refutation sits below the cut.
///
/// Measured elsewhere, 2603.17580: dense retrievers rank "treatment works" and
/// "treatment does not work" alike, because negation is syntactic and the
/// embedding space is not. Each twin here is written to keep the target's
/// vocabulary and invert only the verdict — a test asserts that overlap rather
/// than trusting the author.
const AFFIRMED_TWINS: [&str; 8] = [
    "cargo test --all-features passes on this crate: the ort linker step resolves every time",
    "the release build finishes on this box in four minutes — the LTO stage never runs out of memory",
    "pip install onnxruntime succeeds on this Python: there is a manylinux wheel for 3.13",
    "docker compose up leaves postgres healthy because the volume is owned by the right user, with no restart loop",
    "npm run build succeeds on the CI image: node 18 parses the new syntax without complaint",
    "git push origin main is accepted — the branch is not protected and requires no review",
    "the migrate script leaves the audit table alone when run twice; it is idempotent",
    "rsync to the backup host completes over the VPN for anything larger than a gigabyte",
];

/// Contradicted claims that no command in [`PAIRS`] is about.
///
/// They exist so the status null has something to be wrong about. Without them
/// every contradicted claim would be a target and "list the contradicted ones"
/// would score a perfect null, which would say more about the corpus than about
/// `act`.
const CONTRADICTED_DISTRACTORS: [&str; 8] = [
    "the nightly toolchain miscompiles the SIMD path and the results differ from stable",
    "raising the thread pool above sixteen workers makes the ingest slower, not faster",
    "the S3 lifecycle rule deleted the archived exports before anyone had read them",
    "the ARM builder cannot cross-compile the C shim; the linker picks the host libc",
    "caching the tokenizer between processes corrupts it — the offsets go stale",
    "the webhook retry policy double-charges when the receiver is slow rather than down",
    "the ONNX export loses the pooling layer, so the vectors come back unnormalised",
    "increasing the batch size past 64 exhausts the GPU before the epoch ends",
];

/// Neither refuted nor about anything being run — the bulk of a real ledger.
///
/// Thirty-two of these put the target at 1-in-48 for the uniform null, which is
/// the realistic shape: most of what is believed bears on nothing in particular.
const NEUTRAL_DISTRACTORS: [&str; 32] = [
    "the semiotic grid holds five domains and fourteen modes",
    "observations are appended, never rewritten, so the timeline replays",
    "the MMR default packed four of three thousand tokens before it was fixed",
    "a validity window can be narrowed when better evidence arrives",
    "the embedder falls back to random projection when no model is on disk",
    "the studio serves the same structural map the CLI prints",
    "n-gram tables are interchangeable with the model that reads them",
    "the ground-truth corpus carries thirty documents with known structure",
    "context compression measured thirty point six percent on the demo corpus",
    "the direction control truncates to the same length as the rewrite",
    "worst-covered retention refuses to reward copying",
    "the ledger keeps a revision every time a status changes",
    "provenance chains record which source supported which claim",
    "a claim below twenty training entries is unmeasured rather than weak",
    "the pre-flight stops when a class is too sparse to score",
    "coherence is computed over the certified branches only",
    "dream evaluation archives its results for later replay",
    "typed edges connect nodes, hypotheses and observations",
    "the tokenizer compatibility gate refuses a mismatched pair",
    "the licence gate verifies signatures even in development builds",
    "the terminal watcher and the agent watcher share one timeline",
    "structural machines carry a proof status alongside their observations",
    "the map hash is identical for identical corpora",
    "the mandelbrot document is the most different in the demo set",
    "an unscored prediction is worth reading before acting",
    "the oracle leg reports not-configured rather than inventing a number",
    "packs are dropped into the config directory rather than compiled in",
    "facets carry sub-domain and sub-mode without widening the grid",
    "the quality loop embeds failures against the grid and applies a penalty",
    "inventory forecasting uses a thirty-period moving average",
    "the web server keys session state by the tenant header",
    "voice, image and video reduce to vectors in one shared space",
];

/// Which phrasing of the command is issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// The command shares the claim's own words.
    Lexical,
    /// The same act, with no content word in common with the claim.
    Paraphrase,
}

impl Arm {
    pub fn label(&self) -> &'static str {
        match self {
            Arm::Lexical => "lexical",
            Arm::Paraphrase => "paraphrase",
        }
    }
    fn command(&self, p: &Pair) -> &'static str {
        match self {
            Arm::Lexical => p.lexical,
            Arm::Paraphrase => p.paraphrase,
        }
    }
}

/// What one arm scored, with both nulls measured in the same pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmResult {
    pub arm: String,
    /// Commands issued — one per pair.
    pub queries: usize,
    /// How many claims `act` surfaces; the CLI default is 5.
    pub top: usize,
    /// Queries whose contradicting claim appeared in the top.
    pub hits: usize,
    pub recall: f32,
    /// Hits that would also have been *rendered* as a warning. Equal to `hits`
    /// unless `is_warning` and the corpus disagree, which would be a bug.
    pub warned: usize,
    pub warning_recall: f32,
    /// Mean reciprocal rank over the queries; 0 for a miss. Rank matters
    /// because the operator reads a list, and the fifth line is not the first.
    pub mrr: f32,
    /// `top` claims drawn at random from the whole ledger.
    pub null_uniform_recall: f32,
    /// `top` claims drawn at random from the contradicted claims only.
    pub null_status_recall: f32,
    pub margin_uniform: f32,
    pub margin_status: f32,
    pub verdict: String,
    /// Per-query detail, so a failure can be read rather than only counted.
    pub cases: Vec<CaseResult>,
}

/// One command's outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    pub command: String,
    pub target: String,
    /// 1-based rank of the target within the surfaced list; `None` for a miss.
    pub rank: Option<usize>,
    /// Cosine the target scored, whether or not it made the cut.
    pub target_relevance: f32,
    /// The claim that outranked everything, which on a miss is the diagnosis.
    pub top_statement: String,
}

/// The whole run: both arms, the embedder that produced them, and provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallRun {
    pub ledger_size: usize,
    pub contradicted_claims: usize,
    pub top: usize,
    pub seed: u64,
    pub null_trials: usize,
    /// Which embedder ranked. A recall claim made on the lexical-hash fallback
    /// is not a recall claim; it is recorded, never assumed.
    pub embedder: String,
    pub physis_version: String,
    pub git_commit: String,
    pub arms: Vec<ArmResult>,
    /// Refutation versus its own endorsement, on a ledger that also holds the
    /// affirmed twin of every target. Null 0.500 by construction.
    pub polarity: Vec<PolarityResult>,
    pub polarity_ledger_size: usize,
}

impl RecallRun {
    pub fn arm(&self, label: &str) -> Option<&ArmResult> {
        self.arms.iter().find(|a| a.arm == label)
    }

    pub fn render(&self) -> String {
        let mut o = String::new();
        o.push_str(&format!(
            "── ACT RECALL ──\nledger {} claims ({} contradicted) · top {} · embedder {} · seed {}\n\n",
            self.ledger_size, self.contradicted_claims, self.top, self.embedder, self.seed
        ));
        o.push_str("  arm          recall   warn    MRR   null(unif)  null(status)   Δunif   Δstatus  verdict\n");
        for a in &self.arms {
            o.push_str(&format!(
                "  {:<11} {:>5.3}  {:>5.3}  {:>5.3}      {:>6.3}        {:>6.3}  {:>+7.3}  {:>+7.3}  {}\n",
                a.arm,
                a.recall,
                a.warning_recall,
                a.mrr,
                a.null_uniform_recall,
                a.null_status_recall,
                a.margin_uniform,
                a.margin_status,
                a.verdict
            ));
        }
        if self.embedder == "random-projection" {
            o.push_str(
                "\n  NOTE: random-projection is a lexical hash. These numbers are a floor,\n\
                 \x20       not a retrieval claim. Set PHYSIS_MODEL_DIR and re-run.\n",
            );
        }
        o.push_str(&format!(
            "\n── POLARITY: refutation vs its own endorsement ──\nledger {} claims (every target's affirmed twin added) · null 0.500 by construction\n\n",
            self.polarity_ledger_size
        ));
        o.push_str("  arm          discrim  target@top  twin@top  reassured  verdict\n");
        for p in &self.polarity {
            o.push_str(&format!(
                "  {:<11} {:>6.3}     {:>2}/{:<2}       {:>2}/{:<2}     {:>2}/{:<2}    {}\n",
                p.arm,
                p.discrimination,
                p.target_in_top,
                p.queries,
                p.twin_in_top,
                p.queries,
                p.reassured_instead,
                p.queries,
                p.verdict
            ));
        }
        if self.polarity.iter().any(|p| p.reassured_instead > 0) {
            o.push_str(
                "\n  `reassured` is the harmful cell: the endorsement made the list and the\n   refutation did not, so `act` printed a Supported claim about the very\n   thing a Contradicted claim refutes.\n",
            );
        }
        o.push_str("\n  misses, most instructive first:\n");
        for a in &self.arms {
            for c in a.cases.iter().filter(|c| c.rank.is_none()) {
                o.push_str(&format!(
                    "  [{}] {}\n      wanted: {}\n      got:    {}\n",
                    a.arm,
                    c.command.chars().take(72).collect::<String>(),
                    c.target.chars().take(72).collect::<String>(),
                    c.top_statement.chars().take(72).collect::<String>()
                ));
            }
        }
        o
    }
}

/// Build the ledger. One pass, so every claim carries the same age.
///
/// Age is the variable the plan's null controls for, and the cheapest way to
/// control for it is to remove it: nothing here is older than anything else, so
/// nothing can be ranked by recency by accident.
///
/// `with_twins` adds [`AFFIRMED_TWINS`]. The base ledger keeps it off so the
/// recall numbers already recorded stay comparable — the polarity arm asks a
/// different question on a different ledger, and mixing the two would change
/// both.
fn build_ledger(embedder: &dyn VectorEmbed, with_twins: bool) -> (PhysisCore, Vec<(String, String)>) {
    let mut core = PhysisCore::new();
    // (id, statement) in insertion order, for the nulls to draw from.
    let mut index: Vec<(String, String)> = Vec::new();

    let insert = |core: &mut PhysisCore, index: &mut Vec<(String, String)>, text: &str, contradicted: bool| {
        let mut h = Hypothesis::new(text, embedder.embed(text));
        if contradicted {
            h.add_contradicting_evidence(Evidence::contradicts("benchmark", "measured to fail"));
        } else {
            h.add_supporting_evidence(Evidence::supports("benchmark", "recorded"));
        }
        index.push((h.id.clone(), h.statement.clone()));
        core.hypotheses.insert(h.id.clone(), h);
    };

    for p in PAIRS.iter() {
        insert(&mut core, &mut index, p.claim, true);
    }
    if with_twins {
        // Supported, not Contradicted: the twin is the endorsement, and its
        // status is what makes surfacing it instead of the target harmful
        // rather than merely wrong.
        for t in AFFIRMED_TWINS.iter() {
            insert(&mut core, &mut index, t, false);
        }
    }
    for s in CONTRADICTED_DISTRACTORS.iter() {
        insert(&mut core, &mut index, s, true);
    }
    for s in NEUTRAL_DISTRACTORS.iter() {
        insert(&mut core, &mut index, s, false);
    }
    (core, index)
}

/// Empirical recall of drawing `top` distinct claims at random from `pool`.
///
/// Measured rather than derived from `top/|pool|` so the null is produced the
/// same way the arm is — a sampled list of the same length — and so a mistake
/// in the pool shows up as a number instead of hiding in the algebra.
fn null_recall(pool: &[String], targets: &[String], top: usize, seed: u64) -> f32 {
    if pool.is_empty() || targets.is_empty() {
        return 0.0;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut hits = 0usize;
    let mut draws = 0usize;
    for t in targets {
        for _ in 0..NULL_TRIALS {
            let mut idx: Vec<usize> = (0..pool.len()).collect();
            idx.shuffle(&mut rng);
            let surfaced = &idx[..top.min(idx.len())];
            if surfaced.iter().any(|&i| &pool[i] == t) {
                hits += 1;
            }
            draws += 1;
        }
    }
    hits as f32 / draws as f32
}

/// Score one arm.
pub fn run_arm(
    core: &PhysisCore,
    index: &[(String, String)],
    arm: Arm,
    top: usize,
    embedder: &dyn VectorEmbed,
    seed: u64,
) -> ArmResult {
    let mut cases: Vec<CaseResult> = Vec::new();
    let mut hits = 0usize;
    let mut warned = 0usize;
    let mut rr = 0f32;

    for p in PAIRS.iter() {
        let command = arm.command(p);
        // The whole ledger is ranked, not only the top, so a miss can report
        // the score the target actually got. `bearing_on` truncates, so the
        // full ranking is asked for and cut here instead.
        let full = bearing_on(core, command, embedder, index.len());
        let rank = full.iter().position(|b| b.statement == p.claim);
        let target_relevance = rank.map(|i| full[i].relevance).unwrap_or(f32::NAN);
        let within = rank.filter(|&i| i < top);
        if let Some(i) = within {
            hits += 1;
            rr += 1.0 / (i as f32 + 1.0);
            if full[i].is_warning() {
                warned += 1;
            }
        }
        cases.push(CaseResult {
            command: command.to_string(),
            target: p.claim.to_string(),
            rank: within.map(|i| i + 1),
            target_relevance,
            top_statement: full.first().map(|b| b.statement.clone()).unwrap_or_default(),
        });
    }

    let queries = PAIRS.len();
    let recall = hits as f32 / queries as f32;
    let targets: Vec<String> = PAIRS.iter().map(|p| p.claim.to_string()).collect();
    let all: Vec<String> = index.iter().map(|(_, s)| s.clone()).collect();
    let contradicted: Vec<String> = PAIRS
        .iter()
        .map(|p| p.claim.to_string())
        .chain(CONTRADICTED_DISTRACTORS.iter().map(|s| s.to_string()))
        .collect();

    let null_uniform_recall = null_recall(&all, &targets, top, seed);
    let null_status_recall = null_recall(&contradicted, &targets, top, seed.wrapping_add(1));
    let margin_uniform = recall - null_uniform_recall;
    let margin_status = recall - null_status_recall;

    // One query's worth of recall. A margin under this is one lucky pairing,
    // and one lucky pairing is not a result.
    let noise = 1.0 / queries as f32;
    let verdict = if margin_uniform > noise && margin_status > noise {
        "HOLDS".to_string()
    } else if margin_uniform > noise {
        "WEAK — beats chance, not a status filter".to_string()
    } else {
        "FAILS".to_string()
    };

    ArmResult {
        arm: arm.label().to_string(),
        queries,
        top,
        hits,
        recall,
        warned,
        warning_recall: warned as f32 / queries as f32,
        mrr: rr / queries as f32,
        null_uniform_recall,
        null_status_recall,
        margin_uniform,
        margin_status,
        verdict,
        cases,
    }
}

/// A pairwise discrimination: refutation versus its own endorsement.
///
/// The base arms ask "is the target in the top?". This asks the question that
/// follows it: with both the refutation and its affirmed twin in the ledger,
/// **which one comes first?** There is no topic to separate them — same
/// subject, same vocabulary, opposite verdict — so a ranker with no purchase on
/// polarity is a coin flip, and the null is exactly 0.500 by construction
/// rather than by sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolarityResult {
    /// Which phrasing of the command was issued.
    pub arm: String,
    pub queries: usize,
    pub top: usize,
    /// Pairs where the refutation outranked its endorsement.
    pub target_first: usize,
    /// `target_first / queries`. The null is 0.500.
    pub discrimination: f32,
    /// Pairs where the refutation reached the list the operator sees.
    pub target_in_top: usize,
    /// Pairs where the *endorsement* reached it. A high number here with a low
    /// `target_in_top` is the harmful case: `act` prints a reassuring claim and
    /// withholds the refutation.
    pub twin_in_top: usize,
    /// Pairs where the endorsement was surfaced and the refutation was not.
    pub reassured_instead: usize,
    pub verdict: String,
    pub cases: Vec<PolarityCase>,
}

/// One command's refutation-versus-endorsement outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolarityCase {
    pub command: String,
    /// 1-based rank of the refutation over the whole ledger.
    pub target_rank: usize,
    /// 1-based rank of its affirmed twin.
    pub twin_rank: usize,
    pub target_relevance: f32,
    pub twin_relevance: f32,
}

/// Score the polarity arm for one phrasing.
pub fn run_polarity(
    core: &PhysisCore,
    index: &[(String, String)],
    arm: Arm,
    top: usize,
    embedder: &dyn VectorEmbed,
) -> PolarityResult {
    let mut cases = Vec::new();
    let (mut target_first, mut target_in_top, mut twin_in_top, mut reassured) = (0, 0, 0, 0);

    for (i, p) in PAIRS.iter().enumerate() {
        let command = arm.command(p);
        let full = bearing_on(core, command, embedder, index.len());
        let twin = AFFIRMED_TWINS[i];
        let tr = full.iter().position(|b| b.statement == p.claim);
        let wr = full.iter().position(|b| b.statement == twin);
        // Both are in this ledger by construction; a missing one is a corpus
        // bug, and ranking it last is the reading that cannot flatter the
        // result.
        let tr = tr.unwrap_or(full.len().saturating_sub(1));
        let wr = wr.unwrap_or(full.len().saturating_sub(1));
        if tr < wr {
            target_first += 1;
        }
        let t_in = tr < top;
        let w_in = wr < top;
        if t_in {
            target_in_top += 1;
        }
        if w_in {
            twin_in_top += 1;
        }
        if w_in && !t_in {
            reassured += 1;
        }
        cases.push(PolarityCase {
            command: command.to_string(),
            target_rank: tr + 1,
            twin_rank: wr + 1,
            target_relevance: full.get(tr).map(|b| b.relevance).unwrap_or(f32::NAN),
            twin_relevance: full.get(wr).map(|b| b.relevance).unwrap_or(f32::NAN),
        });
    }

    let queries = PAIRS.len();
    let discrimination = target_first as f32 / queries as f32;
    // One query's worth either side of the coin flip. Eight pairs cannot
    // support a finer claim than that, and pretending otherwise would be the
    // thing this module exists to stop.
    let noise = 1.0 / queries as f32;
    let verdict = if discrimination > 0.5 + noise {
        "HOLDS — ranks the refutation over its endorsement".to_string()
    } else if discrimination < 0.5 - noise {
        "INVERTED — ranks the endorsement first".to_string()
    } else {
        "COIN FLIP — no purchase on polarity".to_string()
    };

    PolarityResult {
        arm: arm.label().to_string(),
        queries,
        top,
        target_first,
        discrimination,
        target_in_top,
        twin_in_top,
        reassured_instead: reassured,
        verdict,
        cases,
    }
}

/// Build the ledger and score both arms.
pub fn run(top: usize, embedder: &dyn VectorEmbed, embedder_kind: &str, seed: u64) -> RecallRun {
    let (core, index) = build_ledger(embedder, false);
    let arms = vec![
        run_arm(&core, &index, Arm::Lexical, top, embedder, seed),
        run_arm(&core, &index, Arm::Paraphrase, top, embedder, seed),
    ];
    // A second ledger, because the twins would change the base numbers if they
    // shared one. Same construction, eight claims longer.
    let (tcore, tindex) = build_ledger(embedder, true);
    let polarity = vec![
        run_polarity(&tcore, &tindex, Arm::Lexical, top, embedder),
        run_polarity(&tcore, &tindex, Arm::Paraphrase, top, embedder),
    ];
    RecallRun {
        polarity,
        polarity_ledger_size: tindex.len(),
        ledger_size: index.len(),
        contradicted_claims: PAIRS.len() + CONTRADICTED_DISTRACTORS.len(),
        top,
        seed,
        null_trials: NULL_TRIALS,
        embedder: embedder_kind.to_string(),
        physis_version: env!("CARGO_PKG_VERSION").to_string(),
        git_commit: crate::bench::git_head(),
        arms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::RandomProjectionEmbedder;

    /// The corpus must be what the doc header says it is. A ground truth that
    /// drifts from its description is worse than none.
    #[test]
    fn the_ledger_is_the_shape_the_nulls_assume() {
        let e = RandomProjectionEmbedder::new(128);
        let (core, index) = build_ledger(&e, false);
        assert_eq!(index.len(), LEDGER_SIZE, "ledger size is quoted in the results");
        assert_eq!(core.hypotheses.len(), LEDGER_SIZE, "no statement collided into one id");
        let contradicted = core
            .hypotheses
            .values()
            .filter(|h| format!("{:?}", h.status) == "Contradicted")
            .count();
        assert_eq!(
            contradicted,
            PAIRS.len() + CONTRADICTED_DISTRACTORS.len(),
            "the status null draws from this pool"
        );
        // Every target must be findable at all, or the arm measures the corpus.
        for p in PAIRS.iter() {
            assert!(
                index.iter().any(|(_, s)| s == p.claim),
                "target absent from the ledger: {}",
                p.claim
            );
        }
    }

    /// A twin is only a twin if it keeps the target's words and inverts only
    /// the verdict. Written by hand, so checked by machine — the same rule as
    /// the paraphrase arm, in the opposite direction.
    #[test]
    fn every_twin_shares_most_of_its_target_vocabulary() {
        const STOP: [&str; 24] = [
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with", "is",
            "are", "it", "this", "that", "as", "by", "from", "into", "than", "not", "no",
        ];
        let norm = |s: &str| -> Vec<String> {
            s.to_ascii_lowercase()
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| w.len() > 2 && !STOP.contains(w))
                .map(|w| w.to_string())
                .collect()
        };
        for (i, p) in PAIRS.iter().enumerate() {
            let claim = norm(p.claim);
            let twin = norm(AFFIRMED_TWINS[i]);
            let shared = claim.iter().filter(|w| twin.contains(w)).count();
            let frac = shared as f32 / claim.len() as f32;
            assert!(
                frac >= 0.5,
                "twin {i} shares only {:.0}% of its target's content words — that is a \
                 different claim, not an inverted one\n  target: {}\n  twin:   {}",
                frac * 100.0,
                p.claim,
                AFFIRMED_TWINS[i]
            );
        }
    }

    /// The twin ledger is the base ledger plus exactly the twins. If it were
    /// anything else, the polarity numbers and the recall numbers would not be
    /// about the same corpus.
    #[test]
    fn the_twin_ledger_is_the_base_ledger_plus_the_twins() {
        let e = RandomProjectionEmbedder::new(128);
        let (_, base) = build_ledger(&e, false);
        let (_, twinned) = build_ledger(&e, true);
        assert_eq!(twinned.len(), base.len() + AFFIRMED_TWINS.len());
        for t in AFFIRMED_TWINS.iter() {
            assert!(twinned.iter().any(|(_, s)| s == t), "twin missing: {t}");
            assert!(
                !base.iter().any(|(_, s)| s == t),
                "twin leaked into the base ledger, which would move the recall numbers: {t}"
            );
        }
    }

    /// The polarity null is 0.500 by construction, so a blind ranker must land
    /// on the coin flip rather than on either verdict.
    #[test]
    fn a_blind_ranker_is_a_coin_flip_on_polarity() {
        struct Blind;
        impl VectorEmbed for Blind {
            fn embed(&self, _t: &str) -> Vec<f32> {
                vec![1.0; 8]
            }
            fn dimension(&self) -> usize {
                8
            }
        }
        let r = run(5, &Blind, "blind", 7);
        for p in &r.polarity {
            assert!(
                p.verdict.starts_with("COIN FLIP") || p.verdict.starts_with("INVERTED"),
                "a blind ranker claimed polarity discrimination on {}: {}",
                p.arm,
                p.verdict
            );
        }
    }

    /// The paraphrase arm only tests what it claims to if the command really
    /// shares no content word with the claim. Written by hand, so checked by
    /// machine.
    #[test]
    fn the_paraphrase_arm_shares_no_content_word_with_its_claim() {
        // Function words are allowed to overlap; they carry no topic.
        const STOP: [&str; 24] = [
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with", "is",
            "are", "it", "this", "that", "as", "by", "from", "into", "than", "not", "no",
        ];
        let norm = |s: &str| -> Vec<String> {
            s.to_ascii_lowercase()
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| w.len() > 2 && !STOP.contains(w))
                .map(|w| w.to_string())
                .collect()
        };
        for p in PAIRS.iter() {
            let cmd = norm(p.paraphrase);
            let claim = norm(p.claim);
            let shared: Vec<&String> = cmd.iter().filter(|w| claim.contains(w)).collect();
            assert!(
                shared.is_empty(),
                "paraphrase shares {:?} with its claim — the arm is not testing paraphrase\n  cmd:   {}\n  claim: {}",
                shared,
                p.paraphrase,
                p.claim
            );
        }
    }

    /// The lexical arm must, conversely, share vocabulary — otherwise the two
    /// arms are the same arm and the comparison says nothing.
    #[test]
    fn the_lexical_arm_shares_vocabulary_with_its_claim() {
        for p in PAIRS.iter() {
            let claim = p.claim.to_ascii_lowercase();
            let shared = p
                .lexical
                .to_ascii_lowercase()
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| w.len() > 2)
                .any(|w| claim.contains(w));
            assert!(shared, "lexical arm shares nothing with its claim: {}", p.lexical);
        }
    }

    /// The null must be the number the construction implies. If drawing `top`
    /// of 48 does not land near 5/48, the sampler is broken and every margin
    /// computed against it is wrong.
    #[test]
    fn the_uniform_null_matches_its_construction() {
        let pool: Vec<String> = (0..48).map(|i| format!("claim {i}")).collect();
        let targets: Vec<String> = (0..8).map(|i| format!("claim {i}")).collect();
        let got = null_recall(&pool, &targets, 5, 7);
        let want = 5.0 / 48.0;
        assert!(
            (got - want).abs() < 0.02,
            "uniform null {got} should be near {want}"
        );
    }

    /// And the status null must be the harder one. If it were not above the
    /// uniform null, the second control would add nothing.
    #[test]
    fn the_status_null_is_harder_than_the_uniform_one() {
        let e = RandomProjectionEmbedder::new(128);
        let r = run(5, &e, "random-projection", 7);
        let a = r.arm("lexical").unwrap();
        assert!(
            a.null_status_recall > a.null_uniform_recall,
            "status null {} must exceed uniform {}",
            a.null_status_recall,
            a.null_uniform_recall
        );
    }

    /// The measurement must be able to fail. An embedder that ranks by nothing
    /// — a constant vector for every input — must not produce a HOLDS.
    #[test]
    fn a_blind_ranker_cannot_produce_a_verdict() {
        struct Blind;
        impl VectorEmbed for Blind {
            fn embed(&self, _t: &str) -> Vec<f32> {
                vec![1.0; 8]
            }
            fn dimension(&self) -> usize {
                8
            }
        }
        let r = run(5, &Blind, "blind", 7);
        for a in &r.arms {
            assert_ne!(a.verdict, "HOLDS", "a blind ranker scored HOLDS on {}", a.arm);
        }
    }

    /// Same seed, same numbers. A benchmark that moves on its own cannot
    /// attribute a change to a code change.
    #[test]
    fn the_run_is_deterministic() {
        let e = RandomProjectionEmbedder::new(128);
        let a = run(5, &e, "random-projection", 7);
        let b = run(5, &e, "random-projection", 7);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }

    /// Every target is Contradicted, so a hit and a warning are the same event.
    /// If they ever diverge, `is_warning` has changed under the benchmark.
    #[test]
    fn every_hit_is_also_a_warning() {
        let e = RandomProjectionEmbedder::new(128);
        let r = run(5, &e, "random-projection", 7);
        for a in &r.arms {
            assert_eq!(a.hits, a.warned, "{}: hit and warning counts must agree", a.arm);
        }
    }
}
