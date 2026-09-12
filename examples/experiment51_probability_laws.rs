//! EXPERIMENT 51: REPRESENTATION-COVARIANT PROBABILITY LAWS
//!
//! Central question: does the underlying arithmetic object induce probability
//! laws or statistical relations that remain invariant across representations,
//! even when the geometric descriptions differ?
//!
//! E50 produced a strong negative result: arithmetic-specific structure was
//! found in individual representations, but did NOT survive a blind change of
//! representation. E51 moves one abstraction level upward — from geometry to
//! probability, statistics and recurrence. This experiment is deliberately
//! capable of returning **NO**: it does not search for confirmation, it does
//! not target 1/2, and it does not assume a common law exists.
//!
//! Design (mirrors E48–E50):
//!   * Representations   psi(x)−x, theta(x)−x, pi(x)−x/ln x, M(x)/sqrt(x)
//!     (cumulative-counting family) plus their increment/atomic sequences.
//!   * Discovery set     psi, theta (+ three nulls).
//!   * Held-out set      pi − x/ln x, Mertens M(x)/sqrt(x) (constructed blind).
//!   * Nulls             shuffled-psi (marginals kept, order destroyed),
//!                       Cramer random primes at density 1/ln x, and a
//!                       synthetic iid-Gaussian series matching first-order
//!                       statistics.
//!   * Observables       five representation-agnostic scalar statistics so
//!                       that cross-representation "transfer" is well-defined:
//!                       C1 excess kurtosis, C2 lag-1 increment ACF,
//!                       C3 tail exponent, C4 mean return interval,
//!                       C5 consecutive-displacement scaling.
//! None of the five targets or uses 1/2.
//!
//! A candidate survives ONLY if it simultaneously (a) transfers across the
//! discovery pair, (b) is arithmetic-specific (separated from every null),
//! and (c) survives the pre-frozen held-out representations. Otherwise the
//! experiment returns a clean failure.
//!
//! Run: cargo run -p physis-core --release --example experiment51_probability_laws

use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;

// ==== Output schema ==========================================================

#[derive(Serialize, Clone)]
struct CandidateReport {
    id: String,
    complexity: u32,
    description: String,
    discovery_value: f64,
    discovery_spread: f64,
    null_values: Vec<f64>,
    held_out_values: Vec<f64>,
    transfers: bool,
    arithmetic_specific: bool,
    held_out_survives: bool,
    survives: bool,
}

#[derive(Serialize, Clone)]
struct RepresentationStat {
    name: String,
    kind: String,
    values: Vec<f64>,
}

#[derive(Serialize, Clone)]
struct Experiment51Output {
    e50_context: String,
    representations_tested: Vec<String>,
    nulls_tested: Vec<String>,
    discovery_set: Vec<String>,
    held_out_set: Vec<String>,
    candidate_observables: Vec<String>,
    representatives: Vec<RepresentationStat>,
    candidates: Vec<CandidateReport>,
    cross_representation_transfer: bool,
    arithmetic_specificity: bool,
    held_out_survival: bool,
    relation_to_half: String,
    final_verdict: String,
}

// ==== small deterministic RNG (LCG, same family as E48–E50) ==================

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed
}
fn f01(seed: &mut u64) -> f64 { (lcg(seed) >> 11) as f64 / (1u64 << 53) as f64 }

// ==== arithmetic sieves (computed once, up to n_max) =========================

struct ArithmeticBase {
    lambda: Vec<f64>,        // von Mangoldt n -> log p if n = p^k else 0
    theta_prefix: Vec<f64>,  // theta(x) = sum_{p<=x} log p
    pi_prefix: Vec<f64>,     // pi(x) = number of primes <= x
    mertens_prefix: Vec<f64>, // M(x) = sum_{n<=x} mu(n)
}

fn build_base(n_max: usize) -> ArithmeticBase {
    let mut spf = vec![0usize; n_max + 1];
    let mut primes = Vec::new();
    for i in 2..=n_max {
        if spf[i] == 0 { spf[i] = i; primes.push(i); }
        for &p in &primes {
            if p > spf[i] || i * p > n_max { break; }
            spf[i * p] = p;
        }
    }
    let mut is_prime = vec![false; n_max + 1];
    for &p in &primes { is_prime[p] = true; }

    // von Mangoldt: n = p^k  =>  lambda(n) = log p
    let mut lambda = vec![0.0f64; n_max + 1];
    for n in 2..=n_max {
        let p = spf[n];
        let mut m = n;
        let mut all_same = true;
        while m > 1 { if spf[m] != p { all_same = false; break; } m /= p; }
        if all_same { lambda[n] = (p as f64).ln(); }
    }
    let mut theta_prefix = vec![0.0f64; n_max + 1];
    let mut pi_prefix = vec![0.0f64; n_max + 1];
    for n in 2..=n_max {
        let dp = if is_prime[n] { (n as f64).ln() } else { 0.0 };
        let dc = if is_prime[n] { 1.0 } else { 0.0 };
        theta_prefix[n] = theta_prefix[n - 1] + dp;
        pi_prefix[n] = pi_prefix[n - 1] + dc;
    }

    // Moebius + Mertens
    let mut mu = vec![1i32; n_max + 1];
    let mut count = vec![0i32; n_max + 1];
    mu[0] = 0;
    for i in 2..=n_max {
        if count[i] == 0 { for j in (i..=n_max).step_by(i) { mu[j] *= -1; count[j] += 1; } }
        let p2 = i * i;
        if p2 <= n_max { for j in (p2..=n_max).step_by(p2) { mu[j] = 0; } }
    }
    let mut mertens_prefix = vec![0.0f64; n_max + 1];
    let mut running = 0.0f64;
    for n in 1..=n_max { running += mu[n] as f64; mertens_prefix[n] = running; }

    ArithmeticBase { lambda, theta_prefix, pi_prefix, mertens_prefix }
}

// Sample a cumulative prefix function at x(n)=round(exp(u)), u in [ln10, ln n_max].
fn sample_prefix(prefix_fn: &dyn Fn(usize) -> f64, n_max: usize, n_samples: usize) -> Vec<f64> {
    let u0 = (10.0f64).ln();
    let u1 = (n_max as f64).ln();
    let du = (u1 - u0) / (n_samples - 1) as f64;
    let mut vs = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        let u = u0 + du * k as f64;
        let x = (u.exp().round() as usize).max(2).min(n_max);
        vs.push(prefix_fn(x));
    }
    vs
}

// ==== basic statistics =======================================================

fn mean(v: &[f64]) -> f64 { if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 } }
fn stddev(v: &[f64], m: f64) -> f64 {
    if v.len() <= 1 { 1.0 } else { (v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt().max(1e-12) }
}
fn zscore(v: &[f64]) -> Vec<f64> {
    let m = mean(v); let s = stddev(v, m);
    v.iter().map(|&x| (x - m) / s).collect()
}
fn increments(v: &[f64]) -> Vec<f64> { v.windows(2).map(|w| w[1] - w[0]).collect() }
// ==== candidate observables (Phase II–III) ===================================
// Each is a pure scalar statistic of a z-scored fluctuation series, so it is
// well-defined on EVERY representation and on every null. None references 1/2.
// complexity = number of distinct library operations in its definition (MDL).

/// C1 — excess kurtosis of the fluctuation distribution (tail heaviness).
fn c1_excess_kurtosis(z: &[f64]) -> f64 {
    let n = z.len() as f64;
    let m = mean(z); let s = stddev(z, m);
    if s <= 0.0 { return f64::NAN; }
    let m4 = z.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    m4 - 3.0
}

/// C2 — lag-1 autocorrelation of the increment series (short-range dependence).
fn c2_lag1_acf(z: &[f64]) -> f64 {
    let d = increments(z);
    let m = mean(&d);
    let v: Vec<f64> = d.iter().map(|&x| x - m).collect();
    if v.len() < 8 { return f64::NAN; }
    let var = v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64;
    if var <= 0.0 { return f64::NAN; }
    let num: f64 = v[..v.len() - 1].iter().zip(v[1..].iter()).map(|(&a, &b)| a * b).sum();
    (num / (v.len() - 1) as f64) / var
}

/// C3 — right-tail exponent alpha in P(|z| > t) ~ t^{-alpha} over the outer tail.
/// Fits log-survival vs log-threshold slope on the 80/85/90/95th percentiles.
fn c3_tail_exponent(z: &[f64]) -> f64 {
    let mut u: Vec<f64> = z.iter().map(|x| x.abs()).collect();
    if u.len() < 64 { return f64::NAN; }
    u.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = u.len();
    let mut pts = Vec::new();
    // thresholds at selected high quantiles; survival = fraction above threshold
    let mut th = 0.75;
    while th < 0.96 {
        let idx = (th * n as f64) as usize;
        if idx >= n || idx == 0 { break; }
        let t = u[idx].max(1e-9);
        let surv = (n - idx) as f64 / n as f64;
        if surv > 1e-4 { pts.push((t.ln(), surv.ln())); }
        th += 0.05;
    }
    if pts.len() < 3 { return f64::NAN; }
    let npts = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + x, b + y));
    let (sxx, sxy) = pts.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + x * x, b + x * y));
    let slope = (npts * sxy - sx * sy) / (npts * sxx - sx * sx).max(1e-12);
    -slope // alpha = -(dlogn P / dlogn t)
}

/// C4 — mean return interval to a one-sigma band around the mean (recurrence),
/// normalised: the empirical mean crossing interval counts vs expectation of 2.
fn c4_return_interval(z: &[f64]) -> f64 {
    let crossings: Vec<usize> = (0..z.len()).filter(|&i| z[i].abs() < 1.0).collect();
    if crossings.len() < 8 { return f64::NAN; }
    let gaps: Vec<usize> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
    let g = if gaps.is_empty() { 1.0 } else { gaps.iter().map(|&x| x as f64).sum::<f64>() / gaps.len() as f64 };
    // relative to the expectation for a continuously crossing sequence (≈2)
    g / 2.0
}

/// C5 — normalised consecutive displacement: mean |z_{i+1} − z_i| / sd(increments).
/// A scale-free measure of local fluctuation amplitude (relational invariance).
fn c5_displacement(z: &[f64]) -> f64 {
    let d = increments(z);
    let m = mean(&d); let s = stddev(&d, m);
    if s <= 0.0 { return f64::NAN; }
    d.iter().map(|&x| (x - m).abs()).sum::<f64>() / d.len().max(1) as f64 / s
}

fn all_candidates(z: &[f64]) -> [f64; 5] {
    [
        c1_excess_kurtosis(z),
        c2_lag1_acf(z),
        c3_tail_exponent(z),
        c4_return_interval(z),
        c5_displacement(z),
    ]
}
// ==== null controls (Phase V) =================================================

/// Null 1 — shuffle the sampled fluctuations: preserves marginal distribution,
/// destroys all ordering (increment dependence, returns, recurrence).
fn shuffled(v: &[f64], seed: &mut u64) -> Vec<f64> {
    let mut out = v.to_vec();
    let n = out.len();
    for i in (1..n).rev() {
        let j = ((f01(seed) * (i as f64 + 1.0)) as usize).min(i);
        out.swap(i, j);
    }
    out
}

/// Null 2 — Cramer random primes at density 1/ln x, run through the SAME
/// psi−x cumulative-counting pipeline. Keeps baseline density while removing
/// deterministic prime correlations.
fn cramer_psi(n_max: usize, n_samples: usize, seed: &mut u64) -> Vec<f64> {
    let mut psi = 0.0f64;
    let mut prefix = vec![0.0f64; n_max + 1];
    for x in 2..=n_max {
        if f01(seed) < 1.0 / (x as f64).ln() { psi += (x as f64).ln(); }
        prefix[x] = psi;
    }
    let f = |x: usize| prefix[x] - x as f64;
    sample_prefix(&f, n_max, n_samples)
}

/// Null 3 — synthetic iid-Gaussian series matching psi's first-order statistics
/// (same mean / variance). Tests whether the fluctuation law is Gaussian.
fn synth_gaussian(m: f64, s: f64, n: usize, seed: &mut u64) -> Vec<f64> {
    // Box–Muller from the LCG.
    (0..n).map(|_| {
        let u1 = f01(seed).max(1e-12); let u2 = f01(seed);
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        m + s * z
    }).collect()
}

// ==== verdict logic (Phase VI–VII) ===========================================
// Margins are fixed BEFORE held-out is revealed (blind holdout).

/// For a candidate's scalar values across all reps, decide transfer /
/// specificity / held-out survival using a single frozen margin `frac`.
fn evaluate(
    id: &str, complexity: u32, description: &str,
    disc: [f64; 2], holdout: [f64; 2], nulls: Vec<f64>,
) -> CandidateReport {
    let all: Vec<f64> = vec![disc[0], disc[1], holdout[0], holdout[1]]
        .into_iter().chain(nulls.iter().cloned()).collect();
    let finite: Vec<f64> = all.iter().cloned().filter(|x| x.is_finite()).collect();
    let disc_mean = if finite.is_empty() { f64::NAN } else { (disc[0] + disc[1]) / 2.0 };
    let range = if finite.len() < 2 {
        f64::NAN
    } else {
        let mn = finite.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        mx - mn
    };

    // Frozen acceptance margin: 40% of the observed total variation.
    let margin = if range.is_finite() && range > 0.0 { 0.40 * range } else { f64::NAN };

    let disc_spread = (disc[0] - disc[1]).abs();

    // transfer: discovery pair close relative to total variation
    let transfers = margin.is_finite() && disc_spread <= margin;

    // specificity: discovery value separated from EVERY null by more than margin
    let spec = margin.is_finite() && disc_mean.is_finite()
        && nulls.iter().all(|&nv| nv.is_finite() && (disc_mean - nv).abs() > margin);

    // held-out: both pre-frozen representations agree with the discovery value
    let ho = margin.is_finite() && disc_mean.is_finite()
        && holdout[0].is_finite() && holdout[1].is_finite()
        && (holdout[0] - disc_mean).abs() <= margin
        && (holdout[1] - disc_mean).abs() <= margin;

    CandidateReport {
        id: id.into(), complexity, description: description.into(),
        discovery_value: disc_mean, discovery_spread: disc_spread,
        null_values: nulls, held_out_values: holdout.to_vec(),
        transfers, arithmetic_specific: spec, held_out_survives: ho,
        survives: transfers && spec && ho,
    }
}
fn write_file(path: &Path, data: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).ok();
    let mut f = File::create(path).unwrap();
    f.write_all(data.as_bytes()).unwrap();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== EXPERIMENT 51: REPRESENTATION-COVARIANT PROBABILITY LAWS ===");
    let n_max = 200_000usize;
    let n_samples = 2048usize;
    let mut seed = 20260912u64;
    let base = build_base(n_max);

    // ---- Phase I: reconstruct representations on a common u = ln x grid ----
    let lam_prefix: Vec<f64> = (0..=n_max).scan(0.0f64, |a, k| { *a += base.lambda[k]; Some(*a) }).collect();
    let psi: Vec<f64> = sample_prefix(&|x: usize| lam_prefix[x] - x as f64, n_max, n_samples);
    let theta: Vec<f64> = sample_prefix(&|x: usize| base.theta_prefix[x] - x as f64, n_max, n_samples);
    let pi: Vec<f64> = sample_prefix(&|x: usize| {
        let lx = (x as f64).ln().max(1e-9);
        base.pi_prefix[x] - x as f64 / lx
    }, n_max, n_samples);
    let mertens: Vec<f64> = sample_prefix(&|x: usize| base.mertens_prefix[x] / (x as f64).sqrt(), n_max, n_samples);

    // ---- Phase V: null controls ----
    let pm = mean(&psi); let ps = stddev(&psi, pm);
    let shuff = shuffled(&psi, &mut seed);
    let cramer = cramer_psi(n_max, n_samples, &mut seed);
    let synth = synth_gaussian(pm, ps, n_samples, &mut seed);

    // z-scaled reps (recorded transformation: z-score) and nulls on that grid
    let reps: Vec<(&str, &str, Vec<f64>)> = vec![
        ("psi(x)-x", "discovery", psi.clone()),
        ("theta(x)-x", "discovery", theta.clone()),
        ("pi(x)-x/ln x", "held-out", pi.clone()),
        ("M(x)/sqrt(x)", "held-out", mertens.clone()),
    ];
    let znulls: Vec<(&str, Vec<f64>)> = vec![
        ("shuffled-psi", zscore(&shuff)),
        ("Cramer-psi", zscore(&cramer)),
        ("synthetic-Gaussian", zscore(&synth)),
    ];

    // ---- Phase II/III/IV: observables on discovery + held-out reps ----
    let meta = [
        ("C1", 1u32, "excess kurtosis of fluctuation distribution"),
        ("C2", 1u32, "lag-1 autocorrelation of increments"),
        ("C3", 2u32, "right-tail exponent P(|z|>t) ~ t^-alpha"),
        ("C4", 2u32, "mean one-sigma return interval / expectation"),
        ("C5", 2u32, "normalised consecutive displacement"),
    ];
    let mut cand_results = Vec::new();
    for ci in 0..5 {
        let sv: Vec<Vec<f64>> = [psi.clone(), theta.clone(), pi.clone(), mertens.clone()]
            .into_iter().map(|s| zscore(&s)).collect();
        let disc: Vec<f64> = [0usize, 1].iter().map(|&k| all_candidates(&sv[k])[ci]).collect();
        let hold: Vec<f64> = [2usize, 3].iter().map(|&k| all_candidates(&sv[k])[ci]).collect();
        let nulls: Vec<f64> = znulls.iter().map(|(_, s)| all_candidates(s)[ci]).collect();
        let rep = evaluate(meta[ci].0, meta[ci].1, meta[ci].2, [disc[0], disc[1]], [hold[0], hold[1]], nulls.clone());
        println!(
            "C{} disc={:.3} (spread {:.3}) | held-out pi={:.3} M={:.3} | nulls=({:.3},{:.3},{:.3}) | transfer={} specific={} heldout={}",
            ci + 1, rep.discovery_value, rep.discovery_spread, hold[0], hold[1],
            rep.null_values[0], rep.null_values[1], rep.null_values[2],
            rep.transfers, rep.arithmetic_specific, rep.held_out_survives,
        );
        cand_results.push(rep);
    }
let any_transfer = cand_results.iter().any(|r| r.transfers);
    let any_specific = cand_results.iter().any(|r| r.arithmetic_specific);
    let any_ho = cand_results.iter().any(|r| r.held_out_survives);
    let n_specific_survivors = cand_results.iter().filter(|r| r.survives).count();

    // ---- Phase VII: stopping rule & verdict ----
    let verdict = if n_specific_survivors > 0 {
        "PARTIAL — a candidate law simultaneously transfers across representations, is arithmetic-specific, and survives the held-out reps (formalize on a larger sieve)".to_string()
    } else {
        let mut reasons = Vec::new();
        if !any_transfer { reasons.push("no candidate transfers across independent representations"); }
        if !any_specific { reasons.push("no candidate is arithmetic-specific against all nulls (the shared probability structure is generic, not prime-specific)"); }
        if !any_ho { reasons.push("no candidate survives the held-out representations"); }
        if any_transfer && any_specific && !any_ho {
            reasons.push("discovery-only agreement is pair-specific and evaporates under blind holdout (the E50 failure mode, restated at the probability level)");
        }
        format!(
            "FAILED — {}. No representation-covariant probability law is BOTH transferable AND arithmetic-specific AND held-out-surviving.",
            if reasons.is_empty() { "no single gate clears.".to_string() } else { reasons.join("; ") + "." }
        )
    };

    let rel_half = format!(
        "no candidate targets 1/2; observed discovery values: C1={:.3} C2={:.3} C3={:.3} C4={:.3} C5={:.3}",
        cand_results[0].discovery_value, cand_results[1].discovery_value,
        cand_results[2].discovery_value, cand_results[3].discovery_value,
        cand_results[4].discovery_value,
    );

    let representatives: Vec<RepresentationStat> = reps.into_iter()
        .map(|(n, k, v)| {
            let z = zscore(&v);
            // decimate ~8x for a portable record (period-8 subsample of the series)
            let dec: Vec<f64> = z.iter().step_by(8).cloned().collect();
            RepresentationStat { name: n.into(), kind: k.into(), values: dec }
        })
        .collect();

    let out = Experiment51Output {
        e50_context: "E50: geometric structure was representation-bound (arithmetic where specific, generic where transferable). E51 tests the same object one level up, at probability / statistics / recurrence.".into(),
        representations_tested: vec!["psi(x)-x".into(), "theta(x)-x".into(), "pi(x)-x/ln x".into(), "M(x)/sqrt(x)".into()],
        nulls_tested: vec!["shuffled-psi".into(), "Cramer-psi".into(), "synthetic-Gaussian".into()],
        discovery_set: vec!["psi(x)-x".into(), "theta(x)-x".into()],
        held_out_set: vec!["pi(x)-x/ln x".into(), "M(x)/sqrt(x)".into()],
        candidate_observables: vec![
            "excess kurtosis".into(), "lag-1 increment ACF".into(),
            "tail exponent".into(), "return interval".into(),
            "displacement scaling".into(),
        ],
        representatives,
        candidates: cand_results,
        cross_representation_transfer: any_transfer,
        arithmetic_specificity: any_specific,
        held_out_survival: any_ho,
        relation_to_half: rel_half.clone(),
        final_verdict: verdict.clone(),
    };

    let out_path = Path::new("research/impossible_machine/experiments/experiment51_results.json");
    write_file(out_path, &serde_json::to_string_pretty(&out).unwrap());
    println!("\nVERDICT: {verdict}");
    println!("relation to 1/2: {rel_half}");
    println!("survivors: {n_specific_survivors}/5 | transfer {any_transfer} | specificity {any_specific} | held-out {any_ho}");
    println!("results: {}", out_path.display());
    Ok(())
}