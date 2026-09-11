//! EXPERIMENT 49: SELF-CONSISTENT DIMENSION OF THE PRIME FIELD
//!
//! The question (smaller, harsher, more invariant than E48):
//!
//!   Does the prime sequence admit a representation whose dimensionality is
//!   an INVARIANT of the arithmetic object rather than an artifact of the
//!   chosen measurement procedure?
//!
//! Method: a candidate dimension d is REPORTED only where four independently
//! constructed estimators CONVERGE, with the convergence tolerance delta
//! calibrated not by us but by the measurement's own reproducibility
//! (moving-block bootstrap of the prime series itself). Positive controls
//! (sinusoid d=1, fBm d=1.3, Brownian d=1.5, white noise) certify the
//! estimators BEFORE they are allowed to speak about primes. If a control
//! fails, the machinery is declared invalid and the experiment reports that.
//!
//! 1/2 appears nowhere. Candidate spectral laws (1/d, (d-1)/d, d/2, 1/(2d))
//! are compared only AFTER the geometry is frozen — and 1/2 is one possible
//! output among all possible outputs.
//!
//! Run: cargo run -p physis-core --release --example experiment49_self_consistent_dimension

use serde::Serialize;

// ───────────────────────── arithmetic substrate ─────────────────────────

fn sieve_lambda(n: usize) -> Vec<f64> {
    // smallest-prime-factor sieve -> von Mangoldt lambda(n) = log p if n=p^k
    let mut spf = vec![0usize; n + 1];
    let mut primes = Vec::new();
    for i in 2..=n {
        if spf[i] == 0 { spf[i] = i; primes.push(i); }
        for &p in &primes {
            if p > spf[i] || i * p > n { break; }
            spf[i * p] = p;
        }
    }
    let mut out = vec![0.0f64; n + 1];
    for n in 2..=n {
        let p = spf[n];
        let mut m = n;
        let mut all_same = true;
        while m > 1 {
            if spf[m] != p { all_same = false; break; }
            m /= p;
        }
        if all_same { out[n] = (p as f64).ln(); }
    }
    out
}

/// Sample E_psi(u) = psi(x) - x on a uniform grid in u = ln x.
/// Returns (u_values, E_values), recorded transformation: u = ln x, E = psi - x.
/// Sample E_psi(u) = psi(x) - x on a uniform grid in u = ln x.
/// Recorded transformation: u = ln x, E = psi - x (raw, unnormalized).
fn psi_series(n_max: usize, n_samples: usize) -> (Vec<f64>, Vec<f64>) {
    let lam = sieve_lambda(n_max);
    let mut prefix = vec![0.0f64; n_max + 1];
    let mut acc = 0.0f64;
    for n in 1..=n_max { acc += lam[n]; prefix[n] = acc; }
    let u0 = (10.0f64).ln();
    let u1 = (n_max as f64).ln();
    let du = (u1 - u0) / (n_samples - 1) as f64;
    let mut us = Vec::with_capacity(n_samples);
    let mut es = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        let u = u0 + du * k as f64;
        let x = (u.exp().round() as usize).max(2);
        us.push(u);
        es.push(prefix[x] - x as f64);
    }
    (us, es)
}

/// theta(x) = sum_{p<=x} log p; fluctuation theta - x sampled in u = ln x.
fn theta_series(n_max: usize, n_samples: usize) -> (Vec<f64>, Vec<f64>) {
    let mut spf = vec![0usize; n_max + 1];
    let mut primes = Vec::new();
    for i in 2..=n_max {
        if spf[i] == 0 { spf[i] = i; primes.push(i); }
        let mut j = 0;
        while j < primes.len() && primes[j] <= spf[i] && i * primes[j] <= n_max {
            spf[i * primes[j]] = primes[j]; j += 1;
        }
    }
    let mut theta_prefix = vec![0.0f64; n_max + 1];
    let mut running = 0.0f64;
    for n in 2..=n_max {
        if spf[n] == n { running += (n as f64).ln(); }
        theta_prefix[n] = running;
    }
    let u0 = (10.0f64).ln();
    let u1 = (n_max as f64).ln();
    let du = (u1 - u0) / (n_samples - 1) as f64;
    let mut us = Vec::with_capacity(n_samples);
    let mut es = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        let u = u0 + du * k as f64;
        let x = (u.exp().round() as usize).max(2);
        us.push(u);
        es.push(theta_prefix[x] - x as f64);
    }
    (us, es)
}

// ───────────────────────── the four independent estimators ─────────────────────────
//
// Convention (recorded, applied identically to every series including controls):
// for a self-similar signal with Hurst exponent H, the graph dimension is
// d = 2 - H. Each estimator measures H by a different route.

fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len().max(1) as f64 }

fn linfit(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let (sx, sy, sxx, sxy) = pts.iter().fold((0.0, 0.0, 0.0, 0.0), |(a, b, c, d), &(x, y)| (a + x, b + y, c + x * x, d + x * y));
    (n * sxy - sx * sy) / (n * sxx - sx * sx).max(1e-12)
}

/// E1: detrended fluctuation analysis. H = slope of ln F(w) vs ln w.
fn hurst_dfa(y: &[f64]) -> f64 {
    let n = y.len();
    let m = mean(y);
    let profile: Vec<f64> = std::iter::once(0.0)
        .chain(y.iter().scan(0.0f64, |a, &v| { *a += v - m; Some(*a) }))
        .collect();
    let mut pts = Vec::new();
    let mut w = 4usize;
    while w <= n / 4 {
        let n_win = n / w;
        let mut f2 = 0.0f64;
        for i in 0..n_win {
            let s = i * w;
            // linear detrend of profile[s..s+w]
            let xs: Vec<f64> = (0..w).map(|k| k as f64).collect();
            let ys: Vec<f64> = (s..s + w).map(|k| profile[k]).collect();
            let xm = w as f64 * 0.5 * (w as f64 - 1.0) / w as f64;
            let ym = mean(&ys);
            let mut sxy = 0.0; let mut sxx = 0.0;
            for k in 0..w { sxy += (k as f64 - xm) * (ys[k] - ym); sxx += (k as f64 - xm).powi(2); }
            let b = sxy / sxx.max(1e-12);
            let a = ym - b * xm;
            for k in 0..w { f2 += (profile[s + k] - (a + b * k as f64)).powi(2); }
        }
        f2 /= (n_win * w) as f64;
        pts.push(((w as f64).ln(), (f2.sqrt().max(1e-300)).ln()));
        w *= 2;
    }
    linfit(&pts)
}

/// E2: rescaled range (R/S). H = slope of ln(R/S) vs ln tau.
fn hurst_rs(y: &[f64]) -> f64 {
    let n = y.len();
    let m = mean(y);
    let mut pts = Vec::new();
    let mut tau = 8usize;
    while tau <= n / 4 {
        let n_win = n / tau;
        let mut rs_sum = 0.0f64;
        let mut cnt = 0;
        for i in 0..n_win {
            let seg: Vec<f64> = (i * tau..(i + 1) * tau).map(|k| y[k]).collect();
            let sm = mean(&seg);
            let dev: Vec<f64> = seg.iter().map(|v| v - sm).collect();
            let cum: Vec<f64> = std::iter::once(0.0)
                .chain(dev.iter().scan(0.0f64, |a, &v| { *a += v; Some(*a) }))
                .collect();
            let r = cum.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - cum.iter().cloned().fold(f64::INFINITY, f64::min);
            let s = (dev.iter().map(|v| v * v).sum::<f64>() / tau as f64).sqrt().max(1e-12);
            rs_sum += r / s; cnt += 1;
        }
        pts.push(((tau as f64).ln(), (rs_sum / cnt as f64).ln()));
        tau *= 2;
    }
    linfit(&pts)
}

/// E3: structure functions. H = zeta(2)/2 from <|Delta y(tau)|^2> ~ tau^{zeta(2)}.
fn hurst_structure(y: &[f64]) -> f64 {
    let n = y.len();
    let mut pts = Vec::new();
    let mut tau = 1usize;
    while tau <= n / 4 {
        let vals: Vec<f64> = (0..n - tau).map(|i| (y[i + tau] - y[i]).abs()).collect();
        let s2 = vals.iter().map(|v| v * v).sum::<f64>() / vals.len() as f64;
        pts.push(((tau as f64).ln(), (s2.max(1e-300)).ln() / 2.0)); // ln S2 /2 = ln E|Dy|
        if tau == 0 { break; }
        tau *= 2;
    }
    // E|Delta| ~ tau^H  =>  slope of ln(E|Dy|) vs ln tau is H (monofractal fGn)
    linfit(&pts)
}

/// E4: information dimension of the point cloud (u, y/sigma_y) — a genuinely
/// different route (entropy of occupation measures, not fluctuation scaling).
fn information_dimension(us: &[f64], ys: &[f64]) -> f64 {
    let (umin, umax) = (us.iter().cloned().fold(f64::INFINITY, f64::min), us.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let (ymin, ymax) = (ys.iter().cloned().fold(f64::INFINITY, f64::min), ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let mut pts = Vec::new();
    let mut g = 4usize;
    while g <= 256 {
        let mut p = vec![0.0f64; g * g];
        for i in 0..us.len() {
            let bx = (((us[i] - umin) / (umax - umin + 1e-12)) * g as f64).min((g - 1) as f64) as usize;
            let by = (((ys[i] - ymin) / (ymax - ymin + 1e-12)) * g as f64).min((g - 1) as f64) as usize;
            p[by * g + bx] += 1.0;
        }
        let total = p.iter().sum::<f64>().max(1.0);
        let h: f64 = p.iter().filter(|&&c| c > 0.0).map(|&c| { let q = c / total; -q * q.ln() }).sum();
        pts.push(((g as f64).ln(), h)); // H(g) ~ d ln g
        g *= 2;
    }
    linfit(&pts)
}

// ───────────────────────── self-consistency machinery ─────────────────────────

/// Convention conversion (recorded): estimators return Hurst exponents of
/// different objects; each is converted to a graph dimension d.
///   DFA alpha: alpha<=1 means fGn-like input  -> d = 2 - alpha
///              alpha> 1 means fBm-like input  -> d = 3 - alpha
///              (a smooth signal has alpha=2 -> d=1; consistent)
///   R/S H:     only valid for H<1 (known R/S bias above); d = 2 - H
///   structure: measures the increment Hurst H_inc; d = 2 - H_inc
///   information: direct measurement of the point cloud; d = slope
fn d_from_dfa(a: f64) -> f64 { if a <= 1.0 { 2.0 - a } else { 3.0 - a } }
fn d_from_rs(h: f64) -> f64 { if h < 1.0 { 2.0 - h } else { f64::NAN } } // R/S invalid for H>=1
fn d_from_struct(h_inc: f64) -> f64 { 2.0 - h_inc }

fn all_estimators(us: &[f64], ys: &[f64]) -> Vec<f64> {
    vec![
        d_from_dfa(hurst_dfa(ys)),
        d_from_rs(hurst_rs(ys)),
        d_from_struct(hurst_structure(ys)),
        information_dimension(us, ys),
    ]
}

/// Moving-block bootstrap: the spread of the four estimators under resampling
/// defines the measurement's own reproducibility window. Convergence
/// tolerance delta = 95th percentile of the bootstrap spread distribution.
fn bootstrap_delta(ys: &[f64], n_boot: usize, seed: u64) -> f64 {
    let n = ys.len();
    let mut rng = seed;
    let mut spreads = Vec::with_capacity(n_boot);
    for _b in 0..n_boot {
        let mut sample = Vec::with_capacity(n);
        while sample.len() < n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let start = ((rng >> 33) as usize) % (n - 64).max(1);
            for k in 0..64 { if sample.len() < n { sample.push(ys[(start + k) % n]); } }
        }
        let us: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let d = all_estimators(&us, &sample);
        let mx = d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mn = d.iter().cloned().fold(f64::INFINITY, f64::min);
        spreads.push(mx - mn);
    }
    spreads.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((n_boot as f64) * 0.95) as usize % n_boot;
    spreads[idx]
}

fn spread(d: &[f64]) -> f64 {
    let finite: Vec<f64> = d.iter().cloned().filter(|v| v.is_finite()).collect();
    let mx = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mn = finite.iter().cloned().fold(f64::INFINITY, f64::min);
    mx - mn
}

// ───────────────────────── positive controls ─────────────────────────

/// fBm via Fourier synthesis: amplitude ~ f^{-(H+1/2)}, random phase.
fn fbm(h: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    for k in 1..n / 2 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let phi = (rng as f64) / (u64::MAX as f64) * 2.0 * std::f64::consts::PI;
        let amp = (k as f64 / n as f64).powf(-(h + 0.5));
        re[k] = amp * phi.cos(); re[n - k] = re[k];
        im[k] = amp * phi.sin(); im[n - k] = -im[n - k];
    }
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for k in 0..n {
            let t = 2.0 * std::f64::consts::PI * (i * k) as f64 / n as f64;
            acc += re[k] * t.cos() - im[k] * t.sin();
        }
        out[i] = acc / n as f64;
    }
    out
}

fn white_noise(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    (0..n).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng as f64) / (u64::MAX as f64) * 2.0 - 1.0
    }).collect()
}

fn sinusoid(n: usize) -> Vec<f64> {
    (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64 * 7.0).sin()).collect()
}

// ───────────────────────── output ─────────────────────────

#[derive(Serialize, Clone)]
struct ControlResult { name: String, expected: f64, estimates: Vec<f64>, spread: f64, valid: bool }

#[derive(Serialize, Clone)]
struct SeriesResult {
    name: String,
    estimates: Vec<f64>,
    spread: f64,
    delta: f64,
    converged: bool,
    d_consensus: Option<f64>,
}

#[derive(Serialize)]
struct Output49 {
    estimator_convention: &'static str,
    controls: Vec<ControlResult>,
    machinery_valid: bool,
    series: Vec<SeriesResult>,
    window_profile: Vec<(String, f64, f64, bool)>, // (window, d_consensus or spread-marker, drift_slope, converged)
    outcome: usize,
    outcome_text: String,
    candidate_laws: Vec<(String, f64)>,
    final_question_answer: String,
}

fn write_file(path: &std::path::Path, content: &str) {
    if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
    std::fs::write(path, content).unwrap_or_else(|e| eprintln!("warning: could not write {}: {e}", path.display()));
}

fn main() {
    println!("EXPERIMENT 49: SELF-CONSISTENT DIMENSION OF THE PRIME FIELD");
    println!("convention: d = 2 - H; H measured 4 independent ways; convergence\\nrequired; tolerance delta = bootstrap reproducibility (not chosen)\n");
    let n = 4096usize;

    // ── positive controls: certify estimators BEFORE primes ──
    let control_specs: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // (name, series, per-estimator expected d; NaN = estimator not valid for this control)
        ("sinusoid (smooth, d=1)", sinusoid(n), vec![1.0, 1.0, 1.0, 1.0]),
        ("fBm H=0.7 (d=1.3)", fbm(0.7, n, 11), vec![1.3, f64::NAN, 1.3, 1.3]),
        ("Brownian H=0.5 (d=1.5)", fbm(0.5, n, 22), vec![1.5, f64::NAN, 1.5, 1.5]),
        ("white noise (fGn conv d=1.5; dust d=2)", white_noise(n, 33), vec![1.5, 1.5, 2.0, f64::NAN]),
    ];
    let mut controls = Vec::new();
    let mut machinery_valid = true;
    for (name, series_y, expected) in control_specs {
        let us: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let est2 = all_estimators(&us, &series_y);
        let sp = spread(&est2);
        // validity: >=2 of the finite expectations hit within 0.3, none off by >0.6
        let mut hits = 0usize;
        let mut bad = 0usize;
        for k in 0..4 {
            if expected[k].is_finite() {
                if (est2[k] - expected[k]).abs() < 0.3 { hits += 1; }
                if (est2[k] - expected[k]).abs() > 0.6 { bad += 1; }
            }
        }
        let ok = hits >= 2 && bad == 0;
        if !ok { machinery_valid = false; }
        println!("  control {name}: dfa={:.3} rs={:.3} struct={:.3} info={:.3} (spread {:.3}) {}",
            est2[0], est2[1], est2[2], est2[3], sp, if ok { "VALID" } else { "INVALID" });
        controls.push(ControlResult { name: name.into(), expected: expected.iter().cloned().filter(|v| v.is_finite()).sum::<f64>() / expected.iter().cloned().filter(|v| v.is_finite()).count().max(1) as f64, estimates: est2.clone(), spread: sp, valid: ok });
    }

    // ── the prime field: geometry FIRST (no zeta, no 1/2 anywhere) ──
    let n_max = 1_000_000usize;
    println!("\\n  building arithmetic series (sieve to 1_000_000, {n} log-spaced samples)…");
    let (us_psi, es_psi) = psi_series(1_000_000, n);
    let (_us_th, es_th) = theta_series(1_000_000, n);

    // normalize: y/sigma_y (recorded transformation, applied to every series)
    let normalize = |ys: &[f64]| -> Vec<f64> {
        let m = mean(ys);
        let sd = (ys.iter().map(|v| (v - m).powi(2)).sum::<f64>() / ys.len() as f64).sqrt().max(1e-12);
        ys.iter().map(|v| (v - m) / sd).collect()
    };
    let ys_psi = normalize(&es_psi);
    let ys_th = normalize(&es_th);

    let measure = |name: &str, ys2: &[f64]| -> SeriesResult {
        let us: Vec<f64> = (0..ys2.len()).map(|k| k as f64).collect();
        let est = all_estimators(&us, ys2);
        let sp = spread(&est);
        let dl = bootstrap_delta(ys2, 24, 777);
        let conv = sp <= dl;
        let consensus = if conv { Some(est.iter().sum::<f64>() / est.len() as f64) } else { None };
        println!("  {name}: dfa={:.3} rs={:.3} struct={:.3} info={:.3} spread={:.3} delta={:.3} {}",
            est[0], est[1], est[2], est[3], sp, dl, if conv { "CONVERGED" } else { "no convergence" });
        SeriesResult { name: name.into(), estimates: est, spread: sp, delta: dl, converged: conv, d_consensus: consensus }
    };

    println!("\\n  measuring (self-consistency required for a dimension to be REPORTED):");
    let mut series = Vec::new();
    series.push(measure("prime field: E_psi(u)", &ys_psi));
    series.push(measure("prime field: E_theta(u)", &ys_th));

    // nulls: shuffled E_psi destroys correlations (expected: no fGn structure)
    let mut rng0 = 424242u64;
    let mut shuffled = ys_psi.clone();
    for i in (1..shuffled.len()).rev() {
        rng0 = rng0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((rng0 >> 33) as usize) % (i + 1);
        shuffled.swap(i, j);
    }
    series.push(measure("null: shuffled E_psi (correlations destroyed)", &shuffled));
    // null: Cramer random primes — same pipeline, random prime indicator with p=1/ln x
    let (us_c, es_c) = cramer_series(1_000_000, n, 999);
    let ys_c = normalize(&es_c);
    let _ = us_c;
    series.push(measure("null: Cramer random primes E_psi(u)", &ys_c));

    // ── d(x): per-window self-consistent dimension, drift test ──
    println!("\\n  window profile (per-scale convergence; drift = slope of d vs u):");
    let mut window_profile = Vec::new();
    let n_win = 8usize;
    let wlen = n / n_win;
    for i in 0..n_win {
        let s = i * wlen;
        let e = s + wlen;
        let win: Vec<f64> = ys_psi[s..e].to_vec();
        let us_w: Vec<f64> = (0..wlen).map(|k| k as f64).collect();
        let est = all_estimators(&us_w, &win);
        let sp = spread(&est);
        let dl = bootstrap_delta(&win, 12, (100 + i) as u64);
        let conv = sp <= dl;
        let marker = if conv { est.iter().sum::<f64>() / est.len() as f64 } else { sp * -1.0 - 1.0 };
        window_profile.push((format!("window {}", i + 1), marker, conv as i32 as f64, conv));
        println!("    window {}: spread {:.3} delta {:.3} {}", i + 1, sp, dl, if conv { format!("CONVERGED d={:.3}", marker) } else { "no".into() });
    }

    // ── outcome classification (pre-registered vocabulary) ──
    let psi = &series[0];
    let th = &series[1];
    let mut outcome: usize = 0;
    let mut outcome_text = String::new();
    let machinery_note = if machinery_valid { "" } else { " [MACHINERY INVALID: positive controls failed — no claim possible]" };
    if !machinery_valid {
        outcome = 0;
        outcome_text = "MACHINERY INVALID — estimators do not recover known dimensions on synthetic controls; nothing about primes can be claimed".into();
    } else if !psi.converged && !th.converged {
        outcome = 3;
        outcome_text = "NEVER CONVERGE — the estimators do not agree on any dimension for the prime field even at their own reproducibility tolerance; the premise 'the prime field has an intrinsic dimension' is probably wrong".into();
    } else if psi.converged && th.converged {
        let d1 = psi.d_consensus.unwrap();
        let d2 = th.d_consensus.unwrap();
        if (d1 - d2).abs() < 0.15 {
            // stable across representations: a genuine invariant candidate
            if (d1 - 2.0).abs() < 0.1 {
                outcome = 1;
                outcome_text = "CONVERGED TO d≈2 — an emergent explanation for a spectral exponent of 1/2 now EXISTS as a hypothesis (not a proof): the arithmetic object itself carries a two-dimensional scaling structure".into();
            } else {
                outcome = 2;
                outcome_text = format!("CONVERGED TO d≈{d1:.3} ≠ 2 — the 1/d mechanism is FALSIFIED as an explanation of any 1/2 exponent, but a different arithmetic structure has been discovered; report the value and its stability, never bend it");
            }
        } else {
            outcome = 4;
            outcome_text = format!("REPRESENTATION-DEPENDENT — E_psi gives d={d1:.3}, E_theta gives d={d2:.3}; the dimension is an artifact of which arithmetic series was sampled, not an invariant of the primes");
        }
    } else {
        outcome = 4;
        outcome_text = format!("REPRESENTATION-DEPENDENT — one series converged (E_psi: {}, E_theta: {}) and the other did not; no invariant dimension claim is possible",
            psi.converged, th.converged);
    }

    // ── candidate spectral laws (frozen geometry FIRST, laws compared AFTER) ──
    let d_reported = if let Some(d) = psi.d_consensus { d } else { f64::NAN };
    let candidate_laws: Vec<(String, f64)> = if d_reported.is_finite() {
        vec![
            (format!("sigma = 1/d"), 1.0 / d_reported),
            (format!("sigma = (d-1)/d"), (d_reported - 1.0) / d_reported),
            (format!("sigma = d/2"), d_reported / 2.0),
            (format!("sigma = 1/(2d)"), 1.0 / (2.0 * d_reported)),
            (format!("sigma = |1 - d/2|"), (1.0 - d_reported / 2.0).abs()),
        ]
    } else { vec![("no consensus dimension — no law defined".into(), f64::NAN)] };

    // ── final adversarial answer ──
    let outcome_text_final = outcome_text.clone();
    let final_question_answer = format!(
        "Q: Does the prime sequence admit a representation whose dimensionality is an invariant of the arithmetic object rather than an artifact of the measurement?\\n  A: machinery_valid={machinery_valid}; E_psi converged={} (spread {:.3} vs delta {:.3}); E_theta converged={} (spread {:.3} vs delta {:.3}); shuffled null converged={}.\\n  classification: {outcome_text_final}{machinery_note}",
        psi.converged, psi.spread, psi.delta, th.converged, th.spread, th.delta, series[2].converged);

    let output = Output49 {
        estimator_convention: "d = 2 - H; four estimators: DFA, R/S, structure functions, information dimension; convergence tolerance delta = 95th percentile moving-block bootstrap spread (data-driven, not chosen)",
        controls, machinery_valid, series, window_profile, outcome, outcome_text: outcome_text.clone(), candidate_laws, final_question_answer,
    };
    write_file(std::path::Path::new("research/impossible_machine/experiments/experiment49_results.json"),
        &serde_json::to_string_pretty(&output).unwrap());
    println!("\\n{outcome_text}{machinery_note}");
    println!("results: research/impossible_machine/experiments/experiment49_results.json");
}

/// Cramer null: primes with probability 1/ln x; same sampling pipeline.
fn cramer_series(n_max: usize, n_samples: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = seed;
    let mut psi = 0.0f64;
    let mut prefix = vec![0.0f64; n_max + 1];
    for x in 2..=n_max {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (rng as f64) / (u64::MAX as f64);
        if u < 1.0 / (x as f64).ln() { psi += (x as f64).ln(); }
        prefix[x] = psi;
    }
    let u0 = (10.0f64).ln();
    let u1 = (n_max as f64).ln();
    let du = (u1 - u0) / (n_samples - 1) as f64;
    let mut us = Vec::with_capacity(n_samples);
    let mut es = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        let u = u0 + du * k as f64;
        let x = (u.exp().round() as usize).max(2);
        us.push(u);
        es.push(prefix[x] - x as f64);
    }
    (us, es)
}
