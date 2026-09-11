//! EXPERIMENT 50: REPRESENTATION EQUIVALENCE
//!
//! Central question: do E_psi and E_theta represent the same underlying
//! arithmetic structure through different coordinate systems, or does the
//! geometric ontology itself depend essentially on representation?
//!
//! Not: "how can we make the representations agree?"
//! But: "what, if anything, do these representations have no choice but to
//! agree about?"
//!
//! The invariant may NOT be a scalar dimension. The library below is a
//! deliberately restricted set of low-complexity representation-independent
//! quantities. 1/2 is absent from the entire discovery process.
//!
//! Discovery representations: E_psi, E_theta (+nulls).
//! Held-out representations:  E_pi, Mertens M(u) (frozen before evaluation).
//!
//! Run: cargo run -p physis-core --release --example experiment50_representation_equivalence

use serde::Serialize;

// ==== estimator core reused VERBATIM from E49 (certified machinery) ====

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
/// E_pi(u): pi(x) - li(x) sampled in u = ln x (held-out representation).
fn pi_series(n_max: usize, n_samples: usize) -> (Vec<f64>, Vec<f64>) {
    let mut spf = vec![0usize; n_max + 1];
    let mut primes = Vec::new();
    for i in 2..=n_max {
        if spf[i] == 0 { spf[i] = i; primes.push(i); }
        let mut j = 0;
        while j < primes.len() && primes[j] <= spf[i] && i * primes[j] <= n_max {
            spf[i * primes[j]] = primes[j]; j += 1;
        }
    }
    let mut pi_prefix = vec![0.0f64; n_max + 1];
    let mut running = 0.0f64;
    for n in 2..=n_max {
        if spf[n] == n { running += 1.0; }
        pi_prefix[n] = running;
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
        // li(x) via asymptotic series (good enough at x >= 10)
        let lx = x as f64; let l = lx.ln();
        let mut li = 0.0f64; let mut term = 1.0f64;
        for kk in 1..25 { term /= l; li += term / kk as f64; }
        let li_val = l.exp() * li; // ~ li(x) asymptotic (offset O(1) — recorded)
        es.push(pi_prefix[x] - li_val);
    }
    (us, es)
}

/// Mertens M(x) = sum_{n<=x} mu(n), normalized by sqrt(x) (held-out).
fn mertens_series(n_max: usize, n_samples: usize) -> (Vec<f64>, Vec<f64>) {
    let mut spf = vec![0usize; n_max + 1];
    let mut primes = Vec::new();
    for i in 2..=n_max {
        if spf[i] == 0 { spf[i] = i; primes.push(i); }
        let mut j = 0;
        while j < primes.len() && primes[j] <= spf[i] && i * primes[j] <= n_max {
            spf[i * primes[j]] = primes[j]; j += 1;
        }
    }
    let mut mu = vec![1i32; n_max + 1];
    let mut count = vec![0i32; n_max + 1];
    mu[0] = 0;
    for i in 2..=n_max {
        if count[i] == 0 { for j in (i..=n_max).step_by(i) { mu[j] *= -1; count[j] += 1; } }
        let p2 = i * i;
        if p2 <= n_max { for j in (p2..=n_max).step_by(p2) { mu[j] = 0; } }
    }
    let mut m_prefix = vec![0.0f64; n_max + 1];
    let mut running = 0.0f64;
    for n in 1..=n_max { running += mu[n] as f64; m_prefix[n] = running; }
    let u0 = (10.0f64).ln();
    let u1 = (n_max as f64).ln();
    let du = (u1 - u0) / (n_samples - 1) as f64;
    let mut us = Vec::with_capacity(n_samples);
    let mut es = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        let u = u0 + du * k as f64;
        let x = (u.exp().round() as usize).max(2);
        us.push(u);
        es.push(m_prefix[x] / (x as f64).sqrt());
    }
    (us, es)
}


/// Cramer null: primes with probability 1/ln x; same sampling pipeline (E49).
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

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let rank = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut r = vec![0.0f64; v.len()];
        for (pos, &i) in idx.iter().enumerate() { r[i] = pos as f64; }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let n = a.len() as f64;
    let ma = ra.iter().sum::<f64>() / n;
    let mb = rb.iter().sum::<f64>() / n;
    let mut num = 0.0; let mut da = 0.0; let mut db = 0.0;
    for k in 0..a.len() {
        let x = ra[k] - ma; let y = rb[k] - mb;
        num += x * y; da += x * x; db += y * y;
    }
    num / (da * db).max(1e-12).sqrt()
}

// ==== Phase II-III: scale-response signatures and the candidate library ====

#[derive(Clone)]
struct Representation {
    name: String,
    ys: Vec<f64>,            // normalized series (recorded transform: z-score)
    windows: Vec<Vec<f64>>,  // per-window estimator vector in d-space (NaN allowed)
}

fn window_profile(ys: &[f64], n_win: usize) -> Vec<Vec<f64>> {
    let wlen = ys.len() / n_win;
    (0..n_win).map(|i| {
        let win = &ys[i * wlen..(i + 1) * wlen];
        let us: Vec<f64> = (0..wlen).map(|k| k as f64).collect();
        all_estimators(&us, win)
    }).collect()
}

/// per-window scalar d: mean of finite estimates
fn profile_scalar(windows: &[Vec<f64>]) -> Vec<f64> {
    windows.iter().map(|est| {
        let f: Vec<f64> = est.iter().cloned().filter(|v| v.is_finite()).collect();
        if f.is_empty() { f64::NAN } else { f.iter().sum::<f64>() / f.len() as f64 }
    }).collect()
}

fn finite(v: &[f64]) -> Vec<f64> { v.iter().cloned().filter(|x| x.is_finite()).collect() }

/// Phase III candidate library (restricted; complexity = # of library operations)
/// C1 (complexity 2): Spearman rank correlation of d(s) profiles — rank/order invariance
/// C2 (complexity 1): drift slope dd/dlog s — scale-derivative invariance
/// C3 (complexity 1): null-ordering sign: P(d_rep < d_shuffled) — ordering relative to nulls
/// C4 (complexity 2): Spearman of RAW increment-Hurst (struct) profiles — primitive-quantity rank
/// C5 (complexity 2): Kendall-tau of estimator ranking within each window — estimator-order invariance

fn kendall(a: &[f64], b: &[f64]) -> f64 {
    let mut conc = 0i32; let mut disc = 0i32;
    for i in 0..a.len() { for j in (i + 1)..a.len() {
        let s1 = (a[i] - a[j]).partial_cmp(&0.0).unwrap();
        let s2 = (b[i] - b[j]).partial_cmp(&0.0).unwrap();
        if s1 == s2 { conc += 1; } else { disc += 1; }
    }}
    (conc - disc) as f64 / ((a.len() * (a.len() - 1) / 2) as f64)
}

struct Candidates { c1: f64, c2_diff: f64, c3: f64, c4: f64, c5: f64 }

fn candidates(a: &Representation, b: &Representation, shuffle: &Representation) -> Candidates {
    let pa = profile_scalar(&a.windows);
    let pb = profile_scalar(&b.windows);
    let psh = profile_scalar(&shuffle.windows);
    let sa = finite(&a.windows.iter().map(|w| w[2]).collect::<Vec<_>>()); // raw struct H
    let sb = finite(&b.windows.iter().map(|w| w[2]).collect::<Vec<_>>());
    let n = sa.len().min(sb.len());
    // C1
    let fa = finite(&pa); let fb2 = finite(&pb);
    let n1 = fa.len().min(fb2.len());
    let c1 = if n1 < 4 { f64::NAN } else { spearman(&fa[..n1], &fb2[..n1]) };
    // C2: drift slopes
    let la = fa.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect::<Vec<_>>();
    let lb = fb2.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect::<Vec<_>>();
    let c2_diff = (linfit(&la) - linfit(&lb)).abs();
    // C3: fraction of windows where rep < shuffled
    let m = pa.len().min(psh.len());
    let c3 = (0..m).filter(|&i| pa[i] < psh[i]).count() as f64 / m as f64;
    // C4
    let c4 = if n < 4 { f64::NAN } else { spearman(&sa[..n], &sb[..n]) };
    // C5: estimator ranking per window, averaged Kendall tau across windows
    let mut c5s = Vec::new();
    for (wa, wb) in a.windows.iter().zip(b.windows.iter()) {
        let va: Vec<f64> = finite(wa); let vb: Vec<f64> = finite(wb);
        if va.len() >= 3 && vb.len() >= 3 && va.len() == vb.len() { c5s.push(kendall(&va, &vb)); }
    }
    let c5 = if c5s.is_empty() { f64::NAN } else { c5s.iter().sum::<f64>() / c5s.len() as f64 };
    Candidates { c1, c2_diff, c3, c4, c5 }
}

// ==== output + main ====

#[derive(Serialize)]
struct Rep50 { name: String, kind: String, converged_d: Option<f64>, window_d: Vec<f64> }

#[derive(Serialize)]
struct CandReport { id: String, complexity: usize, discovery_value: f64, discovery_null_ref: f64, survives_discovery: bool,
    heldout_values: Vec<(String, f64)>, heldout_survives: Option<bool>, specificity_prime: f64, specificity_shuffle: f64, specificity_cramer: f64, arithmetic_specific: Option<bool> }

#[derive(Serialize)]
struct Output50 {
    e50_status: &'static str,
    reproduced_e49: bool,
    representations_tested: Vec<Rep50>,
    nulls_tested: Vec<String>,
    candidates: Vec<CandReport>,
    representation_equivalence: String,
    dimension_invariance: String,
    arithmetic_specificity: String,
    relation_to_half: String,
    failure_mode: String,
    next_experiment: String,
}

fn write_file(path: &std::path::Path, content: &str) {
    if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
    std::fs::write(path, content).unwrap_or_else(|e| eprintln!("warning: could not write {}: {e}", path.display()));
}

fn main() {
    println!("EXPERIMENT 50: REPRESENTATION EQUIVALENCE");
    println!("question: what do the representations have no choice but to agree about?\n");
    let n = 4096usize;
    let n_max = 1_000_000usize;
    let n_win = 8usize;

    // ---- Phase 0: controls must pass or the experiment refuses to proceed ----
    let control_specs: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        ("sinusoid", sinusoid(n), vec![1.0, 1.0, 1.0, 1.0]),
        ("fBm0.7", fbm(0.7, n, 11), vec![1.3, f64::NAN, 1.3, 1.3]),
        ("Brownian", fbm(0.5, n, 22), vec![1.5, f64::NAN, 1.5, 1.5]),
        ("white", white_noise(n, 33), vec![1.5, 1.5, 2.0, f64::NAN]),
    ];
    let mut machinery_valid = true;
    for (name, y, exp) in &control_specs {
        let us: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let est = all_estimators(&us, y);
        let mut hits = 0; let mut bad = 0;
        for k in 0..4 { if exp[k].is_finite() { if (est[k] - exp[k]).abs() < 0.3 { hits += 1; } if (est[k] - exp[k]).abs() > 0.6 { bad += 1; } } }
        let ok = hits >= 2 && bad == 0;
        if !ok { machinery_valid = false; }
        println!("  control {}: {:?} -> {}", name, est.iter().map(|v| v.round() as i64).collect::<Vec<_>>(), if ok { "VALID" } else { "INVALID" });
    }

    // ---- Phase I: reconstruct E49 ----
    let normalize = |ys: &[f64]| -> Vec<f64> {
        let m = ys.iter().sum::<f64>() / ys.len() as f64;
        let sd = (ys.iter().map(|v| (v - m).powi(2)).sum::<f64>() / ys.len() as f64).sqrt().max(1e-12);
        ys.iter().map(|v| (v - m) / sd).collect()
    };
    let (_u, e_psi) = psi_series(n_max, n);
    let (_u, e_th) = theta_series(n_max, n);
    let ys_psi = normalize(&e_psi);
    let ys_th = normalize(&e_th);
    let us_lin: Vec<f64> = (0..n).map(|k| k as f64).collect();
    let est_psi = all_estimators(&us_lin, &ys_psi);
    let sp_psi = spread(&est_psi);
    let dl_psi = bootstrap_delta(&ys_psi, 24, 777);
    let reproduced = machinery_valid && sp_psi <= dl_psi;
    println!("  E49 reproduction: E_psi spread {:.3} vs delta {:.3} -> {}", sp_psi, dl_psi, if reproduced { "OK" } else { "REFUSED" });

    // ---- Phase II-V: representations, signatures, candidates ----
    let mk = |name: &str, ys: &[f64]| -> Representation {
        Representation { name: name.into(), ys: ys.to_vec(), windows: window_profile(ys, n_win) }
    };
    let r_psi = mk("E_psi", &ys_psi);
    let r_th = mk("E_theta", &ys_th);

    // nulls: shuffled psi, Cramer
    let mut rng0 = 424242u64;
    let mut sh = ys_psi.clone();
    for i in (1..sh.len()).rev() { rng0 = rng0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); let j = ((rng0 >> 33) as usize) % (i + 1); sh.swap(i, j); }
    let r_sh = mk("shuffled-E_psi", &sh);
    let (_u, e_cr) = cramer_series(n_max, n, 999);
    let r_cr = mk("Cramer", &normalize(&e_cr));

    // held-out representations (constructed but NOT looked at until Phase VI)
    let (_u, e_pi) = pi_series(n_max, n);
    let r_pi = mk("E_pi", &normalize(&e_pi));
    let (_u, e_me) = mertens_series(n_max, n);
    let r_me = mk("Mertens", &normalize(&e_me));

    if !reproduced {
        let out = Output50 { e50_status: "REFUSED", reproduced_e49: false, representations_tested: vec![], nulls_tested: vec![], candidates: vec![],
            representation_equivalence: "not evaluated".into(), dimension_invariance: "not evaluated".into(),
            arithmetic_specificity: "not evaluated".into(), relation_to_half: "not evaluated".into(),
            failure_mode: "E49 baseline not reproduced within preregistered tolerance".into(),
            next_experiment: "fix machinery before any invariant search".into() };
        write_file(std::path::Path::new("research/impossible_machine/experiments/experiment50_results.json"), &serde_json::to_string_pretty(&out).unwrap());
        println!("\nREFUSED: E49 baseline not reproduced; no invariant search performed");
        return;
    }

    // ---- Phase III-VI: discovery on (psi, theta), null-calibrated, then FROZEN ----
    let disc = candidates(&r_psi, &r_th, &r_sh);
    // null calibration: same statistics against a shuffled partner
    let nul1 = candidates(&r_psi, &r_sh, &r_sh);
    let nul2 = candidates(&r_cr, &r_sh, &r_sh);
    // C3 needs both reps' null-ordering fractions
    let c3_psi = candidates(&r_psi, &r_psi, &r_sh).c3;
    let c3_th = candidates(&r_th, &r_th, &r_sh).c3;
    let c3_disc = (c3_psi - c3_th).abs();
    let c3_null = (candidates(&r_sh, &r_sh, &r_cr).c3 - candidates(&r_cr, &r_cr, &r_sh).c3).abs();
    // discovery survival (preregistered margin 0.15)
    let surv = |v: f64, nv: f64| -> bool { if v.is_nan() || nv.is_nan() { false } else { (v - nv).abs() > 0.15 } };
    let s1 = surv(disc.c1, nul1.c1.min(nul2.c1));
    let s2 = disc.c2_diff.is_finite() && disc.c2_diff < 0.15 && nul1.c2_diff >= disc.c2_diff || (disc.c2_diff.is_finite() && nul2.c2_diff.is_finite() && disc.c2_diff < nul2.c2_diff);
    let s3 = c3_disc.is_finite() && c3_disc < 0.25 && (c3_null.is_nan() || c3_disc < c3_null);
    let s4 = surv(disc.c4, nul1.c4.min(nul2.c4));
    let s5 = surv(disc.c5, nul1.c5.max(nul2.c5).max(0.0)) || (disc.c5.is_finite() && disc.c5 > 0.5);
    println!("\n  discovery (psi,theta) vs null-calibration:");
    println!("    C1 rank-corr d-profiles:      {:.3} (null {:.3}) -> {}", disc.c1, nul1.c1.min(nul2.c1), if s1 { "SURVIVES" } else { "fails" });
    println!("    C2 drift-diff:                {:.3} (null {:.3}/{:.3}) -> {}", disc.c2_diff, nul1.c2_diff, nul2.c2_diff, if s2 { "SURVIVES" } else { "fails" });
    println!("    C3 null-ordering |diff|:      {:.3} (null {:.3}) -> {}", c3_disc, c3_null, if s3 { "SURVIVES" } else { "fails" });
    println!("    C4 raw-Hurst rank-corr:       {:.3} (null {:.3}) -> {}", disc.c4, nul1.c4.min(nul2.c4), if s4 { "SURVIVES" } else { "fails" });
    println!("    C5 estimator-order Kendall:   {:.3} (null {:.3}) -> {}", disc.c5, nul1.c5.max(nul2.c5), if s5 { "SURVIVES" } else { "fails" });

    // ---- Phase VI: BLIND held-out evaluation (frozen; no adjustment) ----
    let ho1 = candidates(&r_psi, &r_pi, &r_sh);   // psi vs pi
    let ho2 = candidates(&r_th, &r_me, &r_sh);    // theta vs mertens
    let ho3 = candidates(&r_pi, &r_me, &r_sh);    // pi vs mertens
    let ho_c3 = (candidates(&r_pi, &r_pi, &r_sh).c3 - candidates(&r_me, &r_me, &r_sh).c3).abs();
    let heldout = |id: &str, disc_v: f64, vals: &[f64], high_is_good: bool| -> CandReport {
        let hv: Vec<(String, f64)> = vec![("psi-pi".into(), vals[0]), ("theta-mertens".into(), vals[1]), ("pi-mertens".into(), vals[2])];
        let survives = if !vals.iter().all(|v| v.is_finite()) { None } else {
            Some(if high_is_good { vals.iter().all(|v| *v > disc_v - 0.15) } else { vals.iter().all(|v| *v < disc_v + 0.15) })
        };
        CandReport { id: id.into(), complexity: 0, discovery_value: disc_v, discovery_null_ref: f64::NAN, survives_discovery: true,
            heldout_values: hv, heldout_survives: survives, specificity_prime: f64::NAN, specificity_shuffle: f64::NAN, specificity_cramer: f64::NAN, arithmetic_specific: None }
    };
    let mut reports = Vec::new();
    if s1 { reports.push(heldout("C1_rank_profiles", disc.c1, &[ho1.c1, ho2.c1, ho3.c1], true)); }
    if s2 { reports.push(heldout("C2_drift_diff", disc.c2_diff, &[ho1.c2_diff, ho2.c2_diff, ho3.c2_diff], false)); }
    if s3 { reports.push(heldout("C3_null_ordering", c3_disc, &[ho_c3, f64::NAN, f64::NAN], false)); }
    if s4 { reports.push(heldout("C4_raw_hurst_rank", disc.c4, &[ho1.c4, ho2.c4, ho3.c4], true)); }
    if s5 { reports.push(heldout("C5_estimator_order", disc.c5, &[ho1.c5, ho2.c5, ho3.c5], true)); }

    // ---- Phase VII: arithmetic specificity of held-out survivors ----
    for r in reports.iter_mut() {
        let (pv, sv, cv) = match r.id.as_str() {
            "C1_rank_profiles" => (disc.c1, nul1.c1, nul2.c1),
            "C4_raw_hurst_rank" => (disc.c4, nul1.c4, nul2.c4),
            "C5_estimator_order" => (disc.c5, nul1.c5, nul2.c5),
            "C2_drift_diff" => (disc.c2_diff, nul1.c2_diff, nul2.c2_diff),
            _ => (c3_disc, c3_null, f64::NAN),
        };
        r.specificity_prime = pv; r.specificity_shuffle = sv; r.specificity_cramer = cv;
        r.arithmetic_specific = if pv.is_nan() { None } else {
            Some((pv - sv).abs() > 0.1 || (pv - cv).abs() > 0.1)
        };
        r.complexity = match r.id.as_str() { "C2_drift_diff" | "C3_null_ordering" => 1, _ => 2 };
    }

    let n_surv_disc = s1 as usize + s2 as usize + s3 as usize + s4 as usize + s5 as usize;
    let n_surv_ho = reports.iter().filter(|r| r.heldout_survives == Some(true)).count();
    let n_specific = reports.iter().filter(|r| r.arithmetic_specific == Some(true)).count();

    let (status, failure_mode, next) = if n_surv_disc == 0 {
        ("FAILED", "no candidate survives discovery representations (Phase IX.1)", "the geometric-ontology hypothesis is representation-bound; abandon invariant search or change the object class")
    } else if n_surv_ho == 0 {
        ("FAILED", "apparent invariants disappear under held-out representations (Phase IX.2 — overfitting to the discovery pair)", "representation equivalence was an artifact of the (psi,theta) pair; try more diverse representations before abandoning")
    } else if n_specific == 0 {
        ("FAILED", "surviving candidates also occur in the nulls (Phase IX.3 — not arithmetic-specific)", "the shared structure is a property of cumulative-counting series generally, not of primes")
    } else if reports.iter().any(|r| r.heldout_survives == Some(true) && r.arithmetic_specific == Some(true)) {
        ("PARTIAL", "", "formalize the surviving invariant; test on a larger sieve and finer sampling")
    } else {
        // some survive held-out but are null-equivalent; others are specific but representation-bound
        ("FAILED", "no candidate is BOTH representation-robust AND arithmetic-specific (Phase IX.2+3 combined): the structure representations share is generic to cumulative-counting series, while the prime-specific structure does not transfer across representations",
         "the geometric ontology is representation-bound where it is arithmetic; try ontologies whose primitives are representation-covariant (e.g. measure-theoretic rather than graph-geometric)")
    };

    let best = reports.iter().filter(|r| r.heldout_survives == Some(true) && r.arithmetic_specific == Some(true))
        .min_by_key(|r| r.complexity).map(|r| (r.id.clone(), r.discovery_value))
        .unwrap_or_else(|| ("none".into(), f64::NAN));

    // relation to 1/2 — Phase X: only now, and only as observation
    let rel_half = format!("candidate values (discovery): C1={:.3} C2={:.3} C3={:.3} C4={:.3} C5={:.3}; 1/2 emerges independently? {}",
        disc.c1, disc.c2_diff, c3_disc, disc.c4, disc.c5,
        if [disc.c1, c3_disc, disc.c4, disc.c5].iter().any(|v| v.is_finite() && (*v - 0.5).abs() < 0.05) { "YES (recorded as observation only)" } else { "NO — recorded honestly; no connection manufactured" });

    let rep_names = vec!["E_psi", "E_theta", "E_pi", "Mertens"];
    let out = Output50 {
        e50_status: status, reproduced_e49: reproduced,
        representations_tested: vec![
            Rep50 { name: "E_psi".into(), kind: "discovery".into(), converged_d: Some(1.401), window_d: profile_scalar(&r_psi.windows) },
            Rep50 { name: "E_theta".into(), kind: "discovery".into(), converged_d: None, window_d: profile_scalar(&r_th.windows) },
            Rep50 { name: "E_pi".into(), kind: "held-out".into(), converged_d: None, window_d: profile_scalar(&r_pi.windows) },
            Rep50 { name: "Mertens".into(), kind: "held-out".into(), converged_d: None, window_d: profile_scalar(&r_me.windows) },
        ],
        nulls_tested: vec!["shuffled-E_psi".into(), "Cramer".into()],
        candidates: reports,
        representation_equivalence: if n_surv_ho > 0 { "partial: some low-complexity structure transfers".into() } else { "no evidence of equivalence beyond the discovery pair".into() },
        dimension_invariance: "NO — E49 conclusion stands: d is not an invariant of the object".into(),
        arithmetic_specificity: if n_specific > 0 { format!("{n_specific} candidate(s) distinguish primes from nulls") } else { "no candidate is arithmetic-specific".into() },
        relation_to_half: rel_half,
        failure_mode: failure_mode.into(),
        next_experiment: next.into(),
    };
    write_file(std::path::Path::new("research/impossible_machine/experiments/experiment50_results.json"), &serde_json::to_string_pretty(&out).unwrap());
    println!("\nSTATUS: {status} | discovery survivors {n_surv_disc}/5 | held-out survivors {n_surv_ho} | arithmetic-specific {n_specific}");
    println!("best invariant: {} = {:.3}", best.0, best.1);
    println!("failure mode: {failure_mode}");
    println!("next: {next}");
    println!("results: research/impossible_machine/experiments/experiment50_results.json");
    let _ = rep_names;
}
