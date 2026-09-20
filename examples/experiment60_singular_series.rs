//! EXPERIMENT 60: SINGULAR-SERIES FIT — quantitative twin-constant check.
//!
//! E52 direction-exists + stable (D ~1.2 vs nulls ~0.10). E60 asks whether
//! the fitted twin constant agrees with 2C2 = 1.32032 at N=1e6.
//! Frozen spec: research/impossible_machine/EXPERIMENT60_SPEC.md.
//! 1/2 never in generation, scoring, ranking.
//!
//! S(k) = 2*C2*prod_{odd p|k}(p-1)/(p-2); fit 2C2_hat = mean R(k)/S_unit(k).
//! Run: cargo run -p physis-core --release --example experiment60_singular_series

use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Serialize, Clone)]
struct E60Output {
    title: String, context: String, n_max: usize, kmax: usize, seed: u64,
    d_disc: f64, d_hold: f64, r2_disc: f64, r2_hold: f64,
    nulls: Vec<f64>, fit_disc: f64, fit_hold: f64,
    ci_disc: (f64, f64), ci_hold: (f64, f64), eff_disc: f64,
    g1: bool, g2: bool, g3: bool, survives: bool,
    rel_half: String, verdict: String, artifacts: Vec<String>,
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed
}
fn f01(seed: &mut u64) -> f64 { (lcg(seed) >> 11) as f64 / (1u64 << 53) as f64 }

fn sieve(n_max: usize) -> Vec<usize> {
    let mut is_p = vec![true; n_max + 1];
    if n_max >= 1 { is_p[0] = false; }
    if n_max >= 1 { is_p[1] = false; }
    let mut p = 2usize;
    while p * p <= n_max {
        if is_p[p] { let mut m = p * p; while m <= n_max { is_p[m] = false; m += p; } }
        p += 1;
    }
    (2..=n_max).filter(|&n| is_p[n]).collect()
}

fn odd_divs(mut k: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut p = 3usize;
    while p * p <= k {
        if k % p == 0 { out.push(p); while k % p == 0 { k /= p; } }
        p += 2;
    }
    if k > 2 { out.push(k); }
    out
}

fn s_unit(k: usize) -> f64 {
    odd_divs(k).iter().map(|&p| (p - 1) as f64 / (p - 2) as f64).product()
}

fn gaps(jumps: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let mut gs = Vec::new();
    let mut ds = Vec::new();
    for i in 0..jumps.len().saturating_sub(1) {
        gs.push((jumps[i + 1] - jumps[i]) as f64);
        ds.push(((jumps[i] as f64).ln()).max(1e-9));
    }
    (gs, ds)
}

fn exp_c(ds: &[f64], kmax: usize) -> Vec<f64> {
    let mut e = vec![0.0f64; kmax / 2];
    for &d in ds {
        let q = 1.0 / d;
        for (i, k) in (2..=kmax).step_by(2).enumerate() {
            e[i] += q * (1.0 - q).powi((k - 1) as i32);
        }
    }
    e
}

fn obs_c(gs: &[f64], kmax: usize) -> Vec<f64> {
    let mut n = vec![0.0f64; kmax / 2];
    for &g in gs {
        let gk = g as usize;
        if (2..=kmax).contains(&gk) && gk % 2 == 0 { n[(gk / 2) - 1] += 1.0; }
    }
    n
}

fn r_prof(gs: &[f64], ds: &[f64], kmax: usize) -> Vec<f64> {
    let n = obs_c(gs, kmax);
    let e = exp_c(ds, kmax);
    n.iter().zip(e.iter()).map(|(&a, &b)| if b > 1e-9 { a / b } else { 0.0 }).collect()
}

fn dev(r: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0f64, 0usize);
    for &v in r { if v.is_finite() { s += (v - 1.0).powi(2); c += 1; } }
    if c == 0 { 0.0 } else { (s / c as f64).sqrt() }
}
fn fit(r: &[f64], kmax: usize) -> f64 {
    let mut s = 0.0;
    let mut c = 0usize;
    for (i, k) in (2..=kmax).step_by(2).enumerate() {
        let v = r[i] / s_unit(k);
        if v.is_finite() { s += v; c += 1; }
    }
    if c == 0 { f64::NAN } else { s / c as f64 }
}

fn bci(r: &[f64], kmax: usize, b: usize, seed: &mut u64) -> (f64, f64) {
    let u: Vec<f64> = (2..=kmax).step_by(2).enumerate()
        .map(|(i, k)| r[i] / s_unit(k)).filter(|v| v.is_finite()).collect();
    if u.is_empty() { return (f64::NAN, f64::NAN); }
    let n = u.len();
    let mut ms = Vec::with_capacity(b);
    for _ in 0..b {
        let mut s = 0.0;
        for _ in 0..n { s += u[(f01(seed) * n as f64) as usize % n]; }
        ms.push(s / n as f64);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ms[(0.025 * b as f64) as usize], ms[((0.975 * b as f64) as usize).min(b - 1)])
}

fn null_cramer(lo: usize, hi: usize, kmax: usize, seed: &mut u64) -> f64 {
    let mut j = Vec::new();
    for x in lo..=hi { if f01(seed) < 1.0 / (x as f64).ln() { j.push(x); } }
    let (gs, ds) = gaps(&j);
    dev(&r_prof(&gs, &ds, kmax))
}

fn null_geo(ds: &[f64], kmax: usize, seed: &mut u64) -> f64 {
    let mut gs = Vec::new();
    for &d in ds {
        let q = 1.0 / d;
        let mut x = 1.0f64;
        while f01(seed) > q { x += 1.0; }
        gs.push(x);
    }
    dev(&r_prof(&gs, ds, kmax))
}

fn wf(path: &Path, data: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).ok();
    let mut f = File::create(path).unwrap();
    f.write_all(data.as_bytes()).unwrap();
}

const C2X2: f64 = 1.32032;
const G3H: f64 = 0.15;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== EXPERIMENT 60: SINGULAR-SERIES FIT ===");
    let n_max = 1_000_000usize;
    let kmax = 40usize;
    let mut seed = 20260912u64;
    let hi = n_max / 2;
    let ps = sieve(n_max);
    let dp: Vec<usize> = ps.iter().cloned().filter(|&p| p > 20 && p <= hi).collect();
    let hp: Vec<usize> = ps.iter().cloned().filter(|&p| p > hi).collect();
    println!("primes {} (disc {} hold {})", ps.len(), dp.len(), hp.len());
    let d = gaps(&dp);
    let h = gaps(&hp);
    let rd = r_prof(&d.0, &d.1, kmax);
    let rh = r_prof(&h.0, &h.1, kmax);
    let (dd, dh) = (dev(&rd), dev(&rh));
    let (r2d, r2h) = (rd[0], rh[0]);
    println!("D disc={:.3} R2={:.3} | hold D={:.3} R2={:.3}", dd, r2d, dh, r2h);
    let (fd, fh) = (fit(&rd, kmax), fit(&rh, kmax));
    let (cd, ch) = (bci(&rd, kmax, 200, &mut seed), bci(&rh, kmax, 200, &mut seed));
    let eff = (fd - 1.0) / ((cd.1 - cd.0) / 2.0).max(1e-9);
    println!("fit disc={:.3} [{:.3},{:.3}] hold={:.3} [{:.3},{:.3}] tgt={:.5}",
        fd, cd.0, cd.1, fh, ch.0, ch.1, C2X2);
    let nc = null_cramer(20, hi, kmax, &mut seed);
    let ng = null_geo(&d.1, kmax, &mut seed);
    let mx = nc.max(ng);
    println!("nulls ({:.3}, {:.3})", nc, ng);
    let g1 = dd > 1.5 * mx.max(1e-6);
    let g2 = (dh - dd).abs() <= 0.5 * dd && r2d > 1.0 && r2h > 1.0;
    let g3 = (fd - C2X2).abs() <= G3H && (fh - C2X2).abs() <= G3H;
    let sv = g1 && g2 && g3;
    let verdict = if sv {
        format!("AGREEMENT — 2C2 disc={:.3} hold={:.3} in {:.5}±{:.2}; G1 D={:.3} vs nulls {:.3}; G2 stable.",
            fd, fh, C2X2, G3H, dd, mx)
    } else {
        let mut r = Vec::new();
        if !g1 { r.push("G1 fail".to_string()); }
        if !g2 { r.push("G2 fail".to_string()); }
        if !g3 { r.push(format!("G3 miss: {:.3}/{:.3} vs {:.5}±{:.2}", fd, fh, C2X2, G3H)); }
        format!("{} — {}. No quantitative claim at N=1e6.", if g1 && g2 { "PARTIAL" } else { "MISS" }, r.join("; "))
    };
    let rel = format!("no 1/2 in gen/score; fit {:.3}/{:.3}; D {:.3}/{:.3}", fd, fh, dd, dh);
    let out = E60Output {
        title: "Singular-series fit (2C2) — quantitative H-L check".into(),
        context: "E52 direction+stable; E60 value vs 2C2=1.32032 at N=1e6".into(),
        n_max, kmax, seed: 20260912, d_disc: dd, d_hold: dh,
        r2_disc: r2d, r2_hold: r2h, nulls: vec![nc, ng],
        fit_disc: fd, fit_hold: fh, ci_disc: cd, ci_hold: ch, eff_disc: eff,
        g1, g2, g3, survives: sv, rel_half: rel.clone(),
        verdict: verdict.clone(),
        artifacts: vec![
            "physis-core/examples/experiment60_singular_series.rs".into(),
            "physis-core/research/impossible_machine/experiments/experiment60_results.json".into(),
            "physis-core/research/impossible_machine/EXPERIMENT60_RESULTS.md".into(),
        ],
    };
    let op = Path::new("research/impossible_machine/experiments/experiment60_results.json");
    wf(op, &serde_json::to_string_pretty(&out).unwrap());
    println!("\nVERDICT: {verdict}");
    println!("rel 1/2: {rel}");
    println!("gates G1={g1} G2={g2} G3={g3} | {}", op.display());
    Ok(())
}
