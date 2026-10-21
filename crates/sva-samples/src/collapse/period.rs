// Concern: a periodic lane of lines summed by one transform, bounded | Non-concern: finding its period (sva-engine) | IO: (Direct, n, bins) -> sample k mod n

use std::f64::consts::TAU;

use sva_formula::C64;

use super::lines::Direct;
use super::run;

pub(crate) struct Period {
    samples: Vec<f64>,
}

impl Period {
    /// Where the lines fill an eighth of the bins or more, and the transform is cheaper than
    /// the ladders and within their bound.
    pub(crate) fn of(direct: &Direct, n: i64, bin: &dyn Fn(f64) -> Option<i64>) -> Option<Period> {
        let n = usize::try_from(n).ok().filter(|n| *n > 1)?;
        let (level, runs) = direct.parts();
        let mut bins = Vec::new();
        let mut held = std::collections::HashMap::new();
        for line in runs.iter().flat_map(|r| r.lines()) {
            let k = bin(line.hz)?.rem_euclid(n as i64) as usize;
            *held.entry(k).or_insert(0usize) += 1;
            bins.push((k, line.amp));
        }
        let shared = held.into_values().max().unwrap_or(1) - 1;
        if n > 8 * bins.len() {
            return None;
        }
        let factors = factors(n);
        let steps: usize = runs.iter().map(run::steps).sum();
        let cheaper = 2 * factors.iter().map(|p| p + 1).sum::<usize>() <= steps;
        let within = bound(n, &factors, norms(&bins), (level, shared)) <= direct.bound();
        (cheaper && within).then(|| Period {
            samples: whole(n, &factors, &bins, level),
        })
    }

    pub(crate) fn at(&self, k: i64) -> f64 {
        self.samples[k.rem_euclid(self.samples.len() as i64) as usize]
    }

    pub(crate) fn bytes(&self) -> usize {
        self.samples.len() * size_of::<f64>()
    }
}

fn whole(n: usize, factors: &[usize], bins: &[(usize, C64)], level: f64) -> Vec<f64> {
    let mut x = vec![C64::ZERO; n];
    for (k, a) in bins {
        x[*k] = x[*k] + a.conj();
    }
    let table: Vec<C64> = (0..n)
        .map(|j| {
            let (sin, cos) = libm::sincos(-TAU * j as f64 / n as f64);
            C64::new(cos, sin)
        })
        .collect();
    let mut out = vec![C64::ZERO; n];
    let mut scratch = Vec::new();
    transform(&x, 1, &mut out, (factors, &table, 1), &mut scratch);
    out.into_iter().map(|y| level + y.re).collect()
}

/// Decimation in time, a stage a prime factor, each a direct DFT.
fn transform(
    x: &[C64],
    stride: usize,
    out: &mut [C64],
    (factors, table, scale): (&[usize], &[C64], usize),
    scratch: &mut Vec<C64>,
) {
    let Some((&p, rest)) = factors.split_first() else {
        out[0] = x[0];
        return;
    };
    let m = out.len() / p;
    for (r, part) in out.chunks_mut(m).enumerate() {
        transform(
            &x[r * stride..],
            stride * p,
            part,
            (rest, table, scale * p),
            scratch,
        );
    }
    scratch.resize(p, C64::ZERO);
    for k in 0..m {
        for r in 0..p {
            scratch[r] = table[scale * r * k] * out[k + r * m];
        }
        for q in 0..p {
            out[k + q * m] = scratch.iter().enumerate().fold(C64::ZERO, |acc, (r, t)| {
                acc + *t * table[scale * m * (r * q % p)]
            });
        }
    }
}

fn factors(mut n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut p = 2;
    while p * p <= n {
        while n.is_multiple_of(p) {
            out.push(p);
            n /= p;
        }
        p += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

fn norms(bins: &[(usize, C64)]) -> (f64, f64) {
    let l1: f64 = bins.iter().map(|(_, a)| a.abs()).sum();
    let l2: f64 = bins.iter().map(|(_, a)| a.norm_sqr()).sum::<f64>().sqrt();
    let grow = 1.0 + 2.0 * (bins.len() + 2) as f64 * U;
    (l1 * grow, l2 * grow)
}

const U: f64 = f64::EPSILON / 2.0;

/// `|at(k) - S(k)|`. A stage of radix `p` is `sqrt(p)` times a unitary map, each output within
/// `eps` of the sum of its inputs' magnitudes, so `eps*sqrt(p)` of it in norm; stages compose
/// as `prod(1 + eps_k*sqrt(p_k)) - 1` of `sqrt(n)*|x|_2` (Higham, Accuracy and Stability, 3.7).
fn bound(n: usize, factors: &[usize], (l1, l2): (f64, f64), (level, shared): (f64, usize)) -> f64 {
    let gamma = |k: f64| k * U / (1.0 - k * U);
    let product = std::f64::consts::SQRT_2 * gamma(2.0);
    let twiddled = MU + product * (1.0 + MU);
    let grows = factors.iter().fold(1.0, |held, p| {
        let p = *p as f64;
        let eps = (1.0 + twiddled) * (1.0 + MU) * (1.0 + product) * (1.0 + gamma(p)) - 1.0;
        held * (1.0 + eps * p.sqrt())
    });
    let x = l2 * ((shared + 1) as f64).sqrt();
    let sample = (grows - 1.0) * (n as f64).sqrt() * x + gamma(shared as f64) * l1;
    let underflow = 8.0 * (n * (factors.len() + 1)) as f64 * f64::from_bits(1);
    (sample + U * (level.abs() + l1 + sample) + underflow) * (1.0 + 4.0 * U)
}

/// `-2 pi j / n`, `j < n`, is three roundings off, then `cos` and `sin` an ulp each.
const MU: f64 = (6.0002 * std::f64::consts::PI + 2.0 * std::f64::consts::SQRT_2) * U;
