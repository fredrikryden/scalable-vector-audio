// Concern: each modulation band's information rate 2B and its proven error bound | Non-concern: the bank and its filters (modulation.rs) | IO: (band, window, hop) -> Rate

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use super::modulation::{Band, Element, LOWPASS_HZ, PRE_LOWPASS_HZ};

const UNIT: f64 = f64::EPSILON / 2.0;
/// Over the roundings `response` and a node's map can take, counted.
const EVALUATION_ROUNDINGS: f64 = 64.0;

/// `min(2 B h, 1)` and how far it can be from that of the exact `B`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rate {
    pub per_frame: f64,
    pub bound: f64,
}

fn sin_pi(x: f64) -> (f64, f64) {
    let k = x.round_ties_even();
    let sign = match k.rem_euclid(2.0) == 0.0 {
        true => 1.0,
        false => -1.0,
    };
    (k, sign * libm::sin(PI * (x - k)))
}

/// `cos^4(pi t/T)`'s transform at `x = nu T` over its value at 0; at 0, 1, 2 its weights.
fn window_squared(x: f64) -> f64 {
    let (k, sin) = sin_pi(x);
    if x == k {
        return match k.abs() {
            0.0 => 1.0,
            1.0 => 2.0 / 3.0,
            2.0 => 1.0 / 6.0,
            _ => 0.0,
        };
    }
    4.0 * sin / (PI * x * (x - 1.0) * (x + 1.0) * (x - 2.0) * (x + 2.0))
}

fn response(band: &Band, window_secs: f64, nu: f64) -> f64 {
    let w = window_squared(nu * window_secs);
    let pre = nu / PRE_LOWPASS_HZ;
    w * w / (1.0 + pre * pre) * band.power(nu)
}

#[derive(Clone, Copy)]
struct Rect {
    x0: f64,
    x1: f64,
    y: f64,
}

impl Rect {
    fn distance(&self, x: f64, y: f64) -> f64 {
        let dx = (self.x0 - x).max(x - self.x1).max(0.0);
        let dy = (y.abs() - self.y).max(0.0);
        libm::hypot(dx, dy)
    }

    fn scaled(&self, by: f64) -> Rect {
        Rect {
            x0: self.x0 * by,
            x1: self.x1 * by,
            y: self.y * by,
        }
    }
}

/// Above the real axis; each has its conjugate below.
fn poles(band: &Band) -> Vec<(f64, f64)> {
    let mut held = vec![(0.0, PRE_LOWPASS_HZ)];
    match band.element {
        Element::Lowpass => {
            let s = LOWPASS_HZ * FRAC_1_SQRT_2;
            held.extend([(s, s), (-s, s)]);
        }
        _ => held.push((band.hz, band.bandwidth_hz / 2.0)),
    }
    held
}

/// `|H(z)|^2` over `rect`: the window's transform is at most `e^(pi T |Im z|)`, its integral
/// over `cos^4` over its own, and `4 cosh(pi Im w) / (pi |w (w^2 - 1)(w^2 - 4)|)`.
fn bound(band: &Band, window_secs: f64, rect: Rect) -> f64 {
    let w = rect.scaled(window_secs);
    let v = PI * w.y;
    let near: f64 = (-2..=2).map(|k| w.distance(k as f64, 0.0)).product();
    let window = sva_formula::exp(v).min(4.0 * libm::cosh(v) / (PI * near));
    let pre = PRE_LOWPASS_HZ / rect.distance(0.0, PRE_LOWPASS_HZ);
    let filter = match band.element {
        Element::Lowpass => {
            let s = LOWPASS_HZ * FRAC_1_SQRT_2;
            let (a, b) = (rect.distance(s, s), rect.distance(-s, s));
            libm::pow(LOWPASS_HZ, 4.0) / (a * a * b * b)
        }
        _ => {
            let half = band.bandwidth_hz / 2.0;
            let d = rect.distance(band.hz, half);
            half * half / (d * d)
        }
    };
    window * window * pre * pre * filter
}

/// Gauss-Legendre on [-1, 1]: Newton from Chebyshev's guess while its step shrinks.
fn gauss(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let mut x = libm::cos(PI * (i as f64 + 0.75) / (n as f64 + 0.5));
            let mut last = f64::INFINITY;
            loop {
                let (p, dp) = legendre(n, x);
                let step = p / dp;
                if step.abs() >= last || step == 0.0 {
                    return (x, 2.0 / ((1.0 - x * x) * dp * dp));
                }
                last = step.abs();
                x -= step;
            }
        })
        .collect()
}

fn legendre(n: usize, x: f64) -> (f64, f64) {
    let (mut before, mut p) = (1.0, x);
    for k in 2..=n {
        let k = k as f64;
        let next = ((2.0 * k - 1.0) * x * p - (k - 1.0) * before) / k;
        before = p;
        p = next;
    }
    (p, n as f64 * (x * p - before) / (x * x - 1.0))
}

fn rounding(m: f64) -> f64 {
    m * UNIT / (1.0 - m * UNIT)
}

struct Integrals<'a> {
    band: &'a Band,
    window_secs: f64,
    rules: Vec<Vec<(f64, f64)>>,
    sums: (f64, f64),
    errors: (f64, f64),
}

impl Integrals<'_> {
    fn rule(&mut self, doubling: usize) -> &[(f64, f64)] {
        while self.rules.len() <= doubling {
            let n = 2 << self.rules.len();
            self.rules.push(gauss(n));
        }
        &self.rules[doubling]
    }

    /// Trefethen's (2008, eq. 4.14) bound, under 2^-53 of the sum.
    fn panel(&mut self, a: f64, b: f64) {
        let flat = Rect {
            x0: a,
            x1: b,
            y: 0.0,
        };
        let nearest = poles(self.band)
            .iter()
            .map(|(x, y)| flat.distance(*x, *y))
            .fold(f64::INFINITY, f64::min);
        let half = 0.5 * (b - a);
        if half > nearest.min(1.0 / self.window_secs) {
            let middle = 0.5 * (a + b);
            self.panel(a, middle);
            self.panel(middle, b);
            return;
        }
        let centre = 0.5 * (a + b);
        let y = (0.5 * nearest).min(1.0 / self.window_secs);
        let reach = libm::hypot(half, y);
        let rho = (reach + y) / half;
        let rect = Rect {
            x0: centre - reach,
            x1: centre + reach,
            y,
        };
        let most = bound(self.band, self.window_secs, rect);
        let disc = y.min(reach - half).min(1.0 / (PI * self.window_secs));
        let near = Rect {
            x0: a - disc,
            x1: b + disc,
            y: disc,
        };
        let steep = bound(self.band, self.window_secs, near) / disc;
        let (band, window_secs) = (self.band, self.window_secs);
        for doubling in 0.. {
            let rule = self.rule(doubling).to_vec();
            let n = rule.len() as f64;
            let (s1, s2) = rule.iter().fold((0.0, 0.0), |(s1, s2), (x, w)| {
                let v = response(band, window_secs, centre + half * x);
                (s1 + w * v, s2 + w * v * v)
            });
            let (s1, s2) = (half * s1, half * s2);
            let shrink = 64.0 / (15.0 * (1.0 - 1.0 / (rho * rho)) * libm::pow(rho, 2.0 * n));
            let truncation = (half * most * shrink, half * most * most * shrink);
            let target = (UNIT * (self.sums.0 + s1), UNIT * (self.sums.1 + s2));
            if truncation.0 <= target.0 && truncation.1 <= target.1 {
                let moved = half * 2.0 * n * UNIT + 3.0 * UNIT * (centre.abs() + half);
                let relative = rounding(EVALUATION_ROUNDINGS) + rounding(5.0 * n + 1.0);
                self.sums = (self.sums.0 + s1, self.sums.1 + s2);
                self.errors.0 += truncation.0 + relative * s1 + 2.0 * half * steep * moved;
                self.errors.1 += truncation.1
                    + (2.0 * relative + UNIT) * s2
                    + 2.0 * half * steep * steep * disc * moved;
                return;
            }
        }
    }

    /// From `nu >= 3/T` and twice the centre: `(81/(10 pi))^2 (nu T)^-10 (150/nu)^2` times
    /// `(2.5/nu)^4` or `(bandwidth/nu)^2`.
    fn tail(&self, from: f64) -> (f64, f64) {
        let window = libm::pow(81.0 / (10.0 * PI), 2.0) * libm::pow(self.window_secs, -10.0);
        let (scale, power) = match self.band.element {
            Element::Lowpass => (libm::pow(LOWPASS_HZ, 4.0), 16.0),
            _ => (self.band.bandwidth_hz * self.band.bandwidth_hz, 14.0),
        };
        let k = window * PRE_LOWPASS_HZ * PRE_LOWPASS_HZ * scale;
        (
            k * libm::pow(from, 1.0 - power) / (power - 1.0),
            k * k * libm::pow(from, 1.0 - 2.0 * power) / (2.0 * power - 1.0),
        )
    }
}

/// B = (int_0^inf |H|^2)^2 / int_0^inf |H|^4, until the tail's bound is under 2^-53 of it.
pub fn rate(band: &Band, window_secs: f64, hop_secs: f64) -> Rate {
    let lower = (band.hz - band.bandwidth_hz / 2.0).max(0.0);
    let start = (3.0 / window_secs).max(2.0 * band.upper_edge_hz());
    let mut breaks = vec![
        0.0,
        lower,
        band.hz,
        band.upper_edge_hz(),
        1.0 / window_secs,
        2.0 / window_secs,
        PRE_LOWPASS_HZ,
        start.max(PRE_LOWPASS_HZ),
    ];
    breaks.sort_by(f64::total_cmp);
    breaks.dedup();
    let mut sum = Integrals {
        band,
        window_secs,
        rules: Vec::new(),
        sums: (0.0, 0.0),
        errors: (0.0, 0.0),
    };
    for pair in breaks.windows(2) {
        sum.panel(pair[0], pair[1]);
    }
    let mut from = breaks[breaks.len() - 1];
    while sum.tail(from).0 > UNIT * sum.sums.0 || sum.tail(from).1 > UNIT * sum.sums.1 {
        sum.panel(from, 2.0 * from);
        from *= 2.0;
    }
    let tail = sum.tail(from);
    let (i1, i2) = sum.sums;
    let (e1, e2) = (sum.errors.0 + tail.0, sum.errors.1 + tail.1);
    let per_frame = |b: f64| (2.0 * b * hop_secs).min(1.0);
    let held = per_frame(i1 * i1 / i2);
    let low = per_frame(libm::pow((i1 - e1).max(0.0), 2.0) / (i2 + e2));
    let high = per_frame(libm::pow(i1 + e1, 2.0) / (i2 - e2).max(0.0));
    Rate {
        per_frame: held,
        bound: (high - held).max(held - low),
    }
}
