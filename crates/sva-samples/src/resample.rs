// Concern: a recording at any instant: its own sample at its own rate, else a windowed sinc of proven bound | Non-concern: decoding, which node reads it | IO: (planes, rates, t) -> a sample, a bound

use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use crate::grid::{Grid, Round};

/// Relative to a tone's amplitude: -120 dB.
pub const BOUND: f64 = 1e-6;
/// Fractions of the lower rate.
pub const PASSBAND: f64 = 0.45;
pub const STOPBAND: f64 = 0.5;
/// `cos^(2P)` windows the sinc.
const P: i32 = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resampled {
    pub from_hz: u32,
    pub to_hz: u32,
    pub passband_hz: f64,
    pub stopband_hz: f64,
    pub ripple_db: f64,
    pub stopband_db: f64,
}

impl Resampled {
    pub fn of(from_hz: u32, to_hz: u32) -> Resampled {
        let lower = f64::from(from_hz.min(to_hz));
        let (_, pass, stop) = design(lower / f64::from(from_hz));
        Resampled {
            from_hz,
            to_hz,
            passband_hz: PASSBAND * lower,
            stopband_hz: STOPBAND * lower,
            ripple_db: 20.0 * (1.0 + pass).log10(),
            stopband_db: -20.0 * stop.log10(),
        }
    }
}

/// Poisson summation reads a tone at `f` back as `sum_m Hh(f - m)`, the passband box
/// convolved with the window's transform `W`; `2P` integrations by parts give
/// `|W(s)| <= TV(w^(2P)) / (2 pi s)^(2P+1)`, so `W`'s mass past `d` is at most `K / d^(2P)`.
fn design(lower: f64) -> (f64, f64, f64) {
    let (fp, fs) = (PASSBAND * lower, STOPBAND * lower);
    let c = (fp + fs) / 2.0;
    let n = f64::from(2 * P);
    let bounds = |h: f64| {
        let tv: f64 = (0..=2 * P)
            .map(|j| {
                let a = f64::from((P - j).abs()) * PI / h;
                binomial(2 * P, j) * (2.0 * a.powf(n) + 2.0 * h * a.powf(n + 1.0))
            })
            .sum::<f64>()
            / 4f64.powi(P);
        let k = tv / ((2.0 * PI).powf(n + 1.0) * n);
        let tail = |d: f64| k / d.powf(n);
        let images = |from: f64| tail(1.0 - from) + k / ((n - 1.0) * (1.0 - from).powf(n - 1.0));
        (
            2.0 * tail(c - fp) + 4.0 * images(fp + c),
            2.0 * tail(fs - c) + 4.0 * images(0.5 + c),
        )
    };
    let mut h = 1.0;
    loop {
        let (pass, stop) = bounds(h);
        if pass <= BOUND && stop <= BOUND {
            return (h, pass, stop);
        }
        h += 1.0;
    }
}

fn binomial(n: i32, k: i32) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * f64::from(n - i) / f64::from(i + 1))
}

#[derive(Debug)]
pub struct Recorded {
    planes: Arc<Vec<Vec<f32>>>,
    rate: u32,
    to: u32,
    frames: usize,
    cutoff: f64,
    half: f64,
}

impl Recorded {
    pub fn new(planes: Arc<Vec<Vec<f32>>>, rate: u32, to: u32) -> Recorded {
        let frames = planes.iter().map(Vec::len).min().unwrap_or(0);
        let lower = f64::from(rate.min(to)) / f64::from(rate);
        let (half, ..) = design(lower);
        Recorded {
            planes,
            rate,
            to,
            frames,
            cutoff: (PASSBAND + STOPBAND) / 2.0 * lower,
            half,
        }
    }

    pub fn width(&self) -> usize {
        self.planes.len()
    }

    pub fn resampled(&self) -> Option<Resampled> {
        (self.rate != self.to).then(|| Resampled::of(self.rate, self.to))
    }

    pub fn secs(&self) -> f64 {
        self.frames as f64 / f64::from(self.rate)
    }

    pub fn at(&self, c: usize, t: f64) -> f64 {
        if !(0.0..self.secs()).contains(&t) {
            return 0.0;
        }
        let plane = &self.planes[c];
        let own = Grid::of(self.rate);
        if self.rate == self.to
            && let Some(n) = own.step_at(t, Round::Even)
            && own.instant(n) == t
        {
            return usize::try_from(n)
                .ok()
                .and_then(|n| plane.get(n))
                .map_or(0.0, |v| f64::from(*v));
        }
        let u = t * f64::from(self.rate);
        let first = (u - self.half).ceil().max(0.0) as usize;
        let last = ((u + self.half).floor() as usize).min(self.frames - 1);
        (first..=last)
            .map(|k| f64::from(plane[k]) * self.kernel(u - k as f64))
            .sum()
    }

    fn kernel(&self, x: f64) -> f64 {
        if x.abs() >= self.half {
            return 0.0;
        }
        let y = 2.0 * self.cutoff * x;
        let sinc = match y == 0.0 {
            true => 1.0,
            false => (PI * y).sin() / (PI * y),
        };
        2.0 * self.cutoff * sinc * (FRAC_PI_2 * x / self.half).cos().powi(2 * P)
    }
}
