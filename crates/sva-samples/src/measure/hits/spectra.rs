// Concern: the averaged spectra and levels the hits' rules read off one clip | Non-concern: what a rule decides (strikes.rs) | IO: (samples, span) -> power, levels

use std::f64::consts::TAU;

use crate::fft::fft;

const SPECTRUM_SAMPLES_48K: f64 = 4096.0;
const LEAST_SAMPLES_48K: f64 = 256.0;
pub const LEVEL_FRAME_SECS: f64 = 0.005;

pub fn sized(samples_48k: f64, sr: f64) -> usize {
    crate::measure::scaled_pow2(samples_48k, sr)
}

pub fn hann(n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..n)
            .map(|k| 0.5 - 0.5 * libm::cos(TAU * k as f64 / (n - 1) as f64))
            .collect(),
    }
}

pub fn power(x: &[f64], n: usize) -> Vec<f64> {
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    re[..x.len().min(n)].copy_from_slice(&x[..x.len().min(n)]);
    fft(&mut re, &mut im);
    (0..=n / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect()
}

pub fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => v[n / 2],
        n => (v[n / 2 - 1] + v[n / 2]) / 2.0,
    }
}

pub fn db_of_power(p: f64) -> f64 {
    10.0 * libm::log10(p)
}

/// One clip's samples, its mean taken out, and the level envelope every rule reads.
pub struct Clip {
    pub x: Vec<f64>,
    pub sr: f64,
    pub n: usize,
    pub level_t: Vec<f64>,
    pub level_db: Vec<f64>,
    pub top_db: f64,
}

fn rms_frames(x: &[f64], k: usize) -> Vec<f64> {
    (0..x.len() / k)
        .map(|f| {
            let s = &x[f * k..(f + 1) * k];
            (s.iter().map(|v| v * v).sum::<f64>() / k as f64 + 1e-20).sqrt()
        })
        .collect()
}

impl Clip {
    pub fn new(x: &[f64], sr: f64) -> Clip {
        let mean = x.iter().sum::<f64>() / x.len().max(1) as f64;
        let x: Vec<f64> = x.iter().map(|v| v - mean).collect();
        let k = ((LEVEL_FRAME_SECS * sr) as usize).max(1);
        let rms = rms_frames(&x, k);
        let level_db: Vec<f64> = rms.iter().map(|r| 20.0 * libm::log10(*r)).collect();
        let n = sized(SPECTRUM_SAMPLES_48K, sr);
        Clip {
            level_t: (0..rms.len())
                .map(|f| (f as f64 + 0.5) * k as f64 / sr)
                .collect(),
            top_db: level_db.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            level_db,
            n,
            x,
            sr,
        }
    }

    pub fn secs(&self) -> f64 {
        self.x.len() as f64 / self.sr
    }

    pub fn at(&self, t: f64) -> usize {
        ((t * self.sr) as i64).clamp(0, self.x.len() as i64) as usize
    }

    /// The 5 ms levels whose centres lie in `[a, b)`, or one -200 dB where none does.
    pub fn levels(&self, a: f64, b: f64) -> Vec<f64> {
        let held: Vec<f64> = self
            .level_t
            .iter()
            .zip(&self.level_db)
            .filter(|(t, _)| **t >= a && **t < b)
            .map(|(_, v)| *v)
            .collect();
        match held.is_empty() {
            true => vec![-200.0],
            false => held,
        }
    }

    pub fn loudest(&self, a: f64, b: f64) -> f64 {
        self.levels(a, b)
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn quietest(&self, a: f64, b: f64) -> f64 {
        self.levels(a, b).into_iter().fold(f64::INFINITY, f64::min)
    }

    /// One Hann over a span shorter than the transform, else Hann frames a quarter apart.
    pub fn spectrum(&self, a: f64, b: f64) -> Option<Vec<f64>> {
        self.spectrum_of(self.at(a), self.at(b), self.n)
    }

    pub fn spectrum_of(&self, a: usize, b: usize, n: usize) -> Option<Vec<f64>> {
        let least = (LEAST_SAMPLES_48K * self.sr / 48_000.0).round() as usize;
        if b < a || b - a < least {
            return None;
        }
        let seg = &self.x[a..b];
        if seg.len() < n {
            let w = hann(seg.len());
            let framed: Vec<f64> = seg.iter().zip(&w).map(|(v, w)| v * w).collect();
            let len = seg.len() as f64;
            return Some(power(&framed, n).into_iter().map(|p| p / len).collect());
        }
        let w = hann(n);
        let mut sum = vec![0.0; n / 2 + 1];
        let mut count = 0usize;
        let mut s = 0;
        while s + n <= seg.len() {
            let framed: Vec<f64> = seg[s..s + n].iter().zip(&w).map(|(v, w)| v * w).collect();
            for (acc, p) in sum.iter_mut().zip(power(&framed, n)) {
                *acc += p;
            }
            count += 1;
            s += n / 4;
        }
        Some(
            sum.into_iter()
                .map(|p| p / count as f64 / n as f64)
                .collect(),
        )
    }
}
