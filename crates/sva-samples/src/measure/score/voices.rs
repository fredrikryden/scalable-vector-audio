// Concern: what sounds in one span: voices by a harmonic sieve, periodicity, coherence, a hit's kind | Non-concern: where a span starts (strikes.rs) | IO: (Clip, peaks or span) -> voices, period, kind

use crate::fft::{dft, fft, idft, ifft};

use super::spectra::{Clip, cents, db_of_power};

const HARMONIC_SHARE: f64 = 0.035;
const HARMONIC_CENTS: f64 = 40.0;
const START_DB: f64 = -30.0;
const MOST_VOICES: usize = 6;
const OVER_DB: f64 = 6.0;
const LEAST_SHARE: f64 = 0.01;
const LONE_SHARE: f64 = 0.15;
const PERIOD_HZ: (f64, f64) = (40.0, 1500.0);
const PERIOD_CLOSE: f64 = 0.9;
const COHERENCE_LAG_SECS: f64 = 0.01;
const COHERENCE_LEAST_HZ: f64 = 20.0;
const COHERENCE_SHARE: f64 = 0.03;
const HIT_SAMPLES_48K: f64 = 1024.0;
const KICK_HZ: f64 = 150.0;
const HAT_HZ: f64 = 5000.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Partial {
    pub hz: f64,
    pub db: f64,
    pub harmonic: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Voice {
    pub hz: f64,
    pub db: f64,
    pub share: f64,
    pub partials: Vec<Partial>,
}

pub fn harmonic(hz: f64, f0: f64) -> usize {
    let k = (hz / f0).round_ties_even();
    let ok = k >= 1.0
        && (hz / f0 - k).abs() <= HARMONIC_SHARE
        && cents(hz, k * f0).abs() <= HARMONIC_CENTS;
    if ok { k as usize } else { 0 }
}

struct Open {
    f0: f64,
    ks: Vec<usize>,
    partials: Vec<Partial>,
    loudest: f64,
}

/// The lowest peak no voice explains starts one, a heard fundamental; a higher peak joins
/// the first voice whose series it continues, k f0 with k - 1 or k - 2 in it, unless it
/// stands 6 dB over that voice's loudest, a loud partial over a weak sub being its own voice.
pub fn voices(peaks: &[(f64, f64)]) -> Vec<Voice> {
    if peaks.is_empty() {
        return Vec::new();
    }
    let top = peaks.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let mut by_hz = peaks.to_vec();
    by_hz.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut open: Vec<Open> = Vec::new();
    for (hz, db) in by_hz {
        let home = open.iter().position(|v| {
            let k = harmonic(hz, v.f0);
            k >= 2
                && (v.ks.contains(&(k - 1)) || v.ks.contains(&(k - 2)) || k <= 2)
                && db <= v.loudest + OVER_DB
        });
        match home {
            Some(i) => {
                let v = &mut open[i];
                let k = harmonic(hz, v.f0);
                if !v.ks.contains(&k) {
                    v.ks.push(k);
                }
                v.partials.push(Partial {
                    hz,
                    db,
                    harmonic: k,
                });
                v.loudest = v.loudest.max(db);
            }
            None if db > top + START_DB && open.len() < MOST_VOICES => open.push(Open {
                f0: hz,
                ks: vec![1],
                partials: vec![Partial {
                    hz,
                    db,
                    harmonic: 1,
                }],
                loudest: db,
            }),
            None => {}
        }
    }
    let total: f64 = peaks.iter().map(|p| libm::pow(10.0, p.1 / 10.0)).sum();
    let mut out: Vec<Voice> = open
        .into_iter()
        .filter_map(|v| {
            let w = |p: &Partial| libm::pow(10.0, p.db / 10.0);
            let held: f64 = v.partials.iter().map(w).sum();
            let k = |p: &Partial| p.harmonic as f64;
            let num: f64 = v.partials.iter().map(|p| w(p) * p.hz * k(p)).sum();
            let den: f64 = v.partials.iter().map(|p| w(p) * k(p) * k(p)).sum();
            let share = held / total;
            let lone = v.partials.len() < 2 && share < LONE_SHARE;
            (share >= LEAST_SHARE && !lone).then(|| Voice {
                hz: num / den,
                db: db_of_power(held),
                share,
                partials: v.partials,
            })
        })
        .collect();
    out.sort_by(|a, b| b.share.total_cmp(&a.share));
    out
}

/// McLeod's normalised square difference over lags for 40 Hz to 1.5 kHz: the first peak
/// within 0.9 of the highest, the shortest period that repeats; its frequency and height.
pub fn period(seg: &[f64], sr: f64) -> Option<(f64, f64)> {
    let n = seg.len();
    let most = ((sr / PERIOD_HZ.0) as usize).min(n / 2);
    let least = (sr / PERIOD_HZ.1) as usize;
    if most < least + 3 {
        return None;
    }
    let size = (2 * n).next_power_of_two();
    let (mut re, mut im) = (vec![0.0; size], vec![0.0; size]);
    re[..n].copy_from_slice(seg);
    fft(&mut re, &mut im);
    for k in 0..size {
        re[k] = re[k] * re[k] + im[k] * im[k];
        im[k] = 0.0;
    }
    ifft(&mut re, &mut im);
    let mut c = Vec::with_capacity(n);
    let mut run = 0.0;
    for v in seg {
        run += v * v;
        c.push(run);
    }
    let d: Vec<f64> = (0..most)
        .map(|lag| {
            let m = c[n - 1] - if lag > 0 { c[lag - 1] } else { 0.0 } + c[n - 1 - lag];
            2.0 * re[lag] / (m + 1e-20)
        })
        .collect();
    let peaks: Vec<usize> = (least.max(1)..most - 1)
        .filter(|&i| d[i] >= d[i - 1] && d[i] > d[i + 1] && d[i] > 0.0)
        .collect();
    let top = peaks
        .iter()
        .map(|&i| d[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let i = *peaks.iter().find(|&&i| d[i] >= PERIOD_CLOSE * top)?;
    Some((sr / i as f64, d[i]))
}

/// A tone's band stays near 1 against itself 10 ms on; noise of bandwidth B falls as
/// e^(-pi B tau).
pub fn coherence(clip: &Clip, (a, b): (f64, f64), partials: &[&Partial]) -> Option<f64> {
    let (a, b) = (clip.at(a), clip.at(b));
    let seg = &clip.x[a..b.max(a)];
    let lag = (COHERENCE_LAG_SECS * clip.sr) as usize;
    if seg.len() < 3 * lag || lag == 0 {
        return None;
    }
    let n = seg.len();
    let (re, im) = dft(seg, &vec![0.0; n]);
    let (mut sum, mut weights) = (0.0, 0.0);
    for p in partials {
        let half = COHERENCE_LEAST_HZ.max(COHERENCE_SHARE * p.hz);
        let (mut zr, mut zi) = (vec![0.0; n], vec![0.0; n]);
        for k in 0..n.div_ceil(2) {
            let f = k as f64 * clip.sr / n as f64;
            if f > p.hz - half && f < p.hz + half {
                zr[k] = 2.0 * re[k];
                zi[k] = 2.0 * im[k];
            }
        }
        let (zr, zi) = idft(&zr, &zi);
        let (mut cr, mut ci, mut e0, mut e1) = (0.0, 0.0, 0.0, 0.0);
        for m in 0..seg.len() - lag {
            let (ar, ai, br, bi) = (zr[m], zi[m], zr[m + lag], zi[m + lag]);
            cr += ar * br + ai * bi;
            ci += ar * bi - ai * br;
            e0 += ar * ar + ai * ai;
            e1 += br * br + bi * bi;
        }
        let w = libm::pow(10.0, p.db / 10.0);
        sum += w * (cr * cr + ci * ci).sqrt() / ((e0 * e1).sqrt() + 1e-30);
        weights += w;
    }
    (weights > 0.0).then(|| sum / weights)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    Kick,
    Snare,
    Hat,
}

impl Hit {
    pub fn key(self) -> &'static str {
        match self {
            Hit::Kick => "kick",
            Hit::Snare => "snare",
            Hit::Hat => "hat",
        }
    }
}

pub fn hit(clip: &Clip, (a, b): (usize, usize)) -> Option<Hit> {
    let n = super::spectra::sized(HIT_SAMPLES_48K, clip.sr);
    let p = clip.spectrum_of(a, b, n)?;
    let total = p.iter().sum::<f64>() + 1e-30;
    let hz = |k: usize| k as f64 * clip.sr / n as f64;
    let low: f64 = (0..p.len())
        .filter(|&k| hz(k) < KICK_HZ)
        .map(|k| p[k])
        .sum();
    let high: f64 = (0..p.len())
        .filter(|&k| hz(k) >= HAT_HZ)
        .map(|k| p[k])
        .sum();
    Some(match (low / total > 0.5, high / total > 0.4) {
        (true, _) => Hit::Kick,
        (false, true) => Hit::Hat,
        _ => Hit::Snare,
    })
}
