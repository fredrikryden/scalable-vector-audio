// Concern: the starter's voices at each hit by a harmonic sieve, no reading, deleted once the starter reads `pitch` | Non-concern: where hits fall | IO: (planes, rate, start, Hits) -> voices per hit

use crate::fft::{dft, fft, idft, ifft};

use super::hits::spectra::{Clip, db_of_power, median};
use super::hits::{Hits, Kind};

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
const PEAK_HI_HZ: f64 = 10_000.0;
const PEAK_LO_HZ: f64 = 25.0;
const PEAK_FLOOR_DB: f64 = -45.0;
const PEAK_PROMINENCE_DB: f64 = 10.0;
const VOICE_SPAN_SECS: f64 = 0.3;
const SETTLE_SECS: f64 = 0.015;
const PRE_SECS: (f64, f64) = (0.07, 0.01);
const NEW_OVER: f64 = 1.5;
const SHORT_SECS: f64 = 0.03;
const HIT_SECS: f64 = 0.12;
const LATE_SHARE: f64 = 0.4;
const LATE_LEAST_SECS: f64 = 0.05;
const BEND_CENTS: f64 = 250.0;
const TONE_COHERENCE: f64 = 0.8;
const PERIOD_CENTS: f64 = 60.0;
const SAME_VOICE_CENTS: f64 = 50.0;
const PERIOD_LEAST_48K: f64 = 1024.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Tonal,
    Noisy,
}

impl Tone {
    pub fn key(self) -> &'static str {
        match self {
            Tone::Tonal => "tonal",
            Tone::Noisy => "noisy",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Voiced {
    pub hz: f64,
    pub tone: Tone,
    /// Sounding newly at the hit: in the power new at a new one, or every voice of a held
    /// start; none of a re-strike, whose voices sound on from before.
    pub new: bool,
    pub partials: Vec<Partial>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Partial {
    pub hz: f64,
    pub db: f64,
    pub harmonic: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct Voice {
    hz: f64,
    share: f64,
    partials: Vec<Partial>,
}

fn harmonic(hz: f64, f0: f64) -> usize {
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
fn voices(peaks: &[(f64, f64)]) -> Vec<Voice> {
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
fn period(seg: &[f64], sr: f64) -> Option<(f64, f64)> {
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
fn coherence(clip: &Clip, (a, b): (f64, f64), partials: &[&Partial]) -> Option<f64> {
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

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * libm::log2(a / b)
}

/// (hz, dB), parabolically interpolated in dB.
fn peaks(clip: &Clip, p: &[f64]) -> Vec<(f64, f64)> {
    let hz = |i: usize| i as f64 * clip.bin_hz;
    let most = p.iter().copied().fold(0.0, f64::max);
    let l: Vec<f64> = p
        .iter()
        .map(|v| db_of_power(v + most * 1e-8 + 1e-30))
        .collect();
    let top = (0..l.len())
        .filter(|&i| hz(i) > PEAK_LO_HZ && hz(i) < PEAK_HI_HZ)
        .map(|i| l[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut out = Vec::new();
    for i in 2..l.len().saturating_sub(2) {
        let f = hz(i);
        if f > PEAK_HI_HZ {
            break;
        }
        let around = l[i.saturating_sub(3)..(i + 4).min(l.len())]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !(f > PEAK_LO_HZ
            && l[i] > top + PEAK_FLOOR_DB
            && l[i] >= l[i - 1]
            && l[i] > l[i + 1]
            && l[i] == around)
        {
            continue;
        }
        let (a, b, c) = (l[i - 1], l[i], l[i + 1]);
        let curve = a - 2.0 * b + c;
        let shift = match curve != 0.0 {
            true => (0.5 * (a - c) / curve).clamp(-0.5, 0.5),
            false => 0.0,
        };
        let lo = f * libm::pow(2.0, -1.0 / 6.0) - 4.0 * clip.bin_hz;
        let hi = f * libm::pow(2.0, 1.0 / 6.0) + 4.0 * clip.bin_hz;
        let near: Vec<f64> = (0..l.len())
            .filter(|&j| hz(j) > lo && hz(j) < hi)
            .map(|j| l[j])
            .collect();
        if b - median(&near) >= PEAK_PROMINENCE_DB {
            out.push((f + shift * clip.bin_hz, b - 0.25 * (a - c) * shift));
        }
    }
    out
}

/// Each hit's voices, in order: a sieve on the power new at it (all of a held start's, none
/// of a re-strike's) and on all sounding over 300 ms, each read again later (a bend), tonal
/// or noisy by its partials' coherence over 10 ms, a noisy one taking a near periodicity.
pub fn measure(planes: &[&[f64]], sr: f64, start: f64, hits: &Hits) -> Vec<Vec<Voiced>> {
    let clip = Clip::new(&super::hits::mono(planes), sr);
    hits.hits
        .iter()
        .map(|h| voiced(&clip, h.kind, (h.at_secs - start, h.next_secs - start)))
        .collect()
}

fn voiced(clip: &Clip, kind: Kind, (a, b): (f64, f64)) -> Vec<Voiced> {
    let span_end = b.min(a + VOICE_SPAN_SECS);
    let post = match b - a > SHORT_SECS {
        true => clip.spectrum(a + SETTLE_SECS, span_end),
        false => clip.spectrum(a, b),
    };
    let Some(post) = post else {
        return Vec::new();
    };
    let pre = clip.spectrum(a - PRE_SECS.0, a - PRE_SECS.1);
    let diff: Vec<f64> = match (&pre, kind) {
        (Some(pre), k) if k != Kind::Held => post
            .iter()
            .zip(pre)
            .map(|(p, q)| (p - NEW_OVER * q).max(0.0))
            .collect(),
        _ => post.clone(),
    };
    let all = voices(&peaks(clip, &post));
    let fresh = match (kind, pre.is_some() && diff.iter().sum::<f64>() > 0.0) {
        (Kind::New, true) => voices(&peaks(clip, &diff)),
        (Kind::New, false) | (Kind::Held, _) => all.clone(),
        (Kind::Restrike, _) => Vec::new(),
    };
    let seg = &clip.x[clip.at(a + SETTLE_SECS)..clip.at(span_end).max(clip.at(a + SETTLE_SECS))];
    let periodic = match seg.len() as f64 > PERIOD_LEAST_48K * clip.sr / 48_000.0 {
        true => period(seg, clip.sr),
        false => None,
    };
    let late_from = a + LATE_LEAST_SECS.max(LATE_SHARE * (b - a));
    let late = match b - a > HIT_SECS {
        true => clip
            .spectrum(late_from, b - 0.01)
            .map(|p| voices(&peaks(clip, &p))),
        false => None,
    };
    let late_seg = &clip.x[clip.at(late_from).min(clip.at(b - 0.01))..clip.at(b - 0.01)];
    let landed =
        match b - a > HIT_SECS && late_seg.len() as f64 > PERIOD_LEAST_48K * clip.sr / 48_000.0 {
            true => period(late_seg, clip.sr),
            false => None,
        }
        .or(periodic);
    let settle = |v: &Voice| -> f64 {
        late.as_ref()
            .and_then(|late| {
                late.iter()
                    .filter(|w| cents(w.hz, v.hz).abs() < BEND_CENTS)
                    .min_by(|x, y| {
                        libm::fabs(libm::log2(x.hz / v.hz))
                            .total_cmp(&libm::fabs(libm::log2(y.hz / v.hz)))
                    })
            })
            .map_or(v.hz, |w| w.hz)
    };
    let fresh: Vec<(&Voice, f64)> = fresh.iter().map(|v| (v, settle(v))).collect();
    let others = all.iter().map(|v| (v, settle(v))).filter(|(_, hz)| {
        fresh
            .iter()
            .all(|(_, n)| cents(*hz, *n).abs() > SAME_VOICE_CENTS)
    });
    fresh
        .iter()
        .map(|&(v, hz)| (v, hz, true))
        .chain(others.map(|(v, hz)| (v, hz, false)))
        .map(|(v, hz, new)| {
            let mut loudest: Vec<&Partial> = v.partials.iter().collect();
            loudest.sort_by(|x, y| y.db.total_cmp(&x.db));
            loudest.truncate(3);
            let coherence = coherence(clip, (a + SETTLE_SECS, span_end), &loudest);
            let tone = match coherence.is_some_and(|c| c >= TONE_COHERENCE) {
                true => Tone::Tonal,
                false => Tone::Noisy,
            };
            let hz = match (tone, landed) {
                (Tone::Noisy, Some((per, _))) => {
                    let k = libm::log2(hz / per).round_ties_even();
                    let octave = per * libm::pow(2.0, k);
                    match k.abs() <= 2.0 && cents(hz, octave).abs() < PERIOD_CENTS {
                        true => per,
                        false => hz,
                    }
                }
                _ => hz,
            };
            Voiced {
                hz,
                tone,
                new,
                partials: v.partials.clone(),
            }
        })
        .collect()
}
