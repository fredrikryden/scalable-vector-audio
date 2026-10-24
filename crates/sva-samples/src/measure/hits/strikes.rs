// Concern: where a clip's hits fall: every peak of a band-wise superflux, new or a re-strike, a re-strike grid | Non-concern: what sounds after one (pitch) | IO: (Clip) -> strikes

use crate::profile::PSYCHOACOUSTIC_V1;

use super::spectra::{Clip, hann, median, power, sized};

const NOVELTY_SAMPLES_48K: f64 = 1024.0;
pub const HOP_SECS: f64 = 0.005;
/// Superflux: each bin's rise over the widest of its neighbours this many hops before.
const FLUX_LAG: usize = 2;
pub(super) const BAND_EDGES_HZ: (f64, f64) = (200.0, 2000.0);
/// A peak is the largest within this many hops either side.
const PEAK_HOPS: usize = 4;
const PICK_MEDIAN_SECS: f64 = 0.15;
const LAST_SECS: f64 = 0.05;
const CROP_SECS: f64 = 0.03;
const CROP_DB: f64 = 30.0;
const AUDIBLE_DB: f64 = 40.0;
const GRID_LEAST: usize = 8;
const GRID_SLACK_SECS: f64 = 0.012;
const GRID_DIP_DB: f64 = 1.5;
/// The periods tried: from, step, and how many, 80 ms to 1 s.
const GRID_PERIOD_SECS: (f64, f64) = (0.08, 0.0025);
const GRID_PERIODS: usize = 368;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    New,
    Restrike,
    Held,
}

impl Kind {
    pub fn key(self) -> &'static str {
        match self {
            Kind::New => "new",
            Kind::Restrike => "restrike",
            Kind::Held => "held",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cue {
    Attack,
    Grid,
    Crop,
}

impl Cue {
    pub fn key(self) -> &'static str {
        match self {
            Cue::Attack => "attack",
            Cue::Grid => "grid",
            Cue::Crop => "crop",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strike {
    pub at: f64,
    pub cue: Cue,
    pub kind: Kind,
    /// The novelty there over its median within 150 ms.
    pub strength: f64,
}

/// Each frame of a Hann `n` long every `hop`, centred on `k hop`, the clip zero past its ends.
fn frames(x: &[f64], n: usize, hop: usize) -> Vec<Vec<f64>> {
    let w = hann(n);
    (0..=x.len() / hop)
        .map(|k| {
            let framed: Vec<f64> = (0..n)
                .map(|i| {
                    let at = (k * hop + i) as i64 - (n / 2) as i64;
                    match at >= 0 && (at as usize) < x.len() {
                        true => x[at as usize] * w[i],
                        false => 0.0,
                    }
                })
                .collect();
            power(&framed, n).into_iter().map(f64::sqrt).collect()
        })
        .collect()
}

/// A log superflux per band, low, mid and high averaged, on one hop: the attacks' novelty.
fn novelty(clip: &Clip) -> (usize, Vec<f64>) {
    let hop = ((HOP_SECS * clip.sr).round() as usize).max(1);
    let n = sized(NOVELTY_SAMPLES_48K, clip.sr);
    let m = frames(&clip.x, n, hop);
    let most = m.iter().flatten().copied().fold(0.0, f64::max) + 1e-20;
    let l: Vec<Vec<f64>> = m
        .iter()
        .map(|f| {
            f.iter()
                .map(|v| libm::log10(1.0 + 1000.0 * v / most))
                .collect()
        })
        .collect();
    let bins = n / 2 + 1;
    let widest: Vec<Vec<f64>> = l
        .iter()
        .map(|f| {
            (0..bins)
                .map(|j| f[(j + bins - 1) % bins].max(f[j]).max(f[(j + 1) % bins]))
                .collect()
        })
        .collect();
    let hz = |j: usize| j as f64 * clip.sr / n as f64;
    let bands = [
        (0.0, BAND_EDGES_HZ.0),
        BAND_EDGES_HZ,
        (BAND_EDGES_HZ.1, f64::INFINITY),
    ];
    let mut attack = vec![0.0; l.len()];
    for (lo, hi) in bands {
        let per: Vec<f64> = (0..l.len())
            .map(|i| match i >= FLUX_LAG {
                true => (0..bins)
                    .filter(|&j| hz(j) >= lo && hz(j) < hi)
                    .map(|j| (l[i][j] - widest[i - FLUX_LAG][j]).max(0.0))
                    .sum(),
                false => 0.0,
            })
            .collect();
        let top = per.iter().copied().fold(0.0, f64::max) + 1e-20;
        attack
            .iter_mut()
            .zip(&per)
            .for_each(|(a, p)| *a += p / top / 3.0);
    }
    (hop, attack)
}

fn over_median(c: &[f64], i: usize, k: usize) -> f64 {
    c[i] - median(&c[i.saturating_sub(k)..(i + k + 1).min(c.len())])
}

/// Each local maximum, the largest within 4 hops, by how far it stands over its median.
fn picked(c: &[f64], k: usize) -> Vec<(usize, f64)> {
    (1..c.len().saturating_sub(1))
        .filter(|&i| {
            let around = c[i.saturating_sub(PEAK_HOPS)..(i + PEAK_HOPS + 1).min(c.len())]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            c[i] >= c[i - 1] && c[i] > c[i + 1] && c[i] == around
        })
        .map(|i| (i, over_median(c, i, k)))
        .filter(|(_, strength)| *strength > 0.0)
        .collect()
}

/// New where the power after it that was not there before outweighs the power carried
/// over; else a re-strike of what sounded.
fn judged(clip: &Clip, a: f64, strength: f64) -> Option<Strike> {
    let post = clip.spectrum(a + 0.015, clip.secs().min(a + 0.085))?;
    let pre = clip.spectrum(a - 0.07, a - 0.01);
    let post_sum: f64 = post.iter().sum();
    let new = match &pre {
        Some(pre) if pre.iter().sum::<f64>() >= post_sum * 1e-3 => {
            let fresh: f64 = post
                .iter()
                .zip(pre)
                .map(|(p, q)| (p - 2.0 * q).max(0.0))
                .sum();
            fresh / post_sum
        }
        _ => 1.0,
    };
    Some(Strike {
        at: a,
        cue: Cue::Attack,
        kind: match new > 1.0 - new {
            true => Kind::New,
            false => Kind::Restrike,
        },
        strength,
    })
}

/// Where most strikes sit on one period's grid, an audible grid point with no strike within
/// 25 ms and a 1.5 dB dip and rise is a re-strike too, however soft.
fn gridded(clip: &Clip, strikes: Vec<Strike>, strength: impl Fn(f64) -> f64) -> Vec<Strike> {
    let ts: Vec<f64> = strikes
        .iter()
        .filter(|s| s.cue != Cue::Crop)
        .map(|s| s.at)
        .collect();
    if ts.len() < GRID_LEAST {
        return strikes;
    }
    let mut best: Option<(usize, f64)> = None;
    for step in 0..GRID_PERIODS {
        let p = GRID_PERIOD_SECS.0 + step as f64 * GRID_PERIOD_SECS.1;
        let mut score = 0usize;
        for i in 0..ts.len() {
            for j in i + 1..ts.len() {
                let d = ts[j] - ts[i];
                let m = (d / p).round_ties_even();
                if (1.0..=4.0).contains(&m) && (d - m * p).abs() < GRID_SLACK_SECS {
                    score += 1;
                }
            }
        }
        let fits = ts
            .windows(2)
            .filter(|w| {
                let d = w[1] - w[0];
                (d - (d / p).round_ties_even() * p).abs() < GRID_SLACK_SECS
            })
            .count();
        if fits as f64 / (ts.len() - 1) as f64 >= 0.6 && best.is_none_or(|(s, _)| score > s) {
            best = Some((score, p));
        }
    }
    let Some((_, p)) = best.filter(|(s, _)| *s >= GRID_LEAST) else {
        return strikes;
    };
    let phase = median(
        &ts.iter()
            .map(|t| (t + p / 2.0).rem_euclid(p) - p / 2.0)
            .collect::<Vec<f64>>(),
    );
    let mut out = strikes;
    let first = phase.rem_euclid(p);
    let points = ((clip.secs() - LAST_SECS - first) / p).ceil().max(0.0) as usize;
    for g in (0..points).map(|i| first + i as f64 * p) {
        if !out.iter().any(|s| (s.at - g).abs() < 0.025) {
            let dip = clip.loudest(g - 0.005, g + 0.035) - clip.quietest(g - 0.03, g + 0.01);
            let audible = clip.loudest(g, g + 0.05) > clip.top_db - AUDIBLE_DB;
            if audible && dip >= GRID_DIP_DB {
                out.push(Strike {
                    at: g,
                    cue: Cue::Grid,
                    kind: Kind::Restrike,
                    strength: strength(g),
                });
            }
        }
    }
    out.sort_by(|a, b| a.at.total_cmp(&b.at));
    out
}

/// None in a silent clip; else every novelty peak before the last 50 ms, completed on a
/// re-strike grid, and the crop's start where the clip opens on sound.
pub fn strikes(clip: &Clip) -> (Vec<Strike>, f64) {
    let (hop, attack) = novelty(clip);
    if clip.top_db <= PSYCHOACOUSTIC_V1.silence_threshold_dbfs {
        return (Vec::new(), hop as f64 / clip.sr);
    }
    let k = ((PICK_MEDIAN_SECS / (hop as f64 / clip.sr)).round() as usize).max(1);
    let strength = |t: f64| {
        let i = ((t * clip.sr / hop as f64).round() as usize).min(attack.len() - 1);
        over_median(&attack, i, k)
    };
    let peaks: Vec<Strike> = picked(&attack, k)
        .into_iter()
        .map(|(i, strength)| ((i * hop) as f64 / clip.sr, strength))
        .filter(|(t, _)| *t <= clip.secs() - LAST_SECS)
        .filter_map(|(t, strength)| judged(clip, t, strength))
        .collect();
    let mut out = gridded(clip, peaks, strength);
    if out.first().is_none_or(|s| s.at > CROP_SECS)
        && clip.loudest(0.0, CROP_SECS) > clip.top_db - CROP_DB
    {
        out.insert(
            0,
            Strike {
                at: 0.0,
                cue: Cue::Crop,
                kind: Kind::Held,
                strength: strength(0.0),
            },
        );
    }
    (out, hop as f64 / clip.sr)
}
