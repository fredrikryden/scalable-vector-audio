// Concern: where a clip's notes strike: candidate attacks, the ones after which something new sounds, a re-strike grid | Non-concern: what sounds after one (voices.rs) | IO: (Clip) -> strikes

use crate::profile::PSYCHOACOUSTIC_V1;

use super::spectra::{Clip, cents, db_of_power, hann, median, power, sized};
use super::voices::voices;

const NOVELTY_SAMPLES_48K: f64 = 1024.0;
const PARTIAL_SAMPLES_48K: f64 = 4096.0;
pub const HOP_SECS: f64 = 0.005;
/// Superflux: each bin's rise over the widest of its neighbours this many hops before.
const FLUX_LAG: usize = 2;
const PARTIAL_LAG: usize = 6;
const PARTIAL_RANGE_DB: f64 = 45.0;
const RISE_CEILING_DB: f64 = 30.0;
const BAND_EDGES_HZ: (f64, f64) = (200.0, 2000.0);
const PICK_WAIT_SECS: f64 = 0.05;
const PICK_MEDIAN_SECS: f64 = 0.15;
const ATTACK_DELTA: f64 = 0.06;
const PARTIAL_DELTA: f64 = 0.12;
const LAST_SECS: f64 = 0.05;
const NEW_SHARE: f64 = 0.45;
const NEW_RISE_DB: f64 = 3.0;
const RESTRIKE_DB: f64 = 3.0;
const FAST_WOBBLE: f64 = 0.5;
/// A dip this deep falls near silence between strikes: a fast train of them, not a beat.
const SILENT_DIP_DB: f64 = 20.0;
const AUDIBLE_DB: f64 = 40.0;
const GLIDE_RISE_DB: f64 = 1.5;
const GLIDE_CENTS: f64 = 300.0;
const GLIDE_SHARE: f64 = 0.8;
const REPEAT_CENTS: f64 = 50.0;
const ONE_PER_SECS: f64 = 0.06;
const CROP_SECS: f64 = 0.03;
const CROP_DB: f64 = 30.0;
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
    Partial,
    Grid,
    Crop,
}

impl Cue {
    pub fn key(self) -> &'static str {
        match self {
            Cue::Attack => "attack",
            Cue::Partial => "partial",
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
    /// The share of the power after it that was not there before.
    pub new: f64,
    pub rise_db: f64,
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

/// Two novelty curves on one hop: a log superflux per band, low, mid and high averaged,
/// for attacks; and the rise of each bin that is a peak now, weighted by its share of the
/// frame, for a partial appearing or rising, a soft or legato attack.
fn novelty(clip: &Clip) -> (usize, Vec<f64>, Vec<f64>) {
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

    let n2 = sized(PARTIAL_SAMPLES_48K, clip.sr);
    let m2 = frames(&clip.x, n2, hop);
    let l2: Vec<Vec<f64>> = m2
        .iter()
        .map(|f| f.iter().map(|v| 20.0 * libm::log10(v + 1e-12)).collect())
        .collect();
    let bins2 = n2 / 2 + 1;
    let mut partial: Vec<f64> = (0..l2.len())
        .map(|i| {
            let f = &l2[i];
            let top = f.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = m2[i].iter().map(|v| v * v).sum::<f64>() + 1e-20;
            (0..bins2)
                .filter(|&j| {
                    f[j] >= f[(j + bins2 - 1) % bins2]
                        && f[j] >= f[(j + 1) % bins2]
                        && f[j] > top - PARTIAL_RANGE_DB
                })
                .map(|j| {
                    let rise = match i >= PARTIAL_LAG {
                        true => {
                            let before = (1..=PARTIAL_LAG)
                                .map(|d| l2[i - d][j])
                                .fold(f64::INFINITY, f64::min);
                            (f[j] - before).clamp(0.0, RISE_CEILING_DB)
                        }
                        false => 0.0,
                    };
                    rise * (m2[i][j] * m2[i][j] / total).sqrt()
                })
                .sum()
        })
        .collect();
    let top = partial.iter().copied().fold(0.0, f64::max) + 1e-20;
    partial.iter_mut().for_each(|v| *v /= top);
    (hop, attack, partial)
}

/// Each local maximum, the largest within 4 hops, standing `delta` over the median within
/// 150 ms; one within 50 ms of the last replaces it where larger.
fn picked(c: &[f64], (hop, sr): (usize, f64), delta: f64) -> Vec<(f64, f64)> {
    let k = ((PICK_MEDIAN_SECS / (hop as f64 / sr)).round() as usize).max(1);
    let mut out: Vec<(f64, f64)> = Vec::new();
    for i in 1..c.len().saturating_sub(1) {
        let around = c[i.saturating_sub(4)..(i + 5).min(c.len())]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let local = median(&c[i.saturating_sub(k)..(i + k + 1).min(c.len())]);
        if !(c[i] >= c[i - 1] && c[i] > c[i + 1] && c[i] == around && c[i] > local + delta) {
            continue;
        }
        let t = (i * hop) as f64 / sr;
        if let Some(last) = out.last_mut()
            && t - last.0 < PICK_WAIT_SECS
        {
            if c[i] > last.1 {
                *last = (t, c[i]);
            }
            continue;
        }
        out.push((t, c[i]));
    }
    out
}

/// The autocorrelation of the 5 ms level about its mean over `[a - half, a + half)`, its
/// highest between lags `lo` and `hi` frames: a regular wobble.
fn wobble(clip: &Clip, a: f64, half: f64, (lo, hi): (usize, usize)) -> f64 {
    let held: Vec<f64> = clip
        .level_t
        .iter()
        .zip(&clip.level_db)
        .filter(|(t, _)| **t >= a - half && **t < a + half)
        .map(|(_, v)| *v)
        .collect();
    if held.len() < 2 * hi {
        return 0.0;
    }
    let mean = held.iter().sum::<f64>() / held.len() as f64;
    let v: Vec<f64> = held.iter().map(|x| x - mean).collect();
    let r = |lag: usize| (0..v.len() - lag).map(|i| v[i] * v[i + lag]).sum::<f64>();
    let zero = r(0) + 1e-20;
    (lo..hi)
        .map(|lag| r(lag) / zero)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The first difference's level about `a` over its median before: the air a strike carries.
fn tick_db(clip: &Clip, a: f64) -> f64 {
    if a <= 0.1 {
        return 0.0;
    }
    let pick = |lo: f64, hi: f64| -> Vec<f64> {
        clip.level_t
            .iter()
            .zip(&clip.tick)
            .filter(|(t, _)| **t >= lo && **t < hi)
            .map(|(_, v)| *v)
            .collect()
    };
    let now = pick(a - 0.005, a + 0.015)
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    let before = median(&pick(a - 0.1, a - 0.01));
    20.0 * libm::log10((now + 1e-20) / (before + 1e-20))
}

/// A strike landing on what already sounded: the new power after it within 300 cents of the
/// harmonics of a voice before, or of a peak before, a glide settling or a swell beating.
fn lands(clip: &Clip, pre: &[f64], post: &[f64]) -> bool {
    let before: Vec<f64> = voices(&clip.peaks(pre, -45.0, 10.0))
        .iter()
        .map(|v| v.hz)
        .collect();
    let fresh: Vec<f64> = post
        .iter()
        .zip(pre)
        .map(|(p, q)| (p - 2.0 * q).max(0.0))
        .collect();
    let fresh = clip.peaks(&fresh, -45.0, 10.0);
    let quiet = clip.peaks(pre, -60.0, 6.0);
    if before.is_empty() || fresh.is_empty() {
        return false;
    }
    let (mut near, mut all) = (0.0, 0.0);
    for &(hz, db) in &fresh {
        let w = libm::pow(10.0, db / 10.0);
        let harmonic = before.iter().any(|&q| {
            let k = (hz / q).round_ties_even().max(1.0);
            cents(hz, q * k).abs() < GLIDE_CENTS
        });
        let peak = quiet.iter().any(|&(q, _)| cents(hz, q).abs() < GLIDE_CENTS);
        all += w;
        if harmonic || peak {
            near += w;
        }
    }
    near / all >= GLIDE_SHARE
}

/// Every voice of the power new after it already sounded before, or a harmonic of it did:
/// a note beating or swelling again, no new one.
fn repeats(clip: &Clip, pre: &[f64], post: &[f64]) -> bool {
    let before: Vec<f64> = voices(&clip.peaks(pre, -45.0, 10.0))
        .iter()
        .map(|v| v.hz)
        .collect();
    let fresh: Vec<f64> = post
        .iter()
        .zip(pre)
        .map(|(p, q)| (p - 2.0 * q).max(0.0))
        .collect();
    let after = voices(&clip.peaks(&fresh, -45.0, 10.0));
    !before.is_empty()
        && !after.is_empty()
        && after
            .iter()
            .all(|v| before.iter().any(|&q| related(v.hz, q)))
}

/// One a harmonic of the other, up to the fourth, within 50 cents.
fn related(a: f64, b: f64) -> bool {
    let (lo, hi) = (a.min(b), a.max(b));
    let k = (hi / lo).round_ties_even();
    (1.0..=4.0).contains(&k) && cents(hi, k * lo).abs() < REPEAT_CENTS
}

fn judged(clip: &Clip, a: f64, cue: Cue) -> Option<Strike> {
    let dur = clip.secs();
    let post = clip.spectrum(a + 0.015, dur.min(a + 0.085))?;
    let pre = clip.spectrum(a - 0.07, a - 0.01);
    let post_sum: f64 = post.iter().sum();
    let (new, rise_db) = match &pre {
        Some(pre) if pre.iter().sum::<f64>() >= post_sum * 1e-3 => {
            let fresh: f64 = post
                .iter()
                .zip(pre)
                .map(|(p, q)| (p - 2.0 * q).max(0.0))
                .sum();
            (
                fresh / post_sum,
                db_of_power(post_sum / pre.iter().sum::<f64>()),
            )
        }
        _ => (1.0, 60.0),
    };
    let dip = clip.loudest(a, a + 0.04) - clip.quietest(a - 0.03, a + 0.005);
    let tick = tick_db(clip, a);
    let fast = wobble(clip, a, 0.15, (4, 17));
    let mut kind = match (
        new >= NEW_SHARE || rise_db >= NEW_RISE_DB,
        dip >= RESTRIKE_DB,
    ) {
        (true, _) => Some(Kind::New),
        (false, true) if tick >= RESTRIKE_DB && (fast < FAST_WOBBLE || dip >= SILENT_DIP_DB) => {
            Some(Kind::Restrike)
        }
        _ => None,
    };
    if let (Some(Kind::New), Some(pre)) = (kind, &pre) {
        let quiet = rise_db < GLIDE_RISE_DB;
        let unstruck = tick < 2.0 && rise_db < 10.0;
        let swell = unstruck && wobble(clip, a, 0.4, (20, 61)) > FAST_WOBBLE;
        let beat = unstruck && fast > FAST_WOBBLE && repeats(clip, pre, &post);
        if ((quiet || swell) && lands(clip, pre, &post)) || beat {
            kind = None;
        }
    }
    let audible = clip.loudest(a, a + 0.05) > clip.top_db - AUDIBLE_DB;
    kind.filter(|_| audible).map(|kind| Strike {
        at: a,
        cue,
        kind,
        new,
        rise_db,
    })
}

/// Where most strikes sit on one period's grid, an audible grid point with a weak cue, a
/// candidate within 20 ms or a 1.5 dB dip and rise, is a re-strike too, however soft.
fn gridded(clip: &Clip, strikes: Vec<Strike>, candidates: &[(f64, Cue)]) -> Vec<Strike> {
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
            let near = candidates.iter().find(|(t, _)| (t - g).abs() < 0.02);
            let dip = clip.loudest(g - 0.005, g + 0.035) - clip.quietest(g - 0.03, g + 0.01);
            let audible = clip.loudest(g, g + 0.05) > clip.top_db - AUDIBLE_DB;
            if audible && (near.is_some() || dip >= GRID_DIP_DB) {
                out.push(Strike {
                    at: near.map_or(g, |c| c.0),
                    cue: Cue::Grid,
                    kind: Kind::Restrike,
                    new: 0.0,
                    rise_db: 0.0,
                });
            }
        }
    }
    out.sort_by(|a, b| a.at.total_cmp(&b.at));
    out
}

/// None in a clip under the profile's silence threshold; else candidates from both novelty
/// curves, kept where something new sounds after them, one per 60 ms (the most new power and
/// rise), completed on a re-strike grid, and the crop's start where the clip opens on sound.
pub fn strikes(clip: &Clip) -> (Vec<Strike>, f64) {
    let (hop, attack, partial) = novelty(clip);
    if clip.top_db <= PSYCHOACOUSTIC_V1.silence_threshold_dbfs {
        return (Vec::new(), hop as f64 / clip.sr);
    }
    let mut candidates: Vec<(f64, Cue)> = picked(&attack, (hop, clip.sr), ATTACK_DELTA)
        .into_iter()
        .map(|(t, _)| (t, Cue::Attack))
        .chain(
            picked(&partial, (hop, clip.sr), PARTIAL_DELTA)
                .into_iter()
                .map(|(t, _)| (t, Cue::Partial)),
        )
        .collect();
    candidates.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then((a.1 == Cue::Partial).cmp(&(b.1 == Cue::Partial)))
    });
    let kept: Vec<Strike> = candidates
        .iter()
        .filter(|(t, _)| *t <= clip.secs() - LAST_SECS)
        .filter_map(|&(t, cue)| judged(clip, t, cue))
        .collect();
    let weight = |s: &Strike| s.new + s.rise_db / 20.0;
    let mut one: Vec<Strike> = Vec::new();
    for s in kept {
        match one.last_mut() {
            Some(last) if s.at - last.at < ONE_PER_SECS => {
                if weight(&s) > weight(last) + 0.05 {
                    *last = s;
                }
            }
            _ => one.push(s),
        }
    }
    let mut out = gridded(clip, one, &candidates);
    if out.first().is_none_or(|s| s.at > CROP_SECS)
        && clip.loudest(0.0, CROP_SECS) > clip.top_db - CROP_DB
    {
        out.insert(
            0,
            Strike {
                at: 0.0,
                cue: Cue::Crop,
                kind: Kind::Held,
                new: 0.0,
                rise_db: 0.0,
            },
        );
    }
    (out, hop as f64 / clip.sr)
}
