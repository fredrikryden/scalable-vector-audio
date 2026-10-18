// Concern: spectral flux onset detection, each onset placed at the sample its strike rises | Non-concern: a tempo grid, what an onset means musically | IO: (planes, rate, start) -> Onsets

use crate::measure::spectrum::{MAX_PINNED_FRAME, magnitudes, pinned_frame};

/// ~23ms: short enough to localize a transient, long enough for low percussion.
const FRAME_SECS: f64 = 0.023;
/// Strikes 25 ms apart peak a few hops from each other.
const HOP_SECS: f64 = FRAME_SECS / 8.0;
/// Below this two novelty peaks are one transient's attack and decay.
const MIN_ONSET_GAP_SECS: f64 = 0.02;
const ADAPTIVE_WINDOW_SECS: f64 = 0.115;
/// A roll's flux peaks barely over its mean; the strike tells it from a swell.
const THRESHOLD_MULTIPLIER: f64 = 1.0;
const THRESHOLD_DELTA: f64 = 1e-6;
const IOI_BUCKETS_PER_SEC: f64 = 40.0;
const IOI_MAX_SECS: f64 = 2.0;
/// Under this share of the loudest frame a rise is leakage.
const LEVEL_FRACTION: f64 = 0.03;
const STRIKE_SPAN_SECS: f64 = 0.012;
/// ~5 dB over the span before, carried on as it moved: a strike jumps off a ring's fall.
const JUMP_ENERGY_RATIO: f64 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Onset {
    pub t_secs: f64,
    pub strength: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IoiBucket {
    pub lo_secs: f64,
    pub hi_secs: f64,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Onsets {
    pub onsets: Vec<Onset>,
    pub ioi_histogram: Vec<IoiBucket>,
    pub resolution_secs: f64,
}

struct SpectralFrame {
    t_secs: f64,
    mags: Vec<Vec<f64>>,
    span_secs: f64,
}

fn hop_of(sample_rate: f64) -> usize {
    ((HOP_SECS * sample_rate).round() as usize).max(1)
}

fn spectral_frames(planes: &[&[f64]], sample_rate: f64, start_secs: f64) -> Vec<SpectralFrame> {
    let frame_len = pinned_frame(FRAME_SECS, sample_rate).min(MAX_PINNED_FRAME);
    let hop = hop_of(sample_rate);
    let len = planes.iter().map(|p| p.len()).min().unwrap_or(0);
    (0..len)
        .step_by(hop)
        .map(|start| {
            let end = (start + frame_len).min(len);
            let mags = planes
                .iter()
                .map(|p| {
                    let mut frame = p[start..end].to_vec();
                    frame.resize(frame_len, 0.0);
                    magnitudes(&frame, sample_rate, Some(FRAME_SECS)).0
                })
                .collect();
            SpectralFrame {
                t_secs: start_secs + start as f64 / sample_rate,
                mags,
                span_secs: frame_len as f64 / sample_rate,
            }
        })
        .collect()
}

/// Every channel, so opposite phases cancel nothing.
pub fn detect(planes: &[&[f64]], sample_rate: f64, start_secs: f64) -> Onsets {
    let frames = spectral_frames(planes, sample_rate, start_secs);
    let floor = LEVEL_FRACTION * frames.iter().map(|f| total(&f.mags)).fold(0.0, f64::max);
    let flux = flux_of(&frames, floor);
    let onsets = pick_peaks(
        &flux,
        &frames,
        &Energy::of(planes, sample_rate, start_secs),
        floor,
    );
    Onsets {
        ioi_histogram: ioi_histogram(&onsets, sample_rate),
        onsets,
        resolution_secs: hop_of(sample_rate) as f64 / sample_rate,
    }
}

fn total(mags: &[Vec<f64>]) -> f64 {
    mags.iter().flatten().sum()
}

/// The rise into each frame; nothing precedes the buffer, so one opening above the floor
/// opens on a rise from silence.
fn flux_of(frames: &[SpectralFrame], floor: f64) -> Vec<f64> {
    let opens_on_sound = frames
        .first()
        .is_some_and(|f| floor > 0.0 && total(&f.mags) > floor);
    (0..frames.len())
        .map(|i| match i {
            0 => match opens_on_sound {
                true => total(&frames[0].mags),
                false => 0.0,
            },
            _ => frames[i - 1]
                .mags
                .iter()
                .zip(&frames[i].mags)
                .map(|(prev, curr)| spectral_flux(prev, curr))
                .sum(),
        })
        .collect()
}

pub(crate) fn spectral_flux(prev: &[f64], curr: &[f64]) -> f64 {
    prev.iter().zip(curr).map(|(&p, &c)| (c - p).max(0.0)).sum()
}

/// A candidate clears the threshold and the floor, peaks over the gap, and holds a strike;
/// a frame past the buffer's end reads silence.
fn pick_peaks(flux: &[f64], frames: &[SpectralFrame], energy: &Energy, floor: f64) -> Vec<Onset> {
    let mut onsets: Vec<Onset> = Vec::new();
    let hop_secs = hop_of(energy.sample_rate) as f64 / energy.sample_rate;
    let window = (ADAPTIVE_WINDOW_SECS / hop_secs).round() as usize;
    let reach = (MIN_ONSET_GAP_SECS / 2.0 / hop_secs).round().max(1.0) as usize;
    for i in 0..flux.len() {
        let lo = i.saturating_sub(window);
        let hi = (i + window + 1).min(flux.len());
        let local = &flux[lo..hi];
        let mean = local.iter().sum::<f64>() / local.len() as f64;
        let threshold = THRESHOLD_DELTA + THRESHOLD_MULTIPLIER * mean;
        let near = &flux[i.saturating_sub(reach)..(i + reach + 1).min(flux.len())];
        if flux[i] <= threshold || flux[i] < floor || near.iter().any(|&v| v > flux[i]) {
            continue;
        }
        let Some(t) = energy.strike(frames[i].t_secs, frames[i].span_secs) else {
            continue;
        };
        if onsets
            .last()
            .is_some_and(|o| t - o.t_secs < MIN_ONSET_GAP_SECS)
        {
            continue;
        }
        onsets.push(Onset {
            t_secs: t,
            strength: flux[i],
        });
    }
    onsets
}

struct Energy {
    running: Vec<f64>,
    sample_rate: f64,
    start_secs: f64,
    reach: usize,
    floor: f64,
}

impl Energy {
    fn of(planes: &[&[f64]], sample_rate: f64, start_secs: f64) -> Energy {
        let len = planes.iter().map(|p| p.len()).min().unwrap_or(0);
        let mut running = Vec::with_capacity(len + 1);
        running.push(0.0);
        for n in 0..len {
            let held = running[running.len() - 1];
            running.push(held + planes.iter().map(|p| p[n] * p[n]).sum::<f64>());
        }
        let reach = (STRIKE_SPAN_SECS * sample_rate).round().max(1.0) as usize;
        let loudest = (0..len)
            .map(|n| running[(n + reach).min(len)] - running[n])
            .fold(0.0, f64::max);
        Energy {
            running,
            sample_rate,
            start_secs,
            reach,
            floor: LEVEL_FRACTION * LEVEL_FRACTION * loudest,
        }
    }

    /// The energy of `[from, to)`, silence outside the buffer.
    fn over(&self, from: usize, to: usize) -> f64 {
        let last = self.running.len() - 1;
        self.running[to.min(last)] - self.running[from.min(last)]
    }

    /// The sample whose energy after most outweighs the energy before it and the floor, if
    /// by a strike's jump over that energy carried on as it moved.
    fn strike(&self, from_secs: f64, span_secs: f64) -> Option<f64> {
        let at = |secs: f64| {
            ((secs - self.start_secs) * self.sample_rate)
                .round()
                .max(0.0) as usize
        };
        let (from, to) = (at(from_secs), at(from_secs + span_secs));
        let before = |n: usize| self.over(n.saturating_sub(self.reach), n) + self.floor;
        let mut best: Option<(f64, usize)> = None;
        for n in from..to.min(self.running.len() - 1) {
            let after = self.over(n, n + self.reach);
            if before(n) == 0.0 {
                continue;
            }
            let ratio = after / before(n);
            if best.is_none_or(|(r, _)| ratio >= r) {
                best = Some((ratio, n));
            }
        }
        let (ratio, n) = best?;
        let carried = before(n) / before(n.saturating_sub(self.reach));
        (ratio / carried >= JUMP_ENERGY_RATIO)
            .then(|| self.start_secs + n as f64 / self.sample_rate)
    }
}

/// Whole samples, so an interval on an edge opens its bucket.
fn ioi_histogram(onsets: &[Onset], sample_rate: f64) -> Vec<IoiBucket> {
    let n = (IOI_MAX_SECS * IOI_BUCKETS_PER_SEC).ceil() as usize + 1;
    let mut counts = vec![0usize; n];
    for w in onsets.windows(2) {
        let samples = ((w[1].t_secs - w[0].t_secs) * sample_rate).round();
        let bucket = (samples * IOI_BUCKETS_PER_SEC / sample_rate).floor() as usize;
        counts[bucket.min(n - 1)] += 1;
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(i, count)| IoiBucket {
            lo_secs: i as f64 / IOI_BUCKETS_PER_SEC,
            hi_secs: if i + 1 == n {
                f64::INFINITY
            } else {
                (i + 1) as f64 / IOI_BUCKETS_PER_SEC
            },
            count,
        })
        .collect()
}
