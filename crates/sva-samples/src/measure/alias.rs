// Concern: scores a render against the same one oversampled, as ASR and per-band NMR | Non-concern: producing either render, the plain spectrum (spectrum.rs) | IO: (base, high, k, sr, start) -> Alias

use crate::fft::fft;
use crate::measure::peaks::FLOOR_DB;
use crate::measure::spectrum::{db, starts};
use crate::stft::hann_periodic as hann;

/// Zwicker's critical bands; the last one runs to Nyquist.
const BARK_EDGES: [f64; 25] = [
    0.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0,
    2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0,
    15500.0,
];

/// The spreading triangle is steep down the Bark scale and shallow up it: a masker reaches
/// more than twice as far above itself as below.
const SPREAD_DOWN_DB_PER_BARK: f64 = 27.0;
const SPREAD_UP_DB_PER_BARK: f64 = 12.0;

/// What a full-scale sine reaches the ear at, so quiet has an energy.
pub const PLAYBACK_DB_SPL: f64 = 90.0;

/// 23 ms at 44.1 kHz, the length NMR is published at.
pub const ALIAS_FRAME: usize = 1024;

/// Under it a frame is fade or tail, its NMR two near-silences' ratio.
const GATE_DB: f64 = -70.0;

/// Above it the alias is audible (Brandenburg).
pub const AUDIBLE_NMR_DB: f64 = -10.0;

#[derive(Clone, Debug, PartialEq)]
pub struct AliasBand {
    pub lo_hz: f64,
    pub hi_hz: f64,
    pub signal_db: f64,
    pub alias_db: f64,
    pub nmr_db: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Alias {
    pub oversample: usize,
    pub sample_rate: f64,
    pub frame_size: usize,
    pub frames: usize,
    pub scored_frames: usize,
    pub playback_db_spl: f64,
    pub asr_db: f64,
    pub nmr_db: f64,
    pub nmr_peak_db: f64,
    pub peak_at_secs: f64,
    pub audible: bool,
    pub instances: usize,
    pub bands: Vec<AliasBand>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnderAFrame {
    pub held: usize,
    pub frame: usize,
}

pub fn worst(scored: impl IntoIterator<Item = Alias>) -> Option<Alias> {
    scored
        .into_iter()
        .reduce(|held, next| match next.nmr_peak_db > held.nmr_peak_db {
            true => next,
            false => held,
        })
}

/// The alias is what the base band holds and the `oversample`x render does not. One bin grid
/// for both, so no decimation filter blurs the top octave this is measuring; 0 and Nyquist,
/// with no mirror folded in, weigh half a bin.
pub fn measure_alias(
    base: &[f64],
    high: &[f64],
    oversample: usize,
    sample_rate: f64,
    start_secs: f64,
) -> Result<Alias, UnderAFrame> {
    assert!(
        oversample.is_power_of_two(),
        "oversample must be a power of two so both transforms are radix-2, got {oversample}"
    );
    let frame = ALIAS_FRAME;
    if base.len() < frame {
        return Err(UnderAFrame {
            held: base.len(),
            frame,
        });
    }
    let wide = frame * oversample;
    let bins = frame / 2 + 1;
    let bin_hz = sample_rate / frame as f64;
    let edges = band_edges(sample_rate / 2.0);

    let w_base = hann(frame);
    let w_high = hann(wide);
    let mut sig_bands = vec![0f64; edges.len() - 1];
    let mut err_bands = vec![0f64; edges.len() - 1];
    let mut sig_total = 0f64;
    let mut err_total = 0f64;
    let mut nmr_sum = 0f64;
    let mut nmr_peak = f64::NEG_INFINITY;
    let mut peak_at = start_secs;
    let mut frames = 0usize;
    let mut scored = 0usize;

    let held = starts(base.len(), frame);
    for start in held
        .into_iter()
        .filter(|s| (s + frame) * oversample <= high.len())
    {
        let (re_b, im_b) = transform(base, start, &w_base, 1);
        let (re_h, im_h) = transform(high, start * oversample, &w_high, oversample);
        frames += 1;

        let mut sig = vec![0f64; bins];
        let mut err = vec![0f64; bins];
        for k in 0..bins {
            let weight = match k == 0 || k == bins - 1 {
                true => 0.5,
                false => 1.0,
            };
            sig[k] = weight * (re_h[k] * re_h[k] + im_h[k] * im_h[k]);
            let (dr, di) = (re_b[k] - re_h[k], im_b[k] - im_h[k]);
            err[k] = weight * (dr * dr + di * di);
        }
        sig_total += sig.iter().sum::<f64>();
        err_total += err.iter().sum::<f64>();

        let s = group(&sig, bin_hz, &edges);
        let e = group(&err, bin_hz, &edges);
        for (acc, v) in sig_bands.iter_mut().zip(&s) {
            *acc += v;
        }
        for (acc, v) in err_bands.iter_mut().zip(&e) {
            *acc += v;
        }

        if db(s.iter().sum::<f64>().sqrt()) < GATE_DB {
            continue;
        }
        let ratio = nmr_of(&s, &e, &edges);
        nmr_sum += ratio;
        scored += 1;
        let frame_db = power_db(ratio);
        if frame_db > nmr_peak {
            nmr_peak = frame_db;
            peak_at = start_secs + start as f64 / sample_rate;
        }
    }

    let mean = if scored > 0 {
        nmr_sum / scored as f64
    } else {
        0.0
    };
    let nmr_db = power_db(mean);
    Ok(Alias {
        oversample,
        sample_rate,
        frame_size: frame,
        frames,
        scored_frames: scored,
        playback_db_spl: PLAYBACK_DB_SPL,
        asr_db: power_db(err_total / sig_total.max(f64::MIN_POSITIVE)),
        nmr_db,
        nmr_peak_db: if scored > 0 { nmr_peak } else { nmr_db },
        peak_at_secs: peak_at,
        audible: scored > 0 && nmr_db > AUDIBLE_NMR_DB,
        instances: 0,
        bands: bands(&sig_bands, &err_bands, &edges, frames.max(1)),
    })
}

/// A power ratio in dB on the floor every level here stops at.
fn power_db(ratio: f64) -> f64 {
    match ratio > 0.0 {
        true => (10.0 * libm::log10(ratio)).max(FLOOR_DB),
        false => FLOOR_DB,
    }
}

fn transform(x: &[f64], start: usize, window: &[f64], stride: usize) -> (Vec<f64>, Vec<f64>) {
    let n = window.len();
    let mut re = vec![0f64; n];
    let mut im = vec![0f64; n];
    for (i, w) in window.iter().enumerate() {
        re[i] = x.get(start + i).copied().unwrap_or(0.0) * w;
    }
    fft(&mut re, &mut im);
    let scale = 4.0 / n as f64;
    let keep = n / (2 * stride) + 1;
    re.truncate(keep);
    im.truncate(keep);
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        *r *= scale;
        *i *= scale;
    }
    (re, im)
}

fn band_edges(nyquist: f64) -> Vec<f64> {
    let mut edges: Vec<f64> = BARK_EDGES
        .iter()
        .copied()
        .filter(|e| *e < nyquist)
        .collect();
    edges.push(nyquist);
    edges
}

fn group(power: &[f64], bin_hz: f64, edges: &[f64]) -> Vec<f64> {
    let mut out = vec![0f64; edges.len() - 1];
    for (k, p) in power.iter().enumerate() {
        let hz = k as f64 * bin_hz;
        let b = edges
            .windows(2)
            .position(|w| hz >= w[0] && hz < w[1])
            .unwrap_or(out.len() - 1);
        out[b] += p;
    }
    out
}

/// Every NMR reads this one, so no band disagrees about what masks it. The offset it drops
/// the spread by is our own, standing in for the tonality estimate this makes none of.
fn thresholds(signal: &[f64], edges: &[f64]) -> Vec<f64> {
    (0..signal.len())
        .map(|j| {
            let zj = j as f64 + 0.5;
            let spread: f64 = signal
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let zi = i as f64 + 0.5;
                    let slope = if zj >= zi {
                        SPREAD_UP_DB_PER_BARK
                    } else {
                        SPREAD_DOWN_DB_PER_BARK
                    };
                    s * libm::pow(10f64, -slope * (zj - zi).abs() / 10.0)
                })
                .sum();
            let offset = if zj <= 12.0 { 3.0 } else { 0.25 * zj };
            let centre = (edges[j] + edges[j + 1]) / 2.0;
            (spread * libm::pow(10f64, -offset / 10.0)).max(quiet_energy(centre))
        })
        .collect()
}

fn nmr_of(signal: &[f64], alias: &[f64], edges: &[f64]) -> f64 {
    let masked = thresholds(signal, edges);
    alias.iter().zip(&masked).map(|(a, m)| a / m).sum::<f64>() / signal.len() as f64
}

/// Terhardt's threshold in quiet, in a band energy's own units.
fn quiet_energy(hz: f64) -> f64 {
    let f = (hz / 1000.0).max(0.02);
    let db_spl = 3.64 * libm::pow(f, -0.8) - 6.5 * libm::exp(-0.6 * (f - 3.3) * (f - 3.3))
        + 0.001 * f.powi(4);
    libm::pow(10f64, (db_spl - PLAYBACK_DB_SPL) / 10.0)
}

fn bands(signal: &[f64], alias: &[f64], edges: &[f64], frames: usize) -> Vec<AliasBand> {
    let mean: Vec<f64> = signal.iter().map(|s| s / frames as f64).collect();
    let masked = thresholds(&mean, edges);
    (0..mean.len())
        .map(|j| {
            let a = alias[j] / frames as f64;
            AliasBand {
                lo_hz: edges[j],
                hi_hz: edges[j + 1],
                signal_db: db(mean[j].sqrt()),
                alias_db: db(a.sqrt()),
                nmr_db: power_db(a / masked[j]),
            }
        })
        .collect()
}
