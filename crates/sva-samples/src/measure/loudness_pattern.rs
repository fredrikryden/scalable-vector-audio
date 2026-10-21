// Concern: specific loudness per auditory band per frame, masking spread in | Non-concern: BS.1770 loudness (loudness.rs), comparing two (sva-core) | IO: (planes, rate, start, hop, spl) -> pattern

use std::f64::consts::TAU;

use super::bands::{BAND_COUNT, FIRST_CAM, erb_hz};
use crate::fft::fft;

pub const MODEL: &str = "specific loudness after Moore, Glasberg and Baer (1997), J. Audio Eng. \
                         Soc. 45(4) 224-240, per frame: a periodic Hann of 2048 samples at 48 kHz \
                         (its length scaled to the rate) centred on each frame time, every \
                         channel's power per bin averaged, a full-scale sine at `spl_db`; the \
                         outer and middle ear as Terhardt's (1979) threshold in quiet, 3.64 \
                         f^-0.8 - 6.5 e^(-0.6 (f - 3.3)^2) + 0.001 f^4 dB (f in kHz), taken \
                         against its value at 1 kHz; each band's excitation the power through a \
                         rounded-exponential auditory filter (1 + p g) e^(-p g), g the \
                         frequency's distance from the centre over the centre, p = 4 fc / \
                         ERB(fc) above it and p - 0.35 (p / p(1 kHz)) (X - 51) below it, X the \
                         level in dB SPL within one ERB of the component, its upward spread of \
                         masking; N' = C ((E + A)^0.2 - A^0.2), A twice the excitation at \
                         threshold, times (2 E / (E + E_T))^1.5 under threshold E_T, C so a 1 \
                         kHz sine at 40 dB SPL is 1 sone; the total the sum over bands 1 Cam \
                         apart. A cell at or under `threshold_sone` is inaudible: a comparison \
                         counts it zero where both sides are. Measured, not ISO 532-2: no \
                         binaural inhibition and no temporal integration, and the threshold in \
                         quiet stands for the low-frequency internal noise.";

const WINDOW_SAMPLES_48K: f64 = 2048.0;
pub const DEFAULT_SPL_DB: f64 = 100.0;
const ALPHA: f64 = 0.2;
const LOWER_SLOPE_DB: f64 = 51.0;
const LOWER_SLOPE: f64 = 0.35;
const LEAST_LOWER_SHARE: f64 = 0.1;
const CALIBRATION_HZ: f64 = 1000.0;
const CALIBRATION_DB: f64 = 40.0;

#[derive(Clone, Debug, PartialEq)]
pub struct PatternFrame {
    pub t_secs: f64,
    pub total_sone: f64,
    /// Sone per Cam, band by band.
    pub specific: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoudnessPattern {
    pub frame_secs: f64,
    pub window_secs: f64,
    pub spl_db: f64,
    pub centres_hz: Vec<f64>,
    pub threshold_sone: f64,
    pub frames: Vec<PatternFrame>,
}

/// Terhardt's threshold in quiet, dB SPL.
fn quiet_db(hz: f64) -> f64 {
    let f = hz.max(20.0) / 1000.0;
    3.64 * libm::pow(f, -0.8) - 6.5 * sva_formula::exp(-0.6 * (f - 3.3) * (f - 3.3))
        + 1e-3 * f * f * f * f
}

/// `bands`' Cam scale inverted, its power read through libm so it is one value native and in
/// wasm.
fn centre_hz(cam: f64) -> f64 {
    (libm::pow(10.0, cam / 21.4) - 1.0) * 1000.0 / 4.37
}

fn intensity(db: f64) -> f64 {
    libm::pow(10.0, db / 10.0)
}

struct Ear {
    centres: Vec<f64>,
    threshold: f64,
    a: f64,
    c: f64,
}

impl Ear {
    fn new(sr: f64) -> Ear {
        let centres: Vec<f64> = (0..BAND_COUNT)
            .map(|k| centre_hz(FIRST_CAM + k as f64))
            .filter(|hz| *hz < sr / 2.0)
            .collect();
        let threshold = intensity(quiet_db(CALIBRATION_HZ));
        let mut ear = Ear {
            centres,
            threshold,
            a: 2.0 * threshold,
            c: 1.0,
        };
        let tone = [(CALIBRATION_HZ, intensity(CALIBRATION_DB))];
        let total: f64 = ear
            .specific(&ear.excitation(&tone, &[CALIBRATION_DB]))
            .iter()
            .sum();
        ear.c = 1.0 / total;
        ear
    }

    /// Each band's excitation off components (hz, intensity at the eardrum), each with the
    /// level per ERB about it.
    fn excitation(&self, parts: &[(f64, f64)], per_erb_db: &[f64]) -> Vec<f64> {
        let p_1k = 4.0 * 1000.0 / erb_hz(1000.0);
        self.centres
            .iter()
            .map(|&fc| {
                let p = 4.0 * fc / erb_hz(fc);
                parts
                    .iter()
                    .zip(per_erb_db)
                    .map(|(&(hz, i), &x)| {
                        let g = (hz - fc).abs() / fc;
                        let slope = match hz < fc {
                            true => (p - LOWER_SLOPE * (p / p_1k) * (x - LOWER_SLOPE_DB))
                                .max(LEAST_LOWER_SHARE * p),
                            false => p,
                        };
                        let pg = slope * g;
                        i * (1.0 + pg) * sva_formula::exp(-pg)
                    })
                    .sum()
            })
            .collect()
    }

    fn specific(&self, excitation: &[f64]) -> Vec<f64> {
        excitation
            .iter()
            .map(|&e| {
                let grown = libm::pow(e + self.a, ALPHA) - libm::pow(self.a, ALPHA);
                let near = match e < self.threshold {
                    true => libm::pow(2.0 * e / (e + self.threshold), 1.5),
                    false => 1.0,
                };
                self.c * near * grown
            })
            .collect()
    }

    fn at_threshold(&self) -> f64 {
        self.specific(&[self.threshold])[0]
    }
}

/// Every frame centred on `start + k hop`, the target zero past its ends.
pub fn measure(
    planes: &[&[f64]],
    sr: f64,
    start: f64,
    hop_secs: f64,
    spl_db: f64,
) -> LoudnessPattern {
    let len = super::held(planes);
    let ear = Ear::new(sr);
    let n = super::scaled_pow2(WINDOW_SAMPLES_48K, sr);
    let hop = ((hop_secs * sr).round() as usize).max(1);
    let w: Vec<f64> = (0..n)
        .map(|k| 0.5 - 0.5 * libm::cos(TAU * k as f64 / n as f64))
        .collect();
    let norm = 2.0 / (n as f64 * n as f64 * 0.375) / 0.5 * intensity(spl_db);
    let bin_hz = sr / n as f64;
    let weights: Vec<f64> = (0..=n / 2)
        .map(|k| intensity(quiet_db(CALIBRATION_HZ) - quiet_db(k as f64 * bin_hz)))
        .collect();
    let frames = (0..=len.saturating_sub(1) / hop)
        .map(|f| {
            let centre = (f * hop) as i64;
            let mut power = vec![0.0; n / 2 + 1];
            for plane in planes {
                let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
                for (i, wi) in w.iter().enumerate() {
                    let at = centre + i as i64 - (n / 2) as i64;
                    if at >= 0 && (at as usize) < len {
                        re[i] = plane[at as usize] * wi;
                    }
                }
                fft(&mut re, &mut im);
                for k in 0..=n / 2 {
                    power[k] += (re[k] * re[k] + im[k] * im[k]) / planes.len() as f64;
                }
            }
            let parts: Vec<(f64, f64)> = (1..n / 2)
                .map(|k| (k as f64 * bin_hz, power[k] * norm * weights[k]))
                .collect();
            let mut running = vec![0.0; parts.len() + 1];
            for (i, p) in parts.iter().enumerate() {
                running[i + 1] = running[i] + p.1;
            }
            let per_erb: Vec<f64> = parts
                .iter()
                .map(|&(hz, _)| {
                    let half = erb_hz(hz) / 2.0;
                    let first = ((hz - half) / bin_hz).ceil().max(1.0) as usize;
                    let last = (((hz + half) / bin_hz).floor() as usize).min(parts.len());
                    let held = running[last] - running[(first - 1).min(last)];
                    10.0 * libm::log10(held + 1e-30)
                })
                .collect();
            let specific = ear.specific(&ear.excitation(&parts, &per_erb));
            PatternFrame {
                t_secs: start + (f * hop) as f64 / sr,
                total_sone: specific.iter().sum(),
                specific,
            }
        })
        .collect();
    LoudnessPattern {
        frame_secs: hop as f64 / sr,
        window_secs: n as f64 / sr,
        spl_db,
        threshold_sone: ear.at_threshold(),
        centres_hz: ear.centres,
        frames,
    }
}
