// Concern: the analysis a fit reads: the spectrogram, the floor a sine is heard over, the harmonic sieve | Non-concern: choosing sources (fit/) | IO: (samples) -> Spectrogram; (peaks) -> f0, members

use std::f64::consts::TAU;
use std::num::NonZeroUsize;

use sva_engine::hearing::{
    CAM_FIRST, CAM_LAST, HOP_SECS, WINDOW_SECS, cam_hz, ear_gain_db, threshold_excitation_db,
};
use sva_engine::{each, fft};

use crate::sources::{Sound, Source, Trajectory};

pub const MODEL: &str = "the analysis: a periodic Hann window as long as the ear's longest at \
     s = 1 (64 ms), its DFT zero-padded to the next power of two, a frame every 1 ms (the ear's \
     hop at s = 1), each bin's power |X|^2; a sine at f is heard where its level in dBFS plus \
     the SPL of full scale plus the ear's gain reaches the threshold excitation at f \
     (hearing.rs), that level its floor; peaks group into a harmonic by a harmonic sieve \
     (Duifhuis, Willems and Sluyter 1982, JASA 71(6) 1568-1580): each peak's Hz over k = 1, \
     2, ... down to the feasible set's lowest is a trial fundamental, a peak its member at \
     harmonic n = round(hz/f0) where it lies in the analysis bin centred on n f0, the nearest \
     per n, the trial scoring the numbers it saves, its members' as partials less its own, \
     where positive and its members outnumber the harmonics up to its highest it lacks, the \
     fundamental its strongest member's Hz over its harmonic number, the highest scoring, \
     ties to the higher fundamental";

/// Power per bin `0..=n/2` of each frame every `hop` samples, centred on the frame's sample,
/// each frame computed where it is read, so no clip holds every frame at once.
pub struct Spectrogram {
    pub(crate) rate: f64,
    pub(crate) n: usize,
    pub(crate) long: usize,
    pub(crate) hop: usize,
    pub(crate) sum_w: f64,
    pub(crate) sum_w2: f64,
    window: Vec<f64>,
    twiddle: Vec<(f64, f64)>,
    samples: Vec<f64>,
}

pub(crate) fn hann(long: usize) -> Vec<f64> {
    (0..long)
        .map(|k| 0.5 - 0.5 * libm::cos(TAU * k as f64 / long as f64))
        .collect()
}

impl Spectrogram {
    pub fn of(samples: &[f64], rate: u32) -> Spectrogram {
        let rate_hz = f64::from(rate);
        let long = libm::round(WINDOW_SECS[0] * rate_hz) as usize;
        let window = hann(long);
        let n = long.next_power_of_two();
        let hop = (libm::round(HOP_SECS * rate_hz) as usize).max(1);
        Spectrogram {
            rate: rate_hz,
            n,
            long,
            hop,
            sum_w: window.iter().sum(),
            sum_w2: window.iter().map(|w| w * w).sum(),
            twiddle: fft::twiddles(n),
            window,
            samples: samples.to_vec(),
        }
    }

    /// How many frames it holds.
    pub(crate) fn frames(&self) -> usize {
        self.samples.len() / self.hop + 1
    }

    /// Frame `m`'s power per bin.
    pub(crate) fn power(&self, m: usize) -> Vec<f64> {
        let n = self.n;
        let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
        let from = (m * self.hop) as i64 - (self.long / 2) as i64;
        for (k, w) in self.window.iter().enumerate() {
            let at = from + k as i64;
            if at >= 0 && (at as usize) < self.samples.len() {
                re[k] = w * self.samples[at as usize];
            }
        }
        fft::fft_with(&mut re, &mut im, &self.twiddle);
        (0..=n / 2).map(|b| re[b] * re[b] + im[b] * im[b]).collect()
    }
}

/// The lowest level in dBFS at which a sine at `hz` reaches threshold, as MODEL states.
pub fn floor_db(hz: f64, spl_db: f64) -> f64 {
    threshold_excitation_db(hz) - ear_gain_db(hz) - spl_db
}

/// The Hz the feasible set spans at `rate`.
pub(crate) fn span_hz(rate: f64) -> (f64, f64) {
    (cam_hz(CAM_FIRST), cam_hz(CAM_LAST).min(rate / 2.0))
}

/// A peak the residual holds: its Hz, start and end.
pub(crate) struct Track {
    pub(crate) hz: f64,
    pub(crate) span: (f64, f64),
}

impl Track {
    fn meets(&self, other: &Track) -> bool {
        self.span.0 < other.span.1 && other.span.0 < self.span.1
    }
}

/// The numbers a fixed source of `harmonics` harmonics with one level knot writes, as born; a
/// partial's at 1.
fn written(harmonics: usize) -> usize {
    let level = Trajectory {
        base: 0.0,
        knots: vec![(0.0, 0.0)],
    };
    let sound = match harmonics {
        1 => Sound::Partial {
            hz: 0.0,
            glide: 0.0,
        },
        _ => Sound::Harmonic {
            hz: 0.0,
            stretch: 0.0,
            glide: 0.0,
        },
    };
    Source::new(0, 0.0, sound, vec![level; harmonics])
        .values()
        .len()
}

/// A harmonic's members as (track, harmonic number).
pub(crate) type Members = Vec<(usize, usize)>;

/// The fundamental whose sieve over `pool` scores most, as `MODEL` states, beside its members
/// as (track, harmonic number), on at most `threads`; `None` where no fundamental saves numbers.
pub(crate) fn sieved(
    tracks: &[Track],
    pool: &[usize],
    analysis_hz: f64,
    (lo, hi): (f64, f64),
    threads: NonZeroUsize,
) -> Option<(f64, Members)> {
    let better = |best: &Option<(usize, f64, Members)>, score: usize, f0: f64| {
        best.as_ref()
            .is_none_or(|(s, f, _)| score > *s || (score == *s && f0 > *f))
    };
    let each_anchor = each(threads, pool.to_vec(), |p| {
        let mut best: Option<(usize, f64, Members)> = None;
        let anchor = &tracks[p];
        let met: Vec<usize> = pool
            .iter()
            .copied()
            .filter(|q| tracks[*q].meets(anchor))
            .collect();
        for k in 1.. {
            let f0 = anchor.hz / f64::from(k);
            if f0 < lo {
                break;
            }
            let mut by_number: Vec<Option<(usize, f64)>> = Vec::new();
            for &q in &met {
                let t = &tracks[q];
                let n = libm::round(t.hz / f0);
                let miss = (t.hz - n * f0).abs();
                if n < 1.0 || 2.0 * miss > analysis_hz {
                    continue;
                }
                let n = n as usize;
                if by_number.len() < n {
                    by_number.resize(n, None);
                }
                if by_number[n - 1].is_none_or(|(_, m)| miss < m) {
                    by_number[n - 1] = Some((q, miss));
                }
            }
            let members: Members = by_number
                .iter()
                .enumerate()
                .filter_map(|(i, m)| m.map(|(q, _)| (q, i + 1)))
                .collect();
            let (saved, own) = (members.len() * written(1), written(by_number.len()));
            if saved <= own || 2 * members.len() <= by_number.len() {
                continue;
            }
            let score = saved - own;
            let &(lead, n) = members.iter().min().expect("a member");
            let f0 = tracks[lead].hz / n as f64;
            if f0 < lo || f0 * by_number.len() as f64 > hi {
                continue;
            }
            if better(&best, score, f0) {
                best = Some((score, f0, members));
            }
        }
        best
    });
    let mut best: Option<(usize, f64, Members)> = None;
    for (score, f0, members) in each_anchor.into_iter().flatten() {
        if better(&best, score, f0) {
            best = Some((score, f0, members));
        }
    }
    best.map(|(_, f0, members)| (f0, members))
}
