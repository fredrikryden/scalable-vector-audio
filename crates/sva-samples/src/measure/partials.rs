// Concern: tracks each partial as frequency and amplitude knot tables, tonal or breathy, with its residual | Non-concern: naming notes (pitch.rs) | IO: (planes, rate, start, Asked) -> Partials

use std::collections::VecDeque;
use std::f64::consts::{FRAC_PI_2, PI, SQRT_2, TAU};

use super::peaks;
use crate::fft::{fft_with, ifft_with, twiddles};
use crate::stft::hann_periodic;

/// Two peaks, or two fragments, are one partial within this many cents or `NEAR_HZ`.
pub const NEAR_CENTS: f64 = 25.0;
pub const NEAR_HZ: f64 = 8.0;
/// A peak counts where it stands this far over the median bin within a sixth of an octave.
pub const PROMINENCE_DB: f64 = 6.0;
/// A peak stands this many times over the Hann sidelobes louder peaks put under it: a
/// partial that decays or swells within a frame raises its skirts over a steady one's.
pub const LEAK_MARGIN: f64 = 4.0;
/// The demodulating Gaussian's half-width: this share of the gap to the nearest track, clamped.
pub const GAP_SHARE: f64 = 0.4;
pub const RESOLUTION_HZ: (f64, f64) = (6.0, 40.0);
/// Groups closer than this share one band at the narrowest Gaussian, and are one track.
pub const LEAST_GAP_HZ: f64 = RESOLUTION_HZ.0 / GAP_SHARE;
/// A hop's frequency is the band's mean phase slope over this many Gaussian time deviations:
/// a change faster than the Gaussian's width is not in the band to be read.
pub const SLOPE_SIGMAS: f64 = 4.0;
/// A hop's frequency is read only where its slope's span lies this many Gaussian time
/// deviations inside the target, the band there untouched by its ends, and held outward.
pub const EDGE_SIGMAS: f64 = 3.0;
/// A phase slope past this many half-widths from the detected frequency reads a neighbour
/// leaking into a band its own track has left silent, and is not the track's.
pub const CAPTURE_SIGMAS: f64 = 2.0;
/// Jacobi passes against the residual: each cancels what the last left of every tonal
/// neighbour's band in a track's own, the neighbours' leak back into it a factor of their
/// Gaussian's weight at the gap smaller each pass.
pub const CANCELLATIONS: usize = 2;
/// A frequency knot table stays this close to the measured curve, in cents, or within twice
/// the curve's own noise where that is wider.
pub const KNOT_CENTS: f64 = 1.0;
/// An amplitude knot table stays within this share of the track's peak, or the floor if higher.
pub const KNOT_SHARE: f64 = 0.02;
/// Flutter: the dB spread of `FLUTTER_FRAME_SECS` means about their `FLUTTER_MEAN_FRAMES`
/// running mean over the loudest `FLUTTER_SPAN_SECS`; over `BREATHY_FLUTTER_DB` is breathy.
pub const FLUTTER_FRAME_SECS: f64 = 0.002;
pub const FLUTTER_MEAN_FRAMES: usize = 15;
pub const FLUTTER_SPAN_SECS: f64 = 0.2;
pub const BREATHY_FLUTTER_DB: f64 = 2.0;
/// Coherence: the band's correlation with itself `COHERENCE_LAG_SECS` on, over the loudest
/// `COHERENCE_SPAN_SECS`, the band at least `COHERENCE_HALF_HZ` either side.
pub const COHERENCE_LAG_SECS: f64 = 0.01;
pub const COHERENCE_SPAN_SECS: f64 = 0.25;
pub const COHERENCE_HALF_HZ: f64 = 15.0;
/// The cancelled reads take a track's flags at this half-width at least: a narrowband noise's
/// envelope under a narrower Gaussian moves slower than the flutter's 30 ms running mean.
pub const FLAG_LEAST_HZ: f64 = 20.0;
/// A breathy track's level is its band's rms over this span.
pub const BREATHY_POWER_SECS: f64 = 0.03;
/// The decimated band is read at least this often.
const LEAST_BAND_RATE_HZ: f64 = 1000.0;
/// Past this many Gaussian deviations a weight is under e^-18 and is left out.
const SIGMAS: f64 = 6.0;
/// Zero padding, in the narrowest Gaussian's time deviations, so no band wraps onto the target.
const PAD_SIGMAS: f64 = 12.0;
/// A sixth of an octave either side, and this many bins more, is a peak's neighbourhood.
const PROMINENCE_BINS: f64 = 4.0;

pub const MODEL: &str = "heterodyne partial tracking, as the fit harness's analysis seed \
                         tracks, read without a score. Detect: every `hop_secs` from the \
                         target's start, the power of every periodic Hann frame of `frame_secs` \
                         centred within `window_secs` of samples (Welch, P. D. (1967), The use \
                         of fast Fourier transform for the estimation of power spectra, IEEE \
                         Trans. Audio Electroacoust. 15(2) 70-73), a frame past the target's \
                         ends spanning the samples it holds, zero-padded to at least its own \
                         length, over every channel's samples averaged, those frames' power \
                         averaged so a noise's chance peaks average away and a held partial's \
                         stay; its peaks the local maxima at or above the floor, in dB against \
                         full scale, standing 4 times over the Hann sidelobes the louder peaks \
                         kept and a DC put under them (`spectrum`'s bound, Harris 1978, with \
                         that margin for a partial that moves within a frame), 6 dB over the \
                         median bin within a sixth of an octave and 4 bins, and 2 frame bins \
                         from 0 Hz and Nyquist. Peaks link hop to hop into fragments, loudest \
                         first, each to the nearest open fragment within 25 cents or 8 Hz, a \
                         fragment open across gaps of half a frame. Fragments sorted by \
                         energy-weighted frequency join a group while within 25 cents or 8 Hz \
                         of its running mean, or 15 Hz, where the narrowest Gaussian below \
                         reads both as one band (one partial across notes; a glide is a group \
                         whose peaks move, which a window longer than the time it takes to \
                         cross a frame's resolution smears). The `peaks` groups of most energy \
                         are tracks; `detected` counts them all. Read: per track the target's \
                         analytic signal demodulated along the track's phase 2 pi \
                         pwl_integral(t, frequency), then a Gaussian lowpass of half-width \
                         sigma = `resolution_hz`, 0.4 of the gap to the nearest track's \
                         frequency clamped to 6-40 Hz, its time deviation `resolution_secs` = 1 \
                         / (2 pi sigma); all of it off one transform of the target zero-padded \
                         by 12 time deviations of the narrowest Gaussian, each band taken flat \
                         to 240 Hz past the track's frequencies, Gaussian past that, and \
                         decimated to at least its width, so the band is exact for every \
                         lowpass here. For a lone sinusoid of constant frequency and amplitude \
                         that band is a e^(i(phase - pi/2)) exactly, wherever the target \
                         reaches past the Gaussian's tails, about 6 time deviations: its \
                         per-hop frequency, amplitude and phase are exact there, and a \
                         neighbour d Hz away moves the band by at most its amplitude times \
                         e^(-d^2 / (2 sigma^2)). Frequency: the group's peaks per hop, plus the \
                         band's mean phase slope over 2 pi across 4 time deviations about the \
                         hop, where the group holds a peak, the band is over the floor, that \
                         span lies 3 time deviations inside the target (anywhere it sounds, \
                         where no hop does) and the slope is within 2 sigma and lands nearer \
                         the track than any other group's peak there, else the band reads a \
                         neighbour's frequency and not its own; held between and outward. \
                         Cancel: then twice, every tonal track's band, Re z(t) e^(i 2 pi \
                         pwl_integral(t)), is taken from the target and each track read again \
                         off that residual plus its own band (Jacobi), so what a tonal \
                         neighbour leaks into a track's band is cancelled, the second pass \
                         cancelling what the first took of the track's own leak into the \
                         neighbour's; a breathy track stays in the residual. Tonal or breathy, \
                         as the harness read them: flutter is the dB spread of 2 ms means of \
                         the band's modulus about their 15-frame running mean, the 7 frames at \
                         either end left out, over the 200 ms from the track's loudest hop, or \
                         the target's last 200 ms where that is later; coherence |sum b(t)* b(t \
                         + 10 ms)| / sqrt(sum |b(t)|^2 sum |b(t + 10 ms)|^2) of the band \
                         brickwalled to max(sigma, 15 Hz) either side over the 250 ms so \
                         placed, 1 under three lags; the first read takes both at sigma, the \
                         cancelled reads at sigma or 20 Hz if wider, where a narrowband noise's \
                         envelope moves faster than the 30 ms running mean and a tone's, its \
                         tonal neighbours cancelled, holds. A tone's modulus holds over 2 ms \
                         frames and a narrowband noise's flutters (Rayleigh, some 5.6 dB): over \
                         2 dB is breathy, its frequency the energy-weighted mean, its \
                         `bandwidth_hz` the larger of -ln(coherence) / (pi 10 ms), a Lorentzian \
                         line's width, and 2 sigma / 3. A tonal track's amplitude is the band's \
                         modulus, the sinusoid's peak amplitude, and `phase` is the \
                         energy-weighted mean of its argument plus pi/2, so amplitude sin(2 pi \
                         pwl_integral(t, frequency) + phase) is the track; a breathy track's \
                         amplitude is its band's rms over 30 ms, and it has no phase. Knots: \
                         per hop an amplitude under the floor is zero, and each table is the \
                         fewest hops, greedily, through which straight lines stay within the \
                         tolerance of the per-hop curve: the amplitude within 2% of the track's \
                         peak or the floor's amplitude, from the track's first sounding hop \
                         less one to its last plus one; the frequency within 1 cent, or twice \
                         its own noise where wider (the median absolute second difference over \
                         0.6745 sqrt 6), where the track sounds, its knots then refit by least \
                         squares weighted by the band's energy. `error` is the largest gap a \
                         table leaves where it is held to the curve. The frequency table spans \
                         the whole target from its first hop, so pwl_integral's phase starts \
                         there. Bounds: `residual_db` is the energy the knot tables leave in a \
                         tonal track's band, the target less every tonal track's tables \
                         demodulated as above, over the energy its band holds, over the \
                         amplitude table's span, the whole of what the track misses there, \
                         noise, a neighbour's error and its own; null for a breathy track, a \
                         band whose waveform no table claims. Before its group's first peak and \
                         after its last a track sounds only within a window and half a frame of \
                         them, and only where its band holds over twice what any other group's \
                         peak there puts through its Gaussian, amp e^(-d^2 / (2 sigma^2)) at \
                         the gap d: a track with no peaks of its own never reads a louder \
                         neighbour filling its band as a second synthesis of it; \
                         `unclaimed_db` is its band's energy over the floor at the hops so \
                         ruled out, over the energy at the hops it keeps, null where none is. \
                         The \
                         target's `residual_db` is its energy less the tonal tracks' tables' \
                         over its energy, breathy energy left in.";

/// What a caller asks of `measure`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Asked {
    pub frame_secs: f64,
    /// The span of samples whose frames' power each hop averages.
    pub window_secs: f64,
    pub hop_secs: f64,
    pub floor_db: f64,
    pub tracks: usize,
}

/// A knot table: `pwl(t, times[0], values[0], ...)`; `error` the most it leaves the curve.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    pub error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Tonal,
    Breathy,
}

impl Kind {
    pub fn key(self) -> &'static str {
        match self {
            Kind::Tonal => "tonal",
            Kind::Breathy => "breathy",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub kind: Kind,
    /// Energy-weighted over the hops it sounds.
    pub hz: f64,
    pub frequency: Curve,
    pub amplitude: Curve,
    pub phase: Option<f64>,
    pub flutter_db: f64,
    pub coherence: f64,
    pub bandwidth_hz: Option<f64>,
    pub resolution_hz: f64,
    pub resolution_secs: f64,
    pub residual_db: Option<f64>,
    /// The band's energy at hops ruled a louder neighbour's, over what the track keeps.
    pub unclaimed_db: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Partials {
    pub frame_secs: f64,
    pub window_secs: f64,
    pub hop_secs: f64,
    pub floor_db: f64,
    pub detected: usize,
    pub residual_db: Option<f64>,
    pub tracks: Vec<Track>,
}

/// Whole samples: a frame of at least 2, a hop of at least 1, a window of whole hops of frames.
pub fn measure(planes: &[&[f64]], sr: f64, start: f64, asked: Asked) -> Partials {
    let x = mono(planes);
    let frame = ((asked.frame_secs * sr).round() as usize).max(2);
    let hop = ((asked.hop_secs * sr).round() as usize).max(1);
    let reach = ((asked.window_secs * sr - frame as f64).max(0.0) / (2 * hop) as f64).round();
    let reach = reach as usize;
    let floor = libm::pow(10.0, asked.floor_db / 20.0);
    let grid = Grid {
        sr,
        samples: x.len(),
        hop,
        hops: x.len().saturating_sub(1).div_ceil(hop) + 1,
        start,
    };
    let peaks = detect(&x, &grid, (frame, reach), (floor, asked.tracks));
    let mut groups = group(link(&peaks, (frame / (2 * hop)).max(1)));
    let detected = groups.len();
    groups.sort_by(|a, b| b.energy.total_cmp(&a.energy).then(a.hz.total_cmp(&b.hz)));
    groups.truncate(asked.tracks);
    groups.sort_by(|a, b| a.hz.total_cmp(&b.hz));
    let (tracks, residual_db) = read(&x, &grid, (&groups, reach + frame.div_ceil(2 * hop)), floor);
    Partials {
        frame_secs: frame as f64 / sr,
        window_secs: (2 * reach * hop + frame) as f64 / sr,
        hop_secs: hop as f64 / sr,
        floor_db: asked.floor_db,
        detected,
        residual_db,
        tracks,
    }
}

fn mono(planes: &[&[f64]]) -> Vec<f64> {
    let len = super::held(planes);
    let count = planes.len() as f64;
    (0..len)
        .map(|n| planes.iter().map(|p| p[n]).sum::<f64>() / count)
        .collect()
}

struct Grid {
    sr: f64,
    /// The target's own samples.
    samples: usize,
    hop: usize,
    hops: usize,
    start: f64,
}

impl Grid {
    fn secs(&self, k: usize) -> f64 {
        (k * self.hop) as f64 / self.sr
    }
}

/// Transforms of one length, its twiddles computed once.
struct Plan {
    twiddle: Vec<(f64, f64)>,
}

impl Plan {
    fn new(n: usize) -> Plan {
        Plan {
            twiddle: twiddles(n),
        }
    }

    fn forward(&self, re: &mut [f64], im: &mut [f64]) {
        fft_with(re, im, &self.twiddle);
    }

    fn inverse(&self, re: &mut [f64], im: &mut [f64]) {
        ifft_with(re, im, &self.twiddle);
    }
}

#[derive(Clone, Copy)]
struct Found {
    hz: f64,
    amp: f64,
}

/// One frame's power per bin, a sinusoid's amplitude squared at its own bin, and the
/// samples its Hann spans; `None` where it spans under two.
type Power = Option<(Vec<f64>, usize)>;

/// Each hop's peaks off the power of every frame centred within `reach` hops, averaged.
fn detect(
    x: &[f64],
    grid: &Grid,
    (frame, reach): (usize, usize),
    (floor, cap): (f64, usize),
) -> Vec<Vec<Found>> {
    let size = frame.next_power_of_two();
    let plan = Plan::new(size);
    let bin_hz = grid.sr / size as f64;
    let span = |k: usize| {
        let centre = k * grid.hop;
        let from = centre.saturating_sub(frame / 2).min(x.len());
        (from, (centre + frame.div_ceil(2)).min(x.len()) - from)
    };
    let (mut re, mut im) = (vec![0.0; size], vec![0.0; size]);
    let mut pair = |k: usize| -> (Power, Power) {
        re.iter_mut().for_each(|v| *v = 0.0);
        im.iter_mut().for_each(|v| *v = 0.0);
        let ((a, na), (b, nb)) = (span(k), span(k + 1));
        if na >= 2 {
            for (i, w) in hann_periodic(na).iter().enumerate() {
                re[i] = x[a + i] * w;
            }
        }
        if k + 1 < grid.hops && nb >= 2 {
            for (i, w) in hann_periodic(nb).iter().enumerate() {
                im[i] = x[b + i] * w;
            }
        }
        plan.forward(&mut re, &mut im);
        let unpacked = |n: usize, second: bool| {
            (n >= 2).then(|| {
                let scale = (4.0 / n as f64) * (4.0 / n as f64);
                let bins = (0..=size / 2).map(|j| {
                    let (zr, zi, wr, wi) =
                        (re[j], im[j], re[(size - j) % size], im[(size - j) % size]);
                    let (pr, pi) = match second {
                        false => (zr + wr, zi - wi),
                        true => (zi + wi, wr - zr),
                    };
                    (pr * pr + pi * pi) * 0.25 * scale
                });
                (bins.collect(), n)
            })
        };
        let second = (k + 1 < grid.hops).then(|| unpacked(nb, true)).flatten();
        (unpacked(na, false), second)
    };
    let mut held: VecDeque<(usize, Power)> = VecDeque::new();
    let mut spare: Option<Power> = None;
    let mut sum = vec![0.0; size / 2 + 1];
    let (mut count, mut next) = (0usize, 0usize);
    let mut out = Vec::with_capacity(grid.hops);
    for k in 0..grid.hops {
        while next < grid.hops && next <= k + reach {
            let p = match spare.take() {
                Some(p) => p,
                None => {
                    let (first, second) = pair(next);
                    spare = Some(second);
                    first
                }
            };
            if let Some((p, _)) = &p {
                sum.iter_mut().zip(p).for_each(|(a, b)| *a += b);
                count += 1;
            }
            held.push_back((next, p));
            next += 1;
        }
        while held.front().is_some_and(|(j, _)| j + reach < k) {
            if let Some((_, Some((p, _)))) = held.pop_front() {
                sum.iter_mut().zip(&p).for_each(|(a, b)| *a -= b);
                count -= 1;
            }
        }
        let spanned = held
            .iter()
            .filter_map(|(_, p)| p.as_ref().map(|p| p.1))
            .min();
        let (Some(spanned), true) = (spanned, count > 0) else {
            out.push(Vec::new());
            continue;
        };
        let mags: Vec<f64> = sum
            .iter()
            .map(|p| (p.max(0.0) / count as f64).sqrt())
            .collect();
        out.push(picked(&mags, (bin_hz, grid.sr), spanned, (floor, cap)));
    }
    out
}

/// Loudest first, the local maxima over the floor standing `LEAK_MARGIN` over the Hann
/// sidelobes the kept peaks, and a DC bin 0 holds as its maximum, put under them; then the
/// prominent ones 2 frame bins clear of 0 Hz and Nyquist.
fn picked(
    mags: &[f64],
    (bin_hz, sr): (f64, f64),
    spanned: usize,
    (floor, cap): (f64, usize),
) -> Vec<Found> {
    let r = spanned as f64 / (2 * (mags.len() - 1)) as f64;
    let half = r / 2.0;
    let dc = match mags {
        [zero, one, ..] if zero >= one => *zero,
        _ => 0.0,
    };
    let edge = 2.0 * sr / spanned as f64;
    let mut kept: Vec<(usize, Found)> = Vec::new();
    for (k, p) in peaks::located(mags, bin_hz) {
        let amp = libm::pow(10.0, p.db / 20.0);
        if amp < floor || kept.len() == 4 * cap {
            break;
        }
        let under = |q: usize| {
            let near = (k.abs_diff(q) as f64 * r - half).max(0.0);
            let image = (k + q) as f64 * r - half;
            mags[q] * (peaks::leakage(near) + peaks::leakage(image))
        };
        let leaked = kept.iter().map(|&(q, _)| under(q)).sum::<f64>()
            + dc * peaks::leakage(k as f64 * r - half);
        if mags[k] > LEAK_MARGIN * leaked / peaks::kernel(half) {
            kept.push((k, Found { hz: p.hz, amp }));
        }
    }
    kept.into_iter()
        .map(|(_, p)| p)
        .filter(|p| p.hz >= edge && p.hz <= sr / 2.0 - edge && prominent(mags, bin_hz, p))
        .collect()
}

fn prominent(mags: &[f64], bin_hz: f64, p: &Found) -> bool {
    let lo = (p.hz * libm::pow(2.0, -1.0 / 6.0) / bin_hz - PROMINENCE_BINS).max(0.0) as usize;
    let hi = (p.hz * libm::pow(2.0, 1.0 / 6.0) / bin_hz + PROMINENCE_BINS) as usize;
    let mut near: Vec<f64> = mags[lo..=hi.min(mags.len() - 1)].to_vec();
    let middle = near.len() / 2;
    let (_, median, _) = near.select_nth_unstable_by(middle, f64::total_cmp);
    p.amp >= *median * libm::pow(10.0, PROMINENCE_DB / 20.0)
}

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * libm::log2(a / b)
}

fn near(a: f64, b: f64) -> bool {
    cents(a, b).abs() < NEAR_CENTS || (a - b).abs() < NEAR_HZ
}

struct Fragment {
    hops: Vec<usize>,
    hz: Vec<f64>,
    amp: Vec<f64>,
}

impl Fragment {
    fn last_hz(&self) -> f64 {
        self.hz[self.hz.len() - 1]
    }
}

fn link(peaks: &[Vec<Found>], gap: usize) -> Vec<Fragment> {
    let mut all: Vec<Fragment> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    for (k, found) in peaks.iter().enumerate() {
        open.retain(|&f| k - all[f].hops[all[f].hops.len() - 1] <= gap);
        let mut order: Vec<&Found> = found.iter().collect();
        order.sort_by(|a, b| b.amp.total_cmp(&a.amp).then(a.hz.total_cmp(&b.hz)));
        let mut taken: Vec<usize> = Vec::new();
        for p in order {
            let off = |f: usize| cents(p.hz, all[f].last_hz()).abs();
            let best = open
                .iter()
                .copied()
                .filter(|f| !taken.contains(f) && near(p.hz, all[*f].last_hz()))
                .min_by(|&a, &b| off(a).total_cmp(&off(b)).then(a.cmp(&b)));
            let f = best.unwrap_or_else(|| {
                all.push(Fragment {
                    hops: Vec::new(),
                    hz: Vec::new(),
                    amp: Vec::new(),
                });
                open.push(all.len() - 1);
                all.len() - 1
            });
            all[f].hops.push(k);
            all[f].hz.push(p.hz);
            all[f].amp.push(p.amp);
            taken.push(f);
        }
    }
    all
}

struct Group {
    hz: f64,
    energy: f64,
    members: Vec<Fragment>,
}

fn group(fragments: Vec<Fragment>) -> Vec<Group> {
    let mut weighed: Vec<(f64, f64, Fragment)> = fragments
        .into_iter()
        .map(|f| {
            let energy: f64 = f.amp.iter().map(|a| a * a).sum();
            let hz = f.hz.iter().zip(&f.amp).map(|(h, a)| h * a * a).sum::<f64>() / energy;
            (hz, energy, f)
        })
        .collect();
    weighed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.hops[0].cmp(&b.2.hops[0])));
    let mut groups: Vec<Group> = Vec::new();
    for (hz, energy, f) in weighed {
        match groups.last_mut() {
            Some(held) if near(hz, held.hz) || hz - held.hz < LEAST_GAP_HZ => {
                held.hz = (held.hz * held.energy + hz * energy) / (held.energy + energy);
                held.energy += energy;
                held.members.push(f);
            }
            _ => groups.push(Group {
                hz,
                energy,
                members: vec![f],
            }),
        }
    }
    groups
}

/// Each hop's own peak, energy-weighted: (frequency, amplitude).
fn peaks_of(g: &Group, hops: usize) -> Vec<Option<(f64, f64)>> {
    let mut sum = vec![0.0; hops];
    let mut weight = vec![0.0; hops];
    for f in &g.members {
        for ((k, hz), a) in f.hops.iter().zip(&f.hz).zip(&f.amp) {
            sum[*k] += hz * a * a;
            weight[*k] += a * a;
        }
    }
    sum.iter()
        .zip(&weight)
        .map(|(s, w)| (*w > 0.0).then(|| (s / w, w.sqrt())))
        .collect()
}

fn coarse(own: &[Option<(f64, f64)>], otherwise: f64) -> Vec<f64> {
    let held: Vec<Option<f64>> = own.iter().map(|p| p.map(|(hz, _)| hz)).collect();
    filled(&held, otherwise)
}

const CLAIM_MARGIN: f64 = 2.0;

/// Per hop: whether the group holds a peak, where alone the phase slope reads its frequency;
/// every other group's peak; the least modulus that is the track's own, 0 within its peaks'
/// span, `CLAIM_MARGIN` times `amp e^(-d^2 / (2 sigma^2))` at the gap d within `reach` of
/// them, infinite past.
struct Claim {
    peak: bool,
    rivals: Vec<f64>,
    least: f64,
}

fn claims(
    held: &[Vec<Option<(f64, f64)>>],
    curves: &[Vec<f64>],
    sigmas: &[f64],
    reach: usize,
) -> Vec<Vec<Claim>> {
    (0..held.len())
        .map(|g| {
            let hops = held[g].len();
            let peaks: Vec<usize> = (0..hops).filter(|&k| held[g][k].is_some()).collect();
            (0..hops)
                .map(|k| {
                    let others: Vec<(f64, f64)> = (0..held.len())
                        .filter(|&h| h != g)
                        .filter_map(|h| held[h][k])
                        .collect();
                    let at = peaks.partition_point(|&j| j < k);
                    let peak = held[g][k].is_some();
                    let near = |j: usize| j.abs_diff(k) <= reach;
                    let reached =
                        (at < peaks.len() && near(peaks[at])) || (at > 0 && near(peaks[at - 1]));
                    let least = if peak || (at > 0 && at < peaks.len()) {
                        0.0
                    } else if !reached {
                        f64::INFINITY
                    } else {
                        let leak = others
                            .iter()
                            .map(|&(hz, amp)| {
                                let d = (hz - curves[g][k]) / sigmas[g];
                                amp * libm::exp(-0.5 * d * d)
                            })
                            .fold(0.0, f64::max);
                        CLAIM_MARGIN * leak
                    };
                    Claim {
                        peak,
                        rivals: others.into_iter().map(|(hz, _)| hz).collect(),
                        least,
                    }
                })
                .collect()
        })
        .collect()
}

/// `None` hops interpolated between their held neighbours, held past the ends.
fn filled(held: &[Option<f64>], otherwise: f64) -> Vec<f64> {
    let known: Vec<(usize, f64)> = held
        .iter()
        .enumerate()
        .filter_map(|(k, v)| v.map(|v| (k, v)))
        .collect();
    let (Some(&first), Some(&last)) = (known.first(), known.last()) else {
        return vec![otherwise; held.len()];
    };
    let mut next = 0;
    (0..held.len())
        .map(|k| {
            while next < known.len() && known[next].0 < k {
                next += 1;
            }
            match (held[k], next) {
                (Some(v), _) => v,
                (None, 0) => first.1,
                (None, n) if n == known.len() => last.1,
                (None, n) => {
                    let ((a, va), (b, vb)) = (known[n - 1], known[n]);
                    va + (vb - va) * (k - a) as f64 / (b - a) as f64
                }
            }
        })
        .collect()
}

/// A knot table read at rising instants: `value` its straight lines, held past its ends;
/// `integral` its running integral from the first knot, extended at the end values.
struct Walk<'a> {
    times: &'a [f64],
    values: &'a [f64],
    prefix: Vec<f64>,
    at: usize,
}

impl<'a> Walk<'a> {
    fn new(times: &'a [f64], values: &'a [f64]) -> Walk<'a> {
        let mut prefix = vec![0.0];
        for k in 1..times.len() {
            let dt = times[k] - times[k - 1];
            prefix.push(prefix[k - 1] + dt * (values[k] + values[k - 1]) * 0.5);
        }
        Walk {
            times,
            values,
            prefix,
            at: 0,
        }
    }

    /// The segment holding `t`, the cursor moved there, back if `t` came before it.
    fn segment(&mut self, t: f64) -> usize {
        let last = self.times.len() - 1;
        if t < self.times[self.at] {
            self.at = self.times.partition_point(|&s| s <= t).saturating_sub(1);
        }
        while self.at < last && self.times[self.at + 1] <= t {
            self.at += 1;
        }
        self.at
    }

    fn value(&mut self, t: f64) -> f64 {
        let k = self.segment(t);
        if t <= self.times[0] || k == self.times.len() - 1 {
            return self.values[k];
        }
        let u = (t - self.times[k]) / (self.times[k + 1] - self.times[k]);
        self.values[k] + (self.values[k + 1] - self.values[k]) * u
    }

    fn integral(&mut self, t: f64) -> f64 {
        let k = self.segment(t);
        let u = t - self.times[k];
        if t <= self.times[0] || k == self.times.len() - 1 {
            return self.prefix[k] + self.values[k] * u;
        }
        let slope = (self.values[k + 1] - self.values[k]) / (self.times[k + 1] - self.times[k]);
        self.prefix[k] + u * (self.values[k] + 0.5 * slope * u)
    }
}

#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}

const ZERO: C = C { re: 0.0, im: 0.0 };

impl C {
    fn of_turns(turns: f64) -> C {
        let (im, re) = libm::sincos(TAU * turns.rem_euclid(1.0));
        C { re, im }
    }
    fn mul(self, o: C) -> C {
        C {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    fn scale(self, s: f64) -> C {
        C {
            re: self.re * s,
            im: self.im * s,
        }
    }
    fn add(self, o: C) -> C {
        C {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
    fn conj(self) -> C {
        C {
            re: self.re,
            im: -self.im,
        }
    }
    fn norm(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    fn abs(self) -> f64 {
        self.norm().sqrt()
    }
    fn arg(self) -> f64 {
        libm::atan2(self.im, self.re)
    }
}

/// One transform of the whole target, zero-padded; each track reads its band off it.
struct Whole {
    re: Vec<f64>,
    im: Vec<f64>,
}

impl Whole {
    fn of(x: &[f64], plan: &Plan, size: usize) -> Whole {
        let mut re = vec![0.0; size];
        let mut im = vec![0.0; size];
        re[..x.len()].copy_from_slice(x);
        plan.forward(&mut re, &mut im);
        Whole { re, im }
    }
}

/// One track's band: decimated `every` samples, shifted down by `centre_bin`, flat over
/// `lo_hz..hi_hz` and 6 of the widest Gaussians more; `sigma` its own Gaussian.
#[derive(Clone, Copy)]
struct Band {
    every: usize,
    centre_bin: usize,
    lo_hz: f64,
    hi_hz: f64,
    sigma: f64,
}

/// What every track's read shares: the grid, the transforms' lengths and plans, the floor.
struct Reader<'a> {
    grid: &'a Grid,
    size: usize,
    whole: Plan,
    bands: Vec<(usize, Plan)>,
    floor: f64,
    hop_secs: Vec<f64>,
}

impl Reader<'_> {
    fn rate(&self, b: &Band) -> f64 {
        self.grid.sr / b.every as f64
    }

    fn centre_hz(&self, b: &Band) -> f64 {
        b.centre_bin as f64 * self.grid.sr / self.size as f64
    }

    fn plan(&self, m: usize) -> &Plan {
        let (_, plan) = self
            .bands
            .iter()
            .find(|(n, _)| *n == m)
            .expect("every band length was planned");
        plan
    }

    /// The band's analytic signal, flat over the track's range and 240 Hz more, Gaussian past
    /// it, shifted down by the centre bin and decimated.
    fn band(&self, whole: &Whole, b: &Band) -> Vec<C> {
        let n = self.size;
        let m = n / b.every;
        let bin_hz = self.grid.sr / n as f64;
        let widest = RESOLUTION_HZ.1;
        let flat = (b.lo_hz - SIGMAS * widest, b.hi_hz + SIGMAS * widest);
        let first = ((flat.0 - SIGMAS * widest) / bin_hz).floor().max(0.0) as usize;
        let last = (((flat.1 + SIGMAS * widest) / bin_hz).ceil() as usize).min(n / 2);
        let (mut re, mut im) = (vec![0.0; m], vec![0.0; m]);
        let gain = 2.0 * m as f64 / n as f64;
        for bin in first..=last {
            let f = bin as f64 * bin_hz;
            let off = (flat.0 - f).max(f - flat.1).max(0.0);
            let w = libm::exp(-0.5 * (off / widest) * (off / widest)) * gain;
            let j = (bin as isize - b.centre_bin as isize).rem_euclid(m as isize) as usize;
            re[j] += whole.re[bin] * w;
            im[j] += whole.im[bin] * w;
        }
        self.plan(m).inverse(&mut re, &mut im);
        re.into_iter()
            .zip(im)
            .map(|(re, im)| C { re, im })
            .collect()
    }

    /// The band times e^(-i 2 pi (pwl_integral(t) - centre t)): demodulated along a table.
    fn mixed(&self, raw: &[C], b: &Band, (times, values): (&[f64], &[f64])) -> Vec<C> {
        let (centre, rate) = (self.centre_hz(b), self.rate(b));
        let mut walk = Walk::new(times, values);
        raw.iter()
            .enumerate()
            .map(|(m, z)| {
                let t = m as f64 / rate;
                z.mul(C::of_turns(centre * t - walk.integral(t)))
            })
            .collect()
    }

    /// The band's own transform, to be lowpassed more than once.
    fn spectrum(&self, z: &[C]) -> (Vec<f64>, Vec<f64>) {
        let mut re: Vec<f64> = z.iter().map(|c| c.re).collect();
        let mut im: Vec<f64> = z.iter().map(|c| c.im).collect();
        self.plan(z.len()).forward(&mut re, &mut im);
        (re, im)
    }

    /// `keep(hz)` each bin's weight, back to the band's samples.
    fn lowpass(
        &self,
        (re, im): &(Vec<f64>, Vec<f64>),
        rate: f64,
        keep: impl Fn(f64) -> f64,
    ) -> Vec<C> {
        let m = re.len();
        let (mut re, mut im) = (re.clone(), im.clone());
        for j in 0..m {
            let signed = if j <= m / 2 {
                j as f64
            } else {
                j as f64 - m as f64
            };
            let w = keep(signed * rate / m as f64);
            re[j] *= w;
            im[j] *= w;
        }
        self.plan(m).inverse(&mut re, &mut im);
        re.into_iter()
            .zip(im)
            .map(|(re, im)| C { re, im })
            .collect()
    }
}

fn gaussian(sigma: f64) -> impl Fn(f64) -> f64 {
    move |f| libm::exp(-0.5 * (f / sigma) * (f / sigma))
}

fn brickwall(half: f64) -> impl Fn(f64) -> f64 {
    move |f| if f.abs() < half { 1.0 } else { 0.0 }
}

/// One track while it is read: its tables, flags, and its band along its table.
struct Reading {
    band: Band,
    freq: Curve,
    kind: Kind,
    amp: Curve,
    phase: f64,
    flutter_db: f64,
    coherence: f64,
    hz: f64,
    unclaimed_db: Option<f64>,
    /// The band demodulated along `freq` and lowpassed by the band's own Gaussian.
    z: Vec<C>,
}

fn read(
    x: &[f64],
    grid: &Grid,
    (groups, reach): (&[Group], usize),
    floor: f64,
) -> (Vec<Track>, Option<f64>) {
    if groups.is_empty() {
        return (Vec::new(), energy_ratio(x, x));
    }
    let held: Vec<Vec<Option<(f64, f64)>>> =
        groups.iter().map(|g| peaks_of(g, grid.hops)).collect();
    let curves: Vec<Vec<f64>> = held
        .iter()
        .zip(groups)
        .map(|(o, g)| coarse(o, g.hz))
        .collect();
    let sigmas: Vec<f64> = groups
        .iter()
        .map(|g| {
            let gap = groups
                .iter()
                .filter(|o| !std::ptr::eq(*o, g))
                .map(|o| (g.hz - o.hz).abs())
                .fold(f64::MAX, f64::min);
            (GAP_SHARE * gap).clamp(RESOLUTION_HZ.0, RESOLUTION_HZ.1)
        })
        .collect();
    let claimed = claims(&held, &curves, &sigmas, reach);
    let narrowest = sigmas.iter().copied().fold(f64::MAX, f64::min);
    let pad = (PAD_SIGMAS / (TAU * narrowest) * grid.sr).ceil() as usize;
    let size = (x.len() + pad).next_power_of_two();
    let bands: Vec<Band> = curves
        .iter()
        .zip(&sigmas)
        .map(|(curve, &sigma)| {
            let lo = curve.iter().copied().fold(f64::MAX, f64::min);
            let hi = curve.iter().copied().fold(0f64, f64::max);
            let width = (hi - lo + 4.0 * SIGMAS * RESOLUTION_HZ.1).max(LEAST_BAND_RATE_HZ);
            let mut every = 1usize;
            while grid.hop.is_multiple_of(2 * every)
                && size.is_multiple_of(2 * every)
                && grid.sr / (2 * every) as f64 >= width
            {
                every *= 2;
            }
            Band {
                every,
                centre_bin: ((lo + hi) / 2.0 * size as f64 / grid.sr).round() as usize,
                lo_hz: lo,
                hi_hz: hi,
                sigma,
            }
        })
        .collect();
    let mut lengths: Vec<usize> = bands.iter().map(|b| size / b.every).collect();
    lengths.sort_unstable();
    lengths.dedup();
    let reader = Reader {
        grid,
        size,
        whole: Plan::new(size),
        bands: lengths.into_iter().map(|m| (m, Plan::new(m))).collect(),
        floor,
        hop_secs: (0..grid.hops).map(|k| grid.secs(k)).collect(),
    };

    let first = Whole::of(x, &reader.whole, size);
    let mut readings: Vec<Reading> = bands
        .iter()
        .zip(curves.iter().zip(&claimed))
        .map(|(b, (curve, claim))| {
            let raw = reader.band(&first, b);
            read_one(&reader, raw, *b, (curve, claim, b.sigma))
        })
        .collect();
    for _ in 0..CANCELLATIONS {
        let left: Vec<f64> = x
            .iter()
            .zip(bands_of(&readings, grid, x.len()))
            .map(|(a, s)| a - s)
            .collect();
        let left = Whole::of(&left, &reader.whole, size);
        readings = readings
            .iter()
            .zip(curves.iter().zip(&claimed))
            .map(|(r, (curve, claim))| {
                let again = add(&reader.band(&left, &r.band), &own(&reader, r));
                let curve = match r.kind {
                    Kind::Tonal => curve.clone(),
                    Kind::Breathy => {
                        let mut held = Walk::new(&r.freq.times, &r.freq.values);
                        reader.hop_secs.iter().map(|t| held.value(*t)).collect()
                    }
                };
                let flags = r.band.sigma.max(FLAG_LEAST_HZ);
                read_one(&reader, again, r.band, (&curve, claim, flags))
            })
            .collect();
    }

    let left: Vec<f64> = x
        .iter()
        .zip(synthesis(&readings, grid, x.len()))
        .map(|(a, s)| a - s)
        .collect();
    let whole_left = Whole::of(&left, &reader.whole, size);
    let tracks = readings
        .iter()
        .filter(|r| r.amp.values.iter().any(|v| *v > 0.0))
        .map(|r| {
            let residual_db = (r.kind == Kind::Tonal)
                .then(|| {
                    let mixed = reader.mixed(
                        &reader.band(&whole_left, &r.band),
                        &r.band,
                        (&r.freq.times, &r.freq.values),
                    );
                    let rest = reader.lowpass(
                        &reader.spectrum(&mixed),
                        reader.rate(&r.band),
                        gaussian(r.band.sigma),
                    );
                    let span = r.span(&reader);
                    let held: f64 = r.z[span.clone()].iter().map(|c| c.norm()).sum();
                    let lost: f64 = rest[span].iter().map(|c| c.norm()).sum();
                    (held > 0.0).then(|| 10.0 * libm::log10(lost / held))
                })
                .flatten();
            track(r, grid.start, residual_db)
        })
        .collect();
    (tracks, energy_ratio(&left, x))
}

fn track(r: &Reading, start: f64, residual_db: Option<f64>) -> Track {
    let shifted = |c: &Curve| Curve {
        times: c.times.iter().map(|t| t + start).collect(),
        ..c.clone()
    };
    Track {
        kind: r.kind,
        hz: r.hz,
        frequency: shifted(&r.freq),
        amplitude: shifted(&r.amp),
        phase: (r.kind == Kind::Tonal).then_some(r.phase),
        flutter_db: r.flutter_db,
        coherence: r.coherence,
        bandwidth_hz: (r.kind == Kind::Breathy).then(|| {
            let lorentz = -libm::log(r.coherence.max(1e-3)) / (PI * COHERENCE_LAG_SECS);
            lorentz.max(2.0 * r.band.sigma / 3.0)
        }),
        resolution_hz: r.band.sigma,
        resolution_secs: 1.0 / (TAU * r.band.sigma),
        residual_db,
        unclaimed_db: r.unclaimed_db,
    }
}

fn energy_ratio(rest: &[f64], whole: &[f64]) -> Option<f64> {
    let held: f64 = whole.iter().map(|v| v * v).sum();
    let lost: f64 = rest.iter().map(|v| v * v).sum();
    (held > 0.0).then(|| 10.0 * libm::log10(lost / held))
}

impl Reading {
    /// The band's samples from the amplitude table's first knot to its last.
    fn span(&self, reader: &Reader) -> std::ops::Range<usize> {
        let rate = reader.rate(&self.band);
        let at = |t: f64| ((t * rate).round() as usize).min(self.z.len() - 1);
        let (a, b) = (self.amp.times[0], self.amp.times[self.amp.times.len() - 1]);
        at(a)..at(b) + 1
    }
}

/// A tonal track's band as `band` reads it: its demodulated band remodulated by its phase
/// and shifted down by the band's centre bin, so the residual's band plus this is its band
/// again; a breathy track is left in the residual.
fn own(reader: &Reader, r: &Reading) -> Vec<C> {
    if r.kind == Kind::Breathy {
        return vec![ZERO; r.z.len()];
    }
    let (centre, rate) = (reader.centre_hz(&r.band), reader.rate(&r.band));
    let mut walk = Walk::new(&r.freq.times, &r.freq.values);
    r.z.iter()
        .enumerate()
        .map(|(m, z)| {
            let t = m as f64 / rate;
            z.mul(C::of_turns(walk.integral(t) - centre * t))
        })
        .collect()
}

/// Every tonal track's band as samples, Re z(t) e^(i 2 pi pwl_integral(t)), z read between
/// the band's samples on straight lines.
fn bands_of(readings: &[Reading], grid: &Grid, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for r in readings.iter().filter(|r| r.kind == Kind::Tonal) {
        let mut walk = Walk::new(&r.freq.times, &r.freq.values);
        let every = r.band.every as f64;
        let last = r.z.len() - 1;
        for (i, slot) in out.iter_mut().enumerate() {
            let at = i as f64 / every;
            let m = (at as usize).min(last);
            let u = at - m as f64;
            let z = r.z[m].scale(1.0 - u).add(r.z[(m + 1).min(last)].scale(u));
            let e = C::of_turns(walk.integral(i as f64 / grid.sr));
            *slot += z.re * e.re - z.im * e.im;
        }
    }
    out
}

/// Every tonal track's knot tables as samples: amplitude sin(2 pi pwl_integral + phase).
fn synthesis(readings: &[Reading], grid: &Grid, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for r in readings.iter().filter(|r| r.kind == Kind::Tonal) {
        let mut phase = Walk::new(&r.freq.times, &r.freq.values);
        let mut amp = Walk::new(&r.amp.times, &r.amp.values);
        let (a0, a1) = (r.amp.times[0], r.amp.times[r.amp.times.len() - 1]);
        let from = ((a0 * grid.sr).ceil().max(0.0) as usize).min(n);
        let to = (((a1 * grid.sr).floor() as usize) + 1).min(n);
        for (i, slot) in out.iter_mut().enumerate().take(to).skip(from) {
            let t = i as f64 / grid.sr;
            let turns = phase.integral(t).rem_euclid(1.0);
            *slot += amp.value(t) * libm::sin(TAU * turns + r.phase);
        }
    }
    out
}

fn add(a: &[C], b: &[C]) -> Vec<C> {
    a.iter().zip(b).map(|(x, y)| x.add(*y)).collect()
}

/// Reads one track off `raw`, its band: frequency refined by the band's phase slope along
/// `curve`, then amplitude, phase and flags, read at `flag_sigma`, along the table it fits.
fn read_one(
    reader: &Reader,
    raw: Vec<C>,
    b: Band,
    (curve, claim, flag_sigma): (&[f64], &[Claim], f64),
) -> Reading {
    let (freq, mean_hz) = refined(reader, &raw, b, (curve, claim));
    let r = measured(reader, &raw, b, (freq, claim), (None, Err(flag_sigma)));
    let r = match r.kind {
        Kind::Tonal => r,
        Kind::Breathy => {
            let ends = reader.hop_secs[reader.hop_secs.len() - 1].max(1.0 / reader.grid.sr);
            let held = Curve {
                times: vec![0.0, ends],
                values: vec![mean_hz; 2],
                error: 0.0,
            };
            let flags = Ok((r.flutter_db, r.coherence));
            measured(reader, &raw, b, (held, claim), (Some(Kind::Breathy), flags))
        }
    };
    Reading { hz: mean_hz, ..r }
}

/// The frequency per hop where the band sounds: `along`'s plus the band's mean phase slope
/// demodulated along it, held between; its table, fitted; and its energy-weighted mean.
fn refined(
    reader: &Reader,
    raw: &[C],
    b: Band,
    (along, claim): (&[f64], &[Claim]),
) -> (Curve, f64) {
    let (grid, hops) = (reader.grid, &reader.hop_secs);
    let rate = reader.rate(&b);
    let per_hop = grid.hop / b.every;
    let mixed = reader.mixed(raw, &b, (hops, along));
    let zc = reader.lowpass(&reader.spectrum(&mixed), rate, gaussian(b.sigma));
    let at_hop = |k: usize| zc[(k * per_hop).min(zc.len() - 1)];
    let sounding: Vec<bool> = (0..grid.hops)
        .map(|k| {
            let m = at_hop(k).abs();
            m >= reader.floor && m > claim[k].least
        })
        .collect();
    let mut turns = vec![0.0; zc.len()];
    for m in 1..zc.len() {
        turns[m] = turns[m - 1] + zc[m].mul(zc[m - 1].conj()).arg() / TAU;
    }
    let deviation = rate / (TAU * b.sigma);
    let reach = ((SLOPE_SIGMAS * deviation / 2.0).round() as usize)
        .max(per_hop.div_ceil(2))
        .max(1);
    let edge = (EDGE_SIGMAS * deviation).ceil() as usize + reach;
    let end = grid.samples.div_ceil(b.every);
    let inside = |k: usize| k * per_hop >= edge && k * per_hop + edge <= end;
    let counted: Vec<bool> = match (0..grid.hops).any(|k| sounding[k] && inside(k)) {
        true => (0..grid.hops).map(|k| sounding[k] && inside(k)).collect(),
        false => sounding.clone(),
    };
    let slope = |k: usize| {
        let lo = (k * per_hop).saturating_sub(reach);
        let hi = (k * per_hop + reach).min(zc.len() - 1).max(lo + 1);
        (turns[hi] - turns[lo]) * rate / (hi - lo) as f64
    };
    let counted: Vec<bool> = (0..grid.hops)
        .map(|k| {
            let (off, at) = (slope(k), along[k] + slope(k));
            let rival = claim[k].rivals.iter().any(|hz| (hz - at).abs() < off.abs());
            counted[k] && claim[k].peak && off.abs() <= CAPTURE_SIGMAS * b.sigma && !rival
        })
        .collect();
    let held: Vec<Option<f64>> = (0..grid.hops)
        .map(|k| counted[k].then(|| along[k] + slope(k)))
        .collect();
    let refined = filled(&held, along[0]);
    let weights: Vec<f64> = (0..grid.hops)
        .map(|k| if counted[k] { at_hop(k).norm() } else { 0.0 })
        .collect();
    let total: f64 = weights.iter().sum();
    let mean_hz = match total > 0.0 {
        true => {
            refined
                .iter()
                .zip(&weights)
                .map(|(f, w)| f * w)
                .sum::<f64>()
                / total
        }
        false => refined[0],
    };
    let cent = mean_hz * (libm::pow(2.0, KNOT_CENTS / 1200.0) - 1.0);
    let tolerance = cent.max(2.0 * spread(&refined, &counted));
    let table = knots(hops, &refined, &counted, tolerance);
    (refit(table, hops, &refined, &weights), mean_hz)
}

/// Amplitude, phase and flags along a frequency table; `kind` where already known, and the
/// flags, flutter and coherence, where already read, else the Gaussian they are read at.
fn measured(
    reader: &Reader,
    raw: &[C],
    b: Band,
    (freq, claim): (Curve, &[Claim]),
    (kind, flags): (Option<Kind>, Result<(f64, f64), f64>),
) -> Reading {
    let grid = reader.grid;
    let rate = reader.rate(&b);
    let per_hop = grid.hop / b.every;
    let spectrum = reader.spectrum(&reader.mixed(raw, &b, (&freq.times, &freq.values)));
    let z = reader.lowpass(&spectrum, rate, gaussian(b.sigma));
    let at_hop = |k: usize| z[(k * per_hop).min(z.len() - 1)];
    let moduli: Vec<f64> = (0..grid.hops).map(|k| at_hop(k).abs()).collect();
    let loudest = (0..grid.hops)
        .max_by(|&a, &b| moduli[a].total_cmp(&moduli[b]).then(b.cmp(&a)))
        .unwrap_or(0);
    let (flutter_db, coherence) = flags.unwrap_or_else(|flag_sigma| {
        let held = grid.samples.div_ceil(b.every).min(z.len());
        let from = |secs: f64| (loudest * per_hop).min(held.saturating_sub((secs * rate) as usize));
        let flagged = match flag_sigma == b.sigma {
            true => z.clone(),
            false => reader.lowpass(&spectrum, rate, gaussian(flag_sigma)),
        };
        let banded = reader.lowpass(
            &spectrum,
            rate,
            brickwall(flag_sigma.max(COHERENCE_HALF_HZ)),
        );
        (
            flutter(&flagged[..held], rate, from(FLUTTER_SPAN_SECS)),
            coherence(&banded[..held], rate, from(COHERENCE_SPAN_SECS)),
        )
    });
    let kind = kind.unwrap_or(match flutter_db > BREATHY_FLUTTER_DB {
        true => Kind::Breathy,
        false => Kind::Tonal,
    });
    let claimed = |k: usize| moduli[k] > claim[k].least;
    let levels: Vec<f64> = match kind {
        Kind::Tonal => moduli.clone(),
        Kind::Breathy => {
            let half = ((BREATHY_POWER_SECS / 2.0) * rate).round() as usize;
            (0..grid.hops)
                .map(|k| {
                    let c = (k * per_hop).min(z.len() - 1);
                    let (lo, hi) = (c.saturating_sub(half), (c + half).min(z.len() - 1));
                    let power =
                        z[lo..=hi].iter().map(|v| v.norm()).sum::<f64>() / (hi - lo + 1) as f64;
                    (power / 2.0).sqrt()
                })
                .collect()
        }
    };
    let peak = |level: f64| match kind {
        Kind::Tonal => level,
        Kind::Breathy => level * SQRT_2,
    };
    let heard = |k: usize| peak(levels[k]) >= reader.floor;
    let amps: Vec<f64> = (0..grid.hops)
        .map(|k| {
            if heard(k) && claimed(k) {
                levels[k]
            } else {
                0.0
            }
        })
        .collect();
    let amp = amplitude_knots((&reader.hop_secs, grid.secs(1)), &amps, reader.floor);
    let weighted = (0..grid.hops)
        .filter(|&k| amps[k] > 0.0)
        .map(|k| at_hop(k).scale(at_hop(k).abs()))
        .fold(ZERO, C::add);
    let (mut kept, mut ruled) = (0.0, 0.0);
    for k in (0..grid.hops).filter(|&k| heard(k)) {
        match claimed(k) {
            true => kept += levels[k] * levels[k],
            false => ruled += levels[k] * levels[k],
        }
    }
    Reading {
        band: b,
        freq,
        kind,
        amp,
        phase: (weighted.arg() + FRAC_PI_2).rem_euclid(TAU),
        flutter_db,
        coherence,
        hz: 0.0,
        unclaimed_db: (ruled > 0.0 && kept > 0.0).then(|| 10.0 * libm::log10(ruled / kept)),
        z,
    }
}

fn flutter(z: &[C], rate: f64, from: usize) -> f64 {
    let frame = ((FLUTTER_FRAME_SECS * rate).round() as usize).max(1);
    let span = (FLUTTER_SPAN_SECS * rate) as usize;
    let frames = (from + span).min(z.len()).saturating_sub(from) / frame;
    let edge = FLUTTER_MEAN_FRAMES / 2;
    if frames <= 2 * edge + 1 || frames * frame * 10 < span {
        return 0.0;
    }
    let levels: Vec<f64> = (0..frames)
        .map(|f| {
            let s = &z[from + f * frame..from + (f + 1) * frame];
            let mean = s.iter().map(|c| c.abs()).sum::<f64>() / frame as f64;
            20.0 * libm::log10(mean + 1e-12)
        })
        .collect();
    let offs: Vec<f64> = (edge..frames - edge)
        .map(|f| {
            let around = levels[f - edge..=f + edge].iter().sum::<f64>();
            levels[f] - around / FLUTTER_MEAN_FRAMES as f64
        })
        .collect();
    let mean = offs.iter().sum::<f64>() / offs.len() as f64;
    (offs.iter().map(|o| (o - mean) * (o - mean)).sum::<f64>() / offs.len() as f64).sqrt()
}

fn coherence(b: &[C], rate: f64, from: usize) -> f64 {
    let lag = ((COHERENCE_LAG_SECS * rate).round() as usize).max(1);
    let to = (from + (COHERENCE_SPAN_SECS * rate) as usize).min(b.len());
    if to < from + 3 * lag {
        return 1.0;
    }
    let (mut cross, mut early, mut late) = (ZERO, 0.0, 0.0);
    for m in from..to - lag {
        cross = cross.add(b[m].conj().mul(b[m + lag]));
        early += b[m].norm();
        late += b[m + lag].norm();
    }
    let denom = (early * late).sqrt();
    if denom > 0.0 {
        cross.abs() / denom
    } else {
        1.0
    }
}

fn amplitude_knots((hop_secs, step): (&[f64], f64), amps: &[f64], floor: f64) -> Curve {
    let mut sounding = (0..amps.len()).filter(|&k| amps[k] > 0.0);
    let (Some(first), Some(last)) = (sounding.clone().next(), sounding.next_back()) else {
        return Curve {
            times: vec![hop_secs[0], hop_secs[hop_secs.len() - 1]],
            values: vec![0.0, 0.0],
            error: 0.0,
        };
    };
    let (a, b) = (first.saturating_sub(1), (last + 1).min(amps.len() - 1));
    let peak = amps.iter().copied().fold(0.0, f64::max);
    let all = vec![true; b - a + 1];
    let tolerance = (KNOT_SHARE * peak).max(floor);
    let mut table = knots(&hop_secs[a..=b], &amps[a..=b], &all, tolerance);
    if table.times.len() == 1 {
        table.times.push(table.times[0] + step);
        table.values.push(0.0);
    }
    table
}

/// A robust deviation of a curve's noise about its trend where `counts`: the median absolute
/// second difference over 0.6745 sqrt 6, as independent errors e give var(e_{k-1} - 2 e_k +
/// e_{k+1}) = 6 var(e); 0 under three counted points in a row.
fn spread(values: &[f64], counts: &[bool]) -> f64 {
    let mut d: Vec<f64> = (1..values.len().saturating_sub(1))
        .filter(|&k| counts[k - 1] && counts[k] && counts[k + 1])
        .map(|k| (values[k - 1] - 2.0 * values[k] + values[k + 1]).abs())
        .collect();
    if d.is_empty() {
        return 0.0;
    }
    let middle = d.len() / 2;
    let (_, median, _) = d.select_nth_unstable_by(middle, f64::total_cmp);
    *median / (0.6745 * 6f64.sqrt())
}

/// The table's values refit, its times held, by least squares weighted by `weights` over
/// every point, the line through two knots continuous; a knot no weight reaches keeps its
/// value. `error` the largest gap it leaves at a weighted point.
fn refit(table: Curve, at: &[f64], measured: &[f64], weights: &[f64]) -> Curve {
    let times = table.times;
    let n = times.len();
    let (mut diag, mut off, mut rhs) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut seg = 0;
    for ((t, y), w) in at.iter().zip(measured).zip(weights) {
        while seg + 2 < n && *t > times[seg + 1] {
            seg += 1;
        }
        if *w <= 0.0 {
            continue;
        }
        let u = ((t - times[seg]) / (times[seg + 1] - times[seg])).clamp(0.0, 1.0);
        diag[seg] += w * (1.0 - u) * (1.0 - u);
        diag[seg + 1] += w * u * u;
        off[seg] += w * (1.0 - u) * u;
        rhs[seg] += w * (1.0 - u) * y;
        rhs[seg + 1] += w * u * y;
    }
    let ridge = diag.iter().copied().fold(0.0, f64::max) * 1e-9;
    if ridge == 0.0 {
        return Curve {
            times,
            values: table.values,
            error: 0.0,
        };
    }
    for k in 0..n {
        diag[k] += ridge;
        rhs[k] += ridge * table.values[k];
    }
    let values = tridiagonal(&diag, &off, &rhs);
    let mut walk = Walk::new(&times, &values);
    let error = at
        .iter()
        .zip(measured)
        .zip(weights)
        .filter(|(_, w)| **w > 0.0)
        .map(|((t, y), _)| (walk.value(*t) - y).abs())
        .fold(0.0, f64::max);
    Curve {
        times,
        values,
        error,
    }
}

/// The symmetric tridiagonal system diag, off (off[k] couples k and k + 1), by Thomas.
fn tridiagonal(diag: &[f64], off: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let (mut c, mut d) = (vec![0.0; n], vec![0.0; n]);
    for k in 0..n {
        let (below, carried, held) = match k {
            0 => (0.0, 0.0, 0.0),
            _ => (off[k - 1], c[k - 1], d[k - 1]),
        };
        let pivot = diag[k] - below * carried;
        c[k] = if k + 1 < n { off[k] / pivot } else { 0.0 };
        d[k] = (rhs[k] - below * held) / pivot;
    }
    let mut x = vec![0.0; n];
    for k in (0..n).rev() {
        x[k] = d[k] - if k + 1 < n { c[k] * x[k + 1] } else { 0.0 };
    }
    x
}

/// The fewest of the points, first and last kept, whose straight lines pass within
/// `tolerance` of every point `counts` marks: each segment runs to the last point whose line
/// from the segment's start every marked point between admits. `error` the largest gap left.
fn knots(times: &[f64], values: &[f64], counts: &[bool], tolerance: f64) -> Curve {
    let n = times.len();
    if n <= 2 {
        return Curve {
            times: times.to_vec(),
            values: values.to_vec(),
            error: 0.0,
        };
    }
    let mut kept = vec![0usize];
    let mut s = 0;
    while s < n - 1 {
        let (mut lo, mut hi) = (f64::MIN, f64::MAX);
        let mut end = s + 1;
        for e in s + 1..n {
            let dt = times[e] - times[s];
            let slope = (values[e] - values[s]) / dt;
            if slope >= lo && slope <= hi {
                end = e;
            }
            if counts[e] {
                lo = lo.max((values[e] - tolerance - values[s]) / dt);
                hi = hi.min((values[e] + tolerance - values[s]) / dt);
            }
            if lo > hi {
                break;
            }
        }
        kept.push(end);
        s = end;
    }
    let mut error = 0f64;
    for w in kept.windows(2) {
        for j in (w[0]..=w[1]).filter(|&j| counts[j]) {
            let u = (times[j] - times[w[0]]) / (times[w[1]] - times[w[0]]);
            let line = values[w[0]] + (values[w[1]] - values[w[0]]) * u;
            error = error.max((values[j] - line).abs());
        }
    }
    Curve {
        times: kept.iter().map(|&k| times[k]).collect(),
        values: kept.iter().map(|&k| values[k]).collect(),
        error,
    }
}
