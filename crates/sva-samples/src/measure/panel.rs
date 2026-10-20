// Concern: the perceptual axes per note and whole: pitch, sharpness, decay, attack, beating | Non-concern: finding notes (score/), comparing two (sva-core) | IO: (planes, rate, start, spl) -> Panel

use std::f64::consts::TAU;

use super::loudness_pattern::{self, LoudnessPattern, sharpness};
use super::score::{self, Score};
use crate::fft::fft;

pub const MODEL: &str = "per note of `score` and over the whole target, off `loudness_pattern` \
                         at 10 ms frames (every channel's power averaged): `sharpness_acum` the loudness-weighted mean of each \
                         frame's sharpness (DIN 45692's weighting over Bark); `decay_phon_per_sec` \
                         the least-squares slope of 10 log2 of the total loudness from two frames \
                         past its loudest until 30 phon under it, null under five frames; \
                         `beating_phon` the rms of each band's 10 log2 specific loudness about its \
                         quadratic Savitzky-Golay trend (up to 51 frames, the most odd the \
                         band's frames hold), over frames within 20 phon of the band's loudest, at least 20 \
                         of them, weighted by the band's mean loudness: what moves at 0.5 to 20 \
                         Hz; `attack_secs` the 10 to 90 % rise of the 1 ms rms to its loudest, \
                         from the last sample at 10 % within 300 ms before it, null where none \
                         is, a sound held or cropped mid-note (silence stands before a strike \
                         at the target's start); per note, `pitches_hz` each voice's f0 moved as \
                         its series' f0 moves read off the peaks of its first 16 harmonics, each \
                         within 1.5 frame bins of k f0 in a Hann over the note's first 300 ms \
                         zero-padded eightfold, parabolically interpolated in dB and weighted by \
                         power: f0 = sum w f k / sum w k^2.";

/// The loudness pattern's frame every axis but pitch and attack is read at.
pub const FRAME_SECS: f64 = 0.01;
const DECAY_SKIP: usize = 2;
const DECAY_RANGE_PHON: f64 = 30.0;
const DECAY_LEAST: usize = 5;
const BEAT_RANGE_PHON: f64 = 20.0;
const BEAT_LEAST: usize = 20;
const BEAT_WIDEST: usize = 51;
const RMS_SECS: f64 = 0.001;
const ATTACK_REACH_SECS: f64 = 0.3;
const ATTACK_LOW: f64 = 0.1;
const ATTACK_HIGH: f64 = 0.9;
const PITCH_SPAN_SECS: f64 = 0.3;
const PITCH_SETTLE_SECS: f64 = 0.015;
const PITCH_PAD: usize = 8;
const PITCH_SEARCH_BINS: f64 = 1.5;
const PITCH_HARMONICS: usize = 16;
const LOG_FLOOR_SONE: f64 = 1e-4;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Axes {
    pub sharpness_acum: Option<f64>,
    pub decay_phon_per_sec: Option<f64>,
    pub beating_phon: Option<f64>,
    pub attack_secs: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    pub at_secs: f64,
    pub length_secs: f64,
    /// Each voice's f0, its own harmonics refined.
    pub pitches_hz: Vec<f64>,
    pub axes: Axes,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Panel {
    pub spl_db: f64,
    pub whole: Axes,
    pub notes: Vec<Note>,
}

pub fn measure(planes: &[&[f64]], sr: f64, start: f64, spl_db: f64) -> Panel {
    let pattern = loudness_pattern::measure(planes, sr, start, FRAME_SECS, spl_db);
    let score = score::measure(planes, sr, start);
    let len = super::held(planes);
    let x: Vec<f64> = (0..len)
        .map(|n| planes.iter().map(|p| p[n]).sum::<f64>() / planes.len() as f64)
        .collect();
    let rms = Rms::of(planes, sr, start);
    let struck = score.events.first().is_some_and(|e| opens(e, start));
    let whole = axes(&pattern, &rms, (start, start + len as f64 / sr), struck);
    let notes = notes(&score, &pattern, &rms, (&x, sr, start));
    Panel {
        spl_db,
        whole,
        notes,
    }
}

fn notes(
    score: &Score,
    pattern: &LoudnessPattern,
    rms: &Rms,
    (x, sr, start): (&[f64], f64, f64),
) -> Vec<Note> {
    score
        .events
        .iter()
        .map(|e| {
            let a = (((e.at_secs - start + PITCH_SETTLE_SECS) * sr) as usize).min(x.len());
            let b = ((e.next_secs.min(e.at_secs + PITCH_SPAN_SECS) - start) * sr) as usize;
            let seg = &x[a..b.min(x.len()).max(a)];
            let struck = opens(e, start);
            Note {
                at_secs: e.at_secs,
                length_secs: e.next_secs - e.at_secs,
                pitches_hz: e
                    .voices
                    .iter()
                    .filter_map(|v| refined(seg, sr, v))
                    .collect(),
                axes: axes(pattern, rms, (e.at_secs, e.next_secs), struck),
            }
        })
        .collect()
}

/// A strike within a millisecond of the target's start, silence before it.
fn opens(e: &score::Event, start: f64) -> bool {
    e.kind != score::Kind::Held && e.at_secs - start <= RMS_SECS
}

/// `struck`: the span opens on a strike at the target's start, silence before it.
fn axes(pattern: &LoudnessPattern, rms: &Rms, (a, b): (f64, f64), struck: bool) -> Axes {
    let frames: Vec<&loudness_pattern::PatternFrame> = pattern
        .frames
        .iter()
        .filter(|f| f.t_secs >= a && f.t_secs < b)
        .collect();
    let weighed: f64 = frames.iter().map(|f| f.total_sone).sum();
    let sharpness_acum = (weighed > 0.0).then(|| {
        frames
            .iter()
            .filter_map(|f| sharpness(&pattern.centres_hz, &f.specific).map(|s| s * f.total_sone))
            .sum::<f64>()
            / weighed
    });
    let level: Vec<f64> = frames
        .iter()
        .map(|f| 10.0 * libm::log2(f.total_sone + LOG_FLOOR_SONE))
        .collect();
    Axes {
        sharpness_acum,
        decay_phon_per_sec: frames
            .iter()
            .any(|f| f.total_sone > pattern.threshold_sone)
            .then(|| decay(&level, pattern.frame_secs))
            .flatten(),
        beating_phon: beating(&frames, pattern.threshold_sone),
        attack_secs: rms.attack(a - rms.start, b - rms.start, struck),
    }
}

fn decay(level: &[f64], step: f64) -> Option<f64> {
    let (p, top) =
        level.iter().enumerate().fold(
            (0, f64::NEG_INFINITY),
            |m, (i, &v)| if v > m.1 { (i, v) } else { m },
        );
    let tail: Vec<f64> = level
        .iter()
        .skip(p + DECAY_SKIP)
        .take_while(|v| **v >= top - DECAY_RANGE_PHON)
        .copied()
        .collect();
    if tail.len() < DECAY_LEAST {
        return None;
    }
    let n = tail.len() as f64;
    let mean_t = (n - 1.0) / 2.0 * step;
    let mean_v = tail.iter().sum::<f64>() / n;
    let (mut num, mut den) = (0.0, 0.0);
    for (i, v) in tail.iter().enumerate() {
        let t = i as f64 * step - mean_t;
        num += t * (v - mean_v);
        den += t * t;
    }
    Some(num / den)
}

/// Quadratic Savitzky-Golay smoothing over `width` (odd) points, the ends fitted on the
/// first and last full windows.
fn trend(y: &[f64], width: usize) -> Vec<f64> {
    let half = width / 2;
    (0..y.len())
        .map(|i| {
            let c = i.clamp(half, y.len() - 1 - half);
            let (mut s, mut st, mut stt, mut sy, mut sty, mut stty, mut s3, mut s4) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for (j, v) in y.iter().enumerate().take(c + half + 1).skip(c - half) {
                let t = j as f64 - c as f64;
                s += 1.0;
                st += t;
                stt += t * t;
                s3 += t * t * t;
                s4 += t * t * t * t;
                sy += v;
                sty += t * v;
                stty += t * t * v;
            }
            let m = [[s, st, stt], [st, stt, s3], [stt, s3, s4]];
            let coef = solve3(m, [sy, sty, stty]);
            let t = i as f64 - c as f64;
            coef[0] + coef[1] * t + coef[2] * t * t
        })
        .collect()
}

fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> [f64; 3] {
    let det = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det(m);
    let mut out = [0.0; 3];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut a = m;
        for row in 0..3 {
            a[row][k] = r[row];
        }
        *slot = det(a) / d;
    }
    out
}

fn beating(frames: &[&loudness_pattern::PatternFrame], floor: f64) -> Option<f64> {
    let bands = frames.first()?.specific.len();
    let (mut sum, mut weights) = (0.0, 0.0);
    for z in 0..bands {
        let y: Vec<f64> = frames
            .iter()
            .map(|f| 10.0 * libm::log2(f.specific[z] + floor))
            .collect();
        let top = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let held: Vec<f64> = y
            .into_iter()
            .filter(|v| *v > top - BEAT_RANGE_PHON)
            .collect();
        if held.len() < BEAT_LEAST {
            continue;
        }
        let width = (held.len() / 2 * 2 - 1).min(BEAT_WIDEST);
        let smooth = trend(&held, width);
        let rms = (held
            .iter()
            .zip(&smooth)
            .map(|(v, s)| (v - s) * (v - s))
            .sum::<f64>()
            / held.len() as f64)
            .sqrt();
        let w = frames.iter().map(|f| f.specific[z]).sum::<f64>() / frames.len() as f64;
        sum += rms * w;
        weights += w;
    }
    (weights > 0.0).then(|| sum / weights)
}

/// The rms over 1 ms about every sample, every channel's power averaged.
struct Rms {
    level: Vec<f64>,
    sr: f64,
    start: f64,
}

impl Rms {
    fn of(planes: &[&[f64]], sr: f64, start: f64) -> Rms {
        let len = super::held(planes);
        let mut running = vec![0.0; len + 1];
        for n in 0..len {
            let p = planes.iter().map(|c| c[n] * c[n]).sum::<f64>() / planes.len() as f64;
            running[n + 1] = running[n] + p;
        }
        let half = ((RMS_SECS * sr / 2.0).round() as usize).max(1);
        let level = (0..len)
            .map(|n| {
                let (lo, hi) = (n.saturating_sub(half), (n + half).min(len));
                ((running[hi] - running[lo]) / (2 * half) as f64).sqrt()
            })
            .collect();
        Rms { level, sr, start }
    }

    fn attack(&self, a: f64, b: f64, struck: bool) -> Option<f64> {
        let at = |t: f64| ((t * self.sr).max(0.0) as usize).min(self.level.len());
        let (lo, hi) = (at(a), at(b));
        if hi <= lo {
            return None;
        }
        let p = (lo..hi).fold(lo, |m, n| if self.level[n] > self.level[m] { n } else { m });
        let peak = self.level[p];
        if peak <= 0.0 {
            return None;
        }
        let from = p.saturating_sub((ATTACK_REACH_SECS * self.sr) as usize);
        let quiet = (from..=p)
            .rev()
            .find(|&n| self.level[n] <= ATTACK_LOW * peak);
        let on = quiet.or((struck && from == 0).then_some(0))?;
        let top = (on..=p).find(|&n| self.level[n] >= ATTACK_HIGH * peak)?;
        Some((top - on).max(1) as f64 / self.sr)
    }
}

/// A voice's f0 moved as its series' f0 moves when read off its first 16 harmonics' peaks,
/// each searched within 1.5 frame bins of k f0.
fn refined(seg: &[f64], sr: f64, voice: &score::Voiced) -> Option<f64> {
    if seg.len() < 2 {
        return None;
    }
    let n = (seg.len() * PITCH_PAD).next_power_of_two();
    let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
    let last = (seg.len() - 1) as f64;
    for (i, v) in seg.iter().enumerate() {
        re[i] = v * (0.5 - 0.5 * libm::cos(TAU * i as f64 / last));
    }
    fft(&mut re, &mut im);
    let db: Vec<f64> = (0..=n / 2)
        .map(|k| 10.0 * libm::log10(re[k] * re[k] + im[k] * im[k] + 1e-30))
        .collect();
    let bin = sr / n as f64;
    let reach = (PITCH_SEARCH_BINS * sr / seg.len() as f64 / bin).ceil() as usize;
    let w = |p: &score::Partial| libm::pow(10.0, p.db / 10.0);
    let k = |p: &score::Partial| p.harmonic as f64;
    let series = voice
        .partials
        .iter()
        .map(|p| w(p) * p.hz * k(p))
        .sum::<f64>()
        / voice
            .partials
            .iter()
            .map(|p| w(p) * k(p) * k(p))
            .sum::<f64>();
    let (mut num, mut den) = (0.0, 0.0);
    for h in (1..=PITCH_HARMONICS).map(|h| h as f64) {
        let centre = (h * series / bin).round() as usize;
        if centre < 1 || centre + reach + 1 >= db.len() {
            break;
        }
        let lo = centre.saturating_sub(reach).max(1);
        let j = (lo..=centre + reach).fold(lo, |m, j| if db[j] > db[m] { j } else { m });
        let (a, b, c) = (db[j - 1], db[j], db[j + 1]);
        let curve = a - 2.0 * b + c;
        let shift = if curve < 0.0 {
            (0.5 * (a - c) / curve).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let power = libm::pow(10.0, b / 10.0);
        num += power * (j as f64 + shift) * bin * h;
        den += power * h * h;
    }
    (den > 0.0 && series > 0.0).then(|| voice.hz * num / den / series)
}
