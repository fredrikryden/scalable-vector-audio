// Concern: the reading `internal` at scale s: excitation per ERB-number channel, its level over threshold, the modulation bands | Non-concern: the curves (hearing.rs) | IO: (planes, spl, s) -> Internal

pub mod modulation;
mod rate;

use std::f64::consts::{LN_10, PI};

use super::hearing::{
    self, CAM_FIRST, CAM_LAST, CAM_SPACING, HOP_SECS, WINDOW_SECS, WINDOW_SPLITS_HZ,
};
use crate::fft;
use modulation::Band;
use rate::Rate;

pub const MODEL: &str = "the ear's internal reading at scale s, through the hearing curves \
     (their own model text): frames every s ms, from the first whose window reaches the clip, \
     silence before it, to the clip's end, the elements from 0; each channel framed by \
     the Hann window of the Glasberg and Moore (2002) range holding its centre, the ranges \
     half-open [lo, hi), the window cos^2(pi t/T) of T = s times that range's length read at \
     the clip's own instants about the frame's time, silence outside the clip, its DFT \
     zero-padded to the next power of two, power per bin one-sided over the window's energy, \
     every channel's power averaged, a full-scale sine at `spl` dB SPL; channels from 1.8 Cam \
     every 0.25 s Cam up to 38.9; excitation E = sum over every bin of roex(p/s) G P, p as \
     ISO 532-2 sets it, X the bin's level per ERB through the level-free roex centred on it, no \
     bin left out but those whose weight is 0 in f64, no g > 4 cut and no floor; V = 10 \
     log10(1 + E/E_T), a model form: the level of the signal plus a noise of excitation E_T \
     over that noise, read as Florentine and Buus's (1981, JASA 70(6) 1646-1654, doi \
     10.1121/1.387219) excitation-level observer; on each channel's V at the hop, Jepsen, \
     Ewert and Dau's (2008, JASA 124 422-438, sec. II.B.4, doi 10.1121/1.2924135) modulation \
     stage whole: a first-order 150 Hz lowpass (Kohlrausch, Fassel and Dau 2000, JASA 108, \
     doi 10.1121/1.429605), then a second-order Butterworth lowpass at 2.5 Hz and complex \
     first-order resonators of 5 Hz bandwidth at 5 and 10 Hz and Q = 2 above, each centre's \
     lower -3 dB edge the last one's upper edge, the top centre at most min(fc/4, 1000 Hz) \
     and every upper edge under the hop's Nyquist, outputs above 10 Hz times 1/sqrt(2), the \
     element the real output at and under 10 Hz and the magnitude above (Dau, Kollmeier and \
     Kohlrausch 1997a, JASA 102, doi 10.1121/1.420344), each filter's poles e^(s_k h) at the \
     hop h, unity gain at its passband's centre; each band's elements at its information rate \
     min(2B, 1/h), B = (int_0^inf |H|^2)^2 / int_0^inf |H|^4 the statistical bandwidth \
     (Bendat and Piersol, Random Data, 4th ed. 2010) of H the transform of the squared window \
     normalised to 1 at 0 times 1/(1 + j nu/150) times the band, 2B independent values per \
     second (Shannon 1949, Proc. IRE 37(1), doi 10.1109/JRPROC.1949.232969), integrated on \
     Gauss-Legendre panels until Trefethen's (2008, SIAM Rev. 50(1) 67-87, doi \
     10.1137/060659831, eq. 4.14) bound over H's continuation off the real axis is under 2^-53 \
     of the integral, the tail bounded in closed form, the f64 roundings counted, and that \
     proven bound on each rate reported beside it; element j of a band at the frame nearest \
     j/rate, ties to even";

/// A clip's sample rate, its `spl`, the scale and what each frame reads off them.
pub struct Ear {
    rate: f64,
    spl_db: f64,
    scale: f64,
    hop_secs: f64,
    ranges: Vec<Range>,
    channels: Vec<Channel>,
}

struct Range {
    window_secs: f64,
    n: usize,
    twiddles: Vec<(f64, f64)>,
    hz: Vec<f64>,
    gain: Vec<f64>,
    around: Vec<(usize, Vec<f64>)>,
}

struct Channel {
    hz: f64,
    range: usize,
    threshold: f64,
    lower: usize,
    upper: (usize, Vec<f64>),
    bands: Vec<(Band, Rate)>,
}

fn nonzero(weights: Vec<f64>) -> (usize, Vec<f64>) {
    let first = weights.iter().position(|w| *w != 0.0).unwrap_or(0);
    let last = weights
        .iter()
        .rposition(|w| *w != 0.0)
        .map_or(first, |i| i + 1);
    (first, weights[first..last].to_vec())
}

fn decibels(x: f64) -> f64 {
    libm::pow(10.0, x / 10.0)
}

impl Range {
    fn of(window_secs: f64, rate: f64, spl_db: f64, top_hz: f64) -> Range {
        let n = (libm::ceil(window_secs * rate).max(1.0) as usize).next_power_of_two();
        let hz: Vec<f64> = (0..=n / 2).map(|j| j as f64 * rate / n as f64).collect();
        let full_scale = decibels(spl_db) / 0.5;
        let gain = hz
            .iter()
            .enumerate()
            .map(|(j, f)| {
                let sides = match j == 0 || j == n / 2 {
                    true => 1.0,
                    false => 2.0,
                };
                decibels(hearing::ear_gain_db(*f)) * full_scale * sides / n as f64
            })
            .collect();
        let around = hz
            .iter()
            .take_while(|f| **f < top_hz)
            .map(|centre| {
                nonzero(
                    hz.iter()
                        .map(|f| hearing::roex_around(*centre, *f))
                        .collect(),
                )
            })
            .collect();
        Range {
            window_secs,
            n,
            twiddles: fft::twiddles(n),
            hz,
            gain,
            around,
        }
    }

    /// The window of the frame centred at `t`.
    fn frame(&self, rate: f64, t: f64) -> Frame<'_> {
        let half = self.window_secs / 2.0;
        let first = libm::floor((t - half) * rate) as i64 + 1;
        let last = (libm::ceil((t + half) * rate) as i64 - 1).min(first + self.n as i64 - 1);
        let weights: Vec<f64> = (first..=last)
            .map(|s| {
                let c = libm::cos(PI * (s as f64 / rate - t) / self.window_secs);
                c * c
            })
            .collect();
        let energy = weights.iter().map(|w| w * w).sum();
        Frame {
            range: self,
            first,
            weights,
            energy,
        }
    }

    /// Power per bin over the window's energy, times the ear's gain and full scale.
    fn eardrum(&self, power: &[f64], energy: f64) -> Vec<f64> {
        power
            .iter()
            .zip(&self.gain)
            .map(|(p, g)| match energy > 0.0 {
                true => p / energy * g,
                false => 0.0,
            })
            .collect()
    }

    fn levels(&self, power: &[f64]) -> Vec<f64> {
        self.around
            .iter()
            .map(|(from, row)| {
                let sum: f64 = row.iter().zip(&power[*from..]).map(|(w, p)| w * p).sum();
                10.0 * libm::log10(sum)
            })
            .collect()
    }
}

/// One range's window at one frame: the samples it weighs and the bins it reads, so samples or
/// a closed form put their power per bin through the ear alike.
pub struct Frame<'a> {
    range: &'a Range,
    first: i64,
    weights: Vec<f64>,
    energy: f64,
}

impl Frame<'_> {
    /// The first sample the window weighs; `weights()[k]` weighs sample `first() + k`.
    pub fn first(&self) -> i64 {
        self.first
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn hz(&self) -> &[f64] {
        &self.range.hz
    }

    /// The transform's length: the window's, zero-padded to a power of two.
    pub fn n(&self) -> usize {
        self.range.n
    }

    /// `|X|^2` per bin of the weighted samples, averaged over `planes`, each holding sample
    /// `from + i` at `i`; a sample a plane does not hold is silence.
    pub fn power(&self, planes: &[&[f64]], from: i64) -> Vec<f64> {
        let bins = self.range.hz.len();
        let mut sum = vec![0.0; bins];
        let (mut re, mut im) = (vec![0.0; self.range.n], vec![0.0; self.range.n]);
        for pair in planes.chunks(2) {
            re.iter_mut().for_each(|v| *v = 0.0);
            im.iter_mut().for_each(|v| *v = 0.0);
            for (k, w) in self.weights.iter().enumerate() {
                let i = self.first + k as i64 - from;
                if (0..pair[0].len() as i64).contains(&i) {
                    re[k] = w * pair[0][i as usize];
                    im[k] = pair.get(1).map_or(0.0, |p| w * p[i as usize]);
                }
            }
            fft::fft_with(&mut re, &mut im, &self.range.twiddles);
            for (j, total) in sum.iter_mut().enumerate() {
                let mirror = (self.range.n - j) % self.range.n;
                *total += 0.5
                    * (re[j] * re[j]
                        + im[j] * im[j]
                        + re[mirror] * re[mirror]
                        + im[mirror] * im[mirror]);
            }
        }
        let count = planes.len() as f64;
        sum.iter().map(|p| p / count).collect()
    }
}

impl Channel {
    fn excitation(&self, scale: f64, hz: &[f64], power: &[f64], levels: &[f64]) -> f64 {
        let below: f64 = (0..self.lower)
            .filter(|i| power[*i] > 0.0)
            .map(|i| {
                let p = hearing::roex_p(self.hz, hz[i], levels[i]) / scale;
                hearing::roex(p, self.hz, hz[i]) * power[i]
            })
            .sum();
        let (from, weights) = &self.upper;
        let above: f64 = weights
            .iter()
            .zip(&power[*from..])
            .map(|(w, p)| w * p)
            .sum();
        below + above
    }

    fn level(&self, excitation: f64) -> f64 {
        10.0 / LN_10 * libm::log1p(excitation / self.threshold)
    }
}

impl Ear {
    pub fn of(rate: f64, spl_db: f64, scale: f64) -> Ear {
        let hop_secs = scale * HOP_SECS;
        let centres: Vec<f64> = (0..)
            .map(|k| CAM_FIRST + k as f64 * CAM_SPACING * scale)
            .take_while(|z| *z <= CAM_LAST)
            .map(hearing::cam_hz)
            .collect();
        let range_of = |hz: f64| WINDOW_SPLITS_HZ.partition_point(|split| *split <= hz);
        let tops: Vec<f64> = (0..WINDOW_SECS.len())
            .map(|r| {
                centres
                    .iter()
                    .filter(|hz| range_of(**hz) == r)
                    .fold(0.0, |a: f64, b| a.max(*b))
            })
            .collect();
        let ranges: Vec<Range> = WINDOW_SECS
            .iter()
            .zip(&tops)
            .map(|(secs, top)| Range::of(scale * secs, rate, spl_db, *top))
            .collect();
        let rates: Vec<Vec<Rate>> = ranges
            .iter()
            .zip(&tops)
            .map(|(range, top)| {
                modulation::bank(*top, hop_secs)
                    .iter()
                    .map(|b| rate::rate(b, range.window_secs, hop_secs))
                    .collect()
            })
            .collect();
        let channels = centres
            .iter()
            .map(|hz| {
                let r = range_of(*hz);
                let bins = &ranges[r].hz;
                let lower = bins.partition_point(|f| f < hz);
                let upper = nonzero(
                    bins[lower..]
                        .iter()
                        .map(|f| hearing::roex(hearing::roex_p(*hz, *f, 0.0) / scale, *hz, *f))
                        .collect(),
                );
                let bands = modulation::bank(*hz, hop_secs)
                    .into_iter()
                    .zip(rates[r].iter().copied())
                    .collect();
                Channel {
                    hz: *hz,
                    range: r,
                    threshold: decibels(hearing::threshold_excitation_db(*hz)),
                    lower,
                    upper: (lower + upper.0, upper.1),
                    bands,
                }
            })
            .collect();
        Ear {
            rate,
            spl_db,
            scale,
            hop_secs,
            ranges,
            channels,
        }
    }

    /// E per channel at frame `m`, centred at `m` hops, of the power per bin `power` answers
    /// each range's frame with, as `Frame::power` answers samples.
    pub fn excitation(&self, m: i64, power: &mut dyn FnMut(&Frame) -> Vec<f64>) -> Vec<f64> {
        let t = m as f64 * self.hop_secs;
        let read: Vec<Option<(Vec<f64>, Vec<f64>)>> = self
            .ranges
            .iter()
            .enumerate()
            .map(|(r, range)| {
                self.channels.iter().any(|c| c.range == r).then(|| {
                    let frame = range.frame(self.rate, t);
                    let power = range.eardrum(&power(&frame), frame.energy);
                    let levels = range.levels(&power);
                    (power, levels)
                })
            })
            .collect();
        self.channels
            .iter()
            .map(|c| {
                let (power, levels) = read[c.range].as_ref().expect("a read range");
                c.excitation(self.scale, &self.ranges[c.range].hz, power, levels)
            })
            .collect()
    }

    /// The first frame any window reaches a clip of `len` samples from, and its last frame.
    pub fn frames(&self, len: usize) -> (i64, i64) {
        let longest = self
            .ranges
            .iter()
            .map(|r| r.window_secs)
            .fold(0.0, f64::max);
        let first = -(libm::ceil(longest / 2.0 / self.hop_secs) as i64);
        let last = libm::floor(len as f64 / (self.hop_secs * self.rate)) as i64;
        (first, last)
    }

    /// Each channel's excitation at threshold, in the order `excitation` answers.
    pub fn thresholds(&self) -> Vec<f64> {
        self.channels.iter().map(|c| c.threshold).collect()
    }

    pub fn read(&self, planes: &[&[f64]], start_secs: f64) -> Internal {
        let (first, last) = self.frames(super::held(planes));
        let mut levels = vec![Vec::new(); self.channels.len()];
        for m in first..=last {
            for ((c, e), v) in self
                .channels
                .iter()
                .zip(self.excitation(m, &mut |frame| frame.power(planes, 0)))
                .zip(&mut levels)
            {
                v.push(c.level(e));
            }
        }
        let from = (-first) as usize;
        let channels = self
            .channels
            .iter()
            .zip(&levels)
            .map(|(c, v)| Heard {
                hz: c.hz,
                window_secs: self.ranges[c.range].window_secs,
                bands: modulation::elements(v, from, self.hop_secs, &c.bands)
                    .into_iter()
                    .zip(&c.bands)
                    .map(|(elements, (band, rate))| Sampled {
                        band: *band,
                        rate_hz: rate.per_frame / self.hop_secs,
                        rate_bound_hz: rate.bound / self.hop_secs,
                        elements,
                    })
                    .collect(),
            })
            .collect();
        Internal {
            rate: self.rate,
            start_secs,
            spl_db: self.spl_db,
            scale: self.scale,
            hop_secs: self.hop_secs,
            channels,
        }
    }
}

/// One band's elements, element `j` at `j / rate_hz` from the clip's start; `rate_bound_hz`
/// bounds how far `rate_hz` is from the one its exact statistical bandwidth gives.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampled {
    pub band: Band,
    pub rate_hz: f64,
    pub rate_bound_hz: f64,
    pub elements: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Heard {
    pub hz: f64,
    pub window_secs: f64,
    pub bands: Vec<Sampled>,
}

/// What `internal` reads off samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Internal {
    pub rate: f64,
    pub start_secs: f64,
    pub spl_db: f64,
    pub scale: f64,
    pub hop_secs: f64,
    pub channels: Vec<Heard>,
}

pub fn measure(planes: &[&[f64]], sr: f64, start: f64, spl_db: f64) -> Internal {
    Ear::of(sr, spl_db, 1.0).read(planes, start)
}
