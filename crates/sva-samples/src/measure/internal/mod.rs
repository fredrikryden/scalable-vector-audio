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
use sva_formula::C64;

pub const MODEL: &str = "the ear's internal reading at scale s, through the hearing curves \
     (their own model text): frames every s ms (Glasberg and Moore's 2002 1 ms hop at s = 1), \
     from the first whose window reaches the clip, silence before it, to the clip's end, the \
     elements from 0; each frame one spectrum composed as Glasberg and Moore (2002, JAES \
     50(5) 331-342) compose it (Moore 2014, Trends Hear. 18, doi 10.1177/2331216514550620): \
     each range's Hann window cos^2(pi t/T) of T = s times that range's length read at the \
     clip's own instants about the frame's time, silence outside the clip, its DFT \
     zero-padded to the next power of two, power per bin one-sided over that window's \
     energy, each range supplying only the bins in its own range, the ranges half-open \
     [lo, hi) so a bin on a split goes to the upper range, the lowest range from 0 Hz and \
     the highest to Nyquist; every channel's power averaged, a full-scale sine at `spl` dB \
     SPL; channels from 1.8 Cam every 0.25 s Cam up to 38.9; excitation E = sum over every \
     composed bin of roex(p/s) G P, p as ISO 532-2 sets it, X the bin's level per ERB through \
     the level-free roex centred on it, no bin left out but those whose weight is 0 in f64, no \
     g > 4 cut and no floor; V = 10 \
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
     of the range holding the channel's centre, normalised to 1 at 0 times 1/(1 + j nu/150) \
     times the band, 2B independent values per \
     second (Shannon 1949, Proc. IRE 37(1), doi 10.1109/JRPROC.1949.232969), integrated on \
     Gauss-Legendre panels until Trefethen's (2008, SIAM Rev. 50(1) 67-87, doi \
     10.1137/060659831, eq. 4.14) bound over H's continuation off the real axis is under 2^-53 \
     of the integral, the tail bounded in closed form, the f64 roundings counted, and that \
     proven bound on each rate reported beside it; element j of a band at the frame nearest \
     j/rate, ties to even; a known limit of the published model: in the short windows a real \
     tone's +f and -f sidelobes interfere, a 2f ripple the 1 ms hop aliases, measured 3.43 dB \
     in the 5302 Hz channel for a 1001.3 Hz tone at 80 dB SL";

/// A clip's sample rate, its `spl`, the scale and what each frame reads off them.
pub struct Ear {
    rate: f64,
    spl_db: f64,
    scale: f64,
    hop_secs: f64,
    ranges: Vec<Range>,
    hz: Vec<f64>,
    around: Vec<(usize, Vec<f64>)>,
    channels: Vec<Channel>,
}

/// One Glasberg and Moore (2002) range: its window, its transform's bins and the bins of
/// those it supplies to the composed spectrum, the ones in its own range.
struct Range {
    window_secs: f64,
    n: usize,
    twiddles: Vec<(f64, f64)>,
    hz: Vec<f64>,
    gain: Vec<f64>,
    supplies: std::ops::Range<usize>,
}

struct Channel {
    hz: f64,
    range: usize,
    threshold: f64,
    lower: usize,
    lower_slope: (f64, f64),
    below: Vec<f64>,
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

/// The range holding `hz`, each half-open [lo, hi).
fn range_of(hz: f64) -> usize {
    WINDOW_SPLITS_HZ.partition_point(|split| *split <= hz)
}

impl Range {
    fn of(r: usize, window_secs: f64, rate: f64, spl_db: f64) -> Range {
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
        let supplies =
            hz.partition_point(|f| range_of(*f) < r)..hz.partition_point(|f| range_of(*f) <= r);
        Range {
            window_secs,
            n,
            twiddles: fft::twiddles(n),
            hz,
            gain,
            supplies,
        }
    }

    /// The first and last samples the window centred at `t` weighs.
    fn span(&self, rate: f64, t: f64) -> (i64, i64) {
        let half = self.window_secs / 2.0;
        let first = libm::floor((t - half) * rate) as i64 + 1;
        let last = (libm::ceil((t + half) * rate) as i64 - 1).min(first + self.n as i64 - 1);
        (first, last)
    }

    fn weight(&self, rate: f64, t: f64, s: i64) -> f64 {
        let c = libm::cos(PI * (s as f64 / rate - t) / self.window_secs);
        c * c
    }

    /// The window of the frame centred at `t`.
    fn frame(&self, rate: f64, t: f64) -> Frame<'_> {
        let (first, last) = self.span(rate, t);
        let weights: Vec<f64> = (first..=last).map(|s| self.weight(rate, t, s)).collect();
        Frame {
            range: self,
            first,
            weights,
        }
    }

    fn energy(&self, rate: f64, t: f64) -> f64 {
        let (first, last) = self.span(rate, t);
        (first..=last)
            .map(|s| {
                let w = self.weight(rate, t, s);
                w * w
            })
            .sum()
    }
}

/// One range's window at one frame: the samples it weighs and the bins it reads, so samples or
/// a closed form put their power per bin through the ear alike.
pub struct Frame<'a> {
    range: &'a Range,
    first: i64,
    weights: Vec<f64>,
}

impl Frame<'_> {
    /// The first sample the window weighs; `weights()[k]` weighs sample `first() + k`.
    pub fn first(&self) -> i64 {
        self.first
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The bins this window supplies, its own range's.
    pub fn hz(&self) -> &[f64] {
        &self.range.hz[self.range.supplies.clone()]
    }

    /// `hz()[i]` is transform bin `bins().start + i`.
    pub fn bins(&self) -> std::ops::Range<usize> {
        self.range.supplies.clone()
    }

    /// The transform's length: the window's, zero-padded to a power of two.
    pub fn n(&self) -> usize {
        self.range.n
    }

    /// `|X|^2` per bin of `bins()` of the weighted samples, averaged over `planes`, each
    /// holding sample `from + i` at `i`; a sample a plane does not hold is silence.
    pub fn power(&self, planes: &[&[f64]], from: i64) -> Vec<f64> {
        let bins = self.bins();
        let mut sum = vec![0.0; bins.len()];
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
            for (j, total) in bins.clone().zip(&mut sum) {
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
    /// `roex(p/s)` at bin `i` under the centre, `p` at level `x`.
    fn skirt(&self, scale: f64, i: usize, x: f64) -> f64 {
        let (p51, slope) = self.lower_slope;
        hearing::roex_apart(
            hearing::lower_p(p51, slope, x) / scale,
            self.below[i],
            self.hz,
        )
    }

    fn excitation(&self, scale: f64, power: &[f64], levels: &[f64]) -> f64 {
        let mut below = 0.0;
        for i in 0..self.lower {
            if power[i] > 0.0 {
                below += self.skirt(scale, i, levels[i]) * power[i];
            }
        }
        let (from, weights) = &self.upper;
        let above: f64 = weights
            .iter()
            .zip(&power[*from..])
            .map(|(w, p)| w * p)
            .sum();
        below + above
    }

    fn weights(&self, scale: f64, toward: &[f64], levels: &[f64]) -> (usize, Vec<f64>) {
        let (from, upper) = &self.upper;
        let first = (0..self.lower).find(|i| toward[*i] > 0.0).unwrap_or(*from);
        let start = first.min(*from);
        let mut row = vec![0.0; from + upper.len() - start];
        for i in first..self.lower {
            if toward[i] > 0.0 {
                row[i - start] = self.skirt(scale, i, levels[i]) * toward[i];
            }
        }
        for (k, w) in upper.iter().enumerate() {
            row[from + k - start] = w * toward[from + k];
        }
        (start, row)
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
            .enumerate()
            .map(|(r, secs)| Range::of(r, scale * secs, rate, spl_db))
            .collect();
        let hz: Vec<f64> = ranges
            .iter()
            .flat_map(|range| range.hz[range.supplies.clone()].iter().copied())
            .collect();
        let top = centres.iter().fold(0.0, |a: f64, b| a.max(*b));
        let around = hz
            .iter()
            .take_while(|f| **f < top)
            .map(|centre| {
                nonzero(
                    hz.iter()
                        .map(|f| hearing::roex_around(*centre, *f))
                        .collect(),
                )
            })
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
            .map(|centre| {
                let r = range_of(*centre);
                let lower = hz.partition_point(|f| f < centre);
                let upper = nonzero(
                    hz[lower..]
                        .iter()
                        .map(|f| {
                            hearing::roex(hearing::roex_p(*centre, *f, 0.0) / scale, *centre, *f)
                        })
                        .collect(),
                );
                let bands = modulation::bank(*centre, hop_secs)
                    .into_iter()
                    .zip(rates[r].iter().copied())
                    .collect();
                let p51 = hearing::p51(*centre);
                Channel {
                    hz: *centre,
                    range: r,
                    threshold: decibels(hearing::threshold_excitation_db(*centre)),
                    lower,
                    lower_slope: (p51, hearing::lower_slope(p51)),
                    below: hz[..lower].iter().map(|f| (f - centre).abs()).collect(),
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
            hz,
            around,
            channels,
        }
    }

    fn supplying(&self) -> impl Iterator<Item = &Range> {
        self.ranges.iter().filter(|r| !r.supplies.is_empty())
    }

    /// Frame `m`'s one spectrum, as Glasberg and Moore (2002) compose it: what `per` answers
    /// each range's frame with, one value per bin of its `Frame::hz()`, in the order of `hz()`.
    pub fn compose<T>(&self, m: i64, per: &mut dyn FnMut(&Frame) -> Vec<T>) -> Vec<T> {
        let t = m as f64 * self.hop_secs;
        let mut out = Vec::with_capacity(self.hz.len());
        for range in self.supplying() {
            let bins = per(&range.frame(self.rate, t));
            assert_eq!(
                bins.len(),
                range.supplies.len(),
                "one value per supplied bin"
            );
            out.extend(bins);
        }
        out
    }

    /// The composed spectrum's bins, in Hz.
    pub fn hz(&self) -> &[f64] {
        &self.hz
    }

    /// E per channel at frame `m` of `power`, `|X|^2` per bin of the composed spectrum as
    /// `compose` answers it.
    pub fn excite(&self, m: i64, power: &[f64]) -> Vec<f64> {
        let (eardrum, levels) = self.heard(m, power);
        self.channels
            .iter()
            .map(|c| c.excitation(self.scale, &eardrum, &levels))
            .collect()
    }

    /// E per channel of each of `parts`, through the filters `power`'s levels set.
    pub fn excite_each(&self, m: i64, power: &[f64], parts: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (_, levels) = self.heard(m, power);
        let toward: Vec<f64> = self
            .factors(m)
            .zip(power)
            .map(|(f, p)| match *p > 0.0 {
                true => f,
                false => 0.0,
            })
            .collect();
        let rows: Vec<(usize, Vec<f64>)> = self
            .channels
            .iter()
            .map(|c| c.weights(self.scale, &toward, &levels))
            .collect();
        parts
            .iter()
            .map(|part| {
                rows.iter()
                    .map(|(from, row)| row.iter().zip(&part[*from..]).map(|(w, p)| w * p).sum())
                    .collect()
            })
            .collect()
    }

    fn factors(&self, m: i64) -> impl Iterator<Item = f64> + '_ {
        let t = m as f64 * self.hop_secs;
        self.supplying().flat_map(move |range| {
            let energy = range.energy(self.rate, t);
            range.gain[range.supplies.clone()]
                .iter()
                .map(move |g| match energy > 0.0 {
                    true => g / energy,
                    false => 0.0,
                })
        })
    }

    /// Each bin's power at the eardrum, beside its level per ERB.
    fn heard(&self, m: i64, power: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let t = m as f64 * self.hop_secs;
        let gains = self.supplying().flat_map(|range| {
            range.gain[range.supplies.clone()]
                .iter()
                .zip(std::iter::repeat(range.energy(self.rate, t)))
        });
        let eardrum: Vec<f64> = power
            .iter()
            .zip(gains)
            .map(|(p, (g, energy))| match energy > 0.0 {
                true => p / energy * g,
                false => 0.0,
            })
            .collect();
        let levels: Vec<f64> = self
            .around
            .iter()
            .map(|(from, row)| {
                let sum: f64 = row.iter().zip(&eardrum[*from..]).map(|(w, p)| w * p).sum();
                10.0 * libm::log10(sum)
            })
            .collect();
        (eardrum, levels)
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

    /// Each channel's bands' outputs, before `Band::element`, of `excitation`'s E per frame.
    pub fn elements(
        &self,
        len: usize,
        excitation: &mut dyn FnMut(i64) -> Vec<f64>,
    ) -> Vec<Vec<Vec<C64>>> {
        let (first, last) = self.frames(len);
        let mut levels = vec![Vec::new(); self.channels.len()];
        for m in first..=last {
            for ((c, e), v) in self.channels.iter().zip(excitation(m)).zip(&mut levels) {
                v.push(c.level(e));
            }
        }
        let from = (-first) as usize;
        self.channels
            .iter()
            .zip(&levels)
            .map(|(c, v)| modulation::outputs(v, from, self.hop_secs, &c.bands))
            .collect()
    }

    /// The frame element `j` of channel `c`'s band `k` reads.
    pub fn sampled(&self, c: usize, k: usize, j: usize) -> i64 {
        modulation::sampled(j, self.channels[c].bands[k].1.per_frame)
    }

    /// `|H(nu)|^2` from E (in V's units) to channel `c`'s band `k`'s output: the squared
    /// window's normalised transform, squared, then the stage's digital filters.
    pub fn kernel(&self, c: usize, k: usize, nu: f64) -> f64 {
        let channel = &self.channels[c];
        let w = rate::window_squared(nu * self.ranges[channel.range].window_secs);
        w * w * channel.bands[k].0.digital(nu, self.hop_secs)
    }

    pub fn read(&self, planes: &[&[f64]], start_secs: f64) -> Internal {
        let outputs = self.elements(super::held(planes), &mut |m| {
            self.excite(m, &self.compose(m, &mut |frame| frame.power(planes, 0)))
        });
        self.reading(outputs, start_secs)
    }

    pub fn reading(&self, outputs: Vec<Vec<Vec<C64>>>, start_secs: f64) -> Internal {
        let channels = self
            .channels
            .iter()
            .zip(outputs)
            .map(|(c, bands)| Heard {
                hz: c.hz,
                window_secs: self.ranges[c.range].window_secs,
                bands: bands
                    .into_iter()
                    .zip(&c.bands)
                    .map(|(ys, (band, rate))| Sampled {
                        band: *band,
                        rate_hz: rate.per_frame / self.hop_secs,
                        rate_bound_hz: rate.bound / self.hop_secs,
                        elements: ys.into_iter().map(|y| band.element(y)).collect(),
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
