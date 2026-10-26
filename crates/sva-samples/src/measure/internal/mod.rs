// Concern: the reading `internal` at scale s: excitation per ERB-number channel, its level over threshold, the modulation bands | Non-concern: the curves (hearing.rs) | IO: (planes, spl, s) -> Internal

pub mod modulation;
mod rate;

use std::collections::HashMap;
use std::f64::consts::{LN_10, PI};
use std::sync::{Arc, Mutex};

use super::hearing::{
    self, CAM_FIRST, CAM_LAST, CAM_SPACING, HOP_SECS, WINDOW_SECS, WINDOW_SPLITS_HZ,
};
use crate::fft;
use modulation::Band;
use rate::Rate;
use sva_formula::{C64, Num};

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
    /// Windows kept where asked.
    windows: Option<Mutex<Windows>>,
}

/// One Glasberg and Moore (2002) range: its window, its transform's bins and the bins of
/// those it supplies to the composed spectrum, the ones in its own range.
struct Range {
    window_secs: f64,
    n: usize,
    twiddles: Vec<(f64, f64)>,
    /// `e^(-2 pi i j k/n)` per supplied bin `j`.
    basis: Vec<Vec<(f64, f64)>>,
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
        let basis = supplies
            .clone()
            .map(|j| {
                (0..n)
                    .map(|k| {
                        let turn = 2.0 * PI * ((j * k) % n) as f64 / n as f64;
                        let (s, c) = libm::sincos(turn);
                        (c, -s)
                    })
                    .collect()
            })
            .collect();
        Range {
            window_secs,
            n,
            twiddles: fft::twiddles(n),
            basis,
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

    /// The window centred at `t`.
    fn window(&self, rate: f64, t: f64) -> Window {
        let (first, last) = self.span(rate, t);
        let weights: Vec<f64> = (first..=last).map(|s| self.weight(rate, t, s)).collect();
        let energy = weights.iter().map(|w| w * w).sum();
        Window {
            first,
            weights,
            energy,
        }
    }
}

/// Windows by range and frame.
type Windows = HashMap<(usize, i64), Arc<Window>>;

/// A range's window at a frame: its first sample, weights, energy.
struct Window {
    first: i64,
    weights: Vec<f64>,
    energy: f64,
}

/// One range's window at one frame: the samples it weighs and the bins it reads, so samples or
/// a closed form put their power per bin through the ear alike.
pub struct Frame<'a> {
    range: &'a Range,
    window: Arc<Window>,
}

impl Frame<'_> {
    /// The first sample the window weighs; `weights()[k]` weighs sample `first() + k`.
    pub fn first(&self) -> i64 {
        self.window.first
    }

    pub fn weights(&self) -> &[f64] {
        &self.window.weights
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

    /// The weighted transform of `x`, sample `first() + k` at `k`.
    pub fn transform(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut re = vec![0.0; self.range.n];
        for (k, (w, x)) in self.window.weights.iter().zip(x).enumerate() {
            re[k] = w * x;
        }
        let mut im = vec![0.0; self.range.n];
        fft::fft_with(&mut re, &mut im, &self.range.twiddles);
        (re, im)
    }

    /// `transform` at `bins()` alone, by direct sums.
    pub fn transform_bins(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let weighted: Vec<f64> = self
            .window
            .weights
            .iter()
            .zip(x)
            .map(|(w, x)| w * x)
            .collect();
        self.range
            .basis
            .iter()
            .map(|basis| {
                weighted
                    .iter()
                    .zip(basis)
                    .fold((0.0, 0.0), |(re, im), (v, (c, s))| (re + v * c, im + v * s))
            })
            .unzip()
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
            for (k, w) in self.window.weights.iter().enumerate() {
                let i = self.window.first + k as i64 - from;
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
    fn skirt<N: Num>(&self, scale: f64, i: usize, x: N) -> N {
        let (p51, slope) = self.lower_slope;
        hearing::roex_apart(
            hearing::lower_p(p51, slope, x) / scale,
            self.below[i],
            self.hz,
        )
    }

    /// `excitation`'s slopes along each eardrum move, `slopes` its levels'.
    fn excitation_along(
        &self,
        scale: f64,
        (power, levels): (&[f64], &[f64]),
        along: &[Vec<f64>],
        slopes: &[Vec<f64>],
    ) -> Vec<f64> {
        let (p51, slope) = self.lower_slope;
        let skirts: Vec<(f64, f64)> = (0..self.lower)
            .map(|i| match power[i] > 0.0 {
                true => {
                    let p = hearing::lower_p(p51, slope, levels[i]);
                    let (v, dp) = hearing::roex_apart_slope(p / scale, self.below[i], self.hz);
                    (
                        v,
                        dp / scale * hearing::lower_p_slope(p51, slope, levels[i]),
                    )
                }
                false => (0.0, 0.0),
            })
            .collect();
        let (from, weights) = &self.upper;
        along
            .iter()
            .zip(slopes)
            .map(|(d, dx)| {
                let mut below = 0.0;
                for (i, (v, dv)) in skirts.iter().enumerate() {
                    if power[i] > 0.0 {
                        below += v * d[i] + dv * dx[i] * power[i];
                    }
                }
                let above: f64 = weights.iter().zip(&d[*from..]).map(|(w, p)| p * w).sum();
                below + above
            })
            .collect()
    }

    fn excitation<N: Num>(&self, scale: f64, power: &[N], levels: &[N]) -> N {
        let mut below = N::ZERO;
        for i in 0..self.lower {
            if power[i].v() > 0.0 {
                below = below + self.skirt(scale, i, levels[i]) * power[i];
            }
        }
        let (from, weights) = &self.upper;
        let above = weights
            .iter()
            .zip(&power[*from..])
            .fold(N::ZERO, |sum, (w, p)| sum + *p * *w);
        below + above
    }

    /// `V`, its tangent through `ln` beside the value `log1p` gives.
    fn level<N: Num>(&self, excitation: N) -> N {
        let value = 10.0 / LN_10 * libm::log1p(excitation.v() / self.threshold);
        ((excitation / self.threshold + 1.0).ln() * (10.0 / LN_10)).rebased(value)
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
            windows: None,
        }
    }

    /// The ear keeping each window it reads.
    pub fn remembering(mut self) -> Ear {
        self.windows = Some(Mutex::default());
        self
    }

    /// Each range supplying bins, by index.
    fn supplying(&self) -> impl Iterator<Item = (usize, &Range)> {
        self.ranges
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.supplies.is_empty())
    }

    /// Range `r`'s window at frame `m`.
    fn window(&self, r: usize, m: i64) -> Arc<Window> {
        let read = || Arc::new(self.ranges[r].window(self.rate, m as f64 * self.hop_secs));
        let Some(windows) = &self.windows else {
            return read();
        };
        if let Some(w) = windows.lock().expect("the windows").get(&(r, m)) {
            return w.clone();
        }
        let w = read();
        windows
            .lock()
            .expect("the windows")
            .insert((r, m), w.clone());
        w
    }

    /// Frame `m`'s one spectrum, as Glasberg and Moore (2002) compose it: what `per` answers
    /// each range's frame with, one value per bin of its `Frame::hz()`, in the order of `hz()`.
    pub fn compose<T>(&self, m: i64, per: &mut dyn FnMut(&Frame) -> Vec<T>) -> Vec<T> {
        let mut out = Vec::with_capacity(self.hz.len());
        for (r, range) in self.supplying() {
            let bins = per(&Frame {
                range,
                window: self.window(r, m),
            });
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
    /// `compose` answers it; a tangent `N` carries its derivative, the lower skirts' `p` moving
    /// with `X`.
    pub fn excite<N: Num>(&self, m: i64, power: &[N]) -> Vec<N> {
        let gains = self.supplying().flat_map(|(r, range)| {
            range.gain[range.supplies.clone()]
                .iter()
                .zip(std::iter::repeat(self.window(r, m).energy))
        });
        let eardrum: Vec<N> = power
            .iter()
            .zip(gains)
            .map(|(p, (g, energy))| match energy > 0.0 {
                true => *p / energy * *g,
                false => N::ZERO,
            })
            .collect();
        let levels: Vec<N> = self
            .around
            .iter()
            .map(|(from, row)| {
                let sum = row
                    .iter()
                    .zip(&eardrum[*from..])
                    .fold(N::ZERO, |sum, (w, p)| sum + *p * *w);
                (sum.ln() * (10.0 / LN_10)).rebased(10.0 * libm::log10(sum.v()))
            })
            .collect();
        self.channels
            .iter()
            .map(|c| c.excitation(self.scale, &eardrum, &levels))
            .collect()
    }

    /// `excite`'s slopes along each move of `power`, as a tangent `N` reads them, the skirts
    /// read once.
    pub fn excite_along(&self, m: i64, power: &[f64], along: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let gains: Vec<f64> = self
            .supplying()
            .flat_map(|(r, range)| {
                let energy = self.window(r, m).energy;
                range.gain[range.supplies.clone()]
                    .iter()
                    .map(move |g| match energy > 0.0 {
                        true => g / energy,
                        false => 0.0,
                    })
            })
            .collect();
        let eardrum: Vec<f64> = power.iter().zip(&gains).map(|(p, g)| p * g).collect();
        let sums: Vec<f64> = self
            .around
            .iter()
            .map(|(from, row)| {
                row.iter()
                    .zip(&eardrum[*from..])
                    .fold(0.0, |s, (w, p)| s + p * w)
            })
            .collect();
        let levels: Vec<f64> = sums.iter().map(|s| 10.0 * libm::log10(*s)).collect();
        let moved: Vec<Vec<f64>> = along
            .iter()
            .map(|d| d.iter().zip(&gains).map(|(d, g)| d * g).collect())
            .collect();
        let slopes: Vec<Vec<f64>> = moved
            .iter()
            .map(|d: &Vec<f64>| {
                self.around
                    .iter()
                    .zip(&sums)
                    .map(|((from, row), s)| {
                        let ds = row.iter().zip(&d[*from..]).fold(0.0, |a, (w, p)| a + p * w);
                        10.0 / LN_10 * ds / s
                    })
                    .collect()
            })
            .collect();
        let mut out = vec![vec![0.0; self.channels.len()]; along.len()];
        for (c, channel) in self.channels.iter().enumerate() {
            let each = channel.excitation_along(self.scale, (&eardrum, &levels), &moved, &slopes);
            for (o, v) in out.iter_mut().zip(each) {
                o[c] = v;
            }
        }
        out
    }

    /// `levels`' slopes along `d`: `10 log10(e)/(E_T + E)`.
    pub fn levels_along(&self, excitation: &[f64], d: &[f64]) -> Vec<f64> {
        self.channels
            .iter()
            .zip(excitation.iter().zip(d))
            .map(|(c, (e, d))| 10.0 / LN_10 * (d / c.threshold) / (e / c.threshold + 1.0))
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

    /// Each channel's bands' outputs, before `Band::element`, of `excitation`'s E per frame.
    pub fn elements(
        &self,
        len: usize,
        excitation: &mut dyn FnMut(i64) -> Vec<f64>,
    ) -> Vec<Vec<Vec<C64>>> {
        let (first, last) = self.frames(len);
        let mut levels = vec![Vec::new(); self.channels.len()];
        for m in first..=last {
            for (e, v) in self.levels(&excitation(m)).into_iter().zip(&mut levels) {
                v.push(e);
            }
        }
        self.filtered(&levels)
    }

    /// V per channel of E per channel.
    pub fn levels<N: Num>(&self, excitation: &[N]) -> Vec<N> {
        self.channels
            .iter()
            .zip(excitation)
            .map(|(c, e)| c.level(*e))
            .collect()
    }

    /// Each channel's bands' outputs of `levels[c]`, its V at every frame from the first: the
    /// stage is linear from rest, so a tangent of V passes as V does.
    pub fn filtered(&self, levels: &[Vec<f64>]) -> Vec<Vec<Vec<C64>>> {
        let from = (-self.frames(0).0) as usize;
        self.channels
            .iter()
            .zip(levels)
            .map(|(c, v)| modulation::outputs(v, from, self.hop_secs, &c.bands))
            .collect()
    }

    /// Channel `c`'s outputs of a V silent but at `at..at + levels.len()`, read there: per band,
    /// its first element there and the outputs on.
    pub fn filtered_within(
        &self,
        c: usize,
        levels: &[f64],
        (at, frames): (usize, usize),
    ) -> Vec<(usize, Vec<C64>)> {
        let from = (-self.frames(0).0) as usize;
        let channel = &self.channels[c];
        modulation::outputs_within(levels, (at, frames), from, self.hop_secs, &channel.bands)
    }

    /// `filtered`'s adjoint: `w` per frame with `sum Re(conj(z) dy) = sum w dV`.
    pub fn pulled(&self, z: &[Vec<Vec<C64>>], frames: usize) -> Vec<Vec<f64>> {
        let from = (-self.frames(0).0) as usize;
        self.channels
            .iter()
            .zip(z)
            .map(|(c, z)| modulation::pulled(z, frames, from, self.hop_secs, &c.bands))
            .collect()
    }

    /// The first and last sample any window of frame `m` weighs.
    pub fn reach(&self, m: i64) -> (i64, i64) {
        let t = m as f64 * self.hop_secs;
        self.supplying()
            .map(|(_, r)| r.span(self.rate, t))
            .fold((i64::MAX, i64::MIN), |(a, b), (f, l)| (a.min(f), b.max(l)))
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
