// Concern: FIT: one sound by forward-backward greedy selection under one DL, then a pass on its residual | Non-concern: the analysis (add.rs), JSON | IO: (samples, rate, spl) -> Fit, Lines

use std::f64::consts::LOG2_E;
use std::ops::Range;

use sva_engine::hearing::{CAM_LAST, cam, cam_hz, erb_hz, level_jnd_db};
use sva_engine::{Abandon, default_threads, each, fft};

use super::add::{Members, Spectrogram, Track, floor_db, sieved, span_hz};
use super::atoms::{Atom, Clip, Feasible, Quantity, Trajectory, birth};
use super::bits::{self, Look, Priced, frequency_delta, level_delta};
use super::grid::Grid;

pub const MODEL: &str = "one sound fitted by forward-backward greedy selection with an orthogonal re-solve (Tropp 2004, doi 10.1109/TIT.2004.834793; Zhang 2011, doi 10.1109/TIT.2011.2146690); \
     DL = B(x) (bits.rs) plus the residual's bits, the one objective; \
     every atom spans the clip, only its level moving; \
     the clip read at its rate over the largest whole factor keeping Cam 38.9 under the Nyquist; \
     scaled to unit RMS, one master gain in dB bringing it back, every floor read at the SPL plus that gain; \
     the residual on the coarse grid's 48 ms frames by its analysis bins (grid.rs), octave bands being too coarse to tell a sine from noise inside one; \
     a cell's power P the averaged periodogram of K looks, K the frame over half the window times the window over the padded transform (Welch 1967); \
     a cell's variance that of a sine of power P_s in noise of power N: a bin's X = s + n, n complex Gaussian with E|n|^2 = N, so |X|^2 = |s|^2 + 2 Re(s* n) + |n|^2 is noncentral chi-square on 2 degrees, its variance N^2 + 2 P_s N per look, (N^2 + 2 P_s N) / K over K; \
     N = P_n + F, the floor F a noise the ear's own; inside a tonal lobe P_n the local noise floor and P_s = max(P - P_n, 0); outside one P_n = P and P_s = 0, so a noise cell's variance stays (P + F)^2 / K and a tonal cell's falls to its true one; \
     the local noise floor a bin's median over one Cam (at least a main lobe's width either side of its own lobe), outside the tonal lobes, on each side of the bin's own main lobe, over the median's share of the mean of K looks, (1 - 1/(9K))^3 (Wilson and Hilferty 1931), the higher side; \
     a tonal lobe the main lobe (2 window bins either side) of a local maximum resolved from every higher bin within a main lobe's width and over 2 ln M times that floor, M the feasible bins: an exponential's tail, the periodogram's at one look and a bound at more, so noise alone passes in one bin with probability 1/M^2 and holds a false lobe in a spectrum with under 1/M; \
     its lobes found first against the lower side's median over all bins, then barred from the floor read against the higher; \
     F a noise at threshold: its power in one ERB a sine's at its threshold (add.rs), so per bin that sine's peak-bin power times f_s sum w^2 / (ERB (sum w)^2), the sine's own peak-bin power overstating a threshold noise's by 2 to 17 dB from 100 Hz to 10 kHz; \
     residual bits 1/2 sum w r^2 log2(e), w the inverse of that variance, r = P less the model's expected power, bins outside the feasible set unweighted; \
     templates in expected power: a sine through the window's closed-form kernel, a noise band's expected periodogram (grid.rs); \
     each column's gain per frame its power; \
     candidates off the residual averaged over the frames; \
     a free partial at each local maximum over its floor, its Hz by QIFFT (Smith and Serra 1987); \
     a voice at each fundamental the sieve (add.rs) finds among those peaks that are tonal lobes of the averaged residual at K times its frames looks, taken from the pool with its members till none is found; \
     a voice's harmonic k at k f0 sqrt(1 + B k^2) (Fletcher 1964), f0 and B by least squares of (f_k / k)^2 = f0^2 + f0^2 B k^2 over its member peaks, B at least 0; \
     members reassigned within half an analysis bin of the stretched Hz and refitted till the set repeats; \
     every harmonic to the highest member its own column and its own level; \
     one noise band per octave band holding none; \
     a candidate scored in closed form: each column's gain per frame its weighted correlation with the residual over its weighted energy, at least 0, written, the DL drop those gains give less its bits; \
     each candidate's frequencies refined against the residual: a partial's Hz by golden section (Kiefer 1953) over half an analysis bin of its peak to the finer of the precision it is written at and its standard deviation under the weights, sigma = (sum w (du/dx)^2)^-1/2, u its expected power, du/dx by central difference at the cube root of epsilon; a voice's harmonics each so as a partial, then f0 and B by that least squares weighted by the inverse variance of (f_k / k)^2, (2 f_k sigma_k / k^2)^-2, kept where the voice's score falls; \
     the best added where that drop is positive, then the new atom's frequencies and those of each kept atom whose templates meet its bins refined in turn against the clip less every other atom's model at the re-solved gains, each kept where the re-solved DL falls, an atom's bins each column's main lobe (2 window bins either side of a partial's, a noise band's band widened by as much), the rest left as they stand; \
     every kept column re-solved per frame by non-negative least squares (Lawson and Hanson 1974), warm-started from the last gains; \
     an add kept only where the written state's DL falls; \
     then the best removal taken while one, re-solved, lowers the DL; \
     stopped where no candidate lowers the DL, so the DL falls at every step and no state repeats; \
     a column's level 10 log10 of its gain, held at its floor; \
     each level through the frame centres from 0 to the clip's end, thinned by Ramer-Douglas-Peucker at the level JND at its peak sensation level (Jesteadt, Wier and Green 1977); \
     the DL the written atoms': levels at the level JND by looks, frequencies at the frequency JND by looks (bits.rs), every frame a look at its level over its floor; \
     a voice's f0 at the JND at f0, its B where its highest harmonic moves by that harmonic's JND; \
     a time at the frame, a noise band's edge one of the grid's band edges; \
     then a second pass, a diagnostic of what the first's scoring missed: the same loop from no atom over the first's residual on the grid it scores, the clip's power less the fit's per bin floored at 0, its own weights read off that residual, its atoms keyed after the first's and sounded on top of them; \
     each pass's steps reported as it goes, its pass beside each";

/// A full-scale sine at 103.95 dB SPL by default (SMPTE RP 200: -20 dBFS pink noise at 83 dB
/// SPL, a sine's RMS 3.01 dB under its peak).
pub const SPL_DB: f64 = 103.95;

/// The planes' mean, the one channel the fit reads, padded with silence to the longest.
pub fn mono(planes: &[Vec<f32>]) -> Vec<f64> {
    let len = planes.iter().map(Vec::len).max().unwrap_or(0);
    let width = planes.len().max(1) as f64;
    (0..len)
        .map(|n| {
            planes
                .iter()
                .map(|p| f64::from(p.get(n).copied().unwrap_or(0.0)))
                .sum::<f64>()
                / width
        })
        .collect()
}

/// The rate `rate` is read at, as `MODEL` states.
pub fn rate_for(rate: u32) -> u32 {
    let need = 2.0 * cam_hz(CAM_LAST);
    let most = (f64::from(rate) / need) as u32;
    (1..=most.max(1))
        .rev()
        .find(|d| rate.is_multiple_of(*d))
        .map_or(rate, |d| rate / d)
}

/// `samples` read at `to`, a whole factor under `rate`, by keeping the bins under its Nyquist.
fn decimated(samples: &[f64], rate: u32, to: u32) -> Vec<f64> {
    let d = (rate / to) as usize;
    if d == 1 || samples.is_empty() {
        return samples.to_vec();
    }
    let n = samples.len().div_ceil(d) * d;
    let mut x = samples.to_vec();
    x.resize(n, 0.0);
    let (re, im) = fft::dft(&x, &vec![0.0; n]);
    let m = n / d;
    let (mut kept_re, mut kept_im) = (vec![0.0; m], vec![0.0; m]);
    for k in 0..m {
        let from = match (2 * k < m, 2 * (m - k) < m) {
            (true, _) => Some(k),
            (_, true) => Some(n - (m - k)),
            _ => None,
        };
        if let Some(j) = from {
            (kept_re[k], kept_im[k]) = (re[j], im[j]);
        }
    }
    let (y, _) = fft::idft(&kept_re, &kept_im);
    y.iter().map(|v| v / d as f64).collect()
}

/// What a step did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Start,
    Add,
    Remove,
}

impl Action {
    pub fn name(self) -> &'static str {
        match self {
            Action::Start => "start",
            Action::Add => "add",
            Action::Remove => "remove",
        }
    }
}

/// The coarse grid's shape, which the first line carries.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub rate_hz: u32,
    pub frame_secs: f64,
    pub first_centre_secs: f64,
    pub band_edges_hz: Vec<(f64, f64)>,
    pub band_floor_db: Vec<f64>,
    pub gain_db: f64,
}

/// One step as a progress line reports it: the atom added or removed, the DL's change and
/// total, the residual's band levels per frame (`None` where the fit covers the clip) and each
/// kept atom's written level per frame, harmonic by harmonic.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    /// The pass, 1 or 2.
    pub pass: u8,
    pub step: usize,
    pub action: Action,
    pub atom: Option<Atom>,
    pub delta_bits: f64,
    pub atom_bits: f64,
    pub residual_bits: f64,
    pub residual_db: Vec<Vec<Option<f64>>>,
    pub envelopes: Vec<(u64, Vec<Vec<f64>>)>,
    pub geometry: Option<Geometry>,
    pub seconds: f64,
}

/// A fit: the clip it read, its atoms in unit-RMS space, the master gain bringing them back,
/// the SPL that space is heard at, its DL's parts, its steps, whether a stop ended it, and the
/// clip's and the fit's coarse grids, each band's power spread flat over its bins.
pub struct Fit {
    pub clip: Clip,
    pub atoms: Vec<Atom>,
    pub gain_db: f64,
    pub spl_db: f64,
    pub atom_bits: f64,
    pub residual_bits: f64,
    pub steps: usize,
    pub stopped: bool,
    pub clip_grid: Grid,
    pub fit_grid: Grid,
    pub second: Second,
}

/// The second pass, as `MODEL` states: its atoms, keyed after the first's, its DL's parts
/// against the first's residual and that residual's DL with no atom, its steps, and whether a
/// stop ended it.
#[derive(Default)]
pub struct Second {
    pub atoms: Vec<Atom>,
    pub start_bits: f64,
    pub atom_bits: f64,
    pub residual_bits: f64,
    pub steps: usize,
    pub stopped: bool,
}

/// What an atom is before its level is written.
#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Partial { hz: f64 },
    Voice { f0: f64, stretch: f64, count: usize },
    Noise { band: (f64, f64) },
}

impl Kind {
    /// Each column's Hz: a noise band's its geometric centre.
    fn hzs(&self) -> Vec<f64> {
        match self {
            Kind::Partial { hz } => vec![*hz],
            Kind::Voice { f0, stretch, count } => (1..=*count)
                .map(|k| {
                    let k = k as f64;
                    f0 * k * libm::sqrt(1.0 + stretch * k * k)
                })
                .collect(),
            Kind::Noise { band } => vec![libm::sqrt(band.0 * band.1)],
        }
    }
}

/// The bins `kind`'s columns reach in `p`, each its main lobe as `MODEL` states, in grid bins.
fn reach(p: &Problem, kind: &Kind) -> Vec<(f64, f64)> {
    let half = 2.0 * p.grid.n as f64 / p.grid.long as f64;
    match kind {
        Kind::Noise { band } => vec![(p.grid.bin_of(band.0) - half, p.grid.bin_of(band.1) + half)],
        _ => kind
            .hzs()
            .iter()
            .map(|hz| (p.grid.bin_of(*hz) - half, p.grid.bin_of(*hz) + half))
            .collect(),
    }
}

/// Whether any of `a`'s reaches meets any of `b`'s.
fn meets(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
    a.iter()
        .any(|(lo, hi)| b.iter().any(|(l, h)| lo < h && l < hi))
}

/// The local noise floor and the tonal lobes over the feasible bins, as `MODEL` states: each
/// bin's median window, the main lobe's half-width in bins, and the test's ratio.
struct Lobes {
    windows: Vec<Range<usize>>,
    half: usize,
    ratio: f64,
}

impl Lobes {
    fn of(grid: &Grid, range: &Range<usize>) -> Lobes {
        let half = (2 * grid.n).div_ceil(grid.long);
        let count = range.len();
        let windows = (0..count)
            .map(|i| {
                let c = cam(grid.hz(range.start + i));
                let at = |cams: f64| grid.bin_of(cam_hz(cams)) - range.start as f64;
                let lo =
                    (libm::ceil(at(c - 0.5)).max(0.0) as usize).min(i.saturating_sub(2 * half));
                let hi = (libm::floor(at(c + 0.5)) as usize + 1).max(i + 2 * half + 1);
                lo..hi.min(count)
            })
            .collect();
        Lobes {
            windows,
            half,
            ratio: 2.0 * libm::log(count.max(1) as f64),
        }
    }

    /// `spectrum`'s noise floor per bin at `looks` looks a bin, and the tonal peaks, as
    /// `MODEL` states: on each side of the bin's own lobe the median over its window outside
    /// the tonal lobes over the median's share of the mean, the higher side the floor.
    fn floor(&self, spectrum: &[f64], looks: f64) -> (Vec<f64>, Vec<usize>) {
        let share = libm::pow(1.0 - 1.0 / (9.0 * looks), 3.0);
        let median = |bins: Range<usize>, barred: &[bool]| -> Option<f64> {
            let mut held: Vec<f64> = bins.filter(|j| !barred[*j]).map(|j| spectrum[j]).collect();
            let middle = held.len() / 2;
            (!held.is_empty())
                .then(|| *held.select_nth_unstable_by(middle, f64::total_cmp).1 / share)
        };
        let medians = |barred: &[bool], side: fn(f64, f64) -> f64| -> Vec<f64> {
            self.windows
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    let left = median(w.start..i.saturating_sub(self.half).max(w.start), barred);
                    let right = median((i + self.half + 1).min(w.end)..w.end, barred);
                    match (left, right) {
                        (Some(l), Some(r)) => side(l, r),
                        (Some(v), None) | (None, Some(v)) => v,
                        (None, None) => {
                            w.clone().map(|j| spectrum[j]).fold(f64::INFINITY, f64::min)
                        }
                    }
                })
                .collect()
        };
        let tonal = |floor: &[f64]| -> Vec<usize> {
            (1..spectrum.len().saturating_sub(1))
                .filter(|i| {
                    let (a, c, e) = (spectrum[i - 1], spectrum[*i], spectrum[i + 1]);
                    let near = (i + 1).saturating_sub(2 * self.half)
                        ..(i + 2 * self.half).min(spectrum.len());
                    c > a
                        && c >= e
                        && c > self.ratio * floor[*i]
                        && near.clone().all(|j| spectrum[j] <= c)
                })
                .collect()
        };
        let first = medians(&vec![false; spectrum.len()], f64::min);
        let floor = medians(&self.inside(&tonal(&first), spectrum.len()), f64::max);
        let peaks = tonal(&floor);
        (floor, peaks)
    }

    /// Whether each of `count` bins lies in the main lobe of a peak among `peaks`.
    fn inside(&self, peaks: &[usize], count: usize) -> Vec<bool> {
        let mut out = vec![false; count];
        for i in peaks {
            let span = i.saturating_sub(self.half)..(i + self.half + 1).min(count);
            out[span].iter_mut().for_each(|b| *b = true);
        }
        out
    }
}

/// The clip on its grid, as `MODEL` states: power, weights and their product with the power
/// over the feasible bins, frame by frame.
struct Problem {
    grid: Grid,
    clip: Clip,
    spl_db: f64,
    top_db: f64,
    range: Range<usize>,
    /// K, the looks one cell holds.
    looks: f64,
    lobes: Lobes,
    power: Vec<Vec<f64>>,
    weight: Vec<Vec<f64>>,
    /// `sum w P^2` per frame.
    energy: Vec<f64>,
    times: Vec<f64>,
}

impl Problem {
    /// `grid` read as `MODEL` states, its power the target.
    fn of(grid: Grid, clip: Clip, spl_db: f64, gain_db: f64) -> Problem {
        let (lo, hi) = span_hz(grid.rate);
        let range = (libm::ceil(grid.bin_of(lo)) as usize).max(1)
            ..(libm::floor(grid.bin_of(hi)) as usize + 1).min(grid.bins() - 1);
        let looks = grid.hop as f64 / (grid.long as f64 / 2.0) * grid.long as f64 / grid.n as f64;
        let floors: Vec<f64> = range
            .clone()
            .map(|b| {
                let hz = grid.hz(b);
                let amplitude = libm::pow(10.0, floor_db(hz, spl_db) / 20.0);
                amplitude * amplitude * grid.rate * grid.sum_w2 / (4.0 * erb_hz(hz))
            })
            .collect();
        let power: Vec<Vec<f64>> = grid
            .power
            .iter()
            .map(|f| f[range.clone()].to_vec())
            .collect();
        let lobes = Lobes::of(&grid, &range);
        let weight: Vec<Vec<f64>> = each(default_threads(), (0..power.len()).collect(), |f| {
            let (noise, tonal) = lobes.floor(&power[f], looks);
            let inside = lobes.inside(&tonal, power[f].len());
            power[f]
                .iter()
                .zip(&noise)
                .zip(&floors)
                .zip(&inside)
                .map(|(((p, pn), fl), inside)| {
                    let pn = match inside {
                        true => pn.max(0.0),
                        false => *p,
                    };
                    let n = pn + fl;
                    let ps = (p - pn).max(0.0);
                    looks / (n * n + 2.0 * ps * n)
                })
                .collect()
        });
        let energy = power
            .iter()
            .zip(&weight)
            .map(|(p, w)| p.iter().zip(w).map(|(p, w)| w * p * p).sum())
            .collect();
        let times = (0..grid.power.len()).map(|f| grid.centre_secs(f)).collect();
        Problem {
            grid,
            clip,
            spl_db,
            top_db: -gain_db,
            range,
            looks,
            lobes,
            power,
            weight,
            energy,
            times,
        }
    }

    fn frames(&self) -> usize {
        self.power.len()
    }

    /// Each column's template over the feasible bins.
    fn templates(&self, kind: &Kind) -> Vec<Vec<f64>> {
        match kind {
            Kind::Noise { band } => vec![self.grid.noise_bins(*band)[self.range.clone()].to_vec()],
            _ => kind
                .hzs()
                .iter()
                .map(|hz| self.range.clone().map(|b| self.grid.sine(*hz, b)).collect())
                .collect(),
        }
    }

    /// `sum_b w a b` in frame `f`.
    fn dot(&self, f: usize, a: &[f64], b: &[f64]) -> f64 {
        self.weight[f]
            .iter()
            .zip(a)
            .zip(b)
            .map(|((w, a), b)| w * a * b)
            .sum()
    }

    /// `(sum_b w t^2, sum_b w t r)` in frame `f`, in one pass.
    fn moments(&self, f: usize, t: &[f64], r: &[f64]) -> (f64, f64) {
        self.weight[f]
            .iter()
            .zip(t)
            .zip(r)
            .fold((0.0, 0.0), |(e, c), ((w, t), r)| {
                let wt = w * t;
                (e + wt * t, c + wt * r)
            })
    }

    /// The frames' looks at a level `level` over `floor` at `hz`.
    fn looks(&self, level: &[f64], floor: f64, hz: f64) -> Vec<Look> {
        level
            .iter()
            .map(|l| Look {
                sl_db: l - floor,
                hz,
            })
            .collect()
    }
}

/// An atom written from its gains: the atom, each column's written gain and level in dB per
/// frame, and its bits.
#[derive(Clone)]
struct Written {
    atom: Atom,
    gains: Vec<Vec<f64>>,
    levels: Vec<Vec<f64>>,
    bits: f64,
}

fn db(power: f64) -> f64 {
    10.0 * libm::log10(power)
}

/// `kind` keyed `key` written from `fitted`, each column's gain per frame, as `MODEL` states.
fn write(p: &Problem, kind: &Kind, key: u64, fitted: &[Vec<f64>]) -> Written {
    let secs = p.clip.secs();
    let hzs = kind.hzs();
    let floors: Vec<f64> = hzs.iter().map(|hz| floor_db(*hz, p.spl_db)).collect();
    let trajectories: Vec<Trajectory> = fitted
        .iter()
        .zip(&floors)
        .map(|(g, floor)| {
            let held: Vec<f64> = g.iter().map(|g| db(*g).max(*floor)).collect();
            let peak = held.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            thinned((0.0, secs), &p.times, &held, level_jnd_db(peak - floor))
        })
        .collect();
    let levels: Vec<Vec<f64>> = trajectories
        .iter()
        .map(|t| p.times.iter().map(|at| level_at(t, 0.0, *at)).collect())
        .collect();
    let gains = levels
        .iter()
        .map(|l| l.iter().map(|l| libm::pow(10.0, l / 10.0)).collect())
        .collect();
    let mut trajectories = trajectories.into_iter();
    let first = trajectories.next().expect("a column");
    let atom = match kind {
        Kind::Partial { hz } => Atom::partial(key, 0.0, Trajectory::steady(*hz), first),
        Kind::Voice { f0, stretch, .. } => Atom::harmonic(
            key,
            0.0,
            (Trajectory::steady(*f0), *stretch),
            std::iter::once(first).chain(trajectories).collect(),
        ),
        Kind::Noise { band } => Atom::noise(key, 0.0, *band, first),
    };
    let bits = priced(p, &atom, &hzs, &levels, &floors);
    Written {
        atom,
        gains,
        levels,
        bits,
    }
}

/// `atom`'s bits at the precisions `MODEL` states, harmonic `k` at `hzs[k]` and `levels[k]`
/// per frame over `floors[k]`.
fn priced(p: &Problem, atom: &Atom, hzs: &[f64], levels: &[Vec<f64>], floors: &[f64]) -> f64 {
    let hop = p.grid.frame_secs();
    let looks = |k: usize| p.looks(&levels[k], floors[k], hzs[k]);
    let lowest = floors.iter().copied().fold(f64::INFINITY, f64::min);
    let feasible = Feasible::of(&p.clip, (lowest, p.top_db));
    let edges = p.grid.bands().len() + 1;
    let mut level = 0;
    let mut deltas = Vec::new();
    for (q, _) in atom.numbers() {
        if q == Quantity::Db {
            level += 1;
        }
        let member = level % hzs.len();
        deltas.push(match q {
            Quantity::Time => hop,
            Quantity::Db | Quantity::DbDeviation => level_delta(hop, &looks(member)),
            Quantity::Hz | Quantity::HzDeviation => frequency_steps(p, hzs, levels, floors).0,
            Quantity::Stretch => frequency_steps(p, hzs, levels, floors).1,
            Quantity::Edge => (feasible.hz.1 - feasible.hz.0) / edges as f64,
        });
    }
    bits::atom(atom, &Priced { feasible, deltas })
}

/// The precisions a fundamental's Hz and its stretch are written at, as `MODEL` states,
/// harmonic `k` at `hzs[k]` and `levels[k]` per frame over `floors[k]`.
fn frequency_steps(p: &Problem, hzs: &[f64], levels: &[Vec<f64>], floors: &[f64]) -> (f64, f64) {
    let hop = p.grid.frame_secs();
    let looks = |k: usize| p.looks(&levels[k], floors[k], hzs[k]);
    let last = hzs.len() - 1;
    let (f0, h, top) = (hzs[0], hzs.len() as f64, hzs[last]);
    let moved = h * h * h * h * f0 * f0 / (2.0 * top);
    (
        frequency_delta(hop, &looks(0)),
        frequency_delta(hop, &looks(last)) / moved,
    )
}

/// The least of `score` over `[lo, hi]` by golden section (Kiefer 1953) till the bracket is
/// narrower than `step`, beside its value, `at` and its value `held` kept where none beats it.
fn golden(
    score: &dyn Fn(f64) -> f64,
    (lo, hi): (f64, f64),
    step: f64,
    held: (f64, f64),
) -> (f64, f64) {
    let ratio = (libm::sqrt(5.0) - 1.0) / 2.0;
    let (mut a, mut b) = (lo, hi);
    let mut c = b - ratio * (b - a);
    let mut d = a + ratio * (b - a);
    let (mut sc, mut sd) = (score(c), score(d));
    let mut best = held;
    while b - a > step && step > 0.0 {
        for (x, v) in [(c, sc), (d, sd)] {
            if v < best.1 {
                best = (x, v);
            }
        }
        if sc < sd {
            (b, d, sd) = (d, c, sc);
            c = b - ratio * (b - a);
            sc = score(c);
        } else {
            (a, c, sc) = (c, d, sd);
            d = a + ratio * (b - a);
            sd = score(d);
        }
    }
    for (x, v) in [(c, sc), (d, sd)] {
        if v < best.1 {
            best = (x, v);
        }
    }
    best
}

/// The standard deviation the residual's weights give one number of a kind whose columns hold
/// `gains` per frame, its kind at the number less and plus `h` being `below` and `above`: the
/// inverse root of `sum w (du/dx)^2`, `u` the kind's expected power, `du/dx` by central
/// difference.
fn spread(p: &Problem, (below, above): (&Kind, &Kind), h: f64, gains: &[Vec<f64>]) -> f64 {
    let (lo, hi) = (p.templates(below), p.templates(above));
    let information: f64 = (0..p.frames())
        .map(|f| {
            (0..p.range.len())
                .map(|b| {
                    let du: f64 = lo
                        .iter()
                        .zip(&hi)
                        .zip(gains)
                        .map(|((l, u), g)| g[f] * (u[b] - l[b]))
                        .sum::<f64>()
                        / (2.0 * h);
                    p.weight[f][b] * du * du
                })
                .sum::<f64>()
        })
        .sum();
    1.0 / libm::sqrt(information)
}

/// `kind` with its frequencies refined against `residual` beside its score, as `MODEL` states.
fn refined(p: &Problem, state: &State, residual: &[Vec<f64>], kind: Kind, key: u64) -> (Kind, f64) {
    let bin = p.grid.rate / p.grid.n as f64;
    let score = |k: &Kind| scored(p, state, residual, k, key).0;
    let (best, written) = scored(p, state, residual, &kind, key);
    // Harmonic `c` at `hz` refined as a partial: its Hz, score and standard deviation.
    let partial = |c: usize, hz: f64, held: f64| {
        let h = bin * libm::cbrt(f64::EPSILON);
        let near = (&Kind::Partial { hz: hz - h }, &Kind::Partial { hz: hz + h });
        let sigma = spread(p, near, h, &written.gains[c..=c]);
        let floor = [floor_db(hz, p.spl_db)];
        let written_step = frequency_steps(p, &[hz], &written.levels[c..=c], &floor).0;
        let at = |h: f64| score(&Kind::Partial { hz: h });
        let range = (hz - bin / 2.0, hz + bin / 2.0);
        let (to, v) = golden(&at, range, written_step.min(sigma), (hz, held));
        (to, v, sigma)
    };
    match kind {
        Kind::Partial { hz } => {
            let (hz, v, _) = partial(0, hz, best);
            (Kind::Partial { hz }, v)
        }
        Kind::Voice { count, .. } => {
            let hzs = kind.hzs();
            let refined = each(default_threads(), (0..count).collect(), |c| {
                let held = score(&Kind::Partial { hz: hzs[c] });
                let (to, _, sigma) = partial(c, hzs[c], held);
                let k = (c + 1) as f64;
                let deviation = 2.0 * to * sigma / (k * k);
                (deviation.is_finite() && deviation > 0.0)
                    .then(|| (c + 1, to, 1.0 / (deviation * deviation)))
            });
            let points: Vec<(usize, f64, f64)> = refined.into_iter().flatten().collect();
            if points.len() < 2 {
                return (kind, best);
            }
            let (f0, stretch) = stretched(&points);
            let fitted = Kind::Voice { f0, stretch, count };
            let v = score(&fitted);
            match v < best {
                true => (fitted, v),
                false => (kind, best),
            }
        }
        Kind::Noise { .. } => (kind, best),
    }
}

/// A level from `on` to `end` through `levels` at `times`, the first held back to `on` and
/// the last on to `end`, thinned by Ramer-Douglas-Peucker at `tolerance` dB.
fn thinned((on, end): (f64, f64), times: &[f64], levels: &[f64], tolerance: f64) -> Trajectory {
    let mut points = vec![(on, levels[0])];
    points.extend(
        times
            .iter()
            .zip(levels)
            .filter(|(t, _)| on < **t && **t < end)
            .map(|(t, l)| (*t, *l)),
    );
    points.push((end, levels[levels.len() - 1]));
    let mut keep = vec![false; points.len()];
    let last = points.len() - 1;
    (keep[0], keep[last]) = (true, true);
    let mut open = vec![(0, last)];
    while let Some((i, j)) = open.pop() {
        let ((ti, li), (tj, lj)) = (points[i], points[j]);
        let gap = |k: usize| {
            let (t, l) = points[k];
            (l - (li + (lj - li) * (t - ti) / (tj - ti))).abs()
        };
        let Some(k) = (i + 1..j).max_by(|x, y| gap(*x).total_cmp(&gap(*y)).then(y.cmp(x))) else {
            continue;
        };
        if gap(k) > tolerance {
            keep[k] = true;
            open.push((i, k));
            open.push((k, j));
        }
    }
    let kept: Vec<(f64, f64)> = points
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(p, _)| *p)
        .collect();
    let base = kept[0].1;
    Trajectory {
        base,
        knots: kept[1..].iter().map(|(t, l)| (*t, l - base)).collect(),
    }
}

/// The level in dB `level` holds at `t`, from `on`.
pub fn level_at(level: &Trajectory, on: f64, t: f64) -> f64 {
    let mut from = (on, level.base);
    for (at, d) in &level.knots {
        let to = (*at, level.base + d);
        if t <= to.0 {
            return match to.0 > from.0 {
                true => from.1 + (to.1 - from.1) * (t - from.0) / (to.0 - from.0),
                false => to.1,
            };
        }
        from = to;
    }
    from.1
}

/// `min 1/2 x'Gx - h'x` over `x >= 0`, `g` row-major `n` by `n`, from the feasible `start`
/// (Lawson and Hanson 1974, their active set warm-started); a gradient under its own rounding
/// bound is nought, and a column whose entry cannot hold or whose system is singular is barred
/// till another entry holds.
fn nnls(g: &[f64], n: usize, h: &[f64], start: &[f64]) -> Vec<f64> {
    let mut x: Vec<f64> = start.iter().map(|v| v.max(0.0)).collect();
    let mut passive: Vec<bool> = x.iter().map(|v| *v > 0.0).collect();
    let mut barred = vec![false; n];
    let mut entered: Option<usize> = None;
    loop {
        loop {
            let set: Vec<usize> = (0..n).filter(|i| passive[*i]).collect();
            let Some(z) = solved(g, n, &set, h) else {
                match entered.take() {
                    Some(j) => {
                        passive[j] = false;
                        barred[j] = true;
                    }
                    None => {
                        x.iter_mut().for_each(|v| *v = 0.0);
                        passive.iter_mut().for_each(|v| *v = false);
                    }
                }
                continue;
            };
            if z.iter().all(|v| *v > 0.0) {
                for (i, v) in set.iter().zip(&z) {
                    x[*i] = *v;
                }
                if entered.take().is_some() {
                    barred.iter_mut().for_each(|b| *b = false);
                }
                break;
            }
            let (mut alpha, mut at) = (1.0f64, None);
            for (i, v) in set.iter().zip(&z) {
                if *v <= 0.0 {
                    let step = x[*i] / (x[*i] - v);
                    if step < alpha || at.is_none() {
                        (alpha, at) = (step.min(alpha), Some(*i));
                    }
                }
            }
            for (i, v) in set.iter().zip(&z) {
                x[*i] += alpha * (v - x[*i]);
            }
            if let Some(i) = at {
                x[i] = 0.0;
            }
            for i in &set {
                if x[*i] <= 0.0 {
                    x[*i] = 0.0;
                    passive[*i] = false;
                    if entered == Some(*i) {
                        barred[*i] = true;
                        entered = None;
                    }
                }
            }
        }
        let best = (0..n)
            .filter(|j| !passive[*j] && !barred[*j])
            .map(|j| {
                let row = &g[j * n..(j + 1) * n];
                let pulled: f64 = row.iter().zip(&x).map(|(a, v)| a * v).sum();
                let bound: f64 = row.iter().zip(&x).map(|(a, v)| (a * v).abs()).sum();
                let gradient = h[j] - pulled;
                let rounding = f64::EPSILON * n as f64 * (h[j].abs() + bound);
                (j, gradient, rounding)
            })
            .filter(|(_, gradient, rounding)| gradient > rounding)
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
        let Some((j, _, _)) = best else {
            return x;
        };
        passive[j] = true;
        entered = Some(j);
    }
}

/// `G[set, set] z = h[set]` by Cholesky, `None` where a pivot falls under its rounding bound.
fn solved(g: &[f64], n: usize, set: &[usize], h: &[f64]) -> Option<Vec<f64>> {
    let m = set.len();
    let mut l = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..=i {
            let mut sum = g[set[i] * n + set[j]];
            for k in 0..j {
                sum -= l[i * m + k] * l[j * m + k];
            }
            if i == j {
                let diagonal = g[set[i] * n + set[i]];
                if sum <= f64::EPSILON * m as f64 * diagonal.abs() || sum <= 0.0 {
                    return None;
                }
                l[i * m + i] = libm::sqrt(sum);
            } else {
                l[i * m + j] = sum / l[j * m + j];
            }
        }
    }
    let mut z: Vec<f64> = set.iter().map(|i| h[*i]).collect();
    for i in 0..m {
        for k in 0..i {
            z[i] -= l[i * m + k] * z[k];
        }
        z[i] /= l[i * m + i];
    }
    for i in (0..m).rev() {
        for k in i + 1..m {
            z[i] -= l[k * m + i] * z[k];
        }
        z[i] /= l[i * m + i];
    }
    Some(z)
}

/// The kept atoms: their kinds and keys, every column's template and owner, each frame's
/// Gram matrix and correlations of the columns with the clip, the last fitted gains, and the
/// atoms as written.
#[derive(Clone)]
struct State {
    kinds: Vec<Kind>,
    keys: Vec<u64>,
    templates: Vec<Vec<f64>>,
    owner: Vec<usize>,
    gram: Vec<Vec<f64>>,
    pulls: Vec<Vec<f64>>,
    fitted: Vec<Vec<f64>>,
    written: Vec<Written>,
    atom_bits: f64,
    residual_bits: f64,
}

impl State {
    fn empty(p: &Problem) -> State {
        let frames = p.frames();
        State {
            kinds: Vec::new(),
            keys: Vec::new(),
            templates: Vec::new(),
            owner: Vec::new(),
            gram: vec![Vec::new(); frames],
            pulls: vec![Vec::new(); frames],
            fitted: vec![Vec::new(); frames],
            written: Vec::new(),
            atom_bits: bits::count(0),
            residual_bits: 0.5 * LOG2_E * p.energy.iter().sum::<f64>(),
        }
    }

    fn dl(&self) -> f64 {
        self.atom_bits + self.residual_bits
    }

    fn atoms(&self) -> Vec<Atom> {
        self.written.iter().map(|w| w.atom.clone()).collect()
    }

    /// This state with `kind` keyed `key` added, re-solved.
    fn with(&self, p: &Problem, kind: Kind, key: u64) -> State {
        let added = p.templates(&kind);
        let atom = self.kinds.len();
        let old = self.templates.len();
        let n = old + added.len();
        let rows = each(default_threads(), (0..p.frames()).collect(), |f| {
            let mut gram = vec![0.0; n * n];
            for i in 0..old {
                gram[i * n..i * n + old].copy_from_slice(&self.gram[f][i * old..(i + 1) * old]);
            }
            let mut pulls = self.pulls[f].clone();
            for (a, t) in added.iter().enumerate() {
                let i = old + a;
                for j in 0..=i {
                    let other = match j < old {
                        true => &self.templates[j],
                        false => &added[j - old],
                    };
                    let v = p.dot(f, t, other);
                    (gram[i * n + j], gram[j * n + i]) = (v, v);
                }
                pulls.push(p.dot(f, t, &p.power[f]));
            }
            let mut fitted = self.fitted[f].clone();
            fitted.resize(n, 0.0);
            (gram, pulls, fitted)
        });
        let mut next = self.clone();
        next.kinds.push(kind);
        next.keys.push(key);
        next.owner.extend(std::iter::repeat_n(atom, added.len()));
        next.templates.extend(added);
        for (f, (gram, pulls, fitted)) in rows.into_iter().enumerate() {
            (next.gram[f], next.pulls[f], next.fitted[f]) = (gram, pulls, fitted);
        }
        next.solved(p)
    }

    /// This state without atom `a`, re-solved.
    fn without(&self, p: &Problem, a: usize) -> State {
        let keep: Vec<usize> = (0..self.templates.len())
            .filter(|c| self.owner[*c] != a)
            .collect();
        let (n, m) = (self.templates.len(), keep.len());
        let mut next = self.clone();
        next.kinds.remove(a);
        next.keys.remove(a);
        next.templates = keep.iter().map(|c| self.templates[*c].clone()).collect();
        next.owner = keep
            .iter()
            .map(|c| self.owner[*c] - usize::from(self.owner[*c] > a))
            .collect();
        for f in 0..self.gram.len() {
            let mut gram = vec![0.0; m * m];
            for (i, ci) in keep.iter().enumerate() {
                for (j, cj) in keep.iter().enumerate() {
                    gram[i * m + j] = self.gram[f][ci * n + cj];
                }
            }
            next.gram[f] = gram;
            next.pulls[f] = keep.iter().map(|c| self.pulls[f][*c]).collect();
            next.fitted[f] = keep.iter().map(|c| self.fitted[f][*c]).collect();
        }
        next.solved(p)
    }

    /// This state's columns re-solved jointly per frame and its atoms written, as `MODEL`
    /// states.
    fn solved(mut self, p: &Problem) -> State {
        let n = self.templates.len();
        let frames: Vec<usize> = (0..p.frames()).collect();
        self.fitted = each(default_threads(), frames, |f| {
            nnls(&self.gram[f], n, &self.pulls[f], &self.fitted[f])
        });
        self.written = (0..self.kinds.len())
            .map(|a| {
                let columns: Vec<Vec<f64>> = (0..n)
                    .filter(|c| self.owner[*c] == a)
                    .map(|c| self.fitted.iter().map(|g| g[c]).collect())
                    .collect();
                write(p, &self.kinds[a], self.keys[a], &columns)
            })
            .collect();
        self.price(p);
        self
    }

    /// The written gains per frame, column by column.
    fn gains(&self, f: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.templates.len());
        for w in &self.written {
            out.extend(w.gains.iter().map(|g| g[f]));
        }
        out
    }

    /// Its DL's parts from its written atoms.
    fn price(&mut self, p: &Problem) {
        let n = self.templates.len();
        self.atom_bits = self
            .written
            .iter()
            .fold(bits::count(self.written.len()), |s, w| s + w.bits);
        self.residual_bits = 0.5
            * LOG2_E
            * (0..p.frames())
                .map(|f| {
                    let g = self.gains(f);
                    let mut quadratic = 0.0;
                    for i in 0..n {
                        let row: f64 = (0..n).map(|j| self.gram[f][i * n + j] * g[j]).sum();
                        quadratic += g[i] * (row - 2.0 * self.pulls[f][i]);
                    }
                    p.energy[f] + quadratic
                })
                .sum::<f64>();
    }

    /// The model's expected power per feasible bin, frame by frame.
    fn model(&self, p: &Problem) -> Vec<Vec<f64>> {
        (0..p.frames())
            .map(|f| {
                let mut m = vec![0.0; p.range.len()];
                for (g, t) in self.gains(f).iter().zip(&self.templates) {
                    for (m, t) in m.iter_mut().zip(t) {
                        *m += g * t;
                    }
                }
                m
            })
            .collect()
    }
}

/// The bins `power` peaks at over their floor, each a track spanning the clip beside its bin,
/// as `MODEL` states.
fn peaks(p: &Problem, power: &[f64]) -> Vec<(Track, usize)> {
    let at = |i: usize| p.range.start + i;
    let span = (0.0, p.clip.secs());
    let mut out = Vec::new();
    for i in 1..power.len().saturating_sub(1) {
        if power[i] <= power[i - 1] || power[i] < power[i + 1] {
            continue;
        }
        let dbs = |v: f64| db(v.max(f64::MIN_POSITIVE));
        let (a, c, e) = (dbs(power[i - 1]), dbs(power[i]), dbs(power[i + 1]));
        let curve = a - 2.0 * c + e;
        let shift = match curve < 0.0 {
            true => 0.5 * (a - e) / curve,
            false => 0.0,
        };
        let hz = p.grid.hz(at(i)) + shift * p.grid.rate / p.grid.n as f64;
        let peak = libm::pow(10.0, (c - 0.25 * (a - e) * shift) / 10.0);
        let level = 20.0 * libm::log10(2.0 * libm::sqrt(peak) / p.grid.sum_w);
        if level > floor_db(hz, p.spl_db) {
            out.push((Track { hz, span }, i));
        }
    }
    out
}

/// `(f0, B)` by least squares of `(f_k / k)^2` on `k^2` over `members`, `B` at least 0.
fn stiffness(tracks: &[Track], members: &Members) -> (f64, f64) {
    let points: Vec<(usize, f64, f64)> = members
        .iter()
        .map(|(t, k)| (*k, tracks[*t].hz, 1.0))
        .collect();
    stretched(&points)
}

/// `(f0, B)` by least squares of `(f_k / k)^2` on `k^2` over `(k, f_k, weight)`, `B` at least 0.
fn stretched(points: &[(usize, f64, f64)]) -> (f64, f64) {
    let points: Vec<(f64, f64, f64)> = points
        .iter()
        .map(|(k, hz, w)| {
            let k = *k as f64;
            (k * k, (hz / k) * (hz / k), *w)
        })
        .collect();
    let total: f64 = points.iter().map(|p| p.2).sum();
    let mean = |pick: fn(&(f64, f64, f64)) -> f64| {
        points.iter().map(|p| p.2 * pick(p)).sum::<f64>() / total
    };
    let (mx, my) = (mean(|p| p.0), mean(|p| p.1));
    let spread: f64 = points.iter().map(|p| p.2 * (p.0 - mx) * (p.0 - mx)).sum();
    let slope = match spread > 0.0 {
        true => {
            points
                .iter()
                .map(|p| p.2 * (p.0 - mx) * (p.1 - my))
                .sum::<f64>()
                / spread
        }
        false => 0.0,
    };
    let (a, c) = match slope > 0.0 {
        true => (my - slope * mx, slope),
        false => (my, 0.0),
    };
    match a > 0.0 {
        true => (libm::sqrt(a), c / a),
        false => (libm::sqrt(my), 0.0),
    }
}

/// The members at the stretched Hz of `(f0, B)` among `pool`: the nearest track within half an
/// analysis bin of each harmonic's, as `MODEL` states.
fn assigned(tracks: &[Track], pool: &[usize], (f0, b): (f64, f64), p: &Problem) -> Members {
    let (bin, top) = (p.grid.rate / p.grid.n as f64, span_hz(p.grid.rate).1);
    let mut out = Vec::new();
    for k in 1.. {
        let kf = k as f64;
        let hz = f0 * kf * libm::sqrt(1.0 + b * kf * kf);
        if hz > top {
            break;
        }
        let near = pool
            .iter()
            .map(|t| (*t, (tracks[*t].hz - hz).abs()))
            .filter(|(_, miss)| 2.0 * miss <= bin)
            .min_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(&y.0)));
        if let Some((t, _)) = near {
            out.push((t, k));
        }
    }
    out
}

/// Every voice the sieve finds among `tracks`, its stretch fitted, as `MODEL` states.
fn voices(p: &Problem, tracks: &[Track]) -> Vec<Kind> {
    let bin = p.grid.rate / p.grid.n as f64;
    let mut pool: Vec<usize> = (0..tracks.len()).collect();
    let mut out = Vec::new();
    while let Some((_, found)) = sieved(tracks, &pool, bin, span_hz(p.grid.rate)) {
        let mut members = found;
        let mut seen: Vec<Members> = Vec::new();
        let mut fitted = stiffness(tracks, &members);
        loop {
            let next = assigned(tracks, &pool, fitted, p);
            if next.is_empty() || next == members || seen.contains(&next) {
                break;
            }
            seen.push(std::mem::replace(&mut members, next));
            fitted = stiffness(tracks, &members);
        }
        pool.retain(|t| !members.iter().any(|(m, _)| m == t));
        let count = members.iter().map(|(_, k)| *k).max().unwrap_or(0);
        let kind = Kind::Voice {
            f0: fitted.0,
            stretch: fitted.1,
            count,
        };
        let level = Trajectory {
            base: 0.0,
            knots: vec![(p.clip.secs(), 0.0)],
        };
        let probe = Atom::harmonic(
            0,
            0.0,
            (Trajectory::steady(fitted.0), fitted.1),
            vec![level; count.max(1)],
        );
        if count >= 2 && probe.writes(&p.clip) {
            out.push(kind);
        }
    }
    out
}

/// Every candidate the residual `residual` proposes, as `MODEL` states.
fn candidates(p: &Problem, state: &State, residual: &[Vec<f64>]) -> Vec<Kind> {
    let mean: Vec<f64> = (0..p.range.len())
        .map(|b| residual.iter().map(|r| r[b]).sum::<f64>() / residual.len() as f64)
        .collect();
    let found = peaks(p, &mean);
    let mut out: Vec<Kind> = found
        .iter()
        .map(|(t, _)| Kind::Partial { hz: t.hz })
        .collect();
    let (_, tonal) = p.lobes.floor(&mean, p.looks * p.frames() as f64);
    let tracks: Vec<Track> = found
        .into_iter()
        .filter(|(_, i)| tonal.binary_search(i).is_ok())
        .map(|(t, _)| t)
        .collect();
    out.extend(voices(p, &tracks));
    for band in p.grid.bands() {
        let held = state
            .kinds
            .iter()
            .any(|k| matches!(k, Kind::Noise { band: b } if *b == band));
        if !held {
            out.push(Kind::Noise { band });
        }
    }
    out
}

/// The DL change adding `kind` keyed `key` makes in closed form against `residual`, as
/// `MODEL` states.
fn scored(
    p: &Problem,
    state: &State,
    residual: &[Vec<f64>],
    kind: &Kind,
    key: u64,
) -> (f64, Written) {
    let templates = p.templates(kind);
    // Each column's weighted energy and correlation with the residual, frame by frame.
    let moments: Vec<Vec<(f64, f64)>> = templates
        .iter()
        .map(|t| {
            (0..p.frames())
                .map(|f| p.moments(f, t, &residual[f]))
                .collect()
        })
        .collect();
    let fitted: Vec<Vec<f64>> = moments
        .iter()
        .map(|m| {
            m.iter()
                .map(|(energy, pull)| match *energy > 0.0 {
                    true => (pull / energy).max(0.0),
                    false => 0.0,
                })
                .collect()
        })
        .collect();
    let written = write(p, kind, key, &fitted);
    // One column's drop in closed form, sum w (2 r g t - g^2 t^2) = 2 g pull - g^2 energy.
    let drop: f64 = match &moments[..] {
        [m] => m
            .iter()
            .zip(&written.gains[0])
            .map(|((energy, pull), g)| 2.0 * g * pull - g * g * energy)
            .sum(),
        _ => (0..p.frames())
            .map(|f| {
                let mut u = vec![0.0; p.range.len()];
                for (t, g) in templates.iter().zip(&written.gains) {
                    for (u, t) in u.iter_mut().zip(t) {
                        *u += g[f] * t;
                    }
                }
                u.iter()
                    .zip(&residual[f])
                    .zip(&p.weight[f])
                    .map(|((u, r), w)| w * (2.0 * r * u - u * u))
                    .sum::<f64>()
            })
            .sum(),
    };
    let n = state.kinds.len();
    let change = written.bits + bits::count(n + 1) - bits::count(n) - 0.5 * LOG2_E * drop;
    (change, written)
}

/// The progress line of `state` after `action` on `atom`, as `Line` states.
fn line(p: &Problem, state: &State, step: usize, action: Action, atom: Option<Atom>) -> Line {
    let model = state.model(p);
    let grid = &p.grid;
    let residual_db = model
        .iter()
        .zip(&p.power)
        .map(|(m, y)| {
            let mut full = vec![0.0; grid.bins()];
            for (i, (m, y)) in m.iter().zip(y).enumerate() {
                full[p.range.start + i] = (y - m).max(0.0);
            }
            grid.pooled(&full)
                .iter()
                .map(|v| (*v > 0.0).then(|| grid.band_db(*v)))
                .collect()
        })
        .collect();
    Line {
        pass: 1,
        step,
        action,
        atom,
        delta_bits: 0.0,
        atom_bits: state.atom_bits,
        residual_bits: state.residual_bits,
        residual_db,
        envelopes: state
            .written
            .iter()
            .map(|w| (w.atom.key, w.levels.clone()))
            .collect(),
        geometry: None,
        seconds: 0.0,
    }
}

/// The best removal's state, where one lowers the DL.
fn pruned(p: &Problem, state: &State) -> Option<(usize, State)> {
    let trials = each(default_threads(), (0..state.kinds.len()).collect(), |a| {
        state.without(p, a)
    });
    trials
        .into_iter()
        .enumerate()
        .filter(|(_, s)| s.dl() < state.dl())
        .min_by(|x, y| x.1.dl().total_cmp(&y.1.dl()).then(x.0.cmp(&y.0)))
}

/// `state` with the atom keyed `added` and each kept atom whose bins meet its own refined in
/// turn against the clip less every other atom's model, each kept where the re-solved state's
/// DL falls, as `MODEL` states.
fn retuned(p: &Problem, mut state: State, added: u64) -> State {
    let new = state
        .keys
        .iter()
        .position(|k| *k == added)
        .expect("the added key");
    let near = reach(p, &state.kinds[new]);
    let keys: Vec<u64> = state
        .keys
        .iter()
        .zip(&state.kinds)
        .filter(|(k, kind)| **k == added || meets(&near, &reach(p, kind)))
        .map(|(k, _)| *k)
        .collect();
    for key in keys {
        let a = state
            .keys
            .iter()
            .position(|k| *k == key)
            .expect("a kept key");
        if matches!(state.kinds[a], Kind::Noise { .. }) {
            continue;
        }
        let columns: Vec<usize> = (0..state.templates.len())
            .filter(|c| state.owner[*c] == a)
            .collect();
        let model = state.model(p);
        let residual: Vec<Vec<f64>> = (0..p.frames())
            .map(|f| {
                let gains = state.gains(f);
                let mut r: Vec<f64> = p.power[f]
                    .iter()
                    .zip(&model[f])
                    .map(|(y, m)| y - m)
                    .collect();
                for c in &columns {
                    for (r, t) in r.iter_mut().zip(&state.templates[*c]) {
                        *r += gains[*c] * t;
                    }
                }
                r
            })
            .collect();
        let (kind, _) = refined(p, &state, &residual, state.kinds[a].clone(), key);
        if kind == state.kinds[a] {
            continue;
        }
        let next = state.without(p, a).with(p, kind, key);
        if next.dl() < state.dl() {
            state = next;
        }
    }
    state
}

/// A greedy pass's end: its state, its steps and whether a stop ended it.
struct Pass {
    state: State,
    steps: usize,
    stopped: bool,
}

/// The greedy loop over `p` from no atom, keys from `born`, each step heard by `hear` marked
/// `pass`, as `MODEL` states.
async fn greedy(
    p: &Problem,
    pass: u8,
    born: u64,
    hear: &mut dyn FnMut(&Line),
    stop: &dyn Abandon,
    clock: &dyn Fn() -> f64,
) -> Pass {
    let mut state = State::empty(p);
    let mut steps = 0;
    let mut stopped = false;
    let mut said = |state: &State, before: f64, action: Action, atom: Atom, steps: &mut usize| {
        *steps += 1;
        let mut heard = line(p, state, *steps, action, Some(atom));
        heard.pass = pass;
        heard.delta_bits = state.dl() - before;
        heard.seconds = clock();
        hear(&heard);
    };
    loop {
        if stop.abandoned().await {
            stopped = true;
            break;
        }
        let model = state.model(p);
        let residual: Vec<Vec<f64>> = model
            .iter()
            .zip(&p.power)
            .map(|(m, y)| y.iter().zip(m).map(|(y, m)| y - m).collect())
            .collect();
        let key = birth(&state.atoms()).max(born);
        let proposed = candidates(p, &state, &residual);
        let scores = each(default_threads(), proposed, |kind| {
            refined(p, &state, &residual, kind, key)
        });
        let best = scores
            .into_iter()
            .enumerate()
            .filter(|(_, (_, s))| *s < 0.0)
            .min_by(|x, y| x.1.1.total_cmp(&y.1.1).then(x.0.cmp(&y.0)));
        let Some((_, (kind, _))) = best else {
            break;
        };
        let grown = retuned(p, state.with(p, kind, key), key);
        if grown.dl() >= state.dl() {
            break;
        }
        let before = state.dl();
        state = grown;
        let added = state.written.last().expect("an atom").atom.clone();
        said(&state, before, Action::Add, added, &mut steps);
        while let Some((a, next)) = pruned(p, &state) {
            let before = state.dl();
            let gone = state.written[a].atom.clone();
            state = next;
            said(&state, before, Action::Remove, gone, &mut steps);
        }
    }
    Pass {
        state,
        steps,
        stopped,
    }
}

/// `model` over `p`'s feasible bins laid into every bin, nought outside.
fn full(p: &Problem, model: &[Vec<f64>]) -> Vec<Vec<f64>> {
    model
        .iter()
        .map(|m| {
            let mut out = vec![0.0; p.grid.bins()];
            out[p.range.clone()].copy_from_slice(m);
            out
        })
        .collect()
}

/// One sound's fit of `samples`, each pass's start and steps heard by `hear` as they end,
/// `stop` asked before each, as `MODEL` states.
pub async fn fit(
    samples: &[f64],
    rate: u32,
    spl_db: f64,
    hear: &mut dyn FnMut(&Line),
    stop: &dyn Abandon,
    clock: &dyn Fn() -> f64,
) -> Fit {
    let to = rate_for(rate);
    let samples = decimated(samples, rate, to);
    let clip = Clip {
        rate: to,
        len: samples.len(),
    };
    let square = samples.iter().map(|s| s * s).sum::<f64>() / samples.len().max(1) as f64;
    let rms = match square > 0.0 {
        true => libm::sqrt(square),
        false => 1.0,
    };
    let unit: Vec<f64> = samples.iter().map(|s| s / rms).collect();
    let gain_db = 20.0 * libm::log10(rms);
    let heard_at = spl_db + gain_db;
    let grid = Grid::of(&Spectrogram::of(&unit, clip.rate), unit.len());
    let p = Problem::of(grid, clip, heard_at, gain_db);
    let mut first = line(&p, &State::empty(&p), 0, Action::Start, None);
    first.geometry = Some(Geometry {
        rate_hz: to,
        frame_secs: p.grid.frame_secs(),
        first_centre_secs: p.grid.centre_secs(0),
        band_edges_hz: p.grid.bands(),
        band_floor_db: p
            .grid
            .band_centres()
            .iter()
            .map(|hz| floor_db(*hz, heard_at))
            .collect(),
        gain_db,
    });
    first.seconds = clock();
    hear(&first);
    let one = greedy(&p, 1, 0, hear, stop, clock).await;
    let model = full(&p, &one.state.model(&p));
    let atoms = one.state.atoms();
    let second = match one.stopped {
        true => Second::default(),
        false => {
            let left = p
                .grid
                .power
                .iter()
                .zip(&model)
                .map(|(y, m)| y.iter().zip(m).map(|(y, m)| (y - m).max(0.0)).collect())
                .collect();
            let q = Problem::of(p.grid.with(left), clip, heard_at, gain_db);
            let mut start = line(&q, &State::empty(&q), 0, Action::Start, None);
            start.pass = 2;
            start.seconds = clock();
            hear(&start);
            let two = greedy(&q, 2, birth(&atoms), hear, stop, clock).await;
            Second {
                atoms: two.state.atoms(),
                start_bits: State::empty(&q).dl(),
                atom_bits: two.state.atom_bits,
                residual_bits: two.state.residual_bits,
                steps: two.steps,
                stopped: two.stopped,
            }
        }
    };
    Fit {
        clip,
        atoms,
        gain_db,
        spl_db: heard_at,
        atom_bits: one.state.atom_bits,
        residual_bits: one.state.residual_bits,
        steps: one.steps,
        stopped: one.stopped || second.stopped,
        clip_grid: p.grid.broad(),
        fit_grid: p.grid.with(model).broad(),
        second,
    }
}
