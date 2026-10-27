// Concern: FIT: one sound by passes of one move each, the best add or level adjust under one DL | Non-concern: the analysis (add.rs), JSON | IO: (samples, rate, spl) -> Fit, Lines

use std::f64::consts::LOG2_E;
use std::ops::Range;

use sva_engine::hearing::{CAM_LAST, cam, cam_hz, erb_hz, level_jnd_db};
use sva_engine::{Abandon, default_threads, each, fft};

use super::add::{Members, Spectrogram, Track, floor_db, sieved, span_hz};
use super::atoms::{Atom, Clip, Feasible, Quantity, Trajectory};
use super::bits::{self, Look, Priced, frequency_delta, level_delta};
use super::grid::Grid;

pub const MODEL: &str = "one sound fitted by passes of one move each: coordinate descent on the atoms' levels beside matching pursuit for new atoms (Mallat and Zhang 1993, doi 10.1109/78.258082); \
     DL = B(x) (bits.rs) plus the residual's bits, the one objective; \
     every pass over one problem: the clip's grid, its weights read once off the clip and never off a residual; \
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
     templates in expected power: a sine through the window's closed-form kernel, a noise band's expected periodogram (grid.rs), a shaped noise's the sum of its bands' at their gains; \
     each column's gain per frame its power; \
     each pass scores every move against the residual and makes the one that lowers the DL most, or none, and the fit stops at the first pass with none, so the DL falls at every pass; \
     a move an add or an adjust; \
     an add a new atom off the residual averaged over the frames, at its best level: \
     a free partial at each local maximum over its floor, its Hz by QIFFT (Smith and Serra 1987); \
     a voice at each fundamental the sieve (add.rs) finds among those peaks that are tonal lobes of the averaged residual at K times its frames looks, taken from the pool with its members till none is found; \
     a voice's harmonic k at k f0 sqrt(1 + B k^2) (Fletcher 1964), f0 and B by least squares of (f_k / k)^2 = f0^2 + f0^2 B k^2 over its member peaks, B at least 0; \
     members reassigned within half an analysis bin of the stretched Hz and refitted till the set repeats; \
     every harmonic to the highest member its own column and its own level; \
     one noise band per octave band no kept noise band holds; \
     one shaped noise: a gain per octave band times one envelope over the frames (rank 1), both at least 0, by alternating least squares against the residual from a flat envelope till an alternation no longer lowers the weighted misfit, which never rises, its bands' cross terms left out of the alternation and in its score; a band whose gain falls to 0 carved out, the rest's gains over the highest's written in dB; \
     an add scored in closed form: each column's gain per frame its weighted correlation with the residual over its weighted energy, at least 0, written, the DL drop those gains give less its bits; \
     each partial's and voice's frequencies refined against the residual as it is scored: a partial's Hz by golden section (Kiefer 1953) over half an analysis bin of its peak to the finer of the precision it is written at and its standard deviation under the weights, sigma = (sum w (du/dx)^2)^-1/2, u its expected power, du/dx by central difference at the cube root of epsilon; a voice's harmonics each so as a partial, then f0 and B by that least squares weighted by the inverse variance of (f_k / k)^2, (2 f_k sigma_k / k^2)^-2, kept where the voice's score falls; \
     a kept atom's frequencies never move again; \
     an adjust a kept atom's level changed by the exact best amount for that atom alone, up or down: each column's written gain per frame plus the residual's weighted correlation with the column over its weighted energy, at least 0, written; a voice's columns each so alone, their main lobes apart; \
     or to 0, which removes it, where that gives the lower DL, as an atom's bits are no smooth function of its level; \
     so like terms collect: an add then an adjust is one atom at their sum; \
     the DL of every move the written state's, in closed form from the residual, its drop exact; the residual derived afresh from the kept atoms at each pass, a move whose repriced DL does not fall, its drop under rounding, no move; \
     ties broken by the move's order: adds in the candidates' order, then adjusts in the atoms'; \
     a column's level 10 log10 of its gain, held at its floor; a shaped noise's floor the least of its bands' floors less their gains; \
     each level through the frame centres from 0 to the clip's end, thinned by Ramer-Douglas-Peucker at the level JND at its peak sensation level (Jesteadt, Wier and Green 1977); \
     the DL the written atoms': levels at the level JND by looks, frequencies at the frequency JND by looks (bits.rs), every frame a look at its level over its floor; \
     a shaped noise's band gain at the level JND by looks of that band's level over its own floor, its envelope by its looks over the envelope's floor; \
     a voice's f0 at the JND at f0, its B where its highest harmonic moves by that harmonic's JND; \
     a time at the frame, a noise band's edge one of the grid's band edges; \
     a key one past the last born; \
     each pass reported as it ends: its move, the atom's level before and after, the DL's drop and total, the residual's grid";

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

/// What a pass did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Start,
    Add,
    Adjust,
}

impl Action {
    pub fn name(self) -> &'static str {
        match self {
            Action::Start => "start",
            Action::Add => "add",
            Action::Adjust => "adjust",
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

/// One pass as a progress line reports it: its move, the DL's change and total, the residual's
/// band levels per frame (`None` where the fit covers the clip), each kept atom's written level
/// per frame, and the atoms it leaves on its clip.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    /// The pass, from 1; 0 for the start.
    pub pass: usize,
    pub action: Action,
    pub atom: Option<Atom>,
    /// The atom's level in dB before and after, as `level_db` states; `None` where it was not,
    /// or is no more.
    pub from_db: Option<f64>,
    pub to_db: Option<f64>,
    pub delta_bits: f64,
    pub atom_bits: f64,
    pub residual_bits: f64,
    pub residual_db: Vec<Vec<Option<f64>>>,
    pub envelopes: Vec<(u64, Vec<Vec<f64>>)>,
    pub geometry: Option<Geometry>,
    pub seconds: f64,
    pub atoms: Vec<Atom>,
    pub clip: Clip,
}

/// A fit: the clip it read, its atoms in unit-RMS space, the master gain bringing them back,
/// the SPL that space is heard at, its DL's parts, its passes and how many added and adjusted,
/// whether a stop ended it, and the clip's and the fit's coarse grids, each band's power spread
/// flat over its bins.
pub struct Fit {
    pub clip: Clip,
    pub atoms: Vec<Atom>,
    pub gain_db: f64,
    pub spl_db: f64,
    pub atom_bits: f64,
    pub residual_bits: f64,
    pub passes: usize,
    pub adds: usize,
    pub adjusts: usize,
    pub stopped: bool,
    pub clip_grid: Grid,
    pub fit_grid: Grid,
}

/// What an atom is before its level is written.
#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Partial {
        hz: f64,
    },
    Voice {
        f0: f64,
        stretch: f64,
        count: usize,
    },
    Noise {
        band: (f64, f64),
    },
    /// Each band beside its power gain, the highest 1.
    Shaped {
        bands: Vec<((f64, f64), f64)>,
    },
}

impl Kind {
    /// Each column's Hz: a noise band's its geometric centre, a shaped noise's its first band's.
    fn hzs(&self) -> Vec<f64> {
        match self {
            Kind::Partial { hz } => vec![*hz],
            Kind::Voice { f0, stretch, count } => (1..=*count)
                .map(|k| {
                    let k = k as f64;
                    f0 * k * libm::sqrt(1.0 + stretch * k * k)
                })
                .collect(),
            Kind::Noise { band } => vec![centre(*band)],
            Kind::Shaped { bands } => vec![centre(bands[0].0)],
        }
    }
}

fn centre((lo, hi): (f64, f64)) -> f64 {
    libm::sqrt(lo * hi)
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

/// The clip on its grid, as `MODEL` states: power and weights over the feasible bins, frame by
/// frame.
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
            Kind::Shaped { bands } => {
                let mut sum = vec![0.0; self.range.len()];
                for (band, gain) in bands {
                    let bins = self.grid.noise_bins(*band);
                    for (s, b) in sum.iter_mut().zip(&bins[self.range.clone()]) {
                        *s += gain * b;
                    }
                }
                vec![sum]
            }
            _ => kind
                .hzs()
                .iter()
                .map(|hz| self.range.clone().map(|b| self.grid.sine(*hz, b)).collect())
                .collect(),
        }
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
    let floors = floors(p, kind);
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
        Kind::Shaped { bands } => Atom::shaped(
            key,
            0.0,
            bands
                .iter()
                .map(|((lo, hi), g)| (*lo, *hi, db(*g)))
                .collect(),
            first,
        ),
    };
    let bits = match kind {
        Kind::Shaped { bands } => shaped_bits(p, &atom, bands, &levels[0], floors[0]),
        _ => priced(p, &atom, &hzs, &levels, &floors),
    };
    Written {
        atom,
        gains,
        levels,
        bits,
    }
}

/// Each column's floor in dB, as `MODEL` states.
fn floors(p: &Problem, kind: &Kind) -> Vec<f64> {
    match kind {
        Kind::Shaped { bands } => vec![
            bands
                .iter()
                .map(|(band, g)| floor_db(centre(*band), p.spl_db) - db(*g))
                .fold(f64::INFINITY, f64::min),
        ],
        _ => kind
            .hzs()
            .iter()
            .map(|hz| floor_db(*hz, p.spl_db))
            .collect(),
    }
}

/// The edge's precision: one of the grid's band edges.
fn edge_step(p: &Problem, feasible: &Feasible) -> f64 {
    (feasible.hz.1 - feasible.hz.0) / (p.grid.bands().len() + 1) as f64
}

/// A shaped noise's bits at the precisions `MODEL` states, its envelope at `level` per frame
/// over `floor`.
fn shaped_bits(
    p: &Problem,
    atom: &Atom,
    bands: &[((f64, f64), f64)],
    level: &[f64],
    floor: f64,
) -> f64 {
    let hop = p.grid.frame_secs();
    let feasible = Feasible::of(&p.clip, (floor, p.top_db));
    let edge = edge_step(p, &feasible);
    let mut deltas = vec![hop];
    for (band, g) in bands {
        let heard: Vec<f64> = level.iter().map(|l| l + db(*g)).collect();
        let at = p.looks(&heard, floor_db(centre(*band), p.spl_db), centre(*band));
        deltas.extend([edge, edge, level_delta(hop, &at)]);
    }
    let envelope = level_delta(hop, &p.looks(level, floor, centre(bands[0].0)));
    for (q, _) in &atom.numbers()[deltas.len()..] {
        deltas.push(match q {
            Quantity::Time => hop,
            _ => envelope,
        });
    }
    bits::atom(atom, &Priced { feasible, deltas })
}

/// `atom`'s bits at the precisions `MODEL` states, harmonic `k` at `hzs[k]` and `levels[k]`
/// per frame over `floors[k]`.
fn priced(p: &Problem, atom: &Atom, hzs: &[f64], levels: &[Vec<f64>], floors: &[f64]) -> f64 {
    let hop = p.grid.frame_secs();
    let looks = |k: usize| p.looks(&levels[k], floors[k], hzs[k]);
    let lowest = floors.iter().copied().fold(f64::INFINITY, f64::min);
    let feasible = Feasible::of(&p.clip, (lowest, p.top_db));
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
            Quantity::Edge => edge_step(p, &feasible),
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
fn refined(p: &Problem, n: usize, residual: &[Vec<f64>], kind: Kind, key: u64) -> (Kind, f64) {
    let bin = p.grid.rate / p.grid.n as f64;
    let score = |k: &Kind| scored(p, n, residual, k, key).0;
    let (best, written) = scored(p, n, residual, &kind, key);
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
        Kind::Noise { .. } | Kind::Shaped { .. } => (kind, best),
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

/// A kept atom: its kind, each column's template, and the atom as written.
#[derive(Clone)]
struct Held {
    kind: Kind,
    templates: Vec<Vec<f64>>,
    written: Written,
}

/// The move a pass made: its action, the atom's key, and its level before and after.
#[derive(Clone)]
struct Moved {
    action: Action,
    key: u64,
    from_db: Option<f64>,
    to_db: Option<f64>,
}

/// The kept atoms, the next key, the DL's parts, and the move that made this state, `None`
/// where a pass found none.
#[derive(Clone)]
struct State {
    held: Vec<Held>,
    born: u64,
    atom_bits: f64,
    residual_bits: f64,
    moved: Option<Moved>,
}

impl State {
    fn empty(p: &Problem) -> State {
        let mut state = State {
            held: Vec::new(),
            born: 0,
            atom_bits: 0.0,
            residual_bits: 0.0,
            moved: None,
        };
        state.price(p);
        state
    }

    fn dl(&self) -> f64 {
        self.atom_bits + self.residual_bits
    }

    fn atoms(&self) -> Vec<Atom> {
        self.held.iter().map(|h| h.written.atom.clone()).collect()
    }

    /// Its DL's parts from its written atoms.
    fn price(&mut self, p: &Problem) {
        self.atom_bits = self
            .held
            .iter()
            .fold(bits::count(self.held.len()), |s, h| s + h.written.bits);
        self.residual_bits = 0.5
            * LOG2_E
            * self
                .residual(p)
                .iter()
                .zip(&p.weight)
                .map(|(r, w)| r.iter().zip(w).map(|(r, w)| w * r * r).sum::<f64>())
                .sum::<f64>();
    }

    /// The clip's power less the model's per feasible bin, frame by frame.
    fn residual(&self, p: &Problem) -> Vec<Vec<f64>> {
        let mut out = p.power.clone();
        for h in &self.held {
            for (t, g) in h.templates.iter().zip(&h.written.gains) {
                for (r, g) in out.iter_mut().zip(g) {
                    r.iter_mut().zip(t).for_each(|(r, t)| *r -= g * t);
                }
            }
        }
        out
    }

    /// The model's expected power per feasible bin, frame by frame.
    fn model(&self, p: &Problem) -> Vec<Vec<f64>> {
        self.residual(p)
            .iter()
            .zip(&p.power)
            .map(|(r, y)| y.iter().zip(r).map(|(y, r)| y - r).collect())
            .collect()
    }
}

/// `written`'s level in dB: its columns' gains summed (a shaped noise's times its bands' gains)
/// and averaged over the frames.
fn level_db(kind: &Kind, written: &Written) -> f64 {
    let frames = written.gains[0].len().max(1) as f64;
    let shape = match kind {
        Kind::Shaped { bands } => bands.iter().map(|(_, g)| g).sum(),
        _ => 1.0,
    };
    let total: f64 = written.gains.iter().flatten().sum();
    db(shape * total / frames)
}

/// `sum w (u^2 - 2 r u)` over every frame, `u` the change `templates` at `after` less
/// `before` make: the residual's weighted square's change.
fn change(
    p: &Problem,
    residual: &[Vec<f64>],
    templates: &[Vec<f64>],
    (after, before): (&[Vec<f64>], &[Vec<f64>]),
) -> f64 {
    (0..p.frames())
        .map(|f| {
            let mut u = vec![0.0; p.range.len()];
            for ((t, a), b) in templates.iter().zip(after).zip(before) {
                let d = a[f] - b[f];
                if d != 0.0 {
                    u.iter_mut().zip(t).for_each(|(u, t)| *u += d * t);
                }
            }
            u.iter()
                .zip(&residual[f])
                .zip(&p.weight[f])
                .map(|((u, r), w)| w * (u * u - 2.0 * r * u))
                .sum::<f64>()
        })
        .sum()
}

/// Atom `a`'s adjust, as `MODEL` states: its DL change beside its written atom, `None` where
/// the adjust takes it to 0.
fn adjusted(p: &Problem, state: &State, residual: &[Vec<f64>], a: usize) -> (f64, Option<Written>) {
    let held = &state.held[a];
    let before = &held.written.gains;
    let fitted: Vec<Vec<f64>> = held
        .templates
        .iter()
        .zip(before)
        .map(|(t, g)| {
            (0..p.frames())
                .map(|f| {
                    let (energy, pull) = p.moments(f, t, &residual[f]);
                    match energy > 0.0 {
                        true => (g[f] + pull / energy).max(0.0),
                        false => g[f],
                    }
                })
                .collect()
        })
        .collect();
    let written = write(p, &held.kind, held.written.atom.key, &fitted);
    let squared =
        |after: &[Vec<f64>]| 0.5 * LOG2_E * change(p, residual, &held.templates, (after, before));
    let kept = written.bits - held.written.bits + squared(&written.gains);
    let n = state.held.len();
    let zeros = vec![vec![0.0; p.frames()]; before.len()];
    let gone = bits::count(n - 1) - bits::count(n) - held.written.bits + squared(&zeros);
    match gone < kept {
        true => (gone, None),
        false => (kept, Some(written)),
    }
}

/// A move: an add of a kind, or an adjust of a kept atom to its written level, `None` being 0.
enum Move {
    Add(Kind),
    Adjust(usize, Option<Written>),
}

/// One pass over `p` from `state`, as `MODEL` states: every move scored against its residual,
/// the best made where it lowers the DL; the state it makes, its `moved` `None` where none does.
fn pass(p: &Problem, state: &State) -> State {
    let key = state.born;
    let n = state.held.len();
    let residual = state.residual(p);
    let adds = each(default_threads(), candidates(p, state, &residual), |kind| {
        let (kind, change) = refined(p, n, &residual, kind, key);
        (change, Move::Add(kind))
    });
    let adjusts = each(default_threads(), (0..n).collect(), |a| {
        let (change, written) = adjusted(p, state, &residual, a);
        (change, Move::Adjust(a, written))
    });
    let best = adds
        .into_iter()
        .chain(adjusts)
        .filter(|(change, _)| *change < 0.0)
        .min_by(|x, y| x.0.total_cmp(&y.0));
    let mut next = state.clone();
    next.moved = None;
    let Some((_, made)) = best else {
        return next;
    };
    let moved = match made {
        Move::Add(kind) => {
            let (_, written) = scored(p, n, &residual, &kind, key);
            let to_db = Some(level_db(&kind, &written));
            next.held.push(Held {
                templates: p.templates(&kind),
                kind,
                written,
            });
            next.born += 1;
            Moved {
                action: Action::Add,
                key,
                from_db: None,
                to_db,
            }
        }
        Move::Adjust(a, written) => {
            let held = &state.held[a];
            let moved = Moved {
                action: Action::Adjust,
                key: held.written.atom.key,
                from_db: Some(level_db(&held.kind, &held.written)),
                to_db: written.as_ref().map(|w| level_db(&held.kind, w)),
            };
            match written {
                Some(w) => next.held[a].written = w,
                None => {
                    next.held.remove(a);
                }
            }
            moved
        }
    };
    next.price(p);
    // A drop under the rounding of the repricing is no drop, as `MODEL` states.
    if next.dl() >= state.dl() {
        next = state.clone();
        next.moved = None;
        return next;
    }
    next.moved = Some(moved);
    next
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
            .held
            .iter()
            .any(|h| matches!(h.kind, Kind::Noise { band: b } if b == band));
        if !held {
            out.push(Kind::Noise { band });
        }
    }
    out.extend(shaped(p, residual));
    out
}

/// The shaped noise `residual` proposes, as `MODEL` states, `None` where every gain is 0.
fn shaped(p: &Problem, residual: &[Vec<f64>]) -> Option<Kind> {
    let bands = p.grid.bands();
    // Each band's weighted energy and correlation with the residual, frame by frame.
    let moments: Vec<Vec<(f64, f64)>> = bands
        .iter()
        .map(|band| {
            let t = &p.grid.noise_bins(*band)[p.range.clone()];
            (0..p.frames())
                .map(|f| p.moments(f, t, &residual[f]))
                .collect()
        })
        .collect();
    let ratio = |(e, c): (f64, f64)| match e > 0.0 {
        true => (c / e).max(0.0),
        false => 0.0,
    };
    // `sum w (u^2 - 2 r u)` over the bands, no cross terms, `u` the gains times the envelope.
    let misfit = |gains: &[f64], envelope: &[f64]| -> f64 {
        moments
            .iter()
            .zip(gains)
            .map(|(m, g)| {
                m.iter()
                    .zip(envelope)
                    .map(|((e, c), v)| g * g * v * v * e - 2.0 * g * v * c)
                    .sum::<f64>()
            })
            .sum()
    };
    let mut envelope = vec![1.0; p.frames()];
    let mut gains = vec![0.0; bands.len()];
    let mut held = f64::INFINITY;
    loop {
        gains = moments
            .iter()
            .map(|m| {
                ratio(
                    m.iter()
                        .zip(&envelope)
                        .fold((0.0, 0.0), |(e, c), ((me, mc), v)| {
                            (e + v * v * me, c + v * mc)
                        }),
                )
            })
            .collect();
        envelope = (0..p.frames())
            .map(|f| {
                ratio(
                    moments
                        .iter()
                        .zip(&gains)
                        .fold((0.0, 0.0), |(e, c), (m, g)| {
                            (e + g * g * m[f].0, c + g * m[f].1)
                        }),
                )
            })
            .collect();
        let now = misfit(&gains, &envelope);
        if now >= held {
            break;
        }
        held = now;
    }
    let top = gains.iter().copied().fold(0.0, f64::max);
    (top > 0.0).then(|| Kind::Shaped {
        bands: bands
            .into_iter()
            .zip(gains)
            .filter(|(_, g)| *g > 0.0)
            .map(|(band, g)| (band, g / top))
            .collect(),
    })
}

/// The DL change adding `kind` keyed `key` to `n` atoms makes in closed form against
/// `residual`, as `MODEL` states.
fn scored(p: &Problem, n: usize, residual: &[Vec<f64>], kind: &Kind, key: u64) -> (f64, Written) {
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
    let change = written.bits + bits::count(n + 1) - bits::count(n) - 0.5 * LOG2_E * drop;
    (change, written)
}

/// The progress line of `state` at pass `pass`, `before` the DL before it, as `Line` states.
fn line(p: &Problem, state: &State, pass: usize, before: f64, removed: Option<Atom>) -> Line {
    let grid = &p.grid;
    let residual_db = state
        .residual(p)
        .iter()
        .map(|r| {
            let mut full = vec![0.0; grid.bins()];
            for (i, r) in r.iter().enumerate() {
                full[p.range.start + i] = r.max(0.0);
            }
            grid.pooled(&full)
                .iter()
                .map(|v| (*v > 0.0).then(|| grid.band_db(*v)))
                .collect()
        })
        .collect();
    let atoms = state.atoms();
    let moved = state.moved.as_ref();
    let atom = moved.and_then(|m| {
        atoms
            .iter()
            .find(|a| a.key == m.key)
            .cloned()
            .or_else(|| removed.clone())
    });
    Line {
        pass,
        action: moved.map_or(Action::Start, |m| m.action),
        atom,
        from_db: moved.and_then(|m| m.from_db),
        to_db: moved.and_then(|m| m.to_db),
        delta_bits: state.dl() - before,
        atom_bits: state.atom_bits,
        residual_bits: state.residual_bits,
        residual_db,
        envelopes: state
            .held
            .iter()
            .map(|h| (h.written.atom.key, h.written.levels.clone()))
            .collect(),
        geometry: None,
        seconds: 0.0,
        atoms,
        clip: p.clip,
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

/// One sound's fit of `samples`, its start and each pass heard by `hear` as they end, `stop`
/// asked before each pass, as `MODEL` states.
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
    let mut state = State::empty(&p);
    let mut first = line(&p, &state, 0, state.dl(), None);
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
    let (mut passes, mut adds, mut adjusts, mut stopped) = (0, 0, 0, false);
    loop {
        if stop.abandoned().await {
            stopped = true;
            break;
        }
        let next = pass(&p, &state);
        let Some(moved) = &next.moved else {
            break;
        };
        passes += 1;
        match moved.action {
            Action::Add => adds += 1,
            _ => adjusts += 1,
        }
        let removed = state
            .atoms()
            .into_iter()
            .find(|a| a.key == moved.key && moved.to_db.is_none());
        let mut heard = line(&p, &next, passes, state.dl(), removed);
        heard.seconds = clock();
        hear(&heard);
        state = next;
    }
    let model = full(&p, &state.model(&p));
    Fit {
        clip,
        atoms: state.atoms(),
        gain_db,
        spl_db: heard_at,
        atom_bits: state.atom_bits,
        residual_bits: state.residual_bits,
        passes,
        adds,
        adjusts,
        stopped,
        clip_grid: p.grid.broad(),
        fit_grid: p.grid.with(model).broad(),
    }
}
