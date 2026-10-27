// Concern: FIT: one sound by passes of one move each, the best add or adjust under one DL | Non-concern: the spectrogram, floor and sieve (add.rs), JSON | IO: (samples, rate, spl) -> Fit, Lines

use std::f64::consts::LOG2_E;
use std::ops::Range;

use sva_engine::hearing::{CAM_FIRST, CAM_LAST, cam, cam_hz, erb_hz, level_jnd_db};
use sva_engine::{Abandon, default_threads, each, fft};

use super::add::{Members, Spectrogram, Track, floor_db, sieved, span_hz};
use super::atoms::{Atom, Clip, Feasible, Quantity, Sound, Trajectory, birth};
use super::bits::{self, Look, Priced, frequency_delta, level_delta};
use super::grid::Grid;
use crate::cli_error::CliError;

pub const MODEL: &str = "one sound fitted by passes of one move each: coordinate descent on the atoms' levels and frequencies beside matching pursuit for new atoms (Mallat and Zhang 1993, doi 10.1109/78.258082); \
     DL = B(x) (bits.rs) plus the residual's bits, the one objective; \
     every pass over one problem: the clip's grid, its weights read once off the clip and never off a residual; \
     every atom spans the clip; a partial's or voice's Hz f(t) = f(0) e^(glide t), glide 0 a fixed tone (atoms.rs); \
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
     a gliding column's template in frame m the periodogram, by one transform, of a unit sine on its path in the window centred on the frame, its sweep across that window held, the frame's other hops left out: that template is the model's prediction, as a fixed sine's lobe is, not an estimate of another one, so its gap from the frame's pooled hops is misfit the residual holds and the DL prices; \
     a gliding column's floor at its Hz at the frames' mean time, its frequency looks at its path's Hz in each frame; \
     each column's gain per frame its power; \
     each pass scores every move against the residual and makes the one that lowers the DL most, or none, and the fit stops at the first pass with none, so the DL falls at every pass, \
     or after the first pass whose drop is no more than the stopping threshold times the larger of the DL before and after it and 1, the relative tolerance optimisers' ftol reads (default 1e-3), a resumed fit's default 0; \
     a fit resumed from a composition holds its atoms as written, each kind and level curve read off its numbers, a glide at the rate floor's precision (its slope's standard error the add's, unwritten), its passes numbered on from those that made it; \
     a move an add or an adjust; \
     the model's power the sum of its atoms' powers, which holds only where no two atoms' tonal lines, a voice's every harmonic a line, lie within the main lobe's half-width (2 window bins) of each other in any frame: inside it two lines beat and their powers do not add; \
     so no add or frequency adjust leaves two atoms' lines within it: a candidate whose lines fall within it of a kept atom's line is no add, that kept atom's own adjusts (its level, frequency or rethin) the moves there, and two voices overlap where any of their lines do; \
     a move whose atom's node the parser or the engine refuses (atoms.rs) skipped for the next best, and named in its pass's line; \
     an add a new atom off the residual averaged over the frames, at its best level: \
     a free partial at each local maximum over its floor, its Hz by QIFFT (Smith and Serra 1987); \
     a partial or voice candidate, refined, glides where its columns' peak tracks glide: a column's track in each frame the highest local maximum over its floor, by QIFFT, in the basin of the averaged residual about its Hz, from the local maximum it climbs to down to the local minimum either side; \
     a glide one slope of ln Hz on time by least squares over its columns' tracks, each its own intercept, its precision the coarser of that slope's standard error and the rate moving the first column's Hz by its written precision over the frames, 0 within that of 0; \
     a glide scored only where it moves some line's Hz over the frames by the main lobe's half-width or more, a narrower sweep staying inside the place two lines there would share; \
     a gliding candidate starts at its tracks' intercepts, is refined as a fixed one is, and is kept where its DL is the lower; \
     a voice at each fundamental the sieve (add.rs) finds among those peaks that are tonal lobes of the averaged residual at K times its frames looks, taken from the pool with its members till none is found; \
     a voice's harmonic k at k f0 sqrt(1 + B k^2) (Fletcher 1964), f0 and B by least squares of (f_k / k)^2 = f0^2 + f0^2 B k^2 over its member peaks, B at least 0; \
     members reassigned within half an analysis bin of the stretched Hz and refitted till the set repeats; \
     every harmonic to the highest member its own column and its own level; \
     one noise band per octave band no kept noise band holds; \
     one shaped noise: a gain per octave band times one envelope over the frames (rank 1), both at least 0, by alternating least squares against the residual from a flat envelope till an alternation no longer lowers the weighted misfit, which never rises, its bands' cross terms left out of the alternation and in its score; a band whose gain falls to 0 carved out, the rest's gains over the highest's written in dB; \
     an add scored in closed form: each column's gain per frame its weighted correlation with the residual over its weighted energy, at least 0, written, the DL drop those gains give less its bits; \
     each partial's and voice's frequencies refined against the residual as it is scored, its glide held: a partial's Hz by golden section (Kiefer 1953) over half an analysis bin of its peak to the finer of the precision it is written at and its standard deviation under the weights, sigma = (sum w (du/dx)^2)^-1/2, u its expected power, du/dx by central difference at the cube root of epsilon; a voice's harmonics each so as a partial, then f0 and B by that least squares weighted by the inverse variance of (f_k / k)^2, (2 f_k sigma_k / k^2)^-2, kept where the voice's score falls; \
     an adjust changes one kept atom in its own written numbers, three for each: \
     level, one gain over its whole level curve, 1 + sum w r u / sum w u^2 in closed form (u its expected power), its levels' bases moved by it in dB; or to 0, which removes it, where that gives the lower DL or the gain is not positive; \
     frequency, for a partial or voice: its tracks read off the residual with the atom added back, in each frame the local maximum over its floor nearest its path's Hz in Cam within a main lobe of it, give a glide as an add's do; \
     at its own glide, at none and at that one where it moves a line so, its Hz at the start moved so its path holds at the frames' mean time, a partial's Hz by golden section over half a bin to the finer of its written precision and its standard deviation, a voice refined as an add's against that residual, its levels kept, the lowest DL kept; \
     rethin, its level curve thinned afresh from each column's written gain per frame plus the residual's weighted correlation with the column over its weighted energy, at least 0; \
     the DL of every move the written state's, in closed form from the residual, its drop exact; the residual derived afresh from the kept atoms at each pass, a move whose repriced DL does not fall, its drop under rounding, no move; \
     ties broken by the move's order: adds in the candidates' order, then adjusts in the atoms', each atom's level, frequency, then rethin; \
     a column's level 10 log10 of its gain, held at its floor; a shaped noise's floor the least of its bands' floors less their gains; \
     each level through the frame centres from 0 to the clip's end, thinned by Ramer-Douglas-Peucker at the level JND at its peak sensation level (Jesteadt, Wier and Green 1977); \
     the DL the written atoms': levels at the level JND by looks, frequencies at the frequency JND by looks (bits.rs), every frame a look at its level over its floor; \
     a shaped noise's band gain at the level JND by looks of that band's level over its own floor, its envelope by its looks over the envelope's floor; \
     a voice's f0 at the JND at f0, its B where its highest harmonic moves by that harmonic's JND; a glide at its precision; \
     a time at the frame, a noise band's edge one of the grid's band edges; \
     a key one past the last born; \
     each pass reported as it ends: its move and what an adjust changed, the atom's level before and after, the DL's drop and total, the residual's grid, the moves it skipped";

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

/// What an adjust changed, as `MODEL` states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tweak {
    Level,
    Remove,
    Frequency,
    Rethin,
}

impl Tweak {
    pub const ALL: [Tweak; 4] = [Tweak::Level, Tweak::Remove, Tweak::Frequency, Tweak::Rethin];

    pub fn name(self) -> &'static str {
        match self {
            Tweak::Level => "level",
            Tweak::Remove => "remove",
            Tweak::Frequency => "frequency",
            Tweak::Rethin => "rethin",
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
    pub tweak: Option<Tweak>,
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
    /// The clock as it ended, and its pass's time.
    pub seconds: f64,
    pub pass_secs: f64,
    /// The empty fit's DL.
    pub start_bits: f64,
    pub atoms: Vec<Atom>,
    pub clip: Clip,
    /// The atoms of the moves better than its own it skipped, their nodes refused.
    pub skipped: Vec<Atom>,
}

/// The default stopping threshold on a pass's relative drop, as `MODEL` states.
pub const TOLERANCE: f64 = 1e-3;

/// The atoms a fit resumes from, the passes that made them, and their master gain if written.
#[derive(Clone, Debug, PartialEq)]
pub struct Resume {
    pub atoms: Vec<Atom>,
    pub passes: usize,
    pub gain_db: Option<f64>,
}

impl Resume {
    /// The atoms a composition's `files` hold after `passes` passes.
    pub fn of(files: &[(String, String)], passes: usize) -> Result<Resume, CliError> {
        let atoms = super::atoms::read(files).ok_or_else(|| {
            CliError::usage(
                "solve.from_unread",
                format!(
                    "the composition to resume holds no `{}` of atoms a fit wrote",
                    super::atoms::SCORE
                ),
            )
        })?;
        Ok(Resume {
            atoms,
            passes,
            gain_db: super::atoms::gain_of(files),
        })
    }
}

/// What a fit is asked beside its clip: its stopping threshold, and the fit it resumes.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub tolerance: f64,
    pub from: Option<Resume>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            tolerance: TOLERANCE,
            from: None,
        }
    }
}

/// Why a fit ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    Converged,
    Tolerance,
    Stopped,
}

impl End {
    pub fn name(self) -> &'static str {
        match self {
            End::Converged => "converged",
            End::Tolerance => "tolerance",
            End::Stopped => "stopped",
        }
    }
}

/// A fit: the clip it read, its atoms in unit-RMS space, the master gain bringing them back,
/// the SPL that space is heard at, its DL's parts, the pass it resumed from and its last, this
/// run's adds and adjusts, these by `Tweak::ALL`, its threshold, why it ended, and both
/// coarse grids, each band's power spread flat.
pub struct Fit {
    pub clip: Clip,
    pub atoms: Vec<Atom>,
    pub gain_db: f64,
    pub spl_db: f64,
    pub atom_bits: f64,
    pub residual_bits: f64,
    pub from_pass: usize,
    pub passes: usize,
    pub adds: usize,
    pub adjusts: usize,
    pub tweaks: [usize; 4],
    pub tolerance: f64,
    pub ended: End,
    pub clip_grid: Grid,
    pub fit_grid: Grid,
}

/// A glide's rate in nepers per second, 0 a fixed tone, and its precision.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Glide {
    rate: f64,
    step: f64,
}

/// What an atom is before its level is written, its Hz at the clip's start.
#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Partial {
        hz: f64,
        glide: Glide,
    },
    Voice {
        f0: f64,
        stretch: f64,
        count: usize,
        glide: Glide,
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
            Kind::Partial { hz, .. } => vec![*hz],
            Kind::Voice {
                f0, stretch, count, ..
            } => (1..=*count)
                .map(|k| {
                    let k = k as f64;
                    f0 * k * libm::sqrt(1.0 + stretch * k * k)
                })
                .collect(),
            Kind::Noise { band } => vec![centre(*band)],
            Kind::Shaped { bands } => vec![centre(bands[0].0)],
        }
    }

    fn glide(&self) -> Glide {
        match self {
            Kind::Partial { glide, .. } | Kind::Voice { glide, .. } => *glide,
            Kind::Noise { .. } | Kind::Shaped { .. } => Glide::default(),
        }
    }

    /// This partial or voice at `to`, its Hz at the start moved so its path holds at `middle`.
    fn glided(&self, to: Glide, middle: f64) -> Kind {
        let moved = |hz: f64| hz * sva_formula::exp((self.glide().rate - to.rate) * middle);
        match self {
            Kind::Partial { hz, .. } => Kind::Partial {
                hz: moved(*hz),
                glide: to,
            },
            Kind::Voice {
                f0, stretch, count, ..
            } => Kind::Voice {
                f0: moved(*f0),
                stretch: *stretch,
                count: *count,
                glide: to,
            },
            Kind::Noise { .. } | Kind::Shaped { .. } => self.clone(),
        }
    }
}

/// Each column's Hz at the clip's start and per frame, its Hz at the frames' mean time, and
/// its floor in dB.
struct Columns {
    starts: Vec<f64>,
    paths: Vec<Vec<f64>>,
    heard: Vec<f64>,
    floors: Vec<f64>,
}

/// A column's template over the feasible bins, one for every frame or one per frame.
#[derive(Clone)]
struct Template(Vec<Vec<f64>>);

impl Template {
    fn at(&self, f: usize) -> &[f64] {
        match self.0.len() {
            1 => &self.0[0],
            _ => &self.0[f],
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
    window: Vec<f64>,
    twiddle: Vec<(f64, f64)>,
}

impl Problem {
    /// `grid` read as `MODEL` states, its power the target.
    fn of(grid: Grid, clip: Clip, spl_db: f64, gain_db: f64) -> Problem {
        let range = feasible(&grid);
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
        let window = (0..grid.long)
            .map(|k| 0.5 - 0.5 * libm::cos(std::f64::consts::TAU * k as f64 / grid.long as f64))
            .collect();
        let twiddle = fft::twiddles(grid.n);
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
            window,
            twiddle,
        }
    }

    fn frames(&self) -> usize {
        self.power.len()
    }
}

/// The bins the feasible set's Hz hold, as `MODEL` states.
fn feasible(grid: &Grid) -> Range<usize> {
    let (lo, hi) = span_hz(grid.rate);
    (libm::ceil(grid.bin_of(lo)) as usize).max(1)
        ..(libm::floor(grid.bin_of(hi)) as usize + 1).min(grid.bins().saturating_sub(1))
}

impl Problem {
    /// Each column's template over the feasible bins, a gliding one's per frame.
    fn templates(&self, kind: &Kind) -> Vec<Template> {
        match kind {
            Kind::Noise { band } => vec![Template(vec![
                self.grid.noise_bins(*band)[self.range.clone()].to_vec(),
            ])],
            Kind::Shaped { bands } => {
                let mut sum = vec![0.0; self.range.len()];
                for (band, gain) in bands {
                    let bins = self.grid.noise_bins(*band);
                    for (s, b) in sum.iter_mut().zip(&bins[self.range.clone()]) {
                        *s += gain * b;
                    }
                }
                vec![Template(vec![sum])]
            }
            _ => {
                let rate = kind.glide().rate;
                let at = |hz: f64| self.range.clone().map(|b| self.grid.sine(hz, b)).collect();
                let bent = match rate != 0.0 {
                    true => self.bent(rate),
                    false => Vec::new(),
                };
                kind.hzs()
                    .iter()
                    .map(|hz| {
                        Template(match rate != 0.0 {
                            true => self
                                .times
                                .iter()
                                .map(|t| self.swept(hz * sva_formula::exp(rate * t), &bent))
                                .collect(),
                            false => vec![at(*hz)],
                        })
                    })
                    .collect()
            }
        }
    }

    /// Radians per Hz at the window's centre a path gliding at `rate` turns by each sample,
    /// `2 pi (e^(rate s) - 1) / rate`, `s` its time from the centre.
    fn bent(&self, rate: f64) -> Vec<f64> {
        let long = self.grid.long;
        (0..long)
            .map(|k| {
                let from = (k as f64 - (long / 2) as f64) / self.grid.rate;
                std::f64::consts::TAU * libm::expm1(rate * from) / rate
            })
            .collect()
    }

    /// The power a unit sine at `hz` at a window's centre, turning `bent` per Hz there, puts
    /// in each feasible bin of that window: its periodogram, positive frequencies alone.
    fn swept(&self, hz: f64, bent: &[f64]) -> Vec<f64> {
        let n = self.grid.n;
        let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
        for (k, (w, turns)) in self.window.iter().zip(bent).enumerate() {
            let (sin, cos) = libm::sincos(hz * turns);
            (re[k], im[k]) = (0.5 * w * cos, 0.5 * w * sin);
        }
        fft::fft_with(&mut re, &mut im, &self.twiddle);
        self.range
            .clone()
            .map(|b| re[b] * re[b] + im[b] * im[b])
            .collect()
    }

    fn middle(&self) -> f64 {
        self.times.iter().sum::<f64>() / self.times.len().max(1) as f64
    }

    fn columns(&self, kind: &Kind) -> Columns {
        let rate = kind.glide().rate;
        let hzs = kind.hzs();
        let heard: Vec<f64> = hzs
            .iter()
            .map(|hz| hz * sva_formula::exp(rate * self.middle()))
            .collect();
        let floors = match kind {
            Kind::Shaped { bands } => vec![
                bands
                    .iter()
                    .map(|(band, g)| floor_db(centre(*band), self.spl_db) - db(*g))
                    .fold(f64::INFINITY, f64::min),
            ],
            _ => heard.iter().map(|hz| floor_db(*hz, self.spl_db)).collect(),
        };
        Columns {
            starts: hzs.clone(),
            paths: hzs
                .iter()
                .map(|hz| {
                    self.times
                        .iter()
                        .map(|t| hz * sva_formula::exp(rate * t))
                        .collect()
                })
                .collect(),
            heard,
            floors,
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

    /// The frames' looks at a level `level` over `floor` at `hz` per frame.
    fn looks(&self, level: &[f64], floor: f64, hz: &[f64]) -> Vec<Look> {
        level
            .iter()
            .zip(hz)
            .map(|(l, hz)| Look {
                sl_db: l - floor,
                hz: *hz,
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
    let floors = p.columns(kind).floors;
    let trajectories: Vec<Trajectory> = fitted
        .iter()
        .zip(&floors)
        .map(|(g, floor)| {
            let held: Vec<f64> = g.iter().map(|g| db(*g).max(*floor)).collect();
            let peak = held.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            thinned((0.0, secs), &p.times, &held, level_jnd_db(peak - floor))
        })
        .collect();
    rewrite(p, kind, key, trajectories)
}

/// `kind` keyed `key` written at each column's level `trajectories`.
fn rewrite(p: &Problem, kind: &Kind, key: u64, trajectories: Vec<Trajectory>) -> Written {
    let columns = p.columns(kind);
    let glide = kind.glide();
    let levels: Vec<Vec<f64>> = trajectories
        .iter()
        .map(|t| levels_at(t, &p.times))
        .collect();
    let gains = levels
        .iter()
        .map(|l| l.iter().map(|l| libm::pow(10.0, l / 10.0)).collect())
        .collect();
    let mut trajectories = trajectories.into_iter();
    let first = trajectories.next().expect("a column");
    let atom = match kind {
        Kind::Partial { hz, .. } => {
            Atom::partial(key, 0.0, Trajectory::steady(*hz), first).glided(glide.rate)
        }
        Kind::Voice { f0, stretch, .. } => Atom::harmonic(
            key,
            0.0,
            (Trajectory::steady(*f0), *stretch),
            std::iter::once(first).chain(trajectories).collect(),
        )
        .glided(glide.rate),
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
        Kind::Shaped { bands } => shaped_bits(p, &atom, bands, &levels[0], columns.floors[0]),
        _ => priced(p, &atom, &columns, &levels, glide.step),
    };
    Written {
        atom,
        gains,
        levels,
        bits,
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
        let hz = vec![centre(*band); level.len()];
        let at = p.looks(&heard, floor_db(centre(*band), p.spl_db), &hz);
        deltas.extend([edge, edge, level_delta(hop, &at)]);
    }
    let hz = vec![centre(bands[0].0); level.len()];
    let envelope = level_delta(hop, &p.looks(level, floor, &hz));
    for (q, _) in &atom.numbers()[deltas.len()..] {
        deltas.push(match q {
            Quantity::Time => hop,
            _ => envelope,
        });
    }
    bits::atom(atom, &Priced { feasible, deltas })
}

/// `atom`'s bits at the precisions `MODEL` states, harmonic `k` at `levels[k]` per frame over
/// its column's floor, its glide at `glide_step`.
fn priced(
    p: &Problem,
    atom: &Atom,
    columns: &Columns,
    levels: &[Vec<f64>],
    glide_step: f64,
) -> f64 {
    let hop = p.grid.frame_secs();
    let looks = |k: usize| p.looks(&levels[k], columns.floors[k], &columns.paths[k]);
    let lowest = columns.floors.iter().copied().fold(f64::INFINITY, f64::min);
    let feasible = Feasible::of(&p.clip, (lowest, p.top_db));
    let level_deltas: Vec<f64> = (0..columns.heard.len())
        .map(|k| level_delta(hop, &looks(k)))
        .collect();
    let mut steps = None;
    let mut steps = || *steps.get_or_insert_with(|| frequency_steps(p, columns, levels));
    let mut level = 0;
    let mut deltas = Vec::new();
    for (q, _) in atom.numbers() {
        if q == Quantity::Db {
            level += 1;
        }
        let member = level % columns.heard.len();
        deltas.push(match q {
            Quantity::Time => hop,
            Quantity::Db | Quantity::DbDeviation => level_deltas[member],
            Quantity::Hz | Quantity::HzDeviation => steps().0,
            Quantity::Stretch => steps().1,
            Quantity::Glide => glide_step,
            Quantity::Edge => edge_step(p, &feasible),
        });
    }
    bits::atom(atom, &Priced { feasible, deltas })
}

/// The precisions a fundamental's Hz and its stretch are written at, as `MODEL` states,
/// harmonic `k` at `levels[k]` per frame over its column's floor.
fn frequency_steps(p: &Problem, columns: &Columns, levels: &[Vec<f64>]) -> (f64, f64) {
    let hop = p.grid.frame_secs();
    let hzs = &columns.heard;
    let looks = |k: usize| p.looks(&levels[k], columns.floors[k], &columns.paths[k]);
    let last = hzs.len() - 1;
    let (f0, h, top) = (hzs[0], hzs.len() as f64, hzs[last]);
    let moved = h * h * h * h * f0 * f0 / (2.0 * top);
    (
        frequency_delta(hop, &looks(0)) * columns.starts[0] / f0,
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
                        .map(|((l, u), g)| g[f] * (u.at(f)[b] - l.at(f)[b]))
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
    let glide = kind.glide();
    let partial_at = |hz: f64| Kind::Partial { hz, glide };
    // Harmonic `c` at `hz` refined as a partial: its Hz, score and standard deviation.
    let partial = |c: usize, hz: f64, held: f64| {
        let h = bin * libm::cbrt(f64::EPSILON);
        let near = (&partial_at(hz - h), &partial_at(hz + h));
        let sigma = spread(p, near, h, &written.gains[c..=c]);
        let columns = p.columns(&partial_at(hz));
        let written_step = frequency_steps(p, &columns, &written.levels[c..=c]).0;
        let at = |h: f64| score(&partial_at(h));
        let range = (hz - bin / 2.0, hz + bin / 2.0);
        let (to, v) = golden(&at, range, written_step.min(sigma), (hz, held));
        (to, v, sigma)
    };
    match kind {
        Kind::Partial { hz, .. } => {
            let (hz, v, _) = partial(0, hz, best);
            (partial_at(hz), v)
        }
        Kind::Voice { count, .. } => {
            let hzs = kind.hzs();
            let refined = each(default_threads(), (0..count).collect(), |c| {
                let held = score(&partial_at(hzs[c]));
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
            let fitted = Kind::Voice {
                f0,
                stretch,
                count,
                glide,
            };
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
/// the last on to `end`, thinned by Ramer-Douglas-Peucker at `tolerance` dB, its knots' times
/// from `on`.
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
        knots: kept[1..].iter().map(|(t, l)| (t - on, l - base)).collect(),
    }
}

/// The level in dB `level` holds at each of the ascending `times`, its knots' times from 0:
/// straight lines from `base` through each knot, held after the last, its knots walked once.
fn levels_at(level: &Trajectory, times: &[f64]) -> Vec<f64> {
    let mut from = (0.0, level.base);
    let mut knots = level
        .knots
        .iter()
        .map(|(at, d)| (*at, level.base + d))
        .peekable();
    times
        .iter()
        .map(|t| {
            while let Some(to) = knots.peek().copied() {
                if *t <= to.0 {
                    return match to.0 > from.0 {
                        true => from.1 + (to.1 - from.1) * (t - from.0) / (to.0 - from.0),
                        false => to.1,
                    };
                }
                from = to;
                knots.next();
            }
            from.1
        })
        .collect()
}

/// A kept atom: its kind, each column's template, and the atom as written.
#[derive(Clone)]
struct Held {
    kind: Kind,
    templates: Vec<Template>,
    written: Written,
}

impl Held {
    fn model(&self, p: &Problem, f: usize) -> Vec<f64> {
        let mut u = vec![0.0; p.range.len()];
        for (t, g) in self.templates.iter().zip(&self.written.gains) {
            u.iter_mut().zip(t.at(f)).for_each(|(u, t)| *u += g[f] * t);
        }
        u
    }
}

/// The move a pass made: its action and tweak, the atom's key, and its level before and after.
#[derive(Clone)]
struct Moved {
    action: Action,
    tweak: Option<Tweak>,
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
    skipped: Vec<Atom>,
}

impl State {
    fn empty(p: &Problem) -> State {
        let mut state = State {
            held: Vec::new(),
            born: 0,
            atom_bits: 0.0,
            residual_bits: 0.0,
            moved: None,
            skipped: Vec::new(),
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
                for (f, (r, g)) in out.iter_mut().zip(g).enumerate() {
                    r.iter_mut().zip(t.at(f)).for_each(|(r, t)| *r -= g * t);
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

/// `sum w (u^2 - 2 r u)` over every frame against `residual`, `u` frame `f`'s change
/// `delta(f)`: the residual's weighted square's change.
fn change(p: &Problem, residual: &[Vec<f64>], delta: &dyn Fn(usize) -> Vec<f64>) -> f64 {
    (0..p.frames())
        .map(|f| {
            delta(f)
                .iter()
                .zip(&residual[f])
                .zip(&p.weight[f])
                .map(|((u, r), w)| w * (u * u - 2.0 * r * u))
                .sum::<f64>()
        })
        .sum()
}

fn rewritten(held: &Held, written: Written) -> Held {
    Held {
        kind: held.kind.clone(),
        templates: held.templates.clone(),
        written,
    }
}

/// Atom `a`'s level adjust, as `MODEL` states: its DL change, its tweak and what it leaves.
fn scaled(
    p: &Problem,
    state: &State,
    residual: &[Vec<f64>],
    a: usize,
) -> (f64, Tweak, Option<Held>) {
    let held = &state.held[a];
    let (energy, pull) = (0..p.frames()).fold((0.0, 0.0), |(e, c), f| {
        held.model(p, f)
            .iter()
            .zip(&residual[f])
            .zip(&p.weight[f])
            .fold((e, c), |(e, c), ((u, r), w)| (e + w * u * u, c + w * u * r))
    });
    let n = state.held.len();
    // Removing it changes the residual by +u: sum w (u^2 + 2 r u).
    let gone = bits::count(n - 1) - bits::count(n) - held.written.bits
        + 0.5 * LOG2_E * (energy + 2.0 * pull);
    let gain = match energy > 0.0 {
        true => 1.0 + pull / energy,
        false => 1.0,
    };
    if gain <= 0.0 {
        return (gone, Tweak::Remove, None);
    }
    let trajectories = held
        .written
        .atom
        .levels()
        .into_iter()
        .map(|t| Trajectory {
            base: t.base + db(gain),
            knots: t.knots.clone(),
        })
        .collect();
    let written = rewrite(p, &held.kind, held.written.atom.key, trajectories);
    // The residual changes by -(gain - 1) u: sum w (d^2 u^2 - 2 d r u).
    let d = gain - 1.0;
    let kept = written.bits - held.written.bits + 0.5 * LOG2_E * (d * d * energy - 2.0 * d * pull);
    match gone < kept {
        true => (gone, Tweak::Remove, None),
        false => (kept, Tweak::Level, Some(rewritten(held, written))),
    }
}

/// Atom `a`'s rethin, as `MODEL` states: its DL change beside the atom it leaves.
fn rethinned(p: &Problem, state: &State, residual: &[Vec<f64>], a: usize) -> (f64, Held) {
    let held = &state.held[a];
    let before = &held.written.gains;
    let fitted: Vec<Vec<f64>> = held
        .templates
        .iter()
        .zip(before)
        .map(|(t, g)| {
            (0..p.frames())
                .map(|f| {
                    let (energy, pull) = p.moments(f, t.at(f), &residual[f]);
                    match energy > 0.0 {
                        true => (g[f] + pull / energy).max(0.0),
                        false => g[f],
                    }
                })
                .collect()
        })
        .collect();
    let written = write(p, &held.kind, held.written.atom.key, &fitted);
    let delta = |f: usize| {
        let mut u = vec![0.0; p.range.len()];
        for ((t, after), before) in held.templates.iter().zip(&written.gains).zip(before) {
            let d = after[f] - before[f];
            if d != 0.0 {
                u.iter_mut().zip(t.at(f)).for_each(|(u, t)| *u += d * t);
            }
        }
        u
    };
    let kept = written.bits - held.written.bits + 0.5 * LOG2_E * change(p, residual, &delta);
    (kept, rewritten(held, written))
}

/// The Hz at the clip's start a track of `(t, ln hz)` gives at `rate`.
fn start(track: &[(f64, f64)], rate: f64) -> Option<f64> {
    (!track.is_empty()).then(|| {
        sva_formula::exp(track.iter().map(|(t, y)| y - rate * t).sum::<f64>() / track.len() as f64)
    })
}

/// Atom `a`'s frequency adjust, as `MODEL` states, `None` for a noise.
fn retuned(p: &Problem, state: &State, residual: &[Vec<f64>], a: usize) -> Option<(f64, Held)> {
    let held = &state.held[a];
    if matches!(held.kind, Kind::Noise { .. } | Kind::Shaped { .. }) {
        return None;
    }
    let key = held.written.atom.key;
    let alone: Vec<Vec<f64>> = residual
        .iter()
        .enumerate()
        .map(|(f, r)| r.iter().zip(held.model(p, f)).map(|(r, u)| r + u).collect())
        .collect();
    let columns = p.columns(&held.kind);
    let tracks: Vec<Vec<(f64, f64)>> = columns
        .paths
        .iter()
        .map(|path| pathed(p, &alone, path))
        .collect();
    let trajectories: Vec<Trajectory> = held.written.atom.levels().into_iter().cloned().collect();
    let score = |kind: &Kind| -> (f64, Held) {
        let templates = p.templates(kind);
        let written = rewrite(p, kind, key, trajectories.clone());
        let delta = |f: usize| {
            let mut u = vec![0.0; p.range.len()];
            for ((after, before), g) in templates.iter().zip(&held.templates).zip(&written.gains) {
                for ((u, x), y) in u.iter_mut().zip(after.at(f)).zip(before.at(f)) {
                    *u += g[f] * (x - y);
                }
            }
            u
        };
        let v = written.bits - held.written.bits + 0.5 * LOG2_E * change(p, residual, &delta);
        let held = Held {
            kind: kind.clone(),
            templates,
            written,
        };
        (v, held)
    };
    let value = |kind: &Kind| score(kind).0;
    let current = held.kind.glide();
    let fitted = slope(&tracks, rate_floor(p, &columns, &held.written.levels));
    let mut glides = vec![current];
    for g in [Glide::default(), fitted] {
        let heard = g.rate == 0.0 || moves(p, &held.kind.glided(g, p.middle()));
        if heard && !glides.iter().any(|h| h.rate == g.rate) {
            glides.push(g);
        }
    }
    let others = state.held.len() - 1;
    glides
        .into_iter()
        .map(|glide| {
            let kind = match held.kind.glided(glide, p.middle()) {
                Kind::Partial { hz, glide } => tuned(p, hz, glide, &value, &held.written),
                voice => refined(p, others, &alone, voice, key).0,
            };
            score(&kind)
        })
        .min_by(|x, y| x.0.total_cmp(&y.0))
}

/// A partial's Hz by golden section against `value` over half an analysis bin of `hz`, its
/// levels `written`'s, to the finer of its written precision and its standard deviation.
fn tuned(
    p: &Problem,
    hz: f64,
    glide: Glide,
    value: &dyn Fn(&Kind) -> f64,
    written: &Written,
) -> Kind {
    let bin = p.grid.rate / p.grid.n as f64;
    let h = bin * libm::cbrt(f64::EPSILON);
    let at = |hz: f64| Kind::Partial { hz, glide };
    let step = frequency_steps(p, &p.columns(&at(hz)), &written.levels).0;
    let sigma = spread(p, (&at(hz - h), &at(hz + h)), h, &written.gains);
    let score = |hz: f64| value(&at(hz));
    let range = (hz - bin / 2.0, hz + bin / 2.0);
    at(golden(&score, range, step.min(sigma), (hz, score(hz))).0)
}

/// The rate that moves `columns`' first Hz by its written precision over the frames, at
/// levels `levels`: the finest glide heard apart from none.
fn rate_floor(p: &Problem, columns: &Columns, levels: &[Vec<f64>]) -> f64 {
    let span = p.times.last().copied().unwrap_or(0.0) - p.times.first().copied().unwrap_or(0.0);
    frequency_steps(p, columns, levels).0 / columns.starts[0] / span
}

/// The glide `tracks` of `(t, ln hz)` give, as `MODEL` states: one slope of ln Hz on time by
/// least squares, each track its own intercept, written at the coarser of its standard error
/// and `floor`; 0 where it is within that of 0 or too few points hold it.
fn slope(tracks: &[Vec<(f64, f64)>], floor: f64) -> Glide {
    type Track<'a> = (&'a [(f64, f64)], (f64, f64));
    let means = |t: &[(f64, f64)]| {
        let n = t.len() as f64;
        (
            t.iter().map(|x| x.0).sum::<f64>() / n,
            t.iter().map(|x| x.1).sum::<f64>() / n,
        )
    };
    let held: Vec<Track> = tracks
        .iter()
        .filter(|t| t.len() >= 2)
        .map(|t| (t.as_slice(), means(t)))
        .collect();
    let (mut sxx, mut sxy, mut count) = (0.0, 0.0, 0);
    for (t, (mt, my)) in &held {
        for (x, y) in t.iter() {
            sxx += (x - mt) * (x - mt);
            sxy += (x - mt) * (y - my);
        }
        count += t.len();
    }
    let dof = count as f64 - held.len() as f64 - 1.0;
    if dof < 1.0 || sxx <= 0.0 {
        return Glide::default();
    }
    let rate = sxy / sxx;
    let rss: f64 = held
        .iter()
        .map(|(t, (mt, my))| {
            t.iter()
                .map(|(x, y)| {
                    let e = y - my - rate * (x - mt);
                    e * e
                })
                .sum::<f64>()
        })
        .sum();
    let step = libm::sqrt(rss / dof / sxx).max(floor);
    match step > 0.0 && rate.abs() > step {
        true => Glide { rate, step },
        false => Glide::default(),
    }
}

/// The Hz and level in dBFS of `power`'s local maximum at `i` by QIFFT (Smith and Serra 1987).
fn qifft(p: &Problem, power: &[f64], i: usize) -> (f64, f64) {
    let dbs = |v: f64| db(v.max(f64::MIN_POSITIVE));
    let (a, c, e) = (dbs(power[i - 1]), dbs(power[i]), dbs(power[i + 1]));
    let curve = a - 2.0 * c + e;
    let shift = match curve < 0.0 {
        true => 0.5 * (a - e) / curve,
        false => 0.0,
    };
    let hz = p.grid.hz(p.range.start + i) + shift * p.grid.rate / p.grid.n as f64;
    let peak = libm::pow(10.0, (c - 0.25 * (a - e) * shift) / 10.0);
    (
        hz,
        20.0 * libm::log10(2.0 * libm::sqrt(peak) / p.grid.sum_w),
    )
}

/// The local maxima of `power` in `bins` over their floors, each its Hz beside its power.
fn maxima(p: &Problem, power: &[f64], bins: Range<usize>) -> Vec<(f64, f64)> {
    let bins = bins.start.max(1)..bins.end.min(power.len().saturating_sub(1));
    bins.filter(|i| power[*i] > power[i - 1] && power[*i] >= power[i + 1])
        .map(|i| (qifft(p, power, i), power[i]))
        .filter(|((hz, level), _)| *level > floor_db(*hz, p.spl_db))
        .map(|((hz, _), power)| (hz, power))
        .collect()
}

fn bins_of(p: &Problem, (lo, hi): (f64, f64)) -> Range<usize> {
    let local = |hz: f64| p.grid.bin_of(hz) - p.range.start as f64;
    let first = libm::ceil(local(lo)).max(0.0) as usize;
    let last = libm::floor(local(hi)).max(-1.0) + 1.0;
    first..(last.max(0.0) as usize).min(p.range.len())
}

fn lobe_hz(p: &Problem) -> f64 {
    p.lobes.half as f64 * p.grid.rate / p.grid.n as f64
}

/// Each tonal line's Hz per frame, a voice's every harmonic one, none for a noise.
fn lines(p: &Problem, kind: &Kind) -> Vec<Vec<f64>> {
    match kind {
        Kind::Noise { .. } | Kind::Shaped { .. } => Vec::new(),
        _ => p.columns(kind).paths,
    }
}

/// Whether some line of `kind` moves over the frames by the main lobe's half-width or more.
fn moves(p: &Problem, kind: &Kind) -> bool {
    let lobe = lobe_hz(p);
    lines(p, kind).iter().any(|path| {
        let (lo, hi) = path
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), hz| {
                (lo.min(*hz), hi.max(*hz))
            });
        hi - lo >= lobe
    })
}

/// Whether any of `a`'s lines lies within the main lobe's half-width of any of `b`'s in some
/// frame, where their powers do not add, as `MODEL` states.
fn collide(p: &Problem, a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    let lobe = lobe_hz(p);
    a.iter().any(|x| {
        b.iter()
            .any(|y| x.iter().zip(y).any(|(x, y)| (x - y).abs() < lobe))
    })
}

/// The peak nearest each frame's Hz on `path` in Cam within a main lobe of it, as
/// `(t, ln hz)`.
fn pathed(p: &Problem, power: &[Vec<f64>], path: &[f64]) -> Vec<(f64, f64)> {
    let lobe = lobe_hz(p);
    path.iter()
        .enumerate()
        .filter_map(|(f, hz)| {
            let target = cam(*hz);
            maxima(p, &power[f], bins_of(p, (hz - lobe, hz + lobe)))
                .into_iter()
                .map(|(h, _)| h)
                .min_by(|a, b| {
                    (cam(*a) - target)
                        .abs()
                        .total_cmp(&(cam(*b) - target).abs())
                })
                .map(|h| (p.times[f], libm::log(h)))
        })
        .collect()
}

/// The basin of `mean` about the bin nearest `hz`: from the local maximum it climbs to, down
/// to the local minimum on each side.
fn basin(p: &Problem, mean: &[f64], hz: f64) -> Range<usize> {
    let bin = libm::round(p.grid.bin_of(hz)) as usize;
    let mut i = bin.clamp(p.range.start, p.range.end - 1) - p.range.start;
    loop {
        let up = [i.checked_sub(1), (i + 1 < mean.len()).then_some(i + 1)]
            .into_iter()
            .flatten()
            .filter(|j| mean[*j] > mean[i])
            .max_by(|a, b| mean[*a].total_cmp(&mean[*b]).then(b.cmp(a)));
        match up {
            Some(j) => i = j,
            None => break,
        }
    }
    let (mut lo, mut hi) = (i, i);
    while lo > 0 && mean[lo - 1] < mean[lo] {
        lo -= 1;
    }
    while hi + 1 < mean.len() && mean[hi + 1] < mean[hi] {
        hi += 1;
    }
    lo..hi + 1
}

/// The peak track a candidate at `hz` holds through `power`, as `MODEL` states: in each frame
/// the highest local maximum over its floor in the basin of `mean` about `hz`, as `(t, ln hz)`.
fn searched(p: &Problem, power: &[Vec<f64>], mean: &[f64], hz: f64) -> Vec<(f64, f64)> {
    let bins = basin(p, mean, hz);
    (0..p.frames())
        .filter_map(|f| {
            maxima(p, &power[f], bins.clone())
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(h, _)| (p.times[f], libm::log(h)))
        })
        .collect()
}

/// `kind`, refined, gliding where its columns' peak tracks glide past the precision its
/// fixed tone's written levels give, as `MODEL` states; `None` where they do not.
fn gliding(
    p: &Problem,
    n: usize,
    residual: &[Vec<f64>],
    mean: &[f64],
    kind: &Kind,
    key: u64,
) -> Option<Kind> {
    if matches!(kind, Kind::Noise { .. } | Kind::Shaped { .. }) {
        return None;
    }
    let tracks: Vec<Vec<(f64, f64)>> = kind
        .hzs()
        .iter()
        .map(|hz| searched(p, residual, mean, *hz))
        .collect();
    let (_, written) = scored(p, n, residual, kind, key);
    let glide = slope(&tracks, rate_floor(p, &p.columns(kind), &written.levels));
    if glide.rate == 0.0 || !moves(p, &kind.glided(glide, p.middle())) {
        return None;
    }
    let starts: Vec<(usize, f64, f64)> = tracks
        .iter()
        .enumerate()
        .filter_map(|(c, t)| start(t, glide.rate).map(|s| (c + 1, s, 1.0)))
        .collect();
    let moved = kind.glided(glide, p.middle());
    Some(match (&moved, &starts[..]) {
        (Kind::Partial { .. }, [(_, hz, _), ..]) => Kind::Partial { hz: *hz, glide },
        (Kind::Voice { count, .. }, [_, _, ..]) => {
            let (f0, stretch) = stretched(&starts);
            Kind::Voice {
                f0,
                stretch,
                count: *count,
                glide,
            }
        }
        _ => moved,
    })
}

/// A move: an add of a kind, or an adjust of a kept atom to what it leaves, `None` removing it.
enum Move {
    Add(Kind),
    Adjust(usize, Option<Box<Held>>, Tweak),
}

/// A move about to be made, an add's atom written.
enum Made {
    Add(Kind, Box<Written>),
    Adjust(usize, Option<Box<Held>>, Tweak),
}

/// `run` over `tasks`, `stop` asked before each turn of a task per thread; `None` if it stops.
async fn paced<T: Send, R: Send>(
    stop: &dyn Abandon,
    tasks: Vec<T>,
    run: impl Fn(T) -> R + Sync,
) -> Option<Vec<R>> {
    let threads = default_threads();
    let mut tasks = tasks.into_iter().peekable();
    let mut out = Vec::new();
    while tasks.peek().is_some() {
        if stop.abandoned().await {
            return None;
        }
        let turn: Vec<T> = tasks.by_ref().take(threads.get()).collect();
        out.extend(each(threads, turn, &run));
    }
    Some(out)
}

/// One pass over `p` from `state`, as `MODEL` states; its `moved` `None` where no move lowers
/// the DL, `None` where `stop` ends it first.
async fn pass(p: &Problem, state: &State, stop: &dyn Abandon) -> Option<State> {
    let key = state.born;
    let n = state.held.len();
    let residual = state.residual(p);
    let mean = averaged(p, &residual);
    let kept: Vec<Vec<Vec<f64>>> = state.held.iter().map(|h| lines(p, &h.kind)).collect();
    // Whether `kind`'s lines lie apart from every kept atom's but atom `but`'s.
    let apart = |kind: &Kind, but: Option<usize>| {
        let mine = lines(p, kind);
        kept.iter()
            .enumerate()
            .all(|(i, theirs)| Some(i) == but || !collide(p, &mine, theirs))
    };
    let proposed: Vec<Kind> = candidates(p, state, &residual)
        .into_iter()
        .filter(|kind| apart(kind, None))
        .collect();
    // Fixed candidates, then gliding variants, so each turn's tasks cost alike.
    let fixed = paced(stop, proposed, |kind| {
        let (fixed, change) = refined(p, n, &residual, kind, key);
        let glide = gliding(p, n, &residual, &mean, &fixed, key);
        (fixed, change, glide)
    })
    .await?;
    let tried: Vec<(usize, Kind)> = fixed
        .iter()
        .enumerate()
        .filter_map(|(i, (_, _, glide))| glide.clone().map(|g| (i, g)))
        .collect();
    let mut glided: Vec<Option<(Kind, f64)>> = vec![None; fixed.len()];
    let refined_glides = paced(stop, tried, |(i, kind)| {
        (i, refined(p, n, &residual, kind, key))
    })
    .await?;
    for (i, made) in refined_glides {
        glided[i] = Some(made);
    }
    let adds = fixed
        .into_iter()
        .zip(glided)
        .map(|((fixed, change, _), glided)| {
            let glided = glided.filter(|(kind, glided)| *glided < change && apart(kind, None));
            let (kind, change) = glided.unwrap_or((fixed, change));
            apart(&kind, None).then_some((change, Move::Add(kind)))
        });
    let adjusts = paced(stop, (0..n).collect(), |a| {
        let (change, tweak, held) = scaled(p, state, &residual, a);
        let mut out = vec![(change, Move::Adjust(a, held.map(Box::new), tweak))];
        let retune = retuned(p, state, &residual, a).filter(|(_, h)| apart(&h.kind, Some(a)));
        if let Some((change, held)) = retune {
            out.push((
                change,
                Move::Adjust(a, Some(Box::new(held)), Tweak::Frequency),
            ));
        }
        let (change, held) = rethinned(p, state, &residual, a);
        out.push((change, Move::Adjust(a, Some(Box::new(held)), Tweak::Rethin)));
        out
    })
    .await?;
    let mut ranked: Vec<(f64, Move)> = adds
        .flatten()
        .chain(adjusts.into_iter().flatten())
        .filter(|(change, _)| *change < 0.0)
        .collect();
    // Stable, so ties keep the moves' order.
    ranked.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut next = state.clone();
    next.moved = None;
    next.skipped = Vec::new();
    let mut best = None;
    for (_, made) in ranked {
        let (atom, made) = match made {
            Move::Add(kind) => {
                let (_, written) = scored(p, n, &residual, &kind, key);
                (
                    Some(written.atom.clone()),
                    Made::Add(kind, Box::new(written)),
                )
            }
            Move::Adjust(a, held, tweak) => (
                held.as_ref().map(|h| h.written.atom.clone()),
                Made::Adjust(a, held, tweak),
            ),
        };
        match atom.filter(|atom| !atom.writes(&p.clip)) {
            Some(refused) => next.skipped.push(refused),
            None => {
                best = Some(made);
                break;
            }
        }
    }
    let Some(made) = best else {
        return Some(next);
    };
    let moved = match made {
        Made::Add(kind, written) => {
            let to_db = Some(level_db(&kind, &written));
            next.held.push(Held {
                templates: p.templates(&kind),
                kind,
                written: *written,
            });
            next.born += 1;
            Moved {
                action: Action::Add,
                tweak: None,
                key,
                from_db: None,
                to_db,
            }
        }
        Made::Adjust(a, made, tweak) => {
            let held = &state.held[a];
            let moved = Moved {
                action: Action::Adjust,
                tweak: Some(tweak),
                key: held.written.atom.key,
                from_db: Some(level_db(&held.kind, &held.written)),
                to_db: made.as_ref().map(|h| level_db(&h.kind, &h.written)),
            };
            match made {
                Some(h) => next.held[a] = *h,
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
        let skipped = std::mem::take(&mut next.skipped);
        next = state.clone();
        next.moved = None;
        next.skipped = skipped;
        return Some(next);
    }
    next.moved = Some(moved);
    Some(next)
}

/// The bins `power` peaks at over their floor, each a track spanning the clip beside its bin,
/// as `MODEL` states.
fn peaks(p: &Problem, power: &[f64]) -> Vec<(Track, usize)> {
    let span = (0.0, p.clip.secs());
    let mut out = Vec::new();
    for i in 1..power.len().saturating_sub(1) {
        if power[i] <= power[i - 1] || power[i] < power[i + 1] {
            continue;
        }
        let (hz, level) = qifft(p, power, i);
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
            glide: Glide::default(),
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

fn averaged(p: &Problem, residual: &[Vec<f64>]) -> Vec<f64> {
    (0..p.range.len())
        .map(|b| residual.iter().map(|r| r[b]).sum::<f64>() / residual.len() as f64)
        .collect()
}

/// Every candidate the residual `residual` proposes, as `MODEL` states.
fn candidates(p: &Problem, state: &State, residual: &[Vec<f64>]) -> Vec<Kind> {
    let mean = averaged(p, residual);
    let found = peaks(p, &mean);
    let mut out: Vec<Kind> = found
        .iter()
        .map(|(t, _)| Kind::Partial {
            hz: t.hz,
            glide: Glide::default(),
        })
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
                .map(|f| p.moments(f, t.at(f), &residual[f]))
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
                    for (u, t) in u.iter_mut().zip(t.at(f)) {
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

/// The progress line of `state` at pass `pass`, `before` the DL before it.
fn line(
    p: &Problem,
    state: &State,
    (pass, before, start_bits): (usize, f64, f64),
    removed: Option<Atom>,
) -> Line {
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
        tweak: moved.and_then(|m| m.tweak),
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
        pass_secs: 0.0,
        start_bits,
        atoms,
        clip: p.clip,
        skipped: state.skipped.clone(),
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

/// The state `from` leaves on `p`, or why it is no fit of this clip.
fn resumed(p: &Problem, from: &Resume, gain_db: f64) -> Result<State, CliError> {
    let unread = |why: &str| {
        CliError::usage(
            "solve.from_unread",
            format!("the composition to resume holds {why}, which no fit writes"),
        )
    };
    let mismatch = |why: String| {
        CliError::usage(
            "solve.from_mismatch",
            format!("the composition to resume was fitted to another clip: {why}"),
        )
    };
    if let Some(written) = from.gain_db.filter(|g| *g != gain_db) {
        return Err(mismatch(format!(
            "its master gain is {written} dB, this clip's {gain_db} dB"
        )));
    }
    let secs = p.clip.secs();
    let mut held = Vec::new();
    for atom in &from.atoms {
        if atom.on != 0.0 || atom.end() != secs {
            return Err(mismatch(format!(
                "atom {} spans {}s to {}s, this clip 0s to {secs}s",
                atom.key,
                atom.on,
                atom.end()
            )));
        }
        if from.atoms.iter().filter(|a| a.key == atom.key).count() > 1 {
            return Err(unread("two atoms of one key"));
        }
        if !atom.values().iter().all(|v| v.is_finite()) {
            return Err(unread("a number that is not finite"));
        }
        let kind = kind_of(atom).ok_or_else(|| unread("an atom of a shape"))?;
        let trajectories: Vec<Trajectory> = atom.levels().into_iter().cloned().collect();
        let kind = match kind.glide().rate != 0.0 {
            true => {
                let levels: Vec<Vec<f64>> = trajectories
                    .iter()
                    .map(|t| levels_at(t, &p.times))
                    .collect();
                let step = rate_floor(p, &p.columns(&kind), &levels);
                kind.glided(
                    Glide {
                        rate: kind.glide().rate,
                        step,
                    },
                    0.0,
                )
            }
            false => kind,
        };
        let written = rewrite(p, &kind, atom.key, trajectories);
        held.push(Held {
            templates: p.templates(&kind),
            kind,
            written,
        });
    }
    let mut state = State {
        held,
        born: birth(&from.atoms),
        atom_bits: 0.0,
        residual_bits: 0.0,
        moved: None,
        skipped: Vec::new(),
    };
    state.price(p);
    Ok(state)
}

/// The kind `atom` is written from, `None` for a shape no fit writes.
fn kind_of(atom: &Atom) -> Option<Kind> {
    let kind = match &atom.sound {
        Sound::Partial { hz, glide } => Kind::Partial {
            hz: hz.base,
            glide: Glide {
                rate: *glide,
                step: 0.0,
            },
        },
        Sound::Harmonic {
            hz, stretch, glide, ..
        } => Kind::Voice {
            f0: hz.base,
            stretch: *stretch,
            count: atom.harmonics(),
            glide: Glide {
                rate: *glide,
                step: 0.0,
            },
        },
        Sound::Noise { lo, hi } => Kind::Noise { band: (*lo, *hi) },
        Sound::Shaped { bands } => Kind::Shaped {
            bands: bands
                .iter()
                .map(|(lo, hi, g)| ((*lo, *hi), libm::pow(10.0, g / 10.0)))
                .collect(),
        },
    };
    let steady = atom.hz().is_none_or(|hz| hz.knots.is_empty());
    let positive = match &kind {
        Kind::Partial { hz, .. } => *hz > 0.0,
        Kind::Voice { f0, stretch, .. } => *f0 > 0.0 && *stretch >= 0.0,
        Kind::Noise { band } => 0.0 < band.0 && band.0 < band.1,
        Kind::Shaped { bands } => {
            !bands.is_empty() && bands.iter().all(|((lo, hi), _)| 0.0 < *lo && lo < hi)
        }
    };
    (steady && positive).then_some(kind)
}

fn refused(clip: &[f64]) -> Option<CliError> {
    match (clip.is_empty(), clip.iter().all(|s| s.is_finite())) {
        (true, _) => Some(CliError::usage(
            "solve.empty_clip",
            "the recording holds no sample, so there is nothing to fit",
        )),
        (_, false) => Some(CliError::usage(
            "solve.not_finite",
            "the recording holds a sample that is not a finite number",
        )),
        _ => None,
    }
}

/// One sound's fit of `samples` under `options`, its start and each pass heard by `hear` as
/// they end, `stop` asked before each pass and between its moves, as `MODEL` states.
pub async fn fit(
    samples: &[f64],
    rate: u32,
    spl_db: f64,
    options: &Options,
    hear: &mut dyn FnMut(&Line),
    stop: &dyn Abandon,
    clock: &dyn Fn() -> f64,
) -> Result<Fit, CliError> {
    if let Some(refused) = refused(samples) {
        return Err(refused);
    }
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
    if feasible(&grid).is_empty() {
        return Err(CliError::usage(
            "solve.rate_too_low",
            format!(
                "at {rate} Hz no analysis bin lies inside the feasible set from {} Hz (Cam {}), \
                 so the clip holds nothing a fit hears; a correctness limit, with no override",
                cam_hz(CAM_FIRST),
                CAM_FIRST
            ),
        ));
    }
    let p = Problem::of(grid, clip, heard_at, gain_db);
    let start_bits = State::empty(&p).dl();
    let (mut state, from_pass) = match &options.from {
        Some(from) => (resumed(&p, from, gain_db)?, from.passes),
        None => (State::empty(&p), 0),
    };
    let mut first = line(&p, &state, (from_pass, state.dl(), start_bits), None);
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
    let (mut passes, mut adds, mut adjusts) = (from_pass, 0, 0);
    let mut tweaks = [0; 4];
    let ended = loop {
        if stop.abandoned().await {
            break End::Stopped;
        }
        let began = clock();
        let Some(next) = pass(&p, &state, stop).await else {
            break End::Stopped;
        };
        let Some(moved) = &next.moved else {
            break End::Converged;
        };
        passes += 1;
        match moved.action {
            Action::Add => adds += 1,
            _ => adjusts += 1,
        }
        if let Some(tweak) = moved.tweak {
            tweaks[Tweak::ALL
                .iter()
                .position(|t| *t == tweak)
                .expect("a tweak")] += 1;
        }
        let removed = state
            .atoms()
            .into_iter()
            .find(|a| a.key == moved.key && moved.to_db.is_none());
        let (before, after) = (state.dl(), next.dl());
        let mut heard = line(&p, &next, (passes, before, start_bits), removed);
        heard.seconds = clock();
        heard.pass_secs = heard.seconds - began;
        hear(&heard);
        state = next;
        // Optimisers' `ftol` (scipy.optimize's L-BFGS-B).
        if before - after <= options.tolerance * before.abs().max(after.abs()).max(1.0) {
            break End::Tolerance;
        }
    };
    let model = full(&p, &state.model(&p));
    Ok(Fit {
        clip,
        atoms: state.atoms(),
        gain_db,
        spl_db: heard_at,
        atom_bits: state.atom_bits,
        residual_bits: state.residual_bits,
        from_pass,
        passes,
        adds,
        adjusts,
        tweaks,
        tolerance: options.tolerance,
        ended,
        clip_grid: p.grid.broad(),
        fit_grid: p.grid.with(model).broad(),
    })
}
