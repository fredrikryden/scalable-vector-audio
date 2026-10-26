// Concern: ADD: births off the residual's spectrogram (harmonics, partials, joined noise, onsets at strikes), solved together, then pruned | Non-concern: the schedule | IO: (clip, atoms, level) -> atoms

use std::collections::HashMap;
use std::f64::consts::{LN_2, TAU};
use sva_engine::hearing::{
    CAM_FIRST, CAM_LAST, HOP_SECS, WINDOW_SECS, cam, cam_hz, ear_gain_db, threshold_excitation_db,
};
use sva_engine::{Abandon, default_threads, each, fft, hits};

use super::atoms::{Atom, Clip, Feasible, Quantity, Sound, Trajectory, birth};
use super::bits::{
    self, Look, Precision, Priced, curvature_delta, frequency_delta, level_delta, precision,
};
use super::calibrate::rendered;
use super::fisher::Fit;
use super::lm::{Limits, Problem, descend};
use super::objective::{NoisePower, Objective, excited, framed, reach};
use super::pyramid::{Action, Run};

pub const MODEL: &str = "ADD from the residual's spectrogram at the level's rate: a periodic \
     Hann window as long as the ear's longest at s = 1 (64 ms), its DFT zero-padded to the next \
     power of two, a frame every 1 ms (the ear's hop at s = 1); the residual's power per bin \
     max(0, |X_clip|^2 - |X_model|^2), the model rendered by the engine from its own nodes, as \
     the phases it draws are not the clip's; a bin is audible where its sine's level, spl + \
     20 log10(2|X|/sum w) dB SPL, plus the ear's gain reaches the threshold excitation at its \
     frequency (hearing.rs), and that level is also the atom's floor; tracks first: each \
     residual bin over its floor that is a local maximum in frequency, furthest over first, \
     starts a track where no earlier one claimed it, its frequency and level by quadratic \
     interpolation of its dB and its neighbours' (QIFFT; Smith and Serra 1987, PARSHL, ICMC), \
     its power at most the residual's over the bins its main lobe covers, as a sine's peak \
     holds no more than its lobe, \
     tracked to each next and earlier frame through the largest unclaimed local maximum within \
     one bin while it stays audible, claiming each cell it passes with the bins a Hann main \
     lobe covers, 2 bins of the window's own length either side; none where moving the first \
     frame's time and the last's in by half the window, which reaches that far, but at the \
     clip's ends, leaves nothing (a run shorter than the window, which no sound makes, as every \
     frame whose window reaches it hears it); it starts at the latest strike the hits reading \
     places (superflux; Boeck and Widmer 2013, DAFx) within its first frame's window, else at \
     that moved time, and ends at the first later strike within its last frame's window, else \
     at that moved time; its base level the peak's and its one knot's deviation the level at \
     the last frame whose window lies whole before its end (the peak's where that is \
     earlier) less it; tracks group into harmonics by a harmonic sieve \
     (Duifhuis, Willems and Sluyter 1982, JASA 71(6) 1568-1580): each track's Hz over k = 1, \
     2, ... down to the feasible set's lowest is a trial fundamental, a track overlapping the \
     trial's in time its member at harmonic n = round(hz/f0) where it lies in the analysis \
     bin centred on n f0, the resolution the analysis placed it to, the nearest per n, the \
     trial scoring the numbers it saves, its members' as partials less its own as born, \
     where positive and its members outnumber the harmonics up to its highest it lacks (a \
     series heard more than not, so a subharmonic of a few tracks is not one), the \
     fundamental its strongest member's Hz over its harmonic number, the most exact the \
     analysis holds, the highest scoring, ties to \
     the higher fundamental, taken from the pool with its members, until none is found; a \
     harmonic spans its strongest member, each harmonic at its member's level or, lacking one, \
     its floor, its level their mean with the strongest member's fall, none whose node the \
     parser refuses (too many harmonics to nest), its members partials still; beside the harmonics, \
     the strongest track in each whole-Cam band from Cam 1.8 is a partial (the \
     critical band, a partial a band from another heard apart; Plomp 1964, JASA 36(9) \
     1628-1636), so pruning picks between a harmonic and its members; noise is proposed from the residual of the atoms the tonal births leave: \
     bands a Cam wide on whole Cam numbers, each band's power 2 sum |X_k|^2/(n sum w^2) over \
     its bins (Parseval), their gains the non-negative least squares fit of those powers, \
     which with disjoint bands is each band's power clipped at 0 (Lawson and Hanson 1974), \
     each unclaimed band's frame furthest over its floor joined by the adjacent bands audible \
     and unclaimed there and in no band proposed this round (a wide band, its floor at its \
     centre), tracked through the frames its joined power is audible as a track is, its edges \
     and levels its joined power's as a track's are; a solve moves the atoms not frozen \
     (lm.rs), one descent per connected component of atoms sharing a frame, its atoms by \
     their last frame, the rest held; each level first solves its atoms, \
     then goes in rounds, each the tonal births and then the noise births: a kind's births are \
     added at once, priced at their start (bits.rs: levels and frequencies by looks, the rest \
     by the Fisher diagonal), their frequencies and edges pinned where their Fisher precision \
     sqrt(12/A_ii) at the level is no finer than the analysis bin, rate/n, and every \
     frequency held at the analysis' value through the level it is born at, where its other \
     numbers start furthest off (a later level moves it), the births and every atom sharing a \
     frame with one revived and solved, then pruned (forward-backward selection; Zhang 2011, IEEE T-IT \
     57(7) 4689-4708): each sweep reads every atom's removal, tries them shortest first \
     against the state as it stands (a removal reads again only the frames the atom reaches, \
     the rest as the state read them, each noise's expected power read once per pruning), and \
     drops each whose removal lowers the residual bits \
     plus B(x), until a sweep drops none, each atom sharing a frame with a dropped one revived \
     and solved again; the state is kept where shorter than before the births; a \
     birth pruning drops is refused for the rest of the level: a partial's frequency (within \
     half the analysis bin) to partials, a band's Cam band to noise, a harmonic's frequencies \
     to the sieve and a joined band's Cam bands to joining, which opens their alternative, \
     narrow bands, proposed at once, before the next round; rounds go on until one keeps no \
     birth at the coarsest level, and one round at each finer (pyramid.rs)";

/// Partials and harmonics summed in samples, as the engine renders them.
pub const COHERENT: bool = true;

/// Power per bin `0..=n/2` of each frame every `hop` samples, centred on the frame's sample.
pub struct Spectrogram {
    rate: f64,
    n: usize,
    long: usize,
    hop: usize,
    sum_w: f64,
    sum_w2: f64,
    power: Vec<Vec<f64>>,
}

impl Spectrogram {
    pub fn of(samples: &[f64], rate: u32) -> Spectrogram {
        let rate_hz = f64::from(rate);
        let long = libm::round(WINDOW_SECS[0] * rate_hz) as usize;
        let window: Vec<f64> = (0..long)
            .map(|k| 0.5 - 0.5 * libm::cos(TAU * k as f64 / long as f64))
            .collect();
        let n = long.next_power_of_two();
        let hop = (libm::round(HOP_SECS * rate_hz) as usize).max(1);
        let twiddle = fft::twiddles(n);
        let frames: Vec<usize> = (0..=samples.len() / hop).collect();
        let power = each(default_threads(), frames, |m| {
            let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
            let from = (m * hop) as i64 - (long / 2) as i64;
            for (k, w) in window.iter().enumerate() {
                let at = from + k as i64;
                if at >= 0 && (at as usize) < samples.len() {
                    re[k] = w * samples[at as usize];
                }
            }
            fft::fft_with(&mut re, &mut im, &twiddle);
            (0..=n / 2)
                .map(|b| re[b] * re[b] + im[b] * im[b])
                .collect::<Vec<f64>>()
        });
        Spectrogram {
            rate: rate_hz,
            n,
            long,
            hop,
            sum_w: window.iter().sum(),
            sum_w2: window.iter().map(|w| w * w).sum(),
            power,
        }
    }

    fn hz(&self, bin: f64) -> f64 {
        bin * self.rate / self.n as f64
    }

    fn secs(&self, frame: usize) -> f64 {
        (frame * self.hop) as f64 / self.rate
    }

    /// The times a sound heard from frame `first` to `last` starts and ends, as `MODEL` states:
    /// `None` where moving each in by half the window leaves nothing; else the latest strike
    /// the first frame's window holds, or that time, and the first later strike the last
    /// frame's window holds, or that time.
    fn span(&self, (first, last): (usize, usize), strikes: &[f64]) -> Option<(f64, f64)> {
        let half = self.long as f64 / 2.0 / self.rate;
        let inward = |frame: usize, by: f64| match frame == 0 || frame + 1 == self.power.len() {
            true => self.secs(frame),
            false => self.secs(frame) + by,
        };
        let (on, end) = (inward(first, half), inward(last, -half));
        if on >= end {
            return None;
        }
        let window = |frame: usize, at: f64| (self.secs(frame) - at).abs() <= half;
        let on = strikes
            .iter()
            .rev()
            .find(|s| window(first, **s))
            .map_or(on, |s| *s);
        let end = strikes
            .iter()
            .find(|s| **s > on && window(last, **s))
            .map_or(end, |s| *s);
        (on < end).then_some((on, end))
    }

    /// The Hz between bins.
    pub fn bin_hz(&self) -> f64 {
        self.rate / self.n as f64
    }

    /// The padded bins a Hann main lobe covers either side of its peak.
    fn lobe_bins(&self) -> usize {
        (2 * self.n).div_ceil(self.long)
    }

    /// The last frame whose whole window lies before `end` seconds, `None` where none does.
    fn covered(&self, end: f64) -> Option<usize> {
        let reach = end * self.rate - (self.long / 2) as f64;
        (reach >= 0.0).then(|| libm::floor(reach / self.hop as f64) as usize)
    }

    /// A sine's level in dBFS whose peak bin holds `power`.
    fn sine_db(&self, power: f64) -> f64 {
        20.0 * libm::log10(2.0 * libm::sqrt(power) / self.sum_w)
    }

    /// A noise's level in dBFS whose bins hold `power` in all.
    fn noise_db(&self, power: f64) -> f64 {
        10.0 * libm::log10(2.0 * 2.0 * power / (self.n as f64 * self.sum_w2))
    }

    /// `max(0, self - model)` per bin.
    fn less(&self, model: &Spectrogram) -> Spectrogram {
        let power = self
            .power
            .iter()
            .zip(&model.power)
            .map(|(c, m)| c.iter().zip(m).map(|(c, m)| (c - m).max(0.0)).collect())
            .collect();
        Spectrogram { power, ..*self }
    }
}

/// What ADD reads off the clip once: its spectrogram, and where the hits reading places each
/// strike, in seconds.
pub struct Heard {
    spectrogram: Spectrogram,
    strikes: Vec<f64>,
}

impl Heard {
    pub fn of(samples: &[f64], rate: u32) -> Heard {
        let hits = hits::measure(&[samples], f64::from(rate), 0.0);
        Heard {
            spectrogram: Spectrogram::of(samples, rate),
            strikes: hits.hits.iter().map(|h| h.at_secs).collect(),
        }
    }

    /// The Hz between the spectrogram's bins.
    pub fn bin_hz(&self) -> f64 {
        self.spectrogram.bin_hz()
    }
}

/// The lowest level in dBFS at which a sine at `hz` reaches threshold, as MODEL states.
pub fn floor_db(hz: f64, spl_db: f64) -> f64 {
    threshold_excitation_db(hz) - ear_gain_db(hz) - spl_db
}

fn audible(hz: f64, db: f64, spl_db: f64) -> bool {
    db > floor_db(hz, spl_db)
}

/// The whole-Cam band holding `hz`.
fn critical_band(hz: f64) -> usize {
    libm::floor(cam(hz) - CAM_FIRST).max(0.0) as usize
}

/// The Hz the feasible set spans at `rate`.
fn span_hz(rate: f64) -> (f64, f64) {
    (cam_hz(CAM_FIRST), cam_hz(CAM_LAST).min(rate / 2.0))
}

/// Each number's precision at `atom` alone at this level, `None` where one is not finite (the
/// atom is heard nowhere).
pub fn priced(objective: &Objective, atom: &Atom, spl_db: f64) -> Option<Priced> {
    priced_and_pinned(objective, atom, spl_db, f64::INFINITY).map(|(p, _)| p)
}

/// `priced`, beside which numbers this level leaves where they are: a frequency or an edge
/// whose Fisher precision `sqrt(12/A_ii)` here is no finer than `analysis_hz`, the bin ADD's
/// analysis placed it to.
pub fn priced_and_pinned(
    objective: &Objective,
    atom: &Atom,
    spl_db: f64,
    analysis_hz: f64,
) -> Option<(Priced, Vec<bool>)> {
    let clip = objective.clip();
    let hz = match atom.sound {
        Sound::Noise { lo, hi } => cam_hz((cam(lo) + cam(hi)) / 2.0),
        _ => atom.hz().map_or(0.0, |t| t.base),
    };
    let feasible = Feasible::of(&clip, floor_db(hz, spl_db).min(atom.level.base));
    let ones = Priced {
        feasible,
        deltas: vec![1.0; atom.values().len()],
    };
    let probe = Fit::new(objective, vec![atom.clone()], vec![ones], COHERENT);
    let a = probe.curvatures(&probe.numbers());
    let ear = objective.ear();
    let thresholds = ear.thresholds();
    let looks = framed(
        std::slice::from_ref(atom),
        clip,
        COHERENT,
        ear.frames(clip.len),
        &|model, m| {
            let e = excited(ear, m, &ear.compose(m, &mut |f| model.bins(f).power));
            let peak = e
                .iter()
                .zip(&thresholds)
                .fold(0.0f64, |p, (e, t)| p.max(e / t));
            Look {
                sl_db: 10.0 * libm::log10(peak),
                hz,
            }
        },
    );
    let hop = objective.heard().hop_secs;
    let numbers = atom.numbers();
    let deltas: Vec<f64> = numbers
        .iter()
        .zip(&a)
        .map(|((q, _), a)| match precision(atom, *q) {
            Precision::Level => level_delta(hop, &looks),
            Precision::Frequency => frequency_delta(hop, &looks),
            Precision::Curvature => curvature_delta(*a),
        })
        .collect();
    let pinned = numbers
        .iter()
        .zip(&a)
        .map(|((q, _), a)| {
            matches!(q, Quantity::Hz | Quantity::HzDeviation | Quantity::Edge)
                && curvature_delta(*a) >= analysis_hz
        })
        .collect();
    deltas
        .iter()
        .all(|d| d.is_finite() && *d > 0.0)
        .then_some((Priced { feasible, deltas }, pinned))
}

/// The residual bits plus B(x).
pub fn length(objective: &Objective, atoms: &[Atom], priced: &[Priced]) -> f64 {
    objective.bits(atoms, COHERENT) + bits::bits(atoms, priced)
}

/// The state ADD grows: atoms beside their prices, the numbers this level pins and whether each
/// is frozen (lm.rs), and its length in bits.
pub struct State {
    pub atoms: Vec<Atom>,
    pub priced: Vec<Priced>,
    pub pinned: Vec<Vec<bool>>,
    pub frozen: Vec<bool>,
    pub length: f64,
}

impl State {
    /// Each atom's reach in frames.
    fn reach(&self, objective: &Objective) -> Vec<(i64, i64)> {
        let (ear, clip) = (objective.ear(), objective.clip());
        self.atoms.iter().map(|a| reach(ear, &clip, a)).collect()
    }

    /// Revives every atom sharing a frame with `at`, a reach.
    fn revive(&mut self, objective: &Objective, at: (i64, i64)) {
        let reach = self.reach(objective);
        for (frozen, (f, t)) in self.frozen.iter_mut().zip(reach) {
            if f <= at.1 && at.0 <= t {
                *frozen = false;
            }
        }
    }
}

/// The unfrozen atoms of `reach` in connected components of sharing a frame, each in time
/// order of its first frame, its atoms by their last frame (lm.rs).
fn components(reach: &[(i64, i64)], frozen: &[bool]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..reach.len()).filter(|a| !frozen[*a]).collect();
    order.sort_by_key(|a| (reach[*a].0, *a));
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut end = i64::MIN;
    for a in order {
        match out.last_mut() {
            Some(last) if reach[a].0 <= end => {
                last.push(a);
                end = end.max(reach[a].1);
            }
            _ => {
                out.push(vec![a]);
                end = reach[a].1;
            }
        }
    }
    for component in &mut out {
        component.sort_by_key(|a| (reach[*a].1, *a));
    }
    out
}

/// A partial the residual proposes: its Hz, start and end, and its level at the peak and its
/// end's deviation from it.
struct Track {
    hz: f64,
    span: (f64, f64),
    levels: (f64, f64),
}

impl Track {
    fn meets(&self, other: &Track) -> bool {
        self.span.0 < other.span.1 && other.span.0 < self.span.1
    }
}

/// Every partial the residual proposes, furthest over its floor first, as `MODEL` states.
fn tracks(residual: &Spectrogram, spl: f64, strikes: &[f64]) -> Vec<Track> {
    let (lo, hi) = span_hz(residual.rate);
    let peak_at = |frame: &[f64], b: usize| {
        b > 0 && b + 1 < frame.len() && frame[b] > frame[b - 1] && frame[b] >= frame[b + 1]
    };
    let mut peaks: Vec<(usize, usize, f64)> = Vec::new();
    for (m, frame) in residual.power.iter().enumerate() {
        for b in 1..frame.len() - 1 {
            let hz = residual.hz(b as f64);
            if hz < lo || hz > hi || !peak_at(frame, b) {
                continue;
            }
            let heard = residual.sine_db(frame[b]) - floor_db(hz, spl);
            if heard > 0.0 {
                peaks.push((m, b, heard));
            }
        }
    }
    peaks.sort_by(|x, y| y.2.total_cmp(&x.2).then((x.0, x.1).cmp(&(y.0, y.1))));
    let bins = residual.n / 2 + 1;
    let lobe = residual.lobe_bins();
    let around = |b: usize| b.saturating_sub(lobe)..(b + lobe + 1).min(bins);
    let mut claimed = vec![vec![false; bins]; residual.power.len()];
    let mut out = Vec::new();
    for (m, b, _) in peaks {
        if claimed[m][b] {
            continue;
        }
        let frame = &residual.power[m];
        let db = |p: f64| 10.0 * libm::log10(p.max(f64::MIN_POSITIVE));
        let (a, c, e) = (db(frame[b - 1]), db(frame[b]), db(frame[b + 1]));
        let curve = a - 2.0 * c + e;
        let p = match curve < 0.0 {
            true => 0.5 * (a - e) / curve,
            false => 0.0,
        };
        let hz = residual.hz(b as f64 + p).clamp(lo, hi);
        let lobe_power: f64 = frame[around(b)].iter().sum();
        let peak = libm::pow(10.0, (c - 0.25 * (a - e) * p) / 10.0).min(lobe_power);
        let track = |step: i64| {
            let (mut at, mut bin) = (m as i64, b);
            let mut cells = vec![(m, b)];
            let mut levels = vec![residual.sine_db(frame[b])];
            loop {
                let next = at + step;
                let Some((n, f)) = usize::try_from(next)
                    .ok()
                    .and_then(|n| residual.power.get(n).map(|f| (n, f)))
                else {
                    break;
                };
                let found = (bin.saturating_sub(1)..=bin + 1)
                    .filter(|j| peak_at(f, *j) && !claimed[n][*j])
                    .max_by(|x, y| f[*x].total_cmp(&f[*y]));
                let Some(j) = found else { break };
                let level = residual.sine_db(f[j]);
                if !audible(residual.hz(j as f64), level, spl) {
                    break;
                }
                (at, bin) = (next, j);
                cells.push((n, j));
                levels.push(level);
            }
            (at as usize, levels, cells)
        };
        let (first, _, before) = track(-1);
        let (end, levels, after) = track(1);
        for (n, j) in before.into_iter().chain(after) {
            for cell in &mut claimed[n][around(j)] {
                *cell = true;
            }
        }
        let Some(span) = residual.span((first, end), strikes) else {
            continue;
        };
        let inside = residual
            .covered(span.1)
            .map_or(0, |f| f.saturating_sub(m).min(levels.len() - 1));
        out.push(Track {
            hz,
            span,
            levels: (residual.sine_db(peak), levels[inside]),
        });
    }
    out
}

/// The numbers a steady atom of `harmonics` harmonics with one level knot writes, as born; a
/// partial's at 1.
fn written(harmonics: usize) -> usize {
    let level = Trajectory {
        base: 0.0,
        knots: vec![(0.0, 0.0)],
    };
    match harmonics {
        1 => Atom::partial(0, 0.0, Trajectory::steady(0.0), level),
        _ => Atom::harmonic(
            0,
            0.0,
            Trajectory::steady(0.0),
            vec![0.0; harmonics - 1],
            level,
        ),
    }
    .values()
    .len()
}

/// A harmonic's members as (track, harmonic number).
type Members = Vec<(usize, usize)>;

/// The fundamental whose sieve over `pool` scores most, as `MODEL` states, beside its members
/// as (track, harmonic number); `None` where no fundamental saves numbers.
fn sieved(
    tracks: &[Track],
    pool: &[usize],
    analysis_hz: f64,
    (lo, hi): (f64, f64),
) -> Option<(f64, Members)> {
    let better = |best: &Option<(usize, f64, Members)>, score: usize, f0: f64| {
        best.as_ref()
            .is_none_or(|(s, f, _)| score > *s || (score == *s && f0 > *f))
    };
    let each_anchor = each(default_threads(), pool.to_vec(), |p| {
        let mut best: Option<(usize, f64, Members)> = None;
        let anchor = &tracks[p];
        for k in 1.. {
            let f0 = anchor.hz / f64::from(k);
            if f0 < lo {
                break;
            }
            let mut by_number: Vec<Option<(usize, f64)>> = Vec::new();
            for &q in pool {
                let t = &tracks[q];
                let n = libm::round(t.hz / f0);
                let miss = (t.hz - n * f0).abs();
                if n < 1.0 || 2.0 * miss > analysis_hz || !t.meets(anchor) {
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

/// Every tonal atom the residual proposes, as `MODEL` states: harmonics the sieve groups, and
/// beside them partials, none from a track within half the analysis bin of a frequency
/// `refused` holds, as a harmonic's to the sieve, as a partial's to a partial.
fn tonal(
    residual: &Spectrogram,
    clip: &Clip,
    spl: f64,
    strikes: &[f64],
    refused: &Refused,
) -> Vec<Atom> {
    let tracks = tracks(residual, spl, strikes);
    let analysis_hz = residual.bin_hz();
    let feasible = span_hz(residual.rate);
    let near = |hz: f64, held: &[f64]| held.iter().any(|r| 2.0 * (hz - r).abs() <= analysis_hz);
    let mut pool: Vec<usize> = (0..tracks.len())
        .filter(|t| !near(tracks[*t].hz, &refused.hz))
        .collect();
    let mut out = Vec::new();
    while let Some((f0, members)) = sieved(&tracks, &pool, analysis_hz, feasible) {
        let count = members.iter().map(|(_, n)| *n).max().expect("a member");
        let strongest = members.iter().map(|(q, _)| *q).min().expect("a member");
        let lead = &tracks[strongest];
        let levels: Vec<f64> = (1..=count)
            .map(|n| {
                members
                    .iter()
                    .find(|(_, m)| *m == n)
                    .map_or(floor_db(f0 * n as f64, spl), |(q, _)| tracks[*q].levels.0)
            })
            .collect();
        let mean = levels.iter().sum::<f64>() / count as f64;
        let offsets: Vec<f64> = levels[..count - 1].iter().map(|l| l - mean).collect();
        let fall = lead.levels.1 - lead.levels.0;
        let harmonic = born(clip, lead.span, (mean, mean + fall), |on, level| {
            Atom::harmonic(0, on, Trajectory::steady(f0), offsets, level)
        });
        if harmonic.writes(clip) {
            out.push(harmonic);
        }
        // Every track is proposed as a partial below, a refused group's members too.
        pool.retain(|t| members.iter().all(|(q, _)| q != t));
    }
    let mut proposed: Vec<usize> = Vec::new();
    for track in &tracks {
        let band = critical_band(track.hz);
        if proposed.contains(&band) || near(track.hz, &refused.partials) {
            continue;
        }
        proposed.push(band);
        out.push(born(clip, track.span, track.levels, |on, level| {
            Atom::partial(0, on, Trajectory::steady(track.hz), level)
        }));
    }
    out
}

/// A sound from `on` to `end` in the clip, at least a sample long, its level from `base` to
/// `last` dB, made by `make` from its onset and level.
fn born(
    clip: &Clip,
    (on, end): (f64, f64),
    (base, last): (f64, f64),
    make: impl FnOnce(f64, Trajectory) -> Atom,
) -> Atom {
    let span = 1.0 / f64::from(clip.rate);
    let on = on.clamp(0.0, clip.secs() - span);
    let end = end.clamp(on + span, clip.secs());
    make(
        on,
        Trajectory {
            base,
            knots: vec![(end, last - base)],
        },
    )
}

/// Every noise band the residual proposes, as `MODEL` states, none joining a Cam band
/// `refused` holds as a joined band's, none holding one it holds as a band's.
fn noises(
    residual: &Spectrogram,
    clip: &Clip,
    spl: f64,
    strikes: &[f64],
    refused: &Refused,
) -> Vec<(Atom, Vec<usize>)> {
    let (lo, hi) = span_hz(residual.rate);
    let edges: Vec<f64> = (0..)
        .map(|z| cam_hz(CAM_FIRST + f64::from(z)))
        .take_while(|hz| *hz <= hi)
        .collect();
    let bands: Vec<(f64, f64, usize, usize, usize)> = edges
        .windows(2)
        .enumerate()
        .map(|(z, e)| {
            let bin = |hz: f64| libm::ceil(hz * residual.n as f64 / residual.rate) as usize;
            (e[0].max(lo), e[1], bin(e[0]), bin(e[1]), z)
        })
        .filter(|(_, _, a, b, _)| b > a)
        .collect();
    let powers: Vec<Vec<f64>> = residual
        .power
        .iter()
        .map(|f| {
            bands
                .iter()
                .map(|(_, _, a, b, _)| f[*a..*b].iter().sum())
                .collect()
        })
        .collect();
    let centre = |(a, b): (usize, usize)| cam_hz((cam(bands[a].0) + cam(bands[b].1)) / 2.0);
    let power = |m: usize, (a, b): (usize, usize)| powers[m][a..=b].iter().sum::<f64>();
    let over =
        |m: usize, ab: (usize, usize)| residual.noise_db(power(m, ab)) - floor_db(centre(ab), spl);
    let mut heard: Vec<(usize, usize, f64)> = Vec::new();
    for m in 0..powers.len() {
        for k in 0..bands.len() {
            let h = over(m, (k, k));
            if h > 0.0 {
                heard.push((m, k, h));
            }
        }
    }
    heard.sort_by(|x, y| y.2.total_cmp(&x.2).then((x.0, x.1).cmp(&(y.0, y.1))));
    let mut claimed = vec![vec![false; bands.len()]; powers.len()];
    let mut proposed = vec![false; bands.len()];
    let mut out = Vec::new();
    for (m, k, _) in heard {
        if claimed[m][k] || proposed[k] || refused.bands.contains(&bands[k].4) {
            continue;
        }
        let joins = |j: usize| {
            over(m, (j, j)) > 0.0
                && !claimed[m][j]
                && !proposed[j]
                && !refused.joined.contains(&bands[j].4)
                && !refused.bands.contains(&bands[j].4)
        };
        let (mut a, mut b) = (k, k);
        if joins(k) {
            while a > 0 && joins(a - 1) {
                a -= 1;
            }
            while b + 1 < bands.len() && joins(b + 1) {
                b += 1;
            }
        }
        let open = |f: &usize| over(*f, (a, b)) > 0.0 && (a..=b).all(|j| !claimed[*f][j]);
        let first = (0..=m).rev().take_while(open).last().unwrap_or(m);
        let end = (m..powers.len()).take_while(open).last().unwrap_or(m);
        for row in &mut claimed[first..=end] {
            for cell in &mut row[a..=b] {
                *cell = true;
            }
        }
        let Some(span) = residual.span((first, end), strikes) else {
            continue;
        };
        for p in &mut proposed[a..=b] {
            *p = true;
        }
        let inside = residual.covered(span.1).map_or(m, |f| f.clamp(m, end));
        let atom = born(
            clip,
            span,
            (
                residual.noise_db(power(m, (a, b))),
                residual.noise_db(power(inside, (a, b))),
            ),
            |on, level| Atom::noise(0, on, (bands[a].0, bands[b].1), level),
        );
        out.push((atom, (a..=b).map(|j| bands[j].4).collect()));
    }
    out
}

/// A birth beside what refuses it where pruning drops it: a partial's Hz, a harmonic's
/// harmonics' Hz, a band's Cam band, a joined band's Cam bands.
enum Joined {
    Partial(f64),
    Harmonic(Vec<f64>),
    Band(usize),
    Bands(Vec<usize>),
}

/// The two kinds a round proposes, tonal first.
#[derive(Clone, Copy)]
enum Kind {
    Tonal,
    Noise,
}

/// What pruning refuses for the rest of a level: a harmonic's Hz to the sieve, a partial's to
/// partials, a joined band's Cam bands to joining, a band's to every band.
#[derive(Default)]
struct Refused {
    hz: Vec<f64>,
    partials: Vec<f64>,
    joined: Vec<usize>,
    bands: Vec<usize>,
}

/// What `kind` proposes from the residual of `atoms`, as `MODEL` states.
fn proposed(
    heard: &Heard,
    atoms: &[Atom],
    clip: &Clip,
    spl: f64,
    refused: &Refused,
    kind: Kind,
) -> Vec<(Atom, Joined)> {
    let model = Spectrogram::of(&rendered(atoms, clip), clip.rate);
    let residual = heard.spectrogram.less(&model);
    match kind {
        Kind::Tonal => tonal(&residual, clip, spl, &heard.strikes, refused)
            .into_iter()
            .map(|a| {
                let joined = match &a.sound {
                    Sound::Harmonic { hz, offsets } => Joined::Harmonic(
                        (1..=offsets.len() + 1)
                            .map(|n| hz.base * n as f64)
                            .collect(),
                    ),
                    _ => Joined::Partial(a.hz().expect("a partial's frequency").base),
                };
                (a, joined)
            })
            .collect(),
        Kind::Noise => noises(&residual, clip, spl, &heard.strikes, refused)
            .into_iter()
            .map(|(a, bands)| {
                let joined = match bands.as_slice() {
                    [band] => Joined::Band(*band),
                    _ => Joined::Bands(bands),
                };
                (a, joined)
            })
            .collect(),
    }
}

/// The unfrozen atoms of `state` moved by one descent per component, the rest held, each
/// pinned number held, each atom frozen as its descent left it.
async fn solved(objective: &Objective, mut state: State, run: &mut Run<'_>) -> State {
    let reach = state.reach(objective);
    for component in components(&reach, &state.frozen) {
        if run.stopped.abandoned().await {
            return state;
        }
        run.activity(Action::Descending, None, Some((component.len(), "atom")));
        let pins: Vec<bool> = component
            .iter()
            .flat_map(|a| state.pinned[*a].iter().copied())
            .collect();
        let others: Vec<Atom> = (0..state.atoms.len())
            .filter(|a| !component.contains(a))
            .map(|a| state.atoms[a].clone())
            .collect();
        let fit = Fit::new(
            objective,
            component.iter().map(|a| state.atoms[*a].clone()).collect(),
            component.iter().map(|a| state.priced[*a].clone()).collect(),
            COHERENT,
        )
        .pinning(&pins)
        .holding(others);
        let start = fit.numbers();
        let limits = Limits {
            steps: None,
            stopped: &run.stopped,
        };
        let Some(descent) = descend(&fit, &start, &fit.deltas(), &limits).await else {
            continue;
        };
        for ((a, atom), frozen) in component
            .iter()
            .zip(fit.atoms_at(&descent.x))
            .zip(&descent.frozen)
        {
            state.atoms[*a] = atom;
            state.frozen[*a] = *frozen;
        }
        let bits = bits::bits(&state.atoms, &state.priced);
        run.descended(objective, &descent, bits);
        state.length = descent.cost / LN_2 + bits;
    }
    state
}

/// `v` without its `i`-th.
fn but<T: Clone>(v: &[T], i: usize) -> Vec<T> {
    let mut out = v.to_vec();
    out.remove(i);
    out
}

/// `state` without atom `i`, `length` long, every atom sharing a frame with it revived.
fn without(objective: &Objective, state: &State, i: usize, length: f64) -> State {
    let at = state.reach(objective)[i];
    let mut out = State {
        atoms: but(&state.atoms, i),
        priced: but(&state.priced, i),
        pinned: but(&state.pinned, i),
        frozen: but(&state.frozen, i),
        length,
    };
    out.revive(objective, at);
    out
}

/// Each noise's composed power per frame, by key, read once while pruning.
type Noises = HashMap<u64, NoisePower>;

/// `atoms`' excitation per frame over `frames`, their partials and harmonics summed in samples
/// beside each noise's power as `noises` holds it.
fn excitations(
    objective: &Objective,
    atoms: &[Atom],
    noises: &Noises,
    frames: (i64, i64),
) -> Vec<Vec<f64>> {
    let tonal: Vec<Atom> = atoms
        .iter()
        .filter(|a| !matches!(a.sound, Sound::Noise { .. }))
        .cloned()
        .collect();
    let noise: Vec<&NoisePower> = atoms.iter().filter_map(|a| noises.get(&a.key)).collect();
    objective.excitations_beside(&tonal, &noise, frames)
}

/// `state`'s length without atom `i`, `full` its excitation per frame: only the frames atom
/// `i` reaches are read again, its partials and harmonics summed in samples beside each
/// noise's power as `noises` holds it.
fn removed(
    objective: &Objective,
    state: &State,
    (full, noises): (&[Vec<f64>], &Noises),
    i: usize,
) -> f64 {
    let (ear, clip) = (objective.ear(), objective.clip());
    let (first, _) = ear.frames(clip.len);
    let (from, to) = reach(ear, &clip, &state.atoms[i]);
    let others = but(&state.atoms, i);
    let local = match from <= to {
        true => excitations(objective, &others, noises, (from, to)),
        false => Vec::new(),
    };
    let residual = objective.bits_of(&mut |m| match from <= m && m <= to {
        true => local[(m - from) as usize].clone(),
        false => full[(m - first) as usize].clone(),
    });
    residual + bits::bits(&others, &but(&state.priced, i))
}

/// Backward steps as `MODEL` states: each sweep reads every atom's removal, then tries them
/// in that order against the state as it stands, dropping each that shortens it, until a sweep
/// drops none.
async fn pruned(objective: &Objective, mut state: State, run: &mut Run<'_>) -> (State, bool) {
    let frames = objective.ear().frames(objective.clip().len);
    let mut dropped = false;
    let noises: Noises = state
        .atoms
        .iter()
        .filter(|a| matches!(a.sound, Sound::Noise { .. }))
        .map(|a| (a.key, objective.noise_power(a)))
        .collect();
    loop {
        let mut full = excitations(objective, &state.atoms, &noises, frames);
        run.evaluated(objective, 1);
        let mut read: Vec<(u64, f64)> = Vec::new();
        for i in 0..state.atoms.len() {
            if run.stopped.abandoned().await {
                return (state, dropped);
            }
            run.activity(
                Action::Measuring,
                Some(&edit(&state.atoms[i])),
                Some((i + 1, "removal")),
            );
            run.evaluated(objective, 1);
            read.push((
                state.atoms[i].key,
                removed(objective, &state, (&full, &noises), i),
            ));
        }
        read.sort_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(&y.0)));
        let mut swept = false;
        for (key, stale) in read {
            let Some(i) = state.atoms.iter().position(|a| a.key == key) else {
                continue;
            };
            let less = match swept {
                false => stale,
                true => {
                    run.evaluated(objective, 1);
                    removed(objective, &state, (&full, &noises), i)
                }
            };
            if less < state.length {
                state = without(objective, &state, i, less);
                full = excitations(objective, &state.atoms, &noises, frames);
                swept = true;
            }
        }
        if !swept {
            return (state, dropped);
        }
        dropped = true;
    }
}

/// One kind's births beside `state`, as `MODEL` states: the state they leave where shorter,
/// whether one survived there, and whether pruning refused anything.
async fn grown(
    objective: &Objective,
    heard: &Heard,
    state: State,
    (kind, refused): (Kind, &mut Refused),
    run: &mut Run<'_>,
) -> (State, bool, bool) {
    let at = objective.clip();
    let mut grown = State {
        atoms: state.atoms.clone(),
        priced: state.priced.clone(),
        pinned: state.pinned.clone(),
        frozen: state.frozen.clone(),
        length: state.length,
    };
    let mut births: Vec<(u64, Joined)> = Vec::new();
    let first = birth(&state.atoms);
    for (mut atom, joined) in proposed(heard, &state.atoms, &at, run.spl_db, refused, kind) {
        if run.stopped.abandoned().await {
            return (state, false, false);
        }
        atom.key = first + births.len() as u64;
        run.proposed += 1;
        let count = run.proposed;
        run.activity(
            Action::Ranking,
            Some(&edit(&atom)),
            Some((count, "proposal")),
        );
        let Some((price, pins)) = priced_and_pinned(objective, &atom, run.spl_db, heard.bin_hz())
        else {
            continue;
        };
        let held: Vec<bool> = atom
            .numbers()
            .iter()
            .zip(pins)
            .map(|((q, _), pin)| pin || matches!(q, Quantity::Hz | Quantity::HzDeviation))
            .collect();
        births.push((atom.key, joined));
        grown.revive(objective, reach(objective.ear(), &at, &atom));
        grown.atoms.push(atom);
        grown.priced.push(price);
        grown.pinned.push(held);
        grown.frozen.push(false);
    }
    if births.is_empty() {
        return (state, false, false);
    }
    let grown = solved(objective, grown, run).await;
    let (mut kept, dropped) = pruned(objective, grown, run).await;
    if dropped {
        kept = solved(objective, kept, run).await;
    }
    let (mut survived, mut opened) = (false, false);
    for (key, joined) in &births {
        match kept.atoms.iter().find(|a| a.key == *key) {
            Some(atom) => {
                survived = true;
                run.edits.push(edit(atom));
            }
            None => match joined {
                Joined::Partial(hz) => refused.partials.push(*hz),
                Joined::Band(band) => refused.bands.push(*band),
                Joined::Harmonic(hz) => refused.hz.extend(hz),
                Joined::Bands(bands) => {
                    refused.joined.extend(bands);
                    opened = true;
                }
            },
        }
    }
    match kept.length < state.length {
        true => (kept, survived, opened),
        false => (state, false, opened),
    }
}

/// From `state`, its unfrozen atoms solved, then rounds of tonal births and then noise
/// births, as `MODEL` states: until one keeps no birth where `whole`, else one.
pub async fn add(
    objective: &Objective,
    heard: &Heard,
    state: State,
    whole: bool,
    run: &mut Run<'_>,
) -> State {
    let mut refused = Refused::default();
    let mut state = solved(objective, state, run).await;
    loop {
        let mut going = false;
        for kind in [Kind::Tonal, Kind::Noise] {
            loop {
                if run.stopped.abandoned().await {
                    return state;
                }
                let (next, survived, opened) =
                    grown(objective, heard, state, (kind, &mut refused), run).await;
                state = next;
                going |= survived;
                if !opened {
                    break;
                }
            }
        }
        if !going || !whole {
            return state;
        }
    }
}

/// An edit as the page names it: the kind, where and at what Hz.
pub fn edit(atom: &Atom) -> String {
    match atom.sound {
        Sound::Noise { lo, hi } => format!(
            "noise {}-{} Hz at {} s",
            libm::round(lo),
            libm::round(hi),
            atom.on
        ),
        Sound::Harmonic { .. } => format!(
            "harmonic {} Hz x{} at {} s",
            libm::round(atom.hz().map_or(0.0, |t| t.base)),
            atom.harmonics(),
            atom.on
        ),
        Sound::Partial { .. } => format!(
            "partial {} Hz at {} s",
            libm::round(atom.hz().map_or(0.0, |t| t.base)),
            atom.on
        ),
    }
}
