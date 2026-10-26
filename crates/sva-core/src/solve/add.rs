// Concern: ADD: births off the residual's spectrogram, partials at QIFFT peaks then noise bands, kept where the length falls | Non-concern: the schedule, the search | IO: (clip, atoms, level) -> atoms

use std::f64::consts::{LN_2, TAU};

use sva_engine::hearing::{
    CAM_FIRST, CAM_LAST, HOP_SECS, WINDOW_SECS, cam, cam_hz, ear_gain_db, threshold_excitation_db,
};
use sva_engine::{default_threads, each, fft};

use super::atoms::{Atom, Clip, Feasible, Quantity, Sound, Trajectory, birth};
use super::bits::{
    self, Look, Precision, Priced, curvature_delta, frequency_delta, level_delta, precision,
};
use super::calibrate::rendered;
use super::fisher::Fit;
use super::lm::{Limits, curvatures, descend};
use super::objective::{Objective, excited, framed};
use super::pyramid::{Action, Run};

pub const MODEL: &str = "ADD from the residual's spectrogram at full resolution: a periodic \
     Hann window as long as the ear's longest at s = 1 (64 ms), its DFT zero-padded to the next \
     power of two, a frame every 1 ms (the ear's hop at s = 1); the residual's power per bin \
     max(0, |X_clip|^2 - |X_model|^2), the model rendered by the engine from its own nodes, as \
     the phases it draws are not the clip's; a bin is audible where its sine's level, spl + \
     20 log10(2|X|/sum w) dB SPL, plus the ear's gain reaches the threshold excitation at its \
     frequency (hearing.rs), and that level is also the atom's floor; partials first: the \
     residual bin furthest over its floor that is a local maximum in frequency over every frame, its \
     frequency and level by quadratic interpolation of its dB and its neighbours' (QIFFT; \
     Smith and Serra 1987, PARSHL, ICMC), tracked to each next and earlier frame through the \
     largest local maximum within one bin while it stays audible, a partial from the first \
     frame's time to the last's, each moved in by half the window, which reaches that far, but at the clip's ends, its base level the peak's and its one knot's deviation the \
     level half a window inside its last frame less it; then noise bands a Cam wide on whole Cam numbers, each band's \
     power 2 sum |X_k|^2/(n sum w^2) over its bins of the residual (Parseval), their gains the \
     non-negative least squares fit of those powers, which with disjoint bands is each band's \
     power clipped at 0 (Lawson and Hanson 1974), the band and frame furthest over the floor at the band's centre tracked \
     through its audible run as a partial is; a candidate's numbers are solved by lm.rs with \
     every other atom held, at precisions priced at its start (bits.rs: levels and \
     frequencies by looks, the rest by the Fisher diagonal), its frequencies and edges pinned where their Fisher precision sqrt(12/A_ii) at the level is no \
     finer than the analysis bin, rate/n, and it is kept iff the residual bits plus B(x) fall; the first that does \
     not ends that kind (matching pursuit's stop: the best atom does not pay)";

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

    /// The times a sound heard from frame `first` to `last`, loudest at `peak`, starts and
    /// ends: each frame's window reaches half its length either side, so the run is that much
    /// shorter at each end, and at least the peak's own frame.
    fn span(&self, first: usize, peak: usize, last: usize) -> (f64, f64) {
        let half = self.long as f64 / 2.0 / self.rate;
        let inward = |frame: usize, by: f64| match frame == 0 || frame + 1 == self.power.len() {
            true => self.secs(frame),
            false => self.secs(frame) + by,
        };
        let (on, end) = (inward(first, half), inward(last, -half));
        match on < end {
            true => (on, end),
            false => (
                self.secs(peak),
                self.secs(peak) + self.hop as f64 / self.rate,
            ),
        }
    }

    /// The Hz between bins.
    pub fn bin_hz(&self) -> f64 {
        self.rate / self.n as f64
    }

    /// The frames half a window spans.
    fn half_frames(&self) -> usize {
        self.long / 2 / self.hop
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

/// The lowest level in dBFS at which a sine at `hz` reaches threshold, as MODEL states.
pub fn floor_db(hz: f64, spl_db: f64) -> f64 {
    threshold_excitation_db(hz) - ear_gain_db(hz) - spl_db
}

fn audible(hz: f64, db: f64, spl_db: f64) -> bool {
    db > floor_db(hz, spl_db)
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
    let a = curvatures(&probe, &probe.numbers());
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

/// The state ADD grows: atoms beside their prices and the numbers this level pins, and its
/// length in bits.
pub struct State {
    pub atoms: Vec<Atom>,
    pub priced: Vec<Priced>,
    pub pinned: Vec<Vec<bool>>,
    pub length: f64,
}

/// `atom` solved with `state` held; the state with it where that shortens it.
fn tried(
    objective: &Objective,
    state: &State,
    atom: Atom,
    analysis_hz: f64,
    run: &mut Run,
) -> Option<State> {
    let (price, pins) = priced_and_pinned(objective, &atom, run.spl_db, analysis_hz)?;
    let fit = Fit::new(objective, vec![atom], vec![price.clone()], COHERENT)
        .holding(state.atoms.clone())
        .pinning(&pins);
    let start = fit.numbers();
    let stopped = run.stopped;
    let descent = descend(
        &fit,
        &start,
        &fit.deltas(),
        &Limits {
            steps: None,
            stopped,
        },
    )?;
    let born = fit.atoms_at(&descent.x).remove(0);
    let mut atoms = state.atoms.clone();
    atoms.push(born);
    let mut priced = state.priced.clone();
    priced.push(price);
    let mut pinned = state.pinned.clone();
    pinned.push(pins);
    let bits = bits::bits(&atoms, &priced);
    run.descended(objective, &descent, bits);
    let length = descent.cost / LN_2 + bits;
    (length < state.length).then_some(State {
        atoms,
        priced,
        pinned,
        length,
    })
}

/// The partial the residual's peak furthest over its floor starts.
fn partial(residual: &Spectrogram, clip: &Clip, key: u64, spl: f64) -> Option<Atom> {
    let (lo, hi) = span_hz(residual.rate);
    let peak_at = |frame: &[f64], b: usize| {
        b > 0 && b + 1 < frame.len() && frame[b] > frame[b - 1] && frame[b] >= frame[b + 1]
    };
    let mut best: Option<(usize, usize, f64)> = None;
    for (m, frame) in residual.power.iter().enumerate() {
        for b in 1..frame.len() - 1 {
            let hz = residual.hz(b as f64);
            if hz < lo || hz > hi || !peak_at(frame, b) {
                continue;
            }
            let heard = residual.sine_db(frame[b]) - floor_db(hz, spl);
            if heard > 0.0 && best.is_none_or(|(_, _, h)| heard > h) {
                best = Some((m, b, heard));
            }
        }
    }
    let (m, b, _) = best?;
    let frame = &residual.power[m];
    let db = |p: f64| 10.0 * libm::log10(p.max(f64::MIN_POSITIVE));
    let (a, c, e) = (db(frame[b - 1]), db(frame[b]), db(frame[b + 1]));
    let curve = a - 2.0 * c + e;
    let p = match curve < 0.0 {
        true => 0.5 * (a - e) / curve,
        false => 0.0,
    };
    let hz = residual.hz(b as f64 + p).clamp(lo, hi);
    let peak = libm::pow(10.0, (c - 0.25 * (a - e) * p) / 10.0);
    let track = |step: i64| {
        let (mut at, mut bin) = (m as i64, b);
        let mut levels = vec![residual.sine_db(frame[b])];
        loop {
            let next = at + step;
            let Some(f) = usize::try_from(next)
                .ok()
                .and_then(|n| residual.power.get(n))
            else {
                break;
            };
            let found = (bin.saturating_sub(1)..=bin + 1)
                .filter(|j| peak_at(f, *j))
                .max_by(|x, y| f[*x].total_cmp(&f[*y]));
            let Some(j) = found else { break };
            let level = residual.sine_db(f[j]);
            if !audible(residual.hz(j as f64), level, spl) {
                break;
            }
            (at, bin) = (next, j);
            levels.push(level);
        }
        (at as usize, levels)
    };
    let (first, _) = track(-1);
    let (end, levels) = track(1);
    let inside = levels.len().saturating_sub(residual.half_frames() + 1);
    Some(born(
        clip,
        Atom::partial,
        key,
        residual.span(first, m, end),
        Trajectory::steady(hz),
        (residual.sine_db(peak), levels[inside]),
    ))
}

/// A sound from `on` to `end` in the clip, at least a sample long, its level from `base` to
/// `last` dB.
fn born<S>(
    clip: &Clip,
    make: fn(u64, f64, S, Trajectory) -> Atom,
    key: u64,
    (on, end): (f64, f64),
    sound: S,
    (base, last): (f64, f64),
) -> Atom {
    let span = 1.0 / f64::from(clip.rate);
    let on = on.clamp(0.0, clip.secs() - span);
    let end = end.clamp(on + span, clip.secs());
    make(
        key,
        on,
        sound,
        Trajectory {
            base,
            knots: vec![(end, last - base)],
        },
    )
}

/// The noise band the residual's band power furthest over its floor starts.
fn noise(residual: &Spectrogram, clip: &Clip, key: u64, spl: f64) -> Option<Atom> {
    let (lo, hi) = span_hz(residual.rate);
    let edges: Vec<f64> = (0..)
        .map(|z| cam_hz(CAM_FIRST + f64::from(z)))
        .take_while(|hz| *hz <= hi)
        .collect();
    let bands: Vec<(f64, f64, usize, usize)> = edges
        .windows(2)
        .map(|e| {
            let bin = |hz: f64| libm::ceil(hz * residual.n as f64 / residual.rate) as usize;
            (e[0].max(lo), e[1], bin(e[0]), bin(e[1]))
        })
        .filter(|(_, _, a, b)| b > a)
        .collect();
    let powers: Vec<Vec<f64>> = residual
        .power
        .iter()
        .map(|f| {
            bands
                .iter()
                .map(|(_, _, a, b)| f[*a..*b].iter().sum())
                .collect()
        })
        .collect();
    let centre = |k: usize| cam_hz((cam(bands[k].0) + cam(bands[k].1)) / 2.0);
    let over = |m: usize, k: usize| residual.noise_db(powers[m][k]) - floor_db(centre(k), spl);
    let heard = |m: usize, k: usize| over(m, k) > 0.0;
    let mut best: Option<(usize, usize, f64)> = None;
    for m in 0..powers.len() {
        for k in 0..bands.len() {
            let h = over(m, k);
            if h > 0.0 && best.is_none_or(|(_, _, b)| h > b) {
                best = Some((m, k, h));
            }
        }
    }
    let (m, k, _) = best?;
    let first = (0..=m).rev().take_while(|f| heard(*f, k)).last()?;
    let end = (m..powers.len()).take_while(|f| heard(*f, k)).last()?;
    Some(born(
        clip,
        Atom::noise,
        key,
        residual.span(first, m, end),
        (bands[k].0, bands[k].1),
        (
            residual.noise_db(powers[m][k]),
            residual.noise_db(powers[end.saturating_sub(residual.half_frames()).max(m)][k]),
        ),
    ))
}

/// Where a kind's next birth starts, from the residual.
type Finder = fn(&Spectrogram, &Clip, u64, f64) -> Option<Atom>;

/// Births until the best of a kind does not pay, partials then noise, from `state`.
pub fn add(objective: &Objective, clip: &Spectrogram, mut state: State, run: &mut Run) -> State {
    let found: [Finder; 2] = [partial, noise];
    let at = objective.clip();
    let mut tried_count = 0;
    for find in found {
        loop {
            if (run.stopped)() {
                return state;
            }
            let model = Spectrogram::of(&rendered(&state.atoms, &at), at.rate);
            let residual = clip.less(&model);
            let Some(atom) = find(&residual, &at, birth(&state.atoms), run.spl_db) else {
                break;
            };
            tried_count += 1;
            run.activity(
                Action::Ranking,
                Some(&edit(&atom)),
                Some((tried_count, "proposal")),
            );
            match tried(objective, &state, atom, clip.bin_hz(), run) {
                Some(grown) => {
                    run.edits.push(edit(grown.atoms.last().expect("a birth")));
                    state = grown;
                }
                None => break,
            }
        }
    }
    state
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
        _ => format!(
            "partial {} Hz at {} s",
            libm::round(atom.hz().map_or(0.0, |t| t.base)),
            atom.on
        ),
    }
}
