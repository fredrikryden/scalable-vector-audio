// Concern: atoms through the ear: power per bin with its proven bound, noise in expectation, an atom's level floor | Non-concern: elements, their distance | IO: (atoms, Frame) -> Bins; (Atom, Ear) -> dB

use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::{SQRT_2, TAU};
use std::rc::Rc;

use sva_engine::internal::{Ear, Frame};
use sva_engine::{Grid, PSYCHOACOUSTIC_V1, fft};
use sva_formula::Knots;
use sva_formula::closed_form::Bound;

use super::atoms::{Atom, Clip, NEPER_PER_DB, Sound, Trajectory};

pub const MODEL: &str = "each atom's samples are its node's formula evaluated as the engine \
     point-samples it: the instant n/rate, the crop's edges met as the grid meets them, phase \
     and level read off the same knot tables, sin and exp as the engine calls them; partials \
     and harmonics are summed sample by sample where `coherent`, else each transformed alone \
     and added in power; a noise is read in expectation over its lines' phases, independent \
     and uniform, as the window's autocorrelation times the lines' summed off one transform; \
     every bin carries a bound on its distance from the same transform of the exact formula \
     at the exact instants, from the rounding of each operation (2^-53 each, 1 ulp for each \
     libm call), the FFT's by Higham's (2002, Accuracy and Stability of Numerical Algorithms, \
     2nd ed., Thm 24.2) 2-norm bound with twiddles within 12 * 2^-53, carried in 2-norm";

const U: f64 = f64::EPSILON / 2.0;

/// `|X|^2` per bin as `Frame::power` answers samples, each within `bound` of the exact formula's.
#[derive(Clone, Debug, PartialEq)]
pub struct Bins {
    pub power: Vec<f64>,
    pub bound: Vec<f64>,
}

impl Bins {
    fn silent(bins: usize) -> Bins {
        Bins {
            power: vec![0.0; bins],
            bound: vec![0.0; bins],
        }
    }

    fn add(mut self, other: Bins) -> Bins {
        for (j, (p, b)) in self.power.iter_mut().zip(&mut self.bound).enumerate() {
            *p += other.power[j];
            *b += other.bound[j] + 2.0 * U * p.abs();
        }
        self
    }
}

/// `atoms` as one state reads them: noise atoms in expectation, the rest summed in samples
/// where `coherent`, else in power; a band's line autocorrelation kept per window length.
pub struct Model<'a> {
    atoms: &'a [Atom],
    clip: Clip,
    coherent: bool,
    correlations: RefCell<HashMap<(u64, u64, usize), Rc<Correlation>>>,
}

impl<'a> Model<'a> {
    pub fn new(atoms: &'a [Atom], clip: Clip, coherent: bool) -> Model<'a> {
        Model {
            atoms,
            clip,
            coherent,
            correlations: RefCell::default(),
        }
    }

    pub fn bins(&self, frame: &Frame) -> Bins {
        let count = frame.weights().len();
        let mut out = Bins::silent(frame.hz().len());
        let windows: Vec<(Vec<f64>, Vec<f64>)> = self
            .atoms
            .iter()
            .filter(|a| !matches!(a.sound, Sound::Noise { .. }))
            .map(|a| samples(a, &self.clip, frame.first(), count))
            .collect();
        match self.coherent {
            true if !windows.is_empty() => {
                let mut sum = vec![0.0; count];
                let mut bound = vec![0.0; count];
                let mut magnitude = vec![0.0; count];
                for (x, b) in &windows {
                    for k in 0..count {
                        sum[k] += x[k];
                        bound[k] += b[k];
                        magnitude[k] += x[k].abs();
                    }
                }
                let terms = windows.len() as f64;
                for k in 0..count {
                    bound[k] += 2.0 * terms * U * magnitude[k];
                }
                out = out.add(transformed(frame, &sum, &bound));
            }
            _ => {
                for (x, b) in &windows {
                    out = out.add(transformed(frame, x, b));
                }
            }
        }
        for atom in self.atoms {
            if let Sound::Noise { lo, hi, .. } = atom.sound {
                let correlation = self
                    .correlations
                    .borrow_mut()
                    .entry((lo.to_bits(), hi.to_bits(), frame.n()))
                    .or_insert_with(|| Rc::new(correlation(&self.clip, (lo, hi), frame.n())))
                    .clone();
                out = out.add(expected(atom, &self.clip, frame, &correlation));
            }
        }
        out
    }

    /// E per channel at frame `m`, bracketed.
    pub fn excitation(&self, ear: &Ear, m: i64) -> Heard {
        let mut held = Vec::new();
        let excitation = ear.excitation(m, &mut |frame| {
            let bins = self.bins(frame);
            let power = bins.power.clone();
            held.push(bins);
            power
        });
        let (least, most) = (bracket(ear, m, &held, -1.0), bracket(ear, m, &held, 1.0));
        Heard {
            excitation,
            least,
            most,
        }
    }
}

/// E sums `w P` over bins, `w = (1 + p g) e^(-p g)` falling in `p` (slope `-p g^2 e^(-p g)`),
/// `p` fixed above the centre and below it falling in `X`, which rises with every `P`. So E
/// rises with every bin's power, and bins moved by their bounds bracket the exact formula's.
#[derive(Clone, Debug, PartialEq)]
pub struct Heard {
    pub excitation: Vec<f64>,
    pub least: Vec<f64>,
    pub most: Vec<f64>,
}

/// The excitation of every bin moved `side` times its bound, never under silence.
fn bracket(ear: &Ear, m: i64, held: &[Bins], side: f64) -> Vec<f64> {
    let mut next = held.iter();
    ear.excitation(m, &mut |_| {
        let bins = next.next().expect("one set of bins per range read");
        moved(bins, side)
    })
}

fn moved(bins: &Bins, side: f64) -> Vec<f64> {
    bins.power
        .iter()
        .zip(&bins.bound)
        .map(|(p, b)| (p + side * b).max(0.0))
        .collect()
}

fn tables(atom: &Atom) -> Option<(Knots, Option<Knots>)> {
    let table = |t: &Trajectory, hold: Option<f64>, integral| {
        let mut times = vec![atom.on];
        let mut values = vec![t.base];
        for (at, d) in &t.knots {
            times.push(*at);
            values.push(t.base + d);
        }
        if let Some(end) = hold {
            times.push(match t.knots.last() {
                Some((at, _)) => at.max(end),
                None => end,
            });
            values.push(*values.last().expect("a base"));
        }
        Knots::new(times, values, integral).ok()
    };
    let level = table(&atom.level, None, false)?;
    let hz = match atom.hz() {
        Some(hz) => Some(table(hz, Some(atom.end()), true)?),
        None => None,
    };
    Some((level, hz))
}

/// Whether sample `n` lies in `[l, r)` as a crop meets the grid: by its instant, or at a tie
/// by the first sample at or past the edge's decimal.
fn inside(grid: &Grid, n: i64, (l, r): (f64, f64)) -> bool {
    let t = grid.instant(n);
    let reached = |edge: f64| match t == edge {
        true => n >= grid.first_at(edge),
        false => t > edge,
    };
    reached(l) && !reached(r)
}

fn amplitude(level: &Knots, steep: f64, t: f64, hint: &mut usize) -> (f64, f64) {
    let v = level.read(t, false, hint);
    let arg = NEPER_PER_DB * v;
    let a = sva_formula::exp(arg);
    let moved = NEPER_PER_DB * (level.rounding() + steep * U * t.abs()) + U * arg.abs();
    (a, a * (2.0 * libm::expm1(moved) + 3.0 * U))
}

/// dB per second, so a rounded instant moves the level at most that.
fn steepest(t: &Trajectory, on: f64) -> f64 {
    let mut last = (on, t.base);
    t.knots.iter().fold(0.0, |m: f64, (at, d)| {
        let v = t.base + d;
        let rate = match *at > last.0 {
            true => (v - last.1).abs() / (at - last.0),
            false => 0.0,
        };
        last = (*at, v);
        m.max(rate)
    })
}

/// A partial's or harmonic's samples `from..from + count`, each beside a bound on its distance
/// from the formula's exact value at the exact instant; an atom whose tables its node refuses
/// is silence, as its crop is.
fn samples(atom: &Atom, clip: &Clip, from: i64, count: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut bound) = (vec![0.0; count], vec![0.0; count]);
    let Some((level, Some(hz))) = tables(atom) else {
        return (x, bound);
    };
    let lines: Vec<(f64, f64, f64)> = match &atom.sound {
        Sound::Partial { phase, .. } => vec![(TAU, *phase, 1.0)],
        Sound::Harmonic {
            offsets, phases, ..
        } => offsets
            .iter()
            .zip(phases)
            .enumerate()
            .map(|(k, (o, ph))| {
                (
                    TAU * (k + 1) as f64,
                    *ph,
                    sva_formula::exp(NEPER_PER_DB * o),
                )
            })
            .collect(),
        Sound::Noise { .. } => unreachable!("a noise is read in expectation"),
    };
    let fastest = hz.values().iter().fold(0.0, |m: f64, v| m.max(v.abs()));
    let steep = steepest(&atom.level, atom.on);
    let grid = Grid::of(clip.rate);
    let (mut at_level, mut at_hz) = (0, 0);
    for k in 0..count {
        let n = from + k as i64;
        if n < 0 || n >= clip.len as i64 || !inside(&grid, n, (atom.on, atom.end())) {
            continue;
        }
        let t = grid.instant(n);
        let (a, a_bound) = amplitude(&level, steep, t, &mut at_level);
        let turned = hz.read(t, false, &mut at_hz);
        let turned_bound = hz.rounding() + fastest * U * t.abs();
        let (mut sum, mut sum_bound, mut magnitude) = (0.0, 0.0, 0.0);
        for (k_tau, ph, gain) in &lines {
            let s = libm::sin(k_tau * turned + ph);
            let angle = k_tau * turned_bound + 4.0 * U * (k_tau * turned.abs() + ph.abs());
            let gain_bound = gain * (3.0 * U + 2.0 * U * libm::log(*gain).abs());
            let term = gain * s;
            sum += term;
            magnitude += term.abs();
            sum_bound += gain * (angle + 2.0 * U) + s.abs() * gain_bound + 2.0 * U * term.abs();
        }
        sum_bound += 2.0 * lines.len() as f64 * U * magnitude;
        x[k] = a * sum;
        bound[k] = a * sum_bound + sum.abs() * a_bound + 2.0 * U * x[k].abs();
    }
    (x, bound)
}

fn fft_bound(n: usize) -> f64 {
    let mu = 12.0 * U;
    let gamma4 = 4.0 * U / (1.0 - 4.0 * U);
    let eta = mu + gamma4 * (SQRT_2 + mu);
    let depth = f64::from(n.trailing_zeros()) * eta;
    depth / (1.0 - depth)
}

fn norm(xs: impl Iterator<Item = f64>) -> f64 {
    xs.map(|x| x * x).sum::<f64>().sqrt()
}

fn transformed(frame: &Frame, x: &[f64], bound: &[f64]) -> Bins {
    let power = frame.power(&[x], frame.first());
    let w = frame.weights();
    let weighted = norm(w.iter().zip(x).map(|(w, x)| w * x));
    let input = norm(
        w.iter()
            .zip(x.iter().zip(bound))
            .map(|(w, (x, b))| w * b + U * (w * x).abs()),
    );
    let root = (frame.n() as f64).sqrt();
    let err = root * (input + fft_bound(frame.n()) * (weighted + input));
    let bound = power
        .iter()
        .map(|p| 5.0 * U * p + (2.0 * (2.0 * p * (1.0 + 5.0 * U)).sqrt() + err) * err)
        .collect();
    Bins { power, bound }
}

/// A noise's expected power per bin: its lines' phases independent and uniform, so
/// `E|X_j|^2 = sum_d R(d) C(d) e^(-2 pi i j d/n)`, with `R` the lines' autocorrelation and `C`
/// the weighted envelope's.
fn expected(atom: &Atom, clip: &Clip, frame: &Frame, correlation: &Correlation) -> Bins {
    let Sound::Noise { lo, hi, .. } = atom.sound else {
        unreachable!("a noise")
    };
    let n = frame.n();
    let silent = Bins::silent(frame.hz().len());
    let Some((level, _)) = tables(atom) else {
        return silent;
    };
    let Correlation {
        values: correlation,
        bounds: correlation_bound,
    } = correlation;
    if correlation.is_empty() {
        return silent;
    }
    let grid = Grid::of(clip.rate);
    let steep = steepest(&atom.level, atom.on);
    if hi <= lo {
        return silent;
    }
    let width = (hi - lo).sqrt();
    let mut hint = 0;
    let (mut y, mut y_bound) = (vec![0.0; 2 * n], vec![0.0; 2 * n]);
    for (k, w) in frame.weights().iter().enumerate() {
        let s = frame.first() + k as i64;
        if s < 0 || s >= clip.len as i64 || !inside(&grid, s, (atom.on, atom.end())) {
            continue;
        }
        let (a, a_bound) = amplitude(&level, steep, grid.instant(s), &mut hint);
        y[k] = w * (a / width);
        y_bound[k] = w * (a_bound / width) + 4.0 * U * y[k].abs();
    }
    if y.iter().all(|v| *v == 0.0) {
        return silent;
    }

    let wide = 2 * n;
    let (wide_root, root) = ((wide as f64).sqrt(), (n as f64).sqrt());
    let held_input = norm(y_bound.iter().copied());
    let mut im = vec![0.0; wide];
    let twiddles = fft::twiddles(wide);
    let y_norm = norm(y.iter().copied());
    fft::fft_with(&mut y, &mut im, &twiddles);
    let spread = wide_root * held_input + fft_bound(wide) * wide_root * (y_norm + held_input);
    let mut power: Vec<f64> = y.iter().zip(&im).map(|(r, i)| r * r + i * i).collect();
    let peak = power.iter().fold(0.0, |m: f64, p| m.max(p.sqrt()));
    let power_norm = norm(power.iter().copied());
    let power_bound = 3.0 * U * power_norm + (2.0 * peak + spread) * spread;
    let mut im = vec![0.0; wide];
    fft::ifft_with(&mut power, &mut im, &twiddles);
    let window = power;
    let window_bound = (wide_root * power_bound
        + fft_bound(wide) * wide_root * (power_norm + power_bound))
        / wide as f64;

    let count = frame.weights().len();
    let mut folded = vec![0.0; n];
    for (i, g) in folded.iter_mut().enumerate() {
        if i < count {
            *g += correlation[i] * window[i];
        }
        if i > 0 && n - i < count {
            *g += correlation[n - i] * window[n - i];
        }
    }
    let r_most = correlation.iter().fold(0.0, |m: f64, r| m.max(r.abs()));
    let r_bound = correlation_bound.iter().fold(0.0, |m: f64, r| m.max(*r));
    let window_norm = norm(window[..count].iter().copied());
    let folded_norm = norm(folded.iter().copied());
    let folded_bound = SQRT_2 * ((r_most + r_bound) * window_bound + r_bound * window_norm)
        + 3.0 * U * SQRT_2 * folded_norm;
    let mut im = vec![0.0; n];
    fft::fft_with(&mut folded, &mut im, &fft::twiddles(n));
    let err = root * folded_bound + fft_bound(n) * root * (folded_norm + folded_bound);
    let bins = frame.hz().len();
    Bins {
        power: folded[..bins].to_vec(),
        bound: vec![err; bins],
    }
}

/// A noise's lines' autocorrelation at lags `0..n` samples, each beside its bound; empty
/// where no line sounds.
struct Correlation {
    values: Vec<f64>,
    bounds: Vec<f64>,
}

fn correlation(clip: &Clip, (lo, hi): (f64, f64), n: usize) -> Correlation {
    let none = Correlation {
        values: Vec::new(),
        bounds: Vec::new(),
    };
    let period = clip.secs();
    let Some((first, Bound::Finite(last))) = sva_formula::noise::band(period, lo, Some(hi)) else {
        return none;
    };
    let turning = TAU / period;
    let omega = TAU * PSYCHOACOUSTIC_V1.ceiling(clip.rate);
    let Some(last) = (first..=last)
        .take_while(|k| (*k as f64 * turning).abs() < omega)
        .last()
    else {
        return none;
    };
    let lines = (last - first + 1) as f64;
    let amplitude = libm::pow(period, -0.5);
    let half_power = amplitude * amplitude / 2.0;
    let rate = f64::from(clip.rate);
    let (values, bounds) = (0..n)
        .map(|d| {
            let lag = d as f64 / rate;
            let r = (first..=last)
                .map(|k| half_power * libm::cos(turning * k as f64 * lag))
                .sum::<f64>();
            let widest = turning * last as f64 * lag;
            (
                r,
                half_power * lines * (3.0 * U * widest + 4.0 * U + lines * U),
            )
        })
        .unzip();
    Correlation { values, bounds }
}

/// Doubles in order as integers, so bisection ends within 64 steps.
fn key(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    bits ^ (((bits >> 63) as u64) >> 1) as i64
}

fn unkey(k: i64) -> f64 {
    f64::from_bits((k ^ (((k >> 63) as u64) >> 1) as i64) as u64)
}

/// Steps doubling from one ulp: at most 64 before the largest double.
fn past(from: f64, reaches: &mut dyn FnMut(f64) -> bool) -> f64 {
    let (top, mut step, mut at) = (key(f64::MAX), 1i64, from);
    while !reaches(at) && at < f64::MAX {
        at = unkey(key(at).saturating_add(step).min(top));
        step = step.saturating_mul(2);
    }
    at
}

fn lowest(lo: f64, hi: f64, reaches: &mut dyn FnMut(f64) -> bool) -> f64 {
    if reaches(lo) {
        return lo;
    }
    let (mut lo, mut hi) = (key(lo), key(hi));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match reaches(unkey(mid)) {
            true => hi = mid,
            false => lo = mid,
        }
    }
    unkey(hi)
}

/// An atom's level floor in dBFS, and a bound on its distance from the exact formula's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Floor {
    pub db: f64,
    pub bound: f64,
}

/// The lowest base level at which `atom` reaches threshold in some channel at some frame. The
/// floors of every bin moved up and down by its bound bracket the exact formula's, as the
/// excitation rises with each bin's power.
pub fn floor_db(atom: &Atom, clip: &Clip, ear: &Ear) -> Floor {
    let model = Model::new(std::slice::from_ref(atom), *clip, true);
    let [db, least, most] =
        [0.0, 1.0, -1.0].map(|side| atom.level.base + crossing(&model, clip, ear, side));
    let gap = |a: f64, b: f64| match a == b {
        true => 0.0,
        false => (a - b).abs(),
    };
    Floor {
        db,
        bound: gap(db, least).max(gap(most, db)) + 3.0 * U * db.abs(),
    }
}

/// The lowest shift of every level at which bins moved `side` times their bound reach
/// threshold, infinite where none does. A shift `c` adds `c` to every `X` and scales every `P`
/// by `10^(c/10)`, so (as for `Heard`) each E moves under `10^(c/10)` times for `c <= 0` and
/// over it above, which bounds each frame's crossing from its loudness at no shift.
fn crossing(model: &Model, clip: &Clip, ear: &Ear, side: f64) -> f64 {
    let thresholds = ear.thresholds();
    let loudest = |e: Vec<f64>| {
        e.iter()
            .zip(&thresholds)
            .fold(0.0, |most: f64, (e, t)| most.max(e / t))
    };
    let ratio = |powers: &[Vec<f64>], c: f64, m: i64| {
        let gain = libm::pow(10.0, c / 10.0);
        let mut next = powers.iter();
        loudest(ear.excitation(m, &mut |_| {
            let held = next.next().expect("one power per range read");
            held.iter().map(|p| p * gain).collect()
        }))
    };
    let powers = |m: i64| {
        let mut held = Vec::new();
        let e = ear.excitation(m, &mut |frame| {
            let power = moved(&model.bins(frame), side);
            held.push(power.clone());
            power
        });
        (loudest(e), held)
    };
    let (first, last) = ear.frames(clip.len);
    let mut heard: Vec<(f64, i64)> = (first..=last)
        .map(|m| (powers(m).0, m))
        .filter(|(r, _)| *r > 0.0)
        .collect();
    heard.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut best = f64::INFINITY;
    for (r, m) in heard {
        let db = -10.0 * libm::log10(r);
        let (below, above) = match r >= 1.0 {
            true => (db, 0.0),
            false => (0.0, db),
        };
        if below >= best {
            break;
        }
        let held = powers(m).1;
        let mut reaches = |c: f64| ratio(&held, c, m) >= 1.0;
        let mut hi = best.min(above);
        if !reaches(hi) {
            hi = match best.is_finite() {
                true => best,
                false => past(hi, &mut reaches),
            };
        }
        if !reaches(hi) || (hi == best && !reaches(best.next_down())) {
            continue;
        }
        best = lowest(below, hi, &mut reaches);
    }
    best
}
