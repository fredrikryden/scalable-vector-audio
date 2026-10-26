// Concern: atoms through the ear: bins with proven bounds, noise in expectation, tangents, its floor | Non-concern: elements, their distance | IO: (atoms, Frame) -> Bins, tangents; (Atom, Ear) -> dB

use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2, TAU};
use std::rc::Rc;

use sva_engine::internal::{Ear, Frame};
use sva_engine::{Grid, PSYCHOACOUSTIC_V1, fft};
use sva_formula::{Knots, Num};

use super::atoms::{Atom, Clip, NEPER_PER_DB, Sound, Trajectory, phase};
use super::dual::Dual;

pub const MODEL: &str = "each atom's samples are its node's formula evaluated as the engine \
     point-samples it, the instant n/rate, phase and level read off the same knot tables, sin \
     and exp as the engine calls them, then gated: sample n weighs its share of [on, end) over \
     its cell [(2n - 1)/(2 rate), (2n + 1)/(2 rate)), the atom held at its base level and \
     turning at its base frequency before on, and at its last level and frequency past end; \
     partials and harmonics are summed sample by sample where `coherent`, else each \
     transformed alone and added in power; a noise is read in expectation over its lines' \
     phases, independent and uniform, as the window's autocorrelation times the band's \
     continuous autocorrelation, (1/2) int_lo^hi' cos(2 pi f tau) df, hi' = min(hi, the \
     audible ceiling); every bin carries `bound`, its distance from the same transform of \
     that formula at the exact instants, from the rounding of each operation (2^-53 each, 1 \
     ulp for each libm call), the FFT's by Higham's (2002, Accuracy and Stability of Numerical \
     Algorithms, 2nd ed., Thm 24.2) 2-norm bound with twiddles within 12 * 2^-53, carried in \
     2-norm, and `sampling`, that formula's distance from the render's own exact power (its \
     crop met as the grid meets it, its noise's discrete lines, in expectation): a share \
     differs from the crop's 0 or 1 by at most 1/2, on one sample per edge, so |h - share| \
     times the ungated sample, times its window weight and summed, eps, bounds each bin's \
     amplitude gap and eps (2|X| + eps) its power's; a noise's crop moves each bin's amplitude \
     by at most eps = sum_k |w_k| |h_k - share_k| y_k sqrt(R_lines(0)), y the ungated \
     envelope and R_lines(0) <= R(0) + h, in RMS over the ensemble (Minkowski), so its power \
     by eps (2 sqrt(P) + eps) (Cauchy-Schwarz); the band's lines k/T, each of power h = \
     1/(2T), sum at lag tau to h sum cos(k theta), theta = 2 pi tau/T, which is the integral \
     over [k_0 - 1/2, k_1 + 1/2] plus h sin(N theta/2) cos(c theta)(1/sin(theta/2) - \
     2/theta), each end within 1/2 of lo T and hi' T, so their gap is at most h(1 + \
     1/sin(theta/2) - 2/theta) for 0 < theta < 2 pi, h at 0, and h(N + T(hi' - lo)), N <= \
     T(hi' - lo) + 1, at any lag, and moves each bin by at most sum_d |dR(d)| |C(d)|, C the \
     weighted envelope's autocorrelation";

const U: f64 = f64::EPSILON / 2.0;

/// `|X|^2` per bin as `Frame::power` answers samples: each within `bound` of the model's exact
/// formula, which is within `sampling` of the render's exact power (a noise's in expectation).
#[derive(Clone, Debug, PartialEq)]
pub struct Bins {
    pub power: Vec<f64>,
    pub bound: Vec<f64>,
    pub sampling: Vec<f64>,
}

impl Bins {
    fn silent(bins: usize) -> Bins {
        Bins {
            power: vec![0.0; bins],
            bound: vec![0.0; bins],
            sampling: vec![0.0; bins],
        }
    }

    fn add(mut self, other: Bins) -> Bins {
        for (j, (p, b)) in self.power.iter_mut().zip(&mut self.bound).enumerate() {
            *p += other.power[j];
            *b += other.bound[j] + 2.0 * U * p.abs();
            self.sampling[j] += other.sampling[j];
        }
        self
    }
}

/// `atoms` as one state reads them: noise atoms in expectation, the rest summed in samples
/// where `coherent`, else in power; a band's autocorrelation kept per window length.
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

    fn tonal(&self) -> impl Iterator<Item = &Atom> {
        self.atoms
            .iter()
            .filter(|a| !matches!(a.sound, Sound::Noise { .. }))
    }

    pub fn bins(&self, frame: &Frame) -> Bins {
        let count = frame.weights().len();
        let mut out = Bins::silent(frame.hz().len());
        let windows: Vec<Window<f64>> = self
            .tonal()
            .map(|a| samples(a, &self.clip, frame.first(), count))
            .collect();
        match self.coherent {
            true if !windows.is_empty() => {
                let mut sum = Window::silent(count);
                let mut magnitude = vec![0.0; count];
                for w in &windows {
                    for (k, m) in magnitude.iter_mut().enumerate() {
                        sum.x[k] += w.x[k];
                        sum.bound[k] += w.bound[k];
                        sum.sampling[k] += w.sampling[k];
                        *m += w.x[k].abs();
                    }
                }
                let terms = windows.len() as f64;
                for (b, m) in sum.bound.iter_mut().zip(&magnitude) {
                    *b += 2.0 * terms * U * m;
                }
                let silent = [&sum.x, &sum.bound, &sum.sampling]
                    .iter()
                    .all(|v| v.iter().all(|x| *x == 0.0));
                if !silent {
                    out = out.add(transformed(frame, &sum));
                }
            }
            _ => {
                for w in &windows {
                    out = out.add(transformed(frame, w));
                }
            }
        }
        for atom in self.atoms {
            if let Sound::Noise { lo, hi } = atom.sound {
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

    /// Each bin's `d|X|^2/dx` for every number `x` of `atoms[i]` in `values()`' order, by forward
    /// mode: `2 Re(conj(X) dX)`, a noise's through its expected power.
    pub fn tangents(&self, frame: &Frame, i: usize) -> Vec<Vec<f64>> {
        let atom = &self.atoms[i];
        let numbers = atom.values().len();
        if let Sound::Noise { .. } = atom.sound {
            return (0..numbers)
                .map(|j| expected_tangent(&atom.seeded(j), &self.clip, frame))
                .collect();
        }
        let count = frame.weights().len();
        let mut sum = vec![0.0; count];
        let heard: Vec<&Atom> = match self.coherent {
            true => self.tonal().collect(),
            false => vec![atom],
        };
        for a in heard {
            for (s, x) in sum
                .iter_mut()
                .zip(samples(a, &self.clip, frame.first(), count).x)
            {
                *s += x;
            }
        }
        let (re, im) = transform(frame, &sum);
        (0..numbers)
            .map(|j| {
                let moved = samples(&atom.seeded(j), &self.clip, frame.first(), count);
                let dx: Vec<f64> = moved.x.iter().map(|x| x.tangent).collect();
                let (d_re, d_im) = transform(frame, &dx);
                frame
                    .bins()
                    .map(|b| 2.0 * (re[b] * d_re[b] + im[b] * d_im[b]))
                    .collect()
            })
            .collect()
    }

    /// E per channel at frame `m`, bracketed for the model's exact formula and for the render.
    pub fn excitation(&self, ear: &Ear, m: i64) -> Heard {
        let bins = composed(ear, m, &mut |frame| self.bins(frame));
        let excitation = ear.excite(m, &bins.power);
        let [least, most, render_least, render_most] =
            [(-1.0, false), (1.0, false), (-1.0, true), (1.0, true)]
                .map(|(side, sampled)| ear.excite(m, &shifted(&bins, side, sampled)));
        Heard {
            excitation,
            least,
            most,
            render_least,
            render_most,
        }
    }
}

/// Frame `m`'s bins composed into the ear's one spectrum, each with its bound and sampling.
fn composed(ear: &Ear, m: i64, bins: &mut dyn FnMut(&Frame) -> Bins) -> Bins {
    let each = ear.compose(m, &mut |frame| {
        let Bins {
            power,
            bound,
            sampling,
        } = bins(frame);
        power
            .into_iter()
            .zip(bound)
            .zip(sampling)
            .map(|((p, b), s)| (p, b, s))
            .collect()
    });
    Bins {
        power: each.iter().map(|b| b.0).collect(),
        bound: each.iter().map(|b| b.1).collect(),
        sampling: each.iter().map(|b| b.2).collect(),
    }
}

/// E sums `w P` over bins, `w = (1 + p g) e^(-p g)` falling in `p` (slope `-p g^2 e^(-p g)`),
/// `p` fixed above the centre and below it falling in `X`, which rises with every `P`. So E
/// rises with every bin's power, and bins moved by their bounds bracket the exact formula's,
/// by their bounds and sampling the render's.
#[derive(Clone, Debug, PartialEq)]
pub struct Heard {
    pub excitation: Vec<f64>,
    pub least: Vec<f64>,
    pub most: Vec<f64>,
    pub render_least: Vec<f64>,
    pub render_most: Vec<f64>,
}

/// Every bin moved `side` times its bound, and its sampling where `sampled`, never under
/// silence.
fn shifted(bins: &Bins, side: f64, sampled: bool) -> Vec<f64> {
    (0..bins.power.len())
        .map(|j| {
            let reach = bins.bound[j] + if sampled { bins.sampling[j] } else { 0.0 };
            (bins.power[j] + side * reach).max(0.0)
        })
        .collect()
}

fn tables<N: Num>(atom: &Atom<N>) -> Option<(Knots<N>, Option<Knots<N>>)> {
    let table = |t: &Trajectory<N>, hold: Option<N>, integral| {
        let mut times = vec![atom.on];
        let mut values = vec![t.base];
        for (at, d) in &t.knots {
            times.push(*at);
            values.push(t.base + *d);
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

/// Sample `n`'s share of `[on, end)` over its cell, edges in f64 so cells tile time, beside
/// its rounding: one subtraction and one product, each within 2^-53 relatively, of a share <= 1.
fn gate<N: Num>(n: i64, rate: f64, on: N, end: N) -> (N, f64) {
    let edge = |n: i64| (2 * n - 1) as f64 / (2.0 * rate);
    let (lo, hi) = (edge(n), edge(n + 1));
    if on.v() <= lo && hi <= end.v() {
        return (N::ONE, 0.0);
    }
    if end.v() <= lo || hi <= on.v() {
        return (N::ZERO, 0.0);
    }
    ((end.min(N::of(hi)) - on.max(N::of(lo))) * rate, 3.0 * U)
}

/// Held at its base before its first knot and at its last value from its last on.
fn level_at<N: Num>(level: &Knots<N>, t: f64, hint: &mut usize) -> N {
    let values = level.values();
    match () {
        () if t < level.first() => values[0],
        () if t >= level.last() => values[values.len() - 1],
        () => level.read(N::of(t), false, hint),
    }
}

/// The running integral, continued at its first value before its first knot and its last
/// past its last; whether `t` lies outside.
fn turned_at<N: Num>(hz: &Knots<N>, t: f64, hint: &mut usize) -> (N, bool) {
    let (times, values) = (hz.times(), hz.values());
    let last = times.len() - 1;
    match () {
        () if t < hz.first() => ((N::of(t) - times[0]) * values[0], true),
        () if t > hz.last() => (hz.held() + (N::of(t) - times[last]) * values[last], true),
        () => (hz.read(N::of(t), false, hint), false),
    }
}

fn amplitude<N: Num>(level: &Knots<N>, steep: f64, t: f64, hint: &mut usize) -> (N, f64) {
    let arg = level_at(level, t, hint) * NEPER_PER_DB;
    let a = arg.exp();
    let moved = NEPER_PER_DB * (level.rounding() + steep * U * t.abs()) + U * arg.v().abs();
    (a, a.v() * (2.0 * libm::expm1(moved) + 3.0 * U))
}

/// dB per second, so a rounded instant moves the level at most that.
fn steepest<N: Num>(t: &Trajectory<N>, on: f64) -> f64 {
    let mut last = (on, t.base.v());
    t.knots.iter().fold(0.0, |m: f64, (at, d)| {
        let (at, v) = (at.v(), t.base.v() + d.v());
        let rate = match at > last.0 {
            true => (v - last.1).abs() / (at - last.0),
            false => 0.0,
        };
        last = (at, v);
        m.max(rate)
    })
}

/// Samples `from..from + count`, each gated, beside its bound and its sampling, per sample.
struct Window<N> {
    x: Vec<N>,
    bound: Vec<f64>,
    sampling: Vec<f64>,
}

impl Window<f64> {
    fn silent(count: usize) -> Window<f64> {
        Window {
            x: vec![0.0; count],
            bound: vec![0.0; count],
            sampling: vec![0.0; count],
        }
    }
}

/// Each harmonic's turning rate, phase, gain (`None` for exactly 1) and the gain's bound: the
/// last at minus the others' sum, as its node folds it.
fn lines<N: Num>(atom: &Atom<N>) -> Vec<(f64, f64, Option<N>, f64)> {
    let line = |k: usize| (TAU * k as f64, phase(atom.key, k));
    let gained = |arg: N, extra: f64| {
        let g = arg.exp();
        let at = g.v();
        (
            Some(g),
            at * (3.0 * U + 2.0 * U * libm::log(at).abs() + extra),
        )
    };
    let Sound::Harmonic { offsets, .. } = &atom.sound else {
        let (k_tau, ph) = line(1);
        return vec![(k_tau, ph, None, 0.0)];
    };
    let mut out: Vec<(f64, f64, Option<N>, f64)> = offsets
        .iter()
        .enumerate()
        .map(|(k, o)| {
            let (k_tau, ph) = line(k + 1);
            let (g, b) = gained(*o * NEPER_PER_DB, 0.0);
            (k_tau, ph, g, b)
        })
        .collect();
    let (k_tau, ph) = line(atom.harmonics());
    match offsets.is_empty() {
        true => out.push((k_tau, ph, None, 0.0)),
        false => {
            let sum = offsets.iter().fold(N::ZERO, |s, o| s + *o);
            let spread = offsets.iter().map(|o| o.v().abs()).sum::<f64>();
            let summing = NEPER_PER_DB * 2.0 * offsets.len() as f64 * U * spread;
            let (g, b) = gained(-sum * NEPER_PER_DB, summing);
            out.push((k_tau, ph, g, b));
        }
    }
    out
}

/// A partial's or harmonic's window of samples; an atom whose tables its node refuses is
/// silence, as its crop is.
fn samples<N: Num>(atom: &Atom<N>, clip: &Clip, from: i64, count: usize) -> Window<N> {
    let mut out = Window {
        x: vec![N::ZERO; count],
        bound: vec![0.0; count],
        sampling: vec![0.0; count],
    };
    let Some((level, Some(hz))) = tables(atom) else {
        return out;
    };
    let lines = lines(atom);
    let fastest = hz.values().iter().fold(0.0, |m: f64, v| m.max(v.v().abs()));
    let steep = steepest(&atom.level, atom.on.v());
    let (grid, rate) = (Grid::of(clip.rate), f64::from(clip.rate));
    let (on, end) = (atom.on, atom.end());
    let (mut at_level, mut at_hz) = (0, 0);
    for k in 0..count {
        let n = from + k as i64;
        if n < 0 || n >= clip.len as i64 {
            continue;
        }
        let (share, share_bound) = gate(n, rate, on, end);
        let heard = inside(&grid, n, (on.v(), end.v()));
        if share.v() == 0.0 && !heard {
            continue;
        }
        let t = grid.instant(n);
        let (a, a_bound) = amplitude(&level, steep, t, &mut at_level);
        let (turned, outside) = turned_at(&hz, t, &mut at_hz);
        let turned_bound = hz.rounding()
            + fastest * U * t.abs()
            + if outside {
                3.0 * U * turned.v().abs()
            } else {
                0.0
            };
        let (mut sum, mut sum_bound, mut magnitude) = (N::ZERO, 0.0, 0.0);
        for (k_tau, ph, gain, gain_bound) in &lines {
            let s = (turned * *k_tau + *ph).sin();
            let angle = k_tau * turned_bound + 4.0 * U * (k_tau * turned.v().abs() + ph.abs());
            let term = gain.map_or(s, |g| g * s);
            let g = gain.map_or(1.0, |g| g.v());
            sum = sum + term;
            magnitude += term.v().abs();
            sum_bound +=
                g * (angle + 2.0 * U) + s.v().abs() * gain_bound + 2.0 * U * term.v().abs();
        }
        sum_bound += 2.0 * lines.len() as f64 * U * magnitude;
        let x = a * sum;
        let x_bound = a.v() * sum_bound + sum.v().abs() * a_bound + 2.0 * U * x.v().abs();
        out.x[k] = share * x;
        out.bound[k] =
            share.v() * x_bound + (x.v().abs() + x_bound) * share_bound + U * out.x[k].v().abs();
        out.sampling[k] = ((f64::from(u8::from(heard)) - share.v()).abs() + share_bound)
            * (x.v().abs() + x_bound);
    }
    out
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

/// The frame's transform of weighted samples, real and imaginary parts.
fn transform(frame: &Frame, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = frame.n();
    let mut re = vec![0.0; n];
    for (k, (w, x)) in frame.weights().iter().zip(x).enumerate() {
        re[k] = w * x;
    }
    let mut im = vec![0.0; n];
    fft::fft_with(&mut re, &mut im, &fft::twiddles(n));
    (re, im)
}

/// Sampling as `MODEL` derives it, `|X|` under `sqrt(power + bound)`.
fn transformed(frame: &Frame, window: &Window<f64>) -> Bins {
    let Window { x, bound, sampling } = window;
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
    let bound: Vec<f64> = power
        .iter()
        .map(|p| 5.0 * U * p + (2.0 * (2.0 * p * (1.0 + 5.0 * U)).sqrt() + err) * err)
        .collect();
    let eps = w
        .iter()
        .zip(sampling)
        .map(|(w, s)| w.abs() * s)
        .sum::<f64>()
        * (1.0 + 4.0 * U * w.len() as f64);
    let sampling = power
        .iter()
        .zip(&bound)
        .map(|(p, b)| eps * (2.0 * (p + b).max(0.0).sqrt() + eps) * (1.0 + 4.0 * U))
        .collect();
    Bins {
        power,
        bound,
        sampling,
    }
}

/// A noise's weighted, gated envelope over a frame, beside its bound and `edge`, `eps`'s sum
/// before `sqrt(R_lines(0))`.
struct Envelope<N> {
    y: Vec<N>,
    bound: Vec<f64>,
    edge: f64,
}

fn envelope<N: Num>(atom: &Atom<N>, clip: &Clip, frame: &Frame) -> Option<Envelope<N>> {
    let Sound::Noise { lo, hi } = atom.sound else {
        unreachable!("a noise")
    };
    let (level, _) = tables(atom)?;
    if hi.v() <= lo.v() {
        return None;
    }
    let width = (hi - lo).sqrt();
    let (grid, rate) = (Grid::of(clip.rate), f64::from(clip.rate));
    let steep = steepest(&atom.level, atom.on.v());
    let (on, end) = (atom.on, atom.end());
    let mut hint = 0;
    let count = frame.weights().len();
    let mut out = Envelope {
        y: vec![N::ZERO; count],
        bound: vec![0.0; count],
        edge: 0.0,
    };
    for (k, w) in frame.weights().iter().enumerate() {
        let s = frame.first() + k as i64;
        if s < 0 || s >= clip.len as i64 {
            continue;
        }
        let (share, share_bound) = gate(s, rate, on, end);
        let heard = inside(&grid, s, (on.v(), end.v()));
        if share.v() == 0.0 && !heard {
            continue;
        }
        let (a, a_bound) = amplitude(&level, steep, grid.instant(s), &mut hint);
        let scaled = a / width;
        let scaled_bound = a_bound / width.v() + 4.0 * U * scaled.v().abs();
        out.y[k] = share * scaled * *w;
        out.bound[k] = w.abs()
            * (share.v() * scaled_bound + (scaled.v().abs() + scaled_bound) * share_bound)
            + 2.0 * U * out.y[k].v().abs();
        out.edge += ((f64::from(u8::from(heard)) - share.v()).abs() + share_bound)
            * w.abs()
            * (scaled.v().abs() + scaled_bound);
    }
    Some(out)
}

/// A noise's expected power per bin, `sum_d R(d) C(d) e^(-2 pi i j d/n)`, `R` the band's
/// autocorrelation and `C` the weighted envelope's; its sampling as `MODEL` derives it.
fn expected(atom: &Atom, clip: &Clip, frame: &Frame, correlation: &Correlation) -> Bins {
    let n = frame.n();
    let bins = frame.hz().len();
    let silent = Bins::silent(bins);
    if correlation.values.is_empty() {
        return silent;
    }
    let Some(Envelope {
        y: held,
        bound: held_bound,
        edge,
    }) = envelope(atom, clip, frame)
    else {
        return silent;
    };
    let eps = edge * correlation.lines.sqrt() * (1.0 + 4.0 * U);
    if held.iter().all(|v| *v == 0.0) {
        return Bins {
            sampling: vec![eps * eps; bins],
            ..silent
        };
    }
    let count = held.len();
    let (mut y, mut y_bound) = (vec![0.0; 2 * n], vec![0.0; 2 * n]);
    y[..count].copy_from_slice(&held);
    y_bound[..count].copy_from_slice(&held_bound);

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

    let Correlation {
        values: correlation,
        bounds: correlation_bound,
        gaps,
        ..
    } = correlation;
    let mut folded = vec![0.0; n];
    for (i, g) in folded.iter_mut().enumerate() {
        if i < count {
            *g += correlation[i] * window[i];
        }
        if i > 0 && n - i < count {
            *g += correlation[n - i] * window[n - i];
        }
    }
    let gap = (0..count)
        .map(|d| match d {
            0 => 1.0,
            _ => 2.0,
        } * gaps[d] * (window[d].abs() + window_bound))
        .sum::<f64>()
        * (1.0 + 4.0 * U * count as f64);
    let r_most = correlation.iter().fold(0.0, |m: f64, r| m.max(r.abs()));
    let r_bound = correlation_bound.iter().fold(0.0, |m: f64, r| m.max(*r));
    let window_norm = norm(window[..count].iter().copied());
    let folded_norm = norm(folded.iter().copied());
    let folded_bound = SQRT_2 * ((r_most + r_bound) * window_bound + r_bound * window_norm)
        + 3.0 * U * SQRT_2 * folded_norm;
    let mut im = vec![0.0; n];
    fft::fft_with(&mut folded, &mut im, &fft::twiddles(n));
    let err = root * folded_bound + fft_bound(n) * root * (folded_norm + folded_bound);
    let power = folded[frame.bins()].to_vec();
    let sampling = power
        .iter()
        .map(|p| gap + eps * (2.0 * (p + err + gap).max(0.0).sqrt() + eps) * (1.0 + 4.0 * U))
        .collect();
    Bins {
        power,
        bound: vec![err; bins],
        sampling,
    }
}

/// `d E|X_j|^2` along a seeded noise's one direction: `C`'s from the envelope's transform,
/// `d|Y|^2 = 2 Re(conj(Y) dY)`, then `sum_d (dR C + R dC) e^(-2 pi i j d/n)`.
fn expected_tangent(atom: &Atom<Dual>, clip: &Clip, frame: &Frame) -> Vec<f64> {
    let n = frame.n();
    let bins = frame.hz().len();
    let Sound::Noise { lo, hi } = atom.sound else {
        unreachable!("a noise")
    };
    let top = hi.min(Dual::of(PSYCHOACOUSTIC_V1.ceiling(clip.rate)));
    let Some(Envelope { y, .. }) = envelope(atom, clip, frame).filter(|_| top.v() > lo.v()) else {
        return vec![0.0; bins];
    };
    let wide = 2 * n;
    let twiddles = fft::twiddles(wide);
    let transformed = |part: &dyn Fn(&Dual) -> f64| {
        let mut re = vec![0.0; wide];
        for (r, v) in re.iter_mut().zip(&y) {
            *r = part(v);
        }
        let mut im = vec![0.0; wide];
        fft::fft_with(&mut re, &mut im, &twiddles);
        (re, im)
    };
    let ((re, im), (d_re, d_im)) = (transformed(&|v| v.value), transformed(&|v| v.tangent));
    let correlate = |power: Vec<f64>| {
        let (mut power, mut im) = (power, vec![0.0; wide]);
        fft::ifft_with(&mut power, &mut im, &twiddles);
        power
    };
    let window = correlate(re.iter().zip(&im).map(|(r, i)| r * r + i * i).collect());
    let d_window = correlate(
        (0..wide)
            .map(|j| 2.0 * (re[j] * d_re[j] + im[j] * d_im[j]))
            .collect(),
    );
    let rate = f64::from(clip.rate);
    let count = y.len();
    let mut folded = vec![0.0; n];
    for (i, g) in folded.iter_mut().enumerate() {
        for d in [i, n - i] {
            if d < count && (d == i || i > 0) {
                let r = band_correlation(lo, top, d as f64 / rate);
                *g += r.tangent * window[d] + r.value * d_window[d];
            }
        }
    }
    let mut im = vec![0.0; n];
    fft::fft_with(&mut folded, &mut im, &fft::twiddles(n));
    folded[frame.bins()].to_vec()
}

/// `(1/2) int_lo^top cos(2 pi f tau) df`, as `cos(pi (top + lo) tau) sin(pi (top - lo) tau) /
/// (2 pi tau)`, which has no difference of near sines.
fn band_correlation<N: Num>(lo: N, top: N, tau: f64) -> N {
    if tau == 0.0 {
        return (top - lo) * 0.5;
    }
    let turn = N::of(PI * tau);
    (turn * (top + lo)).cos() * (turn * (top - lo)).sin() / (TAU * tau)
}

/// A band's autocorrelation at lags `0..n` samples, each beside its rounding and its `gap`
/// from the render's lines'; `lines` bounds the lines' power. Empty where the band is.
struct Correlation {
    values: Vec<f64>,
    bounds: Vec<f64>,
    gaps: Vec<f64>,
    lines: f64,
}

/// Each lag's gap as `MODEL` derives it, the smaller where two hold.
fn correlation(clip: &Clip, (lo, hi): (f64, f64), n: usize) -> Correlation {
    let top = hi.min(PSYCHOACOUSTIC_V1.ceiling(clip.rate));
    if top <= lo {
        return Correlation {
            values: Vec::new(),
            bounds: Vec::new(),
            gaps: Vec::new(),
            lines: 0.0,
        };
    }
    let period = clip.secs();
    let h = 0.5 / period;
    let half = (top - lo) / 2.0;
    let every = h * (1.0 + 2.0 * period * (top - lo));
    let rate = f64::from(clip.rate);
    let mut out = Correlation {
        values: Vec::with_capacity(n),
        bounds: Vec::with_capacity(n),
        gaps: Vec::with_capacity(n),
        lines: (half + h) * (1.0 + 4.0 * U),
    };
    for d in 0..n {
        let tau = d as f64 / rate;
        let theta = TAU * tau / period;
        let gap = match () {
            () if d == 0 => h,
            () if theta < TAU => {
                let x = theta / 2.0;
                let (inverse, over) = (1.0 / libm::sin(x), 1.0 / x);
                (h * (1.0 + inverse - over + 4.0 * U * (inverse + over))).min(every)
            }
            () => every,
        };
        let turning = PI * (top + lo) * tau;
        out.values.push(band_correlation(lo, top, tau));
        out.bounds.push(half * U * (5.0 * turning + 14.0));
        out.gaps.push(gap * (1.0 + 4.0 * U));
    }
    out
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
    let ratio = |held: &[f64], c: f64, m: i64| {
        let gain = libm::pow(10.0, c / 10.0);
        let scaled: Vec<f64> = held.iter().map(|p| p * gain).collect();
        loudest(ear.excite(m, &scaled))
    };
    let powers = |m: i64| {
        let held = shifted(
            &composed(ear, m, &mut |frame| model.bins(frame)),
            side,
            false,
        );
        (loudest(ear.excite(m, &held)), held)
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
