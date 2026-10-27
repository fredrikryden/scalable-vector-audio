// Concern: the coarse model: each frame's excitation per channel in closed form of atom numbers, summed in power, and its tangents | Non-concern: the clip's reading | IO: (atoms, frame) -> E, dE

use std::collections::HashMap;
use std::f64::consts::{LN_10, PI};
use std::sync::Mutex;

use sva_engine::hearing::{self, P_FLOOR, WINDOW_SPLITS_HZ};
use sva_engine::internal::{Ear, range_of};
use sva_engine::{PSYCHOACOUSTIC_V1, default_threads, each};
use sva_formula::Num;

use super::atoms::{Atom, Clip, NEPER_PER_DB, Sound, Trajectory};
use super::dual::Dual;

pub const MODEL: &str = "the coarse model, the cost's own definition at every level `--coarsest` \
     runs, not a reading of the render and so no approximation of one: each frame's excitation \
     per channel is defined from the atoms' numbers alone, no samples and no transforms, as the \
     excitation of their power spectrum: atoms and a harmonic's members sum in power, with no \
     beat between any two (two partials 3 Hz apart define a steady E where their render beats), \
     no window's leakage and no line's lobe, the parts of the render's reading the coarse ear's \
     blur holds and this form leaves out by definition, so a render and this model may differ by \
     any amount there, which the level's clip-coarse.wav and fit-coarse.wav sound side by side; a \
     run not asked coarse reads the render (spectrum.rs); a partial or a harmonic's member k is a \
     line at k times its Hz at the frame's centre (its table held before on and from its last \
     knot), above 0 and under the audible ceiling (else silent, as the engine renders no line \
     past it), of power g_k^2/2 times its envelope's power e^(2 NEPER L(t)), L its pwl level, \
     averaged over [on, end) and the clip against the window of the range holding its Hz, \
     cos^4(pi (t - t_c)/T) over its integral 3T/8, in closed form per straight piece of L; a \
     noise is flat at that mean over 2 (hi - lo) per Hz on [lo, min(hi, ceiling)), each range's \
     share at its own window's mean; each line puts G(f) times full scale times its power at the \
     eardrum, and a band G(f) times full scale times its density per Hz; E = sum over lines of \
     that power times roex(p/s) at the line's Hz, plus each band's integral of its density \
     against roex(p/s) G, p as the ear's, the lower skirt's p at X, X the level per ERB at the \
     eardrum through the level-free roex centred there: at a line its own Hz, for a band's lower \
     skirt the channel's centre; every integral over Hz in closed form, as G is one exponential \
     between the ear tables' rows and roex(u) = (1 + u) e^-u with u linear in f either side of \
     its centre, int (1 + u) e^(l) over a piece Delta e^(l_max) (q_max phi_1(y) + (q_other - \
     q_max) phi_2(y)), phi_n(y) = int_0^1 x^(n-1) e^(-y x) dx, by its series under y = 1 and its \
     closed form above; tangents by dual numbers through the same closed forms: each atom's own \
     parts read along the seed, and every other part linearised at the state by its partials \
     (each band's edges by the fundamental theorem of calculus, its frequency and level slopes by \
     duals), exact to first order";

/// The ranges' count, each half-open between its splits.
const RANGES: usize = WINDOW_SPLITS_HZ.len() + 1;

/// Terms of `phi`'s series under `y = 1`: `1/20!` is under `2^-61`, below the last bit of
/// `phi_1 >= 1 - 1/e` and of `phi_2 >= 1 - 2/e` and of their slopes.
const SERIES_TERMS: usize = 20;

fn roex<N: Num>(u: N) -> N {
    (u + 1.0) * (-u).exp()
}

/// The level-free roex centred on `pt` read at `f`, beside its slopes in `pt` and `f` where
/// `tangents`.
fn around(pt: f64, f: f64, tangents: bool) -> (f64, f64, f64) {
    let (apart, side) = ((f - pt).abs(), if f >= pt { 1.0 } else { -1.0 });
    let width = erb(pt);
    let u = 4.0 * apart / width;
    let e = sva_formula::exp(-u);
    let value = (1.0 + u) * e;
    if !tangents {
        return (value, 0.0, 0.0);
    }
    let du = -u * e;
    let by_f = du * 4.0 * side / width;
    let by_pt = du
        * (-4.0 * side / width
            - 4.0 * apart * (hearing::C1 * hearing::C2 / 1000.0) / (width * width));
    (value, by_pt, by_f)
}

fn erb<N: Num>(hz: N) -> N {
    (hz * (hearing::C2 / 1000.0) + 1.0) * hearing::C1
}

/// `x` moved from its value along its tangent by each partial: the state linearised at a point,
/// its value exact.
fn jet<N: Num>(value: f64, partials: &[(N, f64)]) -> N {
    partials
        .iter()
        .filter(|(x, d)| !x.is_constant() && *d != 0.0)
        .fold(N::of(value), |sum, (x, d)| sum + (*x - x.v()) * *d)
}

/// `(int_0^1 e^(-y x) dx, int_0^1 x e^(-y x) dx)` at `y >= 0`, `e` its `e^-y`.
fn phi<N: Num>(y: N, e: N) -> (N, N) {
    if y.v() < 1.0 {
        let (mut one, mut two) = (N::ZERO, N::ZERO);
        let mut term = N::ONE;
        for k in 0..SERIES_TERMS {
            one = one + term / (k + 1) as f64;
            two = two + term / (k + 2) as f64;
            term = term * -y / (k + 1) as f64;
        }
        return (one, two);
    }
    ((N::ONE - e) / y, (N::ONE - e * (y + 1.0)) / (y * y))
}

/// `int_a^b cos^4(pi (t - tc)/T) e^(l(t)) dt`, `l` straight from `la` at `a` to `lb` at `b`.
fn windowed<N: Num>((a, b): (N, N), (la, lb): (N, N), tc: f64, t: f64) -> N {
    let d = b - a;
    let (top, y) = match la.v() >= lb.v() {
        true => (la, la - lb),
        false => (lb, lb - la),
    };
    let (ea, eb) = ((la - top).exp(), (lb - top).exp());
    let mut sum = d * phi(y, (-y).exp()).0 * (3.0 / 8.0);
    let beta = (lb - la) / d;
    for (k, weight) in [(1.0, 0.5), (2.0, 0.125)] {
        let w = 2.0 * PI * k / t;
        let (sa, ca) = ((a - tc) * w).sincos();
        let (sb, cb) = ((b - tc) * w).sincos();
        let ends = (beta * cb + sb * w) * eb - (beta * ca + sa * w) * ea;
        sum = sum + ends / (beta * beta + w * w) * weight;
    }
    top.exp() * sum
}

/// A table of times and values from `on`, held before its first and from its last.
fn table<N: Num>(t: &Trajectory<N>, on: N) -> Vec<(N, N)> {
    std::iter::once((on, t.base))
        .chain(t.knots.iter().map(|(at, d)| (*at, t.base + *d)))
        .collect()
}

fn held_at<N: Num>(table: &[(N, N)], t: f64) -> N {
    let first = table[0];
    if t < first.0.v() {
        return first.1;
    }
    for w in table.windows(2) {
        let ((t0, v0), (t1, v1)) = (w[0], w[1]);
        if t < t1.v() {
            return match t1.v() > t0.v() {
                true => v0 + (v1 - v0) * ((N::of(t) - t0) / (t1 - t0)),
                false => v1,
            };
        }
    }
    table[table.len() - 1].1
}

/// Which lines and bands move: every one where `whole`, else each holding a number not
/// constant.
fn moving<N: Num>(lines: &[Line<N>], bands: &[(&Band<N>, &Fixed)], whole: bool) -> Moving {
    let lines = lines
        .iter()
        .map(|l| whole || !l.hz.is_constant() || !l.power.is_constant())
        .collect();
    let bands = bands
        .iter()
        .map(|(b, _)| {
            whole
                || !b.lo.is_constant()
                || !b.top.is_constant()
                || b.density.iter().any(|d| !d.is_constant())
        })
        .collect();
    (lines, bands)
}

/// Whether each line and each band moves.
type Moving = (Vec<bool>, Vec<bool>);

/// A line: its Hz at the frame's centre and its mean power over its range's window.
#[derive(Clone, Copy, Debug)]
struct Line<N> {
    hz: N,
    power: N,
}

/// A flat band `[lo, top)`, its density per Hz over each range's window.
#[derive(Clone, Debug)]
struct Band<N> {
    lo: N,
    top: N,
    density: [N; RANGES],
}

/// An atom's lines and bands at one frame.
#[derive(Clone, Debug)]
struct Parts<N> {
    lines: Vec<Line<N>>,
    bands: Vec<Band<N>>,
}

struct Channel {
    hz: f64,
    p51: f64,
    slope: f64,
    /// `p/(s fc)` above the centre.
    upper: f64,
}

/// One gain row: its Hz, `G` in dB there and its slope to the next row.
type Row = (f64, f64, f64);

/// The ear's frames and channels as the closed form reads them.
pub struct Coarse {
    hop: f64,
    secs: f64,
    ceiling: f64,
    scale: f64,
    full_scale: f64,
    windows: Vec<f64>,
    edges: [f64; RANGES + 1],
    rows: Vec<Row>,
    channels: Vec<Channel>,
}

/// A value beside its partials in a band's edges.
#[derive(Clone, Copy, Debug, Default)]
struct Edged {
    v: f64,
    lo: f64,
    top: f64,
}

/// A value beside its partials in one moving number and a band's edges.
#[derive(Clone, Copy, Debug, Default)]
struct Moved {
    v: f64,
    by: f64,
    lo: f64,
    top: f64,
}

/// A band's integrals no frame moves, per channel and range: through the level-free roex at the
/// centre, and the channel's upper skirt.
struct Fixed {
    around: Vec<[Edged; RANGES]>,
    upper: Vec<[Edged; RANGES]>,
    /// This band's integral through the level-free roex centred at each line's Hz read so far,
    /// by that Hz's bits: the band's edges are this one's.
    near: Mutex<HashMap<u64, [Moved; RANGES]>>,
}

/// `X` at each line and centre, whether each centre hears anything, and the sums `X` reads.
type Levels<N> = (Vec<N>, Vec<N>, Vec<bool>, (Vec<f64>, Vec<f64>));

/// One frame's partials at the state, so a seed reads only multiply-adds.
struct Prepared {
    gain: Vec<(f64, f64)>,
    pairs: Vec<Vec<(f64, f64, f64)>>,
    near: Vec<Vec<[Moved; RANGES]>>,
    centred: Vec<Vec<(f64, f64)>>,
    weights: Vec<Vec<(f64, f64, f64)>>,
    lower: Vec<Vec<[Moved; RANGES]>>,
    heard: Vec<bool>,
    /// The level-free sums `X` reads at each line and centre.
    sums: (Vec<f64>, Vec<f64>),
    /// E per channel at the state, `dE_c/dX_b` through each line's lower skirt, and `dE_c/dk_c`
    /// through the bands' lower skirts.
    excitation: Vec<f64>,
    slopes: Vec<Vec<f64>>,
    pulled: Vec<f64>,
}

impl Coarse {
    pub fn of(ear: &Ear, clip: &Clip) -> Coarse {
        let scale = ear.scale();
        let rows: Vec<f64> = hearing::ear_gain_rows();
        let rows = rows
            .iter()
            .enumerate()
            .map(|(i, hz)| {
                let db = hearing::ear_gain_db(*hz);
                let slope = rows
                    .get(i + 1)
                    .map_or(0.0, |next| (hearing::ear_gain_db(*next) - db) / (next - hz));
                (*hz, db, slope)
            })
            .collect();
        let mut edges = [f64::INFINITY; RANGES + 1];
        edges[0] = 0.0;
        edges[1..RANGES].copy_from_slice(&WINDOW_SPLITS_HZ);
        Coarse {
            hop: ear.hop_secs(),
            secs: clip.secs(),
            ceiling: PSYCHOACOUSTIC_V1.ceiling(clip.rate),
            scale,
            full_scale: ear.full_scale(),
            windows: ear.window_secs(),
            edges,
            rows,
            channels: ear
                .centres()
                .into_iter()
                .map(|hz| {
                    let p51 = hearing::p51(hz);
                    Channel {
                        hz,
                        p51,
                        slope: hearing::lower_slope(p51),
                        upper: p51.max(P_FLOOR) / (scale * hz),
                    }
                })
                .collect(),
        }
    }

    /// The gain row line holding `hz`.
    fn row(&self, hz: f64) -> Row {
        let i = self.rows.partition_point(|r| r.0 <= hz).max(1) - 1;
        let (at, db, slope) = self.rows[i];
        match i + 1 == self.rows.len() {
            true => (at, db, 0.0),
            false => (at, db, slope),
        }
    }

    /// `G` in dB at `hz` on the row line holding `at`.
    fn gain_db<N: Num>(&self, hz: N, at: f64) -> N {
        let (row, db, slope) = self.row(at);
        (hz - row) * slope + db
    }

    /// `G roex(k |f - c|)` at `hz`, `G` without full scale.
    fn skirt_at(&self, hz: f64, c: f64, k: f64) -> f64 {
        let g = libm::pow(10.0, self.gain_db(hz, hz) / 10.0);
        g * roex(k * (hz - c).abs())
    }

    /// `int_a^b G(f) roex(k |f - c|) df` over the pieces the rows and `c` cut, `G` without full
    /// scale.
    fn skirt<N: Num>(&self, (a, b): (N, N), c: N, k: N) -> N {
        if b.v() <= a.v() {
            return N::ZERO;
        }
        let mut inner: Vec<N> = self
            .rows
            .iter()
            .map(|r| r.0)
            .filter(|r| *r > a.v() && *r < b.v())
            .map(N::of)
            .collect();
        if c.v() > a.v() && c.v() < b.v() {
            inner.push(c);
        }
        inner.sort_by(|x, y| x.v().total_cmp(&y.v()));
        let mut cuts = vec![a];
        cuts.extend(inner);
        cuts.push(b);
        let ends: Vec<(N, N, N)> = cuts
            .iter()
            .map(|x| {
                let u = (*x - c).abs() * k;
                let l = self.gain_db(*x, x.v()) * (LN_10 / 10.0) - u;
                (u + 1.0, l, l.exp())
            })
            .collect();
        cuts.windows(2)
            .zip(ends.windows(2))
            .map(|(x, ends)| {
                let ((q0, l0, e0), (q1, l1, e1)) = (ends[0], ends[1]);
                let (q_top, q_other, y, top, other) = match l0.v() >= l1.v() {
                    true => (q0, q1, l0 - l1, e0, e1),
                    false => (q1, q0, l1 - l0, e1, e0),
                };
                if top.v() == 0.0 {
                    return N::ZERO;
                }
                let (one, two) = phi(y, other / top);
                (x[1] - x[0]) * top * (q_top * one + (q_other - q_top) * two)
            })
            .fold(N::ZERO, |s, p| s + p)
    }

    /// Range `r`'s share of `[lo, top)`, capped by `within`.
    fn share(&self, r: usize, (lo, top): (f64, f64), (from, to): (f64, f64)) -> (f64, f64) {
        (
            lo.max(self.edges[r]).max(from),
            top.min(self.edges[r + 1]).min(to),
        )
    }

    /// `int G roex(k |f - c|)` over range `r`'s share of `[lo, top)` within `(from, to)`, beside
    /// its partials in `lo` and `top`.
    fn edged(&self, r: usize, band: (f64, f64), within: (f64, f64), (c, k): (f64, f64)) -> Edged {
        let (a, b) = self.share(r, band, within);
        if b <= a {
            return Edged::default();
        }
        Edged {
            v: self.skirt((a, b), c, k),
            lo: match a == band.0 {
                true => -self.skirt_at(a, c, k),
                false => 0.0,
            },
            top: match b == band.1 {
                true => self.skirt_at(b, c, k),
                false => 0.0,
            },
        }
    }

    fn fixed(&self, (lo, top): (f64, f64)) -> Fixed {
        let all = (0.0, f64::INFINITY);
        let each = |within: &dyn Fn(&Channel) -> (f64, f64), k: &dyn Fn(&Channel) -> f64| {
            self.channels
                .iter()
                .map(|ch| {
                    std::array::from_fn(|r| self.edged(r, (lo, top), within(ch), (ch.hz, k(ch))))
                })
                .collect()
        };
        Fixed {
            around: each(&|_| all, &|ch| 4.0 / erb(ch.hz)),
            upper: each(&|ch| (ch.hz, f64::INFINITY), &|ch| ch.upper),
            near: Mutex::default(),
        }
    }

    /// The mean of `e^(2 NEPER L)` over `[on, end)` and the clip against the window of `t`
    /// seconds centred at `tc`.
    fn mean<N: Num>(&self, level: &[(N, N)], tc: f64, t: f64) -> N {
        let (from, to) = ((tc - t / 2.0).max(0.0), (tc + t / 2.0).min(self.secs));
        let mut sum = N::ZERO;
        if to <= from {
            return sum;
        }
        for w in level.windows(2) {
            let ((t0, l0), (t1, l1)) = (w[0], w[1]);
            if t1.v() <= t0.v() {
                continue;
            }
            let (a, b) = (t0.max(N::of(from)), t1.min(N::of(to)));
            if b.v() <= a.v() {
                continue;
            }
            let at = |x: N| (l0 + (l1 - l0) * ((x - t0) / (t1 - t0))) * (2.0 * NEPER_PER_DB);
            sum = sum + windowed((a, b), (at(a), at(b)), tc, t);
        }
        sum / (3.0 * t / 8.0)
    }

    /// `atom`'s lines and bands at frame `m`, as `MODEL` states.
    fn parts<N: Num>(&self, atom: &Atom<N>, m: i64) -> Parts<N> {
        let tc = m as f64 * self.hop;
        let level = table(&atom.level, atom.on);
        let mut out = Parts {
            lines: Vec::new(),
            bands: Vec::new(),
        };
        match &atom.sound {
            Sound::Noise { lo, hi } => {
                let top = hi.min(N::of(self.ceiling));
                if top.v() <= lo.v() {
                    return out;
                }
                let width = (*hi - *lo) * 2.0;
                let density = std::array::from_fn(|r| {
                    let (a, b) = self.share(r, (lo.v(), top.v()), (0.0, f64::INFINITY));
                    match b > a {
                        true => self.mean(&level, tc, self.windows[r]) / width,
                        false => N::ZERO,
                    }
                });
                if density.iter().any(|d: &N| d.v() > 0.0) {
                    out.bands.push(Band {
                        lo: *lo,
                        top,
                        density,
                    });
                }
            }
            Sound::Partial { hz } | Sound::Harmonic { hz, .. } => {
                let f = held_at(&table(hz, atom.on), tc);
                let gains: Vec<N> = match &atom.sound {
                    Sound::Harmonic { offsets, .. } => {
                        let sum = offsets.iter().fold(N::ZERO, |s, o| s + *o);
                        offsets
                            .iter()
                            .copied()
                            .chain(std::iter::once(-sum))
                            .map(|o| (o * (2.0 * NEPER_PER_DB)).exp())
                            .collect()
                    }
                    _ => vec![N::ONE],
                };
                let mut means: [Option<N>; RANGES] = [None; RANGES];
                for (k, g) in gains.iter().enumerate() {
                    let hz = f * (k + 1) as f64;
                    if hz.v() <= 0.0 || hz.v() >= self.ceiling {
                        continue;
                    }
                    let r = range_of(hz.v());
                    let mean =
                        *means[r].get_or_insert_with(|| self.mean(&level, tc, self.windows[r]));
                    let power = mean * *g * 0.5;
                    if power.v() > 0.0 {
                        out.lines.push(Line { hz, power });
                    }
                }
            }
        }
        out
    }

    /// `X` at every line and channel centre, beside whether each centre hears anything: summed
    /// whole where `whole`, else `prepared`'s sums moved by the terms that move.
    fn levels<N: Num>(
        &self,
        prepared: &Prepared,
        (lines, eardrum): (&[Line<N>], &[N]),
        bands: &[(&Band<N>, &Fixed)],
        whole: bool,
    ) -> Levels<N> {
        let to_db = 10.0 / LN_10;
        let (line_moves, band_moves) = moving(lines, bands, whole);
        let start = |sums: &[f64], at: usize| match whole {
            true => N::ZERO,
            false => N::of(sums[at]),
        };
        let less = |term: N| match whole {
            true => term,
            false => term - term.v(),
        };
        let at_lines: Vec<N> = lines
            .iter()
            .enumerate()
            .map(|(b, line)| {
                let mut s = start(&prepared.sums.0, b);
                let all = line_moves[b];
                if whole {
                    let row = &prepared.pairs[b];
                    s = N::of(
                        eardrum
                            .iter()
                            .zip(row)
                            .map(|(e, r)| e.v() * r.0)
                            .sum::<f64>(),
                    );
                }
                for (o, (other, e)) in lines.iter().zip(eardrum).enumerate() {
                    if !whole && (all || line_moves[o]) {
                        let (r, d_at, d_other) = prepared.pairs[b][o];
                        s = s + less(*e * jet(r, &[(line.hz, d_at), (other.hz, d_other)]));
                    }
                }
                for (k, (band, _)) in bands.iter().enumerate() {
                    if !(all || band_moves[k]) {
                        continue;
                    }
                    for (r, j) in prepared.near[b][k].iter().enumerate() {
                        let moved =
                            jet(j.v, &[(line.hz, j.by), (band.lo, j.lo), (band.top, j.top)]);
                        s = s + less(band.density[r] * moved * self.full_scale);
                    }
                }
                s
            })
            .collect();
        let at_centres: Vec<N> = (0..self.channels.len())
            .map(|c| {
                let mut s = start(&prepared.sums.1, c);
                if whole {
                    let row = &prepared.centred[c];
                    s = N::of(
                        eardrum
                            .iter()
                            .zip(row)
                            .map(|(e, r)| e.v() * r.0)
                            .sum::<f64>(),
                    );
                }
                for (b, (line, e)) in lines.iter().zip(eardrum).enumerate() {
                    if !whole && line_moves[b] {
                        let (r, d) = prepared.centred[c][b];
                        s = s + less(*e * jet(r, &[(line.hz, d)]));
                    }
                }
                for (k, (band, fixed)) in bands.iter().enumerate() {
                    if !band_moves[k] {
                        continue;
                    }
                    for (r, j) in fixed.around[c].iter().enumerate() {
                        let moved = jet(j.v, &[(band.lo, j.lo), (band.top, j.top)]);
                        s = s + less(band.density[r] * moved * self.full_scale);
                    }
                }
                s
            })
            .collect();
        let heard: Vec<bool> = at_centres.iter().map(|s| s.v() > 0.0).collect();
        let sums = (
            at_lines.iter().map(|s| s.v()).collect(),
            at_centres.iter().map(|s| s.v()).collect(),
        );
        let at_centres = at_centres
            .into_iter()
            .map(|s| match s.v() > 0.0 {
                true => s.ln() * to_db,
                false => N::ZERO,
            })
            .collect();
        (
            at_lines.into_iter().map(|s| s.ln() * to_db).collect(),
            at_centres,
            heard,
            sums,
        )
    }

    /// Each line's power at the eardrum.
    fn eardrum<N: Num>(&self, prepared: &Prepared, lines: &[Line<N>]) -> Vec<N> {
        lines
            .iter()
            .zip(&prepared.gain)
            .map(|(l, (g, d))| jet(*g, &[(l.hz, *d)]) * l.power)
            .collect()
    }

    fn lower_k<N: Num>(&self, ch: &Channel, x: N) -> N {
        hearing::lower_p(ch.p51, ch.slope, x) / (self.scale * ch.hz)
    }

    /// The roex weight at `hz` of channel `ch`, `x` the level there, beside its slopes in `hz`
    /// and `x` where `tangents`.
    fn weighed(&self, ch: &Channel, hz: f64, x: f64, tangents: bool) -> (f64, f64, f64) {
        let (u, by_hz, by_x) = match hz < ch.hz {
            true => {
                let k = self.lower_k(ch, x);
                let by_x = hearing::lower_p_slope(ch.p51, ch.slope, x) / (self.scale * ch.hz);
                (k * (ch.hz - hz), -k, by_x * (ch.hz - hz))
            }
            false => ((hz - ch.hz) * ch.upper, ch.upper, 0.0),
        };
        let e = sva_formula::exp(-u);
        match tangents {
            true => ((1.0 + u) * e, -u * e * by_hz, -u * e * by_x),
            false => ((1.0 + u) * e, 0.0, 0.0),
        }
    }

    fn near(&self, hz: f64, band: (f64, f64)) -> [Moved; RANGES] {
        let pt = Dual::new(hz, 1.0);
        let k = Dual::of(4.0) / erb(pt);
        let k_at = 4.0 / erb(hz);
        std::array::from_fn(|r| {
            let (a, b) = self.share(r, band, (0.0, f64::INFINITY));
            if b <= a {
                return Moved::default();
            }
            let j = self.skirt((Dual::of(a), Dual::of(b)), pt, k);
            Moved {
                v: j.value,
                by: j.tangent,
                lo: match a == band.0 {
                    true => -self.skirt_at(a, hz, k_at),
                    false => 0.0,
                },
                top: match b == band.1 {
                    true => self.skirt_at(b, hz, k_at),
                    false => 0.0,
                },
            }
        })
    }

    /// The frame's partials at the state, read along each number where `tangents`.
    fn prepare(
        &self,
        lines: &[Line<f64>],
        bands: &[(&Band<f64>, &Fixed)],
        tangents: bool,
    ) -> Prepared {
        let seed = |x: f64| Dual::new(x, f64::from(u8::from(tangents)));
        let gain = lines
            .iter()
            .map(|l| {
                let (_, _, slope) = self.row(l.hz);
                let g = libm::pow(10.0, self.gain_db(l.hz, l.hz) / 10.0) * self.full_scale;
                (g, g * LN_10 / 10.0 * slope * f64::from(u8::from(tangents)))
            })
            .collect();
        let pairs = lines
            .iter()
            .map(|at| {
                lines
                    .iter()
                    .map(|o| around(at.hz, o.hz, tangents))
                    .collect()
            })
            .collect();
        let near = lines
            .iter()
            .map(|line| {
                bands
                    .iter()
                    .map(|(band, fixed)| {
                        let kept = fixed
                            .near
                            .lock()
                            .expect("the near integrals")
                            .get(&line.hz.to_bits())
                            .copied();
                        kept.unwrap_or_else(|| {
                            let read = self.near(line.hz, (band.lo, band.top));
                            fixed
                                .near
                                .lock()
                                .expect("the near integrals")
                                .insert(line.hz.to_bits(), read);
                            read
                        })
                    })
                    .collect()
            })
            .collect();
        let centred = self
            .channels
            .iter()
            .map(|ch| {
                lines
                    .iter()
                    .map(|l| {
                        let (r, _, by) = around(ch.hz, l.hz, tangents);
                        (r, by)
                    })
                    .collect()
            })
            .collect();
        let mut prepared = Prepared {
            gain,
            pairs,
            near,
            centred,
            weights: Vec::new(),
            lower: Vec::new(),
            heard: Vec::new(),
            sums: (Vec::new(), Vec::new()),
            excitation: Vec::new(),
            slopes: Vec::new(),
            pulled: Vec::new(),
        };
        let eardrum = self.eardrum(&prepared, lines);
        let (at_lines, at_centres, heard, sums) =
            self.levels(&prepared, (lines, &eardrum), bands, true);
        prepared.sums = sums;
        prepared.weights = self
            .channels
            .iter()
            .map(|ch| {
                lines
                    .iter()
                    .zip(&at_lines)
                    .map(|(l, x)| self.weighed(ch, l.hz, *x, tangents))
                    .collect()
            })
            .collect();
        prepared.lower = self
            .channels
            .iter()
            .zip(at_centres.iter().zip(&heard))
            .map(|(ch, (x, heard))| {
                let k_at = self.lower_k(ch, *x);
                let k = seed(k_at);
                bands
                    .iter()
                    .map(|(band, _)| {
                        std::array::from_fn(|r| {
                            let (a, b) = self.share(r, (band.lo, band.top), (0.0, ch.hz));
                            if b <= a || !heard {
                                return Moved::default();
                            }
                            let (v, by) = match tangents {
                                true => {
                                    let l =
                                        self.skirt((Dual::of(a), Dual::of(b)), Dual::of(ch.hz), k);
                                    (l.value, l.tangent)
                                }
                                false => (self.skirt((a, b), ch.hz, k_at), 0.0),
                            };
                            Moved {
                                v,
                                by,
                                lo: match a == band.lo {
                                    true => -self.skirt_at(a, ch.hz, k_at),
                                    false => 0.0,
                                },
                                top: match b == band.top {
                                    true => self.skirt_at(b, ch.hz, k_at),
                                    false => 0.0,
                                },
                            }
                        })
                    })
                    .collect()
            })
            .collect();
        prepared.heard = heard;
        prepared.slopes = prepared
            .weights
            .iter()
            .map(|row| row.iter().zip(&eardrum).map(|(w, e)| e * w.2).collect())
            .collect();
        prepared.pulled = prepared
            .lower
            .iter()
            .map(|per_band| {
                per_band
                    .iter()
                    .zip(bands)
                    .map(|(ranges, (band, _))| {
                        ranges
                            .iter()
                            .zip(&band.density)
                            .map(|(l, d)| d * l.by * self.full_scale)
                            .sum::<f64>()
                    })
                    .sum()
            })
            .collect();
        prepared.excitation = self.excite(&prepared, lines, bands, (true, true)).0;
        prepared
    }

    /// E per channel of `lines` and `bands`, every partial read off `prepared`, beside `X` at
    /// each line: summed whole where `whole`, else `prepared`'s excitation
    /// moved by the terms that move, every other band through its slopes in `X` and every other
    /// line too where `coupled`.
    fn excite<N: Num>(
        &self,
        prepared: &Prepared,
        lines: &[Line<N>],
        bands: &[(&Band<N>, &Fixed)],
        (whole, coupled): (bool, bool),
    ) -> (Vec<N>, Vec<N>) {
        let eardrum = self.eardrum(prepared, lines);
        let (at_lines, at_centres, _, _) = self.levels(prepared, (lines, &eardrum), bands, whole);
        let (line_moves, band_moves) = moving(lines, bands, whole);
        let less = |term: N| match whole {
            true => term,
            false => term - term.v(),
        };
        let excitation = self
            .channels
            .iter()
            .enumerate()
            .map(|(c, ch)| {
                let mut e = match whole {
                    true => N::ZERO,
                    false => N::of(prepared.excitation[c]),
                };
                for (b, (line, p)) in lines.iter().zip(&eardrum).enumerate() {
                    let (w, d_hz, d_x) = prepared.weights[c][b];
                    match line_moves[b] {
                        true => {
                            e = e + less(*p * jet(w, &[(line.hz, d_hz), (at_lines[b], d_x)]));
                        }
                        false
                            if coupled
                                && !at_lines[b].is_constant()
                                && prepared.slopes[c][b] != 0.0 =>
                        {
                            e = e + (at_lines[b] - at_lines[b].v()) * prepared.slopes[c][b];
                        }
                        false => {}
                    }
                }
                let k = match prepared.heard[c] {
                    true => self.lower_k(ch, at_centres[c]),
                    false => N::ZERO,
                };
                let mut pulled = match whole {
                    true => 0.0,
                    false => prepared.pulled[c],
                };
                for (i, (band, fixed)) in bands.iter().enumerate() {
                    if !band_moves[i] {
                        continue;
                    }
                    for r in 0..RANGES {
                        if band.density[r].v() == 0.0 && band.density[r].is_constant() {
                            continue;
                        }
                        let (u, l) = (fixed.upper[c][r], prepared.lower[c][i][r]);
                        let skirts = jet(u.v, &[(band.lo, u.lo), (band.top, u.top)])
                            + jet(l.v, &[(k, l.by), (band.lo, l.lo), (band.top, l.top)]);
                        e = e + less(band.density[r] * skirts * self.full_scale);
                        pulled -= band.density[r].v() * l.by * self.full_scale;
                    }
                }
                if !whole && !k.is_constant() && pulled != 0.0 {
                    e = e + (k - k.v()) * pulled;
                }
                e
            })
            .collect();
        (excitation, at_lines)
    }
}

/// One state as the coarse model reads it: each atom's frames and each band's fixed integrals.
pub struct Read<'a> {
    coarse: &'a Coarse,
    atoms: &'a [Atom],
    reach: &'a [(i64, i64)],
    fixed: Vec<Option<Fixed>>,
}

/// A frame's lines, and its bands beside their fixed integrals.
type Flat<'p, N> = (Vec<Line<N>>, Vec<(&'p Band<N>, &'p Fixed)>);

/// Each atom's parts at a frame, the atoms it reaches only.
struct Framed {
    atoms: Vec<usize>,
    parts: Vec<Parts<f64>>,
}

impl<'a> Read<'a> {
    pub fn of(coarse: &'a Coarse, atoms: &'a [Atom], reach: &'a [(i64, i64)]) -> Read<'a> {
        let fixed = atoms
            .iter()
            .map(|a| match a.sound {
                Sound::Noise { lo, hi } => Some(coarse.fixed((lo, hi.min(coarse.ceiling)))),
                _ => None,
            })
            .collect();
        Read {
            coarse,
            atoms,
            reach,
            fixed,
        }
    }

    fn framed(&self, m: i64) -> Framed {
        let atoms: Vec<usize> = (0..self.atoms.len())
            .filter(|a| self.reach[*a].0 <= m && m <= self.reach[*a].1)
            .collect();
        let parts = atoms
            .iter()
            .map(|a| self.coarse.parts(&self.atoms[*a], m))
            .collect();
        Framed { atoms, parts }
    }

    fn flat<'p, N: Num>(&'p self, atoms: &[usize], parts: &'p [Parts<N>]) -> Flat<'p, N> {
        let lines = parts.iter().flat_map(|p| p.lines.iter().copied()).collect();
        let bands = atoms
            .iter()
            .zip(parts)
            .flat_map(|(a, p)| {
                p.bands
                    .iter()
                    .map(move |b| (b, self.fixed[*a].as_ref().expect("a band's integrals")))
            })
            .collect();
        (lines, bands)
    }

    /// E per channel at frame `m`.
    pub fn excitation(&self, m: i64) -> Vec<f64> {
        let framed = self.framed(m);
        let (lines, bands) = self.flat(&framed.atoms, &framed.parts);
        self.coarse.prepare(&lines, &bands, false).excitation
    }

    /// E per channel at frame `m` beside its tangent along each `(atom, number)` seed whose atom
    /// reaches `m`, by dual numbers (`MODEL`); `None` for a seed whose atom does not.
    pub fn tangents(&self, m: i64, seeds: &[(usize, usize)]) -> (Vec<f64>, Vec<Option<Vec<f64>>>) {
        let framed = self.framed(m);
        let (lines, bands) = self.flat(&framed.atoms, &framed.parts);
        let prepared = self.coarse.prepare(&lines, &bands, true);
        let excitation = prepared.excitation.clone();
        let lifted: Vec<Parts<Dual>> = framed
            .parts
            .iter()
            .map(|p| Parts {
                lines: p
                    .lines
                    .iter()
                    .map(|l| Line {
                        hz: Dual::of(l.hz),
                        power: Dual::of(l.power),
                    })
                    .collect(),
                bands: p
                    .bands
                    .iter()
                    .map(|b| Band {
                        lo: Dual::of(b.lo),
                        top: Dual::of(b.top),
                        density: b.density.map(Dual::of),
                    })
                    .collect(),
            })
            .collect();
        let along = seeds
            .iter()
            .map(|(a, j)| {
                let at = framed.atoms.iter().position(|x| x == a)?;
                let mut parts = lifted.clone();
                parts[at] = self.coarse.parts(&self.atoms[*a].seeded(*j), m);
                let (lines, bands) = self.flat(&framed.atoms, &parts);
                let (moved, at_lines) =
                    self.coarse
                        .excite(&prepared, &lines, &bands, (false, false));
                let still: Vec<f64> = lines
                    .iter()
                    .zip(&at_lines)
                    .map(|(l, x)| match l.hz.is_constant() && l.power.is_constant() {
                        true => x.tangent,
                        false => 0.0,
                    })
                    .collect();
                Some(
                    moved
                        .iter()
                        .zip(&prepared.slopes)
                        .map(|(d, slopes)| {
                            let through: f64 = slopes
                                .iter()
                                .zip(&still)
                                .filter(|(w, _)| **w != 0.0)
                                .map(|(w, t)| w * t)
                                .sum();
                            d.tangent + through
                        })
                        .collect(),
                )
            })
            .collect();
        (excitation, along)
    }

    /// E per channel at each frame of `(first, last)`, frames spread over every thread.
    pub fn excitations(&self, (first, last): (i64, i64)) -> Vec<Vec<f64>> {
        each(default_threads(), (first..=last).collect(), |m| {
            self.excitation(m)
        })
    }
}
