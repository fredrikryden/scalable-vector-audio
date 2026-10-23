// Concern: a score line's short-time transform on the reading's frames off its lowering, its tangents back | Non-concern: lowering it, reading the bins | IO: (form, frames) -> bins; adjoint -> gradient

//! A tone's phase is quadratic through a frame's middle and 5 ms either side, with no a priori
//! bound: a priced solve reports the render's reading less this one as `error_bits`, and
//! chooses by the render's. An amplitude or noise gain is quadratic per `Piece`.

use std::sync::{Arc, OnceLock};

use sva_formula::closed_form::{Edge, Unary};
use sva_formula::{Body, C64, Cx, Dual, Num, Tangents};
use sva_samples::Evaluator;
use sva_samples::measure::internal::Grid;

/// A frame reads its line a quarter window either side of its middle.
const SPREAD_SECS: f64 = 0.005;

/// A frame's window and the transform's twiddles, built once a line.
struct Transform {
    w: Vec<f64>,
    twiddles: Vec<(f64, f64)>,
}

impl Transform {
    fn of(g: &Grid) -> Transform {
        Transform {
            w: sva_samples::measure::internal::hann(g.window),
            twiddles: sva_samples::fft::twiddles(g.n),
        }
    }
}

/// A sum of products of leaves.
#[derive(Clone, Debug, Default)]
pub struct SumProd(pub Vec<Vec<usize>>);

impl SumProd {
    fn value(&self, leaf: &dyn Fn(usize) -> f64) -> f64 {
        self.0
            .iter()
            .map(|p| p.iter().map(|l| leaf(*l)).product::<f64>())
            .sum()
    }

    fn back(&self, leaf: &dyn Fn(usize) -> f64, adj: f64, out: &mut dyn FnMut(usize, f64)) {
        for p in &self.0 {
            for (i, l) in p.iter().enumerate() {
                let rest: f64 = p
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, x)| leaf(*x))
                    .product();
                out(*l, adj * rest);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Carrier {
    /// `cos(phase)`, or `sin(phase)` where `sine`.
    Tone { phase: SumProd, sine: bool },
    /// The node's own noise, rendered, its filters read as `H(f)` per bin.
    Noise,
}

#[derive(Clone, Debug)]
pub struct Term {
    pub amp: SumProd,
    pub carrier: Carrier,
}

#[derive(Clone, Debug)]
pub struct Filter {
    pub seed: Option<u32>,
    held: sva_samples::Coeffs,
    /// Each coefficient's derivative in the cutoff, off the render's own design over a dual.
    slope: [f64; 5],
}

impl Filter {
    pub fn new(
        shape: sva_formula::Shape,
        cutoff: f64,
        q: f64,
        seed: Option<u32>,
        sr: f64,
    ) -> Filter {
        let (slope, _) = Tangents::default().active(|| {
            let c = sva_samples::biquad::design(
                shape,
                Dual::seeded(cutoff, &[(0, 1.0)]),
                Dual::of(q),
                Dual::of(1.0),
                sr,
            );
            let d = |x: Dual| x.tangent().first().map_or(0.0, |p| p.1);
            [d(c.b0), d(c.b1), d(c.b2), d(c.a1), d(c.a2)]
        });
        Filter {
            seed,
            held: sva_samples::biquad::design(shape, cutoff, q, 1.0, sr),
            slope,
        }
    }

    fn at(&self, hz: f64, sr: f64) -> C64 {
        sva_samples::biquad::at(&self.held, hz, sr)
    }

    /// `(H, dH/dhz, dH/dcutoff)` at `hz`: `H = N/D` in `z^-1 = e^{-iw}`, differentiated.
    fn response(&self, hz: f64, sr: f64) -> (C64, C64, C64) {
        let c = &self.held;
        let w = std::f64::consts::TAU * hz / sr;
        let z = |k: f64| {
            let (s, co) = libm::sincos(k * w);
            C64::new(co, -s)
        };
        let (z1, z2) = (z(1.0), z(2.0));
        let poly = |x0: f64, x1: f64, x2: f64| C64::new(x0, 0.0) + z1.scale(x1) + z2.scale(x2);
        let (num, den) = (poly(c.b0, c.b1, c.b2), poly(1.0, c.a1, c.a2));
        let h = num / den;
        let i = C64::new(0.0, 1.0);
        let dpoly = |x1: f64, x2: f64| (z1.scale(x1) + z2.scale(2.0 * x2)) * i.scale(-1.0);
        let dw = (dpoly(c.b1, c.b2) - h * dpoly(c.a1, c.a2)) / den;
        let d = &self.slope;
        let dcut = (poly(d[0], d[1], d[2]) - h * poly(0.0, d[3], d[4])) / den;
        (h, dw.scale(std::f64::consts::TAU / sr), dcut)
    }
}

fn gain(filters: &[Filter], hz: f64, sr: f64) -> C64 {
    filters
        .iter()
        .fold(C64::new(1.0, 0.0), |h, f| h * f.at(hz, sr))
}

/// `(H, dH/dhz, dH/dcutoff per varied cutoff)` of a chain.
fn chain(filters: &[Filter], hz: f64, sr: f64) -> (C64, C64, Vec<(u32, C64)>) {
    let parts: Vec<(C64, C64, C64)> = filters.iter().map(|f| f.response(hz, sr)).collect();
    let mut h = C64::new(1.0, 0.0);
    for p in &parts {
        h = h * p.0;
    }
    let mut dhz = C64::new(0.0, 0.0);
    let mut dcut = Vec::new();
    for (i, (f, p)) in filters.iter().zip(&parts).enumerate() {
        let mut rest = C64::new(1.0, 0.0);
        for (j, q) in parts.iter().enumerate() {
            if j != i {
                rest = rest * q.0;
            }
        }
        dhz = dhz + p.1 * rest;
        if let Some(s) = f.seed {
            dcut.push((s, p.2 * rest));
        }
    }
    (h, dhz, dcut)
}

/// One score line lowered and read at its instants; its tangents name local indices.
pub struct Line {
    pub at: f64,
    pub l: Dual,
    pub r: Dual,
    pub rise: Dual,
    pub fall: Dual,
    pub terms: Vec<Term>,
    /// `[leaf][instant]`; a leaf free of `t` holds one.
    pub leaves: Vec<Vec<Dual>>,
    /// Each frame's phase's three, `t* - d, t*, t* + d`, then its pieces'.
    pub instants: Vec<f64>,
    pub frames: Vec<Frame>,
    pub tangents: Tangents,
    /// Its noise's realisations, one a draw where its seed is drawn; none where it holds none.
    pub noise: Vec<Arc<Realisation>>,
    pub noise_filters: Vec<Filter>,
}

/// One noise at the clip's samples, and `basis` per frame: the same bits however its line is
/// cropped around it.
pub struct Realisation {
    samples: Vec<f64>,
    spectra: Vec<OnceLock<[Vec<C64>; 3]>>,
}

impl Realisation {
    pub fn of(samples: Vec<f64>, g: &Grid) -> Realisation {
        Realisation {
            samples,
            spectra: (0..g.frames).map(|_| OnceLock::new()).collect(),
        }
    }

    /// Frame `m`'s windowed spectra under each `lagrange` weight on the window's middle and a
    /// quarter window either side; `m` reads only samples inside.
    fn basis(&self, g: &Grid, tr: &Transform, m: usize) -> &[Vec<C64>; 3] {
        self.spectra[m].get_or_init(|| {
            let s0 = g.first(m) as usize;
            let half = 0.5 * g.window as f64;
            std::array::from_fn(|q| {
                let (mut re, mut im) = (vec![0.0; g.n], vec![0.0; g.n]);
                for (k, wk) in tr.w.iter().enumerate() {
                    let u = (k as f64 - half) / (0.5 * half);
                    re[g.left + k] = wk * lagrange(u)[q] * self.samples[s0 + k];
                }
                sva_samples::fft::fft_with(&mut re, &mut im, &tr.twiddles);
                (0..g.bins()).map(|j| C64::new(re[j], im[j])).collect()
            })
        })
    }
}

/// The support `[lo, hi)` in samples.
fn span(line: &Line, g: &Grid) -> (f64, f64) {
    let (l, r) = (line.l.v, line.r.v);
    ((line.at + l) * g.sr, (line.at + r) * g.sr)
}

/// The samples `[a, b)` frame `m` reads of the support, mirrored reads included.
fn reads(at: f64, l: f64, r: f64, g: &Grid, m: usize) -> Option<(f64, f64)> {
    let (lo, hi) = (
        ((at + l) * g.sr).max(0.0),
        ((at + r) * g.sr).min(g.len as f64),
    );
    if hi <= lo {
        return None;
    }
    let s0 = g.first(m) as f64;
    let s1 = s0 + g.window as f64;
    let mut reads: Vec<(f64, f64)> = Vec::new();
    let clip = |a: f64, b: f64, out: &mut Vec<(f64, f64)>| {
        let (a, b) = (a.max(lo), b.min(hi));
        if b > a {
            out.push((a, b));
        }
    };
    clip(s0, s1, &mut reads);
    if s0 < 0.0 {
        clip(0.0, -s0, &mut reads);
    }
    let end = (g.len - 1) as f64;
    if s1 > end {
        clip(2.0 * end - s1, end, &mut reads);
    }
    let a = reads.iter().map(|x| x.0).fold(f64::INFINITY, f64::min);
    let b = reads.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    (b > a).then_some((a, b))
}

/// What a frame reads between two of `sva_formula::corners`, in local seconds, read at its
/// middle and `H/2` either side (`[i, i + 3)`), `H` its half-width, at most 10 ms; the crop's
/// shoulders apply per sample. A smooth amplitude `a` is within `max|a'''| H^3 / 8`, the bound of
/// `|x (x^2 - H^2/4)| / 6` over `|x| <= H`.
#[derive(Clone, Copy, Debug)]
pub struct Piece {
    lo: f64,
    hi: f64,
    i: usize,
}

impl Piece {
    fn instants(&self) -> [f64; 3] {
        let (mid, d) = (0.5 * (self.lo + self.hi), 0.25 * (self.hi - self.lo));
        [mid - d, mid, mid + d]
    }

    fn weights(&self, t: f64) -> [f64; 3] {
        let (mid, d) = (0.5 * (self.lo + self.hi), 0.25 * (self.hi - self.lo));
        lagrange((t - mid) / d)
    }
}

/// The piece holding local `t`, else the first or last.
fn piece_at(pieces: &[Piece], t: f64) -> usize {
    pieces
        .iter()
        .position(|p| t < p.hi)
        .unwrap_or(pieces.len() - 1)
}

/// One frame a line sounds in: its phase's instants from `i0`, and its pieces.
pub struct Frame {
    m: usize,
    i0: usize,
    pieces: Vec<Piece>,
}

/// Each frame a line at `at` over `[l, r)` sounds in, and the local instants they read.
pub fn layout(at: f64, l: f64, r: f64, corners: &[f64], g: &Grid) -> (Vec<f64>, Vec<Frame>) {
    let mut corners: Vec<f64> = corners.iter().copied().filter(|c| c.is_finite()).collect();
    corners.sort_by(f64::total_cmp);
    corners.dedup();
    let (mut instants, mut frames) = (Vec::new(), Vec::new());
    for m in 0..g.frames {
        let Some((a, b)) = reads(at, l, r, g, m) else {
            continue;
        };
        let (lo, hi) = (a / g.sr - at, b / g.sr - at);
        let t = 0.5 * (a + b) / g.sr - at;
        let i0 = instants.len();
        instants.extend([t - SPREAD_SECS, t, t + SPREAD_SECS]);
        let cuts = corners.iter().copied().filter(|c| *c > lo && *c < hi);
        let ends: Vec<f64> = std::iter::once(lo).chain(cuts).chain([hi]).collect();
        let pieces = ends
            .windows(2)
            .map(|e| {
                let p = Piece {
                    lo: e[0],
                    hi: e[1],
                    i: instants.len(),
                };
                instants.extend(p.instants());
                p
            })
            .collect();
        frames.push(Frame { m, i0, pieces });
    }
    (instants, frames)
}

/// Lagrange's weights on three instants at `u`, in spacings from the middle.
fn lagrange(u: f64) -> [f64; 3] {
    [0.5 * u * (u - 1.0), 1.0 - u * u, 0.5 * u * (u + 1.0)]
}

fn through(v: &[f64; 3], l: &[f64; 3]) -> f64 {
    v[0] * l[0] + v[1] * l[1] + v[2] * l[2]
}

/// One frame's bins: its tones' `a cos(phase)` and `a sin(phase)`, so a phase offset `p`
/// reads `cos p * cos - sin p * quad`; its noise's per realisation, none where it holds none.
pub struct Bins {
    pub m: usize,
    pub cos: Vec<C64>,
    pub quad: Vec<C64>,
    pub noise: Vec<Vec<C64>>,
}

/// Each part's `gamma` per frame and bin.
pub struct Adjoint<'a> {
    pub cos: &'a dyn Fn(usize, usize) -> C64,
    pub quad: &'a dyn Fn(usize, usize) -> C64,
    /// Per realisation, frame and bin.
    pub noise: &'a dyn Fn(usize, usize, usize) -> C64,
}

/// A tone in one frame: its amplitude per piece, the phase as a cosine's, its frequency and the
/// master's gain there, conjugated where the phase falls.
struct Tone {
    amp: Vec<[f64; 3]>,
    phi: [f64; 3],
    hz: f64,
    h: C64,
}

impl Tone {
    /// The phase through its three instants, `p0 + p1 u + p2 u^2`.
    fn poly(&self) -> [f64; 3] {
        let f = &self.phi;
        [f[1], 0.5 * (f[2] - f[0]), 0.5 * (f[2] + f[0]) - f[1]]
    }

    /// The cosine's argument at `u`, the gain's phase in.
    fn phase(&self, u: f64) -> f64 {
        let p = self.poly();
        p[0] + p[1] * u + p[2] * u * u + libm::atan2(self.h.im, self.h.re)
    }
}

/// The piece holding local `t`, its weights there, and its quadratic through `values`.
fn read_at(fr: &Frame, values: &[[f64; 3]], t: f64) -> (usize, [f64; 3], f64) {
    let p = piece_at(&fr.pieces, t);
    let w = fr.pieces[p].weights(t);
    (p, w, through(&values[p], &w))
}

impl Line {
    fn leaf(&self, l: usize, instant: usize) -> f64 {
        let v = &self.leaves[l];
        match v.len() {
            1 => v[0].v,
            _ => v[instant].v,
        }
    }

    fn amp(&self, term: &Term, i: usize) -> f64 {
        term.amp.value(&|l| self.leaf(l, i))
    }

    fn shoulder(&self, t: f64) -> f64 {
        sva_samples::collapse::shoulders(t, self.l.v, self.r.v, self.rise.v, self.fall.v)
    }

    fn per_piece(&self, fr: &Frame, f: impl Fn(usize) -> f64) -> Vec<[f64; 3]> {
        fr.pieces
            .iter()
            .map(|p| std::array::from_fn(|j| f(p.i + j)))
            .collect()
    }

    fn tone(&self, term: &Term, fr: &Frame, master: &[Filter], sr: f64) -> Option<Tone> {
        let Carrier::Tone { phase, sine } = &term.carrier else {
            return None;
        };
        let lag = if *sine {
            std::f64::consts::FRAC_PI_2
        } else {
            0.0
        };
        let amp = self.per_piece(fr, |i| self.amp(term, i));
        let phi = std::array::from_fn(|i| phase.value(&|l| self.leaf(l, fr.i0 + i)) - lag);
        let hz = (phi[2] - phi[0]) / (2.0 * SPREAD_SECS * std::f64::consts::TAU);
        if hz.abs() >= 0.5 * sr {
            return None;
        }
        let h = gain(master, hz.abs(), sr);
        Some(Tone {
            amp,
            phi,
            hz,
            h: if hz < 0.0 { h.conj() } else { h },
        })
    }

    /// Each sample frame `m` reads of the support: its window point, its offset `u` and the
    /// tone's `cos` and `sin` of its phase plus the gain's, by recurrence where no end mirrors.
    fn each_sample(
        &self,
        g: &Grid,
        m: usize,
        tstar: f64,
        t: &Tone,
        mut each: impl FnMut(usize, f64, f64, f64),
    ) {
        let (lo, hi) = span(self, g);
        let (lo, hi) = (lo.max(0.0), hi.min(g.len as f64));
        let s0 = g.first(m);
        let w = g.window as i64;
        let du = 1.0 / (g.sr * SPREAD_SECS);
        let p = t.poly();
        let phase = |u: f64| t.phase(u);
        let u_of = |s: i64| ((s as f64) / g.sr - self.at - tstar) / SPREAD_SECS;
        if s0 >= 0 && s0 + w <= g.len as i64 {
            let ka = ((lo - s0 as f64).ceil().max(0.0) as i64).min(w);
            let kb = ((hi - s0 as f64).ceil().max(0.0) as i64).min(w);
            if kb <= ka {
                return;
            }
            let mut u = u_of(s0 + ka);
            let (s, c) = libm::sincos(phase(u));
            let mut z = C64::new(c, s);
            let (rs, rc) = libm::sincos(phase(u + du) - phase(u));
            let mut r = C64::new(rc, rs);
            let (qs, qc) = libm::sincos(2.0 * p[2] * du * du);
            let q = C64::new(qc, qs);
            for k in ka..kb {
                each(k as usize, u, z.re, z.im);
                z = z * r;
                r = r * q;
                u += du;
            }
            return;
        }
        for k in 0..w {
            let s = g.mirrored(s0 + k);
            if (s as f64) < lo || (s as f64) >= hi {
                continue;
            }
            let u = u_of(s);
            let (sn, c) = libm::sincos(phase(u));
            each(k as usize, u, c, sn);
        }
    }

    /// Its noise filters' and the master's gain at each bin.
    fn noise_gains(&self, g: &Grid, master: &[Filter]) -> Vec<C64> {
        (0..g.bins())
            .map(|j| {
                let hz = j as f64 * g.sr / g.n as f64;
                gain(&self.noise_filters, hz, g.sr) * gain(master, hz, g.sr)
            })
            .collect()
    }

    fn noise_terms(&self) -> impl Iterator<Item = &Term> {
        self.terms
            .iter()
            .filter(|t| matches!(t.carrier, Carrier::Noise))
    }

    /// Its noise terms' summed gain per piece, the shoulders aside.
    fn noise_env(&self, fr: &Frame) -> Vec<[f64; 3]> {
        self.per_piece(fr, |i| {
            self.noise_terms().map(|t| self.amp(t, i)).sum::<f64>()
        })
    }

    /// `read_at`, and the shoulders at `t`.
    fn enveloped(&self, fr: &Frame, values: &[[f64; 3]], t: f64) -> (usize, [f64; 3], f64, f64) {
        let (p, w, q) = read_at(fr, values, t);
        (p, w, q, self.shoulder(t))
    }

    /// `basis` where `fr`'s one piece is its window, inside the clip and clear of the shoulders.
    fn whole<'a>(
        &self,
        noise: &'a Realisation,
        g: &Grid,
        tr: &Transform,
        fr: &Frame,
    ) -> Option<&'a [Vec<C64>; 3]> {
        let s0 = g.first(fr.m);
        let end = s0 + g.window as i64;
        let inside = s0 >= 0 && end as usize <= noise.samples.len();
        let open = (self.at + self.l.v + self.rise.v) * g.sr;
        let close = (self.at + self.r.v - self.fall.v) * g.sr;
        let flat = s0 as f64 >= open && end as f64 <= close;
        (inside && flat && fr.pieces.len() == 1).then(|| noise.basis(g, tr, fr.m))
    }

    /// Frame `fr`'s windowed spectrum of `noise` under the gain `env`.
    fn noise_frame(
        &self,
        noise: &Realisation,
        g: &Grid,
        tr: &Transform,
        fr: &Frame,
        env: &[[f64; 3]],
    ) -> Vec<C64> {
        if let Some(b) = self.whole(noise, g, tr, fr) {
            let e = env[0];
            return (0..g.bins())
                .map(|j| b[0][j].scale(e[0]) + b[1][j].scale(e[1]) + b[2][j].scale(e[2]))
                .collect();
        }
        let (mut re, mut im) = (vec![0.0; g.n], vec![0.0; g.n]);
        each_noise_sample(g, fr.m, span(self, g), noise, |k, s| {
            let (_, _, q, sh) = self.enveloped(fr, env, s as f64 / g.sr - self.at);
            re[g.left + k] = tr.w[k] * q * sh * noise.samples[s];
        });
        sva_samples::fft::fft_with(&mut re, &mut im, &tr.twiddles);
        (0..g.bins()).map(|j| C64::new(re[j], im[j])).collect()
    }

    /// Each frame it sounds in, and its bins there.
    pub fn bins(&self, g: &Grid, master: &[Filter]) -> Vec<Bins> {
        let tr = Transform::of(g);
        let w = &tr.w;
        let shaped = match self.noise.is_empty() {
            true => Vec::new(),
            false => self.noise_gains(g, master),
        };
        let mut out = Vec::with_capacity(self.frames.len());
        for fr in &self.frames {
            let tstar = self.instants[fr.i0 + 1];
            let (mut re, mut im) = (vec![0.0; g.n], vec![0.0; g.n]);
            for term in &self.terms {
                let Some(t) = self.tone(term, fr, master, g.sr) else {
                    continue;
                };
                let abs = t.h.abs();
                self.each_sample(g, fr.m, tstar, &t, |i, u, c, s| {
                    let (_, _, q, sh) = self.enveloped(fr, &t.amp, tstar + u * SPREAD_SECS);
                    let a = w[i] * abs * q * sh;
                    re[g.left + i] += a * c;
                    im[g.left + i] += a * s;
                });
            }
            sva_samples::fft::fft_with(&mut re, &mut im, &tr.twiddles);
            let z = |j: usize| C64::new(re[j % g.n], im[j % g.n]);
            let half = C64::new(0.5, 0.0);
            let cos: Vec<C64> = (0..g.bins())
                .map(|j| (z(j) + z(g.n - j).conj()) * half)
                .collect();
            let quad: Vec<C64> = (0..g.bins())
                .map(|j| (z(j) - z(g.n - j).conj()) * C64::new(0.0, -0.5))
                .collect();
            let env = match self.noise.is_empty() {
                true => Vec::new(),
                false => self.noise_env(fr),
            };
            let noise = self
                .noise
                .iter()
                .map(|n| {
                    let x = self.noise_frame(n, g, &tr, fr, &env);
                    x.iter().zip(&shaped).map(|(x, h)| *x * *h).collect()
                })
                .collect();
            out.push(Bins {
                m: fr.m,
                cos,
                quad,
                noise,
            });
        }
        out
    }

    /// The gradient in local indices from each of its bins' `dL/dRe X + i dL/dIm X`, of its
    /// tones' cosine and quadrature parts and of its noise.
    pub fn back(&self, g: &Grid, master: &[Filter], adj: &Adjoint, out: &mut Vec<(u32, f64)>) {
        let gamma = adj.noise;
        let tr = Transform::of(g);
        let mut leaf_adj: Vec<Vec<f64>> = self.leaves.iter().map(|v| vec![0.0; v.len()]).collect();
        let add_leaf = |l: usize, i: usize, x: f64, adj: &mut Vec<Vec<f64>>| {
            let slot = match adj[l].len() {
                1 => 0,
                _ => i,
            };
            adj[l][slot] += x;
        };
        let (mut l_adj, mut r_adj) = (0.0, 0.0);
        // `dL/dl`, `dL/dr`, `dL/drise`, `dL/dfall` through the shoulders at each sample.
        let mut sh_adj = [0.0; 4];
        let shouldered = |t: f64, x: f64, adj: &mut [f64; 4]| {
            if x != 0.0 {
                for (a, (_, part)) in adj.iter_mut().zip(self.shoulder_slopes(t)) {
                    *a += x * part;
                }
            }
        };
        let mut cut_adj: Vec<(u32, f64)> = Vec::new();
        let (lo, hi) = span(self, g);
        for fr in &self.frames {
            let tstar = self.instants[fr.i0 + 1];
            let xc = samples_back(g, &tr, fr.m, adj.cos);
            let xs = samples_back(g, &tr, fr.m, adj.quad);
            for term in &self.terms {
                let Some(t) = self.tone(term, fr, master, g.sr) else {
                    continue;
                };
                let abs = t.h.abs();
                let mut ga = vec![[0.0; 3]; fr.pieces.len()];
                let mut gphi = [0.0; 3];
                let (mut g_abs, mut g_arg) = (0.0, 0.0);
                let mut ends = (None, None);
                self.each_sample(g, fr.m, tstar, &t, |i, u, c, s| {
                    let lw = lagrange(u);
                    let tl = tstar + u * SPREAD_SECS;
                    let (p, pw, q, sh) = self.enveloped(fr, &t.amp, tl);
                    let a = q * sh;
                    let (along, across) = (xc[i] * c + xs[i] * s, xs[i] * c - xc[i] * s);
                    for j in 0..3 {
                        ga[p][j] += along * abs * sh * pw[j];
                        gphi[j] += across * abs * a * lw[j];
                    }
                    shouldered(tl, along * abs * q, &mut sh_adj);
                    g_abs += along * a;
                    g_arg += across * abs * a;
                    if ends.0.is_none() {
                        ends.0 = Some(i);
                    }
                    ends.1 = Some(i);
                });
                let s0 = g.first(fr.m) as f64;
                let unmirrored = s0 >= 0.0 && s0 + g.window as f64 <= g.len as f64;
                if unmirrored {
                    let value = |i: usize| {
                        let tl = (s0 + i as f64) / g.sr - self.at;
                        let (sn, cs) = libm::sincos(t.phase((tl - tstar) / SPREAD_SECS));
                        let (_, _, q, sh) = self.enveloped(fr, &t.amp, tl);
                        abs * q * sh * (xc[i] * cs + xs[i] * sn)
                    };
                    if let Some(ka) = ends.0
                        && lo > s0
                        && lo.is_finite()
                    {
                        l_adj -= g.sr * value(ka);
                    }
                    if let Some(kb) = ends.1
                        && kb + 1 < g.window
                        && hi < s0 + g.window as f64
                    {
                        r_adj += g.sr * value(kb + 1);
                    }
                }
                let (_, dhz, dcut) = chain(master, t.hz.abs(), g.sr);
                let flip = |d: C64| if t.hz < 0.0 { d.conj() } else { d };
                let through_h = |d: C64| {
                    let q = t.h.conj() * d;
                    match abs > 1e-300 {
                        true => g_abs * q.re / abs + g_arg * q.im / (abs * abs),
                        false => 0.0,
                    }
                };
                let g_hz = through_h(match t.hz < 0.0 {
                    true => dhz.conj().scale(-1.0),
                    false => dhz,
                });
                let k_hz = 1.0 / (2.0 * SPREAD_SECS * std::f64::consts::TAU);
                gphi[2] += g_hz * k_hz;
                gphi[0] -= g_hz * k_hz;
                for (s, d) in dcut {
                    cut_adj.push((s, through_h(flip(d))));
                }
                let Carrier::Tone { phase, .. } = &term.carrier else {
                    unreachable!("a tone's carrier");
                };
                for (p, piece) in fr.pieces.iter().enumerate() {
                    for (j, at) in (piece.i..piece.i + 3).enumerate() {
                        term.amp.back(&|l| self.leaf(l, at), ga[p][j], &mut |l, x| {
                            add_leaf(l, at, x, &mut leaf_adj)
                        });
                    }
                }
                for (i, gp) in gphi.iter().enumerate() {
                    let at = fr.i0 + i;
                    phase.back(&|l| self.leaf(l, at), *gp, &mut |l, x| {
                        add_leaf(l, at, x, &mut leaf_adj)
                    });
                }
            }
        }
        let (chains, shaped) = match self.noise.is_empty() {
            true => (Vec::new(), Vec::new()),
            false => (
                (0..g.bins())
                    .map(|j| {
                        let hz = j as f64 * g.sr / g.n as f64;
                        let (h, _, dcut) = chain(&self.noise_filters, hz, g.sr);
                        let (hm, _, dmcut) = chain(master, hz, g.sr);
                        (h, dcut, hm, dmcut)
                    })
                    .collect::<Vec<_>>(),
                self.noise_gains(g, master),
            ),
        };
        // Each cutoff's sum, in any order, is within (n-1)u*sum|term| of exact (Higham 2002, 4.2).
        let seeds =
            |filters: &[Filter]| filters.iter().filter_map(|f| f.seed).collect::<Vec<u32>>();
        let (cut_seeds, mcut_seeds) = match self.noise.is_empty() {
            true => (Vec::new(), Vec::new()),
            false => (seeds(&self.noise_filters), seeds(master)),
        };
        let mut cut_sums = vec![0.0; cut_seeds.len()];
        let mut mcut_sums = vec![0.0; mcut_seeds.len()];
        for (r, n) in self.noise.iter().enumerate() {
            let gamma = &|m: usize, j: usize| gamma(r, m, j);
            for fr in &self.frames {
                let env = self.noise_env(fr);
                let x = self.noise_frame(n, g, &tr, fr, &env);
                for (j, x) in x.iter().enumerate() {
                    let (h, dcut, hm, dmcut) = &chains[j];
                    let cg = gamma(fr.m, j).conj();
                    for (sum, (_, d)) in cut_sums.iter_mut().zip(dcut) {
                        *sum += (cg * *x * *d * *hm).re;
                    }
                    for (sum, (_, d)) in mcut_sums.iter_mut().zip(dmcut) {
                        *sum += (cg * *x * *h * *d).re;
                    }
                }
                let d_env = match self.whole(n, g, &tr, fr) {
                    Some(basis) => vec![std::array::from_fn(|q| {
                        (0..g.bins())
                            .map(|j| (gamma(fr.m, j).conj() * basis[q][j] * shaped[j]).re)
                            .sum::<f64>()
                    })],
                    None => {
                        let xb =
                            samples_back(g, &tr, fr.m, &|mm, j| shaped[j].conj() * gamma(mm, j));
                        let mut d = vec![[0.0; 3]; fr.pieces.len()];
                        each_noise_sample(g, fr.m, (lo, hi), n, |k, s| {
                            let tl = s as f64 / g.sr - self.at;
                            let (p, w, e, sh) = self.enveloped(fr, &env, tl);
                            let x = xb[k] * n.samples[s];
                            for q in 0..3 {
                                d[p][q] += x * sh * w[q];
                            }
                            shouldered(tl, x * e, &mut sh_adj);
                        });
                        let (l, r) = self.noise_edges(g, fr, n, &env, &xb, (lo, hi));
                        l_adj += l;
                        r_adj += r;
                        d
                    }
                };
                for term in self.noise_terms() {
                    for (p, piece) in fr.pieces.iter().enumerate() {
                        for (q, at) in (piece.i..piece.i + 3).enumerate() {
                            term.amp
                                .back(&|l| self.leaf(l, at), d_env[p][q], &mut |l, x| {
                                    add_leaf(l, at, x, &mut leaf_adj)
                                });
                        }
                    }
                }
            }
        }
        let mut push = |d: Dual, x: f64| {
            if x != 0.0 {
                for (k, t) in self.tangents.of(d) {
                    out.push((*k, t * x));
                }
            }
        };
        for (l, adj) in leaf_adj.iter().enumerate() {
            for (i, x) in adj.iter().enumerate() {
                push(self.leaves[l][i], *x);
            }
        }
        push(self.l, l_adj + sh_adj[0]);
        push(self.r, r_adj + sh_adj[1]);
        push(self.rise, sh_adj[2]);
        push(self.fall, sh_adj[3]);
        out.extend(cut_adj);
        out.extend(cut_seeds.into_iter().zip(cut_sums));
        out.extend(mcut_seeds.into_iter().zip(mcut_sums));
    }

    /// The noise's share of `dL/dl` and `dL/dr` in a frame its crop cuts: the sample at each
    /// edge read in or out, `xb` each sample's `dL/dx` under the gain `env`.
    fn noise_edges(
        &self,
        g: &Grid,
        fr: &Frame,
        noise: &Realisation,
        env: &[[f64; 3]],
        xb: &[f64],
        (lo, hi): (f64, f64),
    ) -> (f64, f64) {
        let s0 = g.first(fr.m) as f64;
        let end = s0 + g.window as f64;
        let cuts = (lo > s0 && lo < end) || (hi > s0 && hi < end);
        if s0 < 0.0 || end > g.len as f64 || !cuts {
            return (0.0, 0.0);
        }
        let read = |k: usize| {
            let (_, _, e, sh) = self.enveloped(fr, env, (s0 + k as f64) / g.sr - self.at);
            e * sh * noise.samples.get(s0 as usize + k).copied().unwrap_or(0.0)
        };
        let mut out = (0.0, 0.0);
        if lo > s0 && lo < end {
            let k = (lo - s0).ceil() as usize;
            if k < g.window {
                out.0 = -g.sr * xb[k] * read(k);
            }
        }
        if hi > s0 && hi < end {
            let k = (hi - s0).ceil() as usize;
            if k < g.window {
                out.1 = g.sr * xb[k] * read(k);
            }
        }
        out
    }

    /// The shoulders' slopes in `l`, `r`, `rise` and `fall` at `t`, where `t` is inside.
    fn shoulder_slopes(&self, t: f64) -> [(Dual, f64); 4] {
        let (l, r, rise, fall) = (self.l.v, self.r.v, self.rise.v, self.fall.v);
        let open = |into: f64, span: f64| match span > 0.0 && into < span {
            true => (
                0.5 - 0.5 * libm::cos(std::f64::consts::PI * into / span),
                0.5 * std::f64::consts::PI * libm::sin(std::f64::consts::PI * into / span) / span,
            ),
            false => (1.0, 0.0),
        };
        let (a, da) = open(t - l, rise);
        let (b, db) = open(r - t, fall);
        let (dl, dr, drise, dfall) = match a <= b {
            true => (-da, 0.0, -da * (t - l) / rise.max(f64::MIN_POSITIVE), 0.0),
            false => (0.0, db, 0.0, -db * (r - t) / fall.max(f64::MIN_POSITIVE)),
        };
        [
            (self.l, dl),
            (self.r, dr),
            (self.rise, drise),
            (self.fall, dfall),
        ]
    }
}

/// Frame `m`'s `dL/dx_k` from its bins' `gamma`: `w_k Re(sum_j gamma_j e^{2 pi i j (left +
/// k)/n})`.
fn samples_back(
    g: &Grid,
    tr: &Transform,
    m: usize,
    gamma: &dyn Fn(usize, usize) -> C64,
) -> Vec<f64> {
    let (mut re, mut im) = (vec![0.0; g.n], vec![0.0; g.n]);
    for j in 0..g.bins() {
        let y = gamma(m, j);
        re[j] = y.re;
        im[j] = -y.im;
    }
    sva_samples::fft::fft_with(&mut re, &mut im, &tr.twiddles);
    (0..g.window).map(|k| tr.w[k] * re[g.left + k]).collect()
}

/// Each sample frame `m` reads of `[lo, hi)` and the clip, by window point.
fn each_noise_sample(
    g: &Grid,
    m: usize,
    (lo, hi): (f64, f64),
    noise: &Realisation,
    mut each: impl FnMut(usize, usize),
) {
    let s0 = g.first(m);
    for k in 0..g.window {
        let s = g.mirrored(s0 + k as i64);
        let sf = s as f64;
        if sf >= lo && sf < hi && (s as usize) < noise.samples.len() {
            each(k, s as usize);
        }
    }
}

/// Whether `b` reads the free variable anywhere.
fn reads_t<N: Num>(b: &Body<N>) -> bool {
    match b {
        Body::Line => true,
        Body::Const(_) | Body::Index(_) | Body::Param(_) => false,
        Body::Add(xs) | Body::Mul(xs) | Body::Fold(_, xs) | Body::Join(xs) => {
            xs.iter().any(|p| reads_t(&p.body))
        }
        Body::Div(a, c) => reads_t(&a.body) || reads_t(&c.body),
        Body::Pow(a, _) | Body::Apply(_, a) | Body::Pv(a) | Body::Channel(a, _) => reads_t(&a.body),
        Body::Knots { at, .. } => reads_t(&at.body),
        _ => true,
    }
}

/// Whether `b` holds an oscillator: a sine or cosine of an argument reading `t`.
fn sounds<N: Num>(b: &Body<N>) -> bool {
    match b {
        Body::Apply(Unary::Sin | Unary::Cos, a) if reads_t(&a.body) => true,
        Body::Add(xs) | Body::Mul(xs) | Body::Fold(_, xs) | Body::Join(xs) => {
            xs.iter().any(|p| sounds(&p.body))
        }
        Body::Div(a, c) => sounds(&a.body) || sounds(&c.body),
        Body::Pow(a, _) | Body::Apply(_, a) | Body::Pv(a) | Body::Channel(a, _) => sounds(&a.body),
        Body::Knots { at, .. } => sounds(&at.body),
        Body::Crop { of, .. } | Body::Shift { of, .. } | Body::Warp { of, .. } => sounds(&of.body),
        _ => false,
    }
}

struct Raw<'a> {
    factors: Vec<Factor<'a>>,
    osc: Option<(Unary, &'a Body<Dual>)>,
}

#[derive(Clone, Copy)]
enum Factor<'a> {
    Of(&'a Body<Dual>),
    Inverse(&'a Body<Dual>),
}

fn expand(b: &Body<Dual>) -> Result<Vec<Raw<'_>>, String> {
    Ok(match b {
        Body::Add(xs) => {
            let mut out = Vec::new();
            for x in xs {
                out.extend(expand(&x.body)?);
            }
            out
        }
        Body::Mul(xs) if sounds(b) => {
            let mut out = vec![Raw {
                factors: Vec::new(),
                osc: None,
            }];
            for x in xs {
                let parts = expand(&x.body)?;
                let mut next = Vec::with_capacity(out.len() * parts.len());
                for o in &out {
                    for p in &parts {
                        if o.osc.is_some() && p.osc.is_some() {
                            return Err("a product of two oscillators".to_string());
                        }
                        let mut factors = o.factors.clone();
                        factors.extend(p.factors.iter().copied());
                        next.push(Raw {
                            factors,
                            osc: o.osc.or(p.osc),
                        });
                    }
                }
                out = next;
            }
            out
        }
        Body::Div(a, c) if sounds(&a.body) && !sounds(&c.body) => {
            let mut out = expand(&a.body)?;
            for o in &mut out {
                o.factors.push(Factor::Inverse(&c.body));
            }
            out
        }
        Body::Apply(op @ (Unary::Sin | Unary::Cos), a) if reads_t(&a.body) => vec![Raw {
            factors: Vec::new(),
            osc: Some((*op, &a.body)),
        }],
        Body::Const(c) if c.re.v == 0.0 && c.im.v == 0.0 && c.re.is_constant() => Vec::new(),
        other if sounds(other) => {
            return Err("an oscillator under a crop, warp, shift, divisor or function".to_string());
        }
        other => vec![Raw {
            factors: vec![Factor::Of(other)],
            osc: None,
        }],
    })
}

struct Leaves<'a> {
    forms: Vec<Body<Dual>>,
    seen: Vec<&'a Body<Dual>>,
}

impl<'a> Leaves<'a> {
    fn id(&mut self, b: &'a Body<Dual>, inverse: bool) -> usize {
        if !inverse && let Some(i) = self.seen.iter().position(|x| *x == b) {
            return i;
        }
        self.seen.push(b);
        let form = match inverse {
            true => Body::Pow(
                sva_formula::closed_form::Part::new(sva_formula::Origin::UNKNOWN, b.clone()),
                -1,
            ),
            false => b.clone(),
        };
        self.forms.push(form);
        self.forms.len() - 1
    }

    fn product(&mut self, b: &'a Body<Dual>, out: &mut Vec<usize>) {
        match b {
            Body::Mul(xs) => xs.iter().for_each(|x| self.product(&x.body, out)),
            other => out.push(self.id(other, false)),
        }
    }

    fn sum(&mut self, b: &'a Body<Dual>) -> SumProd {
        let mut terms = Vec::new();
        match b {
            Body::Add(xs) => {
                for x in xs {
                    let mut p = Vec::new();
                    self.product(&x.body, &mut p);
                    terms.push(p);
                }
            }
            other => {
                let mut p = Vec::new();
                self.product(other, &mut p);
                terms.push(p);
            }
        }
        SumProd(terms)
    }
}

pub struct Split {
    pub l: Dual,
    pub r: Dual,
    pub rise: Dual,
    pub fall: Dual,
    pub terms: Vec<Term>,
    pub leaves: Vec<Body<Dual>>,
}

fn edge(e: Edge<Dual>) -> Dual {
    match e {
        Edge::NegInf => Dual::of(f64::NEG_INFINITY),
        Edge::PosInf => Dual::of(f64::INFINITY),
        Edge::At(x) => x,
    }
}

/// `noisy`: the node's own noise was cut out, and its terms free of oscillators carry it.
pub fn split(body: &Body<Dual>, noisy: bool) -> Result<Split, String> {
    let (of, l, r, rise, fall) = match body {
        Body::Crop {
            of,
            l,
            r,
            rise,
            fall,
        } => (&*of.body, edge(*l), edge(*r), *rise, *fall),
        other => (
            other,
            Dual::of(f64::NEG_INFINITY),
            Dual::of(f64::INFINITY),
            Dual::ZERO,
            Dual::ZERO,
        ),
    };
    let raw = expand(of)?;
    let mut leaves = Leaves {
        forms: Vec::new(),
        seen: Vec::new(),
    };
    let mut terms = Vec::new();
    for t in &raw {
        let mut amp = Vec::new();
        for f in &t.factors {
            match f {
                Factor::Of(b) => leaves.product(b, &mut amp),
                Factor::Inverse(b) => amp.push(leaves.id(b, true)),
            }
        }
        let carrier = match t.osc {
            Some((op, phase)) => Carrier::Tone {
                phase: leaves.sum(phase),
                sine: op == Unary::Sin,
            },
            None if noisy => Carrier::Noise,
            None => Carrier::Tone {
                phase: SumProd::default(),
                sine: false,
            },
        };
        terms.push(Term {
            amp: SumProd(vec![amp]),
            carrier,
        });
    }
    Ok(Split {
        l,
        r,
        rise,
        fall,
        terms,
        leaves: leaves.forms,
    })
}

/// Each leaf at each instant through the render's own column program.
pub fn read_leaves(leaves: &[Body<Dual>], instants: &[f64]) -> Result<Vec<Vec<Dual>>, String> {
    let free: Vec<bool> = leaves.iter().map(|b| !reads_t(b)).collect();
    let pick = |want: bool| -> Vec<&Body<Dual>> {
        leaves
            .iter()
            .zip(&free)
            .filter(|(_, f)| **f == want)
            .map(|(b, _)| b)
            .collect()
    };
    let once = Evaluator::over_each(&pick(true), 0).read_each(&[0.0]);
    let moving = Evaluator::over_each(&pick(false), 0).read_each(instants);
    let (mut once, mut moving) = (once.into_iter(), moving.into_iter());
    leaves
        .iter()
        .zip(&free)
        .map(|(_, f)| {
            let col = match f {
                true => once.next(),
                false => moving.next(),
            }
            .expect("one column per leaf");
            col.into_iter()
                .map(|v| {
                    let v: Cx<Dual> = v.map_err(|e| format!("{e:?}"))?;
                    Ok(v.re)
                })
                .collect()
        })
        .collect()
}
