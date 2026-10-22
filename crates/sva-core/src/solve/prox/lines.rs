// Concern: a score line's short-time transform on the reading's frames off its lowering, its tangents back | Non-concern: lowering it, reading the bins | IO: (form, frames) -> bins; adjoint -> gradient

//! Each tone's amplitude and phase are read at a frame's middle and a quarter window either
//! side and taken quadratic between: exact where both are quadratic over its 20 ms. Elsewhere
//! no bound is proven a priori: a priced solve renders each point and reports its reading off
//! the render less this one as `predicted.error`, and chooses by the render's.

use std::sync::Arc;

use sva_formula::closed_form::{Edge, Unary};
use sva_formula::{Body, C64, Cx, Dual, Num, Tangents};
use sva_samples::Evaluator;
use sva_samples::measure::internal::Grid;

/// A frame reads its line a quarter window either side of its middle.
pub const SPREAD_SECS: f64 = 0.005;

fn hann(n: usize) -> Vec<f64> {
    sva_samples::measure::internal::hann(n)
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
    /// Local instants, three a frame: `t* - d, t*, t* + d`.
    pub instants: Vec<f64>,
    /// Each frame the line sounds in, and its first instant.
    pub frames: Vec<(usize, usize)>,
    pub tangents: Tangents,
    pub noise: Option<Arc<Vec<f64>>>,
    pub noise_filters: Vec<Filter>,
}

/// The support `[lo, hi)` in samples.
fn span(line: &Line, g: &Grid) -> (f64, f64) {
    let (l, r) = (line.l.v, line.r.v);
    ((line.at + l) * g.sr, (line.at + r) * g.sr)
}

/// The local instant frame `m` is linearized about: the middle of what it reads of the
/// support, mirrored reads included.
pub fn centre(at: f64, l: f64, r: f64, g: &Grid, m: usize) -> Option<f64> {
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
    (b > a).then(|| 0.5 * (a + b) / g.sr - at)
}

/// `centre`'s slopes in `l` and `r` where frame `m` reads no mirrored sample: half of each
/// edge inside its window.
fn centre_slopes(g: &Grid, m: usize, lo: f64, hi: f64) -> (f64, f64) {
    let s0 = g.first(m) as f64;
    let s1 = s0 + g.window as f64;
    if s0 < 0.0 || s1 > g.len as f64 {
        return (0.0, 0.0);
    }
    let inside = |x: f64| x > s0 && x < s1;
    (
        if inside(lo) { 0.5 } else { 0.0 },
        if inside(hi) { 0.5 } else { 0.0 },
    )
}

/// Lagrange's weights on a frame's three instants at `u`, its offset in quarter windows.
fn lagrange(u: f64) -> [f64; 3] {
    [0.5 * u * (u - 1.0), 1.0 - u * u, 0.5 * u * (u + 1.0)]
}

fn through(v: &[f64; 3], l: &[f64; 3]) -> f64 {
    v[0] * l[0] + v[1] * l[1] + v[2] * l[2]
}

/// One frame's bins: its tones' `a cos(phase)` and `a sin(phase)`, so a phase offset `p`
/// reads `cos p * cos - sin p * quad`; its noise, empty where it holds none.
pub struct Bins {
    pub m: usize,
    pub cos: Vec<C64>,
    pub quad: Vec<C64>,
    pub noise: Vec<C64>,
}

/// Each part's `gamma` per frame and bin.
pub struct Adjoint<'a> {
    pub cos: &'a dyn Fn(usize, usize) -> C64,
    pub quad: &'a dyn Fn(usize, usize) -> C64,
    pub noise: &'a dyn Fn(usize, usize) -> C64,
}

/// A tone in one frame at its three instants: `a = amp * sh`, the phase as a cosine's, its
/// frequency and the master's gain there, conjugated where the phase falls.
struct Tone {
    amp: [f64; 3],
    sh: [f64; 3],
    a: [f64; 3],
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

    fn tone(&self, term: &Term, i0: usize, master: &[Filter], sr: f64) -> Option<Tone> {
        let Carrier::Tone { phase, sine } = &term.carrier else {
            return None;
        };
        let lag = if *sine {
            std::f64::consts::FRAC_PI_2
        } else {
            0.0
        };
        let mut t = Tone {
            amp: [0.0; 3],
            sh: [0.0; 3],
            a: [0.0; 3],
            phi: [0.0; 3],
            hz: 0.0,
            h: C64::new(1.0, 0.0),
        };
        for i in 0..3 {
            t.sh[i] = self.shoulder(self.instants[i0 + i]);
            t.amp[i] = self.amp(term, i0 + i);
            t.a[i] = t.amp[i] * t.sh[i];
            t.phi[i] = phase.value(&|l| self.leaf(l, i0 + i)) - lag;
        }
        t.hz = (t.phi[2] - t.phi[0]) / (2.0 * SPREAD_SECS * std::f64::consts::TAU);
        if t.hz.abs() >= 0.5 * sr {
            return None;
        }
        let h = gain(master, t.hz.abs(), sr);
        t.h = if t.hz < 0.0 { h.conj() } else { h };
        Some(t)
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

    /// Each frame it sounds in, and its bins there.
    pub fn bins(&self, g: &Grid, master: &[Filter]) -> Vec<Bins> {
        let w = hann(g.window);
        let twiddles = sva_samples::fft::twiddles(g.n);
        let (lo, hi) = span(self, g);
        let mut out = Vec::with_capacity(self.frames.len());
        for (m, i0) in &self.frames {
            let tstar = self.instants[i0 + 1];
            let (mut re, mut im) = (vec![0.0; g.n], vec![0.0; g.n]);
            for term in &self.terms {
                let Some(t) = self.tone(term, *i0, master, g.sr) else {
                    continue;
                };
                let abs = t.h.abs();
                self.each_sample(g, *m, tstar, &t, |i, u, c, s| {
                    let a = w[i] * abs * through(&t.a, &lagrange(u));
                    re[g.left + i] += a * c;
                    im[g.left + i] += a * s;
                });
            }
            sva_samples::fft::fft_with(&mut re, &mut im, &twiddles);
            let z = |j: usize| C64::new(re[j % g.n], im[j % g.n]);
            let half = C64::new(0.5, 0.0);
            let cos: Vec<C64> = (0..g.bins())
                .map(|j| (z(j) + z(g.n - j).conj()) * half)
                .collect();
            let quad: Vec<C64> = (0..g.bins())
                .map(|j| (z(j) - z(g.n - j).conj()) * C64::new(0.0, -0.5))
                .collect();
            let mut bins =
                vec![C64::new(0.0, 0.0); if self.noise.is_some() { g.bins() } else { 0 }];
            if let Some(n) = &self.noise {
                let spec = noise_frame(n, g, *m, lo, hi);
                for term in self
                    .terms
                    .iter()
                    .filter(|t| matches!(t.carrier, Carrier::Noise))
                {
                    let e = self.amp(term, i0 + 1) * self.shoulder(tstar);
                    for (j, x) in spec.iter().enumerate() {
                        let hz = j as f64 * g.sr / g.n as f64;
                        let h = gain(&self.noise_filters, hz, g.sr) * gain(master, hz, g.sr);
                        bins[j] = bins[j] + (*x * h).scale(e);
                    }
                }
            }
            out.push(Bins {
                m: *m,
                cos,
                quad,
                noise: bins,
            });
        }
        out
    }

    /// The gradient in local indices from each of its bins' `dL/dRe X + i dL/dIm X`, of its
    /// tones' cosine and quadrature parts and of its noise.
    pub fn back(&self, g: &Grid, master: &[Filter], adj: &Adjoint, out: &mut Vec<(u32, f64)>) {
        let gamma = adj.noise;
        let w = hann(g.window);
        let mut leaf_adj: Vec<Vec<f64>> = self.leaves.iter().map(|v| vec![0.0; v.len()]).collect();
        let add_leaf = |l: usize, i: usize, x: f64, adj: &mut Vec<Vec<f64>>| {
            let slot = match adj[l].len() {
                1 => 0,
                _ => i,
            };
            adj[l][slot] += x;
        };
        let (mut l_adj, mut r_adj) = (0.0, 0.0);
        let mut sh_adj: Vec<(f64, f64)> = Vec::new();
        let mut cut_adj: Vec<(u32, f64)> = Vec::new();
        let (lo, hi) = span(self, g);
        for (m, i0) in &self.frames {
            let tstar = self.instants[i0 + 1];
            let xc = samples_back(g, &w, *m, adj.cos);
            let xs = samples_back(g, &w, *m, adj.quad);
            for term in &self.terms {
                let Some(t) = self.tone(term, *i0, master, g.sr) else {
                    continue;
                };
                let abs = t.h.abs();
                let (mut ga, mut gphi) = ([0.0; 3], [0.0; 3]);
                let (mut g_abs, mut g_arg) = (0.0, 0.0);
                let mut ends = (None, None);
                self.each_sample(g, *m, tstar, &t, |i, u, c, s| {
                    let lw = lagrange(u);
                    let a = through(&t.a, &lw);
                    let (along, across) = (xc[i] * c + xs[i] * s, xs[i] * c - xc[i] * s);
                    for j in 0..3 {
                        ga[j] += along * abs * lw[j];
                        gphi[j] += across * abs * a * lw[j];
                    }
                    g_abs += along * a;
                    g_arg += across * abs * a;
                    if ends.0.is_none() {
                        ends.0 = Some(i);
                    }
                    ends.1 = Some(i);
                });
                let s0 = g.first(*m) as f64;
                let unmirrored = s0 >= 0.0 && s0 + g.window as f64 <= g.len as f64;
                if unmirrored {
                    let value = |i: usize| {
                        let u = (((s0 + i as f64) / g.sr) - self.at - tstar) / SPREAD_SECS;
                        let (sn, cs) = libm::sincos(t.phase(u));
                        abs * through(&t.a, &lagrange(u)) * (xc[i] * cs + xs[i] * sn)
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
                for i in 0..3 {
                    let at = i0 + i;
                    term.amp
                        .back(&|l| self.leaf(l, at), ga[i] * t.sh[i], &mut |l, x| {
                            add_leaf(l, at, x, &mut leaf_adj)
                        });
                    sh_adj.push((self.instants[at], ga[i] * t.amp[i]));
                    phase.back(&|l| self.leaf(l, at), gphi[i], &mut |l, x| {
                        add_leaf(l, at, x, &mut leaf_adj)
                    });
                }
            }
        }
        if let Some(n) = &self.noise {
            for (m, i0) in &self.frames {
                let tstar = self.instants[i0 + 1];
                let sh = self.shoulder(tstar);
                let spec = noise_frame(n, g, *m, lo, hi);
                let sh_dt = {
                    let [(_, dl), (_, dr), ..] = self.shoulder_slopes(tstar);
                    -(dl + dr)
                };
                let mut d_t = 0.0;
                for term in self
                    .terms
                    .iter()
                    .filter(|t| matches!(t.carrier, Carrier::Noise))
                {
                    let e = self.amp(term, i0 + 1);
                    let e_dt = (self.amp(term, i0 + 2) - self.amp(term, *i0)) / (2.0 * SPREAD_SECS);
                    let mut d_e = 0.0;
                    for (j, x) in spec.iter().enumerate() {
                        let hz = j as f64 * g.sr / g.n as f64;
                        let (h, _, dcut) = chain(&self.noise_filters, hz, g.sr);
                        let (hm, _, dmcut) = chain(master, hz, g.sr);
                        let cg = gamma(*m, j).conj();
                        d_e += (cg * *x * h * hm).re;
                        for (s, d) in dcut {
                            cut_adj.push((s, (cg * *x * d * hm).re * e * sh));
                        }
                        for (s, d) in dmcut {
                            cut_adj.push((s, (cg * *x * h * d).re * e * sh));
                        }
                    }
                    term.amp
                        .back(&|l| self.leaf(l, i0 + 1), d_e * sh, &mut |l, x| {
                            add_leaf(l, i0 + 1, x, &mut leaf_adj)
                        });
                    sh_adj.push((tstar, d_e * e));
                    d_t += d_e * (e_dt * sh + e * sh_dt);
                }
                let (dl, dr) = centre_slopes(g, *m, lo, hi);
                l_adj += d_t * dl;
                r_adj += d_t * dr;
                let (l, r) = self.noise_edges(g, &w, master, n, (*m, *i0), (lo, hi), gamma);
                l_adj += l;
                r_adj += r;
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
        push(self.l, l_adj);
        push(self.r, r_adj);
        for (t, x) in sh_adj {
            if x == 0.0 {
                continue;
            }
            for (d, part) in self.shoulder_slopes(t) {
                push(d, x * part);
            }
        }
        out.extend(cut_adj);
    }

    /// The noise's share of `dL/dl` and `dL/dr` in a frame its crop cuts: the sample at each
    /// edge read in or out.
    #[allow(clippy::too_many_arguments)]
    fn noise_edges(
        &self,
        g: &Grid,
        w: &[f64],
        master: &[Filter],
        noise: &[f64],
        (m, i0): (usize, usize),
        (lo, hi): (f64, f64),
        gamma: &dyn Fn(usize, usize) -> C64,
    ) -> (f64, f64) {
        let s0 = g.first(m) as f64;
        let end = s0 + g.window as f64;
        let cuts = (lo > s0 && lo < end) || (hi > s0 && hi < end);
        if s0 < 0.0 || end > g.len as f64 || !cuts {
            return (0.0, 0.0);
        }
        let e: f64 = self
            .terms
            .iter()
            .filter(|t| matches!(t.carrier, Carrier::Noise))
            .map(|t| self.amp(t, i0 + 1))
            .sum::<f64>()
            * self.shoulder(self.instants[i0 + 1]);
        let gains: Vec<C64> = (0..g.bins())
            .map(|j| {
                let hz = j as f64 * g.sr / g.n as f64;
                (gain(&self.noise_filters, hz, g.sr) * gain(master, hz, g.sr)).scale(e)
            })
            .collect();
        let xb = samples_back(g, w, m, &|mm, j| gains[j].conj() * gamma(mm, j));
        let read = |k: usize| noise.get(s0 as usize + k).copied().unwrap_or(0.0);
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
fn samples_back(g: &Grid, w: &[f64], m: usize, gamma: &dyn Fn(usize, usize) -> C64) -> Vec<f64> {
    let (mut re, mut im) = (vec![0.0; g.n], vec![0.0; g.n]);
    for j in 0..g.bins() {
        let y = gamma(m, j);
        re[j] = y.re;
        im[j] = -y.im;
    }
    sva_samples::fft::fft(&mut re, &mut im);
    (0..g.window).map(|k| w[k] * re[g.left + k]).collect()
}

fn noise_frame(noise: &[f64], g: &Grid, m: usize, lo: f64, hi: f64) -> Vec<C64> {
    let w = hann(g.window);
    let s0 = g.first(m);
    let mut re = vec![0.0; g.n];
    let mut im = vec![0.0; g.n];
    for (k, wk) in w.iter().enumerate() {
        let s = g.mirrored(s0 + k as i64);
        let sf = s as f64;
        if sf >= lo && sf < hi && (s as usize) < noise.len() {
            re[g.left + k] = wk * noise[s as usize];
        }
    }
    sva_samples::fft::fft(&mut re, &mut im);
    (0..g.bins()).map(|j| C64::new(re[j], im[j])).collect()
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
