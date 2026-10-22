// Concern: a reverse-mode tape over small row-major arrays that broadcast rows and columns | Non-concern: what any value means (power.rs, distance.rs) | IO: (ops) -> values; (seeds) -> gradients

use std::sync::Arc;

/// `r` rows of `c` columns, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Arr {
    pub r: usize,
    pub c: usize,
    pub v: Vec<f64>,
}

impl Arr {
    pub fn new(r: usize, c: usize, v: Vec<f64>) -> Arr {
        assert_eq!(r * c, v.len(), "an array holds rows times columns values");
        Arr { r, c, v }
    }

    pub fn scalar(x: f64) -> Arr {
        Arr::new(1, 1, vec![x])
    }

    pub fn zeros(r: usize, c: usize) -> Arr {
        Arr::new(r, c, vec![0.0; r * c])
    }

    pub fn row(v: Vec<f64>) -> Arr {
        Arr::new(1, v.len(), v)
    }

    pub fn column(v: Vec<f64>) -> Arr {
        Arr::new(v.len(), 1, v)
    }

    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.v[i * self.c + j]
    }

    fn get_broadcast(&self, i: usize, j: usize) -> f64 {
        let i = if self.r == 1 { 0 } else { i };
        let j = if self.c == 1 { 0 } else { j };
        self.v[i * self.c + j]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct V(u32);

/// A row's index outside `[0, c)`: zero, or the nearest end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pad {
    Zero,
    Edge,
}

/// `y[r, i] = sum_j k[j] x[r, i*stride + off + j]`.
#[derive(Clone, Debug)]
pub struct Conv {
    pub k: Vec<f64>,
    pub stride: usize,
    pub off: i64,
    pub out: usize,
    pub pad: Pad,
}

impl Conv {
    fn index(&self, c: usize, i: usize, j: usize) -> Option<usize> {
        let at = (i * self.stride) as i64 + self.off + j as i64;
        match self.pad {
            Pad::Zero => (at >= 0 && (at as usize) < c).then_some(at as usize),
            Pad::Edge => Some(at.clamp(0, c as i64 - 1) as usize),
        }
    }
}

/// `y = S x`, `x` and `y` read row-major; entries `(row of y, index into x, weight)`.
#[derive(Clone, Debug)]
pub struct Sparse {
    pub r: usize,
    pub c: usize,
    pub entries: Vec<(u32, u32, f64)>,
}

/// A unit sinusoid's power per band at each frequency of a log grid, read linearly in log2 f.
#[derive(Debug)]
pub struct BandTable {
    pub log2_lo: f64,
    pub step: f64,
    pub bands: usize,
    /// `rows[i * bands + b]`.
    pub rows: Vec<f64>,
    /// At and above this frequency a line holds no power: the profile's ceiling.
    pub ceiling_hz: f64,
    /// The bands rows `i` and `i + 1` reach, first to last.
    reach: Vec<(usize, usize)>,
}

impl BandTable {
    pub fn new(
        log2_lo: f64,
        step: f64,
        bands: usize,
        rows: Vec<f64>,
        ceiling_hz: f64,
    ) -> BandTable {
        let points = rows.len() / bands;
        let span = |i: usize| {
            let row = &rows[i * bands..(i + 1) * bands];
            let lo = row.iter().position(|x| *x != 0.0);
            let hi = row.iter().rposition(|x| *x != 0.0);
            lo.zip(hi)
        };
        let reach = (0..points)
            .map(|i| {
                let (a, b) = (span(i), span((i + 1).min(points - 1)));
                match (a, b) {
                    (Some(a), Some(b)) => (a.0.min(b.0), a.1.max(b.1)),
                    (Some(x), None) | (None, Some(x)) => x,
                    (None, None) => (1, 0),
                }
            })
            .collect();
        BandTable {
            log2_lo,
            step,
            bands,
            rows,
            ceiling_hz,
            reach,
        }
    }

    fn points(&self) -> usize {
        self.rows.len() / self.bands
    }

    /// The row index and fraction a frequency reads, and whether it moves with it.
    fn locate(&self, hz: f64) -> Option<(usize, f64, bool)> {
        if hz.is_nan() || hz <= 0.0 || hz >= self.ceiling_hz {
            return None;
        }
        let x = (libm::log2(hz) - self.log2_lo) / self.step;
        let last = (self.points() - 1) as f64;
        if x <= 0.0 {
            return Some((0, 0.0, false));
        }
        if x >= last {
            return Some((self.points() - 2, 1.0, false));
        }
        let i0 = x.floor() as usize;
        Some((i0, x - i0 as f64, true))
    }

    /// The power per band in `out`, and the bands outside which it is zero.
    pub fn power(&self, hz: f64, out: &mut [f64]) -> (usize, usize) {
        let Some((i0, fr, _)) = self.locate(hz) else {
            return (1, 0);
        };
        let (lo, hi) = self.reach[i0];
        let (a, b) = (i0 * self.bands, (i0 + 1) * self.bands);
        let top = hi.min(self.bands - 1);
        for (k, o) in out.iter_mut().enumerate().take(top + 1).skip(lo) {
            *o = self.rows[a + k] * (1.0 - fr) + self.rows[b + k] * fr;
        }
        (lo, hi)
    }

    /// d(power)/d(hz) per band over the bands `power` names.
    fn slope(&self, hz: f64, out: &mut [f64]) -> bool {
        let Some((i0, _, true)) = self.locate(hz) else {
            return false;
        };
        let (lo, hi) = self.reach[i0];
        let per_hz = 1.0 / (self.step * hz * std::f64::consts::LN_2);
        let (a, b) = (i0 * self.bands, (i0 + 1) * self.bands);
        let top = hi.min(self.bands - 1);
        for (k, o) in out.iter_mut().enumerate().take(top + 1).skip(lo) {
            *o = (self.rows[b + k] - self.rows[a + k]) * per_hz;
        }
        true
    }
}

#[derive(Debug)]
enum Op {
    Leaf,
    Add(V, V),
    Sub(V, V),
    Mul(V, V),
    Div(V, V),
    Pow(V, V),
    Min(V, V),
    Max(V, V),
    Neg(V),
    Exp(V),
    Ln(V),
    Sin(V),
    Cos(V),
    Sqrt(V),
    /// `sqrt(x^2 + e)`.
    Sabs(V),
    /// Held inside `[lo, hi]`; no gradient where held.
    Clamp(V, f64, f64),
    SumAll(V),
    /// Each row's sum, a column.
    SumRows(V),
    /// Each column's sum, a row.
    SumCols(V),
    Cols(V, usize, usize),
    Concat(Vec<V>),
    MatL(Arc<Arr>, V),
    Linear(Arc<Sparse>, V),
    Conv(Arc<Conv>, V),
    /// `x` read through the knots `(xs[j], vals[j])`, zero before the first and from the last.
    Pwl(V, Arc<Vec<f64>>, Vec<V>),
    /// `out[b, f] = sum_k p[k, f] G(hz[k, f])[b]`.
    Bands(V, V, Arc<BandTable>),
}

struct Node {
    val: Arr,
    op: Op,
    live: bool,
}

#[derive(Default)]
pub struct Tape {
    nodes: Vec<Node>,
}

fn shape(a: &Arr, b: &Arr) -> (usize, usize) {
    let dim = |x: usize, y: usize| match (x, y) {
        (x, y) if x == y => x,
        (1, y) => y,
        (x, 1) => x,
        (x, y) => panic!("arrays of {x} and {y} do not broadcast"),
    };
    (dim(a.r, b.r), dim(a.c, b.c))
}

fn zip(a: &Arr, b: &Arr, f: impl Fn(f64, f64) -> f64) -> Arr {
    let (r, c) = shape(a, b);
    if a.r == r && a.c == c && b.r == r && b.c == c {
        return Arr::new(r, c, a.v.iter().zip(&b.v).map(|(x, y)| f(*x, *y)).collect());
    }
    let mut v = Vec::with_capacity(r * c);
    for i in 0..r {
        for j in 0..c {
            v.push(f(a.get_broadcast(i, j), b.get_broadcast(i, j)));
        }
    }
    Arr::new(r, c, v)
}

fn map(a: &Arr, f: impl Fn(f64) -> f64) -> Arr {
    Arr::new(a.r, a.c, a.v.iter().map(|x| f(*x)).collect())
}

/// `g` summed down to `to`'s shape, where `to` broadcast.
fn reduce(g: Arr, r: usize, c: usize) -> Arr {
    if g.r == r && g.c == c {
        return g;
    }
    let mut out = Arr::zeros(r, c);
    for i in 0..g.r {
        for j in 0..g.c {
            let (oi, oj) = (if r == 1 { 0 } else { i }, if c == 1 { 0 } else { j });
            out.v[oi * c + oj] += g.v[i * g.c + j];
        }
    }
    out
}

fn add_into(slot: &mut Option<Arr>, g: Arr) {
    match slot {
        Some(held) => {
            assert_eq!(
                (held.r, held.c),
                (g.r, g.c),
                "a gradient keeps its value's shape"
            );
            held.v.iter_mut().zip(&g.v).for_each(|(h, x)| *h += x);
        }
        None => *slot = Some(g),
    }
}

impl Tape {
    pub fn new() -> Tape {
        Tape::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn push(&mut self, val: Arr, op: Op, live: bool) -> V {
        self.nodes.push(Node { val, op, live });
        V(self.nodes.len() as u32 - 1)
    }

    fn live(&self, x: V) -> bool {
        self.nodes[x.0 as usize].live
    }

    pub fn val(&self, x: V) -> &Arr {
        &self.nodes[x.0 as usize].val
    }

    pub fn scalar_of(&self, x: V) -> f64 {
        let a = self.val(x);
        assert_eq!(a.v.len(), 1, "a scalar holds one value");
        a.v[0]
    }

    /// A value gradients reach when `live`.
    pub fn leaf(&mut self, val: Arr, live: bool) -> V {
        self.push(val, Op::Leaf, live)
    }

    pub fn constant(&mut self, val: Arr) -> V {
        self.leaf(val, false)
    }

    pub fn num(&mut self, x: f64) -> V {
        self.constant(Arr::scalar(x))
    }

    /// The same value with no gradient through it.
    pub fn detach(&mut self, x: V) -> V {
        let val = self.val(x).clone();
        self.constant(val)
    }

    fn binary(&mut self, a: V, b: V, op: fn(V, V) -> Op, f: impl Fn(f64, f64) -> f64) -> V {
        let val = zip(self.val(a), self.val(b), f);
        let live = self.live(a) || self.live(b);
        self.push(val, op(a, b), live)
    }

    fn unary(&mut self, a: V, op: Op, f: impl Fn(f64) -> f64) -> V {
        let val = map(self.val(a), f);
        let live = self.live(a);
        self.push(val, op, live)
    }

    pub fn add(&mut self, a: V, b: V) -> V {
        self.binary(a, b, Op::Add, |x, y| x + y)
    }

    pub fn sub(&mut self, a: V, b: V) -> V {
        self.binary(a, b, Op::Sub, |x, y| x - y)
    }

    pub fn mul(&mut self, a: V, b: V) -> V {
        self.binary(a, b, Op::Mul, |x, y| x * y)
    }

    pub fn div(&mut self, a: V, b: V) -> V {
        self.binary(a, b, Op::Div, |x, y| x / y)
    }

    pub fn pow(&mut self, a: V, b: V) -> V {
        self.binary(a, b, Op::Pow, libm::pow)
    }

    pub fn min(&mut self, a: V, b: V) -> V {
        self.binary(a, b, Op::Min, f64::min)
    }

    pub fn max(&mut self, a: V, b: V) -> V {
        self.binary(a, b, Op::Max, f64::max)
    }

    pub fn neg(&mut self, a: V) -> V {
        self.unary(a, Op::Neg(a), |x| -x)
    }

    pub fn exp(&mut self, a: V) -> V {
        self.unary(a, Op::Exp(a), sva_formula::exp)
    }

    pub fn ln(&mut self, a: V) -> V {
        self.unary(a, Op::Ln(a), libm::log)
    }

    pub fn sin(&mut self, a: V) -> V {
        self.unary(a, Op::Sin(a), libm::sin)
    }

    pub fn cos(&mut self, a: V) -> V {
        self.unary(a, Op::Cos(a), libm::cos)
    }

    pub fn sqrt(&mut self, a: V) -> V {
        self.unary(a, Op::Sqrt(a), f64::sqrt)
    }

    pub fn sabs(&mut self, a: V, e: f64) -> V {
        self.unary(a, Op::Sabs(a), |x| (x * x + e).sqrt())
    }

    pub fn clamp(&mut self, a: V, lo: f64, hi: f64) -> V {
        self.unary(a, Op::Clamp(a, lo, hi), |x| x.clamp(lo, hi))
    }

    pub fn scale(&mut self, a: V, k: f64) -> V {
        let k = self.num(k);
        self.mul(a, k)
    }

    pub fn offset(&mut self, a: V, k: f64) -> V {
        let k = self.num(k);
        self.add(a, k)
    }

    pub fn square(&mut self, a: V) -> V {
        self.mul(a, a)
    }

    pub fn log2(&mut self, a: V) -> V {
        let l = self.ln(a);
        self.scale(l, 1.0 / std::f64::consts::LN_2)
    }

    pub fn sum(&mut self, a: V) -> V {
        let val = Arr::scalar(self.val(a).v.iter().sum());
        let live = self.live(a);
        self.push(val, Op::SumAll(a), live)
    }

    pub fn sum_rows(&mut self, a: V) -> V {
        let x = self.val(a);
        let v = (0..x.r)
            .map(|i| x.v[i * x.c..(i + 1) * x.c].iter().sum())
            .collect();
        let val = Arr::column(v);
        let live = self.live(a);
        self.push(val, Op::SumRows(a), live)
    }

    pub fn sum_cols(&mut self, a: V) -> V {
        let x = self.val(a);
        let mut v = vec![0.0; x.c];
        for i in 0..x.r {
            v.iter_mut()
                .zip(&x.v[i * x.c..(i + 1) * x.c])
                .for_each(|(o, y)| *o += y);
        }
        let val = Arr::row(v);
        let live = self.live(a);
        self.push(val, Op::SumCols(a), live)
    }

    pub fn mean(&mut self, a: V) -> V {
        let n = self.val(a).v.len() as f64;
        let s = self.sum(a);
        self.scale(s, 1.0 / n)
    }

    pub fn cols(&mut self, a: V, from: usize, to: usize) -> V {
        let x = self.val(a);
        assert!(from <= to && to <= x.c, "columns {from}..{to} of {}", x.c);
        let w = to - from;
        let mut v = Vec::with_capacity(x.r * w);
        for i in 0..x.r {
            v.extend_from_slice(&x.v[i * x.c + from..i * x.c + to]);
        }
        let val = Arr::new(x.r, w, v);
        let live = self.live(a);
        self.push(val, Op::Cols(a, from, to), live)
    }

    pub fn concat(&mut self, parts: Vec<V>) -> V {
        let r = self.val(parts[0]).r;
        let c: usize = parts.iter().map(|p| self.val(*p).c).sum();
        let mut v = vec![0.0; r * c];
        let mut at = 0;
        for p in &parts {
            let x = self.val(*p);
            assert_eq!(x.r, r, "concatenated arrays hold one row count");
            for i in 0..r {
                v[i * c + at..i * c + at + x.c].copy_from_slice(&x.v[i * x.c..(i + 1) * x.c]);
            }
            at += x.c;
        }
        let live = parts.iter().any(|p| self.live(*p));
        self.push(Arr::new(r, c, v), Op::Concat(parts), live)
    }

    pub fn matl(&mut self, m: Arc<Arr>, x: V) -> V {
        let a = self.val(x);
        assert_eq!(m.c, a.r, "a matrix of {} columns meets {} rows", m.c, a.r);
        let mut v = vec![0.0; m.r * a.c];
        for i in 0..m.r {
            for k in 0..m.c {
                let w = m.v[i * m.c + k];
                if w == 0.0 {
                    continue;
                }
                let row = &a.v[k * a.c..(k + 1) * a.c];
                v[i * a.c..(i + 1) * a.c]
                    .iter_mut()
                    .zip(row)
                    .for_each(|(o, y)| *o += w * y);
            }
        }
        let val = Arr::new(m.r, a.c, v);
        let live = self.live(x);
        self.push(val, Op::MatL(m, x), live)
    }

    pub fn linear(&mut self, s: Arc<Sparse>, x: V) -> V {
        let a = self.val(x);
        let mut v = vec![0.0; s.r * s.c];
        for (o, i, w) in &s.entries {
            v[*o as usize] += w * a.v[*i as usize];
        }
        let val = Arr::new(s.r, s.c, v);
        let live = self.live(x);
        self.push(val, Op::Linear(s, x), live)
    }

    pub fn conv(&mut self, k: Arc<Conv>, x: V) -> V {
        let a = self.val(x);
        let mut v = vec![0.0; a.r * k.out];
        for r in 0..a.r {
            let row = &a.v[r * a.c..(r + 1) * a.c];
            for i in 0..k.out {
                let mut s = 0.0;
                for (j, w) in k.k.iter().enumerate() {
                    if let Some(at) = k.index(a.c, i, j) {
                        s += w * row[at];
                    }
                }
                v[r * k.out + i] = s;
            }
        }
        let val = Arr::new(a.r, k.out, v);
        let live = self.live(x);
        self.push(val, Op::Conv(k, x), live)
    }

    pub fn pwl(&mut self, x: V, xs: Arc<Vec<f64>>, vals: Vec<V>) -> V {
        assert_eq!(xs.len(), vals.len(), "a knot each value");
        let ys: Vec<f64> = vals.iter().map(|v| self.scalar_of(*v)).collect();
        let val = map(self.val(x), |t| match segment(&xs, t) {
            Some((j, u)) => ys[j] * (1.0 - u) + ys[j + 1] * u,
            None => 0.0,
        });
        let live = self.live(x) || vals.iter().any(|v| self.live(*v));
        self.push(val, Op::Pwl(x, xs, vals), live)
    }

    pub fn bands(&mut self, p: V, hz: V, table: Arc<BandTable>) -> V {
        let (pa, fa) = (self.val(p), self.val(hz));
        assert_eq!(pa.r, fa.r, "a frequency each line");
        assert!(fa.c == 1 || fa.c == pa.c, "a frequency each frame, or one");
        let nb = table.bands;
        let mut out = vec![0.0; nb * pa.c];
        let mut g = vec![0.0; nb];
        let mut span = (1, 0);
        for k in 0..pa.r {
            if fa.c == 1 {
                span = table.power(fa.v[k], &mut g);
            }
            for f in 0..pa.c {
                if fa.c != 1 {
                    span = table.power(fa.v[k * fa.c + f], &mut g);
                }
                let pw = pa.v[k * pa.c + f];
                if pw == 0.0 {
                    continue;
                }
                for b in span.0..=span.1 {
                    out[b * pa.c + f] += pw * g[b];
                }
            }
        }
        let val = Arr::new(nb, pa.c, out);
        let live = self.live(p) || self.live(hz);
        self.push(val, Op::Bands(p, hz, table), live)
    }

    /// The product of `terms`, each divided where flagged, multiplied within its shape first so
    /// the widest arrays are made once.
    pub fn product(&mut self, terms: &[(V, bool)]) -> V {
        let mut groups: [Option<V>; 4] = [None; 4];
        let mut pending: [Vec<V>; 4] = Default::default();
        for (v, inverse) in terms {
            let a = self.val(*v);
            let class = usize::from(a.r > 1) * 2 + usize::from(a.c > 1);
            match (groups[class], inverse) {
                (None, false) => groups[class] = Some(*v),
                (Some(g), false) => groups[class] = Some(self.mul(g, *v)),
                (_, true) => pending[class].push(*v),
            }
        }
        for (class, inverse) in pending.into_iter().enumerate() {
            for v in inverse {
                let g = match groups[class] {
                    Some(g) => g,
                    None => self.num(1.0),
                };
                groups[class] = Some(self.div(g, v));
            }
        }
        let [s, t, k, kt] = groups;
        let (k, t, kt) = match (s, k, t, kt) {
            (Some(s), Some(k), t, kt) => (Some(self.mul(k, s)), t, kt),
            (Some(s), None, Some(t), kt) => (None, Some(self.mul(t, s)), kt),
            (Some(s), None, None, Some(kt)) => (None, None, Some(self.mul(kt, s))),
            (Some(s), None, None, None) => return s,
            (None, k, t, kt) => (k, t, kt),
        };
        let outer = match (k, t) {
            (Some(k), Some(t)) => Some(self.mul(k, t)),
            (k, t) => k.or(t),
        };
        match (outer, kt) {
            (Some(o), Some(kt)) => self.mul(o, kt),
            (o, kt) => o.or(kt).unwrap_or_else(|| self.num(1.0)),
        }
    }

    /// Every live value's gradient of the sum over `seeds` of `<seed, value>`.
    pub fn backward(&self, seeds: &[(V, &Arr)]) -> Grads {
        let mut g: Vec<Option<Arr>> = (0..self.nodes.len()).map(|_| None).collect();
        for (at, seed) in seeds {
            let held = self.val(*at);
            assert_eq!(
                (held.r, held.c),
                (seed.r, seed.c),
                "a seed takes its value's shape"
            );
            if self.live(*at) {
                add_into(&mut g[at.0 as usize], (*seed).clone());
            }
        }
        for n in (0..self.nodes.len()).rev() {
            let Some(gn) = g[n].take() else {
                continue;
            };
            self.back(n, &gn, &mut g);
            g[n] = Some(gn);
        }
        Grads { g }
    }

    fn send(&self, to: V, grad: Arr, g: &mut [Option<Arr>]) {
        if !self.live(to) {
            return;
        }
        let t = self.val(to);
        let grad = reduce(grad, t.r, t.c);
        add_into(&mut g[to.0 as usize], grad);
    }

    fn back(&self, n: usize, gn: &Arr, g: &mut [Option<Arr>]) {
        let node = &self.nodes[n];
        let out = &node.val;
        match &node.op {
            Op::Leaf => {}
            Op::Add(a, b) => {
                self.send(*a, gn.clone(), g);
                self.send(*b, gn.clone(), g);
            }
            Op::Sub(a, b) => {
                self.send(*a, gn.clone(), g);
                self.send(*b, map(gn, |x| -x), g);
            }
            Op::Mul(a, b) => {
                let (va, vb) = (self.val(*a), self.val(*b));
                if self.live(*a) {
                    self.send(*a, zip(gn, vb, |x, y| x * y), g);
                }
                if self.live(*b) {
                    self.send(*b, zip(gn, va, |x, y| x * y), g);
                }
            }
            Op::Div(a, b) => {
                let vb = self.val(*b);
                if self.live(*a) {
                    self.send(*a, zip(gn, vb, |x, y| x / y), g);
                }
                if self.live(*b) {
                    let q = zip(out, vb, |o, y| -o / y);
                    self.send(*b, zip(gn, &q, |x, y| x * y), g);
                }
            }
            Op::Pow(a, b) => {
                let (va, vb) = (self.val(*a), self.val(*b));
                if self.live(*a) {
                    let d = zip(va, vb, |x, y| match y {
                        0.0 => 0.0,
                        _ => y * libm::pow(x, y - 1.0),
                    });
                    self.send(*a, zip(gn, &d, |x, y| x * y), g);
                }
                if self.live(*b) {
                    let la = map(va, |x| if x > 0.0 { libm::log(x) } else { 0.0 });
                    let d = zip(out, &la, |o, l| o * l);
                    self.send(*b, zip(gn, &d, |x, y| x * y), g);
                }
            }
            Op::Min(a, b) | Op::Max(a, b) => {
                let (va, vb) = (self.val(*a), self.val(*b));
                let pick_a = matches!(node.op, Op::Min(..));
                let to_a = zip(va, vb, |x, y| match pick_a {
                    true => f64::from(u8::from(x <= y)),
                    false => f64::from(u8::from(x >= y)),
                });
                if self.live(*a) {
                    self.send(*a, zip(gn, &to_a, |x, w| x * w), g);
                }
                if self.live(*b) {
                    self.send(*b, zip(gn, &to_a, |x, w| x * (1.0 - w)), g);
                }
            }
            Op::Neg(a) => self.send(*a, map(gn, |x| -x), g),
            Op::Exp(a) => self.send(*a, zip(gn, out, |x, o| x * o), g),
            Op::Ln(a) => self.send(*a, zip(gn, self.val(*a), |x, y| x / y), g),
            Op::Sin(a) => {
                let d = map(self.val(*a), libm::cos);
                self.send(*a, zip(gn, &d, |x, y| x * y), g);
            }
            Op::Cos(a) => {
                let d = map(self.val(*a), |y| -libm::sin(y));
                self.send(*a, zip(gn, &d, |x, y| x * y), g);
            }
            Op::Sqrt(a) => self.send(
                *a,
                zip(gn, out, |x, o| if o > 0.0 { x * 0.5 / o } else { 0.0 }),
                g,
            ),
            Op::Sabs(a) => {
                let d = zip(self.val(*a), out, |y, o| y / o);
                self.send(*a, zip(gn, &d, |x, y| x * y), g);
            }
            Op::Clamp(a, lo, hi) => {
                let d = map(self.val(*a), |y| f64::from(u8::from(y >= *lo && y <= *hi)));
                self.send(*a, zip(gn, &d, |x, y| x * y), g);
            }
            Op::SumAll(a) => {
                let t = self.val(*a);
                self.send(*a, Arr::new(t.r, t.c, vec![gn.v[0]; t.r * t.c]), g);
            }
            Op::SumRows(a) => {
                let t = self.val(*a);
                let v = (0..t.r)
                    .flat_map(|i| std::iter::repeat_n(gn.v[i], t.c))
                    .collect();
                self.send(*a, Arr::new(t.r, t.c, v), g);
            }
            Op::SumCols(a) => {
                let t = self.val(*a);
                let v = (0..t.r).flat_map(|_| gn.v.iter().copied()).collect();
                self.send(*a, Arr::new(t.r, t.c, v), g);
            }
            Op::Cols(a, from, to) => {
                let t = self.val(*a);
                let mut v = vec![0.0; t.r * t.c];
                let w = to - from;
                for i in 0..t.r {
                    v[i * t.c + from..i * t.c + to].copy_from_slice(&gn.v[i * w..(i + 1) * w]);
                }
                self.send(*a, Arr::new(t.r, t.c, v), g);
            }
            Op::Concat(parts) => {
                let mut at = 0;
                for p in parts {
                    let t = self.val(*p);
                    if self.live(*p) {
                        let mut v = Vec::with_capacity(t.r * t.c);
                        for i in 0..t.r {
                            v.extend_from_slice(&gn.v[i * gn.c + at..i * gn.c + at + t.c]);
                        }
                        self.send(*p, Arr::new(t.r, t.c, v), g);
                    }
                    at += t.c;
                }
            }
            Op::MatL(m, x) => {
                let t = self.val(*x);
                let mut v = vec![0.0; t.r * t.c];
                for i in 0..m.r {
                    for k in 0..m.c {
                        let w = m.v[i * m.c + k];
                        if w == 0.0 {
                            continue;
                        }
                        v[k * t.c..(k + 1) * t.c]
                            .iter_mut()
                            .zip(&gn.v[i * t.c..(i + 1) * t.c])
                            .for_each(|(o, y)| *o += w * y);
                    }
                }
                self.send(*x, Arr::new(t.r, t.c, v), g);
            }
            Op::Linear(s, x) => {
                let t = self.val(*x);
                let mut v = vec![0.0; t.r * t.c];
                for (o, i, w) in &s.entries {
                    v[*i as usize] += w * gn.v[*o as usize];
                }
                self.send(*x, Arr::new(t.r, t.c, v), g);
            }
            Op::Conv(k, x) => {
                let t = self.val(*x);
                let mut v = vec![0.0; t.r * t.c];
                for r in 0..t.r {
                    for i in 0..k.out {
                        let gi = gn.v[r * k.out + i];
                        if gi == 0.0 {
                            continue;
                        }
                        for (j, w) in k.k.iter().enumerate() {
                            if let Some(at) = k.index(t.c, i, j) {
                                v[r * t.c + at] += w * gi;
                            }
                        }
                    }
                }
                self.send(*x, Arr::new(t.r, t.c, v), g);
            }
            Op::Pwl(x, xs, vals) => {
                let t = self.val(*x);
                let ys: Vec<f64> = vals.iter().map(|v| self.scalar_of(*v)).collect();
                let mut dv = vec![0.0; vals.len()];
                let mut dx = vec![0.0; t.v.len()];
                for (i, at) in t.v.iter().enumerate() {
                    if let Some((j, u)) = segment(xs, *at) {
                        dv[j] += gn.v[i] * (1.0 - u);
                        dv[j + 1] += gn.v[i] * u;
                        dx[i] = gn.v[i] * (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]);
                    }
                }
                for (v, d) in vals.iter().zip(dv) {
                    self.send(*v, Arr::scalar(d), g);
                }
                self.send(*x, Arr::new(t.r, t.c, dx), g);
            }
            Op::Bands(p, hz, table) => {
                let (pa, fa) = (self.val(*p), self.val(*hz));
                let nb = table.bands;
                let mut dp = vec![0.0; pa.v.len()];
                let mut df = vec![0.0; fa.v.len()];
                let (mut gb, mut sb) = (vec![0.0; nb], vec![0.0; nb]);
                let freq_live = self.live(*hz);
                let (mut span, mut moves) = ((1, 0), false);
                for k in 0..pa.r {
                    if fa.c == 1 {
                        span = table.power(fa.v[k], &mut gb);
                        moves = freq_live && table.slope(fa.v[k], &mut sb);
                    }
                    for f in 0..pa.c {
                        if fa.c != 1 {
                            span = table.power(fa.v[k * fa.c + f], &mut gb);
                            moves = freq_live && table.slope(fa.v[k * fa.c + f], &mut sb);
                        }
                        let (mut s, mut d) = (0.0, 0.0);
                        for b in span.0..=span.1 {
                            let go = gn.v[b * pa.c + f];
                            s += go * gb[b];
                            if moves {
                                d += go * sb[b];
                            }
                        }
                        dp[k * pa.c + f] = s;
                        if moves {
                            let at = if fa.c == 1 { k } else { k * fa.c + f };
                            df[at] += d * pa.v[k * pa.c + f];
                        }
                    }
                }
                self.send(*p, Arr::new(pa.r, pa.c, dp), g);
                if freq_live {
                    self.send(*hz, Arr::new(fa.r, fa.c, df), g);
                }
            }
        }
    }
}

/// The knot segment `[xs[j], xs[j+1])` holding `t` and how far along; none outside the table.
fn segment(xs: &[f64], t: f64) -> Option<(usize, f64)> {
    let n = xs.len();
    if n < 2 || t.is_nan() || t < xs[0] || t >= xs[n - 1] {
        return None;
    }
    let j = xs.partition_point(|x| *x <= t).saturating_sub(1).min(n - 2);
    let span = xs[j + 1] - xs[j];
    Some((j, if span > 0.0 { (t - xs[j]) / span } else { 0.0 }))
}

pub struct Grads {
    g: Vec<Option<Arr>>,
}

impl Grads {
    pub fn of(&self, x: V) -> Option<&Arr> {
        self.g[x.0 as usize].as_ref()
    }

    pub fn scalar(&self, x: V) -> f64 {
        self.of(x).map_or(0.0, |a| a.v.iter().sum())
    }
}
