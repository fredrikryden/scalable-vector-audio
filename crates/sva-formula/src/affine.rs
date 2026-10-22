// Concern: reads a formula's polynomial to degree two, and a series term at each index | Non-concern: lowering one to atoms (spectral_sum/build.rs) | IO: (&Body) -> Vec<Coeff>; (&Body, k) -> C64

use crate::closed_form::{Body, IndexId, Unary};
use crate::complex::{C64, Cx};
use crate::num::Num;
use crate::through::{Opaque, Reads};

/// An affine argument makes the line and delta rows, a quadratic one the Gaussian row.
pub const MAX_DEGREE: usize = 2;

/// A series index is an unknown of known axis, which is all a growth row asks of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Real,
    Imaginary,
    Any,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coeff<N: Num = f64> {
    Exact(Cx<N>),
    Unknown(Axis),
}

impl Axis {
    pub fn of_codomain(codomain: crate::ty::Codomain) -> Axis {
        match codomain {
            crate::ty::Codomain::Real => Axis::Real,
            crate::ty::Codomain::Complex => Axis::Any,
        }
    }

    fn of<N: Num>(c: Cx<N>) -> Axis {
        match (c.is_real(), c.re.is_zero()) {
            (true, _) => Axis::Real,
            (false, true) => Axis::Imaginary,
            _ => Axis::Any,
        }
    }

    fn times(self, other: Axis) -> Axis {
        match (self, other) {
            (Axis::Real, a) | (a, Axis::Real) => a,
            (Axis::Imaginary, Axis::Imaginary) => Axis::Real,
            _ => Axis::Any,
        }
    }

    fn plus(self, other: Axis) -> Axis {
        if self == other { self } else { Axis::Any }
    }
}

impl<N: Num> Coeff<N> {
    pub const ZERO: Coeff<N> = Coeff::Exact(Cx::ZERO);
    pub const ONE: Coeff<N> = Coeff::Exact(Cx::ONE);

    pub fn exact(self) -> Option<Cx<N>> {
        match self {
            Coeff::Exact(c) => Some(c),
            Coeff::Unknown(_) => None,
        }
    }

    pub fn axis(self) -> Axis {
        match self {
            Coeff::Exact(c) => Axis::of(c),
            Coeff::Unknown(a) => a,
        }
    }

    pub fn is_zero(self) -> bool {
        matches!(self, Coeff::Exact(c) if c.is_zero())
    }

    fn add(self, other: Coeff<N>) -> Coeff<N> {
        match (self, other) {
            (Coeff::Exact(a), Coeff::Exact(b)) => Coeff::Exact(a + b),
            (a, b) if a.is_zero() => b,
            (a, b) if b.is_zero() => a,
            (a, b) => Coeff::Unknown(a.axis().plus(b.axis())),
        }
    }

    fn mul(self, other: Coeff<N>) -> Coeff<N> {
        match (self, other) {
            (Coeff::Exact(a), Coeff::Exact(b)) => Coeff::Exact(a * b),
            (a, b) if a.is_zero() || b.is_zero() => Coeff::ZERO,
            (a, b) => Coeff::Unknown(a.axis().times(b.axis())),
        }
    }

    fn div(self, other: Coeff<N>) -> Coeff<N> {
        match (self, other) {
            (Coeff::Exact(a), Coeff::Exact(b)) => Coeff::Exact(a / b),
            (a, b) => Coeff::Unknown(a.axis().times(b.axis())),
        }
    }

    fn scale(self, k: N) -> Coeff<N> {
        self.mul(Coeff::Exact(Cx::real(k)))
    }
}

/// Which symbol a polynomial is read in: the closed form's own variable, or one series index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reading {
    Free,
    Index(IndexId),
}

pub fn polynomial_over<N: Num>(f: &Body<N>) -> Option<Vec<Coeff<N>>> {
    polynomial_in_over(f, Reading::Free)
}

pub fn polynomial_in_over<N: Num>(f: &Body<N>, reading: Reading) -> Option<Vec<Coeff<N>>> {
    polynomial_read_over(f, reading, &Opaque)
}

/// Ascending coefficients, or nothing above degree two. Read in the free variable, an index
/// is an unknown real; read in an index, the free variable is not a coefficient at all.
pub fn polynomial_read_over<N: Num>(
    f: &Body<N>,
    reading: Reading,
    reads: &dyn Reads<N>,
) -> Option<Vec<Coeff<N>>> {
    read(f, reading, reads).map(|p| p.as_slice().to_vec())
}

/// A series term compiled once to be read at index after index: at each, the constant `read`
/// spells with the index written in its place, every subterm the index never reaches read
/// once, here.
pub struct ConstantAt<N: Num = f64>(Tape<N>);

impl<N: Num> ConstantAt<N> {
    pub fn of(f: &Body<N>, k: IndexId, reads: &dyn Reads<N>) -> ConstantAt<N> {
        ConstantAt(Tape::of(f, k, reads))
    }

    pub fn at(&self, value: f64) -> Option<Cx<N>> {
        match self.0.exact(value) {
            Exact::Held(c) => Some(c),
            Exact::Failed => None,
            Exact::Unknown => constant(self.0.run(value)?),
        }
    }
}

/// A tape read on constants alone, each op `Coeff`'s arithmetic; `Unknown` where one is no
/// number, which only the polynomials hold.
enum Exact<N: Num> {
    Held(Cx<N>),
    Failed,
    Unknown,
}

/// Postfix: each op pops its operands and pushes what `read` gives of them. `None` where the
/// term fails to read at every index.
struct Tape<N: Num>(Option<Vec<Op<N>>>);

enum Op<N: Num> {
    Push(Poly<N>),
    Index,
    Keyed(u64, Tape<N>),
    Apply(Unary),
    Add(usize),
    Mul(usize),
    Div,
    Pow(u32),
    Shift(N),
}

impl<N: Num> Tape<N> {
    fn of(f: &Body<N>, k: IndexId, reads: &dyn Reads<N>) -> Tape<N> {
        let mut ops = Vec::new();
        Tape(compile(f, k, reads, &mut ops).map(|()| ops))
    }

    fn exact(&self, value: f64) -> Exact<N> {
        let Some(ops) = &self.0 else {
            return Exact::Failed;
        };
        let mut stack: Vec<Coeff<N>> = Vec::with_capacity(8);
        let pop = |stack: &mut Vec<Coeff<N>>| {
            stack
                .pop()
                .expect("a compiled op pops only what an earlier op pushed")
        };
        for op in ops {
            let pushed = match op {
                Op::Push(p) => match p.as_slice() {
                    [c @ Coeff::Exact(_)] => *c,
                    _ => return Exact::Unknown,
                },
                Op::Index => Coeff::Exact(Cx::real(N::of(value))),
                Op::Keyed(seed, of) => match of.exact(value) {
                    Exact::Held(at) => match crate::hash::draw_nearest(*seed, at.re.v()) {
                        Some(drawn) => Coeff::Exact(Cx::real(N::of(drawn))),
                        None => return Exact::Unknown,
                    },
                    Exact::Failed | Exact::Unknown => return Exact::Unknown,
                },
                Op::Apply(unary) => match pop(&mut stack) {
                    Coeff::Exact(x) => Coeff::Exact(apply_scalar_over(*unary, x)),
                    Coeff::Unknown(_) => return Exact::Unknown,
                },
                Op::Add(n) => {
                    let from = stack.len() - n;
                    let sum = stack[from..]
                        .iter()
                        .fold(Coeff::ZERO, |acc, p| Coeff::ZERO.add(acc).add(*p));
                    stack.truncate(from);
                    sum
                }
                Op::Mul(n) => {
                    let from = stack.len() - n;
                    let product = stack[from..]
                        .iter()
                        .fold(Coeff::ONE, |acc, p| Coeff::ZERO.add(acc.mul(*p)));
                    stack.truncate(from);
                    product
                }
                Op::Div => {
                    let den = pop(&mut stack);
                    pop(&mut stack).div(den)
                }
                Op::Pow(n) => {
                    let base = pop(&mut stack);
                    (0..*n).fold(Coeff::ONE, |acc, _| Coeff::ZERO.add(acc.mul(base)))
                }
                Op::Shift(by) => shift(&Poly::one(pop(&mut stack)), *by).c[0],
            };
            stack.push(pushed);
        }
        match pop(&mut stack) {
            Coeff::Exact(c) => Exact::Held(c),
            Coeff::Unknown(_) => Exact::Unknown,
        }
    }

    fn run(&self, value: f64) -> Option<Poly<N>> {
        let mut stack: Vec<Poly<N>> = Vec::with_capacity(8);
        for op in self.0.as_ref()? {
            let pushed = match op {
                Op::Push(p) => *p,
                Op::Index => Poly::one(Coeff::Exact(Cx::real(N::of(value)))),
                Op::Keyed(seed, of) => keyed(*seed, of.run(value)),
                Op::Apply(unary) => applied(*unary, &popped(&mut stack))?,
                Op::Add(n) => {
                    let from = stack.len() - n;
                    let sum = stack[from..]
                        .iter()
                        .fold(Poly::one(Coeff::ZERO), |acc, p| add(&acc, p));
                    stack.truncate(from);
                    sum
                }
                Op::Mul(n) => {
                    let from = stack.len() - n;
                    let product = stack[from..]
                        .iter()
                        .try_fold(Poly::one(Coeff::ONE), |acc, p| mul(&acc, p))?;
                    stack.truncate(from);
                    product
                }
                Op::Div => {
                    let den = popped(&mut stack);
                    divided(popped(&mut stack), &den)?
                }
                Op::Pow(n) => power(&popped(&mut stack), *n)?,
                Op::Shift(by) => shift(&popped(&mut stack), *by),
            };
            stack.push(pushed);
        }
        stack.pop()
    }
}

fn popped<N: Num>(stack: &mut Vec<Poly<N>>) -> Poly<N> {
    stack
        .pop()
        .expect("a compiled op pops only what an earlier op pushed")
}

fn compile<N: Num>(
    f: &Body<N>,
    k: IndexId,
    reads: &dyn Reads<N>,
    ops: &mut Vec<Op<N>>,
) -> Option<()> {
    if !crate::series::mentions_over(f, k) {
        ops.push(Op::Push(read(f, Reading::Free, reads)?));
        return Some(());
    }
    let mut each = |parts: &[&Body<N>]| {
        parts
            .iter()
            .try_for_each(|part| compile(part, k, reads, &mut *ops))
    };
    let op = match f {
        Body::Index(_) => Op::Index,
        Body::Keyed { seed, of } => Op::Keyed(*seed, Tape::of(&of.body, k, reads)),
        Body::Apply(unary, arg) => {
            each(&[&arg.body])?;
            Op::Apply(*unary)
        }
        Body::Add(parts) | Body::Mul(parts) => {
            let bodies: Vec<&Body<N>> = parts.iter().map(|p| &*p.body).collect();
            each(&bodies)?;
            match f {
                Body::Add(_) => Op::Add(parts.len()),
                _ => Op::Mul(parts.len()),
            }
        }
        Body::Div(num, den) => {
            each(&[&num.body, &den.body])?;
            Op::Div
        }
        Body::Pow(base, n) => {
            let n = u32::try_from(*n).ok()?;
            each(&[&base.body])?;
            Op::Pow(n)
        }
        Body::Shift { by, of } => {
            each(&[&of.body])?;
            Op::Shift(*by)
        }
        _ => return None,
    };
    ops.push(op);
    Some(())
}

/// An index read in the free variable is an unknown real; read in itself, its own line.
fn read<N: Num>(f: &Body<N>, reading: Reading, reads: &dyn Reads<N>) -> Option<Poly<N>> {
    let polynomial = |g: &Body<N>| read(g, reading, reads);
    match f {
        Body::Node(id) => Poly::of(&reads.polynomial(*id, reading)?),
        Body::Const(c) => Some(Poly::one(Coeff::Exact(*c))),
        Body::Index(k) => Some(match reading {
            Reading::Index(wanted) if *k == wanted => Poly::of(&[Coeff::ZERO, Coeff::ONE])?,
            _ => Poly::one(Coeff::Unknown(Axis::Real)),
        }),
        Body::Line => match reading {
            Reading::Free => Poly::of(&[Coeff::ZERO, Coeff::ONE]),
            Reading::Index(_) => None,
        },
        Body::Keyed { seed, of } => Some(keyed(*seed, read(&of.body, reading, reads))),
        Body::Apply(op, arg) => applied(*op, &polynomial(&arg.body)?),
        Body::Add(parts) => parts.iter().try_fold(Poly::one(Coeff::ZERO), |acc, p| {
            Some(add(&acc, &polynomial(&p.body)?))
        }),
        Body::Mul(parts) => parts.iter().try_fold(Poly::one(Coeff::ONE), |acc, p| {
            mul(&acc, &polynomial(&p.body)?)
        }),
        Body::Div(num, den) => divided(polynomial(&num.body)?, &polynomial(&den.body)?),
        Body::Pow(base, n) => {
            let b = polynomial(&base.body)?;
            power(&b, u32::try_from(*n).ok()?)
        }
        Body::Shift { by, of } => Some(shift(&polynomial(&of.body)?, *by)),
        _ => None,
    }
}

/// A key the hash cannot draw at is an unknown real.
fn keyed<N: Num>(seed: u64, of: Option<Poly<N>>) -> Poly<N> {
    match of
        .and_then(constant)
        .and_then(|at| crate::hash::draw_nearest(seed, at.re.v()))
    {
        Some(drawn) => Poly::one(Coeff::Exact(Cx::real(N::of(drawn)))),
        None => Poly::one(Coeff::Unknown(Axis::Real)),
    }
}

fn applied<N: Num>(op: Unary, arg: &Poly<N>) -> Option<Poly<N>> {
    match arg.as_slice() {
        [c] => Some(Poly::one(match c.exact() {
            Some(x) => Coeff::Exact(apply_scalar_over(op, x)),
            None => Coeff::Unknown(unary_axis(op, c.axis())),
        })),
        _ => None,
    }
}

fn divided<N: Num>(mut num: Poly<N>, den: &Poly<N>) -> Option<Poly<N>> {
    let [d] = *den.as_slice() else {
        return None;
    };
    num.c[..num.len].iter_mut().for_each(|c| *c = c.div(d));
    Some(num)
}

fn power<N: Num>(base: &Poly<N>, n: u32) -> Option<Poly<N>> {
    (0..n).try_fold(Poly::one(Coeff::ONE), |acc, _| mul(&acc, base))
}

/// Ascending coefficients held in place: a polynomial read here never passes degree two.
#[derive(Clone, Copy)]
struct Poly<N: Num> {
    c: [Coeff<N>; MAX_DEGREE + 1],
    len: usize,
}

impl<N: Num> Poly<N> {
    fn zeros(len: usize) -> Poly<N> {
        Poly {
            c: [Coeff::ZERO; MAX_DEGREE + 1],
            len,
        }
    }

    fn one(c: Coeff<N>) -> Poly<N> {
        let mut p = Poly::zeros(1);
        p.c[0] = c;
        p
    }

    fn of(cs: &[Coeff<N>]) -> Option<Poly<N>> {
        let mut p = Poly::zeros(cs.len());
        p.c.get_mut(..cs.len())?.copy_from_slice(cs);
        Some(p)
    }

    fn as_slice(&self) -> &[Coeff<N>] {
        &self.c[..self.len]
    }
}

fn constant<N: Num>(p: Poly<N>) -> Option<Cx<N>> {
    match p.as_slice() {
        [b] => b.exact(),
        _ => None,
    }
}

/// Every unary this language folds on a scalar, in one place, so `normalize` and `infer`
/// cannot disagree about what a constant argument gives.
pub fn apply_scalar_over<N: Num>(op: Unary, x: Cx<N>) -> Cx<N> {
    match op {
        Unary::Sin => ((Cx::I * x).exp() - (-Cx::I * x).exp()) / Cx::I.scale(N::of(2.0)),
        Unary::Cos => ((Cx::I * x).exp() + (-Cx::I * x).exp()).scale(N::of(0.5)),
        Unary::Exp => x.exp(),
        Unary::Tanh => {
            let (p, m) = (x.exp(), (-x).exp());
            (p - m) / (p + m)
        }
        Unary::Sat => Cx::real(x.re.clamp(-1.0, 1.0)),
        Unary::Abs => Cx::real(x.abs()),
        Unary::Log => Cx::new(x.abs().ln(), x.im.atan2(x.re)),
        Unary::Sqrt => {
            let r = x.abs().sqrt();
            let half = x.im.atan2(x.re) / 2.0;
            Cx::new(r * half.cos(), r * half.sin())
        }
        Unary::Step => Cx::real(N::of(step(x.re.v()))),
    }
}

fn constant_in<N: Num>(f: &Body<N>, reading: Reading) -> Option<Cx<N>> {
    constant_read(f, reading, &Opaque)
}

fn constant_read<N: Num>(f: &Body<N>, reading: Reading, reads: &dyn Reads<N>) -> Option<Cx<N>> {
    constant(read(f, reading, reads)?)
}

pub fn affine_over<N: Num>(f: &Body<N>) -> Option<(Coeff<N>, Coeff<N>)> {
    affine_in_over(f, Reading::Free)
}

pub fn affine_in_over<N: Num>(f: &Body<N>, reading: Reading) -> Option<(Coeff<N>, Coeff<N>)> {
    affine_read_over(f, reading, &Opaque)
}

pub fn affine_read_over<N: Num>(
    f: &Body<N>,
    reading: Reading,
    reads: &dyn Reads<N>,
) -> Option<(Coeff<N>, Coeff<N>)> {
    match polynomial_read_over(f, reading, reads)?[..] {
        [b] => Some((Coeff::ZERO, b)),
        [b, a] => Some((a, b)),
        _ => None,
    }
}

/// The offset of a time spelled as the variable plus a constant, an index counting as one.
pub fn slide_over<N: Num>(at: &Body<N>) -> Option<Coeff<N>> {
    slide_read(at, &Opaque)
}

/// A slope of exactly one, moved by no varied number: anything else stretches time.
pub(crate) fn slide_read<N: Num>(at: &Body<N>, reads: &dyn Reads<N>) -> Option<Coeff<N>> {
    match affine_read_over(at, Reading::Free, reads)? {
        (Coeff::Exact(slope), offset)
            if slope.re.v() == 1.0 && slope.re.is_constant() && slope.im.is_zero() =>
        {
            Some(offset)
        }
        _ => None,
    }
}

pub fn exact_affine_over<N: Num>(f: &Body<N>) -> Option<(Cx<N>, Cx<N>)> {
    exact_affine_read(f, &Opaque)
}

pub(crate) fn exact_affine_read<N: Num>(
    f: &Body<N>,
    reads: &dyn Reads<N>,
) -> Option<(Cx<N>, Cx<N>)> {
    let (a, b) = affine_read_over(f, Reading::Free, reads)?;
    Some((a.exact()?, b.exact()?))
}

pub fn exact_constant_over<N: Num>(f: &Body<N>) -> Option<Cx<N>> {
    constant_in(f, Reading::Free)
}

pub fn exact_constant_read_over<N: Num>(f: &Body<N>, reads: &dyn Reads<N>) -> Option<Cx<N>> {
    constant_read(f, Reading::Free, reads)
}

/// Whether a hash keyed on this formula names more than one value. The free variable moves
/// it; so does a node or a parameter, whose own closed form this judgment cannot see.
pub fn key_moves_over<N: Num>(f: &Body<N>) -> bool {
    key_moves_read(f, &Opaque)
}

pub(crate) fn key_moves_read<N: Num>(f: &Body<N>, reads: &dyn Reads<N>) -> bool {
    match f {
        Body::Line | Body::Param(_) => true,
        Body::Node(id) => reads.key_moves(*id),
        other => crate::closed_form::children(other)
            .iter()
            .any(|p| key_moves_read(&p.body, reads)),
    }
}

/// `exp(q)` with the square completed: a real width and centre, the rest an exponential.
pub struct Squared<N: Num = f64> {
    pub a: N,
    pub mu: N,
    pub omega: N,
    pub amplitude: Cx<N>,
}

pub fn completed_square_over<N: Num>(f: &Body<N>) -> Option<Squared<N>> {
    completed_square_read(f, &Opaque)
}

pub(crate) fn completed_square_read<N: Num>(
    f: &Body<N>,
    reads: &dyn Reads<N>,
) -> Option<Squared<N>> {
    let [a0, a1, a2] = polynomial_read_over(f, Reading::Free, reads)?[..] else {
        return None;
    };
    let (a0, a1, a2) = (a0.exact()?, a1.exact()?, a2.exact()?);
    if !a2.is_real() || a2.re.v() >= 0.0 {
        return None;
    }
    let a = -a2.re;
    let mu = a1.re / (N::of(2.0) * a);
    Some(Squared {
        a,
        mu,
        omega: a1.im,
        amplitude: (a0 + Cx::real(a * mu * mu)).exp(),
    })
}

pub fn is_squared_over<N: Num>(f: &Body<N>) -> bool {
    polynomial_over(f).is_some_and(|p| p.len() == MAX_DEGREE + 1)
}

/// Where a unary lands a subterm sitting on `of`. A root and a logarithm leave the real
/// line on a negative argument, whose sign this judgment does not hold; the rest do not.
fn unary_axis(op: Unary, of: Axis) -> Axis {
    match (op, of) {
        (Unary::Sqrt | Unary::Log, _) => Axis::Any,
        (_, Axis::Real) => Axis::Real,
        _ => Axis::Any,
    }
}

/// Where a whole subterm sits in the complex plane, reading an index or a bare variable as
/// real. `Any` is "could be either", never "is both".
pub fn axis<N: Num>(f: &Body<N>, env: &dyn crate::env::Env) -> Axis {
    axis_read(f, env, &Opaque)
}

/// The same, a ref `reads` knows the form of sitting where that form sits.
pub fn axis_read<N: Num>(f: &Body<N>, env: &dyn crate::env::Env, reads: &dyn Reads<N>) -> Axis {
    let of = |p: &crate::closed_form::Part<N>| axis_read(&p.body, env, reads);
    let axis = |f: &Body<N>| axis_read(f, env, reads);
    match f {
        Body::Const(c) => Axis::of(*c),
        Body::Index(_) | Body::Line | Body::Keyed { .. } | Body::Knots { .. } => Axis::Real,
        Body::Node(id) => reads
            .axis(*id)
            .unwrap_or_else(|| Axis::of_codomain(env.node(*id).codomain)),
        Body::Param(id) => Axis::of_codomain(env.param(*id).codomain),
        Body::Add(parts) => parts
            .iter()
            .map(of)
            .reduce(Axis::plus)
            .unwrap_or(Axis::Real),
        Body::Mul(parts) => parts
            .iter()
            .map(of)
            .reduce(Axis::times)
            .unwrap_or(Axis::Real),
        Body::Div(a, b) => of(a).times(of(b)),
        Body::Shift { of: inner, .. } | Body::Crop { of: inner, .. } => axis(&inner.body),
        Body::Fold(..) => Axis::Real,
        Body::Apply(op, inner) => unary_axis(*op, axis(&inner.body)),
        Body::Pow(base, n) => match (axis(&base.body), n % 2 == 0) {
            (Axis::Real, _) => Axis::Real,
            (Axis::Imaginary, true) => Axis::Real,
            (Axis::Imaginary, false) => Axis::Imaginary,
            _ => Axis::Any,
        },
        _ => Axis::Any,
    }
}

fn add<N: Num>(x: &Poly<N>, y: &Poly<N>) -> Poly<N> {
    let mut out = Poly::zeros(x.len.max(y.len));
    for (k, c) in x
        .as_slice()
        .iter()
        .enumerate()
        .chain(y.as_slice().iter().enumerate())
    {
        out.c[k] = out.c[k].add(*c);
    }
    trim(out)
}

fn mul<N: Num>(x: &Poly<N>, y: &Poly<N>) -> Option<Poly<N>> {
    if x.len + y.len > MAX_DEGREE + 2 {
        return None;
    }
    let mut out = Poly::zeros(x.len + y.len - 1);
    for (i, a) in x.as_slice().iter().enumerate() {
        for (j, b) in y.as_slice().iter().enumerate() {
            out.c[i + j] = out.c[i + j].add(a.mul(*b));
        }
    }
    Some(trim(out))
}

/// `x -> x - by`, by the binomial expansion of each term.
fn shift<N: Num>(p: &Poly<N>, by: N) -> Poly<N> {
    let mut out = Poly::zeros(p.len);
    for (n, c) in p.as_slice().iter().enumerate() {
        let mut binomial = 1.0f64;
        for k in (0..=n).rev() {
            let power = i32::try_from(n - k).unwrap_or(0);
            out.c[k] = out.c[k].add(c.scale(N::of(binomial) * (-by).powi(power)));
            binomial = binomial * k as f64 / (n - k + 1) as f64;
        }
    }
    trim(out)
}

fn trim<N: Num>(mut p: Poly<N>) -> Poly<N> {
    while p.len > 1 && p.c[p.len - 1].is_zero() {
        p.len -= 1;
    }
    p
}

pub fn step(x: f64) -> f64 {
    match x >= 0.0 {
        true => 1.0,
        false => 0.0,
    }
}

pub fn polynomial(f: &Body) -> Option<Vec<Coeff>> {
    polynomial_over(f)
}

pub fn polynomial_in(f: &Body, reading: Reading) -> Option<Vec<Coeff>> {
    polynomial_in_over(f, reading)
}

pub fn polynomial_read(f: &Body, reading: Reading, reads: &dyn Reads) -> Option<Vec<Coeff>> {
    polynomial_read_over(f, reading, reads)
}

pub fn apply_scalar(op: Unary, x: C64) -> C64 {
    apply_scalar_over(op, x)
}

pub fn affine(f: &Body) -> Option<(Coeff, Coeff)> {
    affine_over(f)
}

pub fn affine_in(f: &Body, reading: Reading) -> Option<(Coeff, Coeff)> {
    affine_in_over(f, reading)
}

pub fn affine_read(f: &Body, reading: Reading, reads: &dyn Reads) -> Option<(Coeff, Coeff)> {
    affine_read_over(f, reading, reads)
}

pub fn slide(at: &Body) -> Option<Coeff> {
    slide_over(at)
}

pub fn exact_affine(f: &Body) -> Option<(C64, C64)> {
    exact_affine_over(f)
}

pub fn exact_constant(f: &Body) -> Option<C64> {
    exact_constant_over(f)
}

pub fn exact_constant_read(f: &Body, reads: &dyn Reads) -> Option<C64> {
    exact_constant_read_over(f, reads)
}

pub fn key_moves(f: &Body) -> bool {
    key_moves_over(f)
}

pub fn completed_square(f: &Body) -> Option<Squared> {
    completed_square_over(f)
}

pub fn is_squared(f: &Body) -> bool {
    is_squared_over(f)
}
