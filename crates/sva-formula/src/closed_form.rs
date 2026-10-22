// Concern: declares ClosedForm and its Body, as written before normalization | Non-concern: the canonical atom sum (spectral_sum/) | IO: none

use crate::complex::Cx;
use crate::env::{NodeId, ParamId};
use crate::num::Num;
use crate::origin::Origin;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Var {
    T,
    F,
}

impl Var {
    pub fn as_str(self) -> &'static str {
        match self {
            Var::T => "t",
            Var::F => "f",
        }
    }
}

/// One `Var` for the whole of it: `t` and `f` never meet inside one.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedForm<N: Num = f64> {
    pub var: Var,
    pub body: Body<N>,
    pub origin: Origin,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Part<N: Num = f64> {
    pub origin: Origin,
    pub body: Box<Body<N>>,
}

impl<N: Num> Part<N> {
    pub fn new(origin: Origin, body: Body<N>) -> Part<N> {
        Part {
            origin,
            body: Box::new(body),
        }
    }

    pub fn bare(body: Body<N>) -> Part<N> {
        Part::new(Origin::UNKNOWN, body)
    }
}

/// An infinite bound is a variant, so no sort key or hash ever sees a non-finite `f64`; a
/// finite one is held canonical, `-0.0` as `0.0`.
#[derive(Clone, Copy, Debug)]
pub enum Edge<N: Num = f64> {
    NegInf,
    At(N),
    PosInf,
}

impl<N: Num> PartialEq for Edge<N> {
    fn eq(&self, other: &Edge<N>) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl<N: Num> Eq for Edge<N> {}

/// Numeric: by bit pattern a negative bound sorts above every positive one. Two edges at
/// one value moved differently by the varied numbers are two edges.
impl<N: Num> Ord for Edge<N> {
    fn cmp(&self, other: &Edge<N>) -> std::cmp::Ordering {
        let moved = |e: &Edge<N>| match e {
            Edge::At(x) => Some(x.key()),
            _ => None,
        };
        self.value()
            .total_cmp(&other.value())
            .then_with(|| moved(self).cmp(&moved(other)))
    }
}

impl<N: Num> PartialOrd for Edge<N> {
    fn partial_cmp(&self, other: &Edge<N>) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<N: Num> Edge<N> {
    pub fn at(x: N) -> Edge<N> {
        match x.v() {
            v if v == f64::NEG_INFINITY => Edge::NegInf,
            v if v == f64::INFINITY => Edge::PosInf,
            _ => Edge::At(x.canonical()),
        }
    }

    pub fn value(self) -> f64 {
        match self {
            Edge::NegInf => f64::NEG_INFINITY,
            Edge::At(x) => x.v(),
            Edge::PosInf => f64::INFINITY,
        }
    }

    /// The edge as a number, an infinite one included.
    pub fn number(self) -> N {
        match self {
            Edge::At(x) => x,
            infinite => N::of(infinite.value()),
        }
    }

    pub fn of(e: Edge) -> Edge<N> {
        match e {
            Edge::NegInf => Edge::NegInf,
            Edge::At(x) => Edge::At(N::of(x)),
            Edge::PosInf => Edge::PosInf,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unary {
    Sin,
    Cos,
    Exp,
    Tanh,
    Sat,
    Abs,
    Log,
    Sqrt,
    /// `step(0) = 1`, as a crop's `[l, r)` opens.
    Step,
}

impl Unary {
    pub fn name(self) -> &'static str {
        match self {
            Unary::Sin => "sin",
            Unary::Cos => "cos",
            Unary::Exp => "exp",
            Unary::Tanh => "tanh",
            Unary::Sat => "sat",
            Unary::Abs => "abs",
            Unary::Log => "log",
            Unary::Sqrt => "sqrt",
            Unary::Step => "step",
        }
    }

    pub const ALL: &'static [Unary] = &[
        Unary::Sin,
        Unary::Cos,
        Unary::Exp,
        Unary::Tanh,
        Unary::Sat,
        Unary::Abs,
        Unary::Log,
        Unary::Sqrt,
        Unary::Step,
    ];

    pub fn from_name(name: &str) -> Option<Unary> {
        Unary::ALL.iter().copied().find(|u| u.name() == name)
    }

    /// A is closed under these three on an affine argument and under nothing else.
    pub fn is_closed(self) -> bool {
        matches!(self, Unary::Sin | Unary::Cos | Unary::Exp)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    Max,
    Min,
    Mod,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Finite(i64),
    Infinite,
}

/// A value, not an expansion: truncated once, at collapse.
#[derive(Clone, Debug, PartialEq)]
pub struct Series<N: Num = f64> {
    pub index: IndexId,
    pub lo: i64,
    pub hi: Bound,
    pub term: Part<N>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mode {
    pub omega: f64,
    pub tau: f64,
    pub amp: f64,
    pub phase: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Excitation {
    HammerPulse { f0: f64, t0: f64, contact: f64 },
    Impulse { t0: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModalBank {
    pub modes: Vec<Mode>,
    pub excite: Excitation,
}

/// Residues expand on demand, never stored, so one rational has one spelling.
#[derive(Clone, Debug, PartialEq)]
pub struct Rational<N: Num = f64> {
    pub zeros: Vec<Cx<N>>,
    pub poles: Vec<Cx<N>>,
    pub gain: Cx<N>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Body<N: Num = f64> {
    Const(Cx<N>),
    Line,
    Index(IndexId),
    /// The unit-interval hash `rand` folds to, and the phase a noise line reads.
    Keyed {
        seed: u64,
        of: Part<N>,
    },
    Param(ParamId),
    Node(NodeId),
    Add(Vec<Part<N>>),
    Mul(Vec<Part<N>>),
    Div(Part<N>, Part<N>),
    Pow(Part<N>, i32),
    Apply(Unary, Part<N>),
    Fold(Fold, Vec<Part<N>>),
    Delta {
        at: Part<N>,
        order: u16,
    },
    Pv(Part<N>),
    Shift {
        by: N,
        of: Part<N>,
    },
    /// `of` read at the time `at` names, which is a shift only where `at` is `t` plus a
    /// constant.
    Warp {
        at: Part<N>,
        of: Part<N>,
    },
    Deriv {
        order: u16,
        of: Part<N>,
    },
    /// `rise` and `fall` are raised-cosine shoulders inside `[l, r)`; zero is a hard edge.
    Crop {
        of: Part<N>,
        l: Edge<N>,
        r: Edge<N>,
        rise: N,
        fall: N,
    },
    Join(Vec<Part<N>>),
    Channel(Part<N>, u8),
    Rational(Rational<N>),
    Series(Box<Series<N>>),
    Modal(ModalBank),
    Run(Box<crate::run::Run>),
    Banded(Box<crate::banded::Banded<N>>),
    /// The table read at the instant `at` names, a crop's window read at the free variable.
    Knots {
        at: Part<N>,
        table: std::sync::Arc<crate::knots::Knots<N>>,
    },
}

/// The same form over another number, moved by no varied number.
pub fn lifted<N: Num>(f: &Body) -> Body<N> {
    mapped(f, &N::of)
}

/// The same form, every number's tangent dropped.
pub fn value<N: Num>(f: &Body<N>) -> Body {
    mapped(f, &|x: N| x.v())
}

/// The same form, each number written through `num`.
fn mapped<M: Num, N: Num>(f: &Body<M>, num: &dyn Fn(M) -> N) -> Body<N> {
    let part = |p: &Part<M>| Part::new(p.origin, mapped(&p.body, num));
    let parts = |ps: &[Part<M>]| ps.iter().map(part).collect();
    let series = |s: &Series<M>| Series {
        index: s.index,
        lo: s.lo,
        hi: s.hi,
        term: part(&s.term),
    };
    let edge = |e: Edge<M>| match e {
        Edge::NegInf => Edge::NegInf,
        Edge::At(x) => Edge::At(num(x)),
        Edge::PosInf => Edge::PosInf,
    };
    let complex = |c: Cx<M>| Cx::new(num(c.re), num(c.im));
    match f {
        Body::Const(c) => Body::Const(complex(*c)),
        Body::Line => Body::Line,
        Body::Index(k) => Body::Index(*k),
        Body::Keyed { seed, of } => Body::Keyed {
            seed: *seed,
            of: part(of),
        },
        Body::Param(id) => Body::Param(*id),
        Body::Node(id) => Body::Node(*id),
        Body::Add(ps) => Body::Add(parts(ps)),
        Body::Mul(ps) => Body::Mul(parts(ps)),
        Body::Div(a, b) => Body::Div(part(a), part(b)),
        Body::Pow(of, n) => Body::Pow(part(of), *n),
        Body::Apply(op, of) => Body::Apply(*op, part(of)),
        Body::Fold(op, ps) => Body::Fold(*op, parts(ps)),
        Body::Delta { at, order } => Body::Delta {
            at: part(at),
            order: *order,
        },
        Body::Pv(at) => Body::Pv(part(at)),
        Body::Shift { by, of } => Body::Shift {
            by: num(*by),
            of: part(of),
        },
        Body::Warp { at, of } => Body::Warp {
            at: part(at),
            of: part(of),
        },
        Body::Deriv { order, of } => Body::Deriv {
            order: *order,
            of: part(of),
        },
        Body::Crop {
            of,
            l,
            r,
            rise,
            fall,
        } => Body::Crop {
            of: part(of),
            l: edge(*l),
            r: edge(*r),
            rise: num(*rise),
            fall: num(*fall),
        },
        Body::Join(ps) => Body::Join(parts(ps)),
        Body::Channel(of, k) => Body::Channel(part(of), *k),
        Body::Rational(r) => Body::Rational(Rational {
            zeros: r.zeros.iter().map(|z| complex(*z)).collect(),
            poles: r.poles.iter().map(|p| complex(*p)).collect(),
            gain: complex(r.gain),
        }),
        Body::Series(s) => Body::Series(Box::new(series(s))),
        Body::Modal(m) => Body::Modal(m.clone()),
        Body::Run(run) => Body::Run(run.clone()),
        Body::Banded(b) => Body::Banded(Box::new(crate::banded::Banded {
            series: series(&b.series),
            slope: b.slope.clone(),
            offset: b.offset.clone(),
            omega: b.omega,
            most: b.most,
            widest: b.widest,
            reach: b.reach,
            dropped_db: b.dropped_db,
        })),
        Body::Knots { at, table } => Body::Knots {
            at: part(at),
            table: std::sync::Arc::new(table.mapped(num)),
        },
    }
}

/// The subterms of one node: a walker states its recursion once, and every new variant
/// reaches every walker at once.
pub fn children<N: Num>(f: &Body<N>) -> Vec<&Part<N>> {
    match f {
        Body::Const(_)
        | Body::Line
        | Body::Index(_)
        | Body::Param(_)
        | Body::Node(_)
        | Body::Rational(_)
        | Body::Modal(_)
        | Body::Run(_) => Vec::new(),
        Body::Keyed { of, .. } => vec![of],
        Body::Add(parts) | Body::Mul(parts) | Body::Join(parts) | Body::Fold(_, parts) => {
            parts.iter().collect()
        }
        Body::Div(a, b) | Body::Warp { at: a, of: b } => vec![a, b],
        Body::Pow(of, _)
        | Body::Apply(_, of)
        | Body::Pv(of)
        | Body::Channel(of, _)
        | Body::Delta { at: of, .. }
        | Body::Shift { of, .. }
        | Body::Deriv { of, .. }
        | Body::Crop { of, .. }
        | Body::Knots { at: of, .. } => vec![of],
        Body::Series(s) => vec![&s.term],
        Body::Banded(b) => vec![&b.series.term],
    }
}

pub fn map_children<N: Num>(f: &Body<N>, mut g: impl FnMut(&Part<N>) -> Part<N>) -> Body<N> {
    let one = |p: &Part<N>, g: &mut dyn FnMut(&Part<N>) -> Part<N>| g(p);
    match f {
        Body::Const(_)
        | Body::Line
        | Body::Index(_)
        | Body::Param(_)
        | Body::Node(_)
        | Body::Rational(_)
        | Body::Modal(_)
        | Body::Run(_) => f.clone(),
        Body::Keyed { seed, of } => Body::Keyed {
            seed: *seed,
            of: one(of, &mut g),
        },
        Body::Add(parts) => Body::Add(parts.iter().map(g).collect()),
        Body::Mul(parts) => Body::Mul(parts.iter().map(g).collect()),
        Body::Join(parts) => Body::Join(parts.iter().map(g).collect()),
        Body::Fold(op, parts) => Body::Fold(*op, parts.iter().map(g).collect()),
        Body::Div(a, b) => Body::Div(g(a), g(b)),
        Body::Warp { at, of } => Body::Warp {
            at: g(at),
            of: g(of),
        },
        Body::Pow(of, n) => Body::Pow(g(of), *n),
        Body::Apply(op, of) => Body::Apply(*op, g(of)),
        Body::Pv(of) => Body::Pv(g(of)),
        Body::Channel(of, k) => Body::Channel(g(of), *k),
        Body::Delta { at, order } => Body::Delta {
            at: g(at),
            order: *order,
        },
        Body::Shift { by, of } => Body::Shift { by: *by, of: g(of) },
        Body::Deriv { order, of } => Body::Deriv {
            order: *order,
            of: g(of),
        },
        Body::Knots { at, table } => Body::Knots {
            at: g(at),
            table: table.clone(),
        },
        Body::Crop {
            of,
            l,
            r,
            rise,
            fall,
        } => Body::Crop {
            of: g(of),
            l: *l,
            r: *r,
            rise: *rise,
            fall: *fall,
        },
        Body::Series(s) => Body::Series(Box::new(Series {
            term: g(&s.term),
            ..(**s).clone()
        })),
        Body::Banded(b) => {
            let mut held = (**b).clone();
            held.series.term = g(&b.series.term);
            Body::Banded(Box::new(held))
        }
    }
}

/// `1[l,r)(u - by)` reads the free variable too, so a window moves its own bounds; a warp
/// moves only its time.
pub fn shift_line_over<N: Num>(f: &Body<N>, by: N) -> Body<N> {
    let walk = |p: &Part<N>| Part::new(p.origin, shift_line_over(&p.body, by));
    match f {
        Body::Line => Body::Add(vec![
            Part::bare(Body::Line),
            Part::bare(Body::Const(Cx::real(-by))),
        ]),
        Body::Crop {
            of,
            l,
            r,
            rise,
            fall,
        } => Body::Crop {
            of: walk(of),
            l: Edge::at(l.number() + by),
            r: Edge::at(r.number() + by),
            rise: *rise,
            fall: *fall,
        },
        Body::Warp { at, of } => Body::Warp {
            at: walk(at),
            of: of.clone(),
        },
        Body::Node(_) => Body::Shift {
            by,
            of: Part::bare(f.clone()),
        },
        other => map_children(other, walk),
    }
}

/// An edge holds no formula, so a window is warped whole, and so is a ref.
pub fn read_at_over<N: Num>(f: &Body<N>, at: &Body<N>) -> Body<N> {
    read_at_with_over(f, at, &|_| None)
}

pub fn read_at_with_over<N: Num>(
    f: &Body<N>,
    at: &Body<N>,
    node: &dyn Fn(NodeId) -> Option<NodeId>,
) -> Body<N> {
    match f {
        Body::Line => at.clone(),
        Body::Node(id) if let Some(moved) = node(*id) => Body::Node(moved),
        Body::Crop { .. } | Body::Node(_) => Body::Warp {
            at: Part::bare(at.clone()),
            of: Part::bare(f.clone()),
        },
        Body::Warp { at: inner, of } => Body::Warp {
            at: Part::new(inner.origin, read_at_with_over(&inner.body, at, node)),
            of: of.clone(),
        },
        other => map_children(other, |p| {
            Part::new(p.origin, read_at_with_over(&p.body, at, node))
        }),
    }
}

/// A leaf substitution cannot walk into: the free variable hides behind a name.
pub fn shifts_opaquely<N: Num>(f: &Body<N>) -> bool {
    matches!(
        f,
        Body::Param(_) | Body::Rational(_) | Body::Modal(_) | Body::Run(_) | Body::Banded(_)
    ) || children(f).iter().any(|p| shifts_opaquely(&p.body))
}

pub fn shift_line(f: &Body, by: f64) -> Body {
    shift_line_over(f, by)
}

pub fn read_at(f: &Body, at: &Body) -> Body {
    read_at_over(f, at)
}

pub fn read_at_with(f: &Body, at: &Body, node: &dyn Fn(NodeId) -> Option<NodeId>) -> Body {
    read_at_with_over(f, at, node)
}
