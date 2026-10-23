// Concern: splits a form at a min or max of two lines crossing, and lists where slopes may jump | Non-concern: lowering either half (build.rs) | IO: (&Body) -> two crops or nothing; (&Body) -> instants

use crate::affine::exact_affine_read;
use crate::closed_form::{Body, Edge, Fold, Part, map_children};
use crate::complex::Cx;
use crate::num::Num;
use crate::through::{Opaque, Reads};

pub fn split<N: Num>(f: &Body<N>, reads: &dyn Reads<N>) -> Option<Body<N>> {
    let (kink, (at, below, above)) = first(f, reads)?;
    let half = |with: &Body<N>, l: Edge<N>, r: Edge<N>| {
        let scope = reads.scope();
        Part::bare(Body::Crop {
            of: Part::bare(replaced(f, (scope, &kink, with), reads)),
            l,
            r,
            rise: N::ZERO,
            fall: N::ZERO,
        })
    };
    Some(Body::Add(vec![
        half(&below, Edge::NegInf, Edge::at(at)),
        half(&above, Edge::at(at), Edge::PosInf),
    ]))
}

/// A constant, or a slope and the instant it is zero at: `k*(t - r)` keeps `r` itself. Each
/// fold of a written constant rounds once, at most 2^-53 of its result, as `exact_affine`.
#[derive(Clone, Copy, Debug)]
pub enum Line<N: Num = f64> {
    Flat(N),
    Sloped { slope: N, root: N },
}

impl<N: Num> Line<N> {
    fn slope(self) -> N {
        match self {
            Line::Flat(_) => N::ZERO,
            Line::Sloped { slope, .. } => slope,
        }
    }

    fn scaled(self, k: N, divide: bool) -> Line<N> {
        let by = |v: N| if divide { v / k } else { v * k };
        match self {
            Line::Flat(v) => Line::Flat(by(v)),
            Line::Sloped { slope, root } => Line::Sloped {
                slope: by(slope),
                root,
            },
        }
    }

    fn plus(self, other: Line<N>) -> Line<N> {
        match (self, other) {
            (Line::Flat(a), Line::Flat(b)) => Line::Flat(a + b),
            (Line::Flat(c), Line::Sloped { slope, root })
            | (Line::Sloped { slope, root }, Line::Flat(c)) => match c.is_zero() {
                true => Line::Sloped { slope, root },
                false => Line::Sloped {
                    slope,
                    root: root - c / slope,
                },
            },
            (Line::Sloped { slope: s, root: r }, Line::Sloped { slope: u, root: q }) => {
                match (s + u, r == q) {
                    (sum, _) if sum.is_zero() => Line::Flat(-(s * r + u * q)),
                    (sum, true) => Line::Sloped {
                        slope: sum,
                        root: r,
                    },
                    (sum, false) => Line::Sloped {
                        slope: sum,
                        root: (s * r + u * q) / sum,
                    },
                }
            }
        }
    }
}

pub(crate) fn line<N: Num>(f: &Body<N>, reads: &dyn Reads<N>) -> Option<Line<N>> {
    if let Body::Node(id) = f {
        return reads.line(*id);
    }
    let line = |g: &Body<N>| line(g, reads);
    let real = |c: Cx<N>| c.is_real().then_some(c.re);
    let read = match f {
        Body::Const(c) => Some(Line::Flat(real(*c)?)),
        Body::Line => Some(Line::Sloped {
            slope: N::ONE,
            root: N::ZERO,
        }),
        Body::Shift { by, of } => match line(&of.body)? {
            Line::Sloped { slope, root } => Some(Line::Sloped {
                slope,
                root: root + *by,
            }),
            flat => Some(flat),
        },
        Body::Add(parts) => parts
            .iter()
            .try_fold(Line::Flat(N::ZERO), |acc, p| Some(acc.plus(line(&p.body)?))),
        Body::Mul(parts) => {
            let mut sloped: Option<Line<N>> = None;
            let mut k = N::ONE;
            for p in parts {
                match line(&p.body)? {
                    Line::Flat(v) => k = k * v,
                    held if sloped.is_none() => sloped = Some(held),
                    _ => return None,
                }
            }
            Some(sloped.map_or(Line::Flat(k), |l| l.scaled(k, false)))
        }
        Body::Div(num, den) => match line(&den.body)? {
            Line::Flat(k) if k.v() != 0.0 => Some(line(&num.body)?.scaled(k, true)),
            _ => None,
        },
        _ => None,
    };
    let read = read.or_else(|| {
        let (a, b) = exact_affine_read(f, reads)?;
        match (real(a)?, real(b)?) {
            (a, b) if a.is_zero() => Some(Line::Flat(b)),
            (a, b) => Some(Line::Sloped {
                slope: a,
                root: -b / a,
            }),
        }
    })?;
    match read {
        Line::Sloped { slope, .. } if slope.is_zero() => Some(Line::Flat(N::ZERO)),
        held => Some(held),
    }
    .filter(|l| match *l {
        Line::Flat(v) => v.v().is_finite(),
        Line::Sloped { slope, root } => slope.v().is_finite() && root.v().is_finite(),
    })
}

/// Each half is its own line, read from its root. Against a constant, the crossing is the
/// first double where the line as evaluated passes it, so a `max` never falls under nor a
/// `min` climbs over it, and a line meeting zero crosses at its root.
fn crossing<N: Num>(f: &Body<N>, reads: &dyn Reads<N>) -> Option<Crossing<N>> {
    let Body::Fold(op @ (Fold::Min | Fold::Max), args) = f else {
        return None;
    };
    let [p, q] = args.as_slice() else {
        return None;
    };
    let (l1, l2) = (line(&p.body, reads)?, line(&q.body, reads)?);
    if l1.slope().v() == l2.slope().v() {
        return None;
    }
    let max = *op == Fold::Max;
    match (l1, l2) {
        (Line::Flat(c), sloped @ Line::Sloped { slope, root })
        | (sloped @ Line::Sloped { slope, root }, Line::Flat(c)) => {
            let (k, r, level) = (slope.v(), root.v(), c.v());
            let taken = |t: f64| {
                let v = k * (t - r);
                if max { v >= level } else { v <= level }
            };
            let flat = Body::Const(Cx::real(c));
            let estimate = root + c / slope;
            let at = |found: f64| estimate.rebased(found);
            match (k > 0.0) == max {
                true => Some((
                    at(first_double(taken, estimate.v())?),
                    flat,
                    written(sloped),
                )),
                false => Some((
                    at(first_double(|t| !taken(t), estimate.v())?),
                    written(sloped),
                    flat,
                )),
            }
        }
        (Line::Sloped { slope: s, root: r }, Line::Sloped { slope: u, root: v }) => {
            let at = match r == v {
                true => r,
                false => (s * r - u * v) / (s - u),
            };
            let p_below = (s.v() > u.v()) != max;
            let (below, above) = if p_below { (l1, l2) } else { (l2, l1) };
            at.v()
                .is_finite()
                .then(|| (at, written(below), written(above)))
        }
        (Line::Flat(_), Line::Flat(_)) => None,
    }
}

fn written<N: Num>(l: Line<N>) -> Body<N> {
    match l {
        Line::Flat(c) => Body::Const(Cx::real(c)),
        Line::Sloped { slope, root } => Body::Mul(vec![
            Part::bare(Body::Const(Cx::real(slope))),
            Part::bare(Body::Shift {
                by: root,
                of: Part::bare(Body::Line),
            }),
        ]),
    }
}

/// Doubles in numeric order as integers.
fn ordered(x: i64) -> i64 {
    x ^ (((x >> 63) as u64) >> 1) as i64
}

fn key(x: f64) -> i64 {
    ordered(x.to_bits() as i64)
}

fn double(k: i64) -> f64 {
    f64::from_bits(ordered(k) as u64)
}

/// The least finite double where `holds`, monotone over the doubles, turns true, galloping
/// out from `near` and bisecting; nothing where it never turns within the finite doubles.
fn first_double(holds: impl Fn(f64) -> bool, near: f64) -> Option<f64> {
    let finite = |k: i64| double(k).is_finite().then_some(k);
    let near = finite(key(near))?;
    let mut step = 1i64;
    let (mut no, mut yes) = (near, near);
    match holds(double(near)) {
        true => loop {
            no = finite(yes.checked_sub(step)?)?;
            if !holds(double(no)) {
                break;
            }
            yes = no;
            step = step.checked_mul(2)?;
        },
        false => loop {
            yes = finite(no.checked_add(step)?)?;
            if holds(double(yes)) {
                break;
            }
            no = yes;
            step = step.checked_mul(2)?;
        },
    }
    while yes - no > 1 {
        let mid = no + (yes - no) / 2;
        match holds(double(mid)) {
            true => yes = mid,
            false => no = mid,
        }
    }
    Some(double(yes))
}

/// A shift, warp or series moves the kink; a derivative would add a delta at the seam.
pub(crate) fn in_time<N: Num>(f: &Body<N>) -> bool {
    matches!(
        f,
        Body::Add(_)
            | Body::Mul(_)
            | Body::Div(..)
            | Body::Pow(..)
            | Body::Apply(..)
            | Body::Fold(..)
            | Body::Crop { .. }
            | Body::Join(_)
            | Body::Channel(..)
    )
}

pub type Crossing<N = f64> = (N, Body<N>, Body<N>);

/// Each instant at which `f`'s slope in `t` may jump: where a min or max of two lines crosses,
/// a crop's edges and shoulders' ends, a knot of a table read at a line. Searched as `first`
/// searches, refs unread; an instant listed where the slope holds splits nothing it needs.
pub fn corners<N: Num>(f: &Body<N>) -> Vec<f64> {
    let mut out = Vec::new();
    corners_into(f, &mut out);
    out
}

fn corners_into<N: Num>(f: &Body<N>, out: &mut Vec<f64>) {
    if let Some((at, _, _)) = crossing(f, &Opaque) {
        out.push(at.v());
    }
    match f {
        Body::Crop {
            l, r, rise, fall, ..
        } => {
            let at = |e: &Edge<N>| match e {
                Edge::At(x) => Some(x.v()),
                Edge::NegInf | Edge::PosInf => None,
            };
            let (l, r, rise, fall) = (at(l), at(r), rise.v(), fall.v());
            out.extend(l.into_iter().chain(r));
            if let Some(l) = l.filter(|_| rise > 0.0) {
                out.push(l + rise);
            }
            if let Some(r) = r.filter(|_| fall > 0.0) {
                out.push(r - fall);
            }
            if let (Some(l), Some(r), true) = (l, r, rise > 0.0 && fall > 0.0) {
                out.push((l * fall + r * rise) / (rise + fall));
            }
        }
        Body::Knots { at, table } => {
            if let Some(Line::Sloped { slope, root }) = line(&at.body, &Opaque) {
                out.extend(table.times().iter().map(|k| root.v() + k.v() / slope.v()));
            }
            return;
        }
        _ => {}
    }
    if in_time(f) {
        for p in crate::closed_form::children(f) {
            corners_into(&p.body, out);
        }
    }
}

/// The first min or max of two lines, searched through each ref as through its own form.
pub(crate) fn first<N: Num>(f: &Body<N>, reads: &dyn Reads<N>) -> Option<(Body<N>, Crossing<N>)> {
    if let Body::Node(id) = f {
        return reads.kink(*id);
    }
    if let Some(found) = crossing(f, reads) {
        return Some((f.clone(), found));
    }
    if !in_time(f) {
        return None;
    }
    crate::closed_form::children(f)
        .into_iter()
        .find_map(|p| first(&p.body, reads))
}

/// Which split it is for, the kink, and the line read in its place.
type Replacing<'a, N> = (usize, &'a Body<N>, &'a Body<N>);

fn replaced<N: Num>(f: &Body<N>, at: Replacing<N>, reads: &dyn Reads<N>) -> Body<N> {
    let (scope, kink, with) = at;
    if f == kink {
        return with.clone();
    }
    if let Body::Node(id) = f {
        return reads
            .replaced(*id, scope, kink, with)
            .map_or_else(|| f.clone(), Body::Node);
    }
    if !in_time(f) {
        return f.clone();
    }
    map_children(f, |p| replaced_part(p, at, reads))
}

pub(crate) fn replaced_part<N: Num>(
    p: &Part<N>,
    at: Replacing<N>,
    reads: &dyn Reads<N>,
) -> Part<N> {
    Part::new(p.origin, replaced(&p.body, at, reads))
}
