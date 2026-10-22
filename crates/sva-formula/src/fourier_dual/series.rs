// Concern: duals a series termwise, between a symbolic line and a symbolic delta | Non-concern: how many lines a series yields at a rate (series.rs) | IO: (&Series) -> Series or Left

use std::f64::consts::TAU;

use crate::closed_form::{Body, Part, Series, Unary};
use crate::complex::{C64, Cx};
use crate::num::Num;
use crate::refusal::{AtomSketch, Factor, Left, LeftReason};
use crate::series::mentions_line_read;
use crate::through::{Opaque, Reads, Seen, looked};

/// `amp * exp(2*pi*i*freq*u)`, with both written as index expressions rather than numbers.
pub struct SymLine<N: Num = f64> {
    pub amp: Body<N>,
    pub freq: Body<N>,
}

/// `weight * delta(u - at)`.
pub struct SymDelta<N: Num = f64> {
    pub weight: Body<N>,
    pub at: Body<N>,
}

pub enum Shape<N: Num = f64> {
    Lines(Vec<SymLine<N>>),
    Deltas(Vec<SymDelta<N>>),
}

pub(crate) fn dual<N: Num>(s: &Series<N>, reads: &dyn Reads<N>) -> Result<Series<N>, Left> {
    transform(s, dual_shape, reads)
}

pub(crate) fn reflect<N: Num>(s: &Series<N>, reads: &dyn Reads<N>) -> Result<Series<N>, Left> {
    transform(s, reflect_shape, reads)
}

fn transform<N: Num>(
    s: &Series<N>,
    op: impl Fn(Shape<N>) -> Shape<N>,
    reads: &dyn Reads<N>,
) -> Result<Series<N>, Left> {
    let Some(shape) = read_with(&s.term.body, reads) else {
        return Err(Left::new(
            s.term.origin,
            AtomSketch::of(Factor::Value),
            LeftReason::SeriesNonUniform,
        ));
    };
    Ok(Series {
        term: Part::new(s.term.origin, write(&op(shape))),
        ..s.clone()
    })
}

fn dual_shape<N: Num>(shape: Shape<N>) -> Shape<N> {
    match shape {
        Shape::Lines(lines) => Shape::Deltas(
            lines
                .into_iter()
                .map(|l| SymDelta {
                    weight: l.amp,
                    at: l.freq,
                })
                .collect(),
        ),
        Shape::Deltas(deltas) => Shape::Lines(
            deltas
                .into_iter()
                .map(|d| SymLine {
                    amp: d.weight,
                    freq: negate(d.at),
                })
                .collect(),
        ),
    }
}

fn reflect_shape<N: Num>(shape: Shape<N>) -> Shape<N> {
    match shape {
        Shape::Lines(lines) => Shape::Lines(
            lines
                .into_iter()
                .map(|l| SymLine {
                    freq: negate(l.freq),
                    ..l
                })
                .collect(),
        ),
        Shape::Deltas(deltas) => Shape::Deltas(
            deltas
                .into_iter()
                .map(|d| SymDelta {
                    at: negate(d.at),
                    ..d
                })
                .collect(),
        ),
    }
}

pub(crate) fn write<N: Num>(shape: &Shape<N>) -> Body<N> {
    let parts: Vec<Part<N>> = match shape {
        Shape::Lines(lines) => lines
            .iter()
            .map(|l| {
                Part::bare(Body::Mul(vec![
                    Part::bare(l.amp.clone()),
                    Part::bare(Body::Apply(
                        Unary::Exp,
                        Part::bare(Body::Mul(vec![
                            Part::bare(Body::Const(Cx::of(C64::new(0.0, TAU)))),
                            Part::bare(l.freq.clone()),
                            Part::bare(Body::Line),
                        ])),
                    )),
                ]))
            })
            .collect(),
        Shape::Deltas(deltas) => deltas
            .iter()
            .map(|d| {
                Part::bare(Body::Mul(vec![
                    Part::bare(d.weight.clone()),
                    Part::bare(Body::Delta {
                        at: Part::bare(Body::Add(vec![
                            Part::bare(Body::Line),
                            Part::bare(negate(d.at.clone())),
                        ])),
                        order: 0,
                    }),
                ]))
            })
            .collect(),
    };
    Body::Add(parts)
}

pub fn read<N: Num>(term: &Body<N>) -> Option<Shape<N>> {
    read_with(term, &Opaque)
}

/// The same, each ref the term holds read as the form `reads` names, one level at a time.
pub fn read_with<N: Num>(term: &Body<N>, reads: &dyn Reads<N>) -> Option<Shape<N>> {
    read_deltas(term, reads)
        .map(Shape::Deltas)
        .or_else(|| read_lines(term, reads).map(Shape::Lines))
}

fn read_deltas<N: Num>(term: &Body<N>, reads: &dyn Reads<N>) -> Option<Vec<SymDelta<N>>> {
    match &*looked(term, reads) {
        Body::Add(parts) => {
            let mut out = Vec::new();
            for p in parts {
                out.extend(read_deltas(&p.body, reads)?);
            }
            Some(out)
        }
        other => read_delta(other, reads).map(|d| vec![d]),
    }
}

fn read_delta<N: Num>(term: &Body<N>, reads: &dyn Reads<N>) -> Option<SymDelta<N>> {
    let (weight, singular) = match term {
        Body::Mul(parts) => match parts.as_slice() {
            [weight, singular] => ((*weight.body).clone(), looked(&singular.body, reads)),
            _ => return None,
        },
        // A strike written without a coefficient still weighs one.
        Body::Delta { .. } => (Body::Const(Cx::ONE), Seen::Here(term)),
        _ => return None,
    };
    let Body::Delta { at, order: 0 } = &*singular else {
        return None;
    };
    let Body::Add(offset) = &*looked(&at.body, reads) else {
        return None;
    };
    let [line, shifted] = offset.as_slice() else {
        return None;
    };
    matches!(*looked(&line.body, reads), Body::Line).then(|| SymDelta {
        weight,
        at: negate((*shifted.body).clone()),
    })
}

fn read_lines<N: Num>(term: &Body<N>, reads: &dyn Reads<N>) -> Option<Vec<SymLine<N>>> {
    read_wave(term, Body::Const(Cx::ONE), reads)
}

/// Peels the index-only factors off a term until one wave in the free variable is left.
fn read_wave<N: Num>(
    term: &Body<N>,
    coeff: Body<N>,
    reads: &dyn Reads<N>,
) -> Option<Vec<SymLine<N>>> {
    match &*looked(term, reads) {
        // Written associativity must not decide whether a series reads.
        Body::Add(parts) => {
            let mut out = Vec::new();
            for p in parts {
                out.extend(read_wave(&p.body, coeff.clone(), reads)?);
            }
            Some(out)
        }
        Body::Mul(parts) => {
            let mut core = None;
            let mut acc = coeff;
            for p in parts {
                if mentions_line_read(&p.body, reads) {
                    if core.is_some() {
                        return None;
                    }
                    core = Some(&*p.body);
                } else {
                    acc = product(acc, (*p.body).clone());
                }
            }
            read_wave(core?, acc, reads)
        }
        // A shift written round a term is a shift of the term's own free variable.
        Body::Shift { by, of } => {
            read_wave(&crate::closed_form::shift_line(&of.body, *by), coeff, reads)
        }
        Body::Div(num, den) if !mentions_line_read(&den.body, reads) => read_wave(
            &num.body,
            Body::Div(Part::bare(coeff), Part::bare((*den.body).clone())),
            reads,
        ),
        Body::Apply(op, arg) => {
            let (slope, phase) = split_argument(&arg.body, reads)?;
            Some(match op {
                Unary::Exp => vec![SymLine {
                    amp: product(coeff, exponential(phase)),
                    freq: divide(slope, Cx::of(C64::new(0.0, TAU))),
                }],
                Unary::Cos => quadrature(
                    coeff,
                    slope,
                    phase,
                    Cx::of(C64::real(0.5)),
                    Cx::of(C64::real(0.5)),
                ),
                Unary::Sin => quadrature(
                    coeff,
                    slope,
                    phase,
                    Cx::of(C64::new(0.0, -0.5)),
                    Cx::of(C64::new(0.0, 0.5)),
                ),
                _ => return None,
            })
        }
        _ => None,
    }
}

fn quadrature<N: Num>(
    coeff: Body<N>,
    slope: Body<N>,
    phase: Body<N>,
    up: Cx<N>,
    down: Cx<N>,
) -> Vec<SymLine<N>> {
    let turn = Body::Mul(vec![
        Part::bare(Body::Const(Cx::I)),
        Part::bare(phase.clone()),
    ]);
    let back = Body::Mul(vec![Part::bare(Body::Const(-Cx::I)), Part::bare(phase)]);
    let freq = divide(slope, Cx::of(C64::real(TAU)));
    vec![
        SymLine {
            amp: product(product(coeff.clone(), Body::Const(up)), exponential(turn)),
            freq: freq.clone(),
        },
        SymLine {
            amp: product(product(coeff, Body::Const(down)), exponential(back)),
            freq: negate(freq),
        },
    ]
}

/// `slope * u + phase`, both index-only, or nothing.
fn split_argument<N: Num>(arg: &Body<N>, reads: &dyn Reads<N>) -> Option<(Body<N>, Body<N>)> {
    match &*looked(arg, reads) {
        Body::Add(parts) => {
            let mut slope = None;
            let mut phase = Body::Const(Cx::ZERO);
            for p in parts {
                if mentions_line_read(&p.body, reads) {
                    if slope.is_some() {
                        return None;
                    }
                    // A shifted series writes `(u - by) - k*d`: both offsets are phase.
                    slope = Some(match strip_line(&p.body, reads) {
                        Some(found) => found,
                        None => {
                            let (found, offset) = split_argument(&p.body, reads)?;
                            phase = sum(phase, offset);
                            found
                        }
                    });
                } else {
                    phase = sum(phase, (*p.body).clone());
                }
            }
            Some((slope?, phase))
        }
        Body::Mul(parts) => {
            let mut coeff = Body::Const(Cx::ONE);
            let mut core = None;
            for p in parts {
                if mentions_line_read(&p.body, reads) {
                    if core.is_some() {
                        return None;
                    }
                    core = Some(&*p.body);
                } else {
                    coeff = product(coeff, (*p.body).clone());
                }
            }
            let (slope, phase) = split_argument(core?, reads)?;
            Some((product(coeff.clone(), slope), product(coeff, phase)))
        }
        other => Some((strip_line(other, reads)?, Body::Const(Cx::ZERO))),
    }
}

/// A shifted argument reaches here as `coeff*(t + offset)`, so the coefficient is peeled
/// before the sum inside it is read.
/// The coefficient of the free variable in a product that names it exactly once.
fn strip_line<N: Num>(f: &Body<N>, reads: &dyn Reads<N>) -> Option<Body<N>> {
    match &*looked(f, reads) {
        Body::Line => Some(Body::Const(Cx::ONE)),
        Body::Mul(parts) => {
            let mut rest = Vec::new();
            let mut seen = false;
            for p in parts {
                match &*looked(&p.body, reads) {
                    Body::Line if !seen => seen = true,
                    _ if mentions_line_read(&p.body, reads) => return None,
                    _ => rest.push(Part::bare((*p.body).clone())),
                }
            }
            seen.then(|| collapse(rest))
        }
        _ => None,
    }
}

fn collapse<N: Num>(mut parts: Vec<Part<N>>) -> Body<N> {
    match parts.len() {
        0 => Body::Const(Cx::ONE),
        1 => (*parts.remove(0).body).clone(),
        _ => Body::Mul(parts),
    }
}

fn product<N: Num>(a: Body<N>, b: Body<N>) -> Body<N> {
    match (&a, &b) {
        (Body::Const(x), _) if *x == Cx::ONE => b,
        (_, Body::Const(x)) if *x == Cx::ONE => a,
        _ => Body::Mul(vec![Part::bare(a), Part::bare(b)]),
    }
}

fn sum<N: Num>(a: Body<N>, b: Body<N>) -> Body<N> {
    match &a {
        Body::Const(x) if x.is_zero() => b,
        _ => Body::Add(vec![Part::bare(a), Part::bare(b)]),
    }
}

fn exponential<N: Num>(argument: Body<N>) -> Body<N> {
    match &argument {
        Body::Const(c) if c.is_zero() => Body::Const(Cx::ONE),
        _ => Body::Apply(Unary::Exp, Part::bare(argument)),
    }
}

/// Folding the factor away where it is already written keeps `dual` of `dual` structurally
/// equal to `reflect` rather than merely equal in value.
fn divide<N: Num>(f: Body<N>, by: Cx<N>) -> Body<N> {
    if let Body::Mul(parts) = &f
        && let Some(at) = parts
            .iter()
            .position(|p| matches!(&*p.body, Body::Const(c) if *c == by))
    {
        let mut rest = parts.clone();
        rest.remove(at);
        return collapse(rest);
    }
    Body::Div(Part::bare(f), Part::bare(Body::Const(by)))
}

/// An involution, so a frequency negated twice is the formula it started as.
fn negate<N: Num>(f: Body<N>) -> Body<N> {
    if let Body::Mul(parts) = &f
        && let Some(at) = parts
            .iter()
            .position(|p| matches!(&*p.body, Body::Const(c) if *c == Cx::of(C64::real(-1.0))))
    {
        let mut rest = parts.clone();
        rest.remove(at);
        return collapse(rest);
    }
    Body::Mul(vec![
        Part::bare(Body::Const(Cx::of(C64::real(-1.0)))),
        Part::bare(f),
    ])
}
