// Concern: the atoms each written constructor lowers to, and the window one lifts off | Non-concern: the lane algebra (build.rs) | IO: (a constructor's arguments) -> SpectralSum or a window

use crate::affine::{apply_scalar_over, completed_square_read, exact_affine_read};
use crate::closed_form::{Body, Edge, Fold, Part, Series, Unary, Var};
use crate::complex::Cx;
use crate::num::Num;
use crate::origin::Origin;
use crate::refusal::{AtomSketch, Factor, Left, LeftReason};
use crate::spectral_sum::atom::{
    Exp, Factors, Gauss, Indicator, Pole, Poly, Singular, SpectralAtom,
};
use crate::spectral_sum::merge::simplify_over;
use crate::spectral_sum::{Lane, SpectralSum};
use crate::through::Reads;

pub fn left(origin: Origin, first: Factor, reason: LeftReason) -> Left {
    Left::new(origin, AtomSketch::of(first), reason)
}

/// A constant that is not finite is no atom: a division, a remainder or a logarithm at zero
/// names no number, and A holds none.
pub fn constant<N: Num>(var: Var, c: Cx<N>, origin: Origin) -> Result<SpectralSum<N>, Left> {
    match c.is_finite() {
        true => Ok(one(var, SpectralAtom::constant(c, origin))),
        false => Err(left(origin, Factor::Amplitude, LeftReason::NoValue)),
    }
}

/// A rate, a width or an instant an atom is built on is a finite double or the atom is none:
/// `t/0` and a slope past the largest double name no line to build one on.
pub fn held<N: Num>(origin: Origin, first: Factor, values: &[N]) -> Result<(), Left> {
    match values.iter().all(|v| v.v().is_finite()) {
        true => Ok(()),
        false => Err(left(origin, first, LeftReason::NotFinite)),
    }
}

pub fn one<N: Num>(var: Var, atom: SpectralAtom<N>) -> SpectralSum<N> {
    let mut lane = Lane::of(vec![atom]);
    simplify_over(&mut lane);
    SpectralSum::of(var, vec![lane])
}

pub fn apply<N: Num>(
    op: Unary,
    arg: &Part<N>,
    origin: Origin,
    var: Var,
    reads: &dyn Reads<N>,
) -> Result<SpectralSum<N>, Left> {
    let affine = exact_affine_read(&arg.body, reads);
    if let Some((a, b)) = affine
        && a.is_zero()
    {
        return constant(var, apply_scalar_over(op, b), origin);
    }
    if let Some((a, b)) = affine {
        held(arg.origin, Factor::Value, &[a.re, a.im, b.re, b.im])?;
    }
    if op == Unary::Exp
        && affine.is_none()
        && let Some(q) = completed_square_read(&arg.body, reads)
    {
        held(arg.origin, Factor::Gaussian, &[q.a, q.mu, q.omega])?;
        return Ok(one(
            var,
            SpectralAtom::new(
                q.amplitude,
                Factors {
                    exp: Some(Exp::at(N::ZERO, q.omega)),
                    gauss: Some(Gauss { a: q.a, mu: q.mu }),
                    ..Factors::NONE
                },
                Singular::Regular,
                origin,
            ),
        ));
    }
    let Some((a, b)) = affine.filter(|_| op.is_closed()) else {
        return Err(if op.is_closed() {
            left(
                arg.origin,
                Factor::Exponential,
                LeftReason::NonAffineArgument,
            )
        } else {
            left(origin, Factor::Value, LeftReason::Nonlinearity)
        });
    };
    let atoms = match op {
        Unary::Exp => vec![exp_atom(a, b, origin)?],
        Unary::Cos => vec![
            line_atom((Cx::I * b).exp().scale(N::of(0.5)), Cx::I * a, origin),
            line_atom((-Cx::I * b).exp().scale(N::of(0.5)), -Cx::I * a, origin),
        ],
        _ => vec![
            line_atom(
                (Cx::I * b).exp() * -Cx::I.scale(N::of(0.5)),
                Cx::I * a,
                origin,
            ),
            line_atom(
                (-Cx::I * b).exp() * Cx::I.scale(N::of(0.5)),
                -Cx::I * a,
                origin,
            ),
        ],
    };
    let mut lane = Lane::of(atoms);
    simplify_over(&mut lane);
    Ok(SpectralSum::of(var, vec![lane]))
}

/// `e^{a t + b}`, anchored where it is one where `e^b` passes a double: `e^{sigma (t - mu)}`,
/// `mu = -re(b)/sigma`, `mu`'s rounding carried exactly in the weight's own exponent.
fn exp_atom<N: Num>(a: Cx<N>, b: Cx<N>, origin: Origin) -> Result<SpectralAtom<N>, Left> {
    let weight = b.exp();
    if weight.is_finite() && !weight.is_zero() || a.re.is_zero() {
        return match weight.is_finite() {
            true => Ok(line_atom(weight, a, origin)),
            false => Err(left(origin, Factor::Exponential, LeftReason::Overflow)),
        };
    }
    let mu = -b.re / a.re;
    held(origin, Factor::Exponential, &[mu])?;
    let factors = Factors {
        exp: Some(Exp {
            sigma: a.re,
            omega: a.im,
            mu,
        }),
        ..Factors::NONE
    };
    let residue = Cx::new(a.re.mul_add(mu, b.re), b.im).exp();
    Ok(SpectralAtom::new(
        residue,
        factors,
        Singular::Regular,
        origin,
    ))
}

fn line_atom<N: Num>(c: Cx<N>, alpha: Cx<N>, origin: Origin) -> SpectralAtom<N> {
    SpectralAtom::new(
        c,
        Factors {
            exp: Some(Exp::at(alpha.re, alpha.im)),
            ..Factors::NONE
        },
        Singular::Regular,
        origin,
    )
}

/// `values` is each argument's one number, `None` where one holds none.
pub fn fold<N: Num>(
    op: Fold,
    values: Option<Vec<Cx<N>>>,
    origin: Origin,
    var: Var,
) -> Result<SpectralSum<N>, Left> {
    let Some(values) = values else {
        return Err(left(origin, Factor::Value, LeftReason::Nonlinearity));
    };
    let folded = match (op, values.as_slice()) {
        (Fold::Max, [a, b]) => Cx::real(a.re.max(b.re)),
        (Fold::Min, [a, b]) => Cx::real(a.re.min(b.re)),
        (Fold::Mod, [a, b]) => Cx::real(a.re.rem_euclid(b.re)),
        _ => return Err(left(origin, Factor::Value, LeftReason::Nonlinearity)),
    };
    constant(var, folded, origin)
}

/// `delta^(k)(a*x + b)` places one delta at the argument's zero, scaled by `a^k * |a|`.
pub fn delta<N: Num>(
    at: &Part<N>,
    order: u16,
    var: Var,
    reads: &dyn Reads<N>,
) -> Result<SpectralSum<N>, Left> {
    let (a, b) = real_affine(at, reads)?;
    held(at.origin, Factor::Delta, &[-b / a])?;
    let scale = a.powi(i32::from(order)) * a.abs();
    Ok(one(
        var,
        SpectralAtom::new(
            Cx::real(N::ONE / scale),
            Factors::NONE,
            Singular::Delta { at: -b / a, order },
            at.origin,
        ),
    ))
}

pub fn principal_value<N: Num>(
    at: &Part<N>,
    var: Var,
    reads: &dyn Reads<N>,
) -> Result<SpectralSum<N>, Left> {
    let (a, b) = real_affine(at, reads)?;
    held(at.origin, Factor::PrincipalValue, &[-b / a])?;
    Ok(one(
        var,
        SpectralAtom::new(
            Cx::real(N::ONE / a),
            Factors {
                pole: Some(Pole {
                    at: Cx::real(-b / a),
                    order: 1,
                    pv: true,
                }),
                ..Factors::NONE
            },
            Singular::Regular,
            at.origin,
        ),
    ))
}

fn real_affine<N: Num>(at: &Part<N>, reads: &dyn Reads<N>) -> Result<(N, N), Left> {
    match exact_affine_read(&at.body, reads) {
        Some((a, b)) if a.is_real() && b.is_real() && a.re.v() != 0.0 => Ok((a.re, b.re)),
        _ => Err(left(
            at.origin,
            Factor::Delta,
            LeftReason::NonAffineArgument,
        )),
    }
}
/// The `a*x + b` an atom sum spells, when it spells one.
pub fn affine_atoms<N: Num>(n: &SpectralSum<N>) -> Option<(Cx<N>, Cx<N>)> {
    let [lane] = n.lanes.as_slice() else {
        return None;
    };
    if !lane.is_finite_sum() {
        return None;
    }
    let (mut a, mut b) = (Cx::ZERO, Cx::ZERO);
    for atom in lane.atoms.iter().flat_map(SpectralAtom::flattened) {
        if atom.is_delta() || atom.exp.is_some() || atom.gauss.is_some() {
            return None;
        }
        if atom.ind.is_some() || atom.pole.is_some() {
            return None;
        }
        match atom.poly.degree {
            0 => b = b + atom.c,
            1 => a = a + atom.c,
            _ => return None,
        }
    }
    (!a.is_zero()).then_some((a, b))
}

pub fn shift<N: Num>(n: SpectralSum<N>, by: N) -> Result<SpectralSum<N>, Left> {
    let mut lanes = Vec::with_capacity(n.lanes.len());
    for lane in n.lanes {
        let lane = lane.expanded();
        let mut out = Lane::of(lane.atoms.iter().map(|a| shift_atom(a, by)).collect());
        simplify_over(&mut out);
        for s in &lane.series {
            out.series.push(shift_series(s, by)?);
        }
        lanes.push(out);
    }
    Ok(SpectralSum::of(n.var, lanes))
}

/// A shift of a series is the shift of each of its terms: the free variable moves under the
/// sum the same way it moves under a finite one, so the pair and its term count both stand.
fn shift_series<N: Num>(s: &Series<N>, by: N) -> Result<Series<N>, Left> {
    if crate::closed_form::shifts_opaquely(&s.term.body) {
        return Err(left(
            s.term.origin,
            Factor::Value,
            LeftReason::Unsubstituted,
        ));
    }
    Ok(Series {
        term: Part::new(
            s.term.origin,
            crate::closed_form::shift_line_over(&s.term.body, by),
        ),
        ..s.clone()
    })
}

/// `x -> x - by` in every factor, each keeping the instant it is read from.
fn shift_atom<N: Num>(atom: &SpectralAtom<N>, by: N) -> SpectralAtom<N> {
    if let Singular::Delta { at, order } = atom.sing {
        return SpectralAtom::new(
            atom.c,
            Factors::NONE,
            Singular::Delta { at: at + by, order },
            atom.origin,
        );
    }
    let f = atom.factors();
    let moved = Factors {
        poly: Poly {
            at: f.poly.at + by,
            ..f.poly
        },
        exp: f.exp.map(|e| Exp { mu: e.mu + by, ..e }),
        gauss: f.gauss.map(|g| Gauss { mu: g.mu + by, ..g }),
        ind: f.ind.map(|i| Indicator {
            l: Edge::at(i.l.number() + by),
            r: Edge::at(i.r.number() + by),
        }),
        pole: f.pole.map(|p| Pole {
            at: p.at + Cx::real(by),
            ..p
        }),
    };
    let mut c = atom.c;
    if let Some(e) = f.exp {
        c = c * Cx::new(N::ZERO, -e.omega * by).exp();
    }
    SpectralAtom::new(c, moved, Singular::Regular, atom.origin)
}

pub fn derive<N: Num>(n: SpectralSum<N>) -> Result<SpectralSum<N>, Left> {
    let mut lanes = Vec::with_capacity(n.lanes.len());
    for lane in n.lanes {
        let lane = lane.expanded();
        if !lane.is_finite_sum() {
            return Err(left(
                Origin::UNKNOWN,
                Factor::Value,
                LeftReason::SeriesNonUniform,
            ));
        }
        let mut atoms = Vec::new();
        for atom in &lane.atoms {
            atoms.extend(atom.derivative()?);
        }
        let mut out = Lane::of(atoms);
        simplify_over(&mut out);
        lanes.push(out);
    }
    Ok(SpectralSum::of(n.var, lanes))
}

/// An indicator is a pointwise factor, so it distributes over a sum however long: a cropped
/// series is the series of cropped terms, and the window is the same in every term.
pub fn crop_over<N: Num>(n: SpectralSum<N>, window: Indicator<N>) -> Result<SpectralSum<N>, Left> {
    let mut lanes = Vec::with_capacity(n.lanes.len());
    for lane in n.lanes {
        let lane = lane.expanded();
        let series = lane.series.iter().map(|s| windowed(s, window)).collect();
        let mut atoms = Vec::new();
        for atom in &lane.atoms {
            let met = atom.ind.map_or(window, |i| i.meet(window));
            if met.is_empty() {
                continue;
            }
            if let Singular::Delta { at, .. } = atom.sing {
                if met.contains(at.v()) {
                    atoms.push(*atom);
                }
                continue;
            }
            atoms.push(atom.with(
                atom.c,
                Factors {
                    ind: Some(met),
                    ..atom.factors()
                },
            ));
        }
        let mut out = Lane {
            series,
            ..Lane::of(atoms)
        };
        simplify_over(&mut out);
        lanes.push(out);
    }
    Ok(SpectralSum::of(n.var, lanes))
}

/// A hard-cropped term with its window lifted off: an indicator is a pointwise factor, so
/// what is under it reads as the closed form it was before the crop. A shoulder is not one.
pub fn crop_peeled_over<N: Num>(f: &Body<N>) -> (Body<N>, Option<Indicator<N>>) {
    let Body::Crop {
        of,
        l,
        r,
        rise,
        fall,
    } = f
    else {
        return (f.clone(), None);
    };
    if !(rise.is_zero() && fall.is_zero()) {
        return (f.clone(), None);
    }
    let (body, held) = crop_peeled_over(&of.body);
    let window = Indicator { l: *l, r: *r };
    (body, Some(held.map_or(window, |h| h.meet(window))))
}

fn windowed<N: Num>(s: &Series<N>, window: Indicator<N>) -> Series<N> {
    Series {
        term: Part::new(
            s.term.origin,
            Body::Crop {
                of: s.term.clone(),
                l: window.l,
                r: window.r,
                rise: N::ZERO,
                fall: N::ZERO,
            },
        ),
        ..s.clone()
    }
}

/// The plateau plus two raised-cosine shoulders, seven atoms, each under its own indicator.
pub fn crop_window<N: Num>(
    l: Edge<N>,
    r: Edge<N>,
    rise: N,
    fall: N,
    origin: Origin,
    var: Var,
) -> SpectralSum<N> {
    let (a, b) = (l.number(), r.number());
    let mut atoms = vec![SpectralAtom::new(
        Cx::ONE,
        Factors {
            ind: Some(Indicator {
                l: Edge::at(a + rise),
                r: Edge::at(b - fall),
            }),
            ..Factors::NONE
        },
        Singular::Regular,
        origin,
    )];
    atoms.extend(shoulder(a, a + rise, rise, -0.25, origin));
    atoms.extend(shoulder(b - fall, b, fall, 0.25, origin));
    SpectralSum::of(var, vec![Lane::of(atoms)])
}

/// A shoulder of no length is a hard edge, which the plateau's own indicator already is: a
/// raised cosine over zero seconds has no half-period to turn over.
fn shoulder<N: Num>(from: N, to: N, span: N, quarter: f64, origin: Origin) -> Vec<SpectralAtom<N>> {
    if span.v() <= 0.0 || !span.v().is_finite() {
        return Vec::new();
    }
    raised_cosine(
        from,
        to,
        N::of(std::f64::consts::PI) / span,
        N::of(0.5),
        quarter,
        origin,
    )
}

/// `dc + 2*side*cos(w*(t - from))` over `[from, to)` and nothing outside it, as the three
/// atoms it is: the level, and one line either side of it phased to the window's own start.
pub(crate) fn raised_cosine<N: Num>(
    from: N,
    to: N,
    w: N,
    dc: N,
    side: f64,
    origin: Origin,
) -> Vec<SpectralAtom<N>> {
    let window = Indicator {
        l: Edge::at(from),
        r: Edge::at(to),
    };
    let mut out = vec![SpectralAtom::new(
        Cx::real(dc),
        Factors {
            ind: Some(window),
            ..Factors::NONE
        },
        Singular::Regular,
        origin,
    )];
    for sign in [1.0, -1.0] {
        out.push(SpectralAtom::new(
            Cx::new(N::ZERO, N::of(-sign) * w * from)
                .exp()
                .scale(N::of(side)),
            Factors {
                exp: Some(Exp::at(N::ZERO, w * sign)),
                ind: Some(window),
                ..Factors::NONE
            },
            Singular::Regular,
            origin,
        ));
    }
    out
}

pub fn crop(n: SpectralSum, window: Indicator) -> Result<SpectralSum, Left> {
    crop_over(n, window)
}

pub fn crop_peeled(f: &Body) -> (Body, Option<Indicator>) {
    crop_peeled_over(f)
}
