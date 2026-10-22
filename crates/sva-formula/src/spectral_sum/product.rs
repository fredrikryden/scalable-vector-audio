// Concern: multiplies two atoms into one | Non-concern: summing like atoms (merge.rs) | IO: (SpectralAtom, SpectralAtom) -> Vec<SpectralAtom>

use crate::closed_form::Edge;
use crate::complex::Cx;
use crate::num::Num;
use crate::rational::partial_fractions;
use crate::refusal::{AtomSketch, Factor, Left, LeftReason};
use crate::spectral_sum::atom::{Exp, Factors, Gauss, Pole, Poly, Singular, SpectralAtom};

/// Empty where the two indicators do not meet; several where two distinct poles partial-
/// fraction, or where a delta derivative expands by Leibniz.
pub fn times<N: Num>(
    a: &SpectralAtom<N>,
    b: &SpectralAtom<N>,
) -> Result<Vec<SpectralAtom<N>>, Left> {
    match (a.sing, b.sing) {
        (Singular::Regular, Singular::Regular) => match poly_times(a, a.poly, b.poly)? {
            Some(poly) => smooth_times(a, b, poly),
            None => {
                let mut out = Vec::new();
                for x in a.flattened() {
                    for y in b.flattened() {
                        out.extend(smooth_times(
                            &x,
                            &y,
                            poly_times(a, x.poly, y.poly)?.expect("both read from the origin"),
                        )?);
                    }
                }
                Ok(out)
            }
        },
        (Singular::Delta { at, order }, Singular::Regular) => leibniz(a, b, at, order),
        (Singular::Regular, Singular::Delta { at, order }) => leibniz(b, a, at, order),
        _ => Err(Left::new(
            a.origin,
            AtomSketch::pair(Factor::Delta, Factor::Delta),
            LeftReason::Nonlinearity,
        )),
    }
}

/// Two growing halves meet at one reference, the mean their rates weight, so the product
/// carries no prefactor at all; only a growth that cancels leaves one behind.
fn combine<N: Num>(x: Exp<N>, y: Exp<N>) -> Option<(Exp<N>, Cx<N>)> {
    let sigma = x.sigma + y.sigma;
    let omega = x.omega + y.omega;
    if sigma.is_zero() {
        let carried = Cx::real(-x.sigma * x.mu - y.sigma * y.mu).exp();
        return carried
            .is_finite()
            .then_some((Exp::at(N::ZERO, omega), carried));
    }
    let mu = (x.sigma * x.mu + y.sigma * y.mu) / sigma;
    Some((Exp { sigma, omega, mu }, Cx::ONE))
}

/// Two polynomials read from one instant are one; from two instants, neither is kept. A degree
/// past `u16` refuses.
fn poly_times<N: Num>(
    a: &SpectralAtom<N>,
    x: Poly<N>,
    y: Poly<N>,
) -> Result<Option<Poly<N>>, Left> {
    Ok(match (x.is_one(), y.is_one()) {
        (true, _) => Some(y),
        (_, true) => Some(x),
        _ if x.at != y.at => None,
        _ => Some(Poly {
            degree: x.degree.checked_add(y.degree).ok_or_else(|| {
                let sketch = AtomSketch::pair(Factor::Polynomial, Factor::Polynomial);
                Left::new(a.origin, sketch, LeftReason::Overflow)
            })?,
            at: x.at,
        }),
    })
}

fn smooth_times<N: Num>(
    a: &SpectralAtom<N>,
    b: &SpectralAtom<N>,
    poly: Poly<N>,
) -> Result<Vec<SpectralAtom<N>>, Left> {
    let ind = match (a.ind, b.ind) {
        (Some(x), Some(y)) => {
            let met = x.meet(y);
            if met.is_empty() {
                return Ok(Vec::new());
            }
            Some(met)
        }
        (x, y) => x.or(y),
    };

    let mut c = a.c * b.c;
    let exp = match (a.exp, b.exp) {
        (Some(x), Some(y)) => {
            let (held, carried) = combine(x, y).ok_or_else(|| {
                Left::new(
                    a.origin,
                    AtomSketch::pair(Factor::Exponential, Factor::Exponential),
                    LeftReason::NotTempered,
                )
            })?;
            c = c * carried;
            Some(held)
        }
        (x, y) => x.or(y),
    };
    let gauss = match (a.gauss, b.gauss) {
        (Some(x), Some(y)) => {
            let sum = x.a + y.a;
            let gap = x.mu - y.mu;
            c = c.scale((-x.a * y.a * gap * gap / sum).exp());
            Some(Gauss {
                a: sum,
                mu: (x.a * x.mu + y.a * y.mu) / sum,
            })
        }
        (x, y) => x.or(y),
    };

    let sketch = match (exp, gauss) {
        (Some(e), _)
            if !(e.sigma.v().is_finite() && e.omega.v().is_finite() && e.mu.v().is_finite()) =>
        {
            Some(AtomSketch::pair(Factor::Exponential, Factor::Exponential))
        }
        (_, Some(g)) if !(g.a.v().is_finite() && g.mu.v().is_finite()) => {
            Some(AtomSketch::pair(Factor::Gaussian, Factor::Gaussian))
        }
        _ => None,
    };
    if let Some(sketch) = sketch {
        return Err(Left::new(a.origin, sketch, LeftReason::NotFinite));
    }
    let base = Factors {
        poly,
        exp,
        gauss,
        ind,
        pole: None,
    };
    match (a.pole, b.pole) {
        (None, None) => Ok(vec![SpectralAtom::new(
            c,
            base,
            Singular::Regular,
            a.origin,
        )]),
        (Some(p), None) | (None, Some(p)) => Ok(vec![SpectralAtom::new(
            c,
            Factors {
                pole: Some(p),
                ..base
            },
            Singular::Regular,
            a.origin,
        )]),
        (Some(p), Some(q)) if p.at == q.at => Ok(vec![SpectralAtom::new(
            c,
            Factors {
                pole: Some(Pole {
                    at: p.at,
                    order: p.order.checked_add(q.order).ok_or_else(|| {
                        Left::new(
                            a.origin,
                            AtomSketch::pair(Factor::Pole, Factor::Pole),
                            LeftReason::PoleOrder(u16::MAX),
                        )
                    })?,
                    pv: p.pv || q.pv,
                }),
                ..base
            },
            Singular::Regular,
            a.origin,
        )]),
        (Some(p), Some(q)) => split_poles(c, base, p, q, a),
    }
}

/// Two distinct poles are one rational, and a rational in A is its residues. The principal
/// value stays with the pole it was written on: only that one has no ordinary value.
fn split_poles<N: Num>(
    c: Cx<N>,
    base: Factors<N>,
    p: Pole<N>,
    q: Pole<N>,
    at: &SpectralAtom<N>,
) -> Result<Vec<SpectralAtom<N>>, Left> {
    let mut roots = vec![p.at; usize::from(p.order)];
    roots.extend(std::iter::repeat_n(q.at, usize::from(q.order)));
    let (_, residues) = partial_fractions(&[Cx::ONE], &roots).map_err(|reason| {
        Left::new(
            at.origin,
            AtomSketch::pair(Factor::Pole, Factor::Pole),
            reason,
        )
    })?;
    Ok(residues
        .into_iter()
        .filter(|r| !r.weight.is_zero())
        .map(|r| {
            SpectralAtom::new(
                c * r.weight,
                Factors {
                    pole: Some(Pole {
                        at: r.at,
                        order: r.order,
                        pv: if r.at == p.at { p.pv } else { q.pv },
                    }),
                    ..base
                },
                Singular::Regular,
                at.origin,
            )
        })
        .collect())
}

/// `g(x) * delta^(k)(x - a)` is the alternating binomial sum over `g`'s derivatives at `a`.
fn leibniz<N: Num>(
    delta: &SpectralAtom<N>,
    smooth: &SpectralAtom<N>,
    at: N,
    order: u16,
) -> Result<Vec<SpectralAtom<N>>, Left> {
    if let Some(i) = smooth.ind
        && (i.l == Edge::At(at.canonical()) || i.r == Edge::At(at.canonical()))
    {
        return Err(Left::new(
            delta.origin,
            AtomSketch::pair(Factor::Delta, Factor::Indicator),
            LeftReason::Nonlinearity,
        ));
    }
    if smooth.ind.is_some_and(|i| !i.contains(at.v())) {
        return Ok(Vec::new());
    }
    let mut level = vec![SpectralAtom::new(
        smooth.c,
        Factors {
            ind: None,
            ..smooth.factors()
        },
        Singular::Regular,
        smooth.origin,
    )];

    let mut out = Vec::new();
    let mut binomial = 1i64;
    for j in 0..=order {
        let mut value = Cx::ZERO;
        for atom in &level {
            let Some(v) = atom.smooth_at(at) else {
                return Err(Left::new(
                    delta.origin,
                    AtomSketch::pair(Factor::Delta, Factor::Pole),
                    LeftReason::Nonlinearity,
                ));
            };
            value = value + v;
        }
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        out.push(SpectralAtom::new(
            delta.c * value.scale(N::of(sign * binomial as f64)),
            Factors::NONE,
            Singular::Delta {
                at,
                order: order - j,
            },
            delta.origin,
        ));
        binomial = binomial * i64::from(order - j) / (i64::from(j) + 1);
        let mut next = Vec::new();
        for atom in &level {
            next.extend(atom.derivative()?);
        }
        level = next;
    }
    Ok(out)
}
