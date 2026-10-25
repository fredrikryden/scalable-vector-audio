// Concern: folds a written subterm, or one call of numbers, to the number it holds | Non-concern: lowering a closed form (mod.rs) | IO: (&Body, or a call's arguments) -> Option<f64>

use sva_formula::closed_form::children;
use sva_formula::{Body, ClosedForm, Cx, Fold, NodeId, Num, Origin, Part, Var};

use super::number::Number;

pub fn is_constant<N: Num>(f: &Body<N>) -> bool {
    constant_with(f, &|_| false)
}

/// The same, each ref answered by `node`.
pub(crate) fn constant_with<N: Num>(f: &Body<N>, node: &dyn Fn(NodeId) -> bool) -> bool {
    match f {
        Body::Node(id) => node(*id),
        Body::Line | Body::Param(_) | Body::Index(_) => false,
        other => children(other).iter().all(|p| constant_with(&p.body, node)),
    }
}

/// One bare atom is a number, an empty lane the zero it folded to; else it is not.
pub fn constant_value<N: Number>(body: &Body<N>, var: Var) -> Option<N> {
    if !is_constant(body) {
        return None;
    }
    let form = ClosedForm {
        var,
        body: N::written_out(body)?,
        origin: Origin::UNKNOWN,
    };
    constant_of(&N::normal(&form).ok()?)
}

pub fn constant_of<N: Num>(sum: &sva_formula::SpectralSum<N>) -> Option<N> {
    let [lane] = sum.lanes.as_slice() else {
        return None;
    };
    if !lane.series.is_empty() || !lane.modal.is_empty() {
        return None;
    }
    match lane.atoms.as_slice() {
        [] => Some(N::ZERO),
        [atom] => atom.is_bare().then_some(atom.c.re),
        _ => None,
    }
}

pub fn constant_modulo<N: Number>(a: N, b: N) -> Option<N> {
    folded_number(&Body::Fold(Fold::Mod, vec![number(a), number(b)]))
}

/// Through what `arithmetic` builds; a name it omits names no number.
pub fn constant_call<N: Number>(name: &str, positional: &[N], named: &[(&str, N)]) -> Option<N> {
    let bodies: Vec<Body<N>> = positional
        .iter()
        .map(|x| Body::Const(Cx::real(*x)))
        .collect();
    folded_number(&super::calls::arithmetic(
        name,
        bodies,
        named,
        Var::T,
        Origin::UNKNOWN,
    )?)
}

fn folded_number<N: Number>(body: &Body<N>) -> Option<N> {
    unbounded(body).or_else(|| constant_value(body, Var::T))
}

fn number<N: Num>(x: N) -> Part<N> {
    Part::new(Origin::UNKNOWN, Body::Const(Cx::real(x)))
}

/// A built body of numbers, one infinite, as IEEE arithmetic: no atom holds `inf`.
pub fn unbounded<N: Number>(body: &Body<N>) -> Option<N> {
    (is_constant(body) && holds_infinite(body))
        .then(|| scalar(body))
        .flatten()
}

/// A sum whose numbers add to `inf` and whose every other term is a real polynomial in the free
/// variable, finite wherever it reaches up to `reach`: IEEE arithmetic makes it that `inf`.
pub fn swamped<N: Number>(body: &Body<N>, reach: f64) -> Option<N> {
    let Body::Add(parts) = body else {
        return None;
    };
    let (numbers, terms): (Vec<&Part<N>>, Vec<&Part<N>>) =
        parts.iter().partition(|p| is_constant(&p.body));
    let sum = numbers
        .iter()
        .try_fold(N::ZERO, |held, p| Some(held + scalar(&p.body)?))?;
    let finite = |p: &&Part<N>| {
        let coeffs = N::polynomial(&p.body)?;
        coeffs.iter().enumerate().try_fold(0.0f64, |held, (k, c)| {
            let c = c
                .exact()
                .filter(|c| c.im.is_zero() && c.re.v().is_finite())?;
            Some(held + c.re.v().abs() * reach.powi(k as i32))
        })
    };
    let reached = terms
        .iter()
        .try_fold(0.0, |held, p| Some(held + finite(p)?));
    (sum.v().is_infinite() && reached.is_some_and(|r| r < f64::MAX / 4.0)).then_some(sum)
}

/// The double a two-operand sum or product of reals is in every arithmetic. Not zero (no
/// atom; a written 0 is one), two negatives' product (imaginary -0) or a quotient.
pub fn plain<N: Number>(body: &Body<N>) -> Option<N> {
    plain_or_zero(body).filter(|v| !v.is_zero())
}

fn plain_or_zero<N: Number>(body: &Body<N>) -> Option<N> {
    let pair = |parts: &[Part<N>]| match parts {
        [a, b] => Some((plain_or_zero(&a.body)?, plain_or_zero(&b.body)?)),
        _ => None,
    };
    let held = match body {
        Body::Const(c) if c.im.v().to_bits() == 0 && c.im.is_constant() => c.re,
        Body::Add(parts) => pair(parts).map(|(a, b)| a + b)?,
        Body::Mul(parts) => pair(parts)
            .filter(|(a, b)| !(a.v() < 0.0 && b.v() < 0.0))
            .map(|(a, b)| a * b)?,
        _ => return None,
    };
    held.v().is_finite().then_some(held)
}

pub fn holds_infinite<N: Num>(f: &Body<N>) -> bool {
    matches!(f, Body::Const(c) if !c.is_finite())
        || children(f).iter().any(|p| holds_infinite(&p.body))
}

fn scalar<N: Number>(f: &Body<N>) -> Option<N> {
    let of = |p: &Part<N>| scalar(&p.body);
    Some(match f {
        Body::Const(c) if c.im.is_zero() => c.re,
        Body::Add(parts) => parts.iter().try_fold(N::ZERO, |a, p| Some(a + of(p)?))?,
        Body::Mul(parts) => parts.iter().try_fold(N::ONE, |a, p| Some(a * of(p)?))?,
        Body::Div(a, b) => of(a)? / of(b)?,
        Body::Pow(base, n) => of(base)?.powi(*n),
        Body::Apply(op, x) => N::apply_scalar(*op, Cx::real(of(x)?)).re,
        Body::Fold(op, parts) => match (op, parts.as_slice()) {
            (Fold::Max, [a, b]) => of(a)?.max(of(b)?),
            (Fold::Min, [a, b]) => of(a)?.min(of(b)?),
            (Fold::Mod, [a, b]) => of(a)?.rem_euclid(of(b)?),
            _ => return None,
        },
        _ => return None,
    })
}

/// The one number a node holds, `inf` included.
pub(crate) fn number_of(tys: &crate::typing::Typing, id: sva_formula::NodeId) -> Option<f64> {
    match tys.value(id) {
        crate::typing::Value::ClosedForm(form) => match form.body {
            Body::Const(c) if c.im == 0.0 => Some(c.re),
            _ => constant_value(&form.body, form.var),
        },
        _ => None,
    }
}
