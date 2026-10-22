// Concern: evaluates a spectral sum's atoms at one instant, and the crop and lane rules every evaluator shares | Non-concern: a written form's tree | IO: (&SpectralSum, t, component) -> C64

use sva_formula::closed_form::Unary;
use sva_formula::spectral_sum::atom::{Singular, SpectralAtom};
use sva_formula::{C64, Cx, Lane, Num, SpectralSum};

use crate::error::CollapseError;
use crate::grid::Grid;

/// Where a form is evaluated: a free instant, or a grid sample, whose edges the grid decides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum At {
    Free(f64),
    Sample(Grid, i64),
}

impl At {
    pub fn t(self) -> f64 {
        match self {
            At::Free(t) => t,
            At::Sample(grid, n) => grid.instant(n),
        }
    }

    pub(crate) fn sample(self) -> Option<(Grid, i64)> {
        match self {
            At::Free(_) => None,
            At::Sample(grid, n) => Some((grid, n)),
        }
    }
}

type On = Option<(Grid, i64)>;

/// The six atom factors in closed form. A delta has no ordinary value, and a principal
/// value has none at its own pole.
pub(crate) fn eval_atom_on<N: Num>(
    a: &SpectralAtom<N>,
    t: N,
    on: On,
) -> Result<Cx<N>, CollapseError> {
    debug_assert!(
        on.is_none_or(|(grid, n)| grid.instant(n) == t.v()),
        "a sample at its instant"
    );
    if let Singular::Delta { at, order } = a.sing {
        return Err(CollapseError::SingularInCt {
            at: at.v(),
            order: i32::from(order),
        });
    }
    let held = match (a.ind, on) {
        (Some(ind), Some((grid, n))) => match grid.inside(n, ind.l.value(), ind.r.value()) {
            true => a.smooth_inside(t),
            false => Some(Cx::ZERO),
        },
        _ => a.smooth_at(t),
    };
    held.ok_or(CollapseError::SingularInCt {
        at: a.pole.map_or(t.v(), |p| p.at.re.v()),
        order: -1,
    })
}

pub(crate) fn eval_lane_on<N: Num>(lane: &Lane<N>, t: N, on: On) -> Result<Cx<N>, CollapseError> {
    let mut sum = Cx::ZERO;
    for a in &lane.atoms {
        sum = sum + eval_atom_on(a, t, on)?;
    }
    with_modal(lane, sum, t, on)
}

pub(crate) fn eval_among(
    lane: &Lane,
    live: &[usize],
    (grid, m): (Grid, i64),
) -> Result<C64, CollapseError> {
    let (t, on) = (grid.instant(m), Some((grid, m)));
    let mut sum = C64::ZERO;
    for &i in live {
        sum = sum + eval_atom_on(&lane.atoms[i], t, on)?;
    }
    with_modal(lane, sum, t, on)
}

fn with_modal<N: Num>(
    lane: &Lane<N>,
    mut sum: Cx<N>,
    t: N,
    on: On,
) -> Result<Cx<N>, CollapseError> {
    for bank in &lane.modal {
        for a in sva_formula::modal::atoms(bank, sva_formula::Origin::UNKNOWN) {
            sum = sum + eval_atom_on(&a.lifted::<N>(), t, on)?;
        }
    }
    Ok(sum)
}

pub(crate) fn eval_spectral_sum(n: &SpectralSum, c: usize, t: f64) -> Result<C64, CollapseError> {
    eval_spectral_sum_on(n, c, At::Free(t))
}

/// A finite sum's atoms at `at`; a series line is no atom until enumerated.
pub(crate) fn eval_spectral_sum_on<N: Num>(
    n: &SpectralSum<N>,
    c: usize,
    at: At,
) -> Result<Cx<N>, CollapseError> {
    eval_lane_on(
        &n.lanes[c.min(n.lanes.len() - 1)],
        N::of(at.t()),
        at.sample(),
    )
}

/// FORMAT 15.6's window: one over the plateau, a raised cosine over each shoulder, zero out.
pub fn crop_gain<N: Num>(t: N, l: N, r: N, rise: N, fall: N) -> N {
    if t.v() < l.v() || t.v() >= r.v() {
        return N::ZERO;
    }
    shoulders(t, l, r, rise, fall)
}

/// The window's gain at an instant already known to lie inside it.
pub fn shoulders<N: Num>(t: N, l: N, r: N, rise: N, fall: N) -> N {
    let opening = shoulder(t - l, rise);
    let closing = shoulder(r - t, fall);
    opening.min(closing)
}

fn shoulder<N: Num>(into: N, span: N) -> N {
    if span.v() <= 0.0 || into.v() >= span.v() {
        return N::ONE;
    }
    N::of(0.5) - N::of(0.5) * (N::of(std::f64::consts::PI) * into / span).cos()
}

/// Which operand of a `join` holds one component of the joined value, and which of its own
/// components that is.
pub fn lane_of(widths: &[usize], component: usize) -> Option<(usize, usize)> {
    let mut left = component;
    for (at, width) in widths.iter().enumerate() {
        if left < *width {
            return Some((at, left));
        }
        left -= width;
    }
    None
}

pub fn unary<N: Num>(op: Unary, x: Cx<N>) -> Cx<N> {
    match op {
        Unary::Exp => x.exp(),
        Unary::Sin => {
            let (sin, cos) = x.re.sincos();
            Cx::new(sin * x.im.cosh(), cos * x.im.sinh())
        }
        Unary::Cos => {
            let (sin, cos) = x.re.sincos();
            Cx::new(cos * x.im.cosh(), -sin * x.im.sinh())
        }
        Unary::Tanh => Cx::real(x.re.tanh()),
        Unary::Sat => Cx::real(x.re.clamp(-1.0, 1.0)),
        Unary::Abs => Cx::real(x.abs()),
        Unary::Log => Cx::real(x.re.ln()),
        Unary::Sqrt => Cx::real(x.re.sqrt()),
        Unary::Step => Cx::real(N::of(sva_formula::affine::step(x.re.v()))),
    }
}
