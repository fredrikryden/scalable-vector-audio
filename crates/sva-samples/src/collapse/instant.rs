// Concern: evaluates a written form at one instant over any number, as a fit reads the point row | Non-concern: a column of instants (column.rs) | IO: (&Body<N>, At, component) -> Cx<N>

use sva_formula::closed_form::{Bound, Fold};
use sva_formula::{Body, Cx, IndexId, Num, Part};

use super::column::{INFINITE, near, power, width_of};
use super::point::{At, crop_gain, eval_atom_on, lane_of, shoulders, unary};
use crate::error::CollapseError;
use crate::grid::Grid;

/// A form's value at `at`, each op as the point row's column evaluates it, a finite series
/// summed from +0 in index order. A ref, an infinite series and a run refuse: only an
/// instance's own closed form reaches here.
pub fn form_at<N: Num>(f: &Body<N>, at: At, component: usize) -> Result<Cx<N>, CollapseError> {
    let time = Time {
        t: N::of(at.t()),
        on: at.sample(),
    };
    Walk { bound: Vec::new() }.eval(f, time, component)
}

/// The instant a subterm reads, and the sample it is where it is the root's own.
#[derive(Clone, Copy)]
struct Time<N> {
    t: N,
    on: Option<(Grid, i64)>,
}

struct Walk {
    bound: Vec<(IndexId, f64)>,
}

impl Walk {
    fn eval<N: Num>(&mut self, f: &Body<N>, at: Time<N>, c: usize) -> Result<Cx<N>, CollapseError> {
        let moved = |t: N| Time { t, on: None };
        Ok(match f {
            Body::Const(v) if v.is_finite() => *v,
            Body::Const(_) => return Err(INFINITE),
            Body::Line => Cx::real(at.t),
            Body::Add(parts) => {
                let mut sum = Cx::ZERO;
                for p in parts {
                    sum = sum + self.eval(&p.body, at, c)?;
                }
                finite(sum)?
            }
            Body::Mul(parts) => self.product(parts, at, c)?,
            Body::Div(a, b) => self.eval(&a.body, at, c)? / self.eval(&b.body, at, c)?,
            Body::Pow(x, n) => power(self.eval(&x.body, at, c)?, *n),
            Body::Apply(op, x) => unary(*op, self.eval(&x.body, at, c)?),
            Body::Fold(op, parts) => {
                let mut each = parts.iter();
                let first = each.next().expect("a fold holds one part");
                let mut acc = self.eval(&first.body, at, c)?;
                for p in each {
                    let v = self.eval(&p.body, at, c)?;
                    acc = Cx::real(match op {
                        Fold::Max => acc.re.max(v.re),
                        Fold::Min => acc.re.min(v.re),
                        Fold::Mod => acc.re.rem_euclid(v.re),
                    });
                }
                acc
            }
            Body::Shift { by, of } => self.eval(&of.body, moved(at.t - *by), c)?,
            Body::Warp { at: when, of } => {
                let when = self.eval(&when.body, at, c)?;
                self.eval(&of.body, moved(when.re), c)?
            }
            Body::Crop {
                of,
                l,
                r,
                rise,
                fall,
            } => {
                let (l, r) = (l.number(), r.number());
                let gain = match at.on {
                    Some((grid, n)) if grid.inside(n, l.v(), r.v()) => {
                        shoulders(at.t, l, r, *rise, *fall)
                    }
                    Some(_) => N::ZERO,
                    None => crop_gain(at.t, l, r, *rise, *fall),
                };
                match gain.v() == 0.0 {
                    true => Cx::ZERO,
                    false => self.eval(&of.body, at, c)?.scale(gain),
                }
            }
            Body::Channel(of, k) => self.eval(&of.body, at, usize::from(*k))?,
            Body::Delta { order, .. } => {
                return Err(CollapseError::SingularInCt {
                    at: at.t.v(),
                    order: i32::from(*order),
                });
            }
            Body::Pv(_) => {
                return Err(CollapseError::SingularInCt {
                    at: at.t.v(),
                    order: -1,
                });
            }
            Body::Keyed { seed, of } => {
                let key = self.eval(&of.body, at, c)?;
                Cx::real(N::of(
                    sva_formula::draw_nearest(*seed, key.re.v())
                        .ok_or(CollapseError::NotEvaluable("a key past any step"))?,
                ))
            }
            Body::Join(parts) => {
                let widths: Vec<usize> = parts.iter().map(|p| width_of(&p.body, &[])).collect();
                match lane_of(&widths, c) {
                    Some((lane, component)) => self.eval(&parts[lane].body, at, component)?,
                    None => return Err(CollapseError::NotEvaluable("a component past the width")),
                }
            }
            Body::Modal(bank) => {
                let mut sum = Cx::ZERO;
                for a in sva_formula::modal::atoms(bank, sva_formula::Origin::UNKNOWN) {
                    sum = sum + eval_atom_on(&a.lifted::<N>(), at.t, at.on)?;
                }
                sum
            }
            Body::Knots { at: read, table } => {
                let by = match sva_formula::affine::slide(&read.body) {
                    Some(sva_formula::affine::Coeff::Exact(c)) if c.im.is_zero() => Some(c.re),
                    _ => None,
                };
                let (u, below) = match by {
                    Some(by) => {
                        let u = at.t + by;
                        let (t, c, uv) = (at.t.v(), by.v(), u.v());
                        let back = uv - t;
                        (u, (t - (uv - back)) + (c - back) < 0.0)
                    }
                    None => (self.eval(&read.body, at, c)?.re, false),
                };
                let snapped = match (at.on, by) {
                    (Some((grid, n)), Some(by)) => near(table, u.v(), by.v())
                        .map(|knot| (u.rebased(knot), n < grid.first_past(knot, by.v()))),
                    _ => None,
                };
                let (u, below) = snapped.unwrap_or((u, below));
                Cx::real(table.read(u, below, &mut 0))
            }
            Body::Index(i) => match self.bound.iter().rev().find(|(j, _)| j == i) {
                Some((_, k)) => Cx::real(N::of(*k)),
                None => return Err(CollapseError::NotEvaluable("a free series index")),
            },
            Body::Series(s) => {
                let Bound::Finite(hi) = s.hi else {
                    return Err(CollapseError::NotEvaluable("an infinite series"));
                };
                let mut sum = Cx::ZERO;
                for k in s.lo..=hi {
                    self.bound.push((s.index, k as f64));
                    let term = self.eval(&s.term.body, at, c);
                    self.bound.pop();
                    sum = sum + term?;
                }
                finite(sum)?
            }
            Body::Node(_) => return Err(CollapseError::NotEvaluable("a node")),
            Body::Run(_) | Body::Banded(_) => {
                return Err(CollapseError::NotEvaluable("a truncated series"));
            }
            Body::Param(_) => {
                return Err(CollapseError::NotEvaluable("an unsubstituted parameter"));
            }
            Body::Deriv { .. } => return Err(CollapseError::NotEvaluable("a derivative")),
            Body::Rational(_) => return Err(CollapseError::NotEvaluable("a rational")),
        })
    }

    /// A factor exactly zero zeroes the product whatever another refused, as the column's does.
    fn product<N: Num>(
        &mut self,
        parts: &[Part<N>],
        at: Time<N>,
        c: usize,
    ) -> Result<Cx<N>, CollapseError> {
        let mut held: Result<Cx<N>, CollapseError> = Ok(Cx::ONE);
        for p in parts {
            match (self.eval(&p.body, at, c), &held) {
                (Ok(v), _) if v.is_zero() => return Ok(Cx::ZERO),
                (Ok(v), Ok(acc)) => held = Ok(*acc * v),
                (Err(e), Ok(_)) => held = Err(e),
                (_, Err(_)) => {}
            }
        }
        held
    }
}

fn finite<N: Num>(v: Cx<N>) -> Result<Cx<N>, CollapseError> {
    match v.is_finite() {
        true => Ok(v),
        false => Err(INFINITE),
    }
}
