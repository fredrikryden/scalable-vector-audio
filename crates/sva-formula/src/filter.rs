// Concern: the eight shapes' H(s) as a Rational in s | Non-concern: the biquad recurrence (sva-samples) | IO: (Shape, cutoff, q, gain) -> Rational

use std::f64::consts::TAU;

use crate::closed_form::Rational;
use crate::complex::Cx;
use crate::num::Num;

/// `OnePole` is the 6 dB/oct filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    OnePole,
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Peaking,
    Lowshelf,
    Highshelf,
}

/// In the order sva-ast's `FILTERS` names them.
pub const ALL_SHAPES: [Shape; 8] = [
    Shape::OnePole,
    Shape::Lowpass,
    Shape::Highpass,
    Shape::Bandpass,
    Shape::Notch,
    Shape::Peaking,
    Shape::Lowshelf,
    Shape::Highshelf,
];

impl Shape {
    pub fn default_q(self) -> f64 {
        match self {
            Shape::Bandpass | Shape::Notch | Shape::Peaking => 1.0,
            _ => std::f64::consts::FRAC_1_SQRT_2,
        }
    }

    pub fn takes_gain(self) -> bool {
        matches!(self, Shape::Peaking | Shape::Lowshelf | Shape::Highshelf)
    }
}

/// The RBJ Audio EQ Cookbook's analog prototypes, before any bilinear transform. `gain` is
/// the amplitude ratio at the peak or shelf, what `12db` writes; the cookbook's `A` is its root.
pub fn design<N: Num>(shape: Shape, cutoff: N, q: N, gain: N) -> Rational<N> {
    let w0 = cutoff * TAU;
    let a = gain.sqrt();
    let root_a = a.sqrt();
    match shape {
        Shape::OnePole => Rational {
            zeros: Vec::new(),
            poles: vec![Cx::real(-w0)],
            gain: Cx::real(w0),
        },
        Shape::Lowpass => Rational {
            zeros: Vec::new(),
            poles: quadratic(w0 / q, w0 * w0),
            gain: Cx::real(w0 * w0),
        },
        Shape::Highpass => Rational {
            zeros: vec![Cx::ZERO, Cx::ZERO],
            poles: quadratic(w0 / q, w0 * w0),
            gain: Cx::ONE,
        },
        Shape::Bandpass => Rational {
            zeros: vec![Cx::ZERO],
            poles: quadratic(w0 / q, w0 * w0),
            gain: Cx::real(w0 / q),
        },
        Shape::Notch => Rational {
            zeros: vec![Cx::new(N::ZERO, w0), Cx::new(N::ZERO, -w0)],
            poles: quadratic(w0 / q, w0 * w0),
            gain: Cx::ONE,
        },
        Shape::Peaking => Rational {
            zeros: quadratic(w0 * a / q, w0 * w0),
            poles: quadratic(w0 / (q * a), w0 * w0),
            gain: Cx::ONE,
        },
        Shape::Lowshelf => Rational {
            zeros: quadratic(root_a * w0 / q, a * w0 * w0),
            poles: quadratic(root_a * w0 / (q * a), w0 * w0 / a),
            gain: Cx::ONE,
        },
        Shape::Highshelf => Rational {
            zeros: quadratic(root_a * w0 / (q * a), w0 * w0 / a),
            poles: quadratic(root_a * w0 / q, a * w0 * w0),
            gain: Cx::real(a * a),
        },
    }
}

/// The two roots of `s^2 + b*s + c`, real or conjugate.
fn quadratic<N: Num>(b: N, c: N) -> Vec<Cx<N>> {
    let discriminant = b * b - N::of(4.0) * c;
    let half = Cx::real(-b / 2.0);
    let spread = if discriminant.v() >= 0.0 {
        Cx::real(discriminant.sqrt() / 2.0)
    } else {
        Cx::new(N::ZERO, (-discriminant).sqrt() / 2.0)
    };
    vec![half + spread, half - spread]
}
