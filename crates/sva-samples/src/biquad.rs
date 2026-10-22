// Concern: designs RBJ biquad coefficients and their magnitude response | Non-concern: the shapes and their H(s) (sva-formula), dispatch (filters.rs) | IO: (Shape, cutoff, q, gain, sr) -> Coeffs

use sva_formula::filter::Shape;
use sva_formula::{Cx, Num};

/// Normalized to `a0 = 1`: `y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coeffs<N: Num = f64> {
    pub b0: N,
    pub b1: N,
    pub b2: N,
    pub a1: N,
    pub a2: N,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct State {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl State {
    pub fn held(&self) -> [f64; 4] {
        [self.x1, self.x2, self.y1, self.y2]
    }

    #[inline]
    pub fn step(&mut self, c: &Coeffs, x: f64) -> f64 {
        let y = c.b0 * x + c.b1 * self.x1 + c.b2 * self.x2 - c.a1 * self.y1 - c.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Short of the top tenth, where `tan` runs away.
const HIGHEST_FRACTION_OF_RATE: f64 = 0.45;

/// A sweep past 0 Hz or Nyquist blows the bilinear design up; the clamp is reported.
pub fn clamp_cutoff(hz: f64, sr: f64) -> (f64, bool) {
    let hi = HIGHEST_FRACTION_OF_RATE * sr;
    let lo = hi.min(1.0);
    if hz < lo {
        (lo, true)
    } else if hz > hi {
        (hi, true)
    } else {
        (hz, false)
    }
}

/// -120 dB; a ratio at or under 0 has no root.
pub const MIN_GAIN: f64 = 1e-6;

pub fn clamp_gain(gain: f64) -> (f64, bool) {
    match gain < MIN_GAIN {
        true => (MIN_GAIN, true),
        false => (gain, false),
    }
}

/// Correctness: damping `d = s/c` (`a0, a2 = c ± s`; `b0, b2` for zeros) rounds
/// within `2^-20` of itself from `2^-31`; the ringing bound holds to `2^22`.
pub const DAMPING: (f64, f64) = (1.0 / (1u64 << 31) as f64, (1u64 << 22) as f64);

fn damping_at_unit_q(shape: Shape, cutoff: f64, gain: f64, sr: f64) -> (f64, f64) {
    let w0 = 2.0 * std::f64::consts::PI * cutoff / sr;
    let (cs, half_sn) = (libm::cos(w0), libm::sin(w0) / 2.0);
    let a = gain.sqrt();
    let shelf = |poles_c: f64, zeros_c: f64| {
        let s = 2.0 * a.sqrt() * half_sn;
        (s / poles_c, s / zeros_c)
    };
    match shape {
        Shape::Peaking => (half_sn / a, half_sn * a),
        Shape::Lowshelf => shelf((a + 1.0) + (a - 1.0) * cs, (a + 1.0) - (a - 1.0) * cs),
        Shape::Highshelf => shelf((a + 1.0) - (a - 1.0) * cs, (a + 1.0) + (a - 1.0) * cs),
        _ => (half_sn, half_sn),
    }
}

/// Holds q inside `DAMPING`; zeros no q fits are flagged.
pub fn clamp_q(shape: Shape, cutoff: f64, q: f64, gain: f64, sr: f64) -> (f64, bool) {
    if shape == Shape::OnePole {
        return (q, false);
    }
    let (poles, zeros) = damping_at_unit_q(shape, cutoff, gain, sr);
    let (least, most) = DAMPING;
    let (mut lo, mut hi) = (poles.max(zeros) / most, poles.min(zeros) / least);
    let apart = lo > hi;
    if apart {
        (lo, hi) = (poles / most, poles / least);
    }
    if q < lo || q.is_nan() {
        (lo, true)
    } else if q > hi {
        (hi, true)
    } else {
        (q, apart)
    }
}

/// The RBJ Audio EQ Cookbook's designs, normalized by `a0`; `gain` is a ratio.
pub fn design<N: Num>(shape: Shape, cutoff: N, q: N, gain: N, sr: f64) -> Coeffs<N> {
    let turn = 2.0 * std::f64::consts::PI;
    if shape == Shape::OnePole {
        let rc = N::ONE / (cutoff * turn);
        let dt = 1.0 / sr;
        let alpha = N::of(dt) / (rc + dt);
        return Coeffs {
            b0: alpha,
            b1: N::ZERO,
            b2: N::ZERO,
            a1: alpha - 1.0,
            a2: N::ZERO,
        };
    }

    let w0 = cutoff * turn / sr;
    let cs = w0.cos();
    let sn = w0.sin();
    let alpha = sn / (q * 2.0);
    let a = gain.sqrt();
    let sqrt_a = a.sqrt();
    let one = N::ONE;

    let (b0, b1, b2, a0, a1, a2) = match shape {
        Shape::OnePole => unreachable!("handled above"),
        Shape::Lowpass => (
            (one - cs) / 2.0,
            one - cs,
            (one - cs) / 2.0,
            one + alpha,
            cs * -2.0,
            one - alpha,
        ),
        Shape::Highpass => (
            (one + cs) / 2.0,
            -(one + cs),
            (one + cs) / 2.0,
            one + alpha,
            cs * -2.0,
            one - alpha,
        ),
        Shape::Bandpass => (alpha, N::ZERO, -alpha, one + alpha, cs * -2.0, one - alpha),
        Shape::Notch => (one, cs * -2.0, one, one + alpha, cs * -2.0, one - alpha),
        Shape::Peaking => (
            one + alpha * a,
            cs * -2.0,
            one - alpha * a,
            one + alpha / a,
            cs * -2.0,
            one - alpha / a,
        ),
        Shape::Lowshelf => (
            a * ((a + 1.0) - (a - 1.0) * cs + sqrt_a * 2.0 * alpha),
            a * 2.0 * ((a - 1.0) - (a + 1.0) * cs),
            a * ((a + 1.0) - (a - 1.0) * cs - sqrt_a * 2.0 * alpha),
            (a + 1.0) + (a - 1.0) * cs + sqrt_a * 2.0 * alpha,
            ((a - 1.0) + (a + 1.0) * cs) * -2.0,
            (a + 1.0) + (a - 1.0) * cs - sqrt_a * 2.0 * alpha,
        ),
        Shape::Highshelf => (
            a * ((a + 1.0) + (a - 1.0) * cs + sqrt_a * 2.0 * alpha),
            a * -2.0 * ((a - 1.0) + (a + 1.0) * cs),
            a * ((a + 1.0) + (a - 1.0) * cs - sqrt_a * 2.0 * alpha),
            (a + 1.0) - (a - 1.0) * cs + sqrt_a * 2.0 * alpha,
            ((a - 1.0) - (a + 1.0) * cs) * 2.0,
            (a + 1.0) - (a - 1.0) * cs - sqrt_a * 2.0 * alpha,
        ),
    };

    Coeffs {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

/// `H(e^{i w})` at `hz`: what the recurrence multiplies a held line of that frequency by.
pub fn at<N: Num>(c: &Coeffs<N>, hz: f64, sr: f64) -> Cx<N> {
    let w = 2.0 * std::f64::consts::PI * hz / sr;
    let back = |k: f64| Cx::of(sva_formula::C64::new(libm::cos(k * w), -libm::sin(k * w)));
    let num = Cx::real(c.b0) + Cx::real(c.b1) * back(1.0) + Cx::real(c.b2) * back(2.0);
    let den = Cx::ONE + Cx::real(c.a1) * back(1.0) + Cx::real(c.a2) * back(2.0);
    num / den
}

pub fn magnitude_db(c: &Coeffs, hz: f64, sr: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * hz / sr;
    let (c1, s1) = (libm::cos(w), -libm::sin(w));
    let (c2, s2) = (libm::cos(2.0 * w), -libm::sin(2.0 * w));
    let num_re = c.b0 + c.b1 * c1 + c.b2 * c2;
    let num_im = c.b1 * s1 + c.b2 * s2;
    let den_re = 1.0 + c.a1 * c1 + c.a2 * c2;
    let den_im = c.a1 * s1 + c.a2 * s2;
    let num = (num_re * num_re + num_im * num_im).sqrt();
    let den = (den_re * den_re + den_im * den_im).sqrt();
    if den == 0.0 || num == 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * libm::log10(num / den)
}

pub const RESPONSE_RATIOS: [f64; 9] = [0.125, 0.25, 0.5, 0.707, 1.0, 1.414, 2.0, 4.0, 8.0];

/// A point at or above Nyquist is dropped, not clamped: clamping reports the transform's own
/// zero there as if the filter had put it there.
pub fn response(c: &Coeffs, cutoff: f64, sr: f64) -> Vec<(f64, f64)> {
    RESPONSE_RATIOS
        .iter()
        .map(|r| cutoff * r)
        .filter(|hz| *hz < 0.5 * sr)
        .map(|hz| (hz, magnitude_db(c, hz, sr)))
        .collect()
}
