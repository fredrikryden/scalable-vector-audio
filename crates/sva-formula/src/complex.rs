// Concern: the complex amplitude every atom factor is measured in | Non-concern: what an amplitude multiplies (spectral_sum/atom.rs) | IO: (C64, C64) -> C64

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::num::Num;

#[derive(Clone, Copy, Debug)]
pub struct Cx<N> {
    pub re: N,
    pub im: N,
}

pub type C64 = Cx<f64>;

/// Bitwise after canonicalizing `-0.0`, so equality is transitive.
impl<N: Num> PartialEq for Cx<N> {
    fn eq(&self, other: &Cx<N>) -> bool {
        self.bits() == other.bits()
    }
}
impl<N: Num> Eq for Cx<N> {}

impl<N: Num> Cx<N> {
    pub const ZERO: Cx<N> = Cx {
        re: N::ZERO,
        im: N::ZERO,
    };
    pub const ONE: Cx<N> = Cx {
        re: N::ONE,
        im: N::ZERO,
    };
    pub const I: Cx<N> = Cx {
        re: N::ZERO,
        im: N::ONE,
    };

    pub fn new(re: N, im: N) -> Cx<N> {
        Cx { re, im }
    }

    pub fn real(re: N) -> Cx<N> {
        Cx { re, im: N::ZERO }
    }

    pub fn of(c: C64) -> Cx<N> {
        Cx::new(N::of(c.re), N::of(c.im))
    }

    pub fn value(self) -> C64 {
        C64::new(self.re.v(), self.im.v())
    }

    pub fn bits(self) -> (N::Key, N::Key) {
        (self.re.key(), self.im.key())
    }

    pub fn is_finite(self) -> bool {
        self.re.v().is_finite() && self.im.v().is_finite()
    }

    pub fn is_zero(self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    pub fn is_real(self) -> bool {
        self.im.is_zero()
    }

    pub fn is_constant(self) -> bool {
        self.re.is_constant() && self.im.is_constant()
    }

    pub fn conj(self) -> Cx<N> {
        Cx::new(self.re, -self.im)
    }

    pub fn norm_sqr(self) -> N {
        self.re * self.re + self.im * self.im
    }

    pub fn abs(self) -> N {
        self.re.hypot(self.im)
    }

    /// A component alone on its axis inverts by one division, correctly rounded; only a
    /// value off both axes takes the conjugate over the squared norm.
    pub fn inv(self) -> Cx<N> {
        let d = self.norm_sqr();
        match (self.re.is_zero(), self.im.is_zero()) {
            (false, true) => Cx::new(N::ONE / self.re, -self.im / d),
            (true, false) => Cx::new(self.re / d, -N::ONE / self.im),
            _ => Cx::new(self.re / d, -self.im / d),
        }
    }

    pub fn exp(self) -> Cx<N> {
        let r = self.re.exp();
        let (sin, cos) = self.im.sincos();
        Cx::new(r * cos, r * sin)
    }

    /// `e^z - 1` without the cancellation `exp() - ONE` meets near zero.
    pub fn exp_m1(self) -> Cx<N> {
        let half = (self.im * 0.5).sin();
        Cx::new(
            self.re.expm1() * self.im.cos() - N::of(2.0) * half * half,
            self.re.exp() * self.im.sin(),
        )
    }

    pub fn powi(self, n: u32) -> Cx<N> {
        let mut out = Cx::ONE;
        for _ in 0..n {
            out = out * self;
        }
        out
    }

    pub fn scale(self, k: N) -> Cx<N> {
        Cx::new(self.re * k, self.im * k)
    }

    pub fn over(self, k: N) -> Cx<N> {
        Cx::new(self.re / k, self.im / k)
    }
}

pub fn canonical(x: f64) -> u64 {
    if x == 0.0 {
        0f64.to_bits()
    } else {
        x.to_bits()
    }
}

impl<N: Num> Add for Cx<N> {
    type Output = Cx<N>;
    fn add(self, o: Cx<N>) -> Cx<N> {
        Cx::new(self.re + o.re, self.im + o.im)
    }
}
impl<N: Num> Sub for Cx<N> {
    type Output = Cx<N>;
    fn sub(self, o: Cx<N>) -> Cx<N> {
        Cx::new(self.re - o.re, self.im - o.im)
    }
}
impl<N: Num> Mul for Cx<N> {
    type Output = Cx<N>;
    fn mul(self, o: Cx<N>) -> Cx<N> {
        Cx::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
impl<N: Num> Div for Cx<N> {
    type Output = Cx<N>;
    fn div(self, o: Cx<N>) -> Cx<N> {
        if o.im.is_zero() && o.re.v() != 0.0 {
            return Cx::new(self.re / o.re, self.im / o.re);
        }
        if o.re.is_zero() && o.im.v() != 0.0 {
            return Cx::new(self.im / o.im, -self.re / o.im);
        }
        let d = o.norm_sqr();
        Cx::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}
impl<N: Num> Neg for Cx<N> {
    type Output = Cx<N>;
    fn neg(self) -> Cx<N> {
        Cx::new(-self.re, -self.im)
    }
}
