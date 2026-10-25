// Concern: a number beside its first derivative along one seeded direction, the forward mode every atom tangent is read in | Non-concern: which number is seeded (atoms.rs) | IO: (f64, f64) -> Dual

use std::ops::{Add, Div, Mul, Neg, Sub};

use sva_formula::Num;

/// `value + tangent * e` with `e^2 = 0`: each operation carries its exact derivative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub value: f64,
    pub tangent: f64,
}

impl Dual {
    pub const fn new(value: f64, tangent: f64) -> Dual {
        Dual { value, tangent }
    }

    fn chain(self, value: f64, slope: f64) -> Dual {
        Dual::new(value, slope * self.tangent)
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, o: Dual) -> Dual {
        Dual::new(self.value + o.value, self.tangent + o.tangent)
    }
}

impl Sub for Dual {
    type Output = Dual;
    fn sub(self, o: Dual) -> Dual {
        Dual::new(self.value - o.value, self.tangent - o.tangent)
    }
}

impl Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        Dual::new(
            self.value * o.value,
            self.tangent * o.value + self.value * o.tangent,
        )
    }
}

impl Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        let value = self.value / o.value;
        Dual::new(value, (self.tangent - value * o.tangent) / o.value)
    }
}

impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual::new(-self.value, -self.tangent)
    }
}

impl Add<f64> for Dual {
    type Output = Dual;
    fn add(self, o: f64) -> Dual {
        Dual::new(self.value + o, self.tangent)
    }
}

impl Sub<f64> for Dual {
    type Output = Dual;
    fn sub(self, o: f64) -> Dual {
        Dual::new(self.value - o, self.tangent)
    }
}

impl Mul<f64> for Dual {
    type Output = Dual;
    fn mul(self, o: f64) -> Dual {
        Dual::new(self.value * o, self.tangent * o)
    }
}

impl Div<f64> for Dual {
    type Output = Dual;
    fn div(self, o: f64) -> Dual {
        Dual::new(self.value / o, self.tangent / o)
    }
}

impl Num for Dual {
    type Key = (u64, u64);
    const ZERO: Dual = Dual::new(0.0, 0.0);
    const ONE: Dual = Dual::new(1.0, 0.0);

    fn of(x: f64) -> Dual {
        Dual::new(x, 0.0)
    }
    fn v(self) -> f64 {
        self.value
    }
    fn key(self) -> (u64, u64) {
        (self.value.key(), self.tangent.key())
    }
    fn canonical(self) -> Dual {
        Dual::new(self.value.canonical(), self.tangent.canonical())
    }
    fn rebased(self, v: f64) -> Dual {
        Dual::new(v, self.tangent)
    }
    fn is_zero(self) -> bool {
        self.value == 0.0 && self.tangent == 0.0
    }
    fn is_constant(self) -> bool {
        self.tangent == 0.0
    }

    fn exp(self) -> Dual {
        let e = Num::exp(self.value);
        self.chain(e, e)
    }
    fn expm1(self) -> Dual {
        self.chain(Num::expm1(self.value), Num::exp(self.value))
    }
    fn sin(self) -> Dual {
        let (s, c) = Num::sincos(self.value);
        self.chain(s, c)
    }
    fn cos(self) -> Dual {
        let (s, c) = Num::sincos(self.value);
        self.chain(c, -s)
    }
    fn sincos(self) -> (Dual, Dual) {
        let (s, c) = Num::sincos(self.value);
        (self.chain(s, c), self.chain(c, -s))
    }
    fn ln(self) -> Dual {
        self.chain(Num::ln(self.value), 1.0 / self.value)
    }
    fn sqrt(self) -> Dual {
        let r = Num::sqrt(self.value);
        self.chain(r, 0.5 / r)
    }
    fn powi(self, n: i32) -> Dual {
        let slope = f64::from(n) * Num::powi(self.value, n - 1);
        self.chain(Num::powi(self.value, n), slope)
    }
    fn pow(self, e: Dual) -> Dual {
        let value = Num::pow(self.value, e.value);
        let along_base = match e.value == 0.0 {
            true => 0.0,
            false => e.value * Num::pow(self.value, e.value - 1.0),
        };
        let along_exponent = match e.tangent == 0.0 {
            true => 0.0,
            false => value * Num::ln(self.value),
        };
        Dual::new(
            value,
            along_base * self.tangent + along_exponent * e.tangent,
        )
    }
    fn atan2(self, x: Dual) -> Dual {
        let r2 = self.value * self.value + x.value * x.value;
        Dual::new(
            Num::atan2(self.value, x.value),
            (x.value * self.tangent - self.value * x.tangent) / r2,
        )
    }
    fn sinh(self) -> Dual {
        self.chain(Num::sinh(self.value), Num::cosh(self.value))
    }
    fn cosh(self) -> Dual {
        self.chain(Num::cosh(self.value), Num::sinh(self.value))
    }
    fn tanh(self) -> Dual {
        let t = Num::tanh(self.value);
        self.chain(t, 1.0 - t * t)
    }
    fn hypot(self, y: Dual) -> Dual {
        let h = Num::hypot(self.value, y.value);
        let tangent = match h == 0.0 {
            true => 0.0,
            false => (self.value * self.tangent + y.value * y.tangent) / h,
        };
        Dual::new(h, tangent)
    }
    fn abs(self) -> Dual {
        match self.value < 0.0 {
            true => -self,
            false => self,
        }
    }
    fn mul_add(self, a: Dual, b: Dual) -> Dual {
        Dual::new(
            Num::mul_add(self.value, a.value, b.value),
            self.tangent * a.value + self.value * a.tangent + b.tangent,
        )
    }
    fn rem_euclid(self, b: Dual) -> Dual {
        let value = Num::rem_euclid(self.value, b.value);
        let turns = (self.value - value) / b.value;
        Dual::new(value, self.tangent - turns * b.tangent)
    }
    fn max(self, o: Dual) -> Dual {
        match Num::max(self.value, o.value) == self.value && !self.value.is_nan() {
            true => self,
            false => o,
        }
    }
    fn min(self, o: Dual) -> Dual {
        match Num::min(self.value, o.value) == self.value && !self.value.is_nan() {
            true => self,
            false => o,
        }
    }
    fn clamp(self, lo: f64, hi: f64) -> Dual {
        match self.value {
            v if v < lo => Dual::of(lo),
            v if v > hi => Dual::of(hi),
            _ => self,
        }
    }
}
