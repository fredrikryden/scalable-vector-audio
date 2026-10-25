// Concern: the number a lowering folds in, the operations every form reads it through | Non-concern: what a lowering builds of it (sva-engine) | IO: (f64) -> N; N op N -> N

//! Every operation on `f64` here is the call the lowering made before it was generic, so a
//! render's bits are unchanged.

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Num:
    Copy
    + Debug
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + 'static
{
    /// What two equal numbers share: a double's canonical bits.
    type Key: Copy + Ord + Eq + Hash + Debug;

    const ZERO: Self;
    const ONE: Self;

    fn of(x: f64) -> Self;
    fn v(self) -> f64;
    fn key(self) -> Self::Key;
    /// The same number, `-0.0` read as `0.0`.
    fn canonical(self) -> Self;
    fn rebased(self, v: f64) -> Self;
    fn is_zero(self) -> bool;
    fn is_constant(self) -> bool;

    fn exp(self) -> Self;
    fn expm1(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn sincos(self) -> (Self, Self);
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn pow(self, e: Self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn hypot(self, y: Self) -> Self;
    fn abs(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn rem_euclid(self, b: Self) -> Self;
    /// The larger operand itself, as `f64::max` picks it.
    fn max(self, o: Self) -> Self;
    fn min(self, o: Self) -> Self;
    fn clamp(self, lo: f64, hi: f64) -> Self;
}

impl Num for f64 {
    type Key = u64;
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;

    #[inline(always)]
    fn of(x: f64) -> f64 {
        x
    }
    #[inline(always)]
    fn v(self) -> f64 {
        self
    }
    #[inline(always)]
    fn key(self) -> u64 {
        crate::complex::canonical(self)
    }
    #[inline(always)]
    fn canonical(self) -> f64 {
        f64::from_bits(crate::complex::canonical(self))
    }
    #[inline(always)]
    fn rebased(self, v: f64) -> f64 {
        v
    }
    #[inline(always)]
    fn is_zero(self) -> bool {
        self == 0.0
    }
    #[inline(always)]
    fn is_constant(self) -> bool {
        true
    }
    #[inline(always)]
    fn exp(self) -> f64 {
        crate::exp(self)
    }
    #[inline(always)]
    fn expm1(self) -> f64 {
        libm::expm1(self)
    }
    #[inline(always)]
    fn sin(self) -> f64 {
        libm::sin(self)
    }
    #[inline(always)]
    fn cos(self) -> f64 {
        libm::cos(self)
    }
    #[inline(always)]
    fn sincos(self) -> (f64, f64) {
        libm::sincos(self)
    }
    #[inline(always)]
    fn ln(self) -> f64 {
        libm::log(self)
    }
    #[inline(always)]
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn powi(self, n: i32) -> f64 {
        f64::powi(self, n)
    }
    #[inline(always)]
    fn pow(self, e: f64) -> f64 {
        libm::pow(self, e)
    }
    #[inline(always)]
    fn atan2(self, x: f64) -> f64 {
        libm::atan2(self, x)
    }
    #[inline(always)]
    fn sinh(self) -> f64 {
        libm::sinh(self)
    }
    #[inline(always)]
    fn cosh(self) -> f64 {
        libm::cosh(self)
    }
    #[inline(always)]
    fn tanh(self) -> f64 {
        libm::tanh(self)
    }
    #[inline(always)]
    fn hypot(self, y: f64) -> f64 {
        libm::hypot(self, y)
    }
    #[inline(always)]
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    #[inline(always)]
    fn mul_add(self, a: f64, b: f64) -> f64 {
        f64::mul_add(self, a, b)
    }
    #[inline(always)]
    fn rem_euclid(self, b: f64) -> f64 {
        f64::rem_euclid(self, b)
    }
    #[inline(always)]
    fn max(self, o: f64) -> f64 {
        f64::max(self, o)
    }
    #[inline(always)]
    fn min(self, o: f64) -> f64 {
        f64::min(self, o)
    }
    #[inline(always)]
    fn clamp(self, lo: f64, hi: f64) -> f64 {
        f64::clamp(self, lo, hi)
    }
}
