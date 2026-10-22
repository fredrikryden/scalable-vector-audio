// Concern: the number a lowering folds in: a double, or a double beside its tangents to the varied numbers | Non-concern: which numbers vary (sva-engine) | IO: (f64 or a seed) -> N; N op N -> N

//! A render folds `f64`; a fit folds [`Dual`], the same value beside its first derivatives,
//! so the one lowering yields both and they cannot disagree. Every operation on `f64` here is
//! the call the lowering made before it was generic, so a render's bits are unchanged.

use std::cell::RefCell;
use std::collections::HashMap;
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
    /// What two equal numbers share: a double's canonical bits, and a dual's tangent too.
    type Key: Copy + Ord + Eq + Hash + Debug;

    const ZERO: Self;
    const ONE: Self;

    /// A number no varied number moves.
    fn of(x: f64) -> Self;
    fn v(self) -> f64;
    fn key(self) -> Self::Key;
    /// The same number, `-0.0` read as `0.0`.
    fn canonical(self) -> Self;
    /// `v` moved as this number is: a value found by search beside the derivative of the
    /// formula it rounds.
    fn rebased(self, v: f64) -> Self;
    /// Zero, and moved by no varied number: only this folds away, so a varied zero keeps
    /// its derivative.
    fn is_zero(self) -> bool;
    /// Moved by no varied number.
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
    /// The larger operand itself, as `f64::max` picks it: a tie is the left one's derivative.
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

/// A tangent: each varied number's index beside the derivative, ascending, none zero.
pub type Sparse = Vec<(u32, f64)>;

/// A value and its tangent, held by handle so a dual copies as a double does and the
/// lowering's value semantics stay what they were. The handle names a row of the table
/// [`Tangents::active`] holds on this thread; 0 is no tangent.
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub v: f64,
    t: u32,
}

/// Every tangent one lowering made, each distinct tangent once, so two duals are equal
/// exactly where their values' bits and their handles are.
#[derive(Debug)]
pub struct Tangents {
    rows: Vec<Sparse>,
    ids: HashMap<Vec<(u32, u64)>, u32>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Tangents>> = const { RefCell::new(None) };
}

impl Default for Tangents {
    fn default() -> Tangents {
        Tangents {
            rows: vec![Vec::new()],
            ids: HashMap::new(),
        }
    }
}

impl Tangents {
    /// Runs `f` with `self` the table duals on this thread read and grow, and hands it back.
    pub fn active<R>(self, f: impl FnOnce() -> R) -> (R, Tangents) {
        /// Clears the thread's table however `f` ends, a panic included.
        struct Cleared;
        impl Drop for Cleared {
            fn drop(&mut self) {
                ACTIVE.with(|a| a.borrow_mut().take());
            }
        }
        let held = ACTIVE.with(|a| a.borrow_mut().replace(self));
        assert!(held.is_none(), "one tangent table per thread at a time");
        let cleared = Cleared;
        let out = f();
        let table = ACTIVE
            .with(|a| a.borrow_mut().take())
            .expect("the table set above");
        drop(cleared);
        (out, table)
    }

    pub fn of(&self, x: Dual) -> &[(u32, f64)] {
        &self.rows[x.t as usize]
    }

    fn intern(&mut self, row: Sparse) -> u32 {
        if row.is_empty() {
            return 0;
        }
        let key: Vec<(u32, u64)> = row.iter().map(|(k, d)| (*k, d.to_bits())).collect();
        if let Some(id) = self.ids.get(&key) {
            return *id;
        }
        let id = u32::try_from(self.rows.len()).expect("fewer than 2^32 tangents in one lowering");
        self.rows.push(row);
        self.ids.insert(key, id);
        id
    }
}

/// `a*x + b*y` over two tangents, dropping what cancels to zero.
fn combine(a: f64, x: u32, b: f64, y: u32) -> u32 {
    if (x == 0 || a == 0.0) && (y == 0 || b == 0.0) {
        return 0;
    }
    ACTIVE.with(|cell| {
        let mut held = cell.borrow_mut();
        let table = held
            .as_mut()
            .expect("a dual with a tangent is folded inside Tangents::active");
        let (xs, ys) = (&table.rows[x as usize], &table.rows[y as usize]);
        let mut row = Vec::with_capacity(xs.len() + ys.len());
        let (mut i, mut j) = (0, 0);
        while i < xs.len() || j < ys.len() {
            let (k, d) = match (xs.get(i), ys.get(j)) {
                (Some(p), Some(q)) if p.0 == q.0 => {
                    i += 1;
                    j += 1;
                    (p.0, a * p.1 + b * q.1)
                }
                (Some(p), Some(q)) if p.0 < q.0 => {
                    i += 1;
                    (p.0, a * p.1)
                }
                (Some(_), Some(q)) | (None, Some(q)) => {
                    j += 1;
                    (q.0, b * q.1)
                }
                (Some(p), None) => {
                    i += 1;
                    (p.0, a * p.1)
                }
                (None, None) => unreachable!("the loop runs while either side holds one"),
            };
            if d != 0.0 {
                row.push((k, d));
            }
        }
        table.intern(row)
    })
}

impl Dual {
    /// `v`, moved by the varied numbers `tangent` names, each by its derivative.
    pub fn seeded(v: f64, tangent: &[(u32, f64)]) -> Dual {
        let mut row: Sparse = tangent.iter().copied().filter(|p| p.1 != 0.0).collect();
        row.sort_by_key(|p| p.0);
        let t = ACTIVE.with(|cell| {
            cell.borrow_mut()
                .as_mut()
                .expect("a seed is made inside Tangents::active")
                .intern(row)
        });
        Dual { v, t }
    }

    /// The tangent, read off the table active on this thread.
    pub fn tangent(self) -> Sparse {
        if self.t == 0 {
            return Vec::new();
        }
        ACTIVE.with(|cell| {
            cell.borrow()
                .as_ref()
                .expect("a tangent is read inside Tangents::active")
                .rows[self.t as usize]
                .clone()
        })
    }

    /// `f(v)` beside `f'(v)` times this tangent.
    fn chain(self, value: f64, slope: f64) -> Dual {
        Dual {
            v: value,
            t: combine(slope, self.t, 0.0, 0),
        }
    }
}

impl PartialEq for Dual {
    fn eq(&self, o: &Dual) -> bool {
        self.key() == o.key()
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, o: Dual) -> Dual {
        Dual {
            v: self.v + o.v,
            t: combine(1.0, self.t, 1.0, o.t),
        }
    }
}
impl Sub for Dual {
    type Output = Dual;
    fn sub(self, o: Dual) -> Dual {
        Dual {
            v: self.v - o.v,
            t: combine(1.0, self.t, -1.0, o.t),
        }
    }
}
impl Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        Dual {
            v: self.v * o.v,
            t: combine(o.v, self.t, self.v, o.t),
        }
    }
}
impl Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        let v = self.v / o.v;
        Dual {
            v,
            t: combine(1.0 / o.v, self.t, -v / o.v, o.t),
        }
    }
}
impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        self.chain(-self.v, -1.0)
    }
}
impl Add<f64> for Dual {
    type Output = Dual;
    fn add(self, o: f64) -> Dual {
        Dual {
            v: self.v + o,
            t: self.t,
        }
    }
}
impl Sub<f64> for Dual {
    type Output = Dual;
    fn sub(self, o: f64) -> Dual {
        Dual {
            v: self.v - o,
            t: self.t,
        }
    }
}
impl Mul<f64> for Dual {
    type Output = Dual;
    fn mul(self, o: f64) -> Dual {
        self.chain(self.v * o, o)
    }
}
impl Div<f64> for Dual {
    type Output = Dual;
    fn div(self, o: f64) -> Dual {
        self.chain(self.v / o, 1.0 / o)
    }
}

impl Num for Dual {
    type Key = (u64, u32);
    const ZERO: Dual = Dual { v: 0.0, t: 0 };
    const ONE: Dual = Dual { v: 1.0, t: 0 };

    fn of(x: f64) -> Dual {
        Dual { v: x, t: 0 }
    }
    fn v(self) -> f64 {
        self.v
    }
    fn key(self) -> (u64, u32) {
        (crate::complex::canonical(self.v), self.t)
    }
    fn canonical(self) -> Dual {
        Dual {
            v: f64::from_bits(crate::complex::canonical(self.v)),
            t: self.t,
        }
    }
    fn rebased(self, v: f64) -> Dual {
        Dual { v, t: self.t }
    }
    fn is_zero(self) -> bool {
        self.v == 0.0 && self.t == 0
    }
    fn is_constant(self) -> bool {
        self.t == 0
    }
    fn exp(self) -> Dual {
        let e = crate::exp(self.v);
        self.chain(e, e)
    }
    fn expm1(self) -> Dual {
        self.chain(libm::expm1(self.v), crate::exp(self.v))
    }
    fn sin(self) -> Dual {
        let (s, c) = libm::sincos(self.v);
        self.chain(s, c)
    }
    fn cos(self) -> Dual {
        let (s, c) = libm::sincos(self.v);
        self.chain(c, -s)
    }
    fn sincos(self) -> (Dual, Dual) {
        let (s, c) = libm::sincos(self.v);
        (self.chain(s, c), self.chain(c, -s))
    }
    fn ln(self) -> Dual {
        self.chain(libm::log(self.v), 1.0 / self.v)
    }
    fn sqrt(self) -> Dual {
        let r = self.v.sqrt();
        self.chain(r, 0.5 / r)
    }
    fn powi(self, n: i32) -> Dual {
        let slope = match n {
            0 => 0.0,
            n => f64::from(n) * self.v.powi(n - 1),
        };
        self.chain(self.v.powi(n), slope)
    }
    fn pow(self, e: Dual) -> Dual {
        let v = libm::pow(self.v, e.v);
        let base = match e.v {
            0.0 => 0.0,
            ev => ev * libm::pow(self.v, ev - 1.0),
        };
        let exponent = match e.t {
            0 => 0.0,
            _ => v * libm::log(self.v),
        };
        Dual {
            v,
            t: combine(base, self.t, exponent, e.t),
        }
    }
    fn atan2(self, x: Dual) -> Dual {
        let r = self.v * self.v + x.v * x.v;
        Dual {
            v: libm::atan2(self.v, x.v),
            t: combine(x.v / r, self.t, -self.v / r, x.t),
        }
    }
    fn sinh(self) -> Dual {
        self.chain(libm::sinh(self.v), libm::cosh(self.v))
    }
    fn cosh(self) -> Dual {
        self.chain(libm::cosh(self.v), libm::sinh(self.v))
    }
    fn tanh(self) -> Dual {
        let th = libm::tanh(self.v);
        self.chain(th, 1.0 - th * th)
    }
    fn hypot(self, y: Dual) -> Dual {
        let h = libm::hypot(self.v, y.v);
        Dual {
            v: h,
            t: combine(self.v / h, self.t, y.v / h, y.t),
        }
    }
    fn abs(self) -> Dual {
        match self.v.is_sign_negative() {
            true => -self,
            false => self,
        }
    }
    fn mul_add(self, a: Dual, b: Dual) -> Dual {
        Dual {
            v: self.v.mul_add(a.v, b.v),
            t: combine(1.0, combine(a.v, self.t, self.v, a.t), 1.0, b.t),
        }
    }
    fn rem_euclid(self, b: Dual) -> Dual {
        let q = (self.v - self.v.rem_euclid(b.v)) / b.v;
        Dual {
            v: self.v.rem_euclid(b.v),
            t: combine(1.0, self.t, -q, b.t),
        }
    }
    fn max(self, o: Dual) -> Dual {
        match o.v > self.v || self.v.is_nan() {
            true => o,
            false => self,
        }
    }
    fn min(self, o: Dual) -> Dual {
        match o.v < self.v || self.v.is_nan() {
            true => o,
            false => self,
        }
    }
    fn clamp(self, lo: f64, hi: f64) -> Dual {
        match (self.v < lo, self.v > hi) {
            (true, _) => Dual::of(lo),
            (_, true) => Dual::of(hi),
            _ => self,
        }
    }
}
