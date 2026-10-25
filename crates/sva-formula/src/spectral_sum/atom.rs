// Concern: declares SpectralAtom, its six factors, its derivative and its sort key | Non-concern: multiplying two (product.rs), building one from a Body (build.rs) | IO: (an atom) -> its key

use crate::closed_form::Edge;
use crate::complex::{C64, Cx};
use crate::num::Num;
use crate::origin::Origin;
use crate::refusal::{AtomSketch, Factor, Left, LeftReason};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exp<N: Num = f64> {
    pub sigma: N,
    pub omega: N,
    /// `e^{sigma*(t - mu)}`: the growing half is read from a reference point, so a factor
    /// bounded on its own window is not held as an overflow times an underflow.
    pub mu: N,
}

impl<N: Num> Exp<N> {
    pub fn at(sigma: N, omega: N) -> Exp<N> {
        Exp {
            sigma,
            omega,
            mu: N::ZERO,
        }
    }

    /// The same factor written from the origin, or nothing where that overflows.
    pub fn flattened(self) -> Option<(Exp<N>, Cx<N>)> {
        if self.mu.is_zero() || self.sigma.is_zero() {
            return Some((Exp::at(self.sigma, self.omega), Cx::ONE));
        }
        let carried = Cx::real(-self.sigma * self.mu).exp();
        carried
            .is_finite()
            .then_some((Exp::at(self.sigma, self.omega), carried))
    }
}

/// `(t - at)^degree`: a line keeps the instant it is zero at, where a window opening reads it
/// as exactly zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poly<N: Num = f64> {
    pub degree: u16,
    pub at: N,
}

impl<N: Num> Default for Poly<N> {
    fn default() -> Poly<N> {
        Poly::ONE
    }
}

impl<N: Num> Poly<N> {
    pub const ONE: Poly<N> = Poly {
        degree: 0,
        at: N::ZERO,
    };

    pub fn power(degree: u16) -> Poly<N> {
        Poly {
            degree,
            at: N::ZERO,
        }
    }

    pub fn is_one(self) -> bool {
        self.degree == 0
    }

    pub fn value(self, x: N) -> N {
        (x - self.at).powi(i32::from(self.degree))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gauss<N: Num = f64> {
    pub a: N,
    pub mu: N,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Indicator<N: Num = f64> {
    pub l: Edge<N>,
    pub r: Edge<N>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pole<N: Num = f64> {
    pub at: Cx<N>,
    pub order: u16,
    pub pv: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Singular<N: Num = f64> {
    Regular,
    Delta { at: N, order: u16 },
}

/// The five smooth factors beside the amplitude, absent where `None`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Factors<N: Num = f64> {
    pub poly: Poly<N>,
    pub exp: Option<Exp<N>>,
    pub gauss: Option<Gauss<N>>,
    pub ind: Option<Indicator<N>>,
    pub pole: Option<Pole<N>>,
}

impl<N: Num> Default for Factors<N> {
    fn default() -> Factors<N> {
        Factors::NONE
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpectralAtom<N: Num = f64> {
    pub c: Cx<N>,
    pub poly: Poly<N>,
    pub exp: Option<Exp<N>>,
    pub gauss: Option<Gauss<N>>,
    pub ind: Option<Indicator<N>>,
    pub pole: Option<Pole<N>>,
    pub sing: Singular<N>,
    pub origin: Origin,
}

/// Every field but `c` and `origin`. Equal keys are like atoms, and the hash agrees.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SpectralAtomKey<K = u64, E = Edge> {
    sing: (u8, K, u16),
    pole: (u8, K, K, u16, bool),
    ind: (u8, E, E),
    gauss: (u8, K, K),
    poly: (u16, K),
    exp: (u8, K, K, K),
}

pub type KeyOf<N> = SpectralAtomKey<<N as Num>::Key, Edge<N>>;

/// A real exponential beside a Gaussian is that Gaussian recentred: one spelling survives.
fn absorb_shift<N: Num>(atom: SpectralAtom<N>) -> SpectralAtom<N> {
    let (Some(g), Some(e)) = (atom.gauss, atom.exp) else {
        return atom;
    };
    if e.sigma.is_zero() || !e.mu.is_zero() {
        return atom;
    }
    SpectralAtom {
        c: atom
            .c
            .scale((e.sigma * g.mu + e.sigma * e.sigma / (N::of(4.0) * g.a)).exp()),
        gauss: Some(Gauss {
            mu: g.mu + e.sigma / (N::of(2.0) * g.a),
            ..g
        }),
        exp: (!e.omega.is_zero()).then_some(Exp::at(N::ZERO, e.omega)),
        ..atom
    }
}

/// `origin` is excluded, so two spellings written in different files are one atom.
impl<N: Num> PartialEq for SpectralAtom<N> {
    fn eq(&self, other: &SpectralAtom<N>) -> bool {
        self.c == other.c && self.key() == other.key()
    }
}

impl<N: Num> Indicator<N> {
    pub const ALL: Indicator<N> = Indicator {
        l: Edge::NegInf,
        r: Edge::PosInf,
    };

    /// Half-open, per FORMAT 3.1: two windows that abut share no instant, so a signal cut
    /// into abutting segments sums to itself rather than to a doubled seam.
    pub fn contains(self, x: f64) -> bool {
        self.l.value() <= x && x < self.r.value()
    }

    pub fn is_empty(self) -> bool {
        self.l.value() >= self.r.value()
    }

    /// Two windows meet in one window, which is why a crop of a crop is one atom.
    pub fn meet(self, other: Indicator<N>) -> Indicator<N> {
        Indicator {
            l: self.l.max(other.l),
            r: self.r.min(other.r),
        }
    }
}

impl<N: Num> Factors<N> {
    pub const NONE: Factors<N> = Factors {
        poly: Poly::ONE,
        exp: None,
        gauss: None,
        ind: None,
        pole: None,
    };

    pub fn poly(n: u16) -> Factors<N> {
        Factors {
            poly: Poly::power(n),
            ..Factors::NONE
        }
    }
}

impl SpectralAtom {
    pub fn smooth_at(&self, x: f64) -> Option<C64> {
        self.smooth_at_over(x)
    }

    /// `smooth_at` at an `x` its caller already placed inside the window.
    pub fn smooth_inside(&self, x: f64) -> Option<C64> {
        self.smooth_inside_over(x)
    }

    /// The same atom over another number.
    pub fn lifted<N: Num>(&self) -> SpectralAtom<N> {
        let edge = |e: Edge| Edge::of(e);
        SpectralAtom {
            c: Cx::of(self.c),
            poly: Poly {
                degree: self.poly.degree,
                at: N::of(self.poly.at),
            },
            exp: self.exp.map(|e| Exp {
                sigma: N::of(e.sigma),
                omega: N::of(e.omega),
                mu: N::of(e.mu),
            }),
            gauss: self.gauss.map(|g| Gauss {
                a: N::of(g.a),
                mu: N::of(g.mu),
            }),
            ind: self.ind.map(|i| Indicator {
                l: edge(i.l),
                r: edge(i.r),
            }),
            pole: self.pole.map(|p| Pole {
                at: Cx::of(p.at),
                order: p.order,
                pv: p.pv,
            }),
            sing: match self.sing {
                Singular::Regular => Singular::Regular,
                Singular::Delta { at, order } => Singular::Delta {
                    at: N::of(at),
                    order,
                },
            },
            origin: self.origin,
        }
    }
}

impl<N: Num> SpectralAtom<N> {
    /// Folds every absent factor to `None`, and reduces a zeroth-order delta at its point. An
    /// amplitude past a double is refused where a sum is normalized and where one is read.
    pub fn new(
        c: Cx<N>,
        factors: Factors<N>,
        sing: Singular<N>,
        origin: Origin,
    ) -> SpectralAtom<N> {
        let atom = SpectralAtom {
            c,
            poly: match factors.poly.degree {
                0 => Poly::ONE,
                _ => Poly {
                    at: factors.poly.at.canonical(),
                    ..factors.poly
                },
            },
            exp: factors
                .exp
                .filter(|e| !e.sigma.is_zero() || !e.omega.is_zero())
                .map(|e| match e.sigma.is_zero() {
                    true => Exp::at(N::ZERO, e.omega),
                    false => e,
                }),
            gauss: factors.gauss,
            ind: factors.ind.filter(|i| *i != Indicator::ALL),
            pole: factors.pole.filter(|p| p.order > 0),
            sing: Singular::Regular,
            origin,
        };
        assert!(atom.poly.at.v().is_finite());
        if let Some(e) = atom.exp {
            assert!(e.sigma.v().is_finite() && e.omega.v().is_finite() && e.mu.v().is_finite());
        }
        if let Some(g) = atom.gauss {
            assert!(g.a.v().is_finite() && g.a.v() > 0.0 && g.mu.v().is_finite());
        }
        if let Some(p) = atom.pole {
            assert!(p.at.is_finite());
        }
        let atom = absorb_shift(atom);
        match sing {
            Singular::Regular => atom,
            Singular::Delta { at, order } => {
                assert!(at.v().is_finite());
                let weight = match order {
                    0 => atom.smooth_at_over(at).unwrap_or(Cx::ZERO),
                    _ => {
                        assert!(
                            atom.is_bare(),
                            "a delta derivative reduces by Leibniz before it reaches an atom"
                        );
                        c
                    }
                };
                SpectralAtom {
                    c: weight,
                    poly: Poly::ONE,
                    exp: None,
                    gauss: None,
                    ind: None,
                    pole: None,
                    sing: Singular::Delta {
                        at: at.canonical(),
                        order,
                    },
                    origin,
                }
            }
        }
    }

    pub fn constant(c: Cx<N>, origin: Origin) -> SpectralAtom<N> {
        SpectralAtom::new(c, Factors::NONE, Singular::Regular, origin)
    }

    pub fn is_bare(&self) -> bool {
        self.poly.is_one()
            && self.exp.is_none()
            && self.gauss.is_none()
            && self.ind.is_none()
            && self.pole.is_none()
    }

    pub fn is_delta(&self) -> bool {
        !matches!(self.sing, Singular::Regular)
    }

    pub fn factors(&self) -> Factors<N> {
        Factors {
            poly: self.poly,
            exp: self.exp,
            gauss: self.gauss,
            ind: self.ind,
            pole: self.pole,
        }
    }

    /// `(t - at)^n` written out in powers of `t`.
    pub fn flattened(&self) -> Vec<SpectralAtom<N>> {
        let poly = self.poly;
        if poly.at.is_zero() {
            return vec![*self];
        }
        let mut binomial = 1.0;
        let mut out = Vec::with_capacity(usize::from(poly.degree) + 1);
        for k in (0..=poly.degree).rev() {
            let power = i32::from(poly.degree - k);
            out.push(self.with(
                self.c.scale(N::of(binomial) * (-poly.at).powi(power)),
                Factors {
                    poly: Poly::power(k),
                    ..self.factors()
                },
            ));
            binomial = binomial * f64::from(k) / (f64::from(poly.degree - k) + 1.0);
        }
        out
    }

    pub fn with(&self, c: Cx<N>, factors: Factors<N>) -> SpectralAtom<N> {
        SpectralAtom::new(c, factors, self.sing, self.origin)
    }

    pub fn smooth_at_over(&self, x: N) -> Option<Cx<N>> {
        if self.ind.is_some_and(|i| !i.contains(x.v())) {
            return Some(Cx::ZERO);
        }
        self.smooth_inside_over(x)
    }

    /// `smooth_at` at an `x` its caller already placed inside the window.
    #[inline(always)]
    pub fn smooth_inside_over(&self, x: N) -> Option<Cx<N>> {
        let mut v = self.c;
        if !self.poly.is_one() {
            v = v.scale(self.poly.value(x));
        }
        if let Some(e) = self.exp {
            v = v * Cx::new(e.sigma * (x - e.mu), e.omega * x).exp();
        }
        if let Some(g) = self.gauss {
            v = v.scale((-g.a * (x - g.mu) * (x - g.mu)).exp());
        }
        if let Some(p) = self.pole {
            let d = Cx::real(x) - p.at;
            if d.is_zero() {
                return None;
            }
            v = v / d.powi(u32::from(p.order));
        }
        Some(v)
    }

    /// The product rule over the six factors; the indicator's two terms are edge deltas. An
    /// order or degree past `u16` refuses.
    pub fn derivative(&self) -> Result<Vec<SpectralAtom<N>>, Left> {
        let next = |order: u16, factor: Factor, reason: LeftReason| {
            let left = Left::new(self.origin, AtomSketch::of(factor), reason);
            order.checked_add(1).ok_or(left)
        };
        if let Singular::Delta { at, order } = self.sing {
            return Ok(vec![SpectralAtom::new(
                self.c,
                Factors::NONE,
                Singular::Delta {
                    at,
                    order: next(order, Factor::Delta, LeftReason::Overflow)?,
                },
                self.origin,
            )]);
        }
        let f = self.factors();
        let mut out = Vec::new();
        let poly = self.poly;
        if poly.degree > 0 {
            out.push(self.with(
                self.c.scale(N::of(f64::from(poly.degree))),
                Factors {
                    poly: Poly {
                        degree: poly.degree - 1,
                        ..poly
                    },
                    ..f
                },
            ));
        }
        if let Some(e) = self.exp {
            out.push(self.with(self.c * Cx::new(e.sigma, e.omega), f));
        }
        if let Some(g) = self.gauss {
            let raised = Poly {
                degree: next(poly.degree, Factor::Polynomial, LeftReason::Overflow)?,
                at: match poly.degree {
                    0 => N::ZERO,
                    _ => poly.at,
                },
            };
            out.push(self.with(
                self.c.scale(N::of(-2.0) * g.a),
                Factors { poly: raised, ..f },
            ));
            out.push(self.with(self.c.scale(N::of(2.0) * g.a * (g.mu - raised.at)), f));
        }
        if let Some(p) = self.pole {
            out.push(self.with(
                self.c.scale(N::of(-f64::from(p.order))),
                Factors {
                    pole: Some(Pole {
                        order: next(p.order, Factor::Pole, LeftReason::PoleOrder(u16::MAX))?,
                        ..p
                    }),
                    ..f
                },
            ));
        }
        if let Some(i) = self.ind {
            let open = SpectralAtom::new(
                self.c,
                Factors { ind: None, ..f },
                Singular::Regular,
                self.origin,
            );
            for (edge, sign) in [(i.l, 1.0), (i.r, -1.0)] {
                let Edge::At(at) = edge else { continue };
                let Some(w) = open.smooth_at_over(at) else {
                    continue;
                };
                out.push(SpectralAtom::new(
                    w.scale(N::of(sign)),
                    Factors::NONE,
                    Singular::Delta { at, order: 0 },
                    self.origin,
                ));
            }
        }
        Ok(out)
    }

    /// Every field but `c`, `origin` and the line frequency: atoms alike under this key are
    /// one line spectrum times one common factor. Meaningful only where `sigma` is zero.
    pub fn key_without_line(&self) -> KeyOf<N> {
        let z = N::ZERO.key();
        SpectralAtomKey {
            exp: (0, z, z, z),
            ..self.key()
        }
    }

    /// Groups the atoms a piecewise window fold may combine.
    pub fn key_without_window(&self) -> KeyOf<N> {
        SpectralAtomKey {
            ind: (0, Edge::NegInf, Edge::NegInf),
            ..self.key()
        }
    }

    pub fn key(&self) -> KeyOf<N> {
        let z = N::ZERO.key();
        let (sing_tag, sing_at, sing_order) = match self.sing {
            Singular::Regular => (0u8, z, 0u16),
            Singular::Delta { at, order } => (1, at.key(), order),
        };
        let pole = self.pole.map_or((0u8, z, z, 0u16, false), |p| {
            let (re, im) = p.at.bits();
            (1, re, im, p.order, p.pv)
        });
        let ind = self
            .ind
            .map_or((0u8, Edge::NegInf, Edge::NegInf), |i| (1, i.l, i.r));
        let gauss = self
            .gauss
            .map_or((0u8, z, z), |g| (1, g.a.key(), g.mu.key()));
        let exp = self.exp.map_or((0u8, z, z, z), |e| {
            (1, e.sigma.key(), e.omega.key(), e.mu.key())
        });
        SpectralAtomKey {
            sing: (sing_tag, sing_at, sing_order),
            pole,
            ind,
            gauss,
            poly: (self.poly.degree, self.poly.at.key()),
            exp,
        }
    }

    pub fn present(&self) -> Vec<Factor> {
        let mut out = Vec::new();
        if !self.poly.is_one() {
            out.push(Factor::Polynomial);
        }
        if self.exp.is_some() {
            out.push(Factor::Exponential);
        }
        if self.gauss.is_some() {
            out.push(Factor::Gaussian);
        }
        if self.ind.is_some() {
            out.push(Factor::Indicator);
        }
        match self.pole {
            Some(p) if p.pv => out.push(Factor::PrincipalValue),
            Some(_) => out.push(Factor::Pole),
            None => {}
        }
        if self.is_delta() {
            out.push(Factor::Delta);
        }
        if out.is_empty() {
            out.push(Factor::Amplitude);
        }
        out
    }
}
