// Concern: builds one deterministic noise realization as a conjugate-symmetric line series, in a band of lines | Non-concern: line enumeration (series.rs) | IO: (seed, period, color, band) -> Series

use std::f64::consts::TAU;

use crate::closed_form::{Body, Bound, IndexId, Part, Series, Unary};
use crate::complex::C64;

pub const LINE: IndexId = IndexId(u32::MAX);

pub const EVERY_LINE: (i64, Bound) = (1, Bound::Infinite);

/// Exact by definition: `k >= 1`, `lo*period <= k <= hi*period` in f64; `None` past i64.
pub fn band(period: f64, lo: f64, hi: Option<f64>) -> Option<(i64, Bound)> {
    let index = |x: f64| (x < i64::MAX as f64).then_some(x as i64);
    let first = index(libm::ceil(lo * period))?.max(1);
    let last = match hi {
        Some(hi) => Bound::Finite(index(libm::floor(hi * period))?),
        None => Bound::Infinite,
    };
    Some((first, last))
}

/// Line `k`, at `f = k/period` Hz, is `f^log2(color)/sqrt(period)`: a level a hertz no
/// period moves.
pub fn noise(seed: u64, period: f64, color: f64) -> Series {
    noise_in(seed, period, color, EVERY_LINE)
}

pub fn noise_in(seed: u64, period: f64, color: f64, (first, last): (i64, Bound)) -> Series {
    let tilt = libm::log2(color);
    let amplitude = Body::Mul(vec![
        Part::bare(Body::Const(C64::real(libm::pow(period, -(tilt + 0.5))))),
        Part::bare(Body::Apply(
            Unary::Exp,
            Part::bare(Body::Mul(vec![
                Part::bare(Body::Const(C64::real(tilt))),
                Part::bare(Body::Apply(Unary::Log, Part::bare(Body::Index(LINE)))),
            ])),
        )),
    ]);
    let phase = Body::Mul(vec![
        Part::bare(Body::Const(C64::real(TAU))),
        Part::bare(Body::Keyed {
            seed,
            of: Part::bare(Body::Index(LINE)),
        }),
    ]);
    let turning = Body::Mul(vec![
        Part::bare(Body::Const(C64::real(TAU / period))),
        Part::bare(Body::Index(LINE)),
        Part::bare(Body::Line),
    ]);
    Series {
        index: LINE,
        lo: first,
        hi: last,
        term: Part::bare(Body::Mul(vec![
            Part::bare(amplitude),
            Part::bare(Body::Apply(
                Unary::Cos,
                Part::bare(Body::Add(vec![Part::bare(turning), Part::bare(phase)])),
            )),
        ])),
    }
}
