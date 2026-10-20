// Concern: builds one deterministic noise realization as a conjugate-symmetric line series | Non-concern: line enumeration (series.rs) | IO: (seed, period, color) -> Series

use std::f64::consts::TAU;

use crate::closed_form::{Body, Bound, IndexId, Part, Series, Unary};
use crate::complex::C64;

pub const LINE: IndexId = IndexId(u32::MAX);

/// Line `k`, at `f = k/period` Hz, is `f^log2(color)/sqrt(period)`: a level a hertz no
/// period moves.
pub fn noise(seed: u64, period: f64, color: f64) -> Series {
    let tilt = color.log2();
    let amplitude = Body::Mul(vec![
        Part::bare(Body::Const(C64::real(period.powf(-(tilt + 0.5))))),
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
        lo: 1,
        hi: Bound::Infinite,
        term: Part::bare(Body::Mul(vec![
            Part::bare(amplitude),
            Part::bare(Body::Apply(
                Unary::Cos,
                Part::bare(Body::Add(vec![Part::bare(turning), Part::bare(phase)])),
            )),
        ])),
    }
}
