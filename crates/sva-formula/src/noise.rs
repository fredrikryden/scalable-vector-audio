// Concern: builds one deterministic noise realization as a conjugate-symmetric line series | Non-concern: line enumeration (series.rs) | IO: (seed, period, color) -> Series

use std::f64::consts::TAU;

use crate::closed_form::{Body, Bound, IndexId, Part, Series, Unary};
use crate::complex::C64;

pub const LINE: IndexId = IndexId(u32::MAX);

/// `color` is a ratio per octave, as `-3db`: line `k` is `k^log2(color)`, a summable power.
pub fn noise(seed: u64, period: f64, color: f64) -> Series {
    let amplitude = Body::Apply(
        Unary::Exp,
        Part::bare(Body::Mul(vec![
            Part::bare(Body::Const(C64::real(color.log2()))),
            Part::bare(Body::Apply(Unary::Log, Part::bare(Body::Index(LINE)))),
        ])),
    );
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
