// Concern: states that a closed form read span by span writes the samples one whole read writes | Non-concern: which row a form takes (collapse.rs) | IO: (a ClosedForm) -> bit-equal samples

use crate::helpers::part;
use std::f64::consts::TAU;

use sva_formula::{Body, C64, ClosedForm, Origin, Unary, Var};
use sva_samples::{Grid, PSYCHOACOUSTIC_V1, Rows};

const RATE: u32 = 8_000;
const LEN: usize = 4_000;

fn form(var: Var, body: Body) -> ClosedForm {
    ClosedForm {
        var,
        body,
        origin: Origin::new(0),
    }
}

fn cosine(hz: f64, amp: f64) -> Body {
    Body::Mul(vec![
        part(Body::Const(C64::real(amp))),
        part(Body::Apply(
            Unary::Cos,
            part(Body::Mul(vec![
                part(Body::Const(C64::real(TAU * hz))),
                part(Body::Line),
            ])),
        )),
    ])
}

fn blocks(form: &ClosedForm, start: i64, block: i64) -> Vec<f64> {
    let rows = Rows::of(form, Grid::of(RATE), &PSYCHOACOUSTIC_V1).expect("rows");
    let mut out = Vec::new();
    let mut from = start;
    while from < LEN as i64 {
        let to = (from + block).min(LEN as i64);
        out.extend(rows.planes(from, to).expect("a span").swap_remove(0));
        from = to;
    }
    out
}

fn whole(form: &ClosedForm, start: i64) -> Vec<f64> {
    let rows = Rows::of(form, Grid::of(RATE), &PSYCHOACOUSTIC_V1).expect("rows");
    let mut planes = rows.planes(start, LEN as i64).expect("a whole read");
    planes.swap_remove(0)
}

/// Three summed lines and a point-sampled `tanh` each read one instant at a time, so spans
/// agree with one whole read bit for bit, wherever on the grid either starts.
#[test]
fn a_form_read_one_instant_at_a_time_is_the_same_in_any_span() {
    let chord = form(
        Var::T,
        Body::Add(vec![
            part(cosine(220.0, 0.5)),
            part(cosine(330.0, 0.25)),
            part(cosine(0.0, 0.125)),
        ]),
    );
    let shaped = form(Var::T, Body::Apply(Unary::Tanh, part(cosine(110.0, 3.0))));
    for written in [chord, shaped] {
        for start in [0, 777] {
            let want = whole(&written, start);
            assert!(want.iter().any(|v| *v != 0.0), "silence tests nothing");
            for block in [1, 128, 999, LEN as i64] {
                assert_eq!(
                    blocks(&written, start, block),
                    want,
                    "blocks of {block} from {start}"
                );
            }
        }
    }
}
