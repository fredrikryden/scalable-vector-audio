// Concern: states that a series or spectrum under a crop answers its own law inside the window, nothing outside it | Non-concern: which row a windowed form takes | IO: (a ClosedForm) -> Buffer

use crate::helpers::render;
use sva_formula::{Body, ClosedForm, Edge, Origin, Part, Var};
use sva_samples::{Extent, PSYCHOACOUSTIC_V1};

const RATE: u32 = 48_000;

fn form(body: Body) -> ClosedForm {
    ClosedForm {
        var: Var::T,
        body,
        origin: Origin::new(0),
    }
}

fn cropped(of: Body, to_secs: f64) -> Body {
    Body::Crop {
        of: Part::bare(of),
        l: Edge::at(0.0),
        r: Edge::at(to_secs),
        rise: 0.0,
        fall: 0.0,
    }
}

fn sine(hz: f64) -> Body {
    Body::Apply(
        sva_formula::Unary::Sin,
        Part::bare(Body::Mul(vec![
            Part::bare(Body::Const(sva_formula::C64::real(
                std::f64::consts::TAU * hz,
            ))),
            Part::bare(Body::Line),
        ])),
    )
}

fn decaying(rate_per_sec: f64) -> Body {
    Body::Apply(
        sva_formula::Unary::Exp,
        Part::bare(Body::Mul(vec![
            Part::bare(Body::Const(sva_formula::C64::real(-rate_per_sec))),
            Part::bare(Body::Line),
        ])),
    )
}

/// A decaying lane is swept, and a swept lane is read where its own indicator is 1: past the
/// window it writes zero.
#[test]
fn a_swept_lane_is_read_only_inside_the_window_it_carries() {
    let horizon = (0.0, 4.0);
    let extent = Extent::secs(RATE, horizon.0, horizon.1);
    let len = extent.len();
    let law = form(cropped(
        Body::Mul(vec![Part::bare(decaying(3.0)), Part::bare(sine(440.0))]),
        1.0,
    ));
    let (held, _) = render(&law, RATE, horizon, &PSYCHOACOUSTIC_V1).expect("a windowed decay");
    assert!(held.at(0, 100) != 0.0, "inside the window");
    assert_eq!(held.at(0, len - 1), 0.0, "outside it");
}
