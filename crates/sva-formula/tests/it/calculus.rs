// Concern: proves the symbolic derivative, the Hilbert transform and the envelope stay in A | Non-concern: the transform they are built on (duality.rs) | IO: (a SpectralSum) -> another one

use crate::fixtures::{cosine, line, part, sine};
use sva_formula::{Body, LeftReason, SpectralSum, Var, d_dt, hilbert, normalize};

fn law(f: &Body) -> SpectralSum {
    normalize(f, Var::T).expect("the fixture stays in A")
}

/// An order or degree past `u16` refuses rather than wrapping to a lower one.
#[test]
fn an_order_or_degree_past_its_width_refuses() {
    let pole = law(&Body::Pow(part(line()), -i32::from(u16::MAX)));
    let left = d_dt(&pole).expect_err("the pole order passes u16");
    assert_eq!(left.reason, LeftReason::PoleOrder(u16::MAX));

    let highest = Body::Pow(part(line()), i32::from(u16::MAX));
    assert!(
        normalize(&highest, Var::T).is_ok(),
        "degree u16::MAX itself is held"
    );
    let raised = Body::Mul(vec![
        part(Body::Pow(part(line()), i32::from(u16::MAX))),
        part(line()),
    ]);
    let left = normalize(&raised, Var::T).expect_err("the degree passes u16");
    assert_eq!(left.reason, LeftReason::Overflow);

    let delta = law(&Body::Delta {
        at: part(line()),
        order: u16::MAX,
    });
    let left = d_dt(&delta).expect_err("the delta's order passes u16");
    assert_eq!(left.reason, LeftReason::Overflow);
}

#[test]
fn hilbert_of_a_cosine_is_its_sine() {
    let quadrature = hilbert(&law(&cosine(440.0))).expect("a line pair has a Hilbert transform");
    assert_eq!(quadrature, law(&sine(440.0)));
}
