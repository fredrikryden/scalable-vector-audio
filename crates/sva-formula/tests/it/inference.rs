// Concern: proves a term's type is decided structurally, once, and never by a cache | Non-concern: the transform a pair type promises | IO: (a ClosedForm, an Env) -> its Ty

use crate::fixtures::{DUAL, Fixed, causal, decay, part, sine, term, terms};
use sva_formula::{Body, Code, Var, dual, infer, normalize_closed_form};

fn ty_of(f: Body) -> sva_formula::Ty {
    infer(&term(Var::T, f), &Fixed::holding(DUAL, false)).expect("the fixture types")
}

#[test]
fn a_growing_exponential_has_no_dual_until_cropped() {
    let ringing = Body::Mul(vec![part(sine(440.0)), part(decay(-2.0))]);
    assert!(!ty_of(ringing.clone()).has_dual());
    assert!(ty_of(causal(ringing)).has_dual());
}

#[test]
fn a_join_above_eight_refuses() {
    let wide = Body::Join(
        (0..9)
            .map(|k| part(sine(100.0 * f64::from(k) + 1.0)))
            .collect(),
    );
    let refusal = infer(&term(Var::T, wide), &Fixed::holding(DUAL, false)).unwrap_err();
    assert_eq!(refusal.code, Code::WidthMismatch);
    assert!(refusal.message.contains('8'), "{}", refusal.message);

    let out_of_range = Body::Channel(part(sine(440.0)), 3);
    assert_eq!(
        infer(&term(Var::T, out_of_range), &Fixed::holding(DUAL, false))
            .unwrap_err()
            .code,
        Code::WidthMismatch
    );
}

/// Section 4's invariant, over every fixture: a term types as a pair exactly when its sum
/// form has a dual in A. The two decide through one `dual_class`, so this is what says so.
#[test]
fn infer_agrees_with_normalize_and_dual_on_every_fixture() {
    for (name, subject) in terms() {
        let typed = infer(&subject, &Fixed::holding(DUAL, false))
            .map(|ty| ty.has_dual())
            .unwrap_or(false);
        let transformed = normalize_closed_form(&subject)
            .and_then(|n| dual(&n))
            .is_ok();
        assert_eq!(typed, transformed, "{name}");
    }
}
