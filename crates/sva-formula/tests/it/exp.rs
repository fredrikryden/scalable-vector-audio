// Concern: holds the exponential every crate calls to libm's on every table slot, the subnormal and overflow edges and the specials | Non-concern: where it underflows (underflow.rs) | IO: (x) -> e^x

#![allow(clippy::disallowed_methods)]

use sva_formula::exp;

/// Both within an ulp of `e^x`, the two sit at most one representable step apart.
fn near_libm(x: f64) {
    let (ours, theirs) = (exp(x), libm::exp(x));
    let apart = (ours.to_bits() as i64 - theirs.to_bits() as i64).abs();
    assert!(
        apart <= 1,
        "exp({x:e}) = {ours:e}, libm {theirs:e}: {apart} steps apart"
    );
}

#[test]
fn every_table_slot_lands_within_a_step_of_libm() {
    let step = std::f64::consts::LN_2 / 128.0;
    for k in -128 * 40..128 * 40 {
        for f in [-0.5, -0.31, -0.07, 0.0, 0.13, 0.29, 0.4999] {
            near_libm((k as f64 + f) * step);
        }
    }
}

#[test]
fn the_far_ranges_land_within_a_step_of_libm() {
    let spread = |lo: f64, hi: f64| (0..=20_000).map(move |i| lo + (hi - lo) * i as f64 / 20_000.0);
    spread(-745.13, -708.0).for_each(near_libm);
    spread(-708.0, -500.0).for_each(near_libm);
    spread(500.0, 709.78).for_each(near_libm);
    spread(-1e-6, 1e-6).for_each(near_libm);
}

#[test]
fn the_specials_answer_what_ieee_does() {
    assert_eq!(exp(0.0), 1.0);
    assert_eq!(exp(-0.0), 1.0);
    assert_eq!(exp(f64::NEG_INFINITY).to_bits(), 0.0f64.to_bits());
    assert_eq!(exp(f64::INFINITY), f64::INFINITY);
    assert!(exp(f64::NAN).is_nan());
    assert_eq!(exp(710.0), f64::INFINITY);
    assert_eq!(exp(-746.0).to_bits(), 0.0f64.to_bits());
    assert_eq!(exp(-745.0), 5e-324);
    assert_eq!(exp(1e-300), 1.0);
}
