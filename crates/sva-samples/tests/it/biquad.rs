// Concern: states what the bilinear biquad designs owe the shapes they are named for | Non-concern: H(s) on a closed form (sva-formula) | IO: (Shape, cutoff, q, gain) -> asserted dB

use sva_samples::Shape;
use sva_samples::biquad::{clamp_cutoff, clamp_q, design, magnitude_db};

const SR: f64 = 48000.0;

fn db_at(shape: Shape, cutoff: f64, q: f64, hz: f64) -> f64 {
    magnitude_db(&design(shape, cutoff, q, 1.0, SR), hz, SR)
}

#[test]
fn every_shape_puts_its_energy_where_its_name_says() {
    assert!(db_at(Shape::Lowpass, 1000.0, 0.707, 100.0) > -0.5);
    assert!(db_at(Shape::Lowpass, 1000.0, 0.707, 8000.0) < -30.0);

    assert!(db_at(Shape::Highpass, 1000.0, 0.707, 8000.0) > -0.5);
    assert!(db_at(Shape::Highpass, 1000.0, 0.707, 100.0) < -30.0);

    assert!(db_at(Shape::Bandpass, 1000.0, 2.0, 1000.0).abs() < 0.1);
    assert!(db_at(Shape::Bandpass, 1000.0, 2.0, 100.0) < -20.0);
    assert!(db_at(Shape::Bandpass, 1000.0, 2.0, 10000.0) < -20.0);

    assert!(db_at(Shape::Notch, 1000.0, 2.0, 1000.0) < -60.0);
    assert!(db_at(Shape::Notch, 1000.0, 2.0, 100.0).abs() < 0.5);
}

/// The amplitude ratio `db` writes.
fn ratio(db: f64) -> f64 {
    libm::pow(10f64, db / 20.0)
}

#[test]
fn a_shelf_and_a_bell_reach_their_stated_gain() {
    let peak = magnitude_db(
        &design(Shape::Peaking, 1000.0, 1.0, ratio(9.0), SR),
        1000.0,
        SR,
    );
    assert!((peak - 9.0).abs() < 0.1, "peaking gain {peak}");

    let low = magnitude_db(
        &design(Shape::Lowshelf, 1000.0, 0.707, ratio(-12.0), SR),
        20.0,
        SR,
    );
    assert!((low + 12.0).abs() < 0.3, "lowshelf gain {low}");

    let high = magnitude_db(
        &design(Shape::Highshelf, 1000.0, 0.707, ratio(6.0), SR),
        20000.0,
        SR,
    );
    assert!((high - 6.0).abs() < 0.3, "highshelf gain {high}");
}

#[test]
fn a_cutoff_past_nyquist_or_at_zero_clamps_rather_than_producing_nonsense() {
    let (hz, clamped) = clamp_cutoff(30000.0, SR);
    assert!(clamped && hz < SR / 2.0);
    assert_eq!(clamp_cutoff(0.0, SR), (1.0, true));
    assert_eq!(clamp_cutoff(1000.0, SR), (1000.0, false));
    for shape in [Shape::Lowpass, Shape::Highpass, Shape::Bandpass] {
        let c = design(shape, hz, clamp_q(shape, hz, 0.0, 1.0, SR).0, 1.0, SR);
        assert!(c.b0.is_finite() && c.a1.is_finite() && c.a2.is_finite());
    }
}
