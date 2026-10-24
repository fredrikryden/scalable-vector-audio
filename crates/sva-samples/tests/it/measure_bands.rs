// Concern: states what the ERB envelope bank owes a signal passing through it | Non-concern: the discrete spectrum (src/measure/spectrum.rs) | IO: (samples) -> asserted tracks

use sva_samples::measure::bands::{BandTrack, Bands, analyze, cam, erb_hz, hz_at_cam};

const SR: f64 = 44100.0;

fn band_at(bands: &Bands, hz: f64) -> &BandTrack {
    bands
        .bands
        .iter()
        .min_by(|a, b| {
            (a.centre_hz - hz)
                .abs()
                .total_cmp(&(b.centre_hz - hz).abs())
        })
        .expect("the bank is not empty")
}

#[test]
fn the_scale_is_the_published_one_and_inverts() {
    assert!((erb_hz(1000.0) - 132.6).abs() < 0.1);
    assert!((erb_hz(100.0) - 35.5).abs() < 0.1);
    assert!((cam(1000.0) - 15.6).abs() < 0.1);
    for hz in [50.0, 100.0, 1000.0, 5000.0, 16000.0] {
        assert!((hz_at_cam(cam(hz)) - hz).abs() < 1e-6, "{hz}");
    }
}

#[test]
fn an_exponential_decay_falls_at_the_rate_it_was_built_with() {
    let tau = 0.05;
    let hz = 2000.0;
    let x: Vec<f64> = (0..(2.0 * SR) as usize)
        .map(|i| {
            let t = i as f64 / SR;
            libm::sin(2.0 * std::f64::consts::PI * hz * t) * sva_formula::exp(-t / tau)
        })
        .collect();
    let bands = analyze(&[&x[..]], SR, 0.0);
    let b = band_at(&bands, hz);
    let rate = bands.rate_hz;
    let at = |secs: f64| b.rms[(secs * rate) as usize];
    for (a, c) in [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)] {
        let measured = -(c - a) / libm::log(at(c) / at(a));
        assert!(
            (measured - tau).abs() < 0.004,
            "{a}..{c} s read tau {measured}"
        );
    }
}
