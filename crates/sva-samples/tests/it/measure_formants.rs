// Concern: states what the all-pole model owes resonators whose poles are known | Non-concern: naming a peak (src/measure/pitch.rs) | IO: (samples) -> asserted resonances

use crate::helpers::drawn;
use sva_samples::measure::formants::{Formant, FormantFrame, analyze};

/// A signal whose envelope is known by construction rather than by another measurement.
/// Driven by noise and not an impulse: an envelope models a stationary frame.
fn resonated(x: &[f64], hz: f64, bandwidth_hz: f64, sample_rate: f64) -> Vec<f64> {
    let r = sva_formula::exp(-std::f64::consts::PI * bandwidth_hz / sample_rate);
    let theta = 2.0 * std::f64::consts::PI * hz / sample_rate;
    let (b1, b2) = (2.0 * r * libm::cos(theta), -r * r);
    let (mut y1, mut y2) = (0.0f64, 0.0f64);
    x.iter()
        .map(|&sample| {
            let y = sample + b1 * y1 + b2 * y2;
            y2 = y1;
            y1 = y;
            y
        })
        .collect()
}

fn noise(seed: f64, samples: usize) -> Vec<f64> {
    (0..samples).map(|n| drawn(seed, n as i64)).collect()
}

fn resonator(hz: f64, bandwidth_hz: f64, sample_rate: f64, samples: usize) -> Vec<f64> {
    resonated(&noise(1.0, samples), hz, bandwidth_hz, sample_rate)
}

fn nearest(frame: &FormantFrame, hz: f64) -> Formant {
    *frame
        .formants
        .iter()
        .min_by(|a, b| (a.hz - hz).abs().total_cmp(&(b.hz - hz).abs()))
        .expect("a resonator has a resonance")
}

#[test]
fn a_two_pole_resonator_reports_the_centre_and_bandwidth_it_was_built_with() {
    let sr = 16000.0;
    for (hz, bandwidth) in [(700.0, 90.0), (1220.0, 110.0), (2600.0, 170.0)] {
        let x = resonator(hz, bandwidth, sr, 32768);
        let frame = analyze(&[&x[..]], sr, 2, 4);
        assert_eq!(frame.formants.len(), 1, "one pole pair, one formant");
        let found = frame.formants[0];
        assert!(
            (found.hz - hz).abs() < 5.0,
            "asked {hz} Hz, measured {} Hz",
            found.hz
        );
        assert!(
            (found.bandwidth_hz - bandwidth).abs() < 12.0,
            "asked {bandwidth} Hz wide, measured {} Hz",
            found.bandwidth_hz
        );
    }
}

/// The whole reason this exists: a voiced signal's peaks are the glottal source's harmonics,
/// and the formants sit under them without moving with f0.
#[test]
fn a_voiced_source_through_a_tract_reports_the_tract_and_not_its_harmonics() {
    let sr = 16000.0;
    let targets = [(730.0, 90.0), (1090.0, 110.0), (2440.0, 170.0)];
    for f0 in [98.0, 220.0] {
        let mut x = glottis(f0, sr, 32768);
        for (hz, bandwidth) in targets {
            x = resonated(&x, hz, bandwidth, sr);
        }
        let frame = analyze(&[&x[..]], sr, 14, 8);
        for (hz, _) in targets {
            let found = nearest(&frame, hz);
            assert!(
                (found.hz - hz).abs() < 60.0,
                "f0 {f0}: asked {hz} Hz, nearest measured {} Hz of {:?}",
                found.hz,
                frame.formants
            );
        }
    }
}

fn glottis(f0: f64, sample_rate: f64, samples: usize) -> Vec<f64> {
    (0..samples)
        .map(|n| {
            let x = (f0 * n as f64 / sample_rate).rem_euclid(1.0) / 0.6;
            (x * x - x * x * x).max(0.0)
        })
        .collect()
}
