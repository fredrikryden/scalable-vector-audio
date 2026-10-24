// Concern: states what a level trace owes a signal of known shape | Non-concern: per-band level (measure_bands.rs), loudness weighting (measure_loudness.rs) | IO: (samples) -> asserted frames

use sva_samples::measure::envelope::trace;

const RATE: f64 = 44_100.0;

fn tone(hz: f64, secs: f64, amp: f64) -> Vec<f64> {
    let len = (secs * RATE) as usize;
    (0..len)
        .map(|i| amp * libm::sin(std::f64::consts::TAU * hz * i as f64 / RATE))
        .collect()
}

/// A steady tone has one level, and every frame says so at the second it covers.
#[test]
fn a_steady_tone_traces_one_level_at_its_own_seconds() {
    let frames = trace(&[&tone(440.0, 0.5, 0.5)], RATE, 0.0, 0.05);
    assert_eq!(frames.len(), 10, "half a second in 50 ms steps");
    for (n, frame) in frames.iter().enumerate() {
        assert!(
            (frame.t_secs - n as f64 * 0.05).abs() < 1e-9,
            "frame {n} at {}",
            frame.t_secs
        );
        assert!(
            (frame.rms - 0.5 / 2f64.sqrt()).abs() < 0.01,
            "a sine's RMS is its amplitude over root two: {frame:?}"
        );
        assert!((frame.peak - 0.5).abs() < 0.01, "{frame:?}");
    }
}
