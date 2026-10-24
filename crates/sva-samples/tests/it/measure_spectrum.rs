// Concern: states what the discrete spectrum owes tones of known content | Non-concern: the FFT itself (src/fft.rs), the ERB bank (measure_bands.rs) | IO: (samples) -> asserted Hz and dB

use sva_samples::measure::spectrum::{analyze, db};

const RATE: f64 = 44_100.0;

fn tone(hz: f64, secs: f64, amp: f64) -> Vec<f64> {
    let len = (secs * RATE) as usize;
    (0..len)
        .map(|i| amp * libm::sin(std::f64::consts::TAU * hz * i as f64 / RATE))
        .collect()
}

/// Two partials an octave apart, the upper half the amplitude.
#[test]
fn two_partials_are_both_found_and_ordered_by_level() {
    let low = tone(500.0, 1.0, 0.5);
    let high = tone(1000.0, 1.0, 0.25);
    let mixed: Vec<f64> = low.iter().zip(&high).map(|(a, b)| a + b).collect();
    let s = analyze(&[&mixed[..]], RATE, 4, None);

    let [first, second, ..] = s.peaks.as_slice() else {
        panic!("two partials, two peaks: {:?}", s.peaks);
    };
    assert!((first.hz - 500.0).abs() < s.resolution_hz, "{first:?}");
    assert!((second.hz - 1000.0).abs() < s.resolution_hz, "{second:?}");
    assert!(first.db > second.db, "the louder partial is named first");
    assert!(
        (first.db - second.db - 6.02).abs() < 0.5,
        "half the amplitude is 6 dB down: {first:?} {second:?}"
    );
    assert!(s.centroid_hz > 500.0 && s.centroid_hz < 1000.0);
}

/// A short sine reads at its level and place, its sidelobes no peaks.
#[test]
fn a_short_sine_reads_at_its_level_with_no_sidelobe_peaks() {
    let s = analyze(&[&tone(261.6256, 0.05, 0.5)[..]], RATE, 8, None);
    let [only] = s.peaks.as_slice() else {
        panic!("one sine, one peak: {:?}", s.peaks);
    };
    assert!((only.hz - 261.6256).abs() < 0.5, "{only:?}");
    assert!((only.db - db(0.5)).abs() < 0.5, "{only:?}");
}

/// A band reads its sine's amplitude however its frame pads; 900 Hz's lobe is inside its band.
#[test]
fn a_band_reads_a_sine_at_its_amplitude_at_any_length() {
    for len in [2205, 4096, 4097, 6000, 44_100] {
        let samples = tone(900.0, len as f64 / RATE, 0.5);
        let s = analyze(&[&samples[..]], RATE, 4, None);
        let holding = s
            .bands
            .iter()
            .find(|b| b.lo_hz <= 900.0 && 900.0 < b.hi_hz)
            .expect("a band holds 900 Hz");
        assert!(
            (holding.db - db(0.5)).abs() < 0.1,
            "{len} samples: {holding:?}"
        );
    }
}
