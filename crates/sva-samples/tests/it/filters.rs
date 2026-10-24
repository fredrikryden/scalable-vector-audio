// Concern: states what one filter call site owes a stream and its own parameter trace | Non-concern: coefficient math (src/biquad.rs) | IO: (a swept site) -> asserted dB, frames

use sva_samples::Shape;
use sva_samples::filters::FilterSite;

fn site(shape: Shape, cutoff: f64, q: f64, sr: f64) -> FilterSite {
    FilterSite::new(shape, 1, &[cutoff], &[q], &[1.0], sr)
}

fn step(s: &mut FilterSite, x: f64, cutoff: f64, q: f64, gain: f64, sr: f64, i: usize) -> f64 {
    let mut out = [0.0];
    s.process(&[x], &[cutoff], &[q], &[gain], &mut out, sr, i as i64);
    out[0]
}

fn run(shape: Shape, cutoff: f64, q: f64, input: &[f64]) -> Vec<f64> {
    let sr = 48000.0;
    let mut s = site(shape, cutoff, q, sr);
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| step(&mut s, x, cutoff, q, 1.0, sr, i))
        .collect()
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

fn tone(hz: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| libm::sin(2.0 * std::f64::consts::PI * hz * i as f64 / 48000.0))
        .collect()
}

/// The measured attenuation must match what magnitude_db predicted, or the report lies.
#[test]
fn a_rendered_tone_is_attenuated_by_what_the_response_predicts() {
    let n = 48000;
    for (shape, hz, cutoff) in [
        (Shape::Lowpass, 6000.0, 500.0),
        (Shape::Highpass, 200.0, 2000.0),
        (Shape::Bandpass, 1000.0, 1000.0),
    ] {
        let out = run(shape, cutoff, 0.707, &tone(hz, n));
        let settled = &out[n / 2..];
        let measured = 20.0 * libm::log10(rms(settled) / rms(&tone(hz, n)[n / 2..]));
        let predicted = sva_samples::biquad::magnitude_db(
            &sva_samples::biquad::design(shape, cutoff, 0.707, 1.0, 48000.0),
            hz,
            48000.0,
        );
        assert!(
            (measured - predicted).abs() < 0.5,
            "{shape:?}: measured {measured:.2} dB, predicted {predicted:.2} dB"
        );
    }
}

/// A bandpass at q 1000 runs at that q, unflagged: a tone 1% off the centre sits 26 dB
/// down, as the design says, read once 22 time constants have rung out.
#[test]
fn a_bandpass_keeps_a_q_of_a_thousand() {
    let (n, q) = (384_000, 1000.0);
    let s = site(Shape::Bandpass, 1000.0, q, 48000.0);
    assert!(!s.trace("n", 0, 48000.0)[0].clamped);
    for hz in [1000.0, 1010.0] {
        let out = run(Shape::Bandpass, 1000.0, q, &tone(hz, n));
        let measured = 20.0 * libm::log10(rms(&out[n - 48_000..]) / rms(&tone(hz, 48_000)));
        let predicted = sva_samples::biquad::magnitude_db(
            &sva_samples::biquad::design(Shape::Bandpass, 1000.0, q, 1.0, 48000.0),
            hz,
            48000.0,
        );
        assert!(
            (measured - predicted).abs() < 0.01,
            "{hz} Hz: measured {measured:.3} dB, q {q} predicts {predicted:.3} dB"
        );
    }
}

#[test]
fn a_cutoff_swept_past_nyquist_is_flagged_clamped_and_stays_finite() {
    let sr = 48000.0;
    let mut s = site(Shape::Highpass, 1000.0, 1.0, sr);
    for i in 0..100 {
        let y = step(&mut s, 0.5, 1000.0 + i as f64 * 1000.0, 1.0, 1.0, sr, i);
        assert!(y.is_finite());
    }
    assert!(s.trace("n", 0, sr)[0].clamped);
}

/// One call site, two independent delay lines and two independent equations.
#[test]
fn a_wide_cutoff_gives_one_site_a_lane_and_a_trace_per_component() {
    let sr = 48000.0;
    let cutoff = [400.0, 6000.0];
    let (q, gain) = ([0.707], [0.0]);
    let mut s = FilterSite::new(Shape::Lowpass, 2, &cutoff, &q, &gain, sr);

    let n = 24000;
    let out: Vec<[f64; 2]> = tone(3000.0, n)
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let mut lanes = [0.0; 2];
            s.process(&[x], &cutoff, &q, &gain, &mut lanes, sr, i as i64);
            lanes
        })
        .collect();
    let settled = |c: usize| rms(&out[n / 2..].iter().map(|v| v[c]).collect::<Vec<_>>());
    assert!(
        settled(0) * 4.0 < settled(1),
        "the 400 Hz lane must cut 3 kHz far harder than the 6 kHz one: {} vs {}",
        settled(0),
        settled(1)
    );

    let traces = s.trace("n", 0, sr);
    assert_eq!(traces.len(), 2);
    assert_eq!(traces[0].channel, Some(0));
    assert_eq!(traces[1].frames[0].cutoff, 6000.0);
}
