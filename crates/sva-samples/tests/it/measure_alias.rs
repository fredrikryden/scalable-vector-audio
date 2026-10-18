// Concern: states when a rate change leaves audible alias and when none | Non-concern: producing either signal | IO: (base, oversampled) -> dB

use sva_samples::measure::alias::{ALIAS_FRAME, AUDIBLE_NMR_DB, UnderAFrame, measure_alias};

fn sine(hz: f64, sr: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * hz * i as f64 / sr)
        .map(|p| p.sin())
        .collect()
}

/// One sine at two rates is one sine: nothing is left to call alias.
#[test]
fn a_signal_that_survives_the_rate_change_measures_no_alias() {
    let sr = 44100.0;
    let n = 8192;
    let k = 8;
    let m = measure_alias(
        &sine(440.0, sr, n),
        &sine(440.0, sr * k as f64, n * k),
        k,
        sr,
        0.0,
    )
    .expect("a frame");
    assert!(m.asr_db < -60.0, "asr {}", m.asr_db);
    assert!(m.nmr_db < AUDIBLE_NMR_DB, "nmr {}", m.nmr_db);
    assert!(!m.audible);
    assert!(m.scored_frames > 0);
}

/// A 6 kHz saw has one harmonic under Nyquist; the rest folded down.
#[test]
fn a_folded_harmonic_reads_as_audible_alias() {
    let sr = 44100.0;
    let n = 8192;
    let k = 8;
    let saw = |rate: f64, len: usize| -> Vec<f64> {
        (0..len)
            .map(|i| {
                let p = 6000.0 * i as f64 / rate;
                2.0 * (p - p.floor()) - 1.0
            })
            .collect()
    };
    let m = measure_alias(&saw(sr, n), &saw(sr * k as f64, n * k), k, sr, 0.0).expect("a frame");
    assert!(m.asr_db > -20.0, "asr {}", m.asr_db);
    assert!(m.audible, "nmr {}", m.nmr_db);
}

/// `peak_at_secs` sits beside a window's own `start_secs`, so it counts from the same zero.
#[test]
fn the_loudest_alias_is_named_on_the_renders_own_timeline() {
    let sr = 44100.0;
    let n = 8192;
    let k = 8;
    let saw = |rate: f64, len: usize| -> Vec<f64> {
        (0..len)
            .map(|i| {
                let p = 6000.0 * i as f64 / rate;
                2.0 * (p - p.floor()) - 1.0
            })
            .collect()
    };
    let from_zero =
        measure_alias(&saw(sr, n), &saw(sr * k as f64, n * k), k, sr, 0.0).expect("a frame");
    let from_two =
        measure_alias(&saw(sr, n), &saw(sr * k as f64, n * k), k, sr, 2.0).expect("a frame");
    assert_eq!(
        from_two.peak_at_secs,
        from_zero.peak_at_secs + 2.0,
        "a window starting at 2s names its peak two seconds later"
    );
}

/// A DC is power once, as a sine is: 0.01 of alias over a 0.5 DC is (0.01^2 / 2) / 0.5^2.
#[test]
fn a_dc_counts_its_power_once_against_the_alias() {
    let (sr, n, k) = (44_100.0, 8192, 4);
    let hz = 100.0 * sr / 1024.0;
    let base: Vec<f64> = sine(hz, sr, n).iter().map(|s| 0.5 + 0.01 * s).collect();
    let m = measure_alias(&base, &vec![0.5; n * k], k, sr, 0.0).expect("a frame");
    let known = 10.0 * (0.01f64.powi(2) / 2.0 / 0.25).log10();
    assert!(
        (m.asr_db - known).abs() < 0.01,
        "{} against {known}",
        m.asr_db
    );
}

/// So is an alias at Nyquist: 0.01 at Nyquist over a unit sine is 0.01^2 / (1 / 2).
#[test]
fn an_alias_at_nyquist_counts_its_power_once() {
    let (sr, n, k) = (44_100.0, 8192, 4);
    let hz = 100.0 * sr / 1024.0;
    let nyquist = |i: usize| if i.is_multiple_of(2) { 0.01 } else { -0.01 };
    let base: Vec<f64> = (sine(hz, sr, n).iter().enumerate())
        .map(|(i, s)| s + nyquist(i))
        .collect();
    let m = measure_alias(&base, &sine(hz, sr * k as f64, n * k), k, sr, 0.0).expect("a frame");
    let known = 10.0 * (0.01f64.powi(2) / 0.5).log10();
    assert!(
        (m.asr_db - known).abs() < 0.01,
        "{} against {known}",
        m.asr_db
    );
}

/// Frames run to the last sample: an alias in the last 300 is scored there.
#[test]
fn an_alias_in_the_tail_is_scored() {
    let (sr, n, k) = (44_100.0, ALIAS_FRAME + 300, 4);
    let hz = 100.0 * sr / 1024.0;
    let alias = sine(10_000.0, sr, n);
    let base: Vec<f64> = (sine(hz, sr, n).iter().enumerate())
        .map(|(i, s)| {
            s + if i >= ALIAS_FRAME {
                0.1 * alias[i]
            } else {
                0.0
            }
        })
        .collect();
    let m = measure_alias(&base, &sine(hz, sr * k as f64, n * k), k, sr, 0.0).expect("a frame");
    assert!(m.asr_db > -40.0, "asr {}", m.asr_db);
    assert_eq!(m.peak_at_secs, 300.0 / sr);
}

#[test]
fn a_base_under_one_frame_is_refused() {
    let (sr, k) = (44_100.0, 4);
    let n = ALIAS_FRAME - 1;
    let refused = measure_alias(
        &sine(440.0, sr, n),
        &sine(440.0, sr * 4.0, n * k),
        k,
        sr,
        0.0,
    )
    .expect_err("under a frame");
    assert_eq!(
        refused,
        UnderAFrame {
            held: n,
            frame: ALIAS_FRAME
        }
    );
}
