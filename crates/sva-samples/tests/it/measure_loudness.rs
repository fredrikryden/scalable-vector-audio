// Concern: states the BS.1770 readings a K-weighted measure owes its own conformance tones | Non-concern: per-component level (src/measure/envelope.rs) | IO: (planes) -> asserted LUFS

use sva_samples::measure::loudness::{Loudness, MOMENTARY_SECS, analyze};

fn sine(hz: f64, sr: f64, secs: f64, amp: f64) -> Vec<f64> {
    (0..(secs * sr) as usize)
        .map(|i| amp * libm::sin(2.0 * std::f64::consts::PI * hz * i as f64 / sr))
        .collect()
}

fn of(planes: &[&[f64]], sr: f64) -> Loudness {
    analyze(planes, sr, 0.0, MOMENTARY_SECS).expect("a rate the K-weighting is designed at")
}

/// BS.1770-2 conformance, which pins offset, shelf and highpass at once.
#[test]
fn a_minus_twenty_dbfs_kilohertz_sine_reads_minus_twenty_three_lkfs() {
    let s = sine(1000.0, 48000.0, 5.0, 0.1);
    let l = of(&[&s], 48000.0);
    let integrated = l.integrated_lufs.expect("five seconds is many blocks");
    assert!(
        (integrated - -23.0).abs() < 0.1,
        "read {integrated} LKFS, not -23.0"
    );
    assert!((l.momentary_max_lufs.unwrap() - -23.0).abs() < 0.1);
    assert!((l.short_term_max_lufs.unwrap() - -23.0).abs() < 0.1);
}

#[test]
fn silence_is_gated_out_rather_than_averaged_in() {
    let sr = 48000.0;
    let mut loud = sine(1000.0, sr, 5.0, 0.1);
    let quiet = vec![0.0; (5.0 * sr) as usize];
    let alone = of(&[&loud], sr).integrated_lufs.unwrap();
    loud.extend_from_slice(&quiet);
    let padded = of(&[&loud], sr).integrated_lufs.unwrap();
    assert!(
        (padded - alone).abs() < 0.2,
        "{alone} -> {padded}: only the three blocks straddling the edge are half-full"
    );
}

/// A level whose weighted square passes the largest double, or falls under the smallest, is
/// still a level, up to a peak near the largest double itself.
#[test]
fn a_finite_buffer_too_loud_or_too_quiet_to_square_still_reads_its_loudness() {
    let sr = 48_000.0;
    let unit = sine(1_000.0, sr, 4.0, 1.0);
    let at_scale = of(&[&unit], sr).momentary_max_lufs.expect("a level");
    for gain in [1e-200, 1e160, 1.5e308] {
        let scaled: Vec<f64> = unit.iter().map(|v| v * gain).collect();
        let level = of(&[&scaled], sr)
            .momentary_max_lufs
            .expect("a level past the largest square or under the smallest");
        let lift = 20.0 * libm::log10(gain);
        assert!((level - at_scale - lift).abs() < 1e-6, "{gain}: {level}");
    }
}

/// A block sums its own samples: ten seconds at full scale before it round no quiet block off.
#[test]
fn a_quiet_block_after_a_loud_stretch_reads_its_own_level() {
    let sr = 48_000.0;
    let quiet_amp = 1e-7;
    let alone = of(&[&sine(1_000.0, sr, 2.0, quiet_amp)], sr);
    let steady = alone
        .momentary_max_lufs
        .expect("a level under the absolute gate");
    let mut after = sine(1_000.0, sr, 10.0, 1.0);
    after.extend(sine(1_000.0, sr, 2.0, quiet_amp));
    let read = of(&[&after], sr);
    let quiet: Vec<f64> = (read.momentary.iter())
        .filter(|f| f.t >= 10.5 && f.t + 0.4 <= 12.0)
        .map(|f| f.lufs)
        .collect();
    assert!(!quiet.is_empty());
    for lufs in quiet {
        assert!(
            (lufs - steady).abs() < 0.01,
            "{lufs} LUFS after a loud stretch, {steady} alone"
        );
    }
}
