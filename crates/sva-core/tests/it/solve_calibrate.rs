// Concern: states what N0's calibration owes: Jesteadt's fit and residuals as measured, the frequency DL's stated place-only gap | Non-concern: the cost N0 prices | IO: (rate) -> asserts

use sva_core::solve::calibrate::{
    FREQUENCIES_HZ, SENSATION_LEVELS_DB, calibration, clip, distance, powers, reading, rendered,
    root, tone,
};
use sva_engine::default_threads;
use sva_engine::hearing::frequency_jnd_hz;
use sva_engine::internal::Ear;

const RATE: u32 = 2_500;
const SPL: f64 = 100.0;

/// 48 kHz, too slow here, measured N0 17.15497184, RMS log 0.352988, largest x2.416059.
#[test]
fn n0_reproduces_jesteadt_as_measured() {
    let fitted = calibration(RATE, 1.0, SPL, default_threads());
    let kept = FREQUENCIES_HZ.iter().filter(|hz| **hz < 1250.0).count();
    assert_eq!(fitted.conditions.len(), kept * SENSATION_LEVELS_DB.len());
    let residuals = fitted.residuals(default_threads());
    println!(
        "N0 {}, RMS log {}, largest x{}",
        fitted.n0, residuals.rms, residuals.most
    );
    assert!((fitted.n0 - 18.691_996_931_404).abs() < 1e-9 * fitted.n0);
    assert!((residuals.rms - 0.437_408_761).abs() < 1e-8);
    assert!((residuals.most - 2.782_221_956).abs() < 1e-8);
}

/// No temporal cue closes it, at 1 kHz, 40 dB SL, under the tone's rectangular gate.
#[test]
fn the_frequency_dl_is_the_stated_place_only_gap() {
    let fitted = calibration(RATE, 1.0, SPL, default_threads());
    let ear = Ear::of(f64::from(RATE), SPL, 1.0);
    let clip = clip(RATE);
    let floor = fitted
        .conditions
        .iter()
        .find(|c| c.hz == 1000.0)
        .expect("1 kHz under the ceiling")
        .floor_db;
    let heard = |hz: f64| {
        let samples = rendered(&[tone(hz, floor + 40.0)], &clip);
        reading(&ear, &clip, &powers(&ear, &clip, &samples), 1.0)
    };
    let base = heard(1000.0);
    let mut d = |df: f64| (distance(&heard(1000.0 + df), &base) / fitted.n0).sqrt();
    let wier = frequency_jnd_hz(1000.0, 40.0);
    let gap = root(&mut d, 1.0, wier) / wier;
    println!("FDL over Wier's {gap}");
    assert!((gap - 4.563_296_274).abs() < 1e-8, "{gap}");
}
