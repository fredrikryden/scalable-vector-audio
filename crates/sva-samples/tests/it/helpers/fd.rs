// Concern: drives one finite-difference solver into a buffer and reads its spectrum back | Non-concern: what any one model must hold (the physics_* files) | IO: (Params, rate, secs) -> Buffer, peaks

use sva_samples::measure::spectrum;
use sva_samples::{Buffer, Params, site};

pub fn render(params: &Params, rate: u32, secs: f64) -> Buffer {
    assert!(
        params.valid(),
        "{} got parameters it refuses",
        params.name()
    );
    let len = (secs * f64::from(rate)).round() as usize;
    let mut solver = site(params, f64::from(rate)).expect("a grid this rate can hold");
    let samples = (0..len)
        .map(|_| solver.step(&[]))
        .collect::<Result<Vec<f64>, _>>();
    Buffer::mono(rate, samples.expect("a solve that settles"))
}

pub fn peaks(buffer: &Buffer, max_peaks: usize, frame_secs: Option<f64>) -> Vec<f64> {
    let s = spectrum::analyze(
        &[buffer.plane(0)],
        f64::from(buffer.rate),
        max_peaks,
        frame_secs,
    );
    s.peaks.iter().map(|p| p.hz).collect()
}

pub fn nearest(peaks: &[f64], hz: f64) -> f64 {
    peaks
        .iter()
        .copied()
        .min_by(|a, b| (a - hz).abs().total_cmp(&(b - hz).abs()))
        .expect("a spectrum with peaks")
}

pub fn sounds(buffer: &Buffer) {
    assert!(
        buffer.plane(0).iter().all(|s| s.is_finite()),
        "a diverged grid tests nothing"
    );
    assert!(
        buffer.plane(0).iter().any(|&s| s != 0.0),
        "silence tests nothing"
    );
}

pub fn is_deterministic(params: &Params, rate: u32, secs: f64) {
    let once = render(params, rate, secs);
    sounds(&once);
    assert_eq!(once, render(params, rate, secs));
}

pub fn refuses(base: &Params, broken: Vec<Params>) {
    assert!(
        base.valid(),
        "{} refuses its own reference set",
        base.name()
    );
    for p in broken {
        assert!(!p.valid(), "{} admitted {p:?}", p.name());
    }
}
