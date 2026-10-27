// Concern: states the fit finds a sine's partial, a band's noise, a chord's voices, a stiff string's stretch; its DL never rising; determinism | Non-concern: the text written | IO: (a clip) -> asserts

use std::f64::consts::TAU;

use sva_core::solve::atoms::{Atom, Sound};
use sva_core::solve::fit::{Fit, Line, SPL_DB, fit};
use sva_core::solve::report::progress;
use sva_engine::Never;

const RATE: u32 = 16_000;

/// A third of a second: long enough for several 48 ms frames, short enough to fit quickly.
const LEN: usize = RATE as usize / 3;

fn now<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = std::task::Waker::noop();
    match future
        .as_mut()
        .poll(&mut std::task::Context::from_waker(waker))
    {
        std::task::Poll::Ready(out) => out,
        std::task::Poll::Pending => panic!("nothing to wait on with no signal"),
    }
}

fn fitted(samples: &[f64]) -> (Fit, Vec<Line>) {
    let mut lines = Vec::new();
    let fit = now(fit(
        samples,
        RATE,
        SPL_DB,
        &mut |line| lines.push(line.clone()),
        &Never,
        &|| 0.0,
    ));
    (fit, lines)
}

fn t(n: usize) -> f64 {
    n as f64 / f64::from(RATE)
}

/// A sum of `(hz, amplitude)` sines, each its own phase.
fn tones(parts: &[(f64, f64)]) -> Vec<f64> {
    (0..LEN)
        .map(|n| {
            parts
                .iter()
                .enumerate()
                .map(|(k, (hz, a))| a * libm::sin(TAU * hz * t(n) + k as f64))
                .sum()
        })
        .collect()
}

/// A harmonic tone at `f0`, `count` harmonics at `1/k`, harmonic `k` at `k sqrt(1 + b k^2)`.
fn voice(f0: f64, b: f64, count: usize, level: f64) -> Vec<(f64, f64)> {
    (1..=count)
        .map(|k| {
            let k = k as f64;
            (f0 * k * libm::sqrt(1.0 + b * k * k), level / k)
        })
        .collect()
}

/// Flat noise over `[lo, hi)`: a sine on every 1/T Hz inside, its phase a fixed hash.
fn band(lo: f64, hi: f64, level: f64) -> Vec<f64> {
    let step = f64::from(RATE) / LEN as f64;
    let lines: Vec<f64> = (0..)
        .map(|k| lo + step * k as f64)
        .take_while(|hz| *hz < hi)
        .collect();
    let each = level / libm::sqrt(lines.len() as f64);
    (0..LEN)
        .map(|n| {
            lines
                .iter()
                .enumerate()
                .map(|(k, hz)| {
                    let phase = (k as f64 * 0.618_033_988_75).fract() * TAU * 7.0;
                    each * libm::sin(TAU * hz * t(n) + phase)
                })
                .sum()
        })
        .collect()
}

fn kinds(atoms: &[Atom]) -> (usize, usize, usize) {
    let count = |f: fn(&Sound) -> bool| atoms.iter().filter(|a| f(&a.sound)).count();
    (
        count(|s| matches!(s, Sound::Partial { .. })),
        count(|s| matches!(s, Sound::Harmonic { .. })),
        count(|s| matches!(s, Sound::Noise { .. })),
    )
}

fn shown(fit: &Fit) -> String {
    fit.atoms
        .iter()
        .map(|a| format!("{:?}", a.sound))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn a_sine_is_one_partial() {
    let (fit, _) = fitted(&tones(&[(440.0, 0.3)]));
    assert_eq!(kinds(&fit.atoms), (1, 0, 0), "{}", shown(&fit));
    let Sound::Partial { hz } = &fit.atoms[0].sound else {
        unreachable!("a partial")
    };
    assert!((hz.base - 440.0).abs() < 1.0, "{}", hz.base);
}

#[test]
fn a_sine_beside_a_noise_band_is_one_partial_and_that_band() {
    let samples: Vec<f64> = tones(&[(440.0, 0.3)])
        .iter()
        .zip(band(2000.0, 4000.0, 0.2))
        .map(|(a, b)| a + b)
        .collect();
    let (fit, _) = fitted(&samples);
    assert_eq!(kinds(&fit.atoms), (1, 0, 1), "{}", shown(&fit));
    let noise = fit
        .atoms
        .iter()
        .find_map(|a| match a.sound {
            Sound::Noise { lo, hi } => Some((lo, hi)),
            _ => None,
        })
        .expect("a band");
    assert_eq!(noise, (2000.0, 4000.0));
}

#[test]
fn a_chord_of_two_harmonic_tones_is_two_voices() {
    let mut parts = voice(261.63, 0.0, 8, 0.2);
    parts.extend(voice(329.63, 0.0, 8, 0.2));
    let (fit, _) = fitted(&tones(&parts));
    assert_eq!(kinds(&fit.atoms), (0, 2, 0), "{}", shown(&fit));
}

#[test]
fn a_stiff_string_is_one_voice_its_stretch_within_a_tenth() {
    let b = 4e-4;
    let (fit, _) = fitted(&tones(&voice(220.0, b, 16, 0.3)));
    assert_eq!(kinds(&fit.atoms), (0, 1, 0), "{}", shown(&fit));
    let Sound::Harmonic { stretch, .. } = fit.atoms[0].sound else {
        unreachable!("a voice")
    };
    assert!((stretch - b).abs() <= 0.1 * b, "B {stretch} for {b}");
}

#[test]
fn the_dl_never_rises_from_line_to_line() {
    let samples: Vec<f64> = tones(&voice(220.0, 0.0, 6, 0.2))
        .iter()
        .zip(band(1000.0, 2000.0, 0.05))
        .map(|(a, b)| a + b)
        .collect();
    let (fit, lines) = fitted(&samples);
    assert_eq!(lines.len(), fit.steps + 1, "a line per step and the start");
    let dl: Vec<f64> = lines
        .iter()
        .map(|l| l.atom_bits + l.residual_bits)
        .collect();
    assert!(dl.windows(2).all(|w| w[1] <= w[0]), "{dl:?}");
    for (line, next) in lines.iter().zip(&lines[1..]) {
        let drop = (next.atom_bits + next.residual_bits) - (line.atom_bits + line.residual_bits);
        assert!((next.delta_bits - drop).abs() <= 1e-9 * drop.abs().max(1.0));
    }
}

#[test]
fn a_fit_is_the_same_twice() {
    let samples = tones(&voice(196.0, 0.0, 5, 0.2));
    let (a, heard_a) = fitted(&samples);
    let (b, heard_b) = fitted(&samples);
    assert_eq!(a.atoms, b.atoms);
    let text = |lines: &[Line]| lines.iter().map(progress).collect::<Vec<_>>();
    assert_eq!(text(&heard_a), text(&heard_b));
}
