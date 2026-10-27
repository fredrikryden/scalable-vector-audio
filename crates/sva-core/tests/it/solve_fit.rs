// Concern: states the fit finds a sine's partial, in noise too, a band's noise, voices steady or decaying, a stretch; a DL never rising; determinism | Non-concern: the text | IO: (a clip) -> asserts

use std::f64::consts::TAU;

use sva_core::solve::atoms::{Atom, Sound};
use sva_core::solve::fit::{Action, Fit, Line, SPL_DB, fit};
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

/// A phase in `[0, 2 pi)` hashed from `k` (splitmix64's finaliser), so the lines' phases hold
/// no order: a phase linear in `k` would sum them to a click.
fn hashed(k: u64) -> f64 {
    let mut z = k.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    TAU * (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Flat noise over `[lo, hi)`: a sine on every 1/T Hz inside, its phase hashed.
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
                .map(|(k, hz)| each * libm::sin(TAU * hz * t(n) + hashed(k as u64)))
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
    let lines: Vec<Line> = lines.into_iter().filter(|l| l.pass == 1).collect();
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
fn the_second_pass_reports_its_steps_its_dl_never_rising_within_a_pass() {
    let samples: Vec<f64> = tones(&voice(220.0, 0.0, 6, 0.2))
        .iter()
        .zip(band(1000.0, 2000.0, 0.05))
        .map(|(a, b)| a + b)
        .collect();
    let (fit, lines) = fitted(&samples);
    for (pass, steps) in [(1, fit.steps), (2, fit.second.steps)] {
        let held: Vec<&Line> = lines.iter().filter(|l| l.pass == pass).collect();
        assert_eq!(
            held.len(),
            steps + 1,
            "pass {pass}: a line per step and the start"
        );
        assert_eq!(held[0].action, Action::Start, "pass {pass}");
        assert!(
            held.iter().enumerate().all(|(k, l)| l.step == k),
            "pass {pass}"
        );
        let dl: Vec<f64> = held.iter().map(|l| l.atom_bits + l.residual_bits).collect();
        assert!(dl.windows(2).all(|w| w[1] <= w[0]), "pass {pass}: {dl:?}");
    }
    assert!(fit.second.steps > 0, "the second pass takes a step");
    let second = lines
        .iter()
        .position(|l| l.pass == 2)
        .expect("a second pass");
    assert!(
        lines[second..].iter().all(|l| l.pass == 2),
        "the passes in turn"
    );
    let start = &lines[second];
    assert!(
        (start.atom_bits + start.residual_bits - fit.second.start_bits).abs()
            <= 1e-9 * fit.second.start_bits.abs()
    );
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

/// A sum of `(hz, amplitude, decay per second)` sines, each its own phase.
fn decaying(parts: &[(f64, f64, f64)]) -> Vec<f64> {
    (0..LEN)
        .map(|n| {
            parts
                .iter()
                .enumerate()
                .map(|(k, (hz, a, d))| {
                    a * sva_formula::exp(-d * t(n)) * libm::sin(TAU * hz * t(n) + k as f64)
                })
                .sum()
        })
        .collect()
}

/// White noise of variance `variance`, each sample a Box-Muller draw of two hashed uniforms.
fn white(variance: f64) -> Vec<f64> {
    let uniform = |k: u64| (hashed(k) / TAU).max(f64::MIN_POSITIVE);
    (0..LEN as u64)
        .map(|n| {
            let (u, v) = (uniform(2 * n), uniform(2 * n + 1));
            libm::sqrt(variance) * libm::sqrt(-2.0 * libm::log(u)) * libm::cos(TAU * v)
        })
        .collect()
}

/// White noise 30 dB under the sine's power.
#[test]
fn a_sine_in_white_noise_thirty_db_under_is_one_partial() {
    let (a, hz) = (0.3, 440.0);
    let noise = white(a * a / 2.0 * 1e-3);
    let samples: Vec<f64> = tones(&[(hz, a)])
        .iter()
        .zip(noise)
        .map(|(s, n)| s + n)
        .collect();
    let (fit, _) = fitted(&samples);
    let (partials, voices, _) = kinds(&fit.atoms);
    assert_eq!((partials, voices), (1, 0), "{}", shown(&fit));
}

/// Eight harmonics at 1/k, each decaying at 4 per second (35 dB/s).
#[test]
fn a_decaying_tone_of_eight_harmonics_is_one_voice() {
    let parts: Vec<(f64, f64, f64)> = voice(220.0, 0.0, 8, 0.3)
        .into_iter()
        .map(|(hz, a)| (hz, a, 4.0))
        .collect();
    let (fit, _) = fitted(&decaying(&parts));
    assert_eq!(kinds(&fit.atoms), (0, 1, 0), "{}", shown(&fit));
}

/// A piano's C4: harmonic k decaying at 1 + k/4 per second, 11 to 35 dB/s, the faster the higher.
#[test]
fn a_stiff_string_decaying_harmonic_by_harmonic_is_one_voice_its_stretch_within_a_tenth() {
    let b = 4e-4;
    let parts: Vec<(f64, f64, f64)> = voice(261.63, b, 12, 0.3)
        .into_iter()
        .enumerate()
        .map(|(k, (hz, a))| (hz, a, 1.0 + 0.25 * (k + 1) as f64))
        .collect();
    let (fit, _) = fitted(&decaying(&parts));
    assert_eq!(kinds(&fit.atoms), (0, 1, 0), "{}", shown(&fit));
    let Sound::Harmonic { stretch, .. } = fit.atoms[0].sound else {
        unreachable!("a voice")
    };
    assert!((stretch - b).abs() <= 0.1 * b, "B {stretch} for {b}");
}

/// An add line names the atom that step added: its key kept after the step and not before.
#[test]
fn an_add_line_names_the_atom_it_added() {
    let mut parts = voice(261.63, 0.0, 8, 0.2);
    parts.extend(voice(329.63, 0.0, 8, 0.2));
    let samples: Vec<f64> = tones(&parts)
        .iter()
        .zip(band(2000.0, 4000.0, 0.05))
        .map(|(a, b)| a + b)
        .collect();
    let (_, lines) = fitted(&samples);
    let keys = |l: &Line| l.envelopes.iter().map(|(k, _)| *k).collect::<Vec<u64>>();
    let mut adds = 0;
    for (before, line) in lines.iter().zip(&lines[1..]) {
        if line.action != Action::Add {
            continue;
        }
        adds += 1;
        let key = line.atom.as_ref().expect("an atom").key;
        let held = match before.pass == line.pass {
            true => keys(before),
            false => Vec::new(),
        };
        assert!(keys(line).contains(&key), "step {}: {key} kept", line.step);
        assert!(!held.contains(&key), "step {}: {key} new", line.step);
    }
    assert!(adds >= 2, "{adds} adds");
}
