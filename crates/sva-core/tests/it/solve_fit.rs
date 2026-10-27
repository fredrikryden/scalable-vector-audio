// Concern: states what the fit owes sines, voices, noise bands, a burst; a DL falling every pass; an adjust; determinism | Non-concern: the text | IO: (a clip) -> asserts

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
        count(|s| matches!(s, Sound::Noise { .. } | Sound::Shaped { .. })),
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

/// The fundamentals each of a voice, partials aside: with no joint re-solve, partials beside the
/// voices' harmonics may stay.
#[test]
fn a_chord_of_two_harmonic_tones_holds_a_voice_at_each_fundamental() {
    let mut parts = voice(261.63, 0.0, 8, 0.2);
    parts.extend(voice(329.63, 0.0, 8, 0.2));
    let (fit, _) = fitted(&tones(&parts));
    let mut f0s: Vec<f64> = fit
        .atoms
        .iter()
        .filter_map(|a| match &a.sound {
            Sound::Harmonic { hz, .. } => Some(hz.base),
            _ => None,
        })
        .collect();
    f0s.sort_by(f64::total_cmp);
    assert_eq!(f0s.len(), 2, "{}", shown(&fit));
    assert!(
        (f0s[0] - 261.63).abs() < 1.0 && (f0s[1] - 329.63).abs() < 1.0,
        "{f0s:?}"
    );
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

/// Every pass lowers the DL, its line's change the drop from the last.
#[test]
fn the_dl_falls_at_every_pass() {
    let samples: Vec<f64> = tones(&voice(220.0, 0.0, 6, 0.2))
        .iter()
        .zip(band(1000.0, 2000.0, 0.05))
        .map(|(a, b)| a + b)
        .collect();
    let (fit, lines) = fitted(&samples);
    assert_eq!(lines.len(), fit.passes + 1, "a line per pass and the start");
    assert_eq!(fit.adds + fit.adjusts, fit.passes);
    assert!(lines.iter().enumerate().all(|(k, l)| l.pass == k));
    assert_eq!(lines[0].action, Action::Start);
    let dl: Vec<f64> = lines
        .iter()
        .map(|l| l.atom_bits + l.residual_bits)
        .collect();
    assert!(dl.windows(2).all(|w| w[1] < w[0]), "{dl:?}");
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

/// A piano's C4: harmonic k decaying at 1 + k/4 per second, 11 to 35 dB/s, the faster the higher;
/// every atom a voice at its fundamental, the first's stretch within a tenth. With no backward
/// prune a second voice there may stay.
#[test]
fn a_stiff_string_decaying_harmonic_by_harmonic_is_voiced_its_stretch_within_a_tenth() {
    let b = 4e-4;
    let parts: Vec<(f64, f64, f64)> = voice(261.63, b, 12, 0.3)
        .into_iter()
        .enumerate()
        .map(|(k, (hz, a))| (hz, a, 1.0 + 0.25 * (k + 1) as f64))
        .collect();
    let (fit, _) = fitted(&decaying(&parts));
    let (partials, voices, noises) = kinds(&fit.atoms);
    assert!(
        partials == 0 && noises == 0 && voices >= 1,
        "{}",
        shown(&fit)
    );
    assert!(
        fit.atoms
            .iter()
            .all(|a| a.hz().is_some_and(|hz| (hz.base - 261.63).abs() < 1.0)),
        "{}",
        shown(&fit)
    );
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
        let held = keys(before);
        assert!(keys(line).contains(&key), "pass {}: {key} kept", line.pass);
        assert!(!held.contains(&key), "pass {}: {key} new", line.pass);
    }
    assert!(adds >= 2, "{adds} adds");
}

/// Eight steady harmonics at 1/k.
#[test]
fn a_tone_of_eight_harmonics_is_one_voice() {
    let (fit, _) = fitted(&tones(&voice(220.0, 0.0, 8, 0.3)));
    assert_eq!(kinds(&fit.atoms), (0, 1, 0), "{}", shown(&fit));
}

/// White noise from 0.1 s to 0.25 s, silence around it.
#[test]
fn a_white_noise_burst_is_one_shaped_noise() {
    let samples: Vec<f64> = white(0.05)
        .into_iter()
        .enumerate()
        .map(|(n, v)| match (0.1..0.25).contains(&t(n)) {
            true => v,
            false => 0.0,
        })
        .collect();
    let (fit, _) = fitted(&samples);
    assert_eq!(fit.atoms.len(), 1, "{}", shown(&fit));
    let Sound::Shaped { bands } = &fit.atoms[0].sound else {
        panic!("a shaped noise: {}", shown(&fit))
    };
    assert!(bands.len() > 1, "{bands:?}");
}

/// A sine inside a noise band: the band, added first, holds some of the sine's power, and once
/// the sine is added an adjust lowers the band's level.
#[test]
fn an_adjust_lowers_the_level_of_an_atom_added_first() {
    let samples: Vec<f64> = tones(&[(1500.0, 0.3)])
        .iter()
        .zip(band(1000.0, 2000.0, 0.3))
        .map(|(a, b)| a + b)
        .collect();
    let (fit, lines) = fitted(&samples);
    let first = lines[1].atom.as_ref().expect("an add").key;
    let lowered = lines[2..].iter().any(|l| {
        l.action == Action::Adjust
            && l.atom.as_ref().is_some_and(|a| a.key == first)
            && match (l.from_db, l.to_db) {
                (Some(from), Some(to)) => to < from,
                (Some(_), None) => true,
                _ => false,
            }
    });
    assert!(lowered, "{} adjusts: {}", fit.adjusts, shown(&fit));
    assert_eq!(kinds(&fit.atoms), (1, 0, 1), "{}", shown(&fit));
}
