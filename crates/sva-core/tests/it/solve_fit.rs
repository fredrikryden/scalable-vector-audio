// Concern: states what the fit owes sines, glides, voices, noise bands, a burst; a DL falling every pass; adjusts; determinism; its threshold; a resume | Non-concern: the text | IO: (a clip) -> asserts

use std::f64::consts::TAU;
use std::num::NonZeroUsize;

use sva_core::CliError;
use sva_core::solve::atoms::{Atom, Sound, files_at};
use sva_core::solve::fit::{Action, Fit, Line, Options, Resume, SPL_DB, Tweak, fit};
use sva_core::solve::optimize::{End, TOLERANCE};
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

/// `samples` fitted under `options`, each line it hears.
fn fitted_with(samples: &[f64], options: &Options) -> (Fit, Vec<Line>) {
    let mut lines = Vec::new();
    let fit = now(fit(
        samples,
        RATE,
        SPL_DB,
        options,
        &mut |line| lines.push(line.clone()),
        &Never,
        &|| 0.0,
    ))
    .unwrap_or_else(|e| panic!("the clip fits: {e:?}"));
    (fit, lines)
}

fn fitted(samples: &[f64]) -> (Fit, Vec<Line>) {
    fitted_with(
        samples,
        &Options {
            tolerance: 0.0,
            ..Options::default()
        },
    )
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
    let Sound::Partial { hz, .. } = &fit.atoms[0].sound else {
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

/// The main lobe's half-width.
const LOBE_HZ: f64 = 2.0 / 0.064;

/// Each tonal line's Hz, a voice's every harmonic.
fn lines_of(atom: &Atom) -> Vec<f64> {
    match &atom.sound {
        Sound::Partial { hz, .. } => vec![hz.base],
        Sound::Harmonic { hz, .. } => (1..=atom.harmonics())
            .map(|k| hz.base * atom.ratio(k))
            .collect(),
        Sound::Noise { .. } | Sound::Shaped { .. } => Vec::new(),
    }
}

/// Whether `atom`'s lines lie a lobe from every line of `others`.
fn apart(atom: &Atom, others: &[Atom]) -> bool {
    others.iter().filter(|o| o.key != atom.key).all(|o| {
        lines_of(atom)
            .iter()
            .all(|x| lines_of(o).iter().all(|y| (x - y).abs() >= LOBE_HZ))
    })
}

/// C4 and E4: E4's fourth harmonic lies in C4's fifth's main lobe, so the voices overlap and
/// one is voiced, no two atoms' lines within a main lobe.
#[test]
fn a_chord_of_two_harmonic_tones_holds_one_voice_its_lines_apart() {
    let mut parts = voice(261.63, 0.0, 8, 0.2);
    parts.extend(voice(329.63, 0.0, 8, 0.2));
    let (fit, _) = fitted(&tones(&parts));
    let f0s: Vec<f64> = fit
        .atoms
        .iter()
        .filter_map(|a| match &a.sound {
            Sound::Harmonic { hz, .. } => Some(hz.base),
            _ => None,
        })
        .collect();
    assert_eq!(f0s.len(), 1, "{}", shown(&fit));
    assert!(
        [261.63, 329.63].iter().any(|f| (f0s[0] - f).abs() < 1.0),
        "{f0s:?}"
    );
    assert!(
        fit.atoms.iter().all(|a| apart(a, &fit.atoms)),
        "{}",
        shown(&fit)
    );
}

/// A partial 10 Hz over a voice's third harmonic: no add puts a line there; the voice adjusts.
#[test]
fn a_partial_at_a_voices_harmonic_is_an_adjust_not_an_add() {
    let mut parts = voice(200.0, 0.0, 8, 0.2);
    parts.push((610.0, 0.1));
    let (fit, lines) = fitted(&tones(&parts));
    let held = fit
        .atoms
        .iter()
        .find(|a| matches!(a.sound, Sound::Harmonic { .. }))
        .unwrap_or_else(|| panic!("a voice: {}", shown(&fit)));
    for (before, line) in lines.iter().zip(&lines[1..]) {
        if line.action == Action::Add {
            let added = line.atom.as_ref().expect("an atom");
            assert!(apart(added, &before.atoms), "pass {}: {added:?}", line.pass);
        }
    }
    assert!(
        lines
            .iter()
            .any(|l| l.action == Action::Adjust
                && l.atom.as_ref().is_some_and(|a| a.key == held.key)),
        "{}",
        shown(&fit)
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

/// A fit on one thread and on several finds the same atoms at the same DL, line by line.
#[test]
fn a_fit_is_the_same_on_one_thread_and_on_several() {
    let mut parts = voice(196.0, 0.0, 5, 0.2);
    parts.push((1500.0, 0.1));
    let samples = tones(&parts);
    let on = |threads: usize| {
        let threads = NonZeroUsize::new(threads).expect("a thread");
        let options = Options {
            tolerance: 0.0,
            threads,
            ..Options::default()
        };
        fitted_with(&samples, &options)
    };
    let (one, heard_one) = on(1);
    let (several, heard_several) = on(4);
    assert_eq!(one.atoms, several.atoms);
    assert_eq!(one.atom_bits.to_bits(), several.atom_bits.to_bits());
    assert_eq!(one.residual_bits.to_bits(), several.residual_bits.to_bits());
    let text = |lines: &[Line]| lines.iter().map(progress).collect::<Vec<_>>();
    assert_eq!(text(&heard_one), text(&heard_several));
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

/// Eight harmonics at 1/k, each decaying at 4 per second (35 dB/s): a voice at the fundamental,
/// first. With no backward prune, atoms beside it a level adjust leaves room for may stay.
#[test]
fn a_decaying_tone_of_eight_harmonics_is_voiced_first() {
    let parts: Vec<(f64, f64, f64)> = voice(220.0, 0.0, 8, 0.3)
        .into_iter()
        .map(|(hz, a)| (hz, a, 4.0))
        .collect();
    let (fit, _) = fitted(&decaying(&parts));
    let (_, voices, noises) = kinds(&fit.atoms);
    assert!(voices >= 1 && noises == 0, "{}", shown(&fit));
    let Sound::Harmonic { hz, .. } = &fit.atoms[0].sound else {
        panic!("a voice first: {}", shown(&fit))
    };
    assert!((hz.base - 220.0).abs() < 1.0, "{}", hz.base);
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

/// Each line's DL, the start's first.
fn dls(lines: &[Line]) -> Vec<f64> {
    lines
        .iter()
        .map(|l| l.atom_bits + l.residual_bits)
        .collect()
}

#[test]
fn a_fixed_sine_glides_not_at_all() {
    let (fit, _) = fitted(&tones(&[(440.0, 0.3)]));
    assert_eq!(kinds(&fit.atoms), (1, 0, 0), "{}", shown(&fit));
    assert_eq!(fit.atoms[0].glide(), 0.0, "{}", shown(&fit));
}

/// A plop: a sine rising from 400 to 800 Hz over 0.2 s, exponentially, decaying at 10 per
/// second.
fn plop() -> Vec<f64> {
    let rate = libm::log(2.0) / 0.2;
    (0..LEN)
        .map(|n| {
            let phase = TAU * 400.0 * (sva_formula::exp(rate * t(n)) - 1.0) / rate;
            0.3 * sva_formula::exp(-10.0 * t(n)) * libm::sin(phase)
        })
        .collect()
}

/// Partials only, the first gliding; fixed ones may take up its template's misfit.
#[test]
fn a_plop_is_a_gliding_partial_first_its_glide_within_a_tenth() {
    let rate = libm::log(2.0) / 0.2;
    let (fit, lines) = fitted(&plop());
    let (partials, voices, noises) = kinds(&fit.atoms);
    assert!(partials >= 1 && voices + noises == 0, "{}", shown(&fit));
    let glide = fit.atoms[0].glide();
    assert!(
        (glide - rate).abs() <= 0.1 * rate,
        "glide {glide} for {rate}"
    );
    let dl = dls(&lines);
    assert!(dl.windows(2).all(|w| w[1] < w[0]), "{dl:?}");
}

#[test]
fn a_gliding_fit_is_the_same_twice() {
    let (a, heard_a) = fitted(&plop());
    let (b, heard_b) = fitted(&plop());
    assert_eq!(a.atoms, b.atoms);
    let text = |lines: &[Line]| lines.iter().map(progress).collect::<Vec<_>>();
    assert_eq!(text(&heard_a), text(&heard_b));
}

/// A piano's C4 as above: the fit takes more than one pass, adjusts among them, each counted
/// by what it changed, and the DL falls at every pass.
#[test]
fn a_decaying_stiff_string_is_adjusted_after_it_is_added() {
    let parts: Vec<(f64, f64, f64)> = voice(261.63, 4e-4, 12, 0.3)
        .into_iter()
        .enumerate()
        .map(|(k, (hz, a))| (hz, a, 1.0 + 0.25 * (k + 1) as f64))
        .collect();
    let (fit, lines) = fitted(&decaying(&parts));
    assert!(
        fit.passes > 1 && fit.adjusts >= 1,
        "{} passes, {} adjusts",
        fit.passes,
        fit.adjusts
    );
    assert_eq!(fit.tweaks.iter().sum::<usize>(), fit.adjusts);
    let tweaked = lines.iter().filter(|l| l.action == Action::Adjust);
    assert!(tweaked.clone().all(|l| l.tweak.is_some()));
    assert!(
        tweaked
            .clone()
            .all(|l| Tweak::ALL.contains(&l.tweak.expect("a tweak")))
    );
    let dl = dls(&lines);
    assert!(dl.windows(2).all(|w| w[1] < w[0]), "{dl:?}");
}

/// A piano's C4 decaying harmonic by harmonic, as above.
fn piano() -> Vec<f64> {
    let parts: Vec<(f64, f64, f64)> = voice(261.63, 4e-4, 12, 0.3)
        .into_iter()
        .enumerate()
        .map(|(k, (hz, a))| (hz, a, 1.0 + 0.25 * (k + 1) as f64))
        .collect();
    decaying(&parts)
}

/// At the default threshold each pass but the last drops the DL by more than its share; at
/// none the fit runs on, its first passes the same.
#[test]
fn the_threshold_ends_a_fit_at_the_first_pass_dropping_under_its_share() {
    let samples = piano();
    let (cut, lines) = fitted_with(&samples, &Options::default());
    let (whole, all) = fitted(&samples);
    assert_eq!(whole.ended, End::Converged);
    let dl = dls(&lines);
    let under = |w: &[f64]| w[0] - w[1] <= TOLERANCE * w[0].abs().max(w[1].abs()).max(1.0);
    let passes = &dl[1..];
    assert!(
        passes[..passes.len() - 1].windows(2).all(|w| !under(w)),
        "{dl:?}"
    );
    assert_eq!(
        cut.ended == End::Tolerance,
        under(&dl[dl.len() - 2..]),
        "{dl:?}"
    );
    assert!(cut.passes <= whole.passes);
    let text = |lines: &[Line]| lines.iter().map(progress).collect::<Vec<_>>();
    assert_eq!(text(&lines), text(&all[..lines.len()]));
}

/// A fit resumed from an earlier one's composition starts at its atoms and DL, numbering on.
#[test]
fn a_resumed_fit_goes_on_from_the_composition_it_reads() {
    let samples = piano();
    let (cut, lines) = fitted_with(&samples, &Options::default());
    let (whole, _) = fitted(&samples);
    let files = files_at(&cut.atoms, &cut.clip, cut.gain_db);
    let from = Resume::of(&files, cut.passes).expect("the composition reads back");
    assert_eq!(from.atoms, cut.atoms);
    let (resumed, heard) = fitted_with(
        &samples,
        &Options {
            tolerance: 0.0,
            from: Some(from),
            ..Options::default()
        },
    );
    let last = |l: &Line| l.atom_bits + l.residual_bits;
    let before = last(lines.last().expect("a line"));
    assert!(
        (last(&heard[0]) - before).abs() <= 1e-9 * before,
        "{} {before}",
        last(&heard[0])
    );
    assert_eq!(heard[0].pass, cut.passes);
    assert!(
        heard
            .iter()
            .skip(1)
            .zip(cut.passes + 1..)
            .all(|(l, k)| l.pass == k)
    );
    assert_eq!(resumed.from_pass, cut.passes);
    assert_eq!(resumed.passes, cut.passes + resumed.adds + resumed.adjusts);
    assert_eq!(resumed.ended, End::Converged);
    assert!(
        (resumed.atom_bits + resumed.residual_bits)
            <= whole.atom_bits + whole.residual_bits + 1e-6 * before,
        "{} {}",
        resumed.atom_bits + resumed.residual_bits,
        whole.atom_bits + whole.residual_bits
    );
}

/// The code a usage refusal names.
fn usage(e: &CliError) -> String {
    match e {
        CliError::Usage { code, .. } => code.to_string(),
        other => format!("{other:?}"),
    }
}

/// A composition of another clip's length or gain, or of no score, refuses by name.
#[test]
fn a_resume_from_another_clip_refuses_by_name() {
    let samples = piano();
    let (cut, _) = fitted_with(&samples, &Options::default());
    let refused = |samples: &[f64], from: Resume| {
        let options = Options {
            tolerance: 0.0,
            from: Some(from),
            ..Options::default()
        };
        match now(fit(
            samples,
            RATE,
            SPL_DB,
            &options,
            &mut |_| {},
            &Never,
            &|| 0.0,
        )) {
            Ok(_) => panic!("a resume from another clip refuses"),
            Err(e) => usage(&e),
        }
    };
    let files = files_at(&cut.atoms, &cut.clip, cut.gain_db);
    let from = Resume::of(&files, cut.passes).expect("the composition reads back");
    assert_eq!(
        refused(&samples[..samples.len() / 2], from.clone()),
        "solve.from_mismatch"
    );
    let louder: Vec<f64> = samples.iter().map(|s| 2.0 * s).collect();
    assert_eq!(refused(&louder, from), "solve.from_mismatch");
    let held: Vec<(String, String)> = files.into_iter().filter(|(p, _)| p == "master").collect();
    assert_eq!(
        Resume::of(&held, 0).map(|_| ()).map_err(|e| usage(&e)),
        Err("solve.from_unread".to_string())
    );
}

/// C4 from 0 s, E4 from 0.5 s its upper harmonics 6 dB under: E4's lowest line is added second.
#[test]
fn a_late_second_tone_is_added_second_by_its_lowest_line() {
    let len = RATE as usize;
    let late = len / 2;
    let mut parts: Vec<(f64, f64, usize)> = voice(261.63, 0.0, 8, 0.2)
        .into_iter()
        .map(|(hz, a)| (hz, a, 0))
        .collect();
    parts.extend(
        voice(329.63, 0.0, 8, 0.2)
            .into_iter()
            .enumerate()
            .map(|(k, (hz, a))| (hz, if k == 0 { a } else { a / 2.0 }, late)),
    );
    let samples: Vec<f64> = (0..len)
        .map(|n| {
            parts
                .iter()
                .enumerate()
                .filter(|(_, (_, _, on))| n >= *on)
                .map(|(k, (hz, a, _))| a * libm::sin(TAU * hz * t(n) + hashed(k as u64)))
                .sum()
        })
        .collect();
    let (_, lines) = fitted_with(
        &samples,
        &Options {
            tolerance: TOLERANCE,
            ..Options::default()
        },
    );
    let adds: Vec<&Atom> = lines
        .iter()
        .filter(|l| l.action == Action::Add)
        .filter_map(|l| l.atom.as_ref())
        .collect();
    let lowest = |a: &Atom| lines_of(a).iter().copied().fold(f64::INFINITY, f64::min);
    let shown: Vec<String> = adds.iter().map(|a| format!("{:?}", a.sound)).collect();
    assert!(adds.len() >= 2, "{shown:?}");
    assert!((lowest(adds[0]) - 261.63).abs() < 2.0, "{shown:?}");
    assert!((lowest(adds[1]) - 329.63).abs() < 2.0, "{shown:?}");
}
