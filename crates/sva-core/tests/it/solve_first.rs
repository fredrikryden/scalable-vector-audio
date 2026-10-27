// Concern: states the coarse fit's sine, sine beside a band, three close sines over a floor, decaying voice, same atoms each run | Non-concern: the pyramid | IO: (rendered atoms) -> asserts

use sva_core::solve::atoms::{Atom, Clip, Sound, Trajectory};
use sva_core::solve::calibrate::rendered;
use sva_core::solve::first::fit;

const SPL: f64 = 100.0;
const CLIP: Clip = Clip {
    rate: 16_000,
    len: 16_000,
};

/// The analysis bin at `CLIP`'s rate: a 64 ms window padded to 1024 samples.
const BIN_HZ: f64 = 16_000.0 / 1024.0;

fn steady(db: f64) -> Trajectory {
    Trajectory {
        base: db,
        knots: vec![(CLIP.secs(), 0.0)],
    }
}

fn partial(key: u64, hz: f64, db: f64) -> Atom {
    Atom::partial(key, 0.0, Trajectory::steady(hz), steady(db))
}

fn first(atoms: &[Atom]) -> Vec<Atom> {
    let found = fit(&rendered(atoms, &CLIP), CLIP.rate, SPL).atoms;
    for atom in &found {
        println!("{atom:?}");
    }
    found
}

fn level_at(atom: &Atom, t: f64) -> f64 {
    sva_core::solve::first::level_at(&atom.level, atom.on, t)
}

/// The partials', voices' and noises' counts.
fn counts(atoms: &[Atom]) -> (usize, usize, usize) {
    let count = |kind: fn(&Sound) -> bool| atoms.iter().filter(|a| kind(&a.sound)).count();
    (
        count(|s| matches!(s, Sound::Partial { .. })),
        count(|s| matches!(s, Sound::Harmonic { .. })),
        count(|s| matches!(s, Sound::Noise { .. })),
    )
}

fn partial_hz(atoms: &[Atom]) -> (f64, &Atom) {
    atoms
        .iter()
        .find_map(|a| match &a.sound {
            Sound::Partial { hz } => Some((hz.base, a)),
            _ => None,
        })
        .expect("a partial")
}

fn with_band() -> Vec<Atom> {
    vec![
        partial(0, 440.0, -20.0),
        Atom::noise(1, 0.0, (2000.0, 4000.0), steady(-30.0)),
    ]
}

#[test]
fn a_pure_sine_is_one_partial_and_no_noise() {
    let found = first(&[partial(0, 440.0, -20.0)]);
    assert_eq!(counts(&found), (1, 0, 0), "{found:?}");
    let (hz, tone) = partial_hz(&found);
    assert!((hz - 440.0).abs() <= BIN_HZ / 2.0, "{hz} Hz");
    let level = level_at(tone, 0.5);
    assert!((level + 20.0).abs() <= 0.5, "{level} dB");
}

#[test]
fn a_sine_beside_a_band_is_one_partial_and_the_band() {
    let found = first(&with_band());
    let (partials, voices, noises) = counts(&found);
    assert_eq!((partials, voices), (1, 0), "{found:?}");
    assert!((1..=2).contains(&noises), "{noises} noises");
    let (hz, tone) = partial_hz(&found);
    assert!((hz - 440.0).abs() <= BIN_HZ / 2.0, "{hz} Hz");
    let level = level_at(tone, 0.5);
    assert!((level + 20.0).abs() <= 0.5, "{level} dB");
    let times: Vec<f64> = (1..10).map(|k| 0.1 * f64::from(k)).collect();
    let noise: f64 = found
        .iter()
        .filter(|a| matches!(a.sound, Sound::Noise { .. }))
        .flat_map(|a| {
            times
                .iter()
                .map(|t| libm::pow(10.0, level_at(a, *t) / 10.0))
        })
        .sum();
    let noise = 10.0 * libm::log10(noise / times.len() as f64);
    assert!((noise + 30.0).abs() <= 1.0, "{noise} dB");
}

#[test]
fn three_close_sines_over_a_weak_floor_are_three_partials() {
    let found = first(&[
        partial(0, 829.0, -20.0),
        partial(1, 878.0, -24.0),
        partial(2, 918.0, -28.0),
        Atom::noise(3, 0.0, (100.0, 7000.0), steady(-70.0)),
    ]);
    let (partials, voices, _) = counts(&found);
    assert_eq!((partials, voices), (3, 0), "{found:?}");
    for want in [829.0, 878.0, 918.0] {
        assert!(
            found.iter().any(|a| match &a.sound {
                Sound::Partial { hz } => (hz.base - want).abs() <= BIN_HZ,
                _ => false,
            }),
            "no partial near {want} Hz: {found:?}"
        );
    }
}

#[test]
fn a_decaying_harmonic_tone_is_one_voice_following_its_decay() {
    let level = |t: f64| -20.0 - 20.0 * t;
    let tone = Atom::harmonic(
        0,
        0.0,
        Trajectory::steady(220.0),
        (0..7).map(|k| 7.0 - 2.0 * f64::from(k)).collect(),
        Trajectory {
            base: level(0.0),
            knots: vec![(CLIP.secs(), level(1.0) - level(0.0))],
        },
    );
    let found = first(&[tone]);
    assert_eq!(counts(&found).1, 1, "one voice: {found:?}");
    let voice = found
        .iter()
        .find(|a| matches!(a.sound, Sound::Harmonic { .. }))
        .expect("a voice");
    assert_eq!(voice.harmonics(), 8);
    let f0 = voice.hz().expect("a voice's Hz").base;
    assert!((f0 - 220.0).abs() <= BIN_HZ / 2.0, "{f0} Hz");
    for t in [0.2, 0.4, 0.6, 0.8] {
        let got = level_at(voice, t);
        assert!(
            (got - level(t)).abs() <= 1.0,
            "at {t} s: {got} dB, not {}",
            level(t)
        );
    }
}

#[test]
fn a_fit_is_the_same_every_run() {
    let samples = rendered(&with_band(), &CLIP);
    let once = fit(&samples, CLIP.rate, SPL);
    let again = fit(&samples, CLIP.rate, SPL);
    assert_eq!(once.atoms, again.atoms);
    assert_eq!(once.fit_grid.power, again.fit_grid.power);
}
