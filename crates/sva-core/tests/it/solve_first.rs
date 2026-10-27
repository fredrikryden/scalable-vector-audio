// Concern: states the first fit recovers a partial beside a band, two partials in one Cam band, a decaying voice, the same atoms each run | Non-concern: the pyramid | IO: (rendered atoms) -> asserts

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

fn partials(atoms: &[Atom]) -> Vec<(f64, &Atom)> {
    atoms
        .iter()
        .filter_map(|a| match &a.sound {
            Sound::Partial { hz } => Some((hz.base, a)),
            _ => None,
        })
        .collect()
}

fn with_band() -> Vec<Atom> {
    vec![
        partial(0, 440.0, -20.0),
        Atom::noise(1, 0.0, (2000.0, 4000.0), steady(-30.0)),
    ]
}

#[test]
fn a_partial_and_a_band_come_back_at_their_hz_and_levels() {
    let found = first(&with_band());
    let (hz, tone) = partials(&found)
        .into_iter()
        .min_by(|a, b| (a.0 - 440.0).abs().total_cmp(&(b.0 - 440.0).abs()))
        .expect("a partial");
    assert!((hz - 440.0).abs() <= BIN_HZ / 2.0, "{hz} Hz");
    let level = level_at(tone, 0.5);
    assert!((level + 20.0).abs() <= 0.5, "{level} dB");
    let noise: f64 = found
        .iter()
        .filter(|a| matches!(a.sound, Sound::Noise { .. }))
        .map(|a| libm::pow(10.0, level_at(a, 0.5) / 10.0))
        .sum();
    let noise = 10.0 * libm::log10(noise);
    assert!((noise + 30.0).abs() <= 1.0, "{noise} dB");
}

#[test]
fn two_partials_in_one_cam_band_both_come_back() {
    let found = first(&[partial(0, 1000.0, -20.0), partial(1, 1030.0, -20.0)]);
    let tones = partials(&found);
    for want in [1000.0, 1030.0] {
        let (hz, atom) = tones
            .iter()
            .min_by(|a, b| (a.0 - want).abs().total_cmp(&(b.0 - want).abs()))
            .expect("a partial");
        assert!((hz - want).abs() <= BIN_HZ / 2.0, "{want}: {hz} Hz");
        let level = level_at(atom, 0.5);
        assert!((level + 20.0).abs() <= 1.0, "{want}: {level} dB");
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
    let voices: Vec<&Atom> = found
        .iter()
        .filter(|a| matches!(a.sound, Sound::Harmonic { .. }))
        .collect();
    assert_eq!(voices.len(), 1, "one voice");
    let voice = voices[0];
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
