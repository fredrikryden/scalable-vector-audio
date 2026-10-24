// Concern: what harmonic readings owe textbook chords: roughness minima, Tonnetz paths | Non-concern: closed forms | IO: (partials or samples) -> asserts

use sva_samples::measure::harmony::dissonance::{Dissonance, measure};
use sva_samples::measure::harmony::tonnetz::{Quality, Step, Triad, path};
use sva_samples::measure::harmony::{Framing, Moment, Partial};

const C4: f64 = 261.625_565_300_598_6;
const PAIRS: usize = 1 << 17;

fn exact(hz: f64, amplitude: f64) -> Partial {
    Partial {
        hz,
        amplitude,
        hz_bounds: (hz, hz),
        amplitude_bounds: (amplitude, amplitude),
    }
}

fn rough(moments: &[Moment], framing: Framing, floor_db: f64) -> Dissonance {
    measure(moments, framing, floor_db, PAIRS).expect("few partials")
}

fn hz(midi: i64) -> f64 {
    440.0 * libm::pow(2f64, (midi as f64 - 69.0) / 12.0)
}

/// Six harmonics falling 0.88 per partial, the timbre of Sethares' Fig. 2.
fn harmonic_tone(f0: f64) -> Vec<Partial> {
    (1..=6)
        .map(|n| exact(f0 * f64::from(n), 0.88f64.powi(n - 1)))
        .collect()
}

fn chord(midis: &[i64]) -> Moment {
    Moment {
        t_secs: 0.0,
        partials: midis.iter().flat_map(|m| harmonic_tone(hz(*m))).collect(),
    }
}

const EXACT: Framing = Framing {
    hop_secs: 0.05,
    window: None,
};

fn roughness(moment: Moment) -> f64 {
    rough(&[moment], EXACT, -120.0).frames[0].roughness
}

/// With harmonic tones the fifth and the octave are local minima of Sethares' curve, and the
/// major and minor triads are far smoother than a cluster.
#[test]
fn consonant_intervals_and_triads_are_smoother_than_their_neighbours_and_clusters() {
    let interval = |semis: f64| {
        let mut two = harmonic_tone(C4);
        two.extend(harmonic_tone(C4 * libm::pow(2f64, semis / 12.0)));
        roughness(Moment {
            t_secs: 0.0,
            partials: two,
        })
    };
    for (just, at) in [(1.5f64, "fifth"), (2.0, "octave")] {
        let semis = 12.0 * libm::log2(just);
        assert!(
            interval(semis) < interval(semis - 0.3),
            "{at} below its flat side"
        );
        assert!(
            interval(semis) < interval(semis + 0.3),
            "{at} below its sharp side"
        );
    }
    let major = roughness(chord(&[60, 64, 67]));
    let minor = roughness(chord(&[60, 63, 67]));
    let cluster = roughness(chord(&[60, 61, 62]));
    assert!(major < 0.5 * cluster, "{major} vs {cluster}");
    assert!(minor < 0.5 * cluster, "{minor} vs {cluster}");
}

fn triad(root: u8, quality: Quality) -> Triad {
    Triad { root, quality }
}

const MAJOR: Quality = Quality::Major;
const MINOR: Quality = Quality::Minor;

/// ii-V-I in C: d to G is three steps (d P D R b L G), G to C two (G R e L C).
/// I-V-vi-IV: C to G two, G to a three, a to F one (a L F).
#[test]
fn the_tonnetz_counts_the_textbook_neo_riemannian_paths() {
    let (c, d, f, g, a) = (
        triad(0, MAJOR),
        triad(2, MINOR),
        triad(5, MAJOR),
        triad(7, MAJOR),
        triad(9, MINOR),
    );
    assert_eq!(Step::R.of(c), a);
    assert_eq!(Step::L.of(c), triad(4, MINOR));
    assert_eq!(Step::P.of(c), triad(0, MINOR));
    let walked = |from: Triad, steps: Vec<Step>| steps.into_iter().fold(from, |t, s| s.of(t));
    assert_eq!(path(d, g), vec![Step::P, Step::R, Step::L]);
    assert_eq!(walked(d, path(d, g)), g);
    assert_eq!(walked(g, path(g, a)), a);
    assert_eq!(path(g, c).len(), 2);
    assert_eq!(path(c, g).len(), 2);
    assert_eq!(path(g, a).len(), 3);
    assert_eq!(path(a, f), vec![Step::L]);
    assert!(path(c, c).is_empty());
}
