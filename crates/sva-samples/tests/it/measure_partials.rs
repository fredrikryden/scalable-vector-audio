// Concern: what partials owes a lone sinusoid and a tone against narrowband noise | Non-concern: the knots' count | IO: (samples) -> asserts

use std::f64::consts::TAU;

use sva_samples::measure::partials::{Asked, Curve, Kind, Partials, Track, measure};

const RATE: f64 = 16_000.0;

const ASKED: Asked = Asked {
    frame_secs: 0.1,
    window_secs: 0.5,
    hop_secs: 0.005,
    floor_db: -70.0,
    tracks: 120,
};

fn read(secs: f64, at: impl Fn(f64) -> f64) -> Partials {
    let samples: Vec<f64> = (0..(secs * RATE) as usize)
        .map(|i| at(i as f64 / RATE))
        .collect();
    measure(&[&samples], RATE, 0.0, ASKED)
}

fn nearest(p: &Partials, hz: f64) -> &Track {
    p.tracks
        .iter()
        .min_by(|a, b| (a.hz - hz).abs().total_cmp(&(b.hz - hz).abs()))
        .expect("a track")
}

fn value(c: &Curve, t: f64) -> f64 {
    let k = c
        .times
        .partition_point(|&s| s <= t)
        .clamp(1, c.times.len() - 1);
    let u = (t - c.times[k - 1]) / (c.times[k] - c.times[k - 1]);
    c.values[k - 1] + (c.values[k] - c.values[k - 1]) * u
}

/// `pwl_integral(t, ...)` from the table's first knot, in cycles.
fn cycles(c: &Curve, t: f64) -> f64 {
    let mut held = 0.0;
    for k in 1..c.times.len() {
        let (a, b) = (c.times[k - 1], c.times[k].min(t));
        if b <= a {
            break;
        }
        held += (b - a) * (c.values[k - 1] + value(c, b)) / 2.0;
    }
    held
}

fn interior() -> impl Iterator<Item = f64> {
    (20..=180).map(|k| k as f64 * 0.005)
}

/// One sinusoid is one tonal track: its frequency and phase exact, its amplitude within the
/// table's own `error`.
#[test]
fn a_lone_sinusoid_reads_within_its_stated_bounds() {
    let (hz, amp, phase) = (440.37, 0.3, 0.7);
    let p = read(1.0, |t| amp * libm::sin(TAU * hz * t + phase));
    assert_eq!(p.tracks.len(), 1, "{p:?}");
    let track = &p.tracks[0];
    assert_eq!(track.kind, Kind::Tonal);
    assert!((track.hz - hz).abs() < 1e-3, "{}", track.hz);
    for t in interior() {
        assert!(
            (value(&track.frequency, t) - hz).abs() < 1e-3,
            "{t}: {track:?}"
        );
        let off = (value(&track.amplitude, t) - amp).abs();
        assert!(
            off <= track.amplitude.error + 1e-6,
            "{t}: {off} past {track:?}"
        );
    }
    let read = TAU * cycles(&track.frequency, 0.5) + track.phase.expect("a tonal phase");
    let drift = (read - (TAU * hz * 0.5 + phase)).rem_euclid(TAU);
    assert!(drift.min(TAU - drift) < 1e-3, "{drift}");
    assert!(
        track.residual_db.expect("a tonal bound") < -30.0,
        "{track:?}"
    );
}

/// Noise 160 Hz wide flutters where a tone holds: it reads breathy, the tone tonal.
#[test]
fn narrowband_noise_reads_breathy_and_a_tone_tonal() {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut draw = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let lines: Vec<(f64, f64)> = (0..400)
        .map(|_| (720.0 + 160.0 * draw(), TAU * draw()))
        .collect();
    let scale = 0.05 / (lines.len() as f64 / 2.0).sqrt();
    let p = read(1.0, |t| {
        let noise: f64 = lines
            .iter()
            .map(|(hz, ph)| libm::sin(TAU * hz * t + ph))
            .sum();
        0.1 * libm::sin(TAU * 400.0 * t) + scale * noise
    });
    assert_eq!(nearest(&p, 400.0).kind, Kind::Tonal, "{p:?}");
    let band = p
        .tracks
        .iter()
        .filter(|t| (700.0..900.0).contains(&t.hz))
        .max_by(|a, b| {
            let peak = |t: &Track| t.amplitude.values.iter().copied().fold(0.0, f64::max);
            peak(a).total_cmp(&peak(b))
        })
        .expect("a track in the noise");
    assert_eq!(band.kind, Kind::Breathy, "{band:?}");
    assert!(band.flutter_db > 2.0, "{band:?}");
    assert!(
        band.bandwidth_hz.is_some() && band.phase.is_none(),
        "{band:?}"
    );
}
