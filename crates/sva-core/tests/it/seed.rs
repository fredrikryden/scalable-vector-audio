// Concern: states one reading of a clip writes the starter `seed` writes, the same bytes at any limits | Non-concern: fitting it | IO: (a clip, limits) -> files, dropped
use sva_core::seed::{LIMITS, Limits, Seeded, read, seed, starter};

const RATE: f64 = 48_000.0;

/// FNV-1a of the starter's text at `LIMITS` and at `{1, 1}`.
const PINNED: [u64; 2] = [0x927f_c60d_5460_5c55, 0xdb6b_53e7_d053_c66e];

/// A plucked 220 Hz tone with its octave, 0.3 s.
fn clip() -> Vec<Vec<f32>> {
    vec![
        (0..14_400)
            .map(|n| {
                let t = n as f64 / RATE;
                let w = |hz: f64| libm::sin(2.0 * std::f64::consts::PI * hz * t);
                (0.1 * sva_formula::exp(-6.0 * t) * (w(220.0) + 0.5 * w(440.0))) as f32
            })
            .collect(),
    ]
}

fn bytes(s: &Seeded) -> u64 {
    let text = s
        .files
        .iter()
        .flat_map(|(p, t)| [p.as_str(), t.as_str()])
        .chain(s.dropped.iter().map(String::as_str))
        .chain(s.varies.iter().map(String::as_str))
        .chain([s.cost.as_str()])
        .flat_map(|x| x.bytes().chain([0]));
    text.fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn one_reading_writes_the_starter_seed_writes_at_each_limit() {
    let channels = clip();
    let readings = read(&channels, RATE);
    let limits = [
        LIMITS,
        Limits {
            harmonics: 1,
            lone: 1,
        },
    ];
    let seeded: Vec<u64> = limits
        .iter()
        .map(|l| bytes(&seed(&channels, RATE, "clip.wav", *l)))
        .collect();
    assert_eq!(seeded, PINNED, "seed's starter bytes at {limits:?}");
    let split: Vec<u64> = limits
        .iter()
        .map(|l| bytes(&starter(&readings, "clip.wav", *l)))
        .collect();
    assert_eq!(split, PINNED, "one reading's starters at {limits:?}");
}
