// Concern: states what the objective owes: no bits for the clip itself, a tone's derived bits, a noise's variance over keyed renders, Rice | Non-concern: N0's fit | IO: (atoms, clip) -> asserts

use std::f64::consts::{LN_2, PI};

use sva_core::solve::atoms::{Atom, Clip, Trajectory};
use sva_core::solve::bits::{self, Priced};
use sva_core::solve::calibrate::rendered;
use sva_core::solve::objective::{Objective, rice, total};
use sva_engine::internal::modulation::Element;

const SPL: f64 = 100.0;
/// Every assertion here holds at any N0.
const N0: f64 = 1.0;
const CLIP: Clip = Clip {
    rate: 4_000,
    len: 1_000,
};

/// Edges at (2n - 1)/(2 rate), where the gate's shares are the crop's.
fn tone() -> Atom {
    Atom::partial(
        0,
        401.0 / 8000.0,
        Trajectory::steady(1000.0),
        Trajectory {
            base: -40.0,
            knots: vec![(1601.0 / 8000.0, -6.0)],
        },
    )
}

fn gaussian(a: &[f64], b: &[f64], heard: &sva_engine::internal::Internal) -> f64 {
    let mut i = 0;
    let mut nats = 0.0;
    for channel in &heard.channels {
        for band in &channel.bands {
            for _ in &band.elements {
                let d = a[i] - b[i];
                nats += 0.5 * d * d / (N0 * band.rate_hz);
                i += 1;
            }
        }
    }
    nats / LN_2
}

fn flat(read: &sva_engine::internal::Internal) -> Vec<f64> {
    read.channels
        .iter()
        .flat_map(|c| c.bands.iter().flat_map(|b| b.elements.iter().copied()))
        .collect()
}

#[test]
fn the_clip_against_itself_costs_no_residual_bits() {
    let silence = Objective::new(&vec![0.0; CLIP.len], CLIP, SPL, 1.0, N0);
    assert_eq!(silence.bits(&[], true), 0.0);
    let atoms = [tone()];
    let itself = Objective::new(&rendered(&atoms, &CLIP), CLIP, SPL, 1.0, N0);
    let residual = itself.bits(&atoms, true);
    println!("a tone against its own render: {residual} bits");
    assert!((0.0..1e-12).contains(&residual), "{residual}");
    let priced: Vec<Priced> = atoms
        .iter()
        .map(|a| Priced {
            feasible: sva_core::solve::atoms::Feasible::of(&CLIP, -120.0),
            deltas: vec![0.01; a.numbers().len()],
        })
        .collect();
    assert_eq!(
        total(residual, &atoms, &priced),
        residual + bits::bits(&atoms, &priced)
    );
}

/// Silence's elements are 0, so the tone's own elements' Gaussian code.
#[test]
fn a_known_tone_against_silence_costs_the_derived_bits() {
    let atoms = [tone()];
    let silence = Objective::new(&vec![0.0; CLIP.len], CLIP, SPL, 1.0, N0);
    let bits = silence.bits(&atoms, true);
    let heard = silence.ear().read(&[&rendered(&atoms, &CLIP)], 0.0);
    let derived = gaussian(&flat(&heard), &vec![0.0; flat(&heard).len()], &heard);
    println!("a tone against silence: {bits} bits, derived {derived}");
    assert!(derived > 1.0, "{derived}");
    assert!(
        (bits - derived).abs() <= 1e-6 * derived,
        "{bits} against {derived}"
    );
}

/// Lowpass and real bands are what the lines option was measured on.
#[test]
fn a_noise_atom_sits_within_its_stated_variance_over_keyed_renders() {
    const RENDERS: u64 = 48;
    let clip = Clip {
        rate: 8_000,
        len: 4_000,
    };
    let noise = |key| {
        Atom::noise(
            key,
            0.0,
            (1000.0, 1250.0),
            Trajectory {
                base: -40.0,
                knots: vec![(clip.secs(), 0.0)],
            },
        )
    };
    let model = Objective::new(&vec![0.0; clip.len], clip, SPL, 1.0, N0);
    let expected = model.expect(&[noise(0)], true);
    let reads: Vec<Vec<f64>> = (1..=RENDERS)
        .map(|key| flat(&model.ear().read(&[&rendered(&[noise(key)], &clip)], 0.0)))
        .collect();
    let n = RENDERS as f64;
    let mut sums = [(0.0, 0.0); 3];
    let mut i = 0;
    for (channel, bands) in model.heard().channels.iter().zip(&expected) {
        for (band, elements) in channel.bands.iter().zip(bands) {
            let kind = match band.band.element {
                Element::Lowpass => 0,
                Element::Real => 1,
                Element::Magnitude => 2,
            };
            for (j, predicted) in elements.iter().enumerate() {
                let secs = j as f64 / band.rate_hz;
                if (1000.0..1250.0).contains(&channel.hz) && (0.15..0.35).contains(&secs) {
                    let mean = reads.iter().map(|r| r[i]).sum::<f64>() / n;
                    let spread = reads.iter().map(|r| (r[i] - mean).powi(2)).sum::<f64>();
                    sums[kind].0 += spread / (n - 1.0);
                    sums[kind].1 += predicted.variance;
                }
                i += 1;
            }
        }
    }
    let ratios = sums.map(|(renders, predicted)| renders / predicted);
    println!("renders' variance over predicted: lowpass, real, magnitude {ratios:?}");
    for ratio in &ratios[..2] {
        assert!((ratio - 1.0).abs() < 0.15, "{ratios:?}");
    }
    assert!(
        (ratios[2] - 1.89).abs() < 0.01,
        "objective's MODEL states 1.89: {ratios:?}"
    );
}

/// Against mpmath's `besseli` at 50 digits, `v = 2`, `a = sqrt(2x)`.
#[test]
fn rice_moments_hold_on_both_sides_of_the_switch() {
    let v = 2.0;
    let zero = rice(0.0, v);
    assert!(
        (zero.mean - (PI * v).sqrt() / 2.0).abs() < 1e-15,
        "{zero:?}"
    );
    assert!(
        (zero.variance - (1.0 - PI / 4.0) * v).abs() < 1e-15,
        "{zero:?}"
    );
    let mpmath = [
        (0.01, 1.259_572_887_823_704_7, 0.433_476_140_259_453_1),
        (0.5, 1.548_572_460_551_145_4, 0.601_923_334_422_571_3),
        (2.0, 2.272_383_428_068_742_5, 0.836_273_555_838_550_1),
        (10.0, 4.585_464_850_286_769, 0.973_512_106_784_538_9),
        (30.0, 7.810_792_478_547_886, 0.991_520_857_059_774_2),
        (37.0, 8.660_649_611_399_652, 0.993_148_308_563_053_1),
        (40.0, 9.000_351_714_267_215, 0.993_669_019_487_196_2),
        (45.0, 9.539_686_539_666_86, 0.994_380_724_898_934_7),
        (60.0, 11.000_191_008_212_586, 0.995_797_782_838_960_3),
        (200.0, 20.025_015_684_057_22, 0.998_746_853_262_429_2),
    ];
    for (x, m, s) in mpmath {
        let held = rice((x * v).sqrt(), v);
        let (mean_gap, variance_gap) = ((held.mean - m).abs(), (held.variance - s).abs());
        assert!(
            mean_gap <= held.mean_bound + f64::EPSILON * m,
            "x {x}: {held:?} against {m}"
        );
        assert!(
            variance_gap <= held.variance_bound + f64::EPSILON * s,
            "x {x}: {held:?} against {s}"
        );
        assert!(
            held.mean_bound < 1e-12 * m && held.variance_bound < 1e-10 * s,
            "{held:?}"
        );
    }
}
