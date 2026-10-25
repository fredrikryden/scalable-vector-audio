// Concern: states what the objective owes: no bits for the clip itself, a tone's derived bits | Non-concern: N0's fit | IO: (atoms, clip) -> asserts

use std::f64::consts::LN_2;

use sva_core::solve::atoms::{Atom, Clip, Trajectory};
use sva_core::solve::bits::{self, Priced};
use sva_core::solve::calibrate::rendered;
use sva_core::solve::objective::{Objective, total};

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
