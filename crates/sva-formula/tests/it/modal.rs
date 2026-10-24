// Concern: proves each geometry's eigenfrequencies and the hammer's contact closed form | Non-concern: the transform over the bank they build (duality.rs) | IO: (a geometry) -> its modes

use std::f64::consts::{PI, TAU};

use sva_formula::modal::bar::{BarGeom, beta};
use sva_formula::modal::bore::{BoreGeom, Ends, correction, helmholtz};
use sva_formula::modal::membrane::MembraneGeom;
use sva_formula::modal::room::RoomGeom;
use sva_formula::modal::string::{StringGeom, inharmonicity};
use sva_formula::modal::{HammerGeom, strike};
use sva_formula::{Damping, Excitation, Geometry, ModalBank, Mode, modes};

const LOSSLESS: Damping = Damping {
    dc: 1.0,
    per_square_hz: 0.0,
};

fn cents(a: f64, b: f64) -> f64 {
    (1200.0 * libm::log2(a / b)).abs()
}

fn heard(found: &[Mode]) -> Vec<f64> {
    found.iter().map(|m| m.omega / TAU).collect()
}

/// Each geometry is checked against the literature formula written out again here, so a
/// transcription error in the crate cannot agree with a transcription error in the test.
#[test]
fn string_membrane_bar_bore_room_first_ten_modes_within_five_cents() {
    let (young, diameter, tension, length): (f64, f64, f64, f64) = (2.0e11, 1.0e-3, 700.0, 0.62);
    let b = PI.powi(3) * young * diameter.powi(4) / (64.0 * tension * length * length);
    assert_eq!(b, inharmonicity(young, diameter, tension, length));
    let string = StringGeom {
        f0: 262.0,
        inharmonicity: b,
        strike: 1.0 / 8.0,
        damping: LOSSLESS,
    };
    for (k, hz) in heard(&modes(&Geometry::String(string), 10))
        .iter()
        .enumerate()
    {
        let n = k as f64 + 1.0;
        assert!(cents(*hz, n * 262.0 * (1.0 + b * n * n).sqrt()) < 5.0);
    }

    let membrane = MembraneGeom {
        lx: 0.4,
        ly: 0.3,
        tension: 3000.0,
        density: 0.26,
        strike: (0.3, 0.7),
        damping: LOSSLESS,
    };
    let speed = (3000.0f64 / 0.26).sqrt();
    let lowest = heard(&modes(&Geometry::Membrane(membrane), 10));
    let mut wanted: Vec<f64> = (1..=12)
        .flat_map(|m| (1..=12).map(move |n| (m, n)))
        .map(|(m, n)| {
            0.5 * speed * ((f64::from(m) / 0.4).powi(2) + (f64::from(n) / 0.3).powi(2)).sqrt()
        })
        .collect();
    wanted.sort_by(f64::total_cmp);
    for (got, want) in lowest.iter().zip(wanted.iter()) {
        assert!(cents(*got, *want) < 5.0, "{got} against {want}");
    }

    let bar = BarGeom {
        length: 0.35,
        thickness: 0.02,
        young: 1.4e10,
        density: 800.0,
        damping: LOSSLESS,
    };
    let scale = PI * (0.02 / 12f64.sqrt()) * (1.4e10f64 / 800.0).sqrt() / (8.0 * 0.35 * 0.35);
    for (k, hz) in heard(&modes(&Geometry::Bar(bar), 10)).iter().enumerate() {
        assert!(cents(*hz, scale * beta(k).powi(2)) < 5.0);
    }

    let bore = BoreGeom {
        length: 0.6,
        radius: 0.008,
        speed: 343.0,
        ends: Ends::ClosedOpen,
        damping: LOSSLESS,
    };
    let effective = 0.6 + correction(0.008);
    for (k, hz) in heard(&modes(&Geometry::Bore(bore), 10)).iter().enumerate() {
        let n = k as f64 + 1.0;
        assert!(cents(*hz, (2.0 * n - 1.0) * 343.0 / (4.0 * effective)) < 5.0);
    }

    let room = RoomGeom {
        lx: 5.0,
        ly: 4.0,
        lz: 2.6,
        speed: 343.0,
        damping: LOSSLESS,
    };
    let got = heard(&modes(&Geometry::Room(room), 10));
    let mut wanted: Vec<f64> = (0..=6)
        .flat_map(|l| (0..=6).flat_map(move |m| (0..=6).map(move |n| (l, m, n))))
        .filter(|(l, m, n)| *l + *m + *n > 0)
        .map(|(l, m, n)| {
            0.5 * 343.0
                * ((f64::from(l) / 5.0).powi(2)
                    + (f64::from(m) / 4.0).powi(2)
                    + (f64::from(n) / 2.6).powi(2))
                .sqrt()
        })
        .collect();
    wanted.sort_by(f64::total_cmp);
    for (a, b) in got.iter().zip(wanted.iter()) {
        assert!(cents(*a, *b) < 5.0, "{a} against {b}");
    }

    let (volume, area, neck, radius) = (0.002, 1.5e-4, 0.02, 0.007);
    let effective = neck + 1.7 * radius;
    assert!(
        cents(
            helmholtz(volume, area, neck, radius, 343.0),
            343.0 / TAU * (area / (volume * effective)).sqrt()
        ) < 5.0
    );
}

/// A struck bank is each mode's impulse response convolved with the felt's force: it follows
/// the force while the felt presses, and once it lets go rings on with what the contact gave
/// it. Checked against the convolution integral, summed here by Simpson's rule.
#[test]
fn a_struck_bank_is_its_impulse_response_convolved_with_the_force() {
    let felt = HammerGeom {
        reference_velocity: 3.2,
        reference_contact: 1.5e-3,
        stiffness_exponent: 2.5,
        force: 500.0,
    };
    let t0 = 0.25;
    let excite = strike(3.0, &felt, t0);
    let Excitation::HammerPulse { f0, contact, .. } = excite else {
        panic!("a strike is a hammer's pulse");
    };
    let mode = |hz: f64, tau: f64, amp: f64, phase: f64| Mode {
        omega: TAU * hz,
        tau,
        amp,
        phase,
    };
    // Lossless modes on a force line, beside it and at rest meet it where the two cancel.
    let modes = vec![
        mode(220.0, 0.5, 1.0, 0.0),
        mode(1500.0, 0.05, 0.4, 0.3),
        mode(1.0 / contact, f64::INFINITY, 0.2, 0.0),
        mode((1.0 + 1e-12) / contact, f64::INFINITY, 0.2, 0.1),
        mode(0.0, f64::INFINITY, 0.1, 0.0),
    ];
    let bank = ModalBank {
        modes: modes.clone(),
        excite,
    };
    let atoms = sva_formula::modal::atoms(&bank, sva_formula::Origin::UNKNOWN);
    let value = |t: f64| -> f64 {
        atoms
            .iter()
            .map(|a| a.smooth_at(t).expect("no pole").re)
            .sum()
    };
    let rings = |u: f64| -> f64 {
        modes
            .iter()
            .map(|m| m.amp * sva_formula::exp(-u / m.tau) * libm::cos(m.omega * u + m.phase))
            .sum()
    };
    let force = |s: f64| f0 * 0.5 * (1.0 - libm::cos(TAU * (s - t0) / contact));
    let convolved = |t: f64| -> f64 {
        let end = t.min(t0 + contact);
        if end <= t0 {
            return 0.0;
        }
        let n = 20_000;
        let h = (end - t0) / f64::from(n);
        let at = |k: u32| {
            let s = t0 + f64::from(k) * h;
            force(s) * rings(t - s)
        };
        let inner: f64 = (1..n)
            .map(|k| if k % 2 == 1 { 4.0 } else { 2.0 } * at(k))
            .sum();
        h / 3.0 * (at(0) + inner + at(n))
    };
    let momentum = f0 * contact * 0.5;
    for t in [
        t0 - 0.01,
        t0 + 0.3 * contact,
        t0 + 0.9 * contact,
        t0 + 1.5 * contact,
        t0 + 0.01,
        t0 + 0.4,
    ] {
        let (got, want) = (value(t), convolved(t));
        assert!(
            (got - want).abs() < 1e-9 * momentum,
            "at {t}: {got} against the integral's {want}"
        );
    }
    assert!(
        value(t0 + 0.4).abs() > 1e-3 * momentum,
        "the struck bank still rings long after the felt let go"
    );
}
