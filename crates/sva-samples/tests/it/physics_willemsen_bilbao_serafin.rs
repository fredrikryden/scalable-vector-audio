// Concern: states what a bowed-string solver owes the buffer it steps out | Non-concern: the site's own FD/friction math (src/physics/willemsen_bilbao_serafin.rs) | IO: (Params) -> asserted peaks

use crate::helpers::fd;

use sva_samples::physics::willemsen_bilbao_serafin::WillemsenBilbaoSerafinParams;
use sva_samples::{Params, site};

fn bowed(f0: f64, b: f64, bow_force: f64, bow_pos: f64) -> Params {
    Params::WillemsenBilbaoSerafin(WillemsenBilbaoSerafinParams {
        b,
        bow_pos,
        bow_force,
        ..WillemsenBilbaoSerafinParams::at(f0)
    })
}

/// Schelleng: extreme `bow_force` locks onto an octave-doubled period, empirically.
fn dominant_period(f0: f64, rate: u32, bow_force: f64) -> (f64, usize) {
    let buffer = fd::render(&bowed(f0, 1e-4, bow_force, 0.25), rate, 0.5);
    fd::sounds(&buffer);
    let samples = buffer.plane(0);
    let sr = f64::from(rate);

    let nominal = (sr / f0).round() as usize;
    let tail = &samples[samples.len() - 20 * nominal..];
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let centered: Vec<f64> = tail.iter().map(|&s| s - mean).collect();
    let energy: f64 = centered.iter().map(|c| c * c).sum();
    assert!(energy > 0.0, "a silent tail tests nothing");

    (nominal / 2..nominal + nominal / 2)
        .map(|lag| {
            let corr = centered[lag..]
                .iter()
                .zip(&centered)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / energy;
            (corr, lag)
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .expect("a lag window with at least one lag")
}

#[test]
fn a_moderate_bow_force_sustains_the_nominal_period_an_extreme_one_does_not() {
    let (f0, rate) = (440.0, 44_100u32);
    let (moderate_corr, moderate_lag) = dominant_period(f0, rate, 5.0);
    let (_, extreme_lag) = dominant_period(f0, rate, 400.0);
    let nominal = (f64::from(rate) / f0).round();
    assert!(
        moderate_corr > 0.8,
        "moderate bow force should sustain periodic Helmholtz-like motion, got {moderate_corr}"
    );
    let moderate_off = (moderate_lag as f64 - nominal).abs() / nominal;
    let extreme_off = (extreme_lag as f64 - nominal).abs() / nominal;
    assert!(
        moderate_off < 0.2,
        "moderate bow force's dominant period {moderate_lag} strayed too far from {nominal}"
    );
    assert!(
        extreme_off > moderate_off + 0.2,
        "extreme bow force should lock onto a different period: moderate {moderate_lag}, \
         extreme {extreme_lag}, nominal {nominal}"
    );
}

#[test]
fn a_parameter_out_of_range_is_refused() {
    let base = || WillemsenBilbaoSerafinParams::at(440.0);
    fd::refuses(
        &Params::WillemsenBilbaoSerafin(base()),
        [
            WillemsenBilbaoSerafinParams { f0: 0.0, ..base() },
            WillemsenBilbaoSerafinParams { b: -1.0, ..base() },
            WillemsenBilbaoSerafinParams {
                bow_pos: 0.0,
                ..base()
            },
            WillemsenBilbaoSerafinParams {
                bow_pos: 1.0,
                ..base()
            },
            WillemsenBilbaoSerafinParams {
                bow_force: -1.0,
                ..base()
            },
            WillemsenBilbaoSerafinParams {
                mu_s: 0.0,
                ..base()
            },
            WillemsenBilbaoSerafinParams {
                mu_c: 0.0,
                ..base()
            },
            WillemsenBilbaoSerafinParams {
                stribeck_vel: 0.0,
                ..base()
            },
            WillemsenBilbaoSerafinParams {
                bristle_stiffness: 0.0,
                ..base()
            },
            WillemsenBilbaoSerafinParams {
                f0: f64::NAN,
                ..base()
            },
        ]
        .into_iter()
        .map(Params::WillemsenBilbaoSerafin)
        .collect(),
    );
}

/// A partial 2 past Nyquist has no grid to ring on, so the site refuses instead of diverging.
#[test]
fn a_string_whose_second_partial_passes_nyquist_refuses() {
    let p = Params::WillemsenBilbaoSerafin(WillemsenBilbaoSerafinParams::at(12_000.0));
    let Err(refused) = site(&p, 44_100.0) else {
        panic!("a 12 kHz string rang at 44.1 kHz");
    };
    assert_eq!(refused.code(), "samples.string_past_rate", "{refused:?}");
}

/// A bow a thousand times faster than the reference drives the friction solve past settling:
/// the site refuses at the sample it failed rather than handing the string a force.
#[test]
fn a_friction_solve_that_does_not_settle_refuses() {
    let p = Params::WillemsenBilbaoSerafin(WillemsenBilbaoSerafinParams {
        bow_vel: 1e3,
        ..WillemsenBilbaoSerafinParams::at(440.0)
    });
    let mut solver = site(&p, 44_100.0).expect("a grid this rate holds");
    let refused = (0..44_100)
        .find_map(|_| solver.step(&[]).err())
        .expect("the solve fails to settle within a second");
    assert_eq!(refused.code(), "samples.contact_unsettled", "{refused}");
}
