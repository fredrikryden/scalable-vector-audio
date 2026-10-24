// Concern: states what a darabundit_scavone solver owes the buffer it steps out | Non-concern: the site's own FD math (src/physics/darabundit_scavone.rs) | IO: (Params) -> asserted peaks

use crate::helpers::fd;

use sva_samples::Params;
use sva_samples::physics::darabundit_scavone::{BoreParams, MAX_HOLES, ToneholeSpec};

const C0_M_S: f64 = 347.23;

fn holes(specs: &[(f64, bool, f64)]) -> [Option<ToneholeSpec>; MAX_HOLES] {
    let mut out = [None; MAX_HOLES];
    for (slot, &(pos, open, radius)) in out.iter_mut().zip(specs) {
        *slot = Some(ToneholeSpec {
            pos,
            open,
            radius,
            height: 1.1 * radius,
        });
    }
    out
}

/// The stubby bore spends 10.7% of its corrected length on the end correction, so that term
/// is under test; `(k_3.a)^2 ~ 2` there leaves no clean third peak, so the slender bore
/// covers mode 3.
#[test]
fn a_stepped_spectrum_matches_the_closed_open_quarter_wave_formula() {
    for (case, length, radius, modes) in [("stubby", 0.20, 0.04, 2u32), ("slender", 0.5, 0.01, 3)] {
        let params = Params::DarabunditScavone(BoreParams {
            radius_in: radius,
            radius_out: radius,
            pulse_width: 0.0005,
            damp_dc: 0.0,
            damp_freq: 0.0,
            ..BoreParams::at(length)
        });
        let buffer = fd::render(&params, 44_100, 0.5);
        fd::sounds(&buffer);
        let peaks = fd::peaks(&buffer, 24, None);

        let corrected_length = length + 0.6 * radius;
        for n in 1..=modes {
            let predicted = f64::from(2 * n - 1) * C0_M_S / (4.0 * corrected_length);
            let found = fd::nearest(&peaks, predicted);
            let error = (found - predicted).abs() / predicted;
            assert!(
                error < 0.05,
                "{case} mode {n}: measured {found} predicted {predicted} error {error}"
            );
        }
    }
}

/// Regression: two holes on one grid node under-counted that node's admittance.
#[test]
fn two_holes_rounding_to_the_same_grid_node_still_step_finite_and_stable() {
    let params = Params::DarabunditScavone(BoreParams {
        holes: holes(&[(0.4, true, 0.002), (0.6, false, 0.003)]),
        ..BoreParams::at(0.18)
    });
    fd::sounds(&fd::render(&params, 8_000, 0.1));
}

#[test]
fn a_parameter_out_of_range_is_refused() {
    let base = || BoreParams::at(0.4);
    fd::refuses(
        &Params::DarabunditScavone(base()),
        [
            BoreParams {
                length: 0.0,
                ..base()
            },
            BoreParams {
                radius_in: 0.0,
                ..base()
            },
            BoreParams {
                radius_out: -1.0,
                ..base()
            },
            BoreParams {
                excite_pos: 1.0,
                ..base()
            },
            BoreParams {
                pulse_width: 0.0,
                ..base()
            },
            BoreParams {
                damp_dc: -0.1,
                ..base()
            },
            BoreParams {
                length: f64::NAN,
                ..base()
            },
        ]
        .into_iter()
        .map(Params::DarabunditScavone)
        .collect(),
    );
}
