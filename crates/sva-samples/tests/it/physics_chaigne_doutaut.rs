// Concern: states what a chaigne_doutaut solver must hold over the buffer it steps out | Non-concern: the site's own FD math (src/physics/chaigne_doutaut.rs) | IO: (Params) -> asserted peaks

use crate::helpers::fd;

use sva_samples::Params;
use sva_samples::physics::chaigne_doutaut::ChaigneDoutautParams;

/// Pinned `frame_secs`: two rates, one bin grid.
fn spectrum_peaks(f0: f64, rate: u32, secs: f64) -> Vec<f64> {
    let params = Params::ChaigneDoutaut(ChaigneDoutautParams {
        damp_dc: 0.0,
        damp_freq: 0.0,
        ..ChaigneDoutautParams::at(f0)
    });
    let buffer = fd::render(&params, rate, secs);
    fd::sounds(&buffer);
    fd::peaks(&buffer, 24, Some(secs))
}

/// Fletcher & Rossing's free-free roots; a low `f0` buys nodes for the partials.
#[test]
fn a_stepped_spectrum_matches_the_free_free_transcendental_roots() {
    let peaks = spectrum_peaks(40.0, 44_100, 2.0);
    assert!(peaks.len() >= 2, "not enough peaks found: {peaks:?}");

    let f1 = peaks[0];
    let targets = [1.0, 2.756_508_486, 5.404_001_396, 8.932_762_58];
    let ratios: Vec<f64> = targets
        .iter()
        .map(|&t| fd::nearest(&peaks, f1 * t) / f1)
        .collect();

    // Fewer grid nodes resolve each higher partial, so no bound tightens with order.
    for (i, tolerance) in [(1, 0.01), (2, 0.02), (3, 0.03)] {
        let error = (ratios[i] - targets[i]).abs() / targets[i];
        assert!(
            error < tolerance,
            "partial {} ratio {} is {error} off the transcendental root {}",
            i + 1,
            ratios[i],
            targets[i]
        );
    }
}

#[test]
fn a_parameter_out_of_range_is_refused() {
    let base = || ChaigneDoutautParams::at(261.6);
    fd::refuses(
        &Params::ChaigneDoutaut(base()),
        [
            ChaigneDoutautParams { f0: 0.0, ..base() },
            ChaigneDoutautParams {
                strike_pos: 0.0,
                ..base()
            },
            ChaigneDoutautParams {
                strike_pos: 1.0,
                ..base()
            },
            ChaigneDoutautParams { vel: 0.0, ..base() },
            ChaigneDoutautParams {
                hammer_mass: -1.0,
                ..base()
            },
            ChaigneDoutautParams {
                hammer_k: 0.0,
                ..base()
            },
            ChaigneDoutautParams {
                hammer_p: 0.0,
                ..base()
            },
            ChaigneDoutautParams {
                damp_freq: -0.1,
                ..base()
            },
            ChaigneDoutautParams {
                f0: f64::INFINITY,
                ..base()
            },
        ]
        .into_iter()
        .map(Params::ChaigneDoutaut)
        .collect(),
    );
}
