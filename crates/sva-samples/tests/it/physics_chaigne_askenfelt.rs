// Concern: states what a chaigne_askenfelt solver must hold over the buffer it steps out | Non-concern: the site's own FD math (src/physics/chaigne_askenfelt.rs) | IO: (Params) -> asserted peaks

use crate::helpers::fd;

use sva_samples::physics::chaigne_askenfelt::ChaigneAskenfeltParams;
use sva_samples::{Params, site};

#[test]
fn a_stepped_spectrum_matches_the_stiff_string_partial_formula() {
    let (f0, b) = (220.0, 0.01);
    let params = Params::ChaigneAskenfelt(ChaigneAskenfeltParams {
        b,
        ..ChaigneAskenfeltParams::at(f0)
    });
    let buffer = fd::render(&params, 44_100, 0.5);
    fd::sounds(&buffer);
    let peaks = fd::peaks(&buffer, 24, None);

    for n in 1..=6u32 {
        let predicted = f64::from(n) * f0 * (1.0 + b * f64::from(n * n)).sqrt();
        let found = fd::nearest(&peaks, predicted);
        let error = (found - predicted).abs() / predicted;
        assert!(
            error < 0.05,
            "partial {n}: measured {found} predicted {predicted} error {error}"
        );
    }
}

#[test]
fn a_chaigne_askenfelt_site_steps_the_same_buffer_every_time() {
    let params = Params::ChaigneAskenfelt(ChaigneAskenfeltParams {
        vel: 3.2,
        ..ChaigneAskenfeltParams::at(220.0)
    });
    fd::is_deterministic(&params, 22_050, 0.02);
}

#[test]
fn a_parameter_out_of_range_is_refused() {
    let base = || ChaigneAskenfeltParams::at(220.0);
    fd::refuses(
        &Params::ChaigneAskenfelt(base()),
        [
            ChaigneAskenfeltParams { f0: 0.0, ..base() },
            ChaigneAskenfeltParams { b: -1.0, ..base() },
            ChaigneAskenfeltParams {
                strike_pos: 0.0,
                ..base()
            },
            ChaigneAskenfeltParams {
                strike_pos: 1.0,
                ..base()
            },
            ChaigneAskenfeltParams { vel: 0.0, ..base() },
            ChaigneAskenfeltParams {
                hammer_mass: -1.0,
                ..base()
            },
            ChaigneAskenfeltParams {
                hammer_k: 0.0,
                ..base()
            },
            ChaigneAskenfeltParams {
                hammer_p: 0.0,
                ..base()
            },
            ChaigneAskenfeltParams {
                damp_dc: -0.1,
                ..base()
            },
            ChaigneAskenfeltParams {
                unison_count: 0.0,
                ..base()
            },
            ChaigneAskenfeltParams {
                f0: f64::NAN,
                ..base()
            },
            ChaigneAskenfeltParams {
                bridge_mass: -1.0,
                ..base()
            },
            ChaigneAskenfeltParams {
                string_cents: [0.0, f64::INFINITY, 0.0],
                ..base()
            },
            ChaigneAskenfeltParams {
                string_hammer_k_ratio: [1.0, 1.0, 0.0],
                ..base()
            },
        ]
        .into_iter()
        .map(Params::ChaigneAskenfelt)
        .collect(),
    );
}

/// A partial 2 past Nyquist has no grid to ring on, so the site refuses instead of diverging.
#[test]
fn a_string_whose_second_partial_passes_nyquist_refuses() {
    let p = Params::ChaigneAskenfelt(ChaigneAskenfeltParams::at(12_000.0));
    let Err(refused) = site(&p, 44_100.0) else {
        panic!("a 12 kHz string rang at 44.1 kHz");
    };
    assert_eq!(refused.code(), "samples.string_past_rate", "{refused:?}");
}

/// A light bridge barely damped under stiff, lossy strings: each string alone is stable, but
/// the bridge row leaves the unison's mass form indefinite.
#[test]
fn a_unison_its_bridge_leaves_unstable_refuses() {
    let strings = ChaigneAskenfeltParams {
        b: 1e-4,
        damp_freq: 1e-4,
        bridge_coupling: 0.1,
        bridge_mass: 0.0,
        ..ChaigneAskenfeltParams::at(55.0)
    };
    assert!(site(&Params::ChaigneAskenfelt(strings.clone()), 44_100.0).is_ok());
    let unison = ChaigneAskenfeltParams {
        unison_count: 3.0,
        ..strings
    };
    let Err(refused) = site(&Params::ChaigneAskenfelt(unison), 44_100.0) else {
        panic!("an unstable unison stepped");
    };
    assert_eq!(refused.code(), "samples.bridge_unstable", "{refused:?}");
    for stable in [published_unison(), weinreich_unison()] {
        assert!(site(&Params::ChaigneAskenfelt(stable), 44_100.0).is_ok());
    }
}

/// A three-string unison as a published equation writes one at middle C, beating to nulls.
fn published_unison() -> ChaigneAskenfeltParams {
    ChaigneAskenfeltParams {
        b: 0.000_492_652_403_124_032_3,
        strike_pos: 0.088_514_488_958_233_88,
        vel: 4.543_307_086_614_173,
        hammer_mass: 0.006_412_505_482_227_443,
        hammer_k: 384_167_121.960_794_87,
        hammer_p: 2.399_554_894_193_862_7,
        damp_dc: 0.885_018_811_046_045_9,
        damp_freq: 0.000_441_094_138_384_720_24,
        unison_count: 3.0,
        detune: 1.001_465_045_903_355_8,
        bridge_coupling: 2_603.290_393_684_533_2,
        ..ChaigneAskenfeltParams::at(261.6256)
    }
}

/// The same strings on a massive bridge, each tuned and struck a little differently.
fn weinreich_unison() -> ChaigneAskenfeltParams {
    ChaigneAskenfeltParams {
        detune: 1.0,
        bridge_coupling: 100.0,
        bridge_mass: 1.0,
        string_cents: [-0.35, 0.1, 0.5],
        string_hammer_k_ratio: [1.0, 0.8, 0.6],
        ..published_unison()
    }
}
