// Concern: what hits owes a struck train, a legato glide, a dip-and-tick restrike and a low burst | Non-concern: what sounds at a hit (pitch) | IO: (samples) -> asserts

use sva_samples::measure::hits::{Hit, Hits, Kind, measure};

const RATE: f64 = 48_000.0;

fn read(samples: &[f64]) -> Hits {
    measure(&[samples], RATE, 0.0)
}

/// The hit nearest `t`.
fn nearest(h: &Hits, t: f64) -> &Hit {
    h.hits
        .iter()
        .min_by(|a, b| (a.at_secs - t).abs().total_cmp(&(b.at_secs - t).abs()))
        .unwrap_or_else(|| panic!("a hit near {t}: {h:?}"))
}

/// A `hz` sine struck at `at` with a 1.5 ms rise and a `decay` fall, `secs` long.
fn strike(out: &mut [f64], at: f64, hz: f64, (decay, secs): (f64, f64)) {
    let from = (at * RATE) as usize;
    for (n, s) in out
        .iter_mut()
        .skip(from)
        .take((secs * RATE) as usize)
        .enumerate()
    {
        let t = n as f64 / RATE;
        let shape = (1.0 - sva_formula::exp(-t / 0.0015)) * sva_formula::exp(-t / decay);
        *s += 0.7 * shape * libm::sin(std::f64::consts::TAU * hz * t);
    }
}

/// A train of strikes in silence: each read within the resolution, new, and the strongest hits.
#[test]
fn a_struck_sine_train_reads_each_strike_within_the_resolution() {
    let struck = [0.25, 0.5, 0.75];
    let mut x = vec![0.0; RATE as usize];
    for at in struck {
        strike(&mut x, at, 440.0, (0.02, 0.08));
    }
    let h = read(&x);
    assert_eq!(h.resolution_secs, 0.005, "the novelty's hop");
    let mut strongest: Vec<&Hit> = h.hits.iter().collect();
    strongest.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    let weakest = strongest[..struck.len()]
        .iter()
        .map(|h| h.strength)
        .fold(f64::INFINITY, f64::min);
    for at in struck {
        let hit = nearest(&h, at);
        assert!(
            (hit.at_secs - at).abs() <= h.resolution_secs,
            "{} reads within {} of {at}",
            hit.at_secs,
            h.resolution_secs
        );
        assert_eq!(hit.kind, Kind::New, "out of silence: {hit:?}");
    }
    assert!(
        struck.iter().all(|at| nearest(&h, *at).strength >= weakest),
        "the strikes are the strongest hits: {h:?}"
    );
    assert!(h.hits.iter().all(|h| h.strength > 0.0), "{h:?}");
}

/// A tone gliding 300 cents at one level has no attack: the partial cue is gone, so every
/// hit the glide reads is far weaker than the strike that began the tone.
#[test]
fn a_legato_glide_at_constant_level_reads_no_hit() {
    let secs = 1.2;
    let (from, to) = (0.5, 0.7);
    let mut phase = 0.0;
    let x: Vec<f64> = (0..(secs * RATE) as usize)
        .map(|n| {
            let t = n as f64 / RATE;
            let u = ((t - from) / (to - from)).clamp(0.0, 1.0);
            phase += std::f64::consts::TAU * 440.0 * libm::pow(2.0, 0.25 * u) / RATE;
            let rise = 1.0 - sva_formula::exp(-(t - 0.1).max(0.0) / 0.0015);
            match t >= 0.1 {
                true => 0.5 * rise * libm::sin(phase),
                false => 0.0,
            }
        })
        .collect();
    let h = read(&x);
    let struck = nearest(&h, 0.1);
    assert!((struck.at_secs - 0.1).abs() <= h.resolution_secs, "{h:?}");
    let gliding: Vec<&Hit> = h
        .hits
        .iter()
        .filter(|h| h.at_secs > from - 0.05 && h.at_secs < to + 0.05)
        .collect();
    assert!(
        gliding.iter().all(|g| g.strength < 0.1 * struck.strength),
        "no attack in the glide: {gliding:?} against {struck:?}"
    );
}

/// A held tone that dips 10 dB for 20 ms and comes back with a tick is the same note struck
/// again: nothing new sounds after it.
#[test]
fn a_restrike_on_a_dip_and_tick_reads_restrike() {
    let at = 0.5;
    let x: Vec<f64> = (0..(1.0 * RATE) as usize)
        .map(|n| {
            let t = n as f64 / RATE;
            let level = match t > at - 0.02 && t < at {
                true => 0.3,
                false => 1.0,
            };
            let tick = match t >= at && t < at + 0.001 {
                true => 0.3 * libm::sin(std::f64::consts::TAU * 9000.0 * (t - at)),
                false => 0.0,
            };
            0.5 * level * libm::sin(std::f64::consts::TAU * 440.0 * t) + tick
        })
        .collect();
    let h = read(&x);
    let hit = nearest(&h, at);
    assert!((hit.at_secs - at).abs() <= h.resolution_secs, "{h:?}");
    assert_eq!(hit.kind, Kind::Restrike, "{hit:?}");
}

/// A 60 Hz burst's energy lies below the low band edge.
#[test]
fn a_60_hz_bursts_band_energy_is_loudest_in_the_low_band() {
    let mut x = vec![0.0; RATE as usize];
    strike(&mut x, 0.2, 60.0, (0.1, 0.2));
    let h = read(&x);
    let hit = nearest(&h, 0.2);
    assert!((hit.at_secs - 0.2).abs() <= h.resolution_secs, "{h:?}");
    assert_eq!(h.bands_hz, [200.0, 2000.0]);
    let [low, mid, high] = hit.band_db;
    assert!(low > mid && low > high, "{hit:?}");
}
