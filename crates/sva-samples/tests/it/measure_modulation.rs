// Concern: what modulation owes tremolo across rates, an unread sidelobe, refusals | Non-concern: the band filters | IO: (samples) -> asserts

use std::f64::consts::TAU;

use sva_samples::measure::modulation::{Group, Modulation, Track, measure};
use sva_samples::measure::sliding::Unread;

const RATE: f64 = 16_000.0;

fn tone(secs: f64, at: impl Fn(f64) -> f64) -> Vec<f64> {
    (0..(secs * RATE) as usize)
        .map(|i| at(i as f64 / RATE))
        .collect()
}

fn read(samples: &[f64], bpm: Option<f64>) -> Modulation {
    measure(&[samples], RATE, 0.0, 0.5, 4.0, bpm).expect("a 4 s window over a longer target")
}

fn track(m: &Modulation, group: Group) -> &Track {
    m.tracks
        .iter()
        .find(|t| t.group == group)
        .expect("every group")
}

fn within(bounds: Option<(f64, f64)>, x: f64) -> bool {
    bounds.is_some_and(|(lo, hi)| lo <= x && x <= hi)
}

/// Tremolo on 1320 Hz, its sidebands passed evenly, 0.73 to 19.9 Hz: truth inside the bounds.
#[test]
fn tremolo_across_rates_reads_rate_and_depth_inside_its_bounds() {
    for hz in [0.73, 4.0, 12.9, 19.9] {
        for depth in [0.2, 0.5] {
            let m = read(
                &tone(8.0, |t| {
                    (1.0 + depth * libm::sin(TAU * hz * t)) * libm::sin(TAU * 1320.0 * t)
                }),
                Some(120.0),
            );
            for w in &track(&m, Group::Mid).windows {
                let lead = w.peaks[0];
                if w.to_secs - w.from_secs < m.window_secs {
                    assert!(lead.hz_bounds.is_none() && lead.depth_bounds.is_none());
                    continue;
                }
                if w.from_secs < 0.5 || w.to_secs > 7.5 {
                    continue;
                }
                assert!(within(lead.hz_bounds, hz), "{hz} {depth}: {w:?}");
                assert!(
                    within(lead.period_beats_bounds, 2.0 / hz),
                    "{hz} {depth}: {w:?}"
                );
                assert!(within(lead.depth_bounds, depth), "{hz} {depth}: {w:?}");
                let (lo, hi) = lead.depth_bounds.expect("a whole window");
                assert!(depth > 0.3 || hi - lo < 0.01, "{hz} {depth}: {w:?}");
            }
        }
    }
}

/// Tremolo's Hann sidelobes read bins no main lobe gives, so they bound nothing.
#[test]
fn a_sidelobe_whose_bins_no_main_lobe_reads_is_unbounded() {
    let m = read(
        &tone(8.0, |t| {
            (1.0 + 0.5 * libm::sin(TAU * 4.0 * t)) * libm::sin(TAU * 1000.0 * t)
        }),
        None,
    );
    let mut sidelobes = 0;
    for w in &track(&m, Group::Mid).windows {
        for p in &w.peaks {
            if let Some((lo, hi)) = p.hz_bounds {
                assert!(lo < hi && lo <= p.hz && p.hz <= hi, "{p:?} in {w:?}");
            }
            if w.to_secs - w.from_secs == m.window_secs && (p.hz - 4.0).abs() > 0.3 {
                sidelobes += 1;
                assert!(p.hz_bounds.is_none() && p.depth_bounds.is_none(), "{p:?}");
            }
        }
    }
    assert!(sidelobes > 0);
}

#[test]
fn a_window_under_two_periods_and_a_target_under_the_window_refuse() {
    let samples = vec![0.0; (3.0 * RATE) as usize];
    assert_eq!(
        measure(&[&samples[..]], RATE, 0.0, 0.5, 0.05, None),
        Err(Unread::WindowUnderTwoPeriods {
            window_secs: 0.05,
            least_secs: 0.1
        })
    );
    assert_eq!(
        measure(&[&samples[..]], RATE, 0.0, 0.5, 4.0, None),
        Err(Unread::TargetShorterThanWindow {
            held_secs: 3.0,
            window_secs: 4.0,
            least_secs: 0.1
        })
    );
}
