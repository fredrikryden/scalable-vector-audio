// Concern: what periodicity owes a steady train, a kick under hats, refusals | Non-concern: the band filters | IO: (samples) -> asserts

use sva_samples::measure::periodicity::{Group, Periodicity, Track, Window, measure};
use sva_samples::measure::sliding::Unread;

/// A 2 kHz envelope, so 0.5 s is exactly 1000 lags.
const RATE: f64 = 16_000.0;

fn burst(out: &mut [f64], at: usize, hz: f64, decay_secs: f64, secs: f64) {
    let attack = 0.002 * RATE;
    for (i, s) in out
        .iter_mut()
        .skip(at)
        .take((secs * RATE) as usize)
        .enumerate()
    {
        let t = i as f64 / RATE;
        let rise = (i as f64 / attack).min(1.0);
        *s += rise * sva_formula::exp(-t / decay_secs) * libm::sin(std::f64::consts::TAU * hz * t);
    }
}

/// A 2 ms 3 kHz burst every `period_secs`, placed between samples: exactly periodic.
fn pulses(secs: f64, period_secs: f64) -> Vec<f64> {
    (0..(secs * RATE) as usize)
        .map(|i| {
            let t = i as f64 / RATE;
            let since = t - (t / period_secs).floor() * period_secs;
            match since < 0.002 {
                true => {
                    sva_formula::exp(-since / 0.0005)
                        * libm::sin(std::f64::consts::TAU * 3000.0 * since)
                }
                false => 0.0,
            }
        })
        .collect()
}

fn within(bounds: Option<(f64, f64)>, x: f64) -> bool {
    bounds.is_some_and(|(lo, hi)| lo <= x && x <= hi)
}

fn read(samples: &[f64]) -> Periodicity {
    measure(&[samples], RATE, 0.0, 0.5, 4.0, false).expect("a 4 s window over a longer target")
}

fn track(p: &Periodicity, group: Group) -> &Track {
    p.tracks
        .iter()
        .find(|t| t.group == group)
        .expect("every group")
}

fn whole<'a>(p: &Periodicity, t: &'a Track) -> Vec<&'a Window> {
    t.windows
        .iter()
        .filter(|w| w.to_secs - w.from_secs == p.window_secs)
        .collect()
}

/// 128 bpm, 937.5 lags, reads 128.00 in every whole window, inside bounds under a lag wide.
#[test]
fn a_steady_128_bpm_train_reads_128_inside_its_bounds() {
    let p = read(&pulses(12.0, 60.0 / 128.0));
    let full = whole(&p, track(&p, Group::Broadband));
    assert!(full.len() > 10);
    for w in full {
        let lead = w.peaks[0];
        assert!((lead.bpm - 128.0).abs() < 0.005, "{w:?}");
        assert!(within(lead.bpm_bounds, 128.0), "{w:?}");
        assert!(within(lead.period_bounds_secs, 60.0 / 128.0), "{w:?}");
        let (lo, hi) = lead.period_bounds_secs.expect("a whole window");
        assert!(hi - lo <= p.lag_step_secs * (1.0 + 1e-9), "{w:?}");
    }
}

/// A 55 Hz kick on every beat under 5 kHz hats on every eighth: `low` leads at the beat,
/// `rest` at the eighth.
#[test]
fn a_kick_leads_the_low_band_and_hats_the_rest_at_double_rate() {
    let secs = 10.0;
    let mut out = vec![0.0; (secs * RATE) as usize];
    for k in 0..20 {
        burst(&mut out, k * 8000, 55.0, 0.08, 0.3);
    }
    for k in 0..40 {
        burst(&mut out, k * 4000, 5000.0, 0.02, 0.08);
    }
    let p = read(&out);
    let low = track(&p, Group::Low);
    assert!(low.centres_hz.iter().all(|&hz| hz < 150.0) && !low.centres_hz.is_empty());
    for (track, period) in [(low, 0.5), (track(&p, Group::Rest), 0.25)] {
        for w in whole(&p, track) {
            assert!(within(w.peaks[0].period_bounds_secs, period), "{w:?}");
        }
    }
}

#[test]
fn a_window_under_two_periods_and_a_target_under_the_window_refuse() {
    let samples = vec![0.0; (3.0 * RATE) as usize];
    assert_eq!(
        measure(&[&samples[..]], RATE, 0.0, 0.5, 0.1, false),
        Err(Unread::WindowUnderTwoPeriods {
            window_secs: 0.1,
            least_secs: 0.2
        })
    );
    assert_eq!(
        measure(&[&samples[..]], RATE, 0.0, 0.5, 4.0, false),
        Err(Unread::TargetShorterThanWindow {
            held_secs: 3.0,
            window_secs: 4.0,
            least_secs: 0.2
        })
    );
}
