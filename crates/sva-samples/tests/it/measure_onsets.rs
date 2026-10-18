// Concern: states what onset detection owes a click track, silence and one long transient | Non-concern: the spectrum under the flux (measure_spectrum.rs) | IO: (samples) -> asserted onsets

use sva_samples::measure::onsets::detect;

fn click_track(sr: f64, secs: f64, gap_secs: f64) -> Vec<f64> {
    let n = (secs * sr) as usize;
    let gap = (gap_secs * sr) as usize;
    let mut out = vec![0.0; n];
    let mut at = 0;
    while at + 8 < n {
        for (i, s) in out[at..at + 8].iter_mut().enumerate() {
            *s = (1.0 - i as f64 / 8.0) * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        at += gap;
    }
    out
}

#[test]
fn evenly_spaced_clicks_are_found_at_roughly_their_own_spacing() {
    let (sr, gap) = (44_100.0, 0.25);
    let found = detect(&[&click_track(sr, 4.0, gap)], sr, 0.0);
    assert!(found.onsets.len() >= 12, "{}", found.onsets.len());
    for w in found.onsets.windows(2) {
        let ioi = w[1].t_secs - w[0].t_secs;
        assert!((ioi - gap).abs() < 0.03, "ioi {ioi} far from {gap}");
    }
}

#[test]
fn an_onset_is_placed_on_the_buffers_own_clock() {
    let sr = 8_000.0;
    let from_zero = detect(&[&click_track(sr, 1.0, 0.25)], sr, 0.0);
    let later = detect(&[&click_track(sr, 1.0, 0.25)], sr, 2.0);
    let shifted: Vec<f64> = later.onsets.iter().map(|o| o.t_secs - 2.0).collect();
    let own: Vec<f64> = from_zero.onsets.iter().map(|o| o.t_secs).collect();
    assert!(!own.is_empty());
    for (a, b) in own.iter().zip(&shifted) {
        assert!((a - b).abs() < 1e-9, "{own:?} vs {shifted:?}");
    }
}

#[test]
fn silence_holds_no_onsets() {
    let found = detect(&[&vec![0.0; 44_100]], 44_100.0, 0.0);
    assert!(found.onsets.is_empty());
    assert!(found.ioi_histogram.iter().all(|b| b.count == 0));
}

#[test]
fn a_close_double_trigger_on_one_transient_is_suppressed() {
    let sr = 44_100.0;
    let mut samples = vec![0.0; (sr * 0.2) as usize];
    for (i, s) in samples.iter_mut().enumerate().take(200) {
        *s = (1.0 - i as f64 / 200.0) * if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let found = detect(&[&samples], sr, 0.0);
    assert!(found.onsets.len() <= 1, "{:?}", found.onsets);
}

/// Strikes of a 440 Hz burst rising in 1.5 ms and dying in 12 ms, `gap` seconds apart, each
/// landing on the tail of the one before.
fn bursts(sr: f64, gap: f64, count: usize) -> Vec<f64> {
    let mut out = vec![0.0; ((gap * count as f64 + 0.2) * sr) as usize];
    for k in 0..count {
        let from = (k as f64 * gap * sr).round() as usize;
        for (n, s) in out[from..].iter_mut().enumerate() {
            let t = n as f64 / sr;
            let shape = (1.0 - (-t / 0.0015).exp()) * (-t / 0.012).exp();
            *s += 0.7 * shape * (std::f64::consts::TAU * 440.0 * t).sin();
        }
    }
    out
}

/// A roll's thirty-seconds at 150 bpm are 50 ms apart and at 300 bpm 25 ms: every strike is
/// its own onset, within a millisecond of where it struck.
#[test]
fn strikes_down_to_25_ms_apart_are_each_their_own_onset() {
    let sr = 44_100.0;
    for gap in [0.05, 1.0 / 30.0, 0.025] {
        let found = detect(&[&bursts(sr, gap, 32)], sr, 0.0);
        let at: Vec<f64> = found.onsets.iter().map(|o| o.t_secs).collect();
        assert_eq!(at.len(), 32, "{gap} s apart: {at:?}");
        for (k, heard) in at.iter().enumerate() {
            let struck = k as f64 * gap;
            assert!(
                (heard - struck).abs() <= 0.001,
                "{gap} s apart: strike {k} at {struck} heard at {heard}"
            );
        }
    }
}

/// Every channel is read: strikes in the right channel alone are found where a mono track of
/// them is, and the same strikes inverted in the other channel cancel nothing.
#[test]
fn a_stereo_track_is_read_in_every_channel() {
    let (sr, gap) = (44_100.0, 0.25);
    let clicks = click_track(sr, 2.0, gap);
    let silent = vec![0.0; clicks.len()];
    let inverted: Vec<f64> = clicks.iter().map(|s| -s).collect();
    let at = |planes: &[&[f64]]| -> Vec<f64> {
        detect(planes, sr, 0.0)
            .onsets
            .iter()
            .map(|o| o.t_secs)
            .collect()
    };
    let mono = at(&[&clicks]);
    assert!(mono.len() >= 6, "{mono:?}");
    assert_eq!(
        at(&[&silent, &clicks]),
        mono,
        "strikes in the right channel alone"
    );
    assert_eq!(
        at(&[&clicks, &inverted]),
        mono,
        "opposite phases sum no silence"
    );
}

/// 0.3 s is 13230 samples: every interval lands in the bucket opening there.
#[test]
fn intervals_on_a_bucket_edge_fall_in_the_bucket_it_opens() {
    let sr = 44_100.0;
    let found = detect(&[&bursts(sr, 0.3, 6)], sr, 0.0);
    assert_eq!(found.onsets.len(), 6, "{:?}", found.onsets);
    let filled: Vec<(f64, f64, usize)> = (found.ioi_histogram.iter())
        .filter(|b| b.count > 0)
        .map(|b| (b.lo_secs, b.hi_secs, b.count))
        .collect();
    assert_eq!(filled, vec![(0.3, 0.325, 5)]);
}

#[test]
fn the_resolution_is_the_hop_in_whole_samples() {
    for (sr, hop) in [(44_100.0, 127.0), (48_000.0, 138.0), (8_000.0, 23.0)] {
        let found = detect(&[&bursts(sr, 0.25, 2)], sr, 0.0);
        assert_eq!(found.resolution_secs, hop / sr, "{sr} Hz");
    }
}
