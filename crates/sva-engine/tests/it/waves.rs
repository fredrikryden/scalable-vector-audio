// Concern: proves a named waveform is a function of its fundamental's phase, delayed or modulated whole | Non-concern: a written sum's own series (refs.rs) | IO: (a composition) -> samples

use std::f64::consts::TAU;

use crate::fixtures::graph_of;
use sva_engine::{RenderConfig, Tier, render};

const HALF_LSB: f64 = 1.0 / (1u64 << 24) as f64;

fn plane(files: &[(&str, &str)], root: &str, secs: f64) -> Vec<f64> {
    let g = graph_of(root, files);
    let held = render(
        &g,
        root,
        RenderConfig::seconds(44_100, secs),
        &Tier::default(),
    )
    .expect("a waveform renders");
    let id = held.id(root).expect("the root");
    held.output(id).expect("its samples").plane(0).to_vec()
}

/// `pi/10` at 441 Hz delays the wave by exactly five steps at 44.1 kHz.
#[test]
fn a_constant_phase_delays_the_whole_wave() {
    for name in ["saw", "square", "triangle"] {
        let plain = plane(&[("plain", &format!("{name}(441)\n"))], "plain", 0.06);
        let phased = plane(
            &[("phased", &format!("{name}(441, pi/10)\n"))],
            "phased",
            0.05,
        );
        for (i, s) in phased.iter().enumerate() {
            let want = plain[i + 5];
            assert!(
                (s - want).abs() <= HALF_LSB,
                "{name} sample {i}: {s} against the plain wave's {want}"
            );
        }
    }
}

const VIBRATO: &str = "0.3*sin(2*pi*5*t)";

/// A moving phase is the plain wave read at a moving time, every harmonic under the band kept.
#[test]
fn a_moving_phase_is_the_plain_wave_read_at_a_moving_time() {
    let files = [
        ("plain", "saw(220)\n".to_string()),
        ("warped", format!("@plain(t + {VIBRATO}/(2*pi*220))\n")),
        ("vibrato", format!("saw(220, {VIBRATO})\n")),
    ];
    let files: Vec<(&str, &str)> = files.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let warped = plane(&files, "warped", 0.05);
    let vibrato = plane(&files, "vibrato", 0.05);
    for (i, (s, want)) in vibrato.iter().zip(&warped).enumerate() {
        assert!(
            (s - want).abs() <= HALF_LSB,
            "sample {i}: {s} against the warped read's {want}"
        );
    }
}

#[test]
fn a_saw_whose_pitch_moves_keeps_every_harmonic_under_the_ceiling() {
    let g = graph_of("moving", &[("moving", "saw(1000 + 200*t)\n")]);
    let config = RenderConfig::seconds(44_100, 0.05);
    let ceiling = config.profile.ceiling(44_100);
    let held = render(&g, "moving", config, &Tier::default()).expect("a moving saw renders");
    let id = held.id("moving").expect("the root");
    let samples = held.output(id).expect("its samples").plane(0).to_vec();
    for (i, s) in samples.iter().enumerate() {
        let t = i as f64 / 44_100.0;
        let (turns, hz) = ((1000.0 + 200.0 * t) * t, 1000.0 + 400.0 * t);
        let want: f64 = (1..)
            .take_while(|n| f64::from(*n) * hz < ceiling - 1e-6)
            .map(|n| libm::sin(TAU * f64::from(n) * turns) / f64::from(n))
            .sum::<f64>()
            * 2.0
            / std::f64::consts::PI;
        assert!((s - want).abs() < 1e-9, "sample {i}: {s} against {want}");
    }
    let sva_engine::Detail::Point { tail_db, .. } = held.labels[&id].detail else {
        panic!("a point row, not {:?}", held.labels[&id].detail);
    };
    assert!(tail_db.is_some(), "it states what it may drop");
}
