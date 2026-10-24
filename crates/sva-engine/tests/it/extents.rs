// Concern: proves every node is computed over its own support met with its readers' demand, whatever reads it | Non-concern: where a range ends (stream.rs) | IO: (a composition, a range) -> samples

use crate::fixtures::graph_of;
use sva_ast::Graph;
use sva_engine::{Range, RenderConfig, Tier, render};

const RATE: u32 = 8_000;

fn composition() -> Graph {
    graph_of(
        "extents",
        &[
            ("tone", "0.5*sin(2*pi*440*t)\n"),
            ("held", "sample(@tone)\n"),
            ("short", "sample(crop(@tone, 0s, 0.5s))\n"),
            ("late", "@short(t - 1s)\n"),
            ("early", "@short(t + 0.25s)\n"),
            ("heard_late", "@held(t - 0.5s)\n"),
            ("said_late", "@tone(t - 0.5s)\n"),
            (
                "echo",
                "sample(crop(0.5*sin(2*pi*440*t), 0s, 0.2s)) + 0.5*self[idx(t - 0.25s)]\n",
            ),
            (
                "ring",
                "lowpass(sample(crop(0.5*sin(2*pi*220*t), 0s, 0.05s)), 220, q=30)\n",
            ),
            ("whole", "istft(stft(@held, window=256sp, hop=64sp))\n"),
            ("smoothed", "lp(crop(4, 0s, 2s), cutoff=8)\n"),
            ("steps", "0.5 + 0.5*rand(t - t % 0.125, seed=17)\n"),
            (
                "drift",
                "0.5*(1 - cos(2*pi*0)) + 0.5*lp(sample(@steps), cutoff=55)\n",
            ),
            ("drift_late", "@drift(t - 0.5s)\n"),
        ],
    )
}

fn over(g: &Graph, target: &str, start: i64, end: i64) -> Vec<f64> {
    let config = RenderConfig {
        range: Range {
            start: Some(start),
            end: Some(end),
        },
        ..RenderConfig::at(RATE)
    };
    let held =
        render(g, target, config, &Tier::default()).unwrap_or_else(|e| panic!("{target}: {e}"));
    held.output(held.root).expect("the root").plane(0).to_vec()
}

fn secs(n: f64) -> i64 {
    (n * f64::from(RATE)) as i64
}

/// Each read takes the samples its source holds there, never silence past a window.
#[test]
fn a_shifted_ref_reads_its_source_where_that_source_sounds() {
    let g = composition();
    let short = over(&g, "short", 0, secs(0.5));
    assert!(short.iter().any(|v| *v != 0.0), "silence tests nothing");
    let late = over(&g, "late", 0, secs(2.0));
    let before = &late[..secs(1.0) as usize];
    assert!(before.iter().all(|v| *v == 0.0), "silent before it starts");
    let heard = &late[secs(1.0) as usize..secs(1.5) as usize];
    assert_eq!(heard, &short[..], "a second late, the source's own samples");
    let after = &late[secs(1.5) as usize..];
    assert!(after.iter().all(|v| *v == 0.0), "silent after it ends");
    let early = over(&g, "early", 0, secs(0.5));
    let ahead = &early[..secs(0.25) as usize];
    assert_eq!(
        ahead,
        &short[secs(0.25) as usize..],
        "a read ahead reads ahead"
    );
}

#[test]
fn a_start_past_zero_keeps_the_state_its_history_left() {
    let g = composition();
    for target in ["echo", "ring"] {
        let whole = over(&g, target, 0, secs(1.0));
        let late = over(&g, target, secs(0.3), secs(1.0));
        assert!(late.iter().any(|v| *v != 0.0), "{target}: silence");
        assert_eq!(late[..], whole[secs(0.3) as usize..], "{target}");
    }
}

/// A short-time transform reads its input whole, so an input with no end refuses.
#[test]
fn a_transform_of_an_endless_input_refuses() {
    let config = RenderConfig::seconds(RATE, 0.1);
    let Err(refused) = render(&composition(), "whole", config, &Tier::default()) else {
        panic!("a transform of an endless input rendered");
    };
    assert_eq!(refused.code(), "engine.unbounded_extent", "{refused}");
}

#[test]
fn a_crop_starts_on_the_first_sample_past_its_edge_on_every_route() {
    let g = graph_of(
        "edge",
        &[
            (
                "rows",
                "crop(sin(2*pi*440*t), 0.8333333333333334s, 2s) + 0.25\n",
            ),
            (
                "program",
                "crop(sample(sin(2*pi*440*t)), 0.8333333333333334s, 2s) + 0.25\n",
            ),
            ("read", "@rows\n"),
        ],
    );
    let at = |target: &str| {
        let config = RenderConfig {
            range: Range {
                start: Some(39_990),
                end: Some(40_010),
            },
            ..RenderConfig::at(48_000)
        };
        let held = render(&g, target, config, &Tier::default())
            .unwrap_or_else(|e| panic!("{target}: {e}"));
        held.output(held.root).expect("the root").plane(0).to_vec()
    };
    for target in ["rows", "program", "read"] {
        let samples = at(target);
        assert_eq!(
            samples[10], 0.25,
            "{target}: sample 40000 is before the edge"
        );
        assert_ne!(samples[11], 0.25, "{target}: sample 40001 is past it");
    }
}

/// A step at 100 Hz in a form in `f` is a band edge, never a window from 100 s on: its inverse
/// sounds from the first sample.
#[test]
fn a_form_in_f_reads_its_band_edges_as_hertz() {
    let g = graph_of(
        "band",
        &[
            ("band", "step(f - 100)*exp(0 - f/300)\n"),
            ("heard", "sample(@band)\n"),
        ],
    );
    let samples = over(&g, "heard", 0, secs(0.01));
    assert_ne!(samples[0], 0.0, "the band's inverse sounds at t = 0");
}
