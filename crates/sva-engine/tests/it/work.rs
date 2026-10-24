// Concern: proves a cropped line series writes its own runs inside its window and keeps its lines | Non-concern: what any sample holds (stream.rs) | IO: (a composition) -> Work

use crate::fixtures::graph_of;
use sva_engine::{RenderConfig, Tier, render};

const RATE: u32 = 44_100;

/// A hard crop of a line series sums the series' own runs inside its window: the same bits
/// as the series uncropped there.
#[test]
fn a_cropped_line_series_writes_its_own_runs_inside_its_window() {
    let g = graph_of(
        "cropped-series",
        &[
            ("breath", "noise(7, period=0.04321s)\n"),
            ("cut", "crop(noise(7, period=0.04321s), 0s, 0.05s)\n"),
        ],
    );
    let samples = RATE as usize / 20;
    let held = |target: &str| {
        let r = render(
            &g,
            target,
            RenderConfig::seconds(RATE, 0.05),
            &Tier::default(),
        )
        .expect("a render");
        r.output(r.root).expect("the root").plane(0).to_vec()
    };
    let (whole, cut) = (held("breath"), held("cut"));
    assert_eq!(cut.len(), samples);
    for (n, (a, b)) in cut.iter().zip(&whole).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "sample {n}");
    }
}

#[test]
fn a_cropped_series_keeps_the_uncropped_lines_at_the_ceiling() {
    let g = graph_of(
        "ceiling-line",
        &[
            ("breath", "noise(7, period=0.05s)\n"),
            ("cut", "crop(noise(7, period=0.05s), 0s, 0.05s)\n"),
        ],
    );
    let held = |target: &str| {
        let r = render(
            &g,
            target,
            RenderConfig::seconds(RATE, 0.05),
            &Tier::default(),
        )
        .expect("a render");
        r.output(r.root).expect("the root").plane(0).to_vec()
    };
    let (whole, cut) = (held("breath"), held("cut"));
    for (n, (a, b)) in cut.iter().zip(&whole).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "sample {n}");
    }
}
