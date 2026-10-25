// Concern: proves a noise band renders exactly its lines lo*period..=hi*period, and what it refuses | Non-concern: a line's level (sva-formula noise.rs) | IO: (a composition) -> samples or a refusal

use std::f64::consts::TAU;

use crate::fixtures::{graph_of, samples};
use sva_engine::{RenderConfig, Tier, render};

const RATE: u32 = 48_000;

fn rendered(name: &str, body: &str) -> Vec<f64> {
    let g = graph_of(name, &[("body", body)]);
    let held = render(
        &g,
        "body",
        RenderConfig::seconds(RATE, 0.05),
        &Tier::default(),
    )
    .unwrap_or_else(|e| panic!("{body}: {e}"));
    samples(&held).remove("body").expect("the root's samples")
}

/// Line `k` of `noise(seed, period=T, color=c)` at sample `n`, as its closed form writes it.
fn line(seed: u64, period: f64, color: f64, k: i64, n: usize) -> f64 {
    let k = k as f64;
    let t = n as f64 / f64::from(RATE);
    let phase = sva_formula::draw_nearest(seed, k).expect("a whole key draws");
    libm::pow(k / period, libm::log2(color)) / period.sqrt()
        * libm::cos(TAU / period * k * t + TAU * phase)
}

/// Each band's edges sit on a line or between two; `lo=0` keeps the line at 0 Hz out. The
/// tolerance is the test's: a sum of up to 51 unit lines rounded twice, not the engine's.
#[test]
fn a_band_renders_exactly_its_lines() {
    let cut = libm::pow(10.0, -3.0 / 20.0);
    for (body, (seed, period, color), lines) in [
        (
            "noise(7, period=0.5, lo=100hz, hi=102hz)\n",
            (7, 0.5, 1.0),
            50..=51,
        ),
        (
            "noise(7, period=0.5, lo=99.9, hi=103.9)\n",
            (7, 0.5, 1.0),
            50..=51,
        ),
        ("noise(3, period=1, lo=0, hi=3)\n", (3, 1.0, 1.0), 1..=3),
        (
            "noise(5, period=0.25, color=-3db, lo=1khz, hi=1.2khz)\n",
            (5, 0.25, cut),
            250..=300,
        ),
    ] {
        let found = rendered("band", body);
        let worst = found
            .iter()
            .enumerate()
            .map(|(n, v)| {
                let want: f64 = lines.clone().map(|k| line(seed, period, color, k, n)).sum();
                (v - want).abs()
            })
            .fold(0.0, f64::max);
        assert!(worst < 1e-12, "{body}: off its lines by {worst}");
    }
}

/// `lo=0` and no `hi` is the band no edge writes: the same series, so the same bits.
#[test]
fn an_unbounded_band_renders_the_bits_of_no_band() {
    let plain = rendered("no-band", "noise(1, period=0.1, color=-3db)\n");
    let from_zero = rendered("from-zero", "noise(1, period=0.1, color=-3db, lo=0)\n");
    assert!(plain.iter().any(|v| *v != 0.0));
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&plain), bits(&from_zero));
}

#[test]
fn an_edge_no_band_holds_is_refused_by_its_own_code() {
    for (body, code) in [
        ("noise(1, period=1, lo=-1)\n", "engine.noise_band_negative"),
        ("noise(1, period=1, hi=-2)\n", "engine.noise_band_negative"),
        ("noise(1, period=1, hi=inf)\n", "engine.infinite_value"),
        (
            "noise(1, period=1, lo=inf, hi=5)\n",
            "engine.infinite_value",
        ),
        (
            "noise(1, period=1, lo=200, hi=200)\n",
            "engine.noise_band_inverted",
        ),
        (
            "noise(1, period=1, lo=300, hi=200)\n",
            "engine.noise_band_inverted",
        ),
        (
            "noise(1, period=1, lo=200.1, hi=200.9)\n",
            "engine.noise_band_empty",
        ),
        ("noise(1, period=1, hi=0.5)\n", "engine.noise_band_empty"),
        (
            "noise(1, period=1, lo=1e19, hi=2e19)\n",
            "engine.noise_band_past_index",
        ),
    ] {
        let g = graph_of("refused-band", &[("body", body)]);
        let Err(refused) = render(
            &g,
            "body",
            RenderConfig::seconds(RATE, 0.01),
            &Tier::default(),
        ) else {
            panic!("{body} names no band and renders");
        };
        assert_eq!(refused.code(), code, "{body}: {refused}");
        assert!(
            sva_engine::REGISTRY.iter().any(|(name, _)| *name == code),
            "`{code}` is in the vocabulary `builtins` prints"
        );
    }
}
