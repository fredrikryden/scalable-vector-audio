// Concern: proves each node is sampled at the render's rate, at the instants its reader asks for, or refuses | Non-concern: a stream's blocks (stream.rs) | IO: (a composition, rate) -> samples
use std::f64::consts::TAU;

use crate::fixtures::graph_of;
use sva_engine::{Ask, RenderConfig, Representation, Tier, render};

const RATES: [u32; 4] = [8_000, 44_100, 48_000, 96_000];

fn composition() -> sva_ast::Graph {
    graph_of(
        "grid",
        &[
            ("tone", "sin(2*pi*220*t)\n"),
            ("lfo", "lowpass(sample(0.001*sin(2*pi*3*t)), cutoff=100)\n"),
            ("vibrato", "@tone(t + @lfo)\n"),
            ("scaled", "@tone(0.37*t - 0.0123456s)\n"),
            ("decay", "crop(1, 0s, 1sp) + 0.5*self[idx(t) - 1]\n"),
            (
                "mix",
                "lowpass(sample(0.3*saw(220*t)), cutoff=900) + @vibrato + rand(t - t % 0.01s, \
                 seed=3) + @decay\n",
            ),
        ],
    )
}

fn plane(held: &sva_engine::Render, node: &str) -> Vec<f64> {
    let id = held.id(node).unwrap_or_else(|| panic!("{node} is held"));
    held.output(id).expect("a buffer").plane(0).to_vec()
}

/// An edge written in `sp` counts samples, whatever decimal their instant prints as: at
/// 22.05 and 44.1 kHz `1/rate` prints past sample 1's instant, and `1sp` is still one sample.
#[test]
fn a_crop_edge_in_samples_holds_that_many() {
    for count in [1, 3, 7] {
        let g = graph_of(
            "grid-crop-samples",
            &[
                ("form", &format!("crop(1, 0s, {count}sp)\n")),
                (
                    "value",
                    &format!("crop(sample(cos(2*pi*10*t)), 0s, {count}sp)\n"),
                ),
            ],
        );
        for rate in [8_000, 22_050, 44_100, 48_000, 96_000] {
            for node in ["form", "value"] {
                let held = render(&g, node, RenderConfig::at(rate), &Tier::default())
                    .unwrap_or_else(|e| panic!("{rate}: {e}"));
                assert_eq!(
                    plane(&held, node).len(),
                    count,
                    "{node} {count}sp at {rate}"
                );
            }
        }
    }
}

/// At a scaled, shifted time, and at a time a filter moves.
#[test]
fn a_formula_read_at_any_instant_is_exact() {
    let g = composition();
    for rate in RATES {
        let config = RenderConfig::seconds(rate, 0.02);
        let scaled = render(&g, "scaled", config.clone(), &Tier::default()).expect("a scaled read");
        for (n, v) in plane(&scaled, "scaled").iter().enumerate() {
            let at = 0.37 * n as f64 / f64::from(rate) - 0.012_345_6;
            assert!(
                (v - libm::sin(TAU * 220.0 * at)).abs() < 1e-9,
                "{rate}: {n}"
            );
        }
        // A render holds past its pass only the buffers a reading asks for.
        let config = RenderConfig {
            asks: vec![Ask {
                node: "lfo".to_string(),
                representation: Representation::Samples,
            }],
            ..config
        };
        let held = render(&g, "vibrato", config, &Tier::default()).expect("a moving read");
        let lfo = held
            .buffers
            .get(&held.id("lfo").expect("the lfo"))
            .expect("a filter is held")
            .plane(0)
            .to_vec();
        for (n, v) in plane(&held, "vibrato").iter().enumerate() {
            let at = n as f64 / f64::from(rate) + lfo[n];
            assert!(
                (v - libm::sin(TAU * 220.0 * at)).abs() < 1e-12,
                "{rate}: {n}"
            );
        }
    }
}

/// A form read through a warp whose instant is one of the form's own samples reads that
/// sample's bits, whichever row the form takes: one arithmetic per form.
#[test]
fn a_warp_landing_on_a_sample_reads_its_bits() {
    let forms = [
        ("lines", "sin(2*pi*220*t) + 0.3*cos(2*pi*331*t)"),
        (
            "grouped",
            "exp(-1000*(t - 0.02)*(t - 0.02))*sin(2*pi*220*t)",
        ),
        (
            "swept",
            "exp(-30*t)*sin(2*pi*220*t) + exp(-7*t)*cos(2*pi*97*t)",
        ),
        ("pointed", "tanh(3*sin(2*pi*220*t))"),
        ("added", "tanh(3*sin(2*pi*220*t)) + sin(2*pi*110*t)"),
        ("cropped", "crop(sin(2*pi*440*t), 60/72*0.01s, 0.03s)"),
    ];
    for (name, form) in forms {
        let g = graph_of(
            "grid-landing",
            &[("x", &format!("{form}\n")), ("y", "@x(max(2*t, 0s))\n")],
        );
        for rate in RATES {
            let whole = render(&g, "x", RenderConfig::seconds(rate, 0.04), &Tier::default())
                .unwrap_or_else(|e| panic!("{name} at {rate}: {e}"));
            let read = render(&g, "y", RenderConfig::seconds(rate, 0.02), &Tier::default())
                .unwrap_or_else(|e| panic!("{name} at {rate}: {e}"));
            let (whole, read) = (plane(&whole, "x"), plane(&read, "y"));
            assert!(
                read.iter().any(|v| *v != 0.0),
                "{name}: silence tests nothing"
            );
            for (n, v) in read.iter().enumerate() {
                assert_eq!(
                    v.to_bits(),
                    whole[2 * n].to_bits(),
                    "{name} at {rate}: sample {n} reads {v}, its landing holds {}",
                    whole[2 * n]
                );
            }
        }
    }
}

/// `@kick(t - d)` reads the kick's one value `d` later, rounded to the nearest sample: the
/// kick's own samples moved by that many, bit for bit, at every rate.
#[test]
fn a_stateful_node_read_between_its_steps_reads_the_nearest_whole_sample() {
    let g = graph_of(
        "shifted",
        &[
            (
                "kick",
                "lowpass(crop(sample(sin(2*pi*55*t)), 0s, 0.05s), cutoff=900, q=0.8)\n",
            ),
            ("late", "@kick(t - 0.1234s)\n"),
        ],
    );
    for (rate, moved) in [
        (8_000, 987),
        (44_100, 5_442),
        (48_000, 5_923),
        (96_000, 11_846),
    ] {
        let config = RenderConfig::seconds(rate, 0.3);
        let at = |target: &str| {
            let held = render(&g, target, config.clone(), &Tier::default())
                .unwrap_or_else(|e| panic!("{target} at {rate}: {e}"));
            plane(&held, target)
        };
        let (kick, late) = (at("kick"), at("late"));
        assert!(
            late.iter().any(|v| *v != 0.0),
            "{rate}: silence tests nothing"
        );
        for (n, v) in late.iter().enumerate() {
            let want = n.checked_sub(moved).map_or(0.0, |k| kick[k]);
            assert_eq!(v.to_bits(), want.to_bits(), "{rate}: sample {n}");
        }
    }
}

/// A stateful node at a time that moves has no step to read there; typing says which
/// construct holds the state and offers the nearest step.
#[test]
fn a_stateful_node_read_at_a_moving_time_refuses_at_typing() {
    let g = graph_of(
        "warped",
        &[
            ("pad", "lowpass(sample(0.3*saw(220*t)), cutoff=900)\n"),
            ("d", "0.001*sin(2*pi*3*t)\n"),
            ("warped", "@pad(t - @d)\n"),
        ],
    );
    let Err(refused) = sva_engine::types(&g, "warped") else {
        panic!("a filter has no value at a time that moves");
    };
    assert_eq!(refused.code(), "type.stateful_warp", "{refused}");
    let said = refused.to_string();
    for part in ["`pad`", "`lowpass(…)`", "idx("] {
        assert!(said.contains(part), "{part} in {said}");
    }
}

#[test]
fn a_store_answers_a_stepped_node_only_at_its_own_rate() {
    let g = composition();
    let cache = Tier::default();
    let at = |rate: u32, cache: &Tier| {
        let config = RenderConfig::seconds(rate, 0.05);
        plane(&render(&g, "mix", config, cache).expect("a render"), "mix")
    };
    at(44_100, &cache);
    assert_eq!(at(48_000, &cache), at(48_000, &Tier::default()));
}
