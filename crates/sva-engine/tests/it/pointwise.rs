// Concern: proves a form no atom sum reaches still meets the grid, one instant at a time | Non-concern: which row a form that does reach one takes (sva-samples) | IO: (a composition) -> a Buffer

use std::f64::consts::TAU;

use crate::fixtures::graph_of;
use sva_engine::{Ask, Detail, Output, RenderConfig, Representation, Tier, answer, render};

/// A series has a value at an instant once it is truncated, and FORMAT 6.2 truncates it
/// once: the terms before its tail rounds away under the profile's precision.
#[test]
fn a_neumann_series_under_a_product_renders() {
    let g = graph_of(
        "neumann",
        &[
            ("src", "sin(2*pi*220*t)\n"),
            ("echo", "@src + 0.5*self(t - 0.01s)\n"),
            ("node", "@echo(t)*tanh(t)\n"),
        ],
    );
    let held = render(
        &g,
        "node",
        RenderConfig::seconds(44_100, 0.05),
        &Tier::default(),
    )
    .expect("a series under a product");
    let root = held.id("node").expect("the root");
    let buffer = held.output(root).expect("a point-sampled law");
    let precision = sva_samples::PSYCHOACOUSTIC_V1.half_lsb();
    let taken = (0..)
        .find(|k| 0.5f64.powi(*k) / 0.5 <= precision)
        .expect("a tail under the precision");
    for i in (0..buffer.len()).step_by(37) {
        let t = i as f64 / 44_100.0;
        let want: f64 = (0..taken)
            .map(|k| 0.5f64.powi(k) * libm::sin(TAU * 220.0 * (t - 0.01 * f64::from(k))))
            .sum::<f64>()
            * libm::tanh(t);
        assert!(
            (buffer.at(0, i) - want).abs() < 1e-7,
            "sample {i}: {} against {want}",
            buffer.at(0, i)
        );
    }
}

/// A width-2 closed form with no spectral sum is two lanes, and the pointwise path reads each of them
/// where the join puts it.
#[test]
fn a_panned_ct_law_point_samples_both_lanes() {
    let g = graph_of(
        "panned",
        &[
            ("src", "tanh(3*sin(2*pi*220*t))\n"),
            ("pan", "join(@src*cos(0.3), @src*sin(0.3))\n"),
        ],
    );
    let held = render(
        &g,
        "pan",
        RenderConfig::seconds(44_100, 0.01),
        &Tier::default(),
    )
    .expect("a joined law point-samples");
    let id = held.id("pan").expect("the root");
    let buffer = held.output(id).expect("a rendered pan");
    assert_eq!(buffer.width(), 2, "the join names two components");
    for i in [1usize, 17, 123] {
        let mono = libm::tanh(3.0 * libm::sin(TAU * 220.0 * i as f64 / 44_100.0));
        assert!(
            (buffer.at(0, i) - mono * libm::cos(0.3f64)).abs() < 1e-12,
            "left at {i}: {}",
            buffer.at(0, i)
        );
        assert!(
            (buffer.at(1, i) - mono * libm::sin(0.3f64)).abs() < 1e-12,
            "right at {i}: {}",
            buffer.at(1, i)
        );
    }
}

/// A crop that is shut zeroes the product it is a factor of, even where a factor beside it,
/// a decay read long before its onset, passes the largest double there.
#[test]
fn a_shut_crop_zeroes_a_factor_too_large_for_a_double() {
    let g = graph_of(
        "shut",
        &[(
            "gated",
            "crop(crop(tanh(t - 3s), 3s, inf)*exp(-(t - 3s)*400), 0s, 4s)\n",
        )],
    );
    let held = render(
        &g,
        "gated",
        RenderConfig::seconds(44_100, 4.0),
        &Tier::default(),
    )
    .expect("a shut crop is zero before its window");
    let samples = held
        .output(held.id("gated").expect("the root"))
        .expect("a buffer");
    let (before, after) = samples.plane(0).split_at(3 * 44_100);
    assert!(before.iter().all(|v| *v == 0.0));
    assert!(after.iter().all(|v| v.is_finite()) && after.iter().any(|v| *v != 0.0));
}

/// One node's samples over 50 ms at 8 kHz, beside `ramp`, a form no spectral sum reaches and
/// no reader inlines, as it reads a node of its own.
fn summed(name: &str, body: &str) -> Result<Vec<f64>, sva_engine::EngineError> {
    let root = format!("{body}\n");
    let g = graph_of(
        name,
        &[
            ("edge", "tanh((t - 0.02)/0.01)\n"),
            ("ramp", "650 + 360*@edge(t)\n"),
            ("root", &root),
        ],
    );
    let held = render(
        &g,
        "root",
        RenderConfig::seconds(8_000, 0.05),
        &Tier::default(),
    )?;
    let root = held.id("root").expect("the root");
    Ok(held.output(root).expect("samples").plane(0).to_vec())
}

#[test]
fn a_finite_sum_over_a_ref_is_its_terms_written_out() {
    let by_hand = |read: &str| {
        (1..=4)
            .map(|k| format!("{read}*sin(2*pi*{k}*165*t)/{k}"))
            .collect::<Vec<_>>()
            .join(" + ")
    };
    for read in ["@ramp(t)", "sample(@ramp(t))", "@edge(t)"] {
        let written = summed("sum-hand", &by_hand(read)).expect("terms written out");
        let sum = format!("sum(k, 1, 4, {read}*sin(2*pi*k*165*t)/k)");
        let got = summed("sum-ref", &sum).unwrap_or_else(|e| panic!("{sum}: {e}"));
        assert_eq!(got.len(), written.len());
        for (at, (g, w)) in got.iter().zip(&written).enumerate() {
            assert!(
                (g - w).abs() <= 1e-12 * w.abs().max(1.0),
                "{sum} at {at}: {g} vs {w}"
            );
        }
    }
}

#[test]
fn an_infinite_sum_over_a_ref_refuses_naming_a_finite_bound() {
    for (term, code) in [
        ("sample(@ramp(t))", "type.samples_in_series"),
        ("@ramp(t)", "read.no_spectral_sum"),
    ] {
        let sum = format!("sum(k, 1, inf, {term}*sin(2*pi*k*165*t)/(k*k))");
        let refused = summed("sum-inf", &sum).expect_err("no term count to stop at");
        let sva_engine::EngineError::Refused(d) = &refused else {
            panic!("{sum}: a located refusal, not {refused:?}");
        };
        assert_eq!(d.code, code, "{sum}");
        assert!(d.help.contains("finite upper bound"), "{sum}: {}", d.help);
        assert!(!d.help.contains("sample("), "{sum}: {}", d.help);
    }
}

/// The label is scored at the multiple the reading asked for, so both state one score.
#[test]
fn the_labels_alias_db_is_the_readings_score_at_its_multiple() {
    let g = graph_of(
        "labelled-twice",
        &[
            ("src", "2*sin(2*pi*3000*t)\n"),
            ("body", "sat(lowpass(@src, cutoff=8000, q=0.7))\n"),
        ],
    );
    let asked = Representation::Alias { oversample: 2 };
    let config = RenderConfig::seconds(44_100, 0.1).asking(vec![Ask {
        node: "body".to_string(),
        representation: asked,
    }]);
    let held = render(&g, "body", config, &Tier::default()).expect("a reading");
    let root = held.id("body").expect("the root");
    let Detail::Point { alias_db, .. } = held.labels[&root].detail else {
        panic!("a point sampling carries an alias score");
    };
    let Output::Alias(score) = answer(&held, root, asked).expect("a score").value else {
        panic!("an alias reading answers a score");
    };
    assert_eq!(alias_db, Some(score.asr_db));
}
