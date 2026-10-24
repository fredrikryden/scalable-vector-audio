// Concern: proves what reading another node yields, per the representation it holds | Non-concern: what a node evaluates to | IO: (a composition) -> Read or a refusal

use crate::fixtures::graph_of;
use sva_engine::instantiate::resolve_ref_path;
use sva_engine::{EngineError, Held, RenderConfig, Tier, Var, render, types};

#[test]
fn a_walk_above_the_root_is_a_refusal_the_engine_can_locate() {
    assert_eq!(
        resolve_ref_path("piano/bench", "../../escape").expect_err("above the root"),
        EngineError::RefAboveRoot("piano/bench".to_string(), "../../escape".to_string()),
    );
}

/// A duration on a sampled ref is an index offset wherever it is a whole sample count at
/// the observation's rate, and a bar resolved against a tempo is such a duration.
#[test]
fn a_bar_offset_that_lands_on_the_grid_reads_a_sampled_ref() {
    let mut g = graph_of(
        "bar-offset",
        &[
            ("grid", "sample(sin(2*pi*220*t))\n"),
            ("node", "@grid(t - 1b)*0.5\n"),
            ("variables/bpm", "120\n"),
            ("variables/meter", "4/4\n"),
        ],
    );
    g.lower_reserved(None).expect("a tempo");
    let held = render(
        &g,
        "node",
        RenderConfig::seconds(8_000, 2.5),
        &Tier::default(),
    )
    .expect("a bar offset that lands on the grid");
    let id = held.id("node").expect("the root");
    let buffer = held.output(id).expect("a rendered read");
    let delay = 16_000;
    for i in 0..buffer.len() {
        let at = i as f64 - f64::from(delay);
        let want = 0.5 * libm::sin(std::f64::consts::TAU * 220.0 * at / 8_000.0);
        assert!(
            (buffer.at(0, i) - want).abs() < 1e-9,
            "sample {i}: {} against {want}",
            buffer.at(0, i)
        );
    }
}

#[test]
fn a_law_ref_tiled_by_modulo_point_samples() {
    let g = graph_of(
        "tiled",
        &[("src", "sin(2*pi*100*t)\n"), ("node", "@src(t % 0.0037)\n")],
    );
    let typing = types(&g, "node").expect("a tiled read types");
    let held = typing.id("node").expect("the root");
    let warped = typing.ty(held);
    assert_eq!(
        (warped.held, warped.dual),
        (Held::Form(Var::T), false),
        "a warped time leaves A"
    );

    let rendered = render(
        &g,
        "node",
        RenderConfig::seconds(8_000, 0.05),
        &Tier::default(),
    )
    .expect("a tiled read");
    let root = rendered.id("node").expect("the root");
    let buffer = rendered.output(root).expect("a point-sampled law");
    for i in 0..buffer.len() {
        let tiled = (i as i64 * 10_000) % (37 * 8_000);
        let want = libm::sin(std::f64::consts::TAU * 100.0 * (tiled as f64 / 80_000_000.0));
        assert!(
            (buffer.at(0, i) - want).abs() < 1e-9,
            "sample {i}: {} against {want}",
            buffer.at(0, i)
        );
    }
}

/// A time written one per component is a per-lane substitution: each lane of the callee is
/// read at the time its own lane names.
#[test]
fn a_per_channel_shift_on_a_law_substitutes_per_lane() {
    let g = graph_of(
        "per-lane",
        &[
            ("wide", "join(sin(2*pi*220*t), sin(2*pi*330*t))\n"),
            ("taps", "@wide(join(t - 0.01s, t - 0.02s))\n"),
        ],
    );
    let held = render(
        &g,
        "taps",
        RenderConfig::seconds(44_100, 0.05),
        &Tier::default(),
    )
    .expect("a law read once per component");
    let buffer = held
        .output(held.id("taps").expect("the root"))
        .expect("two lanes");
    assert_eq!(buffer.width(), 2);
    for i in [0usize, 441, 1000] {
        let t = i as f64 / 44_100.0;
        let left = libm::sin(std::f64::consts::TAU * 220.0 * (t - 0.01));
        let right = libm::sin(std::f64::consts::TAU * 330.0 * (t - 0.02));
        assert!(
            (buffer.at(0, i) - left).abs() < 1e-12,
            "left at {i}: {} against {left}",
            buffer.at(0, i)
        );
        assert!(
            (buffer.at(1, i) - right).abs() < 1e-12,
            "right at {i}: {} against {right}",
            buffer.at(1, i)
        );
    }
}

/// The first `n` samples of `body` at 100 Hz, where `form` is the ramp `t` as a closed form
/// and `held` the same ramp as samples.
fn first(name: &str, body: &str, n: usize) -> Result<Vec<f64>, EngineError> {
    let root = format!("{body}\n");
    let g = graph_of(
        name,
        &[("form", "t\n"), ("held", "sample(t)\n"), ("root", &root)],
    );
    let held = render(
        &g,
        "root",
        RenderConfig::seconds(100, n as f64 / 100.0),
        &Tier::default(),
    )?;
    let root = held.id("root").expect("the root");
    Ok(held.output(root).expect("samples").plane(0).to_vec())
}

fn near(got: &[f64], want: &[f64]) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-12)
}

/// A reflected read keeps its sign on either representation: `0.11s - t` counts down.
#[test]
fn a_reflected_read_counts_down_on_a_form_and_on_samples() {
    for (name, body) in [
        ("form-rev", "@form(0.11s - t)"),
        ("held-rev", "@held(0.11s - t)"),
        ("held-neg", "@held(-1*(t - 0.11s))"),
        ("held-sp", "@held(11sp - t)"),
    ] {
        let got = first(name, body, 3).expect("a reflected read renders");
        assert!(near(&got, &[0.11, 0.10, 0.09]), "{body}: {got:?}");
    }
}

/// Samples of a closed form read at any multiple of `t` land where that multiple says: on
/// samples where it is whole there, the closed form itself between them.
#[test]
fn a_scaled_read_of_samples_lands_on_or_between_samples() {
    let got = first("held-2t", "@held(2*t - 0.02s)", 3).expect("a whole scale renders");
    assert!(near(&got, &[-0.02, 0.0, 0.02]), "{got:?}");
    let got = first("held-half", "@held(0.5*t)", 3).expect("half a sample renders");
    let want = [0.0, 0.005, 0.01];
    assert!(
        got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-9),
        "{got:?}"
    );
}

/// Seconds and steps of the rate in use add in one offset.
#[test]
fn an_offset_in_seconds_and_steps_reads_their_sum() {
    let got = first("mixed", "@held(t - 0.02s - 1sp)", 3).expect("a mixed offset renders");
    let step = 1.0 / 100.0;
    assert!(near(&got, &[-0.02 - step, -0.01 - step, -step]), "{got:?}");
}

/// A `sum` binds its index over its own term, so an inner sum naming the same index reads its
/// own: `sum(k, 1, 2, sum(k, 1, 3, k))` is 2*(1 + 2 + 3), never 3*(1 + 2).
#[test]
fn an_inner_sum_reads_its_own_index_where_an_outer_one_shares_its_name() {
    let peak = |body: &str| {
        let g = graph_of("shadowed", &[("root", body)]);
        let held = render(
            &g,
            "root",
            RenderConfig::seconds(8_000, 0.01),
            &Tier::default(),
        )
        .unwrap_or_else(|e| panic!("{body}: {e}"));
        let root = held.id("root").expect("the root");
        let samples = held.output(root).expect("samples").plane(0).to_vec();
        samples.iter().fold(0f64, |m, s| m.max(s.abs()))
    };
    let shadowed = peak("sum(k, 1, 2, sum(k, 1, 3, k))*sin(2*pi*100*t)\n");
    let distinct = peak("sum(j, 1, 2, sum(k, 1, 3, k))*sin(2*pi*100*t)\n");
    assert!((distinct - 12.0).abs() < 1e-3, "{distinct}");
    assert_eq!(shadowed, distinct, "the inner index is the inner sum's own");
}
