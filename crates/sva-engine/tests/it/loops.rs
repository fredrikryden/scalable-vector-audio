// Concern: proves a loop is a continuous series or a discrete loop reading itself by index | Non-concern: running a recurrence (sva-samples) | IO: (a composition) -> Ty, samples or a refusal

use std::f64::consts::TAU;

use crate::fixtures::graph_of;
use sva_engine::{
    Ask, Detail, EngineError, Held, RenderConfig, Representation, Source, Tier, render, types,
};

fn form(name: &str, body: &str) -> Result<Held, EngineError> {
    let g = graph_of(name, &[("loop", body)]);
    let typing = types(&g, "loop")?;
    let id = typing.id("loop").expect("the root");
    Ok(typing.ty(id).held)
}

#[test]
fn unit_gain_refuses() {
    let refused = form("unit", "sin(2*pi*220*t) + self(t - 0.01s)\n").expect_err("no value");
    let EngineError::Refused(d) = &refused else {
        panic!("expected a written refusal, got {refused:?}");
    };
    assert_eq!(d.code, "type.self_gain_unbounded");
    assert!(
        d.help.contains("self[idx("),
        "the discrete loop is offered: {}",
        d.help
    );

    let above = form("above", "sin(2*pi*220*t) + 1.5*self(t - 0.01s)\n").expect_err("no value");
    assert_eq!(above.code(), "type.self_gain_unbounded");
}

const RATES: [u32; 4] = [8_000, 44_100, 48_000, 96_000];

fn at_rate(files: &[(&str, &str)], root: &str, rate: u32, secs: f64) -> sva_engine::Render {
    let g = graph_of("rated", files);
    render(
        &g,
        root,
        RenderConfig::seconds(rate, secs),
        &Tier::default(),
    )
    .unwrap_or_else(|e| panic!("{root} at {rate}: {e}"))
}

/// Over a formula, `x + 0.5*self(t - 17ms)` is its delay-equation series, exact at every
/// rate whether or not 17 ms is a whole number of its samples.
#[test]
fn a_continuous_loop_renders_its_series_exactly_at_any_rate() {
    let x = |t: f64| match (0.0..0.005).contains(&t) {
        true => libm::sin(TAU * 200.0 * t),
        false => 0.0,
    };
    for rate in RATES {
        let held = at_rate(
            &[(
                "loop",
                "crop(sin(2*pi*200*t), 0s, 0.005s) + 0.5*self(t - 17ms)\n",
            )],
            "loop",
            rate,
            0.06,
        );
        let id = held.id("loop").expect("the root");
        for (n, v) in held.output(id).expect("a comb").plane(0).iter().enumerate() {
            let t = n as f64 / f64::from(rate);
            let want: f64 = (0..4)
                .map(|k| 0.5f64.powi(k) * x(t - 0.017 * f64::from(k)))
                .sum();
            assert!(
                (v - want).abs() < 1e-12,
                "{rate}, sample {n}: {v} against {want}"
            );
        }
    }
}

#[test]
fn a_karplus_strong_loop_with_an_allpass_fraction_is_its_recurrence() {
    const RATE: u32 = 44_100;
    const N: usize = 99;
    let fraction = f64::from(RATE) / 440.0 - 0.5 - N as f64;
    let c = (1.0 - fraction) / (1.0 + fraction);
    let lp = 0.498;
    let g = graph_of(
        "rated",
        &[
            ("burst", "crop(sample(rand(t, seed=7)), 0s, 0.002s)\n"),
            (
                "string",
                &format!(
                    "@burst + {c}*{lp}*(self[idx(t) - 99] + self[idx(t) - 100]) + \
                     {lp}*(self[idx(t) - 100] + self[idx(t) - 101]) - \
                     {c}*(self[idx(t) - 1] - @burst[idx(t) - 1])\n"
                ),
            ),
        ],
    );
    // A render holds past its pass only the buffers a reading asks for.
    let config = RenderConfig {
        asks: vec![Ask {
            node: "burst".to_string(),
            representation: Representation::Samples,
        }],
        ..RenderConfig::seconds(RATE, 0.05)
    };
    let held = render(&g, "string", config, &Tier::default()).unwrap_or_else(|e| panic!("{e}"));
    let plane = |node: &str| {
        held.output(held.id(node).expect("held"))
            .expect("a buffer")
            .plane(0)
            .to_vec()
    };
    let (x, y) = (plane("burst"), plane("string"));
    let at = |v: &[f64], n: usize, back: usize| n.checked_sub(back).map_or(0.0, |i| v[i]);
    let mut want = vec![0.0; y.len()];
    for n in 0..y.len() {
        want[n] = x[n]
            + c * lp * (at(&want, n, N) + at(&want, n, N + 1))
            + lp * (at(&want, n, N + 1) + at(&want, n, N + 2))
            - c * (at(&want, n, 1) - at(&x, n, 1));
    }
    assert!(
        y[N * 3..].iter().any(|v| v.abs() > 1e-3),
        "the string rings"
    );
    for (n, (v, w)) in y.iter().zip(&want).enumerate() {
        assert!((v - w).abs() < 1e-9, "sample {n}: {v} against {w}");
    }
}

/// A loop reading the sample it is writing has nothing to read.
#[test]
fn a_zero_delay_loop_refuses() {
    let refused = form("zero-delay", "sin(2*pi*220*t) + 0.5*self(t)\n").expect_err("no value");
    assert_eq!(refused.code(), "samples.zero_delay_loop");
}

/// Reading ahead is not the same mistake as reading the sample being written.
#[test]
fn a_forward_self_read_names_its_own_cause() {
    let refused = form("forward", "sin(2*pi*220*t) + 0.5*self(t + 0.01s)\n").expect_err("no value");
    assert_eq!(refused.code(), "engine.forward_self_read");
}

/// The comb the series denotes is one line carrying every term, and the label says how much
/// of the tail the profile's precision left behind.
#[test]
fn a_closed_loop_renders_its_comb_and_reports_its_tail() {
    let g = graph_of(
        "comb",
        &[("loop", "sin(2*pi*220*t) + 0.5*self(t - 0.01s)\n")],
    );
    let held = render(
        &g,
        "loop",
        RenderConfig::seconds(8_000, 0.05),
        &Tier::default(),
    )
    .expect("a comb");
    let id = held.id("loop").expect("the root");
    let buffer = held.output(id).expect("a rendered comb");
    assert_eq!(held.labels[&id].source, Source::Exact);
    let Detail::Lines {
        summed, tail_db, ..
    } = &held.labels[&id].detail
    else {
        panic!(
            "a comb is a line spectrum, got {:?}",
            held.labels[&id].detail
        );
    };
    assert_eq!(*summed, 1, "every term lands on the one resonance");
    assert!(
        tail_db.is_some_and(|db| db < 0.0),
        "the truncated tail is stated: {tail_db:?}"
    );
    let peak = buffer.plane(0).iter().fold(0.0f64, |a, s| a.max(s.abs()));
    let closed = 1.0
        / libm::hypot(
            1.0 - 0.5 * libm::cos(-std::f64::consts::TAU * 220.0 * 0.01),
            0.5 * libm::sin(-std::f64::consts::TAU * 220.0 * 0.01),
        );
    let off = 20.0 * libm::log10(peak / closed);
    assert!(
        off < 0.0 && off > tail_db.expect("a stated tail"),
        "the render sits between the closed loop and the tail the label states: {off} dB"
    );
}

/// Two nodes each lower a series of their own, and one closed form holds both. An index one of them
/// reused would be captured by the other's expansion, folding two sums into one.
#[test]
fn two_nested_loops_expand_under_indices_of_their_own() {
    let g = graph_of(
        "nested-loops",
        &[
            ("comb", "x + g*self(t - delay)\n"),
            ("inner", "@comb(t, x=1, delay=0.01, g=0.5)\n"),
            ("outer", "@comb(t, x=@inner, delay=0.013, g=0.25)\n"),
        ],
    );
    let held = render(
        &g,
        "outer",
        RenderConfig::seconds(44_100, 0.001),
        &Tier::default(),
    )
    .expect("two nested Neumann series render");
    let buffer = held
        .output(held.id("outer").expect("the root"))
        .expect("a rendered loop");
    let profile = sva_samples::PSYCHOACOUSTIC_V1;
    let taken = |first: f64, g: f64| {
        (0..)
            .find(|n| first * g.powi(*n) / (1.0 - g) <= profile.half_lsb())
            .expect("a tail under the precision")
    };
    let want: f64 = (0..taken(1.0, 0.5)).map(|j| 0.5f64.powi(j)).sum::<f64>()
        * (0..taken(2.0, 0.25)).map(|k| 0.25f64.powi(k)).sum::<f64>();
    assert!(
        (buffer.at(0, 0) - want).abs() < 1e-12,
        "two truncated geometrics multiply: {} against {want}",
        buffer.at(0, 0)
    );
}

/// `join` stacks its operands' channels, so an operand reading `self` whole would make the node
/// wider than itself: no width holds it, and typing says so before any sample is run.
#[test]
fn a_join_over_its_own_loop_refuses_at_typing() {
    for body in [
        "join(crop(sin(2*pi*440*t) + 0.5*self[idx(t - 17ms)], 0s, 1s), \
         crop(sin(2*pi*660*t) + 0.5*self[idx(t - 23ms)], 0s, 1s))\n",
        "join(crop(sin(2*pi*440*t) + 0.5*self[idx(t - 17ms)], 0s, 1s), crop(sin(t), 0s, 1s))\n",
    ] {
        let refused = form("join-over-self", body).expect_err("no width holds it");
        assert_eq!(refused.code(), "type.join_over_self", "{body}");
    }
}
