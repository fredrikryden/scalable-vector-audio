// Concern: proves a discrete node steps at the rate a render asks for, accumulator and all | Non-concern: any solver's own grid (sva-samples) | IO: (a composition, rate) -> a Buffer

use crate::fixtures::graph_of;
use sva_engine::{RenderConfig, Tier, render};

/// A low rate keeps these tests fast; the filter and limiter tests keep 44.1 kHz.
const RATE: u32 = 8_000;

/// No block sizes a sampled loop: a delay-`d` read lands on `i - d`, and `written`
/// refounded at every `i` founds it. Longer than one step, it reads silence until then.
#[test]
fn a_self_read_is_founded_sample_by_sample_without_a_block() {
    let g = graph_of(
        "three-step",
        &[("acc", "sample(1 + 0*t) + 0.5*self[idx(t) - 3]\n")],
    );
    let held = render(
        &g,
        "acc",
        RenderConfig::seconds(RATE, 10.0 / f64::from(RATE)),
        &Tier::default(),
    )
    .expect("a recurrence");
    let root = held.id("acc").expect("the root");
    let buffer = held.output(root).expect("a rendered loop");
    assert_eq!(buffer.len(), 10);

    let mut want = [0.0f64; 10];
    for i in 0..10 {
        want[i] = 1.0 + 0.5 * if i >= 3 { want[i - 3] } else { 0.0 };
    }
    for (i, held) in buffer.plane(0).iter().enumerate() {
        assert!(
            (held - want[i]).abs() < 1e-12,
            "sample {i}: {held} against {}",
            want[i]
        );
    }
    assert_eq!(buffer.at(0, 2), 1.0, "before the first founded read");
    assert_eq!(buffer.at(0, 3), 1.5, "the first founded read");
}

fn plane(g: &sva_ast::Graph, node: &str) -> Vec<f64> {
    let held = render(g, node, RenderConfig::seconds(RATE, 0.05), &Tier::default())
        .unwrap_or_else(|e| panic!("{node}: {e}"));
    let id = held.id(node).unwrap_or_else(|| panic!("{node} typed"));
    held.output(id).expect("a rendered node").plane(0).to_vec()
}

/// The bug this closes: a sampled operand's crop kept its hard edges and dropped both
/// shoulders, so the same signal cropped the same way sounded two different windows.
#[test]
fn a_sampled_crop_takes_the_closed_forms_window() {
    let g = graph_of(
        "crop-shoulders",
        &[
            (
                "closed",
                "crop(sin(2*pi*220*t), 10ms, 40ms, rise=5ms, fall=10ms)\n",
            ),
            (
                "sampled",
                "crop(sample(sin(2*pi*220*t)), 10ms, 40ms, rise=5ms, fall=10ms)\n",
            ),
            ("hard", "crop(sample(sin(2*pi*220*t)), 10ms, 40ms)\n"),
        ],
    );
    let (closed, sampled) = (plane(&g, "closed"), plane(&g, "sampled"));
    assert_eq!(closed.len(), sampled.len());
    for (i, (c, s)) in closed.iter().zip(&sampled).enumerate() {
        assert!(
            (c - s).abs() < 1e-9,
            "sample {i}: the closed form reads {c}, the sampled crop {s}"
        );
        let inside = (RATE as usize / 100..RATE as usize * 4 / 100).contains(&i);
        assert!(inside || *s == 0.0, "sample {i} lies outside [10ms, 40ms)");
    }
    let hard = plane(&g, "hard");
    let first = (0.01 * f64::from(RATE)) as usize + 1;
    assert!(
        sampled[first].abs() < hard[first].abs(),
        "the rise opens the window gradually: {} against {}",
        sampled[first],
        hard[first]
    );
}

/// `sp` is one step of the rate in use: a limiter decaying per step lets go after the same
/// count of samples at 44.1 kHz and at 96 kHz.
#[test]
fn an_sp_limiter_releases_after_the_same_steps_at_every_rate() {
    let g = graph_of(
        "limiter",
        &[
            ("target", "sample(crop(0.5 + 0*t, 0s, 0.1s))\n"),
            ("gr", "max(@target, self[idx(t) - 1]*0.99975)\n"),
        ],
    );
    let at = |rate: u32| {
        let held = render(&g, "gr", RenderConfig::seconds(rate, 0.3), &Tier::default())
            .unwrap_or_else(|e| panic!("{rate}: {e}"));
        held.output(held.root).expect("a gain").plane(0).to_vec()
    };
    for (rate, let_go) in [(44_100, 4_410), (96_000, 9_600)] {
        let released = at(rate)
            .iter()
            .skip(let_go)
            .position(|g| *g <= 0.5 / std::f64::consts::E);
        assert_eq!(
            released,
            Some(3_999),
            "{rate}: 3999 steps after the target lets go"
        );
    }
}

/// `rand(t, seed=k)` at sample `n` is the draw of `k` at step `n`, at any rate and on
/// every render.
#[test]
fn rand_draws_its_seed_at_each_sample_index() {
    let g = graph_of("keyed", &[("noise", "crop(rand(t, seed=5), 0s, 10ms)\n")]);
    for rate in [RATE, 11_025] {
        let read = || {
            let held = render(
                &g,
                "noise",
                RenderConfig::seconds(rate, 0.01),
                &Tier::default(),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            held.output(held.root).expect("a buffer").plane(0).to_vec()
        };
        let drawn = read();
        assert!(!drawn.is_empty());
        for (n, v) in drawn.iter().enumerate() {
            let want = sva_formula::draw(5, n as i64);
            assert_eq!(
                v.to_bits(),
                want.to_bits(),
                "{rate}: sample {n}: {v} against {want}"
            );
        }
        assert_eq!(read(), drawn, "{rate}: a second render draws the same");
    }
}

/// A key between two steps draws at the step nearest it, ties to even.
#[test]
fn rand_between_steps_draws_the_nearest_step() {
    for (key, step) in [("2.5sp", 2), ("3.5sp", 4), ("3.4sp", 3)] {
        let g = graph_of("between", &[("still", &format!("rand({key}, seed=5)\n"))]);
        let held = render(
            &g,
            "still",
            RenderConfig::seconds(RATE, 0.001),
            &Tier::default(),
        )
        .unwrap_or_else(|e| panic!("{key}: {e}"));
        let drawn = held.output(held.root).expect("a buffer").plane(0).to_vec();
        let want = sva_formula::draw(5, step);
        assert!(
            !drawn.is_empty() && drawn.iter().all(|v| v.to_bits() == want.to_bits()),
            "{key}: {drawn:?} against step {step}'s {want}"
        );
    }
}

/// A read whose time wraps by an exact period lands on a sample at every jump.
#[test]
fn a_time_wrapped_by_an_exact_period_reads_whole_samples() {
    let g = graph_of(
        "wrapped",
        &[
            ("noise", "crop(rand(t, seed=1), 0s, 1s)\n"),
            ("looped", "crop(rand(t % 0.03s, seed=1), 0s, 0.1s)\n"),
            ("held", "crop(rand(t - t % 1sp, seed=1), 0s, 0.1s)\n"),
        ],
    );
    let read = |node: &str| {
        let held = render(&g, node, RenderConfig::seconds(RATE, 0.1), &Tier::default())
            .unwrap_or_else(|e| panic!("{e}"));
        held.output(held.root).expect("a buffer").plane(0).to_vec()
    };
    let (noise, looped, held) = (read("noise"), read("looped"), read("held"));
    let period = (0.03 * f64::from(RATE)) as usize;
    for (n, (l, h)) in looped.iter().zip(&held).enumerate() {
        assert!((l - noise[n % period]).abs() < 1e-9, "sample {n}: {l}");
        assert!((h - noise[n]).abs() < 1e-9, "sample {n}: {h}");
    }
}
