// Concern: proves `draw(k, seed=)` is noise's keyed draw at every rate, and what it refuses | Non-concern: the hash (sva-formula hash.rs), `rand` | IO: (a composition, rate) -> samples or a refusal

use crate::fixtures::{graph_of, samples};
use sva_engine::{RenderConfig, Tier, render};

const RATES: [u32; 2] = [44_100, 48_000];

fn rendered(files: &[(&str, &str)], rate: u32) -> Vec<f64> {
    let g = graph_of("draw", files);
    let held = render(
        &g,
        "body",
        RenderConfig::seconds(rate, 0.01),
        &Tier::default(),
    )
    .unwrap_or_else(|e| panic!("{files:?}: {e}"));
    samples(&held).remove("body").expect("the root's samples")
}

fn drawn(seed: u64, k: f64) -> f64 {
    sva_formula::draw_nearest(seed, k).expect("a key within an i64 draws")
}

/// Written as a number, through a row's bound parameters, or as a finite sum's index, every
/// sample is `draw_nearest(seed, k)` to the bit, at either rate.
#[test]
fn a_draw_is_the_keyed_draw_bit_for_bit_at_any_rate() {
    for rate in RATES {
        for (k, seed) in [
            (0.0, 0),
            (1.0, 7),
            (3.0, 7),
            (2.5, 1),
            (-4.0, 12),
            (1e6, 41),
        ] {
            let want = drawn(seed, k).to_bits();
            for files in [
                vec![("body", format!("draw({k}, seed={seed})\n"))],
                vec![("body", format!("draw({k}, {seed})\n"))],
                vec![
                    ("atom", "k = 0\nkey = 0\ndraw(k, seed=key)\n".to_string()),
                    ("body", format!("@atom(t, k={k}, key={seed})\n")),
                ],
            ] {
                let files: Vec<(&str, &str)> =
                    files.iter().map(|(p, b)| (*p, b.as_str())).collect();
                let found = rendered(&files, rate);
                assert!(!found.is_empty());
                assert!(
                    found.iter().all(|v| v.to_bits() == want),
                    "{files:?} at {rate} Hz: {:?} is not {}",
                    &found[..2],
                    drawn(seed, k)
                );
            }
        }
        let found = rendered(&[("body", "sum(j, 2, 2, draw(j, seed=9))\n")], rate);
        assert!(found.iter().all(|v| v.to_bits() == drawn(9, 2.0).to_bits()));
        let found = rendered(&[("body", "sum(j, 1, 3, draw(j, seed=9))\n")], rate);
        let want = drawn(9, 1.0) + drawn(9, 2.0) + drawn(9, 3.0);
        assert!(found.iter().all(|v| (v - want).abs() <= 4.0 * f64::EPSILON));
    }
}

/// A key or seed that reads `t`, or a signal, folds to no number; each refusal is its own
/// code, and every code is one `builtins` prints.
#[test]
fn a_draw_that_reads_time_is_refused_by_its_own_code() {
    for (files, code) in [
        (
            vec![("body", "draw(t, seed=1)\n")],
            "engine.non_constant_argument",
        ),
        (
            vec![("body", "draw(3, seed=t)\n")],
            "engine.non_constant_argument",
        ),
        (
            vec![("body", "draw(3, t)\n")],
            "engine.non_constant_argument",
        ),
        (
            vec![("body", "draw(0*t + 2, seed=1)\n")],
            "engine.non_constant_argument",
        ),
        (
            vec![("sig", "sin(2*pi*t)\n"), ("body", "draw(@sig, seed=1)\n")],
            "engine.non_constant_argument",
        ),
        (
            vec![("body", "draw(rand(t, seed=2), seed=1)\n")],
            "engine.non_constant_argument",
        ),
        (
            vec![("body", "sum(j, 1, inf, draw(j, seed=1)/pow(j, 2))\n")],
            "engine.non_constant_argument",
        ),
        (
            vec![("body", "draw(1e19, seed=1)\n")],
            "engine.draw_past_index",
        ),
        (
            vec![("body", "draw(inf, seed=1)\n")],
            "engine.infinite_value",
        ),
        (
            vec![("body", "draw(3, seed=inf)\n")],
            "engine.infinite_value",
        ),
        (vec![("body", "draw(3)\n")], "grammar.arity"),
        (vec![("body", "draw(seed=3)\n")], "grammar.arity"),
        (
            vec![("body", "draw(3, seed=1, hz=2)\n")],
            "grammar.unknown_named_argument",
        ),
    ] {
        let g = graph_of("refused-draw", &files);
        for rate in RATES {
            let Err(refused) = render(
                &g,
                "body",
                RenderConfig::seconds(rate, 0.01),
                &Tier::default(),
            ) else {
                panic!("{files:?} reads time and renders");
            };
            assert_eq!(refused.code(), code, "{files:?}: {refused}");
        }
        assert!(
            sva_engine::REGISTRY.iter().any(|(name, _)| *name == code),
            "`{code}` is in the vocabulary `builtins` prints"
        );
    }
}
